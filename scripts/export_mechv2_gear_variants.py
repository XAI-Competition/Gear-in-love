"""Gear-class relevance redesign against the official mechanical_v2 (exp-034).

mechanical_v2 scores gear classes (CTF/MTF/RCF/SWF = 1..4) on *physics* bands
(mesh harmonics +- carrier sidebands, speed-dependent), and our calibrated
local gear enrichment is only 0.07-0.23 — the biggest remaining headroom.

Two levers, both composed onto the deployed occ+T8 wrapper:

1. **Window-level band scalar (zero faith cost).** Faithfulness normalizes
   relevance per window, but mechanical_v2 overlap-adds raw values across
   windows — so ``relevance *= s(x)`` with a per-window scalar is invisible to
   faith while shaping the context-level time profile. We use
   ``s = band_fraction ** (gamma_t * p_gear)`` where band_fraction is the
   window's energy fraction inside the speed-union gear-mesh bands (DFT bins)
   and ``p_gear`` is the predicted gear-class probability mass (exponent
   blending keeps non-gear windows at s = 1).

2. **Gear-class channel weighting (faith-coupled).** Within gear-class
   windows, weight channels by their own union-band energy fraction,
   ``(w_ch / max_ch w_ch) ** (gamma_c * p_gear)`` — per-sample, like exp-031
   but class-gated and aimed at the *real* bands.

Union band over speeds 20-50 Hz with geometry defaults: mesh2 ~ 3.65*s ->
[73, 183] Hz, mesh1 ~ 16.67*s -> [333, 833] Hz, mesh1 H2 -> [667, 1667] Hz,
each +-20 Hz plus carrier sidebands (small). DFT bins are 51.2 Hz wide.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from eval_mechanical_v2 import score_mechanical_v2
from export_band_gate_variants import NUM_BINS, hann_dft_matrices
from export_occlusion_gate_variants import FEATURE_LENGTH
from export_temperature_variants import load_base_model
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet

BIN_HZ = 5120.0 / WINDOW_LENGTH  # 51.2 Hz

# Speed-union gear bands (geometry defaults, speeds 20-50 Hz, +-20 Hz width).
GEAR_UNION_BANDS_HZ = ((53.0, 203.0), (313.0, 853.0), (647.0, 1687.0))
GEAR_CLASS_SLICE = slice(1, 5)  # CTF, MTF, RCF, SWF


def gear_union_indicator() -> np.ndarray:
    indicator = np.zeros(NUM_BINS, dtype=np.float32)
    freqs = np.arange(NUM_BINS) * BIN_HZ
    for low, high in GEAR_UNION_BANDS_HZ:
        indicator[(freqs >= low) & (freqs <= high)] = 1.0
    return indicator


@dataclass(frozen=True)
class Variant:
    name: str
    occ_alpha: float
    occ_eps: float
    gamma_t: float  # window-level band scalar exponent (0 = off)
    gamma_c: float  # per-sample channel band weighting exponent (0 = off)


class GearBandModel(nn.Module):
    """occ+T8 wrapper plus gear-class band scalar / channel weighting."""

    def __init__(
        self,
        base: GearXAINet,
        *,
        occ_alpha: float,
        occ_eps: float,
        gamma_t: float,
        gamma_c: float,
        temperature: float = 8.0,
    ):
        super().__init__()
        self.base = base
        self.occ_alpha = float(occ_alpha)
        self.occ_eps = float(occ_eps)
        self.gamma_t = float(gamma_t)
        self.gamma_c = float(gamma_c)
        self.temperature = float(temperature)
        masks = torch.ones(NUM_CHANNELS, 1, NUM_CHANNELS, 1)
        for channel in range(NUM_CHANNELS):
            masks[channel, 0, channel, 0] = 0.0
        self.register_buffer("masks", masks)
        cos_mat, sin_mat = hann_dft_matrices()
        self.register_buffer("dft_cos", torch.from_numpy(cos_mat))
        self.register_buffer("dft_sin", torch.from_numpy(sin_mat))
        self.register_buffer("band_ind", torch.from_numpy(gear_union_indicator()))  # [51]

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat([windows.unsqueeze(0), windows.unsqueeze(0) * self.masks], dim=0)
        flat = stacked.reshape(-1, NUM_CHANNELS, WINDOW_LENGTH)
        feat_all = self.base._features(flat)
        logits_all = self.base._logits_from_features(feat_all)
        probs_all = torch.softmax(logits_all, dim=1)
        probs_r = probs_all.reshape(9, -1, NUM_CLASSES)
        probs = probs_r[0]

        feat = feat_all.reshape(9, -1, self.base.feat_channels, FEATURE_LENGTH)[0]
        relevance = self.base._relevance_from(feat, probs, windows)

        occ_probs = probs_r[1:]
        p_base = (probs * probs).sum(dim=1)
        p_occ = (occ_probs * probs.unsqueeze(0)).sum(dim=2)
        drop = torch.relu(p_base.unsqueeze(0) - p_occ)
        occ_gate = (drop + self.occ_eps).pow(self.occ_alpha).transpose(0, 1)
        relevance = relevance * occ_gate.unsqueeze(2)

        if self.gamma_t > 0 or self.gamma_c > 0:
            real = windows @ self.dft_cos
            imag = windows @ self.dft_sin
            energy = real * real + imag * imag  # [N, 8, 51] band power
            band_ch = energy @ self.band_ind  # [N, 8]
            total_ch = energy.sum(dim=2)  # [N, 8]
            p_gear = probs[:, GEAR_CLASS_SLICE].sum(dim=1)  # [N]

            if self.gamma_c > 0:
                w = (band_ch + 1e-9) / (total_ch + 1e-9)  # [N, 8]
                w_norm = w / (w.max(dim=1, keepdim=True).values + 1e-12)
                exponent = (self.gamma_c * p_gear).unsqueeze(1)  # [N, 1]
                relevance = relevance * w_norm.pow(exponent).unsqueeze(2)

            if self.gamma_t > 0:
                frac = (band_ch.sum(dim=1) + 1e-9) / (total_ch.sum(dim=1) + 1e-9)  # [N]
                scalar = frac.pow(self.gamma_t * p_gear)  # 1.0 for non-gear windows
                relevance = relevance * scalar.view(-1, 1, 1)

        powered = probs.pow(self.temperature)
        probs_out = powered / powered.sum(dim=1, keepdim=True)
        return probs_out, relevance


def default_variants() -> list[Variant]:
    return [
        Variant("base_a1e0p2", 1.0, 0.2, 0.0, 0.0),  # control = P6 family
        Variant("wscalar_g1", 1.0, 0.2, 1.0, 0.0),
        Variant("wscalar_g2", 1.0, 0.2, 2.0, 0.0),
        Variant("wscalar_g4", 1.0, 0.2, 4.0, 0.0),
        Variant("chan_g1", 1.0, 0.2, 0.0, 1.0),
        Variant("chan_g2", 1.0, 0.2, 0.0, 2.0),
        Variant("wscalar_g2_chan_g1", 1.0, 0.2, 2.0, 1.0),
        Variant("a0p5e0p05_wscalar_g2", 0.5, 0.05, 2.0, 0.0),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/exp029_final/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mechv2_exp034"))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--faith-n", type=int, default=2000, help="0 skips the faith check")
    parser.add_argument("--faith-seed", type=int, default=34034)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = {variant.name: variant for variant in default_variants()}
    if args.variants == ["all"]:
        selected = list(available.values())
    else:
        unknown = [name for name in args.variants if name not in available]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(available)}")
        selected = [available[name] for name in args.variants]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_model(args.checkpoint)
    sample = np.random.default_rng(0).normal(size=(8, NUM_CHANNELS, WINDOW_LENGTH))
    sample = sample.astype(np.float32)

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "union_bands_hz": GEAR_UNION_BANDS_HZ,
        "variants": {},
    }
    for variant in selected:
        model = GearBandModel(
            base,
            occ_alpha=variant.occ_alpha,
            occ_eps=variant.occ_eps,
            gamma_t=variant.gamma_t,
            gamma_c=variant.gamma_c,
        )
        variant_dir = args.out_dir / variant.name
        onnx_path = variant_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)

        print(f"=== {variant.name} ===")
        mech = score_mechanical_v2(
            onnx_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            gear_only=True,
        )

        faith_metrics: dict[str, Any] | None = None
        if args.faith_n > 0:
            faith_metrics = evaluate_onnx(
                onnx_path,
                data_dir=Path("data/prepared"),
                n=args.faith_n,
                seed=args.faith_seed,
                batch_size=args.batch_size,
            )
            faith_metrics.pop("raw", None)
            print(summary_line(f"{variant.name} faith-check", faith_metrics))

        result = {
            "variant": variant.__dict__,
            "onnx_path": str(onnx_path),
            "self_check": check,
            "mech_gear": mech,
            "faith_check": faith_metrics,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results["variants"][variant.name] = result

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
