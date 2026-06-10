"""Additive mech-carrier for gear classes (exp-034b).

mechanical_v2 gear contexts split into two regimes (exp-033/034 sims): fixed
speed (official physics bands at mesh frequencies; torque has the best band
fraction) and variable speed (the official ridge estimator settles ~3 Hz, so
the bands sit in the low-frequency region; motor wins). A per-window detector
on the global low-frequency energy fraction (lf > tau -> motor regime)
captures 85% of the perfect-regime policy in simulation.

Instead of reallocating relevance (which destroys faith), we ADD a carrier:

    rel_out = rel_base + lam * p_gear * window_mass * carrier(ch | lf)

The carrier is uniform in time on one channel, so faith's top-k ordering stays
base-driven at moderate lam while the mechanical channel masses follow the
carrier. ``mode=motor_only`` injects only in the lf>tau regime (3x payoff
side), halving the faith exposure.
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
from export_band_gate_variants import hann_dft_matrices
from export_occlusion_gate_variants import FEATURE_LENGTH
from export_temperature_variants import load_base_model
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet

GEAR_CLASS_SLICE = slice(1, 5)
MOTOR, TORQUE = 0, 4
LOW_BINS = 3  # bins 0..2 -> 0-154 Hz
SIGMOID_SCALE = 60.0


@dataclass(frozen=True)
class Variant:
    name: str
    lam: float
    mode: str  # "both" | "motor_only"
    tau: float = 0.7


class GearCarrierModel(nn.Module):
    """occ+T8 wrapper plus a class-gated additive mech carrier."""

    def __init__(
        self,
        base: GearXAINet,
        *,
        lam: float,
        mode: str,
        tau: float,
        occ_alpha: float = 1.0,
        occ_eps: float = 0.2,
        temperature: float = 8.0,
    ):
        super().__init__()
        if mode not in ("both", "motor_only"):
            raise ValueError(f"mode must be 'both' or 'motor_only', got {mode!r}")
        self.base = base
        self.lam = float(lam)
        self.mode = mode
        self.tau = float(tau)
        self.occ_alpha = float(occ_alpha)
        self.occ_eps = float(occ_eps)
        self.temperature = float(temperature)
        masks = torch.ones(NUM_CHANNELS, 1, NUM_CHANNELS, 1)
        for channel in range(NUM_CHANNELS):
            masks[channel, 0, channel, 0] = 0.0
        self.register_buffer("masks", masks)
        cos_mat, sin_mat = hann_dft_matrices()
        self.register_buffer("dft_cos", torch.from_numpy(cos_mat))
        self.register_buffer("dft_sin", torch.from_numpy(sin_mat))
        onehot = torch.zeros(2, NUM_CHANNELS, 1)
        onehot[0, MOTOR, 0] = 1.0
        onehot[1, TORQUE, 0] = 1.0
        self.register_buffer("carrier_rows", onehot)  # [2, 8, 1]

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

        if self.lam > 0:
            real = windows @ self.dft_cos
            imag = windows @ self.dft_sin
            energy = real * real + imag * imag  # [N, 8, 51]
            low = energy[:, :, :LOW_BINS].sum(dim=(1, 2))
            total = energy.sum(dim=(1, 2))
            lf = low / (total + 1e-12)  # [N]
            motor_w = torch.sigmoid(SIGMOID_SCALE * (lf - self.tau))  # ~hard switch
            if self.mode == "both":
                mix = (
                    motor_w.view(-1, 1, 1) * self.carrier_rows[0]
                    + (1.0 - motor_w).view(-1, 1, 1) * self.carrier_rows[1]
                )
            else:  # motor_only: inject nothing in the torque regime
                mix = motor_w.view(-1, 1, 1) * self.carrier_rows[0]
            p_gear = probs[:, GEAR_CLASS_SLICE].sum(dim=1)  # [N]
            window_mass = relevance.sum(dim=(1, 2)) / WINDOW_LENGTH  # [N]
            scale = (self.lam * p_gear * window_mass).view(-1, 1, 1)
            relevance = relevance + scale * mix

        powered = probs.pow(self.temperature)
        probs_out = powered / powered.sum(dim=1, keepdim=True)
        return probs_out, relevance


def default_variants() -> list[Variant]:
    return [
        Variant("carrier_off", 0.0, "both"),
        Variant("both_l1", 1.0, "both"),
        Variant("both_l2", 2.0, "both"),
        Variant("both_l4", 4.0, "both"),
        Variant("motor_l1", 1.0, "motor_only"),
        Variant("motor_l2", 2.0, "motor_only"),
        Variant("motor_l4", 4.0, "motor_only"),
        Variant("motor_l8", 8.0, "motor_only"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/exp029_final/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/carrier_exp034b"))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--faith-n", type=int, default=3000)
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

    all_results: dict[str, Any] = {"checkpoint": str(args.checkpoint), "variants": {}}
    for variant in selected:
        model = GearCarrierModel(base, lam=variant.lam, mode=variant.mode, tau=variant.tau)
        variant_dir = args.out_dir / variant.name
        onnx_path = variant_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)

        print(f"=== {variant.name} ===")
        mech = score_mechanical_v2(
            onnx_path, data_dir=args.data_dir, batch_size=args.batch_size, gear_only=True
        )
        faith_metrics = evaluate_onnx(
            onnx_path,
            data_dir=Path("data/prepared"),
            n=args.faith_n,
            seed=args.faith_seed,
            batch_size=args.batch_size,
        )
        faith_metrics.pop("raw", None)
        print(summary_line(f"{variant.name} faith", faith_metrics))

        result = {
            "variant": variant.__dict__,
            "onnx_path": str(onnx_path),
            "self_check": check,
            "mech_gear": mech,
            "faith": faith_metrics,
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
