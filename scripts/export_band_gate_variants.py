"""In-graph per-sample band-fraction relevance allocation (mech endgame, exp-031).

Class-level channel gates (exp-014..023, probes) can only bet one channel per
class; the per-sample oracle from exp-002b (allocate each window's relevance to
the channel whose spectrum best hits the band) scored proxy EAS 0.563 vs ~0.34
for |x|. This script puts that oracle *inside the ONNX graph*:

    mag[N,8,51]  = |rfft_hann(x)|            (two fixed [100,51] matmuls + abs)
    ind[N,51]    = probs @ B                  (B[9,51]: per-class band indicator)
    w[N,8]       = (mag @ ind) / (mag @ 1)    (per-sample channel band fraction)
    relevance   *= (w / max_ch w) ** gamma    (gamma -> argmax-like allocation)

The band matrix ``B`` is a *configuration*: today it is filled from the local
proxy band configs (mechanism validation only); after the leaderboard probes
identify the official band structure, the same graph re-exports with the real
bands. Composes with the in-graph occlusion gate and output temperature
(exp-027/029 family); probabilities stay bit-identical to the base model.

The devkit's w_ch uses the Hann-windowed single-frame STFT magnitude L1
fraction (metrics.py); the Hann window is baked into the DFT matrices and the
scipy scaling constant cancels in the ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_occlusion_gate_variants import FEATURE_LENGTH
from export_relevance_gate_variants import (
    evaluate_with_optional_band,
    flatten_metrics,
    proxy_band_configs,
)
from export_temperature_variants import load_base_model
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH
from gearxai_workspace.evaluate import sample_validation, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet

NUM_BINS = WINDOW_LENGTH // 2 + 1  # 51 rfft bins, 51.2 Hz spacing
BIN_HZ = 5120.0 / WINDOW_LENGTH


def hann_dft_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Fixed [100, 51] cos/sin matrices with the periodic Hann window baked in."""

    t = np.arange(WINDOW_LENGTH, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / WINDOW_LENGTH)
    k = np.arange(NUM_BINS, dtype=np.float64)
    angle = 2.0 * np.pi * np.outer(t, k) / WINDOW_LENGTH  # [100, 51]
    cos_mat = (hann[:, None] * np.cos(angle)).astype(np.float32)
    sin_mat = (-hann[:, None] * np.sin(angle)).astype(np.float32)
    return cos_mat, sin_mat


def band_indicator_matrix(band_config: dict[str, Any]) -> np.ndarray:
    """[9, 51] per-class band indicator from a devkit-style band config."""

    indicator = np.zeros((NUM_CLASSES, NUM_BINS), dtype=np.float32)
    freqs = np.arange(NUM_BINS, dtype=np.float64) * BIN_HZ
    for class_id_str, bands in band_config["classes"].items():
        row = int(class_id_str)
        for low, high in bands:
            mask = (freqs >= float(low)) & (freqs <= float(high))
            indicator[row, mask] = 1.0
    return indicator


class BandOccGateModel(nn.Module):
    """Frozen GearXAINet + occlusion gate + in-graph per-sample band gate + T."""

    def __init__(
        self,
        base: GearXAINet,
        band_indicator: np.ndarray,
        *,
        gamma: float,
        occ_alpha: float = 0.5,
        occ_eps: float = 0.05,
        temperature: float = 8.0,
        use_occ: bool = True,
    ):
        super().__init__()
        self.base = base
        self.gamma = float(gamma)
        self.occ_alpha = float(occ_alpha)
        self.occ_eps = float(occ_eps)
        self.temperature = float(temperature)
        self.use_occ = use_occ
        masks = torch.ones(NUM_CHANNELS, 1, NUM_CHANNELS, 1)
        for channel in range(NUM_CHANNELS):
            masks[channel, 0, channel, 0] = 0.0
        self.register_buffer("masks", masks)
        cos_mat, sin_mat = hann_dft_matrices()
        self.register_buffer("dft_cos", torch.from_numpy(cos_mat))  # [100, 51]
        self.register_buffer("dft_sin", torch.from_numpy(sin_mat))
        self.register_buffer(
            "band_ind", torch.from_numpy(band_indicator.astype(np.float32))
        )  # [9, 51]

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

        if self.use_occ:
            occ_probs = probs_r[1:]
            p_base = (probs * probs).sum(dim=1)
            p_occ = (occ_probs * probs.unsqueeze(0)).sum(dim=2)
            drop = torch.relu(p_base.unsqueeze(0) - p_occ)
            occ_gate = (drop + self.occ_eps).pow(self.occ_alpha).transpose(0, 1)  # [N, 8]
            relevance = relevance * occ_gate.unsqueeze(2)

        # Per-sample channel band fraction on the devkit's Hann single-frame STFT.
        real = windows @ self.dft_cos  # [N, 8, 51]
        imag = windows @ self.dft_sin
        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        ind = probs @ self.band_ind  # [N, 51] soft class-selected band indicator
        band_mass = torch.bmm(mag, ind.unsqueeze(2)).squeeze(2)  # [N, 8]
        total_mass = mag.sum(dim=2)  # [N, 8]
        w = band_mass / (total_mass + 1e-12)
        # Normalize by the per-sample max so gamma sharpens toward the argmax
        # channel; +eps keeps classes without bands (w == 0) at a uniform gate.
        w_norm = (w + 1e-6) / (w.max(dim=1, keepdim=True).values + 1e-6)
        relevance = relevance * w_norm.pow(self.gamma).unsqueeze(2)

        powered = probs.pow(self.temperature)
        probs_out = powered / powered.sum(dim=1, keepdim=True)
        return probs_out, relevance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/exp029_final/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/band_exp031"))
    parser.add_argument(
        "--band-source",
        default="audit_peaks",
        help="proxy band config used to fill B (mechanism validation; real bands post-probe)",
    )
    parser.add_argument("--gammas", nargs="+", type=float, default=[2.0, 4.0, 8.0])
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27027)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proxies = proxy_band_configs()
    if args.band_source not in proxies:
        raise ValueError(f"Unknown band source {args.band_source}; have {sorted(proxies)}")
    indicator = band_indicator_matrix(proxies[args.band_source])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows, labels = sample_validation(args.data_dir, args.eval_n, seed=args.seed)
    sample = np.array(windows[:8], dtype=np.float32, copy=True)
    base = load_base_model(args.checkpoint)

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "band_source": args.band_source,
        "eval_n": int(len(labels)),
        "seed": args.seed,
        "variants": {},
    }
    for gamma in args.gammas:
        name = f"band_{args.band_source}_g{gamma:g}".replace(".", "p")
        model = BandOccGateModel(base, indicator, gamma=gamma)
        variant_dir = args.out_dir / name
        onnx_path = variant_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_with_optional_band(onnx_path, windows, labels, batch_size=args.batch_size)
        metrics = flatten_metrics(report, n=len(labels))

        proxy_metrics: dict[str, Any] = {}
        for proxy_name, band_config in proxies.items():
            proxy_report = evaluate_with_optional_band(
                onnx_path, windows, labels, batch_size=args.batch_size, band_config=band_config
            )
            proxy_metrics[proxy_name] = {
                "mechanical": proxy_report["mechanical"]["mechanical_score"],
                "expected_band_mass": proxy_report["mechanical"]["expected_band_mass"],
            }

        result = {
            "gamma": gamma,
            "onnx_path": str(onnx_path),
            "self_check": check,
            "public_metrics": metrics,
            "proxy_metrics": proxy_metrics,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results["variants"][name] = result
        print(summary_line(name, metrics))
        for proxy_name, proxy in proxy_metrics.items():
            print(
                f"  proxy {proxy_name}: mech={proxy['mechanical']:.4f} "
                f"band_mass={proxy['expected_band_mass']:.4f}"
            )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
