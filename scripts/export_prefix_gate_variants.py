"""Prefix-64 decoupled channel gating: mech allocation vs faith cost.

The devkit's ``frame_relevance`` pools only ``relevance[ch][0:64]`` (one STFT
frame, ``hop_length=64``), so the mechanical-alignment channel weighting reads
*only the first 64 of 100 time steps*. For any class-conditioned gate matrix G,
multiplying a channel's whole row or just its t<64 prefix produces *identical*
mechanical channel ratios — but the prefix version leaves the t>=64 cells'
cross-channel ordering untouched, bounding the faithfulness damage of
aggressive mechanical bets (exp-014's global ``torque_only_faults`` cost
faith -0.042; the prefix version should cost much less at identical mech).

Each variant applies two gates to the frozen final2 classifier's relevance:

    relevance[:, :, :64] *= (probs @ G_prefix)[:, :, None]
    relevance[:, :, 64:] *= (probs @ G_suffix)[:, :, None]

``G_suffix`` lets the faith-optimal exp-023 gate keep working on the cells the
mechanical metric cannot see. Probabilities are untouched (relevance-only).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_relevance_gate_variants import (
    CHANNELS,
    CLASSES,
    evaluate_with_optional_band,
    flatten_metrics,
    ones,
    proxy_band_configs,
)
from export_temperature_variants import exp023_gate_matrix, load_base_model
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES
from gearxai_workspace.evaluate import sample_validation, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet

MECH_PREFIX = 64  # devkit frame_relevance pools relevance[ch][0:hop_length=64]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    prefix_gates: np.ndarray  # [9, 8]
    suffix_gates: np.ndarray  # [9, 8]


class PrefixGateModel(nn.Module):
    """Frozen GearXAINet with separate class-conditioned prefix/suffix gates."""

    def __init__(self, base: GearXAINet, prefix_gates: np.ndarray, suffix_gates: np.ndarray):
        super().__init__()
        for gates in (prefix_gates, suffix_gates):
            if gates.shape != (NUM_CLASSES, NUM_CHANNELS):
                raise ValueError(f"Expected gate matrix [9, 8], got {gates.shape}.")
        self.base = base
        self.register_buffer("prefix_gates", torch.as_tensor(prefix_gates, dtype=torch.float32))
        self.register_buffer("suffix_gates", torch.as_tensor(suffix_gates, dtype=torch.float32))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities, relevance = self.base(windows)
        prefix_gate = (probabilities @ self.prefix_gates).unsqueeze(2)  # [N, 8, 1]
        suffix_gate = (probabilities @ self.suffix_gates).unsqueeze(2)
        prefix = relevance[:, :, :MECH_PREFIX] * prefix_gate
        suffix = relevance[:, :, MECH_PREFIX:] * suffix_gate
        return probabilities, torch.cat([prefix, suffix], dim=2)


def single_channel_gates(
    channel_by_class: dict[str, str], *, boost: float, other: float
) -> np.ndarray:
    """Per-class hard concentration: ``boost`` on one channel, ``other`` elsewhere."""

    gates = ones()
    for class_name, channel_name in channel_by_class.items():
        class_id = CLASSES.index(class_name)
        gates[class_id, :] = other
        gates[class_id, CHANNELS.index(channel_name)] = boost
    return gates


def default_variants() -> list[Variant]:
    identity = ones()
    exp023 = exp023_gate_matrix()

    # Aggressive mechanical bet used as the stress instrument: every fault class
    # concentrated on its current proxy-best channel (exp-013 audit: torque for
    # most, motor for the low-frequency trio). Values deliberately harsh so the
    # global-vs-prefix faith gap is measurable.
    aggressive_map = {
        "CTF": "torque",
        "MTF": "torque",
        "RCF": "torque",
        "SWF": "torque",
        "BWF": "motor",
        "CWF": "motor",
        "IRF": "torque",
        "ORF": "motor",
    }
    aggressive = single_channel_gates(aggressive_map, boost=4.0, other=0.25)

    return [
        Variant("global_exp023", "control == final3 gate everywhere", exp023, exp023),
        Variant("prefix_exp023_suffix_id", "exp-023 gate on t<64 only", exp023, identity),
        Variant("global_aggressive", "harsh single-channel bet everywhere", aggressive, aggressive),
        Variant(
            "prefix_aggressive_suffix_id",
            "harsh single-channel bet on t<64 only",
            aggressive,
            identity,
        ),
        Variant(
            "prefix_aggressive_suffix_exp023",
            "harsh mech bet on t<64, faith-optimal exp-023 gate on t>=64",
            aggressive,
            exp023,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/prefix_exp028"))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=28028)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-proxies", action="store_true")
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
    windows, labels = sample_validation(args.data_dir, args.eval_n, seed=args.seed)
    sample = np.array(windows[:8], dtype=np.float32, copy=True)
    proxy_configs = proxy_band_configs()
    base = load_base_model(args.checkpoint)

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "eval_n": int(len(labels)),
        "seed": args.seed,
        "variants": {},
    }
    for variant in selected:
        model = PrefixGateModel(base, variant.prefix_gates, variant.suffix_gates)
        variant_dir = args.out_dir / variant.name
        onnx_path = variant_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_with_optional_band(onnx_path, windows, labels, batch_size=args.batch_size)
        metrics = flatten_metrics(report, n=len(labels))

        proxy_metrics: dict[str, Any] = {}
        if not args.skip_proxies:
            for proxy_name, band_config in proxy_configs.items():
                proxy_report = evaluate_with_optional_band(
                    onnx_path, windows, labels, batch_size=args.batch_size, band_config=band_config
                )
                proxy_metrics[proxy_name] = {
                    "mechanical": proxy_report["mechanical"]["mechanical_score"],
                    "expected_band_mass": proxy_report["mechanical"]["expected_band_mass"],
                }

        result = {
            "variant": variant.name,
            "description": variant.description,
            "onnx_path": str(onnx_path),
            "prefix_gates": variant.prefix_gates.tolist(),
            "suffix_gates": variant.suffix_gates.tolist(),
            "self_check": check,
            "public_metrics": metrics,
            "proxy_metrics": proxy_metrics,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results["variants"][variant.name] = result
        print(summary_line(variant.name, metrics))
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
