"""In-graph per-sample channel-occlusion relevance gating (relevance-only).

exp-003a's probe showed that weighting channels by their *causal occlusion
importance* (zero a channel -> how much does the predicted-class confidence
drop) lifts faithfulness to 0.76-0.81, far above any class-conditioned static
gate. exp-003b failed to capture that by distilling occlusion into a
class-level gate; this script instead computes the exact per-sample occlusion
weights *inside the ONNX graph*:

    stacked = concat([x, x with ch0 zeroed, ..., x with ch7 zeroed])  # [9N,8,100]
    probs_all = classifier(stacked)                                   # one shared subgraph
    drop[c]   = relu(p_base - p_occ[c])      (soft predicted-class confidence)
    gate[c]   = (drop[c] + eps) ** alpha
    relevance = base_relevance * gate[:, :, None]

Folding the occlusion variants into the *batch* dimension keeps a single copy
of the conv stack in the graph (~+15 ops instead of ~+25 per extra pass), so
the simplicity cost stays small. The classifier checkpoint is frozen; the
``probabilities`` output is bit-identical to the base model, so macro-F1 is
unchanged and this remains a relevance-only change in the exp-014..024 sense
(held-out transfer of such changes was 1:1 in exp-024).

``eps`` is the graceful-fallback knob: when occlusion barely moves confidence
(all drops ~0) the gate tends to uniform and relevance falls back to the
cam x |x| (x optional static exp-023 gate) baseline ordering.
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
    evaluate_with_optional_band,
    flatten_metrics,
    proxy_band_configs,
)
from export_temperature_variants import (
    exp023_gate_matrix,
    install_direct_channel_gate,
    load_base_model,
)
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH
from gearxai_workspace.evaluate import sample_validation, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet

FEATURE_LENGTH = WINDOW_LENGTH // 8  # three /2 pools: 100 -> 50 -> 25 -> 12


@dataclass(frozen=True)
class Variant:
    name: str
    alpha: float
    eps: float
    static_gate: bool  # also apply the exp-023 class-conditioned motor gate


class OcclusionGateModel(nn.Module):
    """Frozen GearXAINet + in-graph per-sample channel-occlusion relevance gate."""

    def __init__(self, base: GearXAINet, *, alpha: float, eps: float):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)
        self.eps = float(eps)
        masks = torch.ones(NUM_CHANNELS, 1, NUM_CHANNELS, 1)
        for channel in range(NUM_CHANNELS):
            masks[channel, 0, channel, 0] = 0.0
        self.register_buffer("masks", masks)  # [8, 1, 8, 1]

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [9, N, 8, 100]: original plus the eight single-channel-zeroed copies,
        # folded into the batch axis so the conv stack appears once in the graph.
        stacked = torch.cat([windows.unsqueeze(0), windows.unsqueeze(0) * self.masks], dim=0)
        flat = stacked.reshape(-1, NUM_CHANNELS, WINDOW_LENGTH)  # [9N, 8, 100]
        feat_all = self.base._features(flat)
        logits_all = self.base._logits_from_features(feat_all)
        probs_all = torch.softmax(logits_all, dim=1)

        probs_r = probs_all.reshape(9, -1, NUM_CLASSES)  # [9, N, 9]
        probs = probs_r[0]  # [N, 9] — identical to the base model's output
        occ_probs = probs_r[1:]  # [8, N, 9]

        # Soft predicted-class confidence: sum_k p_k * q_k with q the base
        # distribution. At ~0.99 validation confidence this matches the hard
        # argmax probe while avoiding ArgMax/OneHot/Gather ops.
        p_base = (probs * probs).sum(dim=1)  # [N]
        p_occ = (occ_probs * probs.unsqueeze(0)).sum(dim=2)  # [8, N]
        drop = torch.relu(p_base.unsqueeze(0) - p_occ)  # [8, N]
        gate = (drop + self.eps).pow(self.alpha).transpose(0, 1)  # [N, 8]

        feat = feat_all.reshape(9, -1, self.base.feat_channels, FEATURE_LENGTH)[0]
        relevance = self.base._relevance_from(feat, probs, windows)
        return probs, relevance * gate.unsqueeze(2)


def default_variants() -> list[Variant]:
    return [
        Variant("identity_exp023", alpha=0.0, eps=1.0, static_gate=True),  # control == final3
        Variant("occ_a1_eps0p05", alpha=1.0, eps=0.05, static_gate=False),
        Variant("occ_a1_eps0p2", alpha=1.0, eps=0.2, static_gate=False),
        Variant("occ_a0p5_eps0p05", alpha=0.5, eps=0.05, static_gate=False),
        Variant("occ_a2_eps0p05", alpha=2.0, eps=0.05, static_gate=False),
        Variant("occ_a1_eps0p05_exp023", alpha=1.0, eps=0.05, static_gate=True),
        Variant("occ_a1_eps0p2_exp023", alpha=1.0, eps=0.2, static_gate=True),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/occ_exp027"))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27027)
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

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "eval_n": int(len(labels)),
        "seed": args.seed,
        "variants": {},
    }
    for variant in selected:
        base = load_base_model(args.checkpoint)
        if variant.static_gate:
            install_direct_channel_gate(base, exp023_gate_matrix())
        if variant.alpha == 0.0:
            model: nn.Module = base  # control: plain (optionally gated) model
        else:
            model = OcclusionGateModel(base, alpha=variant.alpha, eps=variant.eps)

        variant_dir = args.out_dir / variant.name
        onnx_path = variant_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_with_optional_band(onnx_path, windows, labels, batch_size=args.batch_size)
        metrics = flatten_metrics(report, n=len(labels))
        metrics["deletion_curve"] = report["faithfulness"]["deletion_curve"]
        metrics["insertion_curve"] = report["faithfulness"]["insertion_curve"]

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
            "variant": variant.__dict__,
            "onnx_path": str(onnx_path),
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
