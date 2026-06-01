"""Export and evaluate relevance-only channel-gate variants.

The classifier stays identical to a trained GearXAINet checkpoint. Each variant
only multiplies the relevance map by a class-conditioned channel gate:

    relevance = base_relevance * (probabilities @ gate_matrix)[:, :, None]

This targets the only public mechanical-alignment lever that is controllable on
100-sample windows: per-channel relevance mass. The script evaluates every
candidate with the real public devkit faithfulness/simplicity metrics and with
several explicit proxy band configs. Proxy mechanical scores are not official;
they are used only for Pareto screening.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, load_split
from gearxai_workspace.evaluate import _write_eval_dir, sample_validation, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import ModelConfig, build_model, count_parameters

CHANNELS = ("motor", "rgb_y", "rgb_x", "rgb_z", "torque", "pgb_y", "pgb_x", "pgb_z")
CLASSES = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    gates: np.ndarray


class RelevanceGateWrapper(nn.Module):
    """Wrap a trained base model and alter only its relevance output."""

    def __init__(self, base: nn.Module, gates: np.ndarray):
        super().__init__()
        if gates.shape != (NUM_CLASSES, NUM_CHANNELS):
            raise ValueError(f"Expected gate matrix [9, 8], got {gates.shape}.")
        self.base = base
        self.register_buffer("gates", torch.as_tensor(gates, dtype=torch.float32))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities, relevance = self.base(windows)
        channel_gate = probabilities @ self.gates
        return probabilities, relevance * channel_gate.unsqueeze(2)


def ones() -> np.ndarray:
    return np.ones((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)


def set_gate(gates: np.ndarray, class_name: str, weights: dict[str, float]) -> None:
    class_id = CLASSES.index(class_name)
    for channel_name, weight in weights.items():
        gates[class_id, CHANNELS.index(channel_name)] = weight


def presets() -> dict[str, Preset]:
    """Hand-built relevance-only gates for the first mechanical sweep."""

    identity = ones()

    mild_torque = ones()
    for class_name in ("CTF", "MTF", "RCF", "SWF", "BWF", "IRF", "ORF"):
        set_gate(mild_torque, class_name, {"torque": 1.35})
    set_gate(mild_torque, "CWF", {"motor": 1.2, "pgb_y": 1.2})

    mixed_pgb_torque = ones()
    for class_name in ("CTF", "MTF"):
        set_gate(
            mixed_pgb_torque,
            class_name,
            {"torque": 1.2, "pgb_y": 1.3, "pgb_x": 1.3, "pgb_z": 1.3},
        )
    for class_name in ("RCF", "SWF", "IRF"):
        set_gate(mixed_pgb_torque, class_name, {"torque": 1.35, "pgb_y": 1.2})
    for class_name in ("BWF", "CWF", "ORF"):
        set_gate(mixed_pgb_torque, class_name, {"motor": 1.35, "pgb_x": 1.15})

    lowfreq_motor = ones()
    for class_name in ("BWF", "CWF", "ORF"):
        set_gate(lowfreq_motor, class_name, {"motor": 1.8})
    for class_name in ("CTF", "MTF", "RCF", "SWF", "IRF"):
        set_gate(lowfreq_motor, class_name, {"torque": 1.15})

    sharp_proxy = ones()
    set_gate(sharp_proxy, "CTF", {"torque": 2.0})
    set_gate(sharp_proxy, "MTF", {"rgb_z": 2.0, "pgb_y": 1.4, "pgb_x": 1.4})
    set_gate(sharp_proxy, "RCF", {"torque": 2.0})
    set_gate(sharp_proxy, "SWF", {"torque": 2.0})
    for class_name in ("BWF", "CWF", "ORF"):
        set_gate(sharp_proxy, class_name, {"motor": 2.2})
    set_gate(sharp_proxy, "IRF", {"torque": 2.0, "pgb_y": 1.3})

    torque_only_faults = ones()
    for class_name in CLASSES[1:]:
        set_gate(torque_only_faults, class_name, {name: 0.35 for name in CHANNELS})
        set_gate(torque_only_faults, class_name, {"torque": 2.8})

    return {
        "identity": Preset("identity", "base final2 relevance, exported through wrapper", identity),
        "mild_torque": Preset(
            "mild_torque",
            "small torque bias for most fault classes",
            mild_torque,
        ),
        "mixed_pgb_torque": Preset(
            "mixed_pgb_torque",
            "moderate gear-channel and torque bias, low-frequency classes to motor",
            mixed_pgb_torque,
        ),
        "lowfreq_motor": Preset(
            "lowfreq_motor",
            "motor bias only for low-frequency proxy classes; mild torque elsewhere",
            lowfreq_motor,
        ),
        "sharp_proxy": Preset(
            "sharp_proxy",
            "aggressive proxy best-channel bets from the mechanical audit",
            sharp_proxy,
        ),
        "torque_only_faults": Preset(
            "torque_only_faults",
            "stress test: concentrate all fault classes on torque",
            torque_only_faults,
        ),
    }


def proxy_band_configs() -> dict[str, dict[str, Any]]:
    """Explicit local proxy bands for sensitivity analysis, not official scoring."""

    def band(center: float, radius_bins: int = 1) -> list[list[float]]:
        radius = 51.2 * radius_bins
        return [[max(0.0, center - radius), center + radius]]

    # Centers from the mechanical audit's relevance-induced nonzero peaks.
    audit_centers = {
        "1": band(1024.0),
        "2": band(1689.6),
        "3": band(512.0),
        "4": band(1024.0),
        "5": band(51.2),
        "6": band(51.2),
        "7": band(921.6),
        "8": band(51.2),
    }
    broad_audit = {class_id: [[lo, hi + 51.2]] for class_id, [(lo, hi)] in audit_centers.items()}
    low_frequency = {str(class_id): band(51.2) for class_id in range(1, NUM_CLASSES)}
    high_mid = {
        "1": band(1024.0),
        "2": band(1536.0, radius_bins=2),
        "3": band(512.0),
        "4": band(1024.0),
        "5": band(1024.0),
        "6": band(51.2),
        "7": band(1024.0),
        "8": band(512.0),
    }
    return {
        "audit_peaks": {"classes": audit_centers},
        "audit_peaks_broad": {"classes": broad_audit},
        "low_frequency": {"classes": low_frequency},
        "high_mid": {"classes": high_mid},
    }


def load_base_model(checkpoint: Path) -> nn.Module:
    model = build_model(ModelConfig())
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate_with_optional_band(
    onnx_path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    band_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from gearxai_devkit.evaluator import evaluate_submission

    with tempfile.TemporaryDirectory(prefix="gearxai_gate_eval_") as tmp:
        tmp_path = Path(tmp)
        eval_dir = tmp_path / "data"
        _write_eval_dir(eval_dir, windows, labels)
        band_path = None
        if band_config is not None:
            band_path = tmp_path / "band_config.json"
            band_path.write_text(json.dumps(band_config), encoding="utf-8")
        return evaluate_submission(
            model_path=str(onnx_path),
            data_dir=str(eval_dir),
            split="validation",
            batch_size=batch_size,
            band_config_path=band_path,
        )


def flatten_metrics(report: dict[str, Any], *, n: int) -> dict[str, Any]:
    return {
        "n": n,
        "macro_f1": report["classification"]["macro_f1"],
        "faith": report["faithfulness"]["faith_score"],
        "deletion_auc": report["faithfulness"]["deletion_auc"],
        "insertion_auc": report["faithfulness"]["insertion_auc"],
        "simplicity": report["simplicity"]["simplicity_score"],
        "operator_count": report["simplicity"]["operator_count"],
        "parameter_count": report["simplicity"]["parameter_count"],
        "eligible": report["score"]["eligible"],
        "mechanical": report["mechanical"]["mechanical_score"],
        "expected_band_mass": report["mechanical"]["expected_band_mass"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mech_gate_exp014"))
    parser.add_argument("--presets", nargs="+", default=["all"])
    parser.add_argument("--eval-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=14014)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = presets()
    selected = list(available) if args.presets == ["all"] else args.presets
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Available: {sorted(available)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows, labels = sample_validation(args.data_dir, args.eval_n, seed=args.seed)
    full_windows, _ = load_split(args.data_dir, "validation")
    sample_for_check = np.array(full_windows[:8], dtype=np.float32, copy=True)
    base = load_base_model(args.checkpoint)

    proxy_configs = proxy_band_configs()
    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "eval_n": int(len(labels)),
        "seed": args.seed,
        "presets": {},
    }
    for name in selected:
        preset = available[name]
        variant_dir = args.out_dir / name
        onnx_path = variant_dir / "model.onnx"
        wrapper = RelevanceGateWrapper(base, preset.gates)
        export_onnx(wrapper, onnx_path, sample=sample_for_check)
        check = self_check(onnx_path, sample_for_check)
        report = evaluate_with_optional_band(
            onnx_path,
            windows,
            labels,
            batch_size=args.batch_size,
        )
        metrics = flatten_metrics(report, n=len(labels))
        proxy_metrics: dict[str, Any] = {}
        for proxy_name, band_config in proxy_configs.items():
            proxy_report = evaluate_with_optional_band(
                onnx_path,
                windows,
                labels,
                batch_size=args.batch_size,
                band_config=band_config,
            )
            proxy_metrics[proxy_name] = flatten_metrics(proxy_report, n=len(labels))

        result = {
            "description": preset.description,
            "onnx_path": str(onnx_path),
            "parameters": count_parameters(wrapper),
            "gates": preset.gates.tolist(),
            "self_check": check,
            "public_metrics": metrics,
            "proxy_metrics": proxy_metrics,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results["presets"][name] = result
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
