"""Sweep output-softmax temperature on the frozen final2 + exp-023-gate model.

The devkit's deletion/insertion curves are built from the *predicted-class
softmax probability* (``deleted_probs[rows, class_ids]`` with
``class_ids = y_pred``), and ``validate_submission`` only requires probability
rows to be nonnegative and sum to 1. Multiplying logits by a temperature
``T > 1`` therefore leaves the argmax — and macro-F1 — bit-for-bit unchanged
while sharpening every curve point toward the 0/1 indicator "is the argmax
still the original prediction". On runs/final3 the insertion curve's last six
points (0.74..0.946) each sit 5-25% below that indicator, so sharpening should
lift insertion AUC at a tiny deletion-start cost (0.946 -> 1.0 on point 0).

Two wiring modes per temperature:

* ``out``    — only the ``probabilities`` output is sharpened; the relevance
  head keeps conditioning on the T=1 softmax, so the relevance map (and hence
  the proxy mechanical scores) is mathematically identical to the base model.
  Costs +2 ops (Mul + second Softmax).
* ``shared`` — the sharpened probabilities also condition the CAM and channel
  gate, making the class mixing more one-hot. Relevance changes; costs +1 op.

The classifier checkpoint and the exp-023 focus gate (current best submission,
runs/final3) stay frozen throughout — this is a relevance/probability-only
sweep in the spirit of exp-014..023.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import GearXAINet, ModelConfig, build_model

CHANNELS = ("motor", "rgb_y", "rgb_x", "rgb_z", "torque", "pgb_y", "pgb_x", "pgb_z")
CLASSES = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")
SOFTPLUS_INV_1 = 0.5413248538970947

# exp-023 winner baked into runs/final3: lowfreq_focus b1.2 / c3.4 / o4.2 / other 0.8.
FOCUS_GATE = {"BWF": 1.2, "CWF": 3.4, "ORF": 4.2}
FOCUS_OTHER = 0.8


def exp023_gate_matrix() -> np.ndarray:
    gates = np.ones((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
    for class_name, motor in FOCUS_GATE.items():
        row = CLASSES.index(class_name)
        gates[row, :] = FOCUS_OTHER
        gates[row, CHANNELS.index("motor")] = motor
    return gates


def softplus_inverse(values: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(np.asarray(values, dtype=np.float32))).astype(np.float32)


def install_direct_channel_gate(model: nn.Module, gates: np.ndarray) -> None:
    """Bake class-conditioned gates into the model's channel_gate (exp-015 direct mode)."""

    if getattr(model, "channel_gate", None) is None:
        raise ValueError("Requires a model built with channel_attention=True.")
    preactivation = softplus_inverse(gates) - SOFTPLUS_INV_1  # [9, 8]
    with torch.no_grad():
        model.channel_gate.weight.copy_(torch.from_numpy(preactivation.T))
        model.channel_gate.bias.zero_()


class TemperatureModel(nn.Module):
    """Recompute the GearXAINet forward with a sharpened probability output."""

    def __init__(self, base: GearXAINet, temperature: float, mode: str):
        super().__init__()
        if mode not in ("out", "shared"):
            raise ValueError(f"mode must be 'out' or 'shared', got {mode!r}.")
        self.base = base
        self.temperature = float(temperature)
        self.mode = mode

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.base._features(windows)
        logits = self.base._logits_from_features(feat)
        probs_out = torch.softmax(logits * self.temperature, dim=1)
        if self.mode == "shared":
            conditioning = probs_out
        else:
            conditioning = torch.softmax(logits, dim=1)
        relevance = self.base._relevance_from(feat, conditioning, windows)
        return probs_out, relevance


def load_base_model(checkpoint: Path, widths: tuple[int, ...] = (32, 64, 128)) -> GearXAINet:
    from gearxai_workspace.export import fuse_gearxai_batchnorms

    model = build_model(ModelConfig(widths=widths, kernel_sizes=(7, 5, 3)))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    # Fold post-conv BatchNorms into the conv weights: numerically equivalent
    # (verified max|dP| ~1e-7) and -3 ONNX nodes / ~-700 params for simplicity.
    model = fuse_gearxai_batchnorms(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/temp_exp026"))
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    )
    parser.add_argument("--modes", nargs="+", choices=["out", "shared"], default=["out"])
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=26026)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_model(args.checkpoint)
    install_direct_channel_gate(base, exp023_gate_matrix())
    sample = np.random.default_rng(0).normal(size=(8, NUM_CHANNELS, 100)).astype(np.float32)

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "gate": "exp-023 lowfreq_focus b1.2/c3.4/o4.2/other0.8",
        "eval_n": args.eval_n,
        "seed": args.seed,
        "variants": {},
    }
    for mode in args.modes:
        for temperature in args.temperatures:
            if mode == "shared" and temperature == 1.0:
                continue  # identical to out/T=1 control
            name = f"{mode}_T{temperature:g}".replace(".", "p")
            variant_dir = args.out_dir / name
            onnx_path = variant_dir / "model.onnx"
            model = TemperatureModel(base, temperature, mode)
            export_onnx(model, onnx_path, sample=sample)
            check = self_check(onnx_path, sample, torch_model=model)
            report = evaluate_onnx(
                onnx_path,
                data_dir=args.data_dir,
                n=args.eval_n,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            raw = report.pop("raw")
            report["deletion_curve"] = raw["faithfulness"]["deletion_curve"]
            report["insertion_curve"] = raw["faithfulness"]["insertion_curve"]
            result = {
                "mode": mode,
                "temperature": temperature,
                "onnx_path": str(onnx_path),
                "self_check": check,
                "metrics": report,
            }
            (variant_dir / "metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            all_results["variants"][name] = result
            print(summary_line(name, report))

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
