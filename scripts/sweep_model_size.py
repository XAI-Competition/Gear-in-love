r"""Sweep classifier width/depth to trade accuracy headroom for simplicity.

We clear the 0.80 macro-F1 gate by a huge margin (~0.989), and simplicity is 20%
of the explainability score with ``operator_count`` and ``parameter_count`` as
its only levers (devkit ``simplicity_score``). This sweep retrains progressively
smaller models and measures the real devkit ``macro_f1`` / ``faith`` /
``simplicity`` for each, so we can pick the Pareto point that maximises the
locally-visible objective ``0.40*faith + 0.20*simplicity`` while keeping
macro-F1 well above the gate.

The relevance head is left as the |x|-only baseline (identity channel gate): the
channel gate is an orthogonal, already-validated ~+0.02 faithfulness lever
(progress.md exp-014..024) that re-applies to whichever size wins, so excluding
it here isolates the size effect and keeps the sweep fast.

Run::

    $env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
    uv run --no-sync python scripts\sweep_model_size.py `
      --configs 32,64,128 24,48,96 16,32,64 16,32 `
      --out-dir runs\size_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gearxai_workspace.evaluate import evaluate_onnx
from gearxai_workspace.export import export_onnx
from gearxai_workspace.model import ModelConfig, count_parameters
from gearxai_workspace.train import TrainConfig, train_baseline

# Kernel sizes by depth: wide receptive field early, narrow late.
BASE_KERNELS = (7, 5, 3, 3, 3)


def parse_widths(spec: str) -> tuple[int, ...]:
    return tuple(int(x) for x in spec.replace(":", ",").split(",") if x)


def safe_tag(widths: tuple[int, ...]) -> str:
    return "w" + "_".join(str(w) for w in widths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["32,64,128", "24,48,96", "16,32,64", "16,32"],
        help="space-separated width specs, each comma-separated (e.g. 16,32,64)",
    )
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/size_sweep"))
    parser.add_argument("--train-per-class", type=int, default=8000)
    parser.add_argument("--val-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-n", type=int, default=6000, help="devkit eval subsample size")
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--time-mask-frac", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in args.configs:
        widths = parse_widths(spec)
        kernels = BASE_KERNELS[: len(widths)]
        tag = safe_tag(widths)
        run_dir = args.out_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        config = TrainConfig(
            data_dir=args.data_dir,
            train_per_class=None if args.train_per_class < 0 else args.train_per_class,
            val_per_class=None if args.val_per_class < 0 else args.val_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            noise_std=args.noise_std,
            time_mask_frac=args.time_mask_frac,
            model=ModelConfig(widths=widths, kernel_sizes=kernels),
        )
        result = train_baseline(config)
        model = result["model"]
        params = count_parameters(model)
        torch.save(model.state_dict(), run_dir / "model.pt")

        onnx_path = run_dir / "model.onnx"
        export_onnx(model, onnx_path)
        metrics = evaluate_onnx(
            onnx_path, data_dir=args.data_dir, n=args.eval_n, seed=args.eval_seed
        )
        objective = 0.40 * metrics["faith"] + 0.20 * metrics["simplicity"]

        row = {
            "widths": widths,
            "kernels": kernels,
            "torch_params": int(params),
            "onnx_params": metrics["parameter_count"],
            "operator_count": metrics["operator_count"],
            "macro_f1": metrics["macro_f1"],
            "faith": metrics["faith"],
            "simplicity": metrics["simplicity"],
            "objective_local": objective,
            "onnx_path": str(onnx_path),
        }
        rows.append(row)
        (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"{tag:14s} params={metrics['parameter_count']:6d} ops={metrics['operator_count']:3d} "
            f"f1={metrics['macro_f1']:.4f} faith={metrics['faith']:.4f} "
            f"simp={metrics['simplicity']:.4f} obj={objective:.4f}"
        )

    rows.sort(key=lambda r: r["objective_local"], reverse=True)
    print("\nRanked by local objective (0.40*faith + 0.20*simplicity):")
    print(f"{'config':14s} {'f1':>7s} {'faith':>7s} {'simp':>7s} {'obj':>7s}")
    for r in rows:
        print(
            f"{safe_tag(r['widths']):14s} {r['macro_f1']:7.4f} {r['faith']:7.4f} "
            f"{r['simplicity']:7.4f} {r['objective_local']:7.4f}"
        )
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
