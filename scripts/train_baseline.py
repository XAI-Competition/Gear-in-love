"""Train the GearXAI baseline, export it to ONNX, and self-check the interface.

Example (CPU, a few minutes on a balanced subset):

    uv run --no-sync python scripts/train_baseline.py `
        --train-per-class 8000 --val-per-class 2000 --epochs 12 `
        --out runs/baseline/model.onnx

Then package for the leaderboard with the devkit:

    uv run gearxai package --model runs/baseline/model.onnx `
        --data-dir data/prepared --split validation `
        --out runs/baseline/submission.zip
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from gearxai_workspace.data import load_split
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import count_parameters
from gearxai_workspace.train import TrainConfig, train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + export the GearXAI baseline.")
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--out", default="runs/baseline/model.onnx", type=Path)
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=8000,
        help="balanced subsample size per class (-1 uses all rows)",
    )
    parser.add_argument("--val-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="torch CPU threads (default: leave to torch)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="training device; 'auto' uses cuda when available (default: auto)",
    )
    parser.add_argument(
        "--relevance-weight",
        type=float,
        default=0.0,
        help=(
            "weight of the channel-prior relevance regularizer (0 disables it; "
            "exp-002d found it net-negative, so it is off by default)"
        ),
    )
    parser.add_argument(
        "--occlusion-weight",
        type=float,
        default=0.0,
        help=(
            "weight of the occlusion-distillation loss that trains the channel "
            "gate to predict causal channel importance (exp-003b; 0 disables it)"
        ),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help=(
            "std of additive Gaussian input-noise augmentation (exp-007; 0.1 is "
            "the faithfulness sweet spot, 0 disables it)"
        ),
    )
    parser.add_argument(
        "--time-mask-frac",
        type=float,
        default=0.15,
        help=(
            "fraction of time steps randomly zeroed per window (exp-008; ~0.15 "
            "further raises faithfulness, 0 disables it)"
        ),
    )
    parser.add_argument(
        "--insertion-frac",
        type=float,
        default=0.0,
        help=(
            "exp-029: fraction of each batch replaced by top-q-relevance-cells-only "
            "inputs (insertion-curriculum; 0 disables)"
        ),
    )
    parser.add_argument("--insertion-q-min", type=float, default=0.05)
    parser.add_argument("--insertion-q-max", type=float, default=0.5)
    parser.add_argument(
        "--deletion-entropy-weight",
        type=float,
        default=0.0,
        help=(
            "exp-029: weight of the negative-entropy loss on top-d-relevance-deleted "
            "inputs (deletion anti-objective; 0 disables)"
        ),
    )
    parser.add_argument("--deletion-d-min", type=float, default=0.1)
    parser.add_argument("--deletion-d-max", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        train_per_class=None if args.train_per_class < 0 else args.train_per_class,
        val_per_class=None if args.val_per_class < 0 else args.val_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_threads=args.num_threads,
        device=args.device,
        relevance_weight=args.relevance_weight,
        occlusion_weight=args.occlusion_weight,
        noise_std=args.noise_std,
        time_mask_frac=args.time_mask_frac,
        insertion_frac=args.insertion_frac,
        insertion_q_min=args.insertion_q_min,
        insertion_q_max=args.insertion_q_max,
        deletion_entropy_weight=args.deletion_entropy_weight,
        deletion_d_min=args.deletion_d_min,
        deletion_d_max=args.deletion_d_max,
    )

    result = train_baseline(config)
    model = result["model"]

    # Save torch weights so the relevance head can be retuned and re-exported
    # later without retraining the classifier.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out.parent / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    onnx_path = export_onnx(model, args.out)
    print(f"\nExported ONNX -> {onnx_path}")

    val_w, _ = load_split(args.data_dir, "validation")
    check = self_check(onnx_path, val_w[:8], torch_model=model)
    print("Self-check:", json.dumps(check, indent=2))

    summary = {
        "onnx_path": str(onnx_path),
        "parameters": count_parameters(model),
        "best_val_macro_f1_subset": result["best_val_macro_f1"],
        "device": result.get("device"),
        "history": result["history"],
        "config": {
            "train_per_class": config.train_per_class,
            "val_per_class": config.val_per_class,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "seed": config.seed,
            "device": config.device,
            "relevance_weight": config.relevance_weight,
            "occlusion_weight": config.occlusion_weight,
            "noise_std": config.noise_std,
            "time_mask_frac": config.time_mask_frac,
            "insertion_frac": config.insertion_frac,
            "insertion_q": [config.insertion_q_min, config.insertion_q_max],
            "deletion_entropy_weight": config.deletion_entropy_weight,
            "deletion_d": [config.deletion_d_min, config.deletion_d_max],
            "channel_attention": config.model.channel_attention,
        },
    }
    summary_path = onnx_path.parent / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")

    print(
        "\nNext: package for the leaderboard:\n"
        f"  uv run gearxai package --model {onnx_path} "
        f"--data-dir {args.data_dir} --split validation "
        f"--out {onnx_path.parent / 'submission.zip'}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    raise SystemExit(main())
