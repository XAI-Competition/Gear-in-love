"""Re-export a trained checkpoint with a soft mechanical channel prior.

This does not retrain or change the classifier. It only changes the relevance
head at inference time by applying a mild class-to-channel weighting, so it is a
cheap way to probe hidden mechanical-alignment upside while monitoring the
locally measurable faithfulness cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gearxai_workspace.data import load_split
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check
from gearxai_workspace.model import ModelConfig, build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint with soft mechanical relevance weighting."
    )
    parser.add_argument("--checkpoint", default="runs/final2/model.pt", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument(
        "--prior-variant",
        default="balanced",
        choices=["balanced", "pgb_strong", "error_focus"],
        help="mechanical class-to-channel prior matrix variant",
    )
    parser.add_argument(
        "--disable-channel-attention",
        action="store_true",
        help=(
            "drop the zero-initialized channel gate when re-exporting; useful "
            "when only inference-time mechanical prior is needed"
        ),
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=4000,
        help="validation subsample size for devkit metrics; -1 uses full validation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ModelConfig(
        channel_attention=not args.disable_channel_attention,
        mechanical_prior_strength=args.strength,
        mechanical_prior_variant=args.prior_variant,
    )
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=not args.disable_channel_attention)
    model.eval()

    onnx_path = export_onnx(model, args.out)
    val_w, _ = load_split(args.data_dir, "validation")
    check = self_check(onnx_path, val_w[:8], torch_model=model)

    eval_n = None if args.eval_n < 0 else args.eval_n
    metrics = evaluate_onnx(
        onnx_path,
        data_dir=args.data_dir,
        n=eval_n,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    summary = {
        "onnx_path": str(onnx_path),
        "checkpoint": str(args.checkpoint),
        "mechanical_prior_strength": args.strength,
        "mechanical_prior_variant": args.prior_variant,
        "channel_attention": config.channel_attention,
        "parameters": count_parameters(model),
        "self_check": check,
        "metrics": metrics,
    }
    summary_path = onnx_path.parent / "mechanical_prior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported -> {onnx_path}")
    print(summary_line(f"mechanical_prior={args.strength:g}", metrics))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
