"""Plot GearXAI signal and relevance examples for qualitative inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gearxai_workspace.data import load_split

FAULT_CODES = ["HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF"]
CHANNEL_NAMES = ["motor", "rgb_y", "rgb_x", "rgb_z", "torque", "pgb_y", "pgb_x", "pgb_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot relevance examples.")
    parser.add_argument("--model-a", required=True, type=Path)
    parser.add_argument("--model-b", type=Path)
    parser.add_argument("--label-a", default="final2")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--data-dir", default="data/prepared", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-correct", type=int, default=3)
    parser.add_argument("--num-errors", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def run_model(model_path: Path, windows: np.ndarray, batch_size: int):
    from gearxai_devkit.runtime import run_submission

    runtime = run_submission(model_path, windows, batch_size=batch_size)
    preds = runtime.probabilities.argmax(axis=1).astype(np.int64)
    conf = runtime.probabilities[np.arange(len(preds)), preds]
    return preds, conf, runtime.relevance


def normalize_rows(array: np.ndarray) -> np.ndarray:
    denom = np.percentile(np.abs(array), 99, axis=1, keepdims=True)
    return array / np.maximum(denom, 1e-6)


def plot_one(
    path: Path,
    window: np.ndarray,
    rel_a: np.ndarray,
    rel_b: np.ndarray | None,
    meta: dict[str, Any],
    true_label: int,
    pred_a: int,
    conf_a: float,
    pred_b: int | None,
    conf_b: float | None,
    label_a: str,
    label_b: str,
) -> None:
    rows = 3 if rel_b is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.6 * rows), constrained_layout=True)

    signal = normalize_rows(window)
    axes[0].imshow(signal, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title(
        f"{meta['condition_id']} | true={FAULT_CODES[true_label]} | "
        f"{label_a}={FAULT_CODES[pred_a]} ({conf_a:.3f})"
    )
    axes[0].set_yticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES)
    axes[0].set_ylabel("signal")

    vmax = float(np.percentile(rel_a, 99))
    if rel_b is not None:
        vmax = max(vmax, float(np.percentile(rel_b, 99)))
    vmax = max(vmax, 1e-8)

    axes[1].imshow(rel_a, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"relevance: {label_a}")
    axes[1].set_yticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES)
    axes[1].set_ylabel("rel")

    if rel_b is not None:
        axes[2].imshow(rel_b, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        title = f"relevance: {label_b}"
        if pred_b is not None and conf_b is not None:
            title += f" | pred={FAULT_CODES[pred_b]} ({conf_b:.3f})"
        axes[2].set_title(title)
        axes[2].set_yticks(range(len(CHANNEL_NAMES)), CHANNEL_NAMES)
        axes[2].set_ylabel("rel")

    for ax in axes:
        ax.set_xlabel("time step")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    windows, labels = load_split(args.data_dir, "validation")
    metadata = load_metadata(args.data_dir / "validation_metadata.jsonl")
    preds_a, conf_a, rel_a = run_model(args.model_a, windows, args.batch_size)
    preds_b = conf_b = rel_b = None
    if args.model_b is not None:
        preds_b, conf_b, rel_b = run_model(args.model_b, windows, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    correct_idx = np.flatnonzero(preds_a == labels)[: args.num_correct]
    error_idx = np.flatnonzero(preds_a != labels)[: args.num_errors]
    selected = [("correct", int(idx)) for idx in correct_idx] + [
        ("error", int(idx)) for idx in error_idx
    ]

    manifest = []
    for kind, idx in selected:
        filename = (
            f"{kind}_{idx}_true-{FAULT_CODES[int(labels[idx])]}_"
            f"pred-{FAULT_CODES[int(preds_a[idx])]}.png"
        )
        out_path = args.out_dir / filename
        plot_one(
            out_path,
            windows[idx],
            rel_a[idx],
            None if rel_b is None else rel_b[idx],
            metadata[idx],
            int(labels[idx]),
            int(preds_a[idx]),
            float(conf_a[idx]),
            None if preds_b is None else int(preds_b[idx]),
            None if conf_b is None else float(conf_b[idx]),
            args.label_a,
            args.label_b,
        )
        manifest.append(
            {
                "path": str(out_path),
                "kind": kind,
                "index": idx,
                "condition_id": metadata[idx]["condition_id"],
                "true": FAULT_CODES[int(labels[idx])],
                "pred_a": FAULT_CODES[int(preds_a[idx])],
                "conf_a": float(conf_a[idx]),
                "pred_b": None if preds_b is None else FAULT_CODES[int(preds_b[idx])],
                "conf_b": None if conf_b is None else float(conf_b[idx]),
            }
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} relevance plots -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
