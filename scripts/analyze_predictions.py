"""Analyze GearXAI validation predictions by class and operating condition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gearxai_workspace.data import NUM_CLASSES, load_split
from gearxai_workspace.train import macro_f1

FAULT_CODES = ["HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze validation predictions.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data-dir", default="data/prepared", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def condition_group(condition_id: str) -> str:
    parts = condition_id.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"{parts[1]}Hz"
    if "Variable_speed" in condition_id:
        return "variable_speed"
    return condition_id


def f1_for_class(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> float:
    tp = int(np.sum((y_pred == class_id) & (y_true == class_id)))
    fp = int(np.sum((y_pred == class_id) & (y_true != class_id)))
    fn = int(np.sum((y_pred != class_id) & (y_true == class_id)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    return 0.0 if denom == 0 else float(2 * precision * recall / denom)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        matrix[true, pred] += 1
    return matrix


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    from gearxai_devkit.runtime import run_submission

    windows, labels = load_split(args.data_dir, args.split)
    metadata = load_metadata(args.data_dir / f"{args.split}_metadata.jsonl")
    runtime = run_submission(args.model, windows, batch_size=args.batch_size)
    preds = runtime.probabilities.argmax(axis=1).astype(np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, preds)
    overall = {
        "model": str(args.model),
        "split": args.split,
        "samples": int(len(labels)),
        "macro_f1": macro_f1(labels, preds),
        "accuracy": float(np.mean(labels == preds)),
        "errors": int(np.sum(labels != preds)),
        "confusion_matrix": matrix.tolist(),
    }

    class_rows = []
    for class_id, code in enumerate(FAULT_CODES):
        mask = labels == class_id
        class_rows.append(
            {
                "class_id": class_id,
                "fault_code": code,
                "samples": int(mask.sum()),
                "f1": f1_for_class(labels, preds, class_id),
                "accuracy_within_true_class": float(np.mean(preds[mask] == labels[mask])),
                "errors": int(np.sum(preds[mask] != labels[mask])),
            }
        )

    groups = np.array([condition_group(row["condition_id"]) for row in metadata])
    condition_ids = np.array([row["condition_id"] for row in metadata])
    group_rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_rows.append(
            {
                "group": group,
                "samples": int(mask.sum()),
                "macro_f1": macro_f1(labels[mask], preds[mask]),
                "accuracy": float(np.mean(labels[mask] == preds[mask])),
                "errors": int(np.sum(labels[mask] != preds[mask])),
            }
        )

    condition_rows = []
    for condition in sorted(set(condition_ids.tolist())):
        mask = condition_ids == condition
        condition_rows.append(
            {
                "condition_id": condition,
                "group": condition_group(condition),
                "samples": int(mask.sum()),
                "macro_f1": macro_f1(labels[mask], preds[mask]),
                "accuracy": float(np.mean(labels[mask] == preds[mask])),
                "errors": int(np.sum(labels[mask] != preds[mask])),
            }
        )

    error_pairs = Counter(
        (FAULT_CODES[int(t)], FAULT_CODES[int(p)])
        for t, p in zip(labels, preds, strict=True)
        if t != p
    )
    error_pair_rows = [
        {"true": true, "pred": pred, "count": count}
        for (true, pred), count in error_pairs.most_common()
    ]

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    np.savetxt(args.out_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    write_csv(
        args.out_dir / "per_class.csv",
        class_rows,
        ["class_id", "fault_code", "samples", "f1", "accuracy_within_true_class", "errors"],
    )
    write_csv(
        args.out_dir / "per_group.csv",
        group_rows,
        ["group", "samples", "macro_f1", "accuracy", "errors"],
    )
    write_csv(
        args.out_dir / "per_condition.csv",
        condition_rows,
        ["condition_id", "group", "samples", "macro_f1", "accuracy", "errors"],
    )
    write_csv(args.out_dir / "error_pairs.csv", error_pair_rows, ["true", "pred", "count"])

    print(
        f"{args.model}: samples={len(labels)} macro_f1={overall['macro_f1']:.4f} "
        f"accuracy={overall['accuracy']:.4f} errors={overall['errors']}"
    )
    print(f"Wrote analysis -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
