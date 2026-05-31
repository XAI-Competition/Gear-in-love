"""Reusable devkit-metric evaluation for GearXAI experiments.

Wraps ``gearxai_devkit.evaluator.evaluate_submission`` on a (sub)sample of the
prepared validation split, so every experiment is scored with the *real*
faithfulness / macro-F1 / simplicity the leaderboard uses — not a hand-rolled
proxy. The mechanical score stays ``null`` locally (private band config), as
established in progress.md exp-002a..d.

Typical use after training/export::

    from gearxai_workspace.evaluate import evaluate_onnx
    report = evaluate_onnx("runs/foo/model.onnx", n=4000)
    print(report["faith"], report["macro_f1"], report["simplicity"])
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from gearxai_workspace.data import NUM_CHANNELS, load_split


def _write_eval_dir(out_dir: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    """Write a minimal evaluator-ready prepared dir (windows, labels, stats)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "validation_windows.npy", np.ascontiguousarray(windows, dtype=np.float32))
    np.save(out_dir / "validation_labels.npy", np.ascontiguousarray(labels, dtype=np.int64))
    # Zero baseline: the released windows are pre-standardized (see exp-002a).
    stats = {"format": "[N, 8, 100]", "standardized_channel_mean": [0.0] * NUM_CHANNELS}
    (out_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def sample_validation(
    data_dir: str | Path,
    n: int | None,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a reproducible validation subsample (``None`` uses the full split)."""

    windows, labels = load_split(data_dir, "validation")
    if n is None or n >= len(labels):
        return np.ascontiguousarray(windows), labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(labels), size=n, replace=False))
    return np.ascontiguousarray(windows[idx]), labels[idx]


def evaluate_onnx(
    onnx_path: str | Path,
    *,
    data_dir: str | Path = "data/prepared",
    n: int | None = 4000,
    seed: int = 0,
    batch_size: int = 256,
) -> dict:
    """Score an ONNX model with the devkit on a validation subsample.

    Returns a flat dict of the headline metrics plus the full devkit report
    under ``"raw"``. ``n=None`` evaluates the full validation split (slow: the
    faithfulness deletion/insertion sweep re-runs the model ~22x).
    """

    from gearxai_devkit.evaluator import evaluate_submission

    windows, labels = sample_validation(data_dir, n, seed=seed)
    with tempfile.TemporaryDirectory(prefix="gearxai_eval_") as tmp:
        eval_dir = Path(tmp)
        _write_eval_dir(eval_dir, windows, labels)
        report = evaluate_submission(
            model_path=str(onnx_path),
            data_dir=str(eval_dir),
            split="validation",
            batch_size=batch_size,
        )

    return {
        "n": int(len(labels)),
        "macro_f1": report["classification"]["macro_f1"],
        "faith": report["faithfulness"]["faith_score"],
        "deletion_auc": report["faithfulness"]["deletion_auc"],
        "insertion_auc": report["faithfulness"]["insertion_auc"],
        "simplicity": report["simplicity"]["simplicity_score"],
        "operator_count": report["simplicity"]["operator_count"],
        "parameter_count": report["simplicity"]["parameter_count"],
        "eligible": report["score"]["eligible"],
        "raw": report,
    }


def summary_line(tag: str, metrics: dict) -> str:
    """One-line human summary of an evaluation result."""

    return (
        f"{tag}: n={metrics['n']} macro_f1={metrics['macro_f1']:.4f} "
        f"faith={metrics['faith']:.4f} (del={metrics['deletion_auc']:.3f} "
        f"ins={metrics['insertion_auc']:.3f}) simplicity={metrics['simplicity']:.4f} "
        f"ops={metrics['operator_count']}"
    )
