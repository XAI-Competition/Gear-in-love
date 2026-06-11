"""End-to-end GearXAI evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import psutil

from gearxai_devkit.data import load_split, load_stats
from gearxai_devkit.metrics import (
    combine_scores,
    confusion_matrix,
    deletion_insertion_auc,
    load_band_config,
    macro_f1_score,
    mechanical_score,
    simplicity_score,
)
from gearxai_devkit.runtime import run_submission, validate_submission


def evaluate_submission(
    *,
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "dev",
    band_config_path: str | Path | None = None,
    batch_size: int = 256,
    output_path: str | Path | None = None,
) -> dict:
    """Run validation, prediction, and official scoring for one submission."""

    split_data = load_split(data_dir, split)
    validation = validate_submission(model_path, split_data.windows[: min(8, len(split_data.windows))])
    runtime = run_submission(model_path, split_data.windows, batch_size=batch_size)

    y_true = split_data.labels.astype(np.int64)
    y_pred = runtime.probabilities.argmax(axis=1).astype(np.int64)
    macro_f1 = macro_f1_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return run_submission(model_path, batch, batch_size=batch_size).probabilities

    try:
        stats = load_stats(data_dir)
        channel_mean = np.asarray(stats.get("standardized_channel_mean"), dtype=np.float32)
    except Exception:
        channel_mean = split_data.windows.mean(axis=(0, 2)).astype(np.float32)

    faith = deletion_insertion_auc(
        predict_fn,
        split_data.windows,
        runtime.relevance,
        y_pred,
        channel_mean=channel_mean,
    )

    if band_config_path is not None:
        mech = mechanical_score(
            split_data.windows,
            runtime.relevance,
            y_true,
            load_band_config(band_config_path),
        )
        mech_value = mech["mechanical_score"]
    else:
        mech = {
            "mechanical_score": None,
            "expected_band_mass": None,
            "noise_stability": None,
            "note": "official mechanical score requires a private band config",
        }
        mech_value = None

    simp = simplicity_score(model_path)
    combined = combine_scores(
        macro_f1=macro_f1,
        faith_score=faith["faith_score"],
        mechanical=mech_value,
        simplicity=simp["simplicity_score"],
    )

    process = psutil.Process()
    result = {
        "split": split,
        "samples": int(len(y_true)),
        "validation": validation,
        "runtime": {
            "load_seconds": runtime.load_seconds,
            "inference_seconds": runtime.inference_seconds,
            "output_names": list(runtime.output_names),
            "rss_mb": process.memory_info().rss / (1024 * 1024),
        },
        "classification": {
            "macro_f1": macro_f1,
            "confusion_matrix": matrix.tolist(),
        },
        "faithfulness": faith,
        "mechanical": mech,
        "simplicity": simp,
        "score": combined,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    return result
