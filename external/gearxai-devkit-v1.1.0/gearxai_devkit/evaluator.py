"""End-to-end GearXAI evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
    mechanical_context_indices,
    mechanical_score,
    select_mechanical_contexts,
    simplicity_score,
)
from gearxai_devkit.runtime import run_submission, validate_submission


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _evaluator_commit() -> str:
    configured = os.environ.get("GEARXAI_EVALUATOR_COMMIT", "").strip()
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def deterministic_noisy_windows(windows: np.ndarray, *, seed: int = 42, noise_sigma: float = 0.01) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    rms = np.sqrt(np.mean(windows * windows, axis=(1, 2), keepdims=True))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=windows.shape).astype(np.float32)
    return windows + noise * (noise_sigma * np.maximum(rms, 1e-8))


def stratified_explainability_indices(
    labels: np.ndarray,
    max_samples: int = 4096,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Select a deterministic, approximately balanced explainability subset."""

    labels = np.asarray(labels).astype(np.int64)
    if max_samples <= 0 or max_samples >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    class_ids = np.unique(labels)
    base, remainder = divmod(max_samples, len(class_ids))
    selected = []
    for position, class_id in enumerate(class_ids):
        available = np.flatnonzero(labels == class_id)
        count = min(len(available), base + (position < remainder))
        selected.extend(rng.choice(available, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def evaluate_submission(
    *,
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "dev",
    band_config_path: str | Path | None = None,
    batch_size: int = 256,
    explainability_samples: int = 4096,
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
    explainability_indices = stratified_explainability_indices(y_true, explainability_samples)
    explainability_windows = split_data.windows[explainability_indices]
    explainability_relevance = runtime.relevance[explainability_indices]

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return run_submission(model_path, batch, batch_size=batch_size).probabilities

    try:
        stats = load_stats(data_dir)
        channel_mean = np.asarray(stats.get("standardized_channel_mean"), dtype=np.float32)
    except Exception:
        channel_mean = split_data.windows.mean(axis=(0, 2)).astype(np.float32)

    faith = deletion_insertion_auc(
        predict_fn,
        explainability_windows,
        explainability_relevance,
        y_pred[explainability_indices],
        channel_mean=channel_mean,
    )

    if band_config_path is not None:
        band_config = load_band_config(band_config_path)
        contexts = select_mechanical_contexts(
            y_true,
            split_data.metadata,
            context_length=int(band_config.get("context_length", 512)),
            contexts_per_group=int(band_config.get("contexts_per_group", 2)),
        )
        noisy_indices = mechanical_context_indices(contexts)
        noisy_runtime = run_submission(
            model_path,
            deterministic_noisy_windows(split_data.windows[noisy_indices]),
            batch_size=batch_size,
        )
        mech = mechanical_score(
            split_data.windows,
            runtime.relevance,
            y_true,
            split_data.metadata,
            band_config,
            contexts=contexts,
            noisy_indices=noisy_indices,
            noisy_relevance=noisy_runtime.relevance,
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
        "explainability_samples": int(len(explainability_indices)),
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
        "provenance": {
            "evaluator_commit": _evaluator_commit(),
            "band_config_sha256": (
                _sha256_files([Path(band_config_path)]) if band_config_path is not None else None
            ),
            "dataset_sha256": _sha256_files(
                [
                    Path(data_dir) / f"{split}_windows.npy",
                    Path(data_dir) / f"{split}_labels.npy",
                    Path(data_dir) / f"{split}_metadata.jsonl",
                ]
            ),
            "metric_version": mech.get("metric_version"),
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    return result
