"""Scoring metrics for the GearXAI competition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from scipy.signal import stft

from gearxai_devkit.constants import (
    MIN_MACRO_F1,
    NUM_CHANNELS,
    NUM_CLASSES,
    SAMPLING_RATE_HZ,
    SCORE_WEIGHTS,
    WINDOW_LENGTH,
)
from gearxai_devkit.data import ensure_channels_first


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[true, pred] += 1
    return matrix


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    scores = []
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    for class_id in range(num_classes):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        denom = precision + recall
        scores.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return float(np.mean(scores))


def normalize_relevance(relevance: np.ndarray) -> np.ndarray:
    relevance = np.asarray(relevance, dtype=np.float32)
    if relevance.shape[1:] != (NUM_CHANNELS, WINDOW_LENGTH):
        raise ValueError(f"Expected relevance [N, 8, 100], got {relevance.shape}.")
    relevance = np.maximum(relevance, 0.0)
    total = relevance.sum(axis=(1, 2), keepdims=True)
    return relevance / np.maximum(total, 1e-12)


def topk_mask(relevance: np.ndarray, fraction: float) -> np.ndarray:
    flat = relevance.reshape(relevance.shape[0], -1)
    total_cells = flat.shape[1]
    k = int(round(fraction * total_cells))
    mask = np.zeros_like(flat, dtype=bool)
    if k <= 0:
        return mask.reshape(relevance.shape)
    order = np.argpartition(-flat, kth=min(k - 1, total_cells - 1), axis=1)[:, :k]
    rows = np.arange(flat.shape[0])[:, None]
    mask[rows, order] = True
    return mask.reshape(relevance.shape)


def deletion_insertion_auc(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    windows: np.ndarray,
    relevance: np.ndarray,
    class_ids: np.ndarray,
    *,
    channel_mean: np.ndarray | None = None,
    steps: int = 10,
) -> dict:
    """Compute deletion/insertion AUC on predicted-class confidence."""

    windows = ensure_channels_first(windows)
    relevance = normalize_relevance(relevance)
    if channel_mean is None:
        channel_mean = np.zeros((NUM_CHANNELS,), dtype=np.float32)
    baseline = np.asarray(channel_mean, dtype=np.float32).reshape(1, NUM_CHANNELS, 1)
    base_input = np.broadcast_to(baseline, windows.shape).astype(np.float32).copy()

    fractions = np.linspace(0.0, 1.0, steps + 1)
    deletion_conf = []
    insertion_conf = []
    rows = np.arange(len(windows))
    for fraction in fractions:
        mask = topk_mask(relevance, float(fraction))
        deleted = windows.copy()
        deleted[mask] = np.broadcast_to(baseline, windows.shape)[mask]
        inserted = base_input.copy()
        inserted[mask] = windows[mask]

        deleted_probs = predict_fn(deleted)
        inserted_probs = predict_fn(inserted)
        deletion_conf.append(deleted_probs[rows, class_ids])
        insertion_conf.append(inserted_probs[rows, class_ids])

    deletion_curve = np.stack(deletion_conf, axis=1).mean(axis=0)
    insertion_curve = np.stack(insertion_conf, axis=1).mean(axis=0)
    deletion_auc = float(np.trapz(deletion_curve, fractions))
    insertion_auc = float(np.trapz(insertion_curve, fractions))
    faith_score = float(np.clip((insertion_auc + (1.0 - deletion_auc)) / 2.0, 0.0, 1.0))
    return {
        "faith_score": faith_score,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_curve": deletion_curve.tolist(),
        "insertion_curve": insertion_curve.tolist(),
    }


def load_band_config(path: str | Path) -> dict:
    """Load private mechanical frequency bands without hard-coding them in the devkit."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "classes" not in config:
        raise ValueError("Band config must contain a 'classes' mapping.")
    return config


def frame_relevance(relevance_ch: np.ndarray, frames: int, hop_length: int) -> np.ndarray:
    pooled = np.zeros((frames,), dtype=np.float32)
    for frame_id in range(frames):
        start = frame_id * hop_length
        end = min(start + hop_length, WINDOW_LENGTH)
        if start >= WINDOW_LENGTH:
            break
        pooled[frame_id] = float(relevance_ch[start:end].sum())
    return pooled


def single_mechanical_alignment(
    window: np.ndarray,
    relevance: np.ndarray,
    class_id: int,
    band_config: dict,
) -> float:
    sampling_rate = float(band_config.get("sampling_rate_hz", SAMPLING_RATE_HZ))
    n_fft = int(band_config.get("n_fft", 256))
    hop_length = int(band_config.get("hop_length", 64))
    bands = band_config["classes"].get(str(int(class_id)), [])
    if not bands:
        return 0.0

    relevance = normalize_relevance(relevance[None])[0]
    freq_mass = None
    freqs_ref = None
    for ch in range(NUM_CHANNELS):
        freqs, _, zxx = stft(
            window[ch],
            fs=sampling_rate,
            nperseg=min(n_fft, WINDOW_LENGTH),
            noverlap=max(0, min(n_fft, WINDOW_LENGTH) - hop_length),
            boundary=None,
            padded=False,
        )
        energy = np.abs(zxx).astype(np.float32)
        if energy.ndim != 2 or energy.size == 0:
            continue
        frame_rel = frame_relevance(relevance[ch], energy.shape[1], hop_length)
        denom = energy.sum(axis=0, keepdims=True)
        distributed = (energy / np.maximum(denom, 1e-12)) * frame_rel[None, :]
        freq_mass = distributed if freq_mass is None else freq_mass + distributed
        freqs_ref = freqs

    if freq_mass is None or freqs_ref is None:
        return 0.0

    mask = np.zeros_like(freq_mass, dtype=bool)
    for low, high in bands:
        mask |= ((freqs_ref >= float(low)) & (freqs_ref <= float(high)))[:, None]
    total = float(freq_mass.sum())
    if total <= 1e-12:
        return 0.0
    return float(freq_mass[mask].sum() / total)


def mechanical_score(
    windows: np.ndarray,
    relevance: np.ndarray,
    y_true: np.ndarray,
    band_config: dict,
    *,
    seed: int = 42,
    noise_sigma: float = 0.01,
) -> dict:
    """Compute mechanical alignment plus small-noise stability."""

    windows = ensure_channels_first(windows)
    relevance = normalize_relevance(relevance)
    rng = np.random.default_rng(seed)
    alignments = []
    noisy_alignments = []
    for window, rel, class_id in zip(windows, relevance, y_true.astype(int)):
        score = single_mechanical_alignment(window, rel, int(class_id), band_config)
        rms = float(np.sqrt(np.mean(window * window)))
        noise = rng.normal(0.0, noise_sigma * max(rms, 1e-8), size=window.shape).astype(np.float32)
        noisy = single_mechanical_alignment(window + noise, rel, int(class_id), band_config)
        alignments.append(score)
        noisy_alignments.append(noisy)

    eas = float(np.mean(alignments)) if alignments else 0.0
    noisy_eas = float(np.mean(noisy_alignments)) if noisy_alignments else 0.0
    stability = float(np.clip(1.0 - abs(eas - noisy_eas), 0.0, 1.0))
    mech = float(np.clip(0.75 * eas + 0.25 * stability, 0.0, 1.0))
    return {
        "mechanical_score": mech,
        "expected_band_mass": eas,
        "noise_stability": stability,
        "noisy_expected_band_mass": noisy_eas,
    }


def simplicity_score(model_path: str | Path) -> dict:
    """Score model simplicity from ONNX size, initializer count, and operator count."""

    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    param_count = 0
    for initializer in model.graph.initializer:
        count = 1
        for dim in initializer.dims:
            count *= int(dim)
        param_count += count
    operator_count = len(model.graph.node)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    penalty = (param_count / 1_000_000.0) + (operator_count / 1000.0) + (size_mb / 50.0)
    score = float(np.clip(1.0 / (1.0 + penalty), 0.0, 1.0))
    return {
        "simplicity_score": score,
        "parameter_count": int(param_count),
        "operator_count": int(operator_count),
        "onnx_size_mb": float(size_mb),
    }


def combine_scores(
    *,
    macro_f1: float,
    faith_score: float,
    mechanical: float | None,
    simplicity: float,
) -> dict:
    eligible = bool(macro_f1 >= MIN_MACRO_F1)
    if not eligible:
        return {
            "eligible": False,
            "explainability_score": None,
            "reason": "macro-F1 below 0.80 gate",
        }
    if mechanical is None:
        return {
            "eligible": True,
            "explainability_score": None,
            "reason": "mechanical score unavailable; pass official band config",
        }
    explainability = (
        SCORE_WEIGHTS["faith"] * faith_score
        + SCORE_WEIGHTS["mechanical"] * mechanical
        + SCORE_WEIGHTS["simplicity"] * simplicity
    )
    return {
        "eligible": True,
        "explainability_score": float(explainability),
        "reason": "eligible",
    }
