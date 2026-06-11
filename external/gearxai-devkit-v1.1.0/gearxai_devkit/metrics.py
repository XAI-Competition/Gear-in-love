"""Scoring metrics for the GearXAI competition."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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

MECHANICAL_METRIC_VERSION = "mechanical_v2"
MECHANICAL_CONTEXT_LENGTH = 512
MECHANICAL_CONTEXTS_PER_GROUP = 2


@dataclass(frozen=True)
class MechanicalContext:
    condition_id: str
    class_id: int
    source_start: int
    indices: np.ndarray


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
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    deletion_auc = float(trapezoid(deletion_curve, fractions))
    insertion_auc = float(trapezoid(insertion_curve, fractions))
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


def select_mechanical_contexts(
    labels: np.ndarray,
    metadata: list[dict],
    *,
    context_length: int = MECHANICAL_CONTEXT_LENGTH,
    contexts_per_group: int = MECHANICAL_CONTEXTS_PER_GROUP,
) -> list[MechanicalContext]:
    """Select deterministic, non-overlapping contexts per condition and class."""

    if len(metadata) != len(labels):
        raise ValueError("Mechanical v2 requires one metadata row per hidden-test window.")
    required_windows = context_length - WINDOW_LENGTH + 1
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for index, (label, row) in enumerate(zip(np.asarray(labels).astype(int), metadata)):
        condition_id = str(row.get("condition_id", "")).strip()
        if not condition_id or "source_window_index" not in row:
            raise ValueError("Mechanical v2 metadata requires condition_id and source_window_index.")
        grouped.setdefault((condition_id, int(label)), []).append((int(row["source_window_index"]), index))

    contexts: list[MechanicalContext] = []
    for (condition_id, class_id), rows in sorted(grouped.items()):
        rows.sort()
        runs: list[list[tuple[int, int]]] = []
        run: list[tuple[int, int]] = []
        for item in rows:
            if run and item[0] != run[-1][0] + 1:
                runs.append(run)
                run = []
            run.append(item)
        if run:
            runs.append(run)

        candidates: list[MechanicalContext] = []
        for values in runs:
            if len(values) < required_windows:
                continue
            first = values[:required_windows]
            candidates.append(
                MechanicalContext(condition_id, class_id, first[0][0], np.asarray([item[1] for item in first]))
            )
            last = values[-required_windows:]
            if last[0][0] - first[0][0] >= context_length:
                candidates.append(
                    MechanicalContext(condition_id, class_id, last[0][0], np.asarray([item[1] for item in last]))
                )
        candidates.sort(key=lambda value: value.source_start)
        if len(candidates) > contexts_per_group:
            candidates = [candidates[0], candidates[-1]]
        contexts.extend(candidates[:contexts_per_group])
    if not contexts:
        raise ValueError("Mechanical v2 could not construct any contiguous hidden-test contexts.")
    return contexts


def mechanical_context_indices(contexts: list[MechanicalContext]) -> np.ndarray:
    return np.unique(np.concatenate([context.indices for context in contexts])).astype(np.int64)


def overlap_add_windows(windows: np.ndarray, *, context_length: int = MECHANICAL_CONTEXT_LENGTH) -> np.ndarray:
    """Reconstruct a context from stride-one windows by averaging overlaps."""

    windows = ensure_channels_first(windows)
    expected = context_length - WINDOW_LENGTH + 1
    if len(windows) != expected:
        raise ValueError(f"Expected {expected} stride-one windows for a {context_length}-sample context.")
    reconstructed = np.zeros((NUM_CHANNELS, context_length), dtype=np.float64)
    counts = np.zeros((context_length,), dtype=np.float64)
    for offset, window in enumerate(windows):
        reconstructed[:, offset : offset + WINDOW_LENGTH] += window
        counts[offset : offset + WINDOW_LENGTH] += 1.0
    return (reconstructed / counts[None, :]).astype(np.float32)


def parse_fixed_speed_hz(condition_id: str) -> float | None:
    match = re.match(r"^PGB_(\d+(?:\.\d+)?)_", condition_id)
    return float(match.group(1)) if match else None


def estimate_variable_speed_hz(signal: np.ndarray, band_config: dict) -> float:
    """Estimate local shaft speed from a calibrated motor-vibration spectral ridge."""

    estimator = band_config.get("speed_estimator", {})
    sampling_rate = float(band_config.get("sampling_rate_hz", SAMPLING_RATE_HZ))
    channel = int(estimator.get("channel_index", 0))
    min_hz = float(estimator.get("min_hz", 0.0))
    max_hz = float(estimator.get("max_hz", 40.0))
    harmonics = [float(value) for value in estimator.get("harmonic_candidates", [3, 1, 2, 5, 7])]
    values = np.asarray(signal[channel], dtype=np.float64)
    values = values - values.mean()
    n_fft = int(estimator.get("estimation_n_fft", 8192))
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values)), n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    baseline = float(np.median(spectrum[(frequencies >= 5.0) & (frequencies <= 1000.0)]))
    best_speed = min_hz
    best_prominence = -np.inf
    for harmonic in harmonics:
        low = max(5.0, harmonic * min_hz)
        high = min(sampling_rate / 2.0, harmonic * max_hz)
        mask = (frequencies >= low) & (frequencies <= high)
        if not mask.any():
            continue
        local = spectrum[mask]
        peak = int(np.argmax(local))
        prominence = float(local[peak] / max(baseline, 1e-12))
        speed = float(frequencies[mask][peak] / harmonic)
        if prominence > best_prominence:
            best_speed = speed
            best_prominence = prominence
    return float(np.clip(best_speed, min_hz, max_hz))


def physics_gear_bands(speed_hz: float, band_config: dict) -> list[list[float]]:
    physics = band_config.get("physics", {})
    stages = physics.get(
        "planetary_stages",
        [{"sun_teeth": 20, "ring_teeth": 100}, {"sun_teeth": 28, "ring_teeth": 100}],
    )
    half_width = float(physics.get("band_half_width_hz", 20.0))
    harmonics = int(physics.get("mesh_harmonics", 2))
    sidebands = int(physics.get("carrier_sidebands", 1))
    nyquist = float(band_config.get("sampling_rate_hz", SAMPLING_RATE_HZ)) / 2.0
    centers: list[float] = []
    stage_input = float(speed_hz)
    for stage in stages:
        sun = float(stage["sun_teeth"])
        ring = float(stage["ring_teeth"])
        ratio = 1.0 + ring / sun
        carrier = stage_input / ratio
        mesh = (stage_input - carrier) * sun
        for harmonic in range(1, harmonics + 1):
            base = harmonic * mesh
            centers.append(base)
            for sideband in range(1, sidebands + 1):
                centers.extend((base - sideband * carrier, base + sideband * carrier))
        stage_input = carrier
    bands = []
    for center in sorted(set(round(value, 6) for value in centers if value > 0.0)):
        low = max(0.0, center - half_width)
        high = min(nyquist, center + half_width)
        if low < high:
            bands.append([low, high])
    return bands


def bands_for_context(class_id: int, condition_id: str, signal: np.ndarray, band_config: dict) -> tuple[list, float | None, str]:
    physics = band_config.get("physics", {})
    physics_classes = {int(value) for value in physics.get("gear_fault_classes", [1, 2, 3, 4])}
    if class_id not in physics_classes:
        return band_config["classes"].get(str(class_id), []), None, "data_fallback"
    speed = parse_fixed_speed_hz(condition_id)
    source = "fixed_condition"
    if speed is None:
        speed = estimate_variable_speed_hz(signal, band_config)
        source = "spectral_ridge"
    return physics_gear_bands(speed, band_config), speed, source


def _frequency_mass(signal: np.ndarray, relevance: np.ndarray, band_config: dict, bands: list) -> tuple[float, float]:
    sampling_rate = float(band_config.get("sampling_rate_hz", SAMPLING_RATE_HZ))
    n_fft = int(band_config.get("n_fft", 256))
    hop_length = int(band_config.get("hop_length", 64))
    noverlap = max(0, n_fft - hop_length)
    weighted_total = weighted_inside = signal_total = signal_inside = 0.0
    for channel in range(NUM_CHANNELS):
        frequencies, _, zxx = stft(
            signal[channel],
            fs=sampling_rate,
            nperseg=n_fft,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
        power = np.abs(zxx).astype(np.float64) ** 2
        mask = np.zeros_like(frequencies, dtype=bool)
        for low, high in bands:
            mask |= (frequencies >= float(low)) & (frequencies <= float(high))
        for frame in range(power.shape[1]):
            start = frame * hop_length
            frame_relevance = float(relevance[channel, start : start + n_fft].sum())
            weighted_total += float(power[:, frame].sum()) * frame_relevance
            weighted_inside += float(power[mask, frame].sum()) * frame_relevance
            signal_total += float(power[:, frame].sum())
            signal_inside += float(power[mask, frame].sum())
    alignment = weighted_inside / max(weighted_total, 1e-12)
    control = signal_inside / max(signal_total, 1e-12)
    return float(alignment), float(control)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.maximum(np.asarray(left, dtype=np.float64), 0.0).reshape(-1)
    right = np.maximum(np.asarray(right, dtype=np.float64), 0.0).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, 0.0, 1.0))


def mechanical_score(
    windows: np.ndarray,
    relevance: np.ndarray,
    y_true: np.ndarray,
    metadata: list[dict],
    band_config: dict,
    *,
    contexts: list[MechanicalContext] | None = None,
    noisy_indices: np.ndarray | None = None,
    noisy_relevance: np.ndarray | None = None,
) -> dict:
    """Compute context-aware expected-band enrichment and relevance stability."""

    if band_config.get("metric_version") not in (None, MECHANICAL_METRIC_VERSION):
        raise ValueError(f"Unsupported mechanical metric version: {band_config.get('metric_version')!r}")
    windows = ensure_channels_first(windows)
    relevance = np.maximum(np.asarray(relevance, dtype=np.float32), 0.0)
    contexts = contexts or select_mechanical_contexts(
        y_true,
        metadata,
        context_length=int(band_config.get("context_length", MECHANICAL_CONTEXT_LENGTH)),
        contexts_per_group=int(band_config.get("contexts_per_group", MECHANICAL_CONTEXTS_PER_GROUP)),
    )
    if noisy_indices is None or noisy_relevance is None:
        raise ValueError("Mechanical v2 requires relevance recomputed from deterministic noisy inputs.")
    noisy_indices = np.asarray(noisy_indices, dtype=np.int64)
    noisy_relevance = np.maximum(np.asarray(noisy_relevance, dtype=np.float32), 0.0)
    lookup = {int(index): position for position, index in enumerate(noisy_indices)}
    enrichments = []
    stabilities = []
    scores = []
    speeds = []
    strategies: dict[str, int] = {}
    for context in contexts:
        signal = overlap_add_windows(windows[context.indices])
        clean_rel = overlap_add_windows(relevance[context.indices])
        try:
            noisy_rows = np.asarray([lookup[int(index)] for index in context.indices])
        except KeyError as exc:
            raise ValueError("Noisy relevance is missing a selected mechanical context window.") from exc
        noisy_rel = overlap_add_windows(noisy_relevance[noisy_rows])
        clean_rel /= max(float(clean_rel.sum()), 1e-12)
        noisy_rel /= max(float(noisy_rel.sum()), 1e-12)
        bands, speed, strategy = bands_for_context(context.class_id, context.condition_id, signal, band_config)
        alignment, control = _frequency_mass(signal, clean_rel, band_config, bands)
        enrichment = float(np.clip((alignment - control) / max(1.0 - control, 1e-12), 0.0, 1.0))
        stability = _cosine_similarity(clean_rel, noisy_rel)
        score = enrichment * (0.8 + 0.2 * stability)
        enrichments.append(enrichment)
        stabilities.append(stability)
        scores.append(score)
        strategies[strategy] = strategies.get(strategy, 0) + 1
        if speed is not None:
            speeds.append(speed)

    mech = float(np.mean(scores))
    return {
        "metric_version": MECHANICAL_METRIC_VERSION,
        "mechanical_score": mech,
        "expected_band_enrichment": float(np.mean(enrichments)),
        "relevance_stability": float(np.mean(stabilities)),
        "contexts": len(contexts),
        "strategy_counts": strategies,
        "estimated_speed_hz_min": float(min(speeds)) if speeds else None,
        "estimated_speed_hz_max": float(max(speeds)) if speeds else None,
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
