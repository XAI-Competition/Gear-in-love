"""Probe GearXAI mechanical-alignment levers from prepared validation data.

This is an analysis script, not part of the training path. It mirrors the
public devkit's STFT setup closely enough to answer one question: given that
the official private band config is unavailable, which parts of relevance are
actually controllable and which conclusions are fragile?
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import stft

CHANNELS = ("motor", "rgb_y", "rgb_x", "rgb_z", "torque", "pgb_y", "pgb_x", "pgb_z")
CLASSES = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")
FS = 5120
WINDOW_LENGTH = 100
HOP_LENGTH = 64
NOVERLAP = WINDOW_LENGTH - HOP_LENGTH


def fixed_speed(condition_id: str | None) -> int | None:
    match = re.fullmatch(r"PGB_(20|30|40|50)_\d+", str(condition_id))
    return int(match.group(1)) if match else None


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def mean_pair_corr(arrays: list[np.ndarray]) -> float | None:
    values: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            a = arrays[i]
            b = arrays[j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                values.append(0.0)
            else:
                values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else None


def channel_spectra(window: np.ndarray) -> np.ndarray:
    spectra = np.zeros((len(CHANNELS), 51), dtype=np.float64)
    for ch in range(len(CHANNELS)):
        _, _, zxx = stft(
            window[ch],
            fs=FS,
            nperseg=WINDOW_LENGTH,
            noverlap=NOVERLAP,
            boundary=None,
            padded=False,
        )
        spectra[ch] = np.abs(zxx).sum(axis=1)
    return spectra


def compute_fixed_speed_records(
    windows: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    per_group: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[str, int]]:
    by_group: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for row, (label, meta) in enumerate(zip(labels, metadata, strict=True)):
        speed = fixed_speed(meta.get("condition_id"))
        if speed is not None:
            by_group[(int(label), speed)].append(row)

    group_counts = {
        f"{CLASSES[class_id]}_{speed}": len(by_group[(class_id, speed)])
        for class_id in range(len(CLASSES))
        for speed in (20, 30, 40, 50)
    }
    records: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for class_id in range(len(CLASSES)):
        for speed in (20, 30, 40, 50):
            indices = by_group.get((class_id, speed), [])
            if not indices:
                continue
            sample = rng.choice(indices, size=min(per_group, len(indices)), replace=False)
            aggregate = np.zeros((len(CHANNELS), 51), dtype=np.float64)
            for row in sample:
                aggregate += channel_spectra(np.asarray(windows[row], dtype=np.float32))
            total = aggregate.sum(axis=0)
            channel = aggregate.sum(axis=1)
            records[(class_id, speed)] = {
                "spectrum": total / max(total.sum(), 1e-12),
                "channel": channel / max(channel.sum(), 1e-12),
            }
    return records, group_counts


def summarize_cross_speed(records: dict[tuple[int, int], dict[str, np.ndarray]]) -> dict[str, Any]:
    _, times, zxx = stft(
        np.zeros(WINDOW_LENGTH, dtype=np.float32),
        fs=FS,
        nperseg=WINDOW_LENGTH,
        noverlap=NOVERLAP,
        boundary=None,
        padded=False,
    )
    freqs = np.fft.rfftfreq(WINDOW_LENGTH, d=1 / FS)
    class_stats: dict[str, Any] = {}
    for class_id in range(1, len(CLASSES)):
        speeds = [
            speed
            for speed in (20, 30, 40, 50)
            if (class_id, speed) in records and (0, speed) in records
        ]
        spectrum_delta = [
            records[(class_id, speed)]["spectrum"] - records[(0, speed)]["spectrum"]
            for speed in speeds
        ]
        channel_abs = [records[(class_id, speed)]["channel"] for speed in speeds]
        channel_delta = [
            records[(class_id, speed)]["channel"] - records[(0, speed)]["channel"]
            for speed in speeds
        ]
        top_nonzero_freqs: list[float] = []
        for speed in speeds:
            delta = records[(class_id, speed)]["spectrum"] - records[(0, speed)]["spectrum"]
            top = int(np.argmax(delta[1:]) + 1)
            top_nonzero_freqs.append(float(freqs[top]))

        class_stats[CLASSES[class_id]] = {
            "speeds": speeds,
            "delta_spectrum_cross_speed_corr": round(mean_pair_corr(spectrum_delta) or 0.0, 3),
            "absolute_channel_cross_speed_corr": round(mean_pair_corr(channel_abs) or 0.0, 3),
            "delta_channel_cross_speed_corr": round(mean_pair_corr(channel_delta) or 0.0, 3),
            "top_nonzero_delta_freq_hz_per_speed": top_nonzero_freqs,
            "top_nonzero_delta_freq_span_hz": round(
                float(max(top_nonzero_freqs) - min(top_nonzero_freqs)), 1
            ),
            "dominant_abs_channel_per_speed": [
                CHANNELS[int(np.argmax(records[(class_id, speed)]["channel"]))] for speed in speeds
            ],
            "dominant_delta_channel_per_speed": [
                CHANNELS[
                    int(
                        np.argmax(
                            records[(class_id, speed)]["channel"]
                            - records[(0, speed)]["channel"]
                        )
                    )
                ]
                for speed in speeds
            ],
        }

    return {
        "stft_defaults": {
            "fs": FS,
            "nperseg": WINDOW_LENGTH,
            "noverlap": NOVERLAP,
            "hop_length": HOP_LENGTH,
            "frames": int(zxx.shape[1]),
            "freq_bins": 51,
            "freq_step_hz": 51.2,
            "times": times.tolist(),
        },
        "class_stats": class_stats,
    }


def run_model_relevance(model_path: Path, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.ascontiguousarray(windows, dtype=np.float32)})
    probabilities = next(array for array in outputs if array.ndim == 2)
    relevance = next(array for array in outputs if array.ndim == 3)
    return probabilities, relevance


def normalize_channel_mass(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def summarize_model_relevance(
    model_path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    indices: list[int] = []
    for class_id in range(len(CLASSES)):
        class_indices = np.flatnonzero(labels == class_id)
        indices.extend(
            rng.choice(
                class_indices,
                size=min(per_class, len(class_indices)),
                replace=False,
            )
        )
    selected = np.array(sorted(indices))
    sample_windows = np.ascontiguousarray(windows[selected], dtype=np.float32)
    sample_labels = labels[selected].astype(int)

    probabilities, relevance = run_model_relevance(model_path, sample_windows)
    predictions = probabilities.argmax(axis=1)
    rel_mass = normalize_channel_mass(np.maximum(relevance, 0.0).sum(axis=2))
    abs_mass = normalize_channel_mass(np.abs(sample_windows).sum(axis=2))

    per_class_summary: dict[str, Any] = {}
    for class_id in range(len(CLASSES)):
        mask = sample_labels == class_id
        per_class_summary[CLASSES[class_id]] = {
            "n": int(mask.sum()),
            "accuracy": round(float((predictions[mask] == sample_labels[mask]).mean()), 4),
            "mean_current_relevance_channel": {
                CHANNELS[channel]: round(float(rel_mass[mask, channel].mean()), 4)
                for channel in range(len(CHANNELS))
            },
            "mean_abs_channel": {
                CHANNELS[channel]: round(float(abs_mass[mask, channel].mean()), 4)
                for channel in range(len(CHANNELS))
            },
        }

    return {
        "sampled_windows": int(len(sample_windows)),
        "sample_accuracy": round(float((predictions == sample_labels).mean()), 4),
        "per_class": per_class_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--per-group", type=int, default=300)
    parser.add_argument("--model-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--out", type=Path, default=Path(".tmp/mechanical_alignment_analysis.json"))
    args = parser.parse_args()

    labels = np.load(args.data_dir / "validation_labels.npy")
    windows = np.load(args.data_dir / "validation_windows.npy", mmap_mode="r")
    metadata = load_metadata(args.data_dir / "validation_metadata.jsonl")
    rng = np.random.default_rng(args.seed)

    records, group_counts = compute_fixed_speed_records(
        windows,
        labels,
        metadata,
        per_group=args.per_group,
        rng=rng,
    )
    report = {
        "data_dir": str(args.data_dir),
        "validation_shape": list(windows.shape),
        "sample_size_per_class_speed": args.per_group,
        "fixed_speed_group_counts": group_counts,
        **summarize_cross_speed(records),
    }
    if args.model is not None:
        report["model_relevance"] = summarize_model_relevance(
            args.model,
            windows,
            labels,
            per_class=args.model_per_class,
            rng=rng,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
