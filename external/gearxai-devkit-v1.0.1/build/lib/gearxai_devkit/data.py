"""Official GearXAI data preparation utilities."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gearxai_devkit.constants import (
    CLASS_TO_INDEX,
    NUM_CHANNELS,
    WINDOW_LENGTH,
)


@dataclass(frozen=True)
class SplitData:
    """Prepared competition split."""

    windows: np.ndarray
    labels: np.ndarray
    metadata: list[dict]


def ensure_channels_first(windows: np.ndarray) -> np.ndarray:
    """Return windows as float32 `[N, 8, 100]` arrays."""

    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {windows.shape}.")

    if windows.shape[1:] == (NUM_CHANNELS, WINDOW_LENGTH):
        out = windows
    elif windows.shape[1:] == (WINDOW_LENGTH, NUM_CHANNELS):
        out = np.transpose(windows, (0, 2, 1))
    else:
        raise ValueError(
            "Expected windows shaped [N, 8, 100] or [N, 100, 8], "
            f"got {windows.shape}."
        )
    return out.astype(np.float32, copy=False)


def encode_labels(labels: Sequence) -> np.ndarray:
    """Map DDS class labels or integer labels to `[0, 8]` class ids."""

    encoded = []
    for value in labels:
        if isinstance(value, (np.integer, int)):
            label_id = int(value)
        else:
            text = str(value).strip()
            if text.isdigit():
                label_id = int(text)
            else:
                try:
                    label_id = CLASS_TO_INDEX[text]
                except KeyError as exc:
                    raise ValueError(f"Unknown class label: {value!r}") from exc
        if not 0 <= label_id < len(CLASS_TO_INDEX):
            raise ValueError(f"Class id out of range: {label_id}")
        encoded.append(label_id)
    return np.asarray(encoded, dtype=np.int64)


def load_csv_windows(
    csv_path: str | Path,
    *,
    label_column: str = "Fault",
    sequence_length: int = WINDOW_LENGTH,
    stride: int = 1,
    feature_columns: Sequence[str] | None = None,
) -> SplitData:
    """Create official windows from a DDS-style CSV file.

    Existing DDS scripts group rows by `Fault` before windowing. This function
    keeps that convention and emits channels-first windows for ONNX evaluation.
    """

    csv_path = Path(csv_path)
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column {label_column!r} in {csv_path}.")

    if feature_columns is None:
        feature_columns = [c for c in frame.columns if c != label_column][:NUM_CHANNELS]
    if len(feature_columns) != NUM_CHANNELS:
        raise ValueError(
            f"Expected {NUM_CHANNELS} feature columns, got {len(feature_columns)}."
        )

    windows: list[np.ndarray] = []
    labels: list = []
    metadata: list[dict] = []

    for label, label_frame in frame.groupby(label_column, sort=False):
        values = label_frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        if len(values) < sequence_length:
            continue
        for start in range(0, len(values) - sequence_length + 1, stride):
            window = values[start : start + sequence_length].T
            windows.append(window)
            labels.append(label)
            metadata.append(
                {
                    "source": str(csv_path),
                    "label": str(label),
                    "start": int(start),
                    "sequence_length": int(sequence_length),
                    "stride": int(stride),
                }
            )

    if not windows:
        raise ValueError(f"No windows were created from {csv_path}.")

    return SplitData(
        windows=np.asarray(windows, dtype=np.float32),
        labels=encode_labels(labels),
        metadata=metadata,
    )


def channel_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only per-channel mean and std over samples and time."""

    windows = ensure_channels_first(windows)
    mean = windows.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = windows.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardization(
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel z-score normalization."""

    windows = ensure_channels_first(windows)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, NUM_CHANNELS, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, NUM_CHANNELS, 1)
    return ((windows - mean) / std).astype(np.float32)


def balance_by_class(
    split: SplitData,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
) -> SplitData:
    """Subsample each class without replacement to the same count."""

    rng = np.random.default_rng(seed)
    labels = np.asarray(split.labels)
    class_ids = np.unique(labels)
    if max_per_class is None:
        max_per_class = min(int((labels == c).sum()) for c in class_ids)
    selected: list[np.ndarray] = []
    for class_id in class_ids:
        idx = np.flatnonzero(labels == class_id)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(np.asarray(idx, dtype=np.int64))
    keep = np.concatenate(selected)
    keep.sort()
    metadata = [split.metadata[int(i)] for i in keep]
    return SplitData(split.windows[keep], labels[keep], metadata)


def save_split(output_dir: str | Path, split_name: str, split: SplitData) -> None:
    """Save a prepared split to NPY and JSONL files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split_name}_windows.npy", ensure_channels_first(split.windows))
    np.save(output_dir / f"{split_name}_labels.npy", np.asarray(split.labels, dtype=np.int64))
    with (output_dir / f"{split_name}_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in split.metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_split(data_dir: str | Path, split_name: str) -> SplitData:
    """Load a prepared split from disk."""

    data_dir = Path(data_dir)
    windows_path = data_dir / f"{split_name}_windows.npy"
    labels_path = data_dir / f"{split_name}_labels.npy"
    metadata_path = data_dir / f"{split_name}_metadata.jsonl"
    if not windows_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing prepared split {split_name!r} in {data_dir}."
        )

    metadata: list[dict] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = [json.loads(line) for line in handle if line.strip()]

    return SplitData(
        windows=ensure_channels_first(np.load(windows_path)),
        labels=encode_labels(np.load(labels_path)),
        metadata=metadata,
    )


def save_stats(output_dir: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    """Save normalization statistics used by the official loader."""

    stats = {
        "channel_mean": np.asarray(mean, dtype=float).tolist(),
        "channel_std": np.asarray(std, dtype=float).tolist(),
        "standardized_channel_mean": [0.0] * NUM_CHANNELS,
        "format": "[N, 8, 100]",
    }
    with (Path(output_dir) / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def save_prestandardized_stats(output_dir: str | Path) -> None:
    """Save stats for already scaled release windows."""

    stats = {
        "format": "[N, 8, 100]",
        "source": "GearXAI windows_100 Parquet release",
        "standardized_channel_mean": [0.0] * NUM_CHANNELS,
    }
    with (Path(output_dir) / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def load_stats(data_dir: str | Path) -> dict:
    """Load saved normalization statistics."""

    with (Path(data_dir) / "stats.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_from_csvs(
    *,
    train_csv: str | Path,
    dev_csv: str | Path,
    output_dir: str | Path,
    test_csv: str | Path | None = None,
    balance: bool = True,
    stride: int = 1,
    seed: int = 42,
) -> dict:
    """Prepare official train/dev/test files from DDS-style CSV inputs."""

    train = load_csv_windows(train_csv, stride=stride)
    dev = load_csv_windows(dev_csv, stride=stride)
    test = load_csv_windows(test_csv, stride=stride) if test_csv else None

    if balance:
        train = balance_by_class(train, seed=seed)
        dev = balance_by_class(dev, seed=seed)
        if test is not None:
            test = balance_by_class(test, seed=seed)

    mean, std = channel_stats(train.windows)
    train = SplitData(apply_standardization(train.windows, mean, std), train.labels, train.metadata)
    dev = SplitData(apply_standardization(dev.windows, mean, std), dev.labels, dev.metadata)
    if test is not None:
        test = SplitData(apply_standardization(test.windows, mean, std), test.labels, test.metadata)

    save_split(output_dir, "train", train)
    save_split(output_dir, "dev", dev)
    if test is not None:
        save_split(output_dir, "test", test)
    save_stats(output_dir, mean, std)

    return {
        "train_windows": int(len(train.labels)),
        "dev_windows": int(len(dev.labels)),
        "test_windows": int(len(test.labels)) if test is not None else 0,
        "format": "[N, 8, 100]",
    }


def _split_train_dev_indices(
    labels: np.ndarray,
    *,
    dev_fraction: float,
    gap_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_parts: list[np.ndarray] = []
    dev_parts: list[np.ndarray] = []
    audit = {
        "dev_fraction": float(dev_fraction),
        "gap_windows": int(gap_windows),
        "classes": {},
        "dropped_gap_windows": 0,
    }
    for label in sorted(np.unique(labels).tolist()):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < 3:
            raise ValueError(f"Class {label} has too few rows for train/dev split.")
        dev_count = max(1, int(round(len(label_indices) * dev_fraction)))
        if dev_count >= len(label_indices):
            raise ValueError(f"Class {label} dev split consumes all rows.")
        dev_start_pos = len(label_indices) - dev_count
        train_end_pos = max(0, dev_start_pos - gap_windows)
        if train_end_pos == 0:
            raise ValueError(
                f"Class {label} has no train rows after gap={gap_windows}; "
                "reduce --dev-fraction or --gap-windows."
            )
        train_idx = label_indices[:train_end_pos]
        dev_idx = label_indices[dev_start_pos:]
        gap_count = int(dev_start_pos - train_end_pos)
        train_parts.append(train_idx)
        dev_parts.append(dev_idx)
        audit["classes"][str(int(label))] = {
            "source_train_rows": int(len(label_indices)),
            "train_rows": int(len(train_idx)),
            "dev_rows": int(len(dev_idx)),
            "dropped_gap_windows": gap_count,
        }
        audit["dropped_gap_windows"] += gap_count
    train = np.concatenate(train_parts).astype(np.int64)
    dev = np.concatenate(dev_parts).astype(np.int64)
    train.sort()
    dev.sort()
    return train, dev, audit


def _hash_window(window: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(window.astype(np.float32, copy=False))
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()


def _parquet_paths(windows_dir: Path, split: str) -> list[Path]:
    split_dir = windows_dir / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.parquet"))


def _signals_to_windows(signals: Sequence) -> np.ndarray:
    windows = [
        np.stack([np.asarray(row, dtype=np.float32) for row in signal], axis=0)
        for signal in signals
    ]
    return ensure_channels_first(np.stack(windows, axis=0))


def _write_parquet_selection(
    *,
    paths_and_indices: Sequence[tuple[Path, np.ndarray]],
    output_dir: Path,
    split_name: str,
    total_rows: int,
    hash_sets: dict[str, set[str]],
    label_counts: dict[str, dict[str, int]],
) -> None:
    windows_out = np.lib.format.open_memmap(
        output_dir / f"{split_name}_windows.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, NUM_CHANNELS, WINDOW_LENGTH),
    )
    labels_out = np.lib.format.open_memmap(
        output_dir / f"{split_name}_labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )
    offset = 0
    with (output_dir / f"{split_name}_metadata.jsonl").open("w", encoding="utf-8") as meta_handle:
        for path, indices in paths_and_indices:
            if len(indices) == 0:
                continue
            import pandas as pd

            frame = pd.read_parquet(path)
            selected = frame.iloc[indices]
            windows = _signals_to_windows(selected["signal"])
            labels = encode_labels(selected["fault_code"].to_numpy())
            end = offset + len(labels)
            windows_out[offset:end] = windows
            labels_out[offset:end] = labels
            source_split = "source_train" if split_name in {"train", "dev"} else "source_test"
            for window, (_, row), label in zip(windows, selected.iterrows(), labels):
                hash_sets[split_name].add(_hash_window(window))
                label_counts[split_name][str(int(label))] = label_counts[split_name].get(str(int(label)), 0) + 1
                meta_handle.write(
                    json.dumps(
                        {
                            "condition_id": row.get("condition_id"),
                            "source_split": source_split,
                            "source_window_index": int(row.get("window_index", 0)),
                            "fault_code": row.get("fault_code"),
                            "label": int(label),
                            "format": "[8, 100]",
                            "source": str(path),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            offset = end
    windows_out.flush()
    labels_out.flush()
    if offset != total_rows:
        raise RuntimeError(f"Wrote {offset} {split_name} rows, expected {total_rows}.")


def _write_release_parquet_split(
    *,
    paths: Sequence[Path],
    output_dir: Path,
    split_name: str,
    hash_set: set[str],
    label_counts: dict[str, int],
    per_condition: dict[str, dict],
) -> int:
    import pandas as pd

    total_rows = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=["fault_code", "condition_id", "window_index"])
        total_rows += int(len(frame))
        condition_id = str(frame["condition_id"].iloc[0]) if len(frame) else path.stem
        per_condition.setdefault(condition_id, {})[f"{split_name}_rows"] = int(len(frame))

    windows_out = np.lib.format.open_memmap(
        output_dir / f"{split_name}_windows.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, NUM_CHANNELS, WINDOW_LENGTH),
    )
    labels_out = np.lib.format.open_memmap(
        output_dir / f"{split_name}_labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )

    offset = 0
    with (output_dir / f"{split_name}_metadata.jsonl").open("w", encoding="utf-8") as meta_handle:
        for path in paths:
            frame = pd.read_parquet(path)
            windows = _signals_to_windows(frame["signal"])
            labels = encode_labels(frame["fault_code"].to_numpy())
            end = offset + len(labels)
            windows_out[offset:end] = windows
            labels_out[offset:end] = labels
            for window, (_, row), label in zip(windows, frame.iterrows(), labels):
                hash_set.add(_hash_window(window))
                label_counts[str(int(label))] = label_counts.get(str(int(label)), 0) + 1
                meta_handle.write(
                    json.dumps(
                        {
                            "condition_id": row.get("condition_id"),
                            "source_split": f"public_{split_name}",
                            "source_window_index": int(row.get("window_index", 0)),
                            "fault_code": row.get("fault_code"),
                            "label": int(label),
                            "format": "[8, 100]",
                            "source": str(path),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            offset = end

    windows_out.flush()
    labels_out.flush()
    if offset != total_rows:
        raise RuntimeError(f"Wrote {offset} {split_name} rows, expected {total_rows}.")
    return total_rows


def prepare_public_release_windows(
    *,
    windows_dir: str | Path,
    output_dir: str | Path,
    splits: Sequence[str] = ("train", "validation"),
) -> dict:
    """Prepare evaluator NPY files from the public `windows_100` Parquet release.

    This is the participant-facing path. It accepts only the public train and
    validation splits and refuses a labeled public test directory, because the
    leaderboard test split is intentionally hidden.
    """

    windows_dir = Path(windows_dir)
    output_dir = Path(output_dir)
    requested = tuple(splits)
    allowed = {"train", "validation"}
    unexpected = sorted(set(requested) - allowed)
    if unexpected:
        raise ValueError(f"Only public splits {sorted(allowed)} are supported, got {unexpected}.")
    test_paths = _parquet_paths(windows_dir, "test")
    if test_paths:
        raise ValueError(
            "Found a labeled test directory in the input. Use the current GearXAI public "
            "release, which contains only train and validation splits."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_sets: dict[str, set[str]] = {split: set() for split in requested}
    label_counts: dict[str, dict[str, int]] = {split: {} for split in requested}
    counts: dict[str, int] = {}
    per_condition: dict[str, dict] = {}

    for split_name in requested:
        paths = _parquet_paths(windows_dir, split_name)
        if not paths:
            raise FileNotFoundError(f"No {split_name} Parquet files found under {windows_dir / split_name}.")
        counts[split_name] = _write_release_parquet_split(
            paths=paths,
            output_dir=output_dir,
            split_name=split_name,
            hash_set=hash_sets[split_name],
            label_counts=label_counts[split_name],
            per_condition=per_condition,
        )

    intersections = {}
    for index, left in enumerate(requested):
        for right in requested[index + 1 :]:
            intersections[f"{left}_{right}"] = len(hash_sets[left] & hash_sets[right])
    split_integrity_verified = all(value == 0 for value in intersections.values())
    save_prestandardized_stats(output_dir)
    audit = {
        "windows_dir": str(windows_dir),
        "output_dir": str(output_dir),
        "public_splits": list(requested),
        "hidden_leaderboard_split": "not included",
        "counts": counts,
        "conditions": per_condition,
        "label_counts": label_counts,
        "exact_window_hash_intersections": intersections,
        "split_integrity_verified": split_integrity_verified,
        "hash_algorithm": "blake2b-128 over float32 channels-first windows",
        "format": "[N, 8, 100]",
    }
    with (output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not split_integrity_verified:
        raise RuntimeError(f"Split audit failed: {intersections}")
    return audit


def prepare_from_parquet_windows(
    *,
    windows_dir: str | Path,
    output_dir: str | Path,
    dev_fraction: float = 0.10,
    gap_windows: int = WINDOW_LENGTH - 1,
    include_test: bool = True,
    allow_split_overlap: bool = False,
) -> dict:
    """Prepare evaluator NPY files from the released `windows_100` Parquet layout."""

    windows_dir = Path(windows_dir)
    output_dir = Path(output_dir)
    train_paths = _parquet_paths(windows_dir, "train")
    test_paths = _parquet_paths(windows_dir, "test") if include_test else []
    if not train_paths:
        raise FileNotFoundError(f"No train Parquet files found under {windows_dir / 'train'}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, list[tuple[Path, np.ndarray]]] = {"train": [], "dev": [], "test": []}
    counts = {"train": 0, "dev": 0, "test": 0}
    per_condition: dict[str, dict] = {}

    for path in train_paths:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["fault_code", "condition_id", "window_index"])
        labels = encode_labels(frame["fault_code"].to_numpy())
        train_idx, dev_idx, split_policy = _split_train_dev_indices(
            labels,
            dev_fraction=dev_fraction,
            gap_windows=gap_windows,
        )
        condition_id = str(frame["condition_id"].iloc[0]) if len(frame) else path.stem
        selections["train"].append((path, train_idx))
        selections["dev"].append((path, dev_idx))
        counts["train"] += int(len(train_idx))
        counts["dev"] += int(len(dev_idx))
        per_condition[condition_id] = {
            "source_train_rows": int(len(frame)),
            "train_rows": int(len(train_idx)),
            "dev_rows": int(len(dev_idx)),
            "train_dev_policy": split_policy,
        }

    for path in test_paths:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["fault_code", "condition_id", "window_index"])
        indices = np.arange(len(frame), dtype=np.int64)
        condition_id = str(frame["condition_id"].iloc[0]) if len(frame) else path.stem
        selections["test"].append((path, indices))
        counts["test"] += int(len(indices))
        per_condition.setdefault(condition_id, {})["source_test_rows"] = int(len(frame))
        per_condition[condition_id]["test_rows"] = int(len(indices))

    hash_sets: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    label_counts: dict[str, dict[str, int]] = {"train": {}, "dev": {}, "test": {}}
    for split_name in ["train", "dev", "test"]:
        if split_name == "test" and not include_test:
            continue
        _write_parquet_selection(
            paths_and_indices=selections[split_name],
            output_dir=output_dir,
            split_name=split_name,
            total_rows=counts[split_name],
            hash_sets=hash_sets,
            label_counts=label_counts,
        )

    intersections = {
        "train_dev": len(hash_sets["train"] & hash_sets["dev"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "dev_test": len(hash_sets["dev"] & hash_sets["test"]),
    }
    split_integrity_verified = all(value == 0 for value in intersections.values())
    save_prestandardized_stats(output_dir)
    audit = {
        "windows_dir": str(windows_dir),
        "output_dir": str(output_dir),
        "counts": counts,
        "conditions": per_condition,
        "label_counts": label_counts,
        "exact_window_hash_intersections": intersections,
        "split_integrity_verified": split_integrity_verified,
        "hash_algorithm": "blake2b-128 over float32 channels-first windows",
        "format": "[N, 8, 100]",
    }
    with (output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not split_integrity_verified and not allow_split_overlap:
        raise RuntimeError(f"Split audit failed: {intersections}")
    return audit


def iter_batches(windows: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    """Yield contiguous mini-batches."""

    for start in range(0, len(windows), batch_size):
        yield windows[start : start + batch_size]
