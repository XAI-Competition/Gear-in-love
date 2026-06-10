"""Loading and batching for the prepared GearXAI windows.

The devkit writes evaluator-ready arrays via ``gearxai prepare-data``:

- ``{split}_windows.npy`` : float32 ``[N, 8, 100]`` (channels-first, pre-standardized)
- ``{split}_labels.npy``  : int64 ``[N]`` class ids in ``[0, 8]``

The public release is class-balanced, so for a CPU-friendly baseline we draw a
balanced subset into RAM and iterate it with plain index shuffling (no
DataLoader overhead, which dominates for tiny windows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_CHANNELS = 8
WINDOW_LENGTH = 100
NUM_CLASSES = 9


def load_metadata(data_dir: str | Path, split: str) -> list[dict[str, str]]:
    """Load prepared JSONL metadata for a split."""

    import json

    metadata_path = Path(data_dir) / f"{split}_metadata.jsonl"
    rows = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def split_paths(data_dir: str | Path, split: str) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    return data_dir / f"{split}_windows.npy", data_dir / f"{split}_labels.npy"


def load_split(
    data_dir: str | Path,
    split: str,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, labels)`` for a prepared split.

    ``windows`` is memory-mapped by default so the 2.2 GB train array is not
    pulled into RAM until it is actually indexed.
    """

    windows_path, labels_path = split_paths(data_dir, split)
    if not windows_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing prepared split {split!r} in {data_dir}. "
            "Run: gearxai prepare-data --windows-dir <release> --out data/prepared"
        )
    windows = np.load(windows_path, mmap_mode="r" if mmap else None)
    labels = np.load(labels_path)
    if windows.shape[1:] != (NUM_CHANNELS, WINDOW_LENGTH):
        raise ValueError(f"Expected windows [N, 8, 100], got {windows.shape}.")
    return windows, labels.astype(np.int64)


def balanced_subset_indices(
    labels: np.ndarray,
    per_class: int | None,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Pick up to ``per_class`` sorted indices per class (all rows if ``None``)."""

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == class_id)
        if per_class is not None and len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        selected.append(idx)
    keep = np.concatenate(selected)
    keep.sort()  # sorted access keeps memmap reads sequential
    return keep.astype(np.int64)


def balanced_subset_indices_with_variable_fraction(
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    per_class: int | None,
    *,
    variable_fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """Pick a balanced subset while targeting a variable-speed fraction per class."""

    if not 0.0 <= variable_fraction <= 1.0:
        raise ValueError("variable_fraction must be in [0, 1].")
    if per_class is None:
        return balanced_subset_indices(labels, per_class, seed=seed)

    rng = np.random.default_rng(seed)
    is_variable = np.array(
        ["Variable_speed" in row["condition_id"] for row in metadata], dtype=bool
    )
    selected: list[np.ndarray] = []
    for class_id in range(NUM_CLASSES):
        class_idx = np.flatnonzero(labels == class_id)
        var_idx = class_idx[is_variable[class_idx]]
        fixed_idx = class_idx[~is_variable[class_idx]]

        target_var = int(round(per_class * variable_fraction))
        n_var = min(target_var, len(var_idx), per_class)
        n_fixed = per_class - n_var
        if n_fixed > len(fixed_idx):
            n_fixed = len(fixed_idx)
            n_var = min(per_class - n_fixed, len(var_idx))

        var_keep = (
            rng.choice(var_idx, size=n_var, replace=False)
            if n_var
            else np.array([], dtype=np.int64)
        )
        fixed_keep = (
            rng.choice(fixed_idx, size=n_fixed, replace=False)
            if n_fixed
            else np.array([], dtype=np.int64)
        )
        selected.append(np.concatenate([var_keep, fixed_keep]))

    keep = np.concatenate(selected)
    keep.sort()
    return keep.astype(np.int64)


def materialize(
    windows: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the selected rows off the memmap into contiguous RAM arrays."""

    x = np.ascontiguousarray(windows[indices], dtype=np.float32)
    y = np.ascontiguousarray(labels[indices], dtype=np.int64)
    return x, y


def load_training_subset(
    data_dir: str | Path,
    *,
    train_per_class: int | None,
    val_per_class: int | None,
    train_variable_fraction: float | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Convenience loader returning balanced in-memory train/val subsets."""

    train_w, train_y = load_split(data_dir, "train")
    val_w, val_y = load_split(data_dir, "validation")

    if train_variable_fraction is None:
        train_idx = balanced_subset_indices(train_y, train_per_class, seed=seed)
    else:
        train_meta = load_metadata(data_dir, "train")
        train_idx = balanced_subset_indices_with_variable_fraction(
            train_y,
            train_meta,
            train_per_class,
            variable_fraction=train_variable_fraction,
            seed=seed,
        )
    val_idx = balanced_subset_indices(val_y, val_per_class, seed=seed)

    x_train, y_train = materialize(train_w, train_y, train_idx)
    x_val, y_val = materialize(val_w, val_y, val_idx)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }
