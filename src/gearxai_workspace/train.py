"""CPU-friendly training loop for the GearXAI baseline.

Trains :class:`~gearxai_workspace.model.GearXAINet` on a balanced in-memory
subset, tracking macro-F1 on a validation subset (the same metric the devkit
uses as the eligibility gate) and keeping the best-scoring weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from gearxai_workspace.data import NUM_CLASSES, load_training_subset
from gearxai_workspace.model import GearXAINet, ModelConfig, build_model, count_parameters


@dataclass
class TrainConfig:
    data_dir: str = "data/prepared"
    train_per_class: int | None = 8000
    val_per_class: int | None = 2000
    epochs: int = 12
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    seed: int = 42
    num_threads: int | None = None
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)


def resolve_device(device: str) -> torch.device:
    """Resolve the ``"auto"`` device to cuda when available, else cpu."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Mean per-class F1 (matches ``gearxai_devkit.metrics.macro_f1_score``)."""

    scores = []
    for class_id in range(num_classes):
        tp = int(np.sum((y_pred == class_id) & (y_true == class_id)))
        fp = int(np.sum((y_pred == class_id) & (y_true != class_id)))
        fn = int(np.sum((y_pred != class_id) & (y_true == class_id)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        denom = precision + recall
        scores.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return float(np.mean(scores))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: GearXAINet, x: torch.Tensor, y: np.ndarray, batch_size: int) -> float:
    """Evaluate macro-F1; ``x`` is expected to already live on the model's device."""

    model.eval()
    preds = np.empty(len(y), dtype=np.int64)
    for start in range(0, len(x), batch_size):
        logits = model.classify(x[start : start + batch_size])
        preds[start : start + batch_size] = logits.argmax(dim=1).cpu().numpy()
    return macro_f1(y, preds)


def train_baseline(config: TrainConfig) -> dict:
    """Train the baseline and return the best weights plus a history dict."""

    set_seed(config.seed)
    if config.num_threads:
        torch.set_num_threads(config.num_threads)

    device = resolve_device(config.device)

    data = load_training_subset(
        config.data_dir,
        train_per_class=config.train_per_class,
        val_per_class=config.val_per_class,
        seed=config.seed,
    )
    # The balanced subset is small (~0.8 GB for 270k windows), so keep all
    # tensors resident on the device and index in-place — no per-batch H2D copy.
    x_train = torch.from_numpy(data["x_train"]).to(device)
    y_train = torch.from_numpy(data["y_train"]).to(device)
    x_val = torch.from_numpy(data["x_val"]).to(device)
    y_val_np = data["y_val"]

    model = build_model(config.model).to(device)
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train: {len(x_train):,} windows | Val: {len(x_val):,} windows")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history: list[dict] = []
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] = {}
    n = len(x_train)

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        start_time = time.perf_counter()
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        scheduler.step()

        train_loss = running / n
        val_f1 = evaluate(model, x_val, y_val_np, config.batch_size)
        elapsed = time.perf_counter() - start_time
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})
        print(
            f"epoch {epoch:2d}/{config.epochs} | loss {train_loss:.4f} "
            f"| val_macro_f1 {val_f1:.4f} | {elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Return the best model on CPU so ONNX export stays CPU-only (the evaluator
    # runs ONNX Runtime on CPUExecutionProvider; the submission must not need CUDA).
    model = model.cpu()
    model.load_state_dict(best_state)
    return {
        "model": model,
        "config": config,
        "best_val_macro_f1": best_f1,
        "history": history,
        "device": str(device),
    }
