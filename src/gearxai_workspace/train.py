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
    # Weight of the channel-prior relevance regularizer that trains the
    # channel-attention gate. Default 0 (disabled): the exp-002d A/B showed the
    # prior trades a certain ~0.018 faithfulness loss for only a tiny, uncertain
    # proxy mechanical gain (net negative). Kept as an opt-in lever.
    relevance_weight: float = 0.0
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


# Class-conditioned channel prior [9, 8]: for each fault class, which of the 8
# channels carry its discriminative energy. Estimated empirically (exp-002a) and
# backed by physical intuition for the DDS-SEU rig (planetary/parallel gearbox
# sensors + torque). Kept SOFT and broad on purpose — we don't know the official
# private band config, so we only nudge mass toward plausibly-relevant channels
# rather than overfitting a single channel.
# Channels: 0 motor, 1 rgb_y, 2 rgb_x, 3 rgb_z, 4 torque, 5 pgb_y, 6 pgb_x, 7 pgb_z
# Classes:  0 HEA, 1 CTF, 2 MTF, 3 RCF, 4 SWF, 5 BWF, 6 CWF, 7 IRF, 8 ORF
CHANNEL_PRIOR = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],  # HEA: no preference (healthy)
        [0, 1, 1, 1, 0, 2, 2, 1],  # CTF: pgb vibration (gear tooth)
        [0, 1, 1, 1, 0, 2, 2, 1],  # MTF: pgb vibration (missing tooth)
        [0, 1, 1, 1, 2, 2, 2, 1],  # RCF: torque + pgb (root crack)
        [1, 2, 1, 1, 1, 2, 1, 1],  # SWF: low-band vibration (surface wear)
        [1, 1, 1, 1, 0, 2, 1, 2],  # BWF: pgb bearing (ball fault)
        [2, 1, 1, 1, 0, 1, 1, 1],  # CWF: motor (combination)
        [0, 1, 1, 1, 2, 2, 2, 1],  # IRF: torque + pgb (inner race)
        [2, 1, 1, 1, 0, 1, 1, 1],  # ORF: motor + low band (outer race)
    ],
    dtype=np.float32,
)


def channel_prior_loss(
    relevance: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    """Nudge each sample's relevance channel-mass toward its class channel prior.

    The mechanical metric is time-degenerate on length-100 windows, so only the
    per-channel relevance total matters; concentrating mass on the channels that
    carry a class's fault energy is the robust, speed-invariant lever for
    mechanical alignment. Soft cross-entropy between the relevance channel
    distribution and the (normalized) class prior — minimized when they match.
    """

    eps = 1e-8
    rel_mass = relevance.sum(dim=2)  # [N, 8]
    rel_dist = rel_mass / rel_mass.sum(dim=1, keepdim=True).clamp_min(eps)
    target = prior[labels]  # [N, 8]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(target * (rel_dist + eps).log()).sum(dim=1).mean()


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
    channel_prior = torch.from_numpy(CHANNEL_PRIOR).to(device)

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
            if config.relevance_weight > 0:
                # One forward pass returning logits + relevance, so the
                # channel-attention gate receives gradients from the relevance
                # regularizer while cross-entropy still sees raw logits.
                logits, relevance = model.forward_train(xb)
                cls_loss = criterion(logits, yb)
                rel_loss = channel_prior_loss(relevance, yb, channel_prior)
                loss = cls_loss + config.relevance_weight * rel_loss
            else:
                loss = criterion(model.classify(xb), yb)
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
