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
    # Weight of the occlusion-distillation loss (exp-003b): trains the channel
    # gate to predict causal channel importance (zero a channel -> predicted-class
    # confidence drop), which the faithfulness probe showed is the key lever.
    occlusion_weight: float = 0.0
    # exp-007: additive Gaussian input noise (std, in standardized units) during
    # training. Tests whether augmentation makes the classifier rely on robust,
    # localizable cells and thereby indirectly raises faithfulness. 0 disables it.
    noise_std: float = 0.0
    # exp-008: fraction of time steps randomly zeroed per window during training
    # (time masking). Tests whether forcing the classifier to spread its reliance
    # across time further improves faithfulness. 0 disables it.
    time_mask_frac: float = 0.0
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


@torch.no_grad()
def occlusion_channel_importance(model: GearXAINet, windows: torch.Tensor) -> torch.Tensor:
    """Causal per-channel importance: drop in predicted-class prob when a channel
    is zeroed. Returns a nonnegative ``[N, 8]`` distribution (sums to 1 per row).

    This is the "gold" channel signal from the exp-003a faithfulness probe. It
    costs ``num_channels`` extra forward passes, so it is computed only as a
    distillation target (no grad) for the channel gate, not at inference.
    """

    n, num_ch, _ = windows.shape
    base_probs = torch.softmax(model.classify(windows), dim=1)
    pred = base_probs.argmax(dim=1)
    rows = torch.arange(n, device=windows.device)
    base_conf = base_probs[rows, pred]  # [N]

    importance = torch.zeros(n, num_ch, device=windows.device)
    for ch in range(num_ch):
        occluded = windows.clone()
        occluded[:, ch, :] = 0.0
        conf = torch.softmax(model.classify(occluded), dim=1)[rows, pred]
        importance[:, ch] = (base_conf - conf).clamp_min(0.0)

    total = importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return importance / total


def channel_gate_distill_loss(
    gate: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy pulling the (normalized) channel gate toward the
    occlusion-importance target. ``gate`` is the raw per-channel gate ``[N, 8]``."""

    eps = 1e-8
    gate_dist = gate / gate.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(target * (gate_dist + eps).log()).sum(dim=1).mean()


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
            if config.noise_std > 0:
                xb = xb + torch.randn_like(xb) * config.noise_std
            if config.time_mask_frac > 0:
                # Zero a random fraction of time steps (shared across channels).
                keep = (
                    torch.rand(xb.shape[0], 1, xb.shape[2], device=xb.device)
                    >= config.time_mask_frac
                ).float()
                xb = xb * keep
            optimizer.zero_grad(set_to_none=True)
            use_relevance = config.relevance_weight > 0 or config.occlusion_weight > 0
            if use_relevance:
                # One forward returning logits + relevance + channel gate, so the
                # gate can be trained by the relevance regularizer and/or the
                # occlusion-distillation target while cross-entropy sees logits.
                logits, relevance, ch_gate = model.forward_train(xb)
                loss = criterion(logits, yb)
                if config.relevance_weight > 0:
                    loss = loss + config.relevance_weight * channel_prior_loss(
                        relevance, yb, channel_prior
                    )
                if config.occlusion_weight > 0 and ch_gate is not None:
                    target = occlusion_channel_importance(model, xb)
                    loss = loss + config.occlusion_weight * channel_gate_distill_loss(
                        ch_gate, target
                    )
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
