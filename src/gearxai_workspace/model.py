"""Baseline GearXAI model: 1D-CNN classifier + built-in relevance head.

The competition deliverable is a single ONNX graph that returns *both* class
probabilities ``[N, 9]`` and a relevance map ``[N, 8, 100]``. This module keeps
both in one ``nn.Module`` so the relevance is produced by ordinary forward ops
(exportable to CPU ONNX) rather than a separate attribution pass.

Design choices, tied to how the devkit scores submissions:

* **Classifier** — a small stack of Conv1d blocks that *downsample* time
  (100 -> 50 -> 25 -> 12) while widening channels, then a mean+max global pool
  and a linear head. Downsampling builds a useful hierarchy (and is cheaper per
  sample than full-resolution convs), which is what lets the model clear the
  0.80 macro-F1 gate. The graph stays small, which the *simplicity* metric
  rewards.

* **Relevance** — a forward Grad-CAM. The mean-pool half of the classifier
  weights, weighted by the class probabilities, collapses the final feature map
  into a per-timestep importance ``cam``. We upsample it to length 100 and gate
  it by the per-channel input magnitude ``|x|``. Faithfulness deletes/inserts
  the highest-relevance cells against a **zero** baseline, so marking
  large-magnitude cells at class-relevant times is exactly what it rewards.
  ``softplus`` keeps the map strictly nonnegative and finite.

* **Channel attention** (optional) — a tiny class-conditioned per-channel gate
  multiplies the relevance. The devkit's mechanical-alignment metric is time-
  degenerate on length-100 windows (its STFT has a single frame), so the only
  lever is how much relevance mass each channel receives; this gate lets each
  predicted class shift mass toward the channels carrying its fault-band energy.
  Zero-initialized so it starts as the identity (|x|-only relevance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES, WINDOW_LENGTH

# softplus(_SOFTPLUS_INV_1) == 1.0, so a zero-initialized channel gate starts at
# unit (identity) weighting and the relevance equals the |x|-only baseline.
_SOFTPLUS_INV_1 = 0.5413248538970947


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = NUM_CHANNELS
    num_classes: int = NUM_CLASSES
    widths: tuple[int, ...] = (64, 128, 256)
    kernel_sizes: tuple[int, ...] = (7, 5, 3)
    pool: int = 2
    dropout: float = 0.1
    # Class-conditioned channel attention on the relevance head. The devkit's
    # mechanical-alignment metric is time-degenerate on length-100 windows (its
    # STFT yields a single frame), so the only lever is *how much relevance mass
    # each channel gets*. This adds a tiny [num_classes, in_channels] gate that
    # lets each predicted class emphasize the channels carrying its fault-band
    # energy — without touching the time profile that drives faithfulness.
    channel_attention: bool = True


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool, preserving time length before pooling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class GearXAINet(nn.Module):
    """Joint classifier + relevance network for GearXAI windows."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if len(cfg.widths) != len(cfg.kernel_sizes):
            raise ValueError("widths and kernel_sizes must have the same length.")

        # Per-channel input standardization, learned and baked into the export.
        self.input_norm = nn.BatchNorm1d(cfg.in_channels)

        blocks: list[nn.Module] = []
        in_ch = cfg.in_channels
        for width, kernel in zip(cfg.widths, cfg.kernel_sizes, strict=True):
            blocks.append(ConvBlock(in_ch, width, kernel, cfg.pool))
            in_ch = width
        self.features = nn.Sequential(*blocks)
        self.feat_channels = in_ch

        self.dropout = nn.Dropout(cfg.dropout)
        # Head consumes concatenated [mean-pool, max-pool] features.
        self.head = nn.Linear(2 * in_ch, cfg.num_classes)

        # Class-conditioned channel attention: maps class probabilities to a
        # per-channel gate. Tiny (num_classes x in_channels), so simplicity is
        # barely affected. Initialized to ~uniform so training starts from the
        # |x|-only relevance and learns channel emphasis from there.
        if cfg.channel_attention:
            self.channel_gate = nn.Linear(cfg.num_classes, cfg.in_channels)
            nn.init.zeros_(self.channel_gate.weight)
            nn.init.zeros_(self.channel_gate.bias)
        else:
            self.channel_gate = None

    def _features(self, windows: torch.Tensor) -> torch.Tensor:
        return self.features(self.input_norm(windows))  # [N, C, T']

    def _logits_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([feat.mean(dim=2), feat.amax(dim=2)], dim=1)  # [N, 2C]
        return self.head(self.dropout(pooled))  # [N, 9]

    def classify(self, windows: torch.Tensor) -> torch.Tensor:
        """Return raw class logits ``[N, 9]`` (used for training/eval)."""

        return self._logits_from_features(self._features(windows))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(probabilities[N, 9], relevance[N, 8, 100])``."""

        feat = self._features(windows)  # [N, C, T']
        logits = self._logits_from_features(feat)  # [N, 9]
        probabilities = torch.softmax(logits, dim=1)

        # Forward Grad-CAM using the mean-pool half of the head weights
        # (the part that corresponds to global-average-pooled features).
        mean_weight = self.head.weight[:, : self.feat_channels]  # [9, C]
        class_feat = probabilities @ mean_weight  # [N, C]
        cam = torch.bmm(class_feat.unsqueeze(1), feat).squeeze(1)  # [N, T']

        cam_up = F.interpolate(
            cam.unsqueeze(1), size=WINDOW_LENGTH, mode="linear", align_corners=False
        )  # [N, 1, 100]
        gate = F.softplus(cam_up)  # [N, 1, 100] >= 0
        relevance = gate * windows.abs()  # [N, 8, 100], nonnegative

        if self.channel_gate is not None:
            # Per-channel multiplicative gate, class-conditioned. softplus keeps
            # it positive; centered at 1.0 since channel_gate starts at zero, so
            # the model departs from the |x|-only relevance only as it learns.
            ch_gate = F.softplus(self.channel_gate(probabilities) + _SOFTPLUS_INV_1)
            relevance = relevance * ch_gate.unsqueeze(2)  # [N, 8, 1] broadcast

        return probabilities, relevance


def build_model(config: ModelConfig | None = None) -> GearXAINet:
    return GearXAINet(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
