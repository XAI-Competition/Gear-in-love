"""Public devkit for the GearXAI competition."""

from gearxai_devkit.constants import (
    CHANNEL_NAMES,
    CLASS_LABELS,
    MIN_MACRO_F1,
    NUM_CHANNELS,
    NUM_CLASSES,
    SCORE_WEIGHTS,
    WINDOW_LENGTH,
)

__version__ = "1.0.1"

__all__ = [
    "CHANNEL_NAMES",
    "CLASS_LABELS",
    "MIN_MACRO_F1",
    "NUM_CHANNELS",
    "NUM_CLASSES",
    "SCORE_WEIGHTS",
    "WINDOW_LENGTH",
    "__version__",
]
