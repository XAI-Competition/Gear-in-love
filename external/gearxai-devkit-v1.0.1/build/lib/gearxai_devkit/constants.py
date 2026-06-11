"""Shared constants for the GearXAI evaluator."""

NUM_CHANNELS = 8
WINDOW_LENGTH = 100
NUM_CLASSES = 9
SAMPLING_RATE_HZ = 5120
MIN_MACRO_F1 = 0.80

CHANNEL_NAMES = (
    "motor_vibration",
    "rgb_vibration_y",
    "rgb_vibration_x",
    "rgb_vibration_z",
    "torque",
    "pgb_vibration_y",
    "pgb_vibration_x",
    "pgb_vibration_z",
)

CLASS_LABELS = (
    "HEA",
    "CTF",
    "MTF",
    "RCF",
    "SWF",
    "BWF",
    "CWF",
    "IRF",
    "ORF",
)

CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_LABELS)}

SCORE_WEIGHTS = {
    "faith": 0.40,
    "mechanical": 0.40,
    "simplicity": 0.20,
}
