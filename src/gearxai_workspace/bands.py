"""Physics-informed fault frequency bands for mechanical-alignment optimization.

Route A deliverable (exp-012a). The official mechanical-alignment metric scores
relevance by how much of it lands in class-specific frequency bands, but the
band config is **private**. We reconstruct a best-effort config and, crucially,
flag per-class confidence.

Method: for each fault class we located the most discriminative STFT frequency
(energy deviation from healthy) **independently at all four fixed shaft speeds**
(20/30/40/50 Hz). Classes whose peak stays put across speeds are driven by a
*structural resonance* (a localized defect rings the casing at a fixed natural
frequency on every mesh impact, regardless of rpm) — so a fixed-Hz band is the
right representation and the estimate is trustworthy. Classes whose peak moves
with speed are order-locked or distributed; a single fixed band can't capture
them, so we widen the band and trust it less.

Cross-speed peak drift (Hz) measured on the validation split:

    CTF 51  MTF 52  RCF 51  IRF 51   -> FIXED-Hz (high confidence)
    BWF 358 (mixed)  SWF 614  CWF 666 -> speed-dependent (low confidence)

Bands are in Hz over the 0..2560 Hz range (5120 Hz / 2). HEA (healthy) has no
fault band. Use ``band_config()`` to get a devkit-compatible dict.
"""

from __future__ import annotations

# Per-class bands as (low_hz, high_hz) lists, with a confidence tag.
# High-confidence bands are centered on the cross-speed-stable resonance with a
# +/- one-bin (~51 Hz) margin widened to capture the sideband cluster.
FAULT_BANDS: dict[str, dict] = {
    "HEA": {"bands": [], "confidence": "n/a"},
    "CTF": {"bands": [(1480, 1740)], "confidence": "high"},   # fixed ~1587-1638
    "MTF": {"bands": [(1560, 1760)], "confidence": "high"},   # fixed ~1638-1690
    "RCF": {"bands": [(440, 620)], "confidence": "high"},     # fixed ~512-563
    "IRF": {"bands": [(860, 1040)], "confidence": "high"},    # fixed ~922-973
    # Lower confidence: peak drifts with speed -> use a wider, hedged band.
    "BWF": {"bands": [(850, 1320)], "confidence": "low"},     # mixed 922-1280
    "SWF": {"bands": [(0, 900)], "confidence": "low"},        # 205-819, wide
    "CWF": {"bands": [(0, 960)], "confidence": "low"},        # 256-922, wide
    "ORF": {"bands": [(0, 660)], "confidence": "low"},        # see exp-002a
}

CLASS_LABELS = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

SAMPLING_RATE_HZ = 5120
N_FFT = 256
HOP_LENGTH = 64


def band_config(*, high_confidence_only: bool = False) -> dict:
    """Return a devkit-compatible band config (``classes`` keyed by int id str).

    With ``high_confidence_only`` the low-confidence classes get an empty band,
    so a relevance prior built from this config only commits to the four
    cross-speed-validated resonance bands and stays neutral elsewhere.
    """

    classes: dict[str, list] = {}
    for label, info in FAULT_BANDS.items():
        idx = str(CLASS_TO_INDEX[label])
        if high_confidence_only and info["confidence"] != "high":
            classes[idx] = []
        else:
            classes[idx] = [list(b) for b in info["bands"]]
    return {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "classes": classes,
        "_note": "best-effort reconstruction; official bands are private",
    }


def high_confidence_classes() -> list[str]:
    return [k for k, v in FAULT_BANDS.items() if v["confidence"] == "high"]
