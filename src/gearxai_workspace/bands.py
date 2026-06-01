"""Mechanical-alignment signal analysis (route A, corrected).

**Correction notice (exp-012a-fix):** an earlier version of this file claimed
per-class *fixed-Hz fault bands* with "high confidence". A rigorous cross-speed
check falsified that. This file now reports the honest finding.

What route A tested
-------------------
The official mechanical-alignment metric scores relevance by how much lands in
class-specific frequency bands (private config). To know whether we can
reconstruct those bands, we measured, independently at the four fixed shaft
speeds (20/30/40/50 Hz), two things per fault class:

1. The **deviation-from-healthy spectrum** — and correlated it across speeds.
2. The **per-channel energy distribution** — and correlated it across speeds.

Findings (validation split, 800 windows per class-speed)
--------------------------------------------------------
* **Frequency structure is NOT speed-stable.** Cross-speed correlation of the
  discriminative spectrum is ~0 for every class (CTF 0.01, MTF 0.12, RCF 0.11,
  IRF 0.08, ...). The "clear bands" seen in exp-002a were an artifact of
  averaging across mixed speeds; split by speed, the structure dissolves. So a
  fixed-Hz band config **cannot be reliably reconstructed** — the route-A
  frequency premise fails. (Physically: fault frequencies are order-locked, they
  scale with rpm; and a 100-sample single-STFT-frame window can't localize them.)

* **Channel identity IS speed-stable.** Cross-speed correlation of the
  per-channel energy distribution is **0.90-0.98**, and 7/8 classes keep the
  same dominant channel at all speeds. This is the one trustworthy,
  speed-invariant mechanical signal (it depends on sensor location + fault type,
  not rpm).

Speed-stable dominant channel per fault (the trustworthy signal):

    CTF -> pgb_y    MTF -> pgb_y    RCF -> torque   SWF -> pgb_y
    BWF -> pgb_z*   CWF -> motor    IRF -> torque   ORF -> motor
    (* BWF flips pgb_z/pgb_x at 40 Hz; still corr 0.90)

Bottom line
-----------
The only reconstructable mechanical signal is channel selectivity, not bands.
And exp-002d already showed that steering relevance toward these channels during
training costs a certain ~0.018 faithfulness for a tiny, uncertain proxy gain
(net negative). So mechanical alignment remains **not safely optimizable
locally** — now with direct cross-speed evidence rather than just the
window-length argument. ``DOMINANT_CHANNEL`` is exported for reference only.
"""

from __future__ import annotations

CLASS_LABELS = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

CHANNEL_NAMES = (
    "motor",
    "rgb_y",
    "rgb_x",
    "rgb_z",
    "torque",
    "pgb_y",
    "pgb_x",
    "pgb_z",
)

# Speed-stable dominant channel per fault class (cross-speed corr 0.90-0.98).
# Reference only — exp-002d showed training toward these channels hurts faith.
DOMINANT_CHANNEL: dict[str, str] = {
    "HEA": "motor",
    "CTF": "pgb_y",
    "MTF": "pgb_y",
    "RCF": "torque",
    "SWF": "pgb_y",
    "BWF": "pgb_z",
    "CWF": "motor",
    "IRF": "torque",
    "ORF": "motor",
}

# Cross-speed stability evidence (mean pairwise correlation across 20/30/40/50 Hz).
CHANNEL_DIST_CROSS_SPEED_CORR = {
    "CTF": 0.98, "MTF": 0.97, "RCF": 0.96, "SWF": 0.95,
    "BWF": 0.90, "CWF": 0.97, "IRF": 0.93, "ORF": 0.92,
}
SPECTRUM_CROSS_SPEED_CORR = {
    "CTF": 0.01, "MTF": 0.12, "RCF": 0.11, "SWF": -0.01,
    "BWF": 0.09, "CWF": 0.01, "IRF": 0.08, "ORF": 0.08,
}

# Frequency bands are intentionally NOT provided: cross-speed analysis shows they
# are not reliably reconstructable. Do not fabricate a fixed-Hz band config.
