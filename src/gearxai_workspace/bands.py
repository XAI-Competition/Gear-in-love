"""Mechanical-alignment signal analysis (route A — verified facts only).

**Two corrections happened here (exp-012a, exp-012a-fix), so this file now
records only numbers dumped by a single authoritative script run**
(``.tmp/routeA_authoritative.py`` -> ``routeA_facts.json``), not recalled values.

What route A asked
------------------
The official mechanical-alignment metric scores how much relevance lands in
class-specific frequency bands (private config). exp-002a showed the metric is
time-degenerate on 100-sample windows, so the only usable lever is *per-channel
relevance mass*. Route A asked: across the four fixed shaft speeds (20/30/40/50
Hz), what is (a) reconstructable and (b) class-discriminative?

Verified findings (validation split, 800 windows per class-speed; numbers from
``.tmp/routeA_facts.json``)
-----------------------------------------------------------------
Cross-speed correlation of the deviation-from-healthy spectrum, and of the
per-channel energy distribution:

    class  spectrum_corr  channel_corr  dominant_channel(/speed)        stable
    CTF        0.905          0.910      torque torque torque torque     yes
    MTF        0.288          0.957      torque torque torque torque     yes
    RCF        0.499          0.995      torque torque torque torque     yes
    SWF        0.638          0.948      torque torque torque torque     yes
    BWF        0.913          0.819      motor  torque torque torque     no
    CWF        0.990          0.977      pgb_y  pgb_y  pgb_y  motor      no
    IRF        0.634          0.854      torque torque torque torque     yes
    ORF        0.921          0.884      torque torque torque torque     yes

Two decisive conclusions
------------------------
1. **Frequency bands are only partially speed-stable** (spectrum corr ranges
   0.29-0.99, no clean split). Crucially the exp-002a "clear per-class bands"
   do not hold up: MTF/RCF/IRF/SWF correlate only 0.29-0.64 across speed. So a
   fixed-Hz band config is **not reliably reconstructable** for several classes.

2. **Channel identity is stable but NOT class-discriminative.** ``torque`` is the
   dominant channel for **7 of 8 fault classes** (only CWF differs, and CWF's top
   channel is itself speed-unstable). Because torque is simply the highest-energy
   channel for nearly every class, a channel prior would say "torque" for almost
   everything — it carries essentially no information to tell faults apart. This
   is exactly why the exp-002d channel prior cost faithfulness (-0.018) without a
   real mechanical gain: there is no class-discriminative channel signal to learn.

Bottom line
-----------
Route A (and with it B/C, which depend on a trustworthy band or channel prior) is
**falsified**: neither a fixed-Hz band nor a class-discriminative channel prior is
reconstructable locally. Mechanical alignment is confirmed **not safely
optimizable** without the organizers' private band config. The data below is
exported for reference/audit only; do not build a prior from it.
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

# Verified cross-speed stats (from routeA_facts.json; 800 windows/class/speed).
# (spectrum_corr, channel_corr, dominant_channel, dominant_channel_stable)
CROSS_SPEED_STATS: dict[str, tuple[float, float, str, bool]] = {
    "CTF": (0.905, 0.910, "torque", True),
    "MTF": (0.288, 0.957, "torque", True),
    "RCF": (0.499, 0.995, "torque", True),
    "SWF": (0.638, 0.948, "torque", True),
    "BWF": (0.913, 0.819, "torque", False),
    "CWF": (0.990, 0.977, "pgb_y", False),
    "IRF": (0.634, 0.854, "torque", True),
    "ORF": (0.921, 0.884, "torque", True),
}

# torque dominates 7/8 classes -> channel identity is not class-discriminative.
DOMINANT_CHANNEL: dict[str, str] = {c: s[2] for c, s in CROSS_SPEED_STATS.items()}

# No band config is provided: cross-speed analysis shows fixed-Hz bands are not
# reliably reconstructable, and the channel signal is not class-discriminative.
# Do not fabricate a band config from this module.
