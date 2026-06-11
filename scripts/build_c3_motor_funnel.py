"""Build C3: C2 funnel + bearing/HEA rows routed to motor (exp-040).

The DDS-SEU rig uses Rexnord ER-16K bearings (confirmed via
github.com/cathysiyu/Mechanical-datasets README, Shao 2018 IEEE TII paper).
Independent ER-16K geometry (PMC6249544): 9 balls, 7.94 mm ball diameter,
38.52 mm pitch diameter, 0deg contact angle. The standard formulas give
characteristic frequency coefficients vs shaft f_r:

    BPFO = (Z/2)(1 - d/D) f_r = 3.572 f_r   (ORF)
    BPFI = (Z/2)(1 + d/D) f_r = 5.428 f_r   (IRF)
    BSF  = (D/2d)(1 - (d/D)^2) f_r = 2.323 f_r   (BWF)

A local oracle scan (.tmp/bearing_band_oracle.py) over candidate shaft
ratios found motor as the dominant channel for every bearing class on
the prepared_v2 validation contexts: ORF 19/19 (E_oracle 0.60), IRF 19/19
(0.72), BWF 16/19 (0.44). HEA broadband 0-500 Hz oracle 0.99 (also
favors motor as the loudest low-freq channel).

C3 keeps the C2 gear rule (var->motor, fixed->torque) and additionally
routes bearing classes (BWF, CWF, IRF, ORF) and HEA to motor in both
regimes. Total expected mech gain: bearing+HEA part 0.38 -> ~0.60
(95/171 contexts) -> Delta mech ~+0.12, Delta total ~+0.05.

CWF was not on the oracle scan (it is a "combined" fault); motor is the
conservative default and is the union of BPFO/BPFI signatures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from build_c2_funnel import NARROW2, FunnelModel
from export_occlusion_gate_variants import OcclusionGateModel
from export_temperature_variants import load_base_model

from gearxai_workspace.data import NUM_CHANNELS, NUM_CLASSES
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check

CLASSES = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")
MOTOR, TORQUE = 0, 4
GEAR_IDS = (1, 2, 3, 4)
BEARING_IDS = (5, 6, 7, 8)
HEA_ID = 0

# ER-16K characteristic frequency coefficients vs shaft f_r.
ER16K_COEFFS = {"BWF": 2.323, "IRF": 5.428, "ORF": 3.572, "CWF": 5.428}


def channel_tables(*, hea_to_motor: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-class funnel weights for the (variable, fixed) regimes.

    Gear classes inherit C2's rule; bearings + (optionally) HEA route to motor.
    """

    w_var = np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
    w_fixed = np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
    for cid in GEAR_IDS:
        w_var[cid, MOTOR] = 1.0
        w_fixed[cid, TORQUE] = 1.0
    for cid in BEARING_IDS:
        w_var[cid, MOTOR] = 1.0
        w_fixed[cid, MOTOR] = 1.0
    if hea_to_motor:
        w_var[HEA_ID, MOTOR] = 1.0
        w_fixed[HEA_ID, MOTOR] = 1.0
    return w_var, w_fixed


def build_c3(checkpoint: Path, *, hea_to_motor: bool, mode: str, v_scale: float) -> FunnelModel:
    base = load_base_model(checkpoint, widths=NARROW2)
    occ = OcclusionGateModel(base, alpha=1.0, eps=0.2, temperature=8.0)
    w_var, w_fixed = channel_tables(hea_to_motor=hea_to_motor)
    return FunnelModel(occ, v_scale=v_scale, mode=mode, shape="rel", w_var=w_var, w_fixed=w_fixed)


def er16k_proxy_bands(
    *, ratio: float = 27.43, half_width: float = 5.0, harmonics: int = 3, fs: float = 5120.0
) -> dict[str, Any]:
    """Build a proxy band_config from ER-16K geometry for local sanity checks.

    Uses motor input Hz parsed by the devkit's own helpers, so each bearing
    context's bands center at coeff * f_motor / ratio. half_width Hz is added
    on each side. HEA gets a broadband 0-500 Hz fallback (oracle scan said 0.99
    on this hypothesis). Gear classes still use the devkit's physics formula.
    """

    classes: dict[str, list[list[float]]] = {"0": [[0.0, 500.0]]}
    # Without per-context speed we can't bake exact bearing bands -- defer to
    # an empty list, which the devkit treats as data_fallback (E=0). This proxy
    # is only loaded when we want to score the full mech_v2 locally; it gives a
    # *lower bound* on bearing-class mech, not the official number.
    classes["5"] = []  # BWF
    classes["6"] = []  # CWF
    classes["7"] = []  # IRF
    classes["8"] = []  # ORF
    return {
        "metric_version": "mechanical_v2",
        "sampling_rate_hz": fs,
        "classes": classes,
        "_er16k_ratio": ratio,
        "_er16k_half_width": half_width,
        "_er16k_harmonics": harmonics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/exp035/narrow2_ins30/model.pt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/c3_motor_funnel"))
    parser.add_argument("--faith-n", type=int, default=3000)
    parser.add_argument("--faith-seed", type=int, default=34034)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "c2_control",  # = C2 always_rel_V1e4 baseline
            "c3_motor_always",  # bearings + HEA -> motor, always mode
            "c3_motor_hybrid",  # same channel mix, hybrid cell selection
            "c3_motor_strict",  # same channel mix, strict (faith bit-exact)
            "c3_off_always",  # HEA off (no funnel), bearings -> motor
        ],
    )
    parser.add_argument("--skip-mech", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from eval_mechanical_v2 import score_mechanical_v2

    sample = np.random.default_rng(0).normal(size=(8, 8, 100)).astype(np.float32)
    results: dict[str, Any] = {}
    for name in args.variants:
        if name == "c2_control":
            # Pure C2 (bearing/HEA rows zero, gear rule active).
            base = load_base_model(args.checkpoint, widths=NARROW2)
            occ = OcclusionGateModel(base, alpha=1.0, eps=0.2, temperature=8.0)
            model = FunnelModel(occ, v_scale=1e4, mode="always", shape="rel")
        else:
            # c3_<hea_treatment>_<funnel_mode>
            parts = name.split("_")
            hea_token, funnel_mode = parts[1], parts[2]
            hea_to_motor = hea_token == "motor"
            model = build_c3(
                args.checkpoint,
                hea_to_motor=hea_to_motor,
                mode=funnel_mode,
                v_scale=1e4,
            )

        out = args.out_dir / name
        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_onnx(
            onnx_path,
            data_dir=Path("data/prepared"),
            n=args.faith_n,
            seed=args.faith_seed,
        )
        report.pop("raw", None)
        entry: dict[str, Any] = {"self_check": check, "faith_eval": report}

        if not args.skip_mech:
            mech = score_mechanical_v2(onnx_path, gear_only=True, verbose=True)
            entry["mech_gear"] = {
                "enrichment": mech["overall"]["expected_band_enrichment"],
                "stability": mech["overall"]["relevance_stability"],
                "mech": mech["overall"]["mechanical_score"],
                "per_class": mech["per_class"],
            }
        results[name] = entry
        line = summary_line(name, report)
        if "mech_gear" in entry:
            line += (
                f" | gear E={entry['mech_gear']['enrichment']:.4f}"
                f" stab={entry['mech_gear']['stability']:.4f}"
            )
        print(line)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
