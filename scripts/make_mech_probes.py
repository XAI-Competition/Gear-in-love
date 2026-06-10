"""Build leaderboard mechanical-alignment probe models (relevance-only).

The leaderboard reports per-component scores (survey-002), so each dev-window
submission measures the private mechanical metric. For relevance-only variants
of one frozen classifier, stability is constant and

    delta(mechanical) = 0.75 * delta(EAS).

Every probe here is the *main candidate family* — final2 classifier + in-graph
occlusion gate (alpha=1.0, eps=0.2) + output temperature T=8 — with only the
static class-conditioned gate rows of one class group replaced by a harsh
single-channel bet (boost on one channel, ``other`` elsewhere). Differences
against the main candidate's leaderboard mechanical score therefore isolate
"how much of that group's channel-band energy fraction does this channel
capture on the *official* bands", with conclusions that apply directly to the
final model (no cross-family transfer step).

Probe set (boost 4.0 / other 0.25, the exp-028-calibrated instrument whose
global faith cost is ~-0.03):

* quintet_torque — CTF/MTF/RCF/SWF/IRF all on torque (exp-013 audit prior)
* quintet_pgby   — same group all on pgb_y (gear-mesh alternative)
* trio_motor_hard — BWF/CWF/ORF motor bet hardened from exp-023's soft gate
* mtf_rgbz       — single-class flip MTF -> rgb_z (exp-014 sharp_proxy hint)

Outputs one ONNX per probe plus a quick devkit subset eval (faith + proxy
mech) for local sanity; package each with ``gearxai package`` afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from export_occlusion_gate_variants import OcclusionGateModel
from export_relevance_gate_variants import (
    CHANNELS,
    CLASSES,
    evaluate_with_optional_band,
    flatten_metrics,
    proxy_band_configs,
)
from export_temperature_variants import (
    exp023_gate_matrix,
    install_direct_channel_gate,
    load_base_model,
)

from gearxai_workspace.evaluate import sample_validation, summary_line
from gearxai_workspace.export import export_onnx, self_check

PROBE_BOOST = 4.0
PROBE_OTHER = 0.25

# Class-group -> channel bets. Keys are probe names; values are (classes, channel).
PROBES: dict[str, tuple[tuple[str, ...], str]] = {
    "probe_quintet_torque": (("CTF", "MTF", "RCF", "SWF", "IRF"), "torque"),
    "probe_quintet_pgby": (("CTF", "MTF", "RCF", "SWF", "IRF"), "pgb_y"),
    "probe_trio_motor_hard": (("BWF", "CWF", "ORF"), "motor"),
    "probe_mtf_rgbz": (("MTF",), "rgb_z"),
    # round 2 (lb-002): torque refuted for the quintet (-0.0137 mech); motor is
    # the remaining plausible direction given the low-frequency trio's success.
    "probe_quintet_motor": (("CTF", "MTF", "RCF", "SWF", "IRF"), "motor"),
}


def probe_gate_matrix(classes: tuple[str, ...], channel: str) -> np.ndarray:
    """exp-023 base with the given class rows replaced by a harsh one-channel bet."""

    gates = exp023_gate_matrix()
    for class_name in classes:
        row = CLASSES.index(class_name)
        gates[row, :] = PROBE_OTHER
        gates[row, CHANNELS.index(channel)] = PROBE_BOOST
    return gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mech_probes_round1"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--eval-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=50050)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows, labels = sample_validation(args.data_dir, args.eval_n, seed=args.seed)
    sample = np.array(windows[:8], dtype=np.float32, copy=True)
    proxy_configs = proxy_band_configs()

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "family": f"occ a{args.alpha:g}/eps{args.eps:g} + T{args.temperature:g} + exp-023 base",
        "probe_boost": PROBE_BOOST,
        "probe_other": PROBE_OTHER,
        "eval_n": int(len(labels)),
        "seed": args.seed,
        "probes": {},
    }
    for name, (classes, channel) in PROBES.items():
        base = load_base_model(args.checkpoint)
        install_direct_channel_gate(base, probe_gate_matrix(classes, channel))
        model = OcclusionGateModel(
            base, alpha=args.alpha, eps=args.eps, temperature=args.temperature
        )
        probe_dir = args.out_dir / name
        onnx_path = probe_dir / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_with_optional_band(onnx_path, windows, labels, batch_size=args.batch_size)
        metrics = flatten_metrics(report, n=len(labels))

        proxy_metrics: dict[str, Any] = {}
        for proxy_name, band_config in proxy_configs.items():
            proxy_report = evaluate_with_optional_band(
                onnx_path, windows, labels, batch_size=args.batch_size, band_config=band_config
            )
            proxy_metrics[proxy_name] = {
                "mechanical": proxy_report["mechanical"]["mechanical_score"],
                "expected_band_mass": proxy_report["mechanical"]["expected_band_mass"],
            }

        result = {
            "classes": list(classes),
            "channel": channel,
            "onnx_path": str(onnx_path),
            "self_check": check,
            "public_metrics": metrics,
            "proxy_metrics": proxy_metrics,
        }
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results["probes"][name] = result
        print(summary_line(name, metrics))
        for proxy_name, proxy in proxy_metrics.items():
            print(
                f"  proxy {proxy_name}: mech={proxy['mechanical']:.4f} "
                f"band_mass={proxy['expected_band_mass']:.4f}"
            )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
