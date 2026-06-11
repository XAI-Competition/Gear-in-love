"""Build C1 candidates from the lb-003 readings (narrow2 base, fused BNs).

Decision table from the leaderboard differentials:
- occ a1/eps0.2 over a0.5/eps0.05: +0.0087 hidden mech at -0.0020 faith (P6
  vs S0a) -> adopt.
- exp-023 trio gate: +0.0002 hidden mech (P7 vs S0a) -> drop for good.
- MTF -> rgb_z: +0.008 hidden mech even as a harsh 4.0/0.25 bet (P4); build
  SOFT versions (rgb_z boost only, no suppression) to keep most of the mech
  at a fraction of the faith cost.

All variants load through the fused-BN path (-3 ops, ~-700 params).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from export_occlusion_gate_variants import OcclusionGateModel
from export_relevance_gate_variants import CHANNELS, CLASSES, ones
from export_temperature_variants import install_direct_channel_gate, load_base_model

from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check

NARROW2 = (24, 48, 96)


def mtf_rgbz_gates(boost: float) -> np.ndarray:
    gates = ones()
    gates[CLASSES.index("MTF"), CHANNELS.index("rgb_z")] = boost
    return gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/exp035/narrow2_ins30/model.pt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/c1_final"))
    parser.add_argument("--faith-n", type=int, default=3000)
    parser.add_argument("--faith-seed", type=int, default=34034)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = {
        "c1_occ12": None,  # narrow2 + occ a1/eps0.2 + T8, fused, no static gate
        "c1_occ12_mtf2": mtf_rgbz_gates(2.0),
        "c1_occ12_mtf3": mtf_rgbz_gates(3.0),
        "c1_occ12_mtf4": mtf_rgbz_gates(4.0),
    }
    sample = np.random.default_rng(0).normal(size=(8, 8, 100)).astype(np.float32)
    results = {}
    for name, gates in variants.items():
        base = load_base_model(args.checkpoint, widths=NARROW2)
        if gates is not None:
            install_direct_channel_gate(base, gates)
        model = OcclusionGateModel(base, alpha=1.0, eps=0.2, temperature=8.0)
        out = args.out_dir / name
        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)
        self_check(onnx_path, sample, torch_model=model)
        report = evaluate_onnx(
            onnx_path, data_dir=Path("data/prepared"), n=args.faith_n, seed=args.faith_seed
        )
        report.pop("raw", None)
        results[name] = report
        print(summary_line(name, report))
    (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
