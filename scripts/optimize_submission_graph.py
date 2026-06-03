r"""Fold Constant nodes of an existing ONNX submission into initializers.

This is the zero-cost simplicity lever (exp-025): the devkit's ``operator_count``
counts every graph node, and ``Constant`` nodes are pure no-ops that still cost
``1/1000`` of penalty each. Folding them into ``graph.initializer`` is
numerically identical but drops them from the operator count, so simplicity
rises with *zero* change to faithfulness or macro-F1.

The script proves the equivalence: it runs the original and the folded model
through ONNX Runtime on a validation sample and reports the max output diff
(expected 0.0), then writes the optimized model and prints the before/after
simplicity from the real devkit metric.

Run::

    $env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
    uv run --no-sync python scripts\optimize_submission_graph.py `
      --model runs\mech_gate_exp023_focus_local\lowfreq_focus_b1p2_c3p4_o4p2_other0p8\model.onnx `
      --out runs\final3\model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx

from gearxai_workspace.data import load_split
from gearxai_workspace.export import INPUT_NAME, fold_constants_to_initializers


def simplicity(model_path: Path) -> dict:
    from gearxai_devkit.metrics import simplicity_score

    return simplicity_score(str(model_path))


def run_onnx(model_path: Path, windows: np.ndarray) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {INPUT_NAME: windows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--check-n", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    windows, _ = load_split(args.data_dir, "validation")
    sample = np.ascontiguousarray(windows[: args.check_n], dtype=np.float32)

    proto = onnx.load(str(args.model))
    before_nodes = len(proto.graph.node)
    folded = fold_constants_to_initializers(proto)
    after_nodes = len(proto.graph.node)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(args.out))
    print(f"folded {folded} Constant nodes: {before_nodes} -> {after_nodes} nodes")

    # Prove numerical identity on a real validation sample.
    orig_out = run_onnx(args.model, sample)
    new_out = run_onnx(args.out, sample)
    max_diff = max(float(np.abs(a - b).max()) for a, b in zip(orig_out, new_out, strict=True))
    print(f"max |orig - folded| over {len(sample)} windows = {max_diff:.3e}")
    if max_diff != 0.0:
        raise SystemExit(f"Folding changed outputs (max diff {max_diff}); aborting.")

    s_before = simplicity(args.model)
    s_after = simplicity(args.out)
    print(
        f"simplicity {s_before['simplicity_score']:.5f} -> {s_after['simplicity_score']:.5f} "
        f"(ops {s_before['operator_count']} -> {s_after['operator_count']}, "
        f"params {s_before['parameter_count']} -> {s_after['parameter_count']})"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
