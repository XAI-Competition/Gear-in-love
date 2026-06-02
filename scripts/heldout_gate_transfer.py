"""Held-out transfer check for relevance-only channel-gate candidates.

The gate sweep (progress.md exp-014..023) tuned every gate strength on the same
public validation split it was scored on, so we cannot tell how much of the
``+0.020`` faithfulness gain is real model signal and how much is fitting the
public split's noise. This script splits the public validation set into two
*disjoint*, class-stratified halves:

* ``A`` — the "selection" half (stand-in for where you would pick a gate).
* ``B`` — the "held-out" half (never used to design the gate).

For each candidate gate it reports the real devkit faithfulness on ``A`` and on
``B`` separately. The questions it answers:

1. Does each candidate's faith gain over ``identity`` survive on ``B``? (real
   signal transfers; a public-noise artefact does not.)
2. Does the *fine-tuned* gate (exp-023) keep its edge over the *coarse* gate
   (exp-018/019) on ``B``, or does the edge collapse — i.e. was the fine knob
   twiddling just fitting public-split noise?

Classification and simplicity are invariant by construction (the gate only
reweights relevance channels), so the only metric of interest is faithfulness.

Run::

    $env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
    uv run --no-sync python scripts\heldout_gate_transfer.py `
      --out .tmp\heldout_gate_transfer.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_relevance_gate_variants as gatelib  # noqa: E402

from gearxai_workspace.data import NUM_CLASSES, load_split  # noqa: E402
from gearxai_workspace.export import export_onnx  # noqa: E402

# Candidates spanning the sweep: identity baseline, the coarse single-knob and
# asymmetric configs (most of the gain, fewest assumptions), and the two
# fine-tuned focus configs that the sweep plateaued on.
DEFAULT_CANDIDATES = [
    "identity",
    "lowfreq_motor_only_2p2",  # exp-018: BWF/CWF/ORF motor=2.2 only
    "bwf1p4_cwf2p2_orf3p0",  # exp-019: asymmetric coarse
    "lowfreq_focus:1.4:3.4:4.2:0.8",  # exp-022 main: focus, other=0.8
    "lowfreq_focus:1.2:3.4:4.2:0.8",  # exp-023 main: fine-tuned plateau pick
]


def stratified_halves(labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into two disjoint class-stratified halves (sorted)."""

    rng = np.random.default_rng(seed)
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    for class_id in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        cut = len(idx) // 2
        a_parts.append(idx[:cut])
        b_parts.append(idx[cut:])
    a = np.sort(np.concatenate(a_parts)).astype(np.int64)
    b = np.sort(np.concatenate(b_parts)).astype(np.int64)
    return a, b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final2/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=Path(".tmp/heldout_gate_transfer.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(".tmp/heldout_gate_onnx"))
    parser.add_argument("--seed", type=int, default=70707)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    windows_mm, labels = load_split(args.data_dir, "validation")
    idx_a, idx_b = stratified_halves(labels, seed=args.seed)
    windows_a = np.ascontiguousarray(windows_mm[idx_a], dtype=np.float32)
    labels_a = labels[idx_a]
    windows_b = np.ascontiguousarray(windows_mm[idx_b], dtype=np.float32)
    labels_b = labels[idx_b]
    print(
        f"validation={len(labels)} -> A={len(labels_a)} B={len(labels_b)} "
        f"(disjoint, stratified, seed={args.seed})"
    )

    presets = gatelib.presets()
    selected = gatelib.resolve_selected_presets(args.candidates, presets)
    base = gatelib.load_base_model(args.checkpoint)
    sample = np.array(windows_a[:8], dtype=np.float32, copy=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for preset in selected:
        model = copy.deepcopy(base)
        gatelib.install_direct_channel_gate(model, preset.gates)
        onnx_path = args.work_dir / f"{preset.name}.onnx"
        export_onnx(model, onnx_path, sample=sample)

        report_a = gatelib.evaluate_with_optional_band(
            onnx_path, windows_a, labels_a, batch_size=args.batch_size
        )
        report_b = gatelib.evaluate_with_optional_band(
            onnx_path, windows_b, labels_b, batch_size=args.batch_size
        )
        m_a = gatelib.flatten_metrics(report_a, n=len(labels_a))
        m_b = gatelib.flatten_metrics(report_b, n=len(labels_b))
        rows.append(
            {
                "name": preset.name,
                "faith_a": m_a["faith"],
                "faith_b": m_b["faith"],
                "macro_f1_a": m_a["macro_f1"],
                "macro_f1_b": m_b["macro_f1"],
                "simplicity": m_a["simplicity"],
            }
        )
        print(
            f"{preset.name:36s} faithA={m_a['faith']:.5f} faithB={m_b['faith']:.5f} "
            f"f1A={m_a['macro_f1']:.4f} f1B={m_b['macro_f1']:.4f}"
        )

    base_a = rows[0]["faith_a"]
    base_b = rows[0]["faith_b"]
    print("\nGain over identity (A=selection, B=held-out):")
    print(f"{'candidate':36s} {'dFaithA':>9s} {'dFaithB':>9s} {'transfer':>9s}")
    for row in rows:
        d_a = row["faith_a"] - base_a
        d_b = row["faith_b"] - base_b
        transfer = (d_b / d_a) if abs(d_a) > 1e-9 else float("nan")
        row["d_faith_a"] = d_a
        row["d_faith_b"] = d_b
        row["transfer_ratio"] = transfer
        print(f"{row['name']:36s} {d_a:+9.5f} {d_b:+9.5f} {transfer:>9.2f}")

    # Selection simulation: if you picked the best gate by faith on A, what do
    # you actually get on the held-out half B?
    non_identity = rows[1:]
    best_on_a = max(non_identity, key=lambda r: r["faith_a"])
    best_on_b = max(non_identity, key=lambda r: r["faith_b"])
    print(
        f"\nbest by A-faith: {best_on_a['name']} "
        f"(A={best_on_a['faith_a']:.5f}, its B={best_on_a['faith_b']:.5f})"
    )
    print(f"best by B-faith: {best_on_b['name']} (B={best_on_b['faith_b']:.5f})")

    out = {
        "seed": args.seed,
        "n_validation": int(len(labels)),
        "n_a": int(len(labels_a)),
        "n_b": int(len(labels_b)),
        "checkpoint": str(args.checkpoint),
        "rows": rows,
        "best_by_a_faith": best_on_a["name"],
        "best_by_b_faith": best_on_b["name"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
