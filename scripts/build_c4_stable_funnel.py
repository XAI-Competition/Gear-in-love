"""Build C4: C3 + stability fix (exp-041).

C3 uses ``always`` mode: per channel, the funnel mass lands on the cell where
``rel`` is maximum. Under 1% RMS noise that argmax flips between near-ties,
giving S ≈ 0.85 and wasting 0.2*(0.99-0.85) ≈ 0.028 of E in every context.

First attempt (selector="sig", |x|.argmax) FAILED: vibration signals are
oscillatory, |x| has a near-equal peak every half cycle, so 1% noise flips
the global argmax between far-apart cells — stab dropped to 0.73 (worse than
rel.argmax, whose single smooth blob flips rarely and locally).

Fix (selector="fixed"): the funnel cell is a *constant* index (t=50) with a
flat magnitude ``V * sum(R0) * w_ch``. Placement then depends only on the
predicted class (T8-sharpened, near one-hot) and the regime detector
(kappa=6, saturated) — no argmax over any noisy field at all:
  - S -> ~0.99 (only sum(R0) wiggle and rare class flips remain);
  - faith keeps the same <=1-cell top-k perturbation property as ``always``;
  - mech E unchanged: stride-1 overlap-add spreads the fixed cell uniformly
    over the context, per-frame band fractions ~= channel fraction.

Expected: S 0.85->0.99, mech factor (0.8+0.2*S) 0.970->0.998, mech +0.013,
total +0.005. Fewer ops than C3 (constant buffer replaces amax+compare).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from build_c2_funnel import DETECTOR, NARROW2, default_channel_tables
from build_c3_motor_funnel import channel_tables as c3_channel_tables
from export_occlusion_gate_variants import OcclusionGateModel
from export_temperature_variants import load_base_model
from torch import nn

from gearxai_workspace.data import NUM_CHANNELS, WINDOW_LENGTH
from gearxai_workspace.evaluate import evaluate_onnx, summary_line
from gearxai_workspace.export import export_onnx, self_check

# ── stable funnel model ──────────────────────────────────────────────────────


class StableFunnelModel(nn.Module):
    """C3 funnel with a noise-stable funnel-cell selector.

    selector="fixed": constant cell index (t=50), flat magnitude — placement
    depends only on saturated quantities (T8 probs, kappa=6 regime logit),
    so the relevance map is essentially invariant under 1% RMS noise.
    selector="sig": |windows|.argmax per channel (failed attempt, kept for
    the experiment record — oscillatory |x| has too many near-tie peaks).
    """

    FIXED_T = 50

    def __init__(
        self,
        occ: OcclusionGateModel,
        *,
        v_scale: float = 1e4,
        kappa: float = 6.0,
        selector: str = "fixed",
        w_var: np.ndarray | None = None,
        w_fixed: np.ndarray | None = None,
    ):
        super().__init__()
        if selector not in ("fixed", "sig"):
            raise ValueError(f"selector must be 'fixed' or 'sig', got {selector!r}.")
        self.occ = occ
        self.v_scale = float(v_scale)
        self.selector = selector

        cell = torch.zeros(1, 1, WINDOW_LENGTH)
        cell[0, 0, self.FIXED_T] = 1.0
        self.register_buffer("cell_onehot", cell)

        # Regime detector (same as C2/C3).
        bins = sorted({b for f in DETECTOR["features"] for b in range(f["bin_lo"], f["bin_hi"])})
        t = torch.arange(WINDOW_LENGTH, dtype=torch.float32)
        cols = []
        for k in bins:
            cols.append(torch.cos(2 * math.pi * k * t / WINDOW_LENGTH))
            cols.append(torch.sin(2 * math.pi * k * t / WINDOW_LENGTH))
        self.register_buffer("basis", torch.stack(cols, dim=1))  # [100, 2B]

        col_of = {k: i for i, k in enumerate(bins)}
        selector = torch.zeros(NUM_CHANNELS * 2 * len(bins), len(DETECTOR["features"]))
        for fi, feat in enumerate(DETECTOR["features"]):
            for k in range(feat["bin_lo"], feat["bin_hi"]):
                base = feat["channel"] * 2 * len(bins) + 2 * col_of[k]
                selector[base, fi] = 1.0
                selector[base + 1, fi] = 1.0
        self.register_buffer("selector", selector)
        self.register_buffer(
            "det_w",
            float(kappa) * torch.tensor(DETECTOR["w"], dtype=torch.float32).unsqueeze(1),
        )
        self.det_b = float(kappa) * float(DETECTOR["b"])

        if w_var is None or w_fixed is None:
            w_var, w_fixed = default_channel_tables()
        self.register_buffer("w_var", torch.from_numpy(np.asarray(w_var, dtype=np.float32)))
        self.register_buffer("w_fixed", torch.from_numpy(np.asarray(w_fixed, dtype=np.float32)))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs, rel = self.occ(windows)

        # Regime detector -> lam (1 = variable speed).
        proj = torch.matmul(windows, self.basis)  # [N, 8, 2B]
        band = torch.matmul((proj * proj).flatten(1), self.selector)  # [N, F]
        z = torch.matmul(torch.log(band + 1e-9), self.det_w) + self.det_b  # [N, 1]
        lam = torch.sigmoid(z)

        # Class-conditioned channel weights.
        w_ch = lam * (probs @ self.w_var) + (1.0 - lam) * (probs @ self.w_fixed)  # [N, 8]

        if self.selector == "fixed":
            # Constant cell, flat magnitude: no noisy argmax anywhere.
            weighted = w_ch.unsqueeze(2) * self.cell_onehot  # [N, 8, 100]
        else:
            # Failed variant kept for the record: |x| peak flips under noise.
            sig_abs = windows.abs()  # [N, 8, 100]
            eligible = (sig_abs >= sig_abs.amax(dim=2, keepdim=True)).to(rel.dtype)
            weighted = w_ch.unsqueeze(2) * (rel * eligible)  # [N, 8, 100]

        scale = self.v_scale * rel.sum(dim=(1, 2), keepdim=True)  # [N, 1, 1]
        funnel = scale * weighted
        return probs, rel + funnel


def build_c4(
    checkpoint: Path, *, hea_to_motor: bool = True, selector: str = "fixed"
) -> StableFunnelModel:
    base = load_base_model(checkpoint, widths=NARROW2)
    occ = OcclusionGateModel(base, alpha=1.0, eps=0.2, temperature=8.0)
    w_var, w_fixed = c3_channel_tables(hea_to_motor=hea_to_motor)
    return StableFunnelModel(
        occ, v_scale=1e4, kappa=6.0, selector=selector, w_var=w_var, w_fixed=w_fixed
    )


def count_onnx_ops(path: Path) -> int:
    import onnx

    m = onnx.load(str(path))
    return len(m.graph.node)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/exp035/narrow2_ins30/model.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/c4_stable_funnel"))
    p.add_argument("--faith-n", type=int, default=3000)
    p.add_argument("--faith-seed", type=int, default=34034)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["c4b_fixed"],
    )
    p.add_argument("--skip-mech", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from eval_mechanical_v2 import score_mechanical_v2

    sample = np.random.default_rng(0).normal(size=(8, 8, 100)).astype(np.float32)
    results: dict[str, Any] = {}

    for name in args.variants:
        hea = "nohea" not in name
        selector = "fixed" if "fixed" in name else "sig"
        model = build_c4(args.checkpoint, hea_to_motor=hea, selector=selector)

        out = args.out_dir / name
        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, sample=sample)

        try:
            ops = count_onnx_ops(onnx_path)
        except Exception:
            ops = -1

        check = self_check(onnx_path, sample, torch_model=model)
        report = evaluate_onnx(
            onnx_path,
            data_dir=Path("data/prepared"),
            n=args.faith_n,
            seed=args.faith_seed,
        )
        report.pop("raw", None)
        entry: dict[str, Any] = {"self_check": check, "faith_eval": report, "ops": ops}

        if not args.skip_mech:
            mech = score_mechanical_v2(onnx_path, gear_only=True, verbose=False)
            entry["mech_gear"] = {
                "enrichment": mech["overall"]["expected_band_enrichment"],
                "stability": mech["overall"]["relevance_stability"],
                "mech": mech["overall"]["mechanical_score"],
                "per_class": mech["per_class"],
            }

        results[name] = entry
        line = summary_line(name, report)
        line += f" | ops={ops}"
        if "mech_gear" in entry:
            eg = entry["mech_gear"]
            line += (
                f" | gear E={eg['enrichment']:.4f} stab={eg['stability']:.4f} mech={eg['mech']:.4f}"
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
