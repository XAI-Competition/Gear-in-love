"""Local mechanical_v2 scoring with the official v1.1.0 formula (exp-033).

The June 9 scoring upgrade shipped in devkit v1.1.0: 512-sample overlap-add
contexts, Hann STFT (n_fft=256, hop=64), enrichment over a signal-energy
control, and a relevance noise-stability factor. Gear-fault classes
(CTF/MTF/RCF/SWF = 1..4) take *physics* bands computed from documented
geometry defaults plus the operating speed (fixed speeds parsed from
condition_id, variable speeds from the published motor-channel spectral-ridge
estimator) — so the gear part of mechanical is now locally computable with
the real formula. Healthy/bearing classes use private organizer bands; with
an empty ``classes`` map their contexts score 0, so absolute totals undershoot
the leaderboard but *differences between models* on the gear part are exact.

Outputs the overall local mech plus a per-class breakdown, for calibration
against leaderboard anchor readings (lb-002).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEVKIT_V110 = Path(__file__).resolve().parents[1] / "external" / "gearxai-devkit-v1.1.0"
sys.path.insert(0, str(DEVKIT_V110))

from gearxai_devkit.data import load_split  # noqa: E402
from gearxai_devkit.evaluator import deterministic_noisy_windows  # noqa: E402
from gearxai_devkit.metrics import (  # noqa: E402
    MECHANICAL_METRIC_VERSION,
    mechanical_context_indices,
    mechanical_score,
    select_mechanical_contexts,
)
from gearxai_devkit.runtime import run_submission  # noqa: E402

CLASS_NAMES = ("HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF")


def build_band_config(classes_json: Path | None) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    if classes_json is not None:
        classes = json.loads(classes_json.read_text(encoding="utf-8"))
        if "classes" in classes:  # accept either a bare map or a full config
            classes = classes["classes"]
    return {"metric_version": MECHANICAL_METRIC_VERSION, "classes": classes}


GEAR_CLASS_IDS = (1, 2, 3, 4)  # physics-band classes; locally exact


def score_mechanical_v2(
    model_path: Path,
    *,
    data_dir: Path = Path("data/prepared_v2"),
    split_name: str = "validation",
    band_config: dict[str, Any] | None = None,
    batch_size: int = 256,
    gear_only: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a submission and score it with the official mechanical_v2 formula.

    ``gear_only`` restricts to the physics-band gear-class contexts (the
    locally-exact part) and runs inference only on the windows those contexts
    need (~2.7x faster) — the right mode for optimization sweeps.
    """

    band_config = band_config or build_band_config(None)
    split = load_split(data_dir, split_name)
    y_true = split.labels.astype(np.int64)

    contexts = select_mechanical_contexts(y_true, split.metadata)
    if gear_only:
        contexts = [context for context in contexts if context.class_id in GEAR_CLASS_IDS]
    noisy_indices = mechanical_context_indices(contexts)

    if gear_only:
        # Infer only the windows the selected contexts reference; scatter the
        # relevance back into a full-size array for index-based scoring.
        clean = run_submission(model_path, split.windows[noisy_indices], batch_size=batch_size)
        relevance = np.zeros((len(y_true),) + clean.relevance.shape[1:], dtype=np.float32)
        relevance[noisy_indices] = clean.relevance
    else:
        runtime = run_submission(model_path, split.windows, batch_size=batch_size)
        relevance = runtime.relevance
    noisy_runtime = run_submission(
        model_path,
        deterministic_noisy_windows(split.windows[noisy_indices]),
        batch_size=batch_size,
    )

    def score(selected) -> dict[str, Any]:
        return mechanical_score(
            split.windows,
            relevance,
            y_true,
            split.metadata,
            band_config,
            contexts=selected,
            noisy_indices=noisy_indices,
            noisy_relevance=noisy_runtime.relevance,
        )

    overall = score(contexts)
    per_class: dict[str, Any] = {}
    for class_id in sorted({context.class_id for context in contexts}):
        subset = [context for context in contexts if context.class_id == class_id]
        report = score(subset)
        per_class[CLASS_NAMES[class_id]] = {
            "contexts": len(subset),
            "mech": report["mechanical_score"],
            "enrichment": report["expected_band_enrichment"],
            "stability": report["relevance_stability"],
            "strategies": report["strategy_counts"],
        }
        if verbose:
            print(
                f"  {CLASS_NAMES[class_id]}: mech={report['mechanical_score']:.4f} "
                f"enrich={report['expected_band_enrichment']:.4f} "
                f"stab={report['relevance_stability']:.4f} "
                f"strategies={report['strategy_counts']}"
            )
    if verbose:
        speed_range = [overall["estimated_speed_hz_min"], overall["estimated_speed_hz_max"]]
        print(
            f"OVERALL mech={overall['mechanical_score']:.4f} "
            f"enrich={overall['expected_band_enrichment']:.4f} "
            f"stab={overall['relevance_stability']:.4f} speed_range={speed_range}"
        )
    return {
        "model": str(model_path),
        "gear_only": gear_only,
        "contexts": len(contexts),
        "overall": overall,
        "per_class": per_class,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared_v2"))
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=None,
        help="optional JSON map of fallback-class bands {class_id: [[low,high],...]}",
    )
    parser.add_argument("--gear-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score_mechanical_v2(
        args.model,
        data_dir=args.data_dir,
        split_name=args.split,
        band_config=build_band_config(args.classes_json),
        batch_size=args.batch_size,
        gear_only=args.gear_only,
    )
    result["classes_json"] = str(args.classes_json) if args.classes_json else None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
