"""Helpers for public GearXAI baseline artifacts."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from gearxai_devkit.submission import create_submission_package


DEFAULT_BASELINE_ZIP = Path("baselines") / "gearxai-baselines-v1.0.0.zip"


def _read_manifest_from_zip(baselines_zip: str | Path) -> tuple[dict[str, Any], set[str]]:
    baselines_zip = Path(baselines_zip)
    with zipfile.ZipFile(baselines_zip) as archive:
        names = set(archive.namelist())
        with archive.open("manifest.json") as handle:
            manifest = json.loads(handle.read().decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("baseline manifest.json must be a JSON object")
    return manifest, names


def list_baselines(baselines_zip: str | Path = DEFAULT_BASELINE_ZIP) -> dict[str, Any]:
    """List baseline models that are actually present in the public baseline ZIP."""

    manifest, names = _read_manifest_from_zip(baselines_zip)
    rows = []
    missing = []
    for row in manifest.get("baselines", []):
        onnx_path = row.get("onnx")
        if not onnx_path:
            continue
        summary = {
            "name": row.get("name"),
            "type": row.get("type"),
            "onnx": onnx_path,
            "macro_f1": row.get("macro_f1"),
            "eligible": row.get("score", {}).get("eligible") if isinstance(row.get("score"), dict) else None,
            "valid_competition_submission": row.get("valid_competition_submission"),
        }
        if onnx_path in names:
            rows.append(summary)
        else:
            missing.append(summary)

    return {
        "baselines_zip": str(baselines_zip),
        "count": len(rows),
        "baselines": sorted(rows, key=lambda item: str(item.get("name"))),
        "missing_from_zip": sorted(missing, key=lambda item: str(item.get("name"))),
    }


def package_baseline(
    *,
    name: str,
    data_dir: str | Path,
    output_path: str | Path,
    baselines_zip: str | Path = DEFAULT_BASELINE_ZIP,
    split: str = "dev",
    samples: int = 8,
) -> dict[str, Any]:
    """Create a leaderboard ZIP from a named public baseline."""

    manifest, names = _read_manifest_from_zip(baselines_zip)
    candidates = [row for row in manifest.get("baselines", []) if row.get("name") == name]
    if not candidates:
        raise ValueError(f"Unknown baseline: {name}")
    row = candidates[0]
    onnx_path = row.get("onnx")
    if onnx_path not in names:
        raise ValueError(f"Baseline {name!r} references missing ONNX path: {onnx_path}")

    with tempfile.TemporaryDirectory(prefix="gearxai_baseline_") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(baselines_zip) as archive:
            archive.extract(onnx_path, tmp_dir)
        extracted_model = tmp_dir / onnx_path
        payload = create_submission_package(
            model_path=extracted_model,
            data_dir=data_dir,
            output_path=output_path,
            split=split,
            samples=samples,
        )
    payload["baseline"] = {
        "name": row.get("name"),
        "type": row.get("type"),
        "source_onnx": onnx_path,
    }
    return payload
