"""Submission ZIP packaging and inspection for GearXAI."""

from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gearxai_devkit import __version__
from gearxai_devkit.constants import NUM_CHANNELS, WINDOW_LENGTH
from gearxai_devkit.data import load_split
from gearxai_devkit.runtime import validate_submission


SCHEMA_VERSION = "gearxai.submission.v1"
MODEL_NAME = "model.onnx"
SUBMISSION_METADATA_NAME = "submission.json"
VALIDATION_NAME = "validation.json"
METRICS_NAME = "metrics.json"
README_NAME = "README.md"
REQUIRED_MEMBERS = {MODEL_NAME, SUBMISSION_METADATA_NAME, VALIDATION_NAME, METRICS_NAME}
ALLOWED_MEMBERS = REQUIRED_MEMBERS | {README_NAME}
FORBIDDEN_METADATA_FIELDS = {
    "country",
    "email",
    "institution",
    "name",
    "team",
    "team_members",
    "team_name",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""

    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def leaderboard_metric_contract() -> dict[str, Any]:
    return {
        "macro_f1": {
            "role": "eligibility_gate",
            "threshold": 0.80,
            "higher_is_better": True,
        },
        "faithfulness": {
            "role": "explainability_component",
            "weight": 0.40,
            "higher_is_better": True,
        },
        "mechanical_alignment": {
            "role": "explainability_component",
            "weight": 0.40,
            "higher_is_better": True,
            "public_devkit_note": "official value is recomputed by organizers with private mechanical band templates",
        },
        "simplicity": {
            "role": "explainability_component",
            "weight": 0.20,
            "higher_is_better": True,
        },
        "ranking": "Submissions first pass the macro-F1 eligibility gate; eligible submissions are ranked by 0.40*faithfulness + 0.40*mechanical_alignment + 0.20*simplicity.",
    }


def _validation_windows(
    *,
    data_dir: str | Path | None,
    split: str,
    samples: int,
) -> tuple[np.ndarray, str, int]:
    if data_dir is None:
        sample_count = max(1, int(samples))
        probe = np.zeros((sample_count, NUM_CHANNELS, WINDOW_LENGTH), dtype=np.float32)
        return probe, "synthetic_probe", sample_count

    split_data = load_split(data_dir, split)
    sample_count = min(samples, len(split_data.windows))
    return split_data.windows[:sample_count], split, int(sample_count)


def unavailable_metrics_report(*, model_path: Path, validation_payload: dict[str, Any]) -> dict[str, Any]:
    from gearxai_devkit.metrics import simplicity_score

    return {
        "schema_version": SCHEMA_VERSION,
        "devkit_version": __version__,
        "created_utc": utc_now(),
        "report_type": "leaderboard_metric_contract",
        "official_leaderboard_note": "Official leaderboard metrics are recomputed by organizers on hidden evaluation data.",
        "local_metrics_computed": False,
        "reason": "No --data-dir was provided, so local label-dependent metrics were not computed.",
        "split": validation_payload["split"],
        "samples": validation_payload["samples"],
        "model_sha256": sha256_file(model_path),
        "metric_contract": leaderboard_metric_contract(),
        "classification": {
            "macro_f1": None,
            "eligibility_threshold": 0.80,
        },
        "faithfulness": {
            "faith_score": None,
            "deletion_auc": None,
            "insertion_auc": None,
        },
        "mechanical": {
            "mechanical_score": None,
            "expected_band_mass": None,
            "noise_stability": None,
            "note": "official mechanical score requires private mechanical band templates",
        },
        "simplicity": simplicity_score(model_path),
        "score": {
            "eligible": None,
            "explainability_score": None,
            "reason": "local macro-F1 and hidden mechanical score unavailable",
        },
    }


def build_metrics_report(
    *,
    model_path: Path,
    validation_payload: dict[str, Any],
    data_dir: str | Path | None,
    split: str,
    band_config_path: str | Path | None,
    batch_size: int,
) -> dict[str, Any]:
    if data_dir is None:
        return unavailable_metrics_report(model_path=model_path, validation_payload=validation_payload)

    from gearxai_devkit.evaluator import evaluate_submission

    report = evaluate_submission(
        model_path=model_path,
        data_dir=data_dir,
        split=split,
        band_config_path=band_config_path,
        batch_size=batch_size,
    )
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "devkit_version": __version__,
            "created_utc": utc_now(),
            "report_type": "local_leaderboard_metrics",
            "official_leaderboard_note": "Official leaderboard metrics are recomputed by organizers on hidden evaluation data.",
            "local_metrics_computed": True,
            "model_sha256": sha256_file(model_path),
            "metric_contract": leaderboard_metric_contract(),
        }
    )
    return report


def metrics_summary(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    classification = metrics_payload.get("classification", {})
    faithfulness = metrics_payload.get("faithfulness", {})
    mechanical = metrics_payload.get("mechanical", {})
    simplicity = metrics_payload.get("simplicity", {})
    score = metrics_payload.get("score", {})
    return {
        "local_metrics_computed": metrics_payload.get("local_metrics_computed"),
        "macro_f1": classification.get("macro_f1"),
        "faith_score": faithfulness.get("faith_score"),
        "mechanical_score": mechanical.get("mechanical_score"),
        "simplicity_score": simplicity.get("simplicity_score"),
        "eligible": score.get("eligible"),
        "explainability_score": score.get("explainability_score"),
        "reason": score.get("reason") or metrics_payload.get("reason"),
    }


def preflight_submission(
    *,
    model_path: str | Path,
    data_dir: str | Path | None = None,
    split: str = "dev",
    samples: int = 8,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a model and return evaluator-facing metadata."""

    model_path = Path(model_path)
    windows, validation_split, sample_count = _validation_windows(
        data_dir=data_dir,
        split=split,
        samples=samples,
    )
    validation = validate_submission(model_path, windows)
    payload = {
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "devkit_version": __version__,
        "split": validation_split,
        "samples": int(sample_count),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "validation": validation,
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def build_submission_metadata(*, model_path: Path, validation_payload: dict[str, Any]) -> dict[str, Any]:
    validation = validation_payload["validation"]
    return {
        "schema_version": SCHEMA_VERSION,
        "devkit_version": __version__,
        "created_utc": utc_now(),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "onnx": {
            "input_name": validation.get("input_name"),
            "probability_output": validation.get("probability_output"),
            "relevance_output": validation.get("relevance_output"),
        },
        "package": {
            "command": "gearxai package",
            "format": "model-only leaderboard artifact",
        },
    }


def create_submission_package(
    *,
    model_path: str | Path,
    output_path: str | Path,
    data_dir: str | Path | None = None,
    split: str = "dev",
    samples: int = 8,
    band_config_path: str | Path | None = None,
    batch_size: int = 256,
    readme_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the exact model-only ZIP accepted by the GearXAI submission flow."""

    model_path = Path(model_path)
    output_path = Path(output_path)
    validation_payload = preflight_submission(
        model_path=model_path,
        data_dir=data_dir,
        split=split,
        samples=samples,
    )
    metadata = build_submission_metadata(model_path=model_path, validation_payload=validation_payload)
    metrics_payload = build_metrics_report(
        model_path=model_path,
        validation_payload=validation_payload,
        data_dir=data_dir,
        split=split,
        band_config_path=band_config_path,
        batch_size=batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, MODEL_NAME)
        archive.writestr(
            SUBMISSION_METADATA_NAME,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        archive.writestr(
            VALIDATION_NAME,
            json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        )
        archive.writestr(
            METRICS_NAME,
            json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        )
        if readme_path is not None:
            archive.write(readme_path, README_NAME)

    return {
        "valid": True,
        "package_path": str(output_path),
        "zip_sha256": sha256_file(output_path),
        "model_sha256": metadata["model_sha256"],
        "schema_version": SCHEMA_VERSION,
        "members": sorted(REQUIRED_MEMBERS | ({README_NAME} if readme_path else set())),
        "metrics": metrics_summary(metrics_payload),
    }


def _safe_zip_members(archive: zipfile.ZipFile) -> tuple[list[str], list[str]]:
    names = archive.namelist()
    errors = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"unsafe ZIP member path: {name}")
    return names, errors


def _load_json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    with archive.open(name) as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def _metadata_has_forbidden_fields(payload: dict[str, Any]) -> list[str]:
    found = []
    for key in payload:
        if key.lower() in FORBIDDEN_METADATA_FIELDS:
            found.append(key)
    return found


def inspect_submission_package(
    *,
    package_path: str | Path,
    data_dir: str | Path | None = None,
    split: str = "dev",
    samples: int = 8,
) -> dict[str, Any]:
    """Inspect a participant ZIP with the same checks used by organizers."""

    package_path = Path(package_path)
    errors: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "package_path": str(package_path),
        "zip_sha256": sha256_file(package_path) if package_path.exists() else None,
        "errors": errors,
    }

    if not package_path.exists():
        errors.append(f"package does not exist: {package_path}")
        return result

    with tempfile.TemporaryDirectory(prefix="gearxai_submission_") as tmp:
        tmp_dir = Path(tmp)
        try:
            with zipfile.ZipFile(package_path) as archive:
                names, member_errors = _safe_zip_members(archive)
                errors.extend(member_errors)
                result["members"] = names

                missing = sorted(REQUIRED_MEMBERS - set(names))
                if missing:
                    errors.append(f"missing required ZIP members: {missing}")

                unexpected = sorted(set(names) - ALLOWED_MEMBERS)
                if unexpected:
                    errors.append(f"unexpected ZIP members: {unexpected}")

                if errors:
                    return result

                metadata = _load_json_member(archive, SUBMISSION_METADATA_NAME)
                validation_json = _load_json_member(archive, VALIDATION_NAME)
                metrics_json = _load_json_member(archive, METRICS_NAME)
                if metadata is None:
                    errors.append("submission.json must be a JSON object")
                    return result
                if validation_json is None:
                    errors.append("validation.json must be a JSON object")
                    return result
                if metrics_json is None:
                    errors.append("metrics.json must be a JSON object")
                    return result

                forbidden = _metadata_has_forbidden_fields(metadata)
                if forbidden:
                    errors.append(f"submission.json contains form/team fields: {forbidden}")

                if metadata.get("schema_version") != SCHEMA_VERSION:
                    errors.append(
                        f"unsupported schema_version: {metadata.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
                    )

                archive.extract(MODEL_NAME, tmp_dir)
                model_path = tmp_dir / MODEL_NAME
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            errors.append(f"invalid submission ZIP: {exc}")
            return result

        model_sha256 = sha256_file(model_path)
        result["model_sha256"] = model_sha256
        result["schema_version"] = metadata.get("schema_version") if "metadata" in locals() else None
        result["submission"] = metadata if "metadata" in locals() else None
        result["packaged_validation"] = validation_json if "validation_json" in locals() else None
        result["packaged_metrics"] = metrics_json if "metrics_json" in locals() else None

        if metadata.get("model_sha256") != model_sha256:
            errors.append("model SHA256 does not match submission.json")
        if validation_json.get("model_sha256") != model_sha256:
            errors.append("model SHA256 does not match validation.json")
        if metrics_json.get("model_sha256") != model_sha256:
            errors.append("model SHA256 does not match metrics.json")

        if errors:
            return result

        try:
            validation_payload = preflight_submission(
                model_path=model_path,
                data_dir=data_dir,
                split=split,
                samples=samples,
            )
            result["validation"] = validation_payload
        except Exception as exc:  # noqa: BLE001 - return JSON-friendly inspection errors.
            errors.append(f"model validation failed: {exc}")
            return result

    result["valid"] = not errors
    return result
