from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


REPO_ID = "edi45/gearxai-dds-seu"
DEVKIT_FILENAME = "downloads/gearxai-devkit-v1.0.1.zip"


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "data": root / "data",
        "downloads": root / "downloads",
        "external": root / "external",
        "hf_cache": root / ".tmp" / "hf-cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def download_devkit(root: Path, dirs: dict[str, Path]) -> Path:
    archive = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=DEVKIT_FILENAME,
        local_dir=root,
        local_dir_use_symlinks=False,
        cache_dir=dirs["hf_cache"],
    )
    archive_path = Path(archive)
    target = dirs["external"] / "gearxai-devkit-v1.0.1"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(target)
    return archive_path


def download_dataset_snapshot(root: Path, dirs: dict[str, Path]) -> Path:
    target = dirs["data"] / "hf_snapshot"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=target,
        local_dir_use_symlinks=False,
        cache_dir=dirs["hf_cache"],
        ignore_patterns=["downloads/*.zip"],
    )
    return target


def verify_dataset_configs(dirs: dict[str, Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        dataset = load_dataset(
            REPO_ID,
            "windows_100",
            split=split,
            cache_dir=str(dirs["hf_cache"]),
        )
        counts[split] = len(dataset)
    return counts


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dirs = ensure_dirs(root)
    try:
        devkit = download_devkit(root, dirs)
        snapshot = download_dataset_snapshot(root, dirs)
        counts = verify_dataset_configs(dirs)
    except HfHubHTTPError as exc:
        print("Hugging Face access failed.")
        print("Open https://huggingface.co/datasets/edi45/gearxai-dds-seu, log in, accept the terms,")
        print("then run: uv run huggingface-cli login")
        print(f"Original error: {exc}")
        return 2

    report = {
        "repo_id": REPO_ID,
        "devkit": str(devkit),
        "snapshot": str(snapshot),
        "windows_100_counts": counts,
    }
    report_path = root / "data" / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
