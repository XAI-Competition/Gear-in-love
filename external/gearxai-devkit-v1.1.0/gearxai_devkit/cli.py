"""Command line interface for the GearXAI devkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gearxai_devkit.data import prepare_public_release_windows
from gearxai_devkit.submission import (
    create_submission_package,
    inspect_submission_package,
    preflight_submission,
)


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_preflight(args: argparse.Namespace) -> None:
    payload = preflight_submission(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        samples=args.samples,
        output_path=args.out,
    )
    print_json(payload)


def cmd_package(args: argparse.Namespace) -> None:
    payload = create_submission_package(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.out,
        split=args.split,
        samples=args.samples,
        band_config_path=args.band_config,
        batch_size=args.batch_size,
        readme_path=args.readme,
    )
    print_json(payload)


def cmd_inspect_package(args: argparse.Namespace) -> None:
    payload = inspect_submission_package(
        package_path=args.package,
        data_dir=args.data_dir,
        split=args.split,
        samples=args.samples,
    )
    print_json(payload)
    if not payload["valid"]:
        raise SystemExit(1)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    payload = prepare_public_release_windows(
        windows_dir=args.windows_dir,
        output_dir=args.out,
        splits=args.splits,
    )
    print_json(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gearxai", description="GearXAI participant submission packager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data",
        help="convert the public windows_100 Parquet release into evaluator-ready NPY files",
    )
    prepare.add_argument("--windows-dir", required=True, type=Path, help="path to the released data/windows_100 folder")
    prepare.add_argument("--out", required=True, type=Path, help="prepared output directory")
    prepare.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
        help="public splits to prepare",
    )
    prepare.set_defaults(func=cmd_prepare_data)

    preflight = subparsers.add_parser("preflight", help="run local submission preflight checks")
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--data-dir", type=Path, help="optional prepared split for a stronger local check")
    preflight.add_argument("--split", default="validation")
    preflight.add_argument("--samples", type=int, default=8)
    preflight.add_argument("--out", type=Path)
    preflight.set_defaults(func=cmd_preflight)

    package = subparsers.add_parser("package", help="create the model-only leaderboard ZIP")
    package.add_argument("--model", required=True)
    package.add_argument("--data-dir", type=Path, help="optional prepared split for a stronger local check")
    package.add_argument("--split", default="validation")
    package.add_argument("--samples", type=int, default=8)
    package.add_argument("--out", required=True, type=Path)
    package.add_argument("--band-config", type=Path, help="optional mechanical band config for local scoring")
    package.add_argument("--batch-size", type=int, default=256)
    package.add_argument("--readme", type=Path, help="optional model notes included as README.md")
    package.set_defaults(func=cmd_package)

    inspect_package = subparsers.add_parser(
        "inspect-package",
        help="inspect and validate a model-only leaderboard ZIP",
    )
    inspect_package.add_argument("package", type=Path)
    inspect_package.add_argument("--data-dir", type=Path, help="optional prepared split for a stronger local check")
    inspect_package.add_argument("--split", default="validation")
    inspect_package.add_argument("--samples", type=int, default=8)
    inspect_package.set_defaults(func=cmd_inspect_package)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
