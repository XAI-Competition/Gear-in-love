# GearXAI Workspace

Local workspace for the GearXAI IJCAI-ECAI 2026 explainable gearbox fault diagnosis challenge.

## Environment

This repository uses `uv` and a repo-local cache:

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
uv sync --python 3.12
```

Useful commands:

```powershell
uv run python scripts\download_gearxai_assets.py
uv run python scripts\inspect_dataset.py
uv run gearxai package --model external\gearxai-devkit-v1.0.1\baselines\onnx\logic_lstm.onnx --data-dir data\prepared --split validation --out runs\baseline_logic_lstm_submission.zip
```

The devkit currently expects `numpy.trapz`, so this project pins `numpy<2`.

## Official Task Summary

- Task: 9-class gearbox fault diagnosis from vibration time-series windows.
- Input tensor: `[N, 8, 100]`.
- Output tensors: class probabilities `[N, 9]` and relevance maps `[N, 8, 100]`.
- Submission artifact: one CPU-only ONNX model, packaged by the official devkit into `submission.zip`.
- Public splits: train and validation; hidden leaderboard evaluation data is not released.
- Ranking: valid submissions must pass an 80% macro-F1 hidden-test gate, then are ranked by explainability metrics.

See [docs/competition_brief_zh.md](docs/competition_brief_zh.md) for the Chinese investigation notes, dataset details, submission requirements, and local setup status.

## Local Layout

- `data/`: downloaded dataset files and prepared evaluator files.
- `downloads/`: official devkit ZIP and other source downloads.
- `external/`: unpacked third-party packages, including the devkit.
- `notebooks/`: exploration notebooks.
- `scripts/`: repeatable setup, download, and inspection helpers.
- `src/gearxai_workspace/`: project code.
- `runs/`: training logs and experiment outputs.
