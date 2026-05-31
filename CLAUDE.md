# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local workspace for the **GearXAI IJCAI-ECAI 2026 explainable gearbox fault diagnosis challenge**.
The goal is to produce a competition submission, not a general library. The deliverable is a
**single CPU-only ONNX model** that, from a vibration time-series window, outputs *both*:

- class probabilities `[N, 9]` (9-class fault diagnosis), and
- relevance/explainability maps `[N, 8, 100]` (same shape as the input `[N, 8, 100]`).

Ranking is two-stage and this shapes every design decision: a submission must first clear an
**80% macro-F1 hidden-test gate**, and only then is it ranked by an **explainability score**
(faithfulness 40% + mechanistic relevance 40% + simplicity 20%). So classification accuracy is a
pass/fail threshold, and explainability is the actual competitive differentiator — `captum` is a
first-class dependency for this reason, not an afterthought. The official `logic_lstm.onnx`
baseline is the reference floor (validation macro-F1 ≈ 0.98, faithfulness ≈ 0.70, simplicity
≈ 0.91; see the brief).

Dataset: DDS-SEU planetary gearbox, 8 channels @ 5120 Hz, balanced 9-class, 19 operating
conditions; public **train** + **validation** only (hidden leaderboard test is not released).

The Chinese-language investigation notes, dataset details, and submission requirements live in
[docs/competition_brief_zh.md](docs/competition_brief_zh.md) — read it before substantive work.

## Environment & commands

Package manager is **uv** (Python 3.11–3.12). This repo uses a **repo-local uv cache**, so set
`UV_CACHE_DIR` once per PowerShell session before any uv command:

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
uv sync --python 3.12          # create/sync .venv from pyproject + uv.lock
```

**Heads up:** the installed `.venv` has `torch==2.12.0+cpu`, but `pyproject.toml` pins
`torch>=2.7,<2.8` from the cu126 index. Because of that mismatch, plain `uv run …` tries to
re-resolve and download torch on every call. Use **`uv run --no-sync …`** to run against the
existing venv without re-syncing. With this CPU torch, training runs **on CPU** even though a
GPU is present — see *Compute resources* below before scaling up.

```powershell
uv run pytest                            # run the test suite (testpaths = tests/)
uv run pytest tests/test_import.py       # single file
uv run pytest -k <expr>                  # single test by name expression

uv run ruff check .                      # lint (add --fix to autofix)
uv run ruff format .                     # format

# Fetch dataset snapshot + devkit (needs HF login + accepted dataset terms first; the script
# prints `uv run huggingface-cli login` instructions on a 403):
uv run python scripts\download_gearxai_assets.py
uv run python scripts\inspect_dataset.py            # sanity-check the downloaded dataset
```

The `gearxai` CLI is provided by the devkit (the `gearxai-devkit` path dependency). Submission
flow — prepare data, then package, then verify:

```powershell
uv run gearxai prepare-data --windows-dir data\windows_100 --out data\prepared
uv run gearxai package `
  --model external\gearxai-devkit-v1.0.1\baselines\onnx\logic_lstm.onnx `
  --data-dir data\prepared --split validation `
  --out runs\baseline_logic_lstm_submission.zip
uv run gearxai inspect-package runs\baseline_logic_lstm_submission.zip `
  --data-dir data\prepared --split validation
```

pytest is configured to keep its cache and temp dirs under `.tmp/` (`cache_dir`, `--basetemp`),
so test runs don't litter the repo.

## Key constraints (easy to get wrong)

- **`numpy<2` is pinned on purpose.** The devkit calls `numpy.trapz`, which was removed in NumPy 2.
  Do not bump numpy past 1.x or the devkit breaks.
- **Torch is pinned to the CUDA 12.6 wheel index** (`torch>=2.7,<2.8` from the `pytorch-cu126`
  explicit index). Don't replace it with a default-index torch.
- **The devkit is a local path dependency**, not a PyPI package:
  `gearxai-devkit = { path = "external/gearxai-devkit-v1.0.1" }`. It provides the `gearxai` CLI and
  the ONNX baselines under `external/gearxai-devkit-v1.0.1/baselines/`. Its presence depends on
  having run the download/unpack step.
- **Watch the axis order.** The Hugging Face `signal` field is `100 x 8` (time-major), but the
  ONNX model and evaluator expect channels-first `[N, 8, 100]`. `gearxai prepare-data` performs
  this transpose into the evaluator-ready `.npy` files under `data/prepared` — don't feed raw HF
  windows to the model.
- **Ruff**: line-length 100, target `py312`, rule sets `E, F, I, UP, B`. Run before committing.

## Repository layout & data flow

The Python package is `gearxai_workspace` (src-layout under `src/`). Almost everything else is
**generated/downloaded and gitignored** — `data/`, `downloads/`, `external/`, `runs/`, plus
`*.onnx` and `*.zip` are all ignored, so models, datasets, the unpacked devkit, and submission
artifacts are never committed.

End-to-end pipeline:

1. `scripts/download_gearxai_assets.py` (HF repo `edi45/gearxai-dds-seu`) → dataset snapshot into
   `data/hf_snapshot`, unpacked devkit into `external/`, and a `data/download_report.json`.
2. `gearxai prepare-data` → transposes HF windows into evaluator-ready `data/prepared` `.npy`.
3. `scripts/inspect_dataset.py` → eyeball the dataset schema/shape.
4. Training/experiment code → outputs and logs land in `runs/`.
5. Export to a single CPU ONNX model emitting probabilities `[N, 9]` + relevance `[N, 8, 100]`
   (fix output names/shapes so CPU ONNX Runtime can run it).
6. `gearxai package` → bundles the ONNX model into `runs/...submission.zip`; `inspect-package`
   re-checks it on the public validation split (macro-F1 + explainability) before manual upload.

Other dirs: `notebooks/` (exploration), `docs/` (the competition brief).

## Baseline model

A working baseline lives in `src/gearxai_workspace/` (`data.py`, `model.py`, `train.py`,
`export.py`) with the CLI `scripts/train_baseline.py`; see [docs/baseline.md](docs/baseline.md)
for the design rationale tied to the scoring metrics. `GearXAINet` is one `nn.Module` emitting
**both** outputs: a 1D-CNN classifier (3 pooled conv blocks, mean+max global pool, `softmax`)
plus a forward Grad-CAM relevance head (`relevance = softplus(cam) · |x|`, with `cam` upsampled
to length 100 by a constant matmul) — nonnegative, deterministic, exportable without autograd.
It's deliberately small (~150k params, 34 ONNX ops) because **simplicity** is 20% of the score.
On the full public validation split it scores macro-F1 **0.997**, faithfulness **0.708**,
simplicity **0.836** (see `runs/baseline/submission.zip`).

Two scoring facts to keep in mind when changing the model:
- **Faithfulness** (40%) is deletion/insertion AUC against an **all-zero** baseline (data is
  pre-standardized), so relevance should mark high-`|x|` cells at class-relevant times.
- **Mechanical alignment** (40%) needs a private STFT band config the devkit doesn't ship, so
  `gearxai package` reports `mechanical_score: null` locally — it can't be measured here.

Export uses the **legacy** TorchScript exporter (`torch.onnx.export(..., dynamo=False)`): the
installed torch 2.12 defaults to the dynamo path, which needs `onnxscript` (not installed).

## Compute resources

Training is **not** locked to CPU — it only runs on CPU today because the installed `.venv` has
`torch==2.12.0+cpu`. Hardware available:

- **Local:** NVIDIA **RTX 4060 Laptop, 8 GB VRAM** (driver 566.07). Good for fast iteration on
  the small baseline; 8 GB is the constraint — keep batch/model modest. The current baseline
  (~150k params, windows `[8,100]`) is tiny, so even large batches fit easily.
- **Remote (on request):** the user can provision **multiple H20 GPUs** — ample capacity. **Ask
  the user** when a run genuinely needs more than the 4060 (full 737k-window training, large
  sweeps, bigger models, multi-seed studies). Don't assume it silently; request it.

To actually use a GPU you must first install a **CUDA build of torch** (the venv is CPU-only
right now). `pyproject.toml` already pins the `pytorch-cu126` index, so a CUDA wheel is the
intended target — installing it then makes `torch.cuda.is_available()` true. After that:

- move the model and tensors to `cuda` in `train.py` (the loop is currently CPU/`from_numpy`),
- **export ONNX from the CPU copy** (`model.cpu()`), and keep the submission **CPU-only** — the
  evaluator runs ONNX Runtime on `CPUExecutionProvider`, so the deliverable must not require CUDA.
- after switching torch, plain `uv run` re-sync behavior may differ; keep using `--no-sync` and
  verify `torch.cuda.is_available()` before a long run.

Rule of thumb: prototype on the 4060; **request H20s for any heavy/long run** and log which
hardware each experiment used in `progress.md`.

## Experiment log & git discipline (REQUIRED)

Every experiment must be reproducible from a commit. Follow this loop for any run that
produces a model, metrics, or a notable result:

1. **Commit the code first**, before (or together with) the run, so the experiment maps to an
   exact tree. Keep commits small and focused; never let a run's code drift uncommitted.
2. **Record the result in [progress.md](progress.md)** — the running experiment journal. Append a
   new dated entry; do **not** rewrite history. Each entry must include:
   - the **git commit hash** (`git rev-parse --short HEAD`) the run was produced from,
   - date, goal/hypothesis, and the exact command(s) run,
   - the config that matters (subset sizes, epochs, batch size, seed, arch changes),
   - the **hardware** used (CPU / RTX 4060 / H20×N) and rough wall-clock time,
   - the **devkit metrics** (macro-F1, faith, simplicity; mechanical is `null` locally),
   - artifact paths (e.g. `runs/<name>/submission.zip`) and a one-line takeaway / next step.
3. **Commit again after logging**, so `progress.md` and any kept artifacts land in history. Use a
   message like `exp: <short result>` and include the metrics in the body.

Practical rules:
- Get the hash with `git rev-parse --short HEAD`; log the hash the run was *built from* (commit
  code → run → log that hash). If you committed after the run, note both.
- `runs/`, `data/`, `*.onnx`, `*.zip` are gitignored — they are **not** in git, so `progress.md`
  is the durable record of what each run achieved. Reference artifact paths, don't rely on them
  being committed.
- Keep `progress.md` newest-entry-last (append-only chronological). One entry per experiment.
- Commit messages end with the standard `Co-Authored-By` trailer (see repo commit guidance).
