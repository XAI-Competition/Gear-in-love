# GearXAI Participant Devkit

This ZIP contains only the participant-side packager for **GearXAI: An Explainable Neuro-Symbolic Gearbox Fault Diagnosis Challenge**.

Its job is narrow: take your exported ONNX model, validate the required interface, and create the exact ZIP artifact for leaderboard upload.

## What Is Included

- `gearxai_devkit/`: the Python package and `gearxai` command.
- `baselines/onnx/`: ready ONNX baseline models you can package as examples.
- `pyproject.toml`: package metadata and runtime dependencies.
- `README.md`: this short participant guide.

No dataset files, training reports, split audits, organizer tools, tests, or documentation bundles are included in this devkit ZIP.

## Install

From the extracted devkit folder:

```bash
python -m pip install .
```

## Prepare The Public Data

After downloading or cloning the Hugging Face dataset release, convert the public `windows_100` Parquet files once:

```bash
gearxai prepare-data --windows-dir data/windows_100 --out prepared
```

This creates evaluator-ready `train` and `validation` NPY files under `prepared/`. The hidden leaderboard test split is not part of the public dataset and is recomputed by the organizers.

## Required ONNX Interface

Your model must run on CPU with ONNX Runtime and accept vibration windows shaped:

```text
[N, 8, 100]
```

It must return:

- class probabilities shaped `[N, 9]`
- relevance maps shaped `[N, 8, 100]`

## Create The Submission ZIP

Run one command on your exported ONNX model:

```bash
gearxai package --model model.onnx --data-dir prepared --split validation --out submission.zip
```

The command validates the ONNX interface, computes the public validation metric report, and creates the upload artifact. Without `--data-dir`, it can still perform an interface-only package, but the final participant check should use the prepared public validation split.

The generated `submission.zip` is the file to upload for leaderboard scoring.

It contains only:

- `model.onnx`
- `submission.json`
- `validation.json`
- `metrics.json`

Team name, email, institution, country, and team members are entered in the participant form, not in the ZIP.

## Try A Ready Baseline

Before packaging your own model, you can run the same command on a shipped baseline ONNX:

```bash
gearxai package --model baselines/onnx/logic_lstm.onnx --data-dir prepared --split validation --out baseline_submission.zip
```

That creates a valid submission-style ZIP from an already exported model. Your own submission uses the same command with your own `model.onnx`.

## Inspect Before Upload

The local `metrics.json` is a participant-side report on the public validation split. Official leaderboard metrics are recomputed by the organizers on hidden evaluation data.

You can inspect the ZIP before upload:

```bash
gearxai inspect-package submission.zip --data-dir prepared --split validation
```
