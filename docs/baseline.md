# GearXAI Baseline

A compact, CPU-trainable baseline that produces a valid GearXAI submission: one
ONNX graph returning class probabilities `[N, 9]` and a relevance map
`[N, 8, 100]`.

## Why this design

The devkit (`external/gearxai-devkit-v1.0.1`) scores submissions in two stages,
and every design choice below is driven by it (see
[competition_brief_zh.md](competition_brief_zh.md) and the devkit's
`metrics.py`):

| Component | Weight | What the metric actually measures | Baseline response |
| --- | --- | --- | --- |
| macro-F1 | gate (≥0.80) | argmax accuracy, per-class averaged | 1D-CNN classifier → **0.997** on full validation |
| faithfulness | 0.40 | deletion/insertion AUC of top-relevance cells vs a **zero** baseline | relevance = Grad-CAM × `|x|` (marks large-magnitude, class-relevant cells) → **0.708** |
| mechanical alignment | 0.40 | relevance mass inside class-specific STFT frequency bands (private config) | not optimizable locally; energy-gated relevance is a reasonable prior |
| simplicity | 0.20 | `1/(1 + params/1e6 + ops/1000 + size_mb/50)` | small graph: ~150k params, 34 ops, 0.58 MB → **0.836** |

Two facts matter most:

1. **Faithfulness deletes/inserts against zeros.** The prepared windows are
   pre-standardized (`stats.json` → `standardized_channel_mean = [0]*8`), so the
   perturbation baseline is all-zeros. Relevance that points at high-`|x|` cells
   at class-relevant times is exactly what raises insertion AUC / lowers
   deletion AUC.
2. **Mechanical alignment needs a private band config** the participant devkit
   does not ship, so `gearxai package` reports `mechanical_score: null` locally.
   It is still 40% of the hidden score — improving it is the main open lever.

## Architecture

`src/gearxai_workspace/model.py` — `GearXAINet`:

- `BatchNorm1d(8)` input standardization (baked into the export).
- 3 `Conv1d` blocks (widths 64/128/256, kernels 7/5/3), each
  Conv → BN → ReLU → `MaxPool1d(2)`, downsampling time 100 → 50 → 25 → 12.
- Global **mean+max** pool → `Linear(2·256 → 9)` → softmax = `probabilities`.
- **Relevance head (forward Grad-CAM):** weight the final feature map by the
  prob-weighted *mean-pool* half of the head weights to get a per-timestep
  `cam` at length 12, upsample it to 100 with a constant linear-interpolation
  **matmul** (a fixed buffer — avoids `F.interpolate`, whose ONNX `Resize`
  export crashes the legacy exporter), then `relevance = softplus(cam) · |x|`.
  Nonnegative, finite, deterministic, exportable with ordinary ops (no autograd
  at inference).

## Results (local public validation, 83,790 windows)

From `runs/baseline/submission.zip` → `metrics.json`, trained on a balanced
270k-window subset (30k/class) for 35 epochs on CPU (~25 min):

| metric | value | note |
| --- | --- | --- |
| macro-F1 | **0.9968** | clears the 0.80 gate (`eligible: true`) |
| faithfulness | **0.708** | deletion AUC 0.216 ↓, insertion AUC 0.632 ↑ |
| mechanical | `null` | needs the organizers' private band config |
| simplicity | **0.836** | 34 ops, 150k params, 0.58 MB |

For reference the shipped `logic_lstm` baseline scores macro-F1 ≈ 0.984 and
faith ≈ 0.70, so this baseline is competitive on both.

## Run it

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path

# Train (CPU; balanced subset). --train-per-class -1 uses all 737k windows.
# This config reproduces the results above (~25 min on 16 threads).
uv run --no-sync python scripts\train_baseline.py `
  --train-per-class 30000 --val-per-class 4000 --epochs 35 `
  --batch-size 768 --num-threads 16 --out runs\baseline\model.onnx

# Package the leaderboard ZIP + local metric report (macro-F1, faith, simplicity)
uv run --no-sync gearxai package --model runs\baseline\model.onnx `
  --data-dir data\prepared --split validation --out runs\baseline\submission.zip
```

The training script also runs the devkit's `validate_submission` on the export,
so a green run guarantees the interface checks pass before packaging.

## Code map

- `data.py` — load prepared NPYs (memmap), balanced subsampling into RAM.
- `model.py` — `GearXAINet` (classifier + relevance head).
- `train.py` — CPU training loop, macro-F1 eval, keeps best weights.
- `export.py` — ONNX export (legacy exporter, opset 17) + devkit self-check.
- `scripts/train_baseline.py` — CLI tying it together.

## Next steps (to climb the leaderboard)

- **Faithfulness:** distill input×gradient attributions into the relevance head,
  or add a deletion/insertion-style auxiliary loss during training.
- **Mechanical alignment:** add an STFT-band prior so relevance concentrates in
  fault-characteristic frequencies (the 40% lever we can't see locally).
- **Accuracy:** train on more data/epochs (up to the full 737k windows) and add
  light augmentation; the gate is easy but headroom helps robustness.
