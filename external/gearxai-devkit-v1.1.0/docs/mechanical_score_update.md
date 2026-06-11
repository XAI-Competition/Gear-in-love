# Mechanical Score Update

The June 9, 2026 update keeps the published GearXAI score weights and ONNX interface unchanged. It replaces the
original Mechanical calculation, which had limited frequency resolution and a near-constant stability contribution.

## Evaluation

The organizer evaluator:

- groups hidden-test stride-one windows by operating condition and ground-truth class;
- reconstructs deterministic 512-sample contexts with overlap-add;
- overlap-adds submitted relevance maps into the same contexts;
- uses up to two non-overlapping contexts per class-condition group.

Each context uses a Hann STFT with `n_fft=256` and `hop=64`. Expected frequency bands are condition-aware:

- gear-fault bands come from documented gearbox geometry and operating speed;
- healthy and bearing-fault bands use frozen organizer-training bands;
- fixed speeds are parsed from condition names;
- variable speeds are estimated from a bounded motor-vibration spectral ridge.

For each context:

```text
A = relevance-weighted signal energy inside expected bands
B = unweighted signal-energy fraction inside the same bands
E = clip((A - B) / (1 - B), 0, 1)
S = cosine similarity between clean and noisy recomputed relevance
Mechanical_context = E * (0.8 + 0.2 * S)
```

The final Mechanical score is the balanced mean across contexts. The noisy pass adds deterministic Gaussian noise at
`1%` of each window's RMS and reruns the submitted model. Constant or uniformly random relevance does not receive a
stability floor.

Exact hidden-test data and private band masks are not published during the competition.
