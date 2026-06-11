---
pretty_name: GearXAI DDS-SEU PGB Release
license: other
language:
- en
size_categories:
- 1M<n<10M
tags:
- timeseries
- tabular
- vibration-analysis
- predictive-maintenance
- explainable-ai
- gear-fault-diagnosis
configs:
- config_name: windows_100
  default: true
  data_files:
  - split: train
    path: data/windows_100/train/*.parquet
  - split: validation
    path: data/windows_100/validation/*.parquet
- config_name: samples_scaled
  data_files:
  - split: train
    path: data/samples_scaled/train/*.parquet
  - split: validation
    path: data/samples_scaled/validation/*.parquet
---

# GearXAI DDS-SEU PGB Release

This repository contains the public dataset and participant devkit for **GearXAI: An Explainable Neuro-Symbolic Gearbox Fault Diagnosis Challenge**, accepted to the IJCAI-ECAI 2026 Competitions and Challenges Track.

The release provides processed planetary gearbox (PGB) vibration data from the DDS-SEU drivetrain setup, packaged for direct use with Hugging Face Datasets and the GearXAI evaluator. The competition task is 9-class gearbox fault diagnosis from fixed `100 x 8` multichannel vibration windows, with leaderboard submissions packaged as CPU-only ONNX models.

Competition website: [https://gearxai-ijcai-ecai2026.pages.dev/](https://gearxai-ijcai-ecai2026.pages.dev/)

## What To Use

If you are participating in GearXAI, start with the default `windows_100` config.

| Config | Contents | Recommended use |
| --- | --- | --- |
| `windows_100` | fixed `100 x 8` vibration windows | main competition training and validation |
| `samples_scaled` | scaled per-timestep channel rows | inspection, plotting, and custom preprocessing |

The public dataset exposes `train` and `validation` splits. The official leaderboard evaluation split is held out by the organizers and is not published.

## Quick Start

```python
from datasets import load_dataset

train_windows = load_dataset("edi45/gearxai-dds-seu", "windows_100", split="train")
validation_windows = load_dataset("edi45/gearxai-dds-seu", "windows_100", split="validation")
```

## Participant Devkit

Download the current participant devkit:

[gearxai-devkit-v1.0.1.zip](https://huggingface.co/datasets/edi45/gearxai-dds-seu/resolve/main/downloads/gearxai-devkit-v1.0.1.zip?download=true)

The devkit contains:

- `gearxai prepare-data` for converting public Parquet windows to evaluator-ready NPY files
- `gearxai package` for creating the final `submission.zip`
- `gearxai inspect-package` for checking the ZIP before upload
- ready ONNX baselines that participants can package as examples

Recommended local workflow:

```bash
python -m pip install .
gearxai prepare-data --windows-dir data/windows_100 --out prepared
gearxai package --model model.onnx --data-dir prepared --split validation --out submission.zip
gearxai inspect-package submission.zip --data-dir prepared --split validation
```

The generated `submission.zip` is the model artifact for leaderboard upload. Team name, email, institution, country, and team members are collected separately through the participant form on the competition website.

## Release Summary

- Sampling rate: `5120 Hz`
- Channels: `8`
- Window length: `100`
- Fault classes: `9`
- Operating conditions: `19`

| View | Train | Validation | Public total |
| --- | ---: | ---: | ---: |
| `samples_scaled` | 754,281 rows | 100,719 rows | 855,000 rows |
| `windows_100` | 737,352 windows | 83,790 windows | 821,142 windows |

Each operating condition is stored as its own Parquet shard to keep the structure transparent and easy to inspect.

## Data Schema

### `windows_100`

- `signal`: nested list with shape `100 x 8`
- `fault_code`
- `fault_name`
- `condition_id`
- `speed_hz`
- `load_nm`
- `regime`
- `experiment_id`
- `window_index`

### `samples_scaled`

- `channel_1` to `channel_8`
- `fault_code`
- `fault_name`
- `condition_id`
- `speed_hz`
- `load_nm`
- `regime`
- `experiment_id`
- `sample_index`

## DDS Testbed

The data come from the drivetrain dynamics simulator (DDS) testbed. The setup includes a controller, motor, planetary gearbox, parallel/reduction gearbox, acceleration sensors, and brake. During acquisition, three-axis accelerometers are mounted at the input end of the gearboxes and signals are sampled at `5120 Hz`.

![Annotated DDS testbed](assets/dds-testbed.png)

This release focuses on the processed planetary gearbox task used by GearXAI.

## Channel Layout

Each sample has 8 channels:

- `channel_1`: motor vibration
- `channel_2`, `channel_3`, `channel_4`: RGB vibration (`y`, `x`, `z`)
- `channel_5`: torque
- `channel_6`, `channel_7`, `channel_8`: PGB vibration (`y`, `x`, `z`)

## Fault Labels

| Fault code | Fault name | Source label |
| --- | --- | --- |
| `HEA` | healthy | `0Health` |
| `CTF` | chipped tooth fault | `1Chipped` |
| `MTF` | missing tooth fault | `2Miss` |
| `RCF` | root crack fault | `3Root` |
| `SWF` | surface wear fault | `4Surface` |
| `BWF` | ball fault | `5Ball` |
| `CWF` | combination fault | `6Combination` |
| `IRF` | inner race fault | `7Inner` |
| `ORF` | outer race fault | `8Outer` |

![Representative fault specimens](assets/dds-fault-gallery.png)

## Operating Conditions

This release covers:

- fixed speed/load: `20_0`, `30_0`, `30_1`, `30_2`, `30_3`, `30_4`, `30_5`, `40_0`, `50_0`
- variable speed: `Experiment1` to `Experiment10`

The machine-readable condition table is included in [metadata/conditions.parquet](metadata/conditions.parquet).

## Metadata Files

- `metadata/conditions.parquet`: condition definitions and per-condition counts
- `metadata/fault_map.parquet`: fault-code mapping
- `metadata/release_summary.json`: top-level counts and release metadata
- `metadata/split_policy.json`: public split policy

## Scope

This is a processed competition release, not a raw-data mirror.

Included:

- processed planetary gearbox data
- public train/validation splits
- fixed-window benchmark tensors
- condition and label metadata
- participant devkit and ready ONNX baseline models

Not included:

- raw TXT signal dumps
- the original raw archive
- the RGB / parallel gearbox task
- labeled leaderboard evaluation data

## Acknowledgment

This release was made possible thanks to the efforts of **Professor Dr. Ruqiang Yan**. We gratefully acknowledge his work on the DDS platform and dataset, and his role both as a coauthor of the underlying work and as a member of the GearXAI organizing team.

## Terms and License

The Hugging Face license tag is `other` because the repository combines a custom processed-dataset release with MIT-licensed devkit code.

The license boundary is:

- processed Parquet data, metadata, and dataset documentation: custom GearXAI processed dataset terms in [TERMS.md](TERMS.md)
- participant devkit code inside `downloads/gearxai-devkit-v1.0.1.zip`: MIT license as declared by the devkit package and reproduced in [DEVKIT_LICENSE.md](DEVKIT_LICENSE.md)

In practical terms, you may use, copy, and redistribute this processed Hugging Face release for GearXAI participation, academic/nonprofit/industry research, benchmarking, teaching, and model development, provided attribution and these terms are preserved.

This repository does not transfer ownership of upstream DDS materials and does not grant new rights to redistribute or relicense the original raw DDS archive, raw TXT files, or other upstream source materials.

This release is provided as is, without warranties of any kind.

## Citation

If this release is useful in your work, please cite the GearXAI challenge and the supporting papers below.

### GearXAI Challenge

```bibtex
@misc{hogea2026gearxai,
  title = {GearXAI: An Explainable Neuro-Symbolic Gearbox Fault Diagnosis Challenge},
  author = {Hogea, E. and Onchis, D. M. and Ivascu, T. and Yan, R.},
  year = {2026},
  note = {IJCAI-ECAI 2026 Competitions and Challenges Track},
  url = {https://gearxai-ijcai-ecai2026.pages.dev/}
}
```

### LogicLSTM

```bibtex
@article{hogea2024logiclstm,
  title = {LogicLSTM: Logically-driven long short-term memory model for fault diagnosis in gearboxes},
  author = {Hogea, Eduard and Onchis, Darian M. and Yan, Ruqiang and Zhou, Zheng},
  journal = {Journal of Manufacturing Systems},
  volume = {77},
  pages = {892--902},
  year = {2024},
  doi = {10.1016/j.jmsy.2024.10.003},
  url = {https://www.sciencedirect.com/science/article/pii/S0278612524002280}
}
```

### Rule-Guided Transformer

```bibtex
@article{hogea2026ruleguided,
  title = {Rule guided transformers for dynamic knowledge adaptation in rotating machinery fault diagnosis},
  author = {Hogea, Eduard and Onchis, Darian M. and Yan, Ruqiang and Zhou, Zheng},
  journal = {Advanced Engineering Informatics},
  volume = {72},
  pages = {104444},
  year = {2026},
  doi = {10.1016/j.aei.2026.104444}
}
```
