# 实验记录 (progress.md)

GearXAI 解释型齿轮箱故障诊断 —— 实验日志。**追加式、按时间顺序**（最新在最下方），
每条实验对应一个 git commit。规范见 [CLAUDE.md](CLAUDE.md) 的
"Experiment log & git discipline" 一节。

打分回顾：macro-F1 ≥ 0.80 为准入门槛；过线后按
`0.40·faithfulness + 0.40·mechanical + 0.20·simplicity` 排名。
机械对齐需主办方私有频带配置，本地恒为 `null`。

复现命令模板：

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
uv run --no-sync python scripts\train_baseline.py --train-per-class <N> --val-per-class <M> `
  --epochs <E> --batch-size <B> --num-threads 16 --out runs\<name>\model.onnx
uv run --no-sync gearxai package --model runs\<name>\model.onnx `
  --data-dir data\prepared --split validation --out runs\<name>\submission.zip
```

---

## exp-001 — CNN + Grad-CAM 基线

- **日期**: 2026-05-31
- **commit (代码来源)**: `73c5e3a` — Initial commit: GearXAI baseline
- **目标**: 建立一个能过 0.80 门槛、可解释、ONNX 可导出的最小可用基线，跑通
  训练 → 导出 → devkit 打包 → 自检全流程。
- **模型**: `GearXAINet` —— 3 个卷积块 (Conv→BN→ReLU→MaxPool，通道 64/128/256，
  核 7/5/3，时间 100→50→25→12) + mean+max 全局池化 + 线性头 + softmax；
  可解释头为前向 Grad-CAM，`relevance = softplus(cam)·|x|`，cam 用常量矩阵乘
  上采样到长度 100（避开 `F.interpolate` 导出崩溃）。约 15 万参数，34 个 ONNX 算子。
- **命令**:
  ```powershell
  uv run --no-sync python scripts\train_baseline.py --train-per-class 30000 `
    --val-per-class 4000 --epochs 35 --batch-size 768 --num-threads 16 `
    --out runs\baseline\model.onnx
  uv run --no-sync gearxai package --model runs\baseline\model.onnx `
    --data-dir data\prepared --split validation --out runs\baseline\submission.zip
  ```
- **配置**: train 30000/类 (270k 窗口)，val 4000/类，35 epoch，batch 768，
  lr 1e-3 (AdamW + cosine)，label_smoothing 0.05，seed 42，CPU 16 线程，约 25 分钟。
- **结果** (完整公开验证集 83,790 窗口，来自 `submission.zip` 的 `metrics.json`):

  | 指标 | 数值 |
  | --- | --- |
  | macro-F1 | **0.9968** （过线，eligible: true）|
  | faithfulness | **0.7077** （deletion AUC 0.216↓ / insertion AUC 0.632↑）|
  | mechanical | `null` （本地无私有频带配置）|
  | simplicity | **0.8363** （34 算子 / 150,249 参数 / 0.58 MB）|

- **产物**: `runs/baseline/submission.zip`（model_sha256
  `855499eb02205864b49be4ac6a1604f84ce67ca61e971b761dc25c4edfb0f8fd`），
  `runs/baseline/model.onnx`、`train_summary.json`。
  注：`runs/` 已 gitignore，不入库。
- **结论 / 下一步**: 与官方 `logic_lstm` 基线（macro-F1≈0.984、faith≈0.70）持平或更优，
  分类与忠实度均有竞争力。最大盲区是机械对齐（占 40%，本地不可测）——
  下一步给可解释头加 STFT 故障特征频带先验，是性价比最高的方向；
  忠实度可通过把 input×gradient 归因蒸馏进可解释头继续提升。
