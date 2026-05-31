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

---

## env-001 — 启用 GPU 训练（RTX 4060 / cu126）

- **日期**: 2026-06-01
- **commit**: `14054d6` — feat: enable GPU training, keep ONNX export CPU-only
- **变更**: 用户预装 CUDA 版 torch（`torch==2.10.0+cu126`，pyproject/uv.lock 指向本地 wheel）。
  `TrainConfig.device`（auto/cuda/cpu）+ `resolve_device()`；CLI `--device`；
  整个均衡子集常驻显存训练，`randperm` 放到 device 上；训练结束把最优模型搬回 CPU
  再导出 ONNX，保证提交件 CPU-only。
- **硬件**: RTX 4060 Laptop, 8 GB。
- **验证**: GPU 冒烟训练 `device: cuda`，每 epoch ~0.1–0.2s（CPU 时约 21s，**~100× 加速**）；
  导出 CPU ONNX 通过 devkit `valid: true`，与 torch 输出差异 ~1e-7。ruff/pytest 通过。
- **结论**: 后续实验默认走 GPU，迭代速度大幅提升。

---

## exp-002a — 机械对齐机制分析（无训练，纯探索）

- **日期**: 2026-06-01
- **commit (代码来源)**: `14054d6`
- **目标**: 机械对齐占 40% 但本地不可测。先逆向 devkit 的 `metrics.single_mechanical_alignment`，
  搞清楚到底什么能提分，再决定 relevance 头怎么改。
- **方法**: 复刻 devkit 的 STFT 配置（fs=5120, n_fft=256, hop=64），对验证集做
  逐类频谱分析 + 通道-频带贡献分析（脚本在 `.tmp/`，未入库）。
- **关键发现**:
  1. **时间维退化**：窗口仅 100 点，devkit STFT `nperseg=min(256,100)=100`、`hop=64`
     → **只有 1 个时间帧**。因此 `frame_relevance` 把每个通道的 relevance **在整窗求和**成一个标量，
     **relevance 的时间分布完全不影响机械对齐**——只有「每通道 relevance 总量」起作用。
  2. **机械对齐 = Σ_ch (该通道 relevance 总量) × (该通道能量落在故障频带的比例)**。
     所以**唯一杠杆是通道选择性**：把 relevance 总量分给「频谱能量正好落在该类故障频带」的通道。
  3. 代理指标（用从数据估计的频带配置）验证：单通道集中 |x| (0.354) > 全通道 |x| (0.339)
     > 均匀 (0.322)，证明通道加权确实能提分。
  4. **逐类判别频带**（相对 HEA 的频谱偏离）：CTF/MTF→高频 1480–1740 Hz；RCF→460–620 Hz；
     IRF→840–1024 Hz；CWF→强直流 0 Hz；SWF/BWF/ORF→低频 0–160 Hz。
  5. **逐类主导通道**（判别频带内能量占比 top）：RCF/IRF→`torque`；CTF/MTF→`pgb_y/pgb_x`；
     CWF/ORF→`motor`；符合物理直觉（行星齿轮箱故障在 PGB/扭矩通道更明显）。
- **结论 / 下一步**: 当前 relevance（`softplus(cam)·|x|`）对 8 通道的加权是隐式且均匀偏向高 |x| 的。
  exp-002b 将给 relevance 头加**显式通道注意力**（让模型学每类该信任哪些通道），
  期望在不损失 faithfulness/simplicity 的前提下提升机械对齐。
  ⚠️ 注意：私有频带配置未知，本地仍只能用代理指标参考，不能过拟合到我估计的频带。
