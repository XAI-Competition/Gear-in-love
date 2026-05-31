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

---

## exp-002b — 通道注意力 + 代理机械对齐天花板分析

- **日期**: 2026-06-01
- **commit (代码来源)**: `c01082f` — channel attention head
- **硬件**: RTX 4060（冒烟训练，每 epoch ~0.2s）
- **做了什么**:
  1. 实现类条件**通道注意力门控**（`channel_gate: Linear(9→8)`，softplus，零初始化=恒等，
     仅 +80 参数，simplicity 0.829 几乎无损，导出 devkit valid）。
  2. 加**能量对齐辅助 loss**（`channel_energy_alignment_loss`）训练门控——因为分类路径
     不回传到门控，必须有额外信号。`forward_train` 一次前向同出 logits+relevance。
  3. 跑了**代理机械对齐天花板分析**（用从数据估计的频带配置，非官方私有配置）。
- **关键实证结果**（代理机械对齐，600 验证样本）:

  | relevance 策略 | proxy_mech |
  | --- | ---: |
  | uniform | 0.309 |
  | `\|x\|` only（= exp-001 基线） | 0.325 |
  | oracle 软加权（按频带占比加权 `\|x\|`） | 0.383 |
  | **oracle 单通道（全押频带占比最高的通道）** | **0.563** |

- **结论**（两个重要发现）:
  1. **通道选择性是真实有效的大杠杆**：oracle 单通道把代理机械对齐从 0.325→0.563（+73%）。
     且**集中**比软加权更强（占比型指标偏好把质量压到单一最优通道）。
  2. **我的"能量对齐"辅助 loss 无效**：它把 relevance 通道分布对齐到输入能量分布，
     但 `\|x\|` 本就强烈偏向高能量通道 → 门控学到≈恒等（attn 0.348 ≈ \|x\| 0.348）。
     真正该学的是**类条件、超越能量、且更尖锐（集中）的频带通道偏好**。
- **关键风险 / 待决策**: 要让门控命中频带，监督信号只能来自频带配置；而**官方私有频带未知**，
  本地只有我估计的代理频带 → 直接训练去命中代理频带有**过拟合到错误频带**的风险，
  可能在隐藏测试上不升反降。需要决定：是否依据估计频带做温和的通道集中先验，
  还是保守地不碰、专注 faithfulness/simplicity（确定性收益）。已暂停训练，向用户确认方向。
- **用户决策**: 选择"物理频率公式"方向（用齿轮/轴承运动学推导频带，避免对数据硬估）。

---

## exp-002c — 物理频率先验可行性调研（无训练）

- **日期**: 2026-06-01
- **commit (代码来源)**: `0bcb8ca`
- **目标**: 评估"用物理故障特征频率公式定义机械对齐频带"是否可行、是否稳健。
- **调研发现**:
  1. **台架参数大多未公开**：DDS-SEU = SpectraQuest Drivetrain Diagnostics Simulator，
     2 级行星齿轮箱、**27:1 速比、一级 4 行星 / 二级 3 行星**，但**精确齿数、轴承几何
     （滚珠数/节径/接触角）均未公开** → 无法精确复现官方私有频带（官方频带正是基于这些
     未公开参数）。来源见本条末。
  2. **工况是变速 + 多转速**：验证集 `speed_hz ∈ {20,30,40,50}`（变速工况占 ~53%）。
     推理时 ONNX 只收 `[8,100]` 信号、**看不到转速** → 固定 Hz 频带无法跨工况对齐。
     实测也证实：IRF 峰值频率在 20 Hz 时落 102 Hz、50 Hz 时移到 1740 Hz。
  3. **窗口长度的根本限制（决定性）**：100 样本 = 19.53 ms，频率分辨率仅 **51.2 Hz**；
     在 20 Hz 转速下一个窗口只有 **0.39 转**（不足一圈）→ 低阶轴/齿轮谐波**无法分辨**，
     窗口里只装得下高频齿轮啮合频带。精确阶次/边带分析在 100 点窗口上**物理上不可行**。
- **综合判断（重要）**: 精确物理频带在 100 样本窗口上**既不可行也不必要**。结合 exp-002a
  （机械对齐时间维退化、只看每通道 relevance 总量），机械对齐真正能利用的、**跨工况不变**
  的稳健信号是**通道选择性**（哪些传感器对哪类故障敏感，只取决于传感器位置+故障类型，
  与转速无关）——而这正是 exp-002a 已从数据测出、且有物理直觉支撑的（RCF/IRF→torque，
  CTF/MTF→PGB 振动通道）。
- **结论 / 下一步**: 放弃"精确频带"，改为**稳健的通道-故障先验**：让通道注意力门控学习
  类条件的通道偏好（温和、不过拟合具体 Hz），用 faithfulness/simplicity 守住确定性收益。
  下一步设计 exp-002d 的通道先验正则并做对照训练（含一个 `relevance_weight=0` 的对照组，
  确保不损 faithfulness）。
- **来源**: SEU 数据集说明（github hustcxl/Rotating-machine-fault-data-set、Yxz3930/SEU-datasets）；
  SpectraQuest DDS 产品页（spectraquest.com、mitssolutions.asia）。

---

## exp-002d — 通道先验正则 A/B 对照（GPU，真实 devkit 指标）

- **日期**: 2026-06-01
- **commit (代码来源)**: `0bcb8ca`（+ 工作树内通道先验 loss）
- **硬件**: RTX 4060，每 epoch ~1.2s（GPU；72k 窗口、15 epoch）
- **设计**: 严格 A/B，同一子集（train 8000/类、val 2000/类、15 epoch）：
  - **A** `relevance_weight=0`：带通道注意力结构但门控保持恒等（≡ exp-001 基线）。
  - **B** `relevance_weight=0.3`：启用类条件**通道先验 loss**（`channel_prior_loss`，
    先验矩阵 `CHANNEL_PRIOR[9,8]` 来自 exp-002a + 物理直觉）。
  - 用 devkit `evaluate_submission` 在 3000 验证样本上实测 faithfulness/macro-F1/simplicity，
    外加代理机械对齐。
- **结果**:

  | 指标 | A 无先验 | B 先验0.3 | 变化 |
  | --- | ---: | ---: | ---: |
  | macro-F1 | 0.9881 | 0.9881 | 持平 |
  | **faithfulness** | **0.7083** | **0.6905** | **−0.0178 ❌** |
  | simplicity | 0.8313 | 0.8313 | 持平 |
  | 代理机械对齐 | 0.3428 | 0.3548 | +0.0120（微弱）|

- **结论（决定性，负面）**: 通道先验**得不偿失**。代理机械对齐仅 +0.012，而且是对**我自己估计的**
  代理频带测的——真实私有频带上收益高度不确定，可能更小甚至为负；与此同时 faithfulness
  **确定性损失 0.018**。按竞赛权重 `0.4·faith + 0.4·mech`，即便代理涨幅成立也是
  `0.4×0.012 − 0.4×0.018 = −0.0024` 净负。**验证了 exp-002c 的判断：用确定的 faithfulness
  损失换不确定的机械对齐收益不划算。**
- **决定**: **回退通道先验正则**，保留通道注意力结构（零初始化=恒等，零成本可留作以后扩展）。
  默认 `relevance_weight=0`。exp-002 系列到此收敛：最佳可提交模型仍是 exp-001 路线
  （faithfulness 0.708）。下一步把方向转回**确定性收益**：faithfulness 的 deletion/insertion
  辅助蒸馏，以及 simplicity 的小幅精简。

---

## exp-003a — Faithfulness 多角度探针（固定分类器，无重训）

- **日期**: 2026-06-01
- **commit (代码来源)**: `8933e37`（+ evaluate 模块）
- **硬件**: RTX 4060（探针 + 一个 quick proxy 分类器）
- **目标**: faithfulness（40%，本地**可测**）是确定性收益方向。在固定分类器上复刻 devkit
  deletion/insertion AUC，对比多种 relevance 公式，找最优杠杆。
- **方法**: 复刻 `deletion_insertion_auc`，对同一训练好的分类器测多种 relevance 数组
  （脚本 `.tmp/faith_probe*.py`，未入库）。
- **结果**（800 验证样本）:

  | relevance 策略 | faith | 说明 |
  | --- | ---: | --- |
  | uniform | 0.537 | 下限 |
  | `\|x\|`（当前 baseline 风格） | 0.699–0.707 | 现状 |
  | occ_time × `\|x\|` | 0.667 | **时间维用遮挡反而更差** |
  | gradxinput_ch × `\|x\|` | 0.721 | 梯度近似，有限 |
  | **occ_ch × `\|x\|`（通道因果遮挡 × 输入幅值）** | **0.762–0.812** | **金钥匙** |

- **关键发现**:
  1. **时间维用 `\|x\|` 最好，通道维用因果遮挡重要性最好**：把某通道整段置零、看预测类
     置信掉多少（occ_ch），作为通道权重，再乘时间维的 `\|x\|`，faith 从 0.70→0.76~0.81
     （**+0.06~0.10**，本地实测、收益确定）。
  2. occ_time × `\|x\|`（0.667）比 `\|x\|` 还差 → **不要用遮挡做时间维**。
  3. gradient×input 只能部分逼近 occ_ch（per-sample Spearman 仅 0.66，faith 0.721）→ 梯度近似不够。
- **结论 / 下一步（exp-003b）**: occ_ch 需要 8 次前向，无法塞进单次 ONNX。但它是个 `[N,8]` 量，
  与通道注意力门控输出同形 → **用 occ_ch 作监督目标蒸馏通道门控**，让推理时单次前向就近似出
  高 faith 的通道权重。这把 exp-002 留下的"零成本通道注意力结构"用到了**正确的目标
  （faithfulness 而非机械对齐）**上，且完全本地可测。

---

## exp-003b — Occlusion 蒸馏通道门控（GPU sweep，真实 devkit faith）

- **日期**: 2026-06-01
- **commit (代码来源)**: `c5fd83d` — occlusion distillation
- **硬件**: RTX 4060；含蒸馏时每 epoch ~2.8s（8× 遮挡前向），无蒸馏 ~1.0s
- **设计**: 用因果遮挡通道重要性 `occ_ch[N,8]`（置零某通道→预测类置信掉多少）作监督目标，
  蒸馏 class-conditioned 通道门控（`channel_gate_distill_loss`）。扫 occlusion_weight ∈
  {0, 0.5, 1.0, 2.0}，每个用 evaluate 模块在 4000 验证样本上测**真实 devkit faith**。
- **结果**:

  | occ_weight | macro-F1 | faith | deletion↓ | insertion↑ |
  | ---: | ---: | ---: | ---: | ---: |
  | 0（基线） | 0.9884 | 0.7062 | 0.225 | 0.637 |
  | 0.5 | 0.9719 | 0.7122 | 0.178 | 0.602 |
  | 1.0 | 0.9714 | 0.7140 | 0.178 | 0.605 |
  | 2.0 | 0.9683 | **0.7177** | 0.172 | 0.608 |

- **结论**: occlusion 蒸馏**有效但收益有限**。faith 单调升（0.706→0.718，**+0.012**，
  ×0.4 权重 = +0.0048 解释性分），机制正确：**deletion 大幅改善**（0.225→0.172，门控确实学会
  "哪些通道一删就掉预测"），但 **insertion 同步下降**（0.637→0.608）部分抵消，且 macro-F1
  随权重下滑（0.988→0.968，仍远超 0.80 门槛）。
- **瓶颈分析**: 探针里 per-sample occ_ch 能到 faith 0.81，但这里只到 0.718——差距源于门控是
  **class-conditioned**（输入 `probs[N,9]`，同类样本同权重），只学到类平均通道重要性，
  捕捉不到样本级差异。
- **决定 / 下一步**: 这是**确定的正收益**（不像 exp-002 机械对齐），保留为可选。occ_weight=1.0
  是较好折中（faith +0.008，macro-F1 仅降到 0.971）。exp-003c 候选：给门控加 **per-sample
  通道能量输入** `[N,8]`，逼近 per-sample occ_ch（探针上限 0.81），但要权衡 simplicity。
  先去做 exp-004（simplicity，确定可测），再回头评估 exp-003c。
