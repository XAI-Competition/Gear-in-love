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

---

## exp-004 — Simplicity：宽度 vs 精度权衡（GPU sweep，真实 devkit）

- **日期**: 2026-06-01
- **commit (代码来源)**: `ebfdc7d`
- **硬件**: RTX 4060，每 epoch 0.5–1.0s
- **目标**: simplicity（20%，本地可测）。参数 breakdown 显示 ~96% 参数在卷积层
  （baseline 150k 中 144k）。减窄能大幅提 simplicity，问题是 macro-F1 不能破 0.80 门槛。
- **方法**: 训 5 个宽度配置，evaluate 模块在 4000 验证样本测真实 macro-F1/faith/simplicity。
- **结果**:

  | 配置 | macro-F1 | faith | simplicity | params |
  | --- | ---: | ---: | ---: | ---: |
  | baseline (64,128,256) | 0.9898 | 0.7084 | 0.8313 | 150k |
  | **narrow (32,64,128)** | **0.9697** | 0.7058 | **0.9220** | 40k |
  | narrow2 (24,48,96) | 0.9698 | 0.6887 | 0.9374 | 24k |
  | tiny (16,32,64) | 0.9440 | 0.6880 | 0.9492 | 12k |
  | 2block (32,64) | 0.9311 | 0.6998 | 0.9495 | 14k |

- **结论**: **`narrow (32,64,128)` 是明确甜点**。simplicity 0.831→**0.922（+0.091，×0.2 = +0.018
  解释性分）**，macro-F1 仍 0.970（远超门槛），faith 几乎不变（0.706）。这是**确定、可测的
  正收益，比 exp-003b 的 faith 收益还大**。更窄（narrow2/tiny）开始掉 faith，不划算。
- **决定 / 下一步**: 采用 narrow 作为新默认架构候选。下一步 exp-005：把 **narrow + occlusion
  蒸馏**组合，叠加两个确定收益（simplicity +0.018 与 faith +0.005~0.008），并测 exp-003c
  的 per-sample 能量门控能否进一步提 faith。

---

## exp-005 — 组合 sweep：narrow × 蒸馏 × 能量门控（GPU，真实 devkit）

- **日期**: 2026-06-01
- **commit (代码来源)**: `3b76672`
- **硬件**: RTX 4060；narrow 无蒸馏 ~0.8s/epoch，含蒸馏 ~2.7s/epoch
- **目标**: 把已验证的确定收益组合，找最佳可提交模型。排名用本地可测的
  `expl_partial = 0.4·faith + 0.2·simplicity`（机械项本地未知，已证不可靠优化）。
- **结果**（train 10000/类、20 epoch、eval 5000 验证样本）:

  | 配置 | macro-F1 | faith | simplicity | expl_partial |
  | --- | ---: | ---: | ---: | ---: |
  | **narrow_base** | 0.9793 | **0.7051** | 0.9220 | **0.4664** |
  | narrow_occ1 | 0.9584 | 0.6972 | 0.9220 | 0.4633 |
  | narrow_occ1_energy | 0.9566 | 0.6941 | 0.9169 | 0.4610 |
  | narrow_occ2_energy | 0.9548 | 0.6945 | 0.9169 | 0.4612 |
  | exp-001 baseline（参考） | 0.997 | 0.708 | 0.836 | **0.4504** |

- **决定性结论**:
  1. **`narrow_base` (32,64,128，无蒸馏) 是明确赢家**：expl_partial **0.4664，比 exp-001 baseline
     的 0.4504 高 +0.016**，纯来自 simplicity（0.836→0.922），faith 几乎不变（0.705 vs 0.708），
     macro-F1 0.979 仍远超门槛。
  2. **occlusion 蒸馏在 narrow 架构上一致有害**：所有蒸馏组合 faith（0.694–0.697）都低于
     narrow_base（0.705），能量门控也无帮助还多耗算子（simplicity 0.922→0.917）。
     → **否决 exp-003b/c 的蒸馏路线在小模型上的应用**（蒸馏伤分类与 insertion，小模型更敏感）。
- **决定 / 下一步**: 最优可提交模型 = **narrow (32,64,128)，无蒸馏、无能量门控**。
  下一步：用此配置**全量训练**（更大子集 + 更多 epoch）产出最终 `runs/narrow/submission.zip`，
  作为新的最佳提交，替代 exp-001 baseline。蒸馏代码保留为 opt-in（默认关）。

---

## exp-006 — Relevance 时间维公式探针（固定分类器，CPU，与 GPU 训练并行）

- **日期**: 2026-06-01
- **commit (代码来源)**: `09cbd6c`
- **目标**: exp-003a 显示时间维 `\|x\|` 优于遮挡。本探针在固定 baseline ONNX 上扫时间维的
  各种变换，看 faith 能否再升。与全量训练并行（CPU ORT，不占 GPU）。
- **结果**（1000 验证样本）:

  | 时间维 relevance | faith | 说明 |
  | --- | ---: | --- |
  | `\|x\|`（基线） | 0.7054 | 现状 |
  | `\|x\|^0.5 / ^1.5 / ^2 / ^3` | 0.7054 | **完全相同** |
  | `\|x\|`-rank | 0.7054 | **完全相同** |
  | `\|x\|` channel-normalized | 0.6231 | 更差 |
  | `\|x\|` top-4 能量通道 | 0.6249 | 更差 |

- **关键发现（决定性）**: **任何单调变换给出完全相同的 faith**。原因：devkit `topk_mask` 只按
  relevance **排序**取 top-k cell，单调变换不改排序 → faith 不变。即 **relevance 的数值无关，
  只有 cell 的相对排序重要**。锐化/展平/rank 都无用。channel-normalized 和 top-4 更差，
  因为破坏了**跨通道幅值排序**（把低能量通道 cell 排到高能量通道前）。
- **结论**: **时间维 relevance 在固定 `\|x\|` 通道权重下已是最优，无提升空间**。faith 天花板由
  "哪些 cell 进 top-k"决定，而 `\|x\|` 的全局排序已最优。结合 exp-003（蒸馏在小模型有害）、
  exp-005（narrow 最优），**faithfulness 在当前 forward-Grad-CAM×`\|x\|` 框架下已收敛到
  ~0.705–0.708**。要再升需换框架（如改变分类器使其决策更依赖稀疏可定位的 cell），属高风险大改。
- **决定 / 下一步**: faithfulness 这条线收敛。当前最优可提交模型 = narrow（exp-005，
  expl_partial 0.466，全量训练中产出 `runs/narrow/submission.zip`）。下一步可探索的新方向：
  (a) 全量 737k 数据是否进一步提 macro-F1 裕度；(b) 训练角度（增强/正则）对 faith 的间接影响。

---

## exp-005-final — narrow 全量训练最终提交（GPU 全量 + 全验证集打包）

- **日期**: 2026-06-01
- **commit (代码来源)**: `0efe392`（narrow 为默认架构）
- **硬件**: RTX 4060，全量 360k 训练窗口，40 epoch，~2.5s/epoch（约 2 分钟训练）
- **命令**:
  ```powershell
  uv run --no-sync python scripts\train_baseline.py --train-per-class 40000 `
    --val-per-class 4000 --epochs 40 --batch-size 768 --device cuda --out runs\narrow\model.onnx
  uv run --no-sync gearxai package --model runs\narrow\model.onnx `
    --data-dir data\prepared --split validation --out runs\narrow\submission.zip
  ```
- **最终指标（完整公开验证集 83,790 样本，来自 submission.zip 的 metrics.json）**:

  | 指标 | narrow 最终 | exp-001 baseline | 变化 |
  | --- | ---: | ---: | ---: |
  | macro-F1 | **0.9914** | 0.9968 | −0.005（仍远超 0.80 门槛）|
  | faithfulness | **0.7022** | 0.7077 | −0.006 |
  | simplicity | **0.9220** | 0.8363 | **+0.086** |
  | expl_partial (0.4·f+0.2·s) | **0.4653** | 0.4504 | **+0.0149** |

- **产物**: `runs/narrow/submission.zip`（valid、eligible；model_sha256
  `341fbd70…`，39,689 参数、41 算子）、`runs/narrow/model.onnx`、`model.pt`、`train_summary.json`。
- **结论**: **narrow 是本轮探索的最佳可提交模型**，本地可测解释性比 exp-001 baseline **+0.015**
  （纯 simplicity 贡献，faith 几乎不变，macro-F1 仍 0.991）。这是经 6 轮系统实验
  （exp-002 机械对齐否决、exp-003 蒸馏小模型否决、exp-004 simplicity 甜点、exp-005 组合验证、
  exp-006 时间维已最优）收敛得到的确定性最优。

## 本轮长程探索小结（截至 0efe392）

- **机械对齐（40%）**：本地不可靠（窗口 100 点 < 1 转、变速工况、私有频带未知），
  时间维退化只剩通道杠杆；通道先验经 A/B 实测净负（exp-002）。**放弃**。
- **faithfulness（40% 中本地可测部分）**：占主导的是 cell 排序，`|x|` 全局排序已最优（exp-006）；
  通道遮挡蒸馏在大模型微弱正、小模型负（exp-003/005）。**收敛于 ~0.70**。
- **simplicity（20%）**：**narrow (32,64,128) 是明确赢家，+0.086**（exp-004/005）。**已采纳为默认**。
- **基础设施**：GPU 训练（~18× 加速）、可复用 `evaluate` 模块（真实 devkit 指标）。
- **下一步候选**（尚未做）：全量 737k 数据训练、训练期数据增强/正则对 faith 的间接影响、
  更激进的架构改动（高风险）。当前确定性收益已基本挖尽。

---

## exp-007 — 输入噪声增强对 faithfulness 的间接影响（GPU sweep，真实 devkit）

- **日期**: 2026-06-01
- **commit (代码来源)**: `4906950` — noise augmentation
- **硬件**: RTX 4060，narrow，~0.9s/epoch
- **假设**: exp-006 证明**固定分类器**下时间维 relevance 已最优；但换一个**噪声鲁棒**的
  分类器，可能让决策更依赖稳健可定位的 cell，从而提高 faith 上限。扫 `noise_std`。
- **结果**（narrow，train 10000/类、20 epoch、eval 5000）:

  | noise_std | macro-F1 | faith | insertion | expl_partial |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.0 | 0.9794 | 0.7065 | 0.637 | 0.4670 |
  | 0.05 | 0.9784 | 0.7085 | 0.643 | 0.4678 |
  | **0.1** | 0.9714 | **0.7144** | 0.655 | **0.4702** |
  | 0.2 | 0.9505 | 0.7114 | 0.644 | 0.4689 |

- **关键发现（突破）**: 噪声增强**确实间接提升 faithfulness**。`noise_std=0.1` 是甜点：
  faith **0.7065→0.7144（+0.008）**，insertion 0.637→0.655，macro-F1 仍 0.971（远超门槛）。
  **这修正了 exp-006 的"faith 已收敛"结论**：固定分类器下确实最优，但噪声鲁棒训练抬高了
  faith 天花板（让 relevance 标注的高 |x| cell 对预测更因果）。noise_std=0.2 开始伤分类，过头。
- **决定 / 下一步**: 噪声增强是**真实、可测、确定**的 faith 收益，且与 narrow 的 simplicity 收益
  **正交可叠加**。用 **narrow + noise_std=0.1** 全量训练产出新最佳提交，替代 exp-005-final。

---

## exp-007-final — 最终最佳提交：narrow + noise=0.1（叠加两个确定收益）

- **日期**: 2026-06-01
- **commit (代码来源)**: `0283705`
- **硬件**: RTX 4060，全量 360k 训练窗口，40 epoch，~3s/epoch
- **命令**:
  ```powershell
  uv run --no-sync python scripts\train_baseline.py --train-per-class 40000 `
    --val-per-class 4000 --epochs 40 --batch-size 768 --device cuda --noise-std 0.1 `
    --out runs\final\model.onnx
  uv run --no-sync gearxai package --model runs\final\model.onnx `
    --data-dir data\prepared --split validation --out runs\final\submission.zip
  ```
- **最终指标（完整公开验证集 83,790 样本）**:

  | 指标 | **final (narrow+noise)** | narrow | exp-001 baseline |
  | --- | ---: | ---: | ---: |
  | macro-F1 | **0.9926** | 0.9914 | 0.9968 |
  | faithfulness | **0.7219** | 0.7022 | 0.7077 |
  | simplicity | **0.9220** | 0.9220 | 0.8363 |
  | expl_partial (0.4·f+0.2·s) | **0.4732** | 0.4653 | 0.4504 |

- **产物**: `runs/final/submission.zip`（valid + eligible；model_sha256 `7a8c011a…`，
  39,689 参数、41 算子）、`runs/final/model.onnx`、`model.pt`、`train_summary.json`。
- **结论（本轮探索最终交付）**: **两个正交的确定收益成功叠加**——simplicity（narrow，+0.086）
  + faithfulness（噪声增强，+0.014）。最终 expl_partial **0.4732，比 exp-001 baseline 高 +0.023
  （+5.1%）**，macro-F1 仍 0.993 远超 0.80 门槛。这是本轮 7 方向系统探索收敛得到的最佳可提交模型。

## 本轮长程探索总结（最终，截至 0283705）

| 方向 | 实验 | 结论 | 收益 |
| --- | --- | --- | --- |
| GPU 训练基础设施 | env-001 | torch cu126，~18× 加速 | 使能 |
| 机械对齐（40%） | exp-002 | 本地不可靠（窗口<1转/变速/私有频带）；通道先验净负 | 否决 |
| faith 通道遮挡蒸馏 | exp-003 | 大模型微弱正、小模型负 | 否决（小模型） |
| **simplicity（20%）** | exp-004/005 | **narrow (32,64,128) 甜点** | **+0.086 simplicity** |
| faith 时间维公式 | exp-006 | 固定分类器下 \|x\| 已最优（仅排序重要） | 已最优 |
| **faith 训练增强** | exp-007 | **噪声增强间接提 faith，std=0.1 甜点** | **+0.014 faith** |
| **最终叠加** | exp-007-final | narrow+noise=0.1 | **expl_partial 0.4504→0.4732（+0.023）** |

**最佳可提交模型**: `runs/final/submission.zip`（narrow + noise=0.1）。
**剩余候选方向**（未做，预计收益递减）: 全量 737k 数据训练、mixup/时间mask 等其他增强、
relevance 框架级重构（高风险）。确定性收益已基本挖尽。

---

## exp-008 — Time-masking 增强叠加 noise（GPU sweep，真实 devkit）

- **日期**: 2026-06-01
- **commit (代码来源)**: `1e12352` — time-masking augmentation
- **硬件**: RTX 4060，narrow + noise=0.1 基底，~0.9s/epoch
- **假设**: exp-007 证明噪声增强间接提 faith。time-masking（随机置零部分时间步）能否进一步
  迫使分类器分散时间依赖，叠加提升 faith。扫 `time_mask_frac`，基底 noise=0.1。
- **结果**（narrow，train 10000/类、20 epoch、eval 5000）:

  | time_mask | macro-F1 | faith | expl_partial |
  | ---: | ---: | ---: | ---: |
  | 0.0（纯 noise） | 0.9723 | 0.7169 | 0.4712 |
  | 0.1 | 0.9718 | 0.7280 | 0.4756 |
  | **0.2** | 0.9603 | **0.7296** | **0.4762** |
  | 0.3 | 0.9418 | 0.7243 | 0.4741 |

- **关键发现**: time-masking **确实在 noise 基础上进一步叠加提升 faith**。
  `time_mask=0.2` 是甜点：faith 0.7169→**0.7296（+0.013）**，expl_partial **0.4762**（刷新纪录）。
  但 macro-F1 在子集上降到 0.960（mask 越大降越多，0.3 时 0.942）——全量数据通常会拉回。
  mask=0.1（faith 0.728，macro-F1 0.972 更安全）与 0.2 很接近。
- **决定 / 下一步**: 增强可叠加（noise + time-mask）是又一确定收益。用 **narrow + noise=0.1 +
  time_mask=0.15**（0.1 与 0.2 的折中，平衡 faith 与 macro-F1 裕度）全量训练，产出新最佳提交，
  对比 exp-007-final（faith 0.722）。

---

## exp-008-final — 最终最佳提交：narrow + noise=0.1 + time_mask=0.15（叠加 3 收益）

- **日期**: 2026-06-01
- **commit (代码来源)**: `1a1740e`
- **硬件**: RTX 4060，全量 360k 训练窗口，45 epoch，~2.7s/epoch
- **命令**:
  ```powershell
  uv run --no-sync python scripts\train_baseline.py --train-per-class 40000 `
    --val-per-class 4000 --epochs 45 --batch-size 768 --device cuda `
    --noise-std 0.1 --time-mask-frac 0.15 --out runs\final2\model.onnx
  uv run --no-sync gearxai package --model runs\final2\model.onnx `
    --data-dir data\prepared --split validation --out runs\final2\submission.zip
  ```
- **最终指标（完整公开验证集 83,790 样本）**:

  | 指标 | **final2 (noise+mask)** | final (noise) | exp-001 baseline |
  | --- | ---: | ---: | ---: |
  | macro-F1 | **0.9893** | 0.9926 | 0.9968 |
  | faithfulness | **0.7325** | 0.7219 | 0.7077 |
  | simplicity | **0.9220** | 0.9220 | 0.8363 |
  | expl_partial (0.4·f+0.2·s) | **0.4774** | 0.4732 | 0.4504 |

- **产物**: `runs/final2/submission.zip`（valid + eligible；model_sha256 `d69017ef…`，
  39,689 参数、41 算子）。
- **结论（本轮最佳交付）**: **3 个正交确定收益成功叠加**——simplicity（narrow，+0.086）+
  faithfulness（noise 增强 +0.014，time-mask 再 +0.011）。faith 链路 0.708→0.722→**0.7325**。
  最终 expl_partial **0.4774，比 exp-001 baseline 高 +0.027（+6.0%）**，macro-F1 仍 0.989
  远超门槛（全量训练把子集时的 0.960 拉回到 0.989，印证预判）。这是 8 方向系统探索的最佳可提交模型。
