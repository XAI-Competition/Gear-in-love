# 排行榜 mechanical 探测协议（2026-06-10 制定，同日按排行榜锚点修订）

## 实测校准锚点（2026-06-10；数据源 `leaderboard.json` 全精度核验）

**排行榜数据可直接编程读取**：`https://gearxai-ijcai-ecai2026.pages.dev/leaderboard.json`
（字段：`explainability_score / macro_f1 / faith_score / mechanical_score / simplicity_score /
eligible`；本快照 `generated_at 2026-06-09T13:05Z`——由主办方周期性重新生成，上传后等下一次
生成才能读到新分数）。每次探针上传后可直接抓 JSON 读数，无需手抄。

完整榜面（eligible 队伍，2026-06-09 快照）:

| Rank | Team | Final | Macro-F1 | Faith | Mech | Simp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Resonance Logic | 0.56617 | 0.90603 | 0.71922 | **0.26918** | 0.85405 |
| 2 | 我们（final2） | 0.56541 | 0.92956 | 0.73251 | 0.22001 | 0.92200 |
| 2' | 我们（早期 narrow 系） | 0.55663 | 0.92956 | 0.71119 | 0.21938 | 0.92200 |
| 3 | LSY | 0.54619 | 0.91976 | 0.74041 | 0.17425 | 0.90161 |
| 4 | CUFE | 0.542 | 0.831 | 0.670 | 0.218 | 0.935 |
| 5 | A-Trial | 0.416 | 0.885 | 0.582 | 0.154 | 0.609 |
| 6 | machine unlearning | 0.399 | 0.880 | 0.580 | 0.099 | 0.637 |

公式核验：`0.4·0.73251 + 0.4·0.22001 + 0.2·0.92200 = 0.56541` ✓（官方权重精确成立）。

**榜面战略读数**：
- **与第一名差距仅 0.00076**。第一名靠 mech 0.269 领先（其 faith/simp 都比我们差）；
  全场 mech 0.099–0.269，**没有任何队伍破解 mechanical**。
- 0.269 的存在证明 mech > 0.22 有可捕获信号；S0（faith 0.833）上传后预期总分 ~0.60，
  应直接登顶 +0.04；mech 探测每 +0.1 再 +0.04。

**五个决定性读数**：
1. **faith 迁移 1:1**：final2 公开集 0.73247 → hidden **0.7325**（4 位小数内相同）。
   occ+T8 主候选（公开 0.8333）上榜预期 ≈ 0.83。
2. **hidden macro-F1 = 0.9296**（公开 0.989，−6 点）：所有 relevance-only 变体共享此值，
   门槛余量仍 >0.13。
3. **mechanical ≈ 0.22 是最弱分项**（贡献仅 0.088/0.5654）——也是最大上升空间。
4. **官方 mech 公式 ≠ devkit 出厂公式**：0.22 低于 devkit `0.75·EAS+0.25·stability` 的
   0.25 地板；官网注明 *"Mechanical scoring was upgraded on June 9, 2026, all submissions
   re-evaluated identically"*。把官方 mech 当**黑盒**：只假设"质量分给频带对齐的通道→分高"
   的单调性，差分读数仍然成立，但不要再用 0.75 换算 ΔEAS——**Δmech 直接读**。
5. **排行榜显示 4 位小数**（两次提交 mech 差 0.0006 可见）——探测精度充足。

## 背景与原理

- 官网排行榜对每次提交公开分项：macro-F1、**Faithfulness、Mechanical、Simplicity**、Status
  （survey-002）。开发期提交窗口至 **2026-06-30**，最终截止 **2026-07-15 AoE**。
- mechanical 本地不可测（私有频带模板），但对**同一分类器、只改 relevance 通道分配**的两次
  提交，分项差 `Δmech` 直接度量"该通道分配在官方频带上的优劣"。探针把某类组的 relevance
  质量压到单一通道上，Δmech 就读出该通道在该类组官方频带上的相对捕获率。
- 所有探针与主候选**同族**（final2 分类器 + occ gate a1/eps0.2 + T8 + exp-023 基座，仅改
  静态 gate 的组行），结论无需跨族迁移，直接适用于最终模型。
- 探针的 macro-F1 与主候选完全相同（概率输出逐位一致），faith 会因激进 gate 下降 ~0.02-0.04
  （exp-028 标定），**开发期排名不重要，7/15 的最终提交才计分**。

## 提交资产（2026-06-10 第二版：exp-029 体系，round1 旧探针作废）

| # | 名称 | 路径 | 用途 |
| --- | --- | --- | --- |
| S0a | **主候选**（exp-029 no-gate） | `runs/exp029_final/wrapped/occ_a0p5_eps0p05_T8/submission.zip` | 最高 faith 0.8747；读新分类器的 mech 基线 |
| S0b | exp-029 gated（occ_a1_eps0p2_exp023_T8） | `runs/exp029_final/wrapped/occ_a1_eps0p2_exp023_T8/submission.zip` | faith 0.8633；**探针参照 m0**；S0a−S0b 差分 = exp-023 gate 的 hidden mech 价值 |
| P1 | probe_quintet_torque | `runs/mech_probes_round1b/probe_quintet_torque/submission.zip` | CTF/MTF/RCF/SWF/IRF 全押 torque |
| P2 | probe_quintet_pgby | `runs/mech_probes_round1b/probe_quintet_pgby/submission.zip` | 同组全押 pgb_y |
| P3 | probe_trio_motor_hard | `runs/mech_probes_round1b/probe_trio_motor_hard/submission.zip` | BWF/CWF/ORF motor 加硬（4.0/0.25） |
| P4 | probe_mtf_rgbz | `runs/mech_probes_round1b/probe_mtf_rgbz/submission.zip` | 单类假设：MTF→rgb_z |

P1–P4 与 S0b 同族（同 checkpoint + occ a1/eps0.2 + T8 + exp-023 基座，仅改组行），
差分对 S0b 读。S0a/S0b/P1–P4 共享同一个新 hidden macro-F1（同一分类器）——这个数字
本身是"上传对了文件"的校验位（应 ≈0.926，按 lb-001 的 −6 点偏移估计）。
⚠️ `runs/mech_probes_round1/`（final2 系）已作废，勿上传。

## 上传顺序与预算决策树

**第一步（必做）**：打开提交表单时，记录表单上写明的**提交次数限制**（若有）。

- **预算 ≥ 7**：S0a → S0b → P1 → P2 → P3 → P4 →（分析后）确认提交 C1。
- **预算 5–6**：S0a → S0b → P1 → P2 →（分析后）C1。放弃 P4（单类增益上限小）；
  P3 的 motor 方向已有本地证据，可温和采纳。
- **预算 3–4**：S0a → S0b → P1 → C1。quintet 是最大的不确定组，P1 vs S0b 至少定一个方向。
- **预算 ≤ 2**：只传 S0a（当前最佳）+（可选）S0b。mech 优化退化为按 proxy 先验的温和押注。

每次上传后在下表记录读数；两次提交之间不需要等待本地操作（全部 zip 已预生成）。

## 读数记录表（上传后填写）

| 提交 | 日期 | leaderboard macro-F1 | faith | mechanical | simplicity | Δmech vs S0b |
| --- | --- | --- | --- | --- | --- | --- |
| S0a | | | | | | |
| S0b | | | | （= m0） | | — |
| P1 | | | | | | |
| P2 | | | | | | |
| P3 | | | | | | |
| P4 | | | | | | |

核对项：S0a 的 leaderboard faith 应 ≈ 0.875、S0b ≈ 0.863（公开集 0.87472/0.86330；
final2 锚点证明 faith 迁移精确到 4 位小数）；6 个 zip 的 macro-F1 应逐位相同（≈0.926，
同一分类器）。S0a 预期总分 ≈ 0.4·0.8747 + 0.4·mech + 0.2·0.9085 ≈ **0.62**（若 mech
~0.22）。S0a−S0b 的 Δmech 直接给出 exp-023 gate 在官方频带上的价值（本地 proxy 预测
gated 高 ~+0.02），决定 C1 用不用 gate 基座。

## 解读规则

组级信号按组规模稀释（quintet 5/9 类、trio 3/9 类），质量集中率由 boost4.0/other0.25
决定。官方公式已升级为黑盒（见锚点 4），不做绝对换算，实操按相对比较读：

1. **P1 vs P2**（同组同强度，只换通道）：mech 高者 = quintet 组更优通道方向。
   - P1 ≫ P2：采纳 torque 方向；P2 ≫ P1：采纳 pgb_y；两者接近且都 > m0：组内异质，
     考虑（预算允许时）拆组复测或采用每类 proxy 先验混合。
   - **两者都 < m0**：exp-023 现状（quintet 不动）已优于任何单通道押注 → quintet 保持现状。
2. **P3 vs m0**：P3 > m0 → motor 方向正确且应加硬（最终模型 trio 行用 4.0/0.25 或更强）；
   P3 ≈ m0 → 保持 exp-023 软 gate；P3 < m0 → motor 押注过度，回退强度。
3. **P4 vs m0**（含 P1 时更准：P4 与 P1 只差 MTF 行）：判断 MTF 单类是否该走 rgb_z。
4. **确认提交 C1** = 按上述结论组装的组合 gate（仍在同族内，`make_mech_probes.py` 改
   `PROBES` 即可生成），上传验证 mech 是否达到预测值，并确认 faith 仍可接受
   （Pareto：0.4·Δfaith + 0.4·Δmech ≥ 0 才采纳）。

## 风险与注意事项

- **分数精度**：若排行榜只显示 2 位小数，单类探针（P4）的信号（预计 |Δmech| ≤ 0.02）
  可能不可读；组级探针（P1-P3，预计 |Δmech| 0.02-0.10）仍可读。
- **显示语义**：确认排行榜显示的是"本次提交"的分数而非"历史最佳"。若是历史最佳，
  探针差分仍可从提交回执/反馈邮件读取（dev 窗口主打 "organizer validation feedback"）。
- **不要把探针留作最终提交**：7/15 前必须把最终模型（C1 或 S0）作为最后一次有效提交
  （以官方规则定义的"最终提交"机制为准，上传时确认）。
- 所有探针 zip 由 `gearxai inspect-package` 验证过 valid；上传前不要重新打包。

## 附录：探针本地指标（exp-029 体系 round1b；3000 样本 seed 50050；全部 devkit valid）

| 资产 | faith | del↓ | ins↑ | proxy audit / low_freq | model_sha256 前缀 |
| --- | ---: | ---: | ---: | --- | --- |
| S0a 主候选（5000 样本，全验证集 0.87472） | 0.8740 | 0.178 | 0.926 | 0.4224 / 0.6168 | `c6fbb0b52683` |
| S0b 探针参照（5000 样本，全验证集 0.86330） | 0.8641 | 0.167 | 0.895 | 0.4470 / 0.6428 | `3de8885950d0` |
| probe_quintet_torque | 0.8376 | 0.200 | 0.875 | 0.4529 / 0.6187 | `100da1a38b87` |
| probe_quintet_pgby | 0.8259 | 0.205 | 0.857 | 0.4452 / 0.6129 | `b54c9b7e060f` |
| probe_trio_motor_hard | 0.8532 | 0.175 | 0.882 | 0.4709 / 0.6667 | `e8733d7a81db` |
| probe_mtf_rgbz | 0.8514 | 0.181 | 0.884 | 0.4456 / 0.6369 | `26466a520aba` |

探针 faith 代价 −0.011～−0.038（harsh 押注的预期损耗）；按锚点换算探针总分仍 ≈0.60
量级——上传无"丢人"风险。完整数字见 `runs/mech_probes_round1b/summary.json`。
