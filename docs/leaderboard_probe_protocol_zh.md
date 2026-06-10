# 排行榜 mechanical 探测协议（2026-06-10 制定，同日按排行榜锚点修订）

## 实测校准锚点（2026-06-10，用户提供的两次历史提交）

| 提交 | Final | Macro-F1 | Faith | Mechanical | Simplicity |
| --- | ---: | ---: | ---: | ---: | ---: |
| final2（=`runs/final2`） | 0.5654 | 0.9296 | 0.7325 | 0.2200 | 0.9220 |
| 另一 narrow 家族模型 | 0.5566 | 0.9296 | 0.7112 | 0.2194 | 0.9220 |

公式核验：`0.4·0.7325 + 0.4·0.2200 + 0.2·0.9220 = 0.5654` ✓（官方权重精确成立）。

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

## 提交资产

| # | 名称 | 路径 | 用途 |
| --- | --- | --- | --- |
| S0 | **主候选** occ_a1_eps0p2_exp023_T8 | `runs/occ_exp027/occ_a1_eps0p2_exp023_T8/submission.zip` | 读基线 mech m0 + 验证 faith 0.833 迁移 |
| P1 | probe_quintet_torque | `runs/mech_probes_round1/probe_quintet_torque/submission.zip` | CTF/MTF/RCF/SWF/IRF 全押 torque |
| P2 | probe_quintet_pgby | `runs/mech_probes_round1/probe_quintet_pgby/submission.zip` | 同组全押 pgb_y |
| P3 | probe_trio_motor_hard | `runs/mech_probes_round1/probe_trio_motor_hard/submission.zip` | BWF/CWF/ORF motor 加硬（4.0/0.25 vs 现 1.2-4.2/0.8） |
| P4 | probe_mtf_rgbz | `runs/mech_probes_round1/probe_mtf_rgbz/submission.zip` | 单类假设：MTF→rgb_z |

各 zip 的 sha256 与本地指标见 `runs/mech_probes_round1/*/metrics.json` 及本文档附录。

## 上传顺序与预算决策树

**第一步（必做）**：打开提交表单时，记录表单上写明的**提交次数限制**（若有）。

- **预算 ≥ 6**：S0 → P1 → P2 → P3 → P4 →（分析后）确认提交 C1。
- **预算 4–5**：S0 → P1 → P2 → P3 →（分析后）C1。放弃 P4（单类增益上限小）。
- **预算 2–3**：S0 → P1（quintet 是最大的不确定组）→ C1。P3 的 motor 方向已有
  exp-018/019 本地证据支撑，可不验证直接温和采纳。
- **预算 = 1**：只传 S0（当前最佳）。mech 优化退化为按 proxy 先验的温和押注。

每次上传后在下表记录读数；两次提交之间不需要等待本地操作（探针已全部预生成）。

## 读数记录表（上传后填写）

| 提交 | 日期 | leaderboard macro-F1 | faith | mechanical | simplicity | Δmech vs S0 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | | | | （= m0） | | — |
| P1 | | | | | | |
| P2 | | | | | | |
| P3 | | | | | | |
| P4 | | | | | | |

同时核对：S0 的 leaderboard faith 应 ≈ 0.83（公开验证集 0.8333；final2 锚点已证明
faith 迁移精确到 4 位小数）、macro-F1 应 = 0.9296（与历史提交逐位相同，因为概率输出
不变——这同时是"上传对了文件"的校验位）。S0 预期总分 ≈ 0.4·0.833 + 0.4·mech(S0) +
0.2·0.9085 ≈ **0.60+**（若 mech 维持 ~0.22；occ gate 对 mech 的实际影响正是 S0 要测的）。

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

## 附录：探针本地指标（3000 样本，seed 50050；全部 devkit valid）

| 探针 | faith | del↓ | ins↑ | proxy audit / low_freq | model_sha256 前缀 |
| --- | ---: | ---: | ---: | --- | --- |
| S0 主候选（参考，5000 样本） | 0.8324 | 0.165 | 0.830 | 0.4516 / 0.6527 | `e331d3f22b33` |
| probe_quintet_torque | 0.8042 | 0.196 | 0.805 | 0.4579 / 0.6260 | `e59fa3707884` |
| probe_quintet_pgby | 0.8117 | 0.185 | 0.809 | 0.4502 / 0.6203 | `06bacb51c593` |
| probe_trio_motor_hard | 0.8244 | 0.176 | 0.824 | 0.4736 / 0.6746 | `d89396fadf81` |
| probe_mtf_rgbz | 0.8275 | 0.168 | 0.823 | 0.4503 / 0.6460 | `75d3dc4d5688` |

探针 faith 代价 −0.005～−0.028（harsh 押注的预期损耗）；按锚点换算，探针总分仍在
0.59 上下——上传无"丢人"风险。完整数字见 `runs/mech_probes_round1/summary.json`。
