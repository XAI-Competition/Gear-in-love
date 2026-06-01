# GearXAI Mechanical Alignment 深入审计

日期：2026-06-01

## 结论先行

当前最重要的结论是：**mechanical alignment 在本地不可安全直接优化**。不是因为它不重要，而是因为官方把最关键的 `band_config` 保留为私有；在公开 devkit 下，我们只能知道评分形式，不能知道真正的故障频带模板。

目前应该避免两类高风险动作：

- 不要硬编码从公开 validation 估出来的固定 Hz 频带。
- 不要为了猜 mechanical 把 relevance 强行压到某个类条件通道上，除非同时证明 faithfulness 损失很小。

## 官方评分公式实际奖励什么

公开 devkit 的 `single_mechanical_alignment` 做了以下事情：

1. 对每个通道的 100 点窗口做 STFT。
2. 把 `relevance[ch, :]` 按 STFT 帧聚合成每帧 relevance。
3. 用该帧的频谱能量分布，把 relevance 分配到频率轴上。
4. 看分配后的 relevance 有多少质量落在该类别对应的私有频带里。

因为窗口长度只有 100 点，devkit 默认 `nperseg=min(256, 100)=100`、`hop_length=64`，实际只有 **1 个 STFT 时间帧**。所以 mechanical alignment 对时间定位不敏感，真正起作用的是：

```text
每通道 relevance 总量 × 该通道在私有频带内的频谱能量占比
```

这意味着：

- 时间维 heatmap 的形状不会直接改善 mechanical 分数。
- 主要杠杆是每个样本/类别把 relevance 质量分给哪个通道。
- 如果知道私有频带，单通道集中可能比平滑分布更容易拿高分。
- 但不知道私有频带时，单通道集中很容易赌错并损害 faithfulness。

## 当前数据复核

本轮用 `scripts/analyze_mechanical_alignment.py` 从 `data/prepared` 重新分析了 public validation。prepared metadata 没有 `speed_hz` 字段，固定转速需要从 `condition_id` 解析，例如 `PGB_20_0`、`PGB_30_0`。

关键复核结果：

- public validation 是 `[83790, 8, 100]`。
- 每个固定转速/类别都有样本；30Hz 包含 `PGB_30_0..5`，样本更多。
- devkit STFT 只有 1 个时间帧，频率 bin 间隔是 `51.2 Hz`。
- 若允许 DC/0Hz 参与，很多类别的差异峰会坍到 0Hz；这对“机械故障频带”不一定可信。
- 排除 DC 后，非零峰仍然对抽样、聚合方式和转速很敏感：例如默认复核里 MTF 的 top nonzero peak 在 `102.4Hz` 到 `1126.4Hz` 之间变化，RCF/SWF/IRF 的跨速峰跨度接近或超过 `972.8Hz`。这不足以推出官方私有频带。

## final2 模型的 relevance 现状

对 `runs/final2/model.onnx` 抽样 720 个 validation 窗口后，当前 relevance 的通道质量分布几乎等于 `|x|` 的通道质量分布。这解释了为什么它的 faithfulness 强：deletion/insertion 主要看 top relevance cell 的排序，`|x|` 会优先高亮被置零后影响大的高幅值区域。

这也意味着当前模型没有真正学出强类条件 mechanical 通道偏好。它更像一个高 faithfulness、低风险的能量解释器。

## 为什么“猜通道”不稳

用另一个代理探针假设“官方频带刚好落在当前 final2 relevance 诱导出的若干非零峰附近”，最佳通道仍不形成干净稳定的类别规则：

| class | 代理中心频率 | 最佳通道 | 备注 |
| --- | ---: | --- | --- |
| CTF | 1024.0 | torque | margin 较大 |
| MTF | 1689.6 | rgb_z | 与 pgb_y/pgb_x 很接近 |
| RCF | 512.0 | torque | margin 很小 |
| SWF | 1024.0 | torque | margin 较大 |
| BWF | 51.2 | motor | 低频/DC 附近，风险高 |
| CWF | 51.2 | motor | 低频/DC 附近，风险高 |
| IRF | 921.6 | torque | margin 不大 |
| ORF | 51.2 | motor | 低频/DC 附近，风险高 |

这张表只说明“若频带如此，通道集中可能有用”；它不能证明官方 hidden mechanical 会奖励这些中心频率。特别是 51.2Hz 附近可能混入趋势/DC/工况差异，不应当直接写成机械先验。

## 实用策略

当前最佳方向仍然是：

1. 保留 `runs/final2/submission.zip` 作为强保底提交。
2. 继续优先优化本地可测的 faithfulness 和 simplicity。
3. mechanical 只做低风险代理：
   - 不改变分类路径。
   - 不牺牲 `|x|` cell 排序。
   - 如果尝试通道门控，必须用 devkit 证明 faithfulness 掉幅小于候选 mechanical 代理收益。
4. 不采纳硬编码固定频带或强类条件通道表，除非拿到官方 `band_config` 或新的 hidden leaderboard 反馈能证明方向正确。

## 复现命令

```powershell
$env:UV_CACHE_DIR = (Resolve-Path .\.uv-cache-local).Path
uv run --no-sync python scripts\analyze_mechanical_alignment.py `
  --data-dir data\prepared `
  --model runs\final2\model.onnx `
  --out .tmp\mechanical_alignment_analysis.json
```

输出 JSON 在 `.tmp/`，不会进入 Git。
