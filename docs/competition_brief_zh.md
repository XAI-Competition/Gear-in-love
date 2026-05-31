# GearXAI 解释型齿轮箱故障诊断赛题调查

调查日期：2026-05-30

## 官方入口

- 赛题主页：https://gearxai-ijcai-ecai2026.pages.dev/
- 数据集与 devkit：https://huggingface.co/datasets/edi45/gearxai-dds-seu
- Devkit 版本：`gearxai-devkit-v1.0.1`

## 任务定义

GearXAI 是 IJCAI-ECAI 2026 Competitions and Challenges Track 的单赛道比赛，目标是从多通道振动时间序列中进行齿轮箱故障诊断，并同时输出可解释性结果。

模型输入为振动窗口，官方 evaluator 使用的张量形状是：

```text
[N, 8, 100]
```

模型输出必须包含两个张量：

```text
probabilities: [N, 9]
relevance:     [N, 8, 100]
```

其中 `probabilities` 是 9 类故障概率，`relevance` 是时间维度和通道维度上的 relevance map，用于说明模型关注了哪些信号区域。

## 类别空间

比赛是 9 分类任务：

| code | class |
| --- | --- |
| HEA | healthy |
| CTF | chipped tooth fault |
| MTF | missing tooth fault |
| RCF | root crack fault |
| SWF | surface wear fault |
| BWF | ball fault |
| CWF | combination fault |
| IRF | inner race fault |
| ORF | outer race fault |

## 数据说明

数据来自 DDS-SEU drivetrain setup 的行星齿轮箱 processed release，不是原始 TXT 数据镜像。官方公开了 train 和 validation，隐藏 leaderboard test 不公开。

关键参数：

- sampling rate: `5120 Hz`
- channels: `8`
- window length: `100`
- fault classes: `9`
- operating conditions: `19`

本地已下载的 `windows_100` 数据量：

| split | windows |
| --- | ---: |
| train | 737,352 |
| validation | 83,790 |
| total | 821,142 |

每类样本均衡：

| split | per-class count |
| --- | ---: |
| train | 81,928 |
| validation | 9,310 |

通道含义：

| channel | meaning |
| --- | --- |
| channel_1 | motor vibration |
| channel_2 | RGB vibration y |
| channel_3 | RGB vibration x |
| channel_4 | RGB vibration z |
| channel_5 | torque |
| channel_6 | PGB vibration y |
| channel_7 | PGB vibration x |
| channel_8 | PGB vibration z |

`windows_100` schema：

- `signal`: nested list, shape `100 x 8`
- `fault_code`
- `fault_name`
- `condition_id`
- `speed_hz`
- `load_nm`
- `regime`
- `experiment_id`
- `window_index`

注意：Hugging Face 原始字段中的 `signal` 是 `100 x 8`，而官方 ONNX/evaluator 输入是 channels-first `[N, 8, 100]`。使用 `gearxai prepare-data` 后会转换成 evaluator 需要的 `.npy` 格式。

## 评价与提交要求

官方排序逻辑是先看诊断性能门槛，再看解释性：

- hidden test macro-F1 必须至少 `80%`，否则不进入解释性排名。
- 通过门槛后，按 explainability score 排名。
- explainability 组件包括 faithfulness、mechanistic relevance / mechanical alignment、simplicity。
- 官方页面给出的权重为：faithfulness `40%`，mechanistic relevance `40%`，simplicity `20%`。

提交物：

- 单个 CPU-only ONNX 模型。
- 离线、无网络、ONNX Runtime 推理。
- 用 devkit 生成 `submission.zip` 后，通过官方表单手动上传。
- 队伍信息不写入 ZIP，由提交表单单独填写。

官方命令：

```powershell
gearxai prepare-data --windows-dir data\windows_100 --out prepared
gearxai package --model model.onnx --data-dir prepared --split validation --out submission.zip
gearxai inspect-package submission.zip --data-dir prepared --split validation
```

## 本地已完成状态

工作目录：`D:\Codes\GearXAI`

已完成：

- 初始化 Git 仓库。
- 创建 `uv` 项目和 Python 3.12.12 虚拟环境。
- 使用 repo-local `.uv-cache-local` 作为 `UV_CACHE_DIR`。
- 配置 pytest cache/temp 到 `.tmp`。
- 预装常用包：PyTorch、Lightning、ONNX、ONNX Runtime、datasets、Hugging Face Hub、Captum、scikit-learn、SciPy、pandas、Polars、PyArrow、matplotlib、seaborn、Plotly、TensorBoard、JupyterLab 等。
- 下载官方 Hugging Face dataset snapshot 到 `data/hf_snapshot`。
- 下载并解压 devkit 到 `external/gearxai-devkit-v1.0.1`。
- 使用 devkit 生成 evaluator-ready 数据到 `data/prepared`。
- 用官方 `logic_lstm.onnx` baseline 成功生成并检查 `runs/baseline_logic_lstm_submission.zip`。

本地验证结果：

- pytest: passed。
- baseline package: valid。
- baseline validation macro-F1: `0.9842782072391496`。
- baseline faith score: `0.7006190251559019`。
- baseline simplicity score: `0.9071465366986513`。

## 环境注意事项

devkit 1.0.1 当前调用 `numpy.trapz`。NumPy 2.x 已移除此接口，因此本项目 pin 到：

```text
numpy>=1.26.4,<2
```

不要随意升级到 NumPy 2.x，否则 `gearxai package` 的指标计算会失败。

## 建议建模方向

优先目标不是单纯分类精度，而是在保持 hidden macro-F1 过线的前提下生成有质量的 relevance map。

建议 baseline 路线：

1. 先做一个强分类器：1D CNN / TCN / lightweight Transformer，输入 `[8, 100]`。
2. 模型内置解释输出：用 attention、prototype distance、channel-time gating 或 learned saliency head 直接输出 `[8, 100]` relevance。
3. 训练时加入解释正则：relevance 稀疏性、平滑性、噪声稳定性、通道/频段先验。
4. 导出 ONNX 前固定输出名和形状，确保 CPU ONNX Runtime 可跑。
5. 每次实验都用 `gearxai package` 在 public validation 上检查 macro-F1、faithfulness 和 simplicity。

下一步可以从 `logic_lstm.onnx` 的指标作为最低可用参照，先实现一个可训练 PyTorch baseline，并把导出 ONNX 与 devkit packaging 纳入训练脚本。
