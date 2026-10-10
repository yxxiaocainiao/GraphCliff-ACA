# GraphCliff + ACA 独立试验

建立日期：2026-10-10（Asia/Shanghai）。

公开仓库：<https://github.com/yxxiaocainiao/GraphCliff-ACA>，默认分支 `codex/graphcliff-aca`。用户明确授权后独立发布，保留原作者来源与许可；权重和 artifacts 运行日志仅存本地。根 README.md 为原作者文档，本文件与阶段报告描述此仓库的 ACA 试验。

**目的：在原 GraphCliff 单分子回归模型上，检验 ACANet 的监督式三元组学习能否改善活性预测，特别是 Cliff RMSE。**

ACA 已接入并通过公式、零权重、空三元组、CUDA 前后向、hook 预测一致性与 checkpoint 格式检查。2026-10-10 固定六次先导及独立重载核验全部完成：两个任务的 ACA 在 Overall/Cliff RMSE 上都落后于匹配 MAE 对照，暂停扩大当前直接迁移配方。详细指标、曲线和诊断见 [阶段报告](docs/aca_pilot_20261010/report.md)。这是固定配方的负结果，不能推断所有对比学习无效；ACA 移植不是原创主方法。

用户已提供 30 个 GraphCliff checkpoint。全部通过当前默认模型的 strict 加载和有限值检查；核验记录见 `docs/checkpoint_audit_20261010.json`。现有 `main.py` 会在发现权重时跳过训练并评价 test，后续 ACA 将使用独立入口，保留这些权重原样。加载兼容不代表训练划分或官方来源已验证。

## 来源与目录

- 本地仓库：`D:/GraphCliff-ACA`，分支 `codex/graphcliff-aca`。
- 官方 GraphCliff：<https://github.com/dmis-lab/GraphCliff>。
- 固定起点：`d18857db0303346c0be715a56febf71da1768592`，独立 shallow clone。
- 原作者文档与许可：`README.md`、`LICENSE` 原样保留。
- ACA 来源：<https://github.com/shenwanxiang/ACANet>，固定实现 SHA `4c43e602`，GPL-3.0 原始许可和完整 loss 源码保存在 `experiments/aca/vendor/`，已与官方固定版本下载校验一致。
- 本项目记录：`task_plan.md`、`notes.md`、`AGENTS.md`。

## 最小研究范围

1. 保留原 GraphCliff 的 ShortGINE、LongPoly、门控、SAGPooling 和标量预测任务。
2. 每任务对比 MSE、MAE、MAE+ACA 三组；固定协议见 `experiments/aca/protocol.json`，以 MAE 对照隔离对比项贡献。
3. 从回归头之前的分子表示计算三元组约束；训练后仍对单个分子预测活性。
4. 先验证官方 ACA 移植，再决定是否研究新的 Triplet Loss 变体；移植不声明原创。
5. 不把 FPPool、逐层 Cross-Attention、邻居残差迁移、RelSup 和可靠性候选叠入首轮。

## 旧项目边界

| 目录 | 用途 |
|---|---|
| `D:/GraphCliff-Pair-FPPool-LongPoly` | 保留已有 FPPool/LongPoly、A30、cancel/RelSup 代码与历史证据 |
| `D:/GraphCliff-Pair` | 另一可靠性对话的工作目录 |
| `D:/GraphCliff-main` | 既有 GraphCliff 项目，不作为新试验的可写目录 |
| `D:/GraphCliff-ACA` | 本次独立的 GraphCliff + ACA 方向 |

没有复制旧模型改造和旧训练输出。ACA vendor 源码从旧目录复制后与官方固定版本字节校验一致；现有 30 个用户提供 checkpoint 保留不用。本地 benchmark_data 来自官方克隆，本轮仅分析官方 train 内划分，未评价官方 test。

入口：`python experiments/aca/run.py check`；`run --task CHEMBL234_Ki` 或 `CHEMBL2047_EC50`；完成后 `audit` 独立重载权重与复核预测。输出在 `artifacts/aca_pilot_20261010/`。旧平方 L2 / MSE 配方的负结果和本轮新配方分开记录，本轮不执行超参数搜索。

`python experiments/aca/report.py` 生成阶段报告、逐 epoch 历史、验证预测、环境与源文件记录、两张曲线；这些小型证据纳入本地 Git。新训练权重和运行日志保留 artifacts 中，不提交大文件。复核 RMSE 使用标准库，无新依赖安装。
