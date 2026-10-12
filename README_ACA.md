# GraphCliff + ACA 独立试验

建立日期：2026-10-10（Asia/Shanghai）。

2026-10-12：已完成下列实验并整理代码、结果与复核证据。本次发布分支为 [`codex/route-a-30tasks-20261011`](https://github.com/yxxiaocainiao/GraphCliff-ACA/tree/codex/route-a-30tasks-20261011)，默认分支保留 `codex/graphcliff-aca`。权重、artifacts运行目录、论文PDF和全文摘录不纳入发布。

| 阶段 | 完成情况与结论 | 证据 |
|---|---|---|
| 30任务graph/concat | 60/60；concat平均Cliff改善约1.09%，Overall无改善，未超过论文checkpoint参照 | [报告](docs/route_a_30_20261011/report.md) |
| 作者checkpoint回放 | 30/30；宏平均0.664575/0.756855，逐任务有7个三位小数差异，不能说全部精确一致 | [核验](docs/graphcliff_checkpoint_audit_20261011/report.md) |
| 读出支路 | 四任务三组12/12；浅层支路未获益，最终层全原子支路仅候选 | [结论](docs/readout_pilot_20261011/conclusion.md) |
| cosine-margin | 四任务三seed两组24/24；验证Overall/Cliff改善1.46%/2.31%，任务稳定性门槛未通过 | [结论](docs/cliff_contrastive_20261012/conclusion.md) |
| ECFP×margin | 补24/24，加复用24模型形成48模型四组对照；组合未稳定超过两个单独方案 | [结论](docs/cliff_ecfp_20261012/conclusion.md) |

30任务与读出先导分别评价官方test，但训练/选模协议不同；两个margin阶段只使用官方train内验证，尚未完整train重训和test评价。所有表均区分开发成绩、重训对照与论文checkpoint参照，不声明原创方法或稳定超越论文。下一项系数校准仅为[建议](docs/cliff_ecfp_20261012/next_step_diagnostic.md)，未运行。

本次公开的小型复核证据包括已完成模型的history、prediction CSV、summary和运行manifest；说明及校验记录见[发布说明](docs/publication_20261012/README.md)。独立重载权重需要本地artifacts和原环境；公开CSV可不依赖权重复算RMSE。运行manifest保留实际训练时源码字节hash和本地路径，不作为跨平台新训练配方的运行状态。

后续残差/指纹低LR核验：9次新拟合完成、4个旧对照重审；CHEMBL244图组CUDA中断，后续3组未启动。三个完成任务未支持统一残差/低LR配方，100轮拼接诊断也未超过图对照。见 [残差核验入口](experiments/route_a_residual/README.md) 与 [报告](docs/route_a_residual_20261010/report.md)。停止后续训练，未自动重试或发布。

路线 A 图-指纹融合固定先导已在独立本地分支完成（两个任务、四组、8次训练）。门控组两任务均落后GraphCliff；简单拼接仅在CHEMBL2047有收益，暂停扩大门控配方。见 [路线 A 入口](experiments/route_a/README.md) 和 [完整报告](docs/route_a_20261010/report.md)。仅开发验证、单种子，未评价官方test，未自动发布。

后续单任务结构门控/投影核验见 [报告](docs/aca_local_20261010/report.md)：三组完整、原h消融因两次CUDA故障未完成；完整组合相对匹配MSE没有收益。该阶段历史提交纳入本次发布分支，公开默认分支仍为首次先导版本。

公开仓库：<https://github.com/yxxiaocainiao/GraphCliff-ACA>，默认分支 `codex/graphcliff-aca`。用户明确授权后独立发布，保留原作者来源与许可；权重和 artifacts 运行日志仅存本地。根 README.md 为原作者文档，本文件与阶段报告描述此仓库的 ACA 试验。

**目的：在原 GraphCliff 单分子回归模型上，检验 ACANet 的监督式三元组学习能否改善活性预测，特别是 Cliff RMSE。**

ACA 已接入并通过公式、零权重、空三元组、CUDA 前后向、hook 预测一致性与 checkpoint 格式检查。2026-10-10 固定六次先导及独立重载核验全部完成：两个任务的 ACA 在 Overall/Cliff RMSE 上都落后于匹配 MAE 对照，暂停扩大当前直接迁移配方。详细指标、曲线和诊断见 [阶段报告](docs/aca_pilot_20261010/report.md)。这是固定配方的负结果，不能推断所有对比学习无效；ACA 移植不是原创主方法。

用户已提供 30 个 GraphCliff checkpoint。全部通过当前默认模型的 strict 加载和有限值检查；核验记录见 `docs/checkpoint_audit_20261010.json`。现有 `main.py` 会在发现权重时跳过训练并评价 test，后续 ACA 将使用独立入口，保留这些权重原样。加载兼容不代表训练划分或官方来源已验证。

## 来源与目录

- 本地仓库：`D:/GraphCliff-ACA`；当前发布分支 `codex/route-a-30tasks-20261011`，建库起点分支 `codex/graphcliff-aca`。
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

建库时没有复制旧模型改造和旧训练输出。ACA vendor 源码从旧目录复制后与官方固定版本字节校验一致；本地benchmark_data来自官方克隆。用户提供的30个checkpoint在后续单独回放核验，保留原样、不作新训练初始化；各后续实验是否评价test见上方阶段表。

入口：`python experiments/aca/run.py check`；`run --task CHEMBL234_Ki` 或 `CHEMBL2047_EC50`；完成后 `audit` 独立重载权重与复核预测。输出在 `artifacts/aca_pilot_20261010/`。旧平方 L2 / MSE 配方的负结果和本轮新配方分开记录，本轮不执行超参数搜索。

`python experiments/aca/report.py` 生成阶段报告、逐 epoch 历史、验证预测、环境与源文件记录、两张曲线；这些小型证据纳入本地 Git。新训练权重和运行日志保留 artifacts 中，不提交大文件。复核 RMSE 使用标准库，无新依赖安装。
