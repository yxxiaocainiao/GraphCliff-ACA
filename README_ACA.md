# GraphCliff + ACA 独立试验

建立日期：2026-10-10（Asia/Shanghai）。

**目的：在原 GraphCliff 单分子回归模型上，检验 ACANet 的监督式三元组学习能否改善活性预测，特别是 Cliff RMSE。**

当前仅完成仓库隔离与干净基线导入。ACA 尚未接入，未运行训练，也没有本项目的效果结论。

用户已提供 30 个 GraphCliff checkpoint。全部通过当前默认模型的 strict 加载和有限值检查；核验记录见 `docs/checkpoint_audit_20261010.json`。现有 `main.py` 会在发现权重时跳过训练并评价 test，后续 ACA 将使用独立入口，保留这些权重原样。加载兼容不代表训练划分或官方来源已验证。

## 来源与目录

- 本地仓库：`D:/GraphCliff-ACA`，分支 `codex/graphcliff-aca`。
- 官方 GraphCliff：<https://github.com/dmis-lab/GraphCliff>。
- 固定起点：`d18857db0303346c0be715a56febf71da1768592`，独立 shallow clone。
- 原作者文档与许可：`README.md`、`LICENSE` 原样保留。
- 拟复用的 ACA 来源：<https://github.com/shenwanxiang/ACANet>。具体版本、许可、参数和损失行为待实施前核验。
- 本项目记录：`task_plan.md`、`notes.md`、`AGENTS.md`。

## 最小研究范围

1. 保留原 GraphCliff 的 ShortGINE、LongPoly、门控、SAGPooling 和标量预测任务。
2. 首先对比普通 MSE 与相同回归目标加 ACA 三元组项；如果采用 MAE 配方，另设 MAE 对照。
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

旧代码、权重和实验输出没有复制到此仓库。本地 benchmark_data 来自官方克隆；这里只复制文件，未进行标签分析或测试集评估。

下一步：核对旧 ACA 尝试与官方损失实现，冻结最小试验协议，再接入可关闭的 ACA 路径并验证。
