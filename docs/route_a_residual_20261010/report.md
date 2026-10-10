# 路线 A 残差与学习率固定核验

计划四任务13次新拟合，完成9次；另4个复用对照重新重载审计。244图组CUDA中断，后续3组未启动，未纳入排名。seed42，官方train内验证，无test评价。

| Task | Arm | Overall | Cliff | best/epochs | alpha best |
|---|---|---:|---:|---:|---:|
| CHEMBL234_Ki | graph | 0.585909 | 0.619991 | 53/68 | None |
| CHEMBL234_Ki | concat | 0.633122 | 0.661596 | 19/34 | None |
| CHEMBL234_Ki | residual | 0.606571 | 0.667560 | 23/38 | 0.12780426442623138 |
| CHEMBL234_Ki | residual_slow | 0.587857 | 0.634222 | 26/41 | 0.14847642183303833 |
| CHEMBL234_Ki | concat100 | 0.631870 | 0.667667 | 51/100 | None |
| CHEMBL2047_EC50 | graph | 0.433690 | 0.459170 | 34/49 | None |
| CHEMBL2047_EC50 | concat | 0.409269 | 0.375809 | 32/47 | None |
| CHEMBL2047_EC50 | residual | 0.428083 | 0.470198 | 50/65 | 0.062513567507267 |
| CHEMBL2047_EC50 | residual_slow | 0.439164 | 0.552375 | 48/63 | 0.05515984445810318 |
| CHEMBL2147_Ki | graph | 0.580039 | 0.583213 | 66/81 | None |
| CHEMBL2147_Ki | concat | 0.561773 | 0.571329 | 19/34 | None |
| CHEMBL2147_Ki | residual | 0.581829 | 0.581771 | 45/60 | 0.08439187705516815 |
| CHEMBL2147_Ki | residual_slow | 0.604468 | 0.640338 | 66/81 | 0.0913003608584404 |

## 相对graph（负值改善）

| Task | Arm | Overall | Cliff |
|---|---|---:|---:|
| CHEMBL234_Ki | concat | +8.06% | +6.71% |
| CHEMBL234_Ki | residual | +3.53% | +7.67% |
| CHEMBL234_Ki | residual_slow | +0.33% | +2.30% |
| CHEMBL2047_EC50 | concat | -5.63% | -18.15% |
| CHEMBL2047_EC50 | residual | -1.29% | +2.40% |
| CHEMBL2047_EC50 | residual_slow | +1.26% | +20.30% |
| CHEMBL2147_Ki | concat | -3.15% | -2.04% |
| CHEMBL2147_Ki | residual | +0.31% | -0.25% |
| CHEMBL2147_Ki | residual_slow | +4.21% | +9.79% |

## 早停诊断

前34轮与旧concat最大绝对差=0.0972。
旧轨迹best=19；重跑从epoch7出现差异；不作为旧训练的精确续跑。
本次完整轨迹模拟patience15停止epoch=66；100轮best=51。
在同一完整轨迹内比较模拟停止前/后，避免运行差异混杂；差异来源未定位。延长训练即使有效也不证明指纹收敛更快是原因。

alpha无约束可学习，范数比为每epoch首训练batch的平均样本L2比，非全训练上界。低LR组全程fp/main=.2；alpha沿用main LR。
所有完成模型独立strict重载，预测差<1e-5；CSV用math.fsum复算，初始hash、冻结训练源码/数据hash与分区核验通过。审计只修订CUDA未完成处理与诊断比较，AST核验其他函数/导入/入口未变。
只有单seed、验证集选模，20个2047验证cliff分子；跨任务差异受靶点/噪声/化学空间等影响，不能归因样本量或称统计显著。
残差保留回归头输入/初始权重，但仍改变优化轨迹；未叠加其他模块、未调参、未发布。
