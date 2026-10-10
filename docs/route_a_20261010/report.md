# 路线 A：图与 Morgan 指纹融合先导

两个任务、单seed42、官方train内10%验证；仅validation，无test评价。四组均MSE、均匀采样和相同预算。

| Task | Arm | Overall RMSE | Cliff RMSE | best/epochs | Parameters |
|---|---|---:|---:|---:|---:|
| CHEMBL234_Ki | graph | 0.585909 | 0.619991 | 53/68 | 6021198 |
| CHEMBL234_Ki | fp | 0.597449 | 0.663929 | 51/66 | 1115137 |
| CHEMBL234_Ki | concat | 0.633122 | 0.661596 | 19/34 | 7135822 |
| CHEMBL234_Ki | gate | 0.623614 | 0.644121 | 26/41 | 7595086 |
| CHEMBL2047_EC50 | graph | 0.433690 | 0.459170 | 34/49 | 6021198 |
| CHEMBL2047_EC50 | fp | 0.476743 | 0.571536 | 27/42 | 1115137 |
| CHEMBL2047_EC50 | concat | 0.409269 | 0.375809 | 32/47 | 7135822 |
| CHEMBL2047_EC50 | gate | 0.439782 | 0.465828 | 51/66 | 7595086 |

## 相对变化（负值改善）

| Task | gate vs | Overall | Cliff |
|---|---|---:|---:|
| CHEMBL234_Ki | graph | +6.44% | +3.89% |
| CHEMBL234_Ki | fp | +4.38% | -2.98% |
| CHEMBL234_Ki | concat | -1.50% | -2.64% |
| CHEMBL2047_EC50 | graph | +1.40% | +1.45% |
| CHEMBL2047_EC50 | fp | -7.75% | -18.50% |
| CHEMBL2047_EC50 | concat | +7.46% | +23.95% |

全部checkpoint在独立进程strict重载，逐样本预测与保存CSV误差<1e-5；标准库math.fsum复算指标通过。

单种子且validation参与选模，不能声称统计显著或官方测试收益。新增分支增加参数量；concat首层头输入维度变化，权重无法完全匹配。
这是通用融合试验，不宣称原创；未叠加ACA/FPPool/成对模块，未调参、未自动扩大。
