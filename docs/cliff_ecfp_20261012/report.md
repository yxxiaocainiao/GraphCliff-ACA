# ECFP × cosine-margin 四组开发比较

复用24个纯图模型；新完成24/24；四组完整配对12/12。
仅官方train内验证，不与论文test表混排；图表示承担辅助约束，ECFP只进入回归输入。

| Task | seed | graph MSE Cliff | graph margin Cliff | concat MSE Cliff | concat margin Cliff |
|---|---:|---:|---:|---:|---:|
| CHEMBL234_Ki | 42 | 0.617241 | 0.593930 | 0.687204 | 0.665100 |
| CHEMBL234_Ki | 43 | 0.671953 | 0.644422 | 0.654874 | 0.667209 |
| CHEMBL234_Ki | 44 | 0.660067 | 0.621977 | 0.672364 | 0.661945 |
| CHEMBL2047_EC50 | 42 | 0.472618 | 0.501133 | 0.399663 | 0.430042 |
| CHEMBL2047_EC50 | 43 | 0.435478 | 0.381024 | 0.393493 | 0.458102 |
| CHEMBL2047_EC50 | 44 | 0.462612 | 0.479158 | 0.444687 | 0.421005 |
| CHEMBL2147_Ki | 42 | 0.681719 | 0.627352 | 0.575571 | 0.592803 |
| CHEMBL2147_Ki | 43 | 0.575301 | 0.543855 | 0.545202 | 0.607581 |
| CHEMBL2147_Ki | 44 | 0.599092 | 0.601060 | 0.604798 | 0.510303 |
| CHEMBL244_Ki | 42 | 0.896363 | 0.873594 | 0.879857 | 0.853118 |
| CHEMBL244_Ki | 43 | 0.869164 | 0.881701 | 0.854523 | 0.863263 |
| CHEMBL244_Ki | 44 | 0.827217 | 0.840163 | 0.850988 | 0.868783 |

```json
{
  "reused_models": 24,
  "new_completed": 24,
  "new_total": 24,
  "paired_blocks": 12,
  "independent_reload_audited": true,
  "go": false,
  "macro": {
    "graph_mse": {
      "rmse": 0.5952538012298477,
      "cliff_rmse": 0.6474020992492563
    },
    "graph_margin": {
      "rmse": 0.5865702091701676,
      "cliff_rmse": 0.6324475146110626
    },
    "concat_mse": {
      "rmse": 0.595353276568841,
      "cliff_rmse": 0.6302686879170903
    },
    "concat_margin": {
      "rmse": 0.5898206150674055,
      "cliff_rmse": 0.6332712447581441
    }
  },
  "tasks_beating_both_single_at_least_two_seeds": 0,
  "mean_cliff_interaction": 0.01795714147924746
}
```

先导不是显著性验证；更换输入同时增加参数，未有ECFP-only/容量匹配对照，不声明算法原创。全train/test尚未执行。
