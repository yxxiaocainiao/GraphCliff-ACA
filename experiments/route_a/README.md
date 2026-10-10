# 路线 A：GraphCliff + Morgan 双分支融合

2026-10-10 完成两个任务四组共8次固定先导，单seed42。输入为本仓库 benchmark_data 官方train，独立开发验证；不读取官方test标签、不加载用户旧checkpoint。独立分支 codex/route-a-fusion-20261010；训练代码冻结提交 e391ee3。

阶段结论：当前门控配方相对GraphCliff的Overall/Cliff在两任务均退步，暂停扩大。concat在CHEMBL2047改善Overall 5.63%、Cliff 18.15%，在CHEMBL234分别恶化8.06%、6.71%；只能说明该固定配置存在任务差异，不能宣布通用互补或门控有效。下一候选为匹配graph/concat多种子核验，但本轮没有执行。

完整指标和小型证据：[报告](../../docs/route_a_20261010/report.md)；固定协议：[计划](task_plan.md)。权重和完整日志：artifacts/route_a_20261010（Git忽略）。旧模型、ACA目录、tools和其他项目未改。

```powershell
& D:/Tools/conda-envs/graphcliff/python.exe experiments/route_a/pilot.py check
& D:/Tools/conda-envs/graphcliff/python.exe experiments/route_a/pilot.py run
& D:/Tools/conda-envs/graphcliff/python.exe experiments/route_a/pilot.py audit
```

run要求任务输出目录不存在，防止覆盖已完成结果；当前已完成，勿原地重复run。check只覆盖预检，audit只重载并重建小型报告。所有四组均MSE；fp=2048→512+ReLU；concat输入1024，gate逐通道混合512。保留上游LongPoly内部默认dropout=.1；checkpoint重载与门控端点在eval模式检查。

8个模型strict重载，预测与CSV最大绝对误差8.9e-16；标准库复算RMSE通过。源码和数据SHA256在训练后核验未变。无CUDA中断或非有限值。验证cliff样本仅128/20个，单种子与验证选模不能给统计显著或官方test结论。基线约602万、concat约714万、gate约760万参数；未做参数量匹配消融。

入口复用experiments/aca/run.py的开发划分/初始化/指标工具，沿用GPL-3.0；原GraphCliff MIT来源与许可证保留。
