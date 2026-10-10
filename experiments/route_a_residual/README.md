# 路线 A 残差 / 指纹学习率核验

2026-10-10 用户授权下一轮。冻结训练提交bce1bbe；分支codex/route-a-residual-20261010。预设13次新拟合：完成9次，1次CUDA中断，3次未启动；旧graph/concat四个对照重新审计，未当作新训练。CHEMBL244未完成，禁止使用其部分checkpoint排名。

阶段决策：暂停扩大当前残差/低LR配方。234低LR残差改善标准残差但仍落后graph（Cliff .634222 vs .619991）；2047/2147低LR均退步。concat在2047/2147是当前完成比较中Overall/Cliff都最好的组；234仍graph最好。固定可学习alpha=.01起步并未证明弱残差适用于大样本任务。

234 concat跑满100epoch，best51 Overall/Cliff=.631870/.667667，仍落后graph。重跑前6轮与旧轨迹一致，epoch7起分叉，最大前段差.0972，原因未定位；初始hash、来源代码/数据、划分和设置一致不能保证逐步相同。改为同一完整轨迹模拟早停：patience15在epoch66停止，best51已在停止前出现；之后best72=.638018/.677319，更差。这次诊断不支持延长训练修复退步，也不能证明原运行早停一定正确/错误或把差异归因GPU。

方法：复用原图pooling后512维表示和相同初始回归头；fp=Linear2048→512+ReLU；h=h_graph+alpha*h_fp；可学习alpha=.01初始。仅fp组LR从1e-4降为2e-5，alpha/graph/head仍1e-4，全scheduler保持比例.2。每epoch首训练batch记录残差/graph L2范数比，不保证全程小扰动。其余见[固定计划](task_plan.md)。bit指纹不是count；不是已证实的正则化护栏。

输入：本仓库benchmark_data四任务官方train。输出：artifacts/route_a_residual_20261010（权重/完整日志/冻结源码，仅本地）与[报告和审计](../../docs/route_a_residual_20261010/report.md)。旧模型、ACA、路线A目录、tools与其他项目未修改。审计脚本仅在训练结束后修正未完成记录及诊断比较；冻结代码另保存在docs和artifacts，AST核验所有训练/构造函数未变。

```powershell
& D:/Tools/conda-envs/graphcliff/python.exe experiments/route_a_residual/pilot.py audit
```

当前输出目录已存在，run会拒绝覆盖，勿原地重跑。audit只重载完成模型，不启动训练。13个完成结果strict重载、预测一致、math.fsum复算RMSE、初始/源码/数据hash与旧划分通过。未完整执行原四任务预算；CUDA错误可能异步上报，当前栈不能定位根因。未自动重试、未调环境或驱动、未发布、未评价test。单seed和20个2047验证cliff样本不能给显著性或样本量因果结论。

建议后续优先对graph/concat多种子确认收益；若继续244，应先单独处理CUDA稳定性，另行冻结剩余运行预算。当前不自动续队列。
