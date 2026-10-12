# GraphCliff 结构局部 cosine-margin 先导

当前阶段：24/24有效拟合及独立重载审计完成，go=false，详见conclusion.md。首次标签形状广播错误运行保存在artifacts/cliff_contrastive_20261012_invalid_shape，所有指标/权重作废，不并入结果。

输入：D:/GraphCliff-ACA/benchmark_data/{CHEMBL234_Ki,CHEMBL2047_EC50,CHEMBL2147_Ki,CHEMBL244_Ki}.csv。

冻结协议：D:/GraphCliff-ACA/experiments/cliff_contrastive/task_plan.md。四任务×seed42/43/44×MSE/margin，共24次。官方test成员不变；先导只在官方train内划出验证，正负配对只用开发train。ECFP用来筛结构相近的对，不作为网络输入；推理仍是原单分子GraphCliff。

候选缺陷：原回归训练没有显式的cliff-pair表示约束。试验用标准CosineEmbeddingLoss，正例要求结构和活性都相近、负例要求结构相近但活性差大；无需新增网络。该实现是已有损失的工程对照，不声明原创；论文没有提供这里的margin/权重/采样配方。

- 状态：D:/GraphCliff-ACA/artifacts/cliff_contrastive_20261012/status.json。
- 训练日志/权重：同目录task/seed/arm；每批检查预测/标签形状和标签行对应，保存采样hash。
- 配对证据：本目录*_train_pairs.csv及coverage.json。
- 运行后自动生成：paired_results.csv、results.json、aggregate.json、report.md；完成队列后逐个独立重载审计。
- 全train重训、官方test评价、ECFP与损失组合、30任务扩展尚未执行。

入口（PowerShell，原环境，无新依赖）：

```powershell
& 'D:/Tools/conda-envs/graphcliff/python.exe' 'D:/GraphCliff-ACA/experiments/cliff_contrastive/run.py' report
# 队列结束后可以复核；队列运行时不要启动第二个GPU训练或audit进程。
& 'D:/Tools/conda-envs/graphcliff/python.exe' 'D:/GraphCliff-ACA/experiments/cliff_contrastive/run.py' audit
```

来源：[GraphCliff附录A.1](https://arxiv.org/html/2511.03170v3#A1)、[PyTorch CosineEmbeddingLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)。原模型和旧实验文件保留；新脚本复用现有开发划分、模型、评估、指纹、梯度工具。原MIT及ACA GPL许可保留。
