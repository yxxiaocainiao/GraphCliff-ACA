# 路线 A 固定先导（2026-10-10）

- [x] 实现独立入口和四组：graph、fp、concat、gate；旧模型、ACA、tools不改。
- [x] 检查指纹/样本对应、相同主干初始化、门控端点、有限梯度和checkpoint重载。
- [ ] 两任务各四组顺序运行，共8次；CUDA错误即停止，不自动追加恢复或调参。
- [ ] 独立进程重载8个最佳checkpoint、复算CSV指标，生成阶段报告。

固定：CHEMBL234_Ki、CHEMBL2047_EC50；train内分层10% validation，seed/split42；不读取官方test标签。Morgan radius2/2048 bit，无手性，单层2048→512+ReLU。concat直接1024→128回归头；gate=sigmoid(Linear([h_graph,h_fp]))，512维逐通道混合后沿用原头；不额外归一化。全部MSE、均匀shuffle、batch32；H256/L3/groups4/K3/dropout0；AdamW lr1e-4/wd1e-5/betas(.9,.95)，clip1，warmup10/100epochs/patience15，严格最小validation Overall MSE选模。

图主干及原头同seed42初始化，新增层使用独立seed1042，训练随机序列重置seed42；concat输入维度不同，不能匹配首层头权重。单种子只作筛选，不给统计显著性/官方test结论。主干和fp分支分别使用同初始化校验；参数量差异必须报告。

输出：artifacts/route_a_20261010（权重/运行记录），docs/route_a_20261010（小型阶段证据）。不安装依赖，不自动发布。

预检通过：CHEMBL234 train/valid/cliff=2632/292/128；CHEMBL2047=452/51/20。graph路径eval预测与原模型完全一致；所有主干/指纹分支初始hash一致，门控两端点、有限梯度、strict重载通过。上游LongPoly默认dropout=.1仍保留（构造器dropout0未传入该层）；ShortGINE.norm三个参数未参与forward，不修上游。
