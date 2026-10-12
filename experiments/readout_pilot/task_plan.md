# 全官方train读出支路先导（2026-10-11）

用户授权试验；固定234_Ki、2047_EC50、2147_Ki、244_Ki，seed42，graph/final/shallow三组，共12次新拟合。各组相同图主干初始化、训练顺序和优化器；全量官方train参与拟合，原test行成员保持不变。不是论文历史checkpoint训练复现。

运行前固定100epochs，保存第100轮，普通MSE；不早停，不按test择epoch/任务/分支，不调参。H256/L3/G4/K3、batch128、AdamW lr1e-4/wd1e-5/betas(.9,.95)、clip1、warmup10/cosine100/min_lr1e-6，epoch末scheduler.step(epoch)。seed42，loader generator42。原LongPoly内部dropout .1保留。

graph为原图模型；final新增最终层全部原子max+mean支路；shallow新增第一层ShortGINE的第三个分量v（初次长程混合前）全部原子max+mean支路。两支路均LayerNorm512/Linear512-32/ReLU/Linear32-512，标量alpha初始0，加至原SAGPool读出，原head不变。固定支路随机seed1042，fork_rng不改变主干训练RNG；两支路参数量匹配。没有ECFP或ACA输入/损失。

- [ ] P0 完整train/test行ID、源码/数据hash冻结；真实CUDA批次的零残差输出与梯度、alpha/投影梯度、序列化验证。
- [ ] P1 运行12次100轮拟合，每组独立进程；保存日志/history/最终checkpoint/alpha，无自动重试，异常即停止并报告缺失。
- [ ] P1 固定checkpoint重载后评价官方test，CSV独立重算、12组汇总及浅层vs最终层配对比较。

输出artifacts/readout_pilot_20261011及docs/readout_pilot_20261011。已有实验、原model.py、数据、权重、PDF和其他工程不改。单种子可行性结果，不声明显著性或新颖性；已看过的test须披露，不当作新的盲测。
