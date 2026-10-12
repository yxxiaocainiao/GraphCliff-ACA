# 结构局部的 cosine-margin 损失先导（2026-10-12）

用户授权按上一轮建议先单独验证 cliff-aware 损失。问题：在同一 GraphCliff、同一采样下，对结构相近分子的表示施加活性区分约束，能否降低验证 Cliff RMSE？这是工程先导，不宣称原创或论文复现。

## 冻结协议

- CHEMBL234_Ki、CHEMBL2047_EC50、CHEMBL2147_Ki、CHEMBL244_Ki；初始化 seed=42/43/44，固定 split_seed=42；每个 task×seed 两组，共24次。
- 官方 CSV train/test 行成员不变。仅读取官方 train 标签；复用 base.development 的 train 内10% cliff分层验证。训练+验证行并集必须等于官方 train，且与官方 test 不相交。先导不是完整 train 最终拟合。
- 正例：Morgan radius2/2048bit，无手性，Tanimoto>=0.8 且 |Δy|<0.2；负例：Tanimoto>=0.8 且 |Δy|>=1。独立无向对，无自配对；中间活性差、结构不相近的对不施加约束。这不是完整 MoleculeACE 三种结构判据。
- 用 PyTorch 原生 cosine_embedding_loss，不新增网络或投影头，直接用原 SAGPool 后、回归头前512维表示。正例 1-cos(h_i,h_j)，负例 max(cos(h_i,h_j)-0.5,0)；辅助项=0.5×正例均值+0.5×负例均值。
- 两组：mse 与 margin（MSE+0.01×辅助项）。0.5 cosine margin、0.01系数和配对阈值是预先固定的试验设定，不是 GraphCliff 论文给定参数。无超参搜索。
- 每批124个普通训练样本（末批允许少于124）加随机1个正例对和1个负例对的成员并集；补充成员与普通样本不交叠，补充正负对允许共享一个成员。每个普通训练分子每轮恰好出现一次；额外样本重复完整记录。两臂批次完全一致。
- 主干 H256/L3/groups4/K3，dropout0（LongPoly内部原有0.1保留），原回归头不变；AdamW lr1e-4/wd1e-5/betas(.9,.95)，clip1；warmup10/cosine100/minlr1e-6；scheduler.step(epoch)在每轮结束调用，epoch从1计。
- 最多100轮；patience15/min_delta1e-4，选最低验证 Overall MSE。所有验证指标记为开发结果，不与论文 test 列混排。辅助梯度比/夹角仅首批抽样，不当全程上界。
- 每任务种子内随机打乱两组执行顺序；同初始化、同 batch序列哈希、同参数量。种子重复只估计初始化变动，不是三个独立数据集。
- 记录源文件/数据hash、环境、行ID、配对CSV、采样hash、两项loss、正负有效对、负例激活数、表示余弦、梯度范数比、验证预测。末尾独立重载checkpoint并用math.fsum复算指标。
- 任一任务没有正例/负例、公式/梯度/行成员检查失败、非有限梯度或CUDA故障，停止队列，不自动换阈值或重试。

## 判定与后续边界

先导扩大门槛（操作性规则，非显著性检验）：三种子任务宏平均Cliff至少改善1%，Overall恶化不超过1%；至少3/4任务在至少2/3种子上Cliff改善。否则暂停当前配方。即使达门槛也不自动启动30任务或ECFP组合。下一阶段另冻结完整官方train重训预算、原test最终评价；不得用已看过的test调参。

## 执行步骤

1. 实现最小独立入口，复用原开发划分/模型/评估和指纹/梯度工具；原model.py及历史实验不改。
2. 检查公式、边界、自配对/重叠、零权重梯度、空对行为；四任务覆盖；真实CUDA更新与checkpoint严格重载。
3. 冻结manifest后启动24次独立进程；队列持续写status/history及失败位置。
4. 独立重载24个模型，核验CSV/指标/初始化/采样哈希，给出逐任务逐种子结果和扩大判定。

输出：artifacts/cliff_contrastive_20261012（日志/权重）和docs/cliff_contrastive_20261012（协议、配对、审计、报告）。冻结后本文件和训练源码保持运行版本；完成状态写根task_plan.md。

依据：https://arxiv.org/html/2511.03170v3#A1；PyTorch CosineEmbeddingLoss官方文档及本机torch实现。论文只提出方向，没有给出这里的公式与超参。此前ACA/结构门控负结果保留为历史参照，因损失、批次与选模设置不同不作单因素比较。
