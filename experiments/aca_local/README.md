# 结构负例 + 投影的单任务核验

GPL-3.0；复用 experiments/aca/vendor 的未改动 ACA 源码，其 MIT GraphCliff 骨干保持原样。先导计划见 task_plan.md，配置见 protocol.json。

原作者多视角结构门控包含指纹/骨架/SMILES 的 OR 及结构不相似正例，不等于本试验的单一负例门控。本目录为每个新 ACALoss 实例提供独立的 get_structure_mask：仅 T(anchor,negative)>=.8，与原标签掩码 AND；原标签阈值、动态margin、hinge和全部候选分母不变。直接取原实现 triplet 项另加 MSE，避免 squared=True 同时改变距离。

正例仍仅由 |Δy|<.2 定义，可能跨骨架；严格负例为 |Δy|>=1 且 Morgan(radius2,bits2048,无手性) Tanimoto>=.8。不宣称它等于 MoleculeACE 多视角 cliff 标记。投影采用512→128→64 MLP+ReLU，L2 normalize；距离保留L1（不平方），64维归一化L1最大16，而不是L2距离最大2。

Loss = MSE + .01*triplet。投影头没有 detach，仍回传骨干；不保证 h 纯净或无冲突。每 epoch 首 batch 记录共享 AtomEncoder/GraphCliffEncoder/SAGPooling 参数的梯度范数比和 cosine，排除回归头/投影头；在全参数梯度裁剪前测量，仅是抽样诊断，固定alpha不保证辅助梯度<10%。

普通随机 batch 预检只有5个首epoch候选、82/83批为空。训练四组都使用相同的三元组batch：29普通分子+1完整训练内三元组，普通训练样本每epoch出现一次，额外重复三元组成员；共91批/2905次样本。预检使用与训练一致的float32标签，有440个有向合格负例对（初次float64检查为442，边界精度不同），首epoch588个候选、0空批。普通 MSE 也用此采样，旧均匀采样结果只作历史参考。

投影初始化在 fork_rng 内，骨干初值和后续CPU/CUDA随机状态与无投影组一致。checkpoint 单独保存原骨干与投影权重，原 GraphCliff 可 strict 加载 model_state_dict 并直接单分子预测；推理不执行投影头。

```powershell
python experiments/aca_local/pilot.py check
python experiments/aca_local/pilot.py run --arm mse
python experiments/aca_local/pilot.py run --arm ungated_projected
python experiments/aca_local/pilot.py run --arm gated_raw
python experiments/aca_local/pilot.py run --arm gated_projected
python experiments/aca_local/pilot.py audit
```

输出在 artifacts/aca_local_20261010；已存在的 arm 输出拒绝覆盖。仅 CHEMBL234、seed42、官方train内验证；无官方test、无超参搜索。四组只能分辨在当前配方下门控的条件效应和投影+归一化的组合效应，不能单独归因投影与归一化各自作用，也不能据单种子证明论文方法有效。
