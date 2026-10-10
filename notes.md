# 来源与范围核对

## 2026-10-10：独立仓库建立

- 来源：`https://github.com/dmis-lab/GraphCliff.git`，通过 `git clone --depth 1 --origin upstream` 获取。
- 来源 SHA：`d18857db0303346c0be715a56febf71da1768592`。
- 新目录：`D:/GraphCliff-ACA`；建立前路径不存在；分支：`codex/graphcliff-aca`。
- 官方模型、训练器、数据处理、benchmark_data、README 与 LICENSE 来自该固定快照，没有混入旧项目的工作区改动。
- 当前 `upstream` 指向原作者仓库。尚未创建用户自己的 GitHub 仓库或配置 `origin`，未执行推送。
- 旧目录本轮核对的 HEAD：`D:/GraphCliff-Pair-FPPool-LongPoly`，`codex/residual-fp-centered-cross-20261007`，`5228492867295e5f18c2be49d67ac8723a7f5673`。
- 旧目录未提交路径：`.gitignore` 修改；`a30/`、`experiments/mechanism_retry/a30_protocol_20261008.md`、`experiments/mechanism_retry/cancel/`、`experiments/mechanism_retry/cancellation_candidate_20261009.md`、`tools/make_a30_configs.py` 未跟踪。`a30/README.md` 将这些路径登记在同一研究线。本轮不修改或收集提交这些文件。
- `cancel/model.py` 包含 ScoreDiff、FixedTransfer、RelSup、GatedTransfer。这些是参考残差/相对监督候选，不是 ACA 三元组实现，不复制到新仓库。
- 新旧上游版本可能不同；后续以本仓库同期基线比较 ACA，不把历史分数作为同协议对照。
- 用户指定的 expression-skill、planning-with-files 在标准 `.codex/skills` 和 `.agents/skills` 搜索未找到；本仓库用 task_plan.md/notes.md 记录计划和证据，不声称已使用缺失技能。
- 完成检查：`git diff --exit-code d18857db0303346c0be715a56febf71da1768592 -- .` 通过，97 个上游跟踪文件无内容修改；Git directory 和 common directory 均为 `D:/GraphCliff-ACA/.git`，worktree list 只有本目录。旧项目 HEAD、分支、status 与本轮开始一致。
- 本轮新增且仅提交 `AGENTS.md`、`README_ACA.md`、`task_plan.md`、`notes.md`。未修改旧项目或运行任何训练/推理。

## 待解决问题

1. 早期 ACA 移植到底采用什么 embedding、三元组条件和参数，是否已有可复用实现？
2. 官方实现的 MSE 选项是否同时改变 embedding 距离定义，如何清楚区分移植配方与最小控制变量试验？
3. 当前 batch/任务的有效三元组覆盖是否足够？
4. 未来变体相对 ACA/PrismNet 的新意是什么？尚未确认，先做 ACA 对照。

## 2026-10-10：用户提供 checkpoint 核验

- 位置：`ckpt/<CHEMBL任务>/<时间戳>/best_model.pt`，30 个独立任务各 1 个，合计 723,320,700 bytes。
- 30/30 均能通过 `torch.load(map_location='cpu', weights_only=True)` 加载；封装仅包含 `model_state_dict`。
- 30/30 均可 strict 加载至当前 GraphCliffRegressor(38,13,H256,3 layers,4 groups,K3)，默认模型参数数 6,021,198；所有 state tensor 均为有限值。
- 各任务均有本仓库 benchmark_data 中的对应 CSV。这里只枚举文件名，没有读取训练或测试标签。
- 已记录文件级 SHA256 与形状兼容性结果：`docs/checkpoint_audit_20261010.json`。未核验官方逐文件哈希或这些权重的实际训练划分，布局兼容不等于来源/评价协议已证明。
- 当前 main.py 检测到 `ckpt/<dataset>/*/best_model.pt` 后设置 epochs=0，随后执行官方 test 评价。后续 ACA 应使用单独训练入口和输出位置，防止误加载现有权重后跳过新训练。
- 这些 checkpoint 可用于加载兼容检查和明确标记的历史模型分析。正式 ACA 对照应同初始化、同 train/validation 划分重训；若设计微调实验，需先核验权重训练划分，并让两臂从同一合法权重起点出发。
- 本轮模型 forward、optimizer 步、训练、test 评价均为 0；未修改或移动用户权重。
# 2026-10-10 ACA 接入与先导

确认旧 experiments/aca_pilot 不是未运行方案：仅 CHEMBL234 三种子共六次已完成，MSE+平方 L2 的 Cliff RMSE 平均恶化约 3.01%，旧队列暂停。新轮用 MAE+非平方 L1，固定官方 README 示例；保留普通 MAE 以辨别回归损失变化，全部重新初始化。

ACANet 固定源码 4c43e602 的 clsar/model/loss.py 经官方 raw 在线下载核验 SHA256=ad7e1722383fb85693bfd821b0c5f32827f4cb2867c591865569c445c640df37，与复制的旧 vendor 文件一致。复用原定义而不改写公式。官方当前 HEAD 6274fa43 与固定源码不同，参数示例来源和实现来源分别记录。

第一轮预检通过公式与梯度：两候选三元组的 hinge 均值=1.55；alpha=0 等于 MAE 值和预测梯度；空候选为零对比项。真实 CUDA 批次均有限，CHEMBL234 train/valid=2632/292，valid cliff=128；CHEMBL2047=452/51，valid cliff=20。训练标签均符合 -log10(nM)，train/valid 无 canonical SMILES 交集。提供的 30 个 checkpoint 本轮不用于初始化。
