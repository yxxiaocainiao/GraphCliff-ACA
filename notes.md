# 来源与范围核对

## 2026-10-10：结构门控/投影核验

用户提出严格结构负例、投影归一化、MSE与alpha=.01，固定CHEMBL234四组。直接门控随机batch仅5个候选、82/83批空（float32合格有向负例440），所以所有组统一用29普通分子+一组三元组的训练内采样；91批/2905样本次数，重复三元组公平计入MSE对照。原h仍接受投影反向梯度，标量loss比不等于梯度比。原ACA多视角OR门控与本试验负例T>=.8不是同一语义。

完成MSE=.590215/.650638（Overall/Cliff），无门控投影=.619922/.649969，门控投影=.609107/.659068；完整组合没有优于匹配MSE。原h门控先49epochs后CUDA/cuBLAS错误，一次同配方重试再illegal instruction，按预算停组；两个部分权重不排名。三个完成checkpoint独立重载与CSV指标重算通过。完成模型均原骨干strict加载、推理不调用投影；旧项目/原先导未改，官方test不评价。本轮仅本地topic分支，不自动推送。完整证据在 docs/aca_local_20261010。

## 2026-10-10：公开发布授权与核验

用户明确要求将本方向单独发布为公开 GitHub 仓库。目标为 https://github.com/yxxiaocainiao/GraphCliff-ACA，origin 已配置至该用户仓库；upstream 仍指原作者，未向 upstream 写入。默认分支 codex/graphcliff-aca，不合并其他项目。

发布前补齐官方 shallow 历史：共12个待推送提交，137个当前跟踪文件，历史最大 blob=4,822,725 bytes。当前文件与完整历史凭据模式扫描无匹配；训练权重、artifacts 日志、邻近旧项目均排除。MIT 原许可和 ACA GPL-3.0 源码许可保留。首次推送916e715后已验证 GitHub visibility=PUBLIC；最终说明提交后再次核验本地/远端 SHA 与递归文件树。

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

六次拟合与独立核验完成。Cliff RMSE（MSE / MAE / ACA）：CHEMBL234=.622703/.632913/.694268；CHEMBL2047=.515139/.565489/.597020。ACA 相对 MAE 的 Cliff 恶化9.69%/5.58%，Overall 恶化12.15%/3.38%。两任务每个 ACA epoch 均有候选，回归与三元组 loss 均下降，但没有验证收益。仅单种子先导，暂停扩大该直接配方，不宣称所有监督对比学习无效。

六个重载预测最大误差 <1e-5（实际浮点舍入约1e-15），行 ID、标签和指标均匹配；报告入口另用 math.fsum 从 CSV 重算两种 RMSE。当前 graphcliff 环境无 sklearn，未安装新依赖。30 个用户 checkpoint 对初始审计 SHA 复核全未变；原模型/数据代码无改动。阶段证据与图保存在 docs/aca_pilot_20261010，完整权重和日志在 artifacts，均属于本仓库。

训练后重载发现新入口漏取 checkpoint 的 model_state_dict 外层；CHEMBL2047 MSE 完成 64 epochs、best=49，权重和历史有效。修复只影响重载，不改变训练/loss。recover 仅接收已满足终止条件的历史、校验划分/数据/模型与 loss 来源，恢复已完成 MSE 结果再从固定初始化训练后续两组；不续训部分 epoch、不重跑已完成拟合。恢复过程单独写 recovery.json，原始 provenance 不覆盖；恢复的 MSE 训练耗时记 null。

确认旧 experiments/aca_pilot 不是未运行方案：仅 CHEMBL234 三种子共六次已完成，MSE+平方 L2 的 Cliff RMSE 平均恶化约 3.01%，旧队列暂停。新轮用 MAE+非平方 L1，固定官方 README 示例；保留普通 MAE 以辨别回归损失变化，全部重新初始化。

ACANet 固定源码 4c43e602 的 clsar/model/loss.py 经官方 raw 在线下载核验 SHA256=ad7e1722383fb85693bfd821b0c5f32827f4cb2867c591865569c445c640df37，与复制的旧 vendor 文件一致。复用原定义而不改写公式。官方当前 HEAD 6274fa43 与固定源码不同，参数示例来源和实现来源分别记录。

第一轮预检通过公式与梯度：两候选三元组的 hinge 均值=1.55；alpha=0 等于 MAE 值和预测梯度；空候选为零对比项。真实 CUDA 批次均有限，CHEMBL234 train/valid=2632/292，valid cliff=128；CHEMBL2047=452/51，valid cliff=20。训练标签均符合 -log10(nM)，train/valid 无 canonical SMILES 交集。提供的 30 个 checkpoint 本轮不用于初始化。

## 路线 A 固定先导
输入为 benchmark_data 两任务官方train；Morgan radius2/2048，无手性。复用原 pooling 后512维接口及ACA开发划分工具，不使用ACA loss或采样。graph/fp/concat/gate四组；细节、预算和归因边界见 experiments/route_a/task_plan.md。原代码和旧实验保留；tools/已有未跟踪CUDA诊断不改。expression-skill/planning-with-files 搜索未找到，使用仓库现有计划文件约定。
路线A完成：234 graph/gate Overall=.585909/.623614、Cliff=.619991/.644121；2047 graph/gate=.433690/.439782、.459170/.465828。2047 concat=.409269/.375809（比graph改善5.63%/18.15%）；234 concat=.633122/.661596，均退步。8组重载预测最大差8.9e-16，标准库复算和源码/数据hash通过。暂停扩大门控，未执行多种子或test，未推送。
