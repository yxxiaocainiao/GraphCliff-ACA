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

## 2026-10-12 结构局部 cosine-margin 新先导

用户授权按论文不足先单独试损失，暂不组合ECFP。实验设计采用experimental-design的匹配区组/重复原则；expression-skill/planning-with-files在本机技能根仍未找到，沿用仓库计划文件。新入口experiments/cliff_contrastive/run.py复用base.development/model/evaluate以及aca_local指纹和梯度探针。原模型/历史入口不改；四任务×seed42/43/44×MSE/margin共24次，仅官方train内验证，不读test标签或作者权重。

损失采用PyTorch原生CosineEmbeddingLoss：结构T>=.8、活性差<.2为正例；结构T>=.8、活性差>=1为负例。正负类各均值后等权，margin=.5、系数=.01；直接约束原512维分子表示，无额外参数。这些是冻结试验设定，不是论文给出的配方；与之前ACA相比改变了多个机制，不据新旧差值作单项归因。

预检通过：234/2047/2147/244训练侧无向正负对417/220、83/24、92/28、467/352；开发train/valid为2632/292、452/51、1045/117、2229/247。全部三seed×100轮采样能注入两类；公式/空对/零权重梯度/真实三步CUDA更新/严格重载检查通过。官方train=开发train+valid，test成员不变；辅助约束只用开发train标签。两组匹配批次且普通分子每轮一次，注入成员额外重复；记录采样hash/梯度和几何统计。开发选模min_delta=.0001，与无该阈值的绝对最低误差checkpoint不完全等同。

预算/扩大门槛事前写入experiments/cliff_contrastive/task_plan.md并与源码hash冻结在artifacts/cliff_contrastive_20261012/manifest.json。队列已后台启动，进度写status.json，失败停止无自动重试；结果写docs/cliff_contrastive_20261012。暂不能称已有效。未新增依赖、未push、未启动全train/test或30任务。

首组日志人工检查发现新入口把训练标签另建为[B]，与模型[B,1]输出在MSE中错误广播。立即终止本队列的两个Python进程，首组未完成结果全部作废，旧目录移至artifacts/cliff_contrastive_20261012_invalid_shape保留。修复为直接复用collate_graphs原[B,1]标签，每批断言形状一致且与所采行标签逐值相等；重新预检、冻结和从零启动，不复用无效权重。最初预检使用collate标签，因此没有覆盖训练循环另建标签的错误；运行首组检查捕获了该问题。

完成汇报：24/24有效模型，12个配对块，独立重载审计及24份CSV再次复算通过，队列complete且无剩余进程。Overall均值.595254→.586570（+1.46%改善），Cliff .647402→.632448（+2.31%改善）；Cliff7/12配对块胜出，234/2047/2147/244获益种子数3/1/2/1。预定任务稳定性门槛仅2/4而非>=3/4，因此go=false；平均正信号不足以自动扩大。验证侧严格负对234/2047/2147/244仅4/0/0/5，表示分离的泛化机制证据稀少。当前为官方train内验证开发结果，未做完整train/test或ECFP组合，不宣称超过论文。详细表与边界见docs/cliff_contrastive_20261012/conclusion.md。

## 2026-10-12 用户确认四组 ECFP×margin 比较

新增独立experiments/cliff_ecfp适配，复用原route_a.Predictor('concat')及冻结cliff_contrastive训练循环；用独立模块命名空间替换接口，不改旧源码/运行文件。辅助损失仍约束graph pooled512，训练内Morgan筛配对与原规则一致。纯图24模型复用，仅新24拟合；四任务三seed、官方train内10%验证，不读test标签。所有旧权重/summary/CSV/history及源码hash冻结，新concat图初始化与对应旧纯图一致，两个拼接组全模型初始化一致；FP和新头第一层沿用既有seed1042，未声称三seed同时重抽所有参数。

四任务预检通过：零权重梯度、拼接值、h512辅助入口、辅助梯度不直接进fp/head、真实3步GPU更新、strict重载。两concat模型各7,135,822参数；额外容量约1.11M，拼接vs纯图仍有容量混杂，concat_mse vs concat_margin容量匹配。实际初始哈希由构造时记录，与旧对照核验，不把预期值冒充实测。预检及源码manifest已冻结；24次队列启动，失败停止无自动重试。结果docs/cliff_ecfp_20261012；暂无组合有效结论。组合门槛须同时胜过单独fp和单独loss，详见冻结计划。未新装依赖、未改其他仓库、未push。

四组完成：24个新模型独立重载审计通过，48总模型/12完整配对；状态complete无残余训练进程。Overall/Cliff宏平均graph_mse .595254/.647402、graph_margin .586570/.632448、concat_mse .595353/.630269、concat_margin .589821/.633271。组合Cliff较单独拼接/单独损失分别恶化约.48%/.13%；无任务至少2/3种子同时胜过二者，go=false。当前组合没有稳定互补证据，不据小均值差声称显著或把原因归为梯度冲突。官方test未评价，仍是同固定train内验证的开发结果；不自动扩大。结论docs/cliff_ecfp_20261012/conclusion.md。

六次拟合与独立核验完成。Cliff RMSE（MSE / MAE / ACA）：CHEMBL234=.622703/.632913/.694268；CHEMBL2047=.515139/.565489/.597020。ACA 相对 MAE 的 Cliff 恶化9.69%/5.58%，Overall 恶化12.15%/3.38%。两任务每个 ACA epoch 均有候选，回归与三元组 loss 均下降，但没有验证收益。仅单种子先导，暂停扩大该直接配方，不宣称所有监督对比学习无效。

六个重载预测最大误差 <1e-5（实际浮点舍入约1e-15），行 ID、标签和指标均匹配；报告入口另用 math.fsum 从 CSV 重算两种 RMSE。当前 graphcliff 环境无 sklearn，未安装新依赖。30 个用户 checkpoint 对初始审计 SHA 复核全未变；原模型/数据代码无改动。阶段证据与图保存在 docs/aca_pilot_20261010，完整权重和日志在 artifacts，均属于本仓库。

训练后重载发现新入口漏取 checkpoint 的 model_state_dict 外层；CHEMBL2047 MSE 完成 64 epochs、best=49，权重和历史有效。修复只影响重载，不改变训练/loss。recover 仅接收已满足终止条件的历史、校验划分/数据/模型与 loss 来源，恢复已完成 MSE 结果再从固定初始化训练后续两组；不续训部分 epoch、不重跑已完成拟合。恢复过程单独写 recovery.json，原始 provenance 不覆盖；恢复的 MSE 训练耗时记 null。

确认旧 experiments/aca_pilot 不是未运行方案：仅 CHEMBL234 三种子共六次已完成，MSE+平方 L2 的 Cliff RMSE 平均恶化约 3.01%，旧队列暂停。新轮用 MAE+非平方 L1，固定官方 README 示例；保留普通 MAE 以辨别回归损失变化，全部重新初始化。

ACANet 固定源码 4c43e602 的 clsar/model/loss.py 经官方 raw 在线下载核验 SHA256=ad7e1722383fb85693bfd821b0c5f32827f4cb2867c591865569c445c640df37，与复制的旧 vendor 文件一致。复用原定义而不改写公式。官方当前 HEAD 6274fa43 与固定源码不同，参数示例来源和实现来源分别记录。

第一轮预检通过公式与梯度：两候选三元组的 hinge 均值=1.55；alpha=0 等于 MAE 值和预测梯度；空候选为零对比项。真实 CUDA 批次均有限，CHEMBL234 train/valid=2632/292，valid cliff=128；CHEMBL2047=452/51，valid cliff=20。训练标签均符合 -log10(nM)，train/valid 无 canonical SMILES 交集。提供的 30 个 checkpoint 本轮不用于初始化。

## 路线 A 固定先导
输入为 benchmark_data 两任务官方train；Morgan radius2/2048，无手性。复用原 pooling 后512维接口及ACA开发划分工具，不使用ACA loss或采样。graph/fp/concat/gate四组；细节、预算和归因边界见 experiments/route_a/task_plan.md。原代码和旧实验保留；tools/已有未跟踪CUDA诊断不改。expression-skill/planning-with-files 搜索未找到，使用仓库现有计划文件约定。
路线A完成：234 graph/gate Overall=.585909/.623614、Cliff=.619991/.644121；2047 graph/gate=.433690/.439782、.459170/.465828。2047 concat=.409269/.375809（比graph改善5.63%/18.15%）；234 concat=.633122/.661596，均退步。8组重载预测最大差8.9e-16，标准库复算和源码/数据hash通过。暂停扩大门控，未执行多种子或test，未推送。

## 路线A机制解释审计
旧best19/26实际停34/41；指纹是bit而非count；gate额外157万包含fp105万和gate52万；gate在234优于concat。此前结果不证明收敛失配、小样本规律或动态扰动。新任务2147/244 train=1045/2229；拟合预算与边界见experiments/route_a_residual/task_plan.md。
核验结果：234 residual/slow Cliff=.667560/.634222，均落后graph .619991；2047=.470198/.552375，concat .375809仍最好；2147=.581771/.640338，concat .571329最好。concat100 best51=.631870/.667667；第7轮起轨迹与旧运行分叉，原因未定位，模拟早停66已包含best51，未支持延长训练修复。244图组保存60轮后CUDA launch failure，无重试，后3组未启动；13完成模型独立审计通过。仅审计函数为处理未完成与同轨迹比较而改动，训练AST与冻结源码匹配。

## 30任务夜间队列
run.sh唯一30任务清单，训练内留出后总35020分子；原官方train/test标记保留。两组均使用当前main.py默认超参而非先导batch32。最终test不参与选模；与论文历史训练是否一致未知，不宣称精确复现。每5轮快照和有限CUDA恢复属于操作性恢复，不调参。压力检查两组每批约.09-.10秒，100轮上限外推训练约1.5小时；含启动/写盘/评价按2-4小时预留，08:30不保证。

## 2026-10-11 论文基线差异核验
用户指出本轮图对照与论文不一致。已有30个checkpoint以同一test评价函数回放得到宏平均RMSE0.664575/Cliff0.756855，三位小数对应截图0.665/0.757，234/2047/2147逐任务抽查一致。权重与数据hash、严格载入、CSV指标复算通过。本轮60次从随机初始化训练，图对照0.698532/0.798011，concat0.698618/0.789335；不能宣称超过论文基线。历史训练条件差距尚未查清，不单项归因、不启动追加训练。报告docs/graphcliff_checkpoint_audit_20261011/report.md。原代码/权重/数据/其他工程未改。
