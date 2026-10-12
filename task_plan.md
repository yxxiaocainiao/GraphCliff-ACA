# GraphCliff + ACA 计划

## 后续结构/投影核验

用户给出最小修复建议后，另开 codex/aca-structure-projection-20261010；执行范围和预算固定在 experiments/aca_local/task_plan.md。只做 CHEMBL234 四组，不覆盖原先导，不把机制推测当作已证实原因；新提交暂不自动推送。

本轮形成三组完整结果及原h分支两次CUDA中断记录。完整组合仍落后匹配MSE（Overall+3.20%、Cliff+1.30%），暂停扩配方；原h消融缺失，未完成项明确保留。阶段报告：docs/aca_local_20261010/report.md。

## 公开发布（用户本轮明确授权）

- [x] 核对本仓库干净分支、登录账号及上传范围：仅 GraphCliff-ACA 原始基线、ACA 代码与阶段证据；权重、artifacts 日志与旧项目排除。
- [x] 扫描当前文件及待推送历史中的凭据、大文件，保留 MIT/GPL 来源；补齐 shallow 来源历史以正常推送。12 个提交，137 个当前文件，历史最大 blob=4,822,725 bytes，凭据模式扫描无匹配。
- [x] 创建 yxxiaocainiao/GraphCliff-ACA 公开仓库，推送本方向分支，不合并其他项目。
- [x] 核对 GitHub PUBLIC、默认分支 codex/graphcliff-aca、远端 SHA 与文件树；初次推送 SHA=916e715，最终说明提交后再次核验。

## 当前请求

用户希望尝试在 GraphCliff 上借鉴 ACANet 对比学习，并提出新建仓库以免和历史修改混杂。本轮交付干净、独立的本地仓库；代码接入与试验另按下面阶段推进。

## 阶段与检查点

- [x] P0 核验用户新增的 30 个 checkpoint：CPU weights_only 加载、默认模型 strict state_dict 加载、有限值与 SHA256 检查全部通过；每个任务有对应 CSV。仅记录，不运行推理或测试集评价。
- [x] P0 只读核对旧项目 Git 分支、提交、未提交路径及归属说明。
- [x] P0 在不存在的新路径 `D:/GraphCliff-ACA` 从官方 GraphCliff 独立克隆，建立 `codex/graphcliff-aca` 分支。
- [x] P0 记录来源 SHA、项目边界和后续实验范围，保留原作者文档与许可。
- [x] P0 验证独立 Git 目录、97 个原始跟踪文件未改、旧项目分支/HEAD/status 未变；本轮交付仅新增 4 份本项目说明文件。
- [x] P0 接入前核对旧 ACA 迁移的实现/结果与当前官方 ACA 版本、许可、标签尺度、margin、距离、归约及无有效三元组行为。
- [x] P0 冻结小规模试验：同主干、同初始化/划分/选模规则；MSE、MAE、MAE+ACA 三组。任务、种子、调参预算在运行前写明。
- [x] P1 实现最小可选 ACA 适配；覆盖表示返回、loss 调用、配置、三元组数量与两项 loss 记录，保留纯基线入口。
- [x] P1 验证零权重退化、三元组训练边界、非空/空三元组梯度与一次真实批次前后向。
- [x] P1 完成冻结先导、独立核验预测和指标，再决定是否扩大或设计 Triplet Loss 变体。

## 固定边界

初始化阶段未训练。2026-10-10 用户授权开始下面六次训练；无官方 test 评价、无 GitHub 发布。旧分支与旧任务计划不改。不能以旧结果冒充新基线结果。

## 2026-10-10 固定先导

1. 核对旧 ACA：CHEMBL234 三种子六次训练，MSE+平方 L2 的 Cliff RMSE 均值恶化 3.01%，已暂停。
2. 新试验仅 CHEMBL234_Ki、CHEMBL2047_EC50，split/train seed=42，官方 train 内分层留出 10%；不读取 test 标签。
3. 每任务独立从相同随机初始化跑 MSE、MAE、MAE+ACA 三组。ACA 固定 alpha=.1、lower=.2、upper=1、p=1、squared=False、structure gate=False，参考官方 README 示例，不执行调参。
4. 原主干 H256、3层、4组、K3、dropout=0、batch32；AdamW lr=.0001、wd=.00001、betas=(.9,.95)、clip=1；100 epochs 上限、warmup10、patience15；验证 MSE 严格最小选模（min_delta=0）。
5. 先检查公式、零权重、空三元组、真实 CUDA 批次前后向。非有限值、尺度不符、首个 ACA epoch 全无三元组时停止，不补调参。
6. 六次完成后独立重载 checkpoint 核验验证预测和 RMSE，记录 Cliff RMSE、候选/激活三元组、两项 loss。单种子仅作可行性，不宣布普遍有效。

影响范围：新增 experiments/aca，artifacts/aca_pilot_20261010（忽略大文件），本仓库计划、记录和说明；原始模型、原 checkpoint、旧工程不改。

预检完成，训练源码冻结提交为 1cf5643；两个任务各三组已启动。vendor 源码保留原作者空白字符，diff --check 的 vendor 提示属原样复制；新增代码通过检查。.gitattributes 为 vendor 禁止行尾转换，确保 Windows 重新检出仍能通过固定 SHA 校验。

六次已完成（54/92/89、64/35/82 epochs）。独立重载与 CSV 标准库重算指标通过。ACA 比 MAE 的 Cliff RMSE 在 CHEMBL234/2047 分别恶化 9.69%/5.58%，Overall 也均恶化。阶段决策：暂停扩大当前直接迁移配方，不自动追加种子、任务或调参；保留基线与负结果。若继续研究，优先诊断损失尺度和训练三元组的结构局部性，另冻结单机制试验后再改变体。完整阶段报告在 docs/aca_pilot_20261010/report.md。

## 路线 A（2026-10-10）
用户授权图-指纹融合固定先导；独立分支 codex/route-a-fusion-20261010。执行计划见 experiments/route_a/task_plan.md，两任务四组共8次，不自动扩大或发布。
路线A八组已完成、独立重载/CSV复算/训练源码hash核验通过。gate两任务均落后graph，暂停扩大；concat仅2047有收益。详细见experiments/route_a/README.md和docs/route_a_20261010/report.md。

## 路线A残差与学习率受控核验
用户授权下一步试验；分支codex/route-a-residual-20261010。冻结13次新训练（12次主比较+1次concat100早停诊断），四旧对照复用并重审。执行计划experiments/route_a_residual/task_plan.md。
残差/LR核验本轮按CUDA停止条件结束：完成9次新拟合，重审4个旧对照；244图组60轮后CUDA中断，另3组未启动。三个任务完整，四任务目标尚缺244。暂停扩大配方，未自动恢复；报告docs/route_a_residual_20261010/report.md。

## 30任务GraphCliff与concat正式匹配比较（2026-10-11）
用户授权按原仓库官方划分比较30任务，目标北京时间10月11日08:30。固定单seed42/两模型，当前入口默认batch128/min_delta1e-4；完整协议见experiments/route_a_30/task_plan.md。输出artifacts/route_a_30_20261011和docs/route_a_30_20261011。

## 读出支路受控先导（2026-10-11）

用户授权“试试吧”：四个既有任务、graph/final/shallow、seed42，共12次新拟合，使用完整官方train与原test。固定100轮末模型，不用test早停；预检零残差/梯度/参数匹配及CUDA真实更新通过。协议冻结于experiments/readout_pilot/task_plan.md及artifacts/readout_pilot_20261011/manifest.json。结果单独写入docs/readout_pilot_20261011，不能与此前90% train验证早停成绩直接混排。

2026-10-12核验：12/12完成，四任务配对齐全，CSV复算与初始化hash检查通过。graph宏平均0.627449/0.670354，final0.617793/0.657577，shallow0.626620/0.672519。浅层Cliff恶化0.32%，暂停扩大；final Cliff均值改善1.91%但仅2/4任务胜出，单seed不声明稳定改进。详细见docs/readout_pilot_20261011/conclusion.md。冻结源码/先导计划保留原运行版本。

## 结构局部 cosine-margin 先导（2026-10-12）

用户授权按建议先验证损失，再考虑ECFP组合。四既有任务×三初始化种子×MSE/margin，共24次开发拟合；原官方train/test成员不变，先导仅train内验证。完整冻结预算和Go/No-Go规则见experiments/cliff_contrastive/task_plan.md。P0实现/预检，P1训练与独立重载核验；不自动扩大30任务或推送。

P0完成：配对覆盖、全部计划批次、数值公式、零权重/空对梯度、四任务真实CUDA更新与严格重载通过，manifest已冻结。P1已启动后台24次队列；状态/失败位置见artifacts/cliff_contrastive_20261012/status.json。新增脚本/冻结计划、配对覆盖文档；原模型和历史训练入口保留。训练结果仍待完成与独立审计。

首次启动捕获MSE标签广播错误，已终止并保留invalid_shape归档，首组权重/指标不纳入结果。修复原collate标签及逐批形状/数值断言后重新预检冻结；不改变既定损失、超参、任务或划分。

P1完成并核验：24/24、12个完整配对块，自动独立重载审计通过，汇报时24份CSV再次标准库复算通过。开发宏平均MSE→margin：Overall .595254→.586570（改善1.46%）、Cliff .647402→.632448（改善2.31%）；234/2147获益3/3和2/3种子，2047/244仅1/3。未达至少3/4任务获益至少2/3种子的事前门槛，go=false；不自动扩大或组合ECFP。结论docs/cliff_contrastive_20261012/conclusion.md。冻结训练源码/计划保持原运行版本，官方test未评价。

## 四组 ECFP × cosine-margin 先导（2026-10-12）

用户进一步认可四组比较并确认官方划分。旧损失先导的no-go决定保留，此次是新授权的组合检验，不是自动扩大。复用24个纯图对照，只补24次concat与concat_margin；固定四任务/三种子/原开发划分，test不参与。预算、适配方式和事前组合判断见experiments/cliff_ecfp/task_plan.md；P0实现预检，P1后台训练与48模型来源/指标审计。

P0通过：四任务/三seed图初始化与旧对照一致；512维图辅助表示、零权重梯度、fp/head无直接辅助梯度、真实CUDA更新、严格重载通过；两新组各7,135,822参数，manifest冻结。预检后的初始化来源记录改为实际构造哈希，再完整预检冻结（首份prelaunch manifest保留），未执行无效训练。P1补跑24次后台队列已启动，状态artifacts/cliff_ecfp_20261012/status.json；旧24模型全部只读引用。

P1完成：新24/24及重载审计通过，48模型/12四组配对齐全，旧引用hash、初始化和采样一致。Overall/Cliff宏平均：graph_mse .595254/.647402，graph_margin .586570/.632448，concat_mse .595353/.630269，concat_margin .589821/.633271。组合平均Cliff不及两个单独方案，没有任务至少2/3种子同时胜过二者，go=false；暂停此组合，未自动扩任务、调参或test。结论docs/cliff_ecfp_20261012/conclusion.md。

## 完成实验的分支发布（2026-10-12）

用户询问是否可以提交，按既有origin公开发布授权执行。范围仅本仓库：当前codex/route-a-30tasks-20261011上的8个未推送历史提交，加上此次完成实验的代码/计划/报告/必要CSV与JSON复核证据及README_ACA更新。仅推送origin到同名分支，不合并或改默认分支，不写upstream。

P0/P1完成：整理readout、cosine-margin、ECFP组合、30任务及checkpoint/划分诊断的完成证据；从artifacts仅导出60个有效模型的history/预测/summary和3份manifest，60份预测CSV复算/分区行ID及冻结源hash通过，5份Python源码解析通过。521个发布文件约6.9MB、115个历史blob凭据模式检查零命中，暂存diff检查通过。权重、运行目录、首次无效训练、根论文PDF、paper_text全文及tools未发布。

P2完成：内容提交f5a3b91627a5783f3590dfcc5e92a1cab37fc503已推送到origin/codex/route-a-30tasks-20261011，git ls-remote与本地SHA一致；GitHub tree有756个文件，520个内容允许路径全部存在（另有scope_check.json），禁止路径未出现。默认分支仍codex/graphcliff-aca、仓库PUBLIC。原model/dataset/main/train/README/LICENSE与upstream快照一致。此后仅补本发布完成记录，不新增实验或改变默认分支。
