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
