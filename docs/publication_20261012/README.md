# 2026-10-12 完成实验发布范围

仓库：yxxiaocainiao/GraphCliff-ACA（PUBLIC）。目标分支：codex/route-a-30tasks-20261011。包含当前分支8个尚未推送历史提交及完成实验的新代码、计划、报告和小型复核证据；不合并或修改默认分支，不写作者upstream。

发布核验完成：内容提交[f5a3b91](https://github.com/yxxiaocainiao/GraphCliff-ACA/commit/f5a3b91627a5783f3590dfcc5e92a1cab37fc503)已推送，[分支入口](https://github.com/yxxiaocainiao/GraphCliff-ACA/tree/codex/route-a-30tasks-20261011)。本地/远端提交号一致；GitHub tree有756个文件，新增521个暂存文件包含520个允许内容路径及scope_check自身；禁止发布路径未出现。115个历史blob及新文件凭据模式扫描零命中，60份预测CSV/源hash验证通过。随后仅追加本完成记录，最新分支提交包含这条内容提交。

本次新增实验源码：cliff_contrastive、cliff_ecfp、readout_pilot及route_a_30的checkpoint/局部读出诊断入口。原model.py、dataset、原LICENSE/README及作者权重保持不变。运行中冻结的训练源码和计划未改。

公开结果包括：30任务训练报告/历史/预测、30个作者checkpoint回放预测、读出机制诊断与理论审阅、12个读出模型及48个margin四组模型的小型证据。后两阶段的60个有效模型各导出history、prediction CSV、summary，并导出3份运行manifest；全部60份预测用math.fsum复算并核对分区行ID，冻结源hash验证通过。validation.json记录所复制文件的SHA256。

排除：所有.pt/.pkl权重、artifacts目录和运行日志、首次invalid_shape实验、根论文PDF、paper_gap_audit的paper_text.txt全文、现有tools/诊断文件和__pycache__。这些内容保留本地，不删除。论文分析附原文来源，全文不转载。

这些是完成实验的公开记录，不代表新模型被证实有效。两个margin阶段为开发验证，未进行完整train/test评价。当前组合go=false；下一项系数校准只是未执行建议。历史计划中的“未发布”描述其记录时的状态，最新发布信息以README_ACA和根task_plan为准。

公开CSV和summary足以复算现有指标；重新训练/独立重载需对应环境及本地未公开权重/运行artifacts。历史manifest记录当时实际源码字节hash，Windows行尾变化可能使新检出字节不同；不要把git blob hash或新的复跑结果冒充当时hash或分数。
