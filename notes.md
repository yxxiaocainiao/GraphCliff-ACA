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
