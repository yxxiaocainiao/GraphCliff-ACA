# 30任务匹配比较（2026-10-11，目标北京时间08:30）

- [x] 核对run.sh的30个任务、官方split、源码/输入hash和GPU真实批次压力检查。
- [x] 冻结graph/concat唯一两组和当前main.py默认训练协议，测试快照恢复与指标复算。
- [x] 启动独立后台顺序队列，60次seed42拟合；每组独立进程，自动保留日志/状态/5轮快照。
- [x] 各组训练完成后重载固定最佳checkpoint，验证回放并一次性评价官方test；生成30任务宏平均与胜平负。

## 论文基线差异核验
- [x] 确认本轮60次从随机初始化训练，未使用现有原始checkpoint；本地图对照不能标作论文结果。
- [x] 用已有30个checkpoint和本轮同一test评价函数回放；核验权重/数据hash、严格载入和CSV指标，记录宏平均及逐任务结果。
- [x] 对照论文表格，记录已确认区别与尚未确认的历史训练条件；不启动追加训练或按test调参。

回放宏平均0.664575/0.756855，与截图三位小数一致。报告docs/graphcliff_checkpoint_audit_20261011/report.md；本轮随机训练graph0.698532/0.798011、concat0.698618/0.789335不能称为超过论文GraphCliff。

历史代码核验已发现：作者cf6c281旧版完整train训练、test_loss选模早停；408f783更新引入train内10%验证、valid_loss选模。本轮采用后者。发布checkpoint的确切训练版本仍未确认，不将旧版test选模当成独立测试协议，不直接追加30任务重跑。差异证据已写入核验report；本轮融合结果保留为探索性匹配比较。

任务仅run.sh名单，不扩展其他CSV。两组独立从同图主干初始化训练，不使用作者/用户旧权重，不把旧先导当新结果。Morgan radius2/2048 bit无手性，Linear2048→512+ReLU；concat1024→128回归头。沿用当前main.py：batch128，seed/split42，train内cliff分层留出10%；H256/L3/G4/K3/dropout0（LongPoly内部.1保留）；AdamW lr1e-4/wd1e-5/betas(.9,.95)，clip1，warmup10/100epochs/min_lr1e-6，epoch末scheduler.step(epoch)（epoch从1起），patience15/min_delta1e-4。train/valid DataLoader共用seed42的generator，沿用原入口随机初始化后的RNG，不叠加其他loss/modules。区别于先导batch32/min_delta0/scheduler从0起；不与先导数值直接归因。

官方test不用于训练/调参/选模，仅完成固定checkpoint之后评价，禁止根据test结果选分支、任务、epoch或追加调参。当前仓库划分/入口协议不等于论文历史训练协议已完全复现。

60次主拟合；仅CUDA故障允许每组最多1次操作性恢复，全队列最多3次恢复。保存完整model/optimizer/best/RNG/loader状态，每5epoch覆盖本组latest快照，恢复从快照继续，至多重放4轮；无快照则同seed从头重跑。第二次CUDA错误、非CUDA错误或总恢复预算耗尽即停止队列并报告缺失，不自动改batch/精度/环境/驱动，不以失败尝试择优。恢复日志全部保留。睡眠/关机/ GPU故障可能破坏08:30期限，时间估计不是保证。

输入benchmark_data，输出artifacts/route_a_30_20261011和docs/route_a_30_20261011，均在本仓库；旧模型/ACA/先导/tools/其他项目不改、不发布。状态status.json、日志logs/，完成后汇总summary.csv/report.md。输出JSON原子替换，原权重不覆盖。快照仅加载本队列自己的可信文件（含Python/NumPy RNG对象）。

后台队列已启动：源码3e0a902，父进程PID40476，启动时间2026-10-11 00:16（北京时间）。恢复自检、200次真实CUDA更新通过；当前完成数/故障以status.json为准，报告自动生成，未设置聊天提醒。不要重复启动。
