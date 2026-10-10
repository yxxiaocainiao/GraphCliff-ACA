# 30任务匹配比较（2026-10-11，目标北京时间08:30）

- [x] 核对run.sh的30个任务、官方split、源码/输入hash和GPU真实批次压力检查。
- [x] 冻结graph/concat唯一两组和当前main.py默认训练协议，测试快照恢复与指标复算。
- [ ] 启动独立后台顺序队列，60次seed42拟合；每组独立进程，自动保留日志/状态/5轮快照。
- [ ] 各组训练完成后重载固定最佳checkpoint，验证回放并一次性评价官方test；生成30任务宏平均与胜平负。

任务仅run.sh名单，不扩展其他CSV。两组独立从同图主干初始化训练，不使用作者/用户旧权重，不把旧先导当新结果。Morgan radius2/2048 bit无手性，Linear2048→512+ReLU；concat1024→128回归头。沿用当前main.py：batch128，seed/split42，train内cliff分层留出10%；H256/L3/G4/K3/dropout0（LongPoly内部.1保留）；AdamW lr1e-4/wd1e-5/betas(.9,.95)，clip1，warmup10/100epochs/min_lr1e-6，epoch末scheduler.step(epoch)（epoch从1起），patience15/min_delta1e-4。train/valid DataLoader共用seed42的generator，沿用原入口随机初始化后的RNG，不叠加其他loss/modules。区别于先导batch32/min_delta0/scheduler从0起；不与先导数值直接归因。

官方test不用于训练/调参/选模，仅完成固定checkpoint之后评价，禁止根据test结果选分支、任务、epoch或追加调参。当前仓库划分/入口协议不等于论文历史训练协议已完全复现。

60次主拟合；仅CUDA故障允许每组最多1次操作性恢复，全队列最多3次恢复。保存完整model/optimizer/best/RNG/loader状态，每5epoch覆盖本组latest快照，恢复从快照继续，至多重放4轮；无快照则同seed从头重跑。第二次CUDA错误、非CUDA错误或总恢复预算耗尽即停止队列并报告缺失，不自动改batch/精度/环境/驱动，不以失败尝试择优。恢复日志全部保留。睡眠/关机/ GPU故障可能破坏08:30期限，时间估计不是保证。

输入benchmark_data，输出artifacts/route_a_30_20261011和docs/route_a_30_20261011，均在本仓库；旧模型/ACA/先导/tools/其他项目不改、不发布。状态status.json、日志logs/，完成后汇总summary.csv/report.md。输出JSON原子替换，原权重不覆盖。快照仅加载本队列自己的可信文件（含Python/NumPy RNG对象）。
