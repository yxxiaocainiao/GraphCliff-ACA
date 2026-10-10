# 30任务 GraphCliff vs concat 队列

2026-10-11凌晨用户授权启动，目标北京时间当天08:30。冻结训练提交3e0a902，分支codex/route-a-30tasks-20261011。run.sh的30任务，seed42，两组从头训练，共60次；不使用旧权重或旧先导成绩替代。

输入：benchmark_data官方train/test标记。验证集按原dataset.split_train_valid在官方train内分层留出10%。graph/concat使用相同当前main.py默认参数（batch128/min_delta1e-4/scheduler从epoch1起），因此不同于先导batch32。test仅在每个固定最佳checkpoint训练结束后评价一次，不用于调参或择优。当前原入口协议未声称等于论文历史训练协议。

后台顺序运行，每组独立进程；每5轮保存model/optimizer/best/全部RNG及loader generator快照。CUDA异常每组最多1次恢复，全队列最多3次；其余错误或预算耗尽停止并显式报告缺失。只恢复本队列可信文件，不修改共享环境/驱动/精度/batch。恢复从完整快照继续，没快照则同seed重新开始；失败尝试日志保留，不择优。

输出：

- `artifacts/route_a_30_20261011/status.json`：队列状态与完成模型数。
- `artifacts/route_a_30_20261011/logs/`：每组每次尝试日志。
- `artifacts/route_a_30_20261011/<task>/<arm>/progress.json`：当前epoch与选模状态。
- `docs/route_a_30_20261011/report.md`：自动更新完整配对任务指标，30任务完成才是全基准宏平均。
- `docs/route_a_30_20261011/summary.csv`、`aggregate.json`：逐任务、宏平均和胜平负（平局阈值1e-6）。

预检30任务开发划分通过，总训练分子35020；两组各100次batch128更新、原图eval预测一致、完整快照重载、恢复后下一步更新与采样顺序核验通过。实测约.09-.10秒/批，所有模型100轮训练部分外推约1.5小时；进程启动/存盘/评价后建议预留2-4小时，不保证截止。队列存活期间请求防止系统自动睡眠；手动睡眠、合盖、断电或重启仍可中断。

```powershell
Get-Content D:/GraphCliff-ACA/artifacts/route_a_30_20261011/status.json
Get-Content D:/GraphCliff-ACA/docs/route_a_30_20261011/report.md
```

队列已启动，勿再次运行queue；它会检查已有status拒绝重复启动。report只复核已保存test CSV并重建汇总，不再推理或训练。没有设置聊天提醒或系统计划任务；完成/失败以本地状态与报告为准。源码/环境与输入hash在manifest记录，旧模型、ACA、其他先导、tools和其他项目未改。未发布。单种子完整基准仍不能声称统计显著。
