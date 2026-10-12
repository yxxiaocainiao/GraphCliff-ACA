# ECFP拼接 × cosine-margin 四组比较

阶段：24次补跑及独立重载审计完成；加上24个只读复用纯图模型，48模型四组配对齐全。组合未达事前门槛，go=false，详见conclusion.md。

输入：D:/GraphCliff-ACA/benchmark_data四既有任务CSV；只读取官方train标签，train内10%验证与cliff_contrastive_20261012完全一致。官方test行成员保留，test标签不训练、不选模。本次不是完整train拟合或论文test成绩复现。

新增入口D:/GraphCliff-ACA/experiments/cliff_ecfp/run.py复用原concat结构和冻结损失训练循环。源码不改，模块适配只在新独立进程的命名空间中生效。输出D:/GraphCliff-ACA/artifacts/cliff_ecfp_20261012（日志/权重/状态）与本目录（预检/四组报告/预测指标审计）。详细协议见experiments/cliff_ecfp/task_plan.md。

四组：graph_mse、graph_margin（已有），concat_mse、concat_margin（新）。所有组图主干初始化和逐轮采样匹配；损失作用于拼接前512维图表示，指纹只作回归辅助输入。辅助梯度不直接进入fp/head，但MSE会同时更新两支路。concat新增约1.11M参数，不能将拼接相对纯图的收益全部归因于指纹信息；concat两组容量相同。

预检：四任务三种子初始图hash/行ID核验；零权重梯度；拼接公式与辅助表示；真实3次CUDA更新；严格checkpoint重载预测等价。两新组参数各7,135,822。源、数据、24个旧权重及其记录hash已冻结；旧结果不覆盖。

状态artifacts/cliff_ecfp_20261012/status.json；report.md与four_group_paired.csv逐组完成后更新，48模型完整配对后给出组合事前门槛判定。independent_reload_audited=true只在全部新模型独立重载后成立。

未执行：全train/test、30任务、ECFP-only、容量匹配外部对照和新超参搜索。三种子共享同一验证划分，不作为三个独立数据集；简单组合不声明原创。原模型、数据、作者checkpoint及旧实验源码保留，未推送。
