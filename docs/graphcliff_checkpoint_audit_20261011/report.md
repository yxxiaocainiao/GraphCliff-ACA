# GraphCliff论文表格与本轮训练结果的差异

2026-10-11：已有30个原始格式checkpoint通过严格载入，使用本轮相同官方test评价函数完成推理。checkpoint SHA256与10月10日审计一致，数据SHA256与冻结队列manifest一致；每任务CSV和标准库平方误差复算通过。不训练、不修改原始权重。

| 来源 | 30任务宏平均RMSE | Cliff RMSE |
|---|---:|---:|
| 用户截图论文表格 | 0.665 | 0.757 |
| 已有checkpoint本次回放 | 0.664575 | 0.756855 |
| 本轮随机初始化GraphCliff | 0.698532 | 0.798011 |
| 本轮随机初始化concat | 0.698618 | 0.789335 |

逐任务抽查：CHEMBL234_Ki 0.617522/0.631608，截图0.618/0.632；CHEMBL2047_EC50 0.592385/0.654854，截图0.592/0.655；CHEMBL2147_Ki 0.559971/0.579010，截图0.560/0.579。

结论：已有checkpoint的推理成绩可重现截图，排除了本轮评价方式普遍无法重现表格的问题。本轮60次从随机初始化训练，未使用原始checkpoint，尚未达到论文训练成绩。concat仅比本轮匹配graph对照的Cliff宏平均改善约1.09%，未超过论文GraphCliff；Overall没有改善。

现有checkpoint只验证了本地文件一致性，未验证Zenodo原始压缩包hash及其历史训练划分/超参数/选模记录。当前训练协议采用当前main.py默认配置、官方train内部10%验证、seed42、batch128、验证早停；框架为torch2.7.1/CUDA12.8。作者历史训练环境、随机状态、完整参数及checkpoint选择记录未全部确认，不能将差距归因于某一项，也不依据test差距追加调参。

## 历史训练代码差异（2026-10-11追加核验）

已核对本仓库保存的作者Git历史，并在线读取作者固定旧版本main.py：
- cf6c281127df1325e6c9025e453e058a27927e24（2026-01-16）：dataset使用完整官方train；main.py每轮evaluate(test_loader)，以test_loss保存最佳权重并早停。来源：https://github.com/dmis-lab/GraphCliff/blob/cf6c281127df1325e6c9025e453e058a27927e24/main.py#L177-L197 。
- 408f78311c46cecd25f78a5dd49dae8844428ea0（2026-08-04，KDD2026 update）：引入train内部10%验证划分，改为evaluate(valid_loader)、valid_loss选模及早停；本轮沿用这一规则。
- 此次两版本diff中model.py/train.py/metric.py/run.sh没有差异；main.py/dataset.py中的训练样本和选模规则存在上述实质变化。

这证明旧公开代码和当前协议不同，足以否定“本轮已复现论文历史训练协议”的说法，但不能证明已发布checkpoint一定由该旧版本/默认参数产生，也不能定量解释全部分数差距。旧代码利用test选模有评价泄漏风险；不能为了追上表格直接将当前公平比较改成test早停，再称为独立测试成绩。

本轮30任务匹配比较保留为探索性结果，不作为超过论文GraphCliff的证据。继续路线A之前，先明确历史复现与独立验证比较的协议，复核当前训练实现与官方入口的一致性；现阶段未启动新训练。

新增audit_checkpoints.py只回放已有权重，results.json保存权重路径/hash、逐任务成绩和宏平均；30份predictions.csv保留每个test行的预测。训练代码、数据、权重和其他工程均未修改。下一步是查清作者历史训练协议并复现可靠图对照，再按同一协议测试concat。
