# 路线 A 残差/LR 核验（2026-10-10）

- [x] 实现最小残差入口、分组LR与范数诊断；原GraphCliff/ACA/路线A/tools不改。
- [x] 验证alpha=0退化、非零梯度、主干/头/指纹同初始化、scheduler全程保持LR比例、checkpoint重载。
- [ ] 固定12次新训练：4任务各residual/residual_slow；新增2147/244各graph/concat。旧234/2047的4个对照复用并重新审计，不计作新拟合。
- [x] 追加唯一早停诊断：234 concat100重跑满100epoch，与旧前34轮一致性比较；总13次新训练，不追加调参/恢复。
- [x] 独立审计13个完成结果（9新拟合+4复用），明确4个未完成项，复算CSV、冻结源码hash，报告负结果和边界。

任务train/valid/cliff：234_Ki=2632/292/128，2047_EC50=452/51/20，2147_Ki=1045/117/46，244_Ki=2229/247/118。seed/split42，官方train内10%验证；不读test标签、不使用旧预训练checkpoint。复用前轮Morgan radius2/2048 bit无手性，单层Linear2048→512+ReLU投影。h=h_graph+alpha*h_fp；alpha是无约束可学习标量，初始化.01，不新增512→512层。保持512输入与原回归头初始化。alpha不会被保证单调增加，残差不保证小，记录每epoch首训练batch残差/graph范数比、训练后alpha。

residual标准所有LR=1e-4；residual_slow仅fp参数LR=2e-5，graph/head/alpha=1e-4；AdamW wd1e-5/betas(.9,.95)，clip1，MSE、均匀shuffle、batch32、H256/L3/G4/K3；沿用原scheduler时序（首次epoch为base LR，epoch末step），warmup10/100epoch，slow组LR始终为主组.2，末端min_lr也乘.2。patience15、严格最小Overall验证MSE选模；concat100唯一禁用早停，仍按同指标选模。构造dropout0，保留上游LongPoly内部默认.1；不修unused ShortGINE.norm。

初始主干/头seed42，fp独立seed1042，训练重置seed42。拟合预算预先固定；CUDA/非有限/划分错误即停，不自动重试、降级或更换环境。输出artifacts/route_a_residual_20261010与docs/route_a_residual_20261010；不发布。

待检验假设：残差是否优于concat/graph；降低fp LR是否进一步改善残差；100轮concat能否在早停之后改善。跨任务差异不证明样本量因果；17次比较不等于17个独立任务。旧best19/26并非停止epoch，实际停止34/41。bit指纹表示存在/缺失，不是计数或保证正则化。gate新增157万包括fp约105万和gate约52万，不能全算gate模块；gate在234优于concat，动态扰动原因未经测量。小样本验证cliff仅20个（2047），不能声称显著性。

## 本轮结束状态
原定12次主训练仅完成8次；另concat100诊断完成，共9次新拟合。244 graph保存60轮后在下一轮反向传播报CUDA unspecified launch failure，按冻结停止条件结束，无恢复尝试，3后续组未启动。原四任务验证未完成，不标成全预算达成。完成模型13个独立重载/CSV复算通过。

诊断修订：concat100第7轮起与旧轨迹分叉（前34轮最大差.0972），不能做精确续跑比较；本次完整轨迹内模拟早停在66轮，best51，之后无改善。修订只影响审计方法，AST核验训练/构造函数未改，原冻结源码和协议保留。

决策：234低LR残差仍落后graph，2047/2147低LR退步；concat在2047/2147收益尚待多种子验证。暂停扩大残差/LR，不自动补244、调参、推送或评价test。
