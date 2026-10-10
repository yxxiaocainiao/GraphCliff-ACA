# 路线 A 残差/LR 核验（2026-10-10）

- [x] 实现最小残差入口、分组LR与范数诊断；原GraphCliff/ACA/路线A/tools不改。
- [x] 验证alpha=0退化、非零梯度、主干/头/指纹同初始化、scheduler全程保持LR比例、checkpoint重载。
- [ ] 固定12次新训练：4任务各residual/residual_slow；新增2147/244各graph/concat。旧234/2047的4个对照复用并重新审计，不计作新拟合。
- [ ] 追加唯一早停诊断：234 concat100重跑满100epoch，与旧前34轮一致性比较；总13次新训练，不追加调参/恢复。
- [ ] 独立审计17个结果（16主比较+1诊断），复算CSV、来源hash，报告负结果和边界。

任务train/valid/cliff：234_Ki=2632/292/128，2047_EC50=452/51/20，2147_Ki=1045/117/46，244_Ki=2229/247/118。seed/split42，官方train内10%验证；不读test标签、不使用旧预训练checkpoint。复用前轮Morgan radius2/2048 bit无手性，单层Linear2048→512+ReLU投影。h=h_graph+alpha*h_fp；alpha是无约束可学习标量，初始化.01，不新增512→512层。保持512输入与原回归头初始化。alpha不会被保证单调增加，残差不保证小，记录每epoch首训练batch残差/graph范数比、训练后alpha。

residual标准所有LR=1e-4；residual_slow仅fp参数LR=2e-5，graph/head/alpha=1e-4；AdamW wd1e-5/betas(.9,.95)，clip1，MSE、均匀shuffle、batch32、H256/L3/G4/K3；沿用原scheduler时序（首次epoch为base LR，epoch末step），warmup10/100epoch，slow组LR始终为主组.2，末端min_lr也乘.2。patience15、严格最小Overall验证MSE选模；concat100唯一禁用早停，仍按同指标选模。构造dropout0，保留上游LongPoly内部默认.1；不修unused ShortGINE.norm。

初始主干/头seed42，fp独立seed1042，训练重置seed42。拟合预算预先固定；CUDA/非有限/划分错误即停，不自动重试、降级或更换环境。输出artifacts/route_a_residual_20261010与docs/route_a_residual_20261010；不发布。

待检验假设：残差是否优于concat/graph；降低fp LR是否进一步改善残差；100轮concat能否在早停之后改善。跨任务差异不证明样本量因果；17次比较不等于17个独立任务。旧best19/26并非停止epoch，实际停止34/41。bit指纹表示存在/缺失，不是计数或保证正则化。gate新增157万包括fp约105万和gate约52万，不能全算gate模块；gate在234优于concat，动态扰动原因未经测量。小样本验证cliff仅20个（2047），不能声称显著性。
