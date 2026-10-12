# GraphCliff × ECFP × cosine-margin 四组开发比较（2026-10-12）

用户确认按官方划分继续。复用cliff_contrastive_20261012的24个已审计纯图/MSE和纯图/margin模型，只补4任务×seed42/43/44×concat/concat_margin共24次。旧源码、权重、结果只读。另行存档，不把旧路线A的不同协议成绩混入本次比较。

## 冻结条件

- 任务234_Ki、2047_EC50、2147_Ki、244_Ki；与cliff_contrastive完全相同的官方train内10%验证、row_id、split_seed42、初始化种子、标签尺度、正负对规则、逐轮采样和优化/选模规则。官方test标签不读；完整官方train重训不在本先导内。
- 四组：graph_mse、graph_margin（已有），concat_mse、concat_margin（新）。拼接复用experiments/route_a/pilot.py的Predictor('concat')：Morgan radius2/2048，无手性；2048→512/ReLU与原512维图表示拼接，1024→128回归头第一层，后续原头不变。图主干初始化与对应纯图组严格匹配；两个新组全模型初始化匹配。指纹和新第一层使用既有fork_rng seed1042，后三种子变化是图主干及原头其余层，不声称独立改变每一参数初始化。
- 新损失仍作用于SAGPool后、拼接前的512维图表示，不约束1024维融合表示；其公式/负例margin .5/权重 .01/两类等权与旧损失组相同。零权重恢复concat MSE；图辅助梯度不直接更新指纹支路。
- 复用冻结cliff_contrastive.run.job的训练循环、梯度、验证、记录代码；只用独立模块实例和适配入口替换数据/模型/前向接口，旧文件不改。每批形状和标签对应断言保留。训练首批梯度探针仅比较同一图encoder/pool参数，不把fp/head计入。
- 124普通分子+1正对/1负对成员并集，每轮普通样本一次；两新组及两个旧组在共享轮次的sampler hash完全一致。100轮上限，AdamW1e-4/wd1e-5/betas(.9,.95)/clip1；原warmup10/cosine100调度，patience15/min_delta1e-4/验证Overall MSE选模。
- 不调参、不自动重试、不启动30任务。源/数据/旧checkpoint/hash、环境和分区冻结；旧参考必须24/24且独立重载审计通过。当前torch/CUDA须与旧manifest相同。

## 预检与运行

1. 审核旧manifest及24个模型/summary/CSV/历史哈希，复制来源清单而不修改旧文件。
2. 验证四任务指纹、行成员、图初始化、拼接形状、辅助约束确实来自图表示、零权重梯度、辅助梯度不进fp/head，以及原p.Predictor('concat')等价性；真实GPU训练更新和严格checkpoint重载。
3. 固定顺序按task×seed区组随机交换两新组，执行24次。失败停止并记录。保存进度、预测、训练损失、梯度、几何诊断。
4. 独立重载所有新模型，复算预测和指标，核验旧hash未变/共同初始化和采样，合并48个模型的四组表。

## 事前判断

仅当组合组平均Cliff相对concat_mse及graph_margin各改善至少1%，Overall相对graph_mse及concat_mse各恶化不超过1%，且至少3/4任务在至少2/3种子上Cliff同时胜过concat_mse及graph_margin，才记组合go=true。该规则是操作性筛选，不是显著性检验。需报告每组独立增益和交互差分；不以组合仅胜过纯图便声称互补。即使通过也不自动进行全train/test、ECFP-only或更多任务。简单组合不声明原创。

输出：artifacts/cliff_ecfp_20261012及docs/cliff_ecfp_20261012。冻结后训练入口及本文件不再编辑；完成状态记根task_plan.md与notes.md。
