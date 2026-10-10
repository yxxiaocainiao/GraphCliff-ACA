"""Report four audited fits, candidate coverage and sampled shared gradients."""
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/aca_local_20261010'
DEST = ROOT/'docs/aca_local_20261010'
results = json.loads((SOURCE/'audit.json').read_text())
status = json.loads((SOURCE/'audit_status.json').read_text())
assert len(results)==len(status['completed']) and all(row['replay_max_abs']<1e-5 for row in results)
DEST.mkdir(parents=True,exist_ok=True)
for name in ('audit.json','preflight.json','audit_status.json'):
    shutil.copyfile(SOURCE/name,DEST/name)
aborted = SOURCE/'gated_raw_aborted_cuda'
if aborted.exists():
    shutil.copyfile(aborted/'history.json',DEST/'gated_raw_aborted_history.json')
    shutil.copyfile(aborted/'provenance.json',DEST/'gated_raw_aborted_provenance.json')
    for name in ('history.json','provenance.json'):
        shutil.copyfile(SOURCE/'gated_raw'/name,DEST/f'gated_raw_retry_{name}')
    for name in ('gated_raw.log','gated_raw_retry.log'):
        (DEST/name).write_text((SOURCE/name).read_text(encoding='utf-8').rstrip()+'\n',encoding='utf-8')
frame = pd.DataFrame(results).set_index('arm')
names = dict(mse='MSE（匹配采样）',ungated_projected='投影+归一化，无结构门控',
             gated_raw='严格门控，原 h',gated_projected='严格门控+投影+归一化')
colors = dict(mse='tab:blue',ungated_projected='tab:orange',gated_raw='tab:red',gated_projected='tab:green')
lines = ['# CHEMBL234：结构门控与投影的固定先导', '',
         f'计划四组，完成并核验 {len(results)} 组；未完成：{", ".join(status["incomplete"]) or "无"}。两个CUDA故障导致原h门控组停止，不将部分历史纳入完成模型排名。', '',
         '阶段判断：完整组合相对匹配 MSE 的 Overall/Cliff RMSE 均恶化，暂不扩大该配方。门控改善了无门控投影组的 Overall，但 Cliff 更差，不能把门控称为已验证的修复。原h消融缺失，投影贡献暂不作完整因果归因。', '',
         '仅官方 train 内验证，seed/split42。四组均使用相同训练内三元组采样；不能与旧均匀采样结果作严格机制归因。', '',
         '| 组别 | Overall RMSE | Cliff RMSE | best / epochs |', '|---|---:|---:|---:|']
for row in results:
    lines.append(f'| {names[row["arm"]]} | {row["rmse"]:.6f} | {row["cliff_rmse"]:.6f} | {row["best_epoch"]} / {row["epochs"]} |')
lines += ['', '相对变化=(完整组合/对照−1)×100%，负值改善。', '',
          '| 对照 | Overall 变化 | Cliff 变化 |', '|---|---:|---:|']
for arm in ('mse','ungated_projected','gated_raw'):
    if arm not in frame.index:
        lines.append(f'| {names[arm]} | 未完成，不比较 | 未完成，不比较 |')
        continue
    delta = frame.loc['gated_projected',['rmse','cliff_rmse']]/frame.loc[arm,['rmse','cliff_rmse']]-1
    lines.append(f'| {names[arm]} | {100*delta.rmse:+.2f}% | {100*delta.cliff_rmse:+.2f}% |')
diagnostics = []
fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for result in results:
    arm = result['arm']
    folder = SOURCE/arm
    for name in ('history.json','provenance.json','valid_predictions.csv'):
        shutil.copyfile(folder/name,DEST/f'{arm}_{name}')
    history = json.loads((folder/'history.json').read_text())
    assert all(abs(row['loss']-row['reg']-.01*row['triplet'])<1e-5 for row in history)
    assert all(row['samples']==2905 and row['batches']==91 for row in history)
    if arm.startswith('gated'):
        assert all(row['empty_batches']==0 and row['candidates']>0 for row in history)
    epochs = [row['epoch'] for row in history]
    selected = next(row for row in history if row['epoch']==result['best_epoch'])
    for j,key in enumerate(('rmse','cliff_rmse')):
        line, = axes[0,j].plot(epochs,[row[key] for row in history],label=arm,color=colors[arm])
        axes[0,j].scatter([result['best_epoch']],[selected[key]],color=line.get_color(),edgecolor='black')
        axes[0,j].set(xlabel='Epoch',ylabel='Validation RMSE',title=key)
        axes[0,j].legend()
    if arm!='mse':
        probes = [row['first_batch_gradient'] for row in history]
        ratios = [p['grad_ratio'] for p in probes if p['grad_ratio'] is not None]
        cosines = [p['grad_cosine'] for p in probes if p['grad_cosine'] is not None]
        diagnostics.append(dict(arm=arm,median_ratio=float(np.median(ratios)),max_ratio=max(ratios),
            negative_cosine_fraction=sum(c<0 for c in cosines)/len(cosines) if cosines else None,
            empty_batches=sum(row['empty_batches'] for row in history),
            candidate_occurrences=sum(row['candidates'] for row in history),
            active_occurrences=sum(row['active'] for row in history)))
        axes[1,0].plot(epochs,[row['first_batch_gradient']['grad_ratio'] for row in history],label=arm,color=colors[arm])
        axes[1,1].plot(epochs,[row['first_batch_gradient']['grad_cosine'] for row in history],label=arm,color=colors[arm])
axes[1,0].set(title='Weighted auxiliary / regression gradient norm',xlabel='Epoch',ylabel='Ratio (first batch)')
axes[1,0].axhline(.1,color='black',linestyle='--',label='10% reference, not a cap')
axes[1,1].set(title='Gradient cosine, shared encoder/pool',xlabel='Epoch',ylabel='Cosine (first batch)',ylim=(-1,1))
axes[1,0].legend(); axes[1,1].legend()
fig.suptitle('CHEMBL234_Ki: matched sampling; circles select minimum Overall MSE')
fig.savefig(DEST/'curves_and_gradients.png',dpi=180)
plt.close(fig)
(DEST/'gradient_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
lines += ['', '## 梯度与候选诊断', '',
          '| 组别 | 加权梯度比中位数 | 最大值 | cosine<0 的抽样比例 | 空批次数 |', '|---|---:|---:|---:|---:|']
for row in diagnostics:
    cosine = f'{row["negative_cosine_fraction"]:.1%}' if row['negative_cosine_fraction'] is not None else '未定义'
    lines.append(f'| {names[row["arm"]]} | {row["median_ratio"]:.2%} | {row["max_ratio"]:.2%} | {cosine} | {row["empty_batches"]} |')
lines += ['', '测量对象为共享 AtomEncoder、GraphCliffEncoder、SAGPooling 参数，每epoch仅首batch、裁剪前测量；排除两个头。它不是全程梯度上界、不是全部batch平均，也不能单凭cosine证明验证退步原因。', '',
          '## 实施与预检证据', '',
          '- 原 GraphCliff 结构未修改。投影头512→128→64（ReLU）后L2归一化，三元组保留L1距离与原标签margin；训练 MSE+.01*triplet，推理不计算投影。骨干四组初始化SHA一致。',
          '- 新掩码仅强制负例 Morgan radius2/2048 bit、无手性 Tanimoto>=.8，且Δy>=1；正例Δy<.2，仍可能跨骨架。复用原ACA标签/hinge/候选平均，替换结构掩码，未改 vendor。',
          '- 初始随机batch预检：440个float32合格有向负例对，但首epoch仅5个三元组、82/83批空；不能只开门控而忽略候选密度。最初float64检查442对，与训练标签精度不同。',
          '- 四组同样每批29普通训练样本+1完整训练内三元组；普通样本每epoch一次，三元组成员额外重复，共91批/2905次样本。新预检首epoch588候选、0空批。MSE对照也使用相同过采样，控制采样混杂。',
          '- MSE、投影无门控、原h有门控、完整组合四组；alpha=.01固定，100epochs/patience15，其他模型和优化超参沿用原先导。最低验证Overall MSE选模，不按Cliff选epoch。',
          '- 六点预检：数值公式、门控与空候选、归一化、CPU/CUDA RNG保护、有限梯度、checkpoint可由原模型strict加载且单分子预测相同。三个完成拟合的最佳checkpoint另进程重载；CSV通过math.fsum重算RMSE。', '',
          '- gated_raw 首次完成49个epoch后第50轮loss.backward发生CUDA/cuBLAS内部错误；新进程CUDA烟测通过后按同源码/参数/种子从头重试一次，再因illegal instruction中断。CUDA错误可能异步上报，当前调用栈不能精确定位源算子；不能据此断言梯度爆炸或某个模块必然失败。按预定一次恢复预算停止；两次部分历史和故障日志单独保留，不作为完整结果或择优依据。实际三个完成拟合、两次中断尝试，共五次启动。', '',
          '## 证据边界与下一步', '',
          '此前“损失大所以梯度主导”“投影头让h纯净”“关闭门控必然导致失败”都不能从原报告直接推出。Hinge只作用于未满足的排序约束，投影头仍把梯度传回骨干。完成的三组仅能检查有投影条件下的门控效应和完整组合相对匹配MSE的效果；由于原h组未完成，不能完整分辨投影+归一化相对原h的贡献。正例、学习率、margin与更多种子尚未消融。', '',
          '投影 z=normalize(p(hθ)) 时，辅助梯度为 J(hθ)ᵀ J(normalize∘p)ᵀ ∇z Ltriplet，通常不为零；投影改变辅助梯度的路径与方向，不能保证骨干隔离。Loss 的常数偏移可让数值很大而梯度不变，因此需记录 ||.01∇θLtriplet||/||∇θLMSE||，不能用loss比替代。', '',
          'MAE 对预测的导数为 sign(ŷ−y)/B，MSE 为 2(ŷ−y)/B；MSE更强调大残差，但这不是专门对cliff样本加权。旧MAE对照Cliff RMSE是升高而非降低；该单种子现象不能证明MAE普遍不适合。', '',
          '单任务单种子且验证集用于选模，不能给出官方测试集结论或统计显著性。直接迁移与这些通用修正不声明原创。新采样改变了训练分布，旧均匀采样MSE只能作参考；不能把与旧配方的差值归因于某一个修改。', '',
          '![曲线与梯度](curves_and_gradients.png)']
(DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
frame.to_csv(DEST/'summary.csv')
print(DEST/'report.md')
