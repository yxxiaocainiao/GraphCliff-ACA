"""Build the pilot report only after all six checkpoint replays pass."""
import json
import math
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/aca_pilot_20261010'
DEST = ROOT/'docs/aca_pilot_20261010'
results = json.loads((SOURCE/'audit.json').read_text())
assert len(results) == 6 and all(r['replay_max_abs'] < 1e-5 for r in results)
DEST.mkdir(parents=True, exist_ok=True)
environment = dict(python=sys.version, platform=platform.platform(), packages={})
for package in ('torch','torch-geometric','numpy','pandas','rdkit','scikit-learn','matplotlib','wandb'):
    try:
        environment['packages'][package] = version(package)
    except PackageNotFoundError:
        environment['packages'][package] = 'distribution metadata unavailable'
(DEST/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
shutil.copyfile(SOURCE/'audit.json', DEST/'audit.json')
shutil.copyfile(SOURCE/'preflight.json', DEST/'preflight.json')
frame = pd.DataFrame(results)
frame.to_csv(DEST/'summary.csv', index=False)
tasks = list(frame.task.unique())
matched = frame.pivot(index='task', columns='arm', values=['rmse','cliff_rmse'])
cliff_wins = int((matched[('cliff_rmse','aca')] < matched[('cliff_rmse','mae')]).sum())
overall_wins = int((matched[('rmse','aca')] < matched[('rmse','mae')]).sum())
lines = ['# GraphCliff + ACA 固定先导结果（2026-10-10）', '',
         '六次训练和独立 checkpoint 重载均完成。以下都是官方 train 内验证结果，不是官方 test 分数。', '',
         f'ACA 相对匹配 MAE 的 Cliff 改善 {cliff_wins}/{len(tasks)} 个任务，Overall 改善 {overall_wins}/{len(tasks)} 个任务。本轮固定直接迁移配方不支持扩大训练；保留为对照基线，不作为论文主方法。不能外推所有对比学习或 ACANet 原架构无效。', '',
         '| 任务 | 目标 | Overall RMSE | Cliff RMSE | 最佳 epoch / 完成 epoch |',
         '|---|---|---:|---:|---:|']
for row in results:
    lines.append(f'| {row["task"]} | {row["arm"]} | {row["rmse"]:.6f} | {row["cliff_rmse"]:.6f} | {row["best_epoch"]} / {row["epochs"]} |')
lines += ['', '相对变化为 (ACA / 对照 - 1) × 100%，负值表示改善。', '',
          '| 任务 | 对照 | Overall 变化 | Cliff 变化 |', '|---|---|---:|---:|']
for task in tasks:
    selected = frame[frame.task.eq(task)].set_index('arm')
    for arm in ('mse', 'mae'):
        delta = selected.loc['aca', ['rmse','cliff_rmse']] / selected.loc[arm, ['rmse','cliff_rmse']] - 1
        lines.append(f'| {task} | {arm} | {100*delta.rmse:+.2f}% | {100*delta.cliff_rmse:+.2f}% |')
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
lossfig, lossaxes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
diagnostics = []
for i, task in enumerate(tasks):
    for arm in ('mse','mae','aca'):
        path = SOURCE/task/arm
        history = json.loads((path/'history.json').read_text())
        predictions = pd.read_csv(path/'valid_predictions.csv')
        summary = frame[frame.task.eq(task) & frame.arm.eq(arm)].iloc[0]
        independently_measured = math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(predictions.label,predictions.prediction))/len(predictions))
        cliff = predictions[predictions.cliff.eq(1)]
        independently_cliff = math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(cliff.label,cliff.prediction))/len(cliff))
        assert abs(independently_measured-summary.rmse) < 1e-7
        assert abs(independently_cliff-summary.cliff_rmse) < 1e-7
        assert len(cliff) == summary.valid_cliff
        assert predictions.row_id.tolist() == json.loads((SOURCE/task/'split.json').read_text())['valid']
        shutil.copyfile(path/'history.json', DEST/f'{task}_{arm}_history.json')
        shutil.copyfile(path/'valid_predictions.csv', DEST/f'{task}_{arm}_valid_predictions.csv')
        epochs = [row['epoch'] for row in history]
        selected_epoch = next(row for row in history if row['epoch'] == summary.best_epoch)
        for j, key in enumerate(('rmse','cliff_rmse')):
            line, = axes[i,j].plot(epochs, [row[key] for row in history], label=arm)
            axes[i,j].scatter([summary.best_epoch], [selected_epoch[key]], color=line.get_color(), edgecolor='black', zorder=3)
            axes[i,j].set(title=f'{task}: {key}', xlabel='Epoch', ylabel='Validation RMSE')
            axes[i,j].legend()
        if arm == 'aca':
            assert all(abs(row['loss']-row['reg']-.1*row['triplet']) < 1e-5 for row in history)
            assert all(0 <= row['active'] <= row['candidates'] and row['candidates'] > 0 for row in history)
            lossaxes[i,0].plot(epochs,[row['reg'] for row in history], label='MAE')
            lossaxes[i,0].plot(epochs,[.1*row['triplet'] for row in history], label='0.1 * triplet')
            lossaxes[i,0].set(title=f'{task}: ACA terms', xlabel='Epoch', ylabel='Sample-weighted batch mean')
            lossaxes[i,0].legend()
            lossaxes[i,1].plot(epochs,[row['active']/row['candidates'] for row in history])
            lossaxes[i,1].set(title=f'{task}: active / candidate', xlabel='Epoch', ylabel='Fraction', ylim=(0,1))
            diagnostics.append(dict(task=task, first=history[0], last=history[-1],
                                    total_candidate_occurrences=sum(row['candidates'] for row in history),
                                    total_active_occurrences=sum(row['active'] for row in history)))
    for name in ('split.json','provenance.json','recovery.json'):
        shutil.copyfile(SOURCE/task/name, DEST/f'{task}_{name}')
fig.suptitle('Circles mark checkpoints selected by minimum Overall MSE')
fig.savefig(DEST/'validation_curves.png', dpi=180)
lossfig.savefig(DEST/'aca_diagnostics.png', dpi=180)
plt.close('all')
(DEST/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
lines += ['', '## 训练诊断', '', '| 任务 | MAE（首轮→末轮） | 加权三元组项（首轮→末轮） | 首轮→末轮激活率 |', '|---|---:|---:|---:|']
for item in diagnostics:
    first, last = item['first'], item['last']
    lines.append(f'| {item["task"]} | {first["reg"]:.4f} → {last["reg"]:.4f} | {.1*first["triplet"]:.4f} → {.1*last["triplet"]:.4f} | {first["active"]/first["candidates"]:.1%} → {last["active"]/last["candidates"]:.1%} |')
lines += ['', 'Loss 按 batch 样本数加权；候选/激活计数是训练中重复出现的三元组次数，不是唯一化学关系数。所有 ACA epoch 都有候选三元组，loss 重构成立，训练无非有限值触发；这证明对比项实际参与训练，不能推出泛化收益。', '',
          '## 协议与边界', '',
          '- 相同初始化 SHA、seed=42、同任务划分、相同原始 GraphCliff 主干；MSE、MAE、MAE+ACA 各两任务，共六次拟合。',
          '- 训练/验证：CHEMBL234 2632/292（验证 cliff=128），CHEMBL2047 452/51（cliff=20）。训练标签 y=-log10(nM)，canonical SMILES 无跨训练/验证重复。',
          '- 原主干 H256/3层/4组/K3/dropout0；batch32、AdamW lr1e-4、wd1e-5、betas(.9,.95)、clip1、warmup10、最多100epochs、patience15。验证 Overall MSE 严格最小选模，不按 Cliff RMSE 选 epoch。内部 LongPoly dropout=0.1 保留。',
          '- 固定 ACA alpha=.1、lower=.2、upper=1、p=1、squared=False、similarity_gate=False，来自官方 README 示例；未做 alphaCV 或超参搜索，不是论文调优复现。实现源、数据与配置 SHA 见 provenance。',
          '- 前两个 MSE 训练完成后重载入口漏取 model_state_dict 外层，修复后从已保存的最佳权重恢复；未增加拟合次数或改变训练/loss。原 provenance 保留，recovery 另存。两次 MSE 训练耗时无法准确恢复，记 null。',
          '- 六个最佳权重均另起进程 strict 加载并复核验证预测、标签、行 ID 和指标，最大预测误差见 audit.json；30 个提供的 checkpoint SHA 全部未变。',
          '- 另用标准库 math.fsum 从 CSV 逐项重算 Overall/Cliff RMSE，与 audit 一致。sklearn 未安装，未为这项检查增加依赖。',
          '- 单种子、两个任务、验证集同时用于选模与效果筛查；小任务仅20个验证 cliff，不能给出统计显著性或测试集结论。原始 cliff_mol 标记沿用数据提供值。',
          '- 关闭结构门控：ACA 约束全局活性表示，合格三元组不保证结构相似，不能称为所有“悬崖三元组”。ACA 移植本身不声明原创。', '',
          '参数示例：[ACANet README](https://github.com/shenwanxiang/ACANet/blob/6274fa4307ce365a814ac5e93460f0afabaea200/README.md)；固定实现：[loss.py](https://github.com/shenwanxiang/ACANet/blob/4c43e6024345d0ec4d40fdc88ad57224c898b6da/clsar/model/loss.py)。', '',
          '![验证曲线](validation_curves.png)', '', '![ACA 诊断](aca_diagnostics.png)']
(DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(DEST/'report.md')
