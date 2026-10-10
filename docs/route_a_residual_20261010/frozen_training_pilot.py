"""GPL-3.0: bounded residual/LR pilot reusing the audited route-A predictor utilities."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/route_a'))
import pilot as ra
import numpy as np
import pandas as pd
import torch
from torch import nn

OUT = ROOT / 'artifacts/route_a_residual_20261010'
DOC = ROOT / 'docs/route_a_residual_20261010'
TASKS = (*ra.base.TASKS, 'CHEMBL2147_Ki', 'CHEMBL244_Ki')
ARMS = ('graph', 'concat', 'residual', 'residual_slow')


class Residual(ra.Predictor):
    def __init__(self):
        super().__init__('graph')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1042)
            self.fp = nn.Sequential(nn.Linear(2048, 512), nn.ReLU()).cuda()
        self.alpha = nn.Parameter(torch.tensor(.01, device='cuda'))
        self.ratio = None

    def forward(self, batch):
        graph = self.graph(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
        delta = self.alpha * self.fp(batch['fp'])
        self.ratio = (delta.detach().norm(dim=1) / graph.detach().norm(dim=1).clamp_min(1e-12)).mean()
        return self.head(graph + delta)


def model_new(arm):
    return Residual() if arm.startswith('residual') else ra.Predictor('concat' if arm == 'concat100' else arm)


def optimizer_new(model, slow):
    if slow:
        fp_ids = {id(p) for p in model.fp.parameters()}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in fp_ids], lr=1e-4),
                  dict(params=list(model.fp.parameters()), lr=2e-5)]
    else:
        groups = [dict(params=list(model.parameters()), lr=1e-4)]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5, betas=(.9, .95))
    scheduler = ra.base.WarmupCosineScheduler(optimizer, 10, 100, 1e-4)
    return optimizer, scheduler


def schedule(optimizer, scheduler, epoch, slow):
    scheduler.step(epoch)
    if slow:
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * .2


def path_for(task, arm):
    if task in ra.base.TASKS and arm in ('graph', 'concat'):
        return ra.OUT / task / arm
    return OUT / task / arm


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    graph = ra.Predictor('graph')
    model = Residual()
    assert ra.base.state_hash(graph.graph) == ra.base.state_hash(model.graph)
    assert ra.base.state_hash(graph.head) == ra.base.state_hash(model.head)
    concat = ra.Predictor('concat')
    assert ra.base.state_hash(concat.fp) == ra.base.state_hash(model.fp)
    opt, sched = optimizer_new(model, True)
    assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in model.parameters()}
    assert sum(len(g['params']) for g in opt.param_groups) == len(list(model.parameters()))
    assert abs(opt.param_groups[1]['lr']/opt.param_groups[0]['lr']-.2) < 1e-12
    for epoch in range(100):
        schedule(opt, sched, epoch, True)
        assert abs(opt.param_groups[1]['lr']/opt.param_groups[0]['lr']-.2) < 1e-12
    for task in TASKS:
        tr, va, tg, _ = ra.data(task)
        if task in ra.base.TASKS:
            split = json.loads((ra.OUT/task/'split.json').read_text(encoding='utf-8'))
            assert split['train'] == tr.row_id.tolist() and split['valid'] == va.row_id.tolist()
            provenance = json.loads((ra.OUT/task/'provenance.json').read_text(encoding='utf-8'))
            for name, digest in provenance['source_sha256'].items():
                if name != 'experiments/route_a/task_plan.md':
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
            assert hashlib.sha256((ROOT/'benchmark_data'/f'{task}.csv').read_bytes()).hexdigest() == provenance['data_sha256']
        batch = ra.collate(tg[:32])
        graph.eval()
        model.eval()
        with torch.no_grad():
            model.alpha.zero_()
            assert torch.equal(model(batch), graph(batch)), 'alpha=0 must exactly recover graph'
            model.alpha.fill_(.01)
        model.train()
        nn.functional.mse_loss(model(batch), batch['y']).backward()
        assert model.alpha.grad is not None and abs(float(model.alpha.grad)) > 0
        assert model.fp[0].weight.grad.norm() > 0
        unused = {n for n,p in model.named_parameters() if p.grad is None}
        assert unused == {f'graph.encoder.layers.{i}.short.norm.weight' for i in range(3)}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clone = Residual()
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        clone.eval()
        assert torch.allclose(model(batch), clone(batch), atol=1e-5, rtol=1e-5)
        report[task] = dict(train=len(tr), valid=len(va), valid_cliff=int(va.cliff_mol.sum()),
                            graph_initial_hash=ra.base.state_hash(model.graph),
                            head_initial_hash=ra.base.state_hash(model.head),
                            fp_initial_hash=ra.base.state_hash(model.fp), checks='passed')
        model.zero_grad(set_to_none=True)
        del clone
    ra.base.save(OUT/'preflight.json', report)
    print(json.dumps(report), flush=True)


def fit(task, arm, tr, va, tg, vg):
    destination = path_for(task, arm)
    destination.mkdir()
    started = time.time()
    model = model_new(arm)
    digest = ra.base.state_hash(model)
    preflight = json.loads((OUT/'preflight.json').read_text(encoding='utf-8'))[task]
    assert ra.base.state_hash(model.graph) == preflight['graph_initial_hash']
    if arm != 'concat' and arm != 'concat100':
        assert ra.base.state_hash(model.head) == preflight['head_initial_hash']
    if arm != 'graph':
        assert ra.base.state_hash(model.fp) == preflight['fp_initial_hash']
    ra.base.set_seed(ra.base.SimpleNamespace(seed=42))
    train_loader, valid_loader = ra.loaders(tg, vg)
    slow = arm == 'residual_slow'
    optimizer, scheduler = optimizer_new(model, slow)
    best, stale, history = float('inf'), 0, []
    for epoch in range(100):
        model.train()
        total, count, ratio = 0., 0, None
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch), batch['y'])
            assert torch.isfinite(loss)
            if count == 0 and isinstance(model, Residual):
                ratio = float(model.ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch['y'])
            count += len(batch['y'])
        assert count == len(tr)
        labels, predictions = ra.evaluate(model, valid_loader)
        score = ra.base.metrics(labels, predictions, va.cliff_mol)
        value = float(np.mean((np.asarray(labels)-predictions)**2))
        assert np.isfinite(value)
        history.append(dict(epoch=epoch+1, train_mse=total/count,
                            lr=optimizer.param_groups[0]['lr'],
                            fp_lr=optimizer.param_groups[-1]['lr'] if arm != 'graph' else None,
                            alpha=float(model.alpha.detach()) if isinstance(model, Residual) else None,
                            first_batch_residual_norm_ratio=ratio, **score))
        if value < best:
            best, stale, best_epoch = value, 0, epoch+1
            torch.save(model.state_dict(), destination/'best_model.pt')
        else:
            stale += 1
        ra.base.save(destination/'history.json', history)
        if epoch == 0 or (epoch+1)%5 == 0:
            print(f'{task} {arm} epoch={epoch+1} rmse={score["rmse"]:.6f} cliff={score["cliff_rmse"]:.6f} stale={stale}', flush=True)
        schedule(optimizer, scheduler, epoch, slow)
        if arm != 'concat100' and stale >= 15:
            break
    model.load_state_dict(torch.load(destination/'best_model.pt', weights_only=True), strict=True)
    labels, predictions = ra.evaluate(model, valid_loader)
    pd.DataFrame(dict(row_id=va.row_id, label=labels, prediction=predictions, cliff=va.cliff_mol)).to_csv(destination/'valid_predictions.csv', index=False)
    result = dict(task=task, arm=arm, initial_sha256=digest, best_epoch=best_epoch, epochs=len(history),
                  training_seconds=time.time()-started, parameters=sum(p.numel() for p in model.parameters()),
                  alpha_best=float(model.alpha.detach()) if isinstance(model, Residual) else None,
                  **ra.base.metrics(labels, predictions, va.cliff_mol))
    ra.base.save(destination/'summary.json', result)
    print('FINISHED '+json.dumps(result), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def run():
    assert (OUT/'preflight.json').exists()
    for task in TASKS:
        tr, va, tg, vg = ra.data(task)
        folder = OUT/task
        folder.mkdir()
        ra.base.save(folder/'split.json', dict(train=tr.row_id.tolist(), valid=va.row_id.tolist(), test_labels_read=False))
        names = ['model.py','dataset.py','dataset_utils.py','main.py','experiments/aca/run.py',
                 'experiments/route_a/pilot.py','experiments/route_a_residual/pilot.py','experiments/route_a_residual/task_plan.md']
        ra.base.save(folder/'provenance.json', dict(
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
            data_sha256=hashlib.sha256((ROOT/'benchmark_data'/f'{task}.csv').read_bytes()).hexdigest(),
            protocol=(ROOT/'experiments/route_a_residual/task_plan.md').read_text(encoding='utf-8'),
            torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            pretrained_checkpoint_used=False))
        for arm in ARMS:
            if path_for(task,arm).parent != folder:
                print(f'REUSE {task} {arm}: audited prior fit, not a new training', flush=True)
                continue
            fit(task,arm,tr,va,tg,vg)
        if task == 'CHEMBL234_Ki':
            fit(task,'concat100',tr,va,tg,vg)


def audit():
    DOC.mkdir(parents=True, exist_ok=True)
    results = []
    for task in TASKS:
        _,va,_,vg = ra.data(task)
        loader = torch.utils.data.DataLoader(vg,batch_size=32,collate_fn=ra.collate)
        for arm in (*ARMS, 'concat100') if task == 'CHEMBL234_Ki' else ARMS:
            path = path_for(task,arm)
            result = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            model = model_new(arm)
            assert ra.base.state_hash(model) == result['initial_sha256']
            model.load_state_dict(torch.load(path/'best_model.pt',weights_only=True),strict=True)
            labels,predictions = ra.evaluate(model,loader)
            stored = pd.read_csv(path/'valid_predictions.csv')
            assert stored.row_id.tolist() == va.row_id.tolist() and stored.cliff.tolist() == va.cliff_mol.tolist()
            assert np.max(np.abs(stored.label.to_numpy()-labels)) < 1e-6
            difference = float(np.max(np.abs(stored.prediction.to_numpy()-predictions)))
            assert difference < 1e-5
            for key,value in ra.base.metrics(labels,predictions,va.cliff_mol).items():
                errors = [(float(y)-float(p))**2 for y,p,c in zip(stored.label,stored.prediction,stored.cliff) if key=='rmse' or c]
                assert abs(math.sqrt(math.fsum(errors)/len(errors))-value) < 1e-6
                assert abs(value-result[key]) < 1e-5
            history = json.loads((path/'history.json').read_text(encoding='utf-8'))
            assert len(history) == result['epochs']
            assert min(history,key=lambda r:r['rmse'])['epoch'] == result['best_epoch']
            if arm == 'residual_slow':
                assert all(abs(r['fp_lr']/r['lr']-.2)<1e-12 for r in history)
            result.update(reused=task in ra.base.TASKS and arm in ('graph','concat'),replay_max_abs=difference)
            results.append(result)
            for name in ('valid_predictions.csv','history.json'):
                (DOC/f'{task}_{arm}_{name}').write_bytes((path/name).read_bytes())
            del model
        provenance = json.loads((OUT/task/'provenance.json').read_text(encoding='utf-8'))
        for name,digest in provenance['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((ROOT/'benchmark_data'/f'{task}.csv').read_bytes()).hexdigest()==provenance['data_sha256']
        for name in ('split.json','provenance.json'):
            (DOC/f'{task}_{name}').write_bytes((OUT/task/name).read_bytes())
    assert len(results)==17 and sum(not r['reused'] for r in results)==13
    old = json.loads((ra.OUT/'CHEMBL234_Ki/concat/history.json').read_text(encoding='utf-8'))
    full = json.loads((OUT/'CHEMBL234_Ki/concat100/history.json').read_text(encoding='utf-8'))
    assert len(full)==100
    prefix_error = max(abs(a[k]-b[k]) for a,b in zip(old,full) for k in ('train_mse','rmse','cliff_rmse','lr'))
    assert prefix_error < 1e-5, '100epoch diagnostic prefix does not reproduce earlier training'
    diagnostic = dict(prefix_epochs=len(old),prefix_max_abs=prefix_error,
                      best_before_stop=min(old,key=lambda r:r['rmse']),
                      best_after_stop=min(full[len(old):],key=lambda r:r['rmse']),
                      best_all=min(full,key=lambda r:r['rmse']))
    ra.base.save(DOC/'early_stop_diagnostic.json',diagnostic)
    ra.base.save(DOC/'audit.json',results)
    lines=['# 路线 A 残差与学习率固定核验','',
           '四任务、seed42、官方train内验证，13次新拟合+4个复用对照；复用对照重新重载审计。无test评价。','',
           '| Task | Arm | Overall | Cliff | best/epochs | alpha best |','|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["task"]} | {r["arm"]} | {r["rmse"]:.6f} | {r["cliff_rmse"]:.6f} | {r["best_epoch"]}/{r["epochs"]} | {r.get("alpha_best")} |')
    lines.extend(['','## 相对graph（负值改善）','','| Task | Arm | Overall | Cliff |','|---|---|---:|---:|'])
    for task in TASKS:
        arms={r['arm']:r for r in results if r['task']==task}
        for arm in ('concat','residual','residual_slow'):
            changes=[(arms[arm][k]/arms['graph'][k]-1)*100 for k in ('rmse','cliff_rmse')]
            lines.append(f'| {task} | {arm} | {changes[0]:+.2f}% | {changes[1]:+.2f}% |')
    lines.extend(['','## 早停诊断','',f'前{len(old)}轮与旧concat最大绝对差={prefix_error:.3g}。',
                  f'原best epoch={diagnostic["best_before_stop"]["epoch"]}；完整100轮best={diagnostic["best_all"]["epoch"]}。',
                  '诊断用于检查原早停后是否有改善；延长训练即使有效也不证明指纹收敛更快是原因。','',
                  'alpha无约束可学习，范数比为每epoch首训练batch的平均样本L2比，非全训练上界。低LR组全程fp/main=.2；alpha沿用main LR。',
                  '所有checkpoint独立strict重载，预测差<1e-5；CSV用math.fsum复算，初始hash、源码/数据hash与分区核验通过。',
                  '只有单seed、验证集选模，20个2047验证cliff分子；跨任务差异受靶点/噪声/化学空间等影响，不能归因样本量或称统计显著。',
                  '残差保留回归头输入/初始权重，但仍改变优化轨迹；未叠加其他模块、未调参、未发布。'])
    (DOC/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','run','audit'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    {'check':check,'run':run,'audit':audit}[args.action]()
