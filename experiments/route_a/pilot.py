"""Fixed graph/fingerprint pilot; reuse ACA development split utilities (GPL-3.0)."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/aca'))
import run as base
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.utils.data import DataLoader

OUT = ROOT / 'artifacts/route_a_20261010'
DOC = ROOT / 'docs/route_a_20261010'
ARMS = ('graph', 'fp', 'concat', 'gate')


class Predictor(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        graph = base.model_new()
        self.head = copy.deepcopy(graph.reg_head)
        self.graph = graph if arm != 'fp' else None
        if self.graph is not None:
            self.graph.reg_head = nn.Identity()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1042)
            if arm != 'graph':
                self.fp = nn.Sequential(nn.Linear(2048, 512), nn.ReLU()).cuda()
            if arm == 'concat':
                self.head[0] = nn.Linear(1024, 128).cuda()
            if arm == 'gate':
                self.gate = nn.Linear(1024, 512).cuda()

    def forward(self, batch):
        if self.arm != 'fp':
            graph = self.graph(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
        if self.arm != 'graph':
            fp = self.fp(batch['fp'])
        if self.arm == 'graph':
            h = graph
        elif self.arm == 'fp':
            h = fp
        elif self.arm == 'concat':
            h = torch.cat((graph, fp), dim=1)
        else:
            gate = self.gate(torch.cat((graph, fp), dim=1)).sigmoid()
            h = gate * graph + (1 - gate) * fp
        return self.head(h)


def data(task):
    tr, va, tg, vg = base.development(task)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    for rows, graphs in ((tr, tg), (va, vg)):
        assert len(rows) == len(graphs)
        for smiles, graph in zip(rows.smiles, graphs):
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            array = np.zeros(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
            assert array.sum() > 0 and np.isin(array, (0, 1)).all()
            graph.fp = torch.from_numpy(array.astype(np.float32))
    return tr, va, tg, vg


def collate(graphs):
    batch = base.collate_graphs(graphs)
    batch['fp'] = torch.stack([g.fp for g in graphs])
    return {name: value.cuda() for name, value in batch.items()}


def loaders(tg, vg):
    return (DataLoader(tg, batch_size=32, shuffle=True, collate_fn=collate,
                       generator=torch.Generator().manual_seed(42)),
            DataLoader(vg, batch_size=32, collate_fn=collate))


def evaluate(model, loader):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch['y'].cpu().view(-1).tolist())
            predictions.extend(model(batch).cpu().view(-1).tolist())
    assert np.isfinite(predictions).all()
    return labels, predictions


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in base.TASKS:
        tr, va, tg, vg = data(task)
        batch = collate(tg[:32])
        assert torch.equal(batch['fp'][1], tg[1].fp.cuda())
        graph_hashes, fp_hashes = set(), set()
        for arm in ARMS:
            model = Predictor(arm)
            if model.graph is not None:
                graph_hashes.add(base.state_hash(model.graph))
            if arm != 'graph':
                fp_hashes.add(base.state_hash(model.fp))
            model.eval()
            pred = model(batch)
            assert pred.shape == batch['y'].shape
            if arm == 'graph':
                original = base.model_new()
                original.eval()
                assert torch.equal(pred, original(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']))
                del original
            if arm == 'gate':
                original_bias = model.gate.bias.detach().clone()
                original_weight = model.gate.weight.detach().clone()
                with torch.no_grad():
                    model.gate.weight.zero_()
                    model.gate.bias.fill_(100)
                    h = model.graph(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
                    assert torch.allclose(model(batch), model.head(h), atol=1e-6)
                    model.gate.bias.fill_(-100)
                    assert torch.allclose(model(batch), model.head(model.fp(batch['fp'])), atol=1e-6)
                    model.gate.weight.copy_(original_weight)
                    model.gate.bias.copy_(original_bias)
            model.train()
            nn.functional.mse_loss(model(batch), batch['y']).backward()
            unused = {n for n, p in model.named_parameters() if p.grad is None}
            expected_unused = {f'graph.encoder.layers.{i}.short.norm.weight' for i in range(3)} if arm != 'fp' else set()
            # Upstream ShortGINE declares norm but does not call it in forward.
            assert unused == expected_unused
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            clone = Predictor(arm)
            clone.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            clone.eval()
            assert torch.allclose(model(batch), clone(batch), atol=1e-5, rtol=1e-5)
            del model, clone
        assert len(graph_hashes) == len(fp_hashes) == 1
        report[task] = dict(train=len(tr), valid=len(va), valid_cliff=int(va.cliff_mol.sum()),
                            graph_initial_hash=graph_hashes.pop(), fp_initial_hash=fp_hashes.pop(), checks='passed')
    base.save(OUT / 'preflight.json', report)
    print(json.dumps(report), flush=True)


def run():
    assert (OUT / 'preflight.json').exists()
    for task in base.TASKS:
        tr, va, tg, vg = data(task)
        folder = OUT / task
        folder.mkdir()
        base.save(folder / 'split.json', dict(train=tr.row_id.tolist(), valid=va.row_id.tolist(), test_labels_read=False))
        source = ['model.py', 'dataset.py', 'dataset_utils.py', 'main.py', 'experiments/aca/run.py',
                  'experiments/route_a/pilot.py', 'experiments/route_a/task_plan.md']
        base.save(folder / 'provenance.json', dict(
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source},
            data_sha256=hashlib.sha256((ROOT / 'benchmark_data' / (task + '.csv')).read_bytes()).hexdigest(),
            protocol=(ROOT / 'experiments/route_a/task_plan.md').read_text(encoding='utf-8'),
            torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            pretrained_checkpoint_used=False, label_scale='y=-log10(nM)'))
        for arm in ARMS:
            started = time.time()
            destination = folder / arm
            destination.mkdir()
            model = Predictor(arm)
            initial = base.state_hash(model)
            graph_hash = base.state_hash(model.graph) if model.graph is not None else None
            fp_hash = base.state_hash(model.fp) if arm != 'graph' else None
            preflight = json.loads((OUT / 'preflight.json').read_text())[task]
            assert graph_hash is None or graph_hash == preflight['graph_initial_hash']
            assert fp_hash is None or fp_hash == preflight['fp_initial_hash']
            base.set_seed(base.SimpleNamespace(seed=42))
            train_loader, valid_loader = loaders(tg, vg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
            scheduler = base.WarmupCosineScheduler(optimizer, 10, 100, 1e-4)
            best, stale, history = float('inf'), 0, []
            for epoch in range(100):
                model.train()
                total, count = 0., 0
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(batch), batch['y'])
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch['y'])
                    count += len(batch['y'])
                assert count == len(tr)
                labels, predictions = evaluate(model, valid_loader)
                score = base.metrics(labels, predictions, va.cliff_mol)
                value = float(np.mean((np.asarray(labels) - predictions) ** 2))
                assert np.isfinite(value)
                history.append(dict(epoch=epoch + 1, train_mse=total/count, lr=optimizer.param_groups[0]['lr'], **score))
                if value < best:
                    best, stale, best_epoch = value, 0, epoch + 1
                    torch.save(model.state_dict(), destination / 'best_model.pt')
                else:
                    stale += 1
                base.save(destination / 'history.json', history)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print(f'{task} {arm} epoch={epoch+1} rmse={score["rmse"]:.6f} cliff={score["cliff_rmse"]:.6f} stale={stale}', flush=True)
                scheduler.step(epoch)
                if stale >= 15:
                    break
            model.load_state_dict(torch.load(destination / 'best_model.pt', weights_only=True))
            labels, predictions = evaluate(model, valid_loader)
            pd.DataFrame(dict(row_id=va.row_id, label=labels, prediction=predictions, cliff=va.cliff_mol)).to_csv(destination / 'valid_predictions.csv', index=False)
            result = dict(task=task, arm=arm, initial_sha256=initial, graph_initial_sha256=graph_hash,
                          fp_initial_sha256=fp_hash, parameters=sum(p.numel() for p in model.parameters()),
                          best_epoch=best_epoch, epochs=len(history), training_seconds=time.time()-started,
                          **base.metrics(labels, predictions, va.cliff_mol))
            base.save(destination / 'summary.json', result)
            print('FINISHED ' + json.dumps(result), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()


def audit():
    DOC.mkdir(parents=True, exist_ok=True)
    results = []
    for task in base.TASKS:
        _, va, _, vg = data(task)
        loader = DataLoader(vg, batch_size=32, collate_fn=collate)
        for arm in ARMS:
            path = OUT / task / arm
            result = json.loads((path / 'summary.json').read_text())
            model = Predictor(arm)
            model.load_state_dict(torch.load(path / 'best_model.pt', weights_only=True), strict=True)
            labels, predictions = evaluate(model, loader)
            stored = pd.read_csv(path / 'valid_predictions.csv')
            assert stored.row_id.tolist() == va.row_id.tolist()
            assert np.max(np.abs(stored.label.to_numpy()-labels)) < 1e-6
            assert stored.cliff.tolist() == va.cliff_mol.tolist()
            difference = float(np.max(np.abs(stored.prediction.to_numpy()-predictions)))
            assert difference < 1e-5
            for key, value in base.metrics(labels, predictions, va.cliff_mol).items():
                errors = [(float(y)-float(p))**2 for y, p, c in zip(stored.label, stored.prediction, stored.cliff)
                          if key == 'rmse' or c]
                assert abs(math.sqrt(math.fsum(errors)/len(errors))-value) < 1e-6
                assert abs(value-result[key]) < 1e-5
            result['replay_max_abs'] = difference
            results.append(result)
            stored.to_csv(DOC / f'{task}_{arm}_valid_predictions.csv', index=False)
            for name in ('history.json',):
                (DOC / f'{task}_{arm}_{name}').write_bytes((path / name).read_bytes())
            del model
        for name in ('split.json', 'provenance.json'):
            (DOC / f'{task}_{name}').write_bytes((OUT / task / name).read_bytes())
    base.save(DOC / 'audit.json', results)
    lines = ['# 路线 A：图与 Morgan 指纹融合先导', '',
             '两个任务、单seed42、官方train内10%验证；仅validation，无test评价。四组均MSE、均匀采样和相同预算。', '',
             '| Task | Arm | Overall RMSE | Cliff RMSE | best/epochs | Parameters |',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["task"]} | {r["arm"]} | {r["rmse"]:.6f} | {r["cliff_rmse"]:.6f} | {r["best_epoch"]}/{r["epochs"]} | {r["parameters"]} |')
    lines.extend(['', '## 相对变化（负值改善）', '', '| Task | gate vs | Overall | Cliff |', '|---|---|---:|---:|'])
    for task in base.TASKS:
        arms = {r['arm']: r for r in results if r['task'] == task}
        for reference in ('graph', 'fp', 'concat'):
            changes = [(arms['gate'][key]/arms[reference][key]-1)*100 for key in ('rmse', 'cliff_rmse')]
            lines.append(f'| {task} | {reference} | {changes[0]:+.2f}% | {changes[1]:+.2f}% |')
    lines.extend(['', '全部checkpoint在独立进程strict重载，逐样本预测与保存CSV误差<1e-5；标准库math.fsum复算指标通过。', '',
                  '单种子且validation参与选模，不能声称统计显著或官方测试收益。新增分支增加参数量；concat首层头输入维度变化，权重无法完全匹配。',
                  '这是通用融合试验，不宣称原创；未叠加ACA/FPPool/成对模块，未调参、未自动扩大。'])
    (DOC / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'run', 'audit'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'check': check, 'run': run, 'audit': audit}[args.action]()
