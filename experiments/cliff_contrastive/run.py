"""GPL-3.0: bounded cosine-margin pilot; unchanged GraphCliff, train-only pairs."""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/aca'))
import run as base
spec = importlib.util.spec_from_file_location('aca_local_tools', ROOT / 'experiments/aca_local/pilot.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
torch, np, pd = base.torch, base.np, base.pd
F = torch.nn.functional
OUT = ROOT / 'artifacts/cliff_contrastive_20261012'
DOC = ROOT / 'docs/cliff_contrastive_20261012'
TASKS = ('CHEMBL234_Ki', 'CHEMBL2047_EC50', 'CHEMBL2147_Ki', 'CHEMBL244_Ki')
SEEDS = (42, 43, 44)
ARMS = ('mse', 'margin')
SOURCES = ('model.py', 'dataset.py', 'dataset_utils.py', 'main.py', 'train.py',
           'experiments/aca/run.py', 'experiments/aca/loss_adapter.py',
           'experiments/aca_local/pilot.py', 'experiments/cliff_contrastive/run.py',
           'experiments/cliff_contrastive/task_plan.md')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_new(seed):
    base.set_seed(SimpleNamespace(seed=seed))
    return base.GraphCliffRegressor(38, 13, hidden_size=256, num_layers=3,
                                   groups=4, mid_K=3, dropout=0).cuda()


def pairs(frame):
    sim = local.fingerprints(frame)
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float32, device='cuda')
    delta = (y[:, None] - y[None, :]).abs()
    upper = torch.triu(sim >= .8, diagonal=1)
    pos, neg = upper & (delta < .2), upper & (delta >= 1.)
    return sim, pos, neg


def auxiliary(h, pos, neg):
    terms, details = [], {}
    for name, mask, target in (('positive', pos, 1.), ('negative', neg, -1.)):
        i, j = mask.nonzero(as_tuple=True)
        details[name + '_pairs'] = len(i)
        if len(i):
            values = F.cosine_embedding_loss(h[i], h[j], h.new_full((len(i),), target),
                                            margin=.5, reduction='none')
            terms.append(values.mean())
            details[name + '_cosine'] = float(F.cosine_similarity(h[i], h[j]).detach().mean())
            details[name + '_active'] = int((values.detach() > 0).sum())
        else:
            terms.append(h.sum() * 0.)
            details[name + '_cosine'], details[name + '_active'] = None, 0
    return .5 * (terms[0] + terms[1]), details


def batches(count, positive, negative, seed, epoch):
    rng = np.random.RandomState(seed * 1000 + epoch)
    order = rng.permutation(count)
    result, ordinary = [], []
    for start in range(0, count, 124):
        fill = order[start:start + 124].tolist()
        ordinary.extend(fill)
        occupied = set(fill)
        pc = [pair for pair in positive if not occupied.intersection(pair)]
        nc = [pair for pair in negative if not occupied.intersection(pair)]
        assert pc and nc, 'batch cannot inject both classes without duplicate ordinary members'
        extra = sorted(set(pc[rng.randint(len(pc))]) | set(nc[rng.randint(len(nc))]))
        ids = fill + extra
        assert len(ids) <= 128 and len(ids) == len(set(ids))
        result.append(torch.tensor(ids, dtype=torch.long))
    assert sorted(ordinary) == list(range(count))
    return result


def forward(model, batch):
    return model(batch['x'].cuda(), batch['edge_index'].cuda(), batch['edge_attr'].cuda(), batch['batch'].cuda())


def validate(model, graphs, frame):
    loader = base.DataLoader(graphs, batch_size=128, collate_fn=base.collate_graphs)
    labels, predictions = base.evaluate(model, loader, 'cuda')
    return labels, predictions, base.metrics(labels, predictions, frame.cliff_mol)


def check():
    assert not (OUT / 'manifest.json').exists(), 'already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    # Explicit positive/negative formula, ignored diagonal, zero-weight and empty-pair gradients.
    h = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], device='cuda', requires_grad=True)
    pos = torch.zeros(3, 3, dtype=torch.bool, device='cuda'); pos[0, 1] = True
    neg = pos.clone(); neg.zero_(); neg[0, 2] = True
    aux, info = auxiliary(h, pos, neg)
    expected = .5 * (1. + 1. / math.sqrt(2.) - .5)
    torch.testing.assert_close(aux, aux.new_tensor(expected))
    assert info['positive_pairs'] == info['negative_pairs'] == 1
    assert torch.autograd.grad(aux, h, retain_graph=True)[0].abs().sum() > 0
    reg = h.square().mean()
    torch.testing.assert_close(torch.autograd.grad(reg + 0. * aux, h, retain_graph=True)[0],
                               torch.autograd.grad(reg, h, retain_graph=True)[0], rtol=0, atol=0)
    empty, _ = auxiliary(h, torch.zeros_like(pos), torch.zeros_like(neg))
    assert float(empty.detach()) == 0 and torch.count_nonzero(torch.autograd.grad(empty, h)[0]) == 0
    coverage, timing = [], []
    for task in TASKS:
        tr, va, tg, vg = base.development(task)
        membership = pd.read_csv(ROOT / 'benchmark_data' / f'{task}.csv', usecols=['split'])
        official_train = set(np.flatnonzero(membership.split.eq('train').to_numpy()).tolist())
        official_test = np.flatnonzero(membership.split.eq('test').to_numpy()).tolist()
        assert set(tr.row_id) | set(va.row_id) == official_train
        assert not set(tr.row_id).intersection(va.row_id)
        assert not official_train.intersection(official_test)
        sim, pm, nm = pairs(tr)
        positive, negative = pm.nonzero().cpu().tolist(), nm.nonzero().cpu().tolist()
        assert positive and negative, task + ': insufficient pair classes'
        records = [dict(row_i=int(tr.row_id.iloc[i]), row_j=int(tr.row_id.iloc[j]),
                        kind=kind, similarity=float(sim[i, j]), delta=abs(float(tr.y.iloc[i] - tr.y.iloc[j])))
                   for kind, edges in (('positive', positive), ('negative', negative)) for i, j in edges]
        pd.DataFrame(records).to_csv(DOC / f'{task}_train_pairs.csv', index=False)
        for seed in SEEDS:
            for epoch in range(1, 101):
                for ids in batches(len(tr), positive, negative, seed, epoch):
                    ids = ids.cuda()
                    assert pm[ids[:, None], ids[None, :]].any() and nm[ids[:, None], ids[None, :]].any()
        coverage.append(dict(task=task, train=tr.row_id.tolist(), valid=va.row_id.tolist(),
                             test=official_test, positive_pairs=len(positive), negative_pairs=len(negative),
                             train_pair_sha256=digest(DOC / f'{task}_train_pairs.csv'),
                             data_sha256=digest(ROOT / 'benchmark_data' / f'{task}.csv')))
        ids = batches(len(tr), positive, negative, 42, 1)[0]
        batch = base.collate_graphs([tg[i] for i in ids.tolist()]); ci = ids.cuda()
        model = model_new(42); captured = []
        hook = model.reg_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
        start = time.monotonic()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, batch)
            assert pred.shape == batch['y'].shape, 'MSE shapes must match without broadcasting'
            reg = F.mse_loss(pred, batch['y'].cuda())
            aux, details = auxiliary(captured.pop(), pm[ci[:, None], ci[None, :]], nm[ci[:, None], ci[None, :]])
            loss = reg + .01 * aux
            assert torch.isfinite(loss) and details['positive_pairs'] and details['negative_pairs']
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        hook.remove()
        checkpoint = OUT / f'preflight_{task}.pt'
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        clone = model_new(42)
        clone.load_state_dict(torch.load(checkpoint, weights_only=True)['model_state_dict'], strict=True)
        model.eval(); clone.eval()
        with torch.no_grad(): torch.testing.assert_close(forward(model, batch), forward(clone, batch), atol=1e-6, rtol=1e-5)
        timing.append(dict(task=task, seconds_per_batch=(time.monotonic() - start) / 3))
        print('CHECK', task, len(tr), len(va), len(positive), len(negative), flush=True)
        del model, clone, optimizer
        torch.cuda.empty_cache()
    order = []
    rng = np.random.RandomState(20261012)
    for task in TASKS:
        for seed in SEEDS:
            order.extend(dict(task=task, seed=seed, arm=arm) for arm in rng.permutation(ARMS))
    base.save(OUT / 'manifest.json', dict(splits=coverage, order=order, timing=timing,
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        protocol=(ROOT / 'experiments/cliff_contrastive/task_plan.md').read_text(encoding='utf-8'),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), test_labels_read=False))
    base.save(DOC / 'coverage.json', [{k: v for k, v in item.items() if k not in ('train', 'valid', 'test')} |
                                   dict(train_count=len(item['train']), valid_count=len(item['valid'])) for item in coverage])


def guard():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_sha256'].items(): assert digest(ROOT / name) == expected, name
    for item in manifest['splits']:
        assert digest(ROOT / 'benchmark_data' / f"{item['task']}.csv") == item['data_sha256']
    return manifest


def job(task, seed, arm):
    manifest = guard()
    tr, va, tg, vg = base.development(task)
    split = next(item for item in manifest['splits'] if item['task'] == task)
    assert tr.row_id.tolist() == split['train'] and va.row_id.tolist() == split['valid']
    sim, pm, nm = pairs(tr)
    positive, negative = pm.nonzero().cpu().tolist(), nm.nonzero().cpu().tolist()
    folder = OUT / task / str(seed) / arm; folder.mkdir(parents=True, exist_ok=False)
    model = model_new(seed); initial = base.state_hash(model); captured = []
    hook = model.reg_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    shared = [value for name, value in model.named_parameters() if not name.startswith('reg_head.')]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
    scheduler = base.WarmupCosineScheduler(optimizer, 10, 100, 1e-4)
    best, stale, history = float('inf'), 0, []; started = time.monotonic()
    for epoch in range(1, 101):
        model.train(); regsum = auxsum = samples = positive_count = negative_count = active = 0
        sampler_hash = hashlib.sha256()
        for step, ids in enumerate(batches(len(tr), positive, negative, seed, epoch)):
            sampler_hash.update(ids.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            batch = base.collate_graphs([tg[i] for i in ids.tolist()])
            pred = forward(model, batch)
            ci = ids.cuda()
            assert pred.shape == batch['y'].shape, 'MSE shapes must match without broadcasting'
            y = batch['y'].cuda()
            torch.testing.assert_close(y.view(-1), torch.tensor(tr.y.iloc[ids.tolist()].to_numpy(), dtype=torch.float32, device='cuda'), rtol=0, atol=0)
            reg = F.mse_loss(pred, y)
            aux, details = auxiliary(captured.pop(), pm[ci[:, None], ci[None, :]], nm[ci[:, None], ci[None, :]])
            assert details['positive_pairs'] and details['negative_pairs']
            loss = reg + (.01 * aux if arm == 'margin' else 0.)
            if step == 0:
                probe = local.gradient_probe(reg, .01 * aux, shared)
                first_pairs = details
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            regsum += float(reg.detach()) * len(ids); auxsum += float(aux.detach()) * len(ids); samples += len(ids)
            positive_count += details['positive_pairs']; negative_count += details['negative_pairs']; active += details['negative_active']
        labels, predictions, scores = validate(model, vg, va); captured.clear()
        mse = math.fsum((float(y) - float(p)) ** 2 for y, p in zip(labels, predictions)) / len(labels)
        assert math.isfinite(mse)
        history.append(dict(epoch=epoch, train_mse=regsum / samples, auxiliary=auxsum / samples,
                            samples=samples, ordinary_samples=len(tr), positive_pairs=positive_count,
                            negative_pairs=negative_count, active_negatives=active, sampler_sha256=sampler_hash.hexdigest(),
                            first_batch_gradient=probe, first_batch_pairs=first_pairs, lr=optimizer.param_groups[0]['lr'], **scores))
        if mse < best - 1e-4:
            best, stale, best_epoch = mse, 0, epoch
            torch.save({'model_state_dict': model.state_dict()}, folder / 'best_model.pt')
        else:
            stale += 1
        scheduler.step(epoch)
        base.save(folder / 'history.json', history)
        base.save(folder / 'progress.json', dict(epoch=epoch, stale=stale, best_epoch=best_epoch, phase='training'))
        if epoch == 1 or epoch % 10 == 0: print(task, seed, arm, epoch, scores, 'stale', stale, flush=True)
        if stale >= 15: break
    hook.remove()
    model.load_state_dict(torch.load(folder / 'best_model.pt', weights_only=True)['model_state_dict'], strict=True)
    capture = []
    hook = model.reg_head.register_forward_pre_hook(lambda module, args: capture.append(args[0].detach()))
    labels, predictions, scores = validate(model, vg, va)
    _, vp, vn = pairs(va)
    _, geometry = auxiliary(torch.cat(capture), vp, vn); hook.remove()
    for name, mask in (('positive', vp), ('negative', vn)):
        i, j = mask.nonzero(as_tuple=True)
        geometry[name + '_delta_mae'] = float(np.mean(np.abs((np.asarray(predictions)[i.cpu()] - np.asarray(predictions)[j.cpu()]) -
            (np.asarray(labels)[i.cpu()] - np.asarray(labels)[j.cpu()])))) if len(i) else None
    pd.DataFrame(dict(row_id=va.row_id, label=labels, prediction=predictions, cliff=va.cliff_mol)).to_csv(folder / 'valid_predictions.csv', index=False)
    base.save(folder / 'summary.json', dict(task=task, seed=seed, arm=arm, initial_sha256=initial,
        parameters=sum(v.numel() for v in model.parameters()), best_epoch=best_epoch, epochs=len(history),
        train_count=len(tr), valid_count=len(va), validation_geometry=geometry,
        seconds=time.monotonic() - started, test_labels_read=False, **scores))
    base.save(folder / 'progress.json', dict(epoch=len(history), phase='complete'))
    print('COMPLETE', task, seed, arm, scores, flush=True)


def report(audit=False):
    manifest = guard(); rows = []
    for item in manifest['order']:
        task, seed, arm = item['task'], item['seed'], item['arm']
        folder = OUT / task / str(seed) / arm
        if not (folder / 'summary.json').exists(): continue
        result = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        csv = pd.read_csv(folder / 'valid_predictions.csv')
        split = next(s for s in manifest['splits'] if s['task'] == task)
        assert csv.row_id.tolist() == split['valid']
        for key, subset in (('rmse', csv), ('cliff_rmse', csv[csv.cliff.eq(1)])):
            measured = math.sqrt(math.fsum((float(y) - float(p)) ** 2 for y, p in zip(subset.label, subset.prediction)) / len(subset))
            assert abs(measured - result[key]) < 1e-9
        if audit:
            _, va, _, vg = base.development(task)
            model = model_new(seed)
            model.load_state_dict(torch.load(folder / 'best_model.pt', weights_only=True)['model_state_dict'], strict=True)
            labels, predictions, _ = validate(model, vg, va)
            assert np.max(np.abs(csv.label - labels)) < 1e-6
            assert np.max(np.abs(csv.prediction - predictions)) < 1e-5
            assert csv.cliff.tolist() == va.cliff_mol.tolist()
            del model; torch.cuda.empty_cache()
        rows.append(result)
    deltas = []
    for task in TASKS:
        for seed in SEEDS:
            matched = {item['arm']: item for item in rows if item['task'] == task and item['seed'] == seed}
            if len(matched) != 2: continue
            g, m = matched['mse'], matched['margin']
            assert g['initial_sha256'] == m['initial_sha256'] and g['parameters'] == m['parameters']
            histories = [json.loads((OUT / task / str(seed) / a / 'history.json').read_text(encoding='utf-8')) for a in ARMS]
            assert all(x['sampler_sha256'] == y['sampler_sha256'] for x, y in zip(*histories))
            deltas.append(dict(task=task, seed=seed, mse_rmse=g['rmse'], margin_rmse=m['rmse'],
                               mse_cliff=g['cliff_rmse'], margin_cliff=m['cliff_rmse']))
    pd.DataFrame(deltas).to_csv(DOC / 'paired_results.csv', index=False)
    aggregate = dict(completed=len(rows), total=24, paired_blocks=len(deltas), independent_reload_audited=audit)
    if len(deltas) == 12:
        for key in ('rmse', 'cliff'):
            g = float(np.mean([d['mse_' + key] for d in deltas])); m = float(np.mean([d['margin_' + key] for d in deltas]))
            aggregate[key] = dict(mse=g, margin=m, relative_improvement=1. - m / g)
        supported = sum(sum(d['margin_cliff'] < d['mse_cliff'] for d in deltas if d['task'] == task) >= 2 for task in TASKS)
        aggregate['tasks_winning_at_least_two_seeds'] = supported
        aggregate['go'] = aggregate['cliff']['relative_improvement'] >= .01 and aggregate['rmse']['relative_improvement'] >= -.01 and supported >= 3
    base.save(DOC / 'results.json', rows); base.save(DOC / 'aggregate.json', aggregate)
    lines = ['# 结构局部 cosine-margin 开发先导', '', f"完成{len(rows)}/24；完整配对{len(deltas)}/12个task×seed；独立重载审计={audit}。",
             '固定官方train内验证；无test评价。0.5 margin/.01权重是试验设定；未宣称原创或论文复现。', '',
             '| Task | seed | MSE RMSE | margin RMSE | MSE Cliff | margin Cliff |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {d['task']} | {d['seed']} | {d['mse_rmse']:.6f} | {d['margin_rmse']:.6f} | {d['mse_cliff']:.6f} | {d['margin_cliff']:.6f} |" for d in deltas]
    lines += ['', '```json', json.dumps(aggregate, indent=2, ensure_ascii=False), '```', '',
              '同数据划分重复种子不代表独立数据集；验证参与选模。全train/test与ECFP组合尚未执行。梯度诊断只抽首批。']
    (DOC / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


def queue():
    manifest = guard(); logs = OUT / 'logs'; logs.mkdir(exist_ok=False)
    for done, item in enumerate(manifest['order']):
        base.save(OUT / 'status.json', dict(status='running', completed=done, total=24, **item))
        with (logs / f"{item['task']}_{item['seed']}_{item['arm']}.log").open('w', encoding='utf-8') as stream:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'job', '--task', item['task'],
                                    '--seed', str(item['seed']), '--arm', item['arm']], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if child.returncode:
            base.save(OUT / 'status.json', dict(status='failed', completed=done, total=24, exit_code=child.returncode, **item))
            report(); return
        report(); print('QUEUE', done + 1, '/24', item, flush=True)
    base.save(OUT / 'status.json', dict(status='auditing', completed=24, total=24))
    try:
        aggregate = report(audit=True)
    except Exception as error:
        base.save(OUT / 'status.json', dict(status='failed_audit', completed=24, total=24,
                                          error=repr(error)))
        raise
    base.save(OUT / 'status.json', dict(status='complete', completed=24, total=24, go=aggregate['go']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'job', 'queue', 'report', 'audit'))
    parser.add_argument('--task', choices=TASKS); parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.action == 'job': job(args.task, args.seed, args.arm)
    elif args.action == 'report': report()
    elif args.action == 'audit': report(audit=True)
    else: {'check': check, 'queue': queue}[args.action]()
