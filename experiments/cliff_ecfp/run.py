"""GPL-3.0: reuse frozen contrastive training with the existing ECFP concat adapter."""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


cc = module('contrastive_training', 'experiments/cliff_contrastive/run.py')
p = module('existing_concat', 'experiments/route_a/pilot.py')
torch, np, pd = cc.torch, cc.np, cc.pd
OLD = ROOT / 'artifacts/cliff_contrastive_20261012'
OLD_DOC = ROOT / 'docs/cliff_contrastive_20261012'
OUT = ROOT / 'artifacts/cliff_ecfp_20261012'
DOC = ROOT / 'docs/cliff_ecfp_20261012'
graph_model = cc.model_new
contrastive_report = cc.report
gradient_probe = cc.local.gradient_probe
# Isolated module namespaces: adapt the existing loop without altering frozen files or other jobs.
cc.base = SimpleNamespace(**vars(cc.base))
p.base = SimpleNamespace(**vars(p.base))
cc.local = SimpleNamespace(**vars(cc.local))
core_parameter_ids = set()
actual_initial = {}


class Predictor(p.Predictor):
    def __init__(self, seed):
        def factory():
            original = graph_model(seed)
            self.reference_initial_sha256 = cc.base.state_hash(original)
            return original
        previous = p.base.model_new
        p.base.model_new = factory
        try:
            super().__init__('concat')
        finally:
            p.base.model_new = previous
        self.core_initial_sha256 = cc.base.state_hash(self.graph)

    @property
    def reg_head(self):
        # The frozen loop's pre-hook captures graph features before concatenation.
        return self.graph.reg_head


def model_new(seed):
    global core_parameter_ids, actual_initial
    model = Predictor(seed)
    core_parameter_ids = {id(v) for v in model.graph.parameters()}
    actual_initial = dict(core_initial_sha256=model.core_initial_sha256,
                          reference_initial_sha256=model.reference_initial_sha256)
    return model


def validate(model, graphs, frame):
    loader = cc.base.DataLoader(graphs, batch_size=128, collate_fn=p.collate)
    labels, predictions = p.evaluate(model, loader)
    return labels, predictions, cc.base.metrics(labels, predictions, frame.cliff_mol)


cc.OUT, cc.DOC, cc.__file__ = OUT, DOC, __file__
cc.model_new, cc.validate = model_new, validate
cc.forward = lambda model, batch: model(batch)
cc.base.development, cc.base.collate_graphs = p.data, p.collate
cc.local.gradient_probe = lambda reg, aux, params: gradient_probe(reg, aux, [v for v in params if id(v) in core_parameter_ids])


def reference_files():
    files = [OLD / 'manifest.json', OLD_DOC / 'results.json', OLD_DOC / 'aggregate.json']
    for task in cc.TASKS:
        for seed in cc.SEEDS:
            for arm in cc.ARMS:
                files.extend(OLD / task / str(seed) / arm / name for name in
                             ('summary.json', 'best_model.pt', 'valid_predictions.csv', 'history.json'))
    return files


def check_references(full=False):
    manifest = cc.guard()
    for name, expected in manifest['reference_sha256'].items():
        if full or not name.endswith('.pt'): assert cc.digest(ROOT / name) == expected, name
    return manifest


def check():
    assert not (OUT / 'manifest.json').exists(), 'already frozen'
    old = json.loads((OLD / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in old['source_sha256'].items(): assert cc.digest(ROOT / name) == expected, name
    results = json.loads((OLD_DOC / 'results.json').read_text(encoding='utf-8'))
    audit = json.loads((OLD_DOC / 'aggregate.json').read_text(encoding='utf-8'))
    assert audit['completed'] == 24 and audit['independent_reload_audited'] and len(results) == 24
    assert torch.__version__ == old['torch'] and torch.version.cuda == old['cuda']
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    checks = []
    for task in cc.TASKS:
        tr, va, tg, vg = p.data(task)
        split = next(item for item in old['splits'] if item['task'] == task)
        assert tr.row_id.tolist() == split['train'] and va.row_id.tolist() == split['valid']
        _, pm, nm = cc.pairs(tr)
        ids = cc.batches(len(tr), pm.nonzero().cpu().tolist(), nm.nonzero().cpu().tolist(), 42, 1)[0]
        batch = p.collate([tg[i] for i in ids.tolist()]); ci = ids.cuda()
        core_hashes = {}
        for seed in cc.SEEDS:
            model = model_new(seed)
            reference = next(r for r in results if r['task'] == task and r['seed'] == seed and r['arm'] == 'mse')
            assert model.reference_initial_sha256 == reference['initial_sha256']
            plain = graph_model(seed); plain.reg_head = torch.nn.Identity()
            assert cc.base.state_hash(plain) == model.core_initial_sha256
            core_hashes[str(seed)] = model.core_initial_sha256
            del model, plain
        model = model_new(42); clone = model_new(42)
        assert cc.base.state_hash(model) == cc.base.state_hash(clone)
        model.eval(); clone.eval()
        captured = []
        hook = model.reg_head.register_forward_pre_hook(lambda mod, args: captured.append(args[0]))
        with torch.no_grad():
            predicted = model(batch)
            torch.testing.assert_close(predicted, clone(batch), rtol=0, atol=0)
            expected = model.head(torch.cat((captured.pop(), model.fp(batch['fp'])), dim=1))
            torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
        model.train()
        pred = model(batch); h = captured.pop()
        assert h.shape == (len(ids), 512) and pred.shape == batch['y'].shape
        reg = cc.F.mse_loss(pred, batch['y'])
        aux, details = cc.auxiliary(h, pm[ci[:, None], ci[None, :]], nm[ci[:, None], ci[None, :]])
        noncore = list(model.fp.parameters()) + list(model.head.parameters())
        assert all(v is None for v in torch.autograd.grad(aux, noncore, retain_graph=True, allow_unused=True))
        torch.testing.assert_close(torch.autograd.grad(reg + 0. * aux, pred, retain_graph=True)[0],
                                   torch.autograd.grad(reg, pred, retain_graph=True)[0], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
        for step in range(3):
            if step:
                pred = model(batch); h = captured.pop()
                reg = cc.F.mse_loss(pred, batch['y'])
                aux, details = cc.auxiliary(h, pm[ci[:, None], ci[None, :]], nm[ci[:, None], ci[None, :]])
            optimizer.zero_grad(set_to_none=True)
            loss = reg + .01 * aux; assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        hook.remove()
        file = OUT / f'check_{task}.pt'; torch.save({'model_state_dict': model.state_dict()}, file)
        clone.load_state_dict(torch.load(file, weights_only=True)['model_state_dict'], strict=True)
        model.eval(); clone.eval()
        with torch.no_grad(): torch.testing.assert_close(model(batch), clone(batch), rtol=0, atol=0)
        checks.append(dict(task=task, core_initial_sha256=core_hashes,
                           parameters=sum(v.numel() for v in model.parameters()), checks='passed'))
        print('CHECK', task, checks[-1]['parameters'], flush=True)
        del model, clone, optimizer; torch.cuda.empty_cache()
    manifest = copy.deepcopy(old)
    sources = tuple(old['source_sha256']) + ('experiments/route_a/pilot.py', 'experiments/cliff_ecfp/run.py',
                                            'experiments/cliff_ecfp/task_plan.md')
    manifest.update(source_sha256={name: cc.digest(ROOT / name) for name in sources}, checks=checks,
                    reference_sha256={str(file.relative_to(ROOT)): cc.digest(file) for file in reference_files()},
                    protocol=(ROOT / 'experiments/cliff_ecfp/task_plan.md').read_text(encoding='utf-8'))
    cc.base.save(OUT / 'manifest.json', manifest)
    cc.base.save(DOC / 'preflight.json', checks)
    report()


def report(audit=False):
    manifest = check_references(full=audit)
    fresh = contrastive_report(audit=audit)
    (DOC / 'concat_aggregate.json').write_text(json.dumps(fresh, indent=2), encoding='utf-8')
    rows = json.loads((OLD_DOC / 'results.json').read_text(encoding='utf-8'))
    for row in rows: row['group'] = 'graph_' + row['arm']
    new = json.loads((DOC / 'results.json').read_text(encoding='utf-8'))
    for row in new: row['group'] = 'concat_' + row['arm']
    rows += new
    blocks = []
    for task in cc.TASKS:
        for seed in cc.SEEDS:
            found = {r['group']: r for r in rows if r['task'] == task and r['seed'] == seed}
            if len(found) != 4: continue
            histories = [json.loads((origin / task / str(seed) / arm / 'history.json').read_text(encoding='utf-8'))
                         for origin in (OLD, OUT) for arm in cc.ARMS]
            assert all(len({r['sampler_sha256'] for r in epoch}) == 1 for epoch in zip(*histories))
            for arm in cc.ARMS:
                item = found['concat_' + arm]
                expected = next(c for c in manifest['checks'] if c['task'] == task)['core_initial_sha256'][str(seed)]
                assert item['core_initial_sha256'] == expected
                assert item['reference_initial_sha256'] == found['graph_' + arm]['initial_sha256']
            blocks.append(found)
    aggregate = dict(reused_models=24, new_completed=len(new), new_total=24, paired_blocks=len(blocks),
                     independent_reload_audited=audit, go=False)
    if len(blocks) == 12:
        groups = ('graph_mse', 'graph_margin', 'concat_mse', 'concat_margin')
        means = {g: {key: float(np.mean([b[g][key] for b in blocks])) for key in ('rmse', 'cliff_rmse')} for g in groups}
        supported = sum(sum(b['concat_margin']['cliff_rmse'] < min(b['concat_mse']['cliff_rmse'], b['graph_margin']['cliff_rmse'])
                             for b in blocks if b['graph_mse']['task'] == task) >= 2 for task in cc.TASKS)
        aggregate.update(macro=means, tasks_beating_both_single_at_least_two_seeds=supported,
            mean_cliff_interaction=float(np.mean([b['concat_margin']['cliff_rmse'] - b['concat_mse']['cliff_rmse'] -
                                                   b['graph_margin']['cliff_rmse'] + b['graph_mse']['cliff_rmse'] for b in blocks])))
        c = means['concat_margin']
        aggregate['go'] = (all(c['cliff_rmse'] <= means[g]['cliff_rmse'] * .99 for g in ('concat_mse', 'graph_margin'))
                           and all(c['rmse'] <= means[g]['rmse'] * 1.01 for g in ('graph_mse', 'concat_mse')) and supported >= 3)
    cc.base.save(DOC / 'four_group_results.json', rows); cc.base.save(DOC / 'aggregate.json', aggregate)
    table = []
    for block in blocks:
        item = dict(task=block['graph_mse']['task'], seed=block['graph_mse']['seed'])
        for group, result in block.items():
            item[group + '_rmse'] = result['rmse']; item[group + '_cliff'] = result['cliff_rmse']
        table.append(item)
    pd.DataFrame(table).to_csv(DOC / 'four_group_paired.csv', index=False)
    lines = ['# ECFP × cosine-margin 四组开发比较', '', f'复用24个纯图模型；新完成{len(new)}/24；四组完整配对{len(blocks)}/12。',
             '仅官方train内验证，不与论文test表混排；图表示承担辅助约束，ECFP只进入回归输入。', '',
             '| Task | seed | graph MSE Cliff | graph margin Cliff | concat MSE Cliff | concat margin Cliff |',
             '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['task']} | {r['seed']} | {r['graph_mse_cliff']:.6f} | {r['graph_margin_cliff']:.6f} | {r['concat_mse_cliff']:.6f} | {r['concat_margin_cliff']:.6f} |" for r in table]
    lines += ['', '```json', json.dumps(aggregate, indent=2, ensure_ascii=False), '```', '',
              '先导不是显著性验证；更换输入同时增加参数，未有ECFP-only/容量匹配对照，不声明算法原创。全train/test尚未执行。']
    (DOC / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


cc.report = report


def job(task, seed, arm):
    check_references()
    cc.job(task, seed, arm)
    file = OUT / task / str(seed) / arm / 'summary.json'
    value = json.loads(file.read_text(encoding='utf-8'))
    expected = next(c for c in cc.guard()['checks'] if c['task'] == task)['core_initial_sha256'][str(seed)]
    reference = json.loads((OLD / task / str(seed) / arm / 'summary.json').read_text(encoding='utf-8'))
    assert actual_initial['core_initial_sha256'] == expected
    assert actual_initial['reference_initial_sha256'] == reference['initial_sha256']
    value.update(**actual_initial, auxiliary_representation='graph_512_before_concat')
    cc.base.save(file, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'queue', 'job', 'report', 'audit'))
    parser.add_argument('--task', choices=cc.TASKS); parser.add_argument('--seed', type=int, choices=cc.SEEDS)
    parser.add_argument('--arm', choices=cc.ARMS)
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.action == 'job': job(args.task, args.seed, args.arm)
    elif args.action == 'check': check()
    elif args.action == 'queue': cc.queue()
    else: report(audit=args.action == 'audit')
