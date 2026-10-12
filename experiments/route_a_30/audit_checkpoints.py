"""Evaluate existing upstream-format checkpoints; never train or overwrite weights."""
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location('route30', Path(__file__).with_name('run.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.torch.set_num_threads(4)
out = r.ROOT / 'docs/graphcliff_checkpoint_audit_20261011'
out.mkdir(exist_ok=True)
records = []
audit = json.loads((r.ROOT/'docs/checkpoint_audit_20261010.json').read_text(encoding='utf-8'))
known = {row['task']: row for row in audit['records']}
for task in r.TASKS:
    checkpoint = r.ROOT / known[task]['path']
    assert r.digest(checkpoint) == known[task]['sha256']
    state = r.torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    model = r.p.Predictor('graph')
    mapped = {('head.'+key[len('reg_head.'):] if key.startswith('reg_head.') else 'graph.'+key): value
              for key, value in state.items()}
    model.load_state_dict(mapped, strict=True)
    assert r.digest(r.ROOT/'benchmark_data'/f'{task}.csv') == next(
        row['data_sha256'] for row in json.loads((r.OUT/'manifest.json').read_text(encoding='utf-8'))['tasks'] if row['task']==task)
    frame, metrics = r.final_test(task, model)
    frame.to_csv(out/f'{task}_predictions.csv', index=False)
    records.append(dict(task=task, checkpoint=str(checkpoint), sha256=known[task]['sha256'], **metrics))
    r.save(out/'results.json', dict(records=records, completed=len(records)))
    print(task, metrics, flush=True)
assert len(records)==30
aggregate = {key: math.fsum(row[key] for row in records)/30 for key in ('rmse','cliff_rmse')}
r.save(out/'results.json', dict(records=records, completed=30, aggregate=aggregate,
       upstream_archive_hash_verified=False, training_provenance_verified=False))
print(aggregate, flush=True)
