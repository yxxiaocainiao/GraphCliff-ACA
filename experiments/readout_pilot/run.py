"""GPL-3.0: bounded readout pilot reusing the frozen GraphCliff utilities."""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('route30',ROOT/'experiments/route_a_30/run.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
p=r.p; torch=r.torch; np=r.np; pd=r.pd
from torch_geometric.nn import global_max_pool, global_mean_pool

OUT=ROOT/'artifacts/readout_pilot_20261011'
DOC=ROOT/'docs/readout_pilot_20261011'
TASKS=('CHEMBL234_Ki','CHEMBL2047_EC50','CHEMBL2147_Ki','CHEMBL244_Ki')
ARMS=('graph','final','shallow')
SOURCES=('model.py','dataset.py','dataset_utils.py','main.py','train.py',
         'experiments/aca/run.py','experiments/route_a/pilot.py','experiments/route_a_30/run.py',
         'experiments/readout_pilot/run.py','experiments/readout_pilot/task_plan.md')


class Predictor(p.Predictor):
    def __init__(self,arm):
        super().__init__('graph'); self.readout_arm=arm
        if arm!='graph':
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(1042)
                self.projection=torch.nn.Sequential(torch.nn.LayerNorm(512),torch.nn.Linear(512,32),
                    torch.nn.ReLU(),torch.nn.Linear(32,512)).cuda()
                self.alpha=torch.nn.Parameter(torch.zeros((),device='cuda'))
            if arm=='shallow':
                self.graph.encoder.layers[0].short.register_forward_hook(self.capture_shallow)
            else:
                self.graph.sagpool.register_forward_pre_hook(self.capture_final)

    def capture_shallow(self,module,args,result):
        self.local=torch.chunk(result,3,dim=-1)[2]

    def capture_final(self,module,args):
        self.local=args[0]

    def forward(self,batch):
        embedding=self.graph(batch['x'],batch['edge_index'],batch['edge_attr'],batch['batch'])
        if self.readout_arm!='graph':
            local=torch.cat((global_max_pool(self.local,batch['batch']),
                             global_mean_pool(self.local,batch['batch'])),dim=1)
            embedding=embedding+self.alpha*self.projection(local)
            self.local=None
        return self.head(embedding)


def collate(graphs):
    return {k:v.cuda() for k,v in p.base.collate_graphs(graphs).items()}


def data(task):
    path=ROOT/'benchmark_data'/f'{task}.csv'
    membership=pd.read_csv(path,usecols=['split'])
    ids=np.flatnonzero(membership.split.eq('train').to_numpy())
    keep=set((ids+1).tolist())
    frame=pd.read_csv(path,skiprows=lambda line:line>0 and line not in keep,
                      usecols=['smiles','y','split'])
    assert len(frame)==len(ids) and frame.split.eq('train').all()
    assert np.isfinite(frame.y).all()
    graphs=[p.base.smiles_to_graph(s,float(y)) for s,y in zip(frame.smiles,frame.y)]
    return ids,graphs


def check():
    assert not (OUT/'manifest.json').exists(), 'preflight already frozen'
    OUT.mkdir(parents=True,exist_ok=True); DOC.mkdir(parents=True,exist_ok=True)
    splits=[]
    for task in TASKS:
        ids,graphs=data(task)
        membership=pd.read_csv(ROOT/'benchmark_data'/f'{task}.csv',usecols=['split'])
        test=np.flatnonzero(membership.split.eq('test').to_numpy()).tolist()
        assert not set(ids).intersection(test)
        splits.append(dict(task=task,train=ids.tolist(),test=test,data_sha256=r.digest(ROOT/'benchmark_data'/f'{task}.csv')))
    batch=collate(graphs[:128]); checks=[]
    for arm in ARMS:
        model=Predictor(arm); original=p.base.model_new()
        model.eval(); original.eval()
        with torch.no_grad():
            predicted=model(batch)
            expected=original(batch['x'],batch['edge_index'],batch['edge_attr'],batch['batch'])
        assert torch.equal(predicted,expected)
        rng=torch.cuda.get_rng_state().clone(); model.train(); original.train()
        torch.cuda.set_rng_state(rng)
        torch.nn.functional.mse_loss(original(batch['x'],batch['edge_index'],batch['edge_attr'],batch['batch']),batch['y']).backward()
        torch.cuda.set_rng_state(rng)
        torch.nn.functional.mse_loss(model(batch),batch['y']).backward()
        wrapped=dict(model.named_parameters())
        for key,value in original.named_parameters():
            mapped='head.'+key[9:] if key.startswith('reg_head.') else 'graph.'+key
            other=wrapped[mapped].grad
            assert (value.grad is None)==(other is None)
            if value.grad is not None: torch.testing.assert_close(value.grad,other,atol=1e-6,rtol=1e-5)
        if arm!='graph':
            assert model.alpha.grad.abs()>0
            assert all(v.grad is not None and torch.count_nonzero(v.grad)==0 for v in model.projection.parameters())
            model.zero_grad(set_to_none=True)
            with torch.no_grad(): model.alpha.fill_(.01)
            torch.nn.functional.mse_loss(model(batch),batch['y']).backward()
            assert sum(float(v.grad.abs().sum()) for v in model.projection.parameters())>0
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
        started=time.monotonic()
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(batch),batch['y']); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        torch.cuda.synchronize()
        r.snapshot(OUT/f'check_{arm}.pt',model.state_dict())
        clone=Predictor(arm); clone.load_state_dict(torch.load(OUT/f'check_{arm}.pt',weights_only=True),strict=True)
        model.eval(); clone.eval()
        with torch.no_grad(): torch.testing.assert_close(model(batch),clone(batch),atol=1e-6,rtol=1e-5)
        checks.append(dict(arm=arm,parameters=sum(v.numel() for v in model.parameters()),seconds_per_batch=(time.monotonic()-started)/10))
        print('CHECK',checks[-1],flush=True)
    assert checks[1]['parameters']==checks[2]['parameters']
    r.save(OUT/'manifest.json',dict(splits=splits,checks=checks,protocol=(ROOT/'experiments/readout_pilot/task_plan.md').read_text(encoding='utf-8'),
        source_sha256={n:r.digest(ROOT/n) for n in SOURCES},torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))


def job(task,arm):
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['source_sha256'].items(): assert r.digest(ROOT/name)==digest,name
    ids,graphs=data(task); split=next(s for s in manifest['splits'] if s['task']==task)
    assert ids.tolist()==split['train'] and r.digest(ROOT/'benchmark_data'/f'{task}.csv')==split['data_sha256']
    folder=OUT/task/arm; folder.mkdir(parents=True,exist_ok=False)
    model=Predictor(arm); graph_hash=p.base.state_hash(model.graph)
    generator=torch.Generator().manual_seed(42)
    loader=r.DataLoader(graphs,batch_size=128,shuffle=True,collate_fn=collate,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
    scheduler=p.base.WarmupCosineScheduler(optimizer,10,100,1e-4)
    history=[]; start=time.monotonic()
    for epoch in range(1,101):
        model.train(); total=0.; count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(batch),batch['y']); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            total+=float(loss.detach())*len(batch['y']); count+=len(batch['y'])
        assert count==len(ids)
        lr=scheduler.step(epoch)
        history.append(dict(epoch=epoch,train_mse=total/count,lr_next=lr,
                            alpha=None if arm=='graph' else float(model.alpha.detach())))
        r.save(folder/'history.json',history)
        r.save(folder/'progress.json',dict(epoch=epoch,total=100,phase='training'))
        if epoch%20==0: print(task,arm,history[-1],flush=True)
    r.snapshot(folder/'final_model.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(torch.load(folder/'final_model.pt',weights_only=True),strict=True)
    frame,metrics=r.final_test(task,model)
    assert frame.row_id.tolist()==split['test']
    frame.to_csv(folder/'test_predictions.csv',index=False)
    r.save(folder/'summary.json',dict(task=task,arm=arm,epochs=100,checkpoint_rule='fixed_epoch_100',train_count=len(ids),
        test=metrics,parameters=sum(v.numel() for v in model.parameters()),graph_initial_sha256=graph_hash,
        alpha=None if arm=='graph' else float(model.alpha.detach()),seconds=time.monotonic()-start))
    r.save(folder/'progress.json',dict(epoch=100,total=100,phase='complete'))
    print('COMPLETE',task,arm,metrics,flush=True)


def report():
    rows=[]
    for task in TASKS:
        for arm in ARMS:
            folder=OUT/task/arm
            if not (folder/'summary.json').exists(): continue
            result=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            frame=pd.read_csv(folder/'test_predictions.csv')
            for key,value in result['test'].items():
                errors=[(float(y)-float(v))**2 for y,v,c in zip(frame.label,frame.prediction,frame.cliff) if key=='rmse' or c]
                assert abs(math.sqrt(math.fsum(errors)/len(errors))-value)<1e-9
            rows.append(dict(task=task,arm=arm,**result['test'],alpha=result['alpha'],graph_initial_sha256=result['graph_initial_sha256']))
    pd.DataFrame(rows).to_csv(DOC/'summary.csv',index=False)
    paired=[t for t in TASKS if sum(row['task']==t for row in rows)==3]
    for task in paired: assert len({row['graph_initial_sha256'] for row in rows if row['task']==task})==1
    means={arm:{key:float(np.mean([row[key] for row in rows if row['arm']==arm and row['task'] in paired])) for key in ('rmse','cliff_rmse')} for arm in ARMS} if paired else {}
    r.save(DOC/'aggregate.json',dict(completed=len(rows),total=12,paired_tasks=len(paired),macro=means))
    lines=['# 完整官方train：读出支路单seed先导','',f'已完成{len(rows)}/12，完整配对{len(paired)}/4任务。固定100轮，无test选模；不是论文历史成绩复现。','',
           '| Task | Arm | Test RMSE | Test Cliff RMSE | Alpha |','|---|---|---:|---:|---:|']
    for row in rows: lines.append(f'| {row["task"]} | {row["arm"]} | {row["rmse"]:.6f} | {row["cliff_rmse"]:.6f} | {row["alpha"]} |')
    lines.extend(['','完整配对任务的宏平均：','',json.dumps(means,indent=2),'','graph/final/shallow主干初始化一致；仅支路特征来源不同。单种子，不声明显著性；test已在前期研究中查看，不能称全新盲测。'])
    (DOC/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def queue():
    assert (OUT/'manifest.json').exists() and not (OUT/'status.json').exists()
    logs=OUT/'logs'; logs.mkdir(); completed=0
    r.save(OUT/'status.json',dict(status='running',completed=0,total=12))
    for task in TASKS:
        for arm in ARMS:
            r.save(OUT/'status.json',dict(status='running',completed=completed,total=12,task=task,arm=arm))
            with (logs/f'{task}_{arm}.log').open('w',encoding='utf-8') as stream:
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'job','--task',task,'--arm',arm],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if child.returncode:
                r.save(OUT/'status.json',dict(status='failed',completed=completed,total=12,task=task,arm=arm,exit_code=child.returncode)); report(); return
            completed+=1; report(); print('QUEUE',completed,'/12',task,arm,flush=True)
    r.save(OUT/'status.json',dict(status='complete',completed=12,total=12)); report()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('check','queue','job','report'))
    parser.add_argument('--task',choices=TASKS); parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.action=='job': job(args.task,args.arm)
    else: {'check':check,'queue':queue,'report':report}[args.action]()
