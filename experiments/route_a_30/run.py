"""GPL-3.0: fixed 30-task overnight comparison using existing route-A utilities."""
import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/route_a'))
import pilot as p
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader

OUT = ROOT/'artifacts/route_a_30_20261011'
DOC = ROOT/'docs/route_a_30_20261011'
TASKS = re.findall(r'"(CHEMBL[^" ]+)"', (ROOT/'run.sh').read_text(encoding='utf-8'))
assert len(TASKS)==len(set(TASKS))==30
ARMS = ('graph','concat')


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def snapshot(path, value):
    tmp=path.with_suffix('.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(state,model,optimizer,generator):
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    generator.set_state(state['generator']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])


def check():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in TASKS:
        tr,va,_,_=p.data(task)
        membership=pd.read_csv(ROOT/'benchmark_data'/f'{task}.csv',usecols=['split'])
        assert set(membership.split)=={'train','test'}
        rows.append(dict(task=task,train=len(tr),valid=len(va),valid_cliff=int(va.cliff_mol.sum()),
                         test=int(membership.split.eq('test').sum()),data_sha256=digest(ROOT/'benchmark_data'/f'{task}.csv')))
    _,_,tg,_=p.data('CHEMBL234_Ki')
    batch=p.collate(tg[:128])
    timing={}
    for arm in ARMS:
        model=p.Predictor(arm)
        model.eval()
        initial=p.base.state_hash(model)
        reference=model(batch).detach()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
        model.train()
        torch.cuda.synchronize()
        start=time.monotonic()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(batch),batch['y'])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        timing[arm]=(time.monotonic()-start)/100
        # Serialize/restore a complete local state and verify eval predictions.
        snap=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state(),generator=torch.Generator().manual_seed(42).get_state(),
                  numpy_rng=np.random.get_state(),python_rng=random.getstate())
        snapshot(OUT/f'check_{arm}.pt',snap)
        loaded=torch.load(OUT/f'check_{arm}.pt',weights_only=False)
        clone=p.Predictor(arm)
        clone.load_state_dict(loaded['model'],strict=True)
        model.eval(); clone.eval()
        assert torch.allclose(model(batch),clone(batch),atol=1e-5,rtol=1e-5)
        replay=[]
        orders=[]
        for _ in range(2):
            state=torch.load(OUT/f'check_{arm}.pt',weights_only=False)
            resumed=p.Predictor(arm)
            opt=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
            gen=torch.Generator()
            restore(state,resumed,opt,gen)
            orders.append(torch.randperm(128,generator=gen))
            resumed.train(); opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(resumed(batch),batch['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(resumed.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            replay.append({k:v.detach().cpu().clone() for k,v in resumed.state_dict().items()})
            del resumed,opt,state
        assert torch.equal(orders[0],orders[1])
        assert all(torch.allclose(replay[0][k],replay[1][k],atol=1e-6,rtol=1e-5) for k in replay[0])
        if arm=='graph':
            original=p.base.model_new(); original.eval()
            assert p.base.state_hash(p.Predictor('graph'))==initial
            assert torch.equal(reference,original(batch['x'],batch['edge_index'],batch['edge_attr'],batch['batch']))
            del original
        print(f'CHECK {arm}: 100 updates passed, seconds/batch={timing[arm]:.4f}',flush=True)
        del model,clone,optimizer,snap,loaded,replay
        torch.cuda.empty_cache()
    sources=['model.py','dataset.py','dataset_utils.py','main.py','train.py','run.sh',
             'experiments/aca/run.py','experiments/route_a/pilot.py','experiments/route_a_30/run.py']
    upper_batches=sum(math.ceil(r['train']/128) for r in rows)*100
    estimate=upper_batches*sum(timing.values())
    manifest=dict(tasks=rows,total_train=sum(r['train'] for r in rows),seed=42,split_seed=42,
                  protocol=(ROOT/'experiments/route_a_30/task_plan.md').read_text(encoding='utf-8'),
                  source_sha256={n:digest(ROOT/n) for n in sources},timing_seconds_per_batch=timing,
                  estimated_training_upper_seconds=estimate,preflight='passed',torch=torch.__version__,
                  cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    save(OUT/'manifest.json',manifest)
    print(json.dumps(dict(total_train=manifest['total_train'],estimated_training_upper_hours=estimate/3600)),flush=True)


def final_test(task,model):
    # Called only after training/checkpoint selection has finished for this arm.
    path=ROOT/'benchmark_data'/f'{task}.csv'
    membership=pd.read_csv(path,usecols=['split'])
    ids=np.flatnonzero(membership.split.eq('test').to_numpy())
    keep=set((ids+1).tolist())
    rows=pd.read_csv(path,skiprows=lambda line:line>0 and line not in keep,
                     usecols=['smiles','y','cliff_mol','split'])
    assert len(rows)==len(ids) and rows.split.eq('test').all() and rows.cliff_mol.sum()>0
    graphs=[]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    for smiles,y in zip(rows.smiles,rows.y):
        mol=Chem.MolFromSmiles(smiles); assert mol is not None
        graph=p.base.smiles_to_graph(smiles,float(y))
        bits=np.zeros(2048,dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),bits)
        graph.fp=torch.tensor(bits,dtype=torch.float32)
        graphs.append(graph)
    loader=DataLoader(graphs,batch_size=128,collate_fn=p.collate)
    labels,predictions=p.evaluate(model,loader)
    frame=pd.DataFrame(dict(row_id=ids,label=labels,prediction=predictions,cliff=rows.cliff_mol))
    metrics=p.base.metrics(labels,predictions,rows.cliff_mol)
    for key,value in metrics.items():
        errors=[(float(y)-float(v))**2 for y,v,c in zip(labels,predictions,rows.cliff_mol) if key=='rmse' or c]
        assert abs(math.sqrt(math.fsum(errors)/len(errors))-value)<1e-10
    return frame,metrics


def job(task,arm):
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    info=next(r for r in manifest['tasks'] if r['task']==task)
    assert digest(ROOT/'benchmark_data'/f'{task}.csv')==info['data_sha256']
    folder=OUT/task/arm
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'summary.json').exists(), 'completed fit must not be overwritten'
    tr,va,tg,vg=p.data(task)
    save(OUT/task/'split.json',dict(train=tr.row_id.tolist(),valid=va.row_id.tolist(),official_test_labels_used_for_selection=False))
    model=p.Predictor(arm)
    initial=p.base.state_hash(model)
    shared=p.base.state_hash(model.graph)
    generator=torch.Generator().manual_seed(42)
    train_loader=DataLoader(tg,batch_size=128,shuffle=True,collate_fn=p.collate,generator=generator)
    valid_loader=DataLoader(vg,batch_size=128,collate_fn=p.collate,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
    scheduler=p.base.WarmupCosineScheduler(optimizer,10,100,1e-4)
    best,stale,start_epoch,history,best_state,best_epoch=float('inf'),0,1,[],None,None
    resumed=False
    if (folder/'latest.pt').exists():
        state=torch.load(folder/'latest.pt',weights_only=False)
        assert state['initial_sha256']==initial and state['manifest_sha256']==digest(OUT/'manifest.json')
        restore(state,model,optimizer,generator)
        best,stale,start_epoch=state['best'],state['stale'],state['epoch']+1
        history,best_state,best_epoch=state['history'],state['best_state'],state['best_epoch']
        snapshot(folder/'best_model.pt',best_state)
        resumed=True
        print(f'RESUME {task} {arm} from epoch={start_epoch}',flush=True)
    started=time.monotonic()
    for epoch in range(start_epoch,101):
        if stale>=15:
            break
        model.train(); total,count=0.,0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(batch),batch['y'])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=float(loss.detach())*len(batch['y']); count+=len(batch['y'])
        assert count==len(tr)
        labels,predictions=p.evaluate(model,valid_loader)
        score=p.base.metrics(labels,predictions,va.cliff_mol)
        value=float(np.mean((np.asarray(labels)-predictions)**2)); assert np.isfinite(value)
        if best-value>1e-4:
            best,stale,best_epoch=value,0,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            snapshot(folder/'best_model.pt',best_state)
        else:
            stale+=1
        current_lr=scheduler.step(epoch)
        history.append(dict(epoch=epoch,train_mse=total/count,valid_mse=value,lr_next=current_lr,**score))
        save(folder/'history.json',history)
        save(folder/'progress.json',dict(task=task,arm=arm,epoch=epoch,best_epoch=best_epoch,stale=stale,phase='training'))
        if epoch%5==0 or stale>=15 or epoch==100:
            snapshot(folder/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                     cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),generator=generator.get_state(),
                     numpy_rng=np.random.get_state(),python_rng=random.getstate(),best=best,stale=stale,
                     epoch=epoch,history=history,best_state=best_state,best_epoch=best_epoch,
                     initial_sha256=initial,manifest_sha256=digest(OUT/'manifest.json')))
        if epoch==1 or epoch%5==0:
            print(f'{task} {arm} epoch={epoch} rmse={score["rmse"]:.6f} cliff={score["cliff_rmse"]:.6f} stale={stale}',flush=True)
    model.load_state_dict(torch.load(folder/'best_model.pt',weights_only=True),strict=True)
    labels,predictions=p.evaluate(model,valid_loader)
    validation=p.base.metrics(labels,predictions,va.cliff_mol)
    expected=next(r for r in history if r['epoch']==best_epoch)
    assert abs(validation['rmse']-expected['rmse'])<1e-5
    assert abs(validation['cliff_rmse']-expected['cliff_rmse'])<1e-5
    pd.DataFrame(dict(row_id=va.row_id,label=labels,prediction=predictions,cliff=va.cliff_mol)).to_csv(folder/'valid_predictions.csv',index=False)
    save(folder/'progress.json',dict(task=task,arm=arm,phase='final_test',epoch=len(history),best_epoch=best_epoch))
    frame,metrics=final_test(task,model)
    frame.to_csv(folder/'test_predictions.csv',index=False)
    result=dict(task=task,arm=arm,epochs=len(history),best_epoch=best_epoch,initial_sha256=initial,
                graph_initial_sha256=shared,parameters=sum(v.numel() for v in model.parameters()),
                validation=validation,test=metrics,test_count=len(frame),test_cliff=int(frame.cliff.sum()),
                resumed=resumed,attempt_seconds=time.monotonic()-started)
    save(folder/'summary.json',result)
    print('FINISHED '+json.dumps(result),flush=True)


def report():
    DOC.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in TASKS:
        pair=[]
        for arm in ARMS:
            folder=OUT/task/arm
            if not (folder/'summary.json').exists():
                continue
            result=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            frame=pd.read_csv(folder/'test_predictions.csv')
            for key,value in result['test'].items():
                errors=[(float(y)-float(v))**2 for y,v,c in zip(frame.label,frame.prediction,frame.cliff) if key=='rmse' or c]
                assert abs(math.sqrt(math.fsum(errors)/len(errors))-value)<1e-6
            rows.append(dict(task=task,arm=arm,rmse=result['test']['rmse'],cliff_rmse=result['test']['cliff_rmse'],
                             best_epoch=result['best_epoch'],epochs=result['epochs'],resumed=result['resumed']))
            pair.append(result)
            for name in ('history.json','summary.json','valid_predictions.csv','test_predictions.csv'):
                (DOC/f'{task}_{arm}_{name}').write_bytes((folder/name).read_bytes())
        if len(pair)==2:
            assert pair[0]['graph_initial_sha256']==pair[1]['graph_initial_sha256']
    pd.DataFrame(rows).to_csv(DOC/'summary.csv',index=False)
    save(DOC/'manifest.json',json.loads((OUT/'manifest.json').read_text(encoding='utf-8')))
    lines=['# GraphCliff vs concat：30任务官方test比较','',f'完成模型数：{len(rows)}/60。seed42、固定两组、验证选模，结果为官方test。','',
           '| Task | graph RMSE | concat RMSE | graph Cliff | concat Cliff |','|---|---:|---:|---:|---:|']
    paired=[]
    for task in TASKS:
        arms={r['arm']:r for r in rows if r['task']==task}
        if len(arms)!=2:
            continue
        g,c=arms['graph'],arms['concat']; paired.append((g,c))
        lines.append(f'| {task} | {g["rmse"]:.6f} | {c["rmse"]:.6f} | {g["cliff_rmse"]:.6f} | {c["cliff_rmse"]:.6f} |')
    aggregate=dict(completed_models=len(rows),paired_tasks=len(paired))
    if paired:
        for key in ('rmse','cliff_rmse'):
            aggregate[key]=dict(graph=float(np.mean([g[key] for g,c in paired])),
                                concat=float(np.mean([c[key] for g,c in paired])),
                                wins=sum(c[key]<g[key]-1e-6 for g,c in paired),
                                ties=sum(abs(c[key]-g[key])<=1e-6 for g,c in paired),
                                losses=sum(c[key]>g[key]+1e-6 for g,c in paired))
        lines.extend(['','任务宏平均（仅完整配对任务，未完成时不能称30任务平均）：','',json.dumps(aggregate,ensure_ascii=False,indent=2)])
    lines.extend(['','单种子，不声明统计显著；当前训练协议未声称精确复现论文历史训练。不得按test结果逐任务挑分支。'])
    save(DOC/'aggregate.json',aggregate)
    (DOC/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return aggregate


def queue():
    assert (OUT/'manifest.json').exists()
    logs=OUT/'logs'; logs.mkdir(exist_ok=True)
    assert not (OUT/'status.json').exists(), 'queue already launched; inspect status instead of duplicating'
    retries,completed=0,0
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    save(OUT/'status.json',dict(status='running',completed=0,total=60,started=time.time()))
    try:
        for task in TASKS:
            for arm in ARMS:
                for attempt in range(2):
                    save(OUT/'status.json',dict(status='running',completed=completed,total=60,task=task,arm=arm,attempt=attempt+1,retries=retries))
                    logfile=logs/f'{task}_{arm}_attempt{attempt+1}.log'
                    with logfile.open('w',encoding='utf-8') as stream:
                        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'job','--task',task,'--arm',arm],
                                              cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode==0:
                        completed+=1
                        report()
                        break
                    error=logfile.read_text(encoding='utf-8')
                    is_cuda='CUDA error' in error or 'CUBLAS_STATUS' in error
                    if not is_cuda or attempt==1 or retries>=3:
                        save(OUT/'status.json',dict(status='failed',completed=completed,total=60,task=task,arm=arm,
                                                   log=str(logfile),retries=retries,exit_code=result.returncode))
                        report()
                        return
                    retries+=1
                    print(f'CUDA recovery {retries}/3 for {task}/{arm}; same protocol, fresh process',flush=True)
        save(OUT/'status.json',dict(status='complete',completed=completed,total=60,retries=retries,finished=time.time()))
        report()
    except BaseException as error:
        save(OUT/'status.json',dict(status='failed',completed=completed,total=60,error=str(error),retries=retries))
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','queue','job','report'))
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='job':
        assert args.task and args.arm
        job(args.task,args.arm)
    else:
        {'check':check,'queue':queue,'report':report}[args.action]()
