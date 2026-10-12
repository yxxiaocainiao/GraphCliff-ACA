"""Bounded validation-only screening of the local-signal/readout hypothesis."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/route_a'))
import pilot as p
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS, rdFingerprintGenerator

FULL_TRAIN = '--full-train' in sys.argv
OUT = ROOT/('docs/readout_full_train_20261011' if FULL_TRAIN else 'docs/readout_diagnostic_20261011')
TASKS = ('CHEMBL234_Ki','CHEMBL2047_EC50','CHEMBL2147_Ki','CHEMBL244_Ki')


def save(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def mapped(key):
    return 'head.'+key[9:] if key.startswith('reg_head.') else 'graph.'+key


def parity(graphs):
    batch = {k:v.cuda() for k,v in p.base.collate_graphs(graphs[:32]).items()}
    ref=p.base.model_new(); ref_rng=torch.cuda.get_rng_state().clone()
    wrapped=p.Predictor('graph')
    assert torch.equal(ref_rng,torch.cuda.get_rng_state())
    for key,value in ref.state_dict().items():
        assert torch.equal(value,wrapped.state_dict()[mapped(key)])
    optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95)) for m in (ref,wrapped)]
    records=[]
    for step in range(2):
        rng=torch.cuda.get_rng_state().clone(); outputs=[]; grads=[]
        for model,opt in zip((ref,wrapped),optimizers):
            torch.cuda.set_rng_state(rng); model.train(); opt.zero_grad(set_to_none=True)
            prediction=(model(batch['x'],batch['edge_index'],batch['edge_attr'],batch['batch'])
                        if model is ref else model(batch))
            torch.nn.functional.mse_loss(prediction,batch['y']).backward()
            outputs.append(prediction.detach())
            grads.append({k:None if v.grad is None else v.grad.detach().clone() for k,v in model.named_parameters()})
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        errors=[]
        for key,value in grads[0].items():
            other=grads[1][mapped(key)]
            assert (value is None)==(other is None)
            if value is not None:
                torch.testing.assert_close(value,other,atol=2e-6,rtol=2e-5)
                errors.append(float((value-other).abs().max()))
        torch.testing.assert_close(outputs[0],outputs[1],atol=2e-6,rtol=2e-5)
        weight_error=0.
        for key,value in ref.state_dict().items():
            other=wrapped.state_dict()[mapped(key)]
            torch.testing.assert_close(value,other,atol=2e-6,rtol=2e-5)
            weight_error=max(weight_error,float((value-other).abs().max()))
        records.append(dict(step=step+1,output_max_error=float((outputs[0]-outputs[1]).abs().max()),
                            gradient_max_error=max(errors),weight_max_error=weight_error))
    save('parity.json',dict(passed=True,steps=records,scope='same real batch, CUDA RNG, MSE, clipping, AdamW; not full training equivalence'))


def readout(x):
    return torch.cat((x.max(0).values,x.mean(0)))


def gap(x,y):
    return float((1-torch.nn.functional.cosine_similarity(x[None],y[None])).clamp(min=0))


def run():
    torch.set_num_threads(4); OUT.mkdir(exist_ok=True)
    manifest=[]; pair_rows=[]; totals=[]
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    for task in TASKS:
        tr,va,tg,vg=p.base.development(task)
        saved_split=json.loads((ROOT/'artifacts/route_a_30_20261011'/task/'split.json').read_text())
        assert va.row_id.tolist()==saved_split['valid'] and tr.row_id.tolist()==saved_split['train']
        if task==TASKS[0] and not FULL_TRAIN: parity(tg)
        checkpoint=ROOT/'artifacts/route_a_30_20261011'/task/'graph/best_model.pt'
        model=p.Predictor('graph')
        model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True); model.eval()
        if FULL_TRAIN:
            graph_by_row=dict(zip(tr.row_id.tolist()+va.row_id.tolist(),tg+vg))
            va=pd.concat([tr,va]).sort_values('row_id').reset_index(drop=True)
            vg=[graph_by_row[r] for r in va.row_id]
            membership=pd.read_csv(ROOT/'benchmark_data'/f'{task}.csv',usecols=['split'])
            assert va.row_id.tolist()==membership.index[membership.split.eq('train')].tolist()
        representations=[]; kept=[]; predictions=[]
        captured={}
        hooks=[model.graph.sagpool.register_forward_pre_hook(lambda module,args:captured.update(h=args[0])),
               model.graph.sagpool.register_forward_hook(lambda module,args,result:captured.update(pool=result)),
               model.head.register_forward_pre_hook(lambda module,args:captured.update(z=args[0]))]
        with torch.no_grad():
            for start in range(0,len(vg),128):
                graphs=vg[start:start+128]
                b={k:v.cuda() for k,v in p.base.collate_graphs(graphs).items()}
                pred=model(b); h=captured['h']; z=captured['z']; perm=captured['pool'][4]
                for local,graph in enumerate(graphs):
                    original_nodes=torch.where(b['batch']==local)[0]
                    selected=perm[b['batch'][perm]==local]
                    representations.append((readout(h[original_nodes]).cpu(),readout(h[selected]).cpu(),z[local].cpu()))
                    kept.append(set((selected-original_nodes[0]).cpu().tolist()))
                predictions.extend(pred.view(-1).cpu().tolist())
        for hook in hooks: hook.remove()
        replay_error=None
        if not FULL_TRAIN:
            previous=pd.read_csv(checkpoint.parent/'valid_predictions.csv')
            assert previous.row_id.tolist()==va.row_id.tolist()
            replay_error=float(np.max(np.abs(previous.prediction.to_numpy()-predictions)))
            assert replay_error<2e-5, replay_error
        mols=[Chem.MolFromSmiles(s) for s in va.smiles]
        assert all(m.GetNumAtoms()==g.x.size(0) for m,g in zip(mols,vg))
        fps=[fpgen.GetFingerprint(m) for m in mols]
        candidates=[]
        for i in range(len(mols)):
            for j,similarity in enumerate(DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])):
                if similarity>=.8: candidates.append((float(similarity),j,i))
        candidates.sort(key=lambda v:(-v[0],v[1],v[2]))
        skipped=dict(timeout=0,ambiguous=0,edit_size=0); accepted=0; attempted=0
        for similarity,i,j in candidates[:200 if FULL_TRAIN else 100]:
            if accepted>=(40 if FULL_TRAIN else 30): break
            attempted+=1
            result=rdFMCS.FindMCS([mols[i],mols[j]],timeout=1,ringMatchesRingOnly=True,
                                 completeRingsOnly=True,bondCompare=rdFMCS.BondCompare.CompareOrderExact)
            if result.canceled: skipped['timeout']+=1; continue
            query=Chem.MolFromSmarts(result.smartsString)
            matches=[mols[k].GetSubstructMatches(query,uniquify=True,maxMatches=2) for k in (i,j)]
            if any(len(m)!=1 for m in matches): skipped['ambiguous']+=1; continue
            changed=[set(range(mols[k].GetNumAtoms()))-set(m[0]) for k,m in zip((i,j),matches)]
            if any(not 1<=len(c)<=4 for c in changed): skipped['edit_size']+=1; continue
            accepted+=1; rates=[]; controls=[]
            for k,edited in zip((i,j),changed):
                for atom in sorted(edited):
                    degree=mols[k].GetAtomWithIdx(atom).GetDegree()
                    control=[a.GetIdx() for a in mols[k].GetAtoms() if a.GetIdx() not in edited and a.GetDegree()==degree]
                    if control:
                        rates.append(float(atom in kept[k]))
                        controls.append(sum(c in kept[k] for c in control)/len(control))
            delta=float(va.y.iloc[i]-va.y.iloc[j]); predicted=predictions[i]-predictions[j]
            pair_rows.append(dict(task=task,row_i=int(va.row_id.iloc[i]),row_j=int(va.row_id.iloc[j]),
                similarity=similarity,abs_label_delta=abs(delta),cliff=abs(delta)>=1,near_equal=abs(delta)<=.3,
                edited_i=sorted(changed[0]),edited_j=sorted(changed[1]),
                edited_retention=sum(len(c&kept[k]) for k,c in zip((i,j),changed))/sum(map(len,changed)),
                matched_edit_retention=float(np.mean(rates)) if rates else None,
                degree_matched_control=float(np.mean(controls)) if controls else None,
                pre_gap=gap(representations[i][0],representations[j][0]),
                selected_unweighted_gap=gap(representations[i][1],representations[j][1]),
                actual_post_gap=gap(representations[i][2],representations[j][2]),
                prediction_delta=predicted,pair_delta_abs_error=abs(predicted-delta)))
        totals.append(dict(task=task,partition='official_train' if FULL_TRAIN else 'validation',molecules=len(va),replay_max_error=replay_error,candidates=len(candidates),attempted=attempted,accepted=accepted,skipped=skipped))
        manifest.append(dict(task=task,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             data_sha256=hashlib.sha256((ROOT/'benchmark_data'/f'{task}.csv').read_bytes()).hexdigest()))
        save('pairs.json',pair_rows); save('coverage.json',totals)
        print(totals[-1],flush=True)
    save('provenance.json',dict(tasks=manifest,test_labels_read=False,partition='official_train' if FULL_TRAIN else 'validation',torch=torch.__version__,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pd.DataFrame(pair_rows).to_csv(OUT/'pairs.csv',index=False)
    summary=[]
    for group in ('all','cliff','near_equal'):
        subset=[r for r in pair_rows if group=='all' or r[group]]
        controlled=[r for r in subset if r['degree_matched_control'] is not None]
        summary.append(dict(group=group,pairs=len(subset),controlled_pairs=len(controlled),
          tasks=len({r['task'] for r in subset}),
          edited_retention=float(np.mean([r['edited_retention'] for r in subset])) if subset else None,
          retention_minus_control=float(np.mean([r['matched_edit_retention']-r['degree_matched_control'] for r in controlled])) if controlled else None,
          selected_gap_decreased=sum(r['selected_unweighted_gap']<r['pre_gap'] for r in subset),
          actual_gap_decreased=sum(r['actual_post_gap']<r['pre_gap'] for r in subset)))
    save('summary.json',summary); print(summary,flush=True)


if __name__=='__main__': run()
