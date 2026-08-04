"""Scan all test-test cliff pairs (Layer-2 gates) and rank by how cleanly the
SMALLEST-gate atoms coincide with the non-MCS (cliff-defining) atoms, so the
heatmap visibly localizes on the structural difference.

Helper for picking the pair drawn by analysis/gate_heatmap.py. Requires the
gate/MCS caches, i.e. run analysis/topk_gate_alignment.py --mode save first.

Usage (run from the repository root):
    python analysis/find_cliff_pair.py
"""
import os, glob, numpy as np, torch

GATE_DIR, MCS_DIR = 'analysis_out/gates', 'analysis_out/mcs'

def mean_gate(g):  # g: [N,256] tensor
    return g.mean(dim=-1).numpy()

cands = []
for gp in sorted(glob.glob(f'{GATE_DIR}/*_gates.pt')):
    ds = os.path.basename(gp).replace('_gates.pt','')
    mp = f'{MCS_DIR}/{ds}_mcs.pt'
    if not os.path.exists(mp):
        continue
    gates = torch.load(gp, weights_only=False)
    mcs   = torch.load(mp, weights_only=False)
    for (s1, s2), (mod1, sh1, mod2, sh2) in mcs.items():
        if s1 not in gates or s2 not in gates:
            continue
        if gates[s1]['split'] != 'test' or gates[s2]['split'] != 'test':
            continue
        if not mod1 or not mod2:
            continue
        ok = True
        per = []
        for smi, mod in [(s1, mod1), (s2, mod2)]:
            g = mean_gate(gates[smi]['gate']); N = len(g)
            if max(mod) >= N:  # index mismatch guard
                ok = False; break
            order = list(np.argsort(g))                 # ascending: smallest first
            rank = {a: i for i, a in enumerate(order)}
            # fraction of non-MCS atoms that are in the bottom-|mod| smallest gates
            kk = len(mod)
            bottomk = set(order[:max(kk, 5)])
            frac_in_bottom = len(set(mod) & set(order[:kk])) / kk
            mean_pct = np.mean([rank[a] / (N - 1 + 1e-9) for a in mod])  # 0=all smallest
            per.append((N, kk, frac_in_bottom, mean_pct,
                        len(set(mod) & set(order[:1])) > 0,
                        len(set(mod) & set(order[:3])) > 0))
        if not ok:
            continue
        (N1,k1,f1,p1,h1a,h1b), (N2,k2,f2,p2,h2a,h2b) = per
        # clarity: small clean difference + non-MCS sits at the very lowest gate
        size = k1 + k2
        meanpct = (p1 + p2) / 2          # lower better
        fracbot = (f1 + f2) / 2          # higher better
        top3 = h1b and h2b
        # require a clean, smallish difference on both sides
        if not (1 <= k1 <= 7 and 1 <= k2 <= 7):
            continue
        score = fracbot - meanpct        # high frac in bottom-k AND low percentile
        cands.append((score, fracbot, round(meanpct,3), size, top3, ds, s1, s2, N1, N2, k1, k2))

cands.sort(reverse=True)
print(f'{"score":>6} {"fbot":>5} {"mpct":>5} {"sz":>3} top3  dataset')
for c in cands[:25]:
    score, fbot, mpct, size, top3, ds, s1, s2, N1, N2, k1, k2 = c
    print(f'{score:6.2f} {fbot:5.2f} {mpct:5.2f} {size:3d} {str(top3):5s} {ds}  N=({N1},{N2}) diff=({k1},{k2})')
    print(f'        L: {s1}')
    print(f'        R: {s2}')
