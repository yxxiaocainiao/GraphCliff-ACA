"""
Random baseline for top-k / bottom-k alignment.

For each cliff pair molecule, the baseline hit rate is:
  P(at least 1 of k randomly selected atoms falls in modification set)
  = 1 - C(N - mod_size, k) / C(N, k)    [hypergeometric]

This is computed per molecule from actual N and mod_size in MCS cache,
then macro-averaged across datasets, stratified by split type.

Reproduces the "Random" columns of Table 5 (Top-k gate alignment) in the paper.
Requires analysis/topk_gate_alignment.py --mode save to have been run first,
so that analysis_out/gates/ and analysis_out/mcs/ exist.

Usage (run from the repository root):
    python -u analysis/random_baseline.py
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

GATE_DIR       = 'analysis_out/gates'
MCS_DIR        = 'analysis_out/mcs'
CLIFF_TYPE_DIR = 'analysis_out/cliff_type_cache'
K_VALUES       = [1, 3, 5]
CLIFF_TYPES    = ['tanimoto', 'scaffold', 'levenshtein', 'any']
SPLIT_TYPES    = ['train-train', 'train-test', 'test-test']


def hypergeometric_hit(N, mod_size, k):
    """P(at least 1 hit) when drawing k from N, mod_size are 'successes'."""
    if mod_size == 0 or N == 0:
        return 0.0
    k_eff = min(k, N)
    # P(0 hits) = C(N - mod_size, k) / C(N, k)
    no_hit_num = math.comb(max(N - mod_size, 0), k_eff)
    no_hit_den = math.comb(N, k_eff)
    if no_hit_den == 0:
        return 0.0
    return 1.0 - no_hit_num / no_hit_den


def pair_split_type(s1, s2):
    if s1 == 'train' and s2 == 'train': return 'train-train'
    if s1 == 'test'  and s2 == 'test':  return 'test-test'
    return 'train-test'


def analyze_dataset(dataset_name, gates, mcs_cache, cliff_types):
    baselines = defaultdict(lambda: {k: [] for k in K_VALUES})  # key -> k -> [hit_prob]

    for (smi1, smi2), (mod1, shared1, mod2, shared2) in mcs_cache.items():
        if smi1 not in gates or smi2 not in gates:
            continue

        st    = pair_split_type(gates[smi1]['split'], gates[smi2]['split'])
        types = cliff_types.get((smi1, smi2), set()) if cliff_types else set()
        keys  = [(st, ct) for ct in types] + ([(st, 'any')] if types else [])
        if not keys:
            continue

        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            continue

        for mol, mod in [(mol1, mod1), (mol2, mod2)]:
            N        = mol.GetNumHeavyAtoms()
            mod_size = len(mod)
            if mod_size == 0:
                continue
            for k in K_VALUES:
                p = hypergeometric_hit(N, mod_size, k)
                for key in keys:
                    baselines[key][k].append(p)

    result = {}
    for key, kdict in baselines.items():
        if kdict[K_VALUES[0]]:
            result[key] = {k: np.mean(v) for k, v in kdict.items()}
    return result


def main():
    datasets = sorted(
        os.path.basename(p)[:-len('_gates.pt')]
        for p in os.listdir(GATE_DIR) if p.endswith('_gates.pt')
    ) if os.path.isdir(GATE_DIR) else []
    if not datasets:
        raise SystemExit(
            f'No gate caches found in {GATE_DIR}/. '
            'Run: python -u analysis/topk_gate_alignment.py --mode save'
        )

    # Accumulate per (split_type, cliff_type) key
    all_baselines = defaultdict(lambda: {k: [] for k in K_VALUES})

    for ds in datasets:
        gate_path = os.path.join(GATE_DIR, f'{ds}_gates.pt')
        mcs_path  = os.path.join(MCS_DIR, f'{ds}_mcs.pt')
        ct_path   = os.path.join(CLIFF_TYPE_DIR, f'{ds}_cliff_types.pt')

        if not os.path.exists(gate_path) or not os.path.exists(mcs_path):
            continue

        gates     = torch.load(gate_path,  weights_only=False)
        mcs_cache = torch.load(mcs_path,   weights_only=False)
        cliff_types = torch.load(ct_path,  weights_only=False) if os.path.exists(ct_path) else None

        print(f'Processing {ds}...')
        r = analyze_dataset(ds, gates, mcs_cache, cliff_types)
        for key, kdict in r.items():
            for k, v in kdict.items():
                all_baselines[key][k].append(v)

    # Print summary
    print(f'\n{"":40} ' + '  '.join(f'Rnd-{k}' for k in K_VALUES))
    print('-' * 70)

    rows = []
    for st in SPLIT_TYPES:
        for ct in CLIFF_TYPES:
            key = (st, ct)
            if key not in all_baselines or not all_baselines[key][K_VALUES[0]]:
                continue
            label = f'{st} | {ct}'
            vals  = {k: np.mean(all_baselines[key][k]) for k in K_VALUES}
            print(f'{label:<40} ' + '  '.join(f'{vals[k]*100:>6.1f}%' for k in K_VALUES))
            rows.append({'split_type': st, 'cliff_type': ct,
                         **{f'random_baseline_top{k}': round(vals[k], 4) for k in K_VALUES}})

    os.makedirs('analysis_out', exist_ok=True)
    out_csv = 'analysis_out/random_baseline.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\nSaved -> {out_csv}')


if __name__ == '__main__':
    main()
