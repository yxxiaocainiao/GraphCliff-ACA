"""
Top-k Gate Alignment Analysis for GraphCliff.

Cliff pairs are categorized by their split membership:
  - train-train : both molecules in train set
  - train-test  : one in train, one in test (cross-split)
  - test-test   : both molecules in test set

Phase 1 (--mode save):
  Extract Layer-2 gate weights for ALL cliff_mol=1 molecules (train + test).
  Saved as analysis_out/gates/{dataset}_gates.pt
      ->  {smiles: {'gate': Tensor[N,256], 'split': str}}

Phase 2 (--mode align):
  Find cliff pairs using ActivityCliffs (cliffs.py) on ALL cliff molecules.
  Categorize pairs, compute Top-k alignment per category, save CSV.

Reproduces the "Gate" columns of Table 5 (Top-k gate alignment) in the paper.

Usage (run from the repository root):
    python -u analysis/topk_gate_alignment.py --mode all --k 1 3 5
    python -u analysis/topk_gate_alignment.py --mode all --ckpt_dir ckpt
"""

import os
import glob
import argparse
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from torch.utils.data import Dataset

# Backport nn.RMSNorm for PyTorch < 2.4
if not hasattr(torch.nn, 'RMSNorm'):
    class _RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps
        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return self.weight * x / rms
    torch.nn.RMSNorm = _RMSNorm

from model import GraphCliffRegressor, normalize_edges
from dataset_utils import smiles_to_graph
from cliffs import ActivityCliffs

RDLogger.DisableLog('rdApp.*')

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
GROUPS        = 4
MID_K         = 3
EDGE_DIM      = 13
ATOM_IN_DIM   = 38
CKPT_BASE     = 'ckpt'                       # overridden by --ckpt_dir
GATE_DIR      = 'analysis_out/gates'
MCS_DIR       = 'analysis_out/mcs'
CLIFF_TYPE_DIR = 'analysis_out/cliff_type_cache'
TARGET_LAYER  = 1   # 0-indexed; Layer 2 has the highest gate selectivity
SPLIT_TYPES   = ['train-train', 'train-test', 'test-test']
CLIFF_TYPES   = ['tanimoto', 'scaffold', 'levenshtein', 'any']


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(dataset_name):
    pattern = os.path.join(CKPT_BASE, dataset_name, '*', 'best_model.pt')
    paths = sorted(glob.glob(pattern))
    assert paths, f"No checkpoint found for {dataset_name}"
    model = GraphCliffRegressor(
        atom_in_dim=ATOM_IN_DIM, edge_dim=EDGE_DIM,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        groups=GROUPS, mid_K=MID_K,
    ).to(DEVICE)
    ckpt = torch.load(paths[0], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ── Gate extraction ────────────────────────────────────────────────────────────

def forward_with_gates(model, x, edge_index, edge_attr):
    """Returns gate tensors [N, 256] for each layer."""
    h = model.atom_encoder(x)
    layer_gates = []
    for layer in model.encoder.layers:
        h_norm = layer.pre_norm(h)
        z      = layer.proj(h_norm)
        z      = layer.short(z, edge_index, edge_attr)
        x2, x1, v = torch.chunk(z, 3, dim=-1)
        gate = torch.sigmoid(x1)
        layer_gates.append(gate.detach().cpu())

        if edge_index.numel() > 0:
            ew        = torch.ones(edge_index.size(1), device=DEVICE, dtype=x2.dtype)
            edge_norm = normalize_edges(h.size(0), edge_index, ew)
        else:
            edge_norm = torch.tensor([], device=DEVICE, dtype=h.dtype)
        mid_out = layer.long(x2, edge_index, edge_norm)
        y = mid_out * gate + v
        h = y + h
    return layer_gates  # list of 3 × [N, 256]


# ── Phase 1: Save gates ────────────────────────────────────────────────────────

def save_gates(dataset_name):
    os.makedirs(GATE_DIR, exist_ok=True)
    out_path = os.path.join(GATE_DIR, f'{dataset_name}_gates.pt')

    df      = pd.read_csv(f'./benchmark_data/{dataset_name}.csv')
    cliff_df = df[df['cliff_mol'] == 1].reset_index(drop=True)

    if cliff_df.empty:
        print(f'  [{dataset_name}] no cliff_mol molecules, skip')
        return

    model  = load_model(dataset_name)
    gates  = {}   # smiles -> {'gate': Tensor[N,256], 'split': str}
    errors = 0

    for _, row in cliff_df.iterrows():
        smiles = row['smiles']
        split  = row['split']
        try:
            g = smiles_to_graph(smiles, y=0.0)
        except Exception:
            errors += 1
            continue

        x          = g.x.to(DEVICE)
        edge_index = g.edge_index.to(DEVICE)
        edge_attr  = g.edge_attr.to(DEVICE)

        with torch.no_grad():
            layer_gates = forward_with_gates(model, x, edge_index, edge_attr)

        gates[smiles] = {
            'gate' : layer_gates[TARGET_LAYER],   # [N, 256] CPU tensor
            'split': split,
        }

    torch.save(gates, out_path)
    print(f'  [{dataset_name}] saved {len(gates)} cliff molecules '
          f'(skip {errors}) -> {out_path}')


# ── Phase 2: Alignment ─────────────────────────────────────────────────────────

def compute_mcs_atoms(mol1, mol2):
    """Returns (mod1, shared1, mod2, shared2) atom index sets."""
    mcs = rdFMCS.FindMCS(
        [mol1, mol2],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=5,
    )
    all1, all2 = set(range(mol1.GetNumAtoms())), set(range(mol2.GetNumAtoms()))
    if mcs.numAtoms == 0:
        return all1, set(), all2, set()
    mcs_mol  = Chem.MolFromSmarts(mcs.smartsString)
    shared1  = set(mol1.GetSubstructMatch(mcs_mol))
    shared2  = set(mol2.GetSubstructMatch(mcs_mol))
    return all1 - shared1, shared1, all2 - shared2, shared2


def load_or_build_mcs(dataset_name, smiles_list, pairs):
    """Load cached MCS results or compute and save them.
    Cache: mcs/{dataset}_mcs.pt -> {(smi1,smi2): (mod1,shared1,mod2,shared2)}
    """
    os.makedirs(MCS_DIR, exist_ok=True)
    cache_path = os.path.join(MCS_DIR, f'{dataset_name}_mcs.pt')

    if os.path.exists(cache_path):
        print(f'  [{dataset_name}] loading MCS cache...')
        return torch.load(cache_path, weights_only=False)

    print(f'  [{dataset_name}] computing MCS for {len(pairs)} pairs...')
    cache = {}
    for (i, j) in pairs:
        smi1, smi2 = smiles_list[i], smiles_list[j]
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            continue
        try:
            cache[(smi1, smi2)] = compute_mcs_atoms(mol1, mol2)
        except Exception:
            continue
    torch.save(cache, cache_path)
    print(f'  [{dataset_name}] MCS saved -> {cache_path}')
    return cache


def node_importance(gate):
    """Scalar importance per atom: mean across 256-dim gate. [N]"""
    return gate.mean(dim=-1)


def topk_hit(importance, mod_atoms, k, bottom=False):
    """1 if any modification atom in top-k (or bottom-k), else 0."""
    if not mod_atoms:
        return None
    k_eff = min(k, importance.size(0))
    if bottom:
        idx = set(importance.topk(k_eff, largest=False).indices.tolist())
    else:
        idx = set(importance.topk(k_eff).indices.tolist())
    return 1 if idx & mod_atoms else 0


def pair_split_type(split_i, split_j):
    if split_i == 'train' and split_j == 'train':
        return 'train-train'
    elif split_i == 'test' and split_j == 'test':
        return 'test-test'
    else:
        return 'train-test'


def compute_alignment(dataset_name, k_values, by_cliff_type=False):
    gate_path = os.path.join(GATE_DIR, f'{dataset_name}_gates.pt')
    if not os.path.exists(gate_path):
        print(f'  [{dataset_name}] gates not found, run --mode save first')
        return None

    gates = torch.load(gate_path)   # smiles -> {'gate': [N,256], 'split': str}

    df       = pd.read_csv(f'./benchmark_data/{dataset_name}.csv')
    cliff_df = df[df['cliff_mol'] == 1].reset_index(drop=True)
    if cliff_df.empty:
        return None

    smiles_list = cliff_df['smiles'].tolist()
    activity    = cliff_df['exp_mean [nM]'].tolist()

    if len(smiles_list) < 2:
        return None

    # Find all cliff pairs via ActivityCliffs
    ac           = ActivityCliffs(smiles_list, activity)
    cliff_matrix = ac.find_cliffs(similarity=0.9, potency_fold=10)
    pairs        = list(zip(*np.where(np.triu(cliff_matrix, k=1) == 1)))

    if not pairs:
        print(f'  [{dataset_name}] no cliff pairs found')
        return None

    # Load or build MCS cache
    mcs_cache = load_or_build_mcs(dataset_name, smiles_list, pairs)

    # Optional cliff-type cache
    cliff_types = None
    if by_cliff_type:
        ct_path = os.path.join(CLIFF_TYPE_DIR, f'{dataset_name}_cliff_types.pt')
        if os.path.exists(ct_path):
            cliff_types = torch.load(ct_path, weights_only=False)
        else:
            print(f'  [{dataset_name}] cliff_type cache not found, falling back to split_type only')

    # Stratification keys: (split_type,) or (split_type, cliff_type)
    if by_cliff_type and cliff_types is not None:
        all_keys = [(st, ct) for st in SPLIT_TYPES for ct in CLIFF_TYPES]
    else:
        all_keys = [(st,) for st in SPLIT_TYPES]

    hits        = {key: {k: [] for k in k_values} for key in all_keys}
    hits_bottom = {key: {k: [] for k in k_values} for key in all_keys}
    n_pairs     = {key: 0 for key in all_keys}

    for (i, j) in pairs:
        smi1, smi2 = smiles_list[i], smiles_list[j]
        if smi1 not in gates or smi2 not in gates:
            continue
        if (smi1, smi2) not in mcs_cache:
            continue

        split_i = gates[smi1]['split']
        split_j = gates[smi2]['split']
        st      = pair_split_type(split_i, split_j)

        # Determine which keys this pair contributes to
        if by_cliff_type and cliff_types is not None:
            types = cliff_types.get((smi1, smi2), set())
            keys  = [(st, ct) for ct in types] + ([(st, 'any')] if types else [])
        else:
            keys = [(st,)]

        if not keys:
            continue

        for key in keys:
            n_pairs[key] += 1

        mod1, _, mod2, _ = mcs_cache[(smi1, smi2)]
        imp1 = node_importance(gates[smi1]['gate'])
        imp2 = node_importance(gates[smi2]['gate'])

        for k in k_values:
            for imp, mod in [(imp1, mod1), (imp2, mod2)]:
                hit = topk_hit(imp, mod, k, bottom=False)
                hit_bot = topk_hit(imp, mod, k, bottom=True)
                if hit is not None:
                    for key in keys:
                        hits[key][k].append(hit)
                if hit_bot is not None:
                    for key in keys:
                        hits_bottom[key][k].append(hit_bot)

    result = {}
    for key in all_keys:
        if hits[key][k_values[0]]:
            result[key] = {
                'n_pairs'       : n_pairs[key],
                'scores'        : {k: np.mean(hits[key][k])        for k in k_values},
                'scores_bottom' : {k: np.mean(hits_bottom[key][k]) for k in k_values},
            }

    n_total = sum(n_pairs.get((st,), n_pairs.get((st, 'any'), 0)) for st in SPLIT_TYPES)
    print(f'  [{dataset_name}] total {n_total} pairs  '
          + '  '.join(f'{st}: {n_pairs.get((st,), n_pairs.get((st,"any"),0))}' for st in SPLIT_TYPES))
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def get_datasets():
    return sorted([
        d for d in os.listdir(CKPT_BASE)
        if os.path.isdir(os.path.join(CKPT_BASE, d))
    ])


def main():
    global CKPT_BASE

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',          choices=['save', 'align', 'all'], default='all')
    parser.add_argument('--k',             nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--datasets',      nargs='+', default=None)
    parser.add_argument('--by_cliff_type', action='store_true',
                        help='Stratify by cliff type (tanimoto/scaffold/levenshtein). '
                             'Requires analysis_out/cliff_type_cache/.')
    parser.add_argument('--ckpt_dir', default=CKPT_BASE,
                        help='Directory holding <dataset>/<timestamp>/best_model.pt (default: ckpt)')
    args = parser.parse_args()

    CKPT_BASE = args.ckpt_dir

    datasets = args.datasets or get_datasets()

    # ── Phase 1 ──
    if args.mode in ('save', 'all'):
        print('=== Phase 1: Saving gate weights (all cliff molecules) ===')
        for ds in datasets:
            print(f'Processing {ds}...')
            try:
                save_gates(ds)
            except Exception as e:
                print(f'  [{ds}] ERROR: {e}')

    # ── Phase 2 ──
    if args.mode in ('align', 'all'):
        print('\n=== Phase 2: Computing Top-k Alignment ===')
        all_results = {}
        for ds in datasets:
            print(f'Processing {ds}...')
            try:
                r = compute_alignment(ds, args.k, by_cliff_type=args.by_cliff_type)
                if r:
                    all_results[ds] = r
            except Exception as e:
                print(f'  [{ds}] ERROR: {e}')

        if not all_results:
            print('No results.')
            return

        k_vals = args.k

        # Collect all unique stratification keys
        all_keys = sorted({key for r in all_results.values() for key in r.keys()})

        # Group by split_type (first element)
        from itertools import groupby
        split_groups = {}
        for key in all_keys:
            st = key[0]
            split_groups.setdefault(st, []).append(key)

        print()
        for st in SPLIT_TYPES:
            for key in split_groups.get(st, []):
                label = st if len(key) == 1 else f'{st} | cliff_type={key[1]}'
                entries = [(ds, r[key]) for ds, r in sorted(all_results.items()) if key in r]
                if not entries:
                    continue
                print(f'\n--- {label} ---')
                print(f"{'Dataset':<30} {'#Pairs':>7}  " +
                      '  '.join(f'Top-{k}' for k in k_vals) + '  ' +
                      '  '.join(f'Bot-{k}' for k in k_vals))
                print('-' * 80)
                macro     = {k: [] for k in k_vals}
                macro_bot = {k: [] for k in k_vals}
                for ds, v in entries:
                    sc  = v['scores']
                    scb = v['scores_bottom']
                    row = f"{ds:<30} {v['n_pairs']:>7}  " + \
                          '  '.join(f"{sc[k]*100:>6.1f}%" for k in k_vals) + '  ' + \
                          '  '.join(f"{scb[k]*100:>6.1f}%" for k in k_vals)
                    print(row)
                    for k in k_vals:
                        macro[k].append(sc[k])
                        macro_bot[k].append(scb[k])
                print('-' * 80)
                for k in k_vals:
                    if macro[k]:
                        print(f'Macro avg Top-{k}: {np.mean(macro[k])*100:.1f}%  '
                              f'Bot-{k}: {np.mean(macro_bot[k])*100:.1f}%  '
                              f'(n={len(macro[k])} datasets)')

        # ── Save CSV ──
        rows = []
        for ds, r in sorted(all_results.items()):
            for key, v in r.items():
                row = {'dataset': ds, 'split_type': key[0]}
                if len(key) == 2:
                    row['cliff_type'] = key[1]
                row['n_cliff_pairs'] = v['n_pairs']
                for k in k_vals:
                    row[f'top{k}_alignment']    = round(v['scores'][k], 4)
                    row[f'bot{k}_alignment']    = round(v['scores_bottom'][k], 4)
                rows.append(row)

        os.makedirs('analysis_out', exist_ok=True)
        out_csv = ('analysis_out/topk_alignment_by_clifftype.csv' if args.by_cliff_type
                   else 'analysis_out/topk_alignment_results.csv')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'\nSaved -> {out_csv}')


if __name__ == '__main__':
    main()