"""
Per-atom gate heatmap for a single activity cliff pair (paper Figure 6).

Draws a 2x2 panel using the SMALLEST mean-gate criterion (low gate =
local-bypass-dominant = cliff-defining). Top row: atoms colored by mean
Layer-2 gate sigma(x1) (warm = LOW gate). Bottom row: ground-truth non-MCS
(MCS-difference) atoms in green.

Defaults reproduce the published figure: the test-test cliff pair from
CHEMBL244 (Ki) differing only in a benzimidazole dihalogen
(5,6-dichloro, Ki=140 nM vs. 5,6-difluoro, Ki=6 nM).

Usage (run from the repository root):
    python analysis/gate_heatmap.py
    python analysis/gate_heatmap.py --dataset CHEMBL239_EC50 \
        --smiles1 <SMILES> --smiles2 <SMILES> --out my_figure.png

Use analysis/find_cliff_pair.py to pick other well-localized pairs.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog('rdApp.*')

if not hasattr(torch.nn, 'RMSNorm'):
    class _RMSNorm(torch.nn.Module):
        def __init__(self, ns, eps=1e-8):
            super().__init__()
            ns = (ns,) if isinstance(ns, int) else ns
            self.weight = torch.nn.Parameter(torch.ones(ns)); self.eps = eps
        def forward(self, x):
            return self.weight * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    torch.nn.RMSNorm = _RMSNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GraphCliffRegressor, normalize_edges
from dataset_utils import smiles_to_graph

DEVICE = torch.device('cpu')
HIDDEN_SIZE, NUM_LAYERS, GROUPS, MID_K = 256, 3, 4, 3
EDGE_DIM, ATOM_IN_DIM = 13, 38
LAYER = 1  # 0-indexed -> Layer 2, which shows the highest gate selectivity

# Paper figure: CHEMBL244 (Ki) test-test cliff pair, benzimidazole dihalogen.
SMI_LEFT  = 'N=C(N)c1ccc2[nH]c(Cc3nc4cc(Cl)c(Cl)cc4[nH]3)nc2c1'  # 5,6-dichloro, Ki=140 nM
SMI_RIGHT = 'N=C(N)c1ccc2[nH]c(Cc3nc4cc(F)c(F)cc4[nH]3)nc2c1'    # 5,6-difluoro, Ki=6 nM


def resolve_ckpt(ckpt_dir, dataset):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, dataset, '*', 'best_model.pt')))
    if not paths:
        raise SystemExit(f'No checkpoint under {ckpt_dir}/{dataset}/*/best_model.pt — train first.')
    return paths[0]


def load_model(ckpt_path):
    m = GraphCliffRegressor(atom_in_dim=ATOM_IN_DIM, edge_dim=EDGE_DIM,
                            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                            groups=GROUPS, mid_K=MID_K).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


def gates_all_layers(model, x, edge_index, edge_attr):
    h = model.atom_encoder(x); out = []
    for layer in model.encoder.layers:
        z = layer.short(layer.proj(layer.pre_norm(h)), edge_index, edge_attr)
        x2, x1, v = torch.chunk(z, 3, dim=-1)
        gate = torch.sigmoid(x1); out.append(gate.detach().cpu())
        if edge_index.numel() > 0:
            ew = torch.ones(edge_index.size(1), device=DEVICE, dtype=x2.dtype)
            en = normalize_edges(h.size(0), edge_index, ew)
        else:
            en = torch.tensor([], device=DEVICE, dtype=h.dtype)
        h = layer.long(x2, edge_index, en) * gate + v + h
    return out


def per_atom_gate(model, smi, layer=LAYER):
    g = smiles_to_graph(smi, y=0.0)
    lg = gates_all_layers(model, g.x.to(DEVICE), g.edge_index.to(DEVICE), g.edge_attr.to(DEVICE))
    return lg[layer].mean(dim=-1).numpy()  # [N] mean over 256 channels


def mcs_diff_atoms(m1, m2):
    mcs = rdFMCS.FindMCS([m1, m2], atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder,
                         ringMatchesRingOnly=False, completeRingsOnly=False, timeout=5)
    sm = Chem.MolFromSmarts(mcs.smartsString)
    s1 = set(m1.GetSubstructMatch(sm)); s2 = set(m2.GetSubstructMatch(sm))
    return set(range(m1.GetNumAtoms())) - s1, set(range(m2.GetNumAtoms())) - s2


def draw(mol, atom_colors, atom_radii, size=(360, 230)):
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    op = d.drawOptions(); op.bondLineWidth = 2; op.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors, highlightAtomRadii=atom_radii,
        highlightBonds=[])
    d.FinishDrawing()
    png = d.GetDrawingText()
    import io
    return Image.open(io.BytesIO(png)).convert('RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CHEMBL244_Ki')
    p.add_argument('--ckpt_dir', default='ckpt')
    p.add_argument('--ckpt', default=None, help='Explicit checkpoint path (overrides --ckpt_dir)')
    p.add_argument('--smiles1', default=SMI_LEFT)
    p.add_argument('--smiles2', default=SMI_RIGHT)
    p.add_argument('--layer', type=int, default=LAYER, help='0-indexed filter layer (default 1 = Layer 2)')
    p.add_argument('--out', default='analysis_out/gate_heatmap_pair.png')
    args = p.parse_args()

    ckpt_path = args.ckpt or resolve_ckpt(args.ckpt_dir, args.dataset)
    print(f'Using checkpoint: {ckpt_path}')
    model = load_model(ckpt_path)
    smi_left, smi_right = args.smiles1, args.smiles2
    mL, mR = Chem.MolFromSmiles(smi_left), Chem.MolFromSmiles(smi_right)
    gL = per_atom_gate(model, smi_left, args.layer)
    gR = per_atom_gate(model, smi_right, args.layer)
    diffL, diffR = mcs_diff_atoms(mL, mR)
    print('left atoms', mL.GetNumAtoms(), 'gate range', round(gL.min(),3), round(gL.max(),3),
          'non-MCS', sorted(diffL))
    print('right atoms', mR.GetNumAtoms(), 'gate range', round(gR.min(),3), round(gR.max(),3),
          'non-MCS', sorted(diffR))

    # shared normalization across both molecules
    allg = np.concatenate([gL, gR]); lo, hi = allg.min(), allg.max()
    def importance(g):  # low gate -> high importance (warm)
        return 1.0 - (g - lo) / (hi - lo + 1e-9)
    cmap = cm.get_cmap('Oranges')
    def heat_colors(g):
        imp = importance(g)
        return {i: cmap(0.12 + 0.88 * float(imp[i]))[:3] for i in range(len(g))}
    def heat_radii(g):
        imp = importance(g)
        return {i: 0.25 + 0.30 * float(imp[i]) for i in range(len(g))}

    GREEN = (0.30, 0.75, 0.35)
    def mcs_colors(diff, n):
        return {i: GREEN for i in diff}
    def mcs_radii(diff, n):
        return {i: 0.45 for i in diff}

    topL = draw(mL, heat_colors(gL), heat_radii(gL))
    topR = draw(mR, heat_colors(gR), heat_radii(gR))
    botL = draw(mL, mcs_colors(diffL, mL.GetNumAtoms()), mcs_radii(diffL, mL.GetNumAtoms()))
    botR = draw(mR, mcs_colors(diffR, mR.GetNumAtoms()), mcs_radii(diffR, mR.GetNumAtoms()))

    cellW, cellH = topL.size
    TITLE = 30
    W = cellW * 2
    H = TITLE + cellH + TITLE + cellH
    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(topL, (0, TITLE)); canvas.paste(topR, (cellW, TITLE))
    canvas.paste(botL, (0, TITLE + cellH + TITLE)); canvas.paste(botR, (cellW, TITLE + cellH + TITLE))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except Exception:
        font = ImageFont.load_default()
    d.text((W/2, TITLE//2), 'Learned Gate Value', fill=(0,0,0), font=font, anchor='mm')
    d.text((W/2, TITLE + cellH + TITLE//2), 'Ground-Truth non-MCS', fill=(0,0,0), font=font, anchor='mm')
    out = args.out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    canvas.save(out)
    print('saved', out, canvas.size)


if __name__ == '__main__':
    main()
