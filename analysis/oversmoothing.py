"""
Over-smoothing analysis: layer-wise Mean Average Distance (MAD) and Dirichlet
energy across GNN architectures (paper Figure 4).

Metrics
-------
MAD (per layer, per graph):
    Cosine distance between every pair of node representations inside a graph,
    self-comparisons excluded, averaged first at the node level and then over
    nodes. A decrease with depth means node representations are collapsing.

Dirichlet energy (final layer, per graph):
    E(H) = 1/2 * sum_{(u,v) in E} || h_u / sqrt(1+d_u) - h_v / sqrt(1+d_v) ||^2
    Low energy means adjacent nodes have become nearly identical.

Both are macro-averaged over the graphs of the test split.

Architectures
-------------
    gcn, gin, gine, gat                    plain stacks
    gine_residual                          + residual connections
    gine_dropedge                          + DropEdge (train time only)
    gine_nodenorm                          + NodeNorm
    gine_pairnorm                          + PairNorm
    graphcliff                             the model of this repository

Each baseline is trained with the same recipe as main.py (AdamW, warmup-cosine,
early stopping on the held-out validation split) so the measured embeddings come
from converged models rather than random initialisations.

Usage (run from the repository root):
    python -u analysis/oversmoothing.py --dataset CHEMBL204_Ki
    python -u analysis/oversmoothing.py --dataset CHEMBL204_Ki --num_layers 6 --epochs 100
    python -u analysis/oversmoothing.py --models gine gine_pairnorm graphcliff
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.norm import PairNorm
from torch_geometric.utils import dropout_edge

# Backport nn.RMSNorm for PyTorch < 2.4 (model.py relies on it).
if not hasattr(torch.nn, 'RMSNorm'):
    class _RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps

        def forward(self, x):
            return self.weight * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    torch.nn.RMSNorm = _RMSNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import GraphDataset, collate_graphs
from model import AtomEncoder, GraphCliffRegressor, normalize_edges

ALL_MODELS = ['gcn', 'gin', 'gine', 'gat', 'gine_residual', 'gine_dropedge',
              'gine_nodenorm', 'gine_pairnorm', 'graphcliff']


# ── Normalisation layers ──────────────────────────────────────────────────────

class NodeNorm(nn.Module):
    """NodeNorm: scale each node embedding by its own standard deviation."""

    def __init__(self, p: float = 2.0, eps: float = 1e-5):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp(min=self.eps)
        return x / std.pow(1.0 / self.p)


# ── Baseline encoders ─────────────────────────────────────────────────────────

def _mlp(hidden_size):
    return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, hidden_size))


class BaselineEncoder(nn.Module):
    """Plain L-layer GNN stack with optional anti-oversmoothing mechanism.

    forward() returns the node embeddings after *each* layer, which is what the
    layer-wise MAD curve is measured on.
    """

    def __init__(self, conv_type, hidden_size, edge_dim, num_layers,
                 residual=False, norm=None, dropedge=0.0, dropout=0.0):
        super().__init__()
        self.conv_type = conv_type
        self.residual = residual
        self.dropedge = dropedge
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        convs, norms = [], []
        for _ in range(num_layers):
            if conv_type == 'gcn':
                convs.append(GCNConv(hidden_size, hidden_size))
            elif conv_type == 'gin':
                convs.append(GINConv(_mlp(hidden_size), train_eps=True))
            elif conv_type == 'gine':
                convs.append(GINEConv(_mlp(hidden_size), train_eps=True, edge_dim=edge_dim))
            elif conv_type == 'gat':
                convs.append(GATConv(hidden_size, hidden_size, heads=4, concat=False,
                                     edge_dim=edge_dim))
            else:
                raise ValueError(f'Unknown conv type: {conv_type}')

            if norm == 'pairnorm':
                norms.append(PairNorm())
            elif norm == 'nodenorm':
                norms.append(NodeNorm())
            else:
                norms.append(nn.Identity())

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x, edge_index, edge_attr):
        hiddens = []
        for conv, norm in zip(self.convs, self.norms):
            ei, ea = edge_index, edge_attr
            if self.training and self.dropedge > 0:
                ei, mask = dropout_edge(edge_index, p=self.dropedge)
                ea = edge_attr[mask] if edge_attr is not None else None

            if self.conv_type in ('gine', 'gat'):
                out = conv(x, ei, ea)
            else:
                out = conv(x, ei)
            out = F.relu(out)
            out = norm(out)
            out = self.dropout(out)
            x = out + x if self.residual else out
            hiddens.append(x)
        return hiddens


class BaselineRegressor(nn.Module):
    def __init__(self, atom_in_dim, edge_dim, spec, hidden_size=256,
                 num_layers=3, dropout=0.0):
        super().__init__()
        self.atom_encoder = AtomEncoder(atom_in_dim, hidden_size, dropout)
        self.encoder = BaselineEncoder(
            spec['conv'], hidden_size, edge_dim, num_layers,
            residual=spec.get('residual', False), norm=spec.get('norm'),
            dropedge=spec.get('dropedge', 0.0), dropout=dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def node_embeddings(self, x_f, edge_index, edge_attr):
        return self.encoder(self.atom_encoder(x_f), edge_index, edge_attr)

    def forward(self, x_f, edge_index, edge_attr, batch):
        h = self.node_embeddings(x_f, edge_index, edge_attr)[-1]
        g = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)
        return self.reg_head(g)


BASELINE_SPECS = {
    'gcn':           {'conv': 'gcn'},
    'gin':           {'conv': 'gin'},
    'gine':          {'conv': 'gine'},
    'gat':           {'conv': 'gat'},
    'gine_residual': {'conv': 'gine', 'residual': True},
    'gine_dropedge': {'conv': 'gine', 'dropedge': 0.2},
    'gine_nodenorm': {'conv': 'gine', 'norm': 'nodenorm'},
    'gine_pairnorm': {'conv': 'gine', 'norm': 'pairnorm'},
}

DISPLAY_NAMES = {
    'gcn': 'GCN', 'gin': 'GIN', 'gine': 'GINE', 'gat': 'GAT',
    'gine_residual': 'GINE+Residual', 'gine_dropedge': 'GINE+DropEdge',
    'gine_nodenorm': 'GINE+NodeNorm', 'gine_pairnorm': 'GINE+PairNorm',
    'graphcliff': 'GraphCliff',
}


# ── GraphCliff per-layer embeddings ───────────────────────────────────────────

def graphcliff_node_embeddings(model, x_f, edge_index, edge_attr):
    """Replays GraphCliffEncoder layer by layer, keeping every intermediate."""
    h = model.atom_encoder(x_f)
    hiddens = []
    for layer in model.encoder.layers:
        z = layer.short(layer.proj(layer.pre_norm(h)), edge_index, edge_attr)
        x2, x1, v = torch.chunk(z, 3, dim=-1)
        if edge_index.numel() > 0:
            ew = torch.ones(edge_index.size(1), device=edge_index.device, dtype=x2.dtype)
            edge_norm = normalize_edges(h.size(0), edge_index, ew)
        else:
            edge_norm = torch.tensor([], device=h.device, dtype=h.dtype)
        h = layer.long(x2, edge_index, edge_norm) * torch.sigmoid(x1) + v + h
        hiddens.append(h)
    return hiddens


# ── Metrics ───────────────────────────────────────────────────────────────────

def mad_single_graph(h):
    """Mean Average Distance of one graph's node embeddings [N, D]."""
    n = h.size(0)
    if n < 2:
        return None
    hn = F.normalize(h, p=2, dim=-1, eps=1e-12)
    dist = 1.0 - hn @ hn.t()                     # cosine distance, [N, N]
    dist = dist.clamp(min=0.0)
    dist.fill_diagonal_(0.0)                      # drop self-comparisons
    per_node = dist.sum(dim=1) / (n - 1)          # node-level average
    return per_node.mean().item()


def dirichlet_single_graph(h, edge_index, num_nodes):
    """Symmetrically normalised Dirichlet energy of one graph."""
    if edge_index.numel() == 0:
        return 0.0
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
    deg.index_add_(0, src, torch.ones(src.size(0), device=h.device, dtype=h.dtype))
    scale = (1.0 + deg).pow(-0.5).unsqueeze(-1)
    hs = h * scale
    diff = hs[src] - hs[dst]
    # edge_index stores both directions, so the 1/2 factor is already applied.
    return 0.5 * diff.pow(2).sum().item()


@torch.no_grad()
def measure(model, loader, device, num_layers, is_graphcliff):
    """Returns (mad_per_layer, final_layer_dirichlet), macro-averaged."""
    model.eval()
    mad_sums = [[] for _ in range(num_layers)]
    dirichlet = []

    for batch in loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)
        graph_id = batch['batch'].to(device)

        if is_graphcliff:
            hiddens = graphcliff_node_embeddings(model, x, edge_index, edge_attr)
        else:
            hiddens = model.node_embeddings(x, edge_index, edge_attr)

        for gi in graph_id.unique().tolist():
            node_mask = graph_id == gi
            idx = node_mask.nonzero(as_tuple=True)[0]
            n_nodes = idx.numel()
            # Keep only the edges internal to this graph and reindex them.
            remap = torch.full((graph_id.size(0),), -1, dtype=torch.long, device=device)
            remap[idx] = torch.arange(n_nodes, device=device)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            sub_edges = remap[edge_index[:, edge_mask]]

            for layer_idx, h in enumerate(hiddens):
                value = mad_single_graph(h[idx])
                if value is not None:
                    mad_sums[layer_idx].append(value)
            dirichlet.append(dirichlet_single_graph(hiddens[-1][idx], sub_edges, n_nodes))

    mad = [float(np.mean(v)) if v else float('nan') for v in mad_sums]
    return mad, float(np.mean(dirichlet)) if dirichlet else float('nan')


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def train_model(model, train_loader, valid_loader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['x'].to(device), batch['edge_index'].to(device),
                         batch['edge_attr'].to(device), batch['batch'].to(device))
            loss = F.mse_loss(pred, batch['y'].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for group in optimizer.param_groups:
            group['lr'] = lr_at(epoch, args)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                pred = model(batch['x'].to(device), batch['edge_index'].to(device),
                             batch['edge_attr'].to(device), batch['batch'].to(device))
                preds += pred.view(-1).cpu().tolist()
                labels += batch['y'].view(-1).tolist()
        valid_loss = F.mse_loss(torch.tensor(preds), torch.tensor(labels)).item()

        if best_loss - valid_loss > args.min_delta:
            best_loss, no_improve = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(mad_rows, dirichlet_rows, num_layers, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed — skipping the figure, CSVs are still written.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    layers = list(range(1, num_layers + 1))

    for name, mad in mad_rows:
        style = dict(lw=2.6, marker='o', zorder=5) if name == 'graphcliff' else dict(lw=1.6, marker='o', alpha=0.8)
        axes[0].plot(layers, mad, label=DISPLAY_NAMES.get(name, name), **style)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('MAD')
    axes[0].set_title('(a) Layer-wise Mean Average Distance')
    axes[0].set_xticks(layers)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    names = [DISPLAY_NAMES.get(n, n) for n, _ in dirichlet_rows]
    values = [max(v, 1e-8) for _, v in dirichlet_rows]
    colors = ['tab:red' if n == 'GraphCliff' else 'tab:blue' for n in names]
    axes[1].bar(names, values, color=colors)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Dirichlet energy (final layer)')
    axes[1].set_title('(b) Dirichlet energy')
    axes[1].tick_params(axis='x', rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')
    axes[1].grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved -> {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CHEMBL204_Ki')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--mid_K', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--valid_frac', type=float, default=0.1)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='analysis_out')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    split_kwargs = dict(valid_frac=args.valid_frac, split_seed=args.split_seed)
    train_ds = GraphDataset(args.dataset, split='train', **split_kwargs)
    valid_ds = GraphDataset(args.dataset, split='valid', **split_kwargs)
    test_ds = GraphDataset(args.dataset, split='test')
    print(f'{args.dataset}: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_graphs)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_graphs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_graphs)

    atom_in_dim = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1)

    mad_rows, dirichlet_rows = [], []
    for name in args.models:
        set_seed(args.seed)
        print(f'\n=== {DISPLAY_NAMES.get(name, name)} ===')

        if name == 'graphcliff':
            model = GraphCliffRegressor(
                atom_in_dim, edge_dim, hidden_size=args.hidden_size,
                num_layers=args.num_layers, groups=args.groups,
                mid_K=args.mid_K, dropout=0).to(device)
        else:
            model = BaselineRegressor(
                atom_in_dim, edge_dim, BASELINE_SPECS[name],
                hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)

        model, valid_loss = train_model(model, train_loader, valid_loader, device, args)
        mad, energy = measure(model, test_loader, device, args.num_layers,
                              is_graphcliff=(name == 'graphcliff'))
        print(f'  valid MSE {valid_loss:.4f} | MAD {[round(v, 4) for v in mad]} '
              f'| Dirichlet {energy:.3e}')
        mad_rows.append((name, mad))
        dirichlet_rows.append((name, energy))

    import pandas as pd
    mad_df = pd.DataFrame(
        [{'model': DISPLAY_NAMES.get(n, n),
          **{f'layer{i + 1}': v for i, v in enumerate(mad)}} for n, mad in mad_rows])
    dir_df = pd.DataFrame(
        [{'model': DISPLAY_NAMES.get(n, n), 'dirichlet_energy': v} for n, v in dirichlet_rows])

    mad_csv = os.path.join(args.out_dir, f'oversmoothing_mad_{args.dataset}.csv')
    dir_csv = os.path.join(args.out_dir, f'oversmoothing_dirichlet_{args.dataset}.csv')
    mad_df.to_csv(mad_csv, index=False)
    dir_df.to_csv(dir_csv, index=False)
    print(f'\nSaved -> {mad_csv}\nSaved -> {dir_csv}')
    print(mad_df.to_string(index=False))
    print(dir_df.to_string(index=False))

    plot(mad_rows, dirichlet_rows, args.num_layers,
         os.path.join(args.out_dir, f'oversmoothing_mad_dirichlet_{args.dataset}.png'))


if __name__ == '__main__':
    main()
