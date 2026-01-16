import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset_utils import smiles_to_graph


class GraphDataset(Dataset):
    def __init__(self, dataset_name: str, split: str = 'train'):
        super().__init__()
        data = pd.read_csv(f'./benchmark_data/{dataset_name}.csv')
  
        if split == 'train':
            smiles_list = data[data['split'] == 'train'].smiles.values.tolist()
            y_list = data[data['split'] == 'train'].y.values.tolist()
            
        elif split == 'test':
            smiles_list = data[data['split'] == 'test'].smiles.values.tolist()
            y_list = data[data['split'] == 'test'].y.values.tolist()
            

        assert smiles_list is not None, "Could not find smiles list in MoleculeACE Data object."
        self.samples = []
        for s, y in zip(smiles_list, y_list):
            try:
                graph = smiles_to_graph(s, y)
                graph.smiles = s
                self.samples.append(graph)
            except Exception as e:
                continue
        if 'cliff_mol' in data.columns:
            self.cliff_mols_test = data[data['split'] == 'test']['cliff_mol'].values.tolist()
        else:
            self.cliff_mols_test = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def collate_graphs(batch):
    xs, eis, eas, ys, batch_idx = [], [], [], [], []
    node_offset = 0
    for gi, g in enumerate(batch):
        xs.append(g.x)

        if g.edge_index.numel() > 0:
            eis.append(g.edge_index + node_offset)
            eas.append(g.edge_attr)
        else:
            eis.append(torch.zeros((2,0), dtype=torch.long))
            eas.append(torch.zeros((0, batch[0].edge_attr.size(1)), dtype=torch.float32))

        ys.append([g.y])
        batch_idx.append(torch.full((g.x.size(0),), gi, dtype=torch.long))
        node_offset += g.x.size(0)

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if eas else torch.zeros((0,13), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    batch_vec = torch.cat(batch_idx, dim=0)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'batch': batch_vec
    }
