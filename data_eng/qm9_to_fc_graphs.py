"""Helper functions from an online (not my own) colab to turn qm9 data into fc graphs.

Colab found here, all credits to Cambridge researchers:
https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb#scrollTo=oKraX_u9MXCS&uniqifier=1
"""

import os

from absl import app

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse

TARGET = 14  # Hard code the choice from qm9, enthalpy of atomization at 298K.

class SetTarget(object):
    """
    This transform mofifies the labels vector per data sample to only keep 
    the label for a specific target (there are 19 targets in QM9).

    Note: for this practical, we have hardcoded the target to be target #0,
    i.e. the electric dipole moment of a drug-like molecule.
    (https://en.wikipedia.org/wiki/Electric_dipole_moment)
    """
    def __call__(self, data):
        data.y = data.y[:, TARGET]
        return data


class CompleteGraph(object):
    """
    This transform adds all pairwise edges into the edge index per data sample, 
    then removes self loops, i.e. it builds a fully connected or complete graph
    """
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def load_fc_graphs_from_qm9(
        num_graphs: int = 2000, batch_size: int = 32, path: str = './qm9',
        data_ids = None):
    # Transforms which are applied during data loading:
    # (1) Fully connect the graphs, (2) Select the target/label
    transform = T.Compose([CompleteGraph(), SetTarget()])
    # Load the QM9 dataset with the transforms defined
    dataset = QM9(path, transform=transform)
    # Normalize targets per data sample to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, TARGET].item(), std[:, TARGET].item()

    # This option allows us to get specific indices (e.g. random ones).
    if data_ids is not None:
        # Get a random subset of graphs (num_graphs)
        return dataset[data_ids]
    else:
        # Split datasets (our 3K subset)
        train_dataset = dataset[:num_graphs]
        val_dataset = dataset[num_graphs:num_graphs+1000]
        test_dataset = dataset[num_graphs+1000:num_graphs+2000]

        return train_dataset, val_dataset, test_dataset


def main(argv):
    qm9_data = load_fc_graphs_from_qm9()


if __name__ == '__main__':
    app.run(main)