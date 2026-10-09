"""Shared molecular representation and training-only target transforms."""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import from_smiles

FEATURE_SCHEMA = "pyg-2.6.1-smiles-float-no-explicit-H-v1"


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES or empty molecular graph")
    graph = from_smiles(smiles, with_hydrogen=False)
    graph.x = graph.x.float()
    graph.edge_attr = graph.edge_attr.float()
    return graph


class GraphDataset:
    def __init__(self, graphs):
        if not graphs:
            raise ValueError("Dataset cannot be empty")
        self.graphs = list(graphs)
        self.num_node_features = self.graphs[0].x.size(1)
        self.num_edge_features = self.graphs[0].edge_attr.size(1)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def fit_target_scaler(dataset, train_indices):
    targets = np.asarray([float(dataset[i].y.item()) for i in train_indices], dtype=float)
    if not targets.size or not np.isfinite(targets).all():
        raise ValueError("Training targets must be nonempty and finite")
    std = float(targets.std())
    return {"method": "standard", "mean": float(targets.mean()),
            "std": std if std > 0 else 1.0, "fit_partition": "train"}


def transform_targets(dataset, params):
    graphs = []
    for original in dataset:
        graph = original.clone()
        graph.x, graph.edge_attr = graph.x.float(), graph.edge_attr.float()
        graph.y = (graph.y.float() - params["mean"]) / params["std"]
        graphs.append(graph)
    return GraphDataset(graphs)
