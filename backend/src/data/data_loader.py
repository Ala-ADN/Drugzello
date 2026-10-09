"""ESOL loading using the same PyG representation as inference.

Scaffold assignments and training-only transforms live in scaffold.py and graphs.py.
"""

from pathlib import Path
from torch_geometric.datasets import MoleculeNet


class MolecularDataLoader:
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def load_esol_dataset(self, transform=None):
        # SMILES graphs already have both bond directions. Do not add ToUndirected.
        return MoleculeNet(root=str(self.data_root / "ESOL"), name="ESOL", transform=transform)
