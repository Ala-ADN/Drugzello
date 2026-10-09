"""Inference from a versioned checkpoint using the SAME training representation."""

from threading import RLock

import numpy as np
import torch

from ..data.graphs import smiles_to_graph


class MEGANInference:
    def __init__(self, model, model_config, device="cpu", normalization=None):
        if normalization is None:
            raise ValueError("Checkpoint target normalization is required for logS inference")
        self.model, self.model_config = model.to(device).eval(), model_config
        self.device, self.normalization = device, normalization
        self._lock = RLock()  # Forward stores masks: requests must not interleave.

    def predict(self, smiles, include_explanations=False):
        return self.predict_with_uncertainty(smiles, n_samples=1,
                                             include_explanations=include_explanations)

    def predict_with_uncertainty(self, smiles, n_samples=20, include_explanations=True):
        if not 1 <= n_samples <= 200:
            raise ValueError("n_samples must be between 1 and 200")
        graph = smiles_to_graph(smiles).to(self.device)
        scaler = self.normalization
        with self._lock, torch.no_grad():
            self.model.eval()
            output, edge_scores, node_scores = self.model(
                graph.x, graph.edge_index, graph.edge_attr, return_explanations=True
            )
            value = float(output.item() * scaler["std"] + scaler["mean"])
            result = {"solubility": value, "unit": "logS", "confidence": None,
                      "model_type": "megan_pytorch", "uncertainty": None}
            if include_explanations:
                # Atom IDs follow RDKit/PyG SMILES order. Both directed bond
                # scores are averaged for a single original RDKit bond.
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                edge_lookup = {(int(src), int(dst)): i for i, (src, dst)
                               in enumerate(graph.edge_index.t().cpu().tolist())}
                bonds = []
                for bond in mol.GetBonds():
                    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    bonds.append((edge_scores[edge_lookup[a, b]] + edge_scores[edge_lookup[b, a]]) / 2)
                result.update({"node_importances": node_scores.mean(1).cpu().tolist(),
                               "node_importances_by_channel": node_scores.cpu().tolist(),
                               "edge_importances": [float(b.mean()) for b in bonds],
                               "bond_importances_by_channel": [b.cpu().tolist() for b in bonds],
                               "channel_labels": ["below training mean", "above training mean"]})
            if n_samples > 1:
                outputs, attributions = [], []
                try:
                    # This architecture has no BatchNorm: train() activates its
                    # functional attention dropout AND readout dropout.
                    self.model.train()
                    for _ in range(n_samples):
                        sample, _, nodes = self.model(graph.x, graph.edge_index, graph.edge_attr,
                                                       return_explanations=True)
                        outputs.append(float(sample.item() * scaler["std"] + scaler["mean"]))
                        attributions.append(nodes.cpu().numpy())
                finally:
                    self.model.eval()
                atom_std = np.std(np.asarray(attributions), axis=0).mean(1)
                result["uncertainty"] = {"prediction_std": float(np.std(outputs)),
                                         "uaa_score": float(atom_std.mean()),
                                         "aau_scores": atom_std.tolist()}
                result["uncertainty_method"] = "MC dropout; uncalibrated spread"
            return result
