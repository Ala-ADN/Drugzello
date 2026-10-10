"""Prediction-only GAT: batching invariance and test-label isolation."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graphs import GraphDataset, smiles_to_graph, transform_targets
from src.data.scaffold import scaffold_folds
from src.models.gat import GATConfig, GATRegressor, fit_gat, predict, run_gat


@pytest.fixture
def molecules():
    graphs = []
    for i, smiles in enumerate(["CCO", "CC", "c1ccccc1", "Cc1ccccc1", "c1ccncc1",
                               "C1CCCCC1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]):
        g = smiles_to_graph(smiles)
        g.y = torch.tensor([i / 4 - 2])
        graphs.append(g)
    return GraphDataset(graphs)


def tiny_config():
    return GATConfig(hidden_channels=8, num_layers=2, embedding_dim=4,
                     num_epochs=2, batch_size=3, dropout=0)


def test_gat_batch_node_permutation_and_edgeless_graph(molecules):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = GATRegressor(tiny_config()).eval()
    graphs = [molecules[0], smiles_to_graph("[Na+]"), molecules[-1]]
    graphs[1].y = torch.tensor([0.])
    b = Batch.from_data_list(graphs)
    combined = model(b.x, b.edge_index, b.edge_attr, b.batch)
    individual = torch.cat([model(g.x, g.edge_index, g.edge_attr) for g in graphs])
    torch.testing.assert_close(combined, individual)
    assert torch.isfinite(combined).all()
    g = graphs[-1]
    order = torch.arange(g.num_nodes - 1, -1, -1)
    inverse = torch.argsort(order)
    torch.testing.assert_close(model(g.x, g.edge_index, g.edge_attr),
                               model(g.x[order], inverse[g.edge_index], g.edge_attr))
    combined.sum().backward()
    assert model.node_encoder.tables[0].weight.grad.abs().sum() > 0
    assert model.edge_encoder.tables[0].weight.grad.abs().sum() > 0


def test_gat_training_is_deterministic_and_ignores_test_labels(molecules):
    fold = scaffold_folds([g.smiles for g in molecules], n_folds=3)["folds"][0]
    config = tiny_config()
    first, scaler, fitted = fit_gat(molecules, fold["train_indices"], fold["validation_indices"], config, 42)
    for i in fold["test_indices"]:
        molecules[i].y.fill_(1e6)
    second, again_scaler, again_fitted = fit_gat(molecules, fold["train_indices"], fold["validation_indices"], config, 42)
    assert scaler == again_scaler and fitted == again_fitted
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_gat_exports_original_units_and_replayable_weights(molecules, tmp_path):
    manifest = scaffold_folds([g.smiles for g in molecules], n_folds=3)
    config = tiny_config()
    models, results = tmp_path / "models", tmp_path / "results"
    summary = run_gat(molecules, manifest, config, models, results)
    records = np.genfromtxt(results / "predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
    assert sorted(records["row_idx"].tolist()) == list(range(len(molecules)))
    assert summary["fold_count"] == 3 and summary["metric_unit"] == "logS"
    assert json.loads((results / "config.json").read_text()) == asdict(config)
    checkpoint = torch.load(models / "fold_0_model.pth", weights_only=True)
    model = GATRegressor(GATConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint["normalization"]
    replay = predict(model, transform_targets(molecules, scaler), manifest["folds"][0]["test_indices"], config.batch_size)
    np.testing.assert_allclose(replay * scaler["std"] + scaler["mean"],
                               records["predicted_logS"][records["fold_idx"] == 0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="fresh output"):
        run_gat(molecules, manifest, config, models, results)
