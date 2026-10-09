"""Scientific invariants: graph messages, explanations, leakage and real inference."""

import csv
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graphs import GraphDataset, fit_target_scaler, smiles_to_graph
from src.data.scaffold import scaffold_folds, scaffold_key, validate_scaffold_manifest
from src.models.experiment import checkpoint, model_kwargs, run_experiment
from src.models.megan_architecture import GATv2WithLogits, MEGANCore
from src.models.trainer import MEGANLoss
from src.utils.config import MEGANConfig
from core.model_loader import ModelLoader
from services.inference_service import InferenceService


SMILES = ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1",
          "C1CCCCC1", "CC1CCCCC1", "C1CCCC1", "c1ccsc1", "c1ccoc1", "c1ncc[nH]1"]


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def dataset():
    graphs = []
    for i, smiles in enumerate(SMILES):
        graph = smiles_to_graph(smiles)
        graph.y = torch.tensor([float(i) / 4 - 2])  # Synthetic fixture, not chemistry measurements.
        graphs.append(graph)
    return GraphDataset(graphs)


def test_scaffold_folds_cover_test_once_and_keep_analogues_together():
    manifest = scaffold_folds(SMILES, n_folds=3, seed=42)
    assert manifest == scaffold_folds(SMILES, n_folds=3, seed=42)
    validate_scaffold_manifest(manifest)
    assert scaffold_key("CCO") == scaffold_key("CCCO") == ""
    assert scaffold_key("c1ccccc1") == scaffold_key("Cc1ccccc1")
    for fold in manifest["folds"]:
        for a, b in [(0, 1), (2, 3)]:
            assert any(a in fold[name] and b in fold[name]
                       for name in ("train_indices", "validation_indices", "test_indices"))


def test_manifest_rejects_leakage_and_invalid_smiles():
    manifest = scaffold_folds(SMILES, n_folds=3)
    fold = manifest["folds"][0]
    fold["train_indices"].append(fold["test_indices"][0])
    with pytest.raises(ValueError):
        validate_scaffold_manifest(manifest)
    with pytest.raises(ValueError):
        scaffold_key("not-a-molecule")


def test_scaler_never_reads_validation_or_test_targets(dataset):
    train = [0, 1, 2]
    first = fit_target_scaler(dataset, train)
    dataset[3].y.fill_(1e9)
    assert fit_target_scaler(dataset, train) == first
    assert first["mean"] == pytest.approx(np.mean([-2., -1.75, -1.5]))


def test_messages_depend_on_source_neighbors():
    layer = GATv2WithLogits(1, 1, add_bias=False)
    with torch.no_grad():
        layer.lin_node.weight.fill_(1)
        layer.att.zero_()
    # Uniform attention: node 0 should average its OWN and NEIGHBOR features.
    edges = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    before = layer(torch.tensor([[1.], [3.]]), edges)
    after = layer(torch.tensor([[1.], [7.]]), edges)
    assert before[0].item() == pytest.approx(2.)
    assert after[0].item() == pytest.approx(4.)


def test_explanation_shapes_bounds_and_auxiliary_gradients(dataset):
    model = MEGANCore(9, hidden_channels=8, edge_dim=3, use_edge_features=True,
                      num_layers=2, dropout=0)
    batch = Batch.from_data_list([dataset[0], dataset[2]])
    y, edges, nodes = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                            return_explanations=True)
    assert edges.shape == (batch.edge_index.size(1), 2)
    assert nodes.shape == (batch.x.size(0), 2)
    assert (nodes >= 0).all() and (nodes <= 1).all()
    assert (edges >= 0).all() and (edges <= 1).all()
    losses = MEGANLoss()(y, batch.y.view(-1, 1), model)
    for name in ["exp", "sparsity"]:
        grad = torch.autograd.grad(losses[name], model.attn_layers[0][0].att,
                                   retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_channel_ablation_uses_same_forward(dataset):
    model = MEGANCore(9, hidden_channels=8, edge_dim=3, num_layers=2, dropout=0).eval()
    graph = dataset[2]
    output, _, nodes = model(graph.x, graph.edge_index, graph.edge_attr, return_explanations=True)
    ones = torch.ones_like(nodes)
    identity = model(graph.x, graph.edge_index, graph.edge_attr, importance_mask=ones)
    torch.testing.assert_close(output, identity)
    zeros = torch.zeros_like(nodes)
    ablated = model(graph.x, graph.edge_index, graph.edge_attr, importance_mask=zeros)
    bias_only = model.classifier(torch.zeros(1, 16))
    torch.testing.assert_close(ablated, bias_only)


def test_molecule_permutation_and_batching_preserve_predictions(dataset):
    model = MEGANCore(9, hidden_channels=8, edge_dim=3, use_edge_features=True,
                      num_layers=2, dropout=0).eval()
    graph = dataset[2]
    permutation = torch.tensor([2, 5, 0, 3, 1, 4])
    inverse = torch.argsort(permutation)
    expected = model(graph.x, graph.edge_index, graph.edge_attr)
    permuted = model(graph.x[permutation], inverse[graph.edge_index], graph.edge_attr)
    torch.testing.assert_close(expected, permuted)
    other = dataset[0]
    individual = torch.cat([expected, model(other.x, other.edge_index, other.edge_attr)])
    batch = Batch.from_data_list([graph, other])
    together = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    torch.testing.assert_close(individual, together)


def save_fixture_model(dataset, tmp_path):
    config = MEGANConfig(hidden_channels=8, num_layers=2, dropout=0, gamma_exp=0)
    kwargs = model_kwargs(dataset, config)
    model = MEGANCore(**kwargs).eval()
    scaler = {"method": "standard", "mean": -3., "std": 2., "fit_partition": "train"}
    path = tmp_path / "model.pth"
    torch.save(checkpoint(model, kwargs, scaler, {"config": config.to_dict()}), path)
    return path, model, scaler


def test_real_checkpoint_roundtrip_denormalizes_and_aligns_masks(dataset, tmp_path):
    path, original, scaler = save_fixture_model(dataset, tmp_path)
    loader = ModelLoader()
    assert loader.load_model(path)
    result = loader.get_model().predict(SMILES[0], include_explanations=True)
    graph = dataset[0]
    with torch.no_grad():
        expected = original(graph.x, graph.edge_index, graph.edge_attr).item() * 2 - 3
    assert result["solubility"] == pytest.approx(expected)
    assert result["confidence"] is None
    assert len(result["node_importances_by_channel"]) == 3
    assert len(result["bond_importances_by_channel"]) == 2
    assert result == loader.get_model().predict(SMILES[0], include_explanations=True)


def test_missing_and_legacy_checkpoints_fail_without_mock(tmp_path):
    loader = ModelLoader()
    assert not loader.load_model(tmp_path / "missing.pth")
    assert not loader.is_loaded()
    path = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": {}}, path)
    assert not loader.load_model(path)
    assert "legacy checkpoint" in loader.last_error
    assert loader.get_version() is None


def test_mc_dropout_restores_deterministic_mode(dataset, tmp_path):
    path, _, _ = save_fixture_model(dataset, tmp_path)
    loader = ModelLoader()
    assert loader.load_model(path)
    wrapper = loader.get_model()
    expected = wrapper.predict("CCO")
    result = wrapper.predict_with_uncertainty("CCO", n_samples=10)
    assert result["uncertainty"]["prediction_std"] >= 0
    assert wrapper.predict("CCO") == expected
    assert not wrapper.model.training


def test_api_returns_real_prediction_or_503(dataset, tmp_path, monkeypatch):
    from api.main import app
    from api.endpoints import inference
    from core.config import settings
    from core.model_loader import model_loader
    path, _, _ = save_fixture_model(dataset, tmp_path)
    monkeypatch.setattr(settings, "model_path", str(path))
    monkeypatch.setattr(inference, "_inference_service", None)
    with TestClient(app) as client:
        response = client.post("/predict/enhanced", json={"smiles": "CCO", "include_explanations": True})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"]["model_type"] == "megan_pytorch"
        assert body["prediction"]["confidence"] is None
        assert client.get("/health").json()["model_loaded"]
        assert client.post("/predict", json={"smiles": "garbage"}).status_code == 422
        assert client.post("/solubility", json={"smiles": "CCO", "solvent": "DMSO"}).status_code == 422
        monkeypatch.setattr(settings, "model_path", str(tmp_path / "absent.pth"))
        model_loader.unload_model()
        assert client.post("/predict", json={"smiles": "CCO"}).status_code == 503
        assert client.get("/ready").status_code == 503


def test_nested_experiment_restores_checkpoints_and_publishes_units(dataset, tmp_path):
    manifest = scaffold_folds(SMILES, n_folds=3)
    config = MEGANConfig(hidden_channels=8, num_layers=1, num_epochs=1, batch_size=4,
                         dropout=0, gamma_exp=0.1)
    models, results = tmp_path / "models", tmp_path / "results"
    folds, summary = run_experiment(dataset, config, manifest, models, results,
                                    refit=True, random_forest=False)
    assert summary["metric_unit"] == "logS"
    assert len(folds) == 3
    with (results / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(int(row["row_idx"]) for row in rows) == list(range(len(dataset)))
    loader = ModelLoader()
    for fold in folds:
        assert loader.load_model(models / f"fold_{fold['fold_idx']}_model.pth")
        row = next(row for row in rows if int(row["fold_idx"]) == fold["fold_idx"])
        predicted = loader.get_model().predict(row["smiles"])["solubility"]
        assert predicted == pytest.approx(float(row["predicted_logS"]), abs=1e-5)
    assert loader.load_model(models / "model.pth")


def test_attention_matches_pyg_gatv2_reference(dataset):
    from torch_geometric.nn import GATv2Conv
    graph = dataset[2]
    ours = GATv2WithLogits(9, 8, edge_dim=3, heads=2, use_edge_features=True, dropout=0, add_bias=False)
    reference = GATv2Conv(9, 8, heads=2, concat=False, edge_dim=3, add_self_loops=False, bias=False)
    with torch.no_grad():
        reference.lin_l.weight.copy_(ours.lin_node.weight)
        reference.lin_r.weight.copy_(ours.lin_target.weight)
        reference.lin_edge.weight.copy_(ours.lin_edge.weight)
        reference.att.copy_(ours.att)
    actual = ours(graph.x, graph.edge_index, graph.edge_attr)
    expected = reference(graph.x, graph.edge_index, graph.edge_attr)
    torch.testing.assert_close(actual, expected)
