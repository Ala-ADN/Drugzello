"""Small deterministic CPU checks; no downloads, cloud services, or saved weights."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graphs import fit_target_scaler, transform_targets
from src.models.megan_architecture import MEGANCore
from src.models.trainer import MEGANTrainer
from src.utils.config import MEGANConfig
from src.models.experiment import regression_metrics


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def molecules():
    graphs = []
    for i, smiles in enumerate(["CCO", "CC", "CCC", "CCN", "CO", "CCCO"]):
        graph = from_smiles(smiles)
        graph.x = graph.x.float()
        graph.edge_attr = graph.edge_attr.float()
        graph.y = torch.tensor([float(i) - 3])
        graphs.append(graph)
    return graphs


def small_model():
    return MEGANCore(in_channels=9, hidden_channels=8, edge_dim=3,
                     num_layers=2, K=2, dropout=0.0)


def test_graph_representation_and_batch_predictions(molecules):
    model = small_model().eval()
    batch = Batch.from_data_list(molecules[:2])
    assert batch.x.shape[1] == 9
    assert batch.edge_attr.shape[1] == 3
    prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
    individual = torch.cat([
        model(g.x, g.edge_index, g.edge_attr) for g in molecules[:2]
    ])
    torch.testing.assert_close(prediction, individual)
    assert len(model.get_explanations()) == 2


def test_normalized_targets_roundtrip(molecules):
    original = np.array([g.y.item() for g in molecules])
    params = fit_target_scaler(molecules, range(len(molecules)))
    normalized = transform_targets(molecules, params)
    restored = np.array([g.y.item() for g in normalized]) * params["std"] + params["mean"]
    np.testing.assert_allclose(restored, original, atol=1e-6)


def test_regression_metrics_use_supplied_scale():
    metrics = regression_metrics(
        np.array([-3.0, -2.0, -1.0]), np.array([-2.0, -2.0, -2.0])
    )
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["r2"] == pytest.approx(0)


def test_best_checkpoint_survives_later_optimizer_steps(molecules):
    config = MEGANConfig(hidden_channels=8, num_layers=2, num_epochs=2,
                         dropout=0.0, beta_sparsity=0.0, delta_decor=0.0)
    trainer = MEGANTrainer(config, torch.device("cpu"))
    model = small_model()
    captured = []

    def increasing_validation_loss(model, loader):
        captured.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        loss = float(len(captured))
        return {k: loss for k in ("total", "pred", "exp", "sparsity", "decor")}, loss

    loader = DataLoader(molecules, batch_size=3, shuffle=False)
    with patch.object(trainer, "evaluate", side_effect=increasing_validation_loss):
        result = trainer.train_fold(model, loader, loader, fold_idx=0)
    assert result["best_val_loss"] == 1.0
    for key, tensor in captured[0].items():
        torch.testing.assert_close(result["best_model_state"][key], tensor)
    assert any(not torch.equal(captured[0][k], captured[1][k]) for k in captured[0])
