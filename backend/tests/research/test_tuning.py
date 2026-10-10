"""Embedding, split-step and validation-only selection invariants."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graphs import GraphDataset, smiles_to_graph
from src.data.scaffold import scaffold_folds
from src.models.megan_architecture import MEGANCore, EMBEDDED_ARCHITECTURE_VERSION
from src.models.trainer import MEGANLoss, MEGANTrainer
from src.models.experiment import checkpoint, model_kwargs, evaluate
from src.models import tuning
from src.utils.config import MEGANConfig
from core.model_loader import ModelLoader


@pytest.fixture
def molecules():
    smiles = ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1",
              "C1CCCCC1", "CC1CCCCC1", "C1CCCC1", "c1ccsc1", "c1ccoc1", "c1ncc[nH]1"]
    graphs = []
    for i, s in enumerate(smiles):
        g = smiles_to_graph(s)
        g.y = torch.tensor([i / 4 - 2])
        graphs.append(g)
    return GraphDataset(graphs)


def test_embedding_category_relabeling_and_gradients(molecules):
    torch.manual_seed(42)
    model = MEGANCore(9, hidden_channels=8, num_layers=1, edge_dim=3,
                      use_edge_features=True, feature_encoding="embedding", dropout=0).eval()
    g = molecules[0]
    altered = copy.deepcopy(model)
    expected = model(g.x, g.edge_index, g.edge_attr)
    table = altered.node_encoder.tables[0]
    # Swapping carbon/oxygen category IDs and their table rows changes no meaning.
    with torch.no_grad():
        table.weight[[6, 8]] = table.weight[[8, 6]]
    x = g.x.clone()
    x[:, 0] = torch.where(g.x[:, 0] == 6, 8, torch.where(g.x[:, 0] == 8, 6, g.x[:, 0]))
    torch.testing.assert_close(expected, altered(x, g.edge_index, g.edge_attr))
    model(g.x, g.edge_index, g.edge_attr).sum().backward()
    assert model.node_encoder.tables[0].weight.grad.abs().sum() > 0
    assert model.edge_encoder.tables[0].weight.grad.abs().sum() > 0
    x[0, 0] = 6.5
    with pytest.raises(ValueError, match="integer"):
        model(x, g.edge_index, g.edge_attr)


def test_no_aux_preset_changes_only_loss_weights_and_keeps_mask_gradients(molecules):
    backend = Path(__file__).resolve().parents[2]
    config = MEGANConfig.load(backend / "configs/scaffold_megan_no_aux.yaml")
    original = next(c["config"] for c in tuning.read_candidates(backend / "configs/scaffold_search.yaml")
                    if c["id"] == "embedding_joint_32")
    changed = {key for key, value in config.to_dict().items() if value != original[key]}
    assert changed == {"gamma_exp", "beta_sparsity"}
    assert config.gamma_exp == config.beta_sparsity == config.delta_decor == 0
    torch.manual_seed(42)
    model = MEGANCore(**model_kwargs(molecules, config))
    batch = Batch.from_data_list([molecules[0], molecules[-1]])
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    loss = MEGANLoss(config.gamma_exp, config.beta_sparsity, config.delta_decor)(output, batch.y.view(-1, 1), model)
    expected = torch.nn.functional.mse_loss(output, batch.y.view(-1, 1))
    torch.testing.assert_close(loss["total"], expected, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    actual_grad = torch.autograd.grad(loss["total"], parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, parameters, retain_graph=True)
    for actual, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    mask_grad = torch.autograd.grad(loss["total"], model.node_importance_mlp[0].weight)[0]
    assert mask_grad.abs().sum() > 0


def test_split_step_updates_explanation_before_prediction(molecules):
    torch.manual_seed(42)
    model = MEGANCore(9, hidden_channels=8, num_layers=1, edge_dim=3,
                      use_edge_features=True, feature_encoding="embedding", dropout=0)
    batch = Batch.from_data_list([molecules[0], molecules[-1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original_step = optimizer.step
    head_grads, encoder_states = [], []
    before = model.node_encoder.tables[0].weight.detach().clone()
    def observed_step():
        head_grads.append([p.grad for p in model.classifier.parameters()])
        result = original_step()
        encoder_states.append(model.node_encoder.tables[0].weight.detach().clone())
        return result
    optimizer.step = observed_step
    MEGANTrainer.optimize_batch(model, batch, optimizer, MEGANLoss(gamma_exp=0.1), "split")
    assert len(head_grads) == 2
    assert all(g is None for g in head_grads[0])
    assert all(g is not None for g in head_grads[1])
    assert not torch.equal(before, encoder_states[0])
    assert not torch.equal(encoder_states[0], encoder_states[1])


def test_embedding_checkpoint_loader_roundtrip(molecules, tmp_path):
    config = MEGANConfig(hidden_channels=8, num_layers=1, use_edge_features=True,
                         feature_encoding="embedding", dropout=0)
    kwargs = model_kwargs(molecules, config)
    model = MEGANCore(**kwargs).eval()
    path = tmp_path / "embedded.pth"
    torch.save(checkpoint(model, kwargs, {"method": "standard", "mean": -3., "std": 2.},
                          {"config": config.to_dict()}), path)
    loader = ModelLoader()
    assert loader.load_model(path)
    assert EMBEDDED_ARCHITECTURE_VERSION in loader.get_version()
    g = molecules[0]
    with torch.no_grad():
        expected = model(g.x, g.edge_index, g.edge_attr).item() * 2 - 3
    assert loader.get_model().predict(g.smiles)["solubility"] == pytest.approx(expected)


def test_selector_requires_both_channel_polarities_and_ignores_test():
    def trial(name, mae, deltas):
        return {"candidate_id": name, "validation": {"mae": mae},
                "validation_explanations": {"signed_channel_ablation_mean_logS": deltas},
                "test": {"mae": 999.}}
    trials = [trial("accurate", .5, [-1., 4.]), trial("faithful", .7, [1., 2.])]
    selected = tuning.choose_trials(trials)
    assert selected == {"accuracy": "accurate", "fidelity_constrained": "faithful", "fidelity_constraint_satisfied": True}
    trials[0]["test"]["mae"] = 0
    assert tuning.choose_trials(trials) == selected
    assert not tuning.choose_trials(trials[:1])["fidelity_constraint_satisfied"]


def test_fidelity_reports_molecule_polarity_not_just_channel_means(molecules):
    from torch_geometric.nn import global_mean_pool
    class KnownAblations(torch.nn.Module):
        K = 2
        def forward(self, x, edge_index, edge_attr, batch, return_explanations=False, importance_mask=None):
            nodes = x.new_ones((len(x), 2))
            gates = global_mean_pool(nodes if importance_mask is None else importance_mask, batch)
            # Graph 0 has negative evidence; graph 1's channel 0 is reversed.
            signs = x.new_tensor([-1., 1.])[:len(gates)]
            output = (gates[:, 0] * signs + gates[:, 1] * 2).view(-1, 1)
            edges = x.new_full((edge_index.size(1), 2), .5)
            return (output, edges, nodes) if return_explanations else output
    _, _, scores = evaluate(KnownAblations(), molecules, [0, 2], {"mean": -3., "std": 2.}, batch_size=2)
    assert scores["signed_channel_ablation_mean_logS"] == [0., 4.]
    assert scores["fidelity_star_logS"] == 2.
    assert scores["channel_polarity_fraction"] == [.5, 1.]
    assert scores["both_channels_correct_fraction"] == .5


def test_validation_worker_is_independent_of_outer_test_labels(molecules, tmp_path, monkeypatch):
    manifest = scaffold_folds([g.smiles for g in molecules], n_folds=3)
    fold = manifest["folds"][0]
    monkeypatch.setattr(tuning, "load_raw", lambda _: molecules)
    config = MEGANConfig(hidden_channels=8, num_layers=1, use_edge_features=True,
                         feature_encoding="embedding", num_epochs=1, dropout=0)
    job = {"candidate": {"id": "tiny", "config": config.to_dict()}, "seed": 42,
           "data_root": str(tmp_path), "fold_idx": 0, "trial_dir": str(tmp_path/'trial'),
           "train_indices": fold["train_indices"], "validation_indices": fold["validation_indices"],
           "provenance": "test"}
    first = tuning._fit_trial(job)
    state = torch.load(tmp_path/'trial/model.pth', weights_only=True)["model_state_dict"]
    for i in fold["test_indices"]:
        molecules[i].y.fill_(1e6)
    second = tuning._fit_trial(job)
    repeated = torch.load(tmp_path/'trial/model.pth', weights_only=True)["model_state_dict"]
    assert first["validation"] == second["validation"]
    for key in state:
        torch.testing.assert_close(state[key], repeated[key], rtol=0, atol=0)


def test_tiny_search_freezes_choices_before_test_and_refits(molecules, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    manifest = scaffold_folds([g.smiles for g in molecules], n_folds=3)
    monkeypatch.setattr(tuning, "load_raw", lambda _: molecules)
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump({"base": {"hidden_channels": 8, "num_layers": 1,
                  "num_epochs": 1, "use_edge_features": True, "feature_encoding": "embedding", "dropout": 0},
                  "candidates": [{"id": "joint", "overrides": {}}, {"id": "split", "overrides": {"training_schedule": "split"}},
                                 {"id": "ordinal_long", "overrides": {"feature_encoding": "ordinal"}}]}))
    models, results = tmp_path/'models', tmp_path/'results'
    trials, selected = tuning.run_search(str(tmp_path), manifest, path, models, results, workers=1)
    assert len(trials) == 9 and not (results/'summary.json').exists()
    summary = tuning.evaluate_selections(str(tmp_path), models, results, refit=True)
    assert summary["fold_count"] == 3 and len(summary["models"]) == 7
    assert "ablation_ordinal_long" in summary["models"]
    assert json.loads((results/'selection.json').read_text()) == selected
    loader = ModelLoader()
    assert loader.load_model(models/'model.pth')
    assert np.isfinite(loader.get_model().predict("CCO")["solubility"])
    assert json.loads((results/'deployment.json').read_text())["independent_test_score"] is None
