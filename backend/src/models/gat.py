"""Prediction-only GAT control on the existing ESOL scaffold partitions.

Uses original GAT (PyG GATConv), not MEGAN's GATv2 or explanation gates.
Categorical embeddings, residuals and LayerNorm are explicit variants.
"""

import csv
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
from torch_geometric.utils.smiles import e_map, x_map

from ..data.graphs import fit_target_scaler, transform_targets, FEATURE_SCHEMA
from ..data.scaffold import validate_scaffold_manifest
from .experiment import environment_metadata, regression_metrics, seed_everything, write_json
from .megan_architecture import CategoricalEncoder
from .tuning import validate_data


@dataclass
class GATConfig:
    hidden_channels: int = 32
    num_layers: int = 3
    heads: int = 1
    embedding_dim: int = 16
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    batch_size: int = 64
    num_epochs: int = 150
    patience: int = 20

    def __post_init__(self):
        for key in ("hidden_channels", "num_layers", "heads", "embedding_dim",
                    "batch_size", "num_epochs", "patience"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.dropout < 1 or not np.isfinite(self.dropout):
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


class GATRegressor(nn.Module):
    architecture_version = "gat-regression-embedded-v1"

    def __init__(self, config):
        super().__init__()
        self.node_encoder = CategoricalEncoder([len(v) for v in x_map.values()], config.embedding_dim)
        self.edge_encoder = CategoricalEncoder([len(v) for v in e_map.values()], config.embedding_dim)
        width = config.hidden_channels
        input_width = len(x_map) * config.embedding_dim
        self.input_projection = nn.Linear(input_width, width, bias=False)
        self.layers = nn.ModuleList([
            GATConv(input_width if i == 0 else width, width, heads=config.heads,
                    concat=False, edge_dim=len(e_map) * config.embedding_dim,
                    dropout=config.dropout, add_self_loops=True, fill_value=0.0)
            for i in range(config.num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.layers])
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(),
                                  nn.Dropout(config.dropout), nn.Linear(width, 1))

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=x.device)
        h, bonds = self.node_encoder(x), self.edge_encoder(edge_attr)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = self.input_projection(h) if i == 0 else h
            h = norm(F.elu(layer(h, edge_index, bonds)) + residual)
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        return self.head(pooled)


@torch.no_grad()
def predict(model, dataset, indices, batch_size):
    """Return standardized predictions in the supplied row order."""
    model.eval()
    return torch.cat([
        model(b.x, b.edge_index, b.edge_attr, b.batch).view(-1)
        for b in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    ]).cpu().numpy().astype(float)


def fit_gat(dataset, train_indices, validation_indices, config, seed):
    """No test IDs are accepted; all fitted transforms use training rows only."""
    torch.set_num_threads(1)
    seed_everything(seed)
    scaler = fit_target_scaler(dataset, train_indices)
    normalized = transform_targets(dataset, scaler)
    model = GATRegressor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loader = DataLoader(Subset(normalized, train_indices), batch_size=config.batch_size, shuffle=True)
    val_y = np.asarray([float(dataset[i].y.item()) for i in validation_indices])
    best_mae, best_state, best_epoch, stale = float("inf"), None, None, 0
    history = []
    for epoch in range(config.num_epochs):
        model.train()
        total, count = 0., 0
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(b.x, b.edge_index, b.edge_attr, b.batch).view(-1)
            loss = F.mse_loss(output, b.y.view(-1))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite GAT training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * b.num_graphs
            count += b.num_graphs
        val_p = predict(model, normalized, validation_indices, config.batch_size) * scaler["std"] + scaler["mean"]
        metrics = regression_metrics(val_y, val_p)
        scheduler.step(metrics["mae"])
        history.append({"epoch": epoch, "train_mse_standardized": total / count,
                        "validation": metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        if metrics["mae"] < best_mae:
            best_mae, best_epoch, stale = metrics["mae"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, scaler, {"best_epoch": best_epoch, "validation_mae_logS": best_mae,
                           "epochs_run": len(history), "history": history}


def run_gat(dataset, manifest, config, save_dir, results_dir, data_root=None):
    validate_scaffold_manifest(manifest)
    if manifest["smiles"] != [g.smiles for g in dataset]:
        raise ValueError("Dataset order differs from the frozen split manifest")
    if data_root is not None:
        validate_data(dataset, data_root, manifest)
    save_dir, results_dir = Path(save_dir), Path(results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to silently replace existing research evidence.
    if any(results_dir.glob("*.json")) or any(save_dir.glob("*.pth")):
        raise ValueError("Use fresh output directories for a new GAT run")
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "splits.json", manifest)
    environment = environment_metadata()
    environment["architecture_version"] = GATRegressor.architecture_version
    write_json(results_dir / "environment.json", environment)
    rows, folds = [], []
    for fold in manifest["folds"]:
        i, seed = fold["fold_idx"], manifest["seed"] + fold["fold_idx"]
        model, scaler, fitted = fit_gat(dataset, fold["train_indices"], fold["validation_indices"], config, seed)
        metadata = {"fold_idx": i, "seed": seed, "selection_partition": "inner validation only",
                    "selection_metric": "mae", "best_epoch": fitted["best_epoch"]}
        path = save_dir / f"fold_{i}_model.pth"
        torch.save({"format": "drugzello-gat-v1", "architecture_version": model.architecture_version,
                    "feature_schema": FEATURE_SCHEMA, "config": asdict(config),
                    "model_state_dict": model.state_dict(), "normalization": scaler,
                    "metadata": metadata}, path)
        test = fold["test_indices"]
        normalized = transform_targets(dataset, scaler)
        p = predict(model, normalized, test, config.batch_size) * scaler["std"] + scaler["mean"]
        y = [float(dataset[j].y.item()) for j in test]
        record = {**metadata, **{label + "_size": len(fold[label + "_indices"])
                                 for label in ("train", "validation", "test")},
                  "epochs_run": fitted["epochs_run"], "validation_mae_logS": fitted["validation_mae_logS"],
                  "normalization": scaler, "test": regression_metrics(y, p),
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        folds.append(record)
        write_json(results_dir / f"fold_{i}_history.json", fitted["history"])
        for j, truth, prediction in zip(test, y, p):
            rows.append({"fold_idx": i, "row_idx": j, "smiles": manifest["smiles"][j],
                         "scaffold_id": manifest["scaffold_ids"][j],
                         "true_logS": truth, "predicted_logS": float(prediction)})
        write_json(results_dir / "fold_metrics.json", folds)
        print(f"GAT fold {i}: MAE={record['test']['mae']:.4f}, RMSE={record['test']['rmse']:.4f} logS", flush=True)
    summary = {"model": "gat_only", "metric_unit": "logS", "fold_count": len(folds),
               "selection_partition": "inner validation only", "configuration_selection": "fixed; no search",
               "test_mae_mean": float(np.mean([f["test"]["mae"] for f in folds])),
               "test_mae_std": float(np.std([f["test"]["mae"] for f in folds])),
               "pooled_test": regression_metrics([r["true_logS"] for r in rows], [r["predicted_logS"] for r in rows])}
    write_json(results_dir / "summary.json", summary)
    with (results_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary
