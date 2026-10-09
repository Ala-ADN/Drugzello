"""Auditable nested CV and checkpoint export; no test-dependent model selection."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from ..data.graphs import FEATURE_SCHEMA, fit_target_scaler, transform_targets
from ..data.scaffold import validate_scaffold_manifest
from ..utils.config import MEGANConfig
from .megan_architecture import ARCHITECTURE_VERSION, MEGANCore
from .trainer import MEGANLoss, MEGANTrainer

CHECKPOINT_FORMAT = "drugzello-megan-v2"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def model_kwargs(dataset, config):
    return dict(in_channels=dataset.num_node_features, edge_dim=dataset.num_edge_features,
                out_channels=1, hidden_channels=config.hidden_channels,
                num_layers=config.num_layers, K=config.K, heads_gat=config.heads_gat,
                use_edge_features=config.use_edge_features, dropout=config.dropout,
                layer_norm=config.layer_norm, residual=config.residual,
                feature_encoding=config.feature_encoding, embedding_dim=config.embedding_dim)


def regression_metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def checkpoint(model, kwargs, scaler, metadata):
    return {"format": CHECKPOINT_FORMAT, "architecture_version": model.architecture_version,
            "feature_schema": FEATURE_SCHEMA, "model_kwargs": kwargs,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "normalization": scaler, "metadata": metadata}


def evaluate(model, dataset, indices, scaler, batch_size=32, device="cpu"):
    """Same forward with optional channel masks; report prediction/fidelity in logS."""
    model.eval()
    true, pred, signed_deltas = [], [], []
    node_scores, edge_scores = [], []
    with torch.no_grad():
        for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            output, edges, nodes = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                          return_explanations=True)
            true.extend((batch.y.flatten().cpu().numpy() * scaler["std"] + scaler["mean"]).tolist())
            pred.extend((output.flatten().cpu().numpy() * scaler["std"] + scaler["mean"]).tolist())
            node_scores.extend(nodes.mean(1).cpu().tolist())
            edge_scores.extend(edges.mean(1).cpu().tolist())
            deltas = []
            for k in range(model.K):
                mask = torch.ones_like(nodes)
                mask[:, k] = 0
                ablated = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                importance_mask=mask)
                delta = (ablated - output) if k == 0 else (output - ablated)
                deltas.append(delta.flatten())
            signed_deltas.extend(torch.stack(deltas, 1).cpu().numpy().tolist())
    signed_deltas = np.asarray(signed_deltas) * scaler["std"]
    explanations = {"signed_channel_ablation_mean_logS": signed_deltas.mean(0).tolist(),
                    "fidelity_star_logS": float(signed_deltas.mean()) if model.K == 2 else None,
                    "channel_polarity_fraction": (signed_deltas >= 0).mean(0).tolist(),
                    "both_channels_correct_fraction": float((signed_deltas >= 0).all(1).mean()) if model.K == 2 else None,
                    "mean_node_importance": float(np.mean(node_scores)),
                    "mean_edge_importance": float(np.mean(edge_scores)) if edge_scores else 0.0}
    return true, pred, explanations


def environment_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"])
    except (OSError, subprocess.CalledProcessError):
        commit, diff = None, b""
    root = Path(__file__).resolve().parents[3]
    source_paths = [p for directory in ("src", "scripts", "core", "services", "api")
                    for p in (root / "backend" / directory).rglob("*.py")]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_paths)}
    return {"python": platform.python_version(), "source_commit": commit,
            "source_files_sha256": source_hashes,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "architecture_version": ARCHITECTURE_VERSION, "feature_schema": FEATURE_SCHEMA}


def run_experiment(dataset, config, manifest, save_dir, results_dir, device="cpu", refit=False,
                   random_forest=True):
    validate_scaffold_manifest(manifest)
    if manifest['smiles'] != [g.smiles for g in dataset]:
        raise ValueError("Split molecule order differs from the supplied dataset")
    save_dir, results_dir = Path(save_dir), Path(results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(results_dir / "config.yaml"))
    write_json(results_dir / "splits.json", manifest)
    write_json(results_dir / "environment.json", environment_metadata())
    # Labels and molecule order are immutable audit inputs, BEFORE transformation.
    targets = np.asarray([float(g.y.item()) for g in dataset])
    fingerprint = None
    if random_forest:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fingerprint = np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
                                  for s in manifest["smiles"]])
    summaries, predictions, histories = [], [], []
    kwargs = model_kwargs(dataset, config)
    for fold in manifest["folds"]:
        i, seed = fold["fold_idx"], manifest["seed"] + fold["fold_idx"]
        seed_everything(seed)
        train, val, test = [fold[label + "_indices"] for label in ("train", "validation", "test")]
        scaler = fit_target_scaler(dataset, train)
        normalized = transform_targets(dataset, scaler)
        model = MEGANCore(**kwargs).to(device)
        trainer = MEGANTrainer(config, torch.device(device))
        fitted = trainer.train_fold(
            model, DataLoader(Subset(normalized, train), batch_size=config.batch_size, shuffle=True),
            DataLoader(Subset(normalized, val), batch_size=config.batch_size, shuffle=False), i
        )
        model.load_state_dict(fitted["best_model_state"])
        metadata = {"fold_idx": i, "seed": seed, "config": config.to_dict(),
                    "selection_partition": "validation", "selection_metric": "mae",
                    "best_epoch": fitted["best_epoch"], "split_policy": manifest["policy"]}
        model_path = save_dir / f"fold_{i}_model.pth"
        torch.save(checkpoint(model, kwargs, scaler, metadata), model_path)
        y_true, y_pred, explanations = evaluate(model, normalized, test, scaler,
                                               config.batch_size, device)
        mean_pred = np.repeat(targets[train].mean(), len(test))
        rf_pred = None
        if fingerprint is not None:
            forest = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
            forest.fit(fingerprint[train], targets[train])
            rf_pred = forest.predict(fingerprint[test])
        metrics = regression_metrics(y_true, y_pred)
        summary = {**metadata, "train_size": len(train), "validation_size": len(val),
                   "test_size": len(test), "normalization": scaler,
                   "validation_mae_logS": fitted["best_val_mae"] * scaler["std"],
                   "test": metrics, "mean_baseline_test": regression_metrics(y_true, mean_pred),
                   "explanations": explanations,
                   "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        if rf_pred is not None:
            summary["morgan_random_forest_test"] = regression_metrics(y_true, rf_pred)
        summaries.append(summary)
        histories.append({"fold_idx": i, "train_losses": fitted["train_losses"],
                          "validation_losses": fitted["val_losses"],
                          "validation_mae_standardized": fitted["val_maes"]})
        for j, row in enumerate(test):
            prediction = {"fold_idx": i, "row_idx": row, "smiles": manifest["smiles"][row],
                          "scaffold_id": manifest["scaffold_ids"][row],
                          "true_logS": y_true[j], "predicted_logS": y_pred[j],
                          "train_mean_prediction_logS": float(mean_pred[j])}
            if rf_pred is not None:
                prediction["morgan_rf_prediction_logS"] = float(rf_pred[j])
            predictions.append(prediction)
        print(f"Fold {i}: held-out test MAE={metrics['mae']:.4f} logS")
        # Write progress after each completed fold for recoverability.
        write_json(results_dir / "fold_metrics.json", summaries)
        write_json(results_dir / "training_history.json", histories)
    aggregate = {"metric_unit": "logS", "test_selection": "never used for checkpoint selection",
                 "fold_count": len(summaries),
                 "test_mae_mean": float(np.mean([f["test"]["mae"] for f in summaries])),
                 "test_mae_std": float(np.std([f["test"]["mae"] for f in summaries])),
                 "pooled_test": regression_metrics([r["true_logS"] for r in predictions],
                                                   [r["predicted_logS"] for r in predictions])}
    write_json(results_dir / "summary.json", aggregate)
    with (results_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (results_dir / "fold_metrics.csv").open("w", newline="") as handle:
        fields = ["fold_idx", "seed", "train_size", "validation_size", "test_size",
                  "best_epoch", "test_mae_logS", "test_rmse_logS", "test_r2",
                  "mean_baseline_mae_logS", "morgan_rf_mae_logS", "fidelity_star_logS"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for f in summaries:
            writer.writerow({**{k: f[k] for k in fields[:6]},
                             "test_mae_logS": f["test"]["mae"], "test_rmse_logS": f["test"]["rmse"],
                             "test_r2": f["test"]["r2"],
                             "mean_baseline_mae_logS": f["mean_baseline_test"]["mae"],
                             "morgan_rf_mae_logS": f.get("morgan_random_forest_test", {}).get("mae"),
                             "fidelity_star_logS": f["explanations"]["fidelity_star_logS"]})
    if refit:
        # Test reports are finalized BEFORE fitting deployment weights on all data.
        epochs = max(1, int(np.median([f["best_epoch"] + 1 for f in summaries])))
        train_deployment_model(dataset, config, save_dir / "model.pth", epochs,
                               manifest["seed"], device)
    return summaries, aggregate


def train_deployment_model(dataset, config, path, epochs, seed=42, device="cpu"):
    """Refit for the demo. This checkpoint has NO independent ESOL test score."""
    seed_everything(seed)
    scaler = fit_target_scaler(dataset, range(len(dataset)))
    normalized = transform_targets(dataset, scaler)
    kwargs = model_kwargs(dataset, config)
    model = MEGANCore(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss_fn = MEGANLoss(config.gamma_exp, config.beta_sparsity, config.delta_decor)
    for _ in range(epochs):
        model.train()
        for batch in DataLoader(normalized, batch_size=config.batch_size, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad()
            MEGANTrainer.optimize_batch(model, batch, optimizer, loss_fn, config.training_schedule)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(model, kwargs, scaler,
                          {"purpose": "all-data deployment refit", "epochs": epochs,
                           "seed": seed, "config": config.to_dict(), "test_score": None}), path)
