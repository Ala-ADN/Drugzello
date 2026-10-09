"""Validation-only search, frozen selection, then held-out evaluation.

Two prespecified selectors: minimum validation MAE, and minimum validation MAE
among models whose BOTH channel-mean signed ablations are nonnegative. Neither
reads outer test metrics. Deployment uses the second selector's validation votes.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import yaml

from ..data.data_loader import MolecularDataLoader
from ..data.graphs import GraphDataset, fit_target_scaler, transform_targets
from ..data.scaffold import validate_scaffold_manifest
from ..utils.config import MEGANConfig
from .experiment import (checkpoint, environment_metadata, evaluate, model_kwargs,
                         regression_metrics, seed_everything, train_deployment_model, write_json)
from .megan_architecture import MEGANCore
from .trainer import MEGANTrainer

DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
               "HeavyAtomCount", "LabuteASA", "NumHeteroatoms"]


def load_raw(data_root):
    return GraphDataset([g for g in MolecularDataLoader(data_root).load_esol_dataset()])


def validate_data(data, data_root, manifest):
    if manifest["smiles"] != [g.smiles for g in data]:
        raise ValueError("Dataset order differs from the frozen split manifest")
    if "dataset" in manifest:
        raw = Path(data_root) / "ESOL" / "esol" / "raw" / "delaney-processed.csv"
        if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["dataset"]["raw_csv_sha256"]:
            raise ValueError("Raw target/SMILES checksum differs from the frozen dataset")


def read_candidates(search_path):
    definition = yaml.safe_load(Path(search_path).read_text())
    candidates = []
    for item in definition["candidates"]:
        identifier = item["id"]
        if not identifier.replace("_", "").isalnum():
            raise ValueError("Candidate IDs must contain letters, numbers or underscores")
        values = {**definition["base"], **item["overrides"]}
        config = MEGANConfig(**values)
        if set(values) - set(config.to_dict()):
            raise ValueError("Unknown search hyperparameter")
        candidates.append({"id": identifier, "config": config.to_dict()})
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    return candidates


def choose_trials(trials):
    """Intentionally accepts ONLY validation results; never test measurements."""
    key = lambda t: (t["validation"]["mae"], t["candidate_id"])
    accuracy = min(trials, key=key)
    eligible = [t for t in trials if all(
        x >= 0 for x in t["validation_explanations"]["signed_channel_ablation_mean_logS"])]
    faithful = min(eligible, key=key) if eligible else accuracy
    return {"accuracy": accuracy["candidate_id"], "fidelity_constrained": faithful["candidate_id"],
            "fidelity_constraint_satisfied": bool(eligible)}


def _fit_trial(job):
    """Worker has train/validation IDs only. Outer test IDs are not supplied."""
    torch.set_num_threads(1)
    config = MEGANConfig(**job["candidate"]["config"])
    config.random_seed = job["seed"]
    seed_everything(job["seed"])
    raw = load_raw(job["data_root"])
    scaler = fit_target_scaler(raw, job["train_indices"])
    data = transform_targets(raw, scaler)
    kwargs = model_kwargs(data, config)
    model = MEGANCore(**kwargs)
    trial_dir = Path(job["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (trial_dir / "training.log").open("w", buffering=1) as log, redirect_stdout(log):
        result = MEGANTrainer(config, torch.device("cpu")).train_fold(
            model, DataLoader(Subset(data, job["train_indices"]), batch_size=config.batch_size, shuffle=True),
            DataLoader(Subset(data, job["validation_indices"]), batch_size=config.batch_size, shuffle=False),
            job["fold_idx"])
    model.load_state_dict(result["best_model_state"])
    y, p, explanations = evaluate(model, data, job["validation_indices"], scaler, config.batch_size)
    metadata = {"fold_idx": job["fold_idx"], "candidate_id": job["candidate"]["id"],
                "seed": job["seed"], "config": config.to_dict(), "best_epoch": result["best_epoch"],
                "selection_partition": "validation", "selection_metric": "mae",
                "search_provenance_sha256": job["provenance"]}
    path = trial_dir / "model.pth"
    torch.save(checkpoint(model, kwargs, scaler, metadata), path)
    output = {**metadata, "architecture_version": model.architecture_version,
              "epochs_run": len(result["val_maes"]),
              "validation": regression_metrics(y, p), "validation_explanations": explanations,
              "normalization": scaler, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "seconds": time.monotonic() - started}
    write_json(trial_dir / "history.json", {"training": result["train_losses"],
               "validation": result["val_losses"], "validation_mae_standardized": result["val_maes"]})
    write_json(trial_dir / "result.json", output)
    return output


def descriptor_features(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    descriptors = np.asarray([[getattr(Descriptors, name)(mol) for name in DESCRIPTORS] for mol in mols])
    if not np.isfinite(descriptors).all():
        raise ValueError("Nonfinite molecular descriptor")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = np.asarray([generator.GetFingerprintAsNumPy(mol) for mol in mols])
    return descriptors, np.concatenate([descriptors, fingerprints], axis=1)


def baseline_candidates(seed):
    # Fixed in source before search; each family selects on inner validation MAE.
    return [
        ("descriptor_svr_c10", "descriptor_svr", "descriptors", make_pipeline(StandardScaler(), SVR(C=10, gamma="scale"))),
        ("descriptor_svr_c100", "descriptor_svr", "descriptors", make_pipeline(StandardScaler(), SVR(C=100, gamma="scale"))),
        ("descriptor_svr_c100_gamma001", "descriptor_svr", "descriptors", make_pipeline(StandardScaler(), SVR(C=100, gamma=0.01))),
        ("descriptor_hgb_leaf10", "descriptor_hgb", "descriptors", HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1, early_stopping=False, random_state=seed)),
        ("descriptor_hgb_leaf20", "descriptor_hgb", "descriptors", HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1, early_stopping=False, random_state=seed)),
        ("combined_rf_leaf1", "descriptor_morgan_rf", "combined", RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=0.5, n_jobs=1, random_state=seed)),
        ("combined_rf_leaf2", "descriptor_morgan_rf", "combined", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=1, random_state=seed)),
    ]


def fit_baselines(data, manifest, save_dir, results_dir):
    descriptors, combined = descriptor_features(manifest["smiles"])
    features = {"descriptors": descriptors, "combined": combined}
    targets = np.asarray([float(g.y.item()) for g in data])
    trials = []
    selections = {}
    for fold in manifest["folds"]:
        train, val = fold["train_indices"], fold["validation_indices"]
        i = fold["fold_idx"]
        for identifier, family, encoding, model in baseline_candidates(manifest["seed"] + i):
            model.fit(features[encoding][train], targets[train])
            pred = model.predict(features[encoding][val])
            path = Path(save_dir) / f"fold_{i}" / f"{identifier}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            trials.append({"fold_idx": i, "candidate_id": identifier, "family": family,
                           "feature_set": encoding, "validation": regression_metrics(targets[val], pred),
                           "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        selections[str(i)] = {family: min([t for t in trials if t["fold_idx"] == i and t["family"] == family],
                                     key=lambda t: (t["validation"]["mae"], t["candidate_id"]))["candidate_id"]
                               for family in {t["family"] for t in trials}}
    write_json(Path(results_dir) / "baseline_trials.json", trials)
    return selections


def run_search(data_root, manifest, search_path, save_dir, results_dir, workers=3):
    validate_scaffold_manifest(manifest)
    data = load_raw(data_root)
    validate_data(data, data_root, manifest)
    results_dir, save_dir = Path(results_dir), Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidates(search_path)
    environment = environment_metadata()
    plan = {"candidates": candidates, "source_hashes": environment["source_files_sha256"],
            "ablation_candidates": [name for name in ("ordinal_long", "embedding_joint_32", "embedding_split_32")
                                     if name in {c["id"] for c in candidates}],
            "splits_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "selection_rules": {"accuracy": "minimum inner validation MAE",
                "fidelity_constrained": "minimum inner validation MAE among trials with both channel-mean signed validation ablations >= 0; fallback to accuracy if none"},
            "deployment_rule": "fidelity-constrained selector; mode of its per-fold candidate IDs, ties by mean validation MAE across folds; refit epoch count is median selected epoch count of that candidate across folds",
            "test_use": "outer tests previously inspected for the baseline; no test scores enter this search or its selection"}
    provenance = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["provenance_sha256"] = provenance
    plan_path = results_dir / "search_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing search has a different frozen plan/source; use a new output directory")
    write_json(plan_path, plan)
    write_json(results_dir / "splits.json", manifest)
    write_json(results_dir / "environment.json", environment)
    trials, jobs = [], []
    for fold in manifest["folds"]:
        for candidate in candidates:
            directory = save_dir / f"fold_{fold['fold_idx']}" / candidate["id"]
            result_path = directory / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["search_provenance_sha256"] != provenance:
                    raise ValueError("Stale trial provenance")
                if hashlib.sha256((directory / "model.pth").read_bytes()).hexdigest() != result["checkpoint_sha256"]:
                    raise ValueError("Trial checkpoint checksum mismatch")
                trials.append(result)
                continue
            jobs.append({"data_root": str(Path(data_root).resolve()), "candidate": candidate,
                         "fold_idx": fold["fold_idx"], "seed": manifest["seed"] + fold["fold_idx"],
                         "train_indices": fold["train_indices"], "validation_indices": fold["validation_indices"],
                         "trial_dir": str(directory.resolve()), "provenance": provenance})
    # Process isolation prevents shared RNG/model state from corrupting trials.
    def completed(result):
        trials.append(result)
        write_json(results_dir / "trials.json", sorted(trials, key=lambda t: (t["fold_idx"], t["candidate_id"])))
        print(f"Completed {len(trials)}/{len(candidates)*len(manifest['folds'])}: fold {result['fold_idx']} {result['candidate_id']}, validation MAE {result['validation']['mae']:.4f}, fidelity {result['validation_explanations']['fidelity_star_logS']:.4f}", flush=True)
    if workers == 1:
        for job in jobs:
            completed(_fit_trial(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_fit_trial, job): job for job in jobs}
            for future in as_completed(futures):
                completed(future.result())
    trials.sort(key=lambda t: (t["fold_idx"], t["candidate_id"]))
    write_json(results_dir / "trials.json", trials)
    selection = {str(f["fold_idx"]): choose_trials([t for t in trials if t["fold_idx"] == f["fold_idx"]]) for f in manifest["folds"]}
    baseline_selection = fit_baselines(data, manifest, save_dir, results_dir)
    # Freeze ALL choices before any outer test evaluation.
    frozen = {"megan": selection, "baselines": baseline_selection, "search_provenance_sha256": provenance}
    write_json(results_dir / "selection.json", frozen)
    with (results_dir / "validation_trials.csv").open("w", newline="") as handle:
        fields = ["fold_idx", "candidate_id", "seed", "best_epoch", "epochs_run", "validation_mae_logS", "validation_fidelity_star_logS", "negative_channel_delta_logS", "positive_channel_delta_logS"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            deltas = t["validation_explanations"]["signed_channel_ablation_mean_logS"]
            writer.writerow({**{k: t[k] for k in fields[:5]}, "validation_mae_logS": t["validation"]["mae"],
                             "validation_fidelity_star_logS": t["validation_explanations"]["fidelity_star_logS"],
                             "negative_channel_delta_logS": deltas[0], "positive_channel_delta_logS": deltas[1]})
    return trials, frozen


def evaluate_selections(data_root, save_dir, results_dir, refit=False):
    """Only runs once all choices exist. No hyperparameter selection occurs here."""
    results_dir, save_dir = Path(results_dir), Path(save_dir)
    manifest = json.loads((results_dir / "splits.json").read_text())
    validate_scaffold_manifest(manifest)
    trials = json.loads((results_dir / "trials.json").read_text())
    selections = json.loads((results_dir / "selection.json").read_text())
    plan = json.loads((results_dir / "search_plan.json").read_text())
    expected_trials = len(plan["candidates"]) * len(manifest["folds"])
    if len(trials) != expected_trials or selections["search_provenance_sha256"] != plan["provenance_sha256"]:
        raise ValueError("Incomplete or inconsistent validation search")
    data = load_raw(data_root)
    validate_data(data, data_root, manifest)
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        expected = choose_trials([t for t in trials if t["fold_idx"] == i])
        if selections["megan"][str(i)] != expected:
            raise ValueError("Selection disagrees with the prespecified validation-only rule")
    descriptors, combined = descriptor_features(manifest["smiles"])
    features = {"descriptors": descriptors, "combined": combined}
    baseline_trials = json.loads((results_dir / "baseline_trials.json").read_text())
    targets = np.asarray([float(g.y.item()) for g in data])
    rows, folds = [], []
    for fold in manifest["folds"]:
        i, test = fold["fold_idx"], fold["test_indices"]
        metrics = {"fold_idx": i, "test_size": len(test), "models": {}}
        prediction = {"training_mean": np.repeat(targets[fold["train_indices"]].mean(), len(test))}
        evaluated = {"megan_" + selector: selections["megan"][str(i)][selector]
                     for selector in ("accuracy", "fidelity_constrained")}
        evaluated.update({"ablation_" + identifier: identifier for identifier in plan["ablation_candidates"]})
        for name, identifier in evaluated.items():
            trial = next(t for t in trials if t["fold_idx"] == i and t["candidate_id"] == identifier)
            path = save_dir / f"fold_{i}" / identifier / "model.pth"
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint checksum mismatch")
            state = torch.load(path, weights_only=True, map_location="cpu")
            model = MEGANCore(**state["model_kwargs"])
            model.load_state_dict(state["model_state_dict"], strict=True)
            normalized = transform_targets(data, state["normalization"])
            _, p, explanations = evaluate(model, normalized, test, state["normalization"], trial["config"]["batch_size"])
            prediction[name] = p
            metrics["models"][name] = {"candidate_id": identifier, "test": regression_metrics(targets[test], p),
                                     "explanations": explanations, "checkpoint_sha256": trial["checkpoint_sha256"],
                                     "validation_mae_logS": trial["validation"]["mae"]}
        for family, identifier in selections["baselines"][str(i)].items():
            trial = next(t for t in baseline_trials if t["fold_idx"] == i and t["candidate_id"] == identifier)
            path = save_dir / f"fold_{i}" / f"{identifier}.joblib"
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial["artifact_sha256"]:
                raise ValueError("Baseline artifact checksum mismatch")
            model = joblib.load(path)  # Only locally generated, checksummed artifacts.
            prediction[family] = model.predict(features[trial["feature_set"]][test])
            metrics["models"][family] = {"candidate_id": identifier, "test": regression_metrics(targets[test], prediction[family])}
        metrics["models"]["training_mean"] = {"test": regression_metrics(targets[test], prediction["training_mean"])}
        for j, row_idx in enumerate(test):
            row = {"fold_idx": i, "row_idx": row_idx, "smiles": manifest["smiles"][row_idx],
                   "scaffold_id": manifest["scaffold_ids"][row_idx], "true_logS": float(targets[row_idx])}
            row.update({name + "_logS": float(values[j]) for name, values in prediction.items()})
            rows.append(row)
        folds.append(metrics)
    summary = {"metric_unit": "logS", "fold_count": len(folds), "selection_partition": "inner validation only", "models": {}}
    for name in folds[0]["models"]:
        maes = [f["models"][name]["test"]["mae"] for f in folds]
        summary["models"][name] = {"test_mae_mean": float(np.mean(maes)), "test_mae_std": float(np.std(maes)),
                                  "pooled_test": regression_metrics([r["true_logS"] for r in rows], [r[name+"_logS"] for r in rows])}
        if "explanations" in folds[0]["models"][name]:
            summary["models"][name]["per_fold_fidelity_star_logS"] = [f["models"][name]["explanations"]["fidelity_star_logS"] for f in folds]
    write_json(results_dir / "fold_metrics.json", folds)
    write_json(results_dir / "summary.json", summary)
    with (results_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (results_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_idx", "test_size", "model", "candidate_id", "test_mae_logS", "test_rmse_logS", "test_r2", "fidelity_star_logS"])
        writer.writeheader()
        for f in folds:
            for name, scores in f["models"].items():
                writer.writerow({"fold_idx": f["fold_idx"], "test_size": f["test_size"], "model": name,
                                 "candidate_id": scores.get("candidate_id"), "test_mae_logS": scores["test"]["mae"],
                                 "test_rmse_logS": scores["test"]["rmse"], "test_r2": scores["test"]["r2"],
                                 "fidelity_star_logS": scores.get("explanations", {}).get("fidelity_star_logS")})
    if refit:
        refit_selected(data, trials, selections, save_dir, results_dir, manifest["seed"])
    return summary


def refit_selected(data, trials, selections, save_dir, results_dir, seed):
    from collections import Counter
    counts = Counter(s["fidelity_constrained"] for s in selections["megan"].values())
    identifier = min(counts, key=lambda candidate: (-counts[candidate],
                         np.mean([t["validation"]["mae"] for t in trials if t["candidate_id"] == candidate]), candidate))
    matching = [t for t in trials if t["candidate_id"] == identifier]
    epochs = max(1, int(np.median([t["best_epoch"] + 1 for t in matching])))
    config = MEGANConfig(**matching[0]["config"])
    config.random_seed = seed
    path = Path(save_dir) / "model.pth"
    train_deployment_model(data, config, path, epochs, seed)
    config.save(str(Path(results_dir) / "deployment_config.yaml"))
    write_json(Path(results_dir) / "deployment.json", {"candidate_id": identifier, "selector": "fidelity_constrained",
                "rule": "validation-selection mode, ties by mean validation MAE", "epochs": epochs,
                "training_rows": len(data), "seed": seed, "independent_test_score": None,
                "architecture_version": MEGANCore(**model_kwargs(data, config)).architecture_version,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "remote_release": None})
