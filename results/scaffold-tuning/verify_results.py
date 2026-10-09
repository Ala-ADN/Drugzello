"""Replay local artifacts and independently check the published result tables.

Run from the repository root with the recorded research environment. Checkpoints
and joblib files must be the locally generated, trusted artifacts from this run.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys
import atexit
from contextlib import ExitStack
import tarfile
import tempfile

import joblib
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "backend/models/trained/scaffold-tuned"
TEMP = ExitStack()
atexit.register(TEMP.close)


def unpack(archive, directory, expected_names=None):
    """Read trusted local archives without allowing symlinks or path traversal."""
    with tarfile.open(archive, "r:gz") as handle:
        if expected_names is not None:
            assert set(handle.getnames()) == set(expected_names)
        for member in handle:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.extractfile(member).read())


SOURCE_ROOT = ROOT
source_hashes = json.loads((REPORT / "search_plan.json").read_text())["source_hashes"]
if not all((ROOT / path).is_file() and hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == checksum
           for path, checksum in source_hashes.items()):
    SOURCE_ROOT = Path(TEMP.enter_context(tempfile.TemporaryDirectory(prefix="drugzello-source-")))
    unpack(REPORT / "source.tar.gz", SOURCE_ROOT, source_hashes)
if not (WEIGHTS / "fold_0").is_dir():
    original_weights = WEIGHTS
    WEIGHTS = Path(TEMP.enter_context(tempfile.TemporaryDirectory(prefix="drugzello-folds-")))
    unpack(original_weights / "folds.tar.gz", WEIGHTS)
    (WEIGHTS / "model.pth").write_bytes((original_weights / "model.pth").read_bytes())

sys.path.insert(0, str(SOURCE_ROOT / "backend"))
from src.data.graphs import fit_target_scaler, transform_targets
from src.data.scaffold import validate_scaffold_manifest
from src.models.megan_architecture import MEGANCore
from src.models.tuning import descriptor_features, load_raw, validate_data


def read(name):
    return json.loads((REPORT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metrics(y, p, recorded):
    # Direct formulas, independent of the experiment's sklearn metric helper.
    y, p = np.asarray(y), np.asarray(p)
    actual = {"mae": np.abs(y - p).mean(),
              "rmse": np.sqrt(np.square(y - p).mean()),
              "r2": 1 - np.square(y - p).sum() / np.square(y - y.mean()).sum()}
    for key, value in actual.items():
        np.testing.assert_allclose(value, recorded[key], rtol=0, atol=1e-12)


def main():
    torch.set_num_threads(1)
    manifest, plan, selections = read("splits.json"), read("search_plan.json"), read("selection.json")
    validate_scaffold_manifest(manifest)
    provenance = plan.pop("provenance_sha256")
    assert hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() == provenance
    assert selections["search_provenance_sha256"] == provenance
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == plan["splits_sha256"]
    for path, checksum in plan["source_hashes"].items():
        assert sha(SOURCE_ROOT / path) == checksum, path
    data = load_raw(ROOT / "backend/data")
    validate_data(data, ROOT / "backend/data", manifest)
    targets = np.asarray([float(g.y.item()) for g in data])
    trials, baselines = read("trials.json"), read("baseline_trials.json")
    assert len(trials) == 40 and len(baselines) == 35
    reproduction = []
    old_history = json.loads((ROOT / "results/scaffold-esol/training_history.json").read_text())
    for trial in trials:
        fold = manifest["folds"][trial["fold_idx"]]
        directory = WEIGHTS / f"fold_{trial['fold_idx']}" / trial["candidate_id"]
        assert sha(directory / "model.pth") == trial["checkpoint_sha256"]
        assert trial["search_provenance_sha256"] == provenance
        assert trial["seed"] == manifest["seed"] + trial["fold_idx"]
        assert trial["normalization"] == fit_target_scaler(data, fold["train_indices"])
        history = json.loads((directory / "history.json").read_text())
        assert int(np.argmin(history["validation_mae_standardized"])) == trial["best_epoch"]
        if trial["candidate_id"] == "ordinal_long":
            old = old_history[trial["fold_idx"]]
            for new_key, old_key in (("training", "train_losses"), ("validation", "validation_losses")):
                for loss, values in old[old_key].items():
                    assert history[new_key][loss][:len(values)] == values
            assert history["validation_mae_standardized"][:30] == old["validation_mae_standardized"]
            reproduction.append({"fold_idx": trial["fold_idx"], "epochs_compared": 30,
                                 "all_training_and_validation_losses_identical": True,
                                 "validation_mae_max_absolute_difference": 0.0})
    for trial in baselines:
        path = WEIGHTS / f"fold_{trial['fold_idx']}" / f"{trial['candidate_id']}.joblib"
        assert sha(path) == trial["artifact_sha256"]
    key = lambda t: (t["validation"]["mae"], t["candidate_id"])
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        candidates = [t for t in trials if t["fold_idx"] == i]
        eligible = [t for t in candidates if min(t["validation_explanations"]["signed_channel_ablation_mean_logS"]) >= 0]
        assert selections["megan"][str(i)] == {
            "accuracy": min(candidates, key=key)["candidate_id"],
            "fidelity_constrained": min(eligible or candidates, key=key)["candidate_id"],
            "fidelity_constraint_satisfied": bool(eligible)}
        for family, candidate in selections["baselines"][str(i)].items():
            assert min([t for t in baselines if t["fold_idx"] == i and t["family"] == family], key=key)["candidate_id"] == candidate
    rows = list(csv.DictReader((REPORT / "predictions.csv").open()))
    assert len(rows) == len(data) == 1128
    assert sorted(int(r["row_idx"]) for r in rows) == list(range(len(data)))
    for row in rows:
        idx, i = int(row["row_idx"]), int(row["fold_idx"])
        assert idx in manifest["folds"][i]["test_indices"]
        assert row["smiles"] == manifest["smiles"][idx]
        assert row["scaffold_id"] == manifest["scaffold_ids"][idx]
        assert float(row["true_logS"]) == targets[idx]
    folds, summary = read("fold_metrics.json"), read("summary.json")
    descriptors, combined = descriptor_features(manifest["smiles"])
    features = {"descriptors": descriptors, "combined": combined}
    ablation_rows, max_prediction_difference = [], 0.0
    for f in folds:
        i = f["fold_idx"]
        fold = manifest["folds"][i]
        test = fold["test_indices"]
        subset = [r for r in rows if int(r["fold_idx"]) == i]
        assert [int(r["row_idx"]) for r in subset] == test
        for name, metrics in f["models"].items():
            saved_predictions = np.asarray([float(r[name + "_logS"]) for r in subset])
            check_metrics(targets[test], saved_predictions, metrics["test"])
            if "explanations" in metrics:
                state = torch.load(WEIGHTS / f"fold_{i}" / metrics["candidate_id"] / "model.pth",
                                   weights_only=True, map_location="cpu")
                model = MEGANCore(**state["model_kwargs"])
                model.load_state_dict(state["model_state_dict"], strict=True)
                model.eval()
                scaler = state["normalization"]
                normalized = transform_targets(data, scaler)
                predictions, deltas = [], []
                with torch.no_grad():
                    for batch in DataLoader(Subset(normalized, test), batch_size=64, shuffle=False):
                        output, _, nodes = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                                 return_explanations=True)
                        predictions.extend((output.flatten().numpy() * scaler["std"] + scaler["mean"]).tolist())
                        changes = []
                        for channel in (0, 1):
                            mask = torch.ones_like(nodes)
                            mask[:, channel] = 0
                            ablated = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, importance_mask=mask)
                            delta = ablated - output if channel == 0 else output - ablated
                            changes.append(delta.flatten())
                        deltas.extend(torch.stack(changes, dim=1).numpy().tolist())
                deltas = np.asarray(deltas) * scaler["std"]
                exp = metrics["explanations"]
                np.testing.assert_allclose(deltas.mean(0), exp["signed_channel_ablation_mean_logS"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(deltas.mean(), exp["fidelity_star_logS"], rtol=0, atol=1e-12)
                np.testing.assert_array_equal((deltas >= 0).mean(0), exp["channel_polarity_fraction"])
                assert float((deltas >= 0).all(1).mean()) == exp["both_channels_correct_fraction"]
                ablation_rows.extend({"fold_idx": i, "row_idx": idx, "model": name,
                                      "negative_channel_signed_delta_logS": float(d[0]),
                                      "positive_channel_signed_delta_logS": float(d[1])} for idx, d in zip(test, deltas))
            elif name == "training_mean":
                predictions = np.repeat(targets[fold["train_indices"]].mean(), len(test))
            else:
                trial = next(t for t in baselines if t["fold_idx"] == i and t["candidate_id"] == metrics["candidate_id"])
                model = joblib.load(WEIGHTS / f"fold_{i}" / f"{trial['candidate_id']}.joblib")
                predictions = model.predict(features[trial["feature_set"]][test])
            difference = float(np.max(np.abs(np.asarray(predictions) - saved_predictions)))
            max_prediction_difference = max(max_prediction_difference, difference)
            assert difference == 0.0, (i, name, difference)
    for name, metrics in summary["models"].items():
        check_metrics([float(r["true_logS"]) for r in rows], [float(r[name + "_logS"]) for r in rows], metrics["pooled_test"])
        maes = [f["models"][name]["test"]["mae"] for f in folds]
        assert float(np.mean(maes)) == metrics["test_mae_mean"]
        assert float(np.std(maes)) == metrics["test_mae_std"]
    deployment = read("deployment.json")
    assert sha(WEIGHTS / "model.pth") == deployment["checkpoint_sha256"]
    assert deployment["independent_test_score"] is None
    with (REPORT / "channel_ablations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    evidence = {"date": "2026-10-09", "status": "passed", "molecules": len(rows),
                "source_files_sha256_checked": len(plan["source_hashes"]),
                "megan_checkpoint_sha256_checked": len(trials),
                "baseline_artifact_sha256_checked": len(baselines),
                "validation_only_selection_recomputed": True,
                "scaffold_disjointness_and_one_test_prediction_per_molecule_checked": True,
                "all_fold_and_pooled_metrics_recomputed_with_direct_formulas": True,
                "model_tracks_replayed": len(summary["models"]),
                "held_out_predictions_replayed": len(rows) * len(summary["models"]),
                "max_absolute_prediction_replay_difference_logS": max_prediction_difference,
                "channel_ablation_rows_replayed": len(ablation_rows),
                "deployment_checkpoint_sha256_checked": True,
                "ordinal_first_30_epochs_reproduction": reproduction}
    (REPORT / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
