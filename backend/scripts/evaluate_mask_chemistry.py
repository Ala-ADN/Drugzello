"""Evaluate frozen scaffold MEGAN masks; run from the repository root."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from src.data.graphs import FEATURE_SCHEMA, smiles_to_graph
from src.evaluation.masks import chemical_atoms, diagnose, save_results, sha
from src.models.experiment import CHECKPOINT_FORMAT
from src.models.megan_architecture import MEGANCore


def extract(model, graphs):
    captured = []
    handle = model.node_importance_mlp.register_forward_pre_hook(lambda layer, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            batch = Batch.from_data_list(graphs)
            output, edges, nodes = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_explanations=True)
            h = captured[0]
            pooled = torch.cat([global_add_pool(h * nodes[:, k:k+1], batch.batch) for k in range(model.K)], 1)
            np.testing.assert_allclose(model.classifier(pooled).numpy(), output.numpy(), rtol=0, atol=1e-6)
        return output.numpy().flatten(), nodes.numpy(), edges.numpy(), h.numpy(), pooled.numpy(), batch.ptr.numpy()
    finally:
        handle.remove()


def load_checkpoint(cohort, fold):
    if cohort == "scaffold_prediction_only":
        path = ROOT / f"backend/models/trained/scaffold-megan-no-aux/fold_{fold}_model.pth"
        data = path.read_bytes()
    else:
        path = ROOT / "backend/models/trained/scaffold-tuned/folds.tar.gz"
        with tarfile.open(path) as archive:
            data = archive.extractfile(f"fold_{fold}/embedding_joint_32/model.pth").read()
    import hashlib
    cp = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    assert cp["format"] == CHECKPOINT_FORMAT and cp["feature_schema"] == FEATURE_SCHEMA
    model = MEGANCore(**cp["model_kwargs"])
    assert model.architecture_version == cp["architecture_version"]
    model.load_state_dict(cp["model_state_dict"], strict=True)
    return model.eval(), cp, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["scaffold_prediction_only", "scaffold_joint_auxiliary"], required=True)
    parser.add_argument("--results-dir", default="results/mask-chemistry")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = ROOT / "experiments/mask-chemistry/protocol.json"
    manifest_path = ROOT / "results/scaffold-tuning/splits.json"
    manifest = json.loads(manifest_path.read_text())
    source_result = "scaffold-megan-no-aux" if args.cohort == "scaffold_prediction_only" else "scaffold-tuning"
    prediction_path = ROOT / f"results/{source_result}/predictions.csv"
    expected = {int(r["row_idx"]): r for r in csv.DictReader(prediction_path.open())}
    pred_key = "predicted_logS" if args.cohort == "scaffold_prediction_only" else "ablation_embedding_joint_32_logS"
    recorded_env = json.loads((ROOT / f"results/{source_result}/environment.json").read_text())
    assert sha(ROOT / "backend/src/models/megan_architecture.py") == recorded_env["source_files_sha256"]["backend/src/models/megan_architecture.py"]
    graphs = [smiles_to_graph(s) for s in manifest["smiles"]]
    for graph, smiles in zip(graphs, manifest["smiles"]):
        mol, *_ = chemical_atoms(smiles)
        assert graph.x[:, 0].tolist() == [a.GetAtomicNum() for a in mol.GetAtoms()]
    rows, atoms, folds = [], [], []
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        model, cp, digest = load_checkpoint(args.cohort, i)
        scale, mean = cp["normalization"]["std"], cp["normalization"]["mean"]
        # No chemistry labels or test masks enter channel orientation.
        channel_deltas = []
        for start in range(0, len(fold["train_indices"]), 32):
            ids = fold["train_indices"][start:start+32]
            out, nodes, _, h, pooled, ptr = extract(model, [graphs[j] for j in ids])
            with torch.no_grad():
                for k in range(model.K):
                    ablated = pooled.copy()
                    ablated[:, k * h.shape[1]:(k+1) * h.shape[1]] = 0
                    delta = (out - model.classifier(torch.from_numpy(ablated)).numpy().flatten()) * scale
                    if k == 0:
                        block = np.zeros((len(ids), model.K))
                    block[:, k] = delta
            channel_deltas.extend(block)
        training_delta = np.asarray(channel_deltas)
        signs = np.sign(training_delta.mean(0))
        torch.manual_seed(1042 + i)
        random_model = MEGANCore(**cp["model_kwargs"]).eval()
        record = {"fold_idx": i, "checkpoint_sha256": digest, "randomization_seed": 1042+i,
                  "channel_orientation_partition": "train", "train_channel_mean_effect_logS": training_delta.mean(0).tolist(),
                  "channel_signs": signs.tolist(), "training_channel_sign_consistency": (training_delta * signs >= 0).mean(0).tolist(),
                  "test_size": len(fold["test_indices"]), "max_prediction_replay_error_logS": 0.0}
        for start in range(0, len(fold["test_indices"]), 32):
            ids = fold["test_indices"][start:start+32]
            out, nodes, edges, h, pooled, ptr = extract(model, [graphs[j] for j in ids])
            _, random_nodes, _, _, _, _ = extract(random_model, [graphs[j] for j in ids])
            def readout(values):
                with torch.no_grad():
                    return model.classifier(torch.from_numpy(values.astype("float32"))).numpy().flatten()
            for b, j in enumerate(ids):
                prediction = float(out[b] * scale + mean)
                error = abs(prediction - float(expected[j][pred_key]))
                assert error < 3e-5, (i, j, error)
                record["max_prediction_replay_error_logS"] = max(record["max_prediction_replay_error_logS"], error)
                sl = slice(ptr[b], ptr[b+1])
                row, atom = diagnose(manifest["smiles"][j], nodes[sl], random_nodes[sl], h[sl], pooled[b],
                                     readout, scale, signs, np.random.default_rng(20261010+j))
                rows.append({"fold_idx": i, "row_idx": j, "smiles": manifest["smiles"][j],
                             "true_logS": float(expected[j]["true_logS"]), "predicted_logS": prediction, **row})
                edge_start = sum(graphs[q].num_edges for q in ids[:b])
                atom.update({"fold_idx": i, "row_idx": j, "smiles": manifest["smiles"][j],
                             "directed_edge_indices": graphs[j].edge_index.t().tolist(),
                             "edge_masks": edges[edge_start:edge_start+graphs[j].num_edges].tolist()})
                atoms.append(atom)
        folds.append(record)
        print(f"{args.cohort}: fold {i} completed; train-calibrated signs {signs.tolist()}", flush=True)
    assert sorted(r["row_idx"] for r in rows) == list(range(len(graphs)))
    evidence = {"protocol_sha256": sha(protocol), "split_manifest_sha256": sha(manifest_path),
                "original_predictions_sha256": sha(prediction_path), "runner_sha256": sha(__file__),
                "evaluation_source_sha256": sha(ROOT / "backend/src/evaluation/masks.py"),
                "runtime": {"torch": torch.__version__, "numpy": np.__version__},
                "encoder_unchanged": True, "checkpoint_selection": "previously frozen inner-validation selection"}
    save_results(ROOT / args.results_dir / args.cohort, args.cohort, rows, atoms, folds, evidence)


if __name__ == "__main__":
    main()
