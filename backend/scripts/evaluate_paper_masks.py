"""Chemical-proxy evaluation of frozen original KGCNN ESOL node masks."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np

import reproduce_paper_esol as original
from src.evaluation.masks import chemical_atoms, diagnose, save_results, sha


def extractor(tf, inner):
    @tf.function(reduce_retracing=True)
    def extract(inputs):
        output, nodes, edges = inner(inputs, training=False, return_importances=True)
        h = inputs[0]
        for layer in inner.attention_layers:
            h, _ = layer([h, inputs[1], inputs[2]])
        pools = [inner.lay_pool_out(h * tf.expand_dims(nodes[:, :, k], -1))
                 for k in range(inner.importance_channels)]
        pooled = inner.lay_concat_out(pools)
        replay = pooled
        for layer in inner.final_layers:
            replay = layer(replay)
        tf.debugging.assert_near(replay, output, atol=2e-6)
        return output, nodes, edges, h, pooled
    return extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/mask-chemistry")
    args = parser.parse_args()
    original.source_setup()
    tf = original.runtime()
    from kgcnn.literature.MEGAN import MEGAN, make_model
    from kgcnn.data.moleculenet import MoleculeNetDataset
    config = original.read(original.UPSTREAM / "original_hyper.json")["model"]["config"]
    protocol = ROOT / "experiments/mask-chemistry/protocol.json"
    manifest_path = ROOT / "results/paper-esol/splits.json"
    manifest = original.read(manifest_path)
    cache_path = ROOT / "results/paper-esol/graphs.npz"
    assert sha(cache_path) == original.read(ROOT / "results/paper-esol/features.json")["graph_cache_sha256"]
    arrays = np.load(cache_path, allow_pickle=False)
    bounds = {field: np.concatenate([[0], np.cumsum(arrays[field + "_lengths"])])
              for field in ("node_attributes", "edge_indices")}
    symbols = MoleculeNetDataset.DEFAULT_NODE_ENCODERS["Symbol"]
    for j, smiles in enumerate(manifest["smiles"]):
        mol, *_ = chemical_atoms(smiles)
        lo, hi = bounds["node_attributes"][j:j+2]
        assert hi - lo == mol.GetNumAtoms()
        # Verify chemical identity and heavy-atom order before motif attribution.
        expected_symbols = np.array([symbols(a.GetSymbol()) for a in mol.GetAtoms()])
        np.testing.assert_array_equal(arrays["node_attributes"][lo:hi, :expected_symbols.shape[1]], expected_symbols)
        lo, hi = bounds["edge_indices"][j:j+2]
        actual_edges = set(map(tuple, arrays["edge_indices"][lo:hi].tolist()))
        expected_edges = {(a, b) for bond in mol.GetBonds() for a, b in
                          [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())]}
        assert actual_edges == expected_edges
    prediction_path = ROOT / "results/paper-esol/predictions.csv"
    expected = {int(r["row_idx"]): r for r in csv.DictReader(prediction_path.open())}
    records = original.read(ROOT / "results/paper-esol/fold_metrics.json")
    rows, atoms, folds = [], [], []
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(records[i]["seed"])
        model = make_model(**config)
        path = ROOT / f"backend/models/trained/paper-esol/fold_{i}.weights.npz"
        assert sha(path) == records[i]["checkpoint_sha256"]
        with np.load(path, allow_pickle=False) as w:
            model.set_weights([w[f"arr_{j}"] for j in range(len(w.files))])
        inner = next(layer for layer in model.layers if isinstance(layer, MEGAN))
        extract = extractor(tf, inner)
        @tf.function(input_signature=[tf.TensorSpec([None, 270], tf.float32)])
        def readout_tensor(values):
            for layer in inner.final_layers:
                values = layer(values)
            return values
        def readout(values):
            return readout_tensor(tf.convert_to_tensor(values, dtype=tf.float32)).numpy().flatten()
        scale = records[i]["normalization"]["std"][0]
        mean = records[i]["normalization"]["mean"][0]
        deltas = []
        for start in range(0, len(fold["train_indices"]), 64):
            ids = fold["train_indices"][start:start+64]
            output, _, _, _, pooled = extract(original.tensors(tf, arrays, ids))
            pooled, output = pooled.numpy(), output.numpy().flatten()
            block = np.zeros((len(ids), 3))
            for k in range(3):
                ablated = pooled.copy(); ablated[:, k*90:(k+1)*90] = 0
                block[:, k] = (output - readout(ablated)) * scale
            deltas.extend(block)
        deltas = np.asarray(deltas)
        signs = np.sign(deltas.mean(0))
        tf.keras.utils.set_random_seed(1042+i)
        random_model = make_model(**config)
        random_inner = next(layer for layer in random_model.layers if isinstance(layer, MEGAN))
        random_extract = extractor(tf, random_inner)
        record = {"fold_idx": i, "checkpoint_sha256": sha(path), "randomization_seed": 1042+i,
                  "channel_orientation_partition": "train", "train_channel_mean_effect_logS": deltas.mean(0).tolist(),
                  "channel_signs": signs.tolist(), "training_channel_sign_consistency": (deltas * signs >= 0).mean(0).tolist(),
                  "test_size": len(fold["test_indices"]), "max_prediction_replay_error_logS": 0.0}
        for start in range(0, len(fold["test_indices"]), 64):
            ids = fold["test_indices"][start:start+64]
            inputs = original.tensors(tf, arrays, ids)
            output, node_values, edge_values, h_values, pooled_values = extract(inputs)
            _, random_values, _, _, _ = random_extract(inputs)
            for b, j in enumerate(ids):
                row_id = manifest["row_indices"][j]
                prediction = float(output[b, 0].numpy() * scale + mean)
                error = abs(prediction - float(expected[row_id]["predicted_logS"]))
                assert error < 3e-5, (i, j, error)
                record["max_prediction_replay_error_logS"] = max(record["max_prediction_replay_error_logS"], error)
                row, atom = diagnose(manifest["smiles"][j], node_values[b].numpy(), random_values[b].numpy(),
                                     h_values[b].numpy(), pooled_values[b].numpy(), readout, scale, signs,
                                     np.random.default_rng(20261010+row_id))
                rows.append({"fold_idx": i, "row_idx": row_id, "smiles": manifest["smiles"][j],
                             "true_logS": float(expected[row_id]["true_logS"]), "predicted_logS": prediction, **row})
                atom.update({"fold_idx": i, "row_idx": row_id, "smiles": manifest["smiles"][j],
                             "directed_edge_indices": inputs[2][b].numpy().tolist(), "edge_masks": edge_values[b].numpy().tolist()})
                atoms.append(atom)
        folds.append(record)
        print(f"paper_random_prediction_only: fold {i} completed; train-calibrated signs {signs.tolist()}", flush=True)
    assert sorted(r["row_idx"] for r in rows) == sorted(manifest["row_indices"])
    evidence = {"protocol_sha256": sha(protocol), "split_manifest_sha256": sha(manifest_path),
                "graph_cache_sha256": sha(cache_path), "original_predictions_sha256": sha(prediction_path),
                "runner_sha256": sha(__file__), "evaluation_source_sha256": sha(ROOT / "backend/src/evaluation/masks.py"),
                "runtime": {"tensorflow": tf.__version__, "numpy": np.__version__}, "upstream_sources_verified": True,
                "checkpoint_selection": "previously frozen final epoch 400", "encoder_unchanged": True}
    save_results(ROOT / args.results_dir / "paper_random_prediction_only", "paper_random_prediction_only", rows, atoms, folds, evidence)


if __name__ == "__main__":
    main()
