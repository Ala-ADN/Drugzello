"""Re-run the pinned KGCNN ESOL benchmark from MEGAN Appendix C.

Run with the separate backend/requirements/paper-esol.txt environment.
The original upstream model and preprocessing are imported without modification.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "experiments/megan-esol-paper"
SOURCE = ROOT / ".runtime/kgcnn-esol-paper"
_TF = None
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_setup():
    provenance = read(UPSTREAM / "provenance.json")
    if sha(UPSTREAM / "source.tar.gz") != provenance["archive_sha256"]:
        raise ValueError("Upstream archive checksum mismatch")
    if not SOURCE.exists():
        with tarfile.open(UPSTREAM / "source.tar.gz") as archive:
            for member in archive:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source archive member")
                target = SOURCE / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    for name, digest in provenance["source_files_sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f"Upstream source modified: {name}")
    sys.path.insert(0, str(SOURCE))
    return provenance


def runtime():
    global _TF
    if _TF is not None:
        return _TF
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    _TF = tf
    return tf


def metrics(y, p):
    import numpy as np
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    return {"mae": float(abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def prepare(cache):
    """Use upstream SDF and feature generation on the retained original CSV."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from kgcnn.data.moleculenet import MoleculeNetDataset
    from kgcnn.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018
    raw = ROOT / "backend/data/ESOL/esol/raw/delaney-processed.csv"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, cache / raw.name)
    dataset = MoleculeNetDataset(str(cache), "ESOL", raw.name, verbose=20)
    # Match ESOLDataset's constructor, without downloading into the user's home.
    dataset.prepare_data(**MoleculeNetDataset2018.datasets_prepare_data_info["ESOL"])
    dataset.read_in_memory(**MoleculeNetDataset2018.datasets_read_in_memory_info["ESOL"])
    hyper = read(UPSTREAM / "original_hyper.json")
    for method in hyper["data"]["dataset"]["methods"]:
        for name, kwargs in method.items():
            getattr(dataset, name)(**kwargs)
    for i, graph in enumerate(dataset):
        graph["original_row_idx"] = np.asarray(i)
    dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])
    dataset.clean(hyper["model"]["config"]["inputs"])
    row_ids = [int(g["original_row_idx"]) for g in dataset]
    frame = pd.read_csv(raw)
    smiles = frame["smiles"].iloc[row_ids].tolist()
    labels = np.asarray(dataset.obtain_property("graph_labels"), dtype=float).reshape(-1, 1)
    arrays = {"labels": labels}
    for field in ("node_attributes", "edge_number", "edge_indices"):
        parts = dataset.obtain_property(field)
        arrays[field] = np.concatenate(parts)
        arrays[field + "_lengths"] = np.asarray([len(p) for p in parts], dtype="int64")
    if arrays["node_attributes"].shape[1] != 41:
        raise ValueError("Original benchmark requires 41 atom features")
    np.savez_compressed(cache / "graphs.npz", **arrays)
    kfold = KFold(**hyper["training"]["cross_validation"]["config"])
    folds = [{"fold_idx": i, "train_indices": train.tolist(), "test_indices": test.tolist()}
             for i, (train, test) in enumerate(kfold.split(labels))]
    manifest = {"split_type": "random_kfold", "n_folds": 5, "seed": 42, "shuffle": True,
                "raw_csv_sha256": sha(raw), "raw_rows": len(frame), "retained_rows": len(dataset),
                "row_indices": row_ids, "smiles": smiles,
                "excluded_row_indices": sorted(set(range(len(frame))) - set(row_ids)), "folds": folds}
    write(cache / "splits.json", manifest)
    return manifest


def tensors(tf, arrays, indices):
    """Equivalent to upstream dataset.tensor(): ragged inputs in row order."""
    import numpy as np
    output = []
    for field, dtype in (("node_attributes", tf.float32), ("edge_number", tf.float32),
                         ("edge_indices", tf.int64)):
        lengths = arrays[field + "_lengths"]
        boundaries = np.concatenate([[0], np.cumsum(lengths)])
        values = np.concatenate([arrays[field][boundaries[i]:boundaries[i + 1]] for i in indices])
        output.append(tf.RaggedTensor.from_row_lengths(tf.convert_to_tensor(values, dtype=dtype), lengths[indices]))
    return output


def fit_fold(job):
    source_setup()
    tf = runtime()
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from kgcnn.literature.MEGAN import make_model
    from kgcnn.training.scheduler import LinearLearningRateScheduler
    from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
    hyper = read(UPSTREAM / "original_hyper.json")
    fold = job["fold"]
    i, seed = fold["fold_idx"], 42 + fold["fold_idx"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    arrays = np.load(Path(job["cache"]) / "graphs.npz", allow_pickle=False)
    train, test = fold["train_indices"], fold["test_indices"]
    scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
    y_train = scaler.fit_transform(arrays["labels"][train])
    y_test = scaler.transform(arrays["labels"][test])
    x_train, x_test = tensors(tf, arrays, train), tensors(tf, arrays, test)
    model = make_model(**hyper["model"]["config"])
    mae = ScaledMeanAbsoluteError((1, 1), name="scaled_mean_absolute_error")
    rmse = ScaledRootMeanSquaredError((1, 1), name="scaled_root_mean_squared_error")
    for metric in (mae, rmse):
        metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
    model.compile(optimizer=tf.keras.optimizers.deserialize(hyper["training"]["compile"]["optimizer"]),
                  loss=hyper["training"]["compile"]["loss"], metrics=[mae, rmse])
    results, models = Path(job["results"]), Path(job["models"])
    history_path = results / f"fold_{i}_history.json"
    class Progress(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.records = []
        def on_epoch_end(self, epoch, logs=None):
            self.records.append({"epoch": epoch, **{key: float(value) for key, value in logs.items()}})
            if epoch % 25 == 0 or epoch == job["epochs"] - 1:
                write(history_path, self.records)
                print(f"Paper MEGAN fold {i}: epoch {epoch + 1}/{job['epochs']}, loss={logs['loss']:.5f}", flush=True)
    fit = hyper["training"]["fit"]
    schedule = LinearLearningRateScheduler(**fit["callbacks"][0]["config"])
    started = time.monotonic()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=fit["batch_size"],
              epochs=job["epochs"], validation_freq=fit["validation_freq"], verbose=0,
              callbacks=[schedule, Progress()])
    pred = scaler.inverse_transform(model.predict(x_test, batch_size=fit["batch_size"], verbose=0)).reshape(-1)
    truth = arrays["labels"][test].reshape(-1)
    # Keras 2.12's new HDF5 traversal misses nested attention weights in this
    # historical model. Export its complete, ordered get_weights() arrays.
    path = models / f"fold_{i}.weights.npz"
    weights = model.get_weights()
    assert sum(w.size for w in weights) == model.count_params()
    np.savez_compressed(path, *weights)
    replay = make_model(**hyper["model"]["config"])
    with np.load(path, allow_pickle=False) as saved:
        replay.set_weights([saved[f"arr_{j}"] for j in range(len(saved.files))])
    replay_pred = scaler.inverse_transform(replay.predict(x_test, batch_size=fit["batch_size"], verbose=0)).reshape(-1)
    np.testing.assert_array_equal(replay_pred, pred)
    record = {"fold_idx": i, "seed": seed, "epochs": job["epochs"], "checkpoint_selection": "fixed final epoch",
              "train_size": len(train), "test_size": len(test), "parameter_count": model.count_params(),
              "normalization": {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(), "fit_partition": "train"},
              "test": metrics(truth, pred), "checkpoint_sha256": sha(path),
              "checkpoint_format": "complete ordered NumPy weight arrays", "checkpoint_weight_count": len(weights),
              "seconds": time.monotonic() - started}
    write(results / f"fold_{i}.json", record)
    write(results / f"fold_{i}_predictions.json", {"indices": test, "true_logS": truth.tolist(), "predicted_logS": pred.tolist()})
    print(f"Paper MEGAN fold {i} finished: {record['test']}", flush=True)
    return record


def aggregate(results, manifest, epochs):
    import numpy as np
    folds, rows = [], []
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        folds.append(read(results / f"fold_{i}.json"))
        pred = read(results / f"fold_{i}_predictions.json")
        for j, y, p in zip(pred["indices"], pred["true_logS"], pred["predicted_logS"]):
            rows.append({"fold_idx": i, "row_idx": manifest["row_indices"][j], "smiles": manifest["smiles"][j],
                         "true_logS": y, "predicted_logS": p})
    summary = {"metric_unit": "logS", "fold_count": 5, "epochs": epochs,
               "protocol": "original random five-fold; fixed final epoch; no inner validation",
               "test_mae_mean": float(np.mean([f["test"]["mae"] for f in folds])),
               "test_mae_std": float(np.std([f["test"]["mae"] for f in folds])),
               "test_rmse_mean": float(np.mean([f["test"]["rmse"] for f in folds])),
               "test_rmse_std": float(np.std([f["test"]["rmse"] for f in folds])),
               "pooled_test": metrics([r["true_logS"] for r in rows], [r["predicted_logS"] for r in rows])}
    write(results / "fold_metrics.json", folds)
    write(results / "summary.json", summary)
    with (results / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (results / "fold_metrics.csv").open("w", newline="") as handle:
        fields = ["fold_idx", "seed", "epochs", "train_size", "test_size", "test_mae_logS", "test_rmse_logS", "test_r2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold in folds:
            writer.writerow({**{key: fold[key] for key in fields[:5]},
                             "test_mae_logS": fold["test"]["mae"], "test_rmse_logS": fold["test"]["rmse"],
                             "test_r2": fold["test"]["r2"]})
    return summary


def verify(results, models, manifest):
    tf = runtime()
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from kgcnn.literature.MEGAN import make_model
    hyper = read(UPSTREAM / "original_hyper.json")
    cache = models / "data"
    assert sha(cache / "graphs.npz") == read(results / "features.json")["graph_cache_sha256"]
    arrays = np.load(cache / "graphs.npz", allow_pickle=False)
    rows = list(csv.DictReader((results / "predictions.csv").open()))
    differences = []
    for fold in manifest["folds"]:
        i = fold["fold_idx"]
        record = read(results / f"fold_{i}.json")
        path = models / f"fold_{i}.weights.npz"
        assert sha(path) == record["checkpoint_sha256"]
        scaler = StandardScaler(**hyper["training"]["scaler"]["config"]).fit(arrays["labels"][fold["train_indices"]])
        assert scaler.mean_.tolist() == record["normalization"]["mean"]
        assert scaler.scale_.tolist() == record["normalization"]["std"]
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(record["seed"])
        model = make_model(**hyper["model"]["config"])
        with np.load(path, allow_pickle=False) as saved:
            weights = [saved[f"arr_{j}"] for j in range(len(saved.files))]
            assert len(weights) == record["checkpoint_weight_count"]
            assert sum(w.size for w in weights) == model.count_params()
            model.set_weights(weights)
        output = model.predict(tensors(tf, arrays, fold["test_indices"]), batch_size=64, verbose=0)
        pred = scaler.inverse_transform(output).reshape(-1)
        saved = [r for r in rows if int(r["fold_idx"]) == i]
        assert [int(r["row_idx"]) for r in saved] == [manifest["row_indices"][j] for j in fold["test_indices"]]
        expected = np.asarray([float(r["predicted_logS"]) for r in saved])
        np.testing.assert_allclose(pred, expected, rtol=0, atol=0)
        true = arrays["labels"][fold["test_indices"]].reshape(-1)
        np.testing.assert_array_equal(true, [float(r["true_logS"]) for r in saved])
        assert metrics(true, pred) == record["test"]
        differences.extend(abs(pred - expected))
    assert sorted(int(r["row_idx"]) for r in rows) == sorted(manifest["row_indices"])
    summary = read(results / "summary.json")
    assert metrics([float(r["true_logS"]) for r in rows], [float(r["predicted_logS"]) for r in rows]) == summary["pooled_test"]
    checks = {"predictions_replayed": len(rows), "checkpoint_hashes_verified": 5,
              "max_absolute_prediction_error_logS": float(max(differences)), "train_only_scalers_verified": True,
              "upstream_sources_verified": True, "split_type": "random_kfold"}
    write(results / "verification.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="backend/results/paper-esol")
    parser.add_argument("--save-dir", default="backend/models/trained/paper-esol")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400, help="Use 400 for the benchmark; smaller values are smoke tests")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--verify", action="store_true", help="Replay completed checkpoints without training")
    args = parser.parse_args()
    if args.workers < 1 or args.epochs < 1 or not set(args.folds) <= set(range(5)):
        parser.error("Require positive workers/epochs and fold IDs 0–4")
    provenance = source_setup()
    results, models = Path(args.results_dir).resolve(), Path(args.save_dir).resolve()
    if args.verify:
        print(json.dumps(verify(results, models, read(results / "splits.json")), indent=2))
        return
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    plan = {"upstream_commit": provenance["commit"], "upstream_archive_sha256": provenance["archive_sha256"],
            "config_sha256": sha(UPSTREAM / "original_hyper.json"), "epochs": args.epochs,
            "runner_sha256": sha(__file__), "raw_csv_sha256": sha(ROOT / "backend/data/ESOL/esol/raw/delaney-processed.csv"),
            "training_seeds": [42 + i for i in range(5)], "checkpoint_selection": "fixed final epoch",
            "original_training_seeds": "not recorded by upstream", "python": platform.python_version(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    if (results / "plan.json").exists() and read(results / "plan.json") != plan:
        raise ValueError("Existing experiment differs; choose fresh output directories")
    write(results / "plan.json", plan)
    shutil.copy2(UPSTREAM / "original_hyper.json", results / "config.json")
    cache = models / "data"
    if not (cache / "graphs.npz").exists():
        runtime()
        manifest = prepare(cache)
    else:
        manifest = read(cache / "splits.json")
    if manifest["raw_csv_sha256"] != plan["raw_csv_sha256"]:
        raise ValueError("Prepared graph cache belongs to a different raw dataset")
    write(results / "splits.json", manifest)
    feature_record = {"graph_cache_sha256": sha(cache / "graphs.npz"), "node_feature_width": 41}
    if (results / "features.json").exists() and read(results / "features.json") != feature_record:
        raise ValueError("Prepared graph cache differs from this experiment")
    write(results / "features.json", feature_record)
    jobs = []
    for fold in manifest["folds"]:
        if fold["fold_idx"] in args.folds:
            result = results / f"fold_{fold['fold_idx']}.json"
            if result.exists():
                if sha(models / f"fold_{fold['fold_idx']}.weights.npz") != read(result)["checkpoint_sha256"]:
                    raise ValueError("Completed checkpoint modified")
                continue
            jobs.append({"cache": str(cache), "models": str(models), "results": str(results),
                         "fold": fold, "epochs": args.epochs})
    if args.workers == 1:
        # A child process gets a fresh TF runtime after parent preprocessing.
        for job in jobs:
            with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
                pool.submit(fit_fold, job).result()
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(fit_fold, job) for job in jobs]):
                future.result()
    if all((results / f"fold_{i}.json").exists() for i in range(5)):
        print(json.dumps(aggregate(results, manifest, args.epochs), indent=2))


if __name__ == "__main__":
    main()
