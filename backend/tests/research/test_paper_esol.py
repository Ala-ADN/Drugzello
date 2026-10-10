"""Pinned upstream evidence must match the paper, without TensorFlow in CI."""

import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import yaml

UPSTREAM = Path(__file__).resolve().parents[3] / "experiments/megan-esol-paper"


def test_paper_source_archive_is_unmodified_and_attributed():
    provenance = json.loads((UPSTREAM / "provenance.json").read_text())
    assert hashlib.sha256((UPSTREAM / "source.tar.gz").read_bytes()).hexdigest() == provenance["archive_sha256"]
    with tarfile.open(UPSTREAM / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(provenance["source_files_sha256"])
        for member in archive:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == provenance["source_files_sha256"][member.name]
        assert archive.extractfile("LICENSE").read() == (UPSTREAM / "LICENSE").read_bytes()
        prefix = "training/results/ESOLDataset/MEGAN/"
        assert archive.extractfile(prefix + "MEGAN_hyper.json").read() == (UPSTREAM / "original_hyper.json").read_bytes()
        assert archive.extractfile(prefix + "MEGAN_ESOLDataset_score.yaml").read() == (UPSTREAM / "original_score.yaml").read_bytes()


def test_original_final_epoch_results_round_to_paper_table():
    score = yaml.safe_load((UPSTREAM / "original_score.yaml").read_text())
    assert score["epochs"] == [400] * 5
    # The paper uses final scores, not each fold's best observed test epoch.
    for metric, mean, std in [("mean_absolute_error", 0.44, 0.03), ("root_mean_squared_error", 0.60, 0.05)]:
        values = score["val_scaled_" + metric]
        assert round(float(np.mean(values)), 2) == mean
        assert round(float(np.std(values)), 2) == std


def test_upstream_protocol_uses_fixed_random_folds_and_prediction_only_training():
    hyper = json.loads((UPSTREAM / "original_hyper.json").read_text())
    training = hyper["training"]
    assert training["cross_validation"] == {"class_name": "KFold", "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
    assert training["fit"]["epochs"] == 400
    assert training["compile"]["loss"] == "mean_squared_error"
    assert all(c["class_name"] == "kgcnn>LinearLearningRateScheduler" for c in training["fit"]["callbacks"])
    assert hyper["model"]["config"].get("importance_factor", 0) == 0
    assert hyper["model"]["config"].get("sparsity_factor", 0) == 0
