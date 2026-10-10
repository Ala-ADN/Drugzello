"""Independently audit saved atom metrics and export the comparison tables."""

import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from src.evaluation.masks import chemical_atoms, clustered_interval, sha

COHORTS = ["paper_random_prediction_only", "scaffold_prediction_only", "scaffold_joint_auxiliary"]
BASE = ROOT / "results/mask-chemistry"


def main():
    table, all_rows, checks = [], {}, {}
    for cohort in COHORTS:
        directory = BASE / cohort
        summary = json.loads((directory / "summary.json").read_text())
        evidence = json.loads((directory / "evidence.json").read_text())
        assert evidence["protocol_sha256"] == sha(ROOT / "experiments/mask-chemistry/protocol.json")
        assert evidence["evaluation_source_sha256"] == sha(ROOT / "backend/src/evaluation/masks.py")
        source = "evaluate_paper_masks.py" if cohort.startswith("paper") else "evaluate_mask_chemistry.py"
        assert evidence["runner_sha256"] == sha(ROOT / "backend/scripts" / source)
        if cohort == "scaffold_prediction_only":
            original_folds = json.loads((ROOT / "results/scaffold-megan-no-aux/fold_metrics.json").read_text())
        elif cohort == "paper_random_prediction_only":
            original_folds = json.loads((ROOT / "results/paper-esol/fold_metrics.json").read_text())
        else:
            with tarfile.open(ROOT / "backend/models/trained/scaffold-tuned/folds.tar.gz") as archive:
                original_folds = [json.load(archive.extractfile(f"fold_{i}/embedding_joint_32/result.json")) for i in range(5)]
        assert [f["checkpoint_sha256"] for f in summary["folds"]] == [f["checkpoint_sha256"] for f in original_folds]
        records = list(csv.DictReader((directory / "molecule_metrics.csv").open()))
        with gzip.open(directory / "atom_masks.json.gz", "rt") as f:
            atoms = json.load(f)
        assert len(records) == len(atoms) == summary["molecules"]
        lookup = {a["row_idx"]: a for a in atoms}
        assert len(lookup) == len(atoms)
        for row in records:
            a = lookup[int(row["row_idx"])]
            mol, polar, nonpolar, _ = chemical_atoms(row["smiles"])
            assert a["smiles"] == row["smiles"]
            assert a["symbols"] == [n.GetSymbol() for n in mol.GetAtoms()]
            np.testing.assert_array_equal(a["polar"], polar)
            np.testing.assert_array_equal(a["nonpolar"], nonpolar)
            keep = polar | nonpolar
            if polar.any() and nonpolar.any():
                value = roc_auc_score(polar[keep], np.asarray(a["signed_mask_score"])[keep])
                np.testing.assert_allclose(value, float(row["mask_chemical_auc"]), atol=1e-12, rtol=0)
                effect_auc = roc_auc_score(polar[keep], np.asarray(a["signed_atom_readout_effect_logS"])[keep])
                np.testing.assert_allclose(effect_auc, float(row["signed_atom_effect_chemical_auc"]), atol=1e-12, rtol=0)
            if row["mask_vs_negative_atomic_logp_spearman"]:
                corr = spearmanr(a["signed_mask_score"], -np.asarray(a["atomic_logp"])).statistic
                np.testing.assert_allclose(corr, float(row["mask_vs_negative_atomic_logp_spearman"]), atol=1e-12, rtol=0)
        for key, interval in summary["metric_intervals"].items():
            if interval:
                values = [float(row[key]) for row in records if row[key]]
                assert len(values) == interval["molecule_count"]
                np.testing.assert_allclose(np.mean(values), interval["mean"], rtol=0, atol=1e-12)
        checks[cohort] = {"atom_records_verified": len(atoms), "chemical_auc_recomputed_with_sklearn": True,
                          "atomic_logp_rank_correlation_recomputed_with_scipy": True,
                          "source_hashes_verified": True,
                          "checkpoint_hashes_match_original_experiments": True,
                          "maximum_original_prediction_replay_error_logS": max(f["max_prediction_replay_error_logS"] for f in summary["folds"]),
                          "artifact_hashes": {p.name: sha(p) for p in directory.iterdir() if p.is_file()}}
        row = {"cohort": cohort, "molecules": summary["molecules"]}
        for key in ["mask_chemical_auc", "random_weights_chemical_auc", "mask_minus_random_weights_auc",
                    "permuted_mask_chemical_auc", "mask_vs_negative_atomic_logp_spearman",
                    "signed_atom_effect_chemical_auc", "mask_vs_absolute_atom_effect_spearman",
                    "top_mask_deletion_abs_logS", "random_deletion_abs_logS", "top_minus_random_deletion_logS",
                    "fixed_channel_polarity_chemical_auc"]:
            value = summary["metric_intervals"][key]
            row[key] = value["mean"] if value else None
            row[key + "_ci95_low"] = value["ci95"][0] if value else None
            row[key + "_ci95_high"] = value["ci95"][1] if value else None
        table.append(row)
        all_rows[cohort] = {int(r["row_idx"]): r for r in records}
    with (BASE / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    first, second = all_rows[COHORTS[1]], all_rows[COHORTS[2]]
    assert first.keys() == second.keys()
    paired = []
    rng = np.random.default_rng(20261010)
    for field in ["mask_chemical_auc", "mask_vs_negative_atomic_logp_spearman", "top_minus_random_deletion_logS",
                  "fixed_channel_polarity_chemical_auc", "signed_atom_effect_chemical_auc"]:
        rows = []
        for j in first:
            assert first[j]["smiles"] == second[j]["smiles"] and first[j]["scaffold"] == second[j]["scaffold"]
            if first[j][field] and second[j][field]:
                rows.append({"scaffold": first[j]["scaffold"], "difference": float(first[j][field]) - float(second[j][field])})
        interval = clustered_interval(rows, "difference", rng)
        paired.append({"metric": field, "comparison": "prediction_only_minus_joint_auxiliary", "mean_difference": interval["mean"],
                       "ci95_low": interval["ci95"][0], "ci95_high": interval["ci95"][1],
                       "molecules": interval["molecule_count"], "scaffolds": interval["scaffold_count"]})
    with (BASE / "paired_scaffold_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    (BASE / "verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparison": table, "paired_scaffold_comparison": paired}, indent=2))


if __name__ == "__main__":
    main()
