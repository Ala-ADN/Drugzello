"""Check public result provenance without importing the ML stack."""

import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "export_recorded_results", ROOT / "backend/scripts/export_recorded_results.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class TestRecordedResults(unittest.TestCase):
    def test_preserved_evidence_checksums(self):
        manifest = json.loads((exporter.ARCHIVE / "checksums.json").read_text())
        for name, expected_hash in manifest.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                                 expected_hash)

    def test_csv_matches_original_fold_artifacts(self):
        with (ROOT / "results/recorded_esol.csv").open(newline="") as handle:
            published = list(csv.DictReader(handle))
        expected = exporter.recorded_rows()
        self.assertEqual(len(published), 2)
        for actual, recorded in zip(published, expected):
            for key, value in recorded.items():
                self.assertEqual(actual[key], str(value))

    def test_summary_identifies_best_total_loss_fold(self):
        summary = json.loads((exporter.ARCHIVE / "training_summary.json").read_text())
        best = min(exporter.recorded_rows(), key=lambda row: row["best_validation_total_loss"])
        self.assertEqual(int(summary["best_fold"]), best["fold_idx"])
        self.assertEqual(summary["best_val_mae"],
                         best["validation_mae_standardized_at_best_total_loss"])

    def test_stored_mae_matches_loss_selected_epoch(self):
        folds = json.loads((exporter.ARCHIVE / "fold_results.json").read_text())
        for fold in folds:
            losses = fold["val_losses"]["total"]
            epoch = min(range(len(losses)), key=losses.__getitem__)
            self.assertEqual(losses[epoch], fold["best_val_loss"])
            self.assertEqual(fold["val_maes"][epoch], fold["best_val_mae"])

    def test_logS_conversion_uses_normalization_not_sampled_statistics(self):
        info = json.loads((exporter.ARCHIVE / "data_info.json").read_text())
        for row in exporter.recorded_rows():
            self.assertAlmostEqual(
                row["validation_mae_logS_derived"],
                row["validation_mae_standardized_at_best_total_loss"] * info["norm_params"]["std"],
            )
        self.assertNotEqual(info["norm_params"]["std"], info["stats"]["target_std"])
