"""Export recorded validation metrics without loading serialized model weights."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "169e433107624a058227f117ee4c0a92"
ARCHIVE = ROOT / "experiments" / "legacy-esol"


def recorded_rows(archive=ARCHIVE):
    """Keep recorded MAE separate from its algebraic conversion to logS."""
    info = json.loads((archive / "data_info.json").read_text())
    folds = json.loads((archive / "fold_results.json").read_text())
    if info["norm_params"]["method"] != "standard":
        raise ValueError("Expected standard target normalization")
    if len(folds) != info["splits"]["n_folds"]:
        raise ValueError("Fold count does not match archived metadata")
    scale = info["norm_params"]["std"]
    return [
        {
            "run_id": RUN_ID,
            "dataset": info["dataset_name"],
            "split_type": info["splits"]["split_type"],
            "seed": info["random_state"],
            "fold_idx": fold["fold_idx"],
            "epochs_recorded": len(fold["val_maes"]),
            "validation_mae_standardized_at_best_total_loss": fold["best_val_mae"],
            "validation_mae_logS_derived": fold["best_val_mae"] * scale,
            "best_validation_total_loss": fold["best_val_loss"],
        }
        for fold in folds
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "recorded_esol.csv")
    args = parser.parse_args()
    rows = recorded_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
