"""Train/evaluate MEGAN on nested scaffold folds; optionally refit for inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import MolecularDataLoader
from src.data.graphs import GraphDataset
from src.data.scaffold import scaffold_folds
from src.models.experiment import run_experiment, write_json
from src.models.tuning import validate_data
from src.utils.config import MEGANConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ESOL"], default="ESOL")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--config", default="default")
    parser.add_argument("--config-file")
    parser.add_argument("--split-type", choices=["scaffold"], default="scaffold")
    parser.add_argument("--splits", help="Reuse an exact scaffold manifest instead of generating folds")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-dir", default="models/trained/scaffold")
    parser.add_argument("--results-dir", default="results/scaffold")
    parser.add_argument("--refit", action="store_true", help="After CV, refit a separate all-data demo checkpoint")
    parser.add_argument("--skip-random-forest", action="store_true")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = MEGANConfig.load(args.config_file) if args.config_file else MEGANConfig(args.config)
    for argument, field in [(args.epochs, "num_epochs"), (args.lr, "learning_rate"),
                            (args.batch_size, "batch_size")]:
        if argument is not None:
            setattr(config, field, argument)
    if config.num_epochs < 1 or config.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    config.random_seed = args.seed
    raw = MolecularDataLoader(args.data_root).load_esol_dataset()
    dataset = GraphDataset([graph for graph in raw])
    if args.splits:
        manifest = json.loads(Path(args.splits).read_text())
        validate_data(dataset, args.data_root, manifest)
        if manifest["seed"] != args.seed or manifest["n_folds"] != args.n_folds:
            raise ValueError("Seed/fold arguments must agree with the supplied manifest")
    else:
        manifest = scaffold_folds([g.smiles for g in dataset], args.n_folds,
                                  args.validation_fraction, args.seed)
        raw_path = Path(raw.raw_paths[0])
        manifest["dataset"] = {"name": "ESOL", "raw_csv_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    summaries, aggregate = run_experiment(dataset, config, manifest, args.save_dir,
                                         args.results_dir, args.device, args.refit,
                                         not args.skip_random_forest)
    if not args.disable_mlflow:
        # Log finished artifacts; tracking failure must not rerun training or
        # erase deterministic local evidence.
        try:
            import mlflow
            mlflow.set_tracking_uri("file:" + str(Path("mlruns").resolve()))
            mlflow.set_experiment("MEGAN_ESOL_Scaffold_v2")
            with mlflow.start_run():
                mlflow.log_params({**config.to_dict(), "split_type": "scaffold", "n_folds": args.n_folds})
                mlflow.log_metrics({k: aggregate[k] for k in ("test_mae_mean", "test_mae_std")})
                mlflow.log_artifacts(str(Path(args.results_dir).resolve()), "results")
        except Exception as exc:
            print(f"MLflow logging failed; local results are intact: {exc}")
    print(aggregate)
    return summaries, aggregate


if __name__ == "__main__":
    main()
