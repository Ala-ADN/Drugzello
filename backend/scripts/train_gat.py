"""Train a fixed prediction-only GAT on frozen scaffold folds (CPU)."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.gat import GATConfig, run_gat
from src.models.tuning import load_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="../results/scaffold-tuning/splits.json")
    parser.add_argument("--config-file", default="configs/scaffold_gat.yaml")
    parser.add_argument("--save-dir", default="models/trained/scaffold-gat")
    parser.add_argument("--results-dir", default="results/scaffold-gat")
    args = parser.parse_args()
    config = GATConfig(**yaml.safe_load(Path(args.config_file).read_text()))
    print(json.dumps(asdict(config), indent=2), flush=True)
    summary = run_gat(load_raw(args.data_root), json.loads(Path(args.splits).read_text()),
                      config, args.save_dir, args.results_dir, args.data_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
