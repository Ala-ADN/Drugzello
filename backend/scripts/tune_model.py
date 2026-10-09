"""Reproduce a frozen validation-only scaffold search and its held-out evaluation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.tuning import run_search, evaluate_selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", default="../results/scaffold-esol/splits.json")
    parser.add_argument("--search-config", default="configs/scaffold_search.yaml")
    parser.add_argument("--save-dir", default="models/trained/scaffold-tuned")
    parser.add_argument("--results-dir", default="results/scaffold-tuning")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--stage", choices=["search", "evaluate", "all"], default="all")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.stage in ("search", "all"):
        manifest = json.loads(Path(args.splits).read_text())
        run_search(args.data_root, manifest, args.search_config, args.save_dir, args.results_dir, args.workers)
    if args.stage in ("evaluate", "all"):
        print(json.dumps(evaluate_selections(args.data_root, args.save_dir, args.results_dir, args.refit), indent=2))


if __name__ == "__main__":
    main()
