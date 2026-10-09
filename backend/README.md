# Drugzello backend

Start with the [root research README](../README.md) for installation, results and
training commands.

| Purpose | Entry point |
| --- | --- |
| Single-configuration scaffold experiment | `scripts/train_model.py` |
| Validation-only search and outer evaluation | `scripts/tune_model.py` |
| Presets | `configs/scaffold_esol.yaml`, `configs/scaffold_search.yaml` |
| Molecular graphs and scaffold groups | `src/data/` |
| MEGAN, losses and experiment export | `src/models/` |
| Trained-model API | `api/main.py` (`main.py` is its Uvicorn alias) |
| Deterministic tests | `tests/research/` |

From this directory, with the research environment active:

```bash
MEGAN_MODEL_PATH=models/trained/scaffold-tuned/model.pth uvicorn main:app --reload
python -m pytest -q
```

Missing or incompatible weights return HTTP 503. The API supports aqueous logS
prediction, learned atom/bond masks and uncalibrated MC dropout spread. It does not
serve multi-dataset training, arbitrary solvents or generated MolT5 explanations.

`requirements/base.txt` supplies model/API dependencies; `research.txt` adds tests
and plotting; `notebook.txt` adds notebook tooling. Install `tracking.txt` only for
optional MLflow logging or replaying the archived experiment source.

Generated fold artifacts are compressed locally in
`models/trained/scaffold-tuned/folds.tar.gz`. The inference refit remains directly
loadable as `models/trained/scaffold-tuned/model.pth`. The
[result replay helper](../results/scaffold-tuning/verify_results.py) unpacks fold
artifacts into a temporary directory and removes it on exit. To resume the original
search or run its raw evaluation CLI, first unpack the archive here:

```bash
tar -xzf models/trained/scaffold-tuned/folds.tar.gz -C models/trained/scaffold-tuned
```

The retained experiment reports contain exact source snapshots and checksums.
Obsolete cloud deployment scaffolding, duplicate loaders, manual debug scripts and
unmaintained tests were removed; earlier tracked versions remain in Git history.
