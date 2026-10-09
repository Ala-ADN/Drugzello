# Repository layout and maintenance

Active source is `backend/`, the demo is `frontend/`, learning materials are
`docs/` and `notebooks/`, and reviewed experiment evidence is `results/`.

## Entry points

| Purpose | Command from the indicated directory |
| --- | --- |
| Research tests, root | `make test` |
| Scaffold experiment, backend | `python scripts/train_model.py --config-file configs/scaffold_esol.yaml --disable-mlflow` |
| Validation search, backend | `python scripts/tune_model.py --help` |
| API, root | `make api` |
| Frontend, frontend | `npm ci && npm run dev` |
| Historical metrics, root | `python backend/scripts/export_recorded_results.py` |
| Full result replay, root | `make verify-results` |

The root [README](../README.md) records installation and complete training flags.
A single root `.gitignore` covers real environment files, virtual environments,
graph caches, generated runs/checkpoints, frontend dependencies and build output.
Only `.env.example` templates are retained. Never commit real credentials.

## Evidence and archives

Keep raw ESOL input, dataset checksum, exact splits/seeds, configuration, resolved
environment, all per-fold metrics and predictions. Generated MLflow trees and
model weights stay ignored. Reports link compact tables and explain what the
metrics actually measure. Do not overwrite an earlier experiment's evidence.

The [first scaffold source](../experiments/scaffold-esol-baseline-source/README.md)
and [follow-up source](../results/scaffold-tuning/source.tar.gz) are compressed
snapshots, with hashes in their reports. They preserve the exact code used before
cleanup without leaving obsolete modules on active import paths. The replay helper
uses the follow-up snapshot when maintained source differs and verifies all hashes.
Install the optional tracking requirements to replay that older source.

Local fold artifacts are compressed into
`backend/models/trained/scaffold-tuned/folds.tar.gz`, while both all-data inference
checkpoints remain directly loadable. The replay helper temporarily unpacks them.
For raw CLI resume/evaluation, unpack the fold archive first; fresh output directories
need no unpacking. Source/fold archive creation verified every member byte-for-byte.
The [historical evidence](../experiments/legacy-esol/README.md) retains only four
original text files, checksums and a small [CSV](../results/recorded_esol.csv).
No remote model release has been published.

The [cleanup manifest](cleanup-manifest.json) identifies removed obsolete paths.
Deleted tracked cloud/deployment/debug files remain recoverable from Git history.
Unused frontend screens and template assets were removed; `App.jsx` renders the
maintained editor. The runtime interpreter and virtual environment remain available;
wheels, build output, dependency installations and caches can be regenerated.

## GitHub settings

These remote settings have not been changed locally. With authenticated GitHub CLI:

```bash
gh repo edit Ala-ADN/Drugzello \
  --description 'Reproducible implementation and evaluation of MEGAN for explainable molecular-solubility prediction using PyTorch Geometric.' \
  --add-topic graph-neural-networks --add-topic pytorch-geometric \
  --add-topic explainable-ai --add-topic molecular-machine-learning \
  --add-topic solubility-prediction --add-topic gnn \
  --add-topic mlflow --add-topic reproducible-research
```
