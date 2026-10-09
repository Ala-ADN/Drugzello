# ESOL scaffold experiment — 2026-10-08

A new experiment using the corrected `megan-regression-v2` architecture, not a
relabeling of the archived 2025 result. Five outer scaffold folds share one split
seed (42), with fold training seeds 42–46. Each has a separate scaffold-disjoint
inner validation set and training-only target standardization. This is one CV
experiment, not five independent seed repetitions.

- Test MAE across folds: **0.963558 ± 0.153387 logS** (population standard deviation).
- Pooled out-of-fold test MAE: **0.978541 logS**; RMSE **1.220236 logS**; R² **0.660916**.
- Outer test sizes: 317, 254, 186, 186, 185. All acyclic molecules are grouped together.
- Preset: three layers, 32 hidden units, two explanation channels, bond features,
  residuals/LayerNorm, batch size 64, learning rate 0.001, at most 30 epochs,
  patience 10, gamma 0.1, beta 0.01, no extra decorrelation term.
- Selection: inner validation MAE; the selected checkpoint is restored before test.
- Mean and Morgan random-forest baselines use identical training/test rows.
- **Explanation limitation:** signed fidelity is negative on folds 0 and 2.
  The intended channel polarity is not reliable across all held-out scaffolds.
  ESOL supplies no ground-truth atom/bond explanation labels.

| Artifact | Purpose |
| --- | --- |
| [fold_metrics.csv](fold_metrics.csv) | Compact model, baseline and fidelity table; `best_epoch` is zero-based |
| [fold_metrics.json](fold_metrics.json) | Fold seeds, sizes, normalization, scores and checkpoint SHA-256 |
| [predictions.csv](predictions.csv) | One held-out prediction for each of the 1,128 rows, with SMILES and scaffold ID |
| [splits.json](splits.json) | Exact train/validation/test rows, scaffold policy, dataset checksum |
| [config.yaml](config.yaml) | Resolved experiment hyperparameters |
| [environment.json](environment.json) | Runtime packages, parent Git commit, tracked-diff checksum and source-file hashes |
| [requirements-resolved.txt](requirements-resolved.txt) | Observed package versions, including binary-wheel version tags |
| [training_history.json](training_history.json) | Per-epoch training/validation losses and validation MAE |
| [summary.json](summary.json) | Fold-mean and pooled held-out test metrics in logS |
| [deployment.json](deployment.json) | Local all-data refit checksum, schema and training metadata; no independent test score |

From the repository root, activate the research environment, then:

```bash
cd backend
MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
python scripts/train_model.py \
  --config-file configs/scaffold_esol.yaml --n-folds 5 --epochs 30 --seed 42 \
  --device cpu --save-dir models/trained/scaffold \
  --results-dir results/scaffold-reproduction --refit --disable-mlflow
```

The reviewed run used `--results-dir ../results/scaffold-esol`; use the ignored
output path above to preserve this committed report. The resolved environment also
contained notebook execution dependencies. Core pinned dependencies are in
`backend/requirements/research.txt`; use the recorded manifest for exact environment
reconstruction, with the CPU PyTorch and PyG wheel indexes from the root README.
Source files were uncommitted during this run, so the parent Git commit alone does
not identify the implementation: use the recorded per-file hashes and the reviewed
source together. Once committing, include this change's commit in future reports.

Verification reloaded all five local checkpoints, checked their recorded hashes,
and reproduced all 1,128 CSV predictions and the signed channel-ablation metrics.
The CSV-derived fold and pooled scores agree with the JSON reports. The separate
refit was also checked through the API with explanations and MC dropout enabled.

Local fold checkpoints are generated under `backend/models/trained/scaffold/` and
are ignored. A separate `model.pth` refit uses all ESOL rows for 27 epochs (the median
validation-selected epoch count). That refit is used by the demo and has **no
independent ESOL test score**. Re-run the command to regenerate its weights; no
remote checkpoint release has been published.

This result does not validate the older 0.45 logS CV claim. Improving the score
requires new experiments; never tune repeatedly against these outer test metrics.
