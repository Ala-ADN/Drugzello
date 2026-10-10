# MEGAN with prediction loss only

This fixed ESOL ablation keeps MEGAN's GATv2 encoder, two learned importance channels and channel-weighted graph pooling. It sets `gamma_exp = beta_sparsity = delta_decor = 0`, so the optimization objective is prediction MSE. Masks are still learned through the prediction path; they are not removed or replaced with constants.

## Matched comparison

The configuration matches `embedding_joint_32` in the [earlier frozen search](../../backend/configs/scaffold_search.yaml), except for explanation and sparsity weights (decorrelation was already zero). Both use three 32-unit layers, one attention head, 16-dimensional atom/bond categorical embeddings, residuals, LayerNorm, dropout 0.1, AdamW at 0.001 with weight decay 1e-5, batch size 64, up to 150 epochs and early-stopping patience 20.

Both runs perform one joint update per batch, clip gradient norm at one, select the best checkpoint on inner-validation MAE, and schedule learning rates using validation total loss. With auxiliary weights zero, the scheduler now observes prediction MSE, so its learning-rate decisions may differ. This is a comparison of the loss-weight intervention under the same training rule, not a fixed learning-rate trajectory.

The exact five scaffold folds and training seeds 42–46 match the previous experiments. Normalization fits training labels only; metrics are transformed back to original logS. Atom/bond graphs, the encoder, the loss implementation and the training/optimization/evaluation methods match the earlier source. Repository cleanup removed an unused MLflow constructor argument without changing these methods.

The primary comparator is the **fixed joint 32-unit MEGAN**, not the separately selected MEGAN pipeline or split-update variants. Removing explanation and sparsity penalties together does not isolate explanation co-training alone. A follow-up retaining sparsity while setting only `gamma_exp` to zero would answer that narrower question.

## Results

| Pipeline | Mean fold test MAE ± population SD, logS | Pooled test MAE | Pooled test RMSE |
| --- | ---: | ---: | ---: |
| Matched joint MEGAN, with auxiliary losses | 0.7282 ± 0.1346 | 0.7467 | 0.9541 |
| Same MEGAN, prediction MSE only | **0.6621 ± 0.0537** | **0.6598** | **0.8419** |
| Earlier validation-selected MEGAN | 0.6651 ± 0.0979 | 0.6613 | 0.8465 |
| Fixed GAT only | 0.6021 ± 0.0741 | 0.5941 | 0.7803 |
| Descriptors + Morgan RF | 0.6182 ± 0.0669 | 0.6077 | 0.8069 |

Removing the auxiliary objectives reduced mean fold MAE by 0.0661 logS (about 9.1%) against the matched joint control. Three folds improved and two worsened. Prediction-only MEGAN remains behind the fixed GAT and RF on aggregate error; it performs similarly to the earlier validation-selected MEGAN pipeline. The intervention helped this fixed configuration but does not establish that explanation loss alone caused the original GAT–MEGAN gap.

Prediction-only fold MAEs are 0.6237, 0.6975, 0.5872, 0.7399 and 0.6621. Pooled R² is 0.8386. The channels retain learned masks but have no enforced positive/negative semantics, so accuracy and channel-polarity diagnostics must be interpreted separately.

Download the [comparison table](comparison.csv), [paired fold comparison](paired_fold_comparison.csv), [fold metrics](fold_metrics.csv), [1,128 held-out predictions](predictions.csv), [summary](summary.json), [configuration](config.yaml) and [training histories](training_history.json). [Environment and source hashes](environment.json) and the [source snapshot](source.tar.gz) preserve the run's implementation. Weights remain local in `backend/models/trained/scaffold-megan-no-aux/`.

All 1,128 predictions and channel diagnostics replayed exactly from the five saved checkpoints. Checkpoint hashes, train-only scalers, validation-selected epochs, source hashes and metrics were verified; training and validation total losses equal prediction loss throughout. [Verification record](verification.json). The 33 research tests outside the previously stalled API TestClient startup test passed, including the new exact MSE-gradient and mask-gradient check.

## Reproduction

From the repository root:

```bash
make megan-no-aux
```

The target uses the frozen split manifest and separate reproduction directories. Exact command for the reported run:

```bash
cd backend
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONHASHSEED=42 \
  ../.venv/bin/python -u scripts/train_model.py \
  --config-file configs/scaffold_megan_no_aux.yaml \
  --splits ../results/scaffold-tuning/splits.json --n-folds 5 --seed 42 \
  --device cpu --save-dir models/trained/scaffold-megan-no-aux \
  --results-dir results/scaffold-megan-no-aux \
  --skip-random-forest --disable-mlflow
```

Choose new output directories for another run. No baseline is retrained and no all-data deployment refit is performed. Existing API weights remain unchanged.

## Limitations

This is exploratory on previously inspected outer folds, with one split realization and one training seed per fold. It does not establish statistical significance or optimal performance. Unlike GAT, it retains GATv2 and gated pooling; RF still receives extra chemical descriptors. Those comparisons do not isolate a single architectural factor.

Without explanation supervision, channel polarity has no enforced meaning. Exported channel-ablation diagnostics describe behavior; they do not establish chemically correct explanations. ESOL has no ground-truth atom attribution labels.
