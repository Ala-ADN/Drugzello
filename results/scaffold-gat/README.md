# Prediction-only GAT on ESOL

This fixed control predicts solubility without MEGAN explanation objectives. It uses the same 1,128 molecules, frozen five scaffold folds and original logS reporting scale as the [MEGAN/RF comparison](../scaffold-tuning/README.md).

## Architecture and protocol

The implementation uses [PyG GATConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATConv.html), the original GAT operator with bond features. It has three 32-unit layers, one attention head, 16-dimensional categorical embeddings per atom/bond field, residual connections, LayerNorm, ELU and mean-plus-sum graph pooling. A two-layer prediction head produces one target. Dropout is 0.1; synthetic self-loop bond embeddings are zero.

Loss is standardized-target MSE only. AdamW uses learning rate 0.001, weight decay 1e-5, batch size 64 and gradient clipping at norm 1. ReduceLROnPlateau halves the learning rate after 10 validation-MAE patience epochs. Training stops after 20 epochs without improvement, up to 150 epochs. Best weights are cloned and selected on inner-validation MAE. There is one fixed configuration and no hyperparameter search.

Split seed is 42; training seeds are 42–46. The exact [partition manifest](splits.json) matches the existing MEGAN/RF partitions. Scaler statistics use training rows only. The model receives atom and bond categories, with no fingerprints, calculated molecular descriptors, measured lipophilicity or target-derived inputs. It has no explanation channels, gates, sparsity loss or explanation co-training.

## Verified results

| Pipeline | Mean fold test MAE ± population SD, logS | Pooled test MAE | Pooled test RMSE | Pooled R² |
| --- | ---: | ---: | ---: | ---: |
| Fixed GAT only | **0.6021 ± 0.0741** | **0.5941** | **0.7803** | **0.8613** |
| Descriptors + Morgan RF | 0.6182 ± 0.0669 | 0.6077 | 0.8069 | 0.8517 |
| MAE-selected MEGAN | 0.6651 ± 0.0979 | 0.6613 | 0.8465 | 0.8368 |

GAT fold MAEs are 0.5681, 0.5339, 0.5639, 0.7442 and 0.6006. Best epochs (zero-based) are 79, 49, 78, 58 and 50. The fixed GAT has slightly lower aggregate error than RF in this run; this is not evidence of a statistically significant or universal advantage.

Download [per-fold metrics](fold_metrics.csv), [all 1,128 held-out predictions](predictions.csv), [summary](summary.json), [configuration](config.json) and [environment/source hashes](environment.json). Per-fold histories retain validation metrics and learning rates. Local checkpoint hashes and train-only normalization parameters are in [fold_metrics.json](fold_metrics.json).

All five saved checkpoints replayed all 1,128 predictions exactly, with zero maximum absolute difference; checkpoint hashes, source hashes, original labels, train-only scalers and metrics were verified. [Verification record](verification.json). The [source snapshot](source.tar.gz) preserves the implementation used for this run; generated weights remain ignored and local.

Validation: all three new GAT tests passed, and 32 research tests passed with the existing API TestClient test excluded. That API test stalled during TestClient startup and timed out in a separate 35-second check; the full suite was not verified as passing in this environment. GAT batching, node permutation, isolated atoms, deterministic training, test-label isolation and checkpoint replay are covered by the new tests.

## Reproduction

From the repository root, with the research environment installed:

```bash
make gat
```

This uses fresh, ignored reproduction directories. To reproduce the reported run's command:

```bash
cd backend
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  ../.venv/bin/python -u scripts/train_gat.py \
  --config-file configs/scaffold_gat.yaml \
  --splits ../results/scaffold-tuning/splits.json \
  --save-dir models/trained/scaffold-gat \
  --results-dir results/scaffold-gat
```

Existing JSON reports or checkpoints cause the runner to refuse replacement; choose new output paths to rerun. Local checkpoints remain under `backend/models/trained/scaffold-gat/`; they use a separate GAT format and are not deployed by the MEGAN API. No all-data refit is performed.

## Interpretation limits

This run follows inspection of earlier results, so it is exploratory on previously used outer folds. A fixed GAT configuration is compared with validation-selected MEGAN and RF configurations, not equal search budgets. One split seed and one training seed per fold do not establish a stable ranking or statistical significance.

Original GAT attention, graph pooling and prediction-only loss all differ from MEGAN. A performance difference does not isolate the effect of explanation losses. RF also receives RDKit descriptors and Morgan fingerprints. These results compare complete pipelines; they do not establish that one architecture is universally superior. This is not a reproduction of the quoted seven-model study or its random-split RMSE.

See the [learning guide](../../docs/reproducing-megan.md#11-prediction-only-gat-control) for a dimension-by-dimension walkthrough.
