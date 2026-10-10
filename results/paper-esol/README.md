# MEGAN ESOL paper-protocol reproduction

This experiment targets [Appendix C / Figure 6 of the MEGAN paper](https://arxiv.org/pdf/2211.13236#page=24), using the [pinned original KGCNN 2.1.0 source](../../experiments/megan-esol-paper/README.md). The historical September 2022 artifact reproduces the paper's rounded reference numbers: mean final-fold MAE 0.4369 ± 0.0319 and RMSE 0.6048 ± 0.0461 logS. Those reference values belong to the authors; new measurements are reported separately.

## Measured results

All five folds completed 400 epochs with 206,834 parameters per model. Scores are in original logS units; ± denotes population SD across folds.

| Run | Fold-mean MAE ± SD | Fold-mean RMSE ± SD |
| --- | ---: | ---: |
| Authors' September 2022 artifact | 0.4369 ± 0.0319 | 0.6048 ± 0.0461 |
| New Drugzello reproduction | **0.4424 ± 0.0291** | **0.6206 ± 0.0431** |

The new MAE rounds to the paper's 0.44 ± 0.03. RMSE is slightly higher. Across all 1,127 out-of-fold predictions, pooled MAE is 0.4424, RMSE 0.6221 and R² 0.9119. Pooled metrics and fold means are different aggregations.

Download [per-fold metrics](fold_metrics.csv), [individual predictions](predictions.csv) and the [summary](summary.json). The [fold records](fold_metrics.json) include seeds, training-only scaler statistics and checkpoint hashes. Protocol evidence comprises the [configuration](config.json), [splits](splits.json), [run plan](plan.json), [input checksum](features.json), [resolved dependencies](requirements-resolved.txt), [400-epoch histories](training_histories.json.gz) and [runner source snapshot](runner_source.tar.gz). Weights remain in ignored local model directories.

This closes most of the apparent gap with the paper under its own protocol. It does not identify how much improvement comes from random splitting, features or architecture, because those change together relative to the scaffold experiments.

[Independent checkpoint verification](verification.json) replayed all 1,127 predictions with zero difference, checked five checkpoint hashes and confirmed training-only scalers. A repeated training run matched all 2,000 epoch records and all out-of-fold predictions. Independently recomputed scikit-learn metrics agree with the published summary. The research suite passed 36 tests; one existing API TestClient test was excluded because startup stalls in this environment.

## Protocol

The original configuration specifies random five-fold `KFold(shuffle=True, random_state=42)`, training-fold-only target standardization and 400 fixed epochs. Each outer fold is monitored as Keras validation data every ten epochs. There is no inner validation split, checkpoint selection, early stopping or test-dependent learning-rate scheduler; final-epoch weights determine the reported metrics. The runner does not substitute the lowest observed held-out error.

The original cleaner removes methane (zero-based CSV row 934, SMILES `C`) because it has no edges, leaving 1,127 rows. This upstream filtering rule is preserved; test-fold sizes are 226, 226, 225, 225 and 225. Original CSV row IDs remain attached to all predictions. The earlier PyTorch scaffold runs kept all 1,128 rows, including methane via internal self-loops.

The original MEGAN has three importance channels; attention-head outputs are concatenated. Layer widths are `[60, 50, 40, 30]`, the prediction MLP is `[50, 30, 10, 1]`, and dropout is 0.3. Atom inputs use the original 41 features. Bond-feature attention is disabled. Explanation co-training and sparsity factors default to zero. The architecture and feature generation are imported directly from the archived upstream package without edits.

Adam starts at learning rate 0.001. The original linear scheduler keeps it constant through epoch 200, then decreases toward 1e-5 at epoch 400. Batch size is 64; no added weight decay, gradient clipping or residual/LayerNorm variants are introduced.

## Reproduction

Use an isolated environment, from the repository root:

```bash
python3.11 -m venv .runtime/esol-paper
.runtime/esol-paper/bin/python -m pip install -r backend/requirements/paper-esol.txt
make paper-esol
```

The source archive is bundled, checksummed and extracted into ignored `.runtime/kgcnn-esol-paper/`; the retained raw ESOL CSV is used locally. The original SDF conversion, chemistry features and graph filtering are preserved. Generated features and weights stay in ignored output directories. Fold runs can resume if the configuration, runner, environment and checkpoint hashes match.

For the exact new run's inputs, a small [graph cache](graphs.npz) is retained as a reproducibility artifact. The original conformer generator has no recorded random seed; its SDF stereochemical features can depend on regeneration. This cache contains named numeric arrays only, loads with `allow_pickle=False`, and is checked against native upstream tensor construction. To train on the exact saved inputs, first copy the cache and [split manifest](splits.json) into the reproduction directory:

```bash
mkdir -p backend/models/trained/paper-esol-reproduction/data
cp results/paper-esol/graphs.npz backend/models/trained/paper-esol-reproduction/data/
cp results/paper-esol/splits.json backend/models/trained/paper-esol-reproduction/data/
make paper-esol
```

Without that copy, the runner regenerates the original preprocessing pipeline and records the new graph-cache hash. Exact input reuse and newly generated preprocessing are distinct reproduction modes.

Exact command for the reported run:

```bash
PYTHONHASHSEED=42 OMP_NUM_THREADS=1 \
  .runtime/esol-paper/bin/python -u backend/scripts/reproduce_paper_esol.py \
  --results-dir backend/results/paper-esol \
  --save-dir backend/models/trained/paper-esol --workers 2
```

Replay completed checkpoints:

```bash
.runtime/esol-paper/bin/python backend/scripts/reproduce_paper_esol.py --verify
```

## Differences and limits

This reproduces the historical model, preprocessing configuration, splitting and fixed training schedule. The complete original runtime and training RNG seeds were not published. This run uses TensorFlow CPU 2.12, Python 3.11, RDKit 2022.9.5 and explicit fold seeds 42–46; it cannot recover the authors' original weights bit for bit. The original source remains unchanged, with its MIT license preserved.

The Adam optimizer is deserialized from the original configuration, selecting TensorFlow 2.12's legacy Adam implementation. The local tensor adapter casts bond numbers to the declared float32 input dtype, while native KGCNN tensors retain int64; bond features are disabled, and native/adapted model predictions match exactly. [Adapter verification](adapter_verification.json).

Checkpoints use complete ordered NumPy weight arrays (`allow_pickle=False`). TensorFlow 2.12's newer HDF5 export omitted this historical model's nested attention weights, so it was replaced after replay caught the mismatch. The exporter checks the full parameter count and prediction equality after loading a fresh model; final verification repeats loading in an independent process.

Random folds can share scaffolds across training and testing. These results therefore answer a different generalization question from the existing scaffold experiments. A score close to the paper's random-fold MAE does not establish the same accuracy on unseen scaffolds or support a five-scaffold-fold CV claim. No other model from Figure 6 is retrained in this MEGAN replication.

The fixed 400-epoch protocol is preserved even though the outer folds are monitored. Changes chosen after viewing these results belong in a separate experiment. Report population spread across folds as SD, not a confidence interval.
