# Drugzello: explainable molecular solubility

**Research question:** Can MEGAN predict aqueous solubility on unseen molecular scaffolds while learning useful explanation channels?

Drugzello evaluates MEGAN in PyTorch Geometric using scaffold experiments and baselines. Explore the [model implementation](backend/src/models/megan_architecture.py), [mask-evaluation learning notes](experiments/mask-chemistry/README.md), or trained-model demo.

## Method and data

The reference is Teufel, Torresi, Reiser and Friederich, [_MEGAN: Multi-Explanation Graph Attention Network_](https://arxiv.org/abs/2211.13236) ([xAI 2023 publication](https://doi.org/10.1007/978-3-031-44067-0_18), [authors' implementation](https://github.com/aimat-lab/graph_attention_student)).

The model combines GATv2 attention, differentiable node/edge gates, channel-weighted pooling and explanation co-training. Variants add categorical embeddings and explanation-then-prediction updates.

This applies MEGAN to **ESOL**; it does not replicate the paper's AqSolDB benchmark. See the [architecture](backend/src/models/megan_architecture.py) for residual connections and LayerNorm, and the [trainer](backend/src/models/trainer.py) for auxiliary losses and AdamW.

ESOL contains 1,128 aqueous-solubility targets: `logS = log10(S / (mol/L))`. [Raw data and checksum](backend/data/README.md) are retained.

RDKit/PyG graphs use nine atom and three bond categorical fields, bidirectional bonds and no explicit hydrogens. Models use ordinal inputs or 16-dimensional embeddings per field. Internal self-loops have zero bond features and are excluded from explanations.

## Evaluation and verified results

Five outer test folds use typed, nonchiral Bemis–Murcko scaffolds and scaffold-disjoint inner validation sets. **All acyclic molecules share the empty-scaffold group.** The 269 groups yield test sizes of 317, 254, 186, 186 and 185.

Split seed is 42; training seeds are 42–46. Normalization fits training rows only, and inner validation selects epochs and configurations. Each molecule is tested once.

The search ran **40 MEGAN trials and 35 baseline fits** on the same partitions. Both MEGAN selection rules and three fixed ablations were frozen before outer evaluation.

| Comparison                                      | Fold-mean test MAE ± SD, logS | Pooled MAE |
| ----------------------------------------------- | ----------------------------: | ---------: |
| MEGAN: minimum validation MAE selector          |               0.6651 ± 0.0979 |     0.6613 |
| Fixed joint MEGAN, auxiliary losses disabled    |               0.6621 ± 0.0537 |     0.6598 |
| Fixed embeddings + split updates, 32 units      |               0.6641 ± 0.0865 |     0.6660 |
| MEGAN: validation channel-polarity constraint   |               0.6759 ± 0.1041 |     0.6768 |
| GAT only: fixed prediction MSE control          |           **0.6021 ± 0.0741** | **0.5941** |
| RDKit descriptors + SVR                         |               0.7141 ± 0.0731 |     0.7260 |
| RDKit descriptors + Morgan random forest        |               0.6182 ± 0.0669 |     0.6077 |
| RDKit descriptors + histogram gradient boosting |               0.6695 ± 0.0747 |     0.6592 |

Pooled RMSE/R² are 0.8465/0.8368 for MAE-selected MEGAN and 0.8069/0.8517 for the random forest.

The [prediction-only MEGAN ablation](results/scaffold-megan-no-aux/README.md) retains learned masks and disables all auxiliary losses. Its fold-mean MAE falls from 0.7282 to 0.6621 against the matched joint control; GAT still has lower error. Explanation and sparsity penalties change together, so their individual effects are not isolated.

Download [comparisons](results/scaffold-tuning/comparison.csv), [per-fold metrics](results/scaffold-tuning/fold_metrics.csv) and [held-out predictions](results/scaffold-tuning/predictions.csv).

**Limitations:** Split training uses two updates per batch; comparisons match epochs, not compute. Only one split seed and one training seed per fold were evaluated.

Channel ablation uses the same forward path. The constrained selector has positive average fidelity\* on all folds, with a fallback in fold 1. Both channels have the expected sign for only **66.7%** of test molecules; MAE-only selection has negative fidelity on two folds. [Channel evaluation](results/scaffold-tuning/README.md#explanation-channel-behavior).

[Chemical-proxy tests](results/mask-chemistry/README.md) find signal in the learned masks: training-oriented polar-versus-nonpolar AUC is 0.777 for the original random-fold model and 0.806 for prediction-only scaffold MEGAN, versus about 0.5 after shuffling. High-mask pooling deletions affect predictions more than equal-sized random deletions. Channel polarity remains inconsistent; these checks support chemical plausibility rather than causal explanations.

## Install, test and reproduce

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter==2.1.2 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
python -m pip install -r backend/requirements/research.txt
make test
```

Train one configuration or run the eight-candidate search in fresh output directories:

```bash
make train
make tune
make gat
make megan-no-aux
```

`make gat` runs the fixed [prediction-only GAT control](backend/configs/scaffold_gat.yaml) on the same scaffold folds, without descriptors or explanation losses. Its [report](results/scaffold-gat/README.md) explains the architecture and comparison limits.

`make megan-no-aux` keeps MEGAN's masks and architecture but disables all auxiliary losses. Its [report](results/scaffold-megan-no-aux/README.md) compares prediction-only training against the fixed joint MEGAN control.

For the original KGCNN benchmark, install the [separate TensorFlow environment](results/paper-esol/README.md#reproduction), then run `make paper-esol`.

The [Makefile](Makefile) contains exact flags. The [full report](results/scaffold-tuning/README.md#exact-reproduction-commands) records commands, configurations, splits, seeds, environment and histories. Training regenerates graph caches from the retained CSV.

Compressed source snapshots and fold artifacts preserve the experiments. All **10,152 held-out predictions** replayed exactly after cleanup. See [replay instructions](results/scaffold-tuning/README.md#repository-cleanup-and-archival-replay).

For optional MLflow or archived-source replay, install `backend/requirements/tracking.txt` with pip.

## Repository and demo

```text
backend/src/          Graphs, scaffolds, MEGAN, training and evaluation
backend/scripts/      Training, tuning, GAT, original-paper reproduction and exports
backend/configs/      Authoritative training, search and GAT presets
backend/tests/        Deterministic research and API tests
backend/api/          Trained-model inference API
frontend/             React/Vite molecular editor
results/              Reviewed configurations, splits, metrics and predictions
experiments/          Compact historical evidence and exact source snapshot
```

The 93-epoch inference refit is local at `backend/models/trained/scaffold-tuned/model.pth`; no checkpoint release is published. After reproducing training, set `MODEL_PATH` to your run's `model.pth`.

```bash
make api
```

```bash
cd frontend
npm ci
npm run dev
```

API docs: <http://127.0.0.1:8000/docs> · Demo: <http://localhost:5173>

The API returns aqueous logS, learned masks and MC dropout spread. Missing or incompatible weights return **503**; unsupported solvents and MolT5 requests return **422**. There are no mock predictions or label fallbacks.

Generated dependencies, builds, model artifacts, caches and real environment files are ignored.

## Attribution and license

Credit for MEGAN goes to its original authors. Drugzello provides the PyTorch/PyG implementation, experiments, tests, learning materials and demo.

Code is [MIT licensed](LICENSE); third-party licenses remain.

The [archived original KGCNN source](experiments/megan-esol-paper/README.md) retains the authors' MIT license. Its model and preprocessing are imported unchanged for the paper replication.

```bibtex
@misc{teufel2022megan,
  title={MEGAN: Multi-Explanation Graph Attention Network},
  author={Jonas Teufel and Luca Torresi and Patrick Reiser and Pascal Friederich},
  year={2022}, eprint={2211.13236}, archivePrefix={arXiv},
  doi={10.48550/arXiv.2211.13236}
}
```
