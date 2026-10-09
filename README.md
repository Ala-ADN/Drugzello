# Drugzello — explainable molecular solubility

**Research question:** Can MEGAN predict aqueous solubility on unseen molecular
scaffolds while learning useful explanation channels?

Drugzello implements and evaluates MEGAN in PyTorch Geometric, with reproducible
scaffold experiments, baselines, a [learning guide](docs/reproducing-megan.md), an
[executable notebook](notebooks/megan_pytorch.ipynb), and a trained-model demo.

## Method and data

The reference is Teufel, Torresi, Reiser and Friederich,
[*MEGAN: Multi-Explanation Graph Attention Network*](https://arxiv.org/abs/2211.13236)
([xAI 2023 publication](https://doi.org/10.1007/978-3-031-44067-0_18),
[authors' implementation](https://github.com/aimat-lab/graph_attention_student)).

The maintained model uses dynamic GATv2 attention, differentiable node/edge gates,
channel-weighted pooling and regression explanation co-training. Follow-up variants
use categorical embeddings and explanation-then-prediction optimizer updates.
Residual connections, LayerNorm, the magnitude auxiliary loss and AdamW are
[documented choices](docs/reproducing-megan.md). This reproduces structural ideas
on **ESOL**, rather than the paper's original AqSolDB benchmark or published scores.

ESOL has 1,128 aqueous-solubility targets: `logS = log10(S / (mol/L))`.
[Raw input, provenance and checksum](backend/data/README.md) are retained.
RDKit/PyG graphs have nine atom categorical fields and three bond categorical
fields, both bond directions and no explicit hydrogens. The earlier model used
ordinal values; embedding variants learn a separate 16-dimensional table for each
field. Added self-loops receive zero bond features and are excluded from explanations.

## Evaluation and verified results

Typed, nonchiral Bemis–Murcko scaffold groups define five outer test folds and
scaffold-disjoint inner validation partitions. **All acyclic molecules stay together
in the empty-scaffold group.** There are 269 groups; test sizes are 317, 254, 186,
186 and 185. Split seed is 42, training seeds are 42–46. Target normalization fits
training rows only; epochs and candidate configurations are selected on inner
validation. Every molecule receives exactly one outer-test prediction.

The [2026-10-09 follow-up](results/scaffold-tuning/README.md) ran 40 MEGAN trials
and 35 baseline fits. Both MEGAN selection rules and three fixed ablations were
frozen before outer evaluation. All baseline families use the same partitions.

| Comparison | Fold-mean test MAE ± SD, logS | Pooled MAE |
| --- | ---: | ---: |
| [Earlier ordinal MEGAN](results/scaffold-esol/README.md), at most 30 epochs | 0.9636 ± 0.1534 | 0.9785 |
| MEGAN: minimum validation MAE selector | 0.6651 ± 0.0979 | 0.6613 |
| Fixed embeddings + split updates, 32 units | 0.6641 ± 0.0865 | 0.6660 |
| MEGAN: validation channel-polarity constraint | 0.6759 ± 0.1041 | 0.6768 |
| RDKit descriptors + SVR | 0.7141 ± 0.0731 | 0.7260 |
| RDKit descriptors + Morgan random forest | **0.6182 ± 0.0669** | **0.6077** |
| RDKit descriptors + histogram gradient boosting | 0.6695 ± 0.0747 | 0.6592 |
| Training mean | 1.7653 ± 0.4080 | 1.7640 |

SD is population spread across folds, not a confidence interval or independent
seed repetition. MEGAN's MAE-selected pipeline has pooled RMSE 0.8465 and R² 0.8368;
the stronger random forest has RMSE 0.8069 and R² 0.8517.
[Download all comparisons](results/scaffold-tuning/comparison.csv),
[per-fold metrics](results/scaffold-tuning/fold_metrics.csv) and
[held-out predictions](results/scaffold-tuning/predictions.csv).

**Limitations:** these are exploratory comparisons on previously inspected folds,
not an untouched benchmark or proof of optimal performance. Split training uses
two updates per batch, so comparisons are epoch-matched rather than compute-matched.
Only one split seed and one training seed per fold were evaluated.

Signed channel ablation uses the same forward path. The constrained selector has
positive average fidelity* on all folds, but falls back in fold 1; both channels
have the expected sign for only **66.7%** of test molecules. MAE-only selection has
negative fidelity on two folds. See [channel evaluation](results/scaffold-tuning/README.md#explanation-channel-behavior).
ESOL has no ground-truth attribution labels; learned masks do not establish chemical
mechanisms. MC dropout spread is uncalibrated and confidence is null.

The [historical two-fold evidence](experiments/legacy-esol/README.md) records
validation MAE in standardized units. Neither old nor new experiments substantiate
the historical **0.45 logS** claim. [CV reconciliation](docs/results-and-cv.md).

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

Single-configuration training and the frozen eight-candidate search use fresh
output directories:

```bash
make train
make tune
```

See the [Makefile](Makefile) for exact flags and the
[full report](results/scaffold-tuning/README.md#exact-reproduction-commands) for
executed commands, configurations, splits, seeds, resolved environment and histories.
Training reads the retained raw CSV and regenerates ignored graph caches.

Exact pre-cleanup source snapshots and local fold artifacts are compressed;
[replay instructions](docs/repository-maintenance.md) explain their checksums and
restoration. All 10,152 held-out predictions replayed exactly after cleanup.
Optional MLflow and archived-source imports use
`python -m pip install -r backend/requirements/tracking.txt`; notebook tooling uses
`backend/requirements/notebook.txt`. The notebook's default run teaches mechanics
with invented fixture targets; full ESOL training is opt-in.

## Repository and demo

```text
backend/src/          Graphs, scaffolds, MEGAN, training and evaluation
backend/scripts/      train_model.py, tune_model.py, export_recorded_results.py
backend/configs/      Two authoritative experiment presets
backend/tests/        Deterministic research and API tests
backend/api/          Trained-model inference API
frontend/             React/Vite molecular editor
notebooks/, docs/     Learning materials and reproducibility notes
results/              Reviewed configurations, splits, metrics and predictions
experiments/          Compact historical evidence and exact source snapshot
```

The all-data inference refit used 93 epochs and has **no independent ESOL test
score**. Its weights remain local at `backend/models/trained/scaffold-tuned/model.pth`;
no remote checkpoint release is published. After reproducing training, set
`MODEL_PATH` to that run's `model.pth`.

```bash
make api
# Separate terminal:
cd frontend
npm ci
npm run dev
```

API docs: <http://127.0.0.1:8000/docs>; demo: <http://localhost:5173>.
The API supports aqueous logS, learned masks and MC dropout. Missing/incompatible
weights return HTTP 503. Unsupported solvents and MolT5 requests return HTTP 422.
There are no mock predictions or label fallbacks. Dependencies/build output,
model artifacts, caches and real environment files are ignored.

## Attribution and license

MEGAN belongs to its original authors. Drugzello implements the PyTorch/PyG model
and documented variants, scaffold experiments, tests, learning materials and demo.
PyTorch, PyG, RDKit and scikit-learn provide underlying software; Delaney/MoleculeNet
provide ESOL. Project code is [MIT licensed](LICENSE); third-party licenses remain.

```bibtex
@misc{teufel2022megan,
  title={MEGAN: Multi-Explanation Graph Attention Network},
  author={Jonas Teufel and Luca Torresi and Patrick Reiser and Pascal Friederich},
  year={2022}, eprint={2211.13236}, archivePrefix={arXiv},
  doi={10.48550/arXiv.2211.13236}
}
```
