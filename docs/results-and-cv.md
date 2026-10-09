# Results and CV reconciliation

The archived 2025 run supports a historical **validation** experiment on ESOL with two
shuffled K-folds and seed 42. It does not support the historical claim of 0.45 logS over five
scaffold-based folds. Do not use that claim in applications unless its original
experiment evidence is recovered and published.

No CV source file is present in this checkout; you are updating it yourself.
The wording below describes the historical work only. For the corrected work,
use the new scaffold experiment at the end of this document.

Historical wording:

> Implemented a PyTorch Geometric model inspired by MEGAN for ESOL molecular-solubility
> prediction, with MLflow experiment tracking and two-fold shuffled cross-validation
> (seed 42); archived validation MAE of 0.5573 in standardized target units for the
> fold selected by composite validation loss.

For a shorter CV bullet without metric ambiguity:

> Implemented and evaluated a MEGAN-inspired molecular-solubility model on ESOL using
> PyTorch Geometric, two-fold cross-validation, and MLflow experiment tracking.

The selected-fold metric converts algebraically to approximately **1.168 logS**
using the archived normalization scale. This is not a held-out test result, and
normalization on the full dataset is a limitation. The two fold MAEs are 0.557280
and 0.684546 in standardized units; do not report the best fold as the fold mean.
See [the downloadable table](../results/recorded_esol.csv) and
[original evidence](../experiments/legacy-esol/README.md).

To substantiate a future five-scaffold-fold claim, publish the source commit and
resolved environment, immutable dataset/checksum, exact scaffold assignment code,
per-fold train/validation/test molecule identifiers, all random seeds, training-only
normalization parameters, checkpoint-selection rule, per-fold original-unit
MAE/RMSE/R², aggregate mean and spread, per-molecule test predictions, and checkpoint
artifacts with checksums. Document scaffold overlap checks and keep test folds out
of tuning and checkpoint selection. Do not relabel reconstructed or newly generated
experiments as evidence for the historical CV claim.

## New scaffold experiment (2026-10-08)

The corrected implementation now has a separate five-fold scaffold experiment:
**fold-mean test MAE 0.9636 ± 0.1534 logS**, pooled test MAE 0.9785 logS. This is
a new experiment, not evidence recovering the old 0.45 claim. One split seed (42)
and fold training seeds 42–46 were used; it is not five independent seed repeats.
See [full evidence](../results/scaffold-esol/README.md).

Supported new wording, if you choose to describe the corrected work:

> Implemented and evaluated MEGAN for ESOL aqueous-solubility prediction in PyTorch
> Geometric with scaffold-disjoint training/validation/test partitions; five-fold
> mean test MAE 0.964 ± 0.153 logS, with exact splits, baselines, and reproducible
> experiment artifacts.

The channel-ablation report has negative signed fidelity on two folds, so do not
claim universally reliable chemical explanations. The separate all-data demo refit
has no independent ESOL test score.

## Exploratory follow-up (2026-10-09)

The new eight-candidate validation-only search gives **MEGAN fold-mean test MAE
0.6651 ± 0.0979 logS**, pooled MAE 0.6613. A separate validation-channel-polarity
selector gives 0.6759 ± 0.1041; the descriptor-plus-Morgan random forest gives
0.6182 ± 0.0669. See [full evidence and limitations](../results/scaffold-tuning/README.md).
All candidates use the earlier outer folds, which were already inspected; describe
this as an exploratory follow-up rather than an untouched benchmark or evidence
of the best achievable performance. These new results do not recover the old 0.45 claim.

Supported wording for the follow-up, if you choose to use it:

> Implemented and evaluated MEGAN with categorical embeddings and split optimizer
> updates for ESOL solubility; exploratory five-fold scaffold evaluation with
> validation-only model selection achieved MAE 0.665 ± 0.098 logS, with reproducible
> splits, stronger descriptor baselines, and explanation-channel ablation checks.

Here the spread is population SD across folds, not an uncertainty interval or
independent training-seed repetition. The constrained selector has both expected
channel signs on 66.7% of test molecules and falls back in one fold; do not claim
universally reliable explanations. The 93-epoch all-data refit has no independent
ESOL test score. The random forest's lower MAE should remain visible in the research
report even when the CV describes the MEGAN implementation.
