# Source snapshot for the first scaffold experiment

[source.tar.gz](source.tar.gz) preserves the exact 37 source files used for
[`results/scaffold-esol`](../../results/scaffold-esol/README.md), before categorical
embeddings and split-step tuning were added. Every file matches its recorded
SHA-256 in that experiment's `environment.json`.

This is an archival snapshot, not an active import path. Maintained code lives in
`backend/`; reproduce new searches with `backend/scripts/tune_model.py`. The first
experiment used ordinal features, joint updates and at most 30 epochs. Its local
checkpoints remain separate from the tuned experiment.

To run the archived experiment, use a separate copy of the repository and overlay
the archive's `backend/` directory there (extract with `tar -xzf source.tar.gz`), then use the first experiment's recorded
environment, config and commands. Do not overwrite your maintained working source.
