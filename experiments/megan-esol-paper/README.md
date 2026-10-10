# Original MEGAN ESOL benchmark source

This archive pins [KGCNN commit `ceaaba2d45e2273f12d46f2d0ff9a167b8e1df8e`](https://github.com/aimat-lab/gcnn_keras/tree/ceaaba2d45e2273f12d46f2d0ff9a167b8e1df8e), version 2.1.0, from September 2022. Its recorded final-epoch results round to the [MEGAN paper's Appendix C / Figure 6](https://arxiv.org/pdf/2211.13236#page=24): MAE **0.44 ± 0.03** and RMSE **0.60 ± 0.05**, over random five-fold ESOL cross-validation.

`original_hyper.json` and `original_score.yaml` are unmodified upstream files. `provenance.json` contains their upstream paths, every bundled source hash and the archive hash. `source.tar.gz` bundles the original `kgcnn/` package, training script, configuration, results, requirements and license. The runner verifies all hashes before extracting it into ignored `.runtime/kgcnn-esol-paper/`.

The original final MAEs are 0.4779, 0.4190, 0.4073, 0.4732 and 0.4071 logS; mean 0.4369 and population SD 0.0319. Original final RMSE is 0.6048 ± 0.0461. These are historical upstream results, not new Drugzello measurements. A January 2023 upstream rerun has different scores, so using that newer source would not target the exact table quoted in the paper.

The reproduction imports the original model and chemistry preprocessing without source edits. Drugzello supplies a local-CSV adapter, explicit training seeds, compact result export, resumable fold execution and checkpoint replay. TensorFlow 2.12 / Python 3.11 compatibility and CPU execution are recorded differences: the upstream artifact does not record complete runtime versions, hardware or training RNG seeds. Split seed 42 is recorded upstream.

The original model configuration uses three channels, concatenated attention heads, layer widths `[60, 50, 40, 30]`, MLP widths `[50, 30, 10, 1]`, 41 atom features, no bond features, dropout 0.3, and zero explanation/sparsity factors. It trains for 400 epochs with Adam and a fixed linear learning-rate schedule. It is different from Drugzello's maintained PyTorch implementation.

KGCNN source is MIT licensed; its original [license](LICENSE) and attribution are preserved. See the [reproduction report](../../results/paper-esol/README.md) for commands, measured results and limitations.
