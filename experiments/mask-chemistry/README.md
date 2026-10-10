# How to test molecular explanation masks

The [recorded experiment](../../results/mask-chemistry/README.md) evaluates three frozen MEGAN model sets. Start with the [protocol](protocol.json), then read the small metric functions in [masks.py](../../backend/src/evaluation/masks.py). The model runners handle framework-specific inference; chemistry, controls and aggregation are shared.

## 1. Separate three questions

**Chemical agreement:** Does an atom score follow a specified chemical pattern? Here the pattern is polar N/O atoms versus nonpolar carbon atoms. Passing this check is useful, but does not make these atom classes ground-truth contributions to solubility.

**Readout faithfulness:** Does a highly ranked atom contribute more to this model's output? Compare its pooling removal with matched-size random removals. This is an intervention on a neural network, not on a molecule in water.

**Prediction benefit:** Would a model trained without masks generalize better or worse? That requires a separate matched architecture ablation and cannot be answered by explanation quality alone.

## 2. Trace masks through the prediction

At inference, MEGAN constructs an embedding `h_i` and nonnegative mask `m_ik` for each atom and channel. Its pooled vector contains one block per channel:

```text
z_k = sum_i h_i * m_ik
prediction = readout(concatenate(z_0, ..., z_(K-1)))
```

A large mask has no inherent positive/negative meaning. A signed embedding or the readout can turn its contribution negative. Removing an atom's pooling contribution means subtracting `[h_i*m_i0, ..., h_i*m_i(K-1)]` from the pooled vector while leaving the encoder untouched.

Verify that this reconstruction reproduces the unchanged model prediction before using it to explain anything. The PyTorch runner captures embeddings with a temporary forward hook. The original KGCNN runner uses the model's existing attention, pooling and readout layers. Neither trains or replaces model parameters.

## 3. Give a hypothesis a chance to fail

Choose motifs and metrics before looking at attractive heatmaps. Check that RDKit atom IDs map to model nodes. Restrict the polar-versus-carbon AUC to molecules containing both groups; otherwise the AUC is undefined.

For one molecule, AUC is the fraction of polar/nonpolar atom pairs where the polar atom has a higher score, giving half credit for ties. A constant mask scores 0.5. Average per molecule so a large molecule does not dominate through its number of atom pairs.

If channel polarity is unspecified, determine a convention using training molecules only. Our convention uses the mean prediction change from removing a whole channel, then normalizes masks within each channel before combining them. This avoids choosing a test-favourable channel direction. Check nominal channel labels separately; calibration should not hide their failure.

## 4. Use controls that target different shortcuts

Shuffling scores within a molecule preserves its composition and the score distribution while breaking atom alignment. Randomizing model parameters tests whether apparent chemistry arises from architecture and atom features alone. Random pooling deletions preserve the number of selected atoms and test whether a learned ranking adds information.

Random weights can score above 0.5 because chemical categories are already present in the inputs. Compare the learned model with that control, rather than accepting any nonrandom-looking heatmap. One initialization per fold is a useful initial check; several random seeds would make the control more robust.

## 5. Quantify uncertainty without multiplying sample size

Atoms in a molecule are dependent, as are molecules sharing scaffolds. Compute per-molecule measurements first and bootstrap scaffold clusters. Keep every acyclic molecule in the same empty-scaffold group, matching the split convention.

These intervals condition on the already trained models. They do not cover the variability from retraining with new seeds. Use paired differences on the same molecules for the two matched scaffold models; overlap of separate confidence intervals does not answer a paired comparison.

## 6. Audit before interpreting

Run `make mask-chemistry` with the existing local checkpoints. Run `backend/scripts/summarize_mask_chemistry.py` to audit saved metrics independently with scikit-learn and SciPy. The four new tests check atom mapping, AUC ties/missing classes, known linear-readout interventions, channel-scale invariance and scaffold-cluster aggregation.

For practice, work out the test molecule `CCO` by hand. The first carbon is in the nonpolar proxy; the carbon bonded to oxygen is excluded; oxygen is in the polar proxy. Follow the synthetic linear readout test to verify each signed contribution. Then inspect the deterministically selected real molecular examples, including cases that disagree with the aggregate trend.

The observed result is that chemical signal can emerge from prediction training alone. Chemical plausibility, stable channel semantics and better predictive accuracy are separate claims, each requiring its own evidence.
