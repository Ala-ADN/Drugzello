# Reproducing an architecture without outsourcing the reasoning

Your goal is to turn a paper into a set of tensor operations you can explain,
implement, and falsify. Start with a tiny graph and a shape table. Training a model
before checking these operations can hide a wrong architecture behind plausible
metrics. The [notebook](../notebooks/megan_pytorch.ipynb) is the executable companion;
[the critique](notebook-critique.md) records the original notebook's mistakes.

## 1. Separate the three claims you need to establish

An **architecture reproduction** implements the intended forward computation.
A **training reproduction** also implements the objectives, optimization, and
preprocessing. A **benchmark reproduction** additionally matches data, splits,
selection rules, and reported metrics. Passing a shape assertion establishes only
one small part of the first claim.

The target is [MEGAN, Teufel et al., arXiv v2](https://arxiv.org/html/2211.13236v2),
particularly its encoder, importance masks, channel-weighted readout, co-training,
and ablation scheme. The paper's molecular experiment uses AqSolDB; our ESOL
experiment is a transfer of the method to a different dataset. Optional residuals,
LayerNorm, and the chosen hyperparameters are explicit project variants. We use a
joint gradient update for the first recorded experiment. The follow-up search
also tests explanation-then-prediction split updates. We do not claim an exact
replication of the paper's full training implementation or published scores.

Before writing code, make a ledger: component, equation/figure, input/output shape,
implementation choice, and a check that would fail if it were wrong. For ambiguity,
inspect [the authors' implementation](https://github.com/aimat-lab/graph_attention_student)
and record its revision rather than silently guessing. Its current version has
extensions beyond the 2023 model; the newest code is not automatically the original
experiment. Keep unresolved differences in the reproduction notes.

## 2. Build a tensor contract

For a batch of molecular graphs:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N, F]` | Node features, all graphs concatenated |
| `edge_index` | `[2, E]` | Source indices in row 0, destinations in row 1 |
| `edge_attr` | `[E, D]` | Bond features, in the same order as edges |
| `batch` | `[N]` | Graph ID for each node |
| `A` | `[E_effective, K]` | Raw attention scores for explanation channels |
| `edge_importance` | `[E, K]` | Returned scores for original directed bonds |
| `node_importance` | `[N, K]` | Learned masks used in readout |
| channel graph vectors | `[B, K * H]` | Concatenated weighted graph embeddings |
| prediction | `[B, 1]` | Standardized aqueous solubility during training |

Here `N` is the total node count, not the number of graphs. PyG offsets node IDs
when batching. Never infer atom count from edge count. Molecular graphs already
contain both bond directions; applying `ToUndirected` again can add duplicate
categorical edge indices. We use the same PyG representation for training and
inference, with implicit hydrogens and categorical indices cast to floats. This
is a documented encoding choice, not a claim that ordinal features are optimal.

Added self-loops are internal computational edges with zero bond features. They
contribute to node gating, but are excluded from returned physical-bond scores.
Inference averages the two directed scores for each RDKit bond. Atom IDs remain
in the parsed SMILES order. Changing graph preprocessing requires a new feature
schema and retraining; it is not a compatible checkpoint update.

## 3. Derive attention before reading its implementation

For every edge `s -> d`, project the source/message and destination/query features:

```text
message_s = W_source x_s
query_d   = W_target x_d
A_(s,d)   = aᵀ LeakyReLU(query_d + message_s + W_edge e_(s,d))
alpha     = softmax(A) over edges sharing destination d
new_h_d   = sum over incoming s: alpha_(s,d) * message_s
```

The edge term is optional. Separate source/query projections and the location of
the nonlinearity are important. If you project to a scalar first and then apply a
monotone LeakyReLU, you have the static ranking issue of GAT. Compare the operation
with the [PyG GATv2 source](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/conv/gatv2_conv.html).
Our test copies weights into its reference layer and checks matching outputs.

Hand calculation: two nodes have scalar features 1 and 3. With a self-loop and
a neighboring edge at each destination, uniform attention gives `(1 + 3)/2 = 2`.
Changing the neighbor to 7 must change the destination result to 4. The old
backend multiplied destination features by every incoming attention weight; those
weights sum to one, so its evaluation-mode aggregation returned the destination's
own projected feature. Dimensions were correct while neighbor aggregation was wrong.

Implement this one layer and test it before stacking layers. Then check permutation
equivariance: relabel node IDs, permute features, remap edges, and verify that node
outputs permute consistently. Graph predictions should be invariant. Test batching
by comparing a two-graph batch with the same graphs evaluated individually.

## 4. Add MEGAN's explanation path to the prediction path

Each encoder layer runs `K` attention channels, applies ELU, and averages their
node embeddings. The next layer receives those embeddings. Explanation logits
are kept separately by layer and channel, **without detaching them**.

After the last layer:

```text
E_im  = sigmoid(sum of raw edge logits across layers)
E_pool = mean of incoming effective-edge importance at each node
V_gate = sigmoid(node-importance MLP(final node embeddings))
V_im  = V_gate * E_pool
z_k   = graph-wise sum(final node embeddings * V_im[:, k])
y     = prediction MLP(concat(z_0, z_1, ...))
```

A mean edge-to-node pool keeps the mask bounded; summing incident scores makes
importance grow with degree. A sigmoid edge mask has a different purpose from
attention softmax: attention normalizes competing messages, while importance
acts as an independent gate. Channel averaging belongs in the encoder; the
channel-specific graph vectors are concatenated at readout.

Put an optional `[N, K]` importance mask in this same forward path. A mask of ones
must reproduce the original prediction. A mask of zeros must leave only the
prediction head's response to zero graph embeddings. Avoid reconstructing a
second encoder for ablation: differences in self-loops or normalization would
confound the comparison.

## 5. Make auxiliary objectives actually train the masks

The maintained regression setup uses two channels around a training-set reference.
After training-only standardization, the reference is zero. Pool node importance
into two nonnegative counts per graph. For a below-reference target, train channel
0 toward `reference - target`; for an above-reference target, train channel 1 toward
`target - reference`. Equal-to-reference samples contribute no directional error.

This makes the count represent a **magnitude**. The paper's displayed regression
expression has a sign ambiguity for a nonnegative lower-channel count; we explicitly
use the magnitude convention implied by the intended opposing channel roles. Record
that choice when comparing implementations rather than claiming byte-for-byte loss
identity from the equation alone.

```text
prediction loss = mean squared prediction error
explanation loss = active-channel squared magnitude error summed / (2 * batch_size)
sparsity loss = mean absolute node importance
total = prediction + gamma * explanation + beta * sparsity
```

The optional channel-cosine penalty is a project extension and defaults to zero.
Ground-truth attribution supervision is a different objective; ESOL has no such
labels. Co-training encourages channel roles but does not prove chemical validity.

Check gradients for each loss independently with `torch.autograd.grad`. A nonzero,
finite gradient from explanation/sparsity loss to an attention parameter proves
that a training path exists. It does not prove that optimization will produce useful
explanations. Deliberately insert `detach()` as an exercise: the gradient check
should fail. A loss computed entirely inside `no_grad()` cannot train anything,
even when multiplied by a large coefficient.

## 6. Treat the experiment protocol as part of the implementation

The [splitter](../backend/src/data/scaffold.py) uses RDKit
[Bemis–Murcko scaffolds](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
without chirality. Scaffold atom and bond types are retained; this is not a generic
all-carbon topology split. Large scaffold groups are greedily assigned to the
smallest outer fold; seeded shuffling resolves group-size ties. A separately
seeded inner group selection approximates the validation fraction.

All acyclic molecules have the empty scaffold, so they are kept in one group.
Giving each acyclic molecule a different artificial scaffold would make folds look
balanced while violating this policy. Exact fold sizes can therefore differ greatly.
The manifest stores row IDs, SMILES, scaffold IDs, policy, seed, and all partitions.
Invalid molecules and insufficient scaffold groups fail explicitly. Check disjoint
scaffold sets and complete coverage; checking just disjoint row IDs is insufficient.

For each outer fold:

1. Fix train, validation, and test molecule IDs without reading target values.
2. Fit target mean/std on train only. Apply that transform to the other partitions.
3. Train and choose the checkpoint using validation MAE only. A fixed fold's
   normalization means standardized and original-unit MAE choose the same epoch.
4. Clone checkpoint tensors, restore the selected weights, and then predict test rows.
5. Inverse-transform predictions and report original-unit MAE, RMSE, and R².
6. Train baselines on the same training rows. The supplied comparisons are a training
   mean and a radius-2, 1024-bit Morgan random forest with 200 trees and fixed seed.
7. Save predictions, normalization, histories, config, split IDs, source hashes,
   resolved package versions, and checkpoint checksums.

The implementation seeds Python, NumPy and PyTorch and enables deterministic
algorithms. Each fold uses `base_seed + fold_idx` for training. This is one
cross-validation experiment, not five independent random-seed repetitions. A
scheduler may observe validation loss; it must never observe outer test metrics.

With unequal test sizes, the mean of fold MAEs differs from pooled out-of-fold MAE.
Report both, and state that the supplied fold spread uses population standard deviation.
Do not pick the lowest test fold and call it the result. Do not rename validation
results as tests or normalize the entire dataset before splitting.

## 7. Evaluate channel roles without manufacturing an explanation

Using the same trained forward computation, zero channel 0 and channel 1 in turn.
For two-channel regression, calculate signed original-unit changes:

```text
delta_negative = prediction_without_channel_0 - full_prediction
delta_positive = full_prediction - prediction_without_channel_1
fidelity_star  = mean of the two signed changes, averaged across test graphs
```

Negative values indicate behavior inconsistent with the intended channel polarity.
Do not replace them with absolute changes: that loses the distinction. Also report
mean node/edge mask values as continuous sparsity diagnostics. These are not
thresholded retained-node fractions or attribution accuracy. There are no ESOL
attribution labels to compare against, and a good fidelity number alone would
not establish a chemical mechanism.

MC dropout is an uncalibrated spread estimate. Deterministic inference stays in
`eval()`; repeated dropout samples temporarily enter training mode and always
restore evaluation mode. The API returns `confidence: null` because no calibration
experiment establishes a probability of correctness. Language-generated narratives,
random masks, RDKit descriptors, and actual target labels are not substitutes for
learned explanation masks.

## 8. Keep deployment separate from evaluation

A fold checkpoint was trained only on that fold's training partition. It can be
used to reproduce its saved test predictions. A separate all-data refit trains
for the median validation-selected epoch count after reports are finalized. It
is useful for the demo, but has **no independent ESOL test score**. The checkpoint
records this distinction and contains only tensor state plus primitive metadata.

Inference checks architecture/feature versions, restores weights strictly, uses
the shared SMILES conversion, and applies the saved inverse normalization. Missing,
legacy, or incompatible weights return unavailable status and HTTP 503. Model health
reports actual load state. Never make an API appear healthy by returning a heuristic.

## 9. A practice sequence you can do alone

Close the implementation and write a 20-line scalar attention example. Derive its
outputs by hand. Open the source only after your example works. Then add the
shape table, batching, channel masks, pooling, losses, and checkpoints one piece
at a time. Keep a one-page decision ledger of every choice the paper does not fix.

Before asking an LLM for help, write your expected behavior, actual behavior,
smallest reproducing input, and the exact equation or invariant involved. Ask it
to challenge one specific derivation or failing check. Verify the answer with a
calculation, the reference implementation, or an independent test. Do not accept
“the shapes match” as an architecture review.

The regression suite is the boundary of what we have checked. Different channel
losses, repeated seeds, or the original dataset require new configuration, tests,
and experiments. Those changes should not be silently folded into an old result.

## 10. Derive the follow-up experiments

**Categorical embeddings.** A category index identifies a table row; its numeric
distance from another index has no chemical meaning. For each of PyG's nine atom
fields and three bond fields, create a separate lookup table with 16 columns. Look
up each field and concatenate the rows: atom inputs become `[N, 9 * 16]`, and bond
inputs become `[E, 3 * 16]`. Attention then projects those vectors as before. The
tables learn through the prediction and explanation objectives. A one-hot vector
multiplied by a learned matrix implements the same lookup operation.

Test this without training: swap two category IDs and the corresponding table
rows, then check that predictions are unchanged. This catches unintended reliance
on category ordering. Also check gradients reach both atom and bond tables. Encode
physical bonds before adding self-loops, so synthetic loops get zero vectors rather
than a learned embedding for category zero. Embedding checkpoints have a distinct
architecture version; ordinal checkpoints remain supported.

**Split updates.** In the tested variant, each batch first updates the encoder and
importance path using `gamma * explanation_loss`. The prediction head has no gradient
in that step. Clear gradients, recompute a fresh forward pass with the updated
parameters, then update with `prediction_loss + beta * sparsity_loss` (plus optional
decorrelation, zero in this search). Both steps share an AdamW optimizer and clip
gradient norm to one. The unit test observes both optimizer steps and verifies
which parameters receive gradients. This implements the paper's update order while
retaining our documented magnitude loss, encoding and optimizer choices.

**Validation-only search.** Read `backend/configs/scaffold_search.yaml` before running
anything. It fixes eight candidates, a 150-epoch limit and patience 20. The first
candidate extends the ordinal baseline's training budget; matched 32/64-unit
variants compare joint and split updates. Other candidates vary depth, learning rate,
dropout and auxiliary weight. These last comparisons change several factors, so
they are search candidates rather than isolated causal ablations.

Every outer fold selects independently on its inner validation set. One selector
minimizes MAE. The other minimizes MAE among candidates with **both** channel-mean
signed validation ablations nonnegative, falling back explicitly if none qualifies.
Freeze those choices and baseline choices before evaluating outer tests. Validation
polarity is a selection criterion, not a guarantee about test polarity. Report both
selectors rather than deciding between them using their outer test scores.
The report also measures each channel's correct-polarity fraction and the fraction
of molecules with both channels behaving as intended; a positive average can hide
individual failures. No atom-level attribution accuracy is implied.

The search also freezes three fixed test ablations: longer ordinal training,
32-unit embeddings with joint updates, and 32-unit embeddings with split updates.
These isolate the main changes more clearly than the independently selected pipeline.
Split training performs two optimizer updates per batch, so the comparison is
epoch-matched rather than matched for compute or number of updates. All three
ablations are reported; none is selected using its outer test score.

The deployment rule is fixed separately: use the most frequent fidelity-constrained
validation winner, break ties by its mean validation MAE, then refit on all data for
its median validation-selected epoch count. This refit still has no independent test
score. The earlier outer folds have already been inspected, so the follow-up is an
exploratory comparison on those splits, not a newly untouched benchmark.

The [completed follow-up report](../results/scaffold-tuning/README.md) provides the
fixed configurations, every validation trial, held-out predictions, and channel
ablations. Use its fixed ordinal/joint-embedding/split-embedding comparisons to
trace which changes helped, while remembering that split updates double the
updates per batch. The descriptor-plus-Morgan random forest had lower MAE than
the evaluated MEGAN pipelines. Replaying the artifacts is a useful exercise:
recompute MAE from the prediction CSV, then each molecule's signed channel changes
from the ablation CSV before looking at the recorded summary. Positive average
fidelity did not imply that both channels behaved correctly for every molecule.
