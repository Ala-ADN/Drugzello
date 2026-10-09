# Review of the original MEGAN notebook

Reviewed input SHA-256: `c989830b03f105df2557b97c2fe7152df141dab373ebd4fbc47771d9ec95abb4`. The pre-edit notebook was an untracked local input; its temporary review copy is
no longer retained. Cell numbers below refer to that original notebook (zero-based). The edited notebook imports maintained
modules and clears stale outputs; it is not a source for old benchmark claims.

| Original cells | Finding | Correction |
| --- | --- | --- |
| 0 | Attention scores are projected to a scalar before LeakyReLU. That is GAT-style static attention, despite the GATv2 name. | Separate source/query projections, activation before the attention vector; a parity test against PyG GATv2Conv. |
| 1 | Summing incoming edge importance makes node importance degree-dependent and allows values above one. | Mean pooling of incoming effective edges, learned sigmoid node gates, masks bounded in [0,1]. |
| 2 | Random graphs include self-loops and duplicates; shape checks establish neither semantics nor paper equivalence. | Molecular graphs, explicit self-loop policy, hand-calculated messages and neighbor-perturbation checks. |
| 5–6 | Fidelity reconstructs a different encoder without the original added self-loops; the explanation fidelity loss runs inside no_grad. | One forward method accepts a channel mask; explanation co-training and sparsity retain gradients. |
| 6 | The outer fold named test is repeatedly used for checkpoint selection; saved state_dict tensors can keep changing. | Separate inner validation and outer test; clone checkpoint tensors and restore them before test evaluation. |
| 8–12 | Random KFold settings vary between cells and searches; no scaffold grouping or saved molecular split IDs. | One deterministic scaffold manifest shared by the notebook and CLI, including acyclic policy and exact row/scaffold IDs. |
| 15 | Evaluation constructs a fresh randomly initialized model without loading each fold's saved weights. | Reload the exact saved fold checkpoints; compare CSV predictions against checkpoint inference in a regression test. |
| 18 | create_molt5_prompt is defined twice; generated text is mixed with graph explanation claims. | Learned masks and measured ablations are the explanation output; language generation is outside this experiment. |
| 19 | A missing saved model falls back to the actual dataset label and presents it as a prediction. | Explicit checkpoint failure; no label substitution, heuristic solubility, or random explanation masks. |
| Throughout | Thousands of code lines duplicate application code, use hidden global state, and store large stale outputs. | Short tutorial cells importing tested source, explicit configuration, seeded fixtures, optional full ESOL run, no committed outputs. |

The notebook had more of the intended MEGAN importance/readout structure than the
old backend, and correctly used source-node messages. It still lacked reliable
experiment boundaries. The edited notebook preserves its educational purpose
while making its implementation authoritative through shared modules.

The historical results remain in their separate archive. New scaffold experiments
must be dated and reported independently; adding a splitter does not validate the
old CV number. See [the learning guide](reproducing-megan.md).
