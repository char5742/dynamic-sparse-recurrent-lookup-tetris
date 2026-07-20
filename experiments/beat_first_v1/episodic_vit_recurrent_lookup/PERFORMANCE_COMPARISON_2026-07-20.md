# Final held-teacher performance comparison — 2026-07-20

The corrected episodic ViT recurrent Lookup model was evaluated after training,
using the same immutable real-teacher validation panel as PreAct. This is a
negative accuracy result and a positive CPU-throughput result: the sparse model
does **not** beat PreAct in ranking quality, but it evaluates the panel about
12 times faster on this CPU-native implementation.

## Locked comparison contract

- teacher dataset manifest SHA-256:
  `1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`;
- split seed: `2026071817`;
- fixed validation subset seed: `2026072315`;
- 128 held teacher states; row-list SHA-256:
  `fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`;
- identical `board / candidate / difference / next_hold / aux37` inputs;
- candidate-independent evaluation with teacher Q and rank used only as targets;
- standardized ListNet-plus-margin objective and fixed teacher top-2 margin;
- observed width 76, padded learner width 80;
- Julia 20 threads, BLAS one thread, no game validation or sealed seed opened.

The PreAct best checkpoint and the EVRL update-12,000 checkpoint both consumed
48,000 training states. This is the primary equal-update/equal-state
comparison. EVRL update 20,000 is reported separately as the final trained
model and consumed 80,000 states.

## Results

| Model | Updates | States seen | Parameters | Top-1 | NDCG | Pairwise | Margin | Composite loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PreAct best | 12,000 | 48,000 | 1,481,326 | **0.7891** | **0.99329** | **0.92336** | **0.12332** | **2.56378** | 4.19 |
| EVRL budget-matched | 12,000 | 48,000 | 20,577,480 | 0.3594 | 0.97127 | 0.82028 | 0.04954 | 2.97502 | **53.54** |
| EVRL final | 20,000 | 80,000 | 20,577,480 | 0.3750 | 0.95837 | 0.76881 | 0.08783 | 3.16286 | **50.19** |

Against PreAct, the budget-matched EVRL result is:

- top-1: `-0.42969` absolute;
- NDCG: `-0.02202`;
- pairwise accuracy: `-0.10308`;
- action margin: `-0.07378`;
- composite loss: `+0.41123` (worse);
- CPU held-panel throughput: `12.77x`.

Against PreAct, the final EVRL result is:

- top-1: `-0.41406` absolute;
- NDCG: `-0.03493`;
- pairwise accuracy: `-0.15455`;
- action margin: `-0.03550`;
- composite loss: `+0.59908` (worse);
- CPU held-panel throughput: `11.97x`.

The inference timing is a warm steady-state measurement over three complete
passes of the same 128-state panel. PreAct used its fixed four-state batch;
EVRL used its candidate-independent one-state evaluation batch and physically
sparse routing. Compilation, checkpoint loading, and dataset loading are not
included.

## What was learned

1. **The fixed-batch overfit pass did not imply held generalization.** The
   spatial-credit repair made the architecture trainable and capable of
   memorizing four real-teacher states, but held top-1 remained far below
   PreAct.
2. **More updates did not close the quality gap.** From EVRL update 12,000 to
   20,000, top-1 rose only `0.01563`, while NDCG fell `0.01290`, pairwise
   accuracy fell `0.05147`, and composite loss increased `0.18785`. The model
   sharpened some selected actions without improving the full candidate
   ordering.
3. **Capacity was not the limiting scalar.** EVRL has `13.89x` as many stored
   parameters as PreAct, yet its held ranking is substantially worse. Sparse
   route discovery and credit assignment remain the primary unresolved issue.
4. **The systems claim is supported, the model-quality claim is not.** EVRL's
   input-dependent sparse execution is roughly 12 times faster than this
   PreAct CPU evaluation despite the larger parameter store. It therefore
   establishes a useful speed/accuracy trade-off, not a PreAct replacement.
5. **The final checkpoint must not be described as beating PreAct.** The
   strongest defensible conclusion is that physical sparse episodic/parameter
   lookup is fast and learnable, while the current router and recurrent state
   fail to recover PreAct's spatial ranking quality on unseen teacher states.

## Artifacts

- machine-readable result:
  [`performance_comparison_2026-07-20.json`](performance_comparison_2026-07-20.json);
- reproducible evaluator:
  [`evaluate_teacher_comparison.jl`](evaluate_teacher_comparison.jl);
- PreAct checkpoint SHA-256:
  `f3e40d7b6bd3ea8aa7930b2178b537bdae37eea76cdbf089c3ba489ac99d057e`;
- EVRL update-12,000 SHA-256:
  `a566d0e63eacddbc3e02a8e789f891fd18328953dd4fc443bdc6ac7009e5d858`;
- EVRL final SHA-256:
  `1fc05d63154fc73e5d60367c2b19d63116a975b0a3a772899b7fd0ca382db28e`.

The binary checkpoints and teacher dataset are not committed. Their hashes
and the fixed row-panel hash make the published comparison auditable without
misrepresenting those large local artifacts as repository content.
