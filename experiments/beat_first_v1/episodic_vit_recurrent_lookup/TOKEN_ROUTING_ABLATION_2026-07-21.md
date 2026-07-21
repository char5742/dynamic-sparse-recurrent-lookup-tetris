# EVRL input-token routing ablation — 2026-07-21

## Decision

The register-to-episodic-memory path no longer performs learned hash/WTA
retrieval from 283 tokens to 64 candidates followed by an exact top-16
shortlist.  Every register now attends all 283 input tokens.

This change is deliberately limited to episodic input access.  The exact
PreAct input contract, physical local-8 cell attention, register
self-attention, SwiGLU, LookupFFN parameter routing, active-only sparse bank
updates, shared recurrence, loss, optimizer semantics, and fixed depth-two
comparison setting are unchanged.

The implementation projects the complete token memory to one shared
`32 x 283` K matrix and one shared `32 x 283` V matrix per recurrent step.
Four register queries then evaluate an exact four-head softmax over those
shared rows.  There is no token candidate table, top-k support, dense mask, or
routing STE.  Consequently every episodic token has a direct task-gradient
path through K/V.

## Motivation

The previous token router was introduced to make register input access
physically sparse on CPU.  At the current fixed size of 283 tokens there was
no controlled result showing that its information loss was justified.  Cell
tokenization and recurrent spatial updates were already paid in full, while
only the final path into registers could discard tokens and withhold credit
from unselected positions.  This ablation tests that restriction directly.

## Correctness witness

The canonical serial candidate state machine and the 20-worker barrierless
executor were independently restored from the same new checkpoint.  The
smoke used four real-teacher training states, capped to four candidates per
state only to keep the full parameter/optimizer comparison short.

| Witness | Maximum absolute difference | Relative L2 |
|---|---:|---:|
| Output | 0 | 0 |
| Loss | 0 | 0 |
| Raw output VJP | 0 | 0 |
| Worker gradient | 1.31130219e-6 | 5.69284813e-8 |
| Reduced parameter gradient | 5.21540642e-8 | 5.78742739e-8 |
| Post-optimizer parameter/state | 1.49011612e-8 | 1.75511969e-10 |

Candidate RNG, depth, hard halting, full-token support shape, Lookup row IDs,
active rows, usage, optimizer clocks, sparse row clocks, sampler state, and
post-update RNG state were exact.

## Training

The full-token model was trained from a fresh initialization for the same
12,000 updates / 48,000 teacher states as the budget-matched routed model and
PreAct baseline.

| Field | Value |
|---|---:|
| Parameters | 20,577,224 |
| Updates | 12,000 |
| Consumed states | 48,000 |
| Training throughput | 23.3685 updates/s |
| Candidates/s | 4,077.17 |
| Last composite loss | 3.13755 |
| Last ListNet loss | 3.00577 |
| Last old-Q loss | 0.465372 |
| Last margin loss | 0.0345595 |

Final checkpoint:

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_fixed2_u12000_20260721_r1\checkpoints\checkpoint_000012000.jls
bytes: 253659317
sha256: 80cc8264a03facf5ff4d0c13cde205b0763012281254362b2f15521c262a4f1c
```

The binary checkpoint is not committed because it exceeds the normal GitHub
object limit.

## Held-teacher result

The comparison reuses the exact earlier 128-state panel:

```text
dataset manifest: 1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded
split seed:        2026071817
subset seed:       2026072315
row-list SHA-256:  fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25
```

Teacher Q and rank remain targets only.  Each candidate is evaluated
independently.  No game validation or sealed seed was touched.

| Model at 12k / 48k states | Top-1 | NDCG | Pairwise | Margin | Composite loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|
| PreAct-ECA | 0.78906 | 0.99329 | 0.92336 | 0.12332 | 2.56378 | 4.1925 |
| Routed EVRL (64 -> 16) | 0.35938 | 0.97127 | 0.82028 | 0.04954 | 2.97502 | 53.5423 |
| Full-token EVRL (283) | **0.56250** | **0.97867** | **0.84152** | **0.07205** | **2.80916** | **45.1745** |

Full-token EVRL versus routed EVRL:

- top-1: `+0.203125`;
- NDCG: `+0.0074061`;
- pairwise accuracy: `+0.0212389`;
- action margin: `+0.0225019`;
- composite loss: `-0.1658505`;
- inference throughput: `0.843715x` (15.63% slower).

Full-token EVRL remains behind PreAct by `0.2265625` top-1 and `0.0146179`
NDCG, but it is `10.7749x` faster in this CPU held-panel measurement.

## Conclusion

At 283 tokens, hard input-token routing was a harmful bottleneck rather than
a necessary efficiency mechanism.  Removing it recovered a large amount of
ranking quality for a modest inference cost and no observed training
throughput loss.  LookupFFN parameter sparsity remains intact, so this result
does not argue against dynamic long-memory routing; it argues specifically
against discarding the already-tokenized short-term episodic memory at this
scale.

Machine-readable evidence is in
[`token_routing_ablation_2026-07-21.json`](token_routing_ablation_2026-07-21.json).
