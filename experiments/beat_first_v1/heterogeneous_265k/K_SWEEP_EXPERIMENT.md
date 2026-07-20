# Post-baseline active-width sweep

Status: **UNEXECUTED DESIGN CONTRACT**

This experiment is deliberately separate from the frozen `(26,22,22)`
post-idle A0/H0/H1 gate and from NNUE/search-delta reuse.  The frozen gate is a
correctness and systems baseline; it does **not** satisfy the requested
`k=64/128/256` comparison.

## Fixed model and active-width variants

All variants keep the same three neuron banks, the same 19,924,022 total
parameters, the same feature contract, the same WTA tables, and the same exact
rerank caps.  Only the number of selected neurons changes:

| label | per-layer active counts | total active neurons | active parameters | forward MACs | inclusive parameter-training MACs |
|---|---:|---:|---:|---:|---:|
| `k64` | `(24,20,20)` | 64 | 31,934 | 31,912 | 78,504 |
| `k128` | `(48,40,40)` | 128 | 58,214 | 58,192 | 141,520 |
| `k256` | `(96,80,80)` | 256 | 110,774 | 110,752 | 267,552 |

The 22-parameter difference between active parameters and forward MACs is the
head bias.  Inclusive training counts include the selected parameter VJP and
the two CountSketch traversals.  Routing lookup, bucket traversal, ID
deduplication, sorting, memory gather, and optimizer bookkeeping remain timed
stages rather than hidden MACs.

`initialize_model(...; active_counts=...)` exposes the required model geometry.
The current runtime source appears to derive each WTA target and workspace size
from `layer.active_count`, but that path is unexecuted.  A width-specific smoke
must prove exact selection counts, finite forward/VJP, active-only updates, and
checkpoint continuation before a sweep result is accepted.  The production
exact constructor and frozen validator remain fixed at `(26,22,22)`; no
exploratory result may be passed off as a frozen-gate result.

For every width, forward and parameter VJP may read only selected bank IDs plus
the small head.  The hot optimizer step may update only the stable-reduced
selected bank IDs plus the head.  Unselected bank parameters, moments, event
counters, and timestamps must remain unchanged; whole-bank gradient or
optimizer traversal is a failure even if the final numeric output is correct.

## Comparisons

1. **Pure CPU latency comparison**: `k64`, `k128`, and `k256`, without
   HashStem, use one common frozen full-bank checkpoint, candidate corpus,
   input order, thread placement, and optimizer-free inference.  Changing `k`
   is the only model difference in this comparison.
2. **Pure CPU learning comparison**: all widths start from the same initializer
   bytes, then train independently with the same state order, candidate
   exposures, update budget, seeds, loss, and sparse optimizer semantics.
   Resulting weights are expected to diverge and must not be described as the
   “same weights.”
3. **HashStem device comparison**: the same learned HashStem weights on CPU and
   NPU, followed by the same CPU sparse width.  Packing, submit, transfer, wait,
   tail, routing, gather, and sparse compute are all inside wall time.
4. **System criterion**: NPU HashStem + CPU sparse `k128` must have end-to-end
   p50 latency no greater than the pure-CPU `k64` system while improving held
   teacher ranking and later fixed-development game score.
5. `k256` is promoted only when its ranking/game gain pays for latency and p95;
   total parameter count alone is never an adoption reason.

## Gates

- Run only after the current single heavy Julia process and the frozen
  `(26,22,22)` correctness gate terminate authoritatively.
- Use identical input ordering, seeds, rerank caps, and CPU sets.  Latency and
  device comparisons use identical frozen weights; learning comparisons use
  identical initializer bytes and permit only the subsequent width-specific
  active updates to diverge.
- Record total/active parameters, active edges, learned MACs, routing MAC cap,
  bytes gathered, p50/p95 stage and end-to-end latency, updates/s, ranking
  metrics, and inactive-state invariants.
- Against CPU HashStem at the same `k128`, the NPU path requires at least 1.15x
  end-to-end speedup, no p95 regression, effectively unchanged rankings under
  the numeric tolerance, and no slowdown greater than 1.10x in concurrent
  P-core sparse work.
- Against the pure-CPU `k64` system, NPU HashStem + CPU sparse `k128` requires
  both p50 and p95 end-to-end latency to be no worse.  Held teacher ranking must
  improve by at least +0.02 absolute top-1, have a paired bootstrap 95% lower
  confidence bound above zero, and have non-decreasing NDCG and pairwise
  accuracy.  A later fixed-development game gate requires mean paired score
  improvement of at least +500, a positive difference on a majority of seeds,
  and no completion-rate regression.  These development results are promotion
  evidence only, not a final statistical claim.
- NPU/iGPU results are system evidence, never evidence that the pure-CPU sparse
  model itself improved.
- Use development seeds only after supervised eligibility; validation seeds
  `8001:8008` and sealed seeds `91001:91032` remain untouched.

After the sweep, freeze a new source/result contract for the winning width.
Until then, `(26,22,22)` remains the only frozen heterogeneous baseline and no
`k128` adoption claim is authorized.
