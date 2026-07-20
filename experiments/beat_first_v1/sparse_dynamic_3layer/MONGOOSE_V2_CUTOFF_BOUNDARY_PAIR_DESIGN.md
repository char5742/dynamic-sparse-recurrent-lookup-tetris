# MONGOOSE v2 fixed-k cutoff-boundary pair mining

## Scientific isolation

This arm changes one variable from bounded MONGOOSE v2: the per-layer pair
mined for the existing `pair_bce_gradient!` router surrogate.

The following remain frozen:

- teacher dataset, split, sampler, task loss, and highest-raw-gradient candidate;
- `pair_bce_gradient!`, its BCE form, and all 2,688 router parameters;
- warmup and refresh cadence at 2,000 updates;
- active widths `(48,40,40)` and reserved probes `(6,5,5)`;
- bucket, table/lane, shortlist, exact-score, fill, and probe bounds;
- active-only bank forward/backward and sparse optimizer updates.

No dynamic-k, load-balance regularizer, student hard-negative role, task-loss
change, cap change, dense fallback, NNUE path, game seed, or bank scan belongs
to this arm.

## Exact boundary rule

The unchanged bounded fixed-WTA witness is rerun for the stable
highest-raw-gradient candidate.  For each layer:

1. Start only from rows that the bounded witness retrieved and exact-scored.
2. Keep only rows with positive WTA collision count.  Deterministic underflow
   fills and reserved training probes have collision count zero and are
   excluded from both sides of the pair.
3. Order this natural exploitation pool by exact raw score descending, then
   stable neuron ID ascending.
4. Let `k_exploit = active_count - training_probe_count`, giving
   `(42,35,35)`.
5. The positive is rank `k_exploit`; the negative is rank `k_exploit + 1`.
6. Fail closed if the bounded natural pool has no rank `k_exploit + 1`.

The learner publishes both IDs, both scores, both selected collision counts,
score gap, ranks, exploitation membership, exploitation width, and bounded
eligible-row count for every layer.  Each selected collision count must be at
least one; this makes fill/probe exclusion observable in the run artifact.
For an equal-score boundary, the positive ID must be smaller than the negative
ID, making the stable-ID tie-break independently checkable.

## Identity and continuation boundary

The opt-in mining symbol is
`fixed_wta_exploitation_cutoff_boundary_v1`.  Its complete policy tuple is
stored in the run config and checkpoint metadata, and the training state also
stores the selected mining symbol.  Easy-extrema and cutoff-boundary metadata
have different key sets; restore validates exact config, metadata, and state
identity.  A cutoff run must start fresh.

## Deferred execution

The deterministic test source is
`test_mongoose_v2_cutoff_boundary_pair.jl`.  This implementation task does not
launch Julia; the tests remain a serialized launch gate for the owning agent.
