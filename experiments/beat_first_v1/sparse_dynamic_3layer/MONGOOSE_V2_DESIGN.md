# Bounded MONGOOSE-inspired v2, fixed-k design freeze

Status: preregistered design; implementation and production signal run are not
yet authorized by this document.

## Scientific question

MONGOOSE v1 stopped before its first learned-routing metric because exact
bucket enumeration exceeded the layer-2 visit cap.  That event falsified the
v1 serving implementation, not the learned-routing quality hypothesis.  V2
therefore changes the retrieval/index layout and its accounting, while keeping
the teacher task, learned projection, fixed active widths, and bank update path
unchanged.

## Frozen identity

- Routing mode: `mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2`.
- This is MONGOOSE-inspired learned SimHash, not a reproduction of the ICLR
  implementation.
- Route dimension `64`, two tables, seven bits per table, 128 logical buckets
  per table, and `2,688` learned projection parameters remain unchanged.
- Active widths remain exactly `(48, 40, 40)` and training exploration slots
  remain exactly `(6, 5, 5)`.
- Warmup and live-index refresh cadence remain `2,000` updates.
- The v1 fixed-WTA witness-pair router objective remains unchanged for the
  first bounded signal run.  This isolates bounded retrieval from a new router
  loss.  Cutoff-aware or balance losses require a later, separately frozen
  experiment.
- Dynamic-k, dense-bank scoring, fixed-WTA serving fallback after activation,
  NNUE/difference reuse, and game-seed evaluation are forbidden.

## Bounded learned retrieval

Each logical `(table, bucket)` is physically divided into 16 deterministic
neuron-ID lanes.  Lane assignment is a fixed SplitMix-derived function of the
frozen router seed, layer, table, and neuron ID; it has no trainable parameter.
Every lane is an intrusive list with an O(1) occupancy counter.

For a query:

1. Compute both learned seven-bit table codes.
2. Read the 32 queried lane occupancies without traversing a chain.
3. If the sum of the two logical-bucket occupancies does not exceed the frozen
   layer visit cap, traverse every entry.  The deduplicated candidate set and
   final ordering must equal v1 exactly.
4. On overload, visit one entry at a time in deterministic table/lane
   round-robin order.  Stop at the shared layer visit cap.  No table or lane may
   perform an unaccounted traversal and no chain is scanned to discover its
   length.
5. Deduplicate by neuron ID.  Before exact scoring, cap the fixed shortlist by
   exact-table hit count descending and neuron ID ascending.  Training reserves
   its exploration slots before applying this cap.
6. Rerank the shortlist by raw 64-dimensional key dot product descending,
   exact-table hit count descending, and neuron ID ascending.
7. Deterministically fill any exploitation underflow and add the exact number
   of exploration IDs using a no-repeat affine permutation with a proved
   membership-check bound.  A `1:N` fallback loop is forbidden.
8. Return exactly the frozen fixed-k width with unique, in-range IDs.

Frozen aggregate caps by layer are:

| Layer | bucket entries visited | exact rows scored | active width |
|---|---:|---:|---:|
| 1 | 1,536 | 384 | 48 |
| 2 | 1,280 | 640 | 40 |
| 3 | 1,280 | 640 | 40 |

The implementation must publish per-layer total/table/lane visits, logical
bucket occupancy, overload/truncation state, unique retrieval count, shortlist
drops, exact rows scored, deterministic-fill attempts, and exploration-probe
attempts.  Natural bucket overload activates the bounded scheduler; it is not
itself a numerical failure.

## Update and checkpoint invariants

- Forward tapes, VJP, sparse AdamW state, and incremental bank updates contain
  selected IDs only.  Inactive bank parameters, moments, event counters, and
  lazy-decay timestamps remain unchanged.
- The learned router may update only its existing `2,688` projection
  parameters.
- V2 may use a new index/state representation.  V1 checkpoints must never be
  silently reinterpreted as v2.  Exact v1 continuation stays v1; exact v2
  continuation stays v2; cross-version restore fails closed.
- Incremental lane-index rollback must be bounded in the number of dirty rows.
  Ordinary updates may not snapshot or validate a whole affected bucket.
  Full-bank validation is allowed only when building a scheduled live index.
- Pending projections are trained every update.  All three live projections
  and lane indexes are built, validated, and atomically published at a refresh
  boundary.  Failed publication leaves the prior coherent state untouched.

## Required evidence before a signal run

1. V1 cap-overflow behavior remains covered and unchanged.
2. An adversarial collapsed-bucket test returns fixed width under every cap,
   is deterministic, and activates load-balanced truncation.
3. V1 and v2 selected IDs are identical whenever complete queried occupancy is
   within the visit cap.
4. Exact shortlist, fill, probe, lane, table, and global bounds are tested at
   their boundaries.
5. V2 checkpoint restore produces the exact next update; v1/v2 cross-restore
   is rejected.
6. Active-only bank and sparse-optimizer byte invariants pass unchanged.
7. A real `teacher_v3` two-update CLI smoke completes, publishes a loadable
   checkpoint, and reports finite routing telemetry.  This smoke uses no game
   seeds.

## First scientific signal

Only after a clean `GO / NO-GO / PIVOT` review may one fresh fixed-k v2 signal
run be launched.  It uses the frozen teacher split, starts from update zero,
keeps the 2,000-update warmup, and evaluates learned routing at updates 2,000
and 2,500.  It is a routing/teacher-ranking experiment only.  Promotion
requires zero cap or numerical failures, bounded telemetry, fixed active widths,
finite metrics, active-only updates, and a non-regressing learned-routing signal
relative to the matched fixed-WTA parent.  Validation seeds `8001:8008` and
sealed seeds `91001:91032` remain unopened.
