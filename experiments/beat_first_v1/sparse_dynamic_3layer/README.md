# Tetris-SLIDE-DeepNeuronBank-Q20-B3

Static prototype of the pure-CPU, three-layer dynamic-unstructured track. It
is isolated from the earlier one-layer `sparse_dynamic` prototype, every NNUE
or parent/child search cache, and coarse expert routing. One independent
candidate determines three different neuron-level hard supports; forward,
manual backward, and optimizer updates touch only those supports. There is no
learned dense backbone in this exact model.

This directory was implemented without running Julia while the repository's
single allowed heavy Julia process was reserved by the convergence pipeline.
The tests are therefore supplied but **not claimed as executed**.

## Exact topology

```text
q64, raw x496
  -> WTA/LSH L1: 11,787 neurons, k=26, row=[route64,value496]
  -> CountSketch192 + context22 + NEXT/HOLD42 + constant = x2(257)
     base q64 + CountSketch64(L1 IDs, activations) = q2
  -> WTA/LSH L2: 20,744 neurons, k=22, row=[route64,value257]
  -> CountSketch/context/NEXT composition = x3/q3
  -> WTA/LSH L3: 20,744 neurons, k=22, row=[route64,value257]
  -> [CountSketch192,context22,NEXT/HOLD42] = latent256
  -> dense 256 -> 22 output head
```

The canonical candidate adapter copies context directly from aux rows
`1:20,31,34` and NEXT/HOLD directly from the `7x6` tensor, so Track A's board
sketch additions to `q64` do not contaminate either passthrough.  The explicit
`ThreeLayerInput` accepts three distinct q64 base queries while the fixed
pure-CPU feature adapter supplies the same q64 three times. CountSketch
locations and signs depend on `(layer, neuron ID)`, not on
selected-slot order.  Deeper routing uses `base_q + residual`, so each layer
remains input-dependent without a large dense backbone.

| Component | Shape | Parameters | Active | Forward MACs |
|---|---:|---:|---:|---:|
| Layer 1 bank | `560 x 11,787` | 6,600,720 | 14,560 | 14,560 |
| Layer 2 bank | `321 x 20,744` | 6,658,824 | 7,062 | 7,062 |
| Layer 3 bank | `321 x 20,744` | 6,658,824 | 7,062 | 7,062 |
| Head+bias | `22 x 256 + 22` | 5,654 | 5,654 | 5,632 |
| **Total** | | **19,924,022** | **34,338 (0.17235%)** | **34,316** |

The long-training parameter VJP omits frozen external base/context/raw-feature
cotangents but propagates both deep sketch/route dependencies: 49,804 MACs,
making forward+parameter-VJP **84,120**, only 0.067% above Track A's 84,064.
The separate full manual VJP returns external `dq64` and `dx496`, so it
accounts for 68,632 linear MAC-equivalents; forward plus full VJP is 102,948.
Fixed sketch/scatter work is
reported separately (118 forward signed bucket accumulates and the corresponding
reverse scatters).  Bounded exact reranking permits at most 384/640/640 key
rows, or 106,496 route-dot MACs in total.  Bucket-entry caps are
1,536/1,280/1,280.

The weight-only figures above remain useful for equal-budget comparison. The
literal `muladd` CountSketch work is also exposed: inclusive forward is
**34,434 MACs**, parameter VJP is **49,922 MACs**, and their inclusive training
sum is **84,356 MACs**. Routing rerank MACs remain a separate input-dependent
counter and are added by `RouteTelemetry`.

At those caps, route-key reads are 425,984 bytes and active trainable weights
are 137,352 bytes.  Counting both streams gives 563,336 bytes; deduplicating
the selected route keys already present in active rows gives 545,416 bytes.
`RouteTelemetry` records the actual scored rows, bucket links, rerank MACs,
gross and deduplicated gather bytes, per-layer routing/materialization time,
selected compute time, and end-to-end forward time for every independent
candidate.  The 34,316 active graph edges are distinct from the 34,338 active
parameters because the 22 output biases are not MAC edges.

## Sparse optimizer contract

`EventTimeSparseAdamW_v1` is not presented as dense-equivalent AdamW:

- stable-reduced active rows get one event per learner update;
- their `m`, `v`, and bias-correction event count advance in active-event time;
- inactive momentum/variance/event clocks do not move at all;
- global-time decoupled decay is represented by one common log clock;
- exact WTA reranking multiplies stale row dots by their pending positive
  scalar decay without materializing them;
- only final selected rows are physically materialized; and
- only actual post-decay Adam changes to route-key bits become dirty WTA rows.

Thus an inactive gap deliberately has no residual-momentum movement.  Sparse
gradient accumulation uses an `O(neuron_count)` generation/slot map allocated
once, but reset, reduction, VJP storage, and optimizer work are `O(active)`.
There is no bank-size gradient, mask, zeroing pass, moment traversal, or
optimizer traversal.

`apply_vjp_step!` is a cross-component transaction. It prepares and validates
all three compact bank updates, the complete tiny-head update, and every WTA
bucket/link that dirty route rows could affect before the first runtime write.
Active bank columns, per-row/global clocks, head state, and the affected
intrusive index cells are snapshotted. Any commit/rehash exception restores
those bytes before rethrowing; malformed or non-finite head gradients fail in
the prepare phase with no runtime mutation.

Training may reserve a caller-chosen fixed number of each layer's active slots
for deterministic uniform probes by passing `training_probes=(p1,p2,p3)` and a
stable `probe_token` to `route_forward!`. Probes replace exploitation slots;
they never increase `(26,22,22)` active width or trigger a bank scan. Inference
uses the default `(0,0,0)`.

## Checkpoint identity

Checkpoints use the distinct format
`TETRIS_SLIDE_3L_EVENTTIME_SPARSE_V1`.  The payload preserves all three lazy
physical banks, `m/v/event_count/last_event_step/last_log_decay`, global decay
clocks, dense head optimizer, intrusive WTA arrays/codes, and arbitrary
caller-provided sampler/RNG state.  Saving never materializes dormant rows.

## Static tests

`test_sparse_dynamic_3layer.jl` defines:

- literal topology/accounting checks;
- frozen-support finite differences for a selected bank parameter, `dq`, and
  `dx`;
- proof that an unselected neuron cannot affect the forward;
- generation-map duplicate reduction and one-event semantics;
- bytewise inactive `theta/m/v/event/decay-clock` checks;
- event-time behavior across 98 empty learner steps; and
- checkpoint/reload followed by an exact same routed update and intrusive
  index comparison.

Once the single-heavy-Julia invariant permits it, the intended command is:

```powershell
julia --project=experiments/beat_first_v1 --startup-file=no `
  experiments/beat_first_v1/sparse_dynamic_3layer/test_sparse_dynamic_3layer.jl
```

The exact 19.924M-bank latency/training entry point is intentionally separate
and was also not executed during static construction. It is a synthetic
single-candidate kernel microbenchmark over random `q64/x496`; it cannot
authorize a real-input teacher, one-layer comparison, or game-strength claim:

```powershell
$env:SPARSE3_WARMUP=5
$env:SPARSE3_INFERENCE_ITERS=100
$env:SPARSE3_TRAINING_ITERS=50
$env:SPARSE3_CANDIDATE_POOL=128
julia --project=experiments/beat_first_v1 --startup-file=no `
  experiments/beat_first_v1/sparse_dynamic_3layer/benchmark_sparse_dynamic_3layer.jl
```

## Known risks before promotion

1. The implementation deliberately reaches into Track A WTA module's private
   intrusive-query helpers to add lazy-decay-correct scoring without forking
   its frozen source.  An upstream Track A refactor can break this boundary.
2. WTA index construction is still a full-bank initialization/resume cost;
   hot training only rehashes dirty selected IDs.
3. CountSketch collisions may reduce teacher ranking accuracy even when the
   sparse systems contract is correct; supervised convergence has not run.
4. Full `m` and `v` banks consume roughly 160 MB in addition to ~80 MB theta.
   Computation is active-only, but checkpoint size and cold startup are not.
5. Any learned/NPU stem is explicitly outside this exact pure-CPU model.
   Adding one requires a separately accounted checkpoint/model identity and
   cannot be credited to this comparison.
