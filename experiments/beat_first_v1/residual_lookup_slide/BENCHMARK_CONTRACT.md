# Residual Lookup-SLIDE R0 benchmark contract

Status: **PREPARED, UNEXECUTED**. No Julia process was started while preparing
this harness. It makes no speed, teacher-ranking, game-score, or old-model-beat
claim before the deferred measurement completes.

## Exact implementation under test

The harness imports the production implementation in `ResidualLookupSlide.jl`;
it does not carry a second benchmark-only model. One independent candidate is
mapped to the fixed 256-value carrier

```text
[CountSketch192(raw496), context22, NEXT/HOLD42]
```

and then passes through three blocks. Every block applies fixed RMSNorm, a
fixed signed normalized FHT256, 76 independent three-digit seven-way WTA
addresses, and exactly one 256-value lookup row per table. The rows are summed,
scaled by `1/sqrt(76)`, and added through the learned scalar gate
`0.1*tanh(alpha_logit)`. A learned `256 -> 22` head with 22 biases produces the
output.

Hard addresses have zero derivative. Backward creates gradients only for the
228 selected rows, the dense head and bias, and the three residual gates. The
bank optimizer uses per-column event time and a lazy decoupled-weight-decay
clock; it does not scan or update unselected columns.

This model evaluates a candidate independently. It is neither NNUE parent/child
reuse nor a dense expert hidden behind a routed ID.

## Independent accounting

| Quantity | Derivation | Exact value |
|---|---:|---:|
| Rows per table | `7^3` | 343 |
| Total bank rows | `3 * 76 * 343` | 78,204 |
| Bank parameters | `78,204 * 256` | 20,020,224 |
| Head weights | `256 * 22` | 5,632 |
| Head biases | `22` | 22 |
| Residual gate parameters | `3` | 3 |
| Total parameters | sum above | **20,025,881** |
| Active rows/candidate | `3 * 76` | **228** |
| Active bank parameters | `228 * 256` | 58,368 |
| Active unique parameters | bank + head + bias + gates | **64,025** |
| Trainable edge uses | bank + head + bias + `3*256` gate uses | **64,790** |
| Lookup bytes gathered | `58,368 * 4` | 233,472 B |
| All active parameter bytes | `64,025 * 4` | **256,100 B** |

An active gate is one unique parameter but is used on 256 residual edges, so
active parameter count and trainable edge-use count must not be conflated.

The narrowly defined forward MAC-equivalent counter is **9,968**:

- 496 fixed raw CountSketch signed accumulations;
- 768 RMS square accumulations;
- 768 signed RMS scale operations;
- 768 signed FHT output scales;
- 768 table-mean scales;
- 768 residual-gate muladds;
- 5,632 head MACs.

It deliberately excludes and separately reports 3,072 FHT butterflies / 6,144
scalar add-sub operations, 4,104 WTA comparisons, and 58,368 bank
accumulations. Backward reports 12,032 MAC-equivalents (two 5,632-MAC head VJPs
plus 768 gate-dot MACs), 58,368 bank-gradient scale multiplications, and 496
raw CountSketch-transpose multiplications. These are execution counters, not a
hardware-FLOP claim.

## Exact-forward preflight

The component timer uses a preallocated implementation of the same frozen
operations so it can split forward into:

1. raw CountSketch and carrier composition (`sketch`);
2. RMS normalization and first fixed sign (`norm`);
3. FHT, second fixed sign, and WTA addressing (`hash`);
4. selected-row copies (`gather`);
5. ordered row reduction, table scaling, and gated residual (`residual`);
6. biased output head (`head`).

Before any measurement, its output, addresses, and flat columns must match the
authoritative production `forward` bit-for-bit. Failure aborts the run. The
three-way comparison times the authoritative production forward and its actual
allocations. Component `time_ns()` probes run in a separate diagnostic pass,
so probe overhead cannot contaminate the paired outer samples.

## Same-input three-way comparison

The forward comparison is limited to:

1. freshly initialized, untrained Residual Lookup-SLIDE R0;
2. the exact update-2500 fixed-WTA matched parent with widths `(48,40,40)`;
3. the bounded MONGOOSE-v2 **base** checkpoint at update 2500, serving at the
   same fixed widths.

The fixed parent is pinned to SHA-256
`de9aac395fc9406f2a3c77de4fa2408ada62716155d1d1915a1aaedb62670b85`.
The base-v2 checkpoint is pinned to SHA-256
`c0b8b350be2357c39c27a4cf73fb8efba9b4403ba4de8cee57a2248166eee2af`
and pairing-contract SHA-256
`6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450`.
Its `latest.json` and controller status are also pinned by the harness. The
controller status is explicitly `failed`: the learned-routing teacher signal
regressed below its frozen floor. Therefore this checkpoint is included only
as a bounded serving-latency implementation, never as evidence that MONGOOSE
improved teacher ranking or game strength.

Likewise, R0 has not yet received teacher-ranking or game training. This
benchmark compares architecture/runtime cost only; none of its forward outputs
constitute a strength comparison.

This is not the later cutoff-boundary arm. That arm failed during its initial
evaluation before update 1 and produced no comparable trained checkpoint. The
base-v2 training state must retain `fixed_wta_easy_extrema_v1`; any cutoff-arm
identity is rejected. The loader also rejects mismatched update, sampler state,
split metadata, teacher manifest, model/training configuration, routing mode,
overlay type, activation/index state, parameter count, or active widths. The
same production teacher checkpoint validator is invoked, including optimizer
clocks, policy fields, refresh state, and live/index coherence checks.

Every corpus element begins as one identical candidate object. The fixed
feature adapter is executed once outside all timed regions, producing `q64`
and `x496`. The raw 22 aux-context values and raw 42 NEXT/HOLD values are also
extracted exactly as the production candidate overload does; they are not
sliced back out of the board-CountSketch-contaminated `q64`. R0 receives
`[x496, raw_context22, raw_next_hold42]`; both controls receive a prebuilt
`ThreeLayerInput((q64,q64,q64),x496,raw_context22,raw_next_hold42)`. Thus
architecture-specific views have production semantics, come from the same
candidate, and charge no path for feature packing. Source tensors and every
adapted tensor are included in the printed corpus SHA-256. Measurement order
rotates through the three cyclic orders.

Before warmup, every sparse-bank row in both checkpoint controls has its
pending lazy decay materialized in memory. This is outside all timed regions
and prevents candidate-dependent first-touch decay writes from contaminating
the paired serving samples. Source checkpoint files remain unchanged.

Only forward is paired across architectures. R0 additionally reports its
authoritative selected-only backward, whole sparse/dense optimizer transaction,
and full forward/backward/update throughput. Claiming a MONGOOSE base-bank
update as complete learned-router training would be misleading, so no such
training comparison is emitted.

## Timing and mutation evidence

Every end-to-end stage reports nearest-rank p50/p95 latency, allocated bytes,
and GC time. Aggregate training throughput is completed R0 updates divided by
their measured total duration. The process must have one Julia thread and one
BLAS thread. GC-contaminated raw samples are retained but explicitly marked;
the harness never converts a ratio into a speed claim automatically.

Before timing, one bounded mutation witness snapshots an unselected bank row,
its first and second moments, event count, event step, and lazy-decay clock.
After a whole-model update all inactive snapshots must remain bitwise unchanged,
while a selected row and event count must change. Head, bias, and residual gates
remain correctly classified as always-active parameters.

## Deferred commands

Run only under the single-heavy-Julia invariant. All checkpoint and receipt
paths/hashes are frozen in the harness and cannot be overridden from the CLI:

```powershell
$env:JULIA_NUM_THREADS = '1'
julia --project=experiments/beat_first_v1 `
  experiments/beat_first_v1/residual_lookup_slide/test_accounting.jl

julia --project=experiments/beat_first_v1 `
  experiments/beat_first_v1/residual_lookup_slide/benchmark.jl `
  --samples 30 --warmup-cycles 6
```

This harness uses only a synthetic, domain-separated candidate corpus. Game
validation and sealed test sets are outside its scope.
