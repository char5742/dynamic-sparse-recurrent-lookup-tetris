# Paired sparse CPU benchmark

`paired_cpu_benchmark.jl` is a bounded, synthetic, CPU-only comparison of the
existing exact 19,924,022-parameter one-layer k64 model, the existing exact-
parameter three-layer k64 split `(24,20,20)`, and the already benchmark-only
one-layer dense twin. The three-layer k128 and k256 variants are reported as
accounting-only rows; they are not instantiated or timed.

The harness reuses `sparse_dynamic/benchmark_sparse_q20.jl` for its 64-item
synthetic generator, loss, and dense twin. Every timed
path receives the same corpus item and target at each sample number. The
six-permutation cyclic order balances first/middle/last thermal position
and each directed cross-sample predecessor/successor boundary.

Do not run this while another heavy Julia process owns the benchmark slot. Once
the machine is idle, the default bounded invocation is:

```powershell
$env:JULIA_NUM_THREADS = '1'
julia --project=. experiments/beat_first_v1/sparse_dynamic_comparison/paired_cpu_benchmark.jl
```

Only `PAIR_CPU_WARMUP`, `PAIR_CPU_FORWARD_SAMPLES`,
`PAIR_CPU_TRAINING_SAMPLES`, and `PAIR_CPU_COMPONENT_SAMPLES` are accepted
controls, each with a hard maximum and a required multiple of six in the
contract. The process must have exactly one Julia thread; the harness fixes BLAS
to one thread, so the dense oracle and both sparse paths receive the same CPU
resource envelope. CPU affinity is explicitly `UNPINNED`; the harness does not
claim P-core or any other core-class pinning. Positional arguments, dataset
paths, checkpoints, game states, and evaluation seeds are rejected.

The primary `timing` rows report p50/p95 forward and complete training latency.
`profile` rows report allocations and observed GC. `phase` rows include routing,
materialization, fused gather/compute, parameter VJP, optimizer, and rehash.
`actual` rows distinguish executed model/rerank MACs from routing-inclusive
linear operations (which also count sketch accumulates), and report gathered
bytes. The separate
`gather_copy_diagnostic` is a preallocated, non-additive copy probe; it is never
included in the production forward or complete-training total. Gathered-byte
rows cover routing plus forward parameter reads; they do not pretend to be a
whole-optimizer traffic counter.

Raw timing, allocation, GC, and ratio rows are always retained. Paired latency
ratios are authoritative only when every forward and full-training timed sample
for all three paths reports zero GC time. Any observed timed GC marks every
paired ratio non-authoritative and changes the completion gate to the fail-closed
status below.

A GC-clean completion emits:

```text
gate    paired_latency_ratios           AUTHORITATIVE_NO_TIMED_GC
gate    benchmark_completed             PASS
gate    promotion_or_strength_claim     NOT_AUTHORIZED_SYNTHETIC
```

If timed GC is observed, the raw rows remain available and the first two gates
instead report `NON_AUTHORITATIVE_TIMED_GC_OBSERVED` and
`FAIL_CLOSED_TIMED_GC_OBSERVED`, respectively.

No timing threshold is a promotion gate, and this synthetic benchmark is not
game-strength, validation, or sealed-test evidence.
