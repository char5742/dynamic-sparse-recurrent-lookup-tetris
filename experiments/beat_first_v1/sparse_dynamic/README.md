# Tetris-SLIDE-NeuronBank-Q20 (Track A)

This directory is the isolated prototype for E036. It is deliberately not an
NNUE accumulator, a parent/child search cache, fixed CSR pruning, or a
coarse-grained mixture of dense experts. Each candidate position is sufficient
on its own. Its WTA/LSH result chooses an irregular set of individual neuron
rows, and only those rows may be read by the neural forward path, differentiated,
or passed to the bank optimizer.

## Frozen v1 geometry

```text
independent candidate input
  route features: 64
  value features: 496
           |
           v
WTA/LSH over 32,768 neuron route keys
           |
           v
64 selected neuron rows (IDs vary with every input)
           |
           v
48-value sparse aggregation -> 22-output linear head
```

Every logical neuron is one contiguous Julia column:

```text
route key  64
value     496
outgoing   48
-------------
row       608 Float32 = 2,432 bytes = 38 x 64-byte cache lines
```

The exact trainable count is:

```text
608 * 32,768 + 48 * 22 + 22 = 19,924,022
```

At the default `k=64`, the final selected graph contains 39,990 trainable
values (0.20071% of the model) and 39,968 forward neural MACs. Its
parameter-only VJP performs 44,096 additional linear MAC-equivalents, for
84,064 forward+backward linear MACs per independent candidate before routing
and feature preparation. Routing-key probes,
bucket traversal, sorting, index maintenance, and sparse optimizer work are
reported separately and are always included in end-to-end wall time. The
0.20071% figure describes the final neural graph, not router-key reads.
Production routing fails closed above 1,024 unique scored key rows (3.125% of
the bank, 256 KiB of FP32 route-key data) or 4,096 intrusive bucket entries.
The exact scored-row count contributes `64 * scored_rows` routing MACs, which
is reported alongside neural MACs and their route-plus-forward total. The
routing-inclusive unique parameter-read count is also reported as
`39,990 + max(scored_rows - 64, 0) * 64`: selected route keys already belong
to the final graph and are counted only once. Feature preparation separately
reports 480 signed CountSketch scale-and-accumulate operations, plus the
feature+router+neural total.

## Feature contract

`features.jl` maps one canonical packed candidate to two vectors without any
learned dense stem or parent-position reuse:

- `route64`: HOLD/NEXT 42 plus column heights 10, holes 10, unreachable
  cavities 1, and maximum height 1, with a fixed balanced signed CountSketch
  of all candidate/difference board cells added across the same 64 slots.
- `value496`: post-placement board 240, signed placement difference 240,
  remaining auxiliary values 15, and a constant 1.

The pre-action board is recoverable as `candidate - difference`; no
information claim relies on a cached parent representation.

## Production hard gates

The prototype is rejected as fake sparsity if any production step:

- computes the whole bank and masks afterward;
- scans all neurons to obtain exact top-k;
- allocates, clears, or materializes a `608 x 32,768` gradient;
- runs an optimizer traversal over all bank parameters;
- trains densely and becomes sparse only at inference;
- omits WTA/LSH maintenance from timing; or
- depends on candidate-to-candidate or parent-to-child representation reuse.

Correctness requires a frozen-route reference difference at most `1e-5`, no
writes to inactive parameter/optimizer/timestamp storage, deterministic
row-ID reduction, and checkpoint/resume reproducibility. The first systems
gate requires at least 3x batch-one forward and 2x complete training-step
speed over the same-size dense twin, with index maintenance below 25% of
training wall time. Only after that gate may the model consume a full teacher
epoch. Teacher and game promotion gates remain those preregistered in E036.

## Optimizer semantics

The bank starts with selected-row AdaGrad and bank weight decay fixed to zero.
It is a true sparse optimizer: inactive row state is unchanged. Nonzero lazy
decay is not enabled in v1 because WTA collision tie ranking reads route-key
dot products; a future implementation would have to materialize every probed
row before ranking, not merely every finally selected row. This optimizer is
not presented as mathematically identical to dense AdamW. A later event-time
sparse Adam is eligible only if the AdaGrad correctness and throughput gates
already pass; generic AD/backend benchmarking is outside this experiment.

Production teacher training uses a manual in-place parameter VJP. It writes the
608 values for each selected row directly into the compact sparse accumulator,
adds the tiny-head gradients into reusable buffers, and does not compute the
discarded `dq`/`dx` feature cotangents. The allocating full selected VJP remains
only as an independent numerical reference in tests and diagnostics.

## Dense-twin benchmark

E036 systems timing must use a completed `teacher_v3` training part:

```powershell
julia --project=experiments/beat_first_v1 --startup-file=no `
  experiments/beat_first_v1/sparse_dynamic/benchmark_sparse_q20.jl `
  --dataset=D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3
```

The benchmark verifies the selected part's manifest byte count and SHA-256,
then samples 64 real candidates from the train split. It alternates sparse and
dense order, reports hot/cold-ish p50/p95, allocation/GC, executed linear
operations, logical bytes, routing caps, and real dirty-row rehashing. The
dense twin uses explicit native BLAS for full-bank forward/VJP and a threaded
all-column RowWise AdaGrad update. The fixed-`k` index-scaling gate doubles the
physical bank while preserving its 608-float column stride, alternates base and
doubled queries, and checks scaling in normalized terms. With unchanged hash
geometry, doubling the bank is expected to double average bucket occupancy;
therefore the gate requires scored-row *fraction* and latency per scored row
to stay within 1.50x, while allowing at most 2.50x total route latency. The
absolute 20M-model inference/training gates remain 3x/2x versus the dense twin.
The benchmark also times rehashing inside 17 exact complete sparse-training
steps, reports the per-step maintenance-share p50/p95, and requires the p50 to
remain strictly below 25%. Synthetic input cannot authorize this gate.

`--synthetic-smoke` is only a parser/kernel smoke test. Its output is stamped
`SMOKE_ONLY_NOT_AUTHORIZED` and cannot satisfy the E036 speed gate.
