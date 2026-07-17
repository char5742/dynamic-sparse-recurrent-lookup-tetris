# AD clean retry result

## Decision

This is an 82,240-parameter, fixed-shape, static-activity dense MLP microbenchmark. It
is not a benchmark of the actual Tetris learner. Within this deliberately narrow
workload and machine, the claim that Zygote is the stronger steady-state backend is
rejected: standalone Enzyme static is faster after warmup at every tested batch
(1.08-1.71x), and Reactant + EnzymeMLIR is fastest (2.79-3.39x over Zygote).

The adoption supported by these results is limited to a numerically gated,
static-activity-compatible dense workload with fixed shapes under the pinned versions
below. For that scope, adopt standalone Enzyme for a long-lived native CPU process.
Keep Zygote for short jobs and shape-changing/debug workflows because its first update
is 3-4x faster and its peak working set is lower. Treat Reactant as an opt-in
whole-update fixed-shape stack: it ran successfully on Windows and has the highest
steady throughput, but paid a 37-40 second first update and used about 1.25 GiB peak
RAM. This microbenchmark alone does not justify changing the actual Tetris learner.

No compile-inclusive crossover was observed within the measured 1000 updates: Zygote
had the shortest full loop at every batch. A linear extrapolation using the observed
first-update cost and warm throughput estimates crossover only around 67k-149k updates
for standalone Enzyme and 24.5k-45.5k updates for Reactant, depending on batch. These
are extrapolated ranges, not measured crossover points.

## Measured workload and environment

- Learner step: one Lux forward, one scalar MSE, reverse-mode backward, one AdamW
  update. No separate loss forward.
- Model: `Chain(Dense(256=>256,tanh), Dense(256=>64))`, 82,240 Float32 parameters.
- Shapes: input `(256, batch)`, target `(64, batch)`; one fresh process per backend and
  batch; 1000 updates per process.
- Timing: first update is compile-inclusive. Steady statistics exclude updates 1-10.
  Reactant uses `sync=true` plus explicit loss/parameter synchronization inside the
  timed region. Native operations are synchronous.
- CPU/OS: Intel Core Ultra 7 265K, 20 cores/20 logical processors, 64 GiB RAM,
  Windows 11 Pro build 26100. Julia threads and BLAS threads were both 20.
- Software: Julia 1.12.6, Lux 1.31.4, Zygote 0.7.11, Enzyme 0.13.187,
  Reactant 0.2.274, Optimisers 0.4.7.

## Performance

Allocation is the warm median Julia-heap allocation per update. Peak is the Windows
process peak working set, so it also captures memory outside Julia's allocator.

| Batch | Backend | First update (s) | Warm step/s | Warm median alloc (KiB) | Peak (GiB) | 1000-update loop (s) |
|---:|:---|---:|---:|---:|---:|---:|
| 16 | Zygote | 9.71 | 1064 | 448.02 | 0.79 | 11.03 |
| 16 | standalone Enzyme | 35.82 | 1819 | 169.29 | 1.02 | 36.73 |
| 16 | Reactant + EnzymeMLIR | 39.20 | 3437 | 13.20 | 1.25 | 40.01 |
| 32 | Zygote | 9.28 | 910 | 528.02 | 0.82 | 10.75 |
| 32 | standalone Enzyme | 33.63 | 1094 | 265.29 | 1.03 | 34.89 |
| 32 | Reactant + EnzymeMLIR | 38.11 | 3084 | 13.20 | 1.27 | 38.87 |
| 64 | Zygote | 9.32 | 714 | 688.02 | 0.79 | 11.07 |
| 64 | standalone Enzyme | 32.78 | 823 | 457.29 | 1.03 | 34.34 |
| 64 | Reactant + EnzymeMLIR | 39.54 | 2058 | 13.20 | 1.24 | 40.42 |
| 128 | Zygote | 9.22 | 564 | 1008.04 | 0.83 | 11.33 |
| 128 | standalone Enzyme | 29.31 | 611 | 841.29 | 1.03 | 31.32 |
| 128 | Reactant + EnzymeMLIR | 37.06 | 1575 | 13.20 | 1.25 | 38.14 |

All 12 processes completed 1000 updates with finite losses and parameters. The maximum
absolute loss-trajectory difference from Zygote over all 1000 updates was at most
`4.18e-7` for standalone Enzyme and `2.39e-7` for Reactant.

## Numerical gate

Numerical checks were repeated at batches 16 and 64 from identical initial arrays.

| Batch | Candidate | Loss abs error | Gradient cosine | Gradient max abs error | 1-update parameter max abs error |
|---:|:---|---:|---:|---:|---:|
| 16 | standalone Enzyme | 1.79e-7 | 0.9999999999999865 | 0 | 0 |
| 16 | Reactant + EnzymeMLIR | 5.96e-8 | 0.9999999999998762 | 1.06e-8 | 2.37e-6 |
| 64 | standalone Enzyme | 4.17e-7 | 0.9999999999999892 | 0 | 0 |
| 64 | Reactant + EnzymeMLIR | 0 | 0.9999999999998895 | 6.05e-9 | 1.90e-6 |

Standalone Enzyme's differentiated primal differs from Zygote's forward scalar by a
few Float32 ulps, while its gradient arrays and one-update parameter arrays are bitwise
equal in both checked shapes. Reactant differences are small and consistent with
different floating-point lowering/order.

## Interpretation and limits

Zygote is stronger in the measured compile-inclusive 1000-update comparison because
compile cost dominates: the first update was roughly 9 seconds for Zygote, 29-36
seconds for standalone Enzyme, and 37-40 seconds for Reactant. Enzyme and Reactant lead
only in the separately measured warm steady state; the extrapolated crossover ranges
above require much longer process lifetimes than were run here.

The result applies to this Lux MLP, MSE, AdamW, CPU, and package manifest. It is not a
universal ranking for convolutional models, stateful layers, dynamic shapes, or other
losses. Reactant's 13 KiB allocation figure is Julia-heap allocation only; its higher
process peak is the more representative memory number. Reactant fuses forward, loss,
differentiation, and optimizer update into its compiled whole-update step, while the
native paths call the same `Optimisers.update!` after their AD call. Its speedup is
therefore evidence for that fused Reactant whole-update stack, not evidence that its
isolated backward pass is faster.

Machine-readable results are in `artifacts/summary.json`; raw per-step timings,
allocations, losses, stdout/stderr logs, numerical checks, and environment capture are
in `artifacts/`.
