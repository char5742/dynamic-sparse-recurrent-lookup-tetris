# Episodic ViT recurrent LookupFFN

This directory implements the current Dynamic Sparse Recurrent Lookup Network
for real-teacher Tetris candidate ranking.

## Architecture

Each candidate is evaluated independently from the same input fields used by
the PreAct baseline: board, candidate, difference, next/hold, and `aux37`.
Teacher Q values and ranks are targets only.

The model contains:

1. positioned cell, next/hold, and auxiliary episodic tokens;
2. recurrent cell memory updated through learned physical local-8 spatial
   attention with shared Q/K/V/O projections and relative 3x3 position bias;
3. multiple recurrent registers;
4. exact cross-attention from each register to all 283 episodic tokens, with
   one shared K/V projection per recurrent step;
5. learned register self-attention and SwiGLU transformation;
6. active-only LookupFFN long-term memory;
7. residual recurrent updates and a hard-halting interface.

No dense mask-after-score implementation, CNN, or CountSketch is used.  The
small `4 x 283` register/token score support is evaluated directly, while
LookupFFN long-memory rows remain physically sparse in forward, backward, and
optimizer updates.

## CPU execution

`barrierless_executor.jl` flattens candidates from multiple states into one
global queue.  Twenty native workers consume chunk-8 work dynamically, and
continued candidates are compacted between recurrent steps.  BLAS uses one
thread.  Windows CPU Sets are supported, but the verified fastest setting on
the test machine is no pinning.

`barrierless_postphase.jl` deterministically reduces worker-local dense and
sparse gradients, applies the canonical global clip, and preserves optimizer
clocks and checkpoint compatibility.  The serial/barrierless smoke verifies
outputs, losses, gradients, routing choices, RNG state, optimizer telemetry,
and post-update parameter state.

## Important files

- `EpisodicViTRecurrentLookup.jl`: model, sparse attention, recurrence, VJPs,
  and optimizer semantics.
- `teacher_training.jl`: real-teacher training and checkpoint lifecycle.
- `barrierless_executor.jl`: dynamic candidate execution.
- `barrierless_postphase.jl`: deterministic reduction and optimizer phase.
- `barrierless_correctness_smoke.jl`: single-thread oracle comparison.
- `bounded_mpmc_queue.jl`: bounded allocation-free Windows MPMC queue.
- `windows_cpu_sets.jl`: runtime P/E-core discovery and optional CPU Sets.
- `run_teacher_signal.jl`: training entry point.

## Verified production geometry

```text
carrier/model dim          128
Lookup tables per block     13
WTA choices                 16
rows selected per table      3
attention dim               32
attention heads              4
registers                    4
episodic cross support      283
local spatial neighbours     8
SwiGLU FFN dim              128
fixed recurrent depth        2
```

Hard halting remains implemented but was held at depth two for the 20,000
update representation-learning run.  It should be re-enabled only after the
spatial and routing representations remain stable under the intended final
evaluation protocol.

See [`RESULTS_2026-07-20.md`](RESULTS_2026-07-20.md) for the exact numerical
witnesses and
[`PERFORMANCE_COMPARISON_2026-07-20.md`](PERFORMANCE_COMPARISON_2026-07-20.md)
for the final held-teacher comparison against PreAct.

The subsequent input-routing ablation removed the former `283 -> 64 -> 16`
register memory bottleneck.  At the same 12,000-update budget, full-token
cross-attention improved top-1 from `0.35938` to `0.56250`, with CPU inference
decreasing from `53.54` to `45.17` states/s.  See
[`TOKEN_ROUTING_ABLATION_2026-07-21.md`](TOKEN_ROUTING_ABLATION_2026-07-21.md).
