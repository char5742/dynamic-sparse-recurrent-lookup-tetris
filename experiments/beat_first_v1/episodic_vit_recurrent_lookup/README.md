# Episodic ViT recurrent LookupFFN

This directory implements the current Dynamic Sparse Recurrent Lookup Network
for real-teacher Tetris candidate ranking.

## Architecture

Each candidate is evaluated independently from the same input fields used by
the PreAct baseline: board, candidate, difference, next/hold, and `aux37`.
Teacher Q values and ranks are targets only.

The model contains:

1. positioned cell, next/hold, and auxiliary episodic tokens;
2. a five-stage dilated depthwise/pointwise visual residual over raw
   board/candidate/difference channels, with a `63 x 63` receptive field that
   covers the complete `24 x 10` board;
3. recurrent cell memory updated through learned physical local-8 spatial
   attention with shared Q/K/V/O projections and relative 3x3 position bias;
4. multiple recurrent registers;
5. exact cross-attention from each register to all 283 episodic tokens, with
   one shared K/V projection per recurrent step;
6. learned register self-attention and SwiGLU transformation;
7. active-only LookupFFN long-term memory;
8. residual recurrent updates and a hard-halting interface.

No dense mask-after-score implementation or CountSketch is used.  The
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

Dynamic training now has an optional candidate-local one-step probe path.  It
probes only a bounded number of sampled stops, recomputes ListNet plus margin
after replacing that candidate's Q, and supervises the final halt decision
from `L_stop - L_continue`.  This preserves physical sparsity and replaces the
former state-wide REINFORCE credit signal when enabled.  See
[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md).

See [`RESULTS_2026-07-20.md`](RESULTS_2026-07-20.md) for the exact numerical
witnesses and
[`PERFORMANCE_COMPARISON_2026-07-20.md`](PERFORMANCE_COMPARISON_2026-07-20.md)
for the final held-teacher comparison against PreAct.

The subsequent input-routing ablation removed the former `283 -> 64 -> 16`
register memory bottleneck.  At the same 12,000-update budget, full-token
cross-attention improved top-1 from `0.35938` to `0.56250`, with CPU inference
decreasing from `53.54` to `45.17` states/s.  See
[`TOKEN_ROUTING_ABLATION_2026-07-21.md`](TOKEN_ROUTING_ABLATION_2026-07-21.md).

The current visual extension uses dilations `1,2,4,8,16`, adds only 565
parameters and 135,360 scalar MAC/candidate, and reaches a true `63 x 63`
receptive field.  At 25,000 updates / 100,000 teacher states it obtained
top-1 `0.68750`, NDCG `0.98586`, margin `0.13215`, and `43.72` held CPU
states/s.  See
[`GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md`](GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md).
