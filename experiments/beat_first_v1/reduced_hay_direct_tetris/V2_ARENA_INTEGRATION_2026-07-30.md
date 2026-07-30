# Reduced Hay v2 arena integration — 2026-07-30

## Outcome

`causal_recurrent_v2` now has a production CPU training backend:

```text
ReducedHayV2ArenaTraining.jl
  fixed width-80 tape
  persistent MPMC worker team
  candidate-level barrierless scheduling
  block-local DECOLLE signals
  seven-component Reduced-Hay e-prop traces
  stochastic Plackett-Luce routing
  utility structure learning
  deterministic parallel gradient reduction
  sharded in-place AdamW
  zero hot allocation / zero GC
```

The direct Zygote/BPTT implementation remains the reference control. The
Point-SNN, frozen 11-state and GRU controls remain separate.

## Causal forward contract

The scalar arena backend implements the same sequence as
`_causal_reduced_hay_dynamics`:

1. deliver delayed events from the preceding active spike states;
2. apply located E/I sensory contacts as an initial conductance pulse;
3. update AMPA, NMDA and GABA conductances;
4. compute voltage-dependent NMDA current and axial current;
5. update branch voltage and NMDA-driven plateau;
6. integrate basal state with preceding workspace feedback in apical/soma;
7. export soma/apical/branch analog state;
8. derive the query from the updated state;
9. select workspace blocks;
10. mask the soma events and write the next workspace.

For a deterministic tiny-v2 trajectory, the arena and functional reference
raw outputs agree within `2e-6`; the observed raw maximum error in the focused
check was approximately `2.6e-8`.

## Local credit

For block `b`:

```text
M_b(t) =
    B_b * delta_raw
  + B_b * local_error_b(t)
```

`B_b` is a distinct seed-fixed, non-trainable `22 -> node_dim` matrix.
Local Q, death, quantile and geometry predictions use only the current block's
exported analog states and spike events. Recurrent signal construction does
not read either global head matrix.

Each recurrent weight, gate and delay owns a forward eligibility trace through:

```text
AMPA, NMDA, GABA, branch voltage, plateau, soma, adaptation
```

Cell-local parameters receive the same block signal through the explicit
Reduced-Hay transition derivatives. State-query/key routing uses ordered
Plackett-Luce eligibility and a candidate-centered supervised reward
surrogate. It is not an environment return.

Utility is accumulated from the same third-factor/eligibility product, then
RMS-normalized per source. Consolidation preserves fixed fanout, connection
cost, one swap per source, branch relocation and optimizer-moment reset.

## Focused verification

Command:

```powershell
julia --project=. --threads=4,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/test_reduced_hay_v2_arena_training.jl
```

Result: `27/27` checks passed.

The focused suite covers:

- functional-reference versus fixed-arena forward equivalence;
- routing finite-difference sign;
- exact fixed fanout;
- all 34 parameter groups changing with a live third factor;
- all 30 recurrent/cell/routing groups frozen when the third factor is zero,
  including AdamW decay and structure changes;
- two-worker versus four-worker equivalence (`1.2e-7` observed maximum
  parameter difference);
- hot allocation `0` and GC `0`;
- checkpoint save/restore including mask, branch placement and sampler state.

The existing direct mainline suite also remains green: `72/72` checks.

## Full-width scratch evidence

Run:

```text
D:\tetris-paper-plus\runs\reduced_hay_v2_arena\
  scaled_1k_scratch_20260730_02
```

Configuration:

- `reduced_hay_scaled_v2`;
- width `80`;
- state batch `8`;
- `20` Julia workers, BLAS `1`;
- stochastic training routing;
- `1,000` scratch updates / `8,000` teacher states.

Observed:

| Metric | Initial | Final |
|---|---:|---:|
| composite loss | 6.053182 | 3.776091 |
| local Q loss | 5.453151 | 3.434759 |
| local death loss | 0.693703 | 0.547826 |
| local quantile loss | 2.725258 | 2.575974 |
| local geometry loss | 0.065771 | 0.061002 |
| firing rate | 0.006644 | 0.040790 |
| plateau mean | 0.005538 | 0.130119 |
| routing entropy | 0.314581 | 0.811297 |

Every one of the 34 parameter groups had a nonzero maximum delta. Measured hot
allocation and GC remained zero. The final checkpoint is:

```text
checkpoint_000001000.jld2
SHA-256 0c5151d2679ddc3c9247d6e4bd40a83cb1cf8923786ccc4b047ac776d531f83b
```

## Local replay optimization — 2026-07-31

The production dimensions remain fixed at width 80, state batch 8 and 20
workers. The learning rule and all 34 trainable groups are unchanged. Four
mechanical costs were removed:

- block-local predictor state is formed once per coordinate instead of once
  per output;
- the fixed projection is traversed in contiguous output-major order;
- cell, edge and supervised-head updates run during the first local-signal
  phase, so the post-centering replay contains only routing;
- only eligibility entries touched by the preceding candidate are cleared.

On the identical 32-update scratch schedule, the original implementation and
the optimized implementation measured:

| Metric | Original | Optimized | Ratio |
|---|---:|---:|---:|
| mean states/s, all 32 updates | 50.092 | 89.969 | 1.796x |
| mean states/s, updates 9–32 | 50.136 | 93.295 | 1.861x |
| mean wall/update | 0.161734 s | 0.090287 s | 0.558x |
| mean process CPU/update | 2.004 s | 1.197 s | 0.597x |
| hot allocation / GC | 0 / 0 | 0 / 0 | unchanged |

The optimized focused suite remains `27/27`. Two additional experiments,
packed active-order trace storage and common-coefficient trace expansion, were
rejected because each reduced measured throughput.

Reproduce the phase benchmark without checkpoints using:

```powershell
julia --project=. --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/benchmark_reduced_hay_v2_arena.jl `
  --warmup 8 --repetitions 32 --state-batch 8 --width 80 --workers 20
```

## Held-panel result and boundary

`evaluate_reduced_hay_v2_arena.jl` evaluated the checkpoint on a deterministic
128-state validation panel:

| Metric | Value |
|---|---:|
| composite loss | 3.831504 |
| top-1 | 0.078125 |
| NDCG | 0.824687 |
| pairwise | 0.508192 |
| deterministic forward | 241.997 states/s |
| allocation / GC | 0 bytes / 0 s |

This is evidence that the complete production path trains and can be compared
on held rows. It is not evidence that update 1,000 is competitive with a
100k Point-SNN, GRU, DSRLN or PreAct checkpoint. The next quality decision must
use the identical panel and either a matched update count or matched CPU wall
time.
