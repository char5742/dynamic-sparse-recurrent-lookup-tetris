# Single-root apical-credit shadow — 2026-08-01

## Question

Can Reduced Hay v2 remove the direct Tetris predictor attached to every block
and still produce useful recurrent credit from one global ListNet head?

This is a shadow/control study.  It does not claim that the final
Sacramento/Payeur-style apical predictive circuit or SAL feedback alignment is
complete.

## Added controls

Three opt-in `credit_mode` values were added while leaving the production
default `:block_teacher` unchanged.

- `:workspace_root_control`: exact analytic head cotangent enters only at the
  real normalized workspace and selected-pool root.
- `:workspace_root_reciprocal_control`: the same root signal plus two sparse
  reciprocal block-credit hops.  This is a symmetric teacher control.
- `:workspace_root_adaptive_control`: recurrent signals use an independent
  `22 -> 2 * node_dim` root-feedback map.  The diagnostic fits that map to the
  exact root cotangent with ridge regression; the recurrent signal generation
  itself does not read `head_weight` or `output_weight`.

All root modes freeze the block-local Q/death/quantile/geometry predictors,
including AdamW decay.  The supervised head can be independently frozen for
credit-only experiments.

## Exact/local alignment

The established 128-warm-up block-teacher control was reproduced:

- recurrent cosine: `0.153014102`
- head cosine: `1.000000020`

At initialization, a one-state root control gave:

| credit path | recurrent cosine | credit-core cosine | edge cosine |
|---|---:|---:|---:|
| workspace root | 0.360101 | not recorded | approximately 0.95 per synapse weight |
| root + reciprocal 2-hop | 0.325451 | 0.454799 | 0.886993 |

After four block-teacher warm-up updates, an adaptive root map generalized to
the next batch with:

- root-feedback cosine: `0.968743`
- recurrent cosine: `0.300297`
- credit-core cosine: `0.366532`
- edge cosine: `0.845451`

After 128 warm-up updates on the established four-state comparison, the
adaptive root remained aligned but exposed other bottlenecks:

- root-feedback cosine: `0.860189`
- recurrent cosine: `0.124637`
- credit-core cosine: `0.174649`
- edge cosine: `0.610454`

The edge signal is substantially better than the fixed block-teacher signal,
but routing and several intrinsic-cell groups still reduce the aggregate.

## Head-frozen 64-update comparison

Both arms used:

- tiny recurrent v2
- state batch 4, width 80
- 16 block-teacher warm-up updates
- 64 recurrent-only continuation updates
- recurrent learning-rate multiplier `0.1`
- frozen supervised head
- fixed 32-row validation panel, evaluated in forward-only chunks of 4

| metric | adaptive single root | block teacher |
|---|---:|---:|
| stream loss, last 8 minus first 8 | **-0.156894** | -0.003448 |
| validation loss change | **+0.067796** | +0.258611 |
| final recurrent cosine | **0.127652** | -0.023359 |
| final credit-core cosine | **0.132052** | -0.064308 |
| final edge cosine | **0.696238** | 0.041337 |
| root-feedback cosine | 0.972299 | n/a |

The adaptive root is much more stable and gradient-aligned, but its validation
loss still increased.  This is evidence that the direct-per-block teacher is a
major credit defect, not evidence that the final learning problem is solved.

An aggressive 16-update multiplier-1.0 comparison showed the same relative
stability: validation changed by `+0.015489` for the adaptive root versus
`+0.233554` for the block teacher.

## Focused invariants

`test_reduced_hay_v2_arena_training.jl` passes `100 / 100` focused tests.

- hot allocation: 0 byte after kernel warm-up
- hot GC: 0 seconds
- block-local predictors remain bit-identical in root modes
- exact supervised head remains trainable unless explicitly frozen
- changing `head_weight` and `output_weight` while holding the trajectory,
  root error and adaptive feedback fixed leaves all 30 recurrent gradients
  bit-identical
- the same perturbation changes only the supervised-head gradient

## What remains unresolved

1. The adaptive root map is fitted by an exact-root shadow teacher.  This must
   be replaced with adaptive e-prop or SAL-style local alignment before it can
   be the production learning rule.
2. The block-local predictor still needs to be replaced by a predictor of
   top-down apical feedback, with signed E/I prediction residuals and burst
   gating.  No block should predict Tetris Q/death/geometry in the final path.
3. Routing remains weak or sign-inconsistent.  Its advantage must come from
   propagated apical mismatch rather than the current candidate-level scalar.
4. Intrinsic-cell gradients are badly scaled relative to exact BPTT.  For
   example, `branch_bias` has a large exact norm while the local norm is tiny.
   A compact D-RTRL/pp-prop-style eligibility for trainable cell dynamics is
   required; improving only the spatial third factor cannot repair this.
5. Held validation has not yet decreased, so this mode is not ready for a
   1k/10k/100k production run.

## Next critical path

1. Keep the exact-root ridge fit as a teacher control only.
2. Add a separate sparse feedback parameter set and online adaptive alignment.
3. Add the block-local apical feedback predictor and signed residual/burst
   third factor.
4. Replace routing reward with the block's propagated apical residual.
5. Add compact intrinsic-state eligibility, then repeat the head-frozen
   64-update and 1k scratch comparisons.

## Eligibility and routing repair

The first production-style 100k attempt was stopped after update 37,800.  A
tiny-model comparison showed that the layered feedback transport itself was
already aligned, while the conversion into routing, edge, and intrinsic-cell
updates was incomplete.

Two concrete defects were repaired:

1. Routing used a candidate-centered Plackett-Luce score-function estimator as
   its main supervised gradient.  Root-feedback modes now use the pathwise VJP
   of the actual fixed-mass soft workspace write.  The hard, possibly sampled,
   top-k remains in the forward pass.  The score VJP is also returned to block
   state and query input, rather than stopping at `workspace_key/query`.
2. Each edge kept a seven-scalar trace for only its destination branch.  That
   omitted `branch -> soma -> sibling branch -> soma` paths.  The fixed-memory
   cell-local temporal adjoint now produces E/I drive cotangents, which are
   combined with delayed presynaptic factors in a source-major replay.  No
   inter-cell graph is traversed backwards and no per-edge trajectory is
   stored.

Apical reads are now propagated through workspace decay and earlier workspace
writes during the same reverse local replay.  Pathwise routing credit from
those earlier writes is included as well.

On the same four-state batch after 32 online feedback warm-up updates:

| metric | before | after |
|---|---:|---:|
| recurrent cosine | 0.439797 | **0.902774** |
| credit-core cosine | 0.479527 | **0.909268** |
| edge cosine | 0.654979 | **0.844103** |
| `state_query_weight` cosine | -0.208920 | **0.949246** |
| `workspace_key` cosine | 0.142587 | **0.867712** |
| `delay_logits` cosine | 0.673508 | **0.845925** |
| `soma_leak_logits` cosine | 0.059581 | **0.891528** |

`workspace_decay_logit` is a one-scalar cancellation-sensitive diagnostic.
The symmetric-head control matches it directly (exact `-0.0203290`, local
`-0.0203735`, cosine `1.0`), proving the workspace recurrence adjoint is
correct.  With only 32 online feedback warm-up updates, the residual root-map
error could exceed the small candidate-aggregated decay gradient and reverse
its sign.  At 128 warm-up updates the feedback cosine reached `0.996302` and
workspace decay returned to the exact negative direction (cosine `1.0`).
Production `apical_predictive_online` runs therefore reject feedback warm-up
or recurrent-ramp values below 128.

The old per-edge trace buffers were removed.  A scaled 64-update smoke held
hot allocation and GC at zero and ran at roughly 100--107 states/s.  On the
same eight-state overfit panel, the former stochastic-routing penalty largely
disappeared: final excess loss at 1,000 updates was 0.631 stochastic versus
0.655 deterministic, compared with the earlier approximately 1.038 versus
0.424 result.

Pathwise routing is stronger than the former score-function signal.  The
default entropy/load constraints were therefore raised to 1.0/0.01.  A
500-update stochastic control retained routing entropy 0.376 without harming
the loss trend.  This is a stability constraint, not an environment reward;
the production signal remains a supervised workspace cotangent.

With 128 feedback warm-up, 128 recurrent ramp, and the stronger routing
constraint, an eight-state stochastic scratch run reached at update 2,000:

- final excess loss: `0.478295`
- last-100 mean excess: `0.455898`
- last-500 mean excess: `0.450447`
- minimum observed excess: `0.316508`
- hot allocation after warm-up: `0` bytes
- hot GC: `0` seconds

The former `0.6+` result at update 1,000 was therefore not a demonstrated
architectural floor.  The explicit overfit gate is now excess loss below 0.5;
this run passes it, although it does not claim zero training loss.
