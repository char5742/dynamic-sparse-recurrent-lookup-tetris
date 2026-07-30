# Reduced Hay direct-Tetris mainline

## Decision

The high-dimensional-neuron mainline is no longer:

```text
Hay/NEURON
  -> dense Digital Twin
  -> distilled frozen 11-state cell
  -> Tetris
```

It is now:

```text
Hay-derived mechanism set
  -> CPU-native Reduced Hay cell
  -> direct end-to-end Tetris teacher learning
```

The objective is not biological fidelity or a TwinProp paper reproduction.
The promotion target is Tetris quality per CPU wall-clock second, with stable
learning, bounded memory and useful dynamic sparse execution.

The older Point-SNN, detailed Hay oracle, Digital Twin and frozen distilled
11-state implementation remain unchanged as controls and research history.

## Implementation audit

| Existing component | Decision | Reason |
|---|---|---|
| `ArenaWorkspaceTraining.TrainingArena` | reuse | fixed candidate storage, 1,298 rails and canonical candidate order |
| `loss_and_raw_gradient!` | reuse unchanged | exact shared 22-output ListNet/Q/death/quantile/geometry cotangent |
| Point-SNN source-major graph and fixed gates | reuse design/data layout | appropriate for sparse event delivery and utility consolidation |
| `DendriticCellArena` / SoA layout | reuse design | branch-major fixed storage and allocation-free scalar kernel |
| MPMC barrierless phases and worker-local gradients | reused | candidate jobs, deterministic parallel reduction and sharded AdamW now drive the v2 production trainer |
| Dendritic `local_hybrid` replay | adapted for v2 | the old three-state trace was replaced by a seven-component AMPA/NMDA/GABA/branch/plateau/soma/adaptation trace |
| Dense Digital Twin training | control only | useful paper/oracle experiment, unnecessary dependency for Tetris optimization |
| 11-state distillation/freeze | control only | the final Tetris cell must not freeze internal dynamics because of lineage |
| full Hay cell | oracle only | too expensive for mass placement; useful for mechanism ablation |

There are two deliberately separate credit-assignment tracks:

```text
quality/reference control:
PACK -> FORWARD -> REVERSE-TIME VJP -> AdamW

CPU production candidate:
PACK -> FORWARD -> BLOCK SIGNAL -> LOCAL REPLAY -> AdamW
```

The direct path is retained as the teacher and quality ceiling. The production
v2 trainer now implements the corresponding local path in
`ReducedHayV2ArenaTraining.jl`. It uses the causal state-update-before-route
order, located one-pulse sensory contacts, fixed-fanout recurrent structure,
block-fixed DECOLLE projections and forward Reduced-Hay eligibility traces.

Only hard spike, hard workspace routing, gate state and future compartment
placement are discrete in the reference path. Continuous state and parameters
are differentiated through the complete Tetris trajectory there; the CPU
candidate approximates the same credit with block-local third factors and
forward eligibility traces.

## Minimal Reduced Hay cell

Four basal compartments each retain:

- branch voltage;
- AMPA conductance;
- slow NMDA conductance;
- GABA conductance;
- NMDA-dependent plateau.

The cell also retains apical voltage, soma voltage and adaptation. This gives:

```text
4 * (V + AMPA + NMDA + GABA + plateau)
  + apical + soma + adaptation
= 23 persistent continuous states / cell
```

The NMDA current is explicitly voltage dependent. Excitatory and inhibitory
conductances use different decays and reversal directions. Basal compartments
couple to soma, workspace feedback drives the apical state, apical state
modulates basal integration, and a hard soma event is followed by reset and
adaptation.

Normalized voltage/time units are used intentionally. This is a Hay-derived
CPU model, not a claim that one Tetris cycle equals a biological millisecond.

## Reference direct credit assignment

`ReducedHayDirectTraining.jl` does the following:

1. pack real Tetris candidates with the existing fixed arena;
2. unroll every Reduced Hay cycle;
3. compute the existing canonical 22-output teacher objective;
4. inject its exact raw cotangent into one reverse-mode pullback;
5. update compartment, graph, routing and head parameters with AdamW.

The present reference uses Zygote to establish the BPTT contract. It is
correctness code and a teacher/control, not the only final learning rule.
`ReducedHayCellKernel.jl` already supplies allocation-free SoA state
transition and fired-source-only event delivery. Production experiments must
still compare the direct reference and the v2-specific DECOLLE/e-prop replay
at equal CPU wall-clock. The local backend is now executable at full width;
this does not by itself establish a quality win.

## CPU production credit path

The production update is:

```text
fixed width-80 candidate pack
  -> causal Reduced Hay forward for every candidate
  -> ListNet and 22-output raw cotangent
  -> forward trajectory replay
  -> block-local DECOLLE signal
  -> seven-component e-prop edge eligibility
  -> supervised-reward-surrogate routing update
  -> utility structure consolidation
  -> deterministic parallel reduction
  -> sharded AdamW
```

For every block and cycle:

```text
M_block =
    B_block * delta_raw
  + B_block * local_predictor_error
```

`B_block` is seed-fixed, block-specific and not trainable. The local predictor
uses only that block's exported soma, apical and branch states plus its spike
events. Recurrent third-factor construction does not read `head_weight` or
`output_weight`; exact analytic VJP remains only for `head_weight`,
`head_bias`, `output_weight` and `output_bias`.

Each weight, gate and delay trace follows the local causal path through:

```text
AMPA -> NMDA -> GABA -> branch voltage
     -> plateau -> soma -> adaptation
```

Routing retains stochastic Plackett-Luce top-k during training and hard top-k
during evaluation. Its reward is explicitly a candidate-centered supervised
ListNet/local-loss surrogate, not an environment return. Structure keeps an
exact fixed fanout, mask-aware execution, one source swap per consolidation,
connection cost, branch utility relocation and optimizer-moment reset.

Setting the recurrent third-factor scale to zero freezes all 30 cell/graph/
routing parameter groups including weight decay and structure changes, while
the four supervised head groups continue learning.

## Canonical entrypoints

| Purpose | Entrypoint |
|---|---|
| direct-Tetris Reduced Hay reference | `train_reduced_hay_direct.jl` |
| width-80 barrierless DECOLLE/e-prop training | `train_reduced_hay_v2_arena.jl` |
| held-teacher checkpoint evaluation | `evaluate_reduced_hay_v2_arena.jl` |
| v2 arena integration/equivalence checks | `test_reduced_hay_v2_arena_training.jl` |
| focused unit/finite-difference checks | `runtests.jl` |
| paired parent/current v2 CPU benchmark | `benchmark_v2_reference.jl` |
| four-arm CPU budget contract | `compare_cpu_budget.jl` |
| shared Point/Reduced/GRU preflight | `train_budget_arm.jl` |
| held-teacher budget validation | `validate_budget_arms.jl` |
| exact checkpoint ablation replay | `reevaluate_budget_ablations.jl` |
| TwinProp/paper reproduction history | `../paper_multicompartment_snn/` |

Example:

```powershell
julia --project=. --threads=1 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_direct.jl `
  --preset tiny_recurrent_v2 --updates 16 --state-batch 1 --width 40 `
  --fixed-panel true
```

The canonical builder default is `:reduced_hay_scaled_v2`.  The retained
`:tiny` and `:reduced_hay_scaled_v1` presets are legacy controls.

Production scratch training:

```powershell
julia --project=. --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_v2_arena.jl `
  --preset reduced_hay_scaled_v2 --updates 100000 `
  --state-batch 8 --width 80 --workers 20
```

Held validation-panel evaluation:

```powershell
julia --project=. --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/evaluate_reduced_hay_v2_arena.jl `
  --checkpoint D:\path\to\checkpoint.jld2 `
  --split validation --states 128 --workers 20
```

## Comparison contract

The first equal-budget screen has four arms:

1. Point-SNN (31 blocks x 12 public dimensions, 372 states);
2. Digital-Twin-derived frozen 11-state cell;
3. direct-Tetris Reduced Hay cell;
4. state-matched diagonal GRU.

The concrete model builders are `BudgetMatchedPointSNN.jl`,
`BudgetMatchedFrozenElevenState.jl`, `ReducedHayWorkspaceSNN.jl` and
`BudgetMatchedGRU.jl`. The frozen builder requires an accepted artifact and
fails closed when it is absent. Point, Reduced Hay and GRU use the same
reference trainer via:

```powershell
julia --project=. --threads=1 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_budget_arm.jl `
  --arm point --updates 6 --width 40
```

The static screen matches persistent state scalars within 5% and estimated
per-cycle scalar work within 50%. These estimates are only a precondition.
Every promoted comparison must use the same:

- teacher dataset and train rows;
- candidate batches and candidate order;
- 22-output loss;
- update count and wall-clock cap;
- fixed evaluation panel;
- CPU thread count and affinity policy.

Required measured outputs are loss components, top-1/NDCG/pairwise, updates/s,
states/s, CPU seconds, allocation/update, GC seconds, peak resident memory,
event rate, delivered edges and effective active cells.

## Current validation status

The 1,000-update comparison is recorded in `VALIDATION_2026-07-30.md`.
The 10,000-update learning curve is in
`VALIDATION_10K_2026-07-30.md`; the current result is the exact-checkpoint
continuation to 100,000 updates in `VALIDATION_100K_2026-07-30.md`.
The recurrent-path failure exposed by that comparison and the separate
causal repair arm are recorded in `RECURRENT_REPAIR_2026-07-30.md`.

- All three 10k checkpoints were resumed with exact parameters, AdamW state,
  training-order prefix and fixed panel, then completed to 100k for three
  independent seeds.
- Point and Reduced Hay are tied at 100k: mean composite loss is respectively
  `3.067797` and `3.064013`, while NDCG is `0.949494` and `0.948069`.
- The smaller diagonal GRU saturates at loss `3.174158`, NDCG `0.928019` and
  pairwise `0.700770`, but retains a small top-1 lead.
- Reduced Hay apical ablation costs `+0.135536` loss. Plateau costs only
  `+0.020348` with seed-variable sign, and recurrent-input ablation is
  effectively neutral at `+0.001859`.
- The legacy trajectory had a dense 15,576-parameter direct-input routing
  shortcut, repeated sensory injection, an input-independent first route,
  only three cycles, and collapsing independent recurrent gates.
- `:tiny_recurrent_v2` removes the dense input query, places 15,360 learned
  E/I sensory contacts across four branches, pulses sensory conductance only
  at cycle 1, routes after the compartment update, uses six causal cycles,
  and preserves exactly four learned recurrent contacts per source.
- At 10k, v2 recurrent-off costs `+0.011584` composite loss and reduces NDCG
  by `0.011866` and pairwise by `0.010937`. Plateau-off costs `+0.098793`
  loss.
- The exact v2 continuation reached 100k at loss `3.016713`, top-1
  `0.359375`, NDCG `0.948239`, and pairwise `0.766848`. Against the retained
  same-seed Point checkpoint, the differences are respectively `-0.047389`,
  `+0.109375`, `+0.009201`, and `+0.089252`.
- At 100k, v2 recurrent-off costs `+0.041193` loss and reduces top-1, NDCG,
  and pairwise. Plateau-off costs `+0.307227`; apical-off costs `+0.036818`.
  The repaired mechanisms remain causally useful rather than saturating away.
- This is one-seed equal-update evidence. The retained 100k run used the
  original dense-gather v2 reference at `67.399 updates/s`. The first CPU
  pass now replaces the recurrent and sensory gathers with direct
  source/contact-major kernels and exact custom pullbacks. A warmed paired
  100-update benchmark against parent `5c0af8f` measured `79.310` versus
  `55.625 updates/s` (`1.426x`) and cut allocation from `30.453` to
  `15.134 MB/update` (`50.3%`). This is still not an allocation-free,
  barrierless or equal-wall-clock gameplay victory.
- Reference GRU training remains about `1.77x` faster in the dedicated
  benchmark (`2.09x` in the concurrent continuation), so Reduced Hay has not
  won the equal-wall-clock CPU objective.
- Width 40 exposes 40,630 eligible rows; 100k is about 2.46 passes. The full
  100,243-row corpus requires width 80.
- the frozen 11-state control remains unavailable because no qualified
  artifact exists.

The first full-width local-backend scratch preflight completed on 2026-07-30:

- preset `reduced_hay_scaled_v2`, width 80, state batch 8, 20 workers;
- 1,000 updates / 8,000 teacher states from scratch;
- composite loss `6.053182 -> 3.776091`;
- all 34 parameter groups changed, including all 30 cell/graph/routing groups;
- local Q loss `5.453151 -> 3.434759`, death
  `0.693703 -> 0.547826`, quantile `2.725258 -> 2.575974`, and geometry
  `0.065771 -> 0.061002`;
- final firing rate `0.040790`, plateau mean `0.130119`, routing entropy
  `0.811297`;
- hot allocation `0` bytes and GC `0` seconds for all measured updates;
- checkpoint SHA-256
  `0c5151d2679ddc3c9247d6e4bd40a83cb1cf8923786ccc4b047ac776d531f83b`.

The deterministic 128-state validation panel at update 1,000 measured
top-1 `0.078125`, NDCG `0.824687` and pairwise `0.508192` at
`241.997 states/s`. This is an early learnability/evaluator check, not a
competitive quality result; comparison with Point-SNN, DSRLN, PreAct and GRU
requires matched 10k/100k or equal-wall-clock runs on the identical panel.

Run a retained exact continuation with:

```powershell
julia --project=. --threads=1 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/validate_budget_arms.jl `
  --updates 100000 `
  --resume-dir D:\path\to\10k\artifact `
  --resume-update 10000 `
  --evaluation-milestones 20000,50000,100000 `
  --validation-states 64 --width 40 --arms point,reduced,gru
```

Resume fails closed unless arm, update, seeds, exact schedule prefix and held
panel match. The runner writes parameter and optimizer state checkpoints at
every milestone. This is still an equal-update reference-BPTT result on
teacher-validation rows, not the final equal-wall-clock production or gameplay
evaluation.

## Claim boundary

Paper reproduction and CPU-model claims are separate.

This directory may claim that Hay-derived mechanisms were implemented and
that a direct Tetris teacher cotangent reaches the continuous internal
dynamics. It must not claim:

- reproduction of TwinProp XOR/parity results;
- equivalence to the full Hay cell;
- a speed advantage from the current Zygote reference;
- complete Hay-mechanism efficacy from nonzero internal activity;
- superiority over the frozen 11-state, GRU, DSRLN or PreAct controls before a
  fixed-panel equal-wall-clock comparison.

## Next critical path

1. run the same validation-panel evaluator for Point-SNN and GRU under a
   matched width-80/equal-wall-clock schedule;
2. extend the retained scratch run to 10k, inspect the fixed-panel curve, then
   proceed to 100k only if it continues improving;
3. report direct-BPTT and DECOLLE/e-prop equal-update and equal-wall-clock
   curves separately;
4. tune candidate/local replay load balance if it improves states/s without
   changing serial-equivalence or zero-allocation behavior;
5. reproduce the result on a second model/sampler/routing seed;
6. add the frozen 11-state arm when a qualified artifact exists.
