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

The current post-failure credit repair, exact-gradient measurements, 5k
mechanism ablations, and eight-state memorization gate are recorded in
`CREDIT_REPAIR_V2_2026-08-01.md`.

The local learner is no longer used as the architecture's quality ceiling.
The allocation-free analytic exact-BPTT path, its 10k curve and mechanism
ablations are recorded in `EXACT_BPTT_PERFORMANCE_CEILING_2026-08-02.md`.

The two-timescale exact repair, six-stage representation probe and bounded
brain-internal sleep shadow are recorded in
`BRAIN_INTERNAL_SLEEP_SHADOW_2026-08-02.md`.  Sleep has not been promoted:
the alternating route/recurrent arm failed the wake-only loss/drift gate.

The subsequent meta-audit found that the shadow erased tag magnitude and task
state identity, and that its single saturated-checkpoint comparison was below
the measurement resolution needed for a performance claim.  The replacement
research contract and staged gates are recorded in
`SLEEP_LEARNING_META_RESET_2026-08-02.md`.

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

There are three deliberately separate credit-assignment tracks:

```text
quality/reference control:
PACK -> FORWARD -> REVERSE-TIME VJP -> AdamW

CPU performance control:
PACK -> FORWARD -> ANALYTIC EXACT GRAPH BPTT -> AdamW

CPU local-learning experiment:
PACK -> FORWARD -> BLOCK SIGNAL -> LOCAL REPLAY -> AdamW
```

The Zygote direct path remains the independent correctness oracle.  The
production-width quality ceiling is `credit_mode=:exact_bptt`, implemented by
`ReducedHayV2IntrinsicAdjoint.jl` inside the fixed-arena trainer.  The local
path remains available as an experimental CPU credit rule, but is suspended
for architecture-performance claims.

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

The cell also retains one scalar apical accumulator, soma voltage and
adaptation. This gives:

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

This is the complete state of the current reduced equation, but it is not yet
the smallest paper-mechanism-complete cell. TwinProp's apical/basal and NMDA
ablations motivate replacing the scalar apical accumulator with at least one
active apical compartment carrying its own voltage, AMPA, NMDA, GABA and
plateau state. The next equation-derived comparison is therefore four basal
plus one active apical compartment: 27 persistent states and one spike event,
or 28 observed coordinates per cell and 224 per eight-cell block. Those
numbers follow from the state equation; they are not compatibility widths.

Normalized voltage/time units are used intentionally. This is a Hay-derived
CPU model, not a claim that one Tetris cycle equals a biological millisecond.

## Canonical information, control and readout planes

The historical 48-dimensional block interface was a compatibility dimension,
not a dimension derived from Hay or TwinProp:

```text
8 cells/block * legacy6 coordinates/cell = 48 coordinates/block
```

`legacy6` exported only soma voltage, apical voltage and four branch voltages.
It allowed the first dendritic implementation to reuse the Point-SNN
workspace/head, but it hid AMPA, NMDA, GABA, plateau, adaptation and the soma
event. No paper result supports 48 as the required or sufficient dimension.

The canonical Reduced Hay observation is now `full24`. Its exact per-cell
coordinate order is:

```text
 1  soma voltage
 2  soma spike event
 3  apical voltage
 4  adaptation
 5  branch 1 voltage
 6  branch 1 AMPA conductance
 7  branch 1 NMDA conductance
 8  branch 1 GABA conductance
 9  branch 1 plateau
10  branch 2 voltage
11  branch 2 AMPA conductance
12  branch 2 NMDA conductance
13  branch 2 GABA conductance
14  branch 2 plateau
15  branch 3 voltage
16  branch 3 AMPA conductance
17  branch 3 NMDA conductance
18  branch 3 GABA conductance
19  branch 3 plateau
20  branch 4 voltage
21  branch 4 AMPA conductance
22  branch 4 NMDA conductance
23  branch 4 GABA conductance
24  branch 4 plateau
```

Signed voltages use `tanh`; nonnegative conductance, plateau and adaptation
states use `x/(1+x)`; the spike coordinate is retained directly. These are
monotone observation transforms, not a reduction in coordinate count. The
underlying SoA trajectory still retains the untransformed 23 continuous
states. NMDA current is not an additional independent persistent state: it is
computed from branch voltage, NMDA conductance and the learned voltage-unblock
parameters.

With eight cells per block, the natural block state is therefore:

```text
24 state coordinates/cell * 8 cells/block = 192 coordinates/block
30 blocks * 192 coordinates/block = 5,760 exact information coordinates
```

The three current full-state layouts have distinct roles:

| Preset | Information memory | Routing control | Global readout | Status |
|---|---|---|---|---|
| `reduced_hay_fullstate_bound_v10` | thirty 192D block states are superposed into one 192D signed-permutation workspace | the same 192D bound representation | 576D anchor/temporal/delta sketch | retained compression control |
| `reduced_hay_exact_slots_v11` | exact `30 x 192` block slots | learned `24 -> 4` features per cell, flattened to 32D and combined with fixed Hadamard block roles | learned rank-4 state factor with explicit block/cycle/cell axes | retained CPU low-rank control |
| `reduced_hay_exact_slots_fullrank_v12` | exact `30 x 192` block slots | the same dedicated 32D Hadamard control plane | learned `24 x 24` state transform plus explicit block/cycle/cell axes | retained nonlinear learned-basis control |
| `reduced_hay_exact_slots_direct_v13` | exact `30 x 192` block slots | the same dedicated 32D Hadamard control plane | all 24 state coordinates directly, with no learned state projection or extra `tanh` | current information-path control; matched 10k failed the promotion gate |

The exact-slot workspace never averages blocks. A selected block overwrites
only its own 192D slot; an unselected slot decays in place. The head receives:

```text
A[b]       = complete cycle-1 state of block b
H[t,b]     = state of block b when selected at cycle t
Delta[b]   = final state of block b - A[b]
```

`A` and `Delta` retain all blocks, so the final state is reconstructible as
`A + Delta`. `H` retains block and cycle identity and is sparse: only the
`cycles * workspace_k` selected events are read. In the 30-block, six-cycle,
top-5 coverage-first preset this normally records every block once, but it is
not the complete six-cycle trajectory of every block and it does not retain
route-rank order within one cycle.

Routing is intentionally separated from information storage. `route_dim=32`
is a control plane, not a replacement for the 5,760D memory. Each of the eight
cells projects its 24 observations to four route features. Thirty distinct
Walsh/Hadamard role codes fit in 32 dimensions; the coded block features form
the routing query and per-block score. A separate 32D selected context returns
global modulation to the scalar apical receiver, while each cell also reads
its own exact 24D workspace slot.

Adding a positional vector and immediately pooling was not adopted. If block
tokens are collapsed as

```text
sum_b (x_b + p_b) = sum_b x_b + sum_b p_b
```

the content-position association disappears and the positional sum becomes a
constant. Transformer positional encodings work because the token axis is
retained through attention. v10's multiplicative signed-permutation binding
retains more position information than additive pooling, but superposing 30
block states into 192 dimensions still introduces cross-talk. v11/v12 instead
use the block axis itself as the exact position representation.

The paper/Hay boundary remains strict. [TwinProp](https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full)
supports the importance of spatially separated dendritic voltages,
voltage-dependent NMDA effects, active conductances, apical/basal separation,
E/I timing and soma integration. It does not prescribe a
24-coordinate network interface, exact-slot workspace, Hadamard route code or
factorized Tetris head. The detailed Hay cell has a much richer morphology and
ion-channel set than this four-branch plateau model. `full24` is the complete
observation of this CPU-native Reduced Hay cell, not equivalence to full Hay
and not reproduction of TwinProp. The retained v12 10k checkpoint's eight
learned `24 x 24` projections were numerically full rank but had effective
ranks only about 9.8--12.7. v13 removes that transform entirely. Its matched
result shows that the transform was not the primary bottleneck: direct
full-state readout reduced CPU cost relative to v12 but reduced Tetris
quality.

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
| width-80 barrierless analytic exact BPTT | `train_reduced_hay_v2_arena.jl --credit-mode exact_bptt --deterministic-routing` |
| v10 full24 signed-binding control | `train_reduced_hay_v2_arena.jl --preset reduced_hay_fullstate_bound_v10 --credit-mode exact_bptt --deterministic-routing` |
| v11 exact-slot rank-4 control | `train_reduced_hay_v2_arena.jl --preset reduced_hay_exact_slots_v11 --credit-mode exact_bptt --deterministic-routing` |
| v12 exact-slot full-rank experiment | `train_reduced_hay_v2_arena.jl --preset reduced_hay_exact_slots_fullrank_v12 --credit-mode exact_bptt --deterministic-routing` |
| v13 exact-slot direct-state control | `train_reduced_hay_v2_arena.jl --preset reduced_hay_exact_slots_direct_v13 --credit-mode exact_bptt --deterministic-routing` |
| width-80 barrierless DECOLLE/e-prop training | `train_reduced_hay_v2_arena.jl` |
| held-teacher checkpoint evaluation | `evaluate_reduced_hay_v2_arena.jl` |
| v2 arena integration/equivalence checks | `test_reduced_hay_v2_arena_training.jl` |
| focused unit/finite-difference checks | `runtests.jl` |
| paired parent/current v2 CPU benchmark | `benchmark_v2_reference.jl` |
| four-arm CPU budget contract | `compare_cpu_budget.jl` |
| shared Point/Reduced/GRU preflight | `train_budget_arm.jl` |
| held-teacher budget validation | `validate_budget_arms.jl` |
| exact checkpoint ablation replay | `reevaluate_budget_ablations.jl` |
| six-stage frozen representation probe | `probe_reduced_hay_v2_representation_path.jl` |
| brain-internal sleep A-E shadow | `compare_reduced_hay_v2_sleep_shadow.jl` |
| TwinProp/paper reproduction history | `../paper_multicompartment_snn/` |

Example:

```powershell
julia --project=. --threads=1 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_direct.jl `
  --preset tiny_recurrent_v2 --updates 16 --state-batch 1 --width 40 `
  --fixed-panel true
```

The canonical builder default is `:reduced_hay_scaled_v2`.  The retained
`:tiny` and `:reduced_hay_scaled_v1` presets are legacy controls. The v12
representation is an explicit experiment preset rather than the default until
its matched learning and CPU measurements pass the promotion gate.

Production scratch training:

```powershell
julia --project=. --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_v2_arena.jl `
  --preset reduced_hay_scaled_v2 --updates 100000 `
  --state-batch 8 --width 80 --workers 20
```

Current exact-slot direct-state control:

```powershell
julia --project=. --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_v2_arena.jl `
  --preset reduced_hay_exact_slots_direct_v13 --credit-mode exact_bptt `
  --deterministic-routing --updates 1000 `
  --state-batch 8 --width 80 --workers 20
```

This command describes the v13 information-path control. It is not promoted:
the focused memorization gate passed, but the matched real-data 10k gate did
not.

Every new arena run now copies the actual Julia source closure plus
`Project.toml`, `Manifest.toml` and this README under
`output-dir/source_snapshot`, recording per-file SHA-256, the aggregate
closure hash, Git HEAD/status and the active project. Resume verifies the live
closure against that snapshot and fails closed on drift. The explicit
`--allow-source-drift-on-resume` flag is an unsafe research escape hatch, not
part of a promoted run.

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

The representation mainline has advanced beyond the models used for the
historical 1k/10k/100k results below. Those measurements remain valid for
their named presets and checkpoints, but they are not evidence for v13 or a
future active-apical model.

The current representation status is:

- `legacy6`/48D is retained only for historical compatibility;
- v10 verifies the `full24` observation and a 192D signed-binding compression
  arm, but that single-lane superposition is not the canonical information
  plane;
- v11 preserves the exact 5,760D block-slot information plane and explicit
  block/cycle head axes, while deliberately retaining rank-4 state readout as
  a CPU control;
- v12 keeps the same exact slots and raises the head state factor from 4 to
  24, removing the explicit state-axis dimensional reduction;
- v13 deletes the learned state projection and its extra `tanh`; the head
  directly reads all 24 state coordinates at every explicit block/cell/cycle
  axis;
- route32 remains a separate Hadamard-coded control plane in v11--v13; it
  does not replace the exact block memory;
- exact-slot v11--v13 use analytic exact BPTT. They do not silently
  fall back to the older local-credit contract.

The retained matched real-data comparison uses dataset manifest
`1f63172f...26ded` and the same 128-state panel
`9ad529fd...9fdc6`:

| Model/checkpoint | kernel wall | Excess | KL | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|---:|---:|
| v10 10k | `419.844 s` | `0.664228` | `0.557064` | `0.328125` | `0.945537` | `0.758950` |
| v12 5k equal-wall | `415.441 s` | `0.699962` | `0.641884` | `0.328125` | `0.951041` | `0.767320` |
| v12 10k | `835.210 s` | `0.685918` | `0.629623` | `0.304688` | `0.951089` | `0.772446` |
| v13 5k equal-wall | `418.765 s` | `0.745256` | `0.691138` | `0.289063` | `0.948784` | `0.759452` |
| v13 10k | `804.196 s` | `0.739358` | `0.686994` | `0.289063` | `0.950277` | `0.762884` |

The direct-state hypothesis is therefore rejected as a promotion by itself.
v13 preserves information and is faster than v12 in the warmed production
benchmark (`106.541` versus `59.649 states/s`, both zero allocation/GC), but
it loses the primary excess/KL and top-1 comparisons. The learned v12 basis
was not merely destructive compression; it also supplied a useful nonlinear
feature transform. v13 is retained as the clean information-path control.

The v13 eight-state gate did pass: deterministic evaluation after 1,000
updates measured excess `0.001904`, top-1 `1.0`, NDCG `0.999944` and pairwise
`0.997571`. Thus the matched failure is not a basic gradient or memorization
bug.

The focused v12 mechanism and complete-memorization gate has now passed. The
retained run is:

```text
D:\tetris-paper-plus\runs\reduced_hay_v2_arena\exact_slots_fullrank_v12_overfit8_exact1k_20260802
```

It trained the same eight states for 1,000 exact-BPTT updates from scratch.
Training excess reached `0.001038`; deterministic evaluation of the retained
checkpoint measured excess `0.000970`, top-1 `1.0`, NDCG `0.999796` and
pairwise `0.989005`. All `37/37` parameter groups changed, including the cell,
route, exact-slot feedback and full-rank head groups. The run processed
`65.13 states/s`, with zero reported hot-path allocation and zero GC time.

The matched v11 rank-4 control reached evaluation excess `0.004303` at
`88.68 states/s`. Thus full-rank v12 reduced the focused excess by about
`4.4x`, while the rank-4 control remained about `1.36x` faster. Both achieved
top-1 `1.0` on this memorized panel; these numbers locate a readout-rank
tradeoff rather than establish generalization.

A separate warmed v12 production-shape benchmark measured `59.649 states/s`,
again with zero hot allocation and zero GC. This establishes that the exact
5,760D information plane and full-rank head are executable in the fixed-arena
CPU path without reintroducing GC or hot allocation.

These are focused mechanism, gradient-reachability and memorization results.
The retained v12/v13 real-data curves above now establish the comparison, and
neither exact-slot variant matches the v10 primary quality at equal CPU wall.
This README therefore does not claim that either has matched DSRLN, PreAct,
Point-SNN or GRU, nor that either is the competitive final architecture.

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

The 2026-07-31 local-replay optimization kept width 80, state batch 8,
20 workers and all learning semantics fixed. On an identical 32-update
scratch schedule it improved mean throughput from `50.092` to `89.969
states/s` (`1.796x`); updates 9–32 averaged `93.295 states/s`. Hot allocation
and GC remained zero, and the focused suite remained `27/27`. The retained
benchmark entrypoint is `benchmark_reduced_hay_v2_arena.jl`; details and
rejected variants are recorded in `V2_ARENA_INTEGRATION_2026-07-30.md`.

The deterministic 128-state validation panel at update 1,000 measured
top-1 `0.078125`, NDCG `0.824687` and pairwise `0.508192` at
`241.997 states/s`. This is an early learnability/evaluator check, not a
competitive quality result; comparison with Point-SNN, DSRLN, PreAct and GRU
requires matched 10k/100k or equal-wall-clock runs on the identical panel.

The repaired width-80 local learner subsequently completed a retained 100k
scratch run.  It peaked at 10k on the fixed 128-row panel (top-1 `0.164063`)
and regressed to `0.085938` at 100k, while keeping plateau mean near `0.007`
instead of the old pathological value near `1.0`.  Gradient audit, fixed-data
memorization, exact/local claim boundaries and artifacts are recorded in
`CREDIT_REPAIR_100K_2026-08-01.md`.

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
dynamics. It may also claim that v11--v13 preserve this Reduced Hay model's
complete 24-coordinate observation in exact block slots. It must not claim:

- reproduction of TwinProp XOR/parity results;
- equivalence to the full Hay cell;
- that TwinProp or Hay specifies the project's `full24`, exact-slot,
  Hadamard-routing or factorized-head design;
- that direct full24 readout proves causal use of every mechanism or
  competitive Tetris performance;
- that the scalar apical accumulator is an active Hay/TwinProp apical
  dendritic compartment;
- a speed advantage from the current Zygote reference;
- complete Hay-mechanism efficacy from nonzero internal activity;
- superiority over the frozen 11-state, GRU, DSRLN or PreAct controls before a
  fixed-panel equal-wall-clock comparison.

## Next critical path

The representation gate now precedes further sleep/local-credit work:

1. retain v13 as the closed lossless-information control; its exact gradients,
   complete memorization and failed matched 10k result are all measured;
2. compare binary rails, cycle-1 all-block state, anchor+final,
   anchor+selected-history+final and all-block/all-cycle state with one frozen
   v13 checkpoint and one matched probe;
3. if all-cycle state wins, repair the cycle transition before changing the
   head again, beginning with a spike-only versus low-bandwidth analog event
   payload oracle;
4. if route order is causal, compare route32 against full-state swap-advantage
   prediction before increasing the control dimension;
5. separately test a fixed physics-current observation basis for AMPA, NMDA,
   GABA and plateau recruitment; derived channels belong in the head plane,
   not duplicated persistent memory;
6. implement the equation-derived four-basal plus one-active-apical control
   (`28/cell`, `224/block`) only after the cycle/readout oracle, with scalar
   apical, no-apical-NMDA/plateau and no-coincidence ablations;
7. require equal-update and equal-wall 10k improvement over v10 before any
   100k run;
8. continue reporting v12/v13 as controls rather than the final architecture.

The exact 5,760D information plane is not a request to make every operation
5,760-dimensional. Routing and sparse execution may remain low-dimensional as
long as they are labeled control planes and their selection quality is tested
separately. Conversely, a fast rank-4 head must not be promoted merely because
the full states exist upstream; it must match the rank-24 control closely
enough under the same learning budget.

## Deferred sleep and legacy optimization path

The following path is retained as research history and remains applicable
after the representation gate. It is not the immediate v13 critical path:

1. keep the corrected exact wake multiplier at `0.001`; it reduced the retained
   eight-state excess from `0.440186` to `0.013436`, but the full-width exact
   10k validation curve still did not improve;
2. do not promote the current sleep shadow: alternating route/recurrent sleep
   initially produced the same hard-route boundary drift as simultaneous
   control; the wake-margin guard now removes that drift, but does not improve
   wake-only loss;
3. retain the paired block-silencing result as a negative control: it measures
   internal causal contribution, but its dense-positive advantage did not
   improve awake teacher loss;
4. replace per-array unit-RMS wake tags with raw signed, confidence-bearing,
   state-conditioned consolidation tags captured during real wake updates;
5. establish Float64 measurement and proposal-only task-alignment gates before
   any further replay-length, advantage or learning-rate work;
6. require frozen zero-rail replay to reactivate task-tagged wake prototypes,
   not merely sustain generic firing;
7. run short scratch sleep integration only after alternating sleep beats
   wake-only without destructive head-input drift;
8. run the same validation-panel evaluator for Point-SNN and GRU under a
   matched width-80/equal-wall-clock schedule;
9. report exact-BPTT and DECOLLE/e-prop equal-update and equal-wall-clock
   curves separately;
10. tune candidate/local replay load balance if it improves states/s without
   changing serial-equivalence or zero-allocation behavior;
11. reproduce any promoted result on a second model/sampler/routing seed;
12. add the frozen 11-state arm when a qualified artifact exists.
