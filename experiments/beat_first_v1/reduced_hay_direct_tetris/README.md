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
| MPMC barrierless phases and worker-local gradients | reuse after VJP closure | candidate jobs, deterministic reduction and sharded AdamW remain valid |
| Dendritic `local_hybrid` replay | not mainline credit | recurrent dynamics receive DECOLLE/e-prop, not the global teacher VJP |
| Dense Digital Twin training | control only | useful paper/oracle experiment, unnecessary dependency for Tetris optimization |
| 11-state distillation/freeze | control only | the final Tetris cell must not freeze internal dynamics because of lineage |
| full Hay cell | oracle only | too expensive for mass placement; useful for mechanism ablation |

The executor phase change required for production is:

```text
old dendritic path:
PACK -> FORWARD -> BLOCK SIGNAL -> LOCAL REPLAY -> AdamW

new direct path:
PACK -> FORWARD -> REVERSE-TIME VJP -> AdamW
```

Only hard spike, hard workspace routing, gate state and future compartment
placement are discrete. Continuous state and parameters are differentiated
through the complete Tetris trajectory.

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

## Direct credit assignment

`ReducedHayDirectTraining.jl` does the following:

1. pack real Tetris candidates with the existing fixed arena;
2. unroll every Reduced Hay cycle;
3. compute the existing canonical 22-output teacher objective;
4. inject its exact raw cotangent into one reverse-mode pullback;
5. update compartment, graph, routing and head parameters with AdamW.

The present reference uses Zygote to establish the BPTT contract. It is
correctness code, not the final throughput result. `ReducedHayCellKernel.jl`
already supplies allocation-free SoA state transition and fired-source-only
event delivery. The remaining CPU critical path is an analytic reverse-time
VJP over the fixed tape, followed by integration into the existing MPMC
candidate executor.

## Canonical entrypoints

| Purpose | Entrypoint |
|---|---|
| direct-Tetris Reduced Hay reference | `train_reduced_hay_direct.jl` |
| focused unit/finite-difference checks | `runtests.jl` |
| four-arm CPU budget contract | `compare_cpu_budget.jl` |
| shared Point/Reduced/GRU preflight | `train_budget_arm.jl` |
| held-teacher budget validation | `validate_budget_arms.jl` |
| exact checkpoint ablation replay | `reevaluate_budget_ablations.jl` |
| TwinProp/paper reproduction history | `../paper_multicompartment_snn/` |

Example:

```powershell
julia --project=. --threads=1 `
  experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_direct.jl `
  --preset tiny --updates 16 --state-batch 1 --width 40 --fixed-panel true
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
- Reference GRU training remains about `1.77x` faster in the dedicated
  benchmark (`2.09x` in the concurrent continuation), so Reduced Hay has not
  won the equal-wall-clock CPU objective.
- Width 40 exposes 40,630 eligible rows; 100k is about 2.46 passes. The full
  100,243-row corpus requires width 80.
- the frozen 11-state control remains unavailable because no qualified
  artifact exists.

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

1. do not extend the same width-40 configuration solely for more updates;
2. expose the full width-80 teacher corpus under the same Point/Reduced
   comparison contract;
3. report equal-update and equal-wall-clock curves separately;
4. determine why recurrent input is causally neutral and whether rising soma
   event rate erodes the intended sparse-CPU advantage;
5. only after a quality-per-CPU advantage, close the analytic reverse-time VJP
   and integrate it into the barrierless executor;
6. add the frozen 11-state arm when a qualified artifact exists.
