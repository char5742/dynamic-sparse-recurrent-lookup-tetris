# Float32 Numeric Core Research Record

Date: 2026-08-04

## Purpose

This experiment tests whether a bit-serial Reduced Hay numeric core can learn
its hard arithmetic transitions without allowing a desired output bit or phase
to create the eligibility trace.

The required learning order is:

```text
hard forward trajectory
-> teacher-free local eligibility
-> complete result observation
-> posterior success/failure modulation
-> parameter update
-> freeze
```

The frozen core is intended to become the numeric output and processing layer
of the route-free Tetris SNN.  Tetris integration is performed only after the
standalone numeric result below passes.

## Learning contract

### Boolean transition cells

The full-adder, full-subtractor, sticky-OR and round-to-nearest-even kernels use
hard Reduced Hay cells.  During forward execution, eligibility is generated
from:

- presynaptic hard events;
- postsynaptic AMPA and NMDA changes;
- postsynaptic GABA changes;
- branch voltage and plateau changes;
- soma movement, threshold proximity and the observed hard spike.

The observed action signs the saved E/I trace.  Only after every truth-table
trajectory has completed is the saved trace multiplied by success or failure.
The target bit never selects a synapse and never enters eligibility generation.

### Numeric register cell

The register is a complete 8-basal plus 1-apical Reduced Hay cell.  A cell-local
temporal eligibility replay follows the cell state only; it does not traverse a
network graph and does not read a target.  A subthreshold soma seed keeps silent
hard-zero trajectories plastic.  After every basal/apical case has generated a
tag, the observed hard action is compared with the desired write action and
only the resulting scalar success/failure modulation is applied.

### Phase controller

The hard selected next phase is stored without consulting the desired phase.
Posterior success strengthens the selected action and failure weakens it.  The
desired phase score is never directly incremented.

## Causal checks

The standalone numeric test verifies:

- changing a truth table while holding parameters and inputs fixed does not
  change Boolean eligibility;
- zero modulation changes no Boolean, register or phase parameter;
- positive and negative modulation produce opposite parameter deltas;
- a nonspiking register trajectory still produces nonzero eligibility;
- a successful hard-zero Boolean trajectory weakens excitation and strengthens
  inhibition;
- the register starts from the default Reduced Hay parameters and changes them
  before reaching the hard coincidence truth table;
- the trained hard kernels execute through the actual Reduced Hay state rather
  than silently reading the compiled truth table.

Test result:

```text
teacher-free numeric eligibility and posterior modulation: 20 / 20 passed
complete Float32 numeric-core test suite:                 2258 / 2258 passed
```

## Scratch learning result

Seed:

```text
0x465033324d414348
```

The initial hard kernels were not already correct:

| Kernel | Initial correct bits |
|---|---:|
| full adder | 8 / 16 |
| full subtractor | 8 / 16 |
| sticky OR | 3 / 4 |
| round-to-nearest-even | 7 / 16 |

Posterior-modulated learning reached exact hard behavior in:

| Component | Updates |
|---|---:|
| full adder | 20 |
| full subtractor | 20 |
| sticky OR | 20 |
| round-to-nearest-even | 20 |
| Reduced Hay register | 391 |
| hard phase controller | 6 |

## Arithmetic validation

The same learned transition kernels are shared across all bit positions.

| Width | Samples | Add | Subtract | Multiply | Divide |
|---:|---:|---:|---:|---:|---:|
| 4 | 256 exhaustive | 1.0 | 1.0 | 1.0 | 1.0 |
| 8 | 4,096 | 1.0 | 1.0 | 1.0 | 1.0 |
| 16 | 4,096 | 1.0 | 1.0 | 1.0 | 1.0 |
| 24 | 4,096 | 1.0 | 1.0 | 1.0 | 1.0 |

Random and IEEE-754 boundary-focused Float32 validation used 100,000 operand
pairs for each operation:

| Operation | Exact rate | Maximum ULP |
|---|---:|---:|
| add | 1.0 | 0 |
| subtract | 1.0 | 0 |
| multiply | 1.0 | 0 |
| divide | 1.0 | 0 |

The state-executed addition path, which evaluates every transition through the
actual Reduced Hay membrane, conductance, NMDA, plateau, soma and adaptation
states, matched native Float32 for 2,048 / 2,048 random operand pairs.

## Frozen artifact

```text
results/float32_numeric_core.bin
SHA-256: D189640C010B2B298FFF0D1CF1D363378D307F96B7F09CE3B5692C19ED41A1DF
```

The artifact is the output of standalone numeric learning.  Tetris must load
this frozen object; it must not silently retrain it or restore the removed
teacher-direct update.

## Scope boundary

The learned parts are the hard Boolean transition cells, the Reduced Hay
register gate and the phase controller.  IEEE-754 unpacking, bit-serial loop
ordering, normalization and packing remain explicit finite algorithms composed
from those learned transitions.  This result therefore establishes successful
learning of the shared hard numeric primitives and their exact Float32
composition; it is not a claim that an unconstrained network discovered the
entire IEEE-754 algorithm end to end.

## Historical post-freeze Tetris integration observation (superseded)

After this standalone record was created, the canonical Tetris model was
changed to load the checksum-verified frozen artifact.  Model construction no
longer retrains the numeric core.  The frozen 46-parameter internal Reduced Hay
dynamics of each Q-bit cell is excluded from both AdamW and generic output
homeostasis; Tetris learns only its afferent bit evidence.

The serial/barrierless canonical update comparison passed 25 / 25 checks.

An isolated one-state Tetris adapter control held recurrent updates,
homeostasis and structural rewiring fixed while leaving the posterior Q-bit
modulation active:

```text
update 0:   excess loss 4.146679, hard top-1 0.0
update 250: excess loss 0.346573, hard top-1 1.0
update 500: excess loss 0.316297, hard top-1 1.0
posterior Q modulations applied: 19,304,000
```

This establishes that the frozen numeric core and teacher-free Q eligibility
can learn a Tetris ranking distinction without recurrent parameter movement.

The full-plasticity one-state run reached hard top-1 1.0 at updates 250 and 300,
then lost it after the update-256 recurrent homeostasis/structural event.  Its
update-1,000 result was excess loss 0.607662 and hard top-1 0.0.  Because the
numeric cells remained bit-identical to the frozen artifact, this failure is
assigned to recurrent representation nonstationarity, not to Float32-core
learning or accidental unfreezing.  Full Tetris promotion therefore requires a
separate staged recurrent-plasticity schedule; it does not invalidate the
standalone numeric result above.

## Historical frozen-Q canonical ladder, same-seed rerun (superseded)

The canonical route-free learner was subsequently corrected in three focused
ways before repeating the ladder:

- an inhibitory GABA event now retains teacher-free eligibility even when its
  destination emits a hard spike;
- eligibility is frozen after the first phase-valid Q-bit event, so currents
  after a latched hard decision cannot receive causal credit for it;
- output and recurrent optimizer clocks are independent, and recurrent
  plasticity starts at update 4,096 after the hard numeric adapter stabilizes.

All ladder sizes now use model seed `0x48415939`.  The former state-count-based
seed changed the initial model at every capacity step and was not a valid
comparison.

The final one-state run passed the sustained hard criterion:

| States | Updates | Excess loss | Hard top-1 | Updates/s | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 5,000 | 0.000542 | 1.000 | 41.25 | pass |
| 8 | 5,000 | 0.714509 | 0.250 | 26.18 | fail |

The one-state run activated DECOLLE, subthreshold/nonspiking e-prop,
homeostasis, synaptic scaling, utility and rewiring after the staged recurrent
start and did not collapse afterward.

The eight-state run did not fail because the learning path was inactive.  Its
final Q-bit accuracy was 0.7548, sign/exponent accuracy was 0.9850, and several
mantissa bits collapsed to the same hard value for every one of the 52
candidates.  Recurrent plasticity during updates 4,096--5,000 did not remove
that collapse.  The current evidence therefore places the next bottleneck at
the mapping from recurrent high-dimensional state to one hard Q cell per
low-order Float32 bit, not at the frozen arithmetic core.

The 16-state run was intentionally not started: the canonical gate requires a
stable eight-state pass first.  The old single-batch entrypoint also represented
only the first eight rows when given 16 rows; it now fails closed instead of
reporting a false 16-state result.  A rotating two-batch 16-state path should be
enabled only after the eight-state representation bottleneck is resolved.

## 2026-08-05 Tetris adapter contract correction

A mathematical audit found that the later Tetris adapter had regressed from
the documented IEEE-bit contract. Its 32 physical cells were decoded as four
radix-9 population-count digits. That representation had only `9^4 = 6561`
levels, made all eight cells within a digit interchangeable, and reduced the
posterior teaching signal to four distinct coefficients. The historical
standalone arithmetic-core result above remains valid, but ladder measurements
made through that radix adapter are not measurements of one-cell-per-IEEE-bit
Tetris output.

The canonical adapter now uses:

```text
phase-valid maximum soma margin per output cell
-> one hard bit per Reduced Hay cell
-> authoritative 32-bit IEEE-754 word
-> finite safety map only for transient invalid student words
```

Forward eligibility is the target-free derivative of the maximum margin. The
teacher is unpacked only after all candidates finish and supplies the posterior
per-bit BCE factor. Q task afferents are basal-only; the apical compartment is
again reserved for the frozen phase controller. The retired scalar surrogate-Q
substitution and radix sensitivity do not remain as control paths. Checkpoint
schema 4 rejects both the shape-compatible radix checkpoints and schema-3
checkpoints whose Q-cell internals were frozen.

## 2026-08-05 mathematical bottleneck audit

The corrected 8-state panel contains 52 candidate appearances but only 46
unique binary-rail inputs. Six duplicate pairs have identical Q words and all
auxiliary targets. Accordingly, every per-bit local Jacobian has maximal
identifiable rank `46/52`; the six missing rows are data duplicates, not model
capacity loss. The desired smoothed bit-logit residual lies in each Jacobian's
column space to approximately `1e-13` relative error.

The problem is conditioning and objective geometry:

| Measurement | Result |
|---|---:|
| raw per-bit Jacobian condition | `1.7e4` to `1.37e5` |
| row-normalized condition | essentially unchanged |
| column-normalized condition | `3.3e3` to `1.3e4` |
| initial bit BCE | `0.775990` |
| damped GN trial bit BCE | `0.667618` |
| same GN trial hard top-1 | `0.25` |

Thus the parameters can represent the eight-state targets locally, but the
unpreconditioned local optimizer sees an extremely ill-conditioned problem.
Moreover, IEEE bit BCE and Tetris ranking are not metrically aligned: a sign or
exponent error can dominate many correct mantissa bits. The observed decrease
in bit BCE together with worse hard ranking is therefore expected, not a
contradiction. A diagonal curvature preconditioner and a naive 24-coordinate
dendritic readout were both tested and rejected because they worsened the hard
8-state result; neither remains in the canonical path.

The audit also corrected the scope of the Tetris integration claim. At that
historical point, runtime used only `register_cell` and `phase_controller` from
`BitSerialMachine`:

```text
Tetris state
-> 32 independently evaluated frozen Reduced Hay register cells
-> hard UInt32 word
-> reinterpret(Float32)
```

The learned adder, subtractor, sticky OR, round-to-nearest-even, carry, shift
and normalization machinery is not executed. This is a valid Stage-A hard-bit
adapter, but it is not the requested complete numeric output machine. A true
Stage-B integration must instead have the form

```text
learned Tetris contribution / operand emitter
-> frozen bit-serial accumulator and arithmetic transitions
-> normalization / rounding / pack
-> sole final 32-bit result register
```

and must prohibit any direct recurrent-state-to-final-bit bypass. Before that
larger integration, a trust-region LM oracle was considered as a frozen-cell
capacity test. The direct experiment below superseded that proposal by
demonstrating trainable shared-cell capacity and promoting it canonically.

## 2026-08-05 unfrozen Q-cell capacity experiment and promotion

The frozen-cell diagnosis above was not a valid upper-bound measurement.  It
tested only whether fixed register dynamics plus afferent weights could solve
the panel.  A temporary direct-learning experiment therefore froze the
recurrent graph, payload gains and all auxiliary outputs while training only Q
afferents, the Q basal intercept and the high-dimensional Q-cell dynamics.

The posterior gradient follows this call chain:

```text
fixed recurrent trajectory
-> replay ten-cycle Q-cell trajectory
-> teacher-free afferent / basal / cell-parameter eligibility
-> exact target-bit BCE after all candidates finish
-> optional probabilistic IEEE top-vs-rest ordinal signal
-> temporary Q-only optimizer
```

The winning maximum-margin readout has a direct continuous derivative.  The
intermediate hard spike reset and adaptation paths retain the canonical
triangular surrogate.  This is therefore direct surrogate BPTT through the Q
cell, not a fully conditional exact derivative.  Successful learning is valid
evidence of trainable capacity; failure would not prove representational
impossibility.

The IEEE sortable-key surrogate used by the optional ordinal term handles sign
first, normal lexicographic order for positive magnitudes and reversed order
for negative magnitudes.  Targets are finite; the soft surrogate intentionally
does not reproduce the hard evaluator's temporary NaN safety mapping.  Central
finite differences covered low mantissa, mid-mantissa and sign probabilities.
The largest observed absolute error was `4.2e-5` on a sign derivative and
`1.8e-8` on the tested lower-bit derivatives.

Measured eight-state results from the same seed and 52 candidate appearances:

| Internal dynamics | Objective | Updates | Bit BCE | Bit accuracy | Exact words | Hard top-1 | Sustained top-1=1 interval | Updates/s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| per-bit trainable | bit BCE | 5,000 | 0.076485 | 0.989183 | 0.826923 | 0.875 | none | 30.81 |
| per-bit trainable | bit BCE + 0.1 ordinal | 2,000 | 0.137114 | 0.959135 | 0.461538 | 1.000 | 1,300--2,000 | 32.16 |
| shared trainable | bit BCE + 0.1 ordinal | 2,000 | 0.146323 | 0.959736 | 0.557692 | 1.000 | 1,400--2,000 | 33.21 |

The shared arm has only one trainable 46-parameter internal cell vector copied
identically to all 32 bit cells.  Its columns remained bitwise equal at every
update, while its maximum internal raw-parameter change reached `3.968843`.
The recurrent and auxiliary frozen-state assertions passed after every run.

The conclusion is now unambiguous for this panel:

> the high-dimensional cell has enough directly trainable capacity; freezing
> its internal dynamics was a confound, and per-bit internal specialization is
> not required for eight-state ranking mastery.

Bit BCE alone eventually reconstructs most exact words, but it is a poor early
ranking objective.  A small posterior ordinal term reaches and sustains hard
top-1 mastery while preserving about 96% bit accuracy.  The reported composite
ranking excess remains evaluation-only because death, geometry and other
auxiliary channels are deliberately frozen in this capacity experiment.

The temporary optimizer was then removed after its winning rule was promoted
to the one canonical learner. Canonical training now generates Q afferent,
basal and internal cell eligibility without the teacher, combines posterior
bit BCE with a `0.1` sortable-key ordinal signal, averages the 32 internal-cell
gradients, and applies one shared Adam update to the 46-parameter cell vector.
Parameters and both moments are copied identically across the 32 physical bit
cells after every update. Q afferents and basal intercepts use candidate-mean
Adam with zero Q decay by default; auxiliary output learning remains separate.

### Canonical shared-Q capacity gate

After matching the isolated experiment's objective normalization exactly, the
production `train_update!` path passed the same-seed eight-state gate in 2,000
updates:

| Metric | Result |
|---|---:|
| initial composite excess | 1.158253 |
| final composite excess | 0.900976 |
| hard/tie-aware top-1 | 1.000 |
| sustained top-1=1 interval | 1,300--2,000 |
| IEEE bit accuracy | 0.960337 |
| exact words | 29 / 52 |
| throughput | 25.93 updates/s |

The ordinal objective is `0.1 * mean(pair loss)` alongside mean bit BCE. Its
raw posterior signal therefore compensates the optimizer's candidate mean and
the shared cell's 32-bit mean by `32 * valid_candidate_count`; omitting that
factor weakened ordinal credit by `1/1664` on this panel and failed the gate.
This normalization is now covered by a direct regression test.

The recurrent schedule starts at update 4,096, after this 2,000-update capacity
gate. These measurements validate the canonical shared high-dimensional Q-cell
path, not the later recurrent DECOLLE/e-prop/plasticity phase.

This promotion does not turn the Stage-A result register into the full Stage-B
arithmetic machine.  The separately retained bit-serial arithmetic core stays
frozen for the later accumulator path.  The canonical claim is narrower: one
shared high-dimensional hard-spiking numeric cell can be trained directly from
Tetris supervision without per-bit internal specialization. The production
eight-state gate must be reported separately from the isolated experiment.
