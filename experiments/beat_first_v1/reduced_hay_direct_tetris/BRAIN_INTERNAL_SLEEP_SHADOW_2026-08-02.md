# Brain-internal sleep shadow for Reduced Hay v2

Date: 2026-08-02

## Decision

The sleep mechanism is implemented as a bounded shadow experiment, but it is
not promoted to the production trainer.  The alternating route/recurrent arm
did not beat the wake-only control and caused the same hard-route boundary
change and head-input drift as the literal simultaneous-update negative
control.

The failed promotion gate is useful: the next critical problem is not another
sleep learning-rate sweep.  It is a wake-anchored route-margin constraint or a
causal block advantage whose local sleep improvement predicts preservation of
the awake hard top-k route.

## Prerequisite: exact two-timescale repair

The canonical exact-BPTT CLI accidentally used a recurrent learning-rate
multiplier of `1.0`, although the trainer contract used `0.001`.  The CLI now
defaults to `0.001`.

On the same retained eight-state memorization panel:

| Exact configuration | Updates | Excess loss | Top-1 |
|---|---:|---:|---:|
| old coupled rate | 2,000 | 0.440186 | 0.500 |
| two-timescale, recurrent multiplier 0.001 | 1,000 | 0.013436 | 0.875 |

The latter checkpoint is
`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\exact_bptt_overfit8_default_rlr001_u1000_20260802\checkpoints\checkpoint_000001000.jld2`
with SHA-256
`0e52e10c...` (the complete hash remains in the checkpoint manifest).

This establishes that simultaneous fast movement of the recurrent feature
space was a real failure mode.  It does not establish full-corpus quality.
The full-width scratch control completed 10,000 exact updates at `8.6725`
updates/s (`80,000` teacher states, `1,153.071` seconds), but its fixed
validation panel did not improve:

| Update | Excess loss | Top-1 | NDCG | Pairwise |
|---:|---:|---:|---:|---:|
| 1,000 | 1.060500 | 0.148438 | 0.865264 | 0.579524 |
| 10,000 | 1.073107 | 0.117188 | 0.864955 | 0.581073 |

All recurrent parameter groups changed, so this is not an accidental freeze.
The result supports keeping a slow wake phase, but also motivates a separate
mechanism for growing route/recurrent structure after the supervised head has
stabilized.

## Representation-path check

`probe_reduced_hay_v2_representation_path.jl` freezes the trained model and
fits the same small supervised probe to six stages.  At 2,000 probe updates:

| Representation | Source dimension | Excess loss | Top-1 |
|---|---:|---:|---:|
| binary rails | 1,298 | 0.000026 | 1.000 |
| full internal state | 17,664 | 0.000113 | 1.000 |
| all block exports | 4,608 | 0.000517 | 1.000 |
| ordered top-k blocks | 384 | 0.000051 | 1.000 |
| pooled selected workspace | 48 | 0.004018 | 0.875 |
| current head input | 96 | 0.000837 | 1.000 |

No stage irreversibly destroys the eight-state information.  The earlier
memorization failure therefore came from learning against a moving
representation, not from an incapable frozen readout.

## Sleep boundary

`compare_reduced_hay_v2_sleep_shadow.jl` enforces the requested brain-internal
boundary:

```text
external rails = exactly zero
stored samples = none
dataset reads during sleep = zero
teacher-target reads during sleep = zero
world model = none
separate generator = none
supervised head = frozen

learned recurrent circuit + deterministic internal noise
  -> spontaneous activity
  -> completion / continuation measurement
  -> route and recurrent reorganization
```

Wake contributes only parameter-local state: signed recurrent tags, routing
tags, block utility, a low-dimensional task-consistency tag, firing rate and
workspace-transition statistics.  The sleep function has no dataset argument.
The direct Tetris target is never attached to an individual block.

The sleep block advantage combines internal prediction-error reduction,
pattern completion, wake-tag consistency, activity cost, block-overuse cost
and novelty/replay-count correction.  Plackett-Luce top-k, entropy/load costs,
start-block noise, synaptic downscaling and firing-rate homeostasis are active.

This is a project-specific hybrid.  It is not a reproduction of a published
sleep algorithm.  The recurrent-noise/attractor and anti-Hebbian normalization
motivation is closest to Thiele, Diehl and Cook's wake-sleep SNN proposal
(https://arxiv.org/abs/1703.06290); noisy unsupervised replay is also supported
by Tadros et al. (https://www.nature.com/articles/s41467-022-34938-7), while
heterosynaptic/homeostatic stabilization is consistent with the sequence-SNN
study at https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00693/full.

## A-E shadow result

Retained artifact:
`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\sleep_shadow_arms_m8_t8_noise20_trust_20260802.json`

Configuration: eight stored wake tags, eight internal trajectories, eight
sleep micro-cycles, internal-noise scale `2.0`, recurrent rate `1e-5`, route
rate `2e-6`, downscaling/homeostasis rate `1e-6`.

| Arm | Excess after | Recurrent max delta | Route max delta | Head-input relative drift | Route-mask change |
|---|---:|---:|---:|---:|---:|
| A wake only | 0.013436 | 0 | 0 | 0 | 0 |
| B recurrent-only sleep | 0.013438 | 0.0001261 | 0 | 0.0000304 | 0 |
| C route-only sleep | 0.013436 | 0 | 0.0000167 | 0 | 0 |
| D alternating sleep | 0.013483 | 0.0001261 | 0.0000460 | 0.0072791 | 0.00001024 |
| E literal simultaneous update | 0.013483 | 0.0001261 | 0.0000465 | 0.0072791 | 0.00001024 |

All arms retained top-1 `0.875`; the more sensitive excess loss and
representation metrics reject D.  D changed both parameter families, but it
was worse than A and indistinguishable from the intended negative control E.
Its replay continuation length was `1.5` cycles and completion success was
`0.375`, so the internally generated trajectory was also too short and weak
to support the claimed consolidation mechanism.

The audit recorded `0` nonzero external-rail observations, `0` dataset reads,
`0` teacher-target reads and `0` supervised-head delta during sleep.

## Why the trust region was insufficient

Route-only backtracking accepted only two of eight proposals and preserved the
awake hard mask.  In the alternating arm, however, the recurrent update moved
the query/state distribution first.  The route proposal then improved the
sleep-internal prediction objective yet crossed a tiny number of awake hard
top-k boundaries.  A route-mask change fraction of only `1.02e-5` produced a
head-input relative drift of `0.00728`.

The current sleep-internal predictive loss is therefore not a reliable trust
metric for the awake routing boundary.  The alternating schedule alone does
not solve the closed-loop nonstationarity.

## Promotion gate and next critical path

The required gate was:

> Beat wake-only excess loss while both route and recurrent parameters change,
> without increasing destructive head-input drift.

It failed.  No short scratch or 100k integration is authorized by this result.

The next single experiment should add a compact wake-tagged route-margin
anchor.  Wake stores only the selected/nonselected boundary margin and the
corresponding local query/key tag, not a sample or teacher target.  During
sleep, a route proposal is rejected when its first-order bound predicts that
the signed awake margin will cross zero.  The recurrent phase must use the
same bound before route re-evaluation.  If that still cannot distinguish D
from E, replace the heuristic block advantage with an internal causal
counterfactual: replay the same noise trajectory with one selected block
silenced and measure its contribution to completion/continuation.

This is a mechanism test, not a hyperparameter sweep.  Promotion remains
conditional on the original A-E gate.

## Follow-up: wake-margin guard and causal counterfactual

The proposed follow-up was implemented on 2026-08-02.  Wake now stores, for
each candidate and cycle, only the weakest selected block, strongest
unselected challenger, signed score margin and their local query/state-key
factors.  Two small directional finite-difference tags bound the effect of a
recurrent synapse or soma-threshold proposal.  No row or target is retained.

Both recurrent and route proposals use this same first-order margin guard.
The literal simultaneous arm remains unguarded.

Retained margin-only artifact:
`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\sleep_margin_arms_m8_t8_20260802.json`

| Arm | Excess after | Head-input drift | Mask change | Margin violations |
|---|---:|---:|---:|---:|
| A wake only | 0.013435841 | 0 | 0 | 0 |
| B recurrent-only | **0.013434649** | 0.00000403 | 0 | 0 |
| C route-only | 0.013435841 | 0 | 0 | 0 |
| D alternating | 0.013435841 | 0.00000452 | 0 | 0 |
| E simultaneous | 0.013483286 | 0.00727914 | 0.00001024 | 1 |

The margin tag therefore solves the safety/discrimination problem: D and E
are no longer equivalent, D changes both recurrent and route parameters, and
the observed awake hard mask is preserved.  D still does not beat A, so the
promotion gate remains closed.

The second experiment added an actual paired causal intervention.  For the
same internal-noise seed, the factual trajectory is compared with trajectories
where one highly loaded block has its exported state and event spikes
silenced.  Its advantage is the factual-minus-silenced difference in pattern
completion, continuation, prediction consistency and energy.  The factual
trajectory is restored before every update and teacher-snapshot refresh.

Retained counterfactual artifact:
`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\sleep_counterfactual_arms_m8_t8_20260802.json`

| Arm | Excess after | Change from A | Recurrent max delta | Route max delta | Head-input drift | Mask change |
|---|---:|---:|---:|---:|---:|---:|
| A wake only | 0.013435841 | 0 | 0 | 0 | 0 | 0 |
| B recurrent-only | 0.013436556 | +0.000000715 | 0.0000847 | 0 | 0.0000153 | 0 |
| C route-only | 0.013435841 | 0 | 0 | 0.0000150 | 0 | 0 |
| D alternating | 0.013436556 | +0.000000715 | 0.0000877 | 0.0000489 | 0.0000153 | 0 |
| E simultaneous | 0.013437986 | +0.000002146 | 0.0001255 | 0.0000435 | 0.0000305 | 0 |

The D arm evaluated 192 block interventions across its two alternating phases;
155 had positive internal advantage.  The intervention is causal for the
internally generated trajectory, but the signal is too dense and is not
aligned with lower awake teacher loss.  D remains safe but is slightly worse
than A.

Therefore neither the margin-only nor counterfactual arm passes the original
promotion condition.  No short scratch or production integration follows.
The remaining bottleneck is now narrower: replay must produce a discriminative
internally grounded responsibility signal, not merely reward blocks whose
silencing reduces generic activity continuation.  Replay continuation also
remains short (`1.5 / 6` cycles for D), so a longer self-sustained attractor is
a prerequisite before another sleep-credit experiment.
