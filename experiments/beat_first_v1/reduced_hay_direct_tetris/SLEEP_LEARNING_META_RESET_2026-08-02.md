# Reduced Hay v2 sleep-learning meta reset

Date: 2026-08-02

## Executive decision

The current sleep shadow is frozen as a negative/mechanism control.  It must
not be extended by tuning noise, replay length, advantage coefficients or
learning rates.

The research question is reset from:

> Can an internally generated trajectory invent a useful route/recurrent
> update?

to:

> Can wake-acquired, signed task credit be stored without samples, selectively
> reactivated by an internally generated trajectory, and consolidated under
> function-preserving constraints?

Sleep cannot create new Tetris information when it receives no rail, sample,
target, world model or generator.  Its possible role is deferred credit
application, selective consolidation, interference reduction and dynamical
regularization.  Any task direction must originate in wake.

## What the previous experiments actually established

Confirmed:

- fast joint route/recurrent learning destabilizes the head;
- a recurrent multiplier of `0.001` repairs the eight-state memorization
  failure;
- a wake-route margin tag can prevent destructive hard top-k changes;
- zero-rail internal trajectories and paired block-silencing interventions can
  be executed without dataset or target access.

Not established:

- that internal replay contains task-aligned credit;
- that alternating sleep improves Tetris performance;
- that longer spontaneous activity would improve learning;
- that the one-checkpoint A-E deltas are larger than numerical measurement
  noise.

The previous `0.01343584` excess loss is computed by subtracting two Float32
losses near `2.4427`.  The reported arm differences are only a few Float32
ULPs.  They are mechanism smoke results, not evidence of performance gain.

## Root-cause audit

### 1. Absolute evidence was destroyed

The old shadow takes a single virtual exact update and applies per-array RMS
normalization.  At the retained checkpoint, the raw wake tags were:

| Tag | Raw RMS | Post-normalization RMS | Approximate amplification |
|---|---:|---:|---:|
| synapse | `4.60e-8` | `0.666` | `1.45e7 x` |
| soma threshold | `1.63e-7` | `0.942` | `5.78e6 x` |
| route key | `1.44e-7` | `0.935` | `6.52e6 x` |

Thus a near-converged or noisy wake residual is converted into fixed-strength
plasticity.  The same problem occurs when sleep cost, eligibility,
route-gradient and causal advantage are normalized to unit RMS.  The model is
forced to update even when the correct action is no update.

All future tags must preserve magnitude, sign, accumulation count and
confidence.  Fixed calibration/clipping may bound them; per-observation unit
normalization may not create evidence.

### 2. Task identity was discarded

The old path compresses wake credit into:

```text
one global synapse direction
+ one global route direction
+ unsigned block utility
```

It discards candidate/cycle phase and the local state in which the credit was
earned.  Sleep then gates those tags with unrelated noise activity.  A useful
tag cannot be selectively retrieved because it has no attractor identity.

The minimum retained address is:

```text
block / cell
+ cycle or phase
+ low-rank local-state key
+ separate signed LTP and LTD consolidation tags
+ confidence / decay / replay count
```

This is synaptic and block-local plasticity state, not sample replay.

### 3. The internal objective measured the wrong causal effect

Current completion means that any non-seed block spikes; continuation means
that any cell keeps spiking.  Block silencing therefore measures generic
activity propagation, not reactivation of a task-tagged wake representation.
That is why most interventions produced a positive advantage.

Generic completion, energy, novelty, firing homeostasis and load balancing may
control replay priority or plasticity dose.  They may not determine the sign
of a task update.

### 4. Wake and sleep eligibility were conflated

Wake eligibility is causal for the wake trajectory.  It is not a gradient on
a different sleep trajectory.  The new separation is:

```text
fast wake eligibility
  x wake learning signal
  -> immediate wake update + signed consolidation tag

sleep replay eligibility
  x retrieved signed consolidation tag
  -> consolidation proposal

wake importance / route margin / representation basis
  -> proposal constraint
```

## New architecture contract

Sleep is split into four planes.

### A. Wake engraving plane

During real wake updates, without an extra virtual teacher query:

- apply the normal fast supervised head update;
- form the normal slow recurrent/route wake update;
- accumulate signed parameter-local consolidation tags;
- accumulate a diagonal confidence/importance statistic;
- bind block/cycle task salience to a low-rank local-state prototype;
- store route distribution/margin and a low-rank representation-protection
  basis.

The tag EMA retains raw magnitude.  Consistent wake updates accumulate; noisy
signs cancel.  A zero or low-confidence tag yields zero sleep plasticity.

### B. Frozen task-tagged replay plane

Before enabling sleep plasticity, freeze every parameter and verify:

```text
zero external rail
+ internal noise
+ tag-biased seed/excitability
-> selective reactivation of the matching wake state prototype and sequence
```

The wake tag controls replay probability or temporary sleep-state
excitability, not the sign of an update.  Sleep-specific threshold,
disinhibition, adaptation or plateau modulation is permitted as transient
state, but cannot alter wake parameters at this gate.

Required controls:

- shuffle state keys;
- shuffle tag-to-block assignment;
- reverse temporal order;
- remove priority tags;
- compare tagged-attractor retrieval, coverage and diversity.

Generic continuation length is secondary.  A long sequence that does not
match a tagged wake prototype is a failure.

### C. Constrained consolidation plane

For a proposed sleep update `u`, the task channel and internal channel are
separate:

```text
u_task = retrieved_signed_tag x sleep_eligibility
u_internal = homeostasis / energy / causal replay regularization
```

The internal component is projected so that it cannot oppose accumulated wake
credit.  In descent-direction convention:

```text
dot(wake_descent_tag, u) >= 0
```

The accepted proposal solves the local proximal problem:

```text
maximize  wake_tag_alignment + internal_regularization
subject to
  signed wake route margins remain positive
  pre-sleep route KL <= delta
  importance-weighted parameter drift <= epsilon
  low-rank wake representation drift <= epsilon_repr
```

Synaptic-importance protection is parameter local.  Route KL and the retained
margin guard are safety constraints, not claims of Tetris improvement.

### D. Alternating coadaptation plane

Only after A-C pass:

```text
1. recurrent proposal from retrieved tags
2. task-alignment and function-preservation projection
3. recurrent proposal commit
4. rerun the same sleep seed
5. route proposal from retrieved route tags
6. route KL/margin projection and commit
7. slow teacher/tag update
```

The simultaneous arm receives the same trajectory count, proposal norm and
constraints.  The previous unguarded simultaneous arm remains only a
destructive-stress control.

## Correct evaluation hierarchy

### Gate 0: measurement contract

- compute statewise ListNet/excess directly in Float64 from logits;
- verify a zero-update sham is bitwise identical;
- retain paired per-state deltas;
- define practical improvement as the larger of `1%` relative excess or ten
  times the repeated-measurement width.

### Gate 1: frozen task-tagged replay identity

- early, middle and late wake checkpoints;
- multiple common sleep-noise seeds;
- tagged state/sequence retrieval must beat all shuffle controls;
- no parameter update is permitted.

### Gate 2: proposal alignment shadow

Do not commit the sleep proposal.  Evaluation may read the wake dataset after
sleep has produced the proposal; sleep itself may not.  Measure:

- dot/cosine with accumulated wake descent tags;
- actual finite-difference awake-loss change;
- recurrent, route and joint proposals separately;
- correlation across checkpoint and noise seed;
- sign-reversed tag must reverse or destroy alignment.

If the lower confidence bound is not positive, no replay-length or learning
rate tuning follows.

### Gate 3: fair immediate A-E

All arms use the same real wake minibatches and teacher exposures.  Tags are
captured during those shared wake updates, not through an extra virtual
gradient query.  Wake-only applies its normal slow update; sleep arms defer or
consolidate the same credit.  Compare equal proposal dose and equal safety
constraints.

### Gate 4: repeated wake-sleep interference test

Alternate different subsets of the same eight-state panel:

```text
wake subset A -> sleep -> wake subset B -> sleep -> ...
```

Primary metrics are learning-curve AUC, retention of prior subsets and the
largest stable recurrent/route learning rate.  Sleep-only improvement of an
already saturated checkpoint is not the main objective.

### Gates 5-7

Only after Gates 0-4:

1. multi-seed short scratch;
2. exact 10k at equal teacher exposure and equal wall-clock;
3. 100k promotion.

## Immediate implementation decision

The next code is not another sleep updater.  It is a proposal-only task
alignment harness implementing Gates 0-2.  The current A-E script, margin
guard and block-silencing arm remain retained controls.  Route/recurrent sleep
updates stay out of production until the proposal-alignment gate passes.

This ordering follows the consistent boundary in the primary literature:
sleep can reactivate/protect wake-acquired memory without stored samples, but
does not invent information for an unlearned task; eligibility and synaptic
tags have distinct roles; and importance/KL constraints protect existing
function.  Relevant primary sources include:

- Golden et al. 2022, sleep-like replay without old samples:
  https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1010628
- Thiele et al. 2017, recurrent SNN wake-sleep stabilization:
  https://arxiv.org/abs/1703.06290
- Ding et al. 2024, spontaneous sequence replay and scaling:
  https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012218
- Bellec et al. 2020, trajectory-local eligibility decomposition:
  https://www.nature.com/articles/s41467-020-17236-y
- He et al. 2015, separate LTP/LTD eligibility traces:
  https://pubmed.ncbi.nlm.nih.gov/26593091/
- Zenke et al. 2017, sample-free online synaptic importance:
  https://proceedings.mlr.press/v70/zenke17a.html
- Farajtabar et al. 2020, low-rank function-preserving gradient projection:
  https://proceedings.mlr.press/v108/farajtabar20a.html
- Schulman et al. 2015, KL trust-region geometry:
  https://proceedings.mlr.press/v37/schulman15.html
