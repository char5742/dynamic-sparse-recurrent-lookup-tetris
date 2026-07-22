# EVRL fixed-architecture hyperparameter tuning — 2026-07-22

> Scope correction: Trials 1--4 below are fixed-depth optimizer/regularization
> controls.  They are valid quality baselines, but they do not tune dynamic
> recurrence.  Recurrence-focused trials are recorded separately in
> [`DYNAMIC_RECURRENCE_TUNING_2026-07-22.md`](DYNAMIC_RECURRENCE_TUNING_2026-07-22.md).

## Contract

All trials in this ledger keep the complete EVRL architecture fixed:

- 20,577,789 parameters;
- identical PreAct input boundary and real-teacher dataset;
- 283-token full cross-attention and four recurrent registers;
- five-stage dilated depthwise/pointwise visual path with `63 x 63` receptive field;
- local-8 learned spatial Q/K/V/O relation;
- three shared LookupFFN micro-layers, 13 tables per block, 4,096 rows per table,
  and physical top-3 active-row updates;
- candidate-independent evaluation, ranking objective, sampler seeds, split, and
  128-state held panel;
- 20-worker barrierless executor, no CPU pinning, chunk eight;
- no game validation or sealed seed access.

Only scalar optimizer, weight-decay, or routing-schedule hyperparameters
changed in the trials below; recurrent depth remained fixed at two. Each
accepted comparison runs to update 100,000,
records held metrics every 5,000 updates, and is committed and pushed before
the next trial begins.

## Trial 1 — completed fixed-depth baseline

This trial completed the existing baseline from its exact update-80,000
checkpoint to update 100,000. It is the control for subsequent from-scratch
100,000-update tuning trials; no hyperparameter was changed.

### Hyperparameters

```text
bank LR       2e-4        bank weight decay    0
router LR     4e-4        dense weight decay   1e-4
attention LR  2e-4        gradient clip        5
FFN LR        2e-4        route temperature    1.0 -> 0.25 by update 12k
token LR      2e-4        recurrent depth      fixed 2
register LR   2e-4
head LR       2e-4
```

### Held-panel curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin | Segment updates/s | Segment CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80,000 | 2.596073 | 0.69531 | 0.990361 | 0.901688 | 0.133506 | — | — |
| 85,000 | 2.591489 | **0.73438** | 0.991055 | 0.905273 | 0.134817 | 29.61 | 69.94% |
| 90,000 | 2.582912 | 0.72656 | 0.991496 | **0.906543** | 0.128577 | 30.76 | 72.46% |
| 95,000 | 2.583290 | 0.70312 | 0.991298 | 0.904486 | 0.140228 | 31.57 | 74.01% |
| 100,000 | **2.581711** | 0.71875 | **0.991801** | 0.906227 | **0.146405** | **31.92** | **74.63%** |

Loss and NDCG continued to improve through update 100,000, while discrete
top-1 peaked at update 85,000 and then oscillated. The curve is not a clean
loss plateau, but the top-1 variance motivates a conservative learning-rate
trial rather than a larger step size. The first tuned arm will multiply the
dense representation LRs by `0.75` while leaving the higher router LR and bank
LR unchanged, isolating late representation-update stability from sparse
routing capacity.

### Runtime and witnesses

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_baseline_fixed2_u100000_20260722_r1

training time for updates 80k -> 100k: 626.580054 s
updates/s:                         31.919305
states/s:                          127.677221
average CPU:                       74.6275%
candidate CPU:                     77.6558%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_baseline_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: ba8620d7caa3a648c7b73635005e400cf96944de36708bf17027537a1ca7e553
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  e2c4e4d4f2c8a243452dd65a417e3c80c610f6dc83983b31128e0b791ee1f095
```

Binary checkpoints and teacher data are not committed to Git.

## Trial 2 — dense representation LR 0.75x

This from-scratch trial changed only five dense representation learning rates
from `2e-4` to `1.5e-4`: attention, FFN, token/visual, register, and output
head. Bank LR remained `2e-4`, router LR `4e-4`, Lookup residual-alpha LR
`2e-4`, dense weight decay `1e-4`, and all architecture, seed, data, loss,
routing schedule, and fixed-depth settings matched Trial 1.

### Held-panel curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.856116 | 0.53906 | 0.978386 | 0.837953 | 0.057290 |
| 10,000 | 2.810584 | 0.51562 | 0.979042 | 0.845513 | 0.072452 |
| 15,000 | 2.813538 | 0.50781 | 0.980807 | 0.850272 | 0.070100 |
| 20,000 | 2.791762 | 0.54688 | 0.981539 | 0.853233 | 0.064467 |
| 25,000 | 2.763789 | 0.53906 | 0.981845 | 0.860536 | 0.089816 |
| 30,000 | 2.747787 | 0.54688 | 0.980351 | 0.858367 | 0.089927 |
| 35,000 | 2.742353 | 0.53125 | 0.981982 | 0.865703 | 0.101573 |
| 40,000 | 2.732583 | 0.52344 | 0.980848 | 0.865115 | 0.114253 |
| 45,000 | 2.710017 | 0.53906 | 0.982928 | 0.871285 | 0.114947 |
| 50,000 | 2.706551 | 0.54688 | 0.981505 | 0.869971 | 0.116825 |
| 55,000 | 2.684257 | 0.59375 | 0.985780 | 0.874209 | 0.103992 |
| 60,000 | 2.672819 | 0.59375 | 0.985924 | 0.876116 | 0.107469 |
| 65,000 | 2.655602 | 0.64062 | 0.986628 | 0.883175 | 0.114273 |
| 70,000 | 2.671004 | 0.58594 | 0.986522 | 0.883175 | 0.115489 |
| 75,000 | 2.652258 | 0.66406 | 0.987261 | 0.888353 | 0.132376 |
| 80,000 | 2.641293 | 0.65625 | 0.987924 | 0.891672 | 0.127088 |
| 85,000 | 2.652183 | 0.66406 | 0.987357 | 0.888313 | 0.144588 |
| 90,000 | 2.617375 | 0.67969 | 0.989205 | 0.893652 | 0.119101 |
| 95,000 | 2.614602 | 0.65625 | 0.989280 | 0.894817 | 0.125980 |
| 100,000 | **2.609035** | **0.69531** | **0.989974** | **0.896181** | 0.123212 |

### Decision

Trial 2 is rejected. Relative to the Trial-1 final checkpoint, its loss was
`+0.027324`, top-1 `-0.023438`, NDCG `-0.001828`, pairwise accuracy
`-0.010046`, and margin `-0.023193`. Lowering the dense LR did not remove
top-1 oscillation: top-1 fell from `0.64062` at 65k to `0.58594` at 70k and
from `0.66406` at 85k to `0.65625` at 95k. The intervention mainly delayed
learning and remained behind the baseline at the equal 100k budget.

The next trial restores all baseline learning rates and changes only dense
weight decay from `1e-4` to `3e-4`. The baseline has a held/training loss gap
at 100k, so this tests stronger regularization without slowing early updates.

### Runtime and witnesses

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_dense075_fixed2_u100000_20260722_r1

training time:       3,138.017562 s
updates/s:           31.867253
states/s:            127.469013
average CPU:         74.8478%
candidate CPU:       77.7877%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_dense075_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: 6cac26cd4b7b88ddde9f5f4a941cd8fa5215a2a0d103e43254a989144de9e16e
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  79ec6d082692065f38b704d9c47e9705ce903da18fd33c1ed2e236e3900e1059
```

## Trial 3 — dense weight decay 3e-4

This from-scratch trial restored every Trial-1 learning rate and changed only
dense weight decay from `1e-4` to `3e-4`. Lookup-bank weight decay remained
zero. Architecture, initialization, sampler, data order, loss, routing
schedule, fixed depth, executor, and held panel were identical.

### Held-panel curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.859876 | 0.53125 | 0.978856 | 0.841653 | 0.068698 |
| 10,000 | 2.791346 | 0.52344 | 0.979932 | 0.850689 | 0.085665 |
| 15,000 | 2.763028 | 0.57812 | 0.982595 | 0.861554 | 0.083727 |
| 20,000 | 2.721600 | 0.60156 | 0.983539 | 0.867600 | 0.082771 |
| 25,000 | 2.706555 | 0.58594 | 0.983643 | 0.870668 | 0.096921 |
| 30,000 | 2.679362 | 0.64062 | 0.984503 | 0.876927 | 0.118946 |
| 35,000 | 2.655379 | 0.71875 | 0.987099 | 0.882533 | 0.115824 |
| 40,000 | 2.676071 | 0.64844 | 0.985641 | 0.878742 | 0.136712 |
| 45,000 | 2.619821 | 0.69531 | 0.988658 | 0.886064 | 0.122701 |
| 50,000 | 2.639295 | 0.64844 | 0.986502 | 0.885248 | 0.135180 |
| 55,000 | 2.615425 | 0.74219 | 0.989078 | 0.889951 | 0.125065 |
| 60,000 | 2.611532 | 0.69531 | 0.988770 | 0.890968 | 0.134419 |
| 65,000 | 2.601829 | 0.73438 | 0.989487 | 0.896681 | 0.127673 |
| 70,000 | 2.618289 | 0.68750 | 0.989011 | 0.894113 | 0.126324 |
| 75,000 | 2.605750 | 0.75000 | 0.989422 | 0.897922 | 0.125464 |
| 80,000 | 2.594070 | 0.72656 | 0.990411 | 0.900268 | 0.119268 |
| 85,000 | 2.598925 | 0.73438 | 0.990028 | 0.900217 | 0.136233 |
| 90,000 | **2.591153** | 0.74219 | **0.990525** | **0.901540** | 0.128988 |
| 95,000 | 2.598547 | 0.71875 | 0.989788 | 0.900674 | **0.144499** |
| 100,000 | 2.607862 | **0.80469** | 0.990137 | 0.898083 | 0.144283 |

### Decision

Trial 3 is promising for top-1 but is not a clean all-metric replacement for
Trial 1. Its final top-1 `0.8046875` exceeds Trial 1 by `0.0859375` and the
same-panel PreAct result `0.7890625` by `0.015625`. This is the first EVRL
checkpoint to exceed that PreAct top-1 result under the recorded held-panel
conditions.

The continuous ranking metrics are weaker than Trial 1: final loss is
`+0.026151`, NDCG `-0.001664`, pairwise accuracy `-0.008143`, and margin
`-0.002122`. Across all five evaluations from 80k through 100k, Trial 3
averages top-1 `0.74531` versus Trial 1's `0.71563`, but averages loss
`2.59811` versus `2.58709` and NDCG `0.99018` versus `0.99120`.

Therefore `3e-4` is retained as the current top-1 arm, not declared a universal
winner. Because this held panel has now guided tuning, the PreAct crossing is
development evidence rather than a sealed generalization claim. Game
validation and sealed seeds remain untouched.

The next trial interpolates dense weight decay to `2e-4`, retaining baseline
learning rates. It tests whether the top-1 benefit can be preserved while
recovering Trial-1 loss, NDCG, and pairwise accuracy.

### Runtime and witnesses

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd3e4_fixed2_u100000_20260722_r1

training time:       3,202.446105 s
updates/s:           31.226131
states/s:            124.904522
average CPU:         74.6048%
candidate CPU:       77.3738%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd3e4_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: 51b008ea66041da9cfeb7b005a62e75f6d4f06a0a5a9dde94bc4c47653e51912
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  48480aabeb8a0a8e2762888dfbdb9a90d3b906a1569ae7611295d7f8e96b5440
```

## Trial 4 — dense weight decay 2e-4

This from-scratch trial interpolated dense weight decay between the Trial-1
baseline `1e-4` and Trial-3 top-1 arm `3e-4`. Every other parameter and all
execution/evaluation conditions remained identical.

### Held-panel curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.851713 | 0.53906 | 0.980043 | 0.840504 | 0.065302 |
| 10,000 | 2.784460 | 0.50000 | 0.979888 | 0.846149 | 0.080127 |
| 15,000 | 2.751044 | 0.56250 | 0.983050 | 0.860493 | 0.079198 |
| 20,000 | 2.713456 | 0.60938 | 0.983188 | 0.865943 | 0.094645 |
| 25,000 | 2.693339 | 0.57031 | 0.983044 | 0.870063 | 0.113391 |
| 30,000 | 2.660621 | 0.64062 | 0.986125 | 0.880302 | 0.116080 |
| 35,000 | 2.673281 | 0.64844 | 0.987352 | 0.883035 | 0.104091 |
| 40,000 | 2.649686 | 0.65625 | 0.987742 | 0.884113 | 0.131746 |
| 45,000 | 2.615869 | 0.66406 | 0.989097 | 0.889713 | 0.122493 |
| 50,000 | 2.623402 | 0.69531 | 0.987678 | 0.889993 | 0.129680 |
| 55,000 | 2.632644 | 0.65625 | 0.988862 | 0.891377 | 0.125231 |
| 60,000 | 2.611868 | 0.69531 | 0.989359 | 0.894629 | 0.136155 |
| 65,000 | 2.597276 | 0.75781 | 0.990263 | 0.899451 | 0.140998 |
| 70,000 | 2.611061 | 0.70312 | 0.990018 | 0.898287 | 0.132355 |
| 75,000 | 2.604669 | 0.71875 | 0.989874 | 0.901820 | 0.142173 |
| 80,000 | 2.604071 | 0.68750 | 0.990316 | 0.902574 | 0.139390 |
| 85,000 | **2.586791** | 0.72656 | 0.990715 | 0.904016 | **0.145932** |
| 90,000 | 2.592309 | 0.66406 | 0.990922 | 0.905285 | 0.133420 |
| 95,000 | 2.594969 | 0.67969 | 0.990804 | 0.904189 | 0.140677 |
| 100,000 | 2.590257 | **0.75781** | **0.991009** | **0.906449** | 0.142959 |

### Decision

Trial 4 is a useful intermediate arm but is not selected over both endpoints.
At 100k it improves Trial-1 top-1 by `0.0390625` and pairwise accuracy by
`0.000222`, while worsening loss by `0.008546`, NDCG by `0.000792`, and margin
by `0.003446`. Its final top-1 remains `0.046875` below Trial 3.

Across 80k–100k, Trial 4 averages top-1 `0.70313`, loss `2.59368`, NDCG
`0.99075`, and pairwise accuracy `0.90450`. This is below Trial 3's late top-1
average `0.74531` and below Trial 1's continuous-ranking averages. Thus the
current fixed-architecture sweep has two distinct winners:

- Trial 3 (`dense WD = 3e-4`) for held top-1;
- Trial 1 (`dense WD = 1e-4`) for held loss/NDCG and late continuous-ranking
  stability.

Trial 4 confirms a real regularization tradeoff rather than a monotonic scalar
improvement. Further tuning on this same 128-state panel would increase
selection bias. The next scientifically useful step is replication with a new
permitted development panel or multiple training seeds, while continuing to
leave game validation and sealed seeds untouched.

### Runtime and witnesses

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd2e4_fixed2_u100000_20260722_r1

training time:       3,078.199980 s
updates/s:           32.486518
states/s:            129.946073
average CPU:         74.6836%
candidate CPU:       77.7083%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd2e4_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: ab913b03a0c033341e5613c63010f349382bb3736fb67e06797116be10e7a0f2
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  8add363547a9acf4cd74d18c1d2cb8611c6cedcc5385a596107b341cf794492a
```

## Sweep summary

| Trial | Only changed setting | Final loss | Final top-1 | Final NDCG | Final pairwise | Final margin | Updates/s | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | baseline | **2.581711** | 0.71875 | **0.991801** | 0.906227 | **0.146405** | 31.92 | continuous-ranking winner |
| 2 | dense representation LR `0.75x` | 2.609035 | 0.69531 | 0.989974 | 0.896181 | 0.123212 | 31.87 | rejected |
| 3 | dense WD `3e-4` | 2.607862 | **0.80469** | 0.990137 | 0.898083 | 0.144283 | 31.23 | top-1 winner |
| 4 | dense WD `2e-4` | 2.590257 | 0.75781 | 0.991009 | **0.906449** | 0.142959 | **32.49** | balanced intermediate |

This tuning session executed 320,000 new updates and consumed 1,280,000 new
real-teacher state presentations: 20k to finish Trial 1 and 100k each for
Trials 2–4. Measured training time summed to 10,045.24 seconds. All three
from-scratch trials began from the same model seed and data order.
