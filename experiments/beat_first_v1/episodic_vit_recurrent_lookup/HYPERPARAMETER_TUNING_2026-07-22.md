# EVRL fixed-architecture hyperparameter tuning — 2026-07-22

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

Only scalar optimizer, weight-decay, routing-schedule, or halting
hyperparameters may change. Each accepted comparison runs to update 100,000,
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
