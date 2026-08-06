# Reduced Hay v2 analytic exact-BPTT performance ceiling

Date: 2026-08-02 (JST)

## Decision

Local DECOLLE/e-prop credit assignment is suspended as the architecture's
quality judge.  The production-width performance control is now analytic
exact BPTT in the existing fixed-arena, barrierless trainer.  The local path
remains in the repository as a CPU-learning experiment, but its score must
not be reported as the Reduced Hay cell's representational ceiling.

## Exact path

`credit_mode=:exact_bptt` reuses the causal production forward and reverses:

```text
22-output supervised Tetris cotangent
  -> supervised head and workspace readout
  -> workspace decay and hard-route straight-through derivative
  -> soma/apical/branch/plateau/conductance state history
  -> recurrent weight, delay and fixed-mass gate derivative
  -> earlier source spike and route mask
```

The discrete destination-branch placement is fixed.  The hard recurrent gate
mask is refreshed from the learned gate logits at every optimizer step while
preserving exact fixed fanout.  Local predictors, local third factors, utility
consolidation and stochastic routing are not part of this performance-control
run.  Exact mode therefore requires deterministic routing.

The analytic arena cotangent was compared on the same tiny-model batch with
the retained Zygote functional reference.  Loss agreed to `1e-6`; recurrent,
credit-core, edge and supervised-head cosine were all `1.000000` to printed
precision.  Every reported parameter group, including `workspace_key`,
`state_query_weight`, `workspace_decay_logit`, weights, gates and delays,
also had cosine approximately one.

## CPU result

The production preset was unchanged:

- `reduced_hay_scaled_v2` (`96 x 8 = 768` cells);
- batch 8, width 80, 20 workers, 6 cycles;
- learning rate `5e-4`, weight decay `1e-5`;
- deterministic routing and no routing entropy/load regularizer.

A 64-update scratch smoke completed at about `78 states/s`, with hot
allocation `0` bytes and GC `0` seconds.  The retained 10k run averaged about
`80-82 states/s` in its long windows and also recorded zero hot allocation
and zero GC.  This is about 13% below the earlier roughly `90 states/s` local
path, while avoiding the multi-second/update allocation cost of the Zygote
production-width reference.

Retained run:

```text
D:\tetris-paper-plus\runs\reduced_hay_v2_arena\
  exact_bptt_scaled_scratch_u1000_20260801_2314
```

10k checkpoint SHA-256:

```text
b9e636fbfea1999a220dff3717d0831c59d8667417fcd3312ed9c7ff534b0137
```

## Fixed validation curve

All rows use the same deterministic 128-state validation panel.

| Update | Excess loss | Top-1 | NDCG | Pairwise |
|---:|---:|---:|---:|---:|
| 1k | 0.981170 | 0.203125 | 0.893345 | 0.630504 |
| 2k | 0.943844 | 0.187500 | 0.895195 | 0.638778 |
| 3k | 0.933259 | 0.203125 | 0.903835 | 0.662598 |
| 4k | **0.915060** | 0.187500 | 0.910297 | 0.667412 |
| 5k | 0.939940 | **0.242188** | 0.914540 | 0.667353 |
| 6k | 0.981138 | 0.148438 | **0.916710** | **0.673689** |
| 7k | 0.959812 | 0.164063 | 0.900554 | 0.648970 |
| 8k | 0.943511 | 0.164063 | 0.905718 | 0.666876 |
| 9k | 0.936820 | 0.187500 | 0.908569 | 0.671602 |
| 10k | 0.926098 | 0.203125 | 0.909485 | 0.665282 |

The smooth ranking metrics improve through roughly 4k-6k, then oscillate.
Top-1 is noisy and peaks at 5k.  This is not evidence for continuing the same
schedule directly to 100k: the required short-curve improvement gate is not
met.

## 10k mechanism ablation

| Condition | Excess loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| full | **0.926098** | **0.203125** | **0.909485** | **0.665282** |
| recurrent edges off | 1.000289 | 0.179688 | 0.907191 | 0.664690 |
| plateau off | 1.029099 | 0.164063 | 0.902328 | 0.658648 |
| apical off | 0.960608 | 0.195313 | 0.899774 | 0.655469 |

Unlike the failed local-credit 100k result, the exact run measurably uses the
recurrent graph, plateau and apical path.  Nonzero activity alone is not the
evidence; the fixed-panel losses worsen when each mechanism is removed.

## Eight-state memorization

The same scaled model and optimizer were trained on one fixed eight-state
panel for 2k exact updates.  It reached:

- excess loss `0.440186`;
- top-1 `0.500000`;
- NDCG `0.957056`;
- pairwise `0.735108`.

This passes the earlier `excess < 0.5` gate, but it is not near-zero
memorization.  Exact credit assignment is therefore no longer the primary
failure, yet the current recurrent architecture/optimization still does not
use its nominal capacity efficiently.

## Conclusion and next gate

The result separates two claims:

1. The local-credit failure was real and is now bypassed by an independently
   verified exact gradient path.
2. The current Reduced Hay v2 architecture is still not competitive.  Exact
   learning activates its recurrent and high-dimensional mechanisms, but the
   10k quality curve saturates far below the established Tetris baselines.

Do not spend another 100k on the unchanged schedule.  The next performance
experiment should change one bottleneck at a time, beginning with routing
collapse/interference or the shared readout bottleneck, and must first beat
the retained 10k exact curve and the eight-state memorization result.

