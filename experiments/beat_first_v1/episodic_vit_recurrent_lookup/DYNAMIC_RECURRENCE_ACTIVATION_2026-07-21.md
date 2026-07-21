# EVRL dynamic-recurrence activation study — 2026-07-21

## Question

The global-visual, full-283-token EVRL had been trained with recurrent depth
fixed to two. This study asked whether sampled hard halting could be enabled
after that training, whether the halt head should be reset, or whether dynamic
depth must be present from the beginning.

The model body, prediction head, LookupFFN routing, input, teacher, ranking
losses, candidate-independent evaluation, sparse optimizer, and real-teacher
sampler were kept unchanged. Game validation and sealed seeds were not used.

The trainer now permits only one explicit semantic checkpoint transition when
`EVRL_ENABLE_DYNAMIC_HALTING_TRANSITION=1`: inherited non-zero fixed depth to
sampled hard halting. Optimizer, routing, and loss hyperparameters must remain
identical. `EVRL_DYNAMIC_HALT_RESET_PROBABILITY` optionally resets only the
halt weight, bias, and their Adam moments during that transition.

## Fixed-depth source

The source model continued learning well beyond the earlier 25,000-update
report. At update 75,000 its held-panel result was:

| Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|
| 2.591471 | 0.71875 | 0.990663 | 0.900622 | 0.137788 |

Dynamic-transition branches started from the saved update-80,000 checkpoint:

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_visual_dw5_rf63_fixed2_u100000_20260721_r1\checkpoints\checkpoint_000080000.jls
sha256: 332728081d195814d7b2bcd17728fce83f92fe8a430cd62b2e936ea39d34e104
consumed real-teacher states: 320,000
```

## Results

| Branch | Training intervention | Final update | Held loss | Top-1 | NDCG | Pairwise | Margin | Held deterministic depth | Training sampled depth at end | Updates/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mid-training direct | enable hard halting at 80k | 90k | 2.588522 | 0.72656 | 0.991547 | 0.905774 | 0.141727 | 2 / 2 / 2 | mean 2.25, range 2–5 | 27.48 |
| Mid-training curriculum | random depth 2–6 for 5k, then hard halting | 90k | 2.586021 | 0.71875 | 0.991611 | 0.906437 | 0.145673 | 2 / 2 / 2 | mean 2.23, range 2–4 | 28.16 after release |
| From scratch | random depth 2–6 for first 5k, then hard halting | 20k | 2.735993 | 0.53906 | 0.983724 | 0.866709 | 0.091127 | 2 / 2 / 2 | mean 2.29, range 2–5 | 26.16 |

The `2 / 2 / 2` field is held-panel mean/minimum/maximum. The curriculum
segment itself ran at exactly mean depth 4.0 and 19.06 updates/s. It improved
the body at deeper recurrent states, but releasing hard halting still returned
every deterministic held example to depth two.

Direct mid-training activation reached its best recorded top-1 of `0.7265625`
at updates 85,000 and 90,000. Continuing it to 95,000 reduced top-1 to
`0.7109375`, so the 95,000 checkpoint was not adopted.

The from-scratch branch learned the ranking task normally: top-1 rose from
`0.21875` initially to `0.50781` at 5,000 and `0.53906` at 20,000. Its failure
to acquire deterministic variable depth therefore cannot be explained only by
introducing recurrence late.

## Halt-head reset ablation

Resetting only the halt head to probability `0.4` made deterministic inference
run to the maximum depth 12 immediately, with held loss `7.0786` and top-1
`0.328`. Training then drove sampled depth back toward approximately 2.4–2.5
within a few thousand updates. This branch was stopped and not adopted.

Clearing the prediction head was not tested because it would destroy the
already learned teacher-Q/ranking map without supplying the missing recurrent
credit signal. The halt-only reset is the narrower version of that hypothesis,
and it was insufficient.

## Diagnosis

The current policy-gradient target gives every candidate trajectory in a
state the same realized state-level ranking loss. It does not tell a candidate
whether one additional recurrent step improved that candidate relative to its
competitors. The compute price consequently supplies a clean pressure toward
early stopping, while the benefit of another step is noisy and coupled across
the candidate list.

The evidence supports this interpretation:

1. deeper-state curriculum improves continuous ranking metrics;
2. stochastic training trajectories still explore depths 3–5;
3. deterministic held inference remains depth two in every branch;
4. halt-head reset and from-scratch training do not resolve the collapse.

The next intervention should therefore target halting credit assignment, not
clear the prediction layer or replace the learned model body. A physically
sparse option is to probe a small fraction of candidates for one extra step,
hold the other candidate scores fixed, and supervise continuation from the
exact change in the state ranking loss. Compute price should be reduced during
that acquisition phase and restored only after the halt head separates useful
and useless extra computation.

## Checkpoint witnesses

```text
direct dynamic 85k:
  sha256 d9547600f987138f3c5415400743d817b18f07d7776be5d6ece220ca7f5a5749

direct dynamic 90k:
  sha256 7096d3a625671d5a6754768c4af9adb094b671e6ea61b051d9a142e50981729b

curriculum then dynamic 90k:
  sha256 70bf69636753d8da97c2ebb30ffda0a32eaeb7f446ac46f62d5225ad26944735

from-scratch dynamic 20k:
  sha256 e2a96fb65bb8cedf68b484f409d84aa8d219bd0795e7afb4b188bbdecd1a9e31
```

Binary checkpoints and teacher data are not committed to Git.
