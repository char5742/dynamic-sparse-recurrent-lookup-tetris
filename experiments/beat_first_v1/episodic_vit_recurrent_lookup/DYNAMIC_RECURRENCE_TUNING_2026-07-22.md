# EVRL dynamic-recurrence tuning — 2026-07-22

## Correction and objective

The immediately preceding 100,000-update sweep changed dense learning rates
and weight decay while recurrent depth stayed fixed at two.  Those runs are
valid fixed-depth controls, but they did not tune the dynamic-recurrence
mechanism requested for this stage.  This ledger corrects that scope: every
trial here uses sampled hard halting from scratch after a 5,000-update random
depth curriculum, and changes one halting scalar at a time.

The architecture, 20,577,789 parameters, input contract, real-teacher data and
order, ranking loss, LookupFFN routing, active-only backward, sparse optimizer,
20-worker barrierless executor, held 128-state panel, and initialization seed
remain fixed.  Game validation and sealed seeds remain untouched.

Success requires more than a good final ranking score.  Deterministic held
depth must remain input dependent instead of saturating at the minimum or
maximum, sampled training depth must show the same qualitative behavior, and
the quality/compute result must remain competitive with the fixed-depth
controls.

## Trial R1 — zero compute price

### Isolated change

The prior dynamic setting used `compute_price = 0.02`.  R1 changed only this
value to zero.  It retained:

```text
warmup updates       5,000 (uniform random depth 2--6)
fixed depth          disabled
initial halt prob.   0.5
halt LR              5e-5
policy weight        0.05
entropy weight       0.001
dense weight decay   1e-4
bank/router LR       2e-4 / 4e-4
recurrent range      2--12
```

This tests the narrow hypothesis that the explicit price of extra computation
alone caused the previously observed depth-two collapse.

### Held-panel and depth curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin | Train depth | Held depth | Held range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 | 2.000 | 2.000 | 2--2 |
| 5,000 | 2.857160 | 0.51562 | 0.979035 | 0.840006 | 0.104091 | 2.000 | 2.000 | 2--2 |
| 10,000 | 2.804495 | 0.53125 | 0.979766 | 0.850205 | 0.095223 | 2.126 | 2.138 | 2--8 |
| 15,000 | 2.770932 | 0.53906 | 0.981487 | 0.855166 | 0.088338 | 6.072 | 5.193 | 2--12 |
| 20,000 | 2.735852 | 0.57031 | 0.983592 | 0.863802 | 0.082615 | 2.006 | 2.027 | 2--12 |
| 25,000 | 2.721970 | 0.53906 | 0.981148 | 0.864647 | 0.111630 | 2.001 | 2.005 | 2--4 |
| 30,000 | 2.690626 | 0.55469 | 0.983931 | 0.866516 | 0.107203 | 2.004 | 2.019 | 2--8 |
| 35,000 | 2.691957 | 0.63281 | 0.984052 | 0.874513 | 0.115638 | 2.475 | 2.149 | 2--10 |
| 40,000 | 2.669604 | 0.62500 | 0.986060 | 0.881214 | 0.128812 | 2.052 | 2.116 | 2--12 |
| 45,000 | 2.640631 | 0.61719 | 0.986034 | 0.882523 | 0.136873 | 2.002 | 2.018 | 2--12 |
| 50,000 | 2.644805 | 0.65625 | 0.987395 | 0.885983 | 0.126937 | 2.036 | 2.051 | 2--12 |
| 55,000 | 2.638223 | 0.65625 | 0.988389 | 0.889363 | 0.116839 | 2.134 | 2.209 | 2--12 |
| 60,000 | 2.631392 | 0.64062 | 0.986514 | 0.886535 | 0.141391 | 12.000 | 11.998 | 11--12 |
| 65,000 | 2.622564 | 0.67188 | 0.988940 | 0.892556 | 0.130395 | 12.000 | 12.000 | 12--12 |
| 70,000 | 2.614607 | 0.65625 | 0.989047 | 0.895548 | 0.132581 | 11.939 | 11.947 | 2--12 |
| 75,000 | 2.609085 | 0.69531 | 0.989456 | 0.897448 | 0.133078 | 2.033 | 2.052 | 2--12 |
| 80,000 | 2.596597 | 0.69531 | 0.990671 | 0.899239 | 0.131604 | 2.004 | 2.012 | 2--6 |
| 85,000 | 2.607638 | 0.68750 | 0.989438 | 0.897030 | 0.136655 | 2.000 | 2.000 | 2--2 |
| 90,000 | 2.595474 | 0.70312 | 0.990179 | 0.899312 | 0.139997 | 2.331 | 2.450 | 2--12 |
| 95,000 | 2.597807 | 0.70312 | 0.990683 | 0.898203 | 0.146593 | 2.017 | 2.031 | 2--9 |
| 100,000 | 2.610439 | 0.70312 | 0.989804 | 0.899255 | 0.142586 | 11.955 | 11.913 | 2--12 |

The critical result is the depth trajectory, not only the final checkpoint.
The held mean moved from `5.19` at 15k to `2.03` at 20k, saturated at almost
exactly 12 from 60k through 70k, returned to `2.05` at 75k, and ended at
`11.91`.  Training depth followed the same extrema.  This is not useful
input-dependent allocation; it is a high-variance policy oscillating between
the two depth boundaries.

### Quality and speed decision

R1 is rejected as a dynamic-recurrence configuration.  At 100k it reached
loss `2.610439`, top-1 `0.703125`, NDCG `0.989804`, pairwise `0.899255`, and
margin `0.142586`.  Relative to the fixed-depth Trial-1 control, it is worse by
`+0.028728` loss, `-0.015625` top-1, `-0.001997` NDCG, and `-0.006972`
pairwise accuracy.  It is also `-0.085938` below the recorded PreAct top-1.

The run took `4,035.570673 s` for 100,000 updates (`24.779643 updates/s`),
with average CPU `78.9237%` and candidate-region CPU `81.8531%`.  The fixed
depth control ran at `31.9193 updates/s`; unstable deep saturation therefore
cost about 22.4% of update throughput without improving held quality.

Zeroing the compute price disproves the narrow price-only hypothesis.  The
large boundary-to-boundary oscillation instead points to an overly aggressive
or noisy halt-policy update.  The next one-axis trial will keep R1 otherwise
identical and reduce only halt LR from `5e-5` to `1e-5`.  No dense LR, weight
decay, architecture, loss, or routing parameter will change.

### Runtime witnesses

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_recurrence_cp0_warmup5k_u100000_20260722_r1

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_recurrence_cp0_warmup5k_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
bytes:   253,690,013
sha256:  e39732e972ba32e7b35c4962fd26282a66e7d2cbeeafc356d7341013346589ef
updates: 100,000
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  ad133aeec3d5578c2fa57aa2c674509c4dd2acb55c049ab7c5ab9a43d69dd7ca

summary.json sha256:
  82b021d40bbfeca481042dbfebdf9135cbdf26531fac78923617977b4b16986f
```

Binary checkpoints and teacher data are not committed to Git.
