# Reduced Hay direct-Tetris 10k learning curves — 2026-07-30

## Question

The 1,000-update comparison established that all three trainable controls could
learn, but it could not test whether the smaller diagonal GRU would saturate
before the higher-dimensional recurrent models.

This follow-up increases scratch training by 10x and measures the same held
teacher panels at 1k, 2k, 5k and 10k.

## Protocol

- models: Point-SNN, direct Reduced Hay, diagonal GRU;
- independent model/schedule/panel seeds: 3;
- scratch updates: 10,000 per model and seed;
- held panel: 64 predefined `validation` rows per seed;
- candidate count: at most 40;
- state batch: 1;
- learning rate / weight decay: `2e-4` / `1e-5`;
- Julia threads / BLAS threads: 1 / 1;
- objective: canonical 22-output
  ListNet/Q/death/quantile/geometry loss;
- trainer: warmed Zygote direct-BPTT reference.

Within a seed, every model received the same training-row order and held
panel. Each arm retained parameter and optimizer state checkpoints at all four
milestones. Training timing excludes held evaluation and checkpoint writes.

The frozen 11-state control remains unavailable because no qualified artifact
exists. These are teacher-validation rows, not gameplay-validation or sealed
seeds.

## Learning curve

Composite loss, mean over three seeds:

| Update | Point-SNN | Reduced Hay | diagonal GRU |
|---:|---:|---:|---:|
| 1,000 | 4.770735 | 4.684535 | **4.540472** |
| 2,000 | 4.579693 | 4.479737 | **4.331814** |
| 5,000 | 4.299577 | 4.093651 | **3.655269** |
| 10,000 | 3.794888 | **3.246140** | 3.258115 |

The 5k-to-10k reductions were:

| Arm | Loss reduction | Per 1k updates |
|---|---:|---:|
| Point-SNN | 0.504689 +/- 0.074748 | 0.100938 |
| Reduced Hay | **0.847511 +/- 0.036919** | **0.169502** |
| diagonal GRU | 0.397154 +/- 0.060225 | 0.079431 |

Reduced Hay improved `2.13x` as much as GRU in the second half and crossed the
GRU mean composite loss by 10k. The paired per-seed Reduced-minus-GRU losses
at 10k were:

```text
+0.032585, +0.000596, -0.069104
```

Their mean is `-0.011975 +/- 0.051997`. The crossover is therefore real in the
aggregate curve but not yet statistically decisive with three seeds.

GRU still improved by `0.397` from 5k to 10k, so it is slowing relative to
Reduced Hay but is not fully saturated.

## 10k held quality

Values are mean +/- sample standard deviation over three seeds.

| Arm | Composite loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| Point-SNN | 3.794888 +/- 0.046623 | 0.192708 +/- 0.047735 | **0.923303 +/- 0.009199** | 0.661207 +/- 0.013364 |
| Reduced Hay | **3.246140 +/- 0.058937** | 0.197917 +/- 0.078644 | 0.919309 +/- 0.012697 | 0.655468 +/- 0.017507 |
| diagonal GRU | 3.258115 +/- 0.073585 | **0.208333 +/- 0.032526** | 0.913588 +/- 0.005332 | **0.672456 +/- 0.009688** |

The controls specialize differently:

- Point has the best NDCG despite much worse composite loss;
- Reduced Hay has the best composite loss;
- GRU has the best top-1 and pairwise score.

Reduced Hay's composite crossover must therefore not be reported as uniform
ranking superiority.

The final loss components also expose the difference:

| Arm | ListNet | Q Huber |
|---|---:|---:|
| Point-SNN | **3.088539** | 2.749476 |
| Reduced Hay | 3.109243 | 0.468917 |
| diagonal GRU | 3.148565 | **0.360210** |

Point learns candidate ordering but not calibrated teacher Q. GRU learns Q
calibration most strongly. Reduced Hay occupies the middle and obtains the
best total objective once auxiliary components are included.

## Mechanism recruitment

Reduced Hay internal activity changed materially with longer learning:

| Signal | 1k | 10k |
|---|---:|---:|
| workspace-active spike rate | 0.056255 | 0.080488 |
| soma spike rate | 0.151948 | 0.200289 |
| NMDA state mean | 0.066160 | 0.085202 |
| plateau state mean | 0.007798 | 0.011566 |
| plateau fraction above 0.05 | 0.000743 | 0.076048 |
| branch effective rank | 7.908400 | 7.895029 |

The plateau diagnostic rate increased by about `102x`, while branch-state
effective rank remained high rather than collapsing.

Exact-zero final-checkpoint ablations use
`delta loss = ablated - full`; positive means the mechanism helped:

| Ablation | 10k delta loss | Per-seed result |
|---|---:|---|
| plateau state off | `+0.009515 +/- 0.009055` | helpful in 2 seeds, neutral in 1 |
| apical state off | `+0.017961 +/- 0.006200` | helpful in all 3 |
| recurrent input off | `+0.016232 +/- 0.024230` | variable; strong in 1 seed |

At 1k, plateau removal slightly improved loss and recurrent removal was
neutral. The 10k result reverses that short-horizon interpretation:
Hay-derived mechanisms are gradually recruited, and 1k was too early to judge
their utility.

## Reference CPU cost

These are dense reference-BPTT training rates after warmup. They are not the
future analytic-VJP/barrierless hot path.

| Arm | Updates/s | Allocation/update |
|---|---:|---:|
| Point-SNN | 147.371 +/- 2.858 | 17.195 MB |
| Reduced Hay | 151.710 +/- 2.710 | 10.773 MB |
| diagonal GRU | **268.731 +/- 3.159** | **8.605 MB** |

GRU remains `1.77x` faster per training update than Reduced Hay. Consequently,
the equal-update composite crossover is not yet an equal-CPU-time victory.
An equal-wall-clock continuation must allow GRU proportionally more updates.

## Decision

The user's saturation hypothesis is supported but not fully proven:

1. GRU remains ahead through 5k.
2. GRU's improvement slows after 5k but does not stop.
3. Reduced Hay improves much faster from 5k to 10k.
4. Reduced Hay narrowly crosses mean composite loss and recruits plateau,
   apical and recurrent mechanisms.
5. GRU still leads top-1, pairwise and reference throughput.

The architecture should not be rejected based on the 1k result. The next
learning gate should extend the preserved 10k optimizer checkpoints to at
least 20k, report both equal-update and equal-wall-clock curves, and select
checkpoints by ranking metrics as well as composite loss.

## Artifacts

```text
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u10000_curve_seed1_20260730_190145\artifact
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u10000_curve_seed2_20260730_190700\artifact
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u10000_curve_seed3_20260730_191159\artifact
```

Each directory contains:

- `results.json` with all four held milestones and full traces;
- 12 milestone checkpoints, one per arm and milestone, including optimizer
  state;
- final compatibility checkpoints for all three arms.
