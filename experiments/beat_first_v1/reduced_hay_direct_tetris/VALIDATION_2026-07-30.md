# Reduced Hay direct-Tetris validation — 2026-07-30

## Scope

This is the first held-teacher comparison of the direct-Tetris Reduced Hay
cell. It is an equal-update reference experiment, not a final equal-wall-clock
production benchmark.

- dataset: `teacher_v3`;
- training split: predefined `train`, candidate count at most 40;
- evaluation split: predefined `validation`, 64 held rows per seed;
- scratch updates: 1,000 per model and seed;
- independent model/schedule/panel seeds: 3;
- state batch: 1;
- learning rate: `2e-4`;
- weight decay: `1e-5`;
- Julia threads / BLAS threads: 1 / 1;
- loss: canonical 22-output
  ListNet/Q/death/quantile/geometry objective;
- trainer: warmed Zygote direct-BPTT reference.

Within each seed, Point-SNN, Reduced Hay and GRU received the exact same
training-row order and validation panel. The 64 validation rows were unique
within each seed. The frozen distilled 11-state arm was not run because no
qualified frozen artifact exists; its loader continued to fail closed.

These teacher-validation rows are not gameplay-validation or sealed seeds.

## Static budget

| Arm | Persistent states | Estimated scalar work | Parameters |
|---|---:|---:|---:|
| Point-SNN | 372 | 3,348 | 21,567 |
| frozen 11-state | 374 | 3,366 | unavailable |
| Reduced Hay | 368 | 3,520 | 18,863 |
| diagonal GRU | 360 | 4,320 | 5,894 |

The maximum/minimum state ratio is `1.038889`; the static operation-estimate
ratio is `1.290323`. Point and Reduced Hay expose the same 12-dimensional
block interface.

## Held-validation result

Values are mean +/- sample standard deviation over three independent seeds.
Lower composite loss is better; the other quality metrics are higher-is-better.

| Arm | Composite loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| Point-SNN | 4.770735 +/- 0.045503 | 0.119792 +/- 0.086056 | 0.857926 +/- 0.028394 | 0.403064 +/- 0.140624 |
| Reduced Hay | 4.684535 +/- 0.035661 | 0.104167 +/- 0.032526 | 0.869148 +/- 0.004289 | 0.508210 +/- 0.029732 |
| diagonal GRU | **4.540472 +/- 0.040379** | **0.119792 +/- 0.023868** | **0.882222 +/- 0.003951** | **0.617944 +/- 0.008113** |

Against Point-SNN, Reduced Hay changed the mean metrics by:

- composite loss: `-0.086200`;
- top-1: `-0.015625`;
- NDCG: `+0.011221`;
- pairwise: `+0.105147`.

Reduced Hay therefore reached a better mean ranking loss, NDCG and pairwise
score than Point-SNN, but did not improve top-1. The diagonal GRU remained
better than Reduced Hay on every aggregate quality metric.

## Reference-loop CPU measurements

Compilation was discarded by a separate warmup. These values measure the
dense reference training loop and must not be attributed to the future
allocation-free analytic-VJP/barrierless implementation.

| Arm | Updates/s | Allocation/update |
|---|---:|---:|
| Point-SNN | 132.976 +/- 3.138 | 17.213 MB |
| Reduced Hay | 137.843 +/- 0.986 | 10.587 MB |
| diagonal GRU | **233.675 +/- 2.689** | **8.605 MB** |

Reduced Hay was `1.037x` Point-SNN throughput and used about `38.5%` less
reference-loop allocation. GRU was `1.695x` Reduced Hay throughput and used
about `18.7%` less allocation. All three reference loops still allocated and
performed GC; this is not the production hot-path result.

## Internal activity

Reduced Hay mean activity after 1,000 updates:

| Signal | Mean +/- sample standard deviation |
|---|---:|
| workspace-active spike rate | 0.056255 +/- 0.005893 |
| soma spike rate | 0.151948 +/- 0.011416 |
| NMDA state mean | 0.066160 +/- 0.002138 |
| plateau state mean | 0.007798 +/- 0.000295 |
| plateau fraction above 0.05 | 0.000743 +/- 0.000408 |
| branch-state effective rank | 7.908400 +/- 0.418021 |

The continuous branch state did not collapse, and NMDA state was nonzero.
However, only about `0.074%` of plateau states crossed the diagnostic 0.05
threshold.

## Exact mechanism ablations

The final checkpoint was replayed on the same held panel while multiplying
one mechanism by exactly zero. `Delta loss = ablated - full`, so a positive
value means the mechanism helped.

| Ablation | Delta composite loss | Per-seed sign |
|---|---:|---|
| plateau state off | `-0.004517 +/- 0.004115` | removal improved all 3 seeds |
| apical state off | `+0.003041 +/- 0.001994` | removal hurt all 3 seeds |
| recurrent input off | `-0.000118 +/- 0.001190` | mixed, effectively zero |

The original diagnostic that only minimized the plateau gain logit was
discarded because the cell intentionally has a nonzero minimum plateau gain.
The recorded values above use the corrected exact-zero state/input
interventions in validation schema v2.

The apical path has a small but repeatable favorable effect. The current
plateau path is not useful after 1,000 updates, and the recurrent event graph
has no measurable contribution. A nonzero internal state is therefore not
being mistaken for demonstrated task utility.

## Decision

The current Reduced Hay cell is a better short-run ranking model than the
budget-matched Point-SNN on loss, NDCG and pairwise mean, but it is not yet the
CPU-model winner:

1. top-1 does not improve over Point-SNN;
2. GRU is better in quality, throughput and allocation;
3. plateau dynamics are almost inactive and slightly harmful;
4. the recurrent sparse graph is unused;
5. the frozen 11-state control is unavailable.

The result supports retaining multi-compartment continuous state as a research
arm, but it does not validate the complete Hay-derived mechanism set or justify
a superiority claim.

The next critical architecture step is to repair plateau utilization and
recurrent credit/use, then repeat this same 1k gate. Analytic VJP and
barrierless integration should only become the dominant effort after the cell
shows a task-quality or quality-per-CPU advantage over the GRU control.

## Artifacts and checks

The three checkpoint/result roots are:

```text
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u1000_v64_20260730_171751\artifact
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u1000_v64_seed2_20260730_172142\artifact
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\control_u1000_v64_seed3_20260730_172426\artifact
```

Each root contains the original checkpoint and `results_v2.json` with the
exact-zero ablations. The final source tests passed:

- 33/33 direct-model, finite-difference, canonical-cotangent, budget and
  exact-ablation checks;
- 6/6 allocation-free SoA/event-kernel checks.
