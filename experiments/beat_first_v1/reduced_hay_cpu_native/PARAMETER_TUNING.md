# Historical pre-shared-Q parameter tuning

> This file is an archival record of the structural sweep that selected ten
> cycles and recurrent fanout 48. Its output population, parameter count and
> optimizer contract predate the shared trainable IEEE Q cell and are not the
> current canonical model contract. See `README.md` for the executable source
> of truth.

The tuning objective is hard Tetris ranking quality subject to a sustained
training throughput floor of 20 updates/s.  All accuracy comparisons below use
the same fixed eight-state panel, model seed, batch size 8, candidate width 80,
and the local learner used by that sweep. A transient top-1 peak is not treated
as a pass.

## Configuration adopted by the historical sweep

- 480 high-dimensional Reduced Hay cells (30 blocks x 16 cells)
- 8 basal compartments plus one active apical compartment per cell
- 10 recurrent cycles
- recurrent fanout 48, split local/spatial/global = 16/16/16
- E/I split 36/12 and current/previous-cycle delay split 36/12
- hard output population 8 per channel, output fanout 16
- 20 barrierless workers
- AdamW learning rate 0.002
- learning-rate multiplier 0.5 after update 3,000
- recurrent multiplier 0.001 and sensory multiplier 0.1
- recurrent homeostasis and rewiring end at update 1,000
- output homeostasis ends at update 2,000 in the hard-state ladder
- historical parameter count: 65,277

The adopted 10k hard-eight result is:

| Update | Excess loss | Hard tie-aware top-1 | Updates/s |
|---:|---:|---:|---:|
| 4k | 0.190333 | 0.875 | 30.781 |
| 5k | 0.192207 | 0.875 | 30.952 |
| 6k | 0.164015 | 1.000 | 31.051 |
| 7k | 0.117982 | 0.875 | 31.106 |
| 8k | 0.114536 | 0.875 | 31.148 |
| 9k | 0.114119 | 0.875 | 31.170 |
| 10k | 0.194192 | 0.875 | 31.209 |

This is a material improvement, but it is not a completed hard-eight pass:
top-1 reached 1.0 at 6k but did not remain 1.0 through the final interval.

## Search coverage

Every `LocalLearningConfig` field was considered.  Fifty-three one-factor
arms were measured for 512 updates and eleven selected interactions for 2k.
The full tables are in `results/parameter_screen_512.tsv` and
`results/parameter_interactions_2k.tsv`.  Learning modes `disabled` and
`shadow` are diagnostics rather than accuracy settings and were not promoted
to production learning arms.

Representative structural results:

| Arm | Updates | Excess | Top-1 | Updates/s | Decision |
|---|---:|---:|---:|---:|---|
| cycles 6, fanout 32 | 2k | 0.475268 | 0.500 | 45.335 | too shallow |
| cycles 8, fanout 32 | 2k | 0.297144 | 0.750 | 42.242 | fast baseline |
| cycles 10, fanout 32 | 2k | 0.269927 | 0.750 | 30.875 | insufficient communication |
| cycles 12, fanout 32 | 2k | 0.411491 | 0.625 | 26.730 | recurrent degradation |
| cycles 10, fanout 48 | 2k | 0.266375 | 0.750 | 29.394 | retained |
| cycles 10, fanout 64 | 2k | 0.290885 | 0.375 | 29.298 | over-connected |
| 720 cells, 3 lanes | 2k | 0.272010 | 0.500 | 21.171 | third lane dormant |
| 12 basal compartments | 2k | 0.627941 | 0.500 | 21.474 | drive diluted |
| output population 10, homeostasis 64 | 2k | 0.239179 | 0.625 | 28.229 | later collapse |
| recurrent-only branch role 0.05 | 2k | 0.410868 | 0.625 | 29.546 | over-firing |

The population-10 arm deteriorated from excess 0.239 at 2k to 1.021 at 5k.
Continuous output homeostasis also destabilized population 8; ending output
homeostasis at 2k improved its 5k result from 0.268626/0.750 to
0.226209/0.875.

Optimizer schedule comparison at 10k, using cycles 10 and fanout 48:

| Late learning-rate policy | Final excess | Final top-1 | 6k-10k behavior |
|---|---:|---:|---|
| no decay | 0.437907 | 0.625 | repeated large shocks |
| 0.75 after 5k | 0.235485 | 0.750 | mean excess 0.294 |
| 0.50 after 5k | 0.116410 | 0.750 | mean excess 0.163 |
| 0.50 after 3k | 0.194192 | 0.875 | top-1 0.875 from 7k-10k |

The 3k schedule is selected because the project gate prioritizes sustained
hard top-1 as well as ListNet excess.  It produced the best late top-1
stability, despite the 5k schedule having the lowest single final excess.

Structural ablations with the selected 3k schedule establish that both added
cycles and communication capacity matter:

| Structure | Final excess | Final top-1 | Updates/s |
|---|---:|---:|---:|
| cycles 8, fanout 32 | 0.281565 | 0.625 | 39.041 |
| cycles 10, fanout 32 | 0.291405 | 0.625 | 32.183 |
| cycles 10, fanout 48 | 0.194192 | 0.875 | 31.209 |

Worker count was tested independently at 1k.  Results for 8/12/16/20 workers
were 18.980/23.090/24.088/27.827 updates/s with identical loss and top-1, so
20 workers was retained for the subsequent CPU implementation.

## Fixed rather than swept

- Batch size 8 and candidate width 80 are project constraints from the current
  comparison protocol.
- Thirty blocks are the exact 24 x 10 board tiling (three vertical bands per
  column), not an independent capacity knob.
- Four sensory basal compartments preserve the four board planes.  Extra
  recurrent-only compartments were tested through the 12-basal arm and
  rejected.
- Trainable cell parameters are optimized by the task loss; blindly sweeping
  all 46 per-cell initial raw values would confound architecture selection.
  Initial activity was instead tested through branch role, cell count, basal
  count, firing-rate controls, and sensory/recurrent multipliers.

## Evidence

- `results/parameter_screen_512.tsv`
- `results/parameter_interactions_2k.tsv`
- `results/candidate_lr_decay3k_x05_edge128_homeo_until2k_fanout48_cycles10_10k.log`
- `results/baseline_structure_lr_decay3k_x05_homeo_until2k_cycles8_fanout32_10k.log`
- `results/ablation_lr_decay3k_x05_homeo_until2k_cycles10_fanout32_10k.log`
- `results/final_regression_cycle10_fanout48.log`

All 23 tests in that sweep passed, including analytic/finite-difference cell credit,
fixed-arena allocation checks, topology golden data, checkpoint config
fingerprinting, local-credit causality, and serial/barrierless equivalence.
