# Experiment ledger

The immutable evaluation contract is `configs/evaluation_protocol.toml`. Development
scores never count as G2 evidence. `1313/` is input-only and ignored by Git.

| ID | Hypothesis / change | Success metric | Time limit | Result | Decision / stop reason |
|---|---|---|---:|---|---|
| E000 | Reconstruct the 2024 checkpoint and matching engine | Finite inference; score scale agrees with historical distribution | 2 h | 20,787,454 parameters; seed 5742 reached 15,900/250 | Adopt as G0 baseline |
| E001 | OpenVINO 2026.2.1 can reproduce Lux and accelerate inference | CPU max abs error <1e-4; accelerator <1e-2; end-to-end speedup | 2 h | CPU 5.1e-6, NPU 6.9e-4; 25-step inference 53.2 s → 6.82 s | Adopt NPU for Actor inference |
| E002 | Frozen legacy representation + TD-trained value head improves quickly | Fixed development mean above initial head | 20 min | Initial [15400,16000]; episode 4 [15700,15400]; episode 8 [15600,14400] | Reject; retained initial checkpoint |
| E003 | Five-/ten-piece beam opening finds a perfect clear | Exact empty board with score bonus | 3 min | Width 1000 depth 5 and width 2000 depth 10 found none | Stop; widening has poor expected value |
| E004 | NEXT4/5 score ensemble reduces NEXT5 noise | Full 250-step score above NEXT5 baseline | 5 min | Seed 5742: 14,200 vs 15,900 | Reject |
| E005 | Top-2 Bellman reranking improves the final agent | Improvement on multiple development seeds; compute recorded | 12 min | Seed 5742: 17,000 vs 15,900 (+1,100); seed 5743: 15,600 vs 15,600 (0). About 2.0x candidate evaluations and 2.7x inference time on seed 5743 | Do not promote: mean paired gain is only +550 over two development seeds and one difference is zero; system candidate only, not model improvement; no G2 claim |
| E006 | Gate Bellman reranking by old-Q margin | 100-step score >= ungated 6,300 and lower compute | 2 min | Margin 0.05 scored 5,200 | Reject |
| E007 | Pairwise imitation of a hand-tuned beam heuristic yields a usable standalone MLP policy | Development game score approaches the teacher before RL | 20 min | Five evaluation seeds: [900, 2400, 400, 1100, 400], mean 1,040 despite finite ranking loss | Reject; pairwise accuracy did not preserve full action-set decisions or long-horizon stability |
| E008 | A zero-initialized learned residual can safely improve the hand-tuned heuristic | Fixed five-seed median/mean/worst case exceeds step-0 policy | 20 min | Step 0 scores [11400, 9800, 900, 5400, 10100], median 9,800; every trained snapshot through 800 updates was worse, best correctly remained step 0 | Reject this teacher/objective; evidence supports old-model listwise distillation instead, not a general rejection of end-to-end learning |
| E009 | A 120-state old-model listwise smoke set is enough to establish a compact end-to-end student signal | Held-development top-1 and an unseen development game both improve over random/initial | 5 min | 296,245 parameters, 20 distillation updates: held 40-state top-1 fell 7.5% -> 2.5% (random 2.25%); seed 5745 ended after 23 pieces at score 0 | Reject checkpoint and dataset scale; pipeline is executable, but no learning signal worthy of promotion |
| E010 | Standalone Enzyme accelerates the fixed-shape compact learner while matching Zygote | Gradient cosine near 1, max error <1e-5, and compile-inclusive time-to-100 below Zygote | 10 min | Batch16/133,125p correctness probe: cosine 0.99999999999997, max error 7.1e-8. Under the same concurrent load, Zygote 15.16 update/s and 35 MB/update versus Enzyme 2.08 update/s and 189 MB/update | Reject standalone Enzyme path: numerically correct but about 7.3x slower and 5.4x more allocation; absolute timing is contaminated, direction is decisive |
| E011 | Top-2 one-step Bellman decisions provide a non-degenerate supervised correction signal | At least 100 finite states and non-zero/non-saturated top-2 flip rate | 10 min | 180 states on training-only seeds 72001:72003; 29 flips (16.1%); all three episodes 11.7%–18.3%; 218.9 s wall | Adopt only as residual-gate training data; no game-performance claim |
| E012 | A zero-initialized 16,065-parameter residual gate distills Bellman top-2 flips without lookahead | Held teacher balanced accuracy improves; then unseen development game does not regress | 10 min | Held episode balanced accuracy 0.500→0.706, but uncalibrated seed5750/100 scored 4,800 vs old 5,800 and flipped 28%. Holdout-only calibration chose threshold .272, 1.7% flips, tiny positive proxy gain | Reject uncalibrated gate; one preregistered conservative confirmation on a fresh development seed remains before closing branch |
| E013 | Correct 74-action fixed shape makes compact listwise distillation fast while preserving signal | Reproduce held metrics with finite 300 updates and major wall-time reduction | 5 min | 165,051p: top-1 .022→.222, MRR .104→.366, correlation -.026→.488, CE 4.222→3.776; 45.9 s wall, ~14.9 steady update/s. Prior action-dimension bug padded 74 to 2,000 and took 558 s | Adopt corrected pipeline; compare one learning rate, then promote only the winner to 3,000 updates |
| E014 | Reactant+EnzymeMLIR beats Zygote on compile-inclusive time-to-100 | Matching loss trajectory and projected time-to-100 below clean Zygote | 10 min | Five-step loss trajectory matched Zygote; first full compiled update took 65.34 s. Clean Zygote completed 100 updates in 19.05 s (projected first+99 median 17.46 s). Reactant steady timer was invalid due asynchronous host synchronization | Reject for time-to-target; first compile alone is 3.7x the full Zygote 100-update budget. Revisit only for much longer fixed-shape runs with a synchronized benchmark |
| E015 | Raising compact ListNet learning rate from 3e-4 to 1e-3 improves old-teacher imitation at the same 300 updates | All offline ranking metrics exceed E013 on the identical split/schedule | 2 min | top-1 .370 vs .222, MRR .527 vs .366, correlation .657 vs .488, CE 3.618 vs 3.776; 46.2 s wall | Adopt 1e-3; reject 3e-4 |
| E016 | Extending the winning compact student to 3,000 updates produces a viable no-lookahead game policy | Offline ranking improves and a dataset-unused development game remains stable | 5 min | Offline improved to top-1 .484, MRR .645, correlation .851, CE 3.422 in 223.6 s, but seed5750 ended after 34/50 pieces at score 0 (old score at 50 pieces: 2,100) | Reject checkpoint as a game/model candidate; pure on-teacher-trajectory distillation suffers compounding distribution shift despite strong offline metrics |
| E017 | Sharpening the teacher softmax from T=1.0 to T=0.25 fixes top-action imitation quickly | At 300 updates, top-1 >= .370 and MRR > .527 | 2 min | top-1 .346, MRR .511, CE 3.738, all below T=1.0 reference | Reject and stop temperature sweep; next learning mechanism is DAgger after evaluation freeze |
| E018 | One DAgger round on C11b-visited states repairs the compact policy's early covariate-shift failure | Original-development top-1 >= .45, then both C13 and canonical old baseline complete the one-time seed5751/50-piece smoke | 25 min total | 660 DAgger states / 27,060 labels (30.6% of 2,160-row train aggregate). Preregistered snapshots: update0 top-1/MRR .484/.645; update250 .484/.653; update500 .478/.633. Update250 selected. Seed5751: C13 completed 50, score1,100; old completed50, score2,300; difference -1,200 | **C13-survival-pass only.** Covariate-shift survival gate passed, but score did not improve. Freeze update250 for the separately preregistered three-seed development strength screen; no G2/model-improvement claim. A mistakenly configured max3,000 run was stopped at update450 and excluded before the valid max500 rerun |

## E018 provenance and replay

- Git commit: `0480ad6e20db203ff6673289579bbed68faa82b4`
- Source SHA-256: `dddb0483effad0140f4f34d8269703e28e0fbbfc605e6f6baf9e7bfdfda295e3`
- Manifest SHA-256: `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`
- DAgger dataset: `D:\tetris-paper-plus\datasets\learning\dagger_c13_round1_c11b_5742_5747.jld2`, SHA-256 `ea90138cc340766486a9502e1cd9c8df6a3408c6f431028014c35da30443e05a`
- Aggregate dataset: `D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2`, SHA-256 `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba`
- Selected update250 checkpoint: `D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2`, SHA-256 `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`
- Candidate smoke JSON: `D:\tetris-paper-plus\runs\learning\compact_eval_seed5751_next5_steps50.json`, SHA-256 `8772162009abbf481a6e00e747a47b76975d6632f8f90864b744953ff7cddbd9`
- Baseline smoke JSON: `runs/openvino_checkpoint_npu_seed5751_next5_steps50.json`, SHA-256 `367efbc59717c78eb063f5dca51d95cde991a67ee8e2aa91c434fcda6ba87cdb`

Exact complete commands and configuration are embedded in the DAgger dataset,
aggregate dataset, training checkpoint, and adjacent JSON summaries. Minimal
replay commands, after setting the same `SOURCE_FINGERPRINT_PATH` and
`EXPERIMENT_COMMAND`, are:

```powershell
julia --project=. --threads=20 experiments\learning\generate_teacher_dataset.jl
julia --project=. experiments\learning\merge_teacher_datasets.jl
julia --project=. --threads=20 experiments\learning\train_distillation.jl
$env:COMPACT_EVAL_SEEDS='5751'; $env:COMPACT_EVAL_STEPS='50'; julia --project=. --threads=20 scripts\evaluate_compact_checkpoint.jl
$env:LEGACY_EVAL_SEED='5751'; $env:LEGACY_EVAL_STEPS='50'; $env:OPENVINO_DEVICE='NPU'; julia --project=. --threads=20 scripts\evaluate_openvino_checkpoint.jl
```

## Required record for new experiments

Every promoted experiment must record: experiment ID; source hash; Julia and Manifest
hash; complete config; training and environment seeds; architecture and parameter
count; backend; compile time; steady throughput; score metrics; checkpoint and log paths;
and termination reason.
