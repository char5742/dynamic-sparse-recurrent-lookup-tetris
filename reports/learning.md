# Learning track status

Updated: 2026-07-17 23:28 JST

Only the development seeds in `configs/evaluation_protocol.toml` were used.
Validation below means an offline split inside that development set; the sealed
test seeds were never executed or loaded.

## Teacher dataset C10-data

- Hypothesis: complete candidate-set supervision from the strong 1313 teacher
  is the shortest route to a useful end-to-end compact initialization.
- Mechanism: preserve every candidate and its exact OpenVINO teacher Q, then
  standardize teacher logits independently per state.
- Change: no game/scoring/search change; data only.
- Success: at least 1,000 finite full-candidate states with no test seed.
- Time limit / stop: 15 minutes; stop on a non-finite teacher value, candidate
  overflow, or any seed outside the declared development set.
- Result: **passed**. Seeds 5742--5749, 2,000 states, 86,691 candidates, all
  eight trajectories completed 250 pieces. Scores were 15,900, 15,600, 15,200,
  14,000, 16,200, 15,100, 15,300, and 12,400.
- Cost: 623.554 s wall; 91.415 s candidate generation and 495.030 s NPU
  inference. OpenVINO 2026.2.1, NPU static batch 16, NEXT=5.
- Artifact: `D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2`
  (64,114,637 bytes; SHA-256
  `E0D79E38DAEBB667BD8C248F5F64B8E5241A4ED56A29D31FFB4EE41BD0C26B8D`).

## C10 listwise distillation (first signal)

- Hypothesis: a 165k-parameter CNN trained end-to-end on complete candidate
  lists can recover materially more teacher ranking than random initialization.
- Mechanism: teacher and student logits are standardized within each state,
  then optimized with masked soft-target ListNet cross entropy. This removes
  arbitrary per-state Q offset/scale while retaining the entire ordering.
- Model: 8 channels, one residual block, two projected spatial channels,
  separately encoded 42-value HOLD/NEXT queue, 165,051 parameters. All layers
  train; no frozen 1313 feature extractor.
- Split: episodes 1--6 / seeds 5742--5747 train (1,500 states); episodes 7--8 /
  seeds 5748--5749 offline validation (500 states).
- Optimizer: Lux 1.31.4 + Zygote, AdamW, lr=3e-4, state batch 4, 300 updates,
  TD updates disabled.
- Time limit / stop: 10 minutes; reject non-finite updates or no validation rank
  improvement. The command wrapper timed out at 604 s, but checkpoint and JSON
  had completed and were verified after exit.

| Offline development metric | Initial | After 300 | Delta |
|---|---:|---:|---:|
| top-1 agreement | 0.026 | **0.222** | +0.196 |
| mean reciprocal rank | 0.1057 | **0.3664** | +0.2607 |
| Pearson Q correlation | -0.0263 | **0.4876** | +0.5139 |
| listwise cross entropy | 4.2224 | **3.7755** | -0.4469 |
| random top-1 reference | 0.0255 | 0.0255 | -- |

The learning signal is a clear offline promotion, but this checkpoint is not a
model-performance claim yet. A dimension-index bug made this first run process
2,000 padded action slots instead of the true fixed maximum of 74. Valid
candidates, targets, and masks were preserved, so the ranking result is usable;
the recorded `effective_max_actions=2000` is diagnostic rather than real. The
bug is fixed (`placements` action dimension is 4, state dimension is 5), and a
clean 74-action reproduction is required before a game evaluation.

- Checkpoint: `D:\tetris-paper-plus\checkpoints\learning\C10_listwise_165k_lr3e4_300.jld2`
  (SHA-256 `F054274AFE61ADA9991A2BD9750F93FE527C14B8A1E6F7DEC0BFEEEC4A3E4B9B`).
- Summary: adjacent `C10_listwise_165k_lr3e4_300.json` (non-finite optional TD
  values are serialized as JSON `null`, preventing the previous zero-byte bug).

## Corrected reproduction and learning-rate screen

C10b corrected the action/state dimension and batched eight validation states
per forward pass. It reproduced the C10 signal almost exactly in 45.903 s:
top-1 0.022 -> 0.222, MRR 0.1037 -> 0.3663, correlation -0.0263 ->
0.4876, and CE 4.2224 -> 3.7755. Its true fixed maximum is 74 actions. After
the first compiled update, the remaining 299 updates took about 20.1 seconds
(about 14.9 updates/s).

Changing only AdamW learning rate from 3e-4 to 1e-3 (C11a) improved the same
300-update validation to top-1 **0.370**, MRR **0.5272**, correlation **0.6568**,
and CE **3.6176** in 46.243 s. The 3e-4 setting was rejected.

Extending the winning setting from 300 to 3,000 updates (C11b) improved all
offline metrics rather than overfitting:

| Metric | Random/initial | C11a 300 | C11b 3,000 |
|---|---:|---:|---:|
| top-1 agreement | 0.022 | 0.370 | **0.484** |
| mean reciprocal rank | 0.1037 | 0.5272 | **0.6452** |
| Pearson Q correlation | -0.0263 | 0.6568 | **0.8513** |
| listwise cross entropy | 4.2224 | 3.6176 | **3.4220** |

C11b took 223.632 s and is stored at
`D:\tetris-paper-plus\checkpoints\learning\C11b_listwise_165k_lr1e3_3000_fixed74.jld2`.
It is the best offline checkpoint, but **not a promoted game model**.

## Fixed-budget game rejection

C11b was evaluated with the same candidate generator, NEXT=5, one network call
per decision, and no lookahead on development seed 5750, which was absent from
the teacher dataset. It game-overed after 34 pieces with score 0. The run used
34 calls / 1,450 candidate evaluations; compact inference took 0.427 s and
candidate generation 7.317 s. Artifact:
`D:\tetris-paper-plus\runs\learning\compact_eval_seed5750_next5.json`.

This is direct evidence of covariate-shift / compounding-error failure: high
offline teacher ranking is insufficient for a standalone policy. C11b is
rejected as a game-strength candidate and no claim of beating the old model is
made.

## C12 low-temperature screen

Hypothesis: the temperature-1 teacher distribution was too diffuse, so reducing
the ListNet temperature to 0.25 might improve top-action fidelity. The
pre-registered 300-update gate was top-1 >= C11a's 0.370 and MRR > 0.5272.
C12 reached top-1 0.346 and MRR 0.5113 (CE 3.7381), failing both gates. The
temperature sweep was stopped and the checkpoint rejected.

## C13 DAgger round 1

- Hypothesis: C11b fails mainly because its own rollout leaves the old-teacher
  state distribution; labels on student-visited states should reduce this
  compounding error faster than more on-policy teacher training.
- Mechanism: run C11b on training seeds 5742--5747, record every complete
  candidate set it visits, label all candidates with the unchanged 1313 NPU
  teacher, and mix those rows with the original 1,500 training states. Original
  seeds 5748--5749 remain the unchanged 500-state offline validation split.
- Change: dataset distribution only. The 165,051-parameter model, standardized
  listwise loss, lr=1e-3, fixed candidate budget, rules, and scoring stay fixed.
- Success gate: finite full-candidate data; no original-validation regression
  below top-1 0.45; then survive the 50-piece fixed-budget smoke on the next
  unused development seed before any full game.
- Time limit: 15 minutes teacher relabeling plus 5 minutes training. Stop on
  non-finite values, candidate overflow, validation collapse, or another early
  fixed-budget game-over. No further temperature sweep follows failure.
- Frozen source: commit `0480ad6e20db203ff6673289579bbed68faa82b4`,
  source SHA-256 `dddb0483effad0140f4f34d8269703e28e0fbbfc605e6f6baf9e7bfdfda295e3`,
  Manifest SHA-256 `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`.

### Realized DAgger data

C11b rollout produced 660 states / 27,060 complete candidate labels. Per-seed
terminal results were:

| Seed | Pieces | Score |
|---:|---:|---:|
| 5742 | 105 | 1,800 |
| 5743 | 250 | 6,200 |
| 5744 | 41 | 0 |
| 5745 | 121 | 1,400 |
| 5746 | 57 | 600 |
| 5747 | 86 | 1,100 |

The NPU teacher run took 212.813 s (156.873 s inference, 36.639 s candidate
generation). DAgger fraction was `660 / (1500 + 660) = 0.3056`, inside the
pre-registered interpretable 0.15--0.50 range. Artifacts:

- `D:\tetris-paper-plus\datasets\learning\dagger_c13_round1_c11b_5742_5747.jld2`
  SHA-256 `ea90138cc340766486a9502e1cd9c8df6a3408c6f431028014c35da30443e05a`.
- Merged 2,660-state dataset (2,160 train + unchanged 500 development):
  `D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2`,
  SHA-256 `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba`.

### Preregistered 500-update warm start

One accidentally launched max-3,000 configuration was stopped at update 450,
before its configured first non-preregistered snapshot, and is excluded from
promotion. The clean run was re-executed from C11b with exactly the registered
snapshots 0, 250, and 500:

| Snapshot | Top-1 | MRR | Correlation | CE | Selected |
|---:|---:|---:|---:|---:|---|
| 0 | 0.484 | 0.6452 | 0.8513 | 3.4220 | no |
| 250 | **0.484** | **0.6526** | 0.8491 | 3.4210 | **yes** |
| 500 | 0.478 | 0.6333 | 0.8556 | 3.4178 | no |

Selection is the preregistered lexicographic key (top-1, MRR, then lower CE),
so update 250 wins and passes the top-1 >= 0.45 gate. Training wall time was
60.697 s. Best checkpoint:

`D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2`

SHA-256 `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`.
Its embedded metadata binds the source, merged dataset, C11b initializer,
complete command/config, and update-250 metrics.

### One-time paired 50-piece smoke

Development seed 5751 was used once, with NEXT=5, one logical call per
decision, no lookahead, and the same stable candidate generator.

| Agent | Completed | Score | Candidates | Logical / physical calls | Inference | Wall |
|---|---:|---:|---:|---:|---:|---:|
| C13 update 250 | 50/50 | 1,100 | 2,051 | 50 / 50 | 0.477 s | 9.352 s |
| Old 1313 OpenVINO NPU | 50/50 | 2,300 | 2,151 | 50 / 165 | 13.191 s | 24.855 s |

Both survived, therefore the result is **C13-survival-pass**. The paired score
difference is -1,200, so this is not a strength improvement and not evidence of
beating the old model. Candidate and baseline run hashes are respectively
`8772162009abbf481a6e00e747a47b76975d6632f8f90864b744953ff7cddbd9` and
`367efbc59717c78eb063f5dca51d95cde991a67ee8e2aa91c434fcda6ba87cdb`.

All summaries are appended to `experiments/learning/ledger.jsonl`. Sealed test
seeds remain unused.
