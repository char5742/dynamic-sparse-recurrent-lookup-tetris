# Beat-first experiment ledger

This ledger records both successful and failed runs. Teacher-held rows are not
game-validation seeds; game validation `8001:8008` and sealed seeds
`91001:91032` remain unopened.

## 2026-07-20 — spatial episodic recurrent LookupFFN, 20,000 updates

- Repair: added recurrent physical local-8 cell attention, shared learned
  Q/K/V/O projections, relative spatial bias, mandatory candidate support,
  persistent input residuals, memory BPTT, and exact active-only token VJPs.
- Correctness: strict serial/barrierless smoke passed from the immutable
  update-2,000 checkpoint. Output, loss, and raw VJP were exact; maximum
  parameter-gradient difference was `1.84774399e-6`, and maximum
  post-optimizer parameter difference was `1.84401870e-7`. All routing, RNG,
  sparse-row, optimizer-clock, and sampler witnesses were exact.
- Fixed-batch gate: real-teacher four-state memorization passed. ListNet
  `3.54684→0.99465`, KL `2.56241→0.010223`, old-Q
  `1.88590→0.021857`, top-1 `0→1`, NDCG `0.70592→1`.
- Scheduler repair: typed dense reduction/scaling and single-pass ordered
  `cell_bias` replay reduced allocation `156.6→5.62 MB/update`. Warm measured
  throughput was `19.81 updates/s`, with zero measured GC time.
- Production: resumed update 2,000 and reached update 20,000. The 18,000-update
  resume segment took `862.323 s` at `20.8738 updates/s`.
- Final checkpoint: 253,663,125 bytes, SHA-256
  `1fc05d63154fc73e5d60367c2b19d63116a975b0a3a772899b7fd0ca382db28e`.
- Interpretation: architecture learning and CPU throughput gates pass. The run
  was training-only; no held top-1/NDCG or PreAct superiority claim is made.

## 2026-07-19 — final-cap paired CPU benchmark (`sparse_paired_cpu_benchmark_20260719_r4`)

- Hypothesis: input-dependent sparse execution materially reduces CPU work at
  equal stored parameter count.
- Models: 1L k64, 3L k64 `(24,20,20)`, dense 1L twin; each 19,924,022 stored
  parameters; synthetic paired corpus only.
- Frozen route caps: scored rows `(384,640,640)`, route-key ceiling 425,984
  bytes. 3L k64 active parameters 31,934 and forward MACs 31,912.
- Result: GC-clean authoritative benchmark PASS. p50 forward latency was
  0.1530 ms / 0.1516 ms / 3.6792 ms; p50 full-training latency was
  0.4058 ms / 1.8397 ms / 31.5194 ms for 1L / 3L / dense respectively.
- Artifact: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_paired_cpu_benchmark_20260719_r4\stdout.log`,
  SHA-256 `b37944f8a3c802fa151fc3a8ee9efdb2474b38b5f7c30a86b99b928f78b0f837`.
- Interpretation: CPU engineering evidence only; no strength claim.

## 2026-07-19 — 3L k64 teacher signal, 1,000 updates (`sparse_3l_teacher_signal_1000_v1`)

- Hypothesis: the positive 100-update 3L ranking trend persists under a
  bounded extension.
- Bound: 1,000 updates, 32 held teacher states, one Julia/BLAS thread.
- Result: PASS. Held NDCG `0.798537→0.815667`, pairwise
  `0.486630→0.507503`, composite loss `7.246769→6.910283`, top-1 unchanged
  at `0.03125`. All gradients/metrics finite; no route-cap failure.
- Checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_1000_v1\sparse_3l_20260719T122430\checkpoints\checkpoint_000001000.jls`,
  248,672,483 bytes, SHA-256
  `096090ee6c08a72218e8bc9551de05829cf696804c3959174efc2359aed447d4`.
- Interpretation: promoted to a larger teacher-ranking screen only.

## 2026-07-19 — 3L k64 teacher signal, planned 5,000 updates (`sparse_3l_teacher_signal_5000_v1`)

- Hypothesis: ranking gains become measurable in top-1 on a 512-state held
  panel by update 5,000.
- Time/stop bound: 5,000 updates, 15 minutes, fail closed on route/numeric
  errors. Training stopped authoritatively after checkpoint update 2,000.
- Ranking signal at update 2,000: held top-1 `0.019531→0.042969`, NDCG
  `0.803217→0.837218`, pairwise `0.477261→0.526337`, composite loss
  `7.206525→6.569303`.
- Failure: a layer-2 query produced a 641st distinct WTA candidate and hit the
  frozen exact-score cap 640. The cap was not raised and no dense fallback was
  used. Checkpoint analysis found stable but long-tailed WTA buckets caused by
  overlapping coordinate samples inside each K=4 table, not checkpoint/index
  corruption.
- Last valid checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_5000_v1\sparse_3l_20260719T123823\checkpoints\checkpoint_000002000.jls`,
  248,668,439 bytes, SHA-256
  `0da5407c1a45fda37c34c96d8a134eab7b03766263498312750ccde4554bf5b4`.
- Decision: PIVOT routing only. Preserve active widths, total parameters,
  active-only backward, optimizer, and caps; add an explicit bounded
  collision-count prefilter before exact-dot reranking, then rebaseline.
- Interpretation: teacher-ranking evidence only; checkpoint is not eligible
  for game evaluation.

## 2026-07-19 — 3L k64 CPF1 routing pivot, 2,500 updates (`sparse_3l_teacher_signal_cpf_2500_v1`)

- Hypothesis: a collision-count-descending, stable-ID-ascending shortlist can
  bound exact-dot work without raising either hard cap or erasing the positive
  teacher-ranking trajectory.
- Frozen policy: `wta-collision-count-stable-id-prefilter-v1`; checkpoint and
  training-state formats were advanced to version 2 and bind this policy.
  Bucket-link traversal remains independently capped; training probes reserve
  their exact-score headroom before prefiltering.
- Bound: fresh k64 run, 2,500 updates, 512 episode-separated held teacher
  states, one Julia thread and one BLAS thread. No game validation or sealed
  seed was opened.
- Result: `PASS_RECOVERED_POST_GATE`. Held NDCG
  `0.803217→0.843474` (`+0.040257`), pairwise
  `0.477261→0.537651` (`+0.060391`), top-1
  `0.019531→0.048828` (`+0.029297`), and composite loss
  `7.206525→6.439772` (fraction `0.893603`). All recorded metrics were finite.
- Routing witness: cumulative maximum retrieved rows `(320,646,554)`.
  Layer 2 crossed the reserved 638-row exact-dot budget once; the prefilter
  dropped 8 rows. No bucket-entry or post-prefilter exact-score cap failed.
- Checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_cpf_2500_v1\sparse_3l_20260719T131935\checkpoints\checkpoint_000002500.jls`,
  248,670,913 bytes, SHA-256
  `331571cfb2f94845d6134030d01b50ab81d5b10344f221ec9aced7a9adfc228f`.
- Metrics SHA-256:
  `ff685ea03abdefd9dc6da0fe35152b35573903e4130faabb4074ceae1905c0c3`.
- Recovery note: Julia exited zero and published update 2,500 plus its
  checkpoint, but the first post-gate invocation failed on a PowerShell
  `FileInfo.Parent` bug. No new Julia was launched. The corrected, scoped
  recovery gate re-read and hashed the existing artifacts and wrote
  `post_gate_recovery_status.json`; the original failed controller status is
  preserved rather than rewritten.
- Interpretation: the routing pivot and bounded teacher-ranking screen pass.
  Validation top-1 remains far below the `0.80` game-evaluation eligibility
  gate, so this checkpoint is not a strength result and is not game-evaluated.

## 2026-07-19 — 3L k64 CPF1 teacher signal, 5,000 updates (`sparse_3l_teacher_signal_cpf_5000_v1`)

- Hypothesis: the cap-safe 2,500-update signal continues through the original
  frozen 5,000-update teacher-ranking gate.
- Bound: fresh same-seed k64 run, 5,000 updates, 512 held teacher states, one
  Julia/BLAS thread, checkpoint every 1,000 updates. No game seed was opened.
- Result: PASS. Held NDCG `0.803217→0.868524` (`+0.065307`), pairwise
  `0.477261→0.593774` (`+0.116513`), top-1 `0.019531→0.082031`
  (`+0.062500`), and composite loss `7.206525→5.754268` (fraction
  `0.798480`). The original midpoint, loss-component, Q-spread, margin,
  final-three rebound, bank-coverage, finite-value, and cap gates all passed.
- Routing witness: cumulative maximum retrieved rows `(349,655,554)`;
  prefilter activated twice in layer 2 and dropped 25 rows total. The
  independent bucket cap and reserved exact-score cap remained intact.
- Checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_cpf_5000_v1\sparse_3l_20260719T132818\checkpoints\checkpoint_000005000.jls`,
  248,679,063 bytes, SHA-256
  `5dc7e5d9471a6c452b13e1a86e5ec629f9c13634d4652e3fda7f511b85256efa`.
- Metrics SHA-256:
  `2dc77cd2f612edf819f5c987c7409564314212aac59090027e28d82469b92ab9`.
  Gate-status SHA-256:
  `e6d77325a472a0fb4e09e074a41e7b9ca20a2cd66a1fa80a12db16e8e6037e1d`.
- Interpretation: promoted for sparse active-width comparison only. Final
  validation top-1 `0.08203125` remains below the `0.80` game-evaluation gate.

## 2026-07-19 — 3L k128 CPF1 active-width screen, 2,500 updates

- Hypothesis: doubling active width at fixed 19,924,022 stored parameters
  improves paired held ranking enough to justify its added active compute.
- Bound: same seed/split/dataset and update count as the passed k64 CPF1
  2,500 arm; active counts `(48,40,40)`, 58,214 active parameters, 141,520
  inclusive training MACs per candidate route contract.
- Result: PASS against the SHA-pinned k64 metrics. Final held top-1 was
  `0.0703125` versus k64 `0.048828125`; NDCG `0.844093` versus `0.843474`;
  pairwise `0.540709` versus `0.537651`; composite loss `6.333082` versus
  `6.439772`.
- Cost: 32.445 updates/s and 1,416.05 candidates/s versus k64 45.697
  updates/s and 1,994.42 candidates/s. The top-1 gain is paired with about a
  29% update-throughput reduction.
- Routing witness: maximum retrieved rows `(320,660,564)`; prefilter activated
  once and dropped 25 rows without a hard-cap failure.
- Checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\checkpoints\checkpoint_000002500.jls`,
  248,673,984 bytes, SHA-256
  `de9aac395fc9406f2a3c77de4fa2408ada62716155d1d1915a1aaedb62670b85`.
- Metrics SHA-256:
  `b6298c1ba512414782309e69e018f35a698f28821fe5a9681ec1acc127c9bff6`.
  Gate-status SHA-256:
  `8a8509fb5a5a71a0d718cad840a5336a92f5ccb0f2756fa58839fbb6c9fed207`.
- Interpretation: k128 is promoted over k64 for the final k256 short screen;
  neither checkpoint is eligible for game evaluation.

## 2026-07-19 — 3L k256 CPF1 active-width screen, 2,500 updates

- Hypothesis: doubling active width again materially improves the paired k128
  teacher-ranking result despite lower CPU throughput.
- Bound: same seed/split/dataset/update panel; active counts `(96,80,80)`,
  110,774 active parameters and 267,552 inclusive training MACs. The contract
  also required at least one real CPF1 overflow activation.
- Contract result: FAIL CLOSED. Training exited zero at update 2,500 and all
  paired strength margins were exceeded, but no route crossed the reserved
  exact-dot budget (`prefilter_activation_routes=0`). The preregistered
  overflow-witness condition therefore failed; no rescue run is authorized.
- Informational held metrics: top-1 `0.083984`, NDCG `0.855030`, pairwise
  `0.555905`, composite loss `6.179914`. These are stronger than k128 at the
  same update, but do not convert this failed contract into a formal PASS.
- Cost: 20.017 updates/s and 873.63 candidates/s, versus k128 32.445 and
  1,416.05. Maximum retrieved rows were `(318,604,554)`.
- Last checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k256_teacher_signal_cpf_2500_v1\sparse_3l_20260719T135031\checkpoints\checkpoint_000002500.jls`,
  248,679,788 bytes, SHA-256
  `31e2256fb32831a2c50af26688cb0c050448b1121611bcc13e94f0a3ac81474a`.
  Metrics SHA-256:
  `4188928ecb4e90e475dc6224082bf4f10bfe2c538a3e05b39fac9f376aca952d`.
- Decision: adopt k128 for the next long teacher screen. k256's early ranking
  signal is preserved for a future wall-clock-normalized study, not promoted
  now.

## 2026-07-19 — 3L k128 CPF1 long teacher screen, 20,000 updates

- Hypothesis: the promoted k128 dynamic-sparse model will retain its short-run
  teacher-ranking advantage through a 20,000-update screen while preserving
  finite Q spread, useful action separation, cap-safe routing, and active-only
  sparse training.
- Bound: fresh run from the frozen teacher-v3 split, active counts
  `(48,40,40)`, 19,924,022 stored parameters, 58,214 active parameters and
  141,520 inclusive training MACs per route contract. Evaluations were exactly
  `0:2000:20000`; checkpoints were written at 10,000 and 20,000. No game
  validation or sealed seed was opened.
- Training result: the learner completed all 20,000 updates in 534.25 seconds
  of measured end-to-end training time (37.436 updates/s, 1,631.66
  candidates/s). Held top-1 rose `0.029297→0.228516`, NDCG
  `0.795182→0.937424`, pairwise accuracy `0.465374→0.740755`, and composite
  loss fell `7.211499→3.518782`. The preregistered 10,000-update midpoint and
  final ranking/loss thresholds were exceeded.
- Contract result: FAIL CLOSED. Final held Q standard deviation remained
  finite at `0.217046`, but action margin fell to `0.011132`, below the frozen
  `0.02` floor. The post-gate therefore stopped with
  `CPF1 k128 held action margin collapsed`; no `gate_status.json` was promoted
  and the threshold is not retrospectively relaxed.
- Sparse execution witness: all three banks reached full ever-updated coverage;
  the router remained cap-safe. Maximum retrieved rows were `(356,695,626)`;
  the collision prefilter activated three times and dropped 103 layer-2 rows.
  Per-route inference remained 58,192 active edges plus 216 sketch
  accumulates, rather than evaluating the full 19.9M bank.
- Final checkpoint: `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_20000_v1\sparse_3l_20260719T135857\checkpoints\checkpoint_000020000.jls`,
  248,680,789 bytes, SHA-256
  `0d642c446d89e5d06c482f4c260daad1f60b5a854055093175d17891c3136f3c`.
  The 10,000 checkpoint SHA-256 is
  `44c2fd48a189d1b269f6a73b524f42c5a184b4e92a5e6bc9cbe9018e42087c08`.
  Metrics SHA-256:
  `8ed7ecba6eb01e916ce85c6b7164628b3b0121696e00004c6e6d8bbb540a012d`.
- Interpretation: this is strong teacher-ranking and sparse-training evidence,
  not game-strength evidence. Top-1 remains below the frozen `0.80` game gate,
  and the final margin failure forbids promotion or game evaluation. The
  checkpoint is retained for analysis only pending an independently reviewed,
  preregistered next hypothesis.

## 2026-07-19 — 3L k128 margin-weight pivot, 20,000 updates

- Hypothesis: changing only the sparse teacher objective's effective margin
  weight from `0.15` to `1.0` will preserve ranking learning while preventing
  the parent run's late action-margin collapse. Dense/default loss semantics,
  model geometry, routing, optimizer, seeds, data order, cadence, and every
  success threshold were frozen.
- Implementation evidence: the sparse trainer binds the objective weight into
  train/eval VJPs, every metric row, config, and checkpoint metadata while the
  shared Dense default remains `0.15`. Focused composite/gradient/checkpoint
  tests passed `16/16`, including the actual default-argument path; a direct
  CLI config smoke resolved `objective_margin_weight=1.0`.
- Operational pre-run failure: the first default-root invocation exited before
  update 0 and before dataset loading because four sparse trainer default
  expressions referenced the non-exported `MARGIN_WEIGHT` name unqualified.
  The failed root is preserved at
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1`.
  The only source repair was to qualify those defaults as
  `BeatFirstTrainingCore.MARGIN_WEIGHT`; no scientific condition changed. A
  fresh clean retry review returned GO.
- Bound retry: fresh run at
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1`,
  20,000 updates, 512 held teacher states, `(48,40,40)` active counts,
  19,924,022 stored and 58,214 active parameters. No game validation or sealed
  seed was opened.
- Result: FAIL CLOSED at the unchanged action-margin floor. The learner
  completed all 20,000 updates in 534.65 measured seconds (37.408 updates/s,
  1,630.44 candidates/s). Final held top-1 was `0.230469` versus parent
  `0.228516`; NDCG `0.937192` versus `0.937424`; pairwise `0.740153` versus
  `0.740755`; action margin `0.011338` versus `0.011132`. The `+0.000206`
  margin change remained far below the frozen `0.02` floor. No gate artifact
  was promoted and no earlier checkpoint was selected.
- Sparse execution remained healthy: final Q std `0.217505`, full bank
  coverage, maximum retrieved rows `(357,695,625)`, three real collision
  prefilter activations and 101 dropped layer-2 rows, with no cap or non-finite
  failure.
- Final checkpoint:
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1\sparse_3l_20260719T143553\checkpoints\checkpoint_000020000.jls`,
  248,681,005 bytes, SHA-256
  `1c3effb03b3d5cbff429e78fd175cd9b63271d53ff77a433f03a32c62c6ca0cb`.
  The 10,000 checkpoint SHA-256 is
  `95b9f8dc82d26ca6fef91b9f09ac9cc49b26507718019e85042fbe4bf5447444`.
  Metrics SHA-256:
  `2a5149ff47a4d16053679162b71c2ea589efdd3c5a5d5a80814eb0fdf474d597`.
- Decision: stop margin-weight tuning. The paired result supports the
  scale/negative-selection diagnosis but shows that weighting the fixed teacher
  top-1 versus fixed teacher top-2 term is insufficient. The next permissible
  hypothesis is a separately preregistered hard-negative margin loss against
  the student's strongest currently wrong candidate, while retaining
  active-only forward/backward/optimizer and all existing gates.

## 2026-07-19 — 3L k128 student-hard-negative pivot, 20,000 updates

- Hypothesis: replacing the fixed teacher top-2 margin opponent with the
  student's strongest valid non-top-1 candidate will improve top-1 separation
  without changing routing, active width, model capacity, data order, or the
  effective `0.15` margin weight. The selected negative is stop-gradient and
  stable lowest-ID tie-broken; the one-sided target is the teacher top-1 versus
  selected-negative Q gap.
- Bound: fresh 20,000-update run at
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_hardneg_teacher_signal_cpf_20000_v1`,
  with 19,924,022 stored parameters, active counts `(48,40,40)`, 58,214 active
  parameters, and the same active-only sparse forward/backward/lazy optimizer
  contract. The margin1 metrics were SHA-pinned as the parent. No game
  validation or sealed seed was opened.
- Training result: all 20,000 updates completed in 535.82 measured seconds
  (37.326 updates/s, 1,626.87 candidates/s). Held top-1 rose
  `0.029297→0.234375`, NDCG `0.795182→0.938132`, pairwise accuracy
  `0.465374→0.741503`, and composite loss fell `7.289201→3.524871`.
  Update 10,000 still met the margin floor at `0.022216`; by update 20,000 it
  had fallen to `0.010931`. Final Q standard deviation remained finite at
  `0.216198`. All 512 held states produced a valid hard negative and 421 chose
  a candidate different from the fixed teacher top-2.
- Paired interpretation: versus the fixed-top2 parent, final top-1 improved
  from `0.228516` to `0.234375`, NDCG from `0.937424` to `0.938132`, and
  pairwise from `0.740755` to `0.741503`, while action margin decreased from
  `0.011132` to `0.010931`. Versus margin-weight 1.0, top-1 improved by only
  `0.003906` and action margin decreased by `0.000407`. Hard-negative mining
  is therefore a valid ranking learner but did not repair late Q separation.
- Contract result: FAIL CLOSED at the unchanged final held action-margin floor
  `0.02`. The original in-memory controller first stopped on the known
  serialization mismatch between JSON `Float64(Float32(0.15))` and decimal
  `Float64(0.15)`. A no-Julia recovery gate accepted only that exact failure,
  then reached and preserved the scientific failure
  `Final held action margin collapsed`; no gate artifact was promoted.
- Sparse execution remained healthy: full bank coverage, maximum retrieved
  rows `(356,695,626)`, three collision-prefilter activations, and 102 dropped
  layer-2 rows. Active-only accounting and finite-gradient checks remained
  intact.
- Final checkpoint:
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_hardneg_teacher_signal_cpf_20000_v1\sparse_3l_20260719T150408\checkpoints\checkpoint_000020000.jls`,
  248,681,638 bytes, SHA-256
  `e5b931ee16a018f7e2619f174c2eb3001a7d8d6a3b173bbe042d0d05be96d628`.
  The 10,000 checkpoint SHA-256 is
  `ed0eb350e67c1db4d4b1986127569687dde8a9e57b3e48ff0aa2e563217736a0`.
  Metrics SHA-256:
  `ec91496e392b4664a31f371fae971572734cf8ce11d6d81455721a4205c46746`.
- Decision: do not tune the margin weight or rescue hard-negative loss again.
  The next admissible model hypothesis is one bounded MONGOOSE-style learned
  routing experiment; dynamic-k remains after learned routing. This checkpoint
  is analysis-only because held top-1 is still far below the frozen `0.80`
  game-evaluation gate.

## 2026-07-19 — 3L k128 MONGOOSE-style learned-routing pilot

- Hypothesis: while retaining the fixed teacher top-2 objective at effective
  margin weight `0.15`, replace only the serving router after a 2,000-update
  fixed-WTA warmup with a learned K7/L2 SimHash projection applied to both
  queries and neuron route keys. Exact raw `q·key` remained the final rerank;
  the 19,924,022-parameter bank retained selected-only forward, backward, and
  lazy optimizer updates. The learned overlay contained 2,688 parameters and
  was updated densely only within that projection.
- Verification before launch: focused MONGOOSE tests passed `393/393`; existing
  sparse runtime `724/724`; teacher trainer `1,276/1,276`; margin `16/16`; and
  hard-negative `27/27`, for `2,436` passing tests. A real 20M/teacher_v3
  two-update CLI smoke showed update 0 as inactive `fixed_wta_warmup` and
  update 2 as active learned SimHash with refresh count 1 and optimizer step 2.
  Its checkpoint SHA-256 was
  `5783c51d64eb83fe8ad1aacd89a0c6aa0596e5d4318c57b2c6465579dd95a721`.
- Frozen launch artifacts: controller 50,057 bytes, SHA-256
  `e36b16bffe8ca14df7738f0b705c14419b94c86835811e568909a3af95808e5a`;
  preregistration contract 7,861 bytes, SHA-256
  `90c291aca34b8ddb4d434df47889d69b27b69c3a8c203370f6f1fe4cf4510284`.
  A fresh clean milestone review returned GO for exactly one 20,000-update
  attempt. No validation-game or sealed seed was opened.
- Result: FAIL CLOSED at the first learned-router evaluation. The learner
  completed the 2,000-update warmup, published the learned projection/index,
  and then evaluation raised `SimHash bucket-entry cap 1280 exceeded` before
  the update-2,000 metric or checkpoint could be published. This is the
  preregistered cap stop condition; no cap increase, fallback, resume, or
  rescue run is permitted.
- Preserved output root:
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_teacher_signal_cpf_20000_v1`.
  Only the authoritative update-0 metric exists; it verifies the fixed-WTA
  warmup identity and initial held top-1 `0.029297`, NDCG `0.795182`, pairwise
  `0.465374`, and action margin `0.025808`. Metrics SHA-256:
  `b75fa50b35afa9e3a1d6e581bdc951ce7a06245f9647f05a57f5b5cfda6d338c`.
  Controller status SHA-256:
  `afa0e4e409e280b9e85e420072da0559d375f1e96fc8ec58a336f11cca61da5b`.
  Stderr SHA-256:
  `9cfd77ad9077132f6997e24e238a40a0c4cce3df5e3ed0413ba9089749e25401`.
- Interpretation: the failure is learned-hash bucket concentration/tail risk,
  not evidence that the underlying SLIDE bank cannot learn. Ranking learning
  had already been demonstrated by the fixed and hard-negative runs. This
  exact learned-routing arm is closed; tuning K/L, raising caps, or silently
  reverting to WTA would violate the one-shot contract. The next permitted
  hypothesis is separately preregistered dynamic-k, subject to a fresh
  `GO / NO-GO / PIVOT` review.

## 2026-07-19 — dynamic-k64/k256 implementation and prelaunch evidence

- Clean milestone decision: `PIVOT` to exactly one state-level dynamic-k
  teacher-ranking pilot. The serialized owner remains the exact 19,924,022-
  parameter k128 runtime with active counts `(48,40,40)`. Ephemeral k64 and
  k256 runtime views share the owner's parameter arrays, WTA indexes, and
  optimizer objects; `active_count` is never mutated and only the k128 owner
  is checkpointed.
- Frozen policy: every valid candidate is first routed with fixed WTA at k64
  `(24,20,20)`. The stable valid-only raw-Q top-two margin is compared with
  `Float32(0.02)`. A strict-lower result discards all scout outputs/tapes and
  reruns the entire candidate set at k256 `(96,80,80)`; otherwise the k64
  pass is retained. Only the chosen pass receives VJP and sparse optimizer
  updates. Evaluation uses the same state-level decision with zero training
  probes and does not alter policy counters or optimizer event clocks.
- Exact selected-core accounting excludes variable WTA rerank work: k64 scout
  forward is 32,020 MAC/candidate, a clear k64 training route is 78,504, and
  an expanded route is `32,020 + 267,552 = 299,572`. The first compute gate is
  frozen at update 2,000: candidate-weighted expansion must be at most
  `0.285052563012286` and mean selected-core training cost at most 141,520
  MAC/candidate.
- Implementation sources at test time: `dynamic_k64_k256.jl` SHA-256
  `2ff6d17c59cc616caa90878ac3ff451fbbf8f02d61440ab49d8d1e3f6f1a5d97`;
  `teacher_training.jl` SHA-256
  `478c1adc7321474857663f9f5ed0c916bcef545f2f1f54d91d0e926f771c3e22`;
  focused test SHA-256
  `e276cc95940fb2b1c55b25d3b54acde834823ea3fede8c006f9dc840b53ac097`.
- Focused result: `106,793/106,793` checks passed, including exact threshold
  boundaries, stable ties, singleton and padding handling, full clear and
  expanded routes, no scout VJP, chosen-support-only event/moment updates,
  exact integer MAC accounting, evaluation non-mutation, immutable loaded
  topology `(48,40,40)`, and exact next-step checkpoint continuation. Stdout
  SHA-256 was
  `0b9b1d6036cc73c4323b80bcb004f3cc51457eee97dce6d459bb11ee2c54b271`;
  stderr was empty. Existing fixed-width/MONGOOSE/margin/hard-negative suites
  then passed `2,436/2,436`, all with empty stderr.
- Production-shape smoke: the real `teacher_v3` CLI completed two updates,
  held evaluation, and checkpoint publication at
  `C:\tmp\tetris_dynamic_k_cli_smoke_20260719T1742\sparse_3l_20260719T174306`.
  It exercised one expanded and one clear training state. Candidate-weighted
  expansion was `69/103 = 0.669903`; this observation occurred before, and
  therefore did not consume or relax, the frozen update-2,000 gate. The k128-
  owner checkpoint is 248,662,155 bytes with SHA-256
  `033558816af697083b48e9339efc38cd38ba089d0dd7c5b818cc91e54f4566e4`;
  metrics SHA-256 is
  `c234bc5d98c1af1001c0353cd7aabaaed3ee5b804871cc98d96eb394bbe76d11`.
- Status: implementation and production smoke are verified, but no 20,000-
  update pilot has yet been launched. The isolated contract/controller and a
  fresh clean prelaunch decision remain mandatory. No validation-game or
  sealed seed was opened.

## 2026-07-19 — dynamic-k64/k256 20,000-update one-shot, terminal at update 2,000

- A fresh clean launch review returned `GO`, after which exactly one controller
  and one Julia process were started against the previously absent output root
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_dynamic_k64_k256_teacher_20000_v1`.
  The controller was 42,526 bytes with SHA-256
  `70ed92d5121837c192a5d87397fe7b533bc3c4af9e4d4a15811d56aa62ca41cb`;
  the frozen 8,861-byte contract SHA-256 was
  `63a6150fa2157e5f124d5095ff8ed80e17b703aa558c86679c5c786020f9d48b`.
- The run started at `2026-07-19T18:08:33+09:00` and stopped fail-closed at
  `2026-07-19T18:10:44+09:00`. It completed the first 2,000 training updates,
  then the update-2,000 evaluation entered `assert_dynamic_k_panel!` and raised
  `dynamic-k candidate-weighted expansion cap exceeded`. Both owned processes
  exited; no overlapping Julia process was launched.
- Authoritative terminal status is `failed`; `controller_status.json` is 1,661
  bytes with SHA-256
  `40b14d0b7598784bd13c08a59811b822bca29009571f60d7bb5b0c06d97b6527`.
  Structured trainer stderr is 7,864 bytes with SHA-256
  `14a2f11909c23e6d47cd3ec0918fbe84b5919b80f864a6fda5d25e0e8a271ceb`;
  stdout is empty.
- The assertion precedes metric and checkpoint publication inside
  `_evaluate_record`, so the terminal directory contains an empty
  `metrics.jsonl`, no `latest.json`, no checkpoint, and no `gate.json`. This is
  an evidence-publication defect in the launch harness: the exact update-2,000
  expansion fraction and mean selected-core MAC/candidate were not preserved.
  It does not change the fail-closed decision, because the frozen cap comparison
  itself raised before any continuation was possible.
- Interpretation: the one-shot compute-budget hypothesis failed. A raw-Q margin
  threshold of `0.02` routed too much candidate mass through the k256 pass to
  remain within the preregistered `0.285052563012286` expansion cap. This is not
  evidence that the dynamic sparse bank cannot learn; no held ranking panel was
  published. The exact arm is consumed and closed: no retry, resume, threshold
  tuning, width tuning, rescue, or game evaluation is permitted.

## 2026-07-19 — bounded MONGOOSE-inspired v2 fixed-k preregistration

- Decision: return learned routing to the main line. MONGOOSE v1 stopped at the
  first live-router evaluation because it enumerated more than 1,280 bucket
  links before applying its already bounded shortlist. That is a serving and
  compute-control failure. It produced no live-router ranking metric and does
  not reject the learned-routing hypothesis.
- Frozen causal change: retain K7/L2 learned SimHash, its 2,688 projection
  parameters, the v1 witness-pair objective, k128 widths `(48,40,40)`, probes
  `(6,5,5)`, and the 2,000-update warmup/refresh cadence. Replace only the
  unbounded exact-bucket traversal with a distinct v2 index and bounded query
  policy.
- V2 layout: 16 deterministic neuron-ID lanes per logical bucket. Queries use
  complete traversal when the two queried buckets fit within the existing cap;
  overload uses deterministic table/lane round-robin under the unchanged
  aggregate visit caps `(1536,1280,1280)`, then a fixed shortlist under exact
  score caps `(384,640,640)` and stable exact-dot reranking. Fill and exploration
  use bounded no-repeat permutations. Dynamic-k, WTA/dense serving fallback,
  NNUE, and any all-bank query scan remain forbidden.
- Versioning: v1 behavior and identity remain available unchanged. V2 receives
  a separate routing identity and checkpoint metadata; v1/v2 cross-resume must
  fail closed. Active-only bank backward and lazy sparse optimizer updates are
  unchanged.
- Pre-signal gate: adversarial overload, non-overload v1/v2 equivalence,
  deterministic fixed width, every counter bound, bounded rollback, checkpoint
  exact continuation, inactive-byte invariants, and a real teacher_v3 two-update
  smoke must pass before one clean meta review. The first authorized scientific
  run is fresh, fixed-k, and ends at update 2,500 after observations at updates
  2,000 and 2,500. It makes no game-strength claim and may not touch validation
  seeds `8001:8008` or sealed seeds `91001:91032`.
- Full design freeze:
  `experiments/beat_first_v1/sparse_dynamic_3layer/MONGOOSE_V2_DESIGN.md`.

### MONGOOSE v2 implementation and prelaunch evidence

- V2 is a separate serialized routing implementation. V1 `SimHashIndex`,
  `MongooseOverlayState`, and `query!` remain unchanged. V2 adds a
  `BoundedSimHashIndex` with O(1) bucket/lane occupancy, 16 table-salted
  neuron-ID lanes per table bucket, deterministic overload round-robin, fixed
  shortlist and exact-dot rerank, bounded fill/probe permutations, and an
  O(dirty rows) rollback journal. Ordinary updates no longer snapshot an
  affected bucket by scanning its full chain.
- Frozen implementation hashes: overlay 88,296 bytes,
  `e01f711c240c33f416a94f0940a2c9f7a455583a9b2b6e479ebb4157a499b3a6`;
  runtime 48,643 bytes,
  `5b75e50933895b8789d5713019f06535c695faa8d11477b035cced5fdb73bf29`;
  teacher trainer 135,492 bytes,
  `d27ff2be62a90ac8e6731e9fd47efe0f780dd3fa6932d2c41e041d6abf7900a8`.
- Tests passed: v2 bounded core `120/120`; v2 trainer/runtime/checkpoint
  integration `93/93`; v1 MONGOOSE `393/393`; fixed runtime `724/724`;
  teacher trainer `1,276/1,276`; margin `16/16`; hard-negative `27/27`;
  raw-gap `774/774`; and the closed dynamic-k compatibility suite
  `106,793/106,793`. Total frozen checks: `110,216`.
- A full-shape, real teacher_v3 two-update smoke then exercised the bounded v2
  route at update 2 with all 19,924,022 bank parameters present, fixed active
  widths `(48,40,40)`, probes `(6,5,5)`, and learner width 80. It verified
  inactive bank parameter/moment/clock bytes, saved and restored the v2 index
  checkpoint, restored sampler/Xoshiro state, and reproduced update 3 exactly.
  Accepted summary:
  `C:\tmp\tetris_mongoose_v2_real_teacher_smoke_20260719T203000\summary.json`,
  5,018 bytes, SHA-256
  `f5259fa3f5938805ea11915865d884936b13576ad76a3639a174f72412228dba`.
  Checkpoint: 249,869,840 bytes, SHA-256
  `80956b9056ce8ed42a4d0d146e1354862fd21a704450f7f57d13ceccc34eb0d1`.
  The earlier `...T201500` root is preserved separately; it stopped only on an
  unsupported smoke-driver `mktemp` keyword after publishing its update-2
  checkpoint, and is not scientific evidence.
- The fresh 2,500-update signal is frozen by contract 12,619 bytes,
  `7c710d735c044f295a853ab3be9fba829c4709325f05ee2d6ee1aac9517c6b9e`,
  and controller 51,199 bytes,
  `2e07f4adae22bb420eda766ef7a459b1516cba4705fd7908ba389e263fa86d93`.
  Independent parent `-ValidateOnly` passed with `mutation=false` and
  `launched=false`; the output, reservation, and consumed paths were absent.
  A clean launch meta-review is the remaining authorization gate. No
  validation or sealed game seeds have been touched.

## 2026-07-19 — bounded MONGOOSE v2 fixed-k signal, scientific floor failure

- The earlier prelaunch evidence above was superseded before launch by an
  independent clean `NO-GO`: lane identity omitted router seed/layer, the base
  WTA rollback still scanned affected buckets, non-refresh checkpointing ran a
  full-bank v2 validation, and controller cleanup did not positively confirm
  Julia termination. These were implementation-contract failures, not a test
  of learned routing. They were repaired without changing k, the teacher loss,
  update budget, model parameters, or routing caps.
- Final v2 index format 3 binds lane identity to router seed, layer, table, and
  neuron ID; both the v2 and base-WTA rollback paths are O(dirty rows), and
  non-refresh v2 checkpoints use structural validation while scheduled index
  publication retains full validation. The affected serialized regressions
  passed `109,444/109,444`. A fresh real-teacher exact-continuation smoke passed
  at
  `C:\tmp\tetris_mongoose_v2_real_teacher_smoke_20260719T223000\summary.json`
  (5,018 bytes, SHA-256
  `197aded6320d07c97c3bf330f4513a62bfe921a3e4077509da927d23b8a795bb`);
  its checkpoint is 249,870,121 bytes, SHA-256
  `8b15558fa06b8f6ae2572079badcbf280c8b77c9186ae99f332e9a88d43e03d9`.
- The final controller was 69,232 bytes, SHA-256
  `42d2d18710edbe3431f813c57793b7d3d5b9bf1db2e7caf63ded9801c7e33d59`;
  the contract was 15,421 bytes, SHA-256
  `6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450`.
  Parent `-ValidateOnly` again returned `validated=true`, `mutation=false`,
  `launched=false`. A fresh clean reviewer returned `GO`.
- Exactly one Julia process (PID 36152) ran from 21:37:00 to 21:39:36 JST.
  It exited normally and controller ownership/termination was confirmed. No
  Julia remains. Both update-2,000 and update-2,500 checkpoints were published.
  The update-2,500 checkpoint is 249,896,561 bytes, SHA-256
  `c0b8b350be2357c39c27a4cf73fb8efba9b4403ba4de8cee57a2248166eee2af`;
  metrics are 36,596 bytes, SHA-256
  `e63b16a22971dc37461cb358894f82d56ef1e8729b2e41f1c2e68e52dcd153ea`.
- All numerical, bounded-routing, cumulative telemetry, structured-log, and
  checkpoint gates passed. The run then failed only the preregistered
  teacher-signal floor. From update 0 to 2,500, held NDCG improved
  `0.79518 -> 0.84535`, pairwise `0.46537 -> 0.53999`, and loss
  `7.21150 -> 6.48784`; top-1 reached `0.04492`. Against the matched fixed-WTA
  update-2,500 parent, NDCG was `+0.00126`, but top-1 was
  `0.04492 < 0.07031`, pairwise `0.53999 < 0.54071`, and loss
  `6.48784 > 6.33308`. No game evaluation is authorized.
- This is a scientific failure of the exact bounded-v2 fixed-k arm with the
  unchanged v1 witness-pair router objective and 2,500-update budget. It is not
  a serving/cap failure and does not reject the broader learned-LSH hypothesis.
  Layer-2 routes overloaded on `21,674/21,941 = 98.78%` of learned-phase
  candidates and layer 3 on `14,856/21,941 = 67.71%`; exact scoring averaged
  about `638/640` rows in both deeper layers. The learned hash therefore failed
  to produce a selective shortlist despite the bounded scheduler working as
  designed. Dynamic-k remains closed; the next admissible hypothesis must
  improve fixed-k routing itself under a new frozen contract.

## 2026-07-19 — MONGOOSE v2 cutoff-boundary pair one-shot, terminal eligibility failure

- Exactly one fresh frozen arm,
  `sparse_3l_k128_mongoose_v2_cutoff_pair_teacher_signal_cpf_2500_v1`,
  consumed its sentinel at `2026-07-19T22:55:05.1508373+09:00`. The controller
  log opened at `22:55:05.1703045+09:00`; authoritative failure was published
  at `2026-07-19T22:56:30.7810798+09:00`. Status is `failed`, the owned Julia
  launch was captured, and owned-Julia termination was confirmed.
- Before the first update, the 512-state held panel reported composite loss
  `7.211498738266528`, top-1 `0.029296875`, NDCG
  `0.7951817653961015`, pairwise accuracy `0.46537430670033336`, and action
  margin `0.025808145317569142`. `last_step` remained null, throughput recorded
  zero updates, and no checkpoint was published.
- Exact terminal cause: the first training attempt entered
  `_stable_exploitation_cutoff_pair!` and raised
  `cutoff witness has no bounded natural row outside exploitation top-k` at
  `teacher_training.jl:1063`. The frozen cutoff-pair eligibility required a
  collision-positive natural rank-`k+1` row after excluding fills and probes;
  that witness did not exist for the sampled bounded WTA pool.
- Authoritative evidence is
  `D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_cutoff_pair_teacher_signal_cpf_2500_v1\controller_status.json`
  (982 bytes, SHA-256
  `cc0f882a0673e3f76c11780a7272c7f4200b5636c54b205b93b2351759e6beca`)
  and `controller.stderr.log` in the same directory (3,296 bytes, SHA-256
  `b733440ac602af78e3870d3919072b06560b33aff289647ad28aa89653d0370c`).
  Initial metrics are in
  `sparse_3l_20260719T225531\metrics.jsonl` (8,072 bytes, SHA-256
  `286a4cb0f709a221bc1eedbadd44375563f12d315e9341cb814543e3b7f5aba4`).
- Closure: this is a frozen eligibility/evaluation-design failure before any
  learned-router update or live MONGOOSE evaluation, not a bounded-serving,
  MONGOOSE-routing, or dynamic-sparse-learning hypothesis failure. The exact
  arm is consumed and terminal: no rescue, rerun, resume, parameter adjustment,
  checkpoint promotion, or game evaluation is authorized.

## 2026-07-20 — Final EVRL held-teacher comparison against PreAct

- The final corrected episodic ViT recurrent Lookup model was evaluated on the
  exact 128-state real-teacher validation panel used by PreAct. Dataset manifest
  SHA-256 is
  `1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`;
  row-list SHA-256 is
  `fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`.
  Game validation and sealed seeds remained unopened.
- At the matched 12,000-update / 48,000-state budget, PreAct best reached
  top-1 `0.7890625`, NDCG `0.9932922`, and pairwise accuracy `0.9233594`.
  EVRL reached `0.359375`, `0.9712681`, and `0.8202795`, respectively.
  The accuracy hypothesis therefore failed decisively at equal state count.
- The final EVRL update-20,000 checkpoint reached top-1 `0.375`, NDCG
  `0.9583671`, pairwise accuracy `0.7688135`, and composite loss `3.1628645`.
  Relative to PreAct, gaps are `-0.4140625` top-1, `-0.0349251` NDCG, and
  `+0.5990807` composite loss.
- Warm CPU held-panel inference was `50.19 states/s` for final EVRL versus
  `4.19 states/s` for PreAct, a `11.97x` sparse-model throughput advantage.
  Thus the CPU-native execution claim passed while the model-quality claim did
  not.
- EVRL stores 20,577,480 parameters versus PreAct's 1,481,326 (`13.89x`), so
  the held deficit is not explained by scalar capacity. Fixed-batch
  memorization demonstrated trainability but did not transfer to unseen-state
  routing and full candidate-order credit assignment.
- Reproducible evidence is in
  `episodic_vit_recurrent_lookup/PERFORMANCE_COMPARISON_2026-07-20.md`,
  `performance_comparison_2026-07-20.json`, and
  `evaluate_teacher_comparison.jl`.

## 2026-07-21 — EVRL full-token cross-attention ablation

- Removed only the episodic input-token `283 -> 64 -> 16` hard router; retained
  local spatial attention, LookupFFN routing/active updates, recurrence, input,
  teacher objective, and optimizer semantics.
- Serial/barrierless correctness passed on four real-teacher states.
- Equal budget: 12,000 updates, 48,000 states, 20,577,224 parameters,
  23.3685 updates/s.
- Held 128-state panel: top-1 `0.56250`, NDCG `0.978674`, margin `0.072045`,
  pairwise `0.841518`, composite loss `2.809165`, CPU `45.1745 states/s`.
- Versus routed EVRL: top-1 `+0.203125`, NDCG `+0.007406`, margin
  `+0.022502`, inference throughput `0.843715x`.
- Final checkpoint SHA-256:
  `80cc8264a03facf5ff4d0c13cde205b0763012281254362b2f15521c262a4f1c`.
- Detailed record:
  `episodic_vit_recurrent_lookup/TOKEN_ROUTING_ABLATION_2026-07-21.md`.
