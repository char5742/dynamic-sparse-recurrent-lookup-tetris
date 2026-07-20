# Residual Lookup-SLIDE R0 teacher-training signal

Status: adapter/harness only. R0 has not been executed. Julia, Python,
OpenVINO, and game evaluation remain outside this change.

## Causal question

R0 asks whether the canonical frozen-router Residual Lookup-SLIDE model can
move the existing teacher-v3 held ranking signal in 1,000 updates. It isolates
lookup-value learning from learned routing and dead-row interventions. This is
a teacher-training diagnostic, not a game-strength claim.

## Canonical production model

The adapter includes `ResidualLookupSlide.jl`; it does not define a second
model or optimizer.

- Learner candidate width: 80. Each update packs one complete, prefix-valid
  teacher state. Candidates are never sampled, truncated, or reordered.
- Carrier: 256 values formed exactly as
  `[CountSketch192(raw496), context22, NEXT/HOLD42]`. `raw496` is the existing
  independent-candidate value feature vector; `context22` follows
  `ROUTE_AUX_INDICES`; NEXT/HOLD uses the existing piece/token order.
- Three residual blocks, each with 76 tables. Each table has 343 rows and each
  row has 256 trainable values.
- The fixed router is RMSNorm, layer-seeded signed normalized FHT256, then
  three deterministic WTA7 digits per table. It has no trainable parameter;
  strict first-maximum tie breaking and three immutable router seeds are bound
  by the checkpoint/source identity. Routing is stop-gradient.
- Each block reads one row from every table, sums the 76 vectors, scales by
  `1/sqrt(76)`, and adds `alpha*y`. `alpha = 0.1*tanh(alpha_logit)` and starts
  at exactly `0.01f0` through the canonical checked logit.
- A trainable dense 22-by-256 head and 22 biases produce Q, death logit,
  16 quantiles, and four geometry values.
- Parameters: 20,020,224 lookup values + 5,632 head weights + 22 biases + 3
  alpha logits = 20,025,881. One candidate selects 58,368 bank values; including
  head, bias, and alpha, the canonical active count is 64,025.
- Initialization, selected-only forward/VJP, and accounting come from the
  canonical model. All rows selected by all valid candidates are passed to the
  active-event optimizer; duplicate rows are stably reduced, while unselected
  bank values and their stored moments are not scanned or mutated by the step.
- The canonical whole-model prepare barrier covers all three active-row bank
  updates, dense head/bias, and the three alpha logits before the first write.
  R0 freezes AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, learning rate `1e-4`
  for bank/head/alpha, and zero weight decay.

## Reused teacher-v3 contract

`teacher_training.jl` directly reuses the implementation owned by
`sparse_dynamic_3layer/teacher_training.jl` and its nested training core:

- `load_teacher_dataset` with partial datasets forbidden;
- `episode_separated_split`, including immutable predefined-split preference
  and group/episode leakage checks;
- `fixed_evaluation_subset`;
- `EpochSampler` shuffle-without-replacement order and exact sampler snapshot;
- `pack_batch!`, including complete candidate order, teacher standardization,
  fixed teacher-top2 witnesses, and every auxiliary target;
- `supervised_components` and `evaluation_metrics`.

The objective is the same standardized composite:

- standardized ListNet, temperature 0.5, weight 1.0;
- raw-Q Huber, weight 0.25;
- fixed-teacher-top2 gap Huber, weight 0.15;
- death BCE, weight 0.10;
- quantile teacher loss, weight 0.05;
- normalized geometry loss, weight 0.10;
- Huber delta 1.0.

The adapter asserts these shared-core constants at initialization and restore.

## One-shot schedule

- A production invocation must be fresh. Existing run directories and
  `BEAT_RLS_R0_RESUME` are rejected.
- Exactly 1,000 single-state updates.
- Held evaluations exactly at updates 0, 500, and 1,000.
- Fresh immutable checkpoints exactly after the 500 and 1,000 evaluations.
- Held subset: up to 128 immutable validation rows selected with the shared
  deterministic subset helper. These are data split/sampler seeds, not game
  seeds.
- No game evaluation, no game seeds, and no promotion decision are present.

Only paths and the fresh run ID are configurable:

- `BEAT_RLS_R0_DATASET` (default teacher-v3 path)
- `BEAT_RLS_R0_OUTPUT` (default R0 run root)
- `BEAT_RLS_R0_RUN_ID` (optional safe fresh-directory name)

Geometry, objective, optimizer, seeds, schedules, and intervention flags are
serialized and fail closed against the frozen contract.

## Signal and telemetry

Every scheduled record contains:

- held composite loss, NDCG, pairwise accuracy, top-1 agreement, mean student
  action margin, and every shared loss component;
- for all 228 tables: selection count, used/unused rows, coverage, Shannon and
  normalized entropy, maximum row share, and Gini coefficient;
- route churn/stability against update 0 and the previous evaluation;
- per-block cumulative selected-gradient events, nonzero selected-gradient
  events, and their rate; step records also report local rates and unique rows;
- training updates/second and valid candidates/second;
- parameter, objective, router, and disabled-intervention identities.

Evaluation does not change model parameters, optimizer clocks, or training
route-usage counters. Coverage counts training selections only.

## Failure and exact continuation boundary

The shared optimizer prepares and validates all selected sparse gradients,
dense gradients, parameters, moments, event clocks, and decay clocks before a
whole-model copy-only commit. The adapter validates telemetry capacity before
that commit. Non-finite or malformed supervision therefore fails before model,
optimizer, route counters, or update counters change.

The canonical checkpoint envelope uses a fresh same-directory temporary file
and atomic rename. It stores model, every optimizer moment/clock, immutable
router seeds, and model RNG. Adapter training state adds the exact epoch sampler
and RNG, split/held rows, config, evaluation history, route references,
coverage/gradient counters, and throughput counters. Restore requires exact
topology, seeds, config, split metadata, Julia version, a mandatory
byte-count/SHA-256 receipt, the immutable teacher manifest/content SHA-256, and
a SHA-256 source fingerprint covering this four-file adapter, the canonical
lookup core, and reused teacher contracts. The library restore path exists for
exact-continuation testing; the R0 production CLI remains fresh-only.

## Deliberately absent in R0

Revival and exploration are review-mandated isolation variables and are hard
disabled. There is no dead-row reset, usage-dependent reinitialization, random
route probe, epsilon route, learned hash, auxiliary router loss, or background
bank sweep in this harness.

R1 is explicitly pending. It may introduce a separately preregistered
revival/exploration intervention only after untouched R0 evidence is recorded
and reviewed.

## Execution gate

After static review, the first permitted Julia action should be the focused
teacher adapter test. Only after it passes should the fresh signal run begin:

```powershell
julia --project=. experiments/beat_first_v1/residual_lookup_slide/test_teacher_training.jl
julia --project=. experiments/beat_first_v1/residual_lookup_slide/run_teacher_signal.jl
```

These commands are documentation only; they were not run while creating R0.
