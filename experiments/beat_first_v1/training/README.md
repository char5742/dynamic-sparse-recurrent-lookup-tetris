# Beat-first convergence training

`train_supervised.jl` trains exactly four frozen registry presets, one run at a
time: `preact_eca`, `preact_eca_medium`, `gravity_film`, and
`gravity_film_medium`. Every parameter is passed to the optimizer; startup
fails unless `trainable_parameter_count == total_parameter_count` and a
Zygote witness finds non-zero gradients in the first, middle, and final
backbone paths.

The teacher objective is fixed-width (74 candidates) and combines ListNet,
old-Q Huber, top-1/top-2 margin, selected-action death BCE, and quantile teacher
loss. Train/validation rows are separated by seed when the dataset provides
`seed_ids`/`episode_seeds`, otherwise by complete episode. Geometry caching is
automatically disabled for datasets above 2,048 states, so a 100k+ state file
does not create an unbounded object cache.

Every configurable 100--250 updates, the driver overwrites `latest`, updates
`best`, and appends JSONL containing decomposed train/validation losses,
top-1/NDCG/pairwise, Q mean/std, action margin, global gradient norm, parameter
norm, updates/s, and epoch equivalent. A checkpoint is game-eligible only when
it has completed at least one epoch, validation top-1 is at least 0.80, the
last two evaluations are finite/stable, and Q/margin diagnostics are finite.
The periodic train/validation subsets are deterministic (defaults 256/512
states) and their exact row counts and split groups are recorded.
The expensive host-side full-gradient diagnostic is refreshed every five
evaluations by default (`BEAT_GRAD_EVAL_INTERVAL`) and each record identifies
the update at which that norm was observed; the one-time layer witness always
runs at update zero.

Reactant + EnzymeMLIR is selected by default. Supply the fixed-shape backend
source explicitly; switch only the backend environment variable for the
Native Julia + Zygote fallback.

```powershell
$env:BEAT_VARIANT='gravity_film'
$env:BEAT_TEACHER_DATASET='D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v2'
$env:BEAT_STATE_BATCH='4'
$env:BEAT_BACKEND='reactant'
$env:BEAT_BACKEND_SOURCE='C:\Users\fshuu\Documents\tetris\experiments\beat_first_v1\backend\fixedshape_learner.jl'
$env:BEAT_BACKEND_MODULE='BeatFirstFixedShapeBackend'
$env:BEAT_EVAL_INTERVAL='200'
$env:BEAT_MIN_EPOCHS='1'
$env:BEAT_MAX_EPOCHS='3'
julia --project=experiments/beat_first_v1 experiments/beat_first_v1/training/train_supervised.jl
```

Fallback:

```powershell
$env:BEAT_BACKEND='native'
julia --project=experiments/beat_first_v1 experiments/beat_first_v1/training/train_supervised.jl
```

`rl_stage2.jl` retains the PER, n-step, EMA, teacher-decay, and QR-DQN
primitives. It must not be started until a supervised checkpoint reports
`game_eligible=true` and then passes the fixed development-game gate.

After the N1 insurance run has exited, `post_n1_smoke.jl` performs the bounded
full-parameter preflight and Reactant throughput measurement. Its safe default
is one model (`preact_eca`), state batch 2, and five updates; use comma-separated
values only when deliberately sequencing more variants or batches. Both a
legacy JLD2 teacher file and a sharded dataset directory with `manifest.json`
are accepted through `BEAT_TEACHER_DATASET`.

```powershell
$env:BEAT_SMOKE_VARIANTS='preact_eca'
$env:BEAT_SMOKE_STATE_BATCHES='2'
$env:BEAT_SMOKE_UPDATES='5'
$env:BEAT_SMOKE_OUTPUT='D:\tetris-paper-plus\runs\beat_first_v1\preact_small_smoke.json'
julia --project=experiments/beat_first_v1 experiments/beat_first_v1/training/post_n1_smoke.jl
```

Explicit Medium/batch sweep (run only after N1 and preferably one model per
process to avoid contaminating peak-memory measurements):

```powershell
$env:BEAT_SMOKE_VARIANTS='gravity_film_medium'
$env:BEAT_SMOKE_STATE_BATCHES='2,4,8'
$env:BEAT_SMOKE_UPDATES='10'
$env:BEAT_TEACHER_DATASET='D:\tetris-paper-plus\datasets\learning\teacher_100k'
julia --project=experiments/beat_first_v1 experiments/beat_first_v1/training/post_n1_smoke.jl
```
