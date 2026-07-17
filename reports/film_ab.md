# Compact NEXT-FiLM matched A/B

Status: implementation and the update-zero numerical smoke are complete. The
300-update A/B is **not executed**. Milestone-2 clean review decision D0 pauses
new architecture training until the evaluation freeze/provenance gate is
complete.

## Controlled comparison

The plain arm reproduces the current `CompactCandidateQ` at 8 channels, one
residual block and two projected spatial channels. The FiLM arm adds only a
zero-initialized `Dense(64 => 16)` conditioning projection. All other weights
are copied from the plain initialization. At update zero, the two networks must
therefore agree to `1e-6` max absolute error.

Both arms use the existing 2,000-state teacher dataset, trim its schema to the
actual maximum of 74 candidates, use the same episode split (1--6 train, 7--8
development), identical precomputed state batches, masked per-state
standardized ListNet cross entropy, seed `517812839`, learning rate `1e-3`, and
300 updates. Teacher and student logits both use temperature `0.25`, matching
the learning track's current screen. No validation or sealed test game seeds
are touched.

## Size and compute target

The analytical comparison before execution is:

| arm | expected params | MAC/candidate | delta vs plain |
|---|---:|---:|---:|
| plain | 165,051 | 478,144 | -- |
| NEXT-FiLM | 166,091 | 479,168 | +0.63% params, +0.21% MAC |

This is well inside the required ±25% parameter envelope. Measured smoke and
training results will be appended after execution.

## Numerical smoke

The smoke used two complete states from the real teacher dataset (148 fixed
candidate slots in total) and passed:

| check | result |
|---|---:|
| effective candidate width | 74 |
| plain parameters | 165,051 |
| FiLM parameters | 166,091 |
| update-zero max absolute output difference | 0.0 |
| update-zero ListNet CE (plain / FiLM, T=1 pre-temperature patch) | 3.842589 / 3.842589 |
| all plain / FiLM gradients finite | yes / yes |
| FiLM condition-weight gradient norm | 0.0135619 |

The exact initial equality is structural and remains true at any loss
temperature: the zero-initialized condition projection makes the two forward
passes bit-identical. The nonzero finite condition gradient proves that it can
learn on the chosen loss. The displayed loss/gradient smoke was run before the
`T=0.25` control was added, and was not rerun after D0 froze heavy work. Thus it
is implementation evidence only, not a completed T=0.25 experiment or a
strength result.

## Reproduction commands (deferred)

After D0 is explicitly cleared, re-run the temperature-aware smoke first:

```powershell
$env:FILM_AB_TEMPERATURE='0.25'
julia --project=. --threads=1 experiments\film_ab\smoke.jl
```

Then run the matched A/B in a clean CPU window:

```powershell
$env:FILM_AB_DATASET='D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2'
$env:FILM_AB_SEED='517812839'
$env:FILM_AB_LR='1e-3'
$env:FILM_AB_TEMPERATURE='0.25'
$env:FILM_AB_UPDATES='300'
$env:FILM_AB_STATE_BATCH='2'
$env:FILM_AB_RUN_ID='F01_matched_T025_300'
julia --project=. --threads=20 experiments\film_ab\train_ab.jl
```

The only generated artifacts will be placed in
`D:\tetris-paper-plus\checkpoints\film_ab` and
`D:\tetris-paper-plus\runs\film_ab`. No artifacts were generated there in the
current paused state.

## Deferred decision

This candidate should not consume more CPU merely because its offline
inductive bias is attractive. The clean review found that old-Q distillation
has no direct teacher signal for exceeding the old policy and that evaluation
provenance is not yet frozen. Once D0 is complete, this A/B is useful only if
the project still chooses an old-Q compression branch or replaces the targets
with a demonstrably stronger teacher (for example Bellman re-ranking labels).
