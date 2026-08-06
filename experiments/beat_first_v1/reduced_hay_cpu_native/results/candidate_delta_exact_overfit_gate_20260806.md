# Candidate-delta exact overfit gate (2026-08-06)

This is the focused learnability gate after integrating real-input fan-in
normalization.  Both runs use the complete `teacher_v3` dataset, the same
seeded train-row selection, width 80, all 22 supervised outputs, grouped exact
conditional reverse, and sparse-row AdamW.  No 16/64-state run was started.

## Commands

```powershell
$env:JULIA_NUM_THREADS='1'
julia --project=. --startup-file=no experiments\beat_first_v1\reduced_hay_cpu_native\overfit_candidate_delta_dendritic.jl --states 1 --updates 50 --log-every 10 --buckets 256 --learning-rate 0.003 --clip-norm 2

julia --project=. --startup-file=no experiments\beat_first_v1\reduced_hay_cpu_native\overfit_candidate_delta_dendritic.jl --states 8 --updates 500 --log-every 50 --buckets 256 --learning-rate 0.003 --clip-norm 2
```

## One-state trajectory

| Update | Excess | Legacy top-1 | Tie-aware top-1 | Hard-event rate |
|---:|---:|---:|---:|---:|
| 0 | 6.893998 | 0.000 | 0.000 | 0.233590 |
| 10 | 5.981728 | 0.000 | 0.000 | 0.264359 |
| 20 | 3.195558 | 0.000 | 0.000 | 0.264103 |
| 30 | 1.516584 | 1.000 | 1.000 | 0.273590 |
| 40 | 1.008802 | 1.000 | 1.000 | 0.274359 |
| 50 | 0.678756 | 1.000 | 1.000 | 0.282564 |

Final training-only throughput was `28.802 updates/s` (`28.802 states/s`).
Wall throughput including progress evaluation was `16.077 updates/s`.

## Eight-state trajectory

| Update | Excess | Legacy top-1 | Tie-aware top-1 | Hard-event rate |
|---:|---:|---:|---:|---:|
| 0 | 3.919288 | 0.000 | 0.000 | 0.245052 |
| 50 | 1.778823 | 0.125 | 0.125 | 0.313634 |
| 100 | 1.179434 | 0.500 | 0.500 | 0.318723 |
| 150 | 0.760079 | 0.625 | 0.625 | 0.326479 |
| 200 | 0.314588 | 0.500 | 0.500 | 0.328263 |
| 250 | 0.170663 | 0.750 | 0.750 | 0.326441 |
| 300 | 0.120707 | 0.750 | 0.750 | 0.325784 |
| 350 | 0.099799 | 0.875 | 0.875 | 0.325352 |
| 400 | 0.078472 | 0.750 | 0.750 | 0.324563 |
| 450 | 0.093744 | 1.000 | 1.000 | 0.323906 |
| 500 | 0.052967 | 1.000 | 1.000 | 0.323305 |

Final training-only throughput was `5.042 updates/s` (`40.336 states/s`).
Wall throughput including progress evaluation was `4.947 updates/s`.
The final optimizer step touched 407 program rows, had gradient norm `0.025081`,
and was not clipped.

## Gate result

The continuous exact path memorizes eight real Tetris states while retaining
physical hard events.  It no longer exhibits the previous all-cell event
saturation.  This proves the candidate-delta architecture's small-panel
capacity; it does not yet establish held-out performance or local-learning
quality.
