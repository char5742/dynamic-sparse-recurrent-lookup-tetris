# Candidate-delta exact barrierless capacity gate (2026-08-06)

Configuration shared by both completed runs:

```text
learning_rate=0.003 clip_norm=2 buckets_per_table=1024
workers=20 candidate_chunk=4 threads=20,0 width=80
full_22d_loss=true exact_conditional_reverse=true gradient_scale=1.0
seed=1212242233
```

Commands:

```powershell
julia --project=. --startup-file=no --threads=20,0 experiments\beat_first_v1\reduced_hay_cpu_native\overfit_candidate_delta_dendritic.jl --states 8 --updates 500 --log-every 50 --buckets 1024 --workers 20 --candidate-chunk 4 --learning-rate 0.003 --clip-norm 2

julia --project=. --startup-file=no --threads=20,0 experiments\beat_first_v1\reduced_hay_cpu_native\overfit_candidate_delta_dendritic.jl --states 16 --updates 1000 --log-every 50 --buckets 1024 --workers 20 --candidate-chunk 4 --learning-rate 0.003 --clip-norm 2
```

## Eight states

| Update | Excess | Tie top-1 | Decision event rate |
|---:|---:|---:|---:|
| 0 | 3.919288 | 0.0000 | 0.245052 |
| 50 | 1.364574 | 0.7500 | 0.280131 |
| 100 | 0.326818 | 0.7500 | 0.283681 |
| 150 | 0.129521 | 0.8750 | 0.296113 |
| 200 | 0.118842 | 0.8750 | 0.309033 |
| 250 | 0.066193 | 1.0000 | 0.312657 |
| 300 | 0.018404 | 1.0000 | 0.313897 |
| 350 | 0.015046 | 1.0000 | 0.314723 |
| 400 | 0.012928 | 1.0000 | 0.315136 |
| 450 | 0.012756 | 1.0000 | 0.315418 |
| 500 | 0.011848 | 1.0000 | 0.315437 |

Final training-only throughput: `28.393 updates/s`, `227.144 states/s`.
Final active program rows: `6206`.

## Sixteen states

| Update | Excess | Tie top-1 | Decision event rate |
|---:|---:|---:|---:|
| 0 | 3.788548 | 0.0000 | 0.236223 |
| 50 | 1.371266 | 0.4375 | 0.286548 |
| 100 | 0.389162 | 0.7500 | 0.286804 |
| 150 | 0.243244 | 0.8125 | 0.298372 |
| 200 | 0.117940 | 0.8750 | 0.309630 |
| 250 | 0.089067 | 0.9375 | 0.312620 |
| 300 | 0.064711 | 0.9375 | 0.312867 |
| 350 | 0.047345 | 0.9375 | 0.313260 |
| 400 | 0.054712 | 0.9375 | 0.313480 |
| 450 | 0.039741 | 1.0000 | 0.313763 |
| 500 | 0.037837 | 1.0000 | 0.314568 |
| 550 | 0.035639 | 1.0000 | 0.315135 |
| 600 | 0.038867 | 1.0000 | 0.315519 |
| 650 | 0.032885 | 1.0000 | 0.315793 |
| 700 | 0.031859 | 1.0000 | 0.316223 |
| 750 | 0.030847 | 1.0000 | 0.316324 |
| 800 | 0.030066 | 1.0000 | 0.316690 |
| 850 | 0.029349 | 1.0000 | 0.316735 |
| 900 | 0.028683 | 1.0000 | 0.317037 |
| 950 | 0.028210 | 1.0000 | 0.317192 |
| 1000 | 0.028074 | 1.0000 | 0.317641 |

Final training-only throughput: `20.170 updates/s`, `322.717 states/s`.
Final active program rows: `7808`.

## Stop condition

The reported event rate above covers the 50 decision cells only. During the
16-state run a separate diagnostic found that the spatial dendritic factors
were at 100% hard-spike rate in phases 2 and 3. Consequently the continuous
path passes the 8/16-state memorization gate, but the physical event plane does
not pass. The instructed 64-state run was not started. Before the next gate,
factor hard events must be fixed and logged separately for every phase.
