# Blinded AD performance rerun 2

Date: 2026-07-18

## Independent verdict

The user's skepticism is reasonable in general, but **native standalone Enzyme
does not beat native Lux + Zygote on this frozen, tracked learner**.  The
strongest native Enzyme configuration that lowers the unchanged model is
`set_runtime_activity(ReverseWithPrimal)` with a preallocated, recursively
zeroed derivative shadow.  It is initially gradient-close, but it is 1.69x
slower warm for complete updates, allocates 1.17x more, takes 1.90x as long at
1,000 updates, and fails the AdamW loss/parameter trajectory gate.

Lux + Reactant + EnzymeMLIR is a materially different result.  Its persistent
whole-update compiled path tracks Zygote's loss, is 4.53x faster warm, crosses
the cumulative Zygote trace permanently at update 773, and is 1.145x faster at
1,000 updates.  It pays about 70 seconds for the first compiled update and uses
more peak memory.  This run falls just short of the previously declared 1.15x
production-promotion threshold, and it is still a repeated fixed-batch kernel,
not the complete changing-batch learner pipeline.

Therefore: keep Zygote as the native/default backend under these pins.  Do not
select standalone Enzyme without a package/model change and a new trajectory
gate.  Reactant remains eligible only for long persistent fixed-shape runs and
needs a full pipeline integration measurement before production use.

## Audited workload and methods

- Julia 1.12.6, 20 Julia threads, 10 BLAS threads; Lux 1.31.4, Zygote 0.7.11,
  Enzyme 0.13.186, Reactant 0.2.274, Optimisers 0.4.7.
- Tracked `CompactCandidateQ(channels=8, blocks=1, spatial_channels=2)`,
  165,051 parameters.
- Tracked standardized masked ListNet objective, fixed width 74, production
  state batch 4, C11b warm checkpoint, C13 rows `[1, 750, 1501, 2160]`.
- AdamW: learning rate `1e-3`, betas `(0.9, 0.999)`, weight decay `1e-4`.
- Every timed complete update contains forward, objective, reverse, and
  `Optimisers.update!` (or the equivalent fused compiled Reactant update).
- The first five trajectory updates are excluded from warm medians.  Every
  backend ran in a fresh Julia process with identical thread settings.
- Direct Enzyme uses one `Enzyme.autodiff` differentiated execution, returns
  the primal loss, performs no duplicate forward, uses `Active` scalar return,
  `Duplicated(parameters, shadow)`, `Const` model/state/objective/batch, and
  recursively zeros one preallocated/reused shadow before every call.
- Reactant retains the returned `TrainState`, uses `return_gradients=false`
  and `sync=true`, and observed one cached compiled thunk across 1,000 calls.

The dataset, checkpoint, model, trainer, imported workload, Project, and
Manifest SHA-256 values are embedded in every JSON artifact.  Key input hashes
are dataset `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba`,
checkpoint `3df9f4d233e235addc7bf9b5831c0823cff670261283d92c688efa0c8755e0d0`,
model `793535dfc43e1c16a0b9196305f2e329438afc6aa458fdc7d8521ed1e36b1052`,
trainer `654e7f7c0e05ca6c005ebeb39924769fcbfd5299ad4e4c0acb3bcb6e4fde4ec3`,
and frozen Manifest `09c5730de81a2b9ccfc7259ff6167e6814bb2a303916275ad0b8c7c3a958d3f8`.

## Primary state-batch-4 timing

Times at 100 and 1,000 are sums of timed complete updates from the same
1,000-update fresh-process trace.  “First” is compile-inclusive.

| backend | first update | warm median | warm allocation | n=100 | n=1,000 | peak working set |
|---|---:|---:|---:|---:|---:|---:|
| native Zygote | 27.3747 s | 63.886 ms | 48,471,767 B | 34.585 s | 96.455 s | 1.073 GiB |
| native direct Enzyme runtime | 69.6814 s | 107.706 ms | 56,582,643 B | 81.132 s | 183.743 s | 1.457 GiB |
| Reactant + EnzymeMLIR | 69.9382 s | 14.088 ms | 37,424 B | 71.539 s | 84.256 s | 1.638 GiB |

Native Enzyme/Zygote is 1.686x at warm median, 2.346x at 100, and 1.905x at
1,000 (values greater than one are slower).  Reactant/Zygote is initially
slower, but its permanent observed cumulative crossover is update 773:
Zygote 81.130252 seconds versus Reactant 81.086809 seconds.  At update 1,000,
the summed-timing speedup is `96.454709 / 84.256107 = 1.144780x`.

An explicit post-call Reactant loss-and-all-parameters barrier had a roughly
10 microsecond median.  No recompile, fallback, BLAS warning, or other warning
appeared in the captured Reactant log.

## AD-only timing

This fresh-process benchmark repeats the gradient call at fixed parameters and
does not include optimizer setup or update in its timed region.  It includes
shadow clearing for Enzyme.

| backend | first gradient | warm gradient | warm allocation | n=100 |
|---|---:|---:|---:|---:|
| Zygote | 28.7803 s | 77.092 ms | 48,424,023 B | 37.208 s |
| direct Enzyme runtime | 70.6528 s | 108.163 ms | 56,534,947 B | 82.259 s |

Thus native Enzyme loses even before AdamW is included: 1.40x slower warm and
2.21x slower compile-inclusive at 100 gradient calls.

## Numerical conformance and concrete native blocker

At the initial parameters, direct runtime-activity Enzyme is close to Zygote:

| check | result |
|---|---:|
| loss absolute error | 2.384e-7 |
| gradient cosine | 0.9999999999901593 |
| gradient maximum absolute error | 2.302e-6 |
| gradient relative L2 | 4.659e-6 |
| one-step parameter cosine | 0.9999999994898 |
| one-step parameter maximum absolute error | 9.419e-4 |

There are no gradient sign-bit mismatches, but Zygote has 2,241 exactly-zero
coordinates and Enzyme has 2,240.  AdamW's first-step normalization is highly
sensitive near zero, magnifying a very small derivative difference into an
almost-learning-rate-sized parameter difference.  With identical optimizer
semantics, native Enzyme's losses are
`3.443775, 4.375719, 4.507455, 4.061596, 4.047063` for updates 1--5,
`4.413848` at 100, and `3.132826` at 1,000.  Zygote instead reaches
`3.127614` at 100 and `3.125909` at 1,000.  Native Enzyme's final parameters
versus Zygote have cosine 0.8861545, maximum absolute error 0.5377314, and
relative L2 0.4963645.  This is a concrete trajectory-conformance failure, not
merely a speed loss.

Static `ReverseWithPrimal` fails with `EnzymeRuntimeActivityError` inside the
unchanged tracked model's `reshape`: constant memory can be stored/returned to
an active variable, so Enzyme cannot guarantee static activity correctness.
Runtime activity adds pointer-identity checks and blocks the stronger static
optimization route.  `set_strong_zero(set_runtime_activity(...))` was also
screened; under these pins it produced loss 4.399473 instead of 3.443775 and
gradient cosine 0.304645, so it is incorrect and was not timed as a candidate.

The pinned Lux native `AutoEnzyme` extension is not used as the strongest
standalone baseline: it caches a derivative tree but does not explicitly clear
that tree on repeated calls.  The direct path above does clear every leaf and
still fails.  A manually split Enzyme tape cannot be safely reused across
updates because its saved intermediates depend on changing parameters; the
unsplit call already reuses compiled code while returning the primal without a
second forward.

Reactant losses match Zygote within 3.815e-6 at the recorded 1/2/3/4/5/10/100/
500/1,000 checkpoints.  At 1,000 updates, Reactant parameters versus Zygote
have cosine 0.9999995295, maximum absolute error 0.0116125, and relative L2
0.00097005.  Its speed comes from compiling and caching the entire fixed-shape
forward/backward/AdamW update, enabling fusion and avoiding Julia-side
gradient-tree traffic; it is not evidence that standalone Enzyme reverse mode
is faster.

## Limited scaling screen

The requested stop/timebox arrived after native batch 16 completed.  Batch 16
at 100 updates reproduced the direction: Zygote first/warm/n=100 were
28.994 s / 250.400 ms / 54.371 s, while native Enzyme was
72.555 s / 429.112 ms / 114.757 s.  Native Enzyme ended at loss 4.931584 versus
Zygote 3.127614.  Reactant batch 16 and batches 32/64/128 were not started; no
values are projected for them.

## Exact commands

From `C:\Users\fshuu\Documents\tetris` in PowerShell:

```powershell
$env:AD_BLIND_OUTPUT_ROOT='D:\tetris-paper-plus\runs\ad_blind-rerun-2'
$env:AD_BLIND_STATE_BATCH='4'
$env:AD_BLIND_STEPS='1000'

$env:AD_BLIND_BACKEND='zygote'
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_native.jl

$env:AD_BLIND_BACKEND='enzyme_runtime'
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_native.jl

julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_reactant.jl

$env:AD_BLIND_OUTPUT="D:\tetris-paper-plus\runs\ad_blind-rerun-2\numerics_b4.json"
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/numerics.jl

$env:AD_BLIND_CALLS='100'
$env:AD_BLIND_BACKEND='zygote'
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_gradient.jl
$env:AD_BLIND_BACKEND='enzyme_runtime'
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_gradient.jl
```

Batch 16 used the same native command with `AD_BLIND_STATE_BATCH=16` and
`AD_BLIND_STEPS=100`.

## Artifact hashes

All files are under `D:\tetris-paper-plus\runs\ad_blind-rerun-2`.

| artifact | SHA-256 |
|---|---|
| `numerics_b4.json` | `847ef94460765b3d87a72c3dd9ad09b69af6364613a5fb4596c6435b1cdecbcc` |
| `zygote_b4_n1000.json` | `47144ac44126feab9d32c07c88ad359e44d213d36cb1455fac3de097648e5871` |
| `enzyme_runtime_b4_n1000.json` | `ff121d86f171b8e18e3e102dc246572e176f9e5fc91a5d0da8aa54c4b63685c6` |
| `reactant_b4_n1000.json` | `f41114f545075543a06c414ab3bff31765d746ec38739e3890988e4582483328` |
| `gradient_zygote_b4_n100.json` | `d979b9fe14a8e0c29eae0ec4e2cad89051106f2ead442f0f02e2f29a52c6ac1b` |
| `gradient_enzyme_runtime_b4_n100.json` | `efb2430e0a81fe434ed51eb6d07c7f38b01a45402297ad3b3dea025dd195cdd1` |
| `zygote_b16_n100.json` | `ed72318480a8cba0bc0e1c7e220ae17db94f8eda7f90d127488c4ffbead6060f` |
| `enzyme_runtime_b16_n100.json` | `c1ff9980ae07a67bc1e04e8f8c9da5cd404c3da05dafdd0c4adedca2411e6ed7` |

The active `experiments/online_counterfactual_top2_r1` directory was neither
read for workload inputs nor modified.
