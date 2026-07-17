# Independent actual-learner AD retry

Date: 2026-07-18

## Verdict

**Zygote genuinely wins the native CPU comparison on this machine and tracked
learner. Native Enzyme does not. Persistent Reactant + EnzymeMLIR does win the
long fixed-shape compiled comparison, but only after amortizing compilation.**

Do not collapse those two Enzyme-branded paths into one result:

- Native Julia + Zygote: compatible, stable, and the native winner.
- Native Julia + Enzyme.jl: rejected on the actual loss. Static activity does
  not lower the model; runtime activity is slower than Zygote and fails the
  AdamW trajectory-equivalence gate.
- Reactant CPU + EnzymeMLIR: a persistent ahead-of-time compiled full training
  step. It behaves differently in this experiment: its losses track Zygote,
  its warm step is much faster, and it crosses the native cumulative trace in
  a sufficiently long run. The experiment establishes that difference but
  does not isolate which compiler transformation causes it.

The current project default should remain **Zygote** until a changing-batch,
transfer-, validation-, and checkpoint-inclusive Reactant integration run
passes. Native Enzyme should not be selected under the pinned versions.

## Reproducible workload and host

- Intel Core Ultra 7 265K, 20 physical/logical cores, 64 GiB RAM
- Julia 1.12.6, 20 Julia threads, 10 BLAS threads
- Lux 1.31.4, Zygote 0.7.11, Enzyme 0.13.186, Reactant 0.2.274,
  Optimisers 0.4.7
- tracked 165,051-parameter
  `CompactCandidateQ(channels=8, blocks=1, spatial_channels=2)`
- tracked standardized masked ListNet loss, action width 74, state batch 4
- C11b checkpoint, fixed aggregate-C13 training rows `[1, 750, 1501, 2160]`
- AdamW at `1e-3`, betas `(0.9, 0.999)`, weight decay `1e-4`

Dataset, checkpoint, model, trainer, and Manifest SHA-256 values are in
`artifacts/summary.json`. Each timed backend ran in a fresh Julia process with
no other Julia process present. No game, game score, validation row, sealed
evaluation seed, or held-out test seed was used.

## Native correctness and stability

The direct Enzyme function used `ReverseWithPrimal`, one differentiated
execution (no separate loss forward), an `Active` scalar return, `Duplicated`
parameters, `Const` objective/model/state/batch, a preallocated shadow zeroed
recursively before every call, and in-place AdamW.

Static activity failed after compilation at the tracked model's active
`reshape` with `EnzymeRuntimeActivityError`. The documented runtime-activity
mode completed and initially agreed closely with Zygote:

| numerical check | result |
|---|---:|
| loss absolute error | 2.38e-7 |
| gradient cosine | 0.999999999990 |
| gradient maximum absolute error | 2.30e-6 |
| gradient relative L2 | 4.66e-6 |
| one-update parameter cosine | 0.999999999490 |
| one-update parameter maximum absolute error | 9.42e-4 |

That high gradient cosine was not enough for AdamW trajectory equivalence.
Native Enzyme's losses started
`3.443775, 4.375719, 4.507455, 4.061596, 4.047063` and ended at `4.413848`
after 100 updates; Zygote descended to `3.127614` at update 100. Lux's cached
AutoEnzyme runtime path produced exactly the same 100 losses as the direct
path. Lux zeroes the cached shadow recursively in place, so stale shadow
accumulation does not explain the divergence.

## Timing and memory at the historical shape

Warm statistics exclude updates 1--5. Compile-inclusive totals are sums of
the individually timed complete updates.

| path | first update | warm median | warm allocation | peak working set |
|---|---:|---:|---:|---:|
| Zygote `withgradient` | 26.401 s | 60.914 ms | 48.47 MB | 1.081 GiB |
| Enzyme direct + runtime activity | 63.333 s | 111.628 ms | 56.58 MB | 1.442 GiB |
| Lux AutoEnzyme + runtime activity | 62.743 s | 105.294 ms | 56.59 MB | 1.413 GiB |
| Reactant + EnzymeMLIR | 63.945 s | **13.100 ms** | **37.4 KB** | 1.651 GiB |

Zygote is 1.73x faster warm than the strongest native Enzyme configuration,
while native Enzyme allocates 1.17x as much. These timings are secondary to
native Enzyme's failed stability gate.

Reactant is 4.65x faster warm than Zygote and allocates about 1,295x less per
warm host call, at the cost of 1.53x Zygote's peak working set and a 63.9-second
first compiled update.

| compile-inclusive updates | Zygote | Reactant | Zygote / Reactant |
|---:|---:|---:|---:|
| 100 | **33.340 s** | 65.692 s | 0.508x |
| 500 | **59.956 s** | 71.040 s | 0.844x |
| 1,000 | 92.916 s | **77.523 s** | **1.199x** |

The observed cumulative traces first crossed at update **711** and stayed
crossed: Zygote 73.791 s versus Reactant 73.755 s. This is an observed crossing
for this fixed run, not a universal forecast.

Reactant completed 1,000 finite updates using one cached compiled thunk in all
1,000 observations. `sync=true` covered the timed compiled call; an additional
loss-and-all-parameters barrier had a 10 microsecond median. No recompile or
BLAS/fallback warning appeared in the captured logs. At the recorded
checkpoints, maximum loss error versus Zygote was 3.81e-6. After 1,000 updates,
parameter cosine was 0.99999953, relative L2 was 9.70e-4, and maximum absolute
error was 0.0116; final losses differed by 2.86e-6.

## Why the old answers were confusing

The first repo harness had two material flaws for the broad claim that
"Enzyme cannot win":

1. Its direct native helper called `Enzyme.make_zero(ps)` inside every gradient
   evaluation, so its allocation number included a fresh shadow. The later
   external retry corrected this with a reused shadow and Lux's cache and still
   found native Enzyme about 5.2x slower on the 133k proxy. Therefore this flaw
   weakens the original allocation methodology but does **not** rescue native
   Enzyme.
2. Its Reactant experiment used allocating `single_train_step`, returned
   gradients, did not establish a retained persistent compiled state, and
   rejected later launch timings as unsynchronized. It also used a
   time-to-100 gate, which cannot answer whether a high-compile/low-steady-cost
   backend wins a long run. Lux's official API explicitly recommends
   `single_train_step!`, retaining the returned `TrainState`, and `sync=true`
   for valid Reactant timing.

The later external retry fixed both issues and reported native Enzyme losing
on the proxy and persistent Reactant winning steady state on the actual loss.
This independent retry agrees with that split verdict. It adds that native
Enzyme on the *actual* tracked ListNet workload requires runtime activity and
does not match the optimizer trajectory. Its observed Reactant crossover is
711 rather than the prior run's projected 1,193, so the crossover should be
treated as machine/run/workload-specific.

## Exact adoption rule

1. **Never select native Enzyme** for this tracked model/loss under these pins.
   Reconsider only after a version or model rewrite passes static lowering,
   gradient, 100-update trajectory, and speed gates anew.
2. **Select Zygote** for dynamic experiments, changing shapes, fewer than
   1,000 planned same-shape optimizer updates, and as the production default
   until the integration gate below passes.
3. **Reactant is eligible** when at least 1,000 same-shape updates will reuse
   one persistent `TrainState`. Switch production only if a changing-batch run
   includes sampling/packing, host-device transfer, validation, and checkpoint
   export; matches the registered loss/parameter trajectory; shows one cached
   thunk/no recompilation; and is at least 1.15x faster in measured total wall
   time. The fixed-batch 1,000-update result passes the speed threshold at
   1.199x, but it is not the complete pipeline gate.

## Scope limits

State-batch 16 Zygote completed 100 updates (26.732-second first, 234.344-ms
warm median, 49.798-second total). The remaining state-batch-16 paths and
32/64/128 sweep were deliberately stopped after the actual native correctness
failure and the decisive 1,000-update crossover. This is recorded rather than
filled with projections.

Primary technical references:

- [Enzyme runtime activity FAQ](https://enzymead.github.io/Enzyme.jl/stable/faq/#Runtime-Activity)
- [Enzyme API](https://enzymead.github.io/Enzyme.jl/stable/api/)
- [Lux Training API](https://lux.csail.mit.edu/dev/api/Lux/utilities)
- [Lux profiling Reactant training loops](https://lux.csail.mit.edu/stable/manual/profiling_training_loop)

Raw logs, complete per-update timings/allocations, full parameter vectors,
machine summary, and failure ledger are in this directory and `artifacts/`.
