# Blinded AD performance rerun 2 — corrected adjudication

Date: 2026-07-18

## Corrected verdict

This rerun **does not establish that native Zygote beats the best achievable
standalone Enzyme implementation**.  The attempted direct runtime-activity
Enzyme path is stateful and invalid: in its nominally fixed-parameter,
gradient-only trace, the returned primal loss changes from `3.443775` on call 1
to `4.399473` on call 2 and `6.192273` on call 100, while Zygote remains exactly
`3.443775` for all 100 calls.  No optimizer ran in that diagnostic.  Therefore
the native Enzyme timing and optimizer trajectory are measurements of this
invalid annotation/implementation pattern, not a fair upper bound on
standalone Enzyme.

The valid result is narrower: on the repeated fixed-row, fixed-shape
state-batch-4 ListNet candidate kernel, persistent Lux + Reactant + EnzymeMLIR
has much lower steady timed-call latency than Zygote and finishes 1,000 updates
about 1.143x faster by full-loop wall time.  That is below the preregistered
1.15x promotion threshold.  It does not establish a production learner win.

The appropriate decision from this rerun is to retain Zygote as the existing
default because Reactant did not clear the promotion threshold and native
Enzyme remains unadjudicated—not because this harness proved that Zygote is
intrinsically faster than a correct standalone Enzyme implementation.

## Scope: candidate kernel, not production learner

- Julia 1.12.6, Lux 1.31.4, Zygote 0.7.11, Enzyme 0.13.186, Reactant 0.2.274,
  Optimisers 0.4.7; 20 Julia threads and 10 BLAS threads.
- Tracked `CompactCandidateQ(channels=8, blocks=1, spatial_channels=2)`,
  165,051 parameters, and tracked standardized masked ListNet objective.
- Fixed action width 74, state batch 4, fixed C13 rows
  `[1, 750, 1501, 2160]`, C11b checkpoint, and AdamW `1e-3`.

This is a candidate systems kernel.  The tracked trainer defaults to state
batch 2, samples dynamic rows, and later performs TD plus periodic ListNet
anchor updates.  Data sampling/packing, changing batches, TD/anchor control
flow, validation, and checkpointing are outside this benchmark.  The report's
earlier use of “production state batch 4” was incorrect and is withdrawn.

## Valid Zygote versus Reactant result

“Timed n” is the sum of the individually timed calls.  “Full loop” additionally
includes harness work around those calls.  The first call is compile-inclusive.

| backend | first | warm median | warm allocation | timed n=100 | timed n=1,000 | full loop n=1,000 | peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| native Zygote | 27.3747 s | 63.886 ms | 48,471,767 B | 34.585 s | 96.455 s | 97.210 s | 1.073 GiB |
| Reactant + EnzymeMLIR | 69.9382 s | 14.088 ms | 37,424 B | 71.539 s | 84.256 s | 85.021 s | 1.638 GiB |

The timed-call cumulative traces cross and remain crossed at update 773:
Zygote `81.130252 s`, Reactant `81.086809 s`.  This is only a timed-call
crossover; the harness did not record a per-update full-loop wall trace, so no
full-loop crossover update is claimed.

At 1,000 updates, timed-call speedup is
`96.454709 / 84.256107 = 1.144780x`.  Full-loop endpoint speedup is
`97.209759 / 85.021201 = 1.143359x`.  Both are below 1.15x.  Reactant observed
one cached compiled thunk across 1,000 calls, and its explicit post-call
loss-and-parameter barrier was roughly 10 microseconds median.  No recompile or
fallback warning appeared in the captured log.

## Reactant numerical conformance

Across the complete 1,000-element saved loss traces, maximum absolute Reactant
versus Zygote loss difference is `2.377033e-4` at update 677
(`3.126149416` versus `3.125911713`).  At the recorded checkpoints
1/2/3/4/5/10/100/500/1,000, maximum error is `3.814697e-6`; the earlier report
incorrectly generalized this checkpoint tolerance to the full trace.

At update 1,000, Reactant parameters versus Zygote have cosine
`0.9999995295`, maximum absolute error `0.0116125`, and relative L2
`0.00097005`.  These establish close but non-identical behavior for this fixed
kernel; they are not a complete production-trajectory gate.

## Why the native Enzyme result is invalid

The attempted path used `set_runtime_activity(ReverseWithPrimal)`, `Active`
scalar return, `Duplicated(parameters, shadow)`, `Const` model/state/objective/
batch, one preallocated shadow recursively cleared before each call, and no
duplicate forward.  Static activity failed inside the tracked model's
`reshape` with `EnzymeRuntimeActivityError`.

Despite the intended annotations, the runtime-activity gradient-only trace
proves that the current implementation is stateful or mutating: a correct
fixed-parameter gradient benchmark cannot change its own subsequent primal
loss.  The preserved artifact did not hash each primal object before and after,
so it does not identify whether parameters, state, batch, or Enzyme-internal
state changed.  Its first-call gradient happens to be close to Zygote—cosine
`0.999999999990`, maximum error `2.302e-6`—but that does not validate repeated
execution.  Consequently all of the following earlier claims are withdrawn:

- that this was the strongest correct standalone Enzyme path;
- that its warm/n=100/n=1,000 timings show native Enzyme is slower;
- that its divergent AdamW trajectory is caused by near-zero gradient
  sensitivity rather than the observed statefulness;
- that its batch-16 trace is a valid scaling comparison.

The old direct-Enzyme artifacts are retained as failure evidence.  They report
69.681 s first update, 107.706 ms warm median, and 183.743 s timed n=1,000, but
these numbers must not be used for backend selection.

The old strong-zero result is also withdrawn.  `numerics.jl` evaluated
strong-zero after the runtime mode had already mutated the shared problem, so
loss `4.399473` and cosine `0.304645` do not isolate strong-zero behavior.  No
fresh strong-zero run was added under the correction timebox.  The corrected
numerical driver runs one selected Enzyme mode per fresh process and checks
primal parameters, state, and batch for mutation.  The fixed-input gradient driver
also requires repeat loss stability; native update timing aborts before
optimizer application when its primal-content guard fails.

This failure does not prove that standalone Enzyme cannot be implemented
correctly.  A future adjudication needs a fresh-process path whose primal
content hashes remain unchanged, whose repeated fixed-parameter loss is
constant, and whose one-step and long optimizer trajectories clear declared
tolerances before timing is interpreted.

## Commands and artifact provenance

The preserved raw artifacts were generated by commit `f9c79b0` with commands
of this form from `C:\Users\fshuu\Documents\tetris`:

```powershell
$env:AD_BLIND_OUTPUT_ROOT='D:\tetris-paper-plus\runs\ad_blind-rerun-2'
$env:AD_BLIND_STATE_BATCH='4'
$env:AD_BLIND_STEPS='1000'
$env:AD_BLIND_BACKEND='zygote' # then enzyme_runtime
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_native.jl
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_reactant.jl

$env:AD_BLIND_CALLS='100'
$env:AD_BLIND_BACKEND='zygote' # then enzyme_runtime
julia --threads=20 --project=experiments/ad_backend_retry_2026 experiments/ad_blind_rerun_2/run_gradient.jl
```

The corrected current native scripts intentionally abort on detected primal
mutation, so they will not reproduce the invalid completed native artifacts.
All raw files remain under `D:\tetris-paper-plus\runs\ad_blind-rerun-2`.

| artifact | SHA-256 | interpretation |
|---|---|---|
| `zygote_b4_n1000.json` | `47144ac44126feab9d32c07c88ad359e44d213d36cb1455fac3de097648e5871` | valid fixed kernel |
| `reactant_b4_n1000.json` | `f41114f545075543a06c414ab3bff31765d746ec38739e3890988e4582483328` | valid fixed kernel |
| `gradient_zygote_b4_n100.json` | `d979b9fe14a8e0c29eae0ec4e2cad89051106f2ead442f0f02e2f29a52c6ac1b` | valid fixed gradient |
| `gradient_enzyme_runtime_b4_n100.json` | `efb2430e0a81fe434ed51eb6d07c7f38b01a45402297ad3b3dea025dd195cdd1` | mutation failure evidence |
| `enzyme_runtime_b4_n1000.json` | `ff121d86f171b8e18e3e102dc246572e176f9e5fc91a5d0da8aa54c4b63685c6` | invalid for performance comparison |
| `numerics_b4.json` | `847ef94460765b3d87a72c3dd9ad09b69af6364613a5fb4596c6435b1cdecbcc` | runtime then contaminated strong-zero |
| `zygote_b16_n100.json` | `ed72318480a8cba0bc0e1c7e220ae17db94f8eda7f90d127488c4ffbead6060f` | valid Zygote scaling datum |
| `enzyme_runtime_b16_n100.json` | `c1ff9980ae07a67bc1e04e8f8c9da5cd404c3da05dafdd0c4adedca2411e6ed7` | invalid for comparison |

Input hashes remain recorded inside each artifact.  The active
`experiments/online_counterfactual_top2_r1` directory was not modified.
