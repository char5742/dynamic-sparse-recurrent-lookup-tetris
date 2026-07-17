# Lux / AD / compiler backend report

Status: completed. Lux + Zygote adopted; standalone Enzyme and
Reactant+EnzymeMLIR rejected by the registered time-to-target gates.

The experiment contract is in
`experiments/ad_backend/EXPERIMENT.md`. No result in this document is a game
score or a claim that the historical agent has been surpassed.

## Locked environment

- Julia 1.12.6
- Lux 1.31.4
- Zygote 0.7.11
- Enzyme 0.13.186
- Reactant 0.2.274
- Optimisers 0.4.7

The independent environment and Manifest are under
`experiments/ad_backend/`.  The fixed proxy has Tetris-shaped 24×10×2 boards,
four next candidates, Double-DQN online selection/target evaluation, Huber
loss, backward and Adam update.  It has 133,125 trainable parameters.  It is a
backend screen, not a substitute for the final model benchmark.

## Numerical gate (batch 16)

The first cold process produced the following results:

| check | Zygote | standalone Enzyme | gate |
|---|---:|---:|---:|
| loss | 0.3117593 | 0.31175932 | pass |
| first gradient wall time | 21.961 s | 58.391 s | compile-heavy, informational |
| gradient cosine | — | 0.999999999999966 | pass (>=0.9999) |
| max gradient absolute error | — | 7.08e-8 | pass (<=1e-4) |
| gradient relative L2 | — | 1.66e-7 | pass |
| one-update parameter max error | — | 3.89e-7 | pass (<=1e-4) |

Both paths are finite.  Standalone Enzyme is numerically correct on this fixed
loss; correctness is not the reason for rejection.

## Contaminated functional run

A five-update run overlapped the 20-thread teacher-dataset generator and is
explicitly marked as correctness-only in
`experiments/ad_backend/native_results.json`.  Its absolute rates are not
adoptable host benchmarks:

| backend | finite updates | median warm update | median allocation/update |
|---|---:|---:|---:|
| Zygote | 5/5 | 0.066 s | 34.9 MB |
| standalone Enzyme | 5/5 | 0.481 s | 189.1 MB |

Even under identical contention, Enzyme was about 7.3× slower and allocated
about 5.4× more.  Together with the cold 58.4-second first gradient, this is
already far below the required 1.15× speedup.  Standalone Enzyme is therefore
**rejected**; spending a long 100-update run on it would not change the
time-to-target decision.

## Clean quiet-window results

The main and learning agents held other heavy Julia work while these runs were
executed.  At each launch no other Julia process existed.  Julia used one
thread and BLAS reported ten threads.

### Zygote

Zygote completed 100/100 finite updates at every requested shape:

| batch | cold/warm first update | steady median | updates/s | batch-items/s | measured 100-update sum | first + 99×median | median allocation | cumulative peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 10.686 s (cold) | 0.06839 s | 14.62 | 234.0 | 19.054 s | 17.456 s | 34.9 MB | 1.48 GB |
| 32 | 0.387 s (warm) | 0.12289 s | 8.14 | 260.4 | 15.008 s | 12.552 s | 64.6 MB | 1.65 GB |
| 64 | 0.231 s (warm) | 0.25010 s | 4.00 | 255.9 | 29.674 s | 24.991 s | 123.8 MB | 1.72 GB |
| 128 | 0.498 s (warm) | 0.49641 s | 2.01 | 257.9 | 56.415 s | 49.642 s | 242.4 MB | 1.85 GB |

Only batch 16 includes the first Julia specialization of the update; later
shapes reuse the same array types.  `first + 99×median` is the registered
compile-inclusive learner estimate, not full Julia/package startup.  The RSS
column is cumulative process peak because all four shapes ran in one process.
Batch 32 gave the best item throughput, while batch 16 retains the highest
optimizer-update rate and the historical batch semantics.

Machine-readable evidence:
`experiments/ad_backend/zygote_clean_results.json`.

### Standalone Enzyme

The clean batch-16 short run completed 10/10 finite updates:

| first update | steady median | updates/s | projected first + 99×median | median allocation | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 19.529 s | 0.34333 s | 2.91 | 53.519 s | 189.1 MB | 1.72 GB |

Relative to clean Zygote batch 16, standalone Enzyme was 5.02× slower per warm
update, allocated 5.41× more, and had a 3.07× worse compile-inclusive
time-to-100 projection.  The registered early-stop criterion therefore ended
the run at ten updates.  Its excellent gradient agreement does not compensate
for this time-to-target loss.

Machine-readable evidence:
`experiments/ad_backend/enzyme_clean_results.json`.

### Reactant + EnzymeMLIR

Reactant executed the complete fixed-shape update five times.  Its loss path
was:

```text
0.31175932 -> 0.27252254 -> 0.24093223 -> 0.21491337 -> 0.19014841
```

The native five-update reference ended at `0.19014841318130493`, so the
observed loss trajectory agrees to displayed Float32 precision.  The first
compiled full update took **65.340 s**.  That single update is already 3.74×
slower than Zygote's complete registered first-plus-99-median estimate of
17.456 s.  Even assuming zero cost for Reactant's remaining 99 updates, it
cannot pass the compile-inclusive time-to-100 gate.

Later reported launch times (`0.164 s`, then roughly `0.00025 s`) are not
throughput measurements: host synchronization of the loss occurred outside
the timed region.  They are deliberately excluded.  The compiled optimizer
donated its gradient buffers, so attempting to copy the returned gradient to
host after subsequent updates failed.  A third compatibility attempt was not
made: the compile-time gate had already rejected the backend, and the contract
forbids an open-ended Reactant/Windows chase.

Machine-readable evidence and the instrumentation limitation:
`experiments/ad_backend/reactant_result.json`.

## Final decision

Adopt **Julia 1.12.6 + Lux 1.31.4 + Zygote 0.7.11** for the learner.  It is the
only backend that passed numerical correctness, 100-update stability, memory,
and compile-inclusive time-to-target gates.  Use batch 16 where historical
semantics or update frequency matters; benchmark batch 32 in actual training
because it delivered the best proxy item throughput.

Reject standalone Enzyme on steady speed and allocation.  Reject Reactant on
compile-inclusive time-to-100, without using its invalid asynchronous launch
timings as evidence.  No held-out game test seed was used in this backend
screen.

## 2026-07-18 independent retry and revised long-run decision

The earlier Reactant result correctly rejected its asynchronous timings, but
it did not test the optimized persistent Lux path.  A clean independent retry
therefore separated three questions: Native Enzyme, a synchronized fixed-shape
proxy, and the actual C13 fixed74 ListNet update.  Raw scripts, manifests, logs,
and machine-readable results are under `D:\tetris-paper-plus\ad_retry`.

### Native Enzyme remains rejected

The retry used `ReverseWithPrimal`, a preallocated/reused parameter-gradient
shadow, `Const` model/state/data/target parameters, and no runtime activity.
It also tested Lux's official cached Training API, which already retains and
zeroes its gradient shadow in place.  Numerical agreement remained excellent:
gradient cosine `0.999999999999966`, maximum absolute gradient error
`7.08e-8`, and one-step parameter maximum error `3.89e-7`.

| 100-update Lux Training API metric | Zygote | Native Enzyme |
|---|---:|---:|
| first update | 15.610 s | 19.508 s |
| steady median | 66.004 ms | 351.065 ms |
| measured loop wall | 23.768 s | 58.508 s |
| steady allocation/update | 33.34 MB | 186.97 MB |
| peak working set | 1.697 GB | 1.962 GB |

Native Enzyme was 5.32x slower at steady state and allocated 5.61x more.  The
low-level explicitly preallocated path gave the same conclusion (5.18x slower,
5.61x more allocation).  Native Enzyme is therefore rejected; CPU use alone
does not explain away the result.

Evidence: `D:\tetris-paper-plus\ad_retry\native_enzyme\RESULT.md` and
`summary.json` in the same directory.

### Persistent Reactant wins the long proxy run

The corrected path used `single_train_step!`,
`return_gradients=Val(false)`, `sync=true`, and retained the returned
`TrainState`.  Lux cached one compiled function containing forward, TD loss,
EnzymeMLIR backward, and Adam update.  An additional barrier over loss and all
parameter arrays followed each timed call; its median was 8 microseconds.  One
compiled thunk served all 1,000 updates with no recompilation.

| compile-inclusive proxy updates | Zygote | Reactant + EnzymeMLIR |
|---:|---:|---:|
| 100 | 22.049 s | 67.391 s |
| 500 | 50.085 s | 69.290 s |
| 1,000 | 84.417 s | **71.526 s** |

Reactant paid 66.724 seconds for its first update but reached a 4.580 ms steady
median versus Zygote's 56.015 ms (12.23x faster).  Steady allocation was
28.7 KB versus 33.286 MB (about 1,161x lower).  Five-step loss maximum error
was `4.47e-8`; parameter cosine was `0.999999999999864`, relative L2
`4.44e-7`, and maximum error `5.26e-6`.  Interpolation between the registered
cumulative windows places break-even near update 799; it is not an exact
per-update observed crossing.

Thus the earlier short-horizon rejection remains correct at 100/500 updates,
but the claim that Zygote is the faster long fixed-shape backend is overturned.

Evidence: `D:\tetris-paper-plus\ad_retry\reactant_enzyme\RESULT.md` and
`summary.json` in the same directory.

### Actual C13 fixed74 ListNet screen

The real screen used the tracked 165,051-parameter `CompactCandidateQ`, action
width 74, state batch 4 (296 padded candidates), temperature 1, the C11b warm
checkpoint, the tracked standardized masked ListNet loss, and AdamW at 1e-3.
Both backends consumed the same four fixed aggregate-C13 rows and initial
parameters.  This run used 20 Julia threads and 10 BLAS threads.

| compile-inclusive actual-loss updates | Zygote | Reactant + EnzymeMLIR |
|---:|---:|---:|
| 100 | **29.264 s** | 68.726 s |
| 500 | **55.906 s** | 80.626 s |
| 1,000 | **88.970 s** | 95.850 s |

Reactant did not recover its compile cost within 1,000 updates.  Its steady
median nevertheless improved from 60.238 ms to 30.128 ms (1.999x), and steady
allocation fell from 48.347 MB to 37.9 KB (about 1,276x).  The last-500 measured
rates project cumulative break-even near update 1,193; this is an extrapolation,
not a measured crossing.  Five-step loss maximum error was `9.54e-7`, parameter
cosine `0.999999998381`, relative L2 `5.69e-5`, and maximum error `1.67e-3`.
The update-1,000 loss differed by only `2.86e-6`.  One thunk served all 1,000
finite updates with no recompilation.

This screen deliberately repeats one fixed batch.  It excludes per-update row
sampling, `candidate_batch` construction, changing-batch host/device transfer,
validation, checkpoint export, and the rest of the actual training pipeline.
Those costs can reduce or erase the 2x update-only gain.  It also establishes
compatibility only for this compact model/loss; future FiLM, SE, distributional,
or replay objectives require their own lowering smoke.

Evidence: `D:\tetris-paper-plus\ad_retry\reactant_real_listwise\RESULT.md`,
`summary.json`, and `INTEGRATION_PROPOSAL.md` in the same directory.

### Revised backend policy

- Keep Native Julia + Zygote for short or dynamic experiments and runs below
  roughly 1,200 optimizer updates.
- Keep Native Enzyme rejected.
- Promote Reactant + EnzymeMLIR to a persistent **changing-batch, transfer- and
  checkpoint-inclusive** integration screen for runs of at least 1,200 updates.
- Do not claim production adoption until that complete pipeline is at least
  1.15x faster with matching changing-batch numerical trajectories.

No game, validation seed, or held-out test seed was used in any retry.
