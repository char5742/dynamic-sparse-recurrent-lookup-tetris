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
