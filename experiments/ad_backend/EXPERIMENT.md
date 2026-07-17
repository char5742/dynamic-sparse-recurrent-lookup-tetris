# AD backend experiment contract

## Scope

Compare a fixed-shape Lux learner update (`online forward -> Double-DQN-style
target selection/evaluation -> Huber loss -> backward -> Adam update`) on the
same deterministic model, initial parameters, state, and batch for batch sizes
16, 32, 64, and 128.

The benchmark model is intentionally smaller than the historical 20.8 M
parameter actor network so every backend can be screened quickly on the local
CPU. Backend promotion is about relative learner-update behavior; it is not a
game-score claim.

## Hypotheses

1. Native Lux + Zygote is the compatibility baseline and should complete all
   numerical/stability checks.
2. Standalone Enzyme (direct `Enzyme.autodiff`, not Lux `AutoEnzyme`) can reduce
   steady-state CPU update time and allocations while agreeing numerically with
   Zygote.
3. Reactant + EnzymeMLIR can fuse the complete fixed-shape update. Its one-time
   compile cost may be large, but steady-state updates should amortize it. On
   Windows it is experimental enough to warrant a strict stop limit.

## Success metrics

- 100 consecutive finite updates without recompilation or failure.
- Initial loss relative difference <= 1e-5 versus Zygote.
- Gradient cosine >= 0.9999 and maximum absolute gradient error <= 1e-4.
- One-update parameter maximum absolute difference <= 1e-4.
- Adopt a non-baseline backend only if median steady update throughput is at
  least 1.15x Zygote without materially higher peak RAM or instability, and
  compile-inclusive time-to-100 updates is lower than Zygote.

## Time limits and stop conditions

- Environment resolution/precompile: 20 minutes total.
- Native Enzyme compatibility: stop after 20 minutes or two distinct compiler
  failures on the fixed model.
- Reactant: stop after 15 minutes total, 10 minutes for any single fixed shape,
  or two crashes/recompilation failures. Do not migrate to WSL in this screen.
- Per backend/batch benchmark: warmup plus 100 updates, with a 10-minute cap.
- Abort any run on non-finite loss/parameters, >1e-3 numerical disagreement,
  or an update slower than 10 seconds after compilation.

## Required outputs

`results.json` records versions, machine/thread configuration, compile time,
updates/s, allocation estimates, peak working set, numerical comparisons,
stability, and exit reason. `reports/ad_backend.md` is the human-readable
decision record. Failures are retained.
