# Failure and limitation ledger

This file is append-only for the independent retry. A failed compiler path or
timeboxed shape is retained here with its command and log path.

- `2026-07-18`, native numerical gate, state batch 4: harness launch failed
  before Julia started because PowerShell `Tee-Object` could not create a log
  in the not-yet-created `artifacts/` directory. This is an instrumentation
  failure, not an AD result. The directory was then created and the exact
  workload was relaunched.

- `2026-07-18`, direct native Enzyme static-activity numerical gate, state
  batch 4: failed after 55 seconds with `EnzymeRuntimeActivityError` at
  `CompactCandidateQ`'s active `reshape` (`compact_model.jl:52`). The
  parameter shadow was preallocated and activities were `Active` return,
  `Duplicated` parameters, and `Const` objective/model/state/data. Full
  compiler diagnostic: `artifacts/native_numerics_b4_static_failure.log`. Per Enzyme's
  diagnostic and FAQ, the retry proceeds with the same annotations and
  `set_runtime_activity(ReverseWithPrimal)`; the static failure remains part
  of the compatibility result.

- `2026-07-18`, native Enzyme with runtime activity, state batch 4: both the
  direct preallocated/reused-shadow path and Lux's cached AutoEnzyme path
  completed 100 finite updates, but produced the same divergent trajectory:
  `3.443775 -> 4.375719 -> 4.507455 -> ... -> 4.413848`, versus Zygote ending
  at `3.127614` at update 100. Lux recursively zeroes its cached shadow in
  place; exact equality between direct and Lux losses rules out the original
  suspicion of stale shadow accumulation in the direct helper. Artifacts:
  `enzyme_direct_runtime_b4_n100.*` and `enzyme_lux_runtime_b4_n100.*`.

- State batches 32, 64, and 128 and the remaining state-batch-16 backends were
  not launched. The tracked model supports these shapes, but the sweep was
  stopped as low-ROI after native Enzyme failed the actual numerical/stability
  gate and the 1,000-update historical-shape comparison produced a decisive
  observed crossover. The completed Zygote state-batch-16 run is retained.
