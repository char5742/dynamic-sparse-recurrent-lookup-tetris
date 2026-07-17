# Blinded second AD rerun

This rerun independently audits the tracked 165,051-parameter modern Tetris
learner.  It imports the exact frozen workload definition and package Manifest
from `experiments/ad_backend_retry_2026`, recording both hashes in every output,
but does not import that experiment's conclusions or artifacts.

The primary production candidate is state batch 4 with 74 candidates per state.
Scaling screens use state batches 16, 32, 64, and 128.  Every backend uses the
same checkpoint, fixed rows cycled to the requested shape, objective, AdamW
configuration, Julia/BLAS thread counts, five-update warm exclusion, and full
update timing.

Native Enzyme is direct standalone `Enzyme.autodiff` with
`ReverseWithPrimal`, an `Active` scalar return, `Duplicated` parameters, and
`Const` model/state/batch/objective.  It preallocates one shadow, zeros every
array in that shadow before every derivative, returns the primal loss from the
same differentiated execution, and includes `Optimisers.update!` in timing.
Runtime activity is used only because the unchanged tracked model fails static
activity lowering.  Strong-zero runtime activity is independently screened.

Reactant uses a persistent returned `TrainState`, `return_gradients=false`, and
`sync=true`.  The same optimizer step is fused into the compiled whole learner
update.  All raw outputs are written to
`D:\tetris-paper-plus\runs\ad_blind-rerun-2`.
