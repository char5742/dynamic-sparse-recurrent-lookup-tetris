# Blinded second AD rerun

This rerun independently audits the tracked 165,051-parameter modern Tetris
learner.  It imports the exact frozen workload definition and package Manifest
from `experiments/ad_backend_retry_2026`, recording both hashes in every output,
but does not import that experiment's conclusions or artifacts.

The primary benchmark is a candidate fixed-row kernel: state batch 4 with 74
candidates per state.  It is not the tracked production learner, whose trainer
defaults to state batch 2, samples dynamic rows, and later runs TD plus ListNet
anchor updates.  Every backend here uses the same checkpoint, fixed rows,
objective, AdamW configuration, thread counts, and warm exclusion.

Native Enzyme is direct standalone `Enzyme.autodiff` with
`ReverseWithPrimal`, an `Active` scalar return, `Duplicated` parameters, and
`Const` model/state/batch/objective.  It preallocates one shadow, zeros every
array in that shadow before every derivative, returns the primal loss from the
same differentiated execution, and includes `Optimisers.update!` in timing.
Runtime activity is used only because the unchanged tracked model fails static
activity lowering.  The initial rerun later proved that this direct runtime
path is stateful across nominally fixed-parameter gradient calls; the preserved
artifacts do not identify the mutated object or internal state.  Its timings
are invalid.  Current scripts check primal parameter/state/batch contents,
check repeated fixed-input loss stability, and abort on failure.  The numerical
driver runs exactly one selected Enzyme mode per fresh process to prevent
cross-mode contamination.

Reactant uses a persistent returned `TrainState`, `return_gradients=false`, and
`sync=true`.  The same optimizer step is fused into the compiled whole learner
update.  All raw outputs are written to
`D:\tetris-paper-plus\runs\ad_blind-rerun-2`.
