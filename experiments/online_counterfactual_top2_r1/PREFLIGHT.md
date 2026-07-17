# R1 pre-marker engineering preflight ledger

This ledger records failed as well as successful implementation trials. None
of the entries below is a scientific R1 run. No old model, Tetris game,
preregistered seed, dataset, validation/test seed, or global one-shot marker
was loaded or created.

## PF-001 — Julia analytic-fit process under the frozen 1 GiB cap

- date: 2026-07-18 JST
- purpose: determine whether `fit_ridge.jl` can be the production analytic
  fitter without changing the preregistered fit private-commit limit;
- invocation: the exact `fit_ridge.jl TABLE FREEZE ARTIFACT MILESTONES
  --synthetic` production-argv branch through the same suspended
  `CreateProcessW` -> Job assignment/verification -> resume runner intended for
  the one-shot;
- executable: concrete Julia 1.12.6 binary frozen by path, SHA-256, size, and
  process image;
- OS Job private-commit cap: 1,073,741,824 bytes;
- result: failed before `imports_complete`; child milestones reached only
  `script_enter`, `args_verified`, and `imports_begin`;
- sampled process-tree peak private commit: **1,073,725,440 bytes**;
- sampled process-tree peak resident working set: **286,724,096 bytes**;
- Windows Job `PeakJobMemoryUsed`: **1,377,767,424 bytes**;
- wall: **4.75 seconds**;
- interpretation: Julia package startup alone is not viable under the fixed
  1 GiB production-fit contract. Private commit is committed private address
  space, not resident RAM; resident working set stayed about 274 MiB.

Decision: do not relax the cap and do not call this an R1 scientific failure.
The production analytic ridge/calibration path moves to the already frozen
Python 3.12.13 environment with NumPy 2.4.6 and one BLAS thread. The Julia
implementation remains an independent numeric reference; production is not
eligible until coefficients, predictions, and decisions agree on the same
frozen synthetic schedules.

Synthetic debug artifacts were written under
`D:\tetris-paper-plus\runs\r1-fit-cap-debug`; they are diagnostic only and are
not an input to R1.
