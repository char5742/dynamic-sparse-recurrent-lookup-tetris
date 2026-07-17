# S1 evaluator accounting hardening

## Scope

This change prepares the sole authorized S1 development screen from review
`a52a083` without executing it. The action-selection mechanism remains top-2
one-step Bellman reranking with blend 0.5, discount 0.997, and an infinite
margin threshold. Runtime overrides that differ from those constants are now
rejected instead of silently creating an unregistered variant.

## Accounting definitions

- `root_candidate_evaluations`: candidates scored in each decision's initial
  old-Q candidate set.
- `successor_candidate_evaluations`: candidates scored across the generated
  successor sets for nonterminal selected root branches.
- `lookahead_expansions`: number of nonterminal selected root branches for
  which a successor candidate set was generated, including an empty set.
- `logical_model_passes`: one for a nonempty root set and one for each nonempty
  successor set.
- `physical_backend_requests`: for each independently scored root or successor
  set, `ceil(candidate_count / 16)` OpenVINO requests; a short tail counts as
  one CPU request and an empty set counts as zero.

The legacy `candidate_count` and `lookahead_candidate_count` fields remain and
mean root and successor candidate evaluations respectively.

## Artifact identity and budget

The JSON filename contains device, all S1 constants, seed, NEXT count, and
`max_steps`. JSON records the historical checkpoint's absolute path, bytes, and
SHA-256 fingerprint; backend/device/batch; the fixed search constants; search
budget; generation, inference, and wall time; and the five accounting fields.

## Verification

`julia --project=. --threads=4 test\\runtests.jl` passed all 61 tests. The 13
S1-specific tests cover exact root/successor accounting across full, short-tail,
and empty successor sets; invalid counts; rejection of every altered S1
constant; and collision-resistant 50/100-step filenames. No game, checkpoint
inference, development seed, validation seed, or test seed was executed.
