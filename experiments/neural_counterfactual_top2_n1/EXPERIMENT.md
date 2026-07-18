# N1 pre-marker engineering smoke

This namespace contains a plumbing-only smoke for the scientifically distinct
N1 experiment. It does not create or consume an experiment marker and it does
not authorize training, calibration, development, validation, or sealed-test
evaluation.

Frozen engineering-only inputs:

- seed `73200`;
- old OpenVINO policy on NPU static batch 16 plus actual-size CPU tail;
- root stable top-1/top-2, followed by one forced root placement and up to 11
  old-policy placements per branch;
- unbootstrapped `G12`, `gamma=0.997`;
- C13 selected update-250 checkpoint, whose actual post-swish 64-dimensional
  head penultimate is extracted for the two raw root candidate tensors;
- one discarded 65-parameter logistic update;
- a label-free fail-closed deployment decision.

The return values, label, and return advantage are never printed or persisted.
Only a redacted evidence digest and finite/shape/determinism/identity checks are
written. The static contract test also rejects runtime evaluator loading,
`Core.eval`, `invokelatest`, and any R1 engine-adapter import.

Run tests:

```powershell
C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe --startup-file=no --history-file=no --project=C:\Users\fshuu\Documents\tetris C:\Users\fshuu\Documents\tetris\experiments\neural_counterfactual_top2_n1\test_n1_smoke_core.jl
D:\tetris-paper-plus\python-env\Scripts\python.exe C:\Users\fshuu\Documents\tetris\experiments\neural_counterfactual_top2_n1\test_static_contract.py
```

Run the one engineering smoke in a fresh monitored process:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\fshuu\Documents\tetris\experiments\neural_counterfactual_top2_n1\run_monitored_smoke.ps1
```

## Pre-execution launch record

The first monitored launch at `2026-07-17T23:55:40Z` was rejected by Julia's
parser at `smoke.jl:99`, before any top-level statement, evaluator import,
model, game state, seed, return, or label was reached. It is therefore an
engineering launch failure rather than a smoke observation. The durable output
is
`D:\tetris-paper-plus\runs\n1_engineering_smoke_73200_20260717T235540Z`.
Monitor wall time was 5.798 s, peak process-tree private committed memory was
1,806,356,480 bytes, and peak working set was 647,192,576 bytes. The nested
string interpolation was replaced with explicit `string` construction and the
unit suite now recursively rejects `Expr(:error)`/`Expr(:incomplete)` returned
by `Meta.parseall`.

The next launch at `2026-07-17T23:57:09Z` compiled the real OpenVINO model, but
stopped before `GameState`, seed `73200`, old-Q scoring, C13, a branch return, or
a label. OpenVINO reported scalar `"NPU"`; an overly broad conversion interpreted
it as the three-element string vector `["N", "P", "U"]`, and the strict identity
check rejected it. The durable output is
`D:\tetris-paper-plus\runs\n1_engineering_smoke_73200_20260717T235709Z`.
Monitor wall time was 13.165 s, peak private committed memory was 2,280,521,728
bytes, and peak working set was 1,169,149,952 bytes. Scalar and sequence device
properties now share a tested normalization function.
