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

The first seed-observing source revision stopped at the first continuation
state because it incorrectly required `stable_node_key` to be unique. A clean
identity audit determined that the key orders candidates but does not uniquely
identify them. It authorized one correction: preserve every candidate in exact
stable-sort order and bind it by 1-based ordinal plus key/action/afterstate
digests through raw input, historical Q chunk, selection, and replay. No
candidate is deduplicated or secondarily reordered.

The one authorized corrected smoke passed at
`D:\tetris-paper-plus\runs\n1_engineering_smoke_73200_20260718T001242Z`.
It completed deterministic G12 branches, real C13 `64 x 2` penultimate
extraction, one finite discarded 65-parameter update, and a label-free fallback
decision. Only a redacted label-evidence digest was saved; no return value,
label, or advantage was printed or persisted. The native process monitor
recorded exit code 0 and remained below both resource caps. See `RESULT.md`.

## Frozen N1 calibration-only one-shot design

This section was written before any calibration-pipeline label was generated.
The successful engineering smoke at seed `73200` is plumbing evidence only and
is excluded from all fitting and calibration statistics. No development,
validation, or sealed-test seed is authorized by this design.

### Roles and schedule

- Training episodes: integer seeds `73201:73216` (16 episodes).
- Calibration episodes: integer seeds `73301:73306` (6 episodes).
- In each episode, schedule an opportunity immediately before pieces
  `20,40,...,220`, exactly 11 opportunities per episode.
- Scheduled totals are therefore 176 training and 66 calibration
  opportunities. Every scheduled opportunity must produce exactly one retained
  row or one structured exclusion. If an episode terminates early, all of its
  not-yet-reached scheduled opportunities are written as `early_terminal`
  exclusions. A root with fewer than two candidates is excluded explicitly.
- Training requires 150--176 retained rows inclusive and positive-label
  fraction 3--40% inclusive. Calibration requires 55--66 retained rows
  inclusive. No additional seed may be substituted.
- Between opportunities, and after each sampled root, the canonical episode
  always advances by the frozen old top-1 policy. The learned gate is evaluated
  offline and cannot change training/calibration state distribution.

### Candidate, return, and representation identity

At every retained root, use the exact historical `stable_node_list` order.
`stable_node_key` is an ordering key, not a uniqueness constraint: duplicate
keys and duplicate afterstates remain separate. Each candidate is bound by

```text
(1-based ordinal, stable-key digest, action digest, afterstate digest)
```

The decision context binds the full root-state digest, candidate count, and
ordered candidate-reference-vector digest. The same ordinal binds the raw
candidate tensor, OpenVINO Q output, historical batch-16 chunk position and
actual-size CPU tail, selected node, and replay afterstate. No deduplication,
padding, or secondary sort is permitted. Top-1/top-2 use a strict `>` scan;
ties retain first stable ordinal and the two selected ordinals must differ.

Clone only with `GameState(root)`. From the identical root/future source, force
`a1` or `a2`, then follow frozen old top-1 for 11 additional placements. Stop
the branch immediately on terminal/no-candidate state. With `gamma=0.997`,

```text
G12(a) = sum(k=0:11, gamma^k * score_delta_k / 600)
```

is unbootstrapped. The label is `y=1` iff `G12(a2)>G12(a1)`; ties are zero.
The safety flag records whether each branch terminated within the horizon.

Load the frozen C13 checkpoint read-only and extract its actual
`CompactCandidateQ.head.layers.layer_2` post-swish 64-vector independently for
the raw `a1` and `a2` tensors. The feature is `x=z2-z1`. The first retained
training sample at seed `73201` is immediately repeated from the unchanged root;
ordered candidates, Q/chunk bindings, selected instances, branch/replay traces,
G12 bit patterns, representations, feature, and label must match exactly.

### Frozen model and runtime identities

- Old checkpoint SHA-256:
  `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`.
- Old OpenVINO weights SHA-256:
  `2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba`.
- C13 checkpoint SHA-256:
  `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`.
- C13 model-source SHA-256:
  `793535dfc43e1c16a0b9196305f2e329438afc6aa458fdc7d8521ed1e36b1052`.
- Julia 1.12.6 executable SHA-256:
  `4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043`.
- Manifest SHA-256:
  `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`.

The entrypoint has static top-level evaluator/C13 includes, one fresh Julia
process, one OpenVINO NPU+CPU-tail setup, and one C13 setup. It imports no R1
code/artifact and uses no R1 or development/test seed.

### Frozen fit

Compute each feature's population mean and population standard deviation from
training rows only (`sqrt(sum((x-mean)^2)/n)`). A zero-variance feature
standardizes to exactly zero. Calibration uses these unchanged training
statistics.

Fit 64 slopes and one intercept in Float64 from all-zero initialization by
minimizing the **sum** binary cross-entropy plus
`0.5*lambda*sum(beta.^2)`, with `lambda=1`; the intercept is unpenalized.
Use deterministic Newton steps with Cholesky only, at most 100 iterations,
gradient infinity norm tolerance `1e-10`, initial line-search step 1, Armijo
`c1=1e-4`, shrink factor `0.5`, and at most 30 backtracks. No alternate solver,
jitter, pseudo-inverse, or solver switch is permitted. A non-positive-definite
Hessian, failed line search, non-finite value, or unconverged solve rejects N1.

Gate `a2` iff its logistic probability is finite and `p>=0.90` inclusive;
otherwise fall back to the exact old `a1` candidate instance.

### Frozen calibration gate

On the 55--66 retained calibration rows, all conditions must hold:

- override rate 1--15% inclusive;
- at least 8 overrides across at least 4 of the 6 episodes;
- override precision at least 0.75;
- one-sided 90% whole-episode-bootstrap lower bound for precision strictly
  greater than 0.50;
- mean override G12 advantage `G12(a2)-G12(a1)` strictly positive and its
  one-sided 90% episode-bootstrap lower bound strictly positive;
- zero overrides where `a2` terminates within the horizon and `a1` remains live;
- every fallback selects the exact old top-1 ordinal/instance bit-for-bit;
- median warmed gate-decision time at most 1 ms.

Bootstrap exactly 10,000 replicates. Each replicate uses
`Random.Xoshiro(UInt64(0x4e31_2026))` to sample all six episode IDs with
replacement and concatenates their retained rows; the fitted model is not
refit. A zero-override replicate has precision and advantage `-Inf`. Lower
bounds are the linear/type-7 0.10 quantile. Thresholds and inequalities above
cannot change after inspecting data. Passing is only
`N1-calibration-promoted`; it is not model/G1--G3 evidence and grants no
development run automatically.

### Runtime, projection, artifacts, and one-shot boundary

- The first 16 scheduled training opportunities produce the conservative
  projection `setup_seconds + post_setup_first16_seconds/16*242`; it must be
  <=3300 s. `setup_seconds` starts before static imports/model construction, and
  the first-16 rate includes the one repeat-sentinel cost rather than
  subtracting or amortizing it specially.
- Total setup + training collection + fit + calibration + finalization must be
  <=4500 s.
- One process-tree private committed cap 4 GiB and working-set cap 2 GiB apply.
- Training/calibration rows and exclusions are appended incrementally to JSONL;
  phase summaries, sentinel, fit, calibration metrics, stdout/stderr, resource
  telemetry, and terminal result are retained. Phase accounting must reconcile
  exactly to 176 and 66 scheduled opportunities.
- Production requires a clean exact source commit and a post-commit clean-audit
  readiness artifact. The PowerShell runner validates it, atomically creates
  `.experiment_state/n1_calibration_started.marker`, and refuses any existing
  marker. No label-generating command may run before that boundary.
