# N1 pre-marker engineering smoke result

Status: **`N1-engineering-smoke-pass`** after one clean-audited,
duplicate-preserving identity correction. This is a plumbing result only. No
N1 experiment marker was created, no model was trained or changed, and this is
not evidence for or against a top-2 neural gate's scientific hypothesis.

## Frozen scope

- Engineering-only seed: `73200`.
- Old policy: exact exported 1313 OpenVINO graph, NPU static batch 16 and
  actual-size dynamic CPU tail, NEXT=5, HOLD enabled, stable candidate order.
- Branch contract: `GameState(root)`, force old top-1/top-2 respectively, then
  up to 11 more frozen-old-policy placements, unbootstrapped `G12` with
  `gamma=0.997`.
- Candidate representation: the real C13
  `CompactCandidateQ.head.layers.layer_2` post-swish tensor, shape `64 x B`.
- Learning plumbing: one discarded 64-weight plus one-bias logistic update;
  label-free fallback/top-2 choice.
- Limits: 300 s, process-tree private committed memory <=4 GiB, working set
  <=2 GiB.

The evaluator and C13 source are included at Julia top level, before `main`.
The source contains no runtime `include`, `Core.eval`, `invokelatest`, or R1
engine-adapter import.

## Pre-execution launch corrections

Two failures occurred before seed/model observations and were retained rather
than hidden:

1. `n1_engineering_smoke_73200_20260717T235540Z`: Julia rejected a nested
   interpolation during whole-file parsing. Nothing in the file executed.
   Wall 5.798 s, peak private 1,806,356,480 B, peak working set 647,192,576 B.
2. `n1_engineering_smoke_73200_20260717T235709Z`: the real OpenVINO graph
   compiled, but scalar `"NPU"` was incorrectly converted to
   `["N", "P", "U"]`; the execution-device identity assertion stopped before
   `GameState`, seed, Q values, C13, return, or label. Wall 13.165 s, peak
   private 2,280,521,728 B, peak working set 1,169,149,952 B.

Both were explicitly classified as retryable engineering-launch failures by
the research coordinator. The fixes added recursive Julia parse-error
detection and tested scalar/list OpenVINO device normalization.

## Initial seed-observing smoke

Durable output:

`D:\tetris-paper-plus\runs\n1_engineering_smoke_73200_20260717T235931Z`

The fresh process successfully:

1. loaded the evaluator statically without a world-age workaround;
2. compiled the real old-Q model on OpenVINO NPU with CPU tail;
3. created `GameState(Xoshiro(73200))`;
4. generated and scored the root candidates;
5. obtained distinct stable old top-1/top-2 actions;
6. cloned the root with `GameState(root)`;
7. forced the root top-1 and verified replay against the node afterstate.

At the first old-policy continuation state of the top-1 branch,
`stable_node_key.(nodes)` contained a duplicate and the deliberately strict
`allunique` invariant stopped the smoke:

```text
ERROR: stable candidate keys are not unique
score_nodes -> branch_rollout -> main
```

That source revision was not retried. It therefore did **not** reach a complete
G12 pair, label/advantage,
redacted label digest, C13 checkpoint load, 64x2 extraction, discarded update,
or gate decision. No `smoke_result.json` exists. No label or return value was
printed or persisted.

The independent identity audit found that `stable_node_key` is an ordering key,
not a uniqueness assertion. It authorized exactly one plumbing correction that
preserves candidate multiplicity and historical 16-candidate chunk boundaries.
No candidate could be deleted, merged, or secondarily reordered.

## Authorized duplicate-preserving correction and passing smoke

The correction assigns each candidate a state-local reference:

```text
(1-based ordinal, stable-key digest, action digest, afterstate digest)
```

Each decision binds its root-state digest, candidate count, and ordered vector
digest. The exact same ordinal binds raw candidate tensor position, OpenVINO Q
output, historical chunk/within-chunk/actual-tail coordinates, selected node,
and replay afterstate. Repeat runs compare ordered-vector, Q-binding,
selected-instance, and replay sequence digests. Duplicates remain separate.

A fresh clean agent audited this source and returned GO. The one authorized
fresh smoke then passed:

`D:\tetris-paper-plus\runs\n1_engineering_smoke_73200_20260718T001242Z`

Verified results:

- OpenVINO reports accelerator `NPU` and tail `CPU`.
- The root retained all 34 candidates in exact stable-sort order; Q was finite.
- Strict first-max top-1/top-2 selected distinct ordinals.
- Both forced branches and their frozen-old-policy continuations completed
  finite unbootstrapped G12 computation with deterministic ordered candidate,
  Q/chunk, selected-instance, and replay digests.
- Only redacted label evidence digest
  `60f752e7f3df928dab3fbfb5805d3e14aba3dda594b35ab05780c454dac0f9e8`
  was persisted. Neither G12 values, label, nor advantage was printed/stored.
- The real frozen C13 update-250 checkpoint loaded unchanged. Raw candidate
  inputs had the six expected shapes; its actual post-swish layer-2
  representation was finite, deterministic `64 x 2`, and the final layer
  reconstructed the full forward output exactly.
- One finite 65-parameter logistic-head update ran and was discarded.
- The label-free pre-update gate executed fail-closed fallback; it did not use
  the label or return advantage.
- All three bound model artifacts remained hash-identical and no N1 marker was
  created.

## Tests

Before the corrected fresh smoke:

- Julia pure/core tests: 22/22 pass, including duplicate-key preservation and
  ordinal/chunk binding fixtures.
- Julia complete-source parse check: 1/1 pass, recursively rejecting
  `Expr(:error)` and `Expr(:incomplete)`.
- Python static loading/world-age/source contract tests: pass.
- Independent C13 access check: the frozen update-250 checkpoint has 165,051
  parameters and its actual layer-2 post-swish output is 64-dimensional; on a
  real C13 row, applying the final layer to it exactly reconstructed the full
  forward output (max absolute difference 0). The corrected fresh smoke also
  exercised this exact step on the root top-1/top-2 inputs.

## Runtime evidence

| Field | Corrected passing smoke |
|---|---:|
| Monitor wall | 37.4514132 s |
| Smoke main | 27.7960000 s |
| NPU/CPU model compile | 1.09599996 s |
| Samples | 89 at nominal 200 ms |
| Peak process-tree private | 2,608,627,712 B (2.429 GiB) |
| Peak process-tree working set | 1,301,626,880 B (1.212 GiB) |
| Private cap | 4,294,967,296 B |
| Working-set cap | 2,147,483,648 B |
| Timeout | 300 s |
| Cap/timeout violation | none |
| Process exit code | 0 |

The runner now owns a native `System.Diagnostics.Process` handle. This fixed
the retained `Start-Process` null-exit-code defect; the passing monitor records
integer exit code 0 and `smoke_result_exists: true`.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `smoke.jl` | `b49cb23448258d21ffbf8cfe685ccea0a4d2bb761c076fcf48022b56cea6f006` |
| `n1_smoke_core.jl` | `a69a98c534bc145498e36f029252c93bb4d905d1fe6b18a4d7a00b5c3b30cd18` |
| `run_monitored_smoke.ps1` | `d222bc82798e13d08f709059086d98ab048de1e6c93551ac79d30974843617df` |
| `test_n1_smoke_core.jl` | `b43ebfe59567044ae24bb01d6e4ab655fb57d47ef07d5db7b9cc10379641c4f2` |
| `test_static_contract.py` | `bb66384954e566a2e642885d80e12ad52c38c629b776569270db9a07c8e1a212` |
| passing `monitor_result.json` | `5c619c29ebf7554c3b9613ca4366876fbfc366c24f4e033d0a941d21c2569378` |
| passing `resource_samples.jsonl` | `fb82404ee20c0bebb519d9d83242970f67377409f0fa99f65f2a23f4a012e4d8` |
| passing `smoke_result.json` | `d2b381d59680bace9f365958353c68737cb69c87cb9185af14ca535e94b4fb75` |
| passing empty `stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| passing `stdout.log` | `3c0b465d21f29d546157fd2f1fcb46a809e1b6c5b82e624cf0e05dcb3ada0e56` |

Bound identities remained read-only:

- C13 checkpoint:
  `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`.
- Old checkpoint:
  `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`.
- OpenVINO old weights:
  `2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba`.
- Manifest:
  `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`.
- Julia 1.12.6 executable:
  `4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043`.
