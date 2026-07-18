# N1 pre-marker engineering smoke: terminal result

Status: **`N1-engineering-smoke-FAIL`**. This is a plumbing result only. No
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

## Final actual smoke

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

Per the preregistered stop instruction, this seed/branch-observing smoke was not
retried. It therefore did **not** reach a complete G12 pair, label/advantage,
redacted label digest, C13 checkpoint load, 64x2 extraction, discarded update,
or gate decision. No `smoke_result.json` exists. No label or return value was
printed or persisted.

This failure must not be repaired by silently deleting or merging colliding
candidates: candidate multiplicity and 16-candidate chunk boundaries are part
of the historical LayerNorm policy semantics. A subsequent, scientifically
distinct design must adjudicate whether `stable_node_key` is only an ordering
key or must be extended with a deterministic tie-breaker that preserves every
candidate.

## Tests

Before the final actual smoke:

- Julia pure/core tests: 14/14 pass.
- Julia complete-source parse check: 1/1 pass, recursively rejecting
  `Expr(:error)` and `Expr(:incomplete)`.
- Python static loading/world-age/source contract tests: pass.
- Independent C13 access check: the frozen update-250 checkpoint has 165,051
  parameters and its actual layer-2 post-swish output is 64-dimensional; on a
  real C13 row, applying the final layer to it exactly reconstructed the full
  forward output (max absolute difference 0). The terminal N1 smoke itself did
  not reach this step.

## Runtime evidence

| Field | Final actual smoke |
|---|---:|
| Monitor wall | 20.7047779 s |
| Samples | 49 at nominal 200 ms |
| Peak process-tree private | 2,489,311,232 B (2.318 GiB) |
| Peak process-tree working set | 1,365,622,784 B (1.272 GiB) |
| Private cap | 4,294,967,296 B |
| Working-set cap | 2,147,483,648 B |
| Timeout | 300 s |
| Cap/timeout violation | none |

The PowerShell monitor recorded `exit_code: null` despite the parent shell
receiving exit code 1; the stderr stack and absence of `smoke_result.json`
remain the authoritative failure evidence. This monitor defect does not affect
the measured resource peaks but must be fixed before any future promotion
harness.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `smoke.jl` | `ecc2caedc5d1db66130e814c241783c55eb70861d6c08c764553973966d5c520` |
| `n1_smoke_core.jl` | `cf2d609ef30997bf646fc2f6f33d440e7f1bfc4a87100af75532256f1a53b6e6` |
| `run_monitored_smoke.ps1` | `e45f053229f94b223e01ecedcead8b36ebd4a4ef5f4e7b9f3b124369fb675211` |
| `test_n1_smoke_core.jl` | `5178fdcb1dfa390c79575a6b27f7dcc245a763f3adb2ce989e3959b80c8d4e8c` |
| `test_static_contract.py` | `947e5b81ea833d567a182210d91a891a0561d157856e6eade568375fd78b364b` |
| final `monitor_result.json` | `ffc3a2d7453348321285dd78b37e0441f3d107b6808cb3778d6264524f898024` |
| final `resource_samples.jsonl` | `7172ce17101a7ed1528b24e9c762988c75403aa95104c1d286f5db1c4e045fed` |
| final `stderr.log` | `8517165268d0a04be65e4017d6eb12055dac40ef295231fa7c51be5ac74b58ad` |
| final empty `stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

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
