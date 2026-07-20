# Heterogeneous pipeline runner v2

Status: **UNEXECUTED SOURCE ONLY**. This file and its sibling implementation
make no latency, throughput, residency, NPU, iGPU, or score claim.

`heterogeneous_pipeline_runner_v2.jl` is a new runner; it does not edit or
replace the frozen v1 `pipeline.jl`, `lifecycle.jl`, or
`windows_cpu_sets.jl`. Load v1 first, then include the v2 file as a sibling
module:

```julia
include("Heterogeneous265K.jl")
include("heterogeneous_pipeline_runner_v2.jl")
```

## Dataflow

The overlapped path owns two or three reusable slots and four bounded queues:

```text
E-core sticky worker
  pack/environment
        |
        v
one sticky HashStem broker
  fixed batch 16: one NPU call
  tail 1:15: CPU_TAIL in this same broker
        |
        v
P-core sticky worker
  route + gather + all sparse layers + active-only sparse forward/evaluation in one provider call
        |
        +----> result (default)
        |
        v
optional iGPU stem-training worker (externally gated, disabled by default)
```

This live CLI performs no parameter or optimizer updates; iGPU training remains fail-closed.

The runner never sends individual sparse layers to NPU or iGPU. The provider
interface deliberately exposes one `sparse` callback for the entire dynamic
sparse stack. A tail is not an NPU failure: it is explicitly sent through
`CPU_TAIL` in the same broker, slot, sequence and lineage. A failed full-batch
NPU HashStem call may separately be retried as one whole `CPU_FALLBACK` call
on that same immutable lineage.

## Identity and timing

Every item carries all of:

- slot ID and monotonically increasing slot generation;
- globally increasing sequence;
- caller-bound workload ID, payload SHA-256, order and candidate count;
- model ID, master version, snapshot version and master superstep;
- sparse-bank and sparse-index versions;
- SHA-256 identities for snapshot, bank and index.

A bank/index digest may remain at the same version only when it remains
byte-identical; changing either digest requires its corresponding version to
increase.

Every provider must return the exact same immutable token. Each queue/stage
boundary records raw QPC ticks, including queue entry, begin, end, frequency,
backend and sticky Julia thread ID.

Production providers are not constructible through the public source-test
provider constructor. Zero-field module seals and generic production sealers
are deliberately absent. Production identities use closure-captured identity
capabilities. The module exposes no generic provider or completion issuer. Its
one concrete builder accepts only the exact source-bound live CLI adapter,
binds the whole adapter/config/workload digest, and returns a
capability-authenticated completion receipt after the fixed adapter has
verified device synchronization and output readback. Public callback
constructors remain `MOCK_SOURCE_ONLY` and cannot start the live runner.

`summarize_v2_receipts` rejects failures, mixed lineages, duplicate sequences,
missing stages, binding/readback drift and mixed QPC frequencies. It emits
nearest-rank p50/p95 for complete end-to-end latency, each stage service
interval and each queue interval. It also records the aggregate interval from
minimum submission to maximum completion, item throughput and candidate
throughput. “Actual overlap” retains sequence, stage,
worker role, process, OS thread and Julia thread identities. It requires a
positive half-open interval intersection between different sequences on
different stages and workers, not a pooled maximum or merely multiple queued
items. Missing, ambiguous, mock or non-overlapping evidence closes admission
and drains into phase-separated mode.

Each live worker calls the existing transactional Windows CPU Set API, repeats
an explicit readback, records the topology digest and a point witness, and
fails startup if requested/readback IDs differ. Workers must land on distinct
sticky Julia/OS threads, and the P/E CPU Set readbacks must be disjoint. These
are application-level binding records, not whole-interval residency evidence;
ETW evidence remains required by the existing post-idle gate.

Runtime hooks carry an explicit kind. The live overlapped runner accepts only
the capability-authenticated, exact-identity Windows QPC/CPU-Set callbacks; mock hooks
cannot forge the production kind. Mock hooks are confined to the serial
source-test seam, are marked `MOCK_SOURCE_ONLY` in results, and are rejected by
production timing summaries and the overlap comparator.

## Slow-overlap fallback

`observe_overlap_comparator!` accepts only an external same-lineage end-to-end
comparison with an exactly equal caller-bound workload ID/digest/count/order
sequence that already includes packing, transfer, synchronization and
readback. Overlap must pass p50 and p95 and must not regress either item or
candidate throughput. Otherwise admission closes, existing work drains, then
the runner admits at most one job at a time.
`run_phase_separated_v2!` is the standalone real serial path and also the
hardware-free mock-test seam.

The serial seam requires a persistent `V2PhaseSequenceGuard`; sequence and slot
generation must advance exactly, and candidate workload identity cannot be
reused. The live runner starts end-to-end timing before waiting for its
phase/admission gate or a free slot, so p50/p95 includes bounded-ring
backpressure rather than beginning after admission.

The runner never starts Julia, Python, OpenVINO, or a child process itself. Its
sticky workers are tasks inside the already-authorized caller Julia process, so
launch control remains with the single-heavy-Julia gate.

## Tests and later execution

`test_heterogeneous_pipeline_runner_v2_contract.jl` supplies mock providers,
QPC and CPU Set hooks. It tests lineage rejection, exact binding readback,
single-broker NPU-to-CPU failure fallback, same-broker NPU batch16 plus explicit
CPU tail, one-call sparse execution, the iGPU gate, live two/three-slot
backpressure, a forced actual-overlap witness and positive `KEEP_OVERLAP`,
exact-workload fail-closed comparison, capability spoof rejection, sparse
digest/version coupling, and cleanup after worker binding failure.
It performs no hardware access and uses no dataset seeds.

No Julia test or hardware benchmark was run while the dense convergence job
was active. The next idle gate must run this source test first, then the live
two-slot and three-slot runner under the existing ETW/QPC/IMC evidence matrix.

## Concrete live CLI

`run_heterogeneous_pipeline_v2_live.jl` is the one concrete production entry
point. It is not a generic callback sealer. The v2 runner admits only its exact
`LiveAdapter` type, fixed stage dispatcher and validation function. The CLI
binds its own source SHA-256, the runner/provider/source closure, the exact
sparse checkpoint, HashStem master and snapshot, a teacher-v3 dataset manifest,
an ordered train-only workload manifest, and every item payload digest into one
provider-binding digest.

The ordered workload splits candidate sets into explicit chunks of 1–16
candidates. A complete 16-candidate chunk receives one NPU call; a 1–15 tail
uses `CPU_TAIL` in the same broker. The pack stage recomputes the item digest
from the set ID, candidate indices, action digests, raw sparse features and
packed HashStem matrix before admitting the physical result. The P-core stage
calls all three sparse layers once and records output/route digests plus exact
active-width accounting. It never makes a per-layer accelerator round-trip.

Live execution runs the exact workload first through the bounded overlapped
runner and then through the production phase-separated path. It requires
bitwise-identical sparse output and route digests, the same top action, and
production synchronized/readback receipts before calling the overlap
comparator. `receipts.jsonl` retains raw QPC stage receipts, CPU-set readbacks,
physical call receipts and result digests. `summary.json` retains the exact
workload sequence, source/input/config bindings, aggregate makespan,
items/candidates per second and the `KEEP_OVERLAP` or fail-safe phase decision.

The CLI starts no child process. PythonCall/OpenVINO remains in the caller
Julia process, and the runner uses only sticky tasks. It requires at least one
coordinator plus the three worker threads. The iGPU training option is
fail-closed because there is not yet a production in-process iGPU trainer; only
the explicit value `false` is accepted. No mock provider is reachable from this
entry point.

NPU mode additionally requires the exact byte-pinned `decision.json` emitted
by the direct k-scale gate. The CLI does not infer adoption from a status
string: both the outer and nested decisions must pass with empty reasons, all
route/action/state top-1/top-2 identity totals must match, errors must be
finite, the component speedup must remain at least 1.15x, execution devices
must be exactly `["NPU"]`, and the bound raw records must still hash exactly.
That gate authorizes only the measured `k128` sparse variant. CPU mode uses an
explicit disabled NPU-gate sentinel.

The workload schema is
`heterogeneous-265k-pipeline-v2-train-workload-v1`. It must bind the exact
teacher-v3 workload and dataset manifest, list train source parts and their
environment seeds, certify all development/validation/sealed flags false, and
exclude validation seeds `8001:8008` and sealed seeds `91001:91032`. Each item
must contain the exact contiguous order, unique ID, set index, candidate start,
candidate count and precomputed payload SHA-256.

This source remains **UNEXECUTED**. A successful CLI exit writes
`MEASURED_*_PENDING_INDEPENDENT_VALIDATION` with `adoption_allowed=false`; it
does not prove ETW residency, IMC attribution, model strength, or heterogeneous
adoption. The post-idle gate must first run
`test_heterogeneous_pipeline_live_cli_static.py`, then invoke the CLI exactly
once in the single-heavy-Julia slot with all SHA-256 arguments filled from the
frozen files. The output directory must not already exist.
