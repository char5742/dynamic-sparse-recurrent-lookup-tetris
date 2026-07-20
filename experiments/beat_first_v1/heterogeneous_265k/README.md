# Core Ultra 7 265K heterogeneous HashStem prototype

This is a separate systems prototype. It does **not** replace or redefine the
pure-CPU dynamic-neuron experiment, and it contains no NNUE accumulator or
parent/child search reuse. The three tracks remain distinct:

1. the CPU-SLIDE model proves input-dependent neuron selection, active-only
   forward/backward, and active-only optimizer updates;
2. this directory pipelines that model with a compact fixed-shape NPU stem and
   P/E-core role separation; and
3. NNUE/delta-search reuse, if tested later, is a separate search optimization.

No performance result is claimed by the files in this directory. The OpenVINO
builder and benchmark are written but have not been run.

## Windows CPU Set substrate

`windows_cpu_sets.jl` provides the still-unexecuted Windows affinity substrate.
It calls `GetSystemCpuSetInformation` at each launch and records opaque CPU Set
ID, processor group, group-relative logical/core index, LLC/NUMA index,
efficiency/scheduling class, allocation flags, and allocation tag. CPU Set IDs
are never inferred from logical processor numbers or Julia thread IDs.

P/E naming fails closed unless the live topology contains exactly two intrinsic
efficiency classes, twenty usable one-thread cores, eight records in the higher
class and twelve in the lower class. Parked, real-time, foreign-allocated, SMT,
duplicate, or unexpected multi-group records reject role-specific assignment.
The resulting canonical JSON and SHA-256 are provenance, not placement proof.

Process and dedicated-thread APIs apply and immediately read back CPU Set IDs:

```text
SetProcessDefaultCpuSets  -> GetProcessDefaultCpuSets
SetThreadSelectedCpuSets  -> GetThreadSelectedCpuSets
```

The exported application functions accept `:p_sparse` or `:e_background`, not
raw IDs. They enumerate/classify immediately before assignment, re-enumerate
after readback, require the topology hashes to match, and restore the previous
assignment on failure. This closes stale-ID use across a discovery/assignment
race but still does not turn soft affinity into residency proof.

`GetCurrentProcessorNumberEx` plus `GetCurrentThreadId` creates a point witness.
It does not prove whole-stage residence; an ETW context-switch witness remains a
later requirement. CPU Sets are soft application affinity. Leaving one E-core
out of the application worker set is not an exclusive reservation for Windows
or the NPU kernel driver. The role contract therefore remains explicitly
`UNAPPLIED_UNVERIFIED` until a later exclusive run applies and measures it.

## Independent ETW and RAM-contention evidence

The P/E role split, ETW residency proof, and shared-RAM/IMC counters in this
directory belong only to the **265K heterogeneous systems gate**. They are not
properties of the pure-CPU dynamic sparse model and are not NNUE/search-delta
reuse. A systems result may therefore not be reported as a CPU-SLIDE model
improvement.

`windows_residency_ram_evidence_contract.json` and
`windows_residency_ram_evidence.jl` freeze the independent H0/H1 evidence
boundary. Their status is **UNEXECUTED_STATIC_ONLY**. They do not claim that an
ETW session or an IMC counter has been collected.

Whole-stage CPU residence uses Microsoft Windows Performance Toolkit
`xperf.exe`, the NT Kernel Logger/SystemTraceProvider
`{9e814aad-3204-11d2-9a82-006008a86939}`, `PROC_THREAD+CSWITCH`, and a QPC-clock
trace. A passing extractor must reconstruct every scheduled slice intersecting
each exact raw-QPC `(cell, repetition, sequence, set_id, stage, ordinal)`
interval, including carried-in boundary state. It binds PID plus ETW process
start, run nonce, TID plus ETW thread start, the pre-bound role manifest and
CPU-Set assignment/readback digest. Any lost event/buffer, circular overwrite,
unknown CSwitch version, unclassified benchmark thread, topology ambiguity,
unattributed tick, or disallowed role CPU Set fails closed. Point
`GetCurrentProcessorNumberEx` witnesses never satisfy this gate.

RAM traffic names one provider explicitly: Intel Processor Counter Monitor
(Intel PCM), its Windows driver, and `pcm-memory.exe`. Stock `pcm-memory` CSV is
**not** promoted to evidence. It does not by itself prove the raw per-channel
before/after counts, counter width, reset, overflow, multiplex status, and
enabled/running time required here. A version-bound raw PCM adapter must expose
all of those fields for every populated IMC channel. The validator recomputes
read/write bytes using 64 bytes per CAS count and recomputes bandwidth from the
bound QPC window. Process I/O bytes and theoretical DIMM bandwidth are forbidden
substitutes. IMC scope is recorded honestly as a system-wide exclusive window,
not as a process counter.

The validator hashes and parses the exact role-manifest bytes; every stage must
cover all manifest threads registered for that role, and caller-supplied owner
IDs or CPU Sets cannot select a favorable subset. IMC evidence must
match externally expected digests for the successful availability probe,
tool/driver/adapter, hook source and witness, both raw snapshots, and the
counter inventory/configuration. The raw counter latch may bracket the timed
region by at most 0.1% in total; the guard ticks are reported separately and a
larger window fails closed.

`windows_evidence_runtime_hook.jl` freezes the smallest runner boundary: a
hashed OS-thread role manifest plus an exact timed region bracketed by before
and after raw-counter acknowledgements. The two acknowledgements must retain
the same inventory, programming configuration, and generation. There is no
no-op controller. `probe_windows_residency_ram_tools.ps1` accepts only explicit
absolute paths, binds topology/system/benchmark/runner/provider/substrate/source-
manifest/result and raw-QPC hashes, probes xperf keywords and the two versioned extractor
capabilities, and writes a fresh artifact. In particular, if the required raw
PCM adapter/driver is not present, its only legal result is
`UNAVAILABLE_FAIL_CLOSED` / `INCONCLUSIVE_FAIL_CLOSED` with exit code 2.

`test_windows_residency_ram_contract.jl` is a source-only synthetic contract
test. It includes pass witnesses and rejections for lost ETW data, wrong P/E
residence, multiplexed/wrapped IMC counters, and unavailable tooling. It has not
been executed while the single-heavy-Julia run is active.

## Frozen post-idle result validator

`post_idle_result_validator.py` is an independent, standard-library-only
validator for frozen A0/H0/H1 result bundles. Its implementation status, the
matching `post_idle_validator_contract.json`, and
`test_post_idle_result_validator.py` are **UNEXECUTED_STATIC_ONLY**. The
validator does not import the Julia runner/provider or OpenVINO and does not
accept a runner summary as an adoption decision.

A validation bundle must bind the benchmark contract, frozen source manifest,
input witness, sparse checkpoint, HashStem master and all master members,
snapshot metadata/XML/BIN, system contract, provider/runner/substrate sources,
and A0/H0/H1 repetitions 1, 2, and 3. The validator reopens every referenced
byte, recomputes raw action/route ordering, FP32 outputs and stable full ranking,
physical NPU batch/tail/transfer/wait receipts, raw-QPC stage durations and
nearest-rank p50/p95, numeric tolerances, H1/H0 speed, p95, and CPU-sparse
slowdown. Each non-reference repetition must close over the matching A0 raw
records and summary digest.

ETW and IMC inputs are mandatory external evidence with separately bound
producer sources and raw trace/counter artifacts. Missing, incomplete, lost,
multiplexed, overflowed, or unauditable external evidence returns
`INCONCLUSIVE_FAIL_CLOSED`; complete contradictory/tampered evidence or a failed
numeric/performance gate returns `REJECT`. `ADOPT` is possible only when every
source, raw-record, numeric, rank, performance, ETW, and IMC gate passes, and it
authorizes only the H1 systems gate. It is not A1/A2, pure-CPU model, NNUE,
teacher, or game evidence.

The frozen command shape is:

```text
python post_idle_result_validator.py \
  --bundle-manifest <validation_bundle.json> \
  --bundle-manifest-sha256 <sha256> \
  --validator-contract post_idle_validator_contract.json \
  --validator-contract-sha256 <sha256> \
  --output <fresh-independent-decision.json>
```

`test_post_idle_result_validator.py` is source-only coverage for exact 4096-row
nearest-rank positions, action/route order sensitivity, stable rank/top1 and
finite outputs, H1 batch/tail physical-call and wait formulas, raw-QPC tamper,
inclusive gate boundaries, and missing ETW/IMC fail-closed classification. It
has not been executed in this static freeze.

## Frozen Learned HashStem ABI

One independent candidate is packed into 559 FP32 values:

```text
post-placement board       24 * 10 = 240
placement difference       24 * 10 = 240
NEXT/HOLD                    7 *  6 =  42
auxiliary state                        37
                                      ---
                                      559
```

The model is deliberately small and NPU-friendly:

```text
normalize(board + placement difference)
  -> Conv 3x3, 2 -> 16 -> ReLU
  -> depthwise Conv 5x1, 16 channels -> ReLU
  -> pointwise Conv 1x1, 16 -> 32 -> ReLU
  -> AvgPool 4x2 = 32 x 6 x 5 = 960
  -> concat(aux 37, normalized NEXT/HOLD 42) = 1,039
  -> Dense(1,039, 214)
  -> concat(raw NEXT/HOLD passthrough 42) = 256
```

It has exactly **223,504 trainable parameters** and **433,546 MACs per
candidate**. Frozen input mean/inverse standard deviation are state, not
parameters. A single fixed batch-16 NPU call
produces `[16, 256]`:

```text
  1:64    layer-1 base routing query
 65:128   layer-2 base routing query
129:192   layer-3 base routing query
193:202   auxiliary-supervised context (10)
203:214   remaining latent context (12)
215:256   raw NEXT/HOLD passthrough (42)
```

The first ten context outputs are death, five line-clear logits, maximum height,
hole count, unreachable-cavity count, and a coarse teacher-Q. Deeper sparse
layers add a parameter-free CPU CountSketch residual from the preceding active
representation to their precomputed base query. There is no second NPU call.
With the separately specified 19,924,022-parameter three-layer neuron bank, the
combined trainable total is **20,147,526**.

The HashStem bridge accepts only the named sparse widths frozen by the teacher
trainer; a custom width or the historical `(26,22,22)` baseline may not be
silently relabelled. Combined per-candidate accounting is:

```text
variant  active widths  touched trainable parameters  learned forward MACs
k64      24,20,20       255,438                       465,458
k128     48,40,40       281,718                       491,738
k256     96,80,80       334,278                       544,298
```

The combined model always has **20,147,526 total trainable parameters**. The
bounded exact routing rerank contributes up to another **106,496 dot-product
MACs** and is reported separately from learned forward work; hash lookup, ID
deduplication, CountSketch signed bucket accumulates, packing, and transfers are
timed as non-MAC stages rather than hidden in an accelerator kernel number.
The first scale gate compares NPU-HashStem + CPU-sparse `k128` against
CPU-HashStem + CPU-sparse `k64` and requires no p50/p95 regression plus a
teacher-ranking or game-score gain. The executable three-cell evaluator also
requires a CPU-HashStem + CPU-sparse `k128` control, identical workload and
bank/HashStem weight digests, raw latency distributions, the 1.15x HashStem
component gate, numeric tolerances, exact route IDs, exact action top-1, and no
top-2 swap. Route coverage is exactly three layer-route comparisons per
candidate; action top-1/top-2 coverage equals the timed sample count. Promotion
quality is restricted to higher-is-better teacher top-1, teacher NDCG, or
paired development-game mean score. It evaluates supplied receipts only and
performs no device call.

`hashstem_reference!` is the deterministic scalar FP32 oracle and actual-length
tail fallback. Its `[batch,559]` input and `[batch,256]` output exactly match
the OpenVINO ABI. It is not the throughput CPU implementation.

## Pipeline and synchronization

The bounded ring uses the only legal cycle:

```text
FREE -> PACKED -> HASH_INFLIGHT -> HASH_READY
     -> SPARSE_DONE -> STEM_TRAIN_DONE -> FREE
```

- E-cores prepare the next candidate batch, rebuild dirty WTA indices, prefetch,
  and log. One E-core remains available for Windows and the NPU driver.
- A single NPU broker owns fixed batch-16 HashStem/teacher/actor inference.
- P-cores own WTA/LSH, candidate-ID deduplication, exact reranking, irregular
  gathers, three-layer sparse forward/VJP, lazy sparse optimizer, and the head.
- iGPU stem training is **disabled initially**. It is enabled only if its full
  concurrent pipeline beats CPU stem training by at least 1.15x while slowing
  P-core sparse work by no more than 1.10x.
- A short HashStem tail uses the deterministic CPU path at its actual length.

The first iGPU-training prototype, if the local driver exposes this desktop
GPU, should use a separate native-Windows PyTorch XPU process. Current PyTorch
documents forward/backward, FP32/BF16/FP16, and `torch.compile` support on
Windows Intel GPUs, but its published validated client list names Arc and Core
Ultra mobile parts rather than the 265K desktop's four-Xe-core Intel Graphics.
It is therefore a measurement candidate, not an assumed capability. Native
Windows `oneAPI.jl` is not the primary path because its own project currently
requires WSL2 for Windows and describes support as experimental. OpenVINO stays
on the inference/export side and is not treated as a training framework.

- PyTorch XPU: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- oneAPI.jl platform status: https://github.com/JuliaGPU/oneAPI.jl
- 265K desktop accelerator summary: https://www.intel.com/content/www/us/en/support/articles/000099656/processors.html

Every slot carries input sequence, sparse-bank and sparse-index versions, master
superstep, snapshot version, and QPC timestamps. The timing schema exposes pack,
queueing, submit/wait/copy, routing, gather, sparse compute, sparse update, stem
training, and total residence time; a kernel-only accelerator number cannot
authorize adoption.

NPU work must enter through `try_acquire_npu_slot!`; actual-length CPU work must
enter through `try_acquire_cpu_slot!`. Both hold the version coordinator before
reserving a ring slot and carry the same published lineage. Snapshot publication uses the same
coordinator-then-slot lock order. This closes the race where a packer could
otherwise reserve an old version while refresh admission is closing. The raw
`try_acquire_free_slot!` API now fails closed.
Every mutating call requires the generation-bound `PipelineTicket`; final FREE
publication uses `release_slot_with_record!`, which captures the complete QPC
record under the slot lock before another producer can reuse the buffer.

The stem is frozen within a superstep. NPU outputs and accumulated stem-output
gradients therefore refer to the same weights. Refresh is a drain/barrier:

1. close new CPU-tail and NPU admission;
2. drain every old-version request, ticket, and live ring slot;
3. apply exactly one accumulated CPU master update;
4. export a fresh NPZ/IR under a new version without overwriting the old one;
5. verify hashes, CPU/NPU numerics, route IDs, and end-to-end timing; and
6. atomically publish the new inference version and reopen admission.

Inference metadata binds model ID, master and snapshot versions, weight and
normalization hashes, OpenVINO version, fixed geometry, and XML/BIN hashes.
The fresh NPZ has exactly `conv3`, `conv3_bias`, `depthwise5x1`,
`depthwise_bias`, `pointwise`, `pointwise_bias`, `dense`, `dense_bias`,
`input_mean`, and `input_inv_std`; its whole-file digest is the master/snapshot
weight digest. No pickle/object arrays are accepted.
The normalization digest is SHA-256 over little-endian FP32 `input_mean` bytes
followed immediately by little-endian FP32 `input_inv_std` bytes.
The snapshot struct stores the external SHA-256 of `snapshot_metadata.json`;
the file does not attempt a self-referential digest.
Checkpoint/resume must restore the CPU master, optimizer, sparse bank/index,
published snapshot metadata, ring sequence, and superstep boundary together.
The graph/runtime/affinity timing and contention witness may be reused only
while its system-contract SHA remains identical. Every published weight version
still requires fresh CPU/NPU numeric, route-ID, and top-1 evidence bound to that
snapshot version and NPZ digest.

## Fail-closed adoption gates

NPU HashStem is adopted only when an exclusive, real-input gate shows all of:

- end-to-end candidate throughput at least **1.15x** CPU;
- NPU p95 no worse than CPU p95;
- CPU FP32 maximum absolute error at most `1e-5`;
- NPU mixed-precision maximum absolute error at most `1e-2`;
- exact selected route IDs after identical WTA probe retrieval and reranking;
- exact final action top-1 in the integrated sparse-agent witness; and
- no CPU sparse-stage slowdown above 1.10x under concurrent NPU use.

Failure selects CPU HashStem without relaxing thresholds. NPU mixed precision
is not described as FP32 merely because its output tensor is FP32.

Active sparse blocks are **not offloaded by default**. Three dependent sparse
layers would require repeated gather/pack/submit/wait/scatter boundaries. One
bounded microbenchmark may test CPU, NPU, and iGPU with all boundary costs, but
offload requires at least 1.15x, non-regressed p95, and unchanged route IDs and
scores. Otherwise that path is permanently rejected and NPU remains HashStem,
teacher, and dense-Actor infrastructure.

## Deferred execution

After the current single-heavy-Julia gate finishes, static Julia tests can be
run with:

```powershell
julia --project=experiments/beat_first_v1 --startup-file=no `
  experiments/beat_first_v1/heterogeneous_265k/test_static_contract.jl
```

After a CPU-master NPZ is exported, fresh IRs can be built (not compiled) with:

```powershell
python experiments/beat_first_v1/heterogeneous_265k/hashstem_openvino.py `
  --weights D:\fresh-run\hashstem_master_v1.npz `
  --output-directory D:\fresh-run\hashstem_ir_v1 `
  --master-version 1 --snapshot-version 1 --model-id tetris-slide-b3 `
  --normalization-sha256 <64-hex-digest>
```

The exclusive CPU/NPU gate uses a real witness containing the Julia reference
output, the exact CPU WTA probe candidate sets, and the identical parameter-free
CountSketch query residuals used by layers two and three:

```powershell
python experiments/beat_first_v1/heterogeneous_265k/benchmark_hashstem_openvino.py `
  --fixed-ir D:\fresh-run\hashstem_ir_v1\hashstem_b16.xml `
  --snapshot-metadata D:\fresh-run\hashstem_ir_v1\snapshot_metadata.json `
  --witness D:\fresh-run\real_hashstem_witness.npz `
  --output D:\fresh-run\hashstem_gate.json --k 26 22 22
```

These commands are documentation only in this change; no Julia, Python,
OpenVINO, NPU, or iGPU process was started.

## CPU master training substrate

`hashstem_master.jl` is the single canonical master-weight contract. It uses
the pinned Lux/Reactant/EnzymeMLIR CPU backend already present in the project;
it does not add or resolve dependencies. Its public tensor ABI remains
`[B,559] -> [B,256]`. Separate loss hooks cover all three 64-wide routing
queries, the 22-wide context, the raw 42-wide NEXT/HOLD path, and ten auxiliary
targets. The query/context hooks accept the output cotangent produced by the
selected-only sparse VJP, so the dense HashStem update does not require a dense
backward through the sparse neuron bank.

CPU master parameters and AdamW state remain FP32. The Lux-native convolution
layouts are converted explicitly to the exact ten-array NPZ schema consumed by
`hashstem_openvino.py`. A snapshot source cannot be exported until a witness
has compared the Lux output with the scalar CPU oracle and confirmed shape,
maximum absolute error `<=1e-5`, and identical routed neuron IDs. Export creates
a fresh immutable source directory only; it does not compile OpenVINO, update
the version coordinator, or authorize publication.

Master checkpointing creates a fresh same-volume transaction directory holding
the exact NPZ, JLD2 optimizer/TrainState restore payload, metadata, and SHA-256
manifest, then renames that directory into place. Resume fails closed across
Julia, Lux, or Reactant version drift. It also restores the last published
snapshot version separately from the advancing master version. No training
step implicitly publishes NPU weights, and NPU backward is forbidden.

The optional `:igpu` backend is deliberately an unavailable interface. It stays
fail-closed until a native Windows backend exists and passes isolated plus
concurrent FP32-master, exact-resume, parity, throughput, and P-core-contention
gates. No BF16/FP16 path is silently enabled.

Status of this entire section: **UNEXECUTED_STATIC_ONLY**. Required later gates
are `test_hashstem_master_contract.jl`, one fixed-batch Reactant compile/update,
two-update exact JLD2 continuation, Lux/scalar/OpenVINO forward parity, NPZ key
and orientation validation, immutable snapshot-source roundtrip, and the
existing integrated NPU 1.15x/p95/route-ID/top-1 gate.

The minimum lifecycle P0 substrate now exists, but remains fail-closed and
unexecuted:

1. no writer in this directory can mark a checkpoint `VALIDATED_RUNTIME`; an
   audited promotion step must bind the executable test artifacts before
   `export_hashstem_snapshot_source!` can succeed;
2. `lifecycle.jl` binds every CPU/NPU chunk to one frozen master, snapshot,
   sparse-bank/index version and ordered ring sequence, then permits exactly one
   barrier update before an evidence-bound export/publication;
3. first adoption has a separate probation-only admission path. It cannot run
   after the permanent `ever_adopted` bit is set and does not set adoption until
   bound evidence passes and the whole ring drains; and
4. boundary checkpointing atomically binds the master/optimizer, sparse
   bank/index/optimizer, coordinator, published snapshot/evidence, ring cursor
   and slot generations, and completed superstep.

These are executable/integration gates, not claimed results. The implementation
and `test_lifecycle_static.py` are **UNEXECUTED_STATIC_ONLY**; no performance,
numeric, resume, or adoption claim follows from this static closure.
# HashStem → production 3-layer SLIDE semantic bridge

Status: **UNEXECUTED_STATIC_ONLY**.

`sparse_hashstem_bridge.jl` is the deliberately small boundary between the
canonical learned HashStem and the frozen 19,924,022-parameter production
three-layer sparse model. It does not add a second routing implementation.

The pure-CPU teacher checkpoint is never rewritten in place.
`bind_teacher_checkpoint_for_hashstem_bridge` verifies its external SHA-256,
named variant, active-width topology, training probes, and optimizer/update
clock, then creates a fresh composite checkpoint carrying the HashStem lineage
and immediately reloads it through the strict bridge consumer. The trainer and
bridge share the single `Main.SparseDynamic3Layer` module identity, so an
ordinary teacher envelope is not reinterpreted as a second nested Julia type.
Strict reload rechecks continuation presence, named variant, probes, update,
and every sparse/head optimizer clock rather than trusting binding-time checks.
An opt-in exact
19.924M save/bind/load continuation smoke is defined in
`test_sparse_hashstem_bridge_contract.jl` for the next exclusive idle window.

The bridge takes the exact packed HashStem input and one `[B,256]` stem output.
For every row it constructs:

- three independent learned `q64` vectors from output ranges `1:64`, `65:128`,
  and `129:192`;
- the learned 22-value context from `193:214`;
- the raw 42-value NEXT/HOLD passthrough from `215:256`; and
- the exact sparse `x496` from the same packed row: candidate board 240,
  difference board 240, frozen value-aux rows 15, and constant one.

The raw passthrough must equal packed columns `481:522` exactly. A stale,
permuted, or unrelated HashStem output therefore fails before sparse routing.
The bridge invokes `SparseDynamic3Layer.route_forward!` sequentially. That
production path retains collision-count-first WTA ordering, Float64 exact-dot
reranking with pending lazy decay, selected-row materialization, and fresh
ID-keyed CountSketch residual construction after the actual L1 and L2
activations. No NNUE reuse, coarse expert, worker thread, NPU call, or new
route kernel is present.

`sparse_vjp_to_hashstem_hooks` calls the full production `vjp_selected`, not
the parameter-only shortcut. It returns selected sparse parameter VJPs and
the exact `q1/q2/q3/context` cotangents required by the FP32 HashStem master.
Hard route IDs remain nondifferentiable. NEXT/HOLD cotangents are exposed for
ABI completeness but the current passthrough has no trainable parameters.

Every combined checkpoint must carry a `SparseBridgeLineage` containing:

- sparse model ID, bank version, and frozen sparse source-closure SHA-256;
- HashStem model, master and snapshot versions;
- HashStem weight and normalization SHA-256 values; and
- after commit, an externally expected SHA-256 of the complete sparse
  checkpoint (which already covers banks, indexes, lazy optimizer state,
  head, RNG/sampler state, and metadata).

`production_route_conformance_witness` restores that same sparse checkpoint
twice. It routes a CPU-reference stem output and a caller-supplied
accelerator-perturbed stem output independently, recording all three selected
ID sets, the real routed L2/L3 query digests, output error, retrieval counts,
and selected-row lazy-decay clock changes. It makes no accelerator call itself.

## Remaining promotion gates

The bridge is not authorized for a convergence or heterogeneous claim until:

1. Julia parse/load and `test_sparse_hashstem_bridge_contract.jl` pass under
   the committed Project/Manifest after the current heavy run is idle.
2. `x496` reconstruction is bitwise compared against the frozen canonical
   candidate feature adapter on real teacher batches.
3. A real exact-geometry sparse checkpoint is saved with the lineage metadata,
   reloaded, and its byte/source/topology bindings pass.
4. Full VJP finite differences validate `q1/q2/q3/context`, including both
   deeper CountSketch paths, on fixed routed supports.
5. The production-route witness compares scalar/FP32 HashStem output against
   the OpenVINO NPU output and applies the existing route-ID, error, margin,
   and game-score adoption gates.
6. Sparse-bank gradients and the HashStem hooks are reduced in a deterministic
   superstep; exact checkpoint/resume is proven across both optimizers.
7. Snapshot drain/barrier, CPU tail refresh, and rollback semantics are added
   before any online NPU snapshot refresh.
8. Only then are P/E-core scheduling, NPU HashStem overlap, and optional iGPU
   master training benchmarked end to end. iGPU remains fail-closed.
