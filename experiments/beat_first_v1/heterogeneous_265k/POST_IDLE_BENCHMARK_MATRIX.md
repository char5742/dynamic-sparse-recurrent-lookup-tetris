# Post-idle heterogeneous benchmark matrix

Status: **UNEXECUTED**. This document and
`post_idle_benchmark_contract.json` are static contracts only. They do not
claim that CPU Sets, NPU, iGPU, ETW, QPC sampling, or memory-controller
counters have been exercised.

This matrix deliberately excludes NNUE. It measures whether CPU routing and
irregular gather should remain on the 265K's P-cores while fixed-shape active
blocks, HashStem inference, or HashStem training move to an accelerator.

## Required cells

| ID | Routing / gather | Active sparse block | HashStem | Stem training | Control |
|---|---|---|---|---|---|
| A0 | CPU / CPU | CPU | frozen witness output | CPU selected-only training subcase | reference |
| A1 | CPU / CPU | NPU | same frozen output | forward-only unless explicit NPU VJP compiles | A0 |
| A2 | CPU / CPU | iGPU | same frozen output | separate iGPU forward/backward/update subcase | A0 |
| H0 | CPU / CPU | CPU | CPU | disabled | integrated inference control |
| H1 | CPU / CPU | CPU | NPU batch 16 + actual-length CPU tail | disabled | H0 |
| F0 | CPU / CPU | CPU | NPU batch 16 + CPU tail | CPU | full-pipeline control |
| F1 | CPU / CPU | CPU | NPU batch 16 + CPU tail | iGPU | F0 |

All cells consume the same immutable teacher-v3 **train-split** witness, stable
candidate order, all candidates, identical model/snapshot versions, identical
WTA probe rows, and identical CountSketch residuals. Development, validation,
and sealed game seeds are forbidden. Each cell is a fresh process and cells
run sequentially.

The active-device ABI is fixed batch 16. L1 receives `[16,560]` inputs and
`[16,26,560]` gathered weights; L2/L3 receive `[16,321]` inputs and
`[16,22,321]` gathered weights. Their outputs are `[16,26]`, `[16,22]`, and
`[16,22]`. CPU CountSketch/query composition remains between calls, so every
inter-layer synchronization is observable. The CPU 256-to-22 head follows L3
and remains inside total timing. A short tail executes at its actual length on
CPU and is never hidden accelerator padding.

In FP32 this ABI moves at least 1,912,704 bytes host-to-device and 4,480 bytes
device-to-host per complete batch before runtime metadata or alignment:
931,840 bytes of L1 weights, 451,968 bytes for each deeper layer, and 76,928
bytes of inputs. These bytes must be measured, not omitted from an accelerator
kernel comparison.

## Timing boundary

The adoption number is never a kernel-only number. QueryPerformanceCounter
must span packing, routing, gather, accelerator packing, host/device transfer,
bind, submit, wait/synchronize, output materialization, scatter/CountSketch,
head, VJP, optimizer, and—where applicable—drain/export/compile/verification
and atomic snapshot publication. Raw ticks are retained per sample. Device
events are supplemental only.

The exact QPC field list is in the JSON contract. Existing
`hash_submit_wait_copy_ns` is insufficient for this matrix because it combines
transfer, submission, wait, and copy; the post-idle runner must emit the finer
fields without deleting the existing compatibility record.

Steady measurements use 256 warm-up and 4,096 timed candidate sets, three
fresh repetitions. Full training uses 64 warm-up and 1,024 timed supersteps.
Nearest-rank p50/p95 are reported for every stage and the complete boundary.
Compile and first-call cost stay outside steady percentiles but are reported
and amortized at 100, 1,000, and the full timed count.

## Correctness and adoption

Every route ID at each of the three sparse layers and every final candidate-set
top-1 must match the CPU reference exactly. CPU FP32 output error is bounded by
`1e-5`; accelerator output error by `1e-2`. The iGPU training gate additionally
requires loss error `<=1e-4`, gradient cosine `>=.999`, relative gradient L2
`<=1e-3`, and updated stem-parameter error `<=1e-4` from identical initial
state.

A1/A2 must beat A0 end-to-end by at least 1.15x with no p95 regression. H1 must
beat H0 by 1.15x, with no p95 regression and no more than 1.10x CPU sparse-stage
slowdown under overlap. F1 must beat F0's complete-superstep throughput by
1.15x, with no p95 regression or CPU sparse slowdown above 1.10x. A failed
active-block offload does not reject NPU HashStem.

## Contention and RAM evidence

The seven isolated/concurrent conditions in the JSON contract are mandatory.
They report stage p50/p95, slowdown ratios, throughput, accelerator utilization,
process private/working set, page faults, and same-session DRAM read/write
bytes and bandwidth from uncore/IMC counters. ETW supplies whole-stage CPU
residency; point CPU witnesses alone are insufficient.

Missing, overflowing, or multiplexed DRAM counters make heterogeneous adoption
`INCONCLUSIVE_FAIL_CLOSED`; they are not replaced with process I/O bytes or a
theoretical DIMM rating.

The independent source contract is
`windows_residency_ram_evidence_contract.json`. ETW is explicitly Microsoft
WPT xperf with NT Kernel `PROC_THREAD+CSWITCH` and raw-QPC reconstruction.
DRAM evidence is explicitly Intel PCM `pcm-memory.exe` plus a version-bound raw
IMC adapter. Stock PCM CSV alone cannot pass. Until that adapter proves raw
before/after counts, width, overflow, multiplex/enabled-running status and full
channel coverage, the IMC path is **UNAVAILABLE_FAIL_CLOSED**. All files in that
evidence path remain **UNEXECUTED_STATIC_ONLY**.

## Backend failure

An unavailable or non-compiling NPU/iGPU creates a fresh
`UNAVAILABLE_FAIL_CLOSED` artifact and exits 2. The cell may not silently use
OpenVINO `AUTO`, CPU, another accelerator, or a pre-existing component timing.
The separately named CPU control/fallback may continue, but the requested
heterogeneous cell cannot pass without all timing, numerical, routing, top-1,
contention, residency, and RAM-bandwidth evidence.
