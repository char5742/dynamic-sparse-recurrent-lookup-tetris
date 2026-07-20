# Active-block accelerator microbenchmark v2

Status: **UNEXECUTED STATIC IMPLEMENTATION**.

This is the bounded A1/A2 component experiment requested for the 265K
heterogeneous track. It does not modify or supersede any v1 post-idle file.
It cannot promote a model or prove game strength.

## Exact boundary

The witness has already completed WTA/LSH selection. For every independent
candidate it stores the ordered selected neuron IDs and corresponding
contiguous FP32 row block at the largest support `(96,80,80)`. The three
variants consume nested prefixes:

| Variant | L1/L2/L3 active | Total active |
|---|---:|---:|
| `k64` | `24/20/20` | 64 |
| `k128` | `48/40/40` | 128 |
| `k256` | `96/80/80` | 256 |

WTA lookup, reranking, CountSketch composition, the final Q head, Replay,
training, and game evaluation are outside this component. Therefore a pass is
reported only as `COMPONENT_PASS_PENDING_INTEGRATED_BENCHMARK`.

Each path must still copy the selected prefix into a reusable fixed-shape host
buffer. The timed CPU baseline performs vectorized selected-row dot products
and stable SiLU. The scalar ordered-FP32 implementation is used only outside
timing as a numerical oracle. The accelerator path times, for all three calls:

```text
selected source block
  -> host gather/materialization
  -> input pack
  -> OpenVINO Tensor construction and bind
  -> async submit
  -> wait/synchronize (including opaque transfer/device visibility)
  -> copied FP32 output
```

OpenVINO does not expose a portable host-QPC split between DMA and device
execution. The script therefore does not invent one. The complete boundary
includes transfer, submit, wait, synchronization, and output materialization;
logical FP32 transfer bytes are reported exactly, while `submit+wait` remains
the honest opaque transfer/compute interval.

## Device rules

- NPU compilation is bound to exactly one enumerated `NPU`-family device;
  multiple candidates require an exact `--npu-device` or fail closed.
- The GPU cell requires one explicitly identified **Intel integrated** GPU.
- `AUTO`, `HETERO`, `MULTI`, CPU fallback, discrete GPU substitution, and
  multi-device execution are rejected.
- `EXECUTION_DEVICES` must equal the selected enumerated device string exactly;
  a different device in the same NPU/GPU family is also rejected.
- Missing, ambiguous, non-compiling, or unverifiable devices yield a fresh
  `UNAVAILABLE_FAIL_CLOSED` result and exit code 2.
- Every first-call, warm-up, and timed accelerator invocation emits a physical
  call receipt with execution-device identity, raw QPC intervals, shapes,
  bytes, input/selection/output digests, model name, and monotonic call index.

The component adoption gate is at least 1.15x end-to-end p50 speedup over the
optimized CPU baseline, no p95 regression, accelerator error at most `1e-2`,
and exact physical-call accounting. Even a pass must later survive the
integrated routing/CountSketch/head and contention benchmark.

## Witness ABI

The NPZ contains exactly:

```text
selected_rows_l1  Float32[N,96,560]
selected_rows_l2  Float32[N,80,321]
selected_rows_l3  Float32[N,80,321]
selected_ids_l1   Int32[N,96]
selected_ids_l2   Int32[N,80]
selected_ids_l3   Int32[N,80]
scaled_input_l1   Float32[N,560]
scaled_input_l2   Float32[N,321]
scaled_input_l3   Float32[N,321]
```

`N` is a positive multiple of 16. Inputs are already scaled so a row dot is
the exact sparse-neuron preactivation. IDs are positive, ordered, and unique
within a candidate/layer. The UTF-8 sidecar must use schema
`heterogeneous-265k-active-block-witness-v2`, bind the NPZ and contract
SHA-256, identify a `teacher_v3_train` or `synthetic_component_only` source,
and certify both:

```json
{
  "reserved_seed_free": true,
  "development_validation_sealed_seeds_used": false
}
```

## Deferred invocation

Do not run while another heavy Julia/Python/OpenVINO process is active. After
the dense convergence process exits and an immutable witness exists, invoke in
a fresh output directory with the exact contract digest:

```powershell
python experiments/beat_first_v1/heterogeneous_265k/active_block_accel_microbench_v2.py `
  --contract experiments/beat_first_v1/heterogeneous_265k/active_block_accel_microbench_v2_contract.json `
  --contract-sha256 <sha256> `
  --witness <fresh-witness.npz> `
  --witness-metadata <fresh-witness-metadata.json> `
  --output <fresh-result.json> `
  --raw-timings <fresh-raw-timings.jsonl> `
  --physical-call-receipts <fresh-physical-calls.jsonl>
```

Static source/contract checks live in
`test_active_block_accel_microbench_v2_static.py`; they do not import NumPy or
OpenVINO and do not execute this benchmark.
