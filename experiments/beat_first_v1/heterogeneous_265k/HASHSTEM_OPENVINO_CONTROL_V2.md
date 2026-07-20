# Learned HashStem OpenVINO CPU/NPU control v2

Status: **UNEXECUTED STATIC SOURCE**. This v2 boundary is not part of the
frozen post-idle v1 source manifest and does not alter any v1 result contract.

The performance control is no longer the scalar `hashstem_reference!`. Both
full-batch paths compile the byte-identical fixed OpenVINO IR and embedded
weights:

```text
CPU control:       fixed IR -> explicit CPU, batch 16
NPU candidate:     fixed IR -> explicit NPU, batch 16
remainder (both):  dynamic IR from the same snapshot -> explicit CPU,
                   actual row count 1:15
```

`AUTO`, padded NPU tails, and silent device fallback are rejected. The scalar
output stored in the witness is only a numerical oracle outside the timed
region. Persistent `CompiledModel`, `InferRequest`, input Tensor, output Tensor,
and application buffers are reused after first-call and warm-up measurements.
The CPU path wraps its application pack/output buffers as shared host Tensors
and prebinds both once. It therefore has no redundant per-call bind,
`Tensor.copy_from`, or runtime-output `Tensor.copy_to`. The NPU path likewise
prebinds both tensors; only its explicit host-to-runtime input copy remains.

Every physical call retains raw `perf_counter_ns` intervals for pack, bind,
H2D, submit, wait, D2H, application copy, and complete end-to-end time. A zero
bind interval records persistent reuse; CPU H2D and both D2H intervals are zero
markers rather than invented calls. The one-time prebind duration is retained
in every call receipt. The public portable host-Tensor API cannot independently
synchronize NPU DMA, so receipts say `dma_timing_isolated=false` and report
logical payload bytes only; actual DMA is still inside submit/wait/end-to-end.
It is never presented as a kernel-only or separately measured hardware transfer.

The CLI requires a provenance sidecar with this shape:

```json
{
  "schema": "learned-hashstem-openvino-witness-provenance-v2",
  "split": "teacher_v3_train",
  "witness_sha256": "<64 lowercase hex>",
  "candidate_count": 37,
  "dataset_manifest_sha256": "<externally supplied manifest SHA-256>",
  "witness_generator_source_sha256": "<externally supplied source SHA-256>",
  "environment_seeds": [110001],
  "source_parts": [
    {
      "episode_key": "v3|train|epsilon|110001|3fa999999999999a|",
      "part_sha256": "<SHA-256 copied from the bound manifest>"
    }
  ]
}
```

The manifest and generator source are reopened and compared with SHA-256 values
supplied outside the sidecar. Every bound episode must exist in the manifest as
a train part; its path must remain inside the dataset root, and its byte length,
file SHA-256, seed, and episode key must match. Validation seeds
`8001:8008` and sealed seeds `91001:91032` fail closed. The
route witness is checked with exactly one of the pre-registered downstream
widths:

```text
--sparse-k 64   -> active counts (24,20,20)
--sparse-k 128  -> active counts (48,40,40)
--sparse-k 256  -> active counts (96,80,80)
```

The sparse layers are not executed by this component boundary. Even a passing
artifact remains `COMPONENT_PASS_PENDING_INTEGRATED_SPARSE_K_GATE` and cannot
authorize adoption, a model-strength claim, or a game-score claim.

The component speed gate uses end-to-end latency distributions, not pooled
throughput. It passes only when both exact conditions hold:

```text
115 * NPU p50 ns <= 100 * CPU p50 ns
NPU p95 ns <= CPU p95 ns
```

The pooled total-time speedup remains supplemental telemetry and cannot pass the
gate. HashStem sparse-neuron route-ID conformance is also not candidate/action
top-1 agreement, top-2 swap rate, or action-margin error. This component is
therefore explicitly non-adoptive. Before any system adoption, a downstream
integrated sparse/action evaluator must use the same sparse checkpoint,
routing, gather, and compute path and pass action top-1, top-2 swap,
action-margin, CPU sparse overlap p50/p95, teacher-ranking, and development-game
gates.

Deferred command shape (documentation only; not executed while dense training
is active):

```powershell
python experiments/beat_first_v1/heterogeneous_265k/benchmark_hashstem_openvino_v2.py `
  --fixed-ir <snapshot>/hashstem_b16.xml `
  --dynamic-cpu-ir <snapshot>/hashstem_dynamic_cpu.xml `
  --snapshot-metadata <snapshot>/snapshot_metadata.json `
  --witness <train-only-witness>.npz `
  --witness-provenance <train-only-witness-provenance>.json `
  --dataset-manifest <teacher_v3>/manifest.json `
  --dataset-manifest-sha256 <externally-expected-sha256> `
  --witness-generator-source <generator-source> `
  --witness-generator-source-sha256 <externally-expected-sha256> `
  --sparse-k 128 --warmups 10 --repeats 100 `
  --output <fresh-result>.json
```
