# Direct HashStem + sparse-3L k-scale benchmark

Status: **implemented, static-only, not yet executed**. Run this only in an
exclusive idle window after the current heavy Julia process has terminated.

`run_hashstem_sparse3_k_scale_benchmark.jl` is the minimum production entry
point for the pre-registered three-cell comparison:

1. optimized OpenVINO CPU HashStem + CPU sparse k64;
2. optimized OpenVINO CPU HashStem + CPU sparse k128;
3. exact OpenVINO NPU HashStem batch-16 + actual-length OpenVINO CPU tail +
   CPU sparse k128.

The NPU is invoked through embedded `PythonCall` in the Julia process. No Python child
is launched. Device enumeration and `EXECUTION_DEVICES` must both
name exact `NPU`; AUTO/MULTI/fallback execution fails closed. The same fixed
IR is compiled on exact CPU and NPU. The supplied dynamic IR is compiled on
exact CPU only and handles a non-padded tail.

## Scientific boundary

The input checkpoint must be an **ordinary** 19,924,022-parameter three-layer
teacher checkpoint. Composite bridge checkpoints are rejected. The runner
deep-copies its sparse bank and optimizer clocks, changes only the named active
width and rebuilds the corresponding WTA index. Thus k64 and k128 read the
same source bank bytes, while keeping independent lazy-materialization state.
No optimizer update occurs.

The runner directly reuses:

- `sparse_hashstem_bridge.jl` for named k64/k128 geometry and the exact packed
  HashStem-to-three-layer input semantics;
- `sparse_hashstem_named_inference_view.jl` for byte-pinned, same-bank k64/k128
  views which invoke the production `SparseDynamic3Layer.route_forward!`
  path rather than a benchmark-only routing implementation;
- `hashstem_sparse3_k_scale_gate.jl` for the final promotion decision.

This is an inference/system gate, not a sparse-training result and not a game
strength claim.

## Train-only workload artifact

`--workload` is a byte-pinned JLD2 file with:

```text
schema = "hashstem-sparse3-train-workload-v1"
metadata = Dict(
  "split" => "teacher_v3_train",
  "training_only" => true,
  "reserved_seed_free" => true,
  "dataset_manifest_sha256" => "...",
  "state_order_sha256" => "...",
)
candidate_sets = [
  (
    state_id = "canonical-sortable-id",
    seed = 123,
    input = canonical_candidate_feature_input,
    action_digests = [sha256_per_action...],
    teacher_q = Float32[...],
  ),
  ...,
]
```

State IDs must be unique and sorted. `state_order_sha256` is SHA-256 over a
binary stream containing little-endian `UInt32(number_of_ids)` followed by,
for each UTF-8 state ID, little-endian `UInt32(byte_length)` and the exact
bytes. Every input is validated by `BeatFirstSparseFeatures`. Validation seeds
8001:8008 and sealed seeds 91001:91032 are rejected even if the metadata claims
otherwise. The workload dataset-manifest digest must equal the digest embedded
in the ordinary teacher checkpoint.

## Invocation

Every input has an explicit expected SHA-256. XML and BIN members must be
same-stem siblings and must exactly match the snapshot metadata.

```powershell
$env:JULIA_NUM_THREADS = '1'
julia --project=experiments/beat_first_v1 `
  experiments/beat_first_v1/heterogeneous_265k/run_hashstem_sparse3_k_scale_benchmark.jl `
  --teacher-checkpoint C:\path\teacher_checkpoint.bin `
  --teacher-checkpoint-sha256 <sha256> `
  --hashstem-weights C:\path\weights.npz `
  --hashstem-weights-sha256 <sha256> `
  --snapshot-metadata C:\path\snapshot_metadata.json `
  --snapshot-metadata-sha256 <sha256> `
  --fixed-ir C:\path\hashstem_b16.xml `
  --fixed-ir-sha256 <sha256> `
  --fixed-bin C:\path\hashstem_b16.bin `
  --fixed-bin-sha256 <sha256> `
  --dynamic-ir C:\path\hashstem_dynamic_cpu.xml `
  --dynamic-ir-sha256 <sha256> `
  --dynamic-bin C:\path\hashstem_dynamic_cpu.bin `
  --dynamic-bin-sha256 <sha256> `
  --workload C:\path\train_workload.jld2 `
  --workload-sha256 <sha256> `
  --warmup-repetitions 2 `
  --timed-repetitions 30 `
  --output-directory C:\fresh\k_scale_result
```

At least 30 timed repetitions are mandatory and the count must be a multiple
of six. Cells use a fixed six-permutation cycle to reduce ordering bias. Each
timed cell repacks and evaluates the same entire ordered workload, then hashes
the repacked bytes independently. Any Julia GC observed by the enclosing timed
measurement makes the latency promotion non-authoritative.

## Outputs and gate

The fresh output directory contains exactly the evidence payloads:

- `raw_records.jsonl`: one raw record per cell/repetition, including
  packing, HashStem, routing, gather, selected sparse compute, head, sync, and
  end-to-end nanoseconds; route/output digests; action top-1/top-2; teacher
  top-1/NDCG/pairwise metrics; and actual execution order.
- `decision.json`: the one final decision document, artifact byte/SHA bindings,
  p50/p95 stage summaries, numerical parity, route/action parity, quality, and
  the result of `evaluate_hashstem_sparse3_k_scale_gate`.

Promotion requires all of the frozen gate conditions, including NPU HashStem
p50 speedup >=1.15x versus its CPU control, no p95 regression, NPU-k128
end-to-end p50/p95 no slower than CPU-k64, exact routes/actions/top-2, bounded
FP32-oracle error, sparse ranking-Q maximum absolute error <=1e-2, action-margin
error within max(1e-2, 5% of the CPU margin), no timed GC, and higher teacher
top-1 than CPU-k64. A failed gate writes the raw evidence and a negative
decision, then exits nonzero. Artifact drift, backend mismatch, malformed
workload, or any other exception writes a negative decision and exits nonzero
without inventing a fallback result.
