# Active-block witness producer v2

Status: **UNEXECUTED STATIC IMPLEMENTATION**.

This is the missing producer for the component witness consumed by
`active_block_accel_microbench_v2.py`. It is not a trainer, benchmark, model
promotion gate, or game-strength result.

## Hard prerequisite

The producer requires an explicit, already-trained **k256** three-layer sparse
checkpoint with exactly 19,924,022 parameters and active widths `96/80/80`.
The checkpoint must have at least one completed active-only update, matching
optimizer clocks, and metadata/configuration bound to the supplied teacher_v3
manifest. There is **no initialized-weight fallback** and no generated random
weight fallback. Until such a sparse checkpoint exists, witness production is
expected to fail closed.

## Dataset and sampling boundary

Only manifest entries with split `train` and role `old_policy`, `epsilon`, or
`dagger` are eligible. The producer reads the manifest itself but never opens
validation-part bytes. Every train part used to define the sampling corpus is
required to be:

- a relative path strictly inside the explicit dataset root;
- free of symlink or Windows reparse-point traversal;
- canonically and physically unique (no path or hard-link alias to another entry);
- byte- and SHA-256-identical to its manifest binding;
- outside validation seeds `8001:8008` and sealed seeds `91001:91032`.

The externally supplied manifest SHA-256 must also equal the manifest digest
recorded in the trained checkpoint. Candidate positions are selected without
replacement by a SHA-256-derived coprime stride over the manifest-bound train
candidate corpus. The result is deterministic and contains no unrecorded RNG.

## Exact output ABI

The NPZ contains exactly the nine arrays required by the frozen benchmark:

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

`N` must be a positive multiple of 16 and is bounded at 4,096 so every member
and the complete deterministic archive remain within ZIP32. Production WTA/LSH routing is run once
at k256 with probes disabled. k64 (`24/20/20`) and k128 (`48/40/40`) consume
ordered prefixes of those exact IDs, rows, and per-layer inputs, as required by
the component contract. The same immutable NPZ is therefore consumed by the
CPU, NPU, and iGPU cells; no backend receives alternate weights or inputs.

Julia's regular NPZ writer emits Fortran-order arrays. This producer instead
uses a dependency-free deterministic ZIP32/NPY writer with
`fortran_order=False`, fixed ZIP timestamps, STORE encoding, CRC-32, and
row-major bytes matching NumPy's C-order ABI. The UTF-8 sidecar binds the NPZ,
contract, producer, manifest, train
corpus, checkpoint, sampling plan, source parts, array shapes, and array
digests.

## Deferred CLI

Do not run this while another heavy Julia/OpenVINO workload is active. Once a
trained k256 sparse checkpoint exists, calculate the four source/input digests
first and invoke with absolute paths:

```powershell
$repo = 'C:\Users\fshuu\Documents\tetris'
$dataset = 'D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3'
$producer = "$repo\experiments\beat_first_v1\heterogeneous_265k\active_block_witness_producer_v2.jl"
$contract = "$repo\experiments\beat_first_v1\heterogeneous_265k\active_block_accel_microbench_v2_contract.json"

julia --project="$repo\experiments\beat_first_v1" $producer `
  --dataset-root $dataset `
  --dataset-manifest "$dataset\manifest.json" `
  --dataset-manifest-sha256 <manifest-sha256> `
  --checkpoint <absolute-trained-k256-checkpoint.jls> `
  --checkpoint-sha256 <checkpoint-sha256> `
  --contract $contract `
  --contract-sha256 <contract-sha256> `
  --producer-sha256 <producer-source-sha256> `
  --candidate-count 256 `
  --sampling-domain teacher_v3_train_active_block_v2 `
  --output-npz <absolute-fresh-witness.npz> `
  --output-metadata <absolute-fresh-witness-metadata.json>
```

Both outputs must be fresh. The active-block benchmark independently checks
the sidecar's NPZ/contract digests, schema, train-only split, reserved-seed
attestation, candidate count, exact shapes/dtypes, finiteness, C-contiguity,
positive unique IDs, and read-only witness arrays before any device call.

`test_active_block_witness_producer_v2_static.py` is deliberately text-only;
it does not import NumPy/OpenVINO, launch Julia, read D:, or execute hardware.
