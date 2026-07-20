# Native-Windows Intel iGPU HashStem training probe v1

Status: **UNEXECUTED_STATIC_ONLY**.  These files were added without starting
Python, Julia, OpenVINO, XPU, NPU, or iGPU work.  They do not change or replace
the frozen post-idle v1 matrix.

This is a bounded component probe for the heterogeneous-system track.  It asks
whether a fixed-shape FP32 Learned HashStem training update on the explicitly
enumerated Intel iGPU is faster than the same PyTorch graph on CPU while still
matching the authoritative Lux CPU-master update.  It is not evidence about
the pure-CPU dynamic sparse model, NNUE, teacher quality, game score, or the
full heterogeneous pipeline.

## Fail-closed inputs

The probe accepts only all of the following:

- a byte-bound copy of
  `igpu_hashstem_training_probe_contract_v1.json`;
- a byte-bound live-host system contract with schema
  `heterogeneous-265k-native-windows-system-contract-v1`, status
  `BOUND_LIVE_HOST`, exact Core Ultra CPU name, and exact single-iGPU PyTorch
  XPU name/index/count plus CIM name, Intel PCI PNP ID and driver version.  The
  live `Win32_ComputerSystemProduct` record is normalized and hashed into the
  `sha256:<digest>` host ID, which must exactly match the contract;
- a live `GetSystemCpuSetInformation` enumeration whose canonical topology
  digest exactly matches `cpu.cpu_set_topology_sha256`.  It must classify as
  the audited 8P+12E topology, and the CPU-control thread count is derived as
  the eight P-core CPU Sets.  The mandatory CLI value and
  `cpu.hashstem_control_threads` must both equal that live count;
- a runtime-validated, version-zero HashStem master checkpoint.  Version zero
  is deliberate: v1 can then reproduce the empty AdamW state exactly instead
  of pretending that it converted a nonzero Julia optimizer state;
- immutable snapshot metadata from the exact `hashstem_master.jl` exporter
  contract: exact schema and HashStem schema, status `UNEXECUTED_STATIC_ONLY`,
  `immutable_source=true`, `openvino_compiled=false`,
  `publish_authorized=false`, exact `hashstem_snapshot_weights.npz` filename,
  and a source-checkpoint manifest digest equal to the bound master.  Its
  explicitly supplied NPZ, model ID, master version, weight and normalization
  digests must agree, and every FP32 array must equal the master bitwise;

Exporter values are type-exact JSON values: numeric `1`/`0` cannot stand in
for boolean `true`/`false`.  Live host identity and CPU-topology discovery occur
before caller-supplied identity comparison.  Their partial or complete receipt
is retained in fail-closed validation and numeric-rejection results as well as
successful component results.
- an NPZ training witness and byte-bound sidecar using schema
  `igpu-hashstem-training-witness-v1`;
- for `teacher_v3_train`, the actual teacher manifest and digest plus
  `teacher_split=train`; or, for a synthetic witness, a generator-source digest
  and the isolated `synthetic_component_only_no_game_rng` namespace.

The witness has a positive multiple of 16 rows of `packed`, pre-masked
`output_cotangent`, `auxiliary_target`, `auxiliary_mask`, and `state_mask`.
Its first batch also contains the CPU-master reference output, loss, gradients
and post-AdamW weights for all eight trainable arrays.  Development,
validation, and sealed game seeds are forbidden and the sidecar must say so.

If PyTorch has no XPU, exposes zero or multiple XPUs, cannot compile the exact
fixed graph, cannot pin/copy/synchronize, or cannot match the live system
contract to exactly one healthy Intel CIM adapter, the only result is
`UNAVAILABLE_FAIL_CLOSED` and exit code 2.  CPU is never substituted.

## What is measured

CPU and XPU start from identical canonical NPZ weights, use batch 16, the same
FP32 loss, the same AdamW hyperparameters, the same witness order, and the same
`torch.compile(backend="inductor", fullgraph=True, dynamic=False)` policy.
Wrapper construction and first forward/backward compilation are separate from
steady state.  Each steady update retains raw QPC intervals for:

```text
H2D submit / sync
zero_grad
forward submit / sync
backward submit / sync
optimizer submit / sync
D2H submit / sync
end-to-end total
```

Nearest-rank p50/p95 and aggregate updates/s are derived from those raw rows.
The authoritative speed gate uses the ratio of CPU to XPU end-to-end total
p50; aggregate updates/s is supplemental and cannot satisfy the gate.
The end-to-end boundary starts with the host batch and ends only after a copied
FP32 output/loss, optimizer publication, and explicit synchronization.

A Windows `GetProcessMemoryInfo` + `GlobalMemoryStatusEx` sampler records raw
working-set/private-byte/available-RAM samples around each timed cell.  This is
only a component RAM-pressure receipt.  It is not DRAM-bandwidth evidence.
ETW/IMC evidence and the concurrent P-core sparse slowdown gate remain
external.

## Gates and checkpoint semantics

Before timing, the probe requires:

- local PyTorch CPU output/loss/gradients/updated parameters to match the bound
  Lux CPU-master witness;
- compiled XPU to match the compiled CPU control;
- a real backward and optimizer update: at least one weight bit pattern must
  change, all eight trainable tensors must have step-one AdamW state, and first
  and second moments must be nonzero;
- frozen normalization to remain bitwise unchanged.

The component speed gate requires CPU total p50 divided by XPU total p50 to be
at least 1.15, with no XPU p95 regression.  Aggregate updates/s is reported but
is not a gate.  Passing produces only
`COMPONENT_PASS_PENDING_P_SPARSE_AND_SYSTEM_GATES`.  The later integrated
experiment still must show P-core sparse slowdown no greater than 1.10x and
must supply ETW/IMC and full-pipeline evidence.

The one-step iGPU state is round-tripped through a fresh probe checkpoint.
Reload equality compares dtype, shape, and raw contiguous CPU bytes, not tensor
value equality, so signed-zero and NaN-payload differences cannot pass.  Its
canonical master and candidate-snapshot NPZ files must be byte-identical, its
normalization digest unchanged, and its master/optimizer/candidate snapshot
lineage internally consistent.  Every metadata file explicitly sets production
continuation, publication, and adoption to false.

## Deferred command shape

Do not run this while another heavy Julia/Python/OpenVINO job is active.  After
the single-heavy-process gate is clear, the command shape is:

```text
python igpu_hashstem_training_probe_v1.py
  --contract <contract.json> --contract-sha256 <sha256>
  --system-contract <live-system.json> --system-contract-sha256 <sha256>
  --master-checkpoint <version-zero-master-directory>
  --snapshot-metadata <snapshot-metadata.json>
  --snapshot-metadata-sha256 <sha256>
  --snapshot-weights <snapshot-weights.npz>
  --snapshot-weights-sha256 <sha256>
  --witness <training-witness.npz>
  --witness-metadata <training-witness.json>
  --witness-metadata-sha256 <sha256>
  --teacher-manifest <teacher_v3-manifest.json>
  --teacher-manifest-sha256 <sha256>
  --cpu-threads 8
  --checkpoint-output <fresh-probe-checkpoint-directory>
  --output <fresh-result.json>
```

For the synthetic witness kind, omit both teacher-manifest arguments.  The
probe refuses to overwrite either output.
