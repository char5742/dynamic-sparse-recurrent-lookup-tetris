# Beat-first teacher dataset v2

This is the mainline, resumable teacher/DAgger data path. It reuses the
canonical `stable_node_list`, `legacy_candidate_batch`, `openvino_scores`, and
`apply_node!` helpers. Candidate arrays are saved in their original stable
order with multiplicity intact. Old-Q labels retain the historical static
NPU-batch-16 plus actual-size dynamic CPU-tail semantics.

Each episode is one bounded JLD2 part. A JSON sidecar contains its key, split,
rollout role, seed, counts, byte size, and SHA-256. `manifest.json` is replaced
atomically after every episode. Resume skips manifest entries and reconciles a
completed part+sidecar orphan left by interruption; it never regenerates the
same episode key.

## Frozen schedule

- Validation (generated first): seeds `120001:120024` old-policy and
  `121001:121024` epsilon-policy.
- Training mixture: seeds `100001:100320` old-policy plus
  `110001:110200` epsilon-policy, interleaved. Epsilon cycles deterministically
  through `0.05`, `0.10`, and `0.20`.
- Fallback: `105001:105120` old-policy episodes are consumed only if early
  epsilon deaths leave training below 100,000 states.
- Later DAgger: `130001:130240`, student behavior with every visited candidate
  relabeled by the unchanged old model.

These ranges are disjoint from development `5742:5757`, validation
`8001:8008`, and sealed `91001:91032`. Split membership is episode/seed-level;
no trajectory can straddle train and validation.

## Run and resume

After N1 has stopped and the source is committed:

```powershell
$env:OPENVINO_DEVICE = 'NPU'
$env:OPENVINO_BATCH = '16'
$env:BEAT_DATASET_OVERLAP_TAIL = 'true' # only after the strict gate below says GO
$env:BEAT_DATASET_PLAN = 'base'
$env:BEAT_DATASET_TARGET_TRAIN_STATES = '100000'
julia --project=. --threads=20 experiments\beat_first_v1\dataset\generate_streaming.jl
```

The global OpenVINO predictor remains serial.  This generator also defaults to
serial unless `BEAT_DATASET_OVERLAP_TAIL=true` is set explicitly.  Before that
opt-in, run the 200-state equivalence/throughput gate with no other Julia or NPU
work active:

```powershell
$env:OPENVINO_DEVICE = 'NPU'
$env:OPENVINO_BATCH = '16'
$env:BEAT_DATASET_GATE_STATES = '200'
$env:BEAT_DATASET_GATE_MIN_REDUCTION = '0.25'
$env:BEAT_DATASET_GATE_OUTPUT = 'D:\tetris-paper-plus\runs\beat_first_v1\dataset_overlap_gate_200.json'
julia --project=. --threads=20 experiments\beat_first_v1\dataset\gate_overlap_tail.jl
```

The gate requires bitwise-identical FP32 Q values, identical argmax actions,
candidate counts and stable order, full-batch and tail coverage, and at least a
25% reduction in the complete measured pipeline projection.  A failed gate
writes its artifact, exits nonzero, and requires leaving the generator serial.
The optimized call still uses static NPU batches of 16 and the same actual-size
dynamic CPU tail; only their execution overlaps.

The same command resumes. To test one new episode, set
`BEAT_DATASET_MAX_NEW_EPISODES=1`. DAgger uses a checkpoint through the existing
beat-first candidate adapter:

```powershell
$env:BEAT_DATASET_PLAN = 'dagger'
$env:BEAT_DATASET_STUDENT_CHECKPOINT = 'D:\tetris-paper-plus\checkpoints\beat_first_v1\candidate.jld2'
julia --project=. --threads=20 experiments\beat_first_v1\dataset\generate_streaming.jl
```

## Schema

Every part retains the legacy fields consumed by current training plus:

- full ordered `teacher_q` and stable descending `teacher_rank`;
- top-1/top-2 indices and margin;
- per-candidate line clears, death, maximum height, holes, and unreachable
  cavities;
- selected behavior action and whether epsilon exploration selected it;
- explicit per-state `seed_ids`, episode step, reward, terminal flag, and score.

Short empirical returns are deliberately omitted from the first pass because
they multiply environment work. They can be added later without changing old-Q
labeling.

## Throughput projection

The existing 2,000-state NPU run took 623.554 seconds (86,691 candidate
labels): about **3.21 states/s** or **8.66 hours per 100,000 states** if that
single observed rate holds. Old-policy episodes reached 250 pieces; student
DAgger averaged only 110 pieces in the prior run. The base schedule therefore
front-loads 80,000 potential old-policy states and uses epsilon trajectories
for distribution breadth, with deterministic old-policy fallback to guarantee
the 100k target. Only one generator should run: concurrent Julia training or
benchmarks would contaminate throughput and compete for CPU/NPU resources.

`index.jl` reads only the manifest and visits one part at a time so downstream
training need not concatenate the full dataset in RAM.
