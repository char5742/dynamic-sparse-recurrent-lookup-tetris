# Evaluation artifact hardening

## Hypothesis

Development evaluations with the same seed and NEXT setting but different episode
limits could overwrite one another. In addition, the legacy field names did not
fully expose the D1/G2 compute budget. Including the episode limit in filenames
and recording both logical model passes and physical backend requests makes the
artifacts collision-resistant and directly auditable without changing policy
behavior.

## Changes

- Result filenames now include `steps<max_steps>` for both the OpenVINO baseline
  and compact-model evaluators.
- Both evaluators preserve `candidate_count`; the compact evaluator also preserves
  `network_calls`.
- Both emit the aliases `candidate_evaluations`, `logical_network_calls`, and
  `physical_network_calls` required by the submission exporter.
- The OpenVINO evaluator counts one logical call per scored decision and one
  physical request per candidate chunk, including a short tail:
  `ceil(candidate_count / inference_batch_size)`.
- The compact evaluator counts its actual single Lux forward per scored decision;
  therefore its logical and physical counts both equal the existing
  `network_calls` field.
- Top-level JSON now declares backend, device, and the fixed search budget
  (episode limit, NEXT count, HOLD, candidate order, lookahead, selection, and
  inference batching semantics).

## Success criteria and time limit

- Different `max_steps` values produce different filenames.
- Chunk accounting is correct on empty, exact-boundary, and short-tail cases.
- Existing keys remain present.
- The full lightweight test suite passes without running a game evaluation.
- Stop if the change requires modifying model, engine, scoring, or DAgger code.

## Verification

`julia --project=. --threads=4 test\\runtests.jl` passed all 48 tests:

- engine/RNG replay: 6/6
- freeze/export/G2 validation: 25/25
- artifact accounting, checkpoint identity, and collision-resistant names: 17/17

The artifact tests cover candidate counts 0, 1, 16, 17, 32, and 33 at batch 16,
invalid inputs, exact expected filenames, and filename inequality at 50 versus
250 steps. No game evaluation or model inference was run.

## Checkpoint identity follow-up

An audit found that evaluator JSON identified checkpoints only by path. Both
evaluators now retain their legacy `checkpoint` or `checkpoint_path` key and also
record a `checkpoint_fingerprint` object containing the normalized absolute path,
file size in bytes, and lowercase SHA-256 digest. The fingerprint is computed
before loading/compiling the evaluated checkpoint. A three-byte `abc` fixture
checks all three fields against its known SHA-256; a missing checkpoint is also
rejected. Existing evaluation JSON files were not opened or changed.
