# F: full legacy model continuation feasibility

This directory implements the sole benchmark authorized by
`reports/clean_post_s1_strategy_review.md` at source commit `7cb3e94`.
It is a one-shot throughput and plumbing gate, not a training experiment and
not evidence that a learned policy improved.

The fixed contract is:

- immutable 20,787,454-parameter `LegacyQNetwork` checkpoint;
- source rows `1, 251, 501, 751, 1001, 1251`, episodes 1--6 and implied seeds
  5742--5747 only;
- frozen 3-step `gamma=0.997` targets from stored rewards and selected old-Q;
- selected-action Huber, coupled AdamW `1e-5`, betas `(0.9,0.999)`, weight
  decay `1e-4`, Lux test-mode state, native Zygote only;
- exact historical candidate batching: sequential full chunks of 16 and an
  actual-length CPU tail.  The selected update forwards only the exact chunk
  containing the selected action.  No row is normalized as one 26/43/51-wide
  batch and no tail is padded;
- one temporary NPZ export followed by fresh OpenVINO CPU and NPU+CPU-tail
  compilation and numerical equivalence;
- hard 25-minute, 8-GiB working-set, 300-second first-specialization and
  120-second per-update stops, with no retry or tuning.

`extract_rows.py` uses HDF5 hyperslabs because ordinary JLD2 loading would
materialize forbidden validation rows 1501--2000.  It touches only the six
training trajectories' steps 1--4 (24 rows total).  No game, score,
development evaluation, validation seed or test seed is present in this
harness.

The wrapper first creates `freeze.json` and waits on an explicit start-gate,
so the exact clean commit, source fingerprint, command, process snapshot,
machine memory and constants can be reviewed before the one-shot run starts.
All generated data and results must be under `D:\tetris-paper-plus`; temporary
updated weights are explicitly non-promoted and never overwrite the historical
checkpoint or tracked OpenVINO artifacts.
