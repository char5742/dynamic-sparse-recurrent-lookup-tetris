# R1 terminal one-shot record

## Disposition

`online_counterfactual_top2_R1` is terminally **`R1-calibration-rejected`**.
Its sole production invocation was authorized from source commit
`57298e99cf6d863859cc86774473661ca9da3d15` and consumed the global one-shot
marker. The exact R1 contract prohibits a retry or rescue.

This is an engineering failure, not a scientific result. Julia 1.12 raised a
world-age `MethodError` on the first `initial_state(seed)` call, after
`collection_begin` but before the first state existed. No row, counterfactual
label, retained-row milestone, or repeatability sentinel was produced. The
training table and manifest are absent, so ridge fitting and calibration could
not run. The R1 safety-gate hypothesis is therefore **untested**.

No development, validation, or sealed-test seed was used. No game evaluation
ran. This invocation provides no game-strength evidence, model-improvement
evidence, G1--G3 evidence, or authority for any such claim.

## Observed failure and resources

- Output: `D:\tetris-paper-plus\runs\online_counterfactual_top2_R1__20260717T233744Z`
- Failing phase: `collect_training`, exit code 1, 17.7374487 s
- Monitor wall: 26.9328275 s; stop reason `failed`
- Exception: Julia 1.12 world-age `MethodError`; running world 39366, current
  world 39881; closure defined at `engine_adapter.jl:300`
- Process-tree peak working set (resident): 1,028,902,912 B (0.958 GiB)
- Process-tree peak private committed: 2,142,552,064 B (1.995 GiB)
- Windows Job peak memory used: 2,226,663,424 B (2.074 GiB)
- Contract limits were not exceeded: 2 GiB resident working set and 4 GiB
  private committed for collection
- Missing scientific products: `training_table.json`,
  `training_manifest.json`, `ridge_artifact.json`, `calibration_table.json`,
  `calibration_manifest.json`, and `calibration_assessment.json`

The fail-closed finalizer published `assessment-fail` with promotion
`R1-calibration-rejected`; subsequent phases were skipped except for that
failure assessment.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| Global start gate `D:\tetris-paper-plus\runs\online_counterfactual_top2_R1.start.txt` | `8bbe097f8b2a9f5ffcbd9e744500ba1d891e61a093bdee15c01d1e7ba5d8a0c5` |
| Consumed marker `D:\tetris-paper-plus\runs\online_counterfactual_top2_R1.started.json` | `118339c1478270da6d3dece7192f4191f059dca1d09b9e7a72876ec292c920cf` |
| One-shot freeze `one_shot_freeze.json` | `4f04d1fb1bd2ecd26c2232064f50d1c5870c97a31e31da0b602f6bb8828f7152` |
| Ready gate `ready_for_start_gate.json` | `d5f7f9fab7bd07282bdfdbe8d6d32033ae731797b817ddff73efa06dec8123a4` |
| Source fingerprint `source_fingerprint.json` | `91f7b25df124482d07a0fe878b176ee6cc0a5c2c32e6e715771c5cf860135f67` |
| Source closure recorded in fingerprint | `ce64861194bfd8da5d51fc859d9542b27b7284a271ee217cfc0263a315958bad` |
| Manifest recorded in fingerprint | `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09` |
| Collection phase `collect_training.phase.json` | `6ac745b10d9c7e5846b5345d90a79edb9b4861b8bc33e71752e967fec5592294` |
| Exception log `collect_training.stderr.log` | `f5658e19dafaeee6a9ec1507efd795e2c2624fef989662077cdb41bde49ff8e9` |
| Child milestones | `df061918454c5f20c78af1c5a45d036867df0991a2d9d34795c95f418a4448b9` |
| Process telemetry | `2834b71ed26672112eb28de54242494a6d90244d42cd1e591baa4ab864d42548` |
| Wrapper milestones | `90426bc18fb6efb34ae02d09373e69113e50263c3601737f826ab021ab8cba5f` |
| Assessment `assessment.json` | `37423b5dbfd6e753ec4776754136e3133e287ca3b84900ad354c105068b84645` |
| Monitor `monitor.json` | `0f671e98448d21a44f4676035378367a6d4846f8ba13a71d810ae9c31d614a86` |
| Wrapper result `wrapper_result.json` | `ed35ac8f37f08ee319e92f345d1f390d7da2b7cb5facd133df01bcecc0f33a77` |
| Final result `final_result.json` | `6a2a687bda219f8888323acd4ddb357b2e72ff7511f2a9db39b04436a45e986d` |

The consumed marker records `retry_prohibited: true` and
`rescue_prohibited: true`; neither the source nor any marker/run artifact was
modified while preparing this record.
