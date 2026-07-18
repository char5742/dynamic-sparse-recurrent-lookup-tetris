# Stage 1 teacher-pretraining result (frozen)

All candidates used the same teacher dataset, row schedule seed, 100 updates,
state batch 2 (74 candidates/state), Reactant + EnzymeMLIR CPU, and the same
held-out 500 teacher states. No game validation or sealed seed was used.

| Model | Params | train wall | updates/s | loss first→last | top-1 | NDCG | pairwise | Q Huber |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PreAct-ECA | 1,480,298 | 231.520 s | 0.4319 | 4.8311→2.5705 | 0.378 | 0.947090 | 0.737563 | 2.884462 |
| Tetris-ConvNeXt | 1,211,506 | 425.002 s | 0.2353 | 5.9447→2.5851 | 0.290 | 0.929395 | 0.737252 | 0.135202 |
| Gravity-FiLM | 1,615,314 | 480.983 s | 0.2079 | 6.5688→1.9184 | 0.276 | 0.934346 | 0.748053 | 0.125453 |

The frozen lexicographic teacher promotion key is top-1, NDCG, pairwise, then
negative Q-Huber. It ranks PreAct-ECA first and Tetris-ConvNeXt second. A clean
model-promotion review is pending before stage 2; these results are not game
strength evidence.

## Frozen checkpoints and run records

- PreAct-ECA checkpoint: `D:\tetris-paper-plus\checkpoints\beat_first_v1\beat_teacher_stage1_preact_eca_20260718_preact_eca_stage1.jld2`, SHA-256 `b8519afd13580f6a3fc958fefe71690a466668a93e84b6dc7a5c1f917f91d09b`
- Tetris-ConvNeXt checkpoint: `D:\tetris-paper-plus\checkpoints\beat_first_v1\beat_teacher_stage1_tetris_convnext_20260718_tetris_convnext_stage1.jld2`, SHA-256 `6a6309a2833d3ea1680db3b41aba4dd9ecb44056f1b2752549e38f2f3b7be672`
- Gravity-FiLM checkpoint: `D:\tetris-paper-plus\checkpoints\beat_first_v1\beat_teacher_stage1_gravity_film_20260718_gravity_film_stage1.jld2`, SHA-256 `191208a3faba975894fbe197a326324b22a821b2b0be52c0f4e3159763d69c21`
- Run records: `D:\tetris-paper-plus\runs\beat_first_v1\beat_teacher_stage1_{preact_eca,tetris_convnext,gravity_film}_20260718.json`

The interrupted eager process had already emitted one duplicate PreAct-ECA
checkpoint. Its update and all four metrics were exactly identical to the
single-model run. The single-model checkpoint above is canonical; PreAct-ECA
must not be run a third time for stage 1.

## Source/runtime binding at freeze

- Repository HEAD before beat-first commit: `97c52085d64086cb3a8fc48a222f8b48641064ad`
- `training/core.jl`: `9e96c028ec4d8e9d1093f00dd582f2502b1635796d5b8add45fc0d2b459cae0d`
- `training/train_supervised.jl`: `15d916d5003d01230e34f80f0ff83372f3eb80245ce62d0b530fd61007950814`
- `models/models.jl`: `6cc69c1f09b5d49b8d11acc8aedde9a472de03caaafbd3de67294da262941014`
- `backend/fixedshape_learner.jl`: `537071396bb3f52195f268a0aa989332abd1590e7ea8a955597e150603b92d3c`
- Runtime `experiments/ad_backend_retry_2026/Manifest.toml`: `09c5730de81a2b9ccfc7259ff6167e6814bb2a303916275ad0b8c7c3a958d3f8`
