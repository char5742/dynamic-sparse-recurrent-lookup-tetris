# Learning track (subagent C)

This directory is isolated from the historical checkpoint and from the common
evaluation code. It uses only the development seeds declared in
`configs/evaluation_protocol.toml` (currently `5742:5757`). The main agent's
held-out test seed set is never read here.

Pipeline:

1. `generate_teacher_dataset.jl` records complete candidate sets and the exact
   OpenVINO 1313 teacher scores into a fixed `max_actions=128` schema.
2. `train_distillation.jl` trains every layer of a compact candidate-Q network
   with per-state standardized listwise teacher targets.
3. The same script optionally follows with 3-step Double-DQN updates using a
   target network and correctly importance-weighted proportional PER.
4. `test_learning.jl` is the deterministic infrastructure/numerical smoke test.

Generated datasets and checkpoints live under
`D:/tetris-paper-plus/datasets/learning`; the source `1313` folder is read-only.
