# Paired 100-update sparse learning-signal screen

This is the smallest paired screen for the exact-19,924,022-parameter 1-layer
k64 model and the exact-19,924,022-parameter 3-layer k64 model.  It is not an
architecture-only ablation.  The frozen 1L bank uses row-scalar active-event
AdaGrad, whereas the 3L bank uses per-element event-time AdamW.  Every external
control is paired: teacher_v3 manifest, predefined train/held split, state-row
sampler and order, state batch 1, candidate tensor width 80, shared composite
loss, model/split/sampler seeds, 1e-4 learning rate, zero decay, evaluation
subsets and cadence, and checkpoint cadence.  Both heads use AdamW under the
same learning rate, betas, epsilon, and zero weight decay.

The two arms must run sequentially.  The wrapper acquires the shared
`Local\TetrisBeatFirstV1ExclusiveJulia` named mutex before checking the process
list and holds it through child exit.  While the owned child runs it polls for
non-owned Julia processes every 100 ms; observing one kills and rejects the
current arm.  It therefore serializes concurrent patched Tetris launchers and
cannot silently overlap an already-running dense convergence run.

```powershell
& .\experiments\beat_first_v1\sparse_dynamic_learning_comparison\run_paired_sparse_teacher.ps1 `
  -Model one_layer `
  -Dataset D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3 `
  -OutputRoot D:\tetris-paper-plus\runs\beat_first_v1\sparse_learning_signal_1l3l_k64_v1

& .\experiments\beat_first_v1\sparse_dynamic_learning_comparison\run_paired_sparse_teacher.ps1 `
  -Model three_layer `
  -Dataset D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3 `
  -OutputRoot D:\tetris-paper-plus\runs\beat_first_v1\sparse_learning_signal_1l3l_k64_v1
```

The checkpoints bind the SHA-256 of `paired_100_update_contract.toml`.  Compare
the update-0/25/50/75/100 held metrics as a system-level promotion signal.  A
positive result can justify a later optimizer-matched architecture ablation;
it must not be described as evidence that depth alone caused the difference.

Each evaluation record already embeds the latest update's sparse accounting.
For 1L this includes unique active rows, active parameter touches, routing and
executed MACs, and selected-only optimizer row writes.  For 3L it includes the
same quantities per layer, active edge touches, routing-inclusive/training MACs,
and theta/moment writes for the active support.  Both also record cumulative
per-bank coverage.  No extra timing instrumentation is enabled for this screen.
