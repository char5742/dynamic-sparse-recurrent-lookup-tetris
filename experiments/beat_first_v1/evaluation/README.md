# Beat-first paired evaluation

This is the minimum shared evaluator for a supplied new checkpoint against the
canonical strong 1313 policy.  It reuses `scripts/evaluate_openvino_checkpoint.jl`
for the historical engine, stable candidate ordering, input construction, and
OpenVINO inference.  It does not implement another Tetris engine.

The model adapter must define module `BeatFirstCandidateAdapter` with the three
functions shown in `adapter_template.jl`.  Beat-first training checkpoints use
`beat_first_adapter.jl`, which directly shares the geometry and 37-feature aux
packer from `training/core.jl`.  The evaluator, not the adapter,
generates the candidate set and applies strict first-maximum selection.  Thus a
candidate receives one complete candidate-set scoring call per decision, no
lookahead, NEXT 5, HOLD enabled, and the same 250-piece limit as the old policy.

Example development invocation (do not run while a training benchmark owns the
CPU):

```powershell
julia --startup-file=no --project=. experiments/beat_first_v1/evaluation/evaluate_pair.jl `
  --stage dev `
  --adapter experiments\beat_first_v1\evaluation\beat_first_adapter.jl `
  --checkpoint D:\absolute\path\to\candidate_checkpoint.jld2 `
  --output D:\tetris-paper-plus\runs\beat_first_dev.json
```

Stages are deliberately locked:

- `dev`: seeds 5756--5757.
- `validation`: seeds 8001--8008 and requires `--promoted`.
- `sealed`: seeds 91001--91032 and requires `--authorize-sealed`.  This flag
  may be used only after explicit authorization from the root research agent.

Each output contains the paired score and survival difference, completion,
candidate evaluations, logical model calls, physical backend requests, model
scoring wall time, and a deterministic 10,000-resample paired bootstrap interval.
The progress JSON is updated after each complete pair and the output path is
never silently overwritten.
