# Common development-validation evaluation

`evaluate_development_validation.jl` is the one Q-ranking evaluator for
PreAct, DSRLN, and CD-SDPG.  The panel is the validation split already reused
for model selection.  It is therefore a **development validation panel**, not
a held or sealed test.

The entrypoint fails before inference unless all frozen identities match:

- teacher manifest: `1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`
- ordered 128-row panel: `fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`
- panel contents: 5,619 candidates, 24--73 candidates/state, 7 teacher-tie states
- caller-supplied checkpoint SHA-256

There is no held/sealed stage argument.  Every result writes
`held_test_touched=false` and `sealed_game_seed_touched=false`.

## Common metrics

Only the continuous Q ranking is compared.  Auxiliary heads are excluded.
Both teacher Q and model Q are standardized within each candidate set and use
ListNet temperature `0.5`.  The result reports:

- ListNet cross entropy, teacher entropy, and excess (their difference)
- legacy stable-first-maximum top-1
- teacher-tie-aware top-1
- NDCG and pairwise accuracy
- identical state/candidate counts
- warm inference wall time including each model's input packing
- stored parameter count and resident parameter/runtime bytes

Runtime-resident bytes retain each historical loader's actual workspace.  In
particular, historical DSRLN restores training state and CD-SDPG's canonical
exact executor retains reverse buffers.  Use `parameter_resident_bytes` for
the fair persistent-capacity comparison; treat `inference_resident_bytes` as
the current loaded implementation cost, not a minimal serving lower bound.

## Invocation

```powershell
julia --startup-file=no --project=experiments/beat_first_v1 --threads=20,0 `
  experiments/beat_first_v1/reduced_hay_cpu_native/evaluate_development_validation.jl `
  --model all `
  --checkpoint D:\absolute\candidate_delta_checkpoint.jls `
  --expected-checkpoint-sha256 <64-hex-sha256> `
  --output D:\absolute\development_comparison.json
```

`--model preact`, `--model dsrln`, and `--model candidate-delta` run one arm
through the same entrypoint.  PreAct and DSRLN default to the frozen checkpoints
used in the historical comparison.  A custom checkpoint always requires an
explicit expected SHA-256.

The DSRLN checkpoint serializes concrete types from source commit
`b1af779d8f490098705f77cfdbc354d01b46afd2`.  Current main cannot deserialize
it reliably.  The evaluator therefore extracts that exact Git tree to a
temporary directory and evaluates it in an isolated Julia process; it does not
switch the working tree or create a branch.

## Revalidated baseline

| Model | ListNet excess | Legacy top-1 | Tie-aware top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|---:|
| PreAct | 0.0559635 | 0.7890625 | 0.7968750 | 0.9932922 | 0.9233594 |
| DSRLN | 0.1005548 | 0.8046875 | 0.8046875 | 0.9901367 | 0.8980834 |

These values were freshly recomputed from the checkpoint outputs, rather than
copied from old reports.  CD-SDPG is added only after a source-matching scratch
checkpoint exists.
