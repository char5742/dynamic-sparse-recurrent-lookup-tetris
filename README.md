# CPU-native dynamic sparse recurrent lookup research

This repository contains a Tetris teacher-learning research program built
around a CPU-native, dynamically executed neural architecture.  The current
model combines:

- learned sparse LookupFFN banks as long-term parameter memory;
- sparse episodic attention over board, candidate, next/hold, and auxiliary
  tokens;
- recurrent multi-register working memory;
- physically sparse forward, backward, and optimizer updates;
- hard, input-dependent routing and a recurrent halting interface; and
- a barrierless Windows scheduler for heterogeneous Intel P/E cores.

The central implementation is
[`experiments/beat_first_v1/episodic_vit_recurrent_lookup`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/README.md).
Earlier sparse models, routing experiments, CPU scheduler work, comparison
harnesses, and failed experiments are retained under
[`experiments/beat_first_v1`](experiments/beat_first_v1).

The complete Japanese research narrative, including rejected designs and
failed overfit tests, is preserved in
[`RESEARCH_TRAJECTORY.md`](RESEARCH_TRAJECTORY.md).

## Latest verified milestone

On 2026-07-20 the repaired spatial episodic model completed 20,000 optimizer
updates from the real teacher stream without opening the game-validation or
sealed-seed panels.

- strict serial/barrierless correctness smoke: pass;
- warm 50-update scheduler benchmark: 19.81 updates/s;
- allocation: 5.62 MB/update, down from 156.6 MB/update;
- measured GC time: 0 seconds;
- production segment, updates 2,000 to 20,000: 20.87 updates/s;
- final checkpoint SHA-256:
  `1fc05d63154fc73e5d60367c2b19d63116a975b0a3a772899b7fd0ca382db28e`.

The detailed evidence and limitations are recorded in
[`RESULTS_2026-07-20.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/RESULTS_2026-07-20.md).

## Repository policy

Teacher datasets, run directories, and binary checkpoints are intentionally
excluded.  They are large, machine-local artifacts and are not required to
review the implementation.  Published result records include artifact sizes
and hashes so local artifacts can be verified independently.

## Environment

The project is developed on Windows with Julia 1.12.6.  Instantiate the root
environment with:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

The experiment-specific entry points and environment variables are documented
next to each experiment.  BLAS must remain single-threaded when the native
candidate scheduler is enabled.
