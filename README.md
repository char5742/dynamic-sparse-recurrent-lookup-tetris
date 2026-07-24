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

On 2026-07-25 the fast one-Lookup-block executor was used to restore model
capacity in measured stages.  The largest configuration that retained the
20-updates/s steady-state floor uses learned local spatial attention, three
registers, 16-dimensional one-head register attention, and a width-64 SwiGLU.

- parameters: 6,909,665;
- strict serial/barrierless correctness smoke: pass;
- production segment, updates 10,000 to 100,000: 19.90 updates/s;
- 100k checkpoint steady measurement: 20.30 updates/s;
- steady allocation: 12.849 MB/update; measured GC share: 1.18%;
- fixed training-only panel at update 100,000: loss 2.686149, top-1 0.578125,
  NDCG 0.984347, pairwise 0.874771, margin 0.097056, mean depth 3.023;
- relative to the preceding speed-first 100k model, loss, NDCG, pairwise, and
  margin improved while top-1 decreased by one of 128 panel states;
- game-validation and sealed-seed panels were not opened;
- final checkpoint SHA-256:
  `324492322bf441c9c3b69767e8d997545d0a4b5383e50b0fd431ae12ad9b456f`.

Four registers and 32-dimensional four-head attention were also measured, but
fell below 20 updates/s under dynamic halting and were not adopted.  The full
staged results, 10k checkpoint trajectory, PreAct boundary, and limitations
are recorded in
[`EXPRESSION_RESTORATION_2026-07-25.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/EXPRESSION_RESTORATION_2026-07-25.md).

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
