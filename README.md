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

On 2026-07-24 the current one-Lookup-block model completed 100,000 optimizer
updates with fixed-K=64, register-specific episodic routing.  A cheap all-token
mean path prevents hard-routing input disconnection, while exact multi-head
cross-attention and working-memory writes execute only on selected support.

- parameters: 6,954,877;
- fixed-K structural tests: 25/25 pass;
- strict serial/barrierless correctness smoke: pass;
- production segment, updates 10,000 to 100,000: 12.31 updates/s;
- steady allocation: 9.067 MB/update; measured GC share: 0.83%;
- fixed training-only panel at update 100,000: loss 2.602389, top-1 0.617188,
  NDCG 0.989165, pairwise 0.903184, mean depth 3.026;
- game-validation and sealed-seed panels were not opened;
- final checkpoint SHA-256:
  `094b0767f91d3d488532570b8363c573b2a76716b21db0a7372d5abbeeefa0e9`.

The complete architecture, tuning, 10k checkpoint trajectory, PreAct boundary,
and limitations are recorded in
[`FIXED_K64_EPISODIC_ROUTING_2026-07-24.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/FIXED_K64_EPISODIC_ROUTING_2026-07-24.md).

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
