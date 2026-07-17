# R1 online counterfactual top-2 safety gate

This namespace implements the only experiment authorized by
`reports/clean_post_q1_memory_failure_strategy.md`. It is not a retry or rescue
of Q1. It never trains or modifies the historical neural network: the frozen
old OpenVINO policy remains the default action, and an analytic low-dimensional
gate may replace old top-1 only with old top-2.

## Frozen scientific contract

- training-only episodes: seeds `73001:73012`;
- one-use calibration episodes: seeds `73101:73106`;
- canonical-trajectory sampling positions: `10,20,...,240`;
- validation `8001:8008`, sealed test `91001:91032`, and used development
  `5742:5755` are forbidden in this run;
- stable candidate order, historical candidate batches of 16 plus an
  actual-size tail, NEXT=5, HOLD enabled, and first-index tie breaking;
- for each sampled canonical state, old top-1 and old top-2 are independently
  rolled out from copies of the same root and future source;
- the root action is included in a six-piece return, followed by five
  canonical-old actions, then a non-terminal old-Q bootstrap;
- `G6 = sum(k=0:5, 0.997^k * score_delta_k / 600) +
  0.997^6 * max_old_Q(s6)` and `A6 = G6(a2)-G6(a1)`;
- fit target `clamp(A6,-2,2)`; scientific calibration metrics use unclipped
  `A6`;
- exactly 70 frozen numeric features and an intercept (71 coefficients);
- queue-feature piece order `I,O,S,Z,J,L,T`, with semantic slots
  `hold,next1=end,...,next5=end-4`; the old-Q tensor independently retains its
  historical `[hold,end-4:end]` physical order;
- `covered_cells` is the count of unique occupied cells above at least one
  hole in the same column, on the full 24x10 board;
- 256 analytic episode-cluster bootstrap ridge models, `lambda=1`,
  `Xoshiro(0x5231_2026)`;
- override only when the prediction 10th percentile is strictly `>0.05`;
- no threshold, lambda, feature, horizon, quantile, ensemble, or seed sweep.

The old-Q backend is frozen to OpenVINO 2026.2.1: each complete group of 16
candidates is one static NPU request and an actual-size 1--15 tail is one
dynamic CPU request, with Float32 weights/input/output and no padding.

The production analytic fit/calibration implementation is Python 3.12.13 with
NumPy 2.4.6 and one BLAS thread. Julia's analytic implementation is retained as
an independent numerical reference. This is a resource-contract decision, not
an algorithm change; `PREFLIGHT.md` records why Julia production is rejected.

The equal-future invariant concerns the immutable branch-start source (queue,
RNG snapshot, and future-bag oracle). Branch-final RNG states may legitimately
differ when HOLD changes how many queue entries are consumed. Each branch also
records its final RNG digest independently. The canonical root is not mutated
by either counterfactual and advances with old top-1 only after both returns
have been recorded.

## Promotion boundary

The one-shot run ends after calibration. A pass is named only
`R1-calibration-promoted`; it is not game-strength evidence, G1--G3, or a
model-only improvement. Development is disabled by default and is not part of
this one-shot. Seeds 5756 and 5757 require a later independent freeze review.

`invoke_once.ps1 -ValidateOnly` is synthetic-only: it performs strict
JuliaSyntax parsing, exercises every production-argv branch in a fresh Julia
process, and validates process-tree telemetry without loading a checkpoint,
OpenVINO, the Tetris engine, real datasets, or any reserved seed.
