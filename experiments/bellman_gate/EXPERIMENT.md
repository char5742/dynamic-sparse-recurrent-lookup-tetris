# Bellman residual-gate experiment contract

This track tests one specific mechanism: whether the positive but weak signal
from top-2 one-step Bellman reranking can be distilled into a no-lookahead
model. It does not use the sealed test seeds.

## BG00 — teacher dataset smoke

- Hypothesis: the old policy's second-ranked candidate is preferred by the
  registered `blend=0.5`, `discount=0.997` Bellman teacher often enough to
  provide a learnable correction signal.
- Mechanism: on teacher trajectories, evaluate only the old top two candidates
  with one-step bootstrapping and record the frozen 249-wide features.
- Changed component: training-data labels only. The historical checkpoint,
  engine, NEXT=5, and candidate order remain fixed.
- Success: at least 100 states, finite values, a non-zero/non-degenerate flip
  rate, and complete timing/hash metadata.
- Time limit: 10 minutes for the first smoke dataset.
- Stop: no flips, any non-finite value, or inability to reproduce the registered
  teacher choice.

## BG01 — zero-initialized residual gate

- Hypothesis: a small residual on the frozen historical representation can
  imitate beneficial top-2 flips without destroying the base policy.
- Mechanism: a shared `249 -> hidden -> 1` residual scorer is zero initialized;
  the decision logit is `(q2 + r2) - (q1 + r1)`. Weighted binary cross-entropy
  handles the expected class imbalance.
- Changed component: residual scorer only. At update zero the selected action is
  exactly the old top-1 action.
- Success: on held teacher episodes, balanced accuracy improves over update zero,
  both positive and negative examples are recognized, and the best checkpoint
  remains finite/deterministic.
- Time limit: 5 minutes after dataset generation.
- Stop: no positive labels, no held improvement, non-finite loss, or best update
  remains zero.

## BG02 — model-only game gate

- Hypothesis: the learned residual preserves or improves game score without any
  online lookahead.
- Mechanism: score the unchanged candidate set, apply the residual only to the
  old top two, and select once. No successor state is expanded.
- Model-only accounting: one logical candidate-scoring pipeline per decision,
  zero lookahead expansions, with all current prototype Q/feature backend
  requests reported separately. A fused deployment is required before a final
  efficiency claim.
- Development gate: paired, previously unused development seeds; mean difference
  at least +500, median positive, at least 70% non-negative differences, and no
  completion-rate regression.
- Time limit: first 100-piece episode, then at most 10 full development episodes
  only if the short gate is not catastrophic.
- Stop: early game-over/large regression, no learned flips, or offline gate failed.

The historical model and this model must never be compared by loss alone. Only
BG02 can promote this track to validation, and only the strict G2 validator can
authorize a sealed-test claim.
