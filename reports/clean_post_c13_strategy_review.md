# Post-C13 clean strategy review

Date: 2026-07-18 JST  
Scope: evidence-only strategy audit. No game, training, validation/test seed, or
code was run or changed for this review.

## Decision

**Close plain C13 old-Q DAgger as a strength branch. Permit only option D: one
frozen S1 top-2 one-step Bellman-lookahead system screen.**

C13 round 1 remains useful as evidence that on-policy aggregation repaired the
34-piece catastrophic-survival failure. It is not close to the old policy in
score. Another round with the same old-Q teacher is therefore not the shortest
route to an actual old-model beat. Retain the C13 checkpoint only as
infrastructure or a possible future rollout initializer; do not spend another
training run on unchanged old-Q labels.

Option D is a system candidate, not a model-only improvement. Its extra search
must be reported explicitly and it cannot be used for a G2 model claim. It is
selected because it can test the only existing action-improvement mechanism
directly, before paying to distill that mechanism into a learner.

## Decisive evidence

The strength screen was frozen before execution at commit
`b6e8caa61ded005f141ce8edd89379aa5d00e7c2`. Both agents used NEXT=5, stable
candidate order, HOLD, zero lookahead, one logical network call per decision,
and 100 placed pieces. The compact checkpoint was identical in all three files:
SHA-256
`1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`.
All six episodes completed, so survival cannot explain the score gap.

| Development seed | C13 | old 1313 | paired difference |
|---:|---:|---:|---:|
| 5752 | 1,900 | 6,200 | -4,300 |
| 5753 | 1,100 | 5,100 | -4,000 |
| 5754 | 2,800 | 5,800 | -3,000 |
| mean | 1,933.3 | 5,700.0 | **-3,766.7** |
| median | 1,900 | 5,800 | **-4,000** |

The preregistered strength gate required positive paired mean and median,
at least two non-negative differences, and no completion regression. C13
passed only completion: the win/tie/loss count was `0/0/3`. This is coherent,
same-budget, multi-seed negative evidence, not a marginal or noisy miss. The
earlier 50-piece result was also negative in score (1,100 versus 2,300), though
it was correctly used only as a survival gate.

The mechanism diagnosis is correspondingly narrow:

- C13 established that DAgger coverage can prevent immediate collapse.
- Its target remains the unchanged old-Q ranking. More exposure can improve
  cloning but supplies no preference that is better than that teacher.
- The new three-seed result is uniformly far below the teacher. A second round
  would first need to recover several thousand points merely to match it.

The existing stronger-action evidence is positive but weak. E005 reports that
ungated top-2 Bellman reranking changed two development results by `+1,100` and
`0`, at roughly 2.0 times the candidate evaluations and 2.7 times the inference
time. E011's 29 flips in 180 training-only states shows that the target is
non-degenerate, not that it is stronger. E012's learned residual gate then
scored 4,800 versus 5,800 on its 100-piece game; its calibrated proxy gain was
tiny and was not a listwise learner integration. None of these records alone
authorizes training or a strength claim. They justify only a strict direct
screen of the frozen teacher mechanism.

## Why the other choices are not authorized

| Option | Decision | Time-to-actual-old-beat assessment |
|---|---|---|
| A — C13 round 2 | close | Same old-Q target, after a uniform 3/3 and 3,000--4,300 point deficit. It adds coverage, not a stronger decision rule. |
| B — 800k FiLM with old-Q | reject | FiLM may model board/queue interaction better, but the target ceiling is unchanged. The 800,133-parameter systems result is not a game result, and the matched compact FiLM A/B was never trained. |
| C — on-policy learning from stronger/Bellman labels | defer, not authorized | This is the best later model mechanism, but its proposed teacher has not passed a strong direct screen. Training first would add label generation, optimization, covariate shift, and integration failure modes before establishing that the target beats old Q. |
| D — S1 system screen | **only permitted action** | Directly tests the stronger decision mechanism, needs no learner, and can produce paired game evidence in minutes. |
| E — other | reject | No documented alternative has a shorter measured path with a causal teacher-beating mechanism. |

The Reactant+EnzymeMLIR result does not rescue B or C. Its numerical proxy
matched Zygote and its fixed-shape steady update was about 12.2 times faster,
but the preregistered compile-inclusive gate failed at both 100 updates
(67.39 s versus 22.05 s) and 500 updates (69.29 s versus 50.08 s), with an
interpolated break-even near update 799. More importantly, the artifact limits
its claim to a fixed-shape batch-16 CPU learner proxy. Persistent real replay
and the actual listwise learner have not been integrated or verified. It must
not be credited as realized time savings in this strategy choice.

## Sole authorized experiment: D / S1 short screen

### Hypothesis and mechanism

**Hypothesis:** on previously unused development trajectories, frozen ungated
top-2 one-step Bellman reranking of the historical Q policy yields a material
and directionally consistent paired score gain over the historical argmax-Q
policy within a bounded system cost.

S1 keeps the historical checkpoint fixed and reranks only its two highest-Q
root candidates using immediate reward plus discounted maximum old-Q at the
successor. Freeze `top_k=2`, `blend=0.5`, `discount=0.997`, and
`q_margin_threshold=Inf` (rerank every decision). Do not tune these values.
The historical checkpoint SHA-256 is
`7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`.

The current reviewed S1 evaluator is
`scripts/evaluate_openvino_lookahead.jl`, SHA-256
`0e8547534048f861ed809767756425d0fd417bbf573aa76b79e9d051df205470`.
The protocol SHA-256 is
`4b87f1d00c187466c0eeb4aca85066438b8847776092e5166fb0f88a790c12c5`.
Because `reports/system_candidate_preregistration.md` is absent and the current
prototype JSON does not record all protocol-required logical/physical call
counts or artifact hashes, execution is valid only after a machine-readable
pre-execution freeze binds the exact source, Manifest, protocol, evaluator,
checkpoint, commands, S1 constants, output paths, and accounting fields. This
is provenance/accounting hardening, not permission to change S1 semantics.

### Frozen comparison

- Seeds: development `5755`, `5756`, and `5757`, each exactly once per policy.
  These are the only unused development seeds in the frozen protocol.
- Never load or execute validation seeds `8001--8008` or test seeds
  `91001--91032` in this screen.
- Budget: 100 placed pieces, NEXT=5, HOLD enabled, stable candidate order and
  score rules from the frozen protocol.
- Baseline: historical argmax-Q policy, zero lookahead, one logical complete
  candidate-set score per decision.
- Candidate: the exact S1 constants above. No gate, learned residual, FiLM,
  compact checkpoint, margin rule, beam-width sweep, or fallback configuration.
- Record per seed and policy: score, completion/steps, root and successor
  candidate evaluations, lookahead expansions, logical model passes, physical
  backend requests, generation seconds, inference seconds, and wall seconds.
- Run paired seeds in a fixed preregistered order. Do not replay 5752--5754 and
  do not rerun a noisy or failed pair.

### Success metric

All conditions are required:

1. all six episodes complete 100 pieces with no numerical or runtime failure;
2. S1 has a strictly positive paired difference on **all three** seeds;
3. paired mean difference is at least `+500` points;
4. paired median difference is strictly positive;
5. completion does not regress; and
6. complete system-cost/provenance fields are present and hash-valid.

The `3/3` rule is the integer form of the existing 70% non-negative gate and
prevents the earlier `+1,100, 0` two-seed signal from being treated as adequate.
Even a pass is only `S1-development-promoted`; it is not model-only evidence,
not validation, and not a sealed-test/G2 claim.

### Time limit and immediate stops

Hard limit: **12 minutes total wall time**, including compilation and all six
100-piece episodes. Also stop if any single episode exceeds 150 seconds.

Stop the screen immediately, with no substitution or retry, on any of:

- hash, command, seed-role, or S1-constant mismatch;
- non-finite output, candidate-order/schema mismatch, or missing required
  accounting;
- candidate or baseline game-over before 100 pieces;
- the first completed pair with S1 difference `<= 0`, because the `3/3` gate
  has then become impossible;
- total or per-episode time limit exceeded; or
- any attempt to inspect a validation/test seed or tune from an observed seed.

Failure closes this exact S1 configuration. It does not automatically authorize
A, B, C, another lookahead setting, or reuse of the three seeds. A pass freezes
S1 for a separate later review; no learning run is implicitly authorized.

## Audit limitations

`reports/bellman_gate.md` and `reports/system_candidate_preregistration.md` were
not present in the reviewed worktree. This review therefore relies on the
requested ledger, architecture/FiLM/learning reports, the frozen C13 screen and
episode JSONs, the Reactant proxy summary, and the frozen evaluation protocol.
No conclusion here assumes unrecorded Bellman or system results.
