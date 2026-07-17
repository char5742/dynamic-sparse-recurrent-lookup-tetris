# C13 milestone clean-room review

Date: 2026-07-18 JST
Scope: read-only review of the files named in the audit request. No game,
training, benchmark, or source-code execution was performed. This review did
not assume that the prior C13 decision was correct.

## Executive verdict

**Do not promote C13 to the three-seed strength screen yet.** The round was
genuinely preregistered and its DAgger data stage has substantial provenance,
but the supplied evidence does not establish a valid end-to-end C13 training
run. The requested training summary
`D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.json`
does not exist, and the smoke artifact identifies the resulting `.jld2` only
by path, not by hash. Consequently the warm start, update-0 reproduction,
fixed-snapshot selection, offline guard, and exact checkpoint used in the
smoke cannot be audited.

The seed-5751 artifacts do establish a narrower fact: both policies completed
the 50-piece smoke. That is a **smoke-completion observation**, not yet the
defined `C13-survival-pass`, because the earlier offline/provenance gate is
unproved. It is not evidence of improvement: the compact score was 1,100 and
the baseline score was 2,300.

| Audit question | Verdict |
|---|---|
| Preregistered before execution? | **Yes.** The decision tree was committed at 23:44:46 JST; that commit is an ancestor of the clean source commit recorded by the 23:49 DAgger artifact. |
| Data lineage and seed separation? | **Partially established.** The data-stage source, Manifest, teacher, rollout checkpoint, commands, and development-seed range are recorded; important per-seed and merged-output-hash evidence is absent. |
| Valid C13 training/checkpoint? | **Not established.** The designated summary is missing. |
| Survival-pass wording? | **Only conditionally valid.** “Both reached 50 on development seed 5751” is supported; the formal promotion label is not. |
| Run the 3 seeds x 100 pieces now? | **No.** The candidate is not auditably frozen and the offline gate is not evidenced. |
| Train a different mechanism now? | **No.** First reproduce the already-preregistered C13 training stage; changing the mechanism would evade the failed audit rather than resolve it. |
| Keep Zygote after the Enzyme retry? | **Yes for this workload.** The relative result is large and consistent, though its reproducibility record is incomplete. |

## 1. Preregistration, lineage, and seed separation

### What passes

The preregistration is temporally credible. Git history places
`reports/c13_decision_tree.md` in commit
`799dcc0f493e4d37d4c909bd07942342df6ead8e` at 23:44:46 JST. That commit is an
ancestor of artifact source commit
`0480ad6e20db203ff6673289579bbed68faa82b4` at 23:45:40. The DAgger artifact was
generated at 23:49:45. Thus the two offline/smoke gates, warm-start requirement,
500-update ceiling, fixed-snapshot rule, and interpretation of survival were
written before the observed execution.

The DAgger metadata binds the data stage to:

- a clean source commit plus source-fingerprint and Manifest hashes;
- the canonical old teacher hash
  `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`,
  matching `configs/evaluation_protocol.toml`;
- the C11b rollout checkpoint hash
  `3df9f4d233e235addc7bf9b5831c0823cff670261283d92c688efa0c8755e0d0`;
- an exact command and configuration selecting seeds 5742--5747, NEXT=5,
  six episodes, and the compact rollout policy.

The merge metadata hashes both inputs and preserves 1,500 base-training rows,
adds 660 DAgger rows, and keeps 500 original rows as the final two validation
episodes. The realized DAgger fraction is
`660 / (1500 + 660) = 0.3056`, inside the preregistered interpretable interval
of 0.15--0.50. The named artifacts use only development seeds: 5742--5747 for
DAgger, 5748--5749 for the offline guard by design, and 5751 for the smoke.
Neither validation seeds 8001--8008 nor test seeds 91001--91032 appear in the
recorded commands/configurations, and both dataset summaries state
`held_out_test_seeds_used=false`.

### What does not pass

The training/checkpoint link is missing at the decisive point. The designated
C13 training JSON is absent. None of the remaining allowed artifacts supplies
all of the following preregistered evidence:

- the input C11b hash used by training;
- update-0 top-1, MRR, correlation, and finite loss reproducing C11b;
- results at the fixed snapshots (preferably 250 and 500), the lexicographic
  selection, and the chosen update;
- original-development top-1 at or above 0.45;
- exact training seeds/configuration, source/Manifest binding, termination
  reason, and output-checkpoint hash.

The compact smoke names
`C13_round1_preregistered500_warm_c11b_best.jld2`, but records neither its byte
size nor SHA-256. A matching filename is not enough to prove that the smoked
bytes were the checkpoint selected by the preregistered procedure.

There are additional, secondary record defects:

- the merged output dataset has a path but no recorded size or hash;
- the DAgger summary gives only 660 total rows and 27,060 candidates, although
  the preregistration required per-rollout-seed row counts, terminal step, and
  score;
- the generator records a schema cap of 128 actions while the corrected compact
  pipeline is described as fixed at a true maximum of 74. The supplied metadata
  gives no realized maximum or explanation of this conversion. This is an
  unresolved contract discrepancy, not by itself proof that an invalid action
  set occurred;
- `reports/learning.md` still says no C13 rollout was executed, and
  `reports/experiment_ledger.md` ends at E017 without the required promoted-run
  record. Their timestamps explain how they became stale, but the contradiction
  confirms that the milestone record was not closed coherently.

The `held_out_test_seeds_used=false` flags and development-only commands are
positive evidence, but they cannot replace the missing training record.
Therefore seed separation is credible for data generation and the smoke, not
audited end to end for the trained checkpoint.

## 2. Meaning of the 50-piece result

The two smoke artifacts agree on seed 5751, NEXT=5, HOLD enabled, stable
candidate order, no lookahead, one logical scoring pass per decision, and a
50-piece cap.

| Policy | Score | Pieces | Game over | Candidate evals | Logical calls | Physical calls |
|---|---:|---:|---|---:|---:|---:|
| C13 compact, Lux CPU | 1,100 | 50 | false | 2,051 | 50 | 50 |
| Canonical old model, OpenVINO NPU/CPU-tail | 2,300 | 50 | false | 2,151 | 50 | 165 |

The differing candidate totals are compatible with different policy
trajectories. The differing physical-call totals follow the declared compact
all-candidate forward versus the historical model's 16-candidate chunks and do
not violate the common logical budget.

Both policies survived, so the smoke achieved its narrow kill-test purpose.
The paired score difference is **-1,200**, and a single short development seed
has no strength resolution anyway. The wording “survival pass only, no strength
claim” is methodologically sound in the preregistration. Applying the formal
`C13-survival-pass` label to the present evidence is premature, however,
because that label also requires a valid warm-start run and the offline top-1
gate. No improvement claim is accepted.

## 3. Milestone decision and the only permitted next computation

Do not spend the next budget on the unused three-seed x 100-piece strength
screen, FiLM, stronger labels, a temperature change, or another DAgger round.
The current failure is evidentiary, not evidence that the learning mechanism
failed.

**Permit exactly one computation next: one clean replay of the preregistered
C13 500-update warm-start training stage on the already-created aggregate, with
no game execution and no change to data, architecture, temperature, optimizer,
learning rate, loss, or checkpoint-selection rule.** Before updates, bind the
merged dataset and C11b input by SHA-256 and reproduce the C11b update-0 metrics;
evaluate only the preregistered fixed snapshots; emit a summary containing all
metrics, seeds, commands, source/Manifest hashes, selected update, termination
reason, and output-checkpoint SHA-256. This replay is the entire authorization;
it does not implicitly authorize a smoke rerun or the three-seed screen.

If that replay cannot establish the warm start and offline gate, close C13 as
invalid/rejected according to the preregistered branch. If it succeeds, a new
independent decision must determine how to handle the already-opened seed 5751
and whether any existing smoke can be cryptographically tied to identical
checkpoint bytes.

## 4. Enzyme retry fairness

The retry is sufficiently like-for-like to support the local decision. Both
backends used Julia 1.12.6, Lux 1.31.4, one Julia thread, ten BLAS threads,
batch 16, and the same 133,125-parameter workload. Numerical agreement passed:
gradient cosine was essentially 1, maximum gradient error was `7.08e-8`, and
one-step parameter maximum error was `3.89e-7`. The summary also records
reverse-with-primal use, preallocated/zeroed gradient shadow, no observed BLAS
fallback warning, and the same first-call discovery rule for both Lux API
backends.

The performance result is not marginal:

| Lux training API, 100 updates | Zygote | Enzyme | Enzyme relative result |
|---|---:|---:|---:|
| steady updates/s | 15.151 | 2.848 | 0.188x |
| steady median allocation/update | 33.34 MB | 186.97 MB | 5.61x more |
| wall time | 23.77 s | 58.51 s | 2.46x longer |

The lower-level preallocated-shadow comparison independently shows a 5.18x
Zygote steady-speed advantage, 5.61x Enzyme allocation multiplier, and 3.10x
Enzyme wall-time multiplier. These results fail both arms of the preregistered
Enzyme adoption gate by wide margins and agree in direction with E010.

Limit the inference: the supplied summary does not bind raw logs, source, or
Manifest hashes, the interrupted allocating-path retry is only described, and
the benchmark model has 133,125 rather than C13's 165,051 parameters. It does
not prove that Enzyme is inferior for every model or very long compiled run.
Those limitations are far too small to support an improvement claim or reverse
the present 5x-scale deficit. **Maintaining Zygote is justified for the current
compact-learning workload; no further Enzyme computation is authorized by this
review.**
