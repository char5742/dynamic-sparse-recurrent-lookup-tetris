# C13 round 1: preregistered promotion decision tree

Date: 2026-07-18 JST  
Scope: independent design review based only on the existing learning,
architecture, and FiLM records. This review ran no training, benchmark, or
game. It defines how to interpret C13 before reading its result.

## Executive decision

Use **C11b warm-start, not scratch training**, for C13 round 1. C13 is a
distribution-correction experiment: the architecture, old-teacher target,
temperature, optimizer, and search budget must remain unchanged. Starting from
scratch would spend most of the budget relearning the already demonstrated
teacher-trajectory ranking and would confound the effect of adding
student-visited states.

The round has two gates, in this order:

1. Preserve the original 5748--5749 offline-development top-1 agreement at
   `>= 0.45` after training on the aggregate.
2. If and only if gate 1 passes, run one fixed-budget 50-piece smoke on the next
   unused development seed. Completing 50 pieces is permission to continue,
   not evidence of beating the old model.

Do not introduce FiLM, temperature changes, TD loss, lookahead, a different
candidate budget, or a different learning rate in this round. If round 1
fails, follow the failure-specific branch below rather than tuning against the
same seed.

## What C13 can and cannot prove

C13 labels student-visited states with the unchanged 1313 teacher. It can show
that DAgger repairs compounding error and produces a fast approximation of the
old policy. It does **not** supply a preference that is stronger than the old
policy. Consequently:

- passing the offline and 50-piece gates means only that the compact policy is
  viable enough for a multi-seed development screen;
- matching or exceeding one short old-model score is not a model-improvement
  claim;
- if the compact model eventually exceeds the old model, this must be verified
  on paired unseen seeds and described as an empirical generalization result,
  not as an expected consequence of the teacher objective;
- a robust route beyond the teacher ultimately needs stronger labels or
  returns, such as frozen one-step/Bellman re-ranking on the same on-policy
  states.

## Frozen round-1 design

### Initialization

Warm-start from the exact C11b checkpoint:

`D:\tetris-paper-plus\checkpoints\learning\C11b_listwise_165k_lr1e3_3000_fixed74.jld2`

The model is the plain 165,051-parameter compact network: 8 channels, one
residual block, two projected spatial channels. Load `ps` and `st`; start a new
AdamW optimizer at learning rate `1e-3`. Reusing the optimizer state is neither
necessary nor possible from the current C11b artifact.

The existing `train_distillation.jl` defines `load_initial_checkpoint`, but in
the reviewed source that function is not connected to `TrainState`. Therefore
the run is valid as warm-start only if its provenance explicitly records the
input checkpoint hash and the pre-update parameters reproduce C11b metrics.
Merely setting an environment variable is not sufficient evidence.

Required pre-update check on the unchanged 5748--5749 rows:

- top-1 agreement approximately `0.484`;
- MRR approximately `0.6452`;
- correlation approximately `0.8513`;
- finite output and loss.

A materially random-looking initial metric invalidates the run as C13. It is a
scratch run and must not be compared as the preregistered distribution-only
change.

### Dataset and mixture

- Rollout policy: frozen C11b student.
- Rollout seeds: 5742--5747 only.
- Labeler: unchanged 1313 OpenVINO teacher, all valid candidates labelled.
- Base training rows: original episodes 1--6 / 1,500 states.
- Offline-development rows: original seeds 5748--5749 / 500 states, unchanged
  and appended only as the last two episode IDs.
- Sealed validation and test seeds: never loaded or executed.

Record the number of DAgger states per rollout seed, total candidates, terminal
step, score, and the realised training-state fraction

`dagger_states / (1500 + dagger_states)`.

Uniform row sampling makes the effective on-policy dose depend on how early
the student dies. Interpret that dose as follows:

| DAgger fraction | Interpretation |
|---:|---|
| `< 0.15` | Inadequate exposure. A negative result does not reject DAgger. |
| `0.15--0.50` | Useful round-1 dose; the result is interpretable. |
| `> 0.50` | Forgetting risk; the original-development guard is decisive. |

No post-result reweighting is allowed in round 1. If exposure is inadequate,
the only permitted retry is a preregistered stratified aggregate with a fixed
25--50% DAgger minibatch share; do not silently duplicate rows until a metric
improves.

### Target scale and loss

Keep the C11b temperature `1.0`. C12 already showed that temperature `0.25`
was worse at the 300-update gate (top-1 0.346 versus 0.370 and MRR 0.5113
versus 0.5272). Changing temperature during C13 would confound the distribution
intervention.

The current objective standardizes teacher Q and student output independently
inside every complete candidate set and applies masked ListNet cross entropy.
This is suitable for policy-ranking distillation because it removes the
teacher's arbitrary per-state offset and scale. It also has important limits:

- the compact output is a **ranking logit**, not a calibrated Q value;
- absolute teacher margins and cross-state value scale are discarded;
- affine changes to student output are invisible to the loss;
- the objective cannot be mixed directly with Bellman targets and interpreted
  as one consistently scaled Q head;
- candidate-count and near-tie behaviour can influence the listwise entropy.

Those limits do not justify changing C13. They do mean that a later RL phase
should use a separate calibrated return/Q head or an explicitly fixed global
target transform. A stronger-teacher phase should preserve ranking and margin
information rather than treating the standardized logit as a true Q value.

### Updates and checkpoint selection

Use only the preregistered short warm-start window (at most 500 updates and at
most 5 minutes after compilation). Evaluate at update 0 and at fixed snapshots
chosen before execution, preferably 250 and 500. Select among those snapshots
lexicographically by original-development top-1, MRR, then lower listwise CE.
Do not add another checkpoint after seeing the curve.

Immediate stops:

- any NaN/Inf;
- candidate count above the fixed schema;
- a rollout, training, or validation seed outside the declared sets;
- provenance/hash mismatch;
- original-development top-1 below 0.45 at all fixed snapshots;
- wall time above 5 minutes after compilation.

The 0.45 gate tolerates a small decrease from C11b's 0.484 while preventing
catastrophic forgetting. Training-set or DAgger-row agreement is diagnostic
only and cannot replace this gate.

## Post-run decision tree

```text
provenance and warm-start checks valid?
|-- no  -> INVALID RUN; fix wiring/provenance and rerun the same C13 once
`-- yes
    |
    |-- no finite aggregate or DAgger fraction < 0.15
    |      -> DATA-DOSE FAILURE; one stratified 25--50% DAgger retry only
    |
    `-- DAgger fraction >= 0.15
        |
        |-- original-dev top-1 < 0.45
        |      -> FORGETTING/OPTIMIZATION FAILURE
        |         one shorter fixed-mixture retry; then close plain old-Q DAgger
        |
        `-- original-dev top-1 >= 0.45
            |
            `-- freeze chosen snapshot; paired 50-piece smoke on seed 5751 once
                |
                |-- student game-over before 50
                |      -> POLICY FAILURE; no FiLM sweep on the same labels
                |         move to stronger on-policy re-ranking labels
                |
                `-- both complete 50
                       -> SURVIVAL PASS ONLY
                          freeze candidate and run a preregistered multi-seed
                          paired development screen; no tuning from seed 5751
```

### One 50-piece development seed

One seed is appropriate only as a cheap kill test for the previously observed
34-piece catastrophic failure. Run the candidate and canonical baseline under
identical NEXT=5, candidate generation, one logical network pass per decision,
no lookahead, and a 50-piece cap. Record score, completion, candidate
evaluations, logical calls, physical backend requests, inference time, and
wall time for both.

Passing criterion: both policies reach 50 pieces and the candidate has no
numerical/runtime failure. Record the paired score but do not use its sign to
tune C13 and do not call the result an improvement. Seed 5751 becomes a
development screening seed after this one use. Prefer unused development seeds
for the subsequent multi-seed screen rather than replaying 5751 until it
passes.

The next strength-oriented gate should use a frozen candidate on three unused
development seeds, identical 100-piece budgets, paired mean and median
difference above zero, at least two of three non-negative paired differences,
and no completion regression. This is still branch selection, not G2.

## Failure-specific next action

### A. Too little on-policy data

If early deaths yield less than 15% DAgger rows, do not change the model. Run
one additional aggregation round with a fixed stratified sampler so that
25--50% of each minibatch is drawn from student-visited states and the rest
from the original teacher trajectories. Warm-start from the best finite C13
snapshot, use new training-only rollout states, retain the same 5748--5749
guard, and keep the total retry within 20 minutes. Failure after this dose closes
plain old-Q DAgger.

### B. Offline guard fails with adequate DAgger dose

This is catastrophic forgetting or unstable optimization, not an architecture
victory. Permit one shorter warm-start retry with a fixed base/DAgger mixture
and no new hyperparameter sweep. If top-1 still stays below 0.45, close the
branch. Scratch training is not the next action: it is slower and makes the
distribution intervention less identifiable.

### C. Offline guard passes but the 50-piece smoke fails

This is evidence that higher old-teacher agreement on the available states is
still insufficient. Do not spend the next budget on FiLM or more epochs over
the same old-Q labels. Generate a stronger, frozen target on student-visited
states (for example old-model one-step/Bellman re-ranking at a fixed expansion
budget), first prove that target's selected actions have positive paired proxy
gain against old Q, then distill those complete candidate rankings. This is the
shortest branch that introduces a mechanism capable of beating the teacher.

### D. Survival passes but the multi-seed strength gate fails

The compact old-policy clone is useful infrastructure, but not the final model.
Use it as the rollout initializer for stronger on-policy targets. At most one
additional DAgger round is justified if the failures correlate with uncovered
student states; otherwise stop cloning the old Q.

## FiLM timing

Do **not** run the existing FiLM A/B before C13 establishes a viable plain
policy. FiLM changes representation while C13 changes state distribution, so
running both together would destroy attribution. Moreover, the existing F01
plan uses temperature `0.25` and says it matches the strongest learning screen;
the current learning record shows the opposite. Any future FiLM A/B must use
the winning temperature `1.0` unless a new, separately preregistered reason is
given.

FiLM is eligible only after either:

1. plain C13 survives the short game but ranking plateaus in a way consistent
   with missing queue/board interaction; or
2. stronger re-ranking labels exist and representation capacity becomes the
   measured bottleneck.

Then perform a single matched A/B on the same aggregate, same warm-started
shared weights, zero-initialized FiLM projection, same row schedule, and same
update budget. Require higher original-development top-1 and MRR with no
disproportionate latency, followed by a fresh paired game gate. Offline FiLM
gain alone is not promotion.

## Time and leakage budget

| Stage | Maximum | Stop condition |
|---|---:|---|
| DAgger generation | 15 min | invalid value/schema/seed/hash |
| Warm-start training | 5 min after compile | non-finite or all snapshots `<0.45` |
| One paired 50-piece smoke | 5 min target | candidate early game-over or runtime mismatch |
| One dose/mixture retry, if justified | 20 min total | second failure closes plain old-Q DAgger |

Development seeds 5742--5747 are training rollouts, 5748--5749 are the reused
offline-development guard, and 5751 is a one-time short smoke only. Do not use
validation 8001--8008 or sealed test 91001--91032 at this stage. Do not inspect
a test result and return to architecture, loss, temperature, mixture, or update
selection. Every artifact must bind source/Manifest, dataset, old teacher,
rollout checkpoint, output checkpoint, exact command, rules, backend, and seed
hashes before execution.

## Promotion labels

- **C13-invalid:** provenance/warm-start/data contract failed.
- **C13-rejected:** adequate data and valid run, but offline guard or 50-piece
  survival failed.
- **C13-survival-pass:** offline guard and one 50-piece smoke passed; no strength
  claim.
- **development-promoted:** frozen checkpoint passes the later three-seed
  paired strength gate.
- **old-model exceeded:** reserved for the separately frozen validation/G2
  protocol, never assigned from C13 itself.

