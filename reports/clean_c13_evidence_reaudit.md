# C13 evidence clean-room re-audit

Date: 2026-07-18 JST  
Scope: designated reports, machine-readable ledger, C13 training/data
artifacts, and the two seed-5751 smoke artifacts. This audit performed only
read-only file, Git-history, metadata, size, and SHA-256 inspection. It ran no
game, training, or benchmark and made no source-code change.

## Executive verdict

The earlier negative audit is **not upheld**. Its decisive premise came from a
requester-supplied filename that is not an artifact of this run:
`C13_round1_preregistered500_warm_c11b_best.json`. The actual run-level summary
is `C13_round1_preregistered500_warm_c11b.json`; it names the selected
`_best.jld2` inside the record. Absence of `_best.json` is therefore an input
mistake, not a C13 failure.

| Question | Independent verdict |
|---|---|
| Valid preregistered 500-update warm start? | **Yes.** The correct summary records C11b input bytes/hash, exact update-0 reproduction, 500 actual updates, only snapshots 0/250/500, and normal budget completion. |
| Offline gate and best update? | **Pass; update 250.** Top-1 is 0.484 at update 250, above the frozen 0.45 gate. The registered lexicographic rule selects 250 over 0 by MRR and over 500 by top-1. |
| Can the smoke be joined to the selected checkpoint bytes? | **Sufficient for this development milestone, but not by a self-contained cryptographic attestation in the smoke JSON.** The ensemble of exact path, file chronology, committed post-run hash, and current-byte match is strong non-adversarial evidence. |
| Genuine defect severity | **Low and non-blocking for `C13-survival-pass`; medium if left unfixed for later promotion evidence.** Both training and compact-evaluator JSON omit a selected-output fingerprint. |
| Retrain or replay seed 5751? | **No.** Neither would repair the historical metadata omission, and replaying the one-time smoke would violate its intended use. |
| Permit the development 3-seed x 100-piece screen? | **Yes, once, with the frozen bytes and hash recorded directly in its outputs.** |

C13 therefore retains the precise label **`C13-survival-pass`**. This is not a
strength or old-model-improvement claim: on seed 5751 the candidate completed
50 pieces but scored 1,100 versus the old model's 2,300.

## 1. Correction of the prior audit

The correct training summary exists, is 6,419 bytes, and parses as a complete
run record. Its material fields are also present in the final line of
`experiments/learning/ledger.jsonl`. In particular it supplies every training
fact that the earlier review said was unavailable: initializer fingerprint,
update-0 metrics, fixed validation history, selected update, offline gate
metric, source/Manifest provenance, exact command and configuration, learning
seed, dataset fingerprint, termination reason, and wall time.

The prior review also says that `reports/learning.md` still reported no C13
rollout and `reports/experiment_ledger.md` ended at E017. The files actually
designated for this re-audit contain the realized C13 data/training/smoke record
and E018. Git places their C13 commit at 00:06:59 JST and the machine-readable
ledger commit at 00:07:58 JST. Those statements from the prior review cannot be
used as evidence about the present artifact set.

This correction does not erase real record weaknesses. It separates them from
the nonexistent-filename error: the selected checkpoint hash was written into
the human ledger after evaluation, not emitted by either the training summary
or compact smoke evaluator itself.

## 2. Training, warm-start, and offline gate

Preregistration commit `799dcc0f493e4d37d4c909bd07942342df6ead8e`
(23:44:46 JST) is an ancestor of the clean run-source commit
`0480ad6e20db203ff6673289579bbed68faa82b4` (23:45:40 JST). The relevant
registered constraints and the observed record agree:

- The aggregate actually used by training is 85,281,241 bytes with SHA-256
  `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba`.
  This exactly matches the training summary's dataset fingerprint. The merge
  record contains 1,500 base-training rows, 660 DAgger rows, and the unchanged
  500-row development split. The realized DAgger share is 30.56%, within the
  preregistered 15--50% interpretable range.
- The recorded initializer is the C11b checkpoint, 1,351,340 bytes, SHA-256
  `3df9f4d233e235addc7bf9b5831c0823cff670261283d92c688efa0c8755e0d0`.
  A fresh read-only hash of that file matches both recorded values.
- Update 0 reproduces C11b: top-1 0.484, MRR 0.6451578, correlation 0.8512917,
  and finite CE 3.4219624. This is affirmative evidence that the initializer
  was loaded; it is not a random-looking scratch initialization.
- `actual_distill_steps=500`, `distill_stop_reason="configured update budget
  completed"`, and the validation history contains exactly updates 0, 250,
  and 500. Wall time is 60.697 seconds, below the five-minute post-compile
  ceiling.

| Update | Top-1 | MRR | Correlation | CE | Registered selection result |
|---:|---:|---:|---:|---:|---|
| 0 | 0.484 | 0.6452 | 0.8513 | 3.4220 | superseded by 250's higher MRR |
| 250 | **0.484** | **0.6526** | 0.8491 | 3.4210 | **selected; offline gate passes** |
| 500 | 0.478 | 0.6333 | 0.8556 | 3.4178 | loses first on top-1 |

The selection follows the frozen lexicographic key exactly: top-1, then MRR,
then lower CE. The separately mentioned mistakenly configured 3,000-update run
does not contaminate this record; the audited run itself contains an explicit
500-update configuration, 500 actual updates, and the registered snapshot set.

## 3. Exact artifact inspection

All timestamps below are filesystem last-write times in JST. Case differences
between recorded lowercase and `Get-FileHash` uppercase representations are
not value differences.

| Artifact | Bytes | Last write (JST) | Actual SHA-256 |
|---|---:|---|---|
| `reports/c13_decision_tree.md` | 13,450 | 2026-07-17 23:44:20.591 | `8588c711c1cd6666cd0e831114182803ba8139266ac73d0a8e2306e3a0748ef0` |
| `reports/clean_c13_milestone_review.md` | 10,465 | 2026-07-18 00:09:17.179 | `4c8e1e5cf970b2d3d5ecda811ca4289bfe0e1554271851ce8a39bb425cd00a02` |
| `reports/learning.md` | 10,234 | 2026-07-18 00:05:46.041 | `6c09c182c9af09f729a5db7065d92c727c8c64d3c60c2b482c5db5f1ab64ccf4` |
| `reports/experiment_ledger.md` | 9,945 | 2026-07-18 00:06:25.291 | `9bd30ff7c010440b720c81765af8e37e3ad7e9770d4602b6d83a24f9e267bd21` |
| `experiments/learning/ledger.jsonl` | 13,493 | 2026-07-18 00:03:19.716 | `10bcac52eaf68cd52c05888e7631711e4023b73504d8e27de9a17a3bde75dcdc` |
| training summary `.json` | 6,419 | 2026-07-18 00:03:19.397 | `19afe7f5bd49aa9d804cb5d504fb741b846367d3ee813c6178c37ee70664694b` |
| selected update-250 `_best.jld2` | 701,561 | 2026-07-18 00:03:18.249 | `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270` |
| aggregate dataset `.json` | 2,803 | 2026-07-17 23:59:59.085 | `180c7c5f8e551f8f00bdf8decd36259eecb443011743da5d0b02e3f1ef2b7e13` |
| aggregate dataset `.jld2` | 85,281,241 | 2026-07-17 23:59:58.645 | `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba` |
| compact seed-5751 smoke `.json` | 1,686 | 2026-07-18 00:04:07.337 | `8772162009abbf481a6e00e747a47b76975d6632f8f90864b744953ff7cddbd9` |
| old-model seed-5751 smoke `.json` | 1,206 | 2026-07-18 00:04:45.876 | `367efbc59717c78eb063f5dca51d95cde991a67ee8e2aa91c434fcda6ba87cdb` |

The aggregate, selected checkpoint, and two smoke hashes read from disk match
the values committed in the C13 reports. The two smoke JSONs also agree on
seed 5751, NEXT=5, HOLD, stable candidate order, no lookahead, one logical call
per decision, and a 50-piece cap. Both complete all 50 pieces without game
over.

## 4. How strongly the smoke is bound to checkpoint bytes

The candidate smoke records the exact selected path
`D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2`.
That is the same path named by the training summary. The file was created at
00:02:57.860 JST and last written at 00:03:18.249; the training JSON was then
written at 00:03:19.397. The smoke's internal timestamp is 00:04:06.569 and
its file was written at 00:04:07.337. The selected checkpoint's current bytes
still hash to the value committed in the C13 record at 00:06:59.

This chain is sufficient to identify the smoke input as the recorded
update-250 bytes under the normal, non-adversarial standard used for a
development kill test. It supports applying `C13-survival-pass` without a
retrain or a smoke replay.

It is not a cryptographic proof captured at evaluator load time. Specifically:

- the training JSON names the best path but does not emit its size/hash;
- the compact smoke JSON names that path but does not emit its size/hash;
- no adjacent `_best.sha256`, `_best.jld2.sha256`, or `_best.json` exists;
- the first explicit selected-checkpoint hash is in a post-smoke committed
  report, so a hypothetical replacement with restored filesystem timestamps
  cannot be excluded from the run JSON alone.

This is a genuine provenance-completeness defect, but it is not evidence that
replacement occurred. Treating it as run-invalidating would conflate lack of
same-run cryptographic attestation with positive evidence of wrong bytes. Its
present severity is low/non-blocking because the path is unique, the time
ordering is coherent, the recorded and current hashes agree, and the result is
only a development survival gate. It becomes medium and promotion-blocking if
the next evaluator again omits the candidate and baseline fingerprints.

## 5. Remediation and the only next computation

Do **not** retrain C13. A retrain would create different bytes and cannot improve
the historical smoke's identity evidence. Do **not** replay seed 5751: the
existing smoke already served its one-time kill-test role, and another look at
the same seed adds tuning exposure without resolving the historical omission.

Record the frozen candidate now as 701,561 bytes and SHA-256
`1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270` in a
checksum sidecar or equivalent immutable run manifest. Future training and
evaluation exporters should compute and embed input path, byte size, and
SHA-256 at load/write time for both candidate and baseline. A sidecar created
now does not retroactively strengthen the smoke JSON, but it prevents ambiguity
from this point onward. These are evidence-hardening actions, not grounds for
new learning.

**Authorize exactly one next computation: one paired development strength
screen on the next unused development seeds 5752, 5753, and 5754, with a
100-piece cap per policy per seed.** Use the frozen C13 bytes above and the
canonical old baseline; NEXT=5, HOLD, stable candidate order, no lookahead, and
one logical network call per decision must remain identical. The output must
directly record both checkpoint fingerprints, rules/search budget, source and
Manifest identity, per-seed completion and score, candidate evaluations,
logical/physical calls, inference and wall time, and termination reason.

Apply the already registered gate without post-result tuning: paired mean and
median score difference above zero, at least two of three paired differences
non-negative, and no completion regression. Passing permits only the label
`development-promoted`; failing sends C13 to the preregistered stronger-target
branch. This authorization includes no training, FiLM/temperature/mixture
change, seed-5751 replay, validation/test seed, or additional screen.
