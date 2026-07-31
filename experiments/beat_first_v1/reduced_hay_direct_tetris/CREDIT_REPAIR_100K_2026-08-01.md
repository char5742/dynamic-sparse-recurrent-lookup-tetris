# Reduced Hay v2 local-credit repair and 100k scratch run — 2026-08-01

## Outcome

The local DECOLLE/e-prop training path is now mechanically consistent enough
to memorize one deterministic teacher state while keeping every recurrent
parameter group trainable.  The exact analytic head VJP and the exact-BPTT
reference both pass their focused derivative tests.

The repaired width-80 production model nevertheless does **not** keep improving
through 100,000 updates.  Held ranking quality peaks around 10k, is roughly
flat through 50k, and is worse at 100k.  This is a negative architecture and
credit-assignment result, not a successful 100k quality result.

## Credit-assignment repairs

The retained repair changes are:

1. Ordered Plackett–Luce routing receives one candidate/cycle supervised
   advantage, centred and RMS-capped across candidates, with the required
   inverse-candidate reduction.  It is recorded as a
   `supervised_reward_surrogate`, not an environment return.
2. The recurrent third factor no longer depends on `head_weight` or
   `output_weight`.  Each block combines a seed-fixed, non-trainable
   `22 -> node_dim` projection with trainable block-local Q, death, quantile and
   geometry predictors.
3. Third-factor RMS handling is a cap, not forced unit RMS.  A small error can
   therefore produce a small recurrent update near a fitted example.
4. Recurrent, local-predictor and supervised-head gradients are clipped as
   separate groups.  The recurrent trust step is `0.001` of the base AdamW
   step; no recurrent parameter is frozen.
5. The analytic supervised VJP remains confined to the four head/output
   parameter fields.  Recurrent and cell parameters remain DECOLLE/e-prop.
6. Non-overfit runs now store an empty `overfit_rows` field instead of copying
   the complete training split into every checkpoint and incorrectly enabling
   `--split overfit`.

Exact BPTT itself did not contain a derivative error: the finite-difference and
Zygote checks passed.  “BP repair” in this result therefore means correcting
the local-credit approximation and its optimizer boundary, not replacing
DECOLLE/e-prop with full BPTT.

## Focused verification

The following critical-path checks passed before the retained run:

| Check | Result |
|---|---:|
| exact Reduced-Hay/BPTT and finite-difference suite | 66 / 66 |
| allocation-free SoA/event kernel suite | 6 / 6 |
| fixed-arena DECOLLE/e-prop suite | 246 / 246 |

The local suite was rerun after the final checkpoint-metadata correction and
remained `246 / 246`.

### Fixed-data memorization

ListNet cross-entropy has a non-zero teacher-entropy floor, so the zeroable
quantities are `composite - teacher_entropy` and ListNet KL.

One deterministic teacher state, full recurrent learning, 2,000 updates:

| Metric | Result |
|---|---:|
| excess loss | 0.000222683 |
| ListNet KL | 0.000150919 |
| top-1 | 1.000000 |
| NDCG | 0.999968 |
| pairwise | 0.994958 |

Eight deterministic states at 2,000 updates reached excess loss `0.042208`,
ListNet KL `0.038203` and top-1 `0.875`.  This is not a zero-loss result, but
it confirms that the one-state pass is not produced by freezing the recurrent
path.

### Exact/local shadow gradient

After 128 local-predictor warm-up updates on the tiny recurrent control:

- exact and local forward/loss values were bit-aligned;
- supervised-head cosine was `1.0`;
- aggregate recurrent exact/local cosine was `0.153014`;
- 28 of 30 recurrent fields had positive cosine;
- `apical_leak_logits` was `-0.2425`, but its exact norm was only `0.00296`;
- `feedback_gain` was `-0.0469`.

This is evidence that the approximation is directionally useful in aggregate,
not evidence that it reproduces exact BPTT.  The shadow test uses the tiny
control because exact BPTT of the 768-cell, batch-8, width-80 production model
is not a practical CPU audit.

## Retained 100k protocol and integrity

- source revision: `57e8e09d6abfaab392fbc95c3d1bc958906d2e96`;
- preset: `reduced_hay_scaled_v2`;
- width / state batch: `80 / 8`;
- workers / BLAS threads: `20 / 1`;
- stochastic Plackett–Luce routing during training;
- base learning rate / recurrent multiplier: `5e-4 / 0.001`;
- credit warm-up / ramp: `128 / 512` updates;
- trained updates / teacher states: `100,000 / 800,000`;
- wall time: `13,379.674 s`;
- end-to-end throughput: `7.474 updates/s` or `59.792 states/s`;
- measured hot GC: `0 s`;
- stderr: empty;
- final checkpoint SHA-256:
  `4fa084e370646c02d6b04693e7663c28b2883e79a5750b8e2eaff2da1cc9277e`.

The final update's scalar loss is not used as the quality result because batch
difficulty varies.  All retained comparisons below use deterministic routing
on the same 128 predefined validation rows, with panel SHA-256
`9ad529fd3a439d27d7b041a29a5e228bdad233577aefd35225f7058eb989fdc6`.

## Held learning curve

| Update | Composite | Excess | ListNet KL | Top-1 | NDCG | Pairwise |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 3.767540 | 1.081593 | 0.968964 | 0.085938 | 0.858752 | 0.568949 |
| 10,000 | **3.748589** | **1.062643** | **0.947991** | **0.164063** | 0.870819 | 0.580845 |
| 50,000 | 3.752322 | 1.066376 | 0.955676 | 0.132813 | **0.873459** | **0.582351** |
| 100,000 | 3.767749 | 1.081803 | 0.969869 | 0.085938 | 0.862767 | 0.574327 |

The 100k training-panel check is similar rather than much better: composite
`3.733874`, top-1 `0.093750`, NDCG `0.863567` and pairwise `0.574839`.
Consequently the regression is not explained by a conventional large
train/validation generalization gap.

Training-window means tell the same story.  Composite loss falls from
`3.794736` in updates 1–10k to about `3.761` during 20–60k, then remains flat;
the 90–100k mean is `3.767477`.  Mean gradient norm rises from `17.56` to
`21.31`, while firing rate rises gradually from `0.00700` to `0.00923` and
plateau mean from `0.00584` to `0.00699`.  The old plateau explosion is absent.

## Model-gradient audit at 100k

Every recurrent parameter field changed from initialization and retained a
non-zero Adam moment.  Maximum absolute deltas include:

| Field | Max absolute delta |
|---|---:|
| workspace key | 0.012768 |
| state-query weight | 0.014083 |
| typical compartment logits | about 0.045–0.048 |
| synapse weight | 0.106836 |
| delay logits | 0.119658 |
| gate logits | 0.589061 |

Thus the recurrent path is not frozen and has not lost all learning signal.
On the first copied batch after the final checkpoint, the raw group norms were:

| Group | Raw norm | Applied clip scale |
|---|---:|---:|
| recurrent/cell | 24.4459 | 0.204533 |
| local predictors | <= 5 | 1.0 |
| supervised head | <= 5 | 1.0 |

The large total trace norm is therefore recurrent e-prop pressure, not head
gradient clipping.  The `0.001` recurrent learning-rate multiplier prevents
the earlier destructive update, but it also leaves the supervised head to
adapt much faster than the features.  By 100k, maximum absolute changes are
`29.0173` for `local_readout` and `17.5945` for `output_weight`.  The hidden
head is RMS-normalized before a fixed-scale `tanh`, so this is not the old
uncontrolled hidden-tanh saturation bug; it is long-horizon predictor/output
drift on features that improve too slowly.

## Comparison boundary

### Same architecture and identical panel

The halted pre-repair local run at update 54k used the same scaled preset and
the same 128-row panel:

| Run | Update | Loss | Top-1 | NDCG | Pairwise | Firing | Plateau |
|---|---:|---:|---:|---:|---:|---:|---:|
| old local | 54k | **3.659756** | 0.148438 | **0.891168** | **0.627478** | 0.040145 | 1.018187 |
| repaired local, best top-1 | 10k | 3.748589 | **0.164063** | 0.870819 | 0.580845 | 0.007540 | 0.006210 |
| repaired local, final | 100k | 3.767749 | 0.085938 | 0.862767 | 0.574327 | 0.009719 | 0.007385 |

The old run has better ranking metrics but a pathological plateau magnitude;
the repaired run has biologically/arithmetically stable state but worse final
quality.  Neither is a satisfactory production result.

### Retained budget controls

The retained three-seed 100k budget comparison used width 40, batch 1 and
exact reference BPTT, so it is context rather than a direct winner/loser table:

| Arm | Loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|---:|
| Point-SNN | 3.067797 | 0.239583 | **0.949494** | **0.734008** |
| direct Reduced Hay | **3.064013** | 0.234375 | 0.948069 | 0.730139 |
| diagonal GRU | 3.174158 | **0.244792** | 0.928019 | 0.700770 |

The Digital-Twin-derived frozen 11-state control remains unavailable because
no production-qualified frozen artifact exists.  It must not be reported as a
measured comparison.

## Decision

1. The requested repaired 100k scratch run completed and is retained.
2. Exact BPTT and the analytic head VJP are correct under focused derivative
   tests; the remaining failure is in approximation quality and long-horizon
   optimization, not a demonstrated reverse-mode implementation bug.
3. The repaired local learner can memorize one state and learns useful ranking
   signal by 10k, but it does not maintain that gain through 100k.
4. Stable low plateau/firing rates show that the earlier explosive dynamics are
   fixed.  They do not establish useful high-dimensional-neuron superiority.
5. Full-width Point/GRU retraining was not repeated in this run.  Existing
   width-40 controls are deliberately kept separate, and frozen 11-state stays
   unmeasured.
6. Another identical 100k extension is not justified.  The next critical
   experiment should keep a decayed/frozen best head while improving recurrent
   exact/local alignment, then require a 10k-to-50k held gain before paying for
   another 100k run.

## Artifacts

```text
D:\tetris-paper-plus\runs\reduced_hay_v2_arena\bp_repair_100k_scratch_20260801_021231
```

Artifact hashes:

```text
results.json             c6e3ac90f6532e41cfdc634c308f782c56b8c1d4528f38b49bbdfc9d01b24f23
training_trace.tsv       c847c3557e25331f56577cb7ef9cb893755129fccd4c07d976c8253cb7a4aec2
checkpoint_manifest.tsv  3bac5fb4ed720085660b8ab9771ff02c0fa8bc84e4d59e1dfa3114ca6daf5a60
```

No gameplay-validation or sealed seed was opened.
