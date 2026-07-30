# Reduced Hay recurrent-path repair — 2026-07-30

## Outcome

The original `:legacy_v1` Reduced Hay arm did not test a functioning
high-dimensional recurrent graph.  It learned primarily as a dense
input-conditioned workspace router followed by a short compartmental
readout.  The sparse basal graph could be removed at 100k with almost no mean
effect.

The retained legacy arm is unchanged.  A separate
`:causal_recurrent_v2` arm repairs the causal trajectory and prevents the two
observed structural bypasses.  At 10k, removing recurrence from v2 increases
held composite loss by `+0.011584` and reduces NDCG by `-0.011866` and
pairwise accuracy by `-0.010937`.  In the legacy arm at the same checkpoint,
removing recurrence instead changes loss by `-0.002415`.

At 100k, recurrent-off costs `+0.041193` loss and reduces top-1 by
`-0.015625`, NDCG by `-0.006874`, and pairwise by `-0.006121`.  The same-seed
v2 checkpoint also exceeds Point-SNN on all four held metrics.  This
establishes that the repaired recurrent path is used and that the repair can
separate from Point at equal update count.  It does not establish a
multi-seed or equal-wall-clock victory.

## Failure analysis

The legacy result was not explained by a generic lack of model size.

1. **No branch-local input conjunction.**  Each basal branch received only
   one excitatory rail and one inhibitory rail.  The branch therefore had no
   collection of located inputs on which to perform the intended NMDA-like
   coincidence computation.
2. **The first route was input independent.**  Every cell state was zero when
   the first workspace route was selected.
3. **The graph had too little causal depth.**  With three cycles, the zero
   first spike scan and one-cycle delivery semantics left only two useful
   propagation opportunities.
4. **Sensory evidence bypassed recurrence.**  The same sensory drive was
   reinjected on every cycle, so later cell states did not require graph
   delivery.
5. **Routing had a much larger direct shortcut.**  The trainable
   `query_weight` contained `12 x 1,298 = 15,576` dense rail weights, while
   the recurrent graph had only 128 candidate edge slots.
6. **Independent gates collapsed.**  Seed 1 retained 64/128 gates at
   initialization, 56 at 10k, and 44 at 100k.  The other two 100k seeds ended
   at 43 and 45 gates.

The 100k seed-1 gradient audit also showed the resulting credit imbalance:

| Parameter group | Gradient norm |
|---|---:|
| synapse weight | 0.0681 |
| gate | 0.00447 |
| delay | 0.00333 |
| workspace feedback | 0.458 |
| workspace key | 2.467 |
| dense query | 7.972 |

The query gradient was about 117 times the synapse-weight gradient.  The
learned query RMS grew by about 29 times from initialization, whereas the
sparse graph changed a held raw output by only about 0.45% in relative norm.

## Causal recurrent v2

The repair is intentionally a new arm rather than a silent change to the
retained research control.

- Each of 16 cells has four branches and 120 fixed-location trainable
  excitatory contacts plus 120 inhibitory contacts per branch.
- The 15,360 located contacts cover all 1,298 binary rails.
- Sensory conductance is delivered only on cycle 1.
- The compartment state advances before the first workspace route.
- Routing is derived from the current exported cell state; the
  15,576-parameter direct input `query_weight` is absent.
- Selected soma spikes are delivered through the graph on following cycles.
- Workspace feedback also affects the following cycle.
- Six cycles expose multiple recurrent hops.
- Each source keeps exactly four of eight learned recurrent contacts.
  A fixed-mass top-k straight-through relaxation lets contacts exchange
  responsibility without changing total fanout.

After the mechanism gate passed, `build_reduced_hay_model()` and
`train_reduced_hay_direct.jl` were promoted to the causal v2 defaults.
`:tiny` and `:reduced_hay_scaled_v1` remain explicit legacy controls, while
the scaled canonical preset is `:reduced_hay_scaled_v2`.  Its initialization
check reports 244,823 parameters, 49,152 located sensory contacts, and 18,432
enabled recurrent contacts (24 per source).

The v2 reference has 18,663 trainable parameters versus 18,863 in legacy v1
and 21,567 in the Point control.  It retains direct BPTT for continuous state,
the existing spike and route estimators, signed E/I delivery, learned delays,
voltage-dependent NMDA unblock, plateau, apical modulation, soma integration,
and adaptation.

## Verification

The focused tests check:

- all 1,298 rails are represented by located contacts;
- the dense input query is absent;
- every source has exactly four active recurrent contacts;
- recurrent delivery is nonzero;
- recurrent-off changes the output;
- Tetris teacher gradients reach sensory contacts, state routing, recurrent
  weights, gates, and delays;
- the retained legacy initialization keeps its exact prior random draw order.

The existing Reduced Hay kernel/event tests remain green.

## 10k scratch result

One exact seed was trained on the same width-40 schedule and fixed 64-state
panel.  The v2 run was continued exactly from its retained 1k checkpoint.

| Arm | Parameters | Composite loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|---:|
| Point-SNN | 21,567 | 3.762591 | 0.140625 | **0.924101** | 0.654632 |
| legacy Reduced Hay | 18,863 | **3.207165** | **0.187500** | **0.926419** | 0.658816 |
| causal recurrent v2 | 18,663 | 3.380272 | 0.078125 | 0.896973 | **0.662576** |

The overall ranking result is mixed: v2 has lower loss and higher pairwise
accuracy than Point, but worse top-1 and NDCG.  It has not demonstrated a
quality victory.

V2 mechanism activity at 10k:

| Signal | Value |
|---|---:|
| recurrent absolute input / branch-cycle | 0.013574 |
| recurrent / sensory absolute input | 0.208468 |
| branch-cycles with recurrent input | 0.141043 |
| recurrent gate density | 0.500000 |
| workspace-active spike rate | 0.124788 |
| plateau-active fraction | 0.300846 |
| branch effective rank | 3.483308 |

Exact-zero ablations use `delta = ablated - full`:

| V2 ablation | Delta loss | Delta top-1 | Delta NDCG | Delta pairwise |
|---|---:|---:|---:|---:|
| plateau off | +0.098793 | 0.000000 | -0.012347 | -0.011276 |
| apical off | +0.012389 | +0.078125 | -0.006119 | +0.001575 |
| recurrent input off | +0.011584 | +0.015625 | -0.011866 | -0.010937 |

Plateau and recurrent input now make a positive contribution to the optimized
loss and to two continuous ranking metrics.  The top-1 panel contains only 64
states and moves by one state per `0.015625`, so its sign is recorded but is
not used alone to accept or reject the mechanism.

## CPU boundary

Reference continuation throughput was:

| Arm | Updates/s |
|---|---:|
| Point-SNN | 151.458 |
| legacy Reduced Hay | 149.349 |
| causal recurrent v2 | 71.291 |

V2 runs six cycles rather than three and still uses allocation-heavy Zygote
reference BPTT.  The present result repairs semantics, not CPU efficiency.
Before v2 can replace the production arm, the same causal order must be moved
into the existing SoA event kernel and barrierless executor, and compared at
equal wall-clock.

## 100k exact continuation

The v2 10k checkpoint was continued exactly to 100k with parameter and AdamW
state restoration.  The controls below are the already-retained checkpoints
with the same model, schedule and held-panel seed.

| Arm | Composite loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| Point-SNN | 3.064102 | 0.250000 | 0.939038 | 0.677596 |
| legacy Reduced Hay | **2.995290** | 0.250000 | 0.946011 | 0.718495 |
| diagonal GRU | 3.106588 | 0.250000 | 0.929343 | 0.705286 |
| causal recurrent v2 | 3.016713 | **0.359375** | **0.948239** | **0.766848** |

V2-minus-Point is:

| Metric | Difference |
|---|---:|
| composite loss | -0.047389 |
| top-1 | +0.109375 |
| NDCG | +0.009201 |
| pairwise | +0.089252 |

V2 has slightly higher composite loss than legacy v1 (`+0.021424`) but better
top-1 (`+0.109375`), NDCG (`+0.002228`) and pairwise (`+0.048354`).  This is
the first retained same-seed checkpoint in this line where the repaired
high-dimensional arm separates from Point across all reported quality
metrics.

V2 activity at 100k:

| Signal | Value |
|---|---:|
| recurrent absolute input / branch-cycle | 0.017305 |
| recurrent / sensory absolute input | 0.272344 |
| branch-cycles with recurrent input | 0.157484 |
| recurrent gate density | 0.500000 |
| workspace-active spike rate | 0.178311 |
| soma spike rate | 0.445042 |
| plateau-active fraction | 0.367233 |
| plateau mean | 0.083577 |
| NMDA mean | 0.179219 |
| branch effective rank | 3.457779 |

Exact-zero 100k v2 ablations:

| Ablation | Delta loss | Delta top-1 | Delta NDCG | Delta pairwise |
|---|---:|---:|---:|---:|
| plateau off | +0.307227 | -0.140625 | -0.022736 | -0.056779 |
| apical off | +0.036818 | -0.078125 | -0.002460 | -0.006449 |
| recurrent input off | +0.041193 | -0.015625 | -0.006874 | -0.006121 |

All three retained Hay-derived mechanisms now have the expected causal sign
on loss, top-1, NDCG and pairwise.  In particular, the recurrent contribution
grew rather than disappearing between 10k and 100k.

The 10k-to-100k v2 reference continuation ran at `67.399 updates/s`, allocated
`31.938 MB/update`, and spent `193.636 s` in GC.  The retained legacy seed-1
continuation ran at `148.780 updates/s`.  This approximately `2.21x` reference
throughput deficit is a required target for the SoA/event integration.

## Claim boundary

The accepted claim is:

> The original tie with Point-SNN was confounded by a dense routing shortcut,
> repeated sensory bypass, gate collapse, and insufficient causal depth.  In
> the repaired arm, branch plateau and sparse recurrent input have measurable
> causal value under the Tetris teacher objective.

The result is not a reproduction of the full Hay cell or TwinProp, and one
100k seed does not establish general high-dimensional-neuron superiority.

## Artifacts

```text
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\recurrent_repair_u64_20260730_205012
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\recurrent_repair_u1000_20260730_205514
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\recurrent_repair_u10000_20260730_205902
D:\tetris-paper-plus\runs\reduced_hay_direct_validation\recurrent_repair_v2_u100000_20260730_211247
```

```text
results.json SHA-256:
6F8C1E9086DC04467388A995B69907FF11E27D8CDDFEB8582D939776459C4B67

checkpoint_reduced_v2_u100000.jld2 SHA-256:
FDB6EA1C36CC217EA2E2B8B6F4EEB5B7ACD29AE2B6A731B5B8C458D77FD3A88B
```
