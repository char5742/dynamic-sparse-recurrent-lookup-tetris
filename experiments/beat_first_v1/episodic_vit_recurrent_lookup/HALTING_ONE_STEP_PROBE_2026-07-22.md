# Candidate-local one-step halting probe — 2026-07-22

## Decision

The state-wide REINFORCE halting update is replaced, when probe mode is
enabled, by physically sparse candidate-local supervision.  The architecture,
20,577,789 parameters, input and teacher contracts, normal task loss, hard
routing, active-only backward, sparse optimizer, checkpoint format, candidate
RNG order, and candidate-independent scoring are unchanged.

For at most `P` sampled-stop candidates per state:

1. execute exactly one additional recurrent step;
2. replace only that candidate's scalar Q with its `t+1` Q;
3. recompute the same ListNet plus margin ranking value;
4. define `delta = L_stop - L_continue`;
5. train the final halt logit toward continue when `delta > c`, otherwise stop.

Unprobed candidates receive no halting gradient.  The original score matrix is
restored before task backward, so the task loss and task VJP do not depend on
the probe.  Probe forward does not update route-usage counters and does not
create a probe trajectory for backward.

Runtime controls:

```text
EVRL_HALT_PROBES_PER_STATE   number of stopped candidates probed per state
EVRL_HALT_PROBE_WEIGHT       BCE weight on the candidate-local halt target
EVRL_COMPUTE_PRICE           continue threshold c in probe mode
EVRL_ENABLE_HALT_PROBE_TRANSITION=1
                             explicit legacy-checkpoint transition gate
```

The default probe count is zero, preserving old checkpoint behavior.  Missing
probe fields in a version-1 checkpoint normalize to count zero and weight one;
the checkpoint serialization format is unchanged.

## One-step primitive witness

A forced-depth-3 trajectory followed by `probe_one_step!` was compared with a
normal forced-depth-4 forward on the same model and input.  Maximum output
difference was `2.3841858e-7`, below the `2e-6` acceptance tolerance.

## Real-teacher serial/barrierless correctness smoke

The production 20-worker smoke used the current 20,577,789-parameter R2
update-10,000 checkpoint, four training states, all valid candidates (counts
34, 53, 51, and 68), and two probes per state.  Validation rows and sealed game
seeds were not constructed or touched.

Result: **pass**.

```text
output max abs                         0
loss max abs                           0
raw task VJP max abs                   0
probe target exact                     true
probe delta exact                      true
halt RNG state / next values exact     true / true
sampler state / next rows exact        true / true
parameter gradient max abs             4.0382147e-6
parameter gradient relative L2         1.7158563e-6
post-optimizer parameter max abs        4.0419400e-7
optimizer clocks                       exact at update 10,001
```

Checkpoint witness:

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
  evrl_recurrence_cp0_haltlr1e5_warmup5k_u100000_20260722_r2\checkpoints\
  checkpoint_000010000.jls
sha256 3bd4140707a10cd63781bd39c65d21255ae8dbaa0ea022c78ab501b3f014041b
```

## 100-update throughput preflight

The same checkpoint was run for 10 warmup plus 100 measured real-teacher
updates with two probes per state, barrierless scheduling, no pinning, chunk 8,
20 Julia workers, and BLAS threads 1.  Benchmark updates were not checkpointed.

```text
measured updates             100
updates/s                    20.6586
states/s                     82.6343
candidate CPU                60.5010%
overall CPU                  60.1197%
last update probes           8
continue / stop targets      2 / 6
last mean delta             -6.0588e-5
minimum speed criterion      15 updates/s
result                       pass
```

This preflight is slower than the no-probe R1 aggregate because it executes up
to eight exact additional recurrent steps per update and recomputes four small
state-local ranking losses.  It remains above the explicitly accepted 15
updates/s floor.  It is not a quality result: benchmark-only mode performed no
new held evaluation and wrote no checkpoint.

## Trial status

The scalar halt-LR Trial R2 was stopped at its already-complete update-10,000
boundary when the candidate-local credit-assignment correction was specified.
It is retained only as the immutable smoke parent above; it is not a completed
100k tuning result.  The next full trial enables probes from the beginning of
the dynamic training schedule rather than grafting them onto the finished R1
policy.

## Trial P1 — completed 100,000-update probe training

P1 trained the probe-aware policy from scratch.  It kept the full architecture,
20,577,789 parameters, real-teacher order, optimizer, task loss, hard routing,
and executor fixed.  The only halting-credit change from Recurrence R1 was the
candidate-local probe target described above.

```text
probe candidates/state       2
probe BCE weight             1
continue threshold c         0
halt learning rate           5e-5
random-depth warmup          updates 1--5,000, depth 2--6
learned recurrent range      2--12
teacher states/update        4
total updates/states         100,000 / 400,000
```

The orchestration command reached its two-hour host timeout after the complete
75,000-update checkpoint had been written.  No model error occurred.  Training
resumed from that checkpoint with its exact optimizer, sampler, halt RNG, and
route-usage state, and completed at update 100,000.  The initial evaluation in
the resume run reproduced the update-75,000 metrics exactly.

### Held-panel and depth curve

| Update | Loss | Top-1 | NDCG | Pairwise | Margin | Train depth | Held depth | Held range | Probe continue/stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 | 2.000 | 2.000 | 2--2 | -- |
| 5,000 | 2.840448 | 0.53125 | 0.979206 | 0.839419 | 0.105218 | 2.000 | 2.000 | 2--2 | warmup |
| 10,000 | 2.816590 | 0.52344 | 0.978723 | 0.843999 | 0.103889 | 3.607 | 3.538 | 3--12 | 4 / 4 |
| 15,000 | 2.774000 | 0.50000 | 0.980272 | 0.852634 | 0.097204 | 3.048 | 3.006 | 2--12 | 3 / 5 |
| 20,000 | 2.762955 | 0.59375 | 0.983187 | 0.863068 | 0.089299 | 5.121 | 4.886 | 3--7 | 8 / 0 |
| 25,000 | 2.723159 | 0.53125 | 0.981245 | 0.864550 | 0.113911 | 2.027 | 2.055 | 2--12 | 4 / 4 |
| 30,000 | 2.687534 | 0.58594 | 0.984653 | 0.870740 | 0.113859 | 2.025 | 2.060 | 2--12 | 6 / 2 |
| 35,000 | 2.678400 | 0.62500 | 0.984968 | 0.873602 | 0.126336 | 5.014 | 4.940 | 4--12 | 4 / 4 |
| 40,000 | 2.663871 | 0.61719 | 0.985075 | 0.876878 | 0.140843 | 2.068 | 2.105 | 2--12 | 4 / 4 |
| 45,000 | 2.654798 | 0.61719 | 0.985994 | 0.883263 | 0.119168 | 2.371 | 2.490 | 2--12 | 5 / 3 |
| 50,000 | 2.643391 | 0.61719 | 0.985995 | 0.884732 | 0.142789 | 5.307 | 5.347 | 3--12 | 5 / 3 |
| 55,000 | 2.636955 | 0.63281 | 0.988076 | 0.887374 | 0.125715 | 2.059 | 2.096 | 2--12 | 4 / 4 |
| 60,000 | 2.647861 | 0.67188 | 0.988406 | 0.886657 | 0.120213 | 2.076 | 2.122 | 2--12 | 4 / 4 |
| 65,000 | 2.622504 | 0.67188 | 0.989416 | 0.894942 | 0.140595 | 3.095 | 3.076 | 2--4 | 3 / 5 |
| 70,000 | 2.626701 | 0.65625 | 0.988544 | 0.892699 | 0.145571 | 2.203 | 2.270 | 2--12 | 5 / 3 |
| 75,000 | 2.614308 | 0.67188 | 0.989111 | 0.897401 | 0.132339 | 3.460 | 3.399 | 2--6 | resume boundary |
| 80,000 | 2.604949 | 0.69531 | 0.990553 | 0.899782 | 0.123521 | 2.044 | 2.042 | 2--12 | 4 / 4 |
| 85,000 | 2.613532 | 0.70312 | 0.988633 | 0.896584 | 0.133625 | 2.433 | 2.529 | 2--12 | 3 / 5 |
| 90,000 | 2.592303 | **0.74219** | 0.991142 | 0.902739 | 0.119525 | 3.021 | 3.021 | 2--12 | 4 / 4 |
| 95,000 | **2.587874** | 0.73438 | **0.991345** | **0.904401** | 0.141909 | 2.205 | 2.194 | 2--12 | 3 / 5 |
| 100,000 | 2.605494 | 0.71094 | 0.990369 | 0.902199 | **0.151411** | 2.030 | 2.011 | 2--12 | 5 / 3 |

The continue/stop column is telemetry from the single update at each reporting
boundary, not a population count over the preceding 5,000 updates.  Even this
bounded witness demonstrates candidate-specific targets in both directions;
the policy is no longer forced to apply one state-wide advantage sign to every
candidate.

### Decision

P1 is a positive credit-assignment result and a partial dynamic-depth result.
The balanced checkpoint is update 95,000: it has the lowest held composite
loss and highest held NDCG of P1 while retaining top-1 `0.73438`, pairwise
`0.90440`, margin `0.14191`, and mean depth `2.19`.  Update 90,000 is the P1
top-1 winner (`0.74219`, mean depth `3.02`), while update 100,000 is the margin
winner (`0.15141`).

Relative to the rejected state-wide R1 final checkpoint, the P1 final improves
loss by `0.004945`, top-1 by `0.007812`, NDCG by `0.000565`, pairwise accuracy
by `0.002944`, and margin by `0.008825`.  More importantly, it does not end in
R1's near-total depth-12 saturation (`11.91` mean): P1 ends at `2.01` and shows
intermediate held means near 3--5 at multiple checkpoints.

This does **not** yet prove ideal adaptive computation.  P1 still oscillates,
and its selected/final checkpoints remain biased toward the minimum depth.
Compared with the fixed-depth Trial-1 final control, P1 update 95,000 has
`+0.006163` loss, `+0.015625` top-1, `-0.000456` NDCG, `-0.001826` pairwise,
and `-0.004496` margin.  It therefore supplies better candidate-local
halting credit, not a clean all-metric quality win over fixed depth.

Against the recorded PreAct panel result, P1 update 95,000 remains lower by
`0.054688` top-1, `0.001945` NDCG, and `0.018959` pairwise accuracy, while its
margin is higher by `0.018589` and composite loss is higher by `0.024094`.
The held panel has guided development and is not a sealed generalization set.

### Runtime and artifacts

```text
aggregate training time       5,015.947635 s
aggregate updates/s           19.936412
aggregate average CPU         59.8515%
aggregate candidate CPU       60.9202%
resume 75k--100k time         1,298.462386 s
resume updates/s              19.253542

balanced checkpoint:
  ...\evrl_haltprobe_p2_c0_w1_warmup5k_u100000_20260722_p1_resume2\
  checkpoints\checkpoint_000095000.jls
  sha256 33de0fce4f1e7b6b734069bebe00a3eb822635a4e1d3131619f84f8c206cef02

final checkpoint:
  ...\evrl_haltprobe_p2_c0_w1_warmup5k_u100000_20260722_p1_resume2\
  checkpoints\checkpoint_000100000.jls
  bytes 253,691,534
  sha256 224b8f0adb7c41b3c5b9830a1d06b2ffe8ab00266644db5c29a1c29c07e26458

first-segment metrics sha256:
  9ba8da7da37fa637dc166b76410612faa7b4d5f2ee55e4a06a5f2059919bcb9c
resume metrics sha256:
  1ef8b116b594d55272388f12ef3573f19cbdefba078e1b927a1ef1eb8ed55321
resume summary sha256:
  55fede26a83af61d4d45fdc994f58f70e4c1de0b34da62a7ab546e428d6937ab
```

Binary checkpoints and teacher data remain local and are not committed.
