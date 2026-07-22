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
