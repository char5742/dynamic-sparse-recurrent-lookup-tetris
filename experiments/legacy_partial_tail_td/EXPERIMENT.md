# P1: one-shot legacy partial-tail anchored TD

This directory preregisters and implements the **only** experiment authorized
by `reports/clean_post_f_strategy_review.md` (SHA-256
`a079330917571824fdbb0dd92d37db92dc1df9701012206bb27bc672d24ca906`).
It is a new P1 hypothesis and namespace, not an F repair.  Nothing here edits,
invokes, resets, or reuses `experiments/legacy_full_feasibility`, its global
marker, or `1313/`.

## Immutable hypothesis and model subset

The initializer is `1313/mainmodel copy 3.jld2`, SHA-256
`7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`.
The dataset is `teacher_dev_5742_5749_2000.jld2`, SHA-256
`e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d`.

Only these parameter paths enter AD and AdamW:

- `board_net.resblocks.layer_29` (1,214,272 parameters)
- `board_net.resblocks.layer_31` (1,214,272 parameters)
- `board_net.conv2` (2,305 parameters)
- `board_net.norm2` (2 parameters)
- `score_net` (518,657 parameters)

The exact total is 2,949,508 array elements in 34 array leaves.  AdamW must
have exactly 5,899,016 moment-array elements.  The other 17,837,946 parameters
in 250 array leaves and all 17,666 running-state elements in 69 array leaves
remain SHA-256 identical.  Parameterless activation layers 30 and 32 remain in
the tail forward.  The empty pool subtree in each trainable residual block may
have a `nothing` or empty-tuple gradient; every actual parameter array must
have a shape-matched finite array gradient on every update.

For each historical candidate chunk, the board trunk through layer 28 and the
entire queue mixer are evaluated outside the AD closure.  The closure receives
only the fixed prefix values, the five trainable subtrees, and fixed test-mode
state.  Forward evaluation always uses consecutive full chunks of 16 followed
by an actual-length tail.  There is no padding or whole-list normalization.

At step 0, the full candidate list for the first deterministic frozen row
(source row 1055, episode 5 step 55, 52 candidates, selected action 11) is the
preregistered witness.  Its chunks are exactly 16+16+16+4; split-tail output must match the original
full Lux graph within `1e-6` and stored old-Q within `1e-2`.  Before update 1, central finite
differences for `score_net.layer_3.bias[1]`,
`score_net.layer_1.weight[1]`, and
`board_net.resblocks.layer_31.layer_1.weight[1]` must each satisfy
`abs_error <= 1e-3 + 0.02*max(abs(fd),abs(ad))`.  `Lux.testmode` is applied
once before AD, never inside it.

## Data freeze, target, loss, optimizer

Eligibility is determined with HDF5 hyperslabs over rows 1--1500 only.  A row
is eligible only when `t` through `t+3` have the same episode and consecutive
steps and no terminal occurs through `t+2`.  This must yield exactly 1,482
rows: 247 per episode for episodes 1--6 / seeds 5742--5747.  The eligibility
list is shuffled once with Julia `Xoshiro(0x1313_2026)`; the first 300 rows,
without replacement and preserving shuffled order, are written to
`row_freeze.json` before any target extraction or learning.  Every row is then
consumed exactly once and no environment rollout is generated.
The comma-separated ordered-row digest is fixed as
`7f8a24abc5000ad1cc13ee4c4d7b5227caf57923686fd17aea83ef664550efae`.

The frozen target uses stored score-delta `/600` rewards, `n=3`, and
`gamma=0.997`.  A terminal truncates rewards and removes bootstrap; otherwise
the stored old-policy selected Q at `t+3` is the bootstrap.  No updated model
value is used to regenerate a target.

For the exact 16-wide/actual-tail chunk containing the selected action, one
row loss is:

`Huber(q[selected], y3) + mean(Huber(q[a], old_q[a]) for a in selected_chunk)`

Both Huber deltas and the anchor weight are 1.  No padding, ListNet, DAgger,
auxiliary loss, other chunk, or label is present.  Optimisation is Julia
1.12.6, Lux 1.31.4, Zygote 0.7.11, coupled AdamW with learning rate `1e-5`,
betas `(0.9,0.999)`, weight decay `1e-4`, exactly 300 updates, and no sweep,
rollback, rescue, or checkpoint selection.

## Gates and evaluation order

After the fixed update 300 is freshly merged, it is written to new JLD2 and
NPZ files; no old checkpoint or weight file is overwritten.  Fresh fixed-16
and dynamic-tail OpenVINO IRs are saved and compiled.  Lux versus fresh
OpenVINO must be within CPU `1e-4` and NPU+CPU-tail `1e-2`.  All constructors
receive the new NPZ explicitly; the historical default NPZ constructor is not
used for the candidate.

Only after training and fresh OpenVINO equivalence are complete may HDF5 read
rows 1501--2000, episodes 7--8 / seeds 5748--5749.  The offline gate explicitly
binds the fresh NPZ to the historical OpenVINO chunked path (rather than paying
for 500 redundant full Lux CPU forwards).  The fixed update-300 candidate must produce finite Q values
and at least 0.95 old-policy top-1 agreement.  This split never selects or
rolls back a checkpoint.

Only after the training, export/equivalence, and offline gates pass are
development seeds 5756 and then 5757 permitted.  Each candidate/canonical-old
pair uses NEXT=5, HOLD, `stable_node_key` order, 100 pieces, zero lookahead,
one logical full-candidate score call per decision, fixed chunks of 16 and an
actual-length CPU tail.  Candidate/baseline candidate evaluations, logical and
physical calls, generation time, inference time, and wall time are recorded.
Game-over stops immediately.  A nonpositive seed-5756 difference stops before
5757.  P1-development-pass requires two completed pairs, 2/2 strictly positive
differences, paired mean at least +500, and positive paired median.

Seeds 5750--5755 are not reused.  Validation seeds 8001--8008 and sealed test
seeds 91001--91032 must not be loaded or run.  A development pass is not a
statistical claim and does not authorize sealed testing; candidate freeze and
sealed evaluation require a separate review.

## One-shot and external stops

`invoke_once.ps1` creates
`D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json` atomically
after an inspectable freeze and explicit start gate.  The marker is independent
of output directory.  A stopped or failed run cannot be retried under a new
path.  Outputs must be in one fresh directory below `D:\tetris-paper-plus`.

The PowerShell parent monitors the complete descendant process tree.  It stops
the tree at a 35-minute wall or 8-GiB aggregate working set.  The first update,
including optimizer setup and finite differences, has a 180-second external
stop.  Every later update has a 15-second internal and external stop; the first
six of those later updates must have a median at most 4.5 seconds.  After those
six, the measured projection `one-shot elapsed through update 7 +
293*warm_median + 100 + 120 + 400*0.411` must not exceed 2,100
seconds.  Any contract, hash, gradient, finite, frozen-state, time, memory,
export, offline, accounting, or game gate failure stops the one shot without a
changed row, seed, update count, hyperparameter, or backend.

The audited implementation base is commit
`c9ab1a94342752dc135725beabf4a6b10d73f92d`.  A runnable hardening revision
must be its single direct child, may change only this experiment directory,
and must be passed back to the wrapper as the full
`-AuthorizedHardeningCommit`.  The explicit start-gate text includes that full
commit.  Before the marker and again at the gate, the supplied source
fingerprint is recomputed against the live repository: repository root,
Manifest hash, exact source file path set, every byte count and SHA-256, file
count, aggregate source SHA-256, clean tree, parent, authorized HEAD, and the
hardening diff scope must all match.  The separate harness aggregate covers
all files here, including PowerShell files outside the source-fingerprint
suffix set.

Finalization is deliberately two-stage.  The monitored finalizer writes only
`assessment.json` after strictly reconciling every artifact, role, hash link,
provenance field, seed/config order, and nonnegative accounting value.  The
wrapper then records that finalizer phase and computes the completed monitor
and result before terminal publication.  It durably flushes and atomically
renames `monitor.json` with `complete=true`, durably publishes the
non-authoritative `wrapper_result.json`, and publishes `final_result.json`
atomically as the final fallible filesystem operation.  A pass requires the
exact ordered phase ledger, no skipped phase or failure, a passing assessment,
and exact nonnegative top-level and per-phase working-set/private-byte
accounting.  An incomplete/pre-finalizer monitor or any pre-final publication
failure cannot leave a passing final result.

`-ValidateOnly` and the files named `test_*` perform syntax, contract, and
synthetic checks only.  They do not read the real checkpoint/dataset, learn a
model, compile OpenVINO, or load any evaluation seed.
