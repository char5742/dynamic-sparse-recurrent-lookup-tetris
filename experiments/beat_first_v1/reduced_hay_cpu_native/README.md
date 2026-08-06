# CPU-native high-dimensional candidate-delta motif graph

This directory has one canonical model: the **HD Candidate-Delta Motif Graph
(HD-CDMG)**. Older point-SNN, recurrent Reduced-Hay, Float32 numeric-core,
local-credit, route, and dendritic-forest experiments remain in Git or as
historical controls, but the canonical root does not include or export them.

The research claim is deliberately narrower than “a biological Hay model” or
"TwinProp reproduced." HD-CDMG uses compact, trainable Reduced-Hay cells to test
a CPU-native hypothesis: irregular candidate-local changes can update a small
copy-on-write relation closure instead of evaluating a dense candidate network.
Whether this beats PreAct or DSRLN is an empirical gate, not an architectural
assertion.

## Canonical model

```text
state-common binary rails
  -> collision-free 16D spatial program packets
     (240 positions x before/after plane = 480 sources)
  -> 48 structured Reduced Hay relation cells
     (24 row + 10 column + 14 cross/action)
  -> fixed semantic relation-to-motif graph
  -> 48 structured Reduced Hay motif cells
     (24 vertical + 10 well + 8 local-geometry + 6 action/global)
  -> source-bound, lower-bounded structured motif readout
  -> 22 typed Reduced Hay output cells
  -> continuous 22D ranking output

candidate-local program signed delta + auxiliary context + raw placement
  -> affected relation-cell copy-on-write overlays
  -> candidate relation-packet delta
  -> exact affected motif-cell copy-on-write closure
  -> candidate motif-packet delta
  -> 22 output cells
```

Each relation, motif, and output cell keeps 48 continuous and event coordinates:

- eight basal and one active apical compartment;
- voltage, AMPA, NMDA, GABA, and plateau state per compartment;
- soma voltage, adaptation, and hard spike;
- one mandatory analog transition at each graph level.

Program lookup emits a 16-Float32 spatial payload. Cell-to-cell communication
uses a separate 47-Float32 continuous transition packet: all nine
compartments expose voltage, AMPA, NMDA, GABA and plateau (`9 x 5 = 45`), then
the exact pre-reset soma margin and adaptation occupy lanes 46 and 47. Each
field uses a fixed physical-unit `softsign`; hard spike remains on the separate
event plane.

The fixed source-to-relation anatomy exposes every one of the 16 packet lanes
exactly once per spatial source. Each lane owns an opponent pair, so the fanout
is 32 physical contacts/source. The 16 lanes cycle over the four fixed topology
destinations (row, column, tile-local, small-world stripe), while their two
octaves cover all eight basal compartments. Raw occupied placement bits use
four hard-positive contacts, one to each topology family; this preserves the
pre-clear action identity without a direct placement-to-output path.

Every relation cell reaches four geometry-defined motif destinations. A
five-state physical compartment bundle is always co-located at one motif;
signed voltage uses an AMPA/GABA opponent pair while the physically
non-negative conductance, plateau and adaptation lanes need one contact each.
This yields 57 unique typed contacts/source and drives all nine destination
compartments, including apical. Candidate execution forms the exact motif
closure of the affected relations; hard spike/plateau events are diagnostics
and future sparse-wave controls, never gates on this mandatory analog DAG.

Output ownership is deliberately strict:

- every content source—program packets, candidate auxiliary context, and raw
  placement—must first change relation cells and then motif cells;
- no program-packet-to-output or trainable all-to-all relation-to-output graph
  exists;
- `common_output` remains only as state-common conditioning;
- a fixed structured readout carries the motif packet, plus a fixed-scale
  `1/8` relation residual, into typed output compartments.

The readout schedule is fixed by output role: Q reads all 47 fields, death
reads 24 stratified fields, each quantile reads eight rotated fields whose
union covers all 47, and each geometry output reads 16 stratified fields.
Compartment identity is phase-rotated by semantic family, apical state stays
apical, and AMPA/NMDA/GABA/plateau retain physical receptor identity. Each of
the 48 semantic source IDs owns an independent bounded-positive gain for each
output. A family-wide average is forbidden because it would make rows or
columns permutation invariant. There is no candidate `leaf_output`, trainable
`relation_output`, auxiliary-to-output, placement-to-output, dense observation
head, or runtime router.

Most stored capacity remains the 208,448-row semantic program bank with a
16-value payload per row. Runtime access is sparse; stored parameter count is
not the per-candidate compute count.

## Canonical call chains

Forward:

```text
CandidateDeltaRelationGraph.prepare_state!
  -> materialize the common 16 x 480 program-packet grid once
  -> leaf_relation + common_relation
  -> advance the 48 base relation cells once
  -> relation_motif
  -> advance the 48 base motif cells once
  -> structured motif readout + 1/8 relation residual + common_output
  -> prepare the immutable 22-cell base output anchor

CandidateDeltaRelationGraph.forward_candidate!
  -> rematerialize changed after-plane packets only
  -> program signed delta + auxiliary context + raw placement
  -> affected relation-cell copy-on-write transitions
  -> relation-packet delta through relation_motif
  -> affected motif-cell copy-on-write transitions
  -> structured motif delta + 1/8 relation residual
  -> one transition of the 22 candidate output cells
  -> continuous 22D supervised output
```

Training:

```text
RelationGraphTraining.train_update!
  -> fixed-width ListNet/auxiliary supervised objective
  -> exact conditional candidate output reverse
  -> structured motif/readout reverse
  -> grouped copy-on-write motif reverse
  -> relation_motif reverse
  -> grouped copy-on-write relation reverse
  -> program/context/placement relation-input reverse
  -> shared base reverse once per state
  -> RelationGraphOptimizer.apply_adamw!
```

“Exact conditional” means the continuous state path uses the analytic reverse
and hard events use the declared conditional treatment. It is not the finite
difference derivative of the discontinuous hard-forward function.

## CPU execution contract

- fixed `8 x 80` ranking batch;
- fixed SoA state/worker storage and cached typed-afferent kernels;
- state-common program, relation, motif, and output-anchor work computed once;
- candidate-local copy-on-write relation and exact motif closures;
- persistent barrierless workers and deterministic reduction order;
- no central workspace route, query/key, top-k policy, or replay route;
- no allocation in the measured hot kernels.

This is a CPU-oriented opportunity, not a claim that GPUs cannot execute the
model. Promotion requires measured wall-clock and ranking quality against the
same panels and data budget.

## Entrypoints

- `train_scratch.jl` is the public full-data command and selects no alternate
  model path.
- `overfit_relation_graph.jl` is the fixed-state HD-CDMG capacity gate.
- `test_candidate_delta_relation_graph.jl` verifies forward/reverse causality,
  sparse/full-overlay equality, grouped reverse, finite differences, and the
  zero-allocation hot path.

Launch training from the repository project with an empty interactive thread
pool, for example:

```powershell
julia --project=. --threads=20,0 experiments/beat_first_v1/reduced_hay_cpu_native/train_scratch.jl --updates 1000
```

The checkpoint schema rejects incompatible model/source/data identities on
resume.

## Promotion boundary

Do not start a 100k run merely because the module loads. The direct gates are:

1. normal and bounds-checked unit/conditional-gradient tests;
2. sparse/full-overlay and grouped-reverse equality plus zero-allocation checks;
3. serial/barrierless equality;
4. stable fixed-state capacity tests;
5. full-data 1k quality while retaining the agreed CPU throughput floor;
6. only then 5k/10k trend checks and a monitored 100k comparison with PreAct
   and DSRLN on identical evaluation panels.

Design rationale, input-alias evidence, and literature boundaries are recorded
in `CPU_NATIVE_DENDRITIC_REDESIGN.md`.
