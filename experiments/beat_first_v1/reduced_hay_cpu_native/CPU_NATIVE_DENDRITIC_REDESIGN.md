# CPU-native route-free multiscale dendritic event graph

Status: **canonical design contract, approved 2026-08-08**.

This document is the only forward-looking architecture contract for
`reduced_hay_cpu_native`. Historical relation-only, relation-to-motif,
Digital-Twin, frozen 11-state, and Point-SNN implementations remain available
through Git history and controls; they do not amend this contract.

The objective is not biological fidelity or TwinProp reproduction. The
objective is:

> use high-dimensional dendritic cells, candidate-delta copy-on-write, and
> hard-event refinement to exceed PreAct and DSRLN on Tetris quality per CPU
> wall-clock while preserving exact, falsifiable implementation contracts.

No central workspace router, query/key top-k, all-source dense head, giant
position lookup, or compatibility shim belongs to the canonical path.

## 1. Scope and claims

The canonical model is an original CPU-native Reduced-Hay architecture.

- Hay et al. motivates active basal/apical compartments, NMDA-like voltage
  dependence, soma integration, and multiple time scales.
- TwinProp motivates testing whether these mechanisms provide useful local
  nonlinear computation; it does not prescribe this topology, learning rule,
  Tetris input, or output geometry.
- e-prop supplies the eligibility-times-learning-signal factorization.
- DECOLLE motivates fixed local predictors; the resulting method is described
  as **DECOLLE-inspired**, not as exact DECOLLE.
- exact conditional reverse mode is an oracle, not the production learner.

References:

- Hay et al. 2011: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002107
- TwinProp preprint: https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full
- e-prop: https://www.nature.com/articles/s41467-020-17236-y
- DECOLLE: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full
- EventProp oracle reference: https://www.nature.com/articles/s41598-021-91786-z

Success is not claimed until the final equal-wall-clock comparison gate is
passed with confidence intervals. Intermediate overfit, throughput, or
training-only results are not victory claims.

## 2. Teacher-sufficient input contract

For one state `s` and candidate `a`, the information contract is

```text
X(s,a) = (B, P_a, REN, B2B, TSpin_a, HOLD, NEXT1..5)
```

- `B`: 24 x 10 pre-action board.
- `P_a`: up to four raw placement coordinates before line clearing.
- `REN`: exact integer representation, not a saturating eight-threshold code.
- `B2B`: explicit FALSE/TRUE token.
- `TSpin_a`: explicit FALSE/TRUE token.
- `HOLD`: role-bound NONE/I/O/T/S/Z/J/L token.
- `NEXT1..5`: five ordered role-bound tetromino tokens.

The input builder must not access teacher Q, rank, reward, selected action,
death target, geometry target, seed, split, or candidate ordinal.

### 2.1 Explicit negative symbols

The following are distinct types and must never share the numerical meaning
"zero":

```text
EMPTY / FALSE / NONE    observed semantic value
OUTSIDE                 board boundary
NO_EVENT                scheduler control state
ABSENT                   unused placement slot
```

Every observed categorical input emits an explicit non-silent typed token.

### 2.2 Exact line-clear map

Let rows be top-to-bottom `1:H`, columns `1:W`, with `H=24`, `W=10`.

```math
U_{r,c} = B_{r,c} \lor P_{r,c}
```

```math
F_r = \prod_{c=1}^{W} U_{r,c}, \qquad d = \sum_r F_r
```

The source-row to after-row map is

```math
\mu(r)=
\begin{cases}
0, & F_r=1 \\
d+\sum_{q\le r}(1-F_q), & F_r=0.
\end{cases}
```

With inverse `pi`, the after board is

```math
A_{y,c}=
\begin{cases}
0, & y\le d \\
U_{\pi(y),c}, & y>d.
\end{cases}
```

`B[y,c]`, `A[y,c]`, and `P[y,c]` are never bound into one same-coordinate
semantic object. On a clear, `A[y,c]` can originate from another source row.
The canonical neural graph therefore has separate before and after planes;
the raw placement remains a separate four-slot path.

Raw placement is mandatory. The current dataset audit found 4,303 candidate
pairs with identical post-clear board and T-spin but different teacher Q, all
explained by different raw placements; equal raw placement produced no such
conflict.

## 3. Canonical Reduced-Hay cell

Each cell owns 47 continuous state coordinates plus hard control events:

```math
z_i = [(V,g_A,g_N,g_G,p)_{1:9}, V_{soma}, a] \in \mathbb{R}^{47}.
```

The nine compartments are eight basal and one apical. Each compartment keeps
voltage, AMPA, NMDA, GABA, and plateau state. Soma and adaptation are separate.

The numerical kernel must retain:

- E/I-separated conductance dynamics;
- voltage-dependent NMDA unblock;
- ligand- and voltage-dependent plateau dynamics;
- ordered basal compartment identity;
- apical modulation;
- continuous pre-reset soma margin;
- soma reset and adaptation;
- hard soma spike and plateau onset/offset events.

The hard spike is not the scalar task answer. It controls optional computation.
The continuous pre-reset soma margin is the local task readout.

## 4. Twelve-dimensional axon packet

The 47 continuous coordinates stay inside the cell. They are not copied over
every edge.

For branch-pair groups `G_j={2j-1,2j}`, `j=1:4`, a cell exports

```math
p_i=[F_1,N_1,I_1,\ldots,F_4,N_4,I_4]\in\mathbb{R}_{+}^{12}.
```

- `F_j`: bounded fast excitation and positive voltage evidence.
- `N_j`: bounded voltage-unblocked NMDA and plateau evidence.
- `I_j`: bounded GABA, adaptation, and negative/opponent evidence.

Soma and apical state modulate these terms. Hard soma and four plateau-group
events travel on a separate control plane.

This packet is a testable communication hypothesis, not a claim that all 47
state dimensions are losslessly transmitted. Let `r_task` be the reachable
task-tangent rank, `m=12` the packet width, and `T` the number of observation
waves. A necessary observability condition is

```math
mT \ge r_{task}.
```

The initial maximum is `T=4`, giving at most 48 temporal observations. The
stacked observability Jacobian and its singular spectrum must be measured. If
the condition fails in practice, this packet hypothesis is falsified; fields
are not patched one by one.

### 4.1 Why every lossless merge is binary

The cell has 27 typed inputs: eight basal plus one apical compartment, each
with AMPA/NMDA/GABA. Three apical channels are reserved for context, leaving 24
basal channels.

An ordered binary merge fits exactly:

```text
left  12D -> basal compartments 1:4
right 12D -> basal compartments 5:8
```

```math
2\times12=24\le27.
```

A four-child merge cannot preserve four packets in one transition:

```math
4\times12=48>24.
```

Consequently all information-spine aggregation is an ordered binary tree.
Four-ary pooling, mean pooling, and same-bin child accumulation are forbidden.

Semantic edges after the ordered spine may use a small role-specific typed
projection `12 -> 3`, one source per basal branch, with at most eight sources
per receiver. This is task-specific compression, not a claimed lossless merge.

## 5. Fixed multiscale information spine

The initial core has 1,458 high-dimensional cells derived from board geometry,
not from an inherited width such as 48.

| Component | Cells |
|---|---:|
| before spatial plane | 240 |
| after spatial plane | 240 |
| row binary trees, two planes | `24*(10-1)*2 = 432` |
| column binary trees, two planes | `10*(24-1)*2 = 460` |
| first-tier semantic motifs | 32 |
| evidence cells | 32 |
| private output cells | 22 |
| total | **1,458** |

### 5.1 Spatial planes

Each board coordinate owns one before cell and one after cell. A spatial cell
receives its eight ordered neighbors on eight basal branches and its center,
plane identity, and phase on apical input. `OUTSIDE` is explicit.

Absolute position is fixed anatomy: node identity plus ordered path. There is
no learned position lookup.

### 5.2 Row and column trees

Each plane has:

- one ordered binary tree per row: nine internal nodes for ten leaves;
- one ordered binary tree per column: 23 internal nodes for 24 leaves.

Left/right and upper/lower children always occupy disjoint basal groups. No
permutation-invariant reduction is introduced.

The crossed row and column covers provide multiple independent cuts through
the board. Task-relevant Jacobian rank and minimum singular value are explicit
promotion measurements.

### 5.3 Semantic motifs and evidence bank

The 32 first-tier motif cells are eight families with four canonical slots:

1. placement-centered 3x3 local patch;
2. landing and support;
3. touched row;
4. touched column;
5. row band;
6. column shard;
7. raw footprint and clear/remap;
8. queue, REN, B2B, and T-spin context.

The four raw placement slots are sorted by a deterministic canonical rule and
retain absolute coordinates. `ABSENT` is explicit.

The following 32 evidence cells apply a fixed balanced mixing schedule over
the eight motif families. Every evidence cell receives at most eight sources,
and no single geometry hub is allowed.

### 5.4 Capacity scaling

The fixed information spine is never rewired. Additional semantic memory may
be represented by dormant Reduced-Hay cells and cold candidate contacts, but
only source-major active edges are scanned. Stored capacity may grow without
forcing all capacity through every candidate.

The capacity count is selected by the exact 1/8/16/32/64 ladder and the
20-updates/s gate. It is not selected by preserving a legacy dimension.

## 6. Output geometry

There is no raw input-to-output, leaf-to-output, relation residual, or
all-source-by-22 dense path.

Each private output cell receives at most eight evidence sources. One semantic
edge applies a role-specific typed `12 -> 3` projection and deposits the result
onto one basal branch. The scalar emitted by an output cell is

```math
m_j=V^{pre}_{soma,j}-\theta_j.
```

The 22 physical output cells are structured as:

| Role | Cells |
|---|---:|
| state value opponent population | 2 |
| candidate advantage population | 8 |
| death opponent population | 2 |
| four geometry opponent populations | 8 |
| uncertainty opponent population | 2 |

Candidate Q uses a dueling decomposition over all candidates of one state:

```math
\hat q_a=V(s)+A(s,a)-\frac{1}{|C_s|}\sum_{b\in C_s} A(s,b).
```

The 16 current quantile targets are equal to the same deterministic teacher Q;
they are not treated as 16 independent return-distribution dimensions. The
external ABI remains 22D via

```math
\hat q_{\tau}=\hat q+c_{\tau}\,\operatorname{softplus}(\sigma),
```

where fixed ordered `c_tau` coefficients make the deterministic optimum
`sigma -> 0`.

### 6.1 ListNet scale

Student-self-standardization is forbidden because it removes score amplitude
and conflicts with raw-Q calibration. A teacher-derived state scale is shared:

```math
s_T=\sqrt{\operatorname{Var}(q_T)+\sigma_0^2},
```

```math
p_T=\operatorname{softmax}\!\left(\frac{q_T-\bar q_T}{\tau s_T}\right),
\qquad
p_S=\operatorname{softmax}\!\left(\frac{\hat q-\bar{\hat q}}{\tau s_T}\right).
```

The ranking loss is `KL(p_T || p_S)`. Raw-Q Huber calibrates common shift and
absolute scale.

## 7. Candidate-delta execution

### 7.1 Mandatory analog sweep

Every affected cell in the candidate DAG is advanced once even when it does
not spike:

```text
changed typed input
-> affected spatial cells
-> row/column ancestors
-> semantic motifs
-> evidence cells
-> private outputs
```

Subthreshold evidence is therefore never gated by a hard event.

### 7.2 Optional hard-event refinement

Additional computation uses a fixed source-major event graph:

```text
soma spike / plateau onset or offset
-> active outgoing edges
-> typed destination inbox
-> next synchronous wave
```

Wave semantics are Jacobi-synchronous:

1. sources and edges are visited in deterministic sorted order;
2. all contributions are accumulated;
3. each unique destination is advanced once;
4. newly generated events enter only the next wave.

The wavefront stops when empty or after four waves. Same-wave leakage is an
error. A hard event must causally change a later visited state or output; a
diagnostic-only event is an implementation failure.

### 7.3 COW complexity

The state-common before plane is computed once per state. Candidates own only
generation-stamped copy-on-write overlays.

For no-clear candidates with `k` changed sites, expected work is

```math
W(a)=O(k\log 240+E_{event}).
```

Line clearing can change `Theta(240)` destination cells. The initial canonical
implementation therefore uses an explicit exact slow path: materialize the
24x10 after bitboard and rebuild the candidate after-plane forest. It does not
make a false `O(4 log N)` claim.

Persistent row-remap optimization is allowed only after an associative summary
contract is proven. Mutable nonlinear Hay trajectories are not assumed to be
reusable monoids.

## 8. Learning contract

Continuous analog credit, hard-event control credit, homeostasis, and
structural plasticity are different estimators on different clocks.

### 8.1 Teacher-free eligibility

Eligibility is generated during forward or forward replay without teacher
access:

```math
E_{i,p}^{t+1}=A_i^tE_{i,p}^{t}+\frac{\partial F_i^t}{\partial p}.
```

Multi-compartment traces cover voltage, AMPA, NMDA, GABA, plateau, apical,
soma, adaptation, packet projection, and synaptic parameters. A cell touched
by the mandatory analog sweep may learn even if its hard spike is zero. A truly
unvisited cell receives zero task update.

### 8.2 Two-pass ListNet learning

```text
pass 1: all candidates forward, teacher-free trajectory
boundary: compute 22D raw loss derivatives for the whole candidate list
pass 2: deterministic forward replay, regenerate eligibility, apply modulation
```

The block/cell learning signal is

```math
M_i=B_i\delta_{raw}+C_i^T\epsilon_i^{local}.
```

- `B_i` and `C_i` are seed-fixed, cell-family-specific, non-trainable maps.
- `epsilon_local` predicts next packet, pattern completion, event continuation,
  and local energy; it does not directly predict teacher Q per block.
- teacher Q may affect `delta_raw`, hence the third factor.
- teacher Q never affects eligibility generation.
- trainable output weights are not used to create recurrent third factors.

The local update is

```math
\widehat g_p=\sum_{i,t} M_i^t J_{\chi_i}^t E_{i,p}^t.
```

This is random-feedback multi-compartment e-prop with DECOLLE-inspired local
prediction, not full BPTT and not strict DECOLLE.

### 8.3 Separate hard-event learner

Hard control eligibility uses a bounded surrogate around the pre-reset margin:

```math
T_{i,p}=\widetilde H'(m_i)\frac{\partial m_i}{\partial p}.
```

Its learning signal contains causal continuation benefit and an explicit event
energy cost. It is measured and clipped separately from analog credit. The
canonical ListNet analog gradient does not silently include event surrogate
terms.

### 8.4 Exact oracle

Exact conditional analytic VJP is retained only for:

- finite differences on event-stable trajectories;
- capacity testing;
- exact/local alignment;
- sparse/dense and COW/full oracles.

Production learning keeps exact analytic VJP for the small output population
and structured decoder only. Interior cells and graph contacts use local
eligibility.

### 8.5 Plasticity clocks

```text
every update:       output and decoder
medium interval:    recurrent analog local update
separate interval:  hard-event control update
slow interval:      intrinsic homeostasis and synaptic scaling
very slow interval: utility-based structural rewiring
```

The mandatory information spine is immutable. Only optional recurrent/event
contacts may rewire. Rewiring preserves fanout, duplicate prohibition,
receptor budgets, one-node/one-swap limits, and resets optimizer moments and
eligibility for changed contacts.

Raw-space AdamW is not applied to inverse-softplus conductance parameters,
because shrinking a negative raw value toward zero increases the physical
conductance. Positive conductances use physical-space priors/projection.

Sleep learning is not part of the initial canonical model. It may be tested
only after wake learning passes all gates.

## 9. CPU execution contract

- Fixed SoA/AoSoA arena; initial SIMD candidate width is measured, not assumed.
- State-common before plane is computed once per state.
- Candidate COW uses generation stamps and fixed active lists.
- Hot edge arrays are source-major and contain only forward fields.
- Optimizer moments, utility, and dormant contacts are cold arrays.
- A worker owns all waves for one candidate microbatch.
- No spike-level global jobs, global atomics, or all-edge scans.
- The two global mathematical boundaries are ListNet completion and
  deterministic gradient reduction/optimizer application.
- Sparse and dense kernels are numerical oracles of each other.
- Lazy decay is used only where it exactly reproduces repeated zero-input cell
  and eligibility transitions; otherwise the cell remains active.
- Forward and production local replay have zero hot allocation and zero hot GC.
- Serial and barrierless paths call the same optimizer-boundary projection.

The user-specified external throughput gate is `>=20 updates/s` at batch 8 on
the current 20-worker CPU configuration. Static stored cell count is not
scanned per candidate; wall time must be explained by visited cells, visited
edges, active compartments, and bytes.

## 10. Falsification and promotion gates

The implementation order is fixed. New audit categories are not added during
a gate unless an existing invariant fails.

### G0: design freeze

This document, source dimensions, equations, and output semantics are frozen
before implementation.

### G1: information, numerical, and causal tests

- new input representation has no teacher-disagreeing collision;
- raw placement, queue order, REN, NONE, OUTSIDE, and no-event remain distinct;
- line-clear remap matches a materialized oracle;
- binary child order and typed receptor identity affect state;
- COW equals full recompute;
- sparse equals dense;
- serial equals barrierless;
- event intervention changes a later state/output;
- event-stable finite differences satisfy a rounding/conditioning-derived bound;
- hot allocation and GC are zero.

### G2: exact capacity ladder

Run nested scratch panels `1 -> 8 -> 16 -> 32 -> 64`.

- tie-aware top-1 must reach 1.0;
- excess must reach the free-logit oracle floor within numerical/statistical
  uncertainty;
- an equally long confirmation window must not collapse.

Exact failure means representation, communication, topology, or output is
wrong. Local credit is not modified to hide it.

### G3: local-credit gate

On identical snapshots and batches across multiple seeds:

- every material parameter group has a positive one-sided confidence bound for
  exact/local alignment;
- no significant inverse group remains;
- optimally scaled local steps reduce loss relative to no-step and shuffled
  controls;
- nonspiking analog-touched cells are measured separately;
- exact success plus local failure falsifies the local rule, not the model.

### G4: CPU gate

- batch 8 steady-state throughput is at least 20 updates/s;
- sparse/dense crossover is measured per kernel;
- scaling with dirty sites and event visits matches the work model;
- fixed arena, zero hot allocation, zero hot GC, and deterministic reduction
  remain true.

### G5: only three architectural ablations

1. active Reduced-Hay versus Point/passive cell;
2. event refinement versus mandatory analog sweep only;
3. ordered multiscale topology versus degree-preserving shuffle/local-only.

An added mechanism is retained only if it improves the paired equal-wall-clock
quality/CPU Pareto frontier with confidence intervals.

### G6: 1k, 10k, then 100k

The 1k run must have a significantly negative loss slope, active mechanisms,
and maintain the CPU gate. The 10k run logs only the predeclared telemetry every
1k. A 100k run starts only if the 10k curve is not Pareto-dominated by Point or
GRU controls and its extrapolation can still reach the comparison frontier.

Non-finite values, causal-contract violations, configured mechanisms with zero
activation, or sustained statistically significant regression stop the run.

### G7: final comparison

- same teacher, candidates, input information, width, and loss;
- development and sealed panels separated;
- at least three training seeds;
- primary comparison at equal wall-clock;
- secondary comparison at equal teacher exposure;
- paired confidence intervals for excess, tie-aware top-1, NDCG, pairwise,
  updates/s, states/s, RSS, allocation, and GC.

PreAct or DSRLN is beaten only if the new model dominates the relevant
quality/time Pareto frontier with uncertainty included.

## 11. Loop-prevention rule

One failed gate permits one root-cause design correction at the layer named by
the failure classification, followed by the same gate again. A second failure
freezes that hypothesis as a falsified control. A third local patch is
forbidden.

```text
exact failure          -> representation/topology/output hypothesis
exact pass/local fail  -> credit-assignment hypothesis
small-N pass/large-N fail -> capacity/interference hypothesis
quality pass/CPU fail  -> execution hypothesis
```

## 12. Canonical implementation boundary

Retain and adapt:

- Reduced-Hay numerical equations and analytic cell VJP;
- candidate input/data contracts after correcting line-clear/meta encoding;
- typed afferent primitives;
- fixed arenas, source-major adjacency, barrierless workers;
- checkpoint fail-closed behavior;
- exact oracle infrastructure.

Replace in the canonical call chain:

- giant program bank and program packets;
- current 48-relation/48-motif topology;
- full-47D inter-cell transport;
- relation residual and current structured readout;
- one-phase analog orchestration;
- exact reverse as production training;
- duplicated serial/barrierless optimizer-boundary logic.

Implementation is performed in the existing directory on `main`. No `v2`,
`v3`, `clean`, new branch, stale API shim, or compatibility path is created.
