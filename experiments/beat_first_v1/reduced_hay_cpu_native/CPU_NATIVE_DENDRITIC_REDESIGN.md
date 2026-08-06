# CPU-native dendritic redesign decision record

Status: 2026-08-06. This is a redesign contract, not a success report.

> **PreAct/DSRLN超えは未達である。** 旧100k runは34,000更新で停止し、
> 後続のcorrected Dendritic Delta Forest (DDF) も64-state短期試験で全階層の
> hard eventが0のまま、excess `1.900604`、tie top-1 `0.328125`に留まった。
> 固定tree cut、低次元message、signed scalarのE/I変換を正本から外す。
> 以後の唯一の正本候補は **HD Candidate-Delta Relation Graph (HD-CDRG)**
> である。これはTwinPropの論文再現ではなく、高次元Reduced Hay cellと
> candidate差分実行がCPU実時間あたりのTetris性能を改善するかを反証可能に
> 検査する独自のCPU仮説である。

### 2026-08-06 canonical amendment: relation-to-motif DAG

The first relation-only 10k gate plateaued at excess `0.234` and tie-aware
top-1 `0.547`, materially behind the comparable PreAct/DSRLN controls. The
measured output representation also lost effective rank and the trainable
relation-to-output conductances concentrated on a small subset. This falsified
the shallow `relation -> output` form; it did not falsify Reduced-Hay cells.

The canonical path is therefore now:

```text
program packets
-> 48 relation Reduced-Hay cells
-> fixed geometry-derived relation-to-motif anatomy
-> 48 motif Reduced-Hay cells
-> source-specific bounded structured readout
-> 22 output Reduced-Hay cells
-> cell-local 47-lane continuous readout
```

Both the direct program-to-output graph and the independently trainable
all-to-all relation-to-output graph were deleted. A fixed `1/8` structured
relation residual remains to preserve first-order evidence without bypassing
the motif interface. The motif closure is the exact union of four fixed
destinations for every affected relation, so candidate-local COW remains
bounded and no runtime router is introduced. Source-specific `48 x 22` gains
retain row/column/tile/stripe identity; family-wide averaging is explicitly
forbidden. This amendment supersedes later relation-only forward diagrams in
this historical record wherever they conflict with the canonical README and
code.

The program-bank payload remains 16D, but a cell packet is now 47D:
`9 compartments x (V, AMPA, NMDA, GABA, plateau) + exact pre-reset margin +
adaptation`. Relation-to-motif anatomy co-locates each five-state compartment
bundle and uses 57 typed contacts/source. Any later historical paragraph that
calls 16D the inter-cell packet shape is superseded by this amendment; 16D now
means only the spatial program payload.

## 1. 停止時点を固定する

停止したrunは `results/training/` に保存されている。

| 項目 | 保存値 |
|---|---:|
| update | 34,000 |
| teacher-state exposure | 272,000 (`batch=8`) |
| train excess / top-1 | 1.969940 / 0.169750 |
| fixed panel excess / top-1 | 2.088422 / 0.171875 |
| validation excess / top-1 | 1.820668 / 0.148438 |
| firing rate | 0.00137063 |
| preceding 30k--33k throughput | 4.784--5.162 updates/s |
| final 34k row throughput | 3.475 updates/s |

The final-row speed includes the stopping boundary and is not used as the
steady-state estimate. The retained files are:

```text
results/training/latest.jls
SHA256 DB0719F61E968B00048809AC996F6958556BBDB1A1162DF7B82B14EE48192F3B

results/training/progress.tsv
SHA256 94FA1A4C44D5C2BFA48576DB75C10CEA38DF533BEF806105334D6DBDFF2C650F
```

The earlier continuous/batched readout control in
`results/highdim8_local_batched_head_100k/progress.tsv` reached, at 100k,
train top-1 `0.540375`, fixed-panel top-1 `0.554688`, validation top-1
`0.531250`, and `24.060 updates/s`. It is not a fair winner comparison because
the model and learning path differ. It is nevertheless a decisive control:
high-dimensional cell dynamics alone do not explain the present regression,
and replacing a ranking score by a hard IEEE word did not produce an efficient
Tetris learner.

### 1.1 Candidate-delta controlで確定したこと

旧hash bankは、1,048,576 physical rowsのうち到達可能なのが154,242 rowsだけで、
約85.3%が学習不能だった。これを局所patternと絶対位置から直接addressする
collision-free bankへ置換した。正本bankは208,448 rows、payload 16、
3,335,168 trainable valuesで、全binary local domainの列挙が厳密に全rowへ到達する。
model全体は3,440,710 parametersである。

compact版の過学習結果は次のとおりだった。

| states | updates | initial excess | final excess | final tie top-1 | train states/s |
|---:|---:|---:|---:|---:|---:|
| 8 | 500 | 4.618441 | 0.015859 | 1.000000 | 230.56 |
| 16 | 1,000 | 4.463010 | 0.032975 | 1.000000 | 338.89 |
| 64 | 3,000 | 5.100877 | 0.015868 | 1.000000 | 406.66 |

したがって実装故障や小パネル容量不足は、flat版の失敗原因ではない。

full-data scratch 1,000 updatesでは、固定128-state development panelが
excess `1.127302 -> 0.489725`、tie top-1 `0.203125 -> 0.375000`へ改善した。
旧34k validationの`1.820668 / 0.148438`は大幅に上回った。一方で同一panelの
PreActは`0.055964 / 0.796875`、DSRLNは`0.100555 / 0.804688`であり、まだ遠い。

推論速度は同じ評価器で1,435.14 states/sだった。これはDSRLNの40.51 states/s、
PreActの3.32 states/sより大幅に速いが、品質未達なので勝利とは扱わない。
checkpointと評価は以下に固定した。

```text
results/candidate_delta_compact_semantic_1k_20260806_0227/latest.jls
SHA256 10AFF7D2F31DF22DBA787D7139C12752A493B2B301C02F39A24318B4E1EBC40F

results/candidate_delta_compact_semantic_1k_20260806_0227/development_validation_all.json
```

最重要の反証は、decision hard eventがupdate 500以降0でも学習が継続したこと
である。flat版は実質的に、27D factor broadcast、50 continuous soma margins、
22 x 50 projectionであり、graph-level event-driven SNNではなかった。5k延長は
中止し、このcheckpointをHD-CDRGの比較controlとして保存する。

### 1.2 教師入力の不可逆aliasを実データで確定した

さらに、flat版と初期DDF案に共通する、学習では修復不能な入力欠落を発見した。
`teacher_v3`全110,366 states・4,817,295 candidates（width 80）を走査すると、
post-clear boardとT-spinが同じ候補対が119,725組あり、そのうち4,303組は
teacher Qが異なった。最大差は`0.94140625`で、4,303組すべてがline clearを
伴う異なるraw placementだった。同じraw placementでのQ不一致は0件だった。

```text
dataset manifest SHA256
1F63172F33F8CEE17B7ADA88D4F35CDFA94B8D7DD5751C8E8244008CAA526DED
```

before board、queue、REN、B2Bはstate共通で、旧aux・added・removedは
before/post-clear after/T-spinから決定される。そのため旧studentは、line clearで
消えた配置形状を復元できず、同じ入力へ異なる正解を要求されていた。DDF出力には
`placement_weight[240,22]`を追加し、実行時には実際の4 occupied positionsだけを
22個の出力cellへ配送する。追加接触は88/candidateであり、240位置のdense scanは
行わない。

## 2. Mathematical diagnosis

### 2.1 The optimized geometry is wrong for ranking

For candidates in one state, ListNet uses

\[
p_i=\frac{\exp(q_i)}{\sum_j\exp(q_j)}.
\]

It is invariant to a common additive score: `softmax(q + c) = softmax(q)`.
Only relative candidate evidence matters. IEEE-754 bit BCE instead optimizes
Hamming-like agreement between encodings. Adjacent real values can change many
bits at exponent, sign, normalization, or rounding boundaries, while a small
bit error can cause a large ordinal change. Exact Float32 arithmetic is useful
as a separate learned machine, but it is not the correct metric space for a
Tetris ranker.

The new output is therefore one continuous ranking logit per candidate. Hard
spikes remain execution events; they are not required to be the scalar answer.

### 2.2 Candidate-common computation is recomputed 43.662 times

The canonical loader reads
`D:\\tetris-paper-plus\\datasets\\beat_first_v1\\teacher_v3`. Its frozen split
audit has 100,243 training states and 4,376,773 candidates:

\[
\bar C=4{,}376{,}773/100{,}243=43.662
\quad\text{candidates/state}.
\]

Current forward advances 480 recurrent cells for 10 cycles for every candidate:

\[
4{,}800\ \text{cell advances/candidate},
\]

\[
43.662\times4{,}800\approx209{,}578
\ \text{cell advances/state}
\]

before counting output-cell work and reverse replay. Yet the board, queue, B2B,
REN, and most spatial relations are common to every candidate in that state.
The implementation discards this algebraic sharing.

Let `x` be state-common information and `d_a` the deterministic placement/clear
delta for action `a`. The intended decomposition is

\[
h_a=F_{\mathrm{delta}}(h_0(x),d_a),
\qquad q_a=b(x)+r(h_a).
\]

`h_0(x)` is computed once and shared immutably. Candidate-specific state is
copy-on-write. The common scalar `b(x)` cancels exactly in ListNet, while
interactions between `x` and `d_a` remain available through `h_0`.

### 2.3 Event sparsity currently does not skip the dominant work

At 34k the observed hard firing rate is about `0.137%`. Sparse source-major
delivery skips most recurrent **edges**, but all cells still integrate all
continuous compartment states on all ten cycles. Thus low firing reduces only
one term of the cost:

\[
T_{\mathrm{old}}
=O\!\left(C(N_{cell}T_{cycle}+E_{fired}+N_{out})\right).
\]

The intended event-wavefront cost is

\[
T_{\mathrm{new}}
=O\!\left(N_{base}+\sum_a
  (|V_a^{\mathrm{touched}}|+|E_a^{\mathrm{delivered}}|)
\right).
\]

A cell is lazily decayed when touched. There is no global ten-cycle scan.

### 2.4 Transient state is large, stored capacity is small

The current system evolves 480 cells with 48 states each, but has only 67,882
independent parameters. It pays for a large dense transient trajectory without
obtaining PreAct/DSRLN-scale long-term capacity. The redesign separates:

- **stored capacity:** many dormant dendritic branch/program entries in CPU RAM;
- **active compute:** a fixed small number of entries and graph factors touched
  by one candidate delta.

This is the CPU-native opportunity. A GPU can implement the same algorithm;
the claim is not impossibility. The target regime is irregular, branchy,
cache-resident sparse work with small per-candidate frontiers, where dense GPU
occupancy and batched matrix multiplication are poor matches.

### 2.5 Input placement does not reproduce the relevant dendritic mechanism

The present sensory encoder gives a rail a deterministic primary contact and an
associative-lane contact. TwinProp instead optimizes synaptic strengths and
effective dendritic locations over multiple contacts. Fixed rail duplication
is not an implementation of learned dendritic placement. The redesign retains
rail identity as typed local factors and learns which dormant branch programs
are useful, without a central top-k router.

### 2.6 Learning-rule work has been confounded with architecture search

DECOLLE/e-prop can be valuable after the representational and computational
graph is sound. It cannot repair the wrong output geometry or repeated common
work. The new architecture is first trained with an exact sparse reverse over
the actually visited graph. Local eligibility is promoted only after it has
positive alignment and does not erase the exact model's quality.

## 3. TwinProp re-audit and correction

[TwinProp](https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full)
does **not** train a reduced 11-state cell end-to-end for Tetris, and it does
not distill or freeze such a cell. Its differentiable digital twin is trained
against somatic subthreshold voltage and somatic spikes, then the twin's
internal parameters are frozen while task optimization changes input synaptic
strengths and effective dendritic locations. Its task readout is a somatic
spike/no-spike decision in a time window, not a continuous Q value or an
IEEE-754 word. The task optimization uses conventional gradient descent on
GPUs; the paper explicitly does not claim a biological local learning rule.

The related single-cell result that an NMDA-rich L5 pyramidal cell requires a
deep temporal surrogate is reported by
[Beniaguev et al.](https://doi.org/10.1016/j.neuron.2021.07.002). Structured
sparse dendrite-to-soma connectivity and restricted receptive fields, rather
than morphology alone, are also central in
[Thorat et al.](https://doi.org/10.1038/s41467-025-56297-9). Branch-specific
timescales are supported by
[DH-SNN](https://doi.org/10.1038/s41467-023-44614-z).

[Wu et al.](https://doi.org/10.1016/j.patter.2026.101520) supplies the crucial
engineering correction. At matched parameter and compute budgets, active
dendrites did not provide a free increase in learning capacity; their main
advantage was local nonlinear aggregation that reduced downstream
communication and memory traffic. Therefore the design must not expose every
branch voltage/NMDA/plateau coordinate to a wide downstream projection. The
high-dimensional state stays local and earns its cost by compressing many
typed inputs into a compact message.

What is retained from this literature:

- multiple persistent compartment states;
- E/I separation and voltage-dependent NMDA/plateau nonlinearity;
- multiple contacts and learned slow structural placement;
- branch-local spatiotemporal pattern detection;
- soma spike as an event and analog dendritic state as computation.

What is not imported as a claim:

- biological fidelity as the final objective;
- a full Hay or digital-twin cell for every Tetris site;
- 11-state distillation/freezing as a paper result;
- hard Float32 bits as the paper's representation;
- DECOLLE/e-prop as TwinProp's learning rule.

Event-driven recomputation of only affected graph nodes is independently
supported by [AEGNN](https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html),
and exact event-based adjoints are demonstrated by
[EventProp](https://doi.org/10.1038/s41598-021-91786-z). These are feasibility
references, not evidence that the proposed Tetris model already wins.

EventPropのexact性はcontinuous-time LIFのevent時刻とjump conditionに対する
随伴法である。DDFの離散3-phase hard eventは明示的surrogateであり、exactと
呼ぶのはrecorded event sequenceを条件にしたcontinuous compartment VJPだけとする。

## 4. Canonical architecture contract

The only canonical candidate is the **HD Candidate-Delta Relation Graph
(HD-CDRG)**. The crossed DDF remains a falsified control and is not reachable
from the canonical entrypoint.

```text
state-common binary rails
  -> collision-free 16D spatial program packets
     (240 positions x before/after plane = 480 sources)
  -> 48 structured Reduced Hay relation cells
     (24 row + 10 column + 14 cross/action)
  -> 22 typed Reduced Hay output cells
  -> continuous 22D ranking output

candidate placement / deterministic clear delta
  -> changed after-plane program packets only
  -> affected relation-cell COW overlays only
  -> mandatory direct source-to-output delta jump
  -> affected relation-to-output delta jump
```

There is no reduction tree, hierarchy root, global reducer, central workspace,
query/key, learned top-k gate, or runtime route selection. The fixed graph says
where evidence may flow; actual candidate work is selected by changed sources,
their fixed relation closure, and local cell dynamics.

### 4.1 Fixed dimensions and input ownership

The dimensions below are part of the checkpoint fingerprint, not tuning aliases:

```text
BOARD_POSITION_COUNT       = 240
SEMANTIC_PLANE_COUNT       = 2
SPATIAL_SOURCE_COUNT       = 480
PROGRAM_PACKET_DIM         = 16

ROW_RELATION_COUNT         = 24
COLUMN_RELATION_COUNT      = 10
CROSS_ACTION_COUNT         = 14
RELATION_COUNT             = 48

OUTPUT_COUNT               = 22
REDUCED_HAY_STATE_DIM      = 48
REDUCED_HAY_PARAMETER_DIM  = 46
COMPARTMENT_COUNT          = 9   # 8 basal + 1 apical
RECEPTOR_COUNT             = 3   # AMPA, NMDA, GABA
CELL_INPUT_DIM             = 27
CELL_PACKET_DIM            = 16
CELL_PACKET_BYTES          = 64
```

Relation IDs are immutable:

```text
1:24   row relations
25:34  column relations
35:48  cross/action relations
```

No teacher-derived feature is admitted. Every candidate delta is a
deterministic transformation of information available to all comparison
models.

- state-common: board before placement, HOLD + NEXT5 role tokens, B2B, REN,
  and the other shared rails;
- candidate-local: raw four-cell placement, post-placement/post-clear board
  changes, row-index remapping, and action identity;
- spatial source: one 16D semantic program packet for each of the 240 positions
  in each of the before/after planes;
- relation state: 48 private high-dimensional Reduced Hay cells;
- output state: 22 private high-dimensional Reduced Hay cells.

The 480 spatial sources are semantic program embeddings, **not** 480
`ActiveApicalCell` trajectories. The dynamic high-dimensional population is
exactly 70 cells (48 relation + 22 output). Its long-term spatial capacity is
the `208,448 x 16 = 3,335,168`-parameter dormant program bank; a candidate reads
only rows required by its changed packets.

Line clears use exact changed positions and index remapping. Candidate execution
does not scan or re-encode the full 240-position after plane.

### 4.2 High-dimensional cell packet

A relation cell retains the full 48-coordinate Reduced Hay state internally:

```text
8 branch voltages
8 x AMPA / NMDA / GABA conductances
8 plateau states
apical voltage / conductances
soma voltage
adaptation state
hard soma event history
```

Its canonical analog message is the implemented
`HighDimensionalCellPacket`: **16 Float32 coordinates, exactly 64 bytes**.
Branch identity is never averaged away.

```text
packet[1:8]
    = softly bounded, resting-centered voltage of each basal branch

packet[9:16]
    = softly bounded signed slow state of the same branch
    = AMPA + voltage-unblocked NMDA + plateau - GABA
```

This 16D analog vector is the only inter-cell packet shape. AMPA, NMDA, and
GABA are fixed contact metadata and destination inbox slots, not packet axes.
The former soma-margin/mean-plateau/event triplet is removed. Packet pullback is
the exact analytic cotangent to the selected final 48-state column. The packet
map owns no trainable projection, so parameter dependence remains in the
recorded cell trajectory.

Hard soma and plateau events are separate 0/1 control outputs of the single
mandatory transition. They are reported but do not add, replace, zero, or gate
canonical analog propagation. Spike-only communication remains forbidden
because it discards the subthreshold computation that motivates the cell.

### 4.3 Receptor-typed nonnegative contacts

The historical conversion

```text
signed scalar > 0 -> AMPA + NMDA
signed scalar < 0 -> GABA
```

is forbidden. A conductance-based cell depends independently on total E/I
conductance and net drive; no global scalar can make that sign switch symmetric.

Every logical contact permanently owns:

```text
source packet field
source opponent polarity (+1 or -1)
destination cell and compartment
fixed receptor identity (AMPA, NMDA, or GABA)
trainable raw conductance
```

For contact `e`,

\[
a_e=\max(p_e x_{f_e},0),
\qquad
g_e=\operatorname{softplus}(\gamma_e),
\qquad
I_e=g_e a_e.
\]

`I_e` is deposited only into the contact's fixed receptor slot. Physical
conductance is therefore always nonnegative, and gradient updates cannot turn
AMPA into GABA or vice versa. Opponent contacts carry semantic sign without
changing receptor anatomy. A zero source delta activates neither polarity and
is true silence. Separate contacts can deliver AMPA, NMDA, and GABA
simultaneously to the same destination cell.

Canonical candidate delivery first forms the signed 16D packet delta
`candidate_packet - base_packet` in caller-owned scratch. Every logical edge
then owns an opponent contact pair with the **same receptor identity**:

```text
positive contact: polarity +1 -> one fixed destination compartment
negative contact: polarity -1 -> a separate fixed destination compartment
```

The signed packet delta is passed as an ordinary source packet to the
source-major `deposit_sources!` path. Only one side of a pair activates for a
nonzero lane, and neither side activates for zero. This never subtracts a
conductance, never constructs a negative receptor inbox, and never changes
receptor identity. Canonical forward/reverse must not use algebraic
`candidate_inbox = base_inbox + deposit_delta!(...)`: conductance-based state is
advanced from the common base state with a delta-only, nonnegative typed inbox.
The paired selected-source pullback returns the signed packet-delta cotangent;
the spatial COW layer alone owns its equal-and-opposite distribution to
candidate and base program packets.

### 4.4 Collision-free program-packet memory

Long-term spatial capacity lives in a large bank of dormant 16D program rows.
Destination, branch, receptor, and polarity are deliberately absent from the
row payload; those anatomical identities belong to the fixed typed-contact
graphs. There is no learned global query/key dot product and no top-k gate.
Four collision-free semantic rows are summed with fixed `1/sqrt(4)` scaling
and a soft bound to materialize one 16D, one-cache-line local program packet.

The spatial address must expose that dormant capacity; resident bytes alone
are not model capacity. Each spatial source therefore performs exactly four
multiresolution lookups:

```text
3x3 morphology
morphology + row
morphology + column
morphology + row + column + before/after plane
```

For the 24-by-10 binary board, the exact semantic domains of these four rows
are `832`, `14,272`, `5,312`, and `188,032`. They are assigned consecutive,
collision-free physical row ranges. The canonical bank therefore has exactly
`208,448` reachable rows and `3,335,168` Float32 payload parameters; resident
and address-reachable capacity are identical. A source scans its nine local
sites once, packs the valid 4/6/9 occupancy bits, and adds that mask to four
fixed Int32 position bases. Hashes, buckets, slots, deduplication and the old
capacity CLI are absent. State preparation materializes the common `16 x 480`
base grid once. A candidate rematerializes only changed after-plane columns and
forms the exact overlay-minus-base packet delta. Experiments still report
actually touched rows separately because untouched semantic programs remain
dormant.

### 4.5 Fixed relation topology and COW execution

The 480 spatial sources feed a shallow fixed bipartite topology. Every source
has exactly four independent relation memberships:

```text
one physical-row relation
one physical-column relation
one 6-row x 5-column tile-local relation
one balanced small-world stripe relation
```

This gives 24 row, 10 column, and 14 cross/action relation cells. The immutable
`RelationLayout` owns source membership, source plane, destination relation,
destination basal compartment, and local role. There is no relation-to-relation
edge. Cross/action IDs 35:42 are the eight Cartesian products of four six-row
bands and two five-column halves; IDs 43:48 are six balanced small-world
stripes. None is a disguised global reducer.

The direct source-to-output jump is mandatory. It preserves a short path from
changed 16D program packets to every output role, so the 48-cell relation bank
cannot become another information cut. A separate fixed typed graph sends all
relation packets to the output bank. Context and raw four-cell placement use
their own sparse typed contacts; neither is collapsed to one apical scalar.

Execution ownership is:

- the state owns immutable base program packets, base relation inbox/state/
  packets, and the common output input snapshot;
- a candidate owns generation-tagged COW overlays only for changed after-plane
  packets, affected relation cells, and its 22 output cells;
- the affected-relation closure is a bounded bitset/list generated directly
  from changed source IDs, without a heap, tree traversal, or global queue;
- signed source-packet deltas traverse opponent-paired contacts into fresh
  delta-only relation and mandatory direct-output inboxes; signed
  relation-packet deltas enter that candidate output inbox the same way;
- untouched base values are read through the immutable snapshot and are not
  copied or advanced;
- hard events are separate control observations and never gate mandatory
  source-delta, relation-delta, or output propagation;
- SoA pools, fixed arenas, source-major traversal, barrierless candidate jobs,
  hot allocation `0`, and hot GC `0` remain mandatory.

Every relation and output cell starts from its caller-supplied common base state
and executes exactly **one mandatory candidate-delta transition**. There is no
hidden rest reset, relaxation phase, repeated input injection, or event-based
skip. The resulting hard event is the event of that one transition.

### 4.6 Typed output and ranking geometry

The output bank owns 22 independent high-dimensional Reduced Hay cells. It does
not use an auxiliary-to-Q cascade, a dense observation head, or one scalar
apical context bottleneck. Fixed source-to-output and relation-to-output typed
contacts distribute packet fields across basal/apical compartments while
preserving receptor identity.

For output `k`, the continuous supervised value is

\[
y_k = g^{margin}_k
      (m^{final}_k-m^{rest}_k)
    + g^{plateau}_k\,\overline p^{final}_k
    + b_k.
\]

Both gains and the bias are trainable. The hard event of the single mandatory
transition is reported separately as a control event and is deliberately absent
from the continuous task pullback. Thus a ranking answer does not require an
output spike, while spikes remain available for future conditional execution.

All 22 supervised roles (Q, death, quantile, geometry, and the existing teacher
vector) are retained. ListNet is computed only after all candidates of the same
state are available; candidate-common additive score terms cancel, but
candidate-by-base interactions remain available through the shared base state
and typed delta contacts.

### 4.7 Canonical forward call chain

```text
train_update!(trainer, state_batch)

for each state:
    encode_state_common_rails!
    prepare_base_state!
        materialize the collision-free 16 x 480 base packet grid
        deposit all base source packets into 48 typed relation inboxes
        advance all 48 base relation cells and emit 16D packets
        deposit direct source + relation + common-context packets
        prepare the immutable base output snapshot

    parallel for each candidate:
        encode_candidate_delta!                 # no teacher access
        rematerialize changed after packets only
        form exact candidate-minus-base 16D deltas
        collect affected relation IDs
        fork affected relation COW overlays
        deposit signed deltas through opponent-paired deposit_sources!
            into a fresh nonnegative delta-only typed inbox
        advance affected relations once from their common base states
        fork the candidate output overlay
        deposit mandatory direct source deltas through opponent pairs
        form/deposit signed relation-packet deltas through opponent pairs
        deposit sparse raw-placement/action contacts
        advance 22 typed output cells once from common base states
        emit the continuous 22D candidate result
```

There is exactly one canonical call chain. No route flag, forest fallback,
compact-message fallback, dense reducer, or historical checkpoint shim selects
another information path.

### 4.8 Exact conditional reverse call chain

```text
all candidate forwards
-> ListNet / supervised 22D cotangents
-> parallel candidate output pullback
     output state -> relation packet deltas
                  -> direct source packet deltas
                  -> action/context contacts
-> affected relation-cell pullback
     relation state -> changed source fields
                    -> shared base-state cotangents
-> changed spatial-program pullback
-> fixed per-worker gradient reduction
-> reduce shared base cotangents once per state
-> base output pullback
-> base relation pullback
-> base spatial-program pullback
-> optimizer step
```

Continuous one-step compartment transitions, the 16D packet map, and typed
contact deposition use analytic VJPs conditional on the recorded hard event.
The canonical task VJP does not differentiate that hard bit; any separately
named surrogate/event cotangent belongs to a later control learner. Only
visited COW overlays are taped. Shared base computation is reversed once per
state, not once per candidate.

DECOLLE/e-prop remains a later CPU-training optimization. It may be promoted
only after same-batch shadow comparison shows useful direction/norm alignment
without erasing the exact model's quality. It is not allowed to mask an
architecture-capacity failure.

### 4.9 CPU active-work contract

Stored dormant capacity may grow without increasing the per-candidate frontier.
The first canonical implementation must satisfy:

```text
average changed after-plane sources       <= 20
average affected relation cells           <= 24
p95 affected relation cells               <= 40

average relation/output cell advances     <= 66 per candidate
p95 relation/output cell advances         <= 128 per candidate
total cell advances                        <= 3,500 per state

average logical contact deliveries        <= 2,048 per candidate
p95 logical contact deliveries            <= 4,096 per candidate

hot allocation                            = 0 bytes
hot GC                                    = 0 seconds
training throughput                       >= 20 updates/s
```

Counts include the mandatory direct source-to-output jump. A larger program
bank, more dormant contacts, or more inactive parameter storage may not relax
these active-work bounds. A GPU can execute the same graph; the hypothesis is
only that small-batch, branchy COW frontiers and cache-line packets favor CPU
MIMD execution enough to improve quality per wall-clock.

## 5. Removed canonical paths

The following are historical controls only and are not reachable from the
HD-CDRG entrypoint:

- `DendriticDeltaForest` and the distinct before/after crossed trees;
- row-half, column-group, ancestor, anchor, or root reduction;
- the two-analog-plus-hard-event compact message;
- signed aggregate drive that switches between excitatory and inhibitory
  receptors;
- global E/I calibration constants such as `EXCITATORY_DRIVE_SCALE`;
- auxiliary-to-Q root cascade or a global dense observation head;
- one-scalar apical context compression;
- central workspace routing, query/key, or Plackett--Luce top-k;
- a fixed global recurrent clock or per-candidate all-cell integration;
- spike-only recurrent communication;
- hard IEEE-754 result bits or bit BCE as the Q representation;
- a full Hay/digital-twin runtime at every graph site;
- a frozen 11-state distilled core as the final Tetris model;
- treating local-credit repair as a substitute for architecture validation;
- beginning another 100k run before the gates below pass;
- claiming GPU impossibility, TwinProp reproduction, or biological fidelity;
- compatibility flags, aliases, or new versioned directories that can select a
  removed forest/route/compact path.

Point-SNN, frozen distilled cells, the stopped Reduced Hay runs, DDF, and
Float32 numeric-core experiments remain research history and comparison
artifacts. They are not called from the canonical entrypoint.

## 6. Implementation order and falsification gates

Only this critical path is authorized. A failed gate stops scale-up and
requires a model-level decision; it does not trigger another isolated gain or
learning-rate sweep.

### Gate A: program packets and typed-contact causality

Pass only if:

- the base grid is exactly `16 x 480`, each source activates exactly four
  collision-free rows, and every bank row `1:208,448` is reachable;
- candidate-after packet plus its recorded base delta reconstructs the full
  candidate packet exactly;
- equal E/I coactivation produces a different destination state from silence;
- equal net signed drive with different total conductance produces different
  voltage/time-constant evolution;
- receptor identity is unchanged by optimizer updates and every physical
  conductance remains nonnegative;
- positive/negative packet deltas traverse same-receptor opponent pairs in
  separate compartments through selected-source deposition; canonical
  candidate execution never subtracts a receptor inbox;
- AMPA, NMDA, and GABA delivery counts are all nonzero;
- program-packet, cell-packet, and typed-contact finite differences match their
  analytic continuous VJPs.

### Gate B: sparse base/COW equivalence and CPU contract

Compare the sparse candidate execution with a full-overlay oracle using the
same signed source deltas, opponent-paired contacts, relation topology, common
base cell states, cell parameters, and output bank.

Pass only if:

- forward outputs match within `1e-6`;
- continuous parameter gradients match within `1e-5`;
- serial and barrierless outputs, gradients, and actual updates match;
- the relation closure contains exactly the 24 row + 10 column + 14 cross/action
  IDs implied by changed sources;
- the mandatory direct source-to-output jump is present in both paths;
- the active-work limits in Section 4.9 pass, including at least
  `20 updates/s`, hot allocation `0`, and hot GC `0`;
- no teacher target or dataset row is read while constructing base state or
  candidate overlays.

### Gate C: 64-state exact-capacity gate

Train HD-CDRG from scratch with the exact conditional reverse. The old DDF
result is failure evidence, not a target to tune around.

Pass only if:

```text
tie-aware top-1              = 1.0
excess loss                  < 0.05
tail tie-aware top-1         = 1.0
tail excess does not rebound
```

Additionally:

- program bank, all three typed receptor groups, all three relation classes,
  relation-cell parameters, and output-cell parameters all change;
- direct source-to-output ablation worsens capacity;
- relation-path ablation worsens capacity;
- the candidate-to-Q input Jacobian is not dominated by one singular direction;
- hard events and plateau events are nonzero but nonsaturated;
- no hidden tree, global reducer, route, or dense fallback is invoked.

Failure stops full-data training. It does not authorize another 64-state gain
sweep or removal of the direct path.

### Gate D: full-data 1k quality/CPU gate

Use the immutable width-80 train/development manifest and train from scratch.
The retained compact control is:

```text
development excess loss   = 0.489725
development tie top-1     = 0.375000
```

Pass only if HD-CDRG achieves:

```text
development excess loss   < 0.489725
development tie top-1     > 0.375000
training throughput       >= 20 updates/s
```

It must also have better quality-vs-time AUC than the compact control, continue
improving over the final 200 updates, avoid hard-event/plateau/E-I collapse,
and satisfy every Section 4.9 active-work bound. The checkpoint, exact source
fingerprint, progress log, and execution counters are retained.

Only after Gate D may the same canonical model proceed to a 10k equal-wall-clock
comparison. A 100k run remains forbidden until the 10k curve justifies it.

## 7. PreAct/DSRLN comparison contract

An earlier SNN report copied the PreAct `top-1=0.789063` validation result next
to a different training-only panel.  That is not a same-panel comparison.  The
reproducible historical held panel is:

```text
validation subset seed 2026072315
128 states
row hash fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25
width 80, original candidate order and multiplicity
```

On that panel the currently reproducible controls are:

| control | parameters | teacher states | loss | top-1 | NDCG | pairwise | train states/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PreAct checkpoint | 1,481,326 | 48,000 | 2.563784 | 0.789063 | 0.993292 | 0.923359 | 1.97 |
| DSRLN checkpoint | 20,577,789 | 400,000 | 2.607862 | 0.804688 | 0.990137 | 0.898083 | 124.90 |

The stronger historical PreAct log reached loss `2.550905`, NDCG `0.994468`,
and pairwise `0.930203`, but its exact checkpoint has been overwritten.  It is
therefore a historical envelope, not a rerunnable binary.  Conversely, the
DSRLN `top-1=0.804688` is a single 100k checkpoint spike; its 80k--100k
five-point mean is about `0.7453`.  Neither fact licenses weakening the new
model's gate.

DSRLN's relevant architecture fact is that `20,447,232` parameters (`99.37%`)
live in its Lookup bank while only `14,976` bank parameters (`0.0732%`) are
touched per macro-step. HD-CDRG must improve on this stored-capacity/active-
compute separation, not merely exceed the old 6.8M Pareto variant.

Final comparison must freshly evaluate all models under one immutable
manifest:

- identical train/validation/test row IDs and candidate lists;
- identical teacher targets and tie-aware ranking definitions;
- identical input information; deterministic delta transforms are allowed,
  teacher-derived inputs are not;
- both equal teacher-state exposures and equal wall-clock budgets;
- same CPU affinity, worker count policy, BLAS thread count, and warm-up;
- three seeds, with checkpoints and per-1k progress retained;
- excess loss, tie-aware top-1, NDCG, pairwise, quality-vs-time AUC,
  candidates/s, states/s, peak resident memory, hot allocation/GC, and CPU
  utilization.

"PreAct/DSRLNを超えた" is permitted only when the new model:

1. exceeds the freshly evaluated best baseline on the common held panel and
   test split, with a paired confidence interval reported;
2. has better equal-wall-clock quality AUC;
3. preserves the advantage across all three seeds;
4. demonstrates that increasing dormant capacity does not proportionally
   increase active compute.

The provisional historical envelope is deliberately strict:

```text
loss      < 2.550905
top-1     > 0.804688
NDCG      > 0.994468
pairwise  > 0.930203
```

One state on the 128-state panel changes top-1 by `0.0078125`; the PreAct/DSRLN
top-1 gap is only two states.  A victory claim therefore additionally requires
a pre-frozen panel of at least 1,024 held states, state-paired confidence
intervals, and at least three training seeds.

Until those conditions hold, the accurate statement is:

> The current model failed the efficiency objective; HD-CDRG is a falsifiable
> CPU hypothesis intended to exploit candidate sharing, COW sparse work,
> 16D branch-preserving communication from high-dimensional relation/output
> cells, and CPU-resident dormant program capacity. It is not a TwinProp
> reproduction and has not yet beaten PreAct or DSRLN.

## Primary research references

- [What can a neuron compute? The role of multi-contact synapses and dendritic nonlinearities (TwinProp)](https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full)
- [Dendritic nonlinearities mitigate communication costs](https://doi.org/10.1016/j.patter.2026.101520)
- [Single cortical neurons as deep artificial neural networks](https://doi.org/10.1016/j.neuron.2021.07.002)
- [Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning](https://doi.org/10.1038/s41467-025-56297-9)
- [Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics](https://doi.org/10.1038/s41467-023-44614-z)
- [AEGNN: Asynchronous Event-Based Graph Neural Networks](https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html)
- [InkStream: Incremental computation of dynamic graph embeddings](https://www.comp.nus.edu.sg/~tulika/IPDPS25.pdf)
- [Delta Networks for optimized recurrent network computation](https://proceedings.mlr.press/v70/neil17a.html)
- [SparseProp: Efficient event-based simulation and learning in sparse spiking neural networks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0b443d358a391166d1fbf551fb53de02-Abstract-Conference.html)
- [Large Memory Layers with Product Keys](https://proceedings.neurips.cc/paper/2019/hash/9d8df73a3cfbf3c5b47bc9b50f214aff-Abstract.html)
- [Event-based backpropagation can compute exact gradients for spiking neural networks](https://doi.org/10.1038/s41598-021-91786-z)
- [A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop)](https://doi.org/10.1038/s41467-020-17236-y)
- [Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)](https://doi.org/10.3389/fnins.2020.00424)
