# High-Dimensional Dendritic Workspace SNN

`dendritic_workspace_snn` is an independent experimental successor to
`serial_workspace_snn`. It keeps the point-neuron SNN unchanged as the control
and replaces its scalar LIF nodes with reduced, active multi-compartment cells.

The directory now contains both the differentiable capability reference and
the fixed-memory production trainer used for real teacher-v3 Tetris learning.
The point-neuron SNN remains unchanged as the control model.

## Why this is more than branch routing

The model implements three distinct ingredients:

1. a fixed synapse-to-basal-branch placement,
2. persistent branch voltage and slow plateau state,
3. branch-local voltage-dependent recruitment followed by soma integration.

This distinction follows the mechanism isolated in the 2026 TwinProp
preprint: dendritic morphology alone was insufficient; distributed voltage
dynamics, NMDA-like nonlinearity, and voltage-dependent conductances carried
the additional computation. TwinProp itself is an optimization instrument,
not a biological local learning rule, and is not copied here.

Primary paper:
[Beniaguev et al., “What can a neuron compute?”](https://www.biorxiv.org/content/10.64898/2026.06.08.730984v1.full)

## Cell state

For four basal branches, one cell owns 11 persistent scalars:

```text
branch voltage × 4
plateau state × 4
apical/context state
soma voltage
adaptation
```

The branch update is:

```text
u_b(t+1) =
    branch_leak_b * u_b(t)
  + excitatory_b(t) - inhibitory_b(t)
  + plateau_feedback_b * p_b(t)

coincidence_b =
  hard_sigmoid(
    plateau_slope_b * (u_b(t+1) - plateau_threshold_b)
  )

p_b(t+1) =
    plateau_decay_b * p_b(t)
  + plateau_gain_b
  * max(excitatory_b(t) - inhibitory_b(t), 0)
  * coincidence_b
```

The soma integrates all branch voltages and plateaus, modulated by a persistent
apical/context state. A soma event subtracts one threshold unit from the soma;
it does not reset the branch or plateau states.

## Analog information plane and event control plane

Each cell exports six analog values:

```text
tanh(soma)
tanh(apical)
tanh(branch_voltage_1..4)
```

The soma spike remains a one-bit graph communication and execution-control
event. The continuous compartment state is not compressed into that bit.

The scaled v1 topology is:

```text
96 blocks
× 8 dendritic cells per block
× 6 analog exports per cell
= 48-dimensional block interface

768 cells total
4 basal branches per cell
11 persistent states per cell
48 candidate outgoing edges per cell
36,864 candidate edges
```

This preserves the point SNN's 48-dimensional workspace/head interface while
replacing 4,608 scalar nodes with 768 high-dimensional cells.

## Production local learning

`DendriticCellKernel.jl` and `DendriticArenaTraining.jl` use a factorized,
forward-only local eligibility:

```text
edge
  -> branch-voltage eligibility
  -> plateau eligibility
  -> soma eligibility
```

This uses three scalars per edge rather than a full 11-by-11 Jacobian. A block
learning signal can therefore produce:

```text
parameter update = block signal * soma eligibility
utility increment = abs(block signal * soma eligibility)
```

The recurrent learner has no parameter-tree input while it constructs its
third factor.  It therefore cannot read `head_weight` or `output_weight`.
Each block uses a different seed-fixed, non-trainable 22-to-48 projection and
predicts Q, death, quantiles, and geometry only from its own six-dimensional
cell readouts and spikes.

The ListNet update remains a fixed-memory two-pass replay:

```text
all candidate forwards
  -> ListNet/raw cotangent
  -> block-local signal and supervised reward surrogate
  -> candidate-centered routing advantage
  -> forward eligibility replay
```

The four supervised head groups use the exact analytic VJP.  The 23 recurrent,
compartment, routing, and graph groups use DECOLLE-style local signals and
e-prop traces.  The routing reward is explicitly a candidate-centered
`supervised_reward_surrogate`; it is not an environment return.

Training uses stochastic ordered Plackett-Luce top-k, exact ordered routing
eligibility, and entropy/load regularization.  Inference keeps deterministic
hard top-k.

## Structural learning

ON/OFF state is a fixed-fanout `BitMatrix`, independent of the continuous gate
logit.  AdamW cannot accidentally change the fanout by moving a logit through
zero.  Only utility consolidation can perform one ON/OFF swap per source:

```text
ON utility  = |weight update| + |gate update| + |delay update|
OFF utility = max(-gate loss-gradient, 0)
```

The second expression is the counterfactual evidence that turning an OFF edge
on would lower the supervised loss.  Consolidation retains the connection
cost, resets moments only for swapped edges, and preserves exactly 24 of 48
ON edges for every source.

Branch placement is also learned from block-signal-by-eligibility utility.
At most one edge per source changes branch in each branch-consolidation pass,
and only the moved edge has its moments reset.

## CPU representation

The scalar cell type is used for capability and finite-difference tests. CPU
execution uses `DendriticCellArena`, a structure-of-arrays layout:

```text
branch_voltage[cell, branch]
plateau[cell, branch]
apical[cell]
soma[cell]
adaptation[cell]
spike[cell]
```

Cells are the contiguous matrix dimension, so each branch compartment across
the population is SIMD-friendly. The population sweep and eligibility update
allocate zero bytes after compilation.

Run the kernel benchmark:

```powershell
julia --project=. dendritic_workspace_snn/benchmark_cell_kernel.jl
```

The benchmark compares one sweep of 768 four-branch cells with one sweep of
4,608 point LIF nodes. It measures only neuron-state transitions, not graph
delivery, workspace routing, backward, or end-to-end training.

Measured on the development CPU with Julia 1.12.6, 2,000 sweeps per sample:

| kernel | median | hot allocation |
|---|---:|---:|
| 4,608 point LIF nodes | 0.001309 s | 0 bytes |
| 768 cells × 4 branches | 0.001630 s | 0 bytes |

The reduced dendritic population kernel was 1.2453 times the point-neuron
state-transition wall time in this narrow benchmark.

The differentiable scaled reference has 240,311 parameters. A warmed
single-candidate forward measured 0.002694 seconds but allocated 7,873,347
bytes. It remains the capability/reference implementation.

`DendriticArenaTraining.jl` is the production path. It uses:

- fixed candidate and trajectory arenas,
- source-major fired-event delivery,
- active-edge-only eligibility replay,
- persistent barrierless MPMC workers,
- parallel in-place AdamW,
- allocation-free utility and branch consolidation,
- atomic JLD2 checkpoints with optimizer, sampler, utility, mask, and branch
  state.

On the development CPU, a scaled one-state warmed update was about 0.021
seconds with 0 hot bytes and 0 GC. Batch-8, 20-worker training sustained about
120 to 130 states/s after warm-up.

## Verification

Run:

```powershell
julia --project=. dendritic_workspace_snn/runtests.jl
```

The tests cover:

- active-dendrite XOR and passive-branch ablation,
- multi-branch plateau-state rank,
- persistent voltage/plateau state after a soma event,
- scalar/SoA arena numerical equivalence,
- zero allocation in scalar, arena, and eligibility hot kernels,
- zero third-factor stopping local updates,
- 0/1 sensory rails and the canonical 22-output contract,
- 11-state and 48-dimensional topology invariants,
- serial versus vectorized branch-specific edge delivery,
- nonzero gradients for compartment, graph, routing, and head parameters,
- finite-difference agreement,
- active-plateau output ablation,
- serial trace versus vectorized candidate output.

Current test result:

```text
new dendritic tests: 90 / 90 pass
production arena integration: 24 / 24 pass
existing point-SNN tests: 28 / 28 pass
```

Run real scratch training:

```powershell
julia --project=. --threads=20,0 `
  dendritic_workspace_snn/train_dendritic_10k.jl `
  --updates 10000 --state-batch 8 --workers 20
```

The driver writes a per-update TSV trace, SHA-256 checkpoint manifest, ten
periodic checkpoints by default, and `results.json`. Resume restores the exact
optimizer powers/moments, epoch sampler, structural utilities, gate mask, and
branch placement.

## Full scratch 10k result

The final mask-aware run trained 10,000 updates / 80,000 teacher states:

```text
initial single-batch loss  6.644819
final single-batch loss    4.060909
first 100 mean loss        5.91
last 100 mean loss         3.80
updates/s                  about 14.5
hot allocation / GC        0 bytes / 0 seconds
```

All four local predictor losses fell from the first to the last 100-update
window. The run performed real ON/OFF and branch structural updates while
retaining 24/48 ON edges for every source. Routing entropy did not collapse to
zero, but its last-100 mean was about 0.63, below the configured 0.70 floor;
that is a measured remaining tuning issue, not hidden as a completed quality
claim.

## Remaining research boundaries

The implementation is complete for the fixed-four-cycle dendritic SNN
training path. The following are separate research/evaluation work:

- held-out/sealed Tetris quality comparison against point-SNN, DSRLN, and
  PreAct,
- equal-wall-clock ablations for plateau, apical state, and branch relocation,
- stronger long-run routing-entropy control,
- candidate-specific dynamic cycle halting,
- longer 100k training after the 10k architecture result is reviewed.
