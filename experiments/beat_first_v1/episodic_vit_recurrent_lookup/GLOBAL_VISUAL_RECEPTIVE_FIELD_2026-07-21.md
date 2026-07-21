# EVRL global visual receptive field — 2026-07-21

## Decision

The cell tokenizer now has a deliberately small convolutional visual path in
parallel with its original per-cell linear projection. A single local filter
was rejected because it would not let one output cell depend on the complete
`24 x 10` board. The accepted path is:

```text
raw board / candidate / difference (3 channels)
  -> depthwise 3x3, dilation 1  -> SiLU -> pointwise 3x3-channel mix + residual
  -> depthwise 3x3, dilation 2  -> SiLU -> pointwise 3x3-channel mix + residual
  -> depthwise 3x3, dilation 4  -> SiLU -> pointwise 3x3-channel mix + residual
  -> depthwise 3x3, dilation 8  -> SiLU -> pointwise 3x3-channel mix + residual
  -> depthwise 3x3, dilation 16 -> SiLU -> pointwise 3x3-channel mix + residual
  -> pointwise 3 -> 128 projection
  -> learned residual gate into the existing cell token
```

The theoretical receptive field is

```text
1 + 2 * (1 + 2 + 4 + 8 + 16) = 63
```

in both axes. This covers the complete `24 x 10` board. A direct perturbation
test confirmed a non-zero path from the top-left board cell to the bottom-right
token (`L2 = 2.1776226e-7` at initialization).

The existing recurrent learned local-8 Q/K/V/O cell attention remains in
place. Thus the visual stack supplies cheap hierarchical spatial features,
while recurrent attention can still form input-dependent relations. The
283-token register cross-attention, LookupFFN parameter routing, active-only
bank updates, loss, optimizer semantics, and fixed depth-two comparison
setting are unchanged.

## Cost

| Field | Value |
|---|---:|
| Base full-token EVRL parameters | 20,577,224 |
| Visual parameters added | 565 |
| Total parameters | 20,577,789 |
| Visual scalar MAC/candidate | 135,360 |
| Receptive field | 63 x 63 |
| Dilation schedule | 1, 2, 4, 8, 16 |

There is no per-stage gate. Multiplying five initially small stage gates made
the corner-to-corner signal vanish in Float32. Instead, each stage is an
identity residual and only the final visual-to-token residual is gated.

## Gradient and executor correctness

Selected finite-difference checks passed:

| Parameter | Relative error |
|---|---:|
| Depthwise stage 1 | 5.87e-5 |
| Depthwise stage 5 | 4.89e-5 |
| Channel mix stage 3 | 2.04e-4 |
| Final pointwise projection | 2.99e-6 |
| Output residual scale | 1.68e-5 |

The production serial/barrierless smoke compared four real-teacher training
states, 16 candidates per state, and all post-update state. It used 20 Julia
workers, no pinning, and chunk 8.

| Witness | Maximum absolute difference | Relative L2 |
|---|---:|---:|
| Output | 0 | 0 |
| Loss | 0 | 0 |
| Raw VJP | 0 | 0 |
| Worker gradient | 5.61774e-6 | 1.60486e-6 |
| Reduced parameter gradient | 1.72108e-6 | 1.60504e-6 |
| Post-optimizer parameter/state | 2.08616e-7 | 2.13909e-9 |

Candidate RNG, forced and realized depth, hard-halting decisions, token edges,
Lookup row IDs, active masks, sparse events, route usage, sampler state,
optimizer clocks, and post-update RNG state were exact. The smoke checkpoint
SHA-256 was
`f00a0cb84ea1bd9eaf6dfb372912830022eda3f8f97075bfec4eaf9ab2d44fcc`.

## Speed

A fresh 10-warmup + 100-measured-update real-teacher benchmark produced:

```text
31.5813 updates/s
126.325 states/s
5,540.62 candidates/s
74.09% whole-process CPU
76.98% candidate CPU
```

The 12,000-update production segment completed its update work in `389.788 s`
at `30.7859 updates/s`. The resumed 12,000 -> 25,000 segment completed its
update work in `506.448 s` at `25.6690 updates/s`. Both exceed the required
15 updates/s floor.

Held-panel inference at update 25,000 was `43.7214 states/s`, 3.22% below the
previous no-visual full-token checkpoint's `45.1745 states/s`, and 10.428x the
PreAct measurement on the same panel.

## Learning result

The model consumed exactly 100,000 real-teacher states at update 25,000. The
same fixed 128-state panel was reconstructed from dataset manifest
`1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`
with row-list SHA-256
`fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`.
Teacher Q and rank were targets only and candidates were evaluated
independently.

| Model | Updates / states | Top-1 | NDCG | Pairwise | Margin | Loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PreAct | 12k / 48k | 0.78906 | 0.99329 | 0.92336 | 0.12332 | 2.56378 | 4.1925 |
| Full-token EVRL, no visual stack | 12k / 48k | 0.56250 | 0.97867 | 0.84152 | 0.07205 | 2.80916 | 45.1745 |
| Global-visual EVRL | 12k / 48k | 0.50000 | 0.97936 | 0.85128 | 0.07982 | 2.81943 | not remeasured |
| Global-visual EVRL | 25k / 100k | 0.68750 | 0.98586 | 0.87751 | 0.13215 | 2.66337 | 43.7214 |

At the equal 12k budget, the visual model is a mixed result: top-1 is lower by
`0.0625`, while NDCG, pairwise accuracy, and margin are slightly higher. It
would be incorrect to attribute the 25k improvement solely to the visual
architecture because the no-visual model was not trained to the same 25k
budget.

The important convergence result is that 12k was not a plateau. From 12k to
25k, the same visual model improved top-1 by `0.1875`, NDCG by `0.006501`,
pairwise accuracy by `0.026229`, margin by `0.052324`, and loss by `0.156063`.
At 25k it still trails PreAct by `0.1015625` top-1 and `0.007428` NDCG, while
exceeding PreAct's action margin by `0.008825` and remaining 10.428x faster in
this CPU inference measurement.

Final checkpoint:

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_visual_dw5_rf63_fixed2_u25000_20260721_r1\checkpoints\checkpoint_000025000.jls
bytes: 253673177
sha256: a571db8dbb8c865a0c05a1695f58e7d9cd5db9b78475bcc759d196168c016b6e
```

The binary checkpoint and teacher dataset are not committed. Machine-readable
evaluation evidence is in
[`visual_receptive_field_evaluation_2026-07-21.json`](visual_receptive_field_evaluation_2026-07-21.json).

## Subsequent dynamic-recurrence study

The fixed-depth model was later continued to update 80,000 and used for a
controlled hard-halting activation study. Direct activation, a five-thousand
update random-depth curriculum, halt-head reset, and dynamic training from
scratch were all tested. Every final held-panel deterministic trajectory still
stopped at depth two. The result isolates candidate-specific halting credit as
the next required change; it does not invalidate the visual, episodic-memory,
or LookupFFN body. See
[`DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md`](DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md).
