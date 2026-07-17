# Architecture screen for the 24×10 Tetris value model

Status: finite forward/backward systems screen completed.  Game-score promotion
is still pending end-to-end training; none of the numbers below is a strength
claim.

## Scope and frozen interface

This screen changes only the value network.  Every candidate consumes the same
fixed-shape inputs: board plus candidate placement `(24, 10, 2, batch)`, six
HOLD/NEXT one-hot tokens `(7, 6, batch)`, and three scalar features
`(REN/30, B2B, T-spin)`.  It emits one scalar Q value per candidate.  The
historical 20,787,454-parameter checkpoint remains untouched.

The screen uses Julia 1.12.6, Lux 1.31.4 and Zygote 0.7.11 from the root locked
environment.  Scores are not used at this stage: only candidates which pass
this systems screen should receive scarce end-to-end training time.

## Registered experiments (before execution)

### A1 — pre-activation ResNet + SE

- Hypothesis: a six-block, 64-channel pre-activation residual tower retains the
  useful locality of the old network while SE supplies cheap global channel
  context; it should remove over 90% of the old parameters without losing the
  representational structure that already worked.
- Expected mechanism: identity-path optimization, small-batch GroupNorm, and
  channel gating.
- Change: `PreActSEValueNet`, with a separate light queue encoder joined at the
  value head.
- Success: finite forward/backward, at most 2.5M parameters, at least 2× the
  estimated candidate throughput of the old Lux CPU checkpoint.
- Time limit: 20 minutes including compilation and two batch sizes.
- Stop: numerical error, more than 2.5M parameters, or steady-state backward
  slower than the historical-scale tower.

### A2 — ConvNeXt / depthwise inverted bottleneck

- Hypothesis: on a tiny 24×10 grid, depthwise 5×5 spatial mixing followed by
  pointwise expansion provides the best CPU/OpenVINO speed per receptive field.
- Expected mechanism: lower MAC count and standard grouped-convolution ops.
- Change: `ConvNeXtValueNet`, eight 64-channel depthwise-inverted blocks, light
  queue encoder, shared value head.
- Success: finite forward/backward, lowest MAC count, and at least 1.25× the
  backward throughput of A1 without a parameter increase.
- Time limit: 20 minutes.
- Stop: grouped convolution is slower than A1 by over 20% on this CPU, or AD is
  unstable.

### A3 — board CNN + NEXT encoder + FiLM

- Hypothesis: HOLD/NEXT affects the value of board patterns throughout feature
  extraction, so feature-wise affine conditioning at each residual block should
  be more sample-efficient than concatenating queue features only at the head.
- Expected mechanism: queue-conditioned channel scaling and shifting at every
  board block, preserving convolutional spatial priors.
- Change: `FiLMValueNet`, six 64-channel residual blocks conditioned by a
  shared queue embedding.
- Success: finite forward/backward, at most 2.5M parameters, and no more than a
  30% throughput penalty versus the fastest candidate.  This candidate is
  preferred for training if it meets that systems budget because it tests the
  strongest Tetris-specific inductive bias.
- Time limit: 25 minutes.
- Stop: the conditioning path dominates allocations, or throughput falls below
  70% of the fastest candidate.

## Common benchmark protocol

- Random but fixed seed 0x1313, identical inputs and MSE target.
- The registered plan requested batches 16 and 64.  The finite batch-16 screen
  already showed 21–29 second reverse passes for the 64-channel models, so the
  batch-64 pass was stopped under the time budget and actual student training
  was prioritized.
- 12 BLAS threads, no concurrent architecture benchmark.
- Record parameters, analytical multiply-accumulate count (MAC), analytical
  major activation footprint, median latency, throughput and allocation.
- OpenVINO feasibility is audited from the actual operation set; conversion is
  not counted as successful until a later numerical-equivalence test.

## Evidence basis

- ConvNeXt re-examines pure ConvNet design and uses depthwise spatial mixing plus
  inverted bottlenecks: <https://arxiv.org/abs/2201.03545> and the
  [official implementation](https://github.com/facebookresearch/ConvNeXt).
- ECA shows that channel attention can add negligible parameter/MAC overhead;
  this screen uses the more readily exportable SE gate as the conservative
  channel-attention representative: <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html>.
- FiLM defines conditioning as a feature-wise affine transform, matching the
  desired interaction between the six-piece queue and board features:
  <https://aaai.org/papers/11671-film-visual-reasoning-with-a-general-conditioning-layer/>.
- OpenVINO's official supported-operation table includes convolution, grouped /
  depthwise convolution, normalization, pooling, reshape, concatenation and
  elementwise operations used here:
  <https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-operations.html>.

## Measurements

The machine was Julia 1.12.6 / Lux 1.31.4 / Zygote 0.7.11.  All candidates
used Float32 and batch 16.  `activation MiB` is the analytical sum of major
forward activations, not process RSS and not the larger set of tensors retained
by reverse mode.  MAC is per candidate.

| candidate | params | MAC/candidate | major activation MiB (b16) | first forward / backward | steady forward | steady loss+backward | finite |
|---|---:|---:|---:|---:|---:|---:|:---:|
| PreAct-SE 64×6 | 756,789 | 106.819 M | 17.92 | 1.763 s / 88.548 s | 77.580 ms (206.2 cand/s) | 29.049 s | yes |
| ConvNeXt 64×8 | 584,837 | 66.631 M | 46.04 | 1.452 s / 76.250 s | 148.239 ms (107.9 cand/s) | 25.254 s | yes |
| NEXT-FiLM 64×6 | 800,133 | 106.862 M | 17.92 | 0.478 s / 49.694 s | 89.468 ms (178.8 cand/s) | 21.161 s | yes |

The exact machine-readable records are in
`experiments/architecture/benchmark_results.json`.

### Contamination and limits

- No other Julia process was observed when the PreAct-SE run started.
- The ConvNeXt and FiLM run overlapped the 20-thread teacher-dataset generator
  (`experiments/learning/generate_teacher_dataset.jl`, PID 6544).  Their latency
  and throughput are therefore **contaminated reference values**, not clean
  host benchmarks.  They are useful only as a same-run directional comparison.
- Parameter counts, analytical MACs/activations, and finite forward/backward
  results are unaffected by this contention.
- OpenVINO execution was not measured.  The candidates intentionally use
  standard convolution/grouped convolution, normalization, pooling, dense,
  reshape and elementwise operators, but operator support alone is not a
  latency result.  Export plus numerical-equivalence testing remains mandatory
  before any OpenVINO speed claim.
- No game score was measured in this systems screen, and no held-out test seed
  was used.

The absolute 64-channel reverse-mode times are too slow for the shortest local
time-to-target.  That is itself an actionable result: the first student should
use the compact channel/depth regime and only scale after a score signal.

## Decision

**Promote the NEXT-conditioned FiLM architecture, in compact form, as the one
architecture candidate for end-to-end teacher distillation.**  It preserves
the board×queue interaction that a head-only queue concatenation cannot learn,
stays under 1 M parameters at 64×6, and was the fastest loss+backward candidate
within the two candidates measured under identical contention.  The first
training configuration should reduce channels/depth (for example 16–32
channels and 2–4 blocks), then scale only if development policy agreement or
game score is capacity-limited.

ConvNeXt is rejected for the present Windows CPU path.  Its analytical MAC
count was 37.6% lower than FiLM, yet its grouped/depthwise implementation had
65.7% worse forward latency in the same contaminated run and retained much
larger expansion activations.  Its theoretical efficiency did not translate
to this host.

PreAct-SE remains a compatibility fallback, not the first training candidate.
It has a sound residual/attention bias but no NEXT interaction inside the board
tower, and the clean batch-16 backward took 29.0 seconds for the 64×6 version.

Final promotion still requires the same teacher data, update budget and
development seeds against the compact unconditioned baseline.  Fixed-search
game score, rather than loss alone, decides whether FiLM survives.
