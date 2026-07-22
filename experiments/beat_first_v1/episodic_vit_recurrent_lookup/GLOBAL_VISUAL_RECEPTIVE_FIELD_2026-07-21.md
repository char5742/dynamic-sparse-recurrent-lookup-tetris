# EVRLの全盤面視覚受容野 — 2026-07-21

## 判断

cell tokenizerへ、従来のcell単位linear projectionと並列する小型convolutional visual pathを追加した。単一local filterでは1つのoutput cellが`24 x 10`盤面全体に依存できないため不採用とし、次の経路を採用した。

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

理論上の受容野は両軸とも次のとおりで、`24 x 10`盤面全体を覆う。

```text
1 + 2 * (1 + 2 + 4 + 8 + 16) = 63
```

直接perturbation testでも、左上board cellから右下tokenへの非zero経路を確認した（初期化時`L2 = 2.1776226e-7`）。

既存のrecurrent learned local-8 Q/K/V/O cell attentionは維持した。visual stackが安価な階層的空間特徴を供給し、recurrent attentionは引き続き入力依存relationを形成できる。283-token register cross-attention、LookupFFN parameter routing、active-only bank update、loss、optimizer semantics、固定depth 2の比較条件は変更していない。

## 計算量

| 項目 | 値 |
|---|---:|
| Base full-token EVRL parameters | 20,577,224 |
| 追加visual parameters | 565 |
| Total parameters | 20,577,789 |
| Visual scalar MAC/candidate | 135,360 |
| Receptive field | 63 x 63 |
| Dilation schedule | 1, 2, 4, 8, 16 |

stageごとのgateは置いていない。初期値が小さい5つのstage gateを乗算すると、corner-to-corner信号がFloat32で消失したためである。各stageはidentity residualとし、最後のvisual-to-token residualだけをgateした。

## Gradientとexecutorの正当性

選択したfinite-difference checkはすべて合格した。

| Parameter | Relative error |
|---|---:|
| Depthwise stage 1 | 5.87e-5 |
| Depthwise stage 5 | 4.89e-5 |
| Channel mix stage 3 | 2.04e-4 |
| Final pointwise projection | 2.99e-6 |
| Output residual scale | 1.68e-5 |

production serial/barrierless smokeでは4件のreal-teacher training states、各16 candidates、update後state全体を比較した。Julia 20 workers、pinningなし、chunk 8で実行した。

| 確認項目 | 最大絶対差 | Relative L2 |
|---|---:|---:|
| Output | 0 | 0 |
| Loss | 0 | 0 |
| Raw VJP | 0 | 0 |
| Worker gradient | 5.61774e-6 | 1.60486e-6 |
| Reduced parameter gradient | 1.72108e-6 | 1.60504e-6 |
| Optimizer後parameter/state | 2.08616e-7 | 2.13909e-9 |

candidate RNG、forced／realized depth、hard-halting判定、token edge、Lookup row ID、active mask、sparse event、route usage、sampler state、optimizer clock、update後RNG stateは完全一致した。smoke checkpoint SHA-256は`f00a0cb84ea1bd9eaf6dfb372912830022eda3f8f97075bfec4eaf9ab2d44fcc`。

## 速度

新規に10 warmup＋100 measured-updateのreal-teacher benchmarkを実施した。

```text
31.5813 updates/s
126.325 states/s
5,540.62 candidates/s
74.09% whole-process CPU
76.98% candidate CPU
```

12,000-update production segmentは`389.788 s`、`30.7859 updates/s`で完了した。12,000から25,000へのresume segmentは`506.448 s`、`25.6690 updates/s`だった。どちらも要求下限15 updates/sを上回る。

update 25,000のheld-panel inferenceは`43.7214 states/s`で、直前のno-visual full-token checkpointの`45.1745 states/s`より3.22%低く、同じpanelのPreActより`10.428x`高速だった。

## 学習結果

update 25,000で正確に100,000 real-teacher statesを消費した。同じ固定128-state panelをdataset manifest `1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`、row-list SHA-256 `fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`から再構築した。teacher Qと順位は教師信号としてのみ使い、candidateは独立に評価した。

| モデル | Updates / states | Top-1 | NDCG | Pairwise | Margin | Loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PreAct | 12k / 48k | 0.78906 | 0.99329 | 0.92336 | 0.12332 | 2.56378 | 4.1925 |
| Full-token EVRL, no visual stack | 12k / 48k | 0.56250 | 0.97867 | 0.84152 | 0.07205 | 2.80916 | 45.1745 |
| Global-visual EVRL | 12k / 48k | 0.50000 | 0.97936 | 0.85128 | 0.07982 | 2.81943 | 未再測定 |
| Global-visual EVRL | 25k / 100k | 0.68750 | 0.98586 | 0.87751 | 0.13215 | 2.66337 | 43.7214 |

同一12k予算ではmixed resultで、visual modelのtop-1は`0.0625`低い一方、NDCG、pairwise accuracy、marginはわずかに高い。no-visual modelを同じ25kまで学習していないため、25kでの改善全体をvisual architectureだけの効果としてはならない。

重要な収束上の結果は、12kがplateauではなかったことである。同じvisual modelは12kから25kでtop-1 `+0.1875`、NDCG `+0.006501`、pairwise `+0.026229`、margin `+0.052324`、loss `-0.156063`を達成した。25k時点でもPreActよりtop-1 `0.1015625`、NDCG `0.007428`低いが、marginは`0.008825`高く、CPU推論は`10.428x`高速だった。

最終checkpoint:

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_visual_dw5_rf63_fixed2_u25000_20260721_r1\checkpoints\checkpoint_000025000.jls
bytes: 253673177
sha256: a571db8dbb8c865a0c05a1695f58e7d9cd5db9b78475bcc759d196168c016b6e
```

binary checkpointとteacher datasetはcommitしていない。machine-readable evaluationは[`visual_receptive_field_evaluation_2026-07-21.json`](visual_receptive_field_evaluation_2026-07-21.json)に保存した。

## 後続の動的再帰試験

固定depth modelは後にupdate 80,000まで継続し、hard-halting有効化のcontrolled studyに使用した。直接有効化、5,000-update random-depth curriculum、halt-head reset、from-scratch dynamic trainingのすべてで、最終held-panel deterministic trajectoryはdepth 2で停止した。これは次に必要な変更をcandidate固有halting creditへ絞り込む結果であり、visual、episodic-memory、LookupFFN bodyを否定するものではない。詳細は[`DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md`](DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md)を参照。
