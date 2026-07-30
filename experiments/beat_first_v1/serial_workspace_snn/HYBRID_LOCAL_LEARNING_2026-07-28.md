# SWSNN hybrid local learning検証

## 結論

scaled SWSNNへ次を実装した。

- e-prop型のforward eligibility trace
- 22出力誤差をhead Jacobianでnode/blockへ送るDECOLLE型local signal
- 学習時counter-based Gumbel top-kと、推論時決定論的hard top-k
- workspace key/queryのlocal soft routing credit
- `abs(third factor × eligibility)`を蓄積するutility構造学習
- 12 reducer / 20 workerによる勾配buffer削減

最終`local_hybrid`では、head 4 parameter群だけ教師あり解析VJPを残す。
recurrent側11 parameter群はすべて局所更新され、凍結parameterはない。

参考にした分解は
[e-prop](https://www.nature.com/articles/s41467-020-17236-y)と
[DECOLLE](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full)
である。ただし、本実装はSWSNNのcontinuous membrane head、hard workspace、
ListNet候補比較へ合わせた独自のhybridであり、論文実装の再現ではない。

## 局所学習則

edgeでは各候補の保存trajectoryをforward方向に再生する。

```text
trace(t+1) =
    local_state_recurrence(t) * trace(t)
  + parameter_local_drive(t)

local_gradient += block_learning_signal * trace
```

対象は`weight`、`gate`、`delay`、`leak`、`threshold`、`feedback_gain`、
`input_gain`、`input_bias`、`workspace_decay`である。routingはworkspaceの
低次元状態とblock signalだけを局所的に戻し、`workspace_key`と
`query_weight`を更新する。下流の全シナプスを逆向きに走査しない。

ListNet誤差は全候補を比較してから確定するため、実装は次の二段階である。

1. 全候補をforwardして22出力の誤差を確定
2. 各候補trajectoryをforward方向へ再生してeligibilityと局所更新を生成

## 128状態shadow検証

対象checkpointは
`trained/arena_scaled_u100000_20260727/checkpoints/checkpoint_000100000.jld2`。
固定training panel 128状態で、解析VJPとのaggregate cosineを測った。

| parameter群 | aligned | candidate shuffle |
|---|---:|---:|
| synapse weight | 0.374568 | 0.124118 |
| input gain | 0.700569 | 0.226396 |
| input bias | 0.608456 | 0.179081 |
| gate logits | 0.277754 | 0.097109 |
| delay logits | 0.163858 | -0.001421 |
| leak logits | 0.134306 | 0.045522 |
| threshold logits | 0.083224 | 0.015951 |
| feedback gain | 0.366467 | 0.119617 |
| workspace key | 0.607575 | 0.093819 |
| query weight | 0.777585 | 0.237694 |
| workspace decay | 1.000000 | 1.000000 |

alignedは全群で正、主要な多次元群はcandidate shuffleで低下した。
workspace decayは1 scalarなのでcosineが符号しか表さず、shuffle controlの
識別力はない。第三因子をzeroにした条件では全局所勾配が0になった。

## 64更新hybrid比較

同じ100k checkpointのコピーを用い、固定128状態panelを循環して64更新した。
学習率は`1e-4`、20 worker、12 e-prop reducer、構造intervalは8。
評価時は両方とも決定論的hard top-kである。

| 指標 | 解析VJP | local hybrid |
|---|---:|---:|
| 初期panel loss | 2.720101 | 2.720101 |
| 最終panel loss | 2.668758 | 2.663746 |
| loss差 | -0.051343 | -0.056355 |
| updates/s | 6.1278 | 6.9050 |
| 全20 thread平均CPU | 59.52% | 57.31% |
| worker学習storage | 36.31 MB | 28.06 MB |
| hot allocation | 0 byte | 0 byte |
| hot GC | 0.0 s | 0.0 s |
| 平均発火率 | 0.006731 | 0.006718 |
| workspace route entropy | 0.250118 | 0.248231 |
| workspace RMS | 0.180593 | 0.179138 |
| gate density | 0.501074 | 0.501071 |

この短いtraining-only panel実験では、local hybridはVJPより約12.7%高速で、
worker学習storageを約22.7%減らし、lossを継続して下げた。これは汎化、
ゲーム強度、長期収束、PreAct/DSRLN超えを示さない。

## 構造学習

utilityはupdateごとに次で更新する。

```text
utility_ij =
    decay * utility_ij
  + mean_worker(abs(third_factor * eligibility_ij))
```

周期的なconsolidationでは、現在ONの最低utility接続と、接続costを引いた
最高utilityのOFF接続を比較する。1ノード1 swapまで、かつnode集合を巡回分割し、
一括churnを禁止する。gateのAdamW momentは実際に反転した接続だけresetする。

100k checkpointの64更新では、既存接続を上回るOFF接続がなく反転は0だった。
新規初期化の最も厳しい「毎更新consolidation」2更新smokeでは477反転に抑え、
旧一括補正の30,531反転を廃止した。

## 再現

```powershell
$env:SWSNN_LEARNING_MODE = "local_hybrid"
$env:SWSNN_EPROP_REDUCERS = "12"
$env:SWSNN_STRUCTURAL_INTERVAL = "25"
julia --threads=20,0 --project=. experiments/beat_first_v1/serial_workspace_snn/train_arena_100k.jl
```

shadow control:

```powershell
$env:SWSNN_EPROP_STATES = "128"
$env:SWSNN_EPROP_FULL_STATE_ONLY = "1"
julia --threads=20,0 --project=. experiments/beat_first_v1/serial_workspace_snn/benchmark_eprop_shadow.jl
```

hybrid比較:

```powershell
$env:SWSNN_HYBRID_UPDATES = "64"
$env:SWSNN_HYBRID_ROUTING_ONLY = "1"
$env:SWSNN_EPROP_REDUCERS = "12"
julia --threads=20,0 --project=. experiments/beat_first_v1/serial_workspace_snn/benchmark_hybrid_learning.jl
```

保存済みJSONのSHA-256:

- `eprop_shadow_benchmark.json`:
  `4253045613B2CCEB08DFF520E4854B5FAC25F6C08F439448D9F06DABFC2A14D0`
- `hybrid_learning_benchmark.json`:
  `25F57099AF63CFA03C2F6ABEC64E5C951A3A6C9E2EC4843AC54ABF8CD47735DE`
