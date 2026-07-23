# EVRL halting収束修正

## 目的

根本修正版EVRLではLookup collapse、register別memory retrieval、書込み可能working
memoryを解消した一方、training-only固定128状態の平均再帰深度が、60kから100kの間に
`2.820 -> 4.273 -> 2.000 -> 3.027 -> 4.880`と振動した。

今回の変更は、モデルの入力、teacher、順位task loss、LookupFFN、register、working
memory、hard halting、active-only backward、sparse optimizer、checkpoint形式を維持した
まま、haltingの信用割当を収束しやすい形へ直すことを目的とする。validationとsealed
seedは使用しない。

## 原因

旧実装のstate-level policy gradientは、停止時点で得た同じ最終ranking lossを、その
trajectory中の全stop／continue判断へ与えていた。この方法では、悪いtrajectoryに対して
途中のcontinueと最後のstopが同時に抑制される。特にhalt logitが0付近に集まると、小さな
global bias更新だけで多数候補の深度が最小側または最大側へ一斉に移り、深度振動を起こす。

既存の1-step probeは最終stopだけを教師あり学習しており、最終stopへ到達する前の
continue判断には候補固有の反実仮想信用を与えていなかった。

## 修正

### 各判断時点の候補固有教師

probe対象候補について、既にtrajectoryへ保存済みの各step出力を再利用する。候補のscore
だけをそのstepの値へ差し替え、同じListNetとmargin lossを再計算する。

途中step `t`のcontinue判断には、

```text
delta_t = L(stop at t) - L(realized final step)
```

最終stop判断には、従来どおり物理的に1stepだけ追加実行して、

```text
delta_T = L(stop at T) - L(continue to T+1)
```

を使う。`delta`が追加計算価格を上回るほどcontinue、下回るほどstopとなるsoft targetを
作り、各stochastic decisionへBCE勾配を与える。全深度PonderNetの展開は行わず、追加
forwardは従来と同じprobe候補の最終1stepだけである。

### 高分散policy gradientの限定

trace教師を得たtrajectoryでは、state-wide terminal REINFORCEを重ねない。probeされない
trajectoryだけは探索信号を失わないよう、従来policy gradientを`0.1`倍で残す。これにより
正確な候補固有教師と高分散なterminal creditが競合しない。

### 単調な停止hazard prior

halt logitへ次の固定priorを加え、学習headをcandidate固有の残差判断にする。

```text
prior(step) = 0.5 * (step - 4)
```

最初の試行ではlearned residualを無制限に加えたため、学習後にpriorを上書きできた。
最終版は残差を次のように有界化する。

```text
halt_logit = prior(step) + 0.75 * tanh(learned_logit)
```

決定論的推論ではlearned headだけで停止境界を無制限に押し出せない一方、入力ごとに中心
深度より早く止めることも、遅く続行することもできる。学習時の確率的haltingと最大12
stepの安全上限は維持する。モデルparameter数、checkpoint内のparameter配置、optimizer
stateは変わらない。

### halting専用schedule

- entropy正則化は20k更新までに0へannealする
- halt learning rateは20kまで`5e-5`を保ち、60kへ向けcosine decayする
- 60k以降の下限はbase learning rateの2%、すなわち`1e-6`

後半まで一定のhalt learning rateとentropyを与え続けて方策を揺らす状態を避ける。

## 正当性

有界化前は90k checkpointの次の同一real-teacher 4状態、有界化後は60k checkpointの
次の同一real-teacher 4状態を使い、serial oracleとproduction barrierless executorを
比較した。以下は最終版である有界residual smokeの値である。

- 出力、loss、raw VJP：完全一致
- worker gradient最大絶対差：`2.762e-6`
- parameter gradient最大絶対差：`1.471e-6`
- optimizer後parameter最大絶対差：`1.935e-5`
- optimizer telemetry最大差：`5.617e-7`
- RNG、sampler、hard halting、全stepのhalt trace target、Lookup row、token edge：
  完全一致

checkpointとoptimizer semanticsの互換性を保ったまま、数値一致smokeに合格した。

## 速度preflight

90k checkpointから20 warmup＋100測定更新を行った。これはbenchmark用であり、生成した
90,120 checkpointは採用しない。

| 項目 | 値 |
|---|---:|
| updates/s | `11.300` |
| segment updates/s | `11.462` |
| 平均CPU使用率 | `58.12%` |
| executor CPU使用率 | `65.14%` |
| allocation/update | `11.365 MB` |
| GC占有率 | `0.77%` |
| probe数 | `16` |
| trace教師数 | `32` |
| trace stop target平均 | `0.5723` |
| trace confidence平均 | `0.4318` |

許容下限`10 updates/s`を維持し、stopとcontinueの両方を含む非自明な教師を生成した。

## 既存checkpointへの安定化効果

追加学習を行わず、既存60k～100k checkpointを新しい停止priorの下で同じ
training-only固定128状態へ再適用した。

| checkpoint | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60k | 2.563492 | 0.632812 | 0.990885 | 0.906149 | 0.128296 | 4.000 | 4～4 |
| 70k | 2.554880 | 0.664062 | 0.992015 | 0.913429 | 0.122752 | 4.177 | 4～5 |
| 80k | 2.558170 | 0.617188 | 0.991528 | 0.913289 | 0.108601 | 4.000 | 4～4 |
| 90k | 2.551701 | 0.671875 | 0.991979 | 0.913475 | 0.116506 | 4.004 | 4～5 |
| 100k | 2.559245 | 0.679688 | 0.991217 | 0.908847 | 0.136969 | 4.547 | 4～5 |

学習前から、旧方策の`2.000～4.880`というcheckpoint間振動を`4.000～4.547`へ抑えた。
ただし固定priorだけでは入力依存深度の幅が狭いため、各stepのtrace教師でlearned residual
を再学習させる必要がある。

## 第1試行：無制限residual

60k checkpointから20,000更新した。速度とGCは合格したが、halting収束は不合格だった。

| 項目 | 値 |
|---|---:|
| 実学習時間 | `1,804.961秒` |
| 学習updates/s | `11.081` |
| scheduler全体updates/s | `10.901` |
| 平均CPU使用率 | `52.81%` |
| executor CPU使用率 | `59.41%` |
| allocation/update | `8.570 MB` |
| GC占有率 | `0.744%` |

同一training-only固定128状態の結果は次のとおりだった。

| checkpoint | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60k | 2.563492 | 0.632812 | 0.990885 | 0.906149 | 0.128296 | 4.000 | 4～4 |
| 65k | 2.568041 | 0.632812 | 0.991441 | 0.912559 | 0.128987 | 3.182 | 2～12 |
| 70k | 2.552855 | 0.632812 | 0.991380 | 0.912487 | 0.129254 | 3.413 | 2～12 |
| 75k | 2.554220 | 0.656250 | 0.991380 | 0.912335 | 0.117424 | 2.006 | 2～4 |
| 80k | 2.553490 | 0.632812 | 0.991362 | 0.911733 | 0.121897 | 4.133 | 2～11 |

平均深度は`3.182 -> 3.413 -> 2.006 -> 4.133`と再び振動した。prior自体は正しく
単調でも、無制限のlearned logitがpriorを上書きしたためである。このrunとcheckpointは
原因を切り分ける不採用armとして保存し、本学習の再開点には使わない。

## 第2試行：有界residual

同一teacher sampler位置を持つ根本修正版60k checkpointから20,000更新し、65k、70k、
75k、80kを保存する。全checkpointを同じtraining-only固定128状態で評価し、順位品質だけ
でなく平均、最小、最大深度の推移を同時に判定した。

### 実行結果

- run ID：`evrl_halt_bounded_convergent_from60k_u80000_20260723`
- 開始checkpoint：根本修正版60k
- 更新数：20,000
- 消費teacher state：80,000
- validation／sealed seed：未使用

| 項目 | 値 |
|---|---:|
| 実学習時間 | `1,788.273秒` |
| 学習updates/s | `11.184` |
| scheduler全体updates/s | `10.993` |
| 平均CPU使用率 | `54.31%` |
| executor CPU使用率 | `60.31%` |
| allocation/update | `8.620 MB` |
| GC時間 | `13.877秒` |
| GC占有率 | `0.763%` |
| 最終batchの確率的平均深度 | `3.082` |
| 最終batchの確率的深度範囲 | 2～9 |
| 最終trace stop target平均 | `0.5262` |
| 最終trace confidence平均 | `0.5397` |
| 最終halt learning rate | `1e-6` |

速度は許容下限`10 updates/s`を維持し、GC占有率も1%未満だった。確率的学習では2～9
stepを探索しつつ、決定論的評価の停止境界は有界residualにより3～6へ制限される。

### 同一パネル推移

| checkpoint | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60k | 2.563492 | 0.632812 | 0.990885 | 0.906149 | 0.128296 | 4.000 | 4～4 |
| 65k | 2.576409 | 0.648438 | 0.991098 | 0.906943 | 0.132685 | 3.001 | 3～6 |
| 70k | 2.565254 | **0.664062** | 0.991174 | 0.906647 | **0.141094** | 3.066 | 3～6 |
| 75k | **2.558603** | 0.648438 | 0.991193 | 0.910392 | 0.126238 | 3.006 | 3～6 |
| 80k | 2.560529 | 0.617188 | **0.991465** | **0.911968** | 0.125583 | 3.036 | 3～6 |

65k以降の平均深度は`3.001～3.066`に収まり、最大振幅は`0.065`だった。無制限
residual試行の最大振幅`2.127`に対して約33分の1、96.9%の縮小である。halt weight
normも`0.0763 -> 0.0790 -> 0.0815 -> 0.0840`と緩やかに変化し、biasは
`-0.01093 -> -0.01077`に留まった。

NDCGは65kから80kまで単調改善し、pairwiseも最終的に`0.911968`へ伸びた。従って
haltingを安定させるために順位学習が停止したわけではない。収束arm内のcomposite loss
最良点は75kである。

## 最終checkpoint選択

収束armだけで既存の主採用90kを置き換えず、旧90kと100kも最終halting式で同じ固定
128状態へ再評価した。

| checkpoint | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **旧90k** | **2.551701** | 0.671875 | **0.991979** | **0.913475** | 0.116506 | 4.004 | 4～5 |
| 旧100k | 2.559245 | **0.679688** | 0.991217 | 0.908847 | **0.136969** | 4.547 | 4～5 |
| 新75k | 2.558603 | 0.648438 | 0.991193 | 0.910392 | 0.126238 | 3.006 | 3～6 |
| 新80k | 2.560529 | 0.617188 | 0.991465 | 0.911968 | 0.125583 | 3.036 | 3～6 |

最小loss、最高NDCG、最高pairwiseを同時に持つ旧90kを主採用checkpointとして維持する。
最終式による旧90kの深度は4～5に収まり、旧実装で観測した2～12への不安定な振れを
起こさない。新75kは収束学習のwitnessとして保存するが、品質が90kより低いため主採用
にはしない。

主採用checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_rootfix_register_memory_halt_u100000_20260723\checkpoints\
checkpoint_000090000.jls
```

SHA-256：

```text
3ea19a64fd72521c1e679c53b348525194214db3073e0937e8e515c864e28c71
```

収束arm最良75k SHA-256：

```text
beab318c333472530160bf2b6612fbaebc7bfa43e476cbb1c2cca1ab8c78c65e
```

## 判定

haltingの後半振動は、入力依存性を完全に捨てる固定深度化ではなく、次の組合せで解消した。

1. 各stochastic decisionへの候補固有trace教師
2. probe済みtrajectoryとterminal REINFORCEの競合除去
3. 単調hazard prior
4. learned residualの有界化
5. entropyとhalt learning rateの後半収束

決定論的深度は構造上3～6に有界であり、学習時は2～12のhard halting探索を維持する。
同じ20k更新で平均深度振幅を96.9%減らし、NDCGとpairwiseを改善したため、
halting収束修正は合格と判定する。
