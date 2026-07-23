# 固定K=64の疎いエピソード記憶読出し

## 目的

全283 tokenに対する密なcross-attentionを、入力情報を切断しない物理的に疎な
エピソード記憶読出しへ置き換えた。変更対象はregisterと入力固有memoryの間だけであり、
盤面token化、8近傍の空間関係、depthwise視覚処理、register self-attention、
単一LookupFFN、再帰、hard halting、active-only backward、疎optimizer、teacher lossは
維持した。

今回の設計は次の三経路から成る。

1. 各registerがlearned WTA/hash routerによって283 tokenから固定64 tokenを選ぶ。
2. 選択した64 tokenだけにK/V射影と正確なmulti-head attentionを行う。
3. 全283 tokenの正規化平均を小さな常設経路として全registerへ加える。

Kは学習時も推論時も64で固定した。Kのcurriculum、annealing、64から16への再選別、
dense score計算後のmaskは導入していない。

## 実装

### register別の固定K読出し

token側とregister側に共有のblock-structured射影を持たせ、入力ごと、registerごとに
異なる64 token IDを取得する。routerは全tokenの低次元descriptorだけを走査し、
高価なK/V射影とQK計算は選択されたsupportにしか行わない。

support選択はhardだが、選択された64 token内では通常のsoftmax attentionを用いる。
attention scoreのVJPからstraight-throughのrouter勾配を作るため、token routerと
register routerの両方をtask lossから更新できる。

### 入力切断を防ぐ平均経路

hard routerが初期に重要tokenを外しても入力全体が消えないよう、正規化済み283 tokenの
平均を各registerへ残差注入する。この経路は283個の加算と定数倍だけであり、
全tokenへの最低限の信用割当を保証する。

### 同一supportへのworking-memory write

registerからepisodic memoryへの書込みは、読出しで選択した64 tokenの和集合だけに
限定した。未選択tokenへのdense reverse attentionは行わない。複数registerが同じtokenを
選んだ場合だけ、該当register間で書込み重みを正規化する。forward、backward、
trajectory保存はいずれも同じ物理supportを使う。

### 規模

- model次元：128
- register数：4
- attention次元：32、4 head
- episodic token数：283
- support：registerごとに64
- LookupFFN：共有再帰block内に1 block
- parameter数：6,954,877

## 正当性

### 構造smoke

`test_fixed_k64_episodic_lookup.jl`で次を確認した。

- 各registerのsupportは重複なしの64 token
- working-memory write先は読出しsupportの和集合の部分集合
- 全token平均は実際の正規化token平均と一致
- cross Q、token router、register router、平均経路scaleの勾配が非ゼロ
- 平均経路により全283 tokenへ勾配が到達
- 出力と勾配normが有限

結果は25項目すべて合格した。既存の単一Lookup＋再帰depthwise回帰試験も8項目すべて
合格した。

### serial対barrierless

実teacherのupdate 10,000 checkpointから同じ4 training state、同じcandidate、
同じRNG順序で1更新を比較した。

| 項目 | 結果 |
|---|---:|
| 出力の最大絶対差 | 0 |
| lossの最大絶対差 | 0 |
| raw VJPの最大絶対差 | 0 |
| parameter gradientの相対L2差 | 1.0079e-6 |
| 更新後parameter stateの相対L2差 | 2.2065e-9 |
| optimizer telemetryの相対L2差 | 5.3741e-8 |

hard halting、選択token、Lookup row、probe教師、active token mask、sparse row clock、
sampler状態、次のsampler row、optimizer clockも一致した。

## 実行性能

20 Julia worker、BLAS 1、barrierless、pinningなし、chunk 8で40更新のsteady-stateを
測定した。

| 指標 | 結果 |
|---|---:|
| updates/s | 11.7391 |
| 全体CPU使用率 | 59.54% |
| candidate処理中CPU使用率 | 59.96% |
| allocation/update | 9.067 MB |
| GC時間 | 0.0287秒 / 40更新 |
| GC占有率 | 0.83% |

従来の全283 token readと比べてcross readとworking-memory writeを物理的に疎化した。
K/V投影は選択memoryを連続scratchへ集めてBLASで処理し、scalar loop版で発生した
大幅な速度低下を解消した。許容下限の10 updates/sを満たしている。

## 学習率調整

共通条件はstate batch 4、dense LR 2e-4、dense weight decay 3e-4、halt LR 5e-5、
2 probe/state、compute price 0である。新しく復活したepisodic routerのLRだけを比較した。
各試行は同一seed、実teacher 1,000更新、4,000 training stateである。

### 固定training-only 128状態

| router LR | 更新 | loss | top-1 | NDCG | pairwise | margin | 学習速度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4e-4 | 500 | 3.349055 | 0.289062 | 0.936698 | 0.741312 | 0.118830 | 11.51 updates/s |
| 4e-4 | 1,000 | 3.276297 | 0.421875 | 0.955092 | 0.773679 | 0.110220 | 11.43 updates/s |
| 2e-4 | 500 | 3.374201 | 0.304688 | 0.938292 | 0.737469 | 0.109145 | 11.34 updates/s |
| 2e-4 | 1,000 | 3.270660 | 0.367188 | 0.948351 | 0.763083 | 0.110444 | 11.35 updates/s |

1,000更新では4e-4が2e-4に対してtop-1を0.054687、NDCGを0.006741、
pairwiseを0.010596改善した。lossだけは2e-4が0.005637低いが、順位学習を主目的とし、
速度差もないためrouter LR 4e-4を採用する。

評価行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。validation rowとsealed seedには触れていない。

## 採用設定

- episodic support：固定64
- router LR：4e-4
- bank／attention／FFN／token／register／head LR：2e-4
- dense weight decay：3e-4
- halt LR：5e-5
- warmup：5,000更新
- halt probe：2 candidate/state
- compute price：0
- scheduler：barrierless、pinningなし、chunk 8

## 10,000更新

採用した1,000更新checkpointから10,000更新まで継続した。追加9,000更新の結果は
次のとおりである。

| 指標 | 結果 |
|---|---:|
| 総更新数 | 10,000 |
| 消費teacher state | 40,000 |
| 追加区間の学習実時間 | 783.561秒 |
| 追加区間の速度 | 11.4605 updates/s |
| 全体CPU使用率 | 52.51% |
| candidate処理中CPU使用率 | 55.59% |
| 最終batch loss | 3.078363 |
| 最終batch平均深度 | 3.126 |
| 最終batch深度範囲 | 2～9 |
| 最終batch probe | continue 4 / stop 4 |

checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_fixed_k64_wta_mean_wm_u10000_20260723\checkpoints\
checkpoint_000010000.jls
```

SHA-256：

```text
c834611e07cec1743658cea90118253406c45bdb5b5c8db625259229df89906a
```

### 同一training-only 128状態の推移

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 3.276297 | 0.421875 | 0.955092 | 0.773679 | 0.110220 | 3.000 | 3 |
| 9,000 | 2.869512 | 0.437500 | 0.973726 | 0.832315 | 0.094285 | 3.041 | 3～6 |
| 10,000 | 2.858829 | 0.453125 | 0.974179 | 0.834829 | 0.081635 | 3.061 | 3～6 |

1,000から10,000更新でlossは0.417468低下し、top-1は0.031250、NDCGは
0.019086、pairwiseは0.061150改善した。9,000から10,000更新でも全4順位指標が
同時に改善しており、10,000更新時点はplateauではない。

halt weight normは`0 -> 0.3039`となり、固定パネル上の深度も単一の3から3～6へ
分岐した。最終training batchでは2～9を使用し、probe教師もcontinueとstopに
均等に分かれた。したがって、hard haltingはwarmup後に入力依存の反復を学習し始めている。

同じ40,000 teacher stateを使った直前の全tokenモデルはloss`2.810665`、
top-1`0.531250`、NDCG`0.979387`だった。今回の固定K64版はそれぞれ
`+0.048164`、`-0.078125`、`-0.005208`であり、10,000更新時点のsample
efficiencyでは全token版にまだ届いていない。一方、固定K64版は入力読出しと
working-memory書込みの物理的疎性を実現しつつ、最低速度10 updates/sと継続学習信号を
維持した。長期学習前の段階で精度超過を主張しない。

validation rowとsealed seedには全過程で触れていない。

## 100,000更新

採用したrouter LR `4e-4`の10,000更新checkpointから、モデル構造、入力、teacher、
loss、optimizer、hard halting、probe数、schedulerを変えずに100,000更新まで継続した。
10,000更新以降は90,000更新、360,000 teacher stateを追加した。

| 指標 | 結果 |
|---|---:|
| 総更新数 | 100,000 |
| 総消費teacher state | 400,000 |
| 10k→100kの学習時間 | 7,310.216秒 |
| 10k→100kの最終記録速度 | 12.3088 updates/s |
| 全体CPU使用率 | 50.94% |
| candidate処理中CPU使用率 | 53.36% |
| steady benchmark allocation | 9.067 MB/update |
| steady benchmark GC占有率 | 0.83% |

20 Julia worker、BLAS 1、barrierless、pinningなし、chunk 8を全区間で維持した。
同時に重いJuliaを複数起動していない。100,000更新checkpoint保存後に学習Juliaは正常終了した。

### 10k刻みの追加学習なし再評価

全checkpointを同じtraining-only固定128状態、同じ評価式で一括評価した。
評価行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
であり、validation rowとsealed seedには触れていない。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.858831 | 0.453125 | 0.974178 | 0.834830 | 0.081634 | 3.061 | 3～6 |
| 20,000 | 2.755108 | 0.500000 | 0.981385 | 0.856700 | 0.095351 | 3.070 | 3～6 |
| 30,000 | 2.720215 | 0.554688 | 0.983597 | 0.870289 | 0.111738 | 3.002 | 3～6 |
| 40,000 | 2.679639 | 0.546875 | 0.983974 | 0.877964 | 0.125210 | 3.087 | 3～6 |
| 50,000 | 2.648654 | 0.585938 | 0.986037 | 0.883243 | 0.116143 | 3.084 | 3～6 |
| 60,000 | 2.628023 | 0.617188 | 0.987335 | 0.891481 | 0.138669 | 3.049 | 3～6 |
| 70,000 | 2.668506 | 0.609375 | 0.985697 | 0.885012 | 0.122491 | 3.055 | 3～6 |
| 80,000 | 2.629101 | 0.593750 | 0.989182 | 0.899729 | 0.100275 | 4.133 | 3～6 |
| 90,000 | 2.610506 | 0.617188 | **0.989632** | 0.902576 | 0.111613 | 3.001 | 3～6 |
| 100,000 | **2.602389** | **0.617188** | 0.989165 | **0.903184** | 0.121353 | 3.026 | 3～6 |

100,000更新がlossとpairwiseで最良であり、top-1も60,000、90,000更新と同率最良
だったため、主採用checkpointは100,000更新とする。NDCGだけは90,000更新が
100,000更新を`0.000468`上回った。60,000更新以降は単調ではないが、
90,000から100,000更新でもlossは`0.008117`低下し、pairwiseは`0.000608`
改善した。したがってlossについて完全な頭打ちとは判定しない一方、順位品質の改善幅は
小さく、収束近傍の振動域に入っている。

平均深度は80,000更新で`4.133`まで一時的に広がった後、100,000更新では`3.026`へ
戻った。halt weight normは10,000更新の`0.3039`から100,000更新の`0.4736`へ増えたが、
平均深度は安定した単調推移ではない。入力依存の3～6 step選択は残っているものの、
halting方策の収束は未完了である。

### PreActとの差

既存PreActの最高品質点は12,750更新、51,000 teacher stateで、loss`2.550905`、
top-1`0.789062`、NDCG`0.994468`、pairwise`0.930203`、margin`0.116617`である。
固定K64版100,000更新との差は次のとおり。

| 指標 | 固定K64 100k | PreAct 12.75k | 固定K64－PreAct |
|---|---:|---:|---:|
| loss | 2.602389 | 2.550905 | +0.051484 |
| top-1 | 0.617188 | 0.789062 | -0.171875 |
| NDCG | 0.989165 | 0.994468 | -0.005303 |
| pairwise | 0.903184 | 0.930203 | -0.027019 |
| margin | 0.121353 | 0.116617 | +0.004736 |

固定K64版は400,000 teacher stateを消費しており、PreActの約7.84倍である。それでも
loss、top-1、NDCG、pairwiseでPreActへ届かなかったため、sample efficiencyと到達品質の
PreAct超えは実証できなかった。marginだけは固定K64版が高い。

一方、学習速度は固定K64版`12.3088 updates/s`、記録済みPreAct
`0.4894 updates/s`で、同じ4 state/update基準では固定K64版が約25.2倍高い。
これはCPU上のtraining throughput上の優位であり、精度優位を意味しない。

### checkpoint

主採用checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_fixed_k64_wta_mean_wm_u100000_20260724\checkpoints\
checkpoint_000100000.jls
```

SHA-256：

```text
094b0767f91d3d488532570b8363c573b2a76716b21db0a7372d5abbeeefa0e9
```

10k刻みのSHA-256：

```text
10k  c834611e07cec1743658cea90118253406c45bdb5b5c8db625259229df89906a
20k  ead2c5469110d5d8c39c0bf59426b3a89fc011a7586669179832490d477d09c8
30k  4be6ff20876e95d44d15626b149366c9e298a651dd1ac51da11eadfcdee0e572
40k  855fd1a7e42934efdc9ca44b6f41fc377086957cba574509ab82f4dc026ec2b0
50k  eb1523069dc60198ea1b17424d98809c04fae8213df5d9d3bafa3accec9247c9
60k  5864e83a81dec1a6e49eaa6728b206ecac5534dcd7cf15da0aa03e8a86e8a16a
70k  24aa05db2200ae0f036f87c871eb42a99f86126011ac21801504c8d2bf28ba76
80k  36af91aa78af73ed57e527206bc6732adcbc1bcdd8651dcc3a146c75c1f20ab4
90k  f285c550e5390436b49a6f93706437ee0c1bf2ff448bf12dc1c7125dcc1ecace
100k 094b0767f91d3d488532570b8363c573b2a76716b21db0a7372d5abbeeefa0e9
```

## 結論

固定K=64の物理的疎な入力読出しは、全tokenへの安価な平均経路と組み合わせることで、
100,000更新まで学習信号を保ち、10 updates/s以上で学習できた。router LRは`4e-4`を
採用し、主損失は100,000更新まで改善した。

ただし、全283 tokenを高帯域に読む構成と比べた情報制約は依然大きく、PreActの精度を
上回らなかった。またhalting深度は一時的に広がるものの安定収束していない。今回確認
できたのは、入力欠損を回避した固定K疎読出しが実用速度で学習可能であることと、
その現行品質上限であり、PreAct超えではない。
