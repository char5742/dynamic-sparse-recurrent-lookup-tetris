# EVRL 学習安定化調整（2026-07-23）

## 目的

動的再帰P1で見られたloss、top-1、平均深度の振動を、モデル全体のアーキテクチャを変えずに学習率、weight decay、state batchの調整で改善する。その前提として、同じ実teacher系列を長時間安全に学習できることを確認する。

数値一致smokeと速度benchmarkはtraining splitだけを使用した。品質曲線には既存の
固定teacher evaluation panelを用いたが、sealed game seedは構築も参照もしていない。

## 調整前に判明したbackwardの範囲外書込み

通常実行では200～400更新後にJuliaのGCが`EXCEPTION_ACCESS_VIOLATION`で停止した。scheduler、worker数、GC worker数、resume checkpointを変えても再現したため、`--check-bounds=yes`で既知P1 checkpointを再生した。

その結果、`_spatial_attention_vjp!`が8近傍attentionのedge 5～8を、長さ4の`BackwardScratch.dweights`へ書き込んでいたことが判明した。

```text
BoundsError: 4-element Vector{Float32} at index [5]
source: EpisodicViTRecurrentLookup.jl:_spatial_attention_vjp!
```

scratch容量はepisodic shortlist、register数、旧spatial shortlistの最大値だけで決められており、`LOCAL_SPATIAL_NEIGHBORS = 8`が含まれていなかった。修正は共有softmax VJP scratchの容量上限へ8近傍数を追加するだけであり、forward、backwardの数式、入力、teacher、loss、hard halting、RNG順序、active-only backward、sparse optimizer、checkpoint形式は変更していない。

## 数値一致smoke

10,000更新checkpointと同じreal-teacher 4状態を使い、single-worker oracleと20-worker barrierlessを`--check-bounds=yes`で比較した。

| 項目 | 結果 |
|---|---:|
| 出力最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw task VJP最大絶対差 | 0 |
| parameter gradient最大絶対差 | 4.59794e-6 |
| parameter gradient relative L2 | 2.04267e-6 |
| optimizer後parameter/state最大絶対差 | 4.60073e-7 |
| 離散経路、halt probe、RNG、sampler、optimizer clock | すべて一致 |
| 判定 | 合格 |

## 1,000更新の長時間preflight

実際に100,000更新を完了したP1の75,000更新checkpointから、checkpointを書き出さず1,000更新を再生した。20 Julia worker、BLAS 1 thread、barrierless、pinningなし、chunk 8を使用した。

| 項目 | 結果 |
|---|---:|
| 実行更新 | 1,000 |
| throughput測定更新 | 990（最初の10更新を除外） |
| 測定実時間 | 63.8505秒 |
| updates/s | 15.5050 |
| 全体CPU使用率 | 49.6387% |
| candidate executor CPU使用率 | 51.3630% |
| allocation | 7.739 MB/update |
| GC時間 | 0.5377秒（測定時間の0.84%） |
| GC／範囲外アクセス停止 | なし |
| 最低速度15 updates/s | 合格 |

## 既存結果の扱い

過去のP1 100,000更新は完走しているが、範囲外書込みを含むbackwardで作成されている。したがって既存checkpointは比較用の履歴として保持する一方、修正後モデルの最終性能根拠にはしない。以後のLR、weight decay、batch調整は修正版からscratch学習し、同一teacher-state予算で比較する。

## 次の調整順序

1. 修正版P1設定をscratchから再実行し、安定性の基準曲線を作る。
2. candidate-local probeは固定し、halt learning rateを一軸で下げて平均深度の振動を比較する。
3. dense learning rateとdense weight decayを一軸ずつ比較する。
4. state batch変更は、serial/barrierlessの数値一致とcheckpoint互換性を確認できた場合だけ採用する。
5. loss、top-1、NDCG、margin、平均深度が同時に安定する構成だけを100,000更新まで継続する。

## state batch 8の数値一致と速度判定

`EVRL_STATE_BATCH`を4または8から選べるようにし、workspace、flattened candidate
capacity、barrierless state runtime、optimizer平均化係数、checkpointのstate-batch
整合性を同じ定数から構築した。batch 8ではqueue capacityを640候補の2倍以上となる
2,048へ自動拡張した。モデル構造、loss、optimizer semanticsは変更していない。

8状態、実候補数`52, 26, 26, 51, 34, 54, 52, 53`の全328候補を使った
`--check-bounds=yes` smokeでは次の結果を得た。

| 項目 | 結果 |
|---|---:|
| 出力最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw task VJP最大絶対差 | 0 |
| parameter gradient最大絶対差 | 3.50922e-6 |
| parameter gradient relative L2 | 2.21206e-6 |
| optimizer後parameter/state最大絶対差 | 3.50643e-7 |
| 離散経路、probe、RNG、sampler、optimizer clock | すべて一致 |
| 判定 | 合格 |

smokeの可変長trajectory witnessは、巨大な`NTuple`をclosure型へ埋め込むとJuliaの
compiler stack overflowを起こしたため、同じ内容をheap上の可変長`Vector`として
保持するようにした。これは診断表現だけの変更であり、学習経路には入らない。

10更新warmup後の100更新benchmarkをbatch 4の修正後preflightと比較した。

| state batch | updates/s | states/s | CPU使用率 | allocation/update | 判定 |
|---:|---:|---:|---:|---:|---|
| 4 | 15.5050 | 62.020 | 49.64% | 7.739 MB | 採用 |
| 8 | 7.5291 | 60.233 | 56.99% | 15.426 MB | 不採用 |

batch 8はCPU使用率を上げたが、state throughputを改善せず、最低15 updates/sも
満たさなかった。大batchによる勾配分散低減の可能性より実行条件を優先し、本線は
batch 4へ戻した。

## 修正後の動的再帰20,000更新基準

修正後モデルをscratchから学習し、最初の5,000更新は深度2～6のrandom-depth
warmup、その後はcandidate-local 1-step probe付きhard haltingとした。設定は
dense LR `2e-4`、router LR `4e-4`、halt LR `5e-5`、dense weight decay
`1e-4`、compute price `0`、2 probes/stateである。

| 更新 | held loss | held top-1 | held NDCG | held margin | held平均深度 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 2.867476 | 0.507812 | 0.978324 | 0.099650 | 2.000 | - |
| 10,000 | 2.810665 | 0.531250 | 0.979387 | 0.088824 | 2.107 | 15.506 |
| 15,000 | 2.758752 | 0.523438 | 0.981218 | 0.104578 | 2.396 | 15.343 |
| 20,000 | 2.752324 | 0.578125 | 0.982200 | 0.103318 | 3.556 | 15.592 |

lossとNDCGは全評価点で改善した。top-1は15,000更新で一度小幅に下がったが、
20,000更新では開始点より`+0.070313`上昇した。平均深度は`2.000 -> 2.107 ->
2.396 -> 3.556`と単調に増え、評価候補によって最小2、最大12を使い分けた。
範囲外書込み修正後は、旧P1で見られた両端への急激な深度振動を20,000更新まで
再現していない。

20,000更新checkpointは次である。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd1e4_u20000_20260723_b2
checkpoint: checkpoint_000020000.jls
sha256: 5cfa14c342acdb450911acd10b70c2a7e65c6a120d0a2e0c114834ee3bd1ff52
teacher states: 80,000
```

この結果から、修正後の現行LR `2e-4`とweight decay `1e-4`を安定基準として採用する。
後半の振動を抑えるため、50,000更新以降はepisodic dense学習率だけを0.5倍にする
予定とし、それ以前のoptimizer設定は変更しない。

## 25,000更新でのhalt LR paired trial

現行halt LR `5e-5`のまま20,000更新checkpointを25,000更新まで延長すると、held
lossとmarginは改善した一方、平均深度が`3.5563`から`2.0005`へ急落した。そこで同じ
20,000更新checkpoint、同じsampler系列から、halt LRだけを`1e-5`へ下げて5,000更新
進めた。他のLR、weight decay、loss、probe、RNG、optimizer stateは同一である。

checkpoint resumeでこの一軸変更だけを許可する`EVRL_ENABLE_HALT_LR_TRANSITION=1`を
追加した。明示指定がない場合、またはhalt LR以外も異なる場合は従来どおりresumeを
拒否する。transition後の全候補serial/barrierless smokeは、出力、loss、raw VJPが
完全一致し、parameter gradient最大差`1.53e-6`、optimizer後state最大差`1.53e-7`
で合格した。

| 20k→25k arm | held loss | held top-1 | held NDCG | held margin | held平均深度 | 深度範囲 | updates/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| halt LR 5e-5 | 2.710472 | 0.578125 | 0.982187 | 0.122002 | 2.0005 | 2～3 | 15.320 |
| halt LR 1e-5 | 2.710228 | 0.578125 | 0.981946 | 0.122518 | 2.0844 | 2～12 | 15.418 |

`1e-5`はloss、margin、平均深度、速度をわずかに改善し、top-1を維持した。NDCGは
`-0.000241`の微減である。5,000更新だけでは深度安定化の証拠として不十分なため、
このarmを50,000更新まで延長してから最終採否を決める。

```text
run: evrl_boundsfix_p2_c0_halt1e5_lr2e4_wd1e4_u25000_20260723_hlr1
checkpoint: checkpoint_000025000.jls
sha256: db042e9c35ffc49e121988bc4a6b57281a2cdd38e7e06be584f029802f35c2d8
```

## halt LR 1e-5の40,000更新追跡

25,000更新checkpointから同じarmを延長した。40,000更新境界は保存済みであり、その後
の未保存更新は深度振動の判定後に破棄した。

| 更新 | held loss | held top-1 | held NDCG | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | 2.710228 | 0.578125 | 0.981946 | 0.122518 | 2.084 | 2～12 | - |
| 30,000 | 2.691209 | 0.632812 | 0.983490 | 0.116039 | 2.647 | 2～12 | 15.271 |
| 35,000 | 2.671507 | 0.671875 | 0.985029 | 0.132433 | 11.713 | 6～12 | 15.663 |
| 40,000 | 2.644841 | 0.656250 | 0.987263 | 0.134669 | 2.131 | 2～12 | 15.748 |

品質は40,000更新まで明確に改善した。しかし平均深度は`2.084 -> 2.647 -> 11.713
-> 2.131`と、下限側から上限側へ移動した後、再び下限側へ反転した。halt LRを5分の1
にしても、旧試行で見られた両端振動は解消しなかった。したがって`1e-5`は品質arm
としては有望だが、平均深度安定化の解としては不採用である。次は同じ20,000更新
checkpointから`1e-6`を短区間だけ比較し、方策更新幅の限界を確認する。

```text
run: evrl_boundsfix_p2_c0_halt1e5_lr2e4_wd1e4_u50000_20260723_hlr2
last accepted checkpoint: checkpoint_000040000.jls
sha256: d73a4a31bf8c12efcef16186926b6e7a09a6b9f47e60ac62f9da4fd1ec85844c
```
