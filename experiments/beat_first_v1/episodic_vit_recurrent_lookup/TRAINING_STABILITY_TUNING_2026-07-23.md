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

## halt LR 1e-6の30,000更新比較

halt方策の更新幅をさらに10分の1へ下げた場合を確認するため、同じ20,000更新
checkpoint、同じsampler系列からhalt LRだけを`1e-6`へ変更し、30,000更新まで進めた。
入力、teacher、task loss、probe、dense LR、router LR、weight decay、hard halting、
optimizer stateは同一である。

| 更新 | held loss | held top-1 | held NDCG | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 2.752324 | 0.578125 | 0.982200 | 0.103318 | 3.556 | 2～12 | - |
| 25,000 | 2.713433 | 0.570312 | 0.981893 | 0.117283 | 2.086 | 2～12 | 15.312 |
| 30,000 | 2.692056 | 0.632812 | 0.983332 | 0.120413 | 2.709 | 2～12 | 15.636 |

25,000更新ではhalt LR `1e-5`のarmとほぼ同じ平均深度になったが、loss、top-1、
NDCG、marginはいずれもわずかに劣った。30,000更新ではtop-1を`0.632812`まで回復し、
平均深度も`2.709`へ増えたが、`1e-5`の同時点と比べてlossは`+0.000847`、NDCGは
`-0.000158`、marginは`+0.004374`であり、明確な品質優位はない。

また、halt LR `1e-5`も30,000更新時点では平均深度`2.647`と穏当だったにもかかわらず、
35,000更新で`11.713`へ飽和した。このため、`1e-6`の30,000更新時点だけを根拠に
深度安定化へ成功したとは判定しない。今回の一軸比較から、halt LRの縮小だけでは
candidate-local probe教師の時間変動を十分に抑えられないことが分かった。batch 8も
速度条件を満たさなかったため、次の調整はbatch 4を維持し、dense weight decayを
一軸で比較する。

```text
run: evrl_boundsfix_p2_c0_halt1e6_lr2e4_wd1e4_u30000_20260723_hlr3
checkpoint: checkpoint_000030000.jls
sha256: 2525f8397e6bb60ecd58e8f4d39b3ebc14a0220da1e269aa32364d202fcad2ee
teacher states: 120,000
```

## dense weight decay 3e-4の30,000更新比較

batch 8とhalt LRの一軸比較に続き、dense weight decayだけを`1e-4`から`3e-4`へ
変更した。比較元は同じ20,000更新checkpointであり、halt LRは両armとも`5e-5`、
dense LRは`2e-4`である。resume時にdense weight decayだけの変更を許可する
`EVRL_ENABLE_DENSE_WD_TRANSITION=1`を追加し、他のhyperparameterが一つでも異なる
場合は従来どおり拒否するようにした。

変更後のreal-teacher serial／barrierless bounds-check smokeは合格した。

| 項目 | 結果 |
|---|---:|
| 出力最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw task VJP最大絶対差 | 0 |
| parameter gradient最大絶対差 | 1.52737e-6 |
| parameter gradient relative L2 | 1.42275e-6 |
| optimizer後parameter/state最大絶対差 | 1.52737e-7 |
| 離散経路、probe、RNG、sampler、optimizer clock | すべて一致 |
| 判定 | 合格 |

20,000更新から30,000更新までのpaired結果は次のとおりである。

| dense WD | 更新 | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 25,000 | 2.710472 | 0.578125 | 0.982187 | 0.865256 | 0.122002 | 2.001 | 15.320 |
| 3e-4 | 25,000 | 2.713781 | 0.570312 | 0.981433 | 0.865202 | 0.122533 | 2.113 | 15.235 |
| 1e-4 | 30,000 | 2.692367 | 0.632812 | 0.983506 | 0.866969 | 0.118709 | 3.372 | 15.120 |
| 3e-4 | 30,000 | 2.689768 | 0.625000 | 0.983862 | 0.867854 | 0.117013 | 2.110 | 15.532 |

30,000更新では`3e-4`がlossを`0.002599`、NDCGを`0.000356`、pairwiseを
`0.000885`改善した。一方、top-1は`0.007812`、marginは`0.001696`低い。
平均深度は`1e-4`が`2.001 -> 3.372`と増えたのに対し、`3e-4`は`2.113 ->
2.110`でほぼ一定だった。強いweight decayは短区間の深度振動を抑えたが、より長く
考える能力も抑えている可能性があるため、この時点では採用を確定しない。40,000更新
まで延長し、品質を維持したまま深度が入力依存の範囲を残すかで判定する。

```text
WD 1e-4 control run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd1e4_u30000_20260723_b4
checkpoint: checkpoint_000030000.jls
sha256: 28e18aeaa92da23da916f597bb4ac3b6e0251d6dfee1fefac55e3667abba72d2

WD 3e-4 run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd3e4_u30000_20260723_wd1
checkpoint: checkpoint_000030000.jls
sha256: e28618f8fe9d1bbf84c2cb02f384d7c0e7c575a339a5cb8f403ec34a34baa0f8
```

## dense weight decay 3e-4の40,000更新追跡

30,000更新時点の短期安定性が一時的なものかを確認するため、同じarmを40,000更新まで
延長した。

| 更新 | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30,000 | 2.689768 | 0.625000 | 0.983862 | 0.867854 | 0.117013 | 2.110 | 2～12 | - |
| 35,000 | 2.664932 | 0.695312 | 0.985194 | 0.875882 | 0.141134 | 3.147 | 2～12 | 15.241 |
| 40,000 | 2.645313 | 0.671875 | 0.988018 | 0.881705 | 0.134168 | 2.178 | 2～12 | 15.557 |

loss、NDCG、pairwiseは30,000、35,000、40,000更新の全点で改善した。top-1とmarginは
35,000更新が最高だが、40,000更新も30,000更新より高い。平均深度は`2.110 ->
3.147 -> 2.178`と変動したものの、halt LR `1e-5`で見られた35,000更新の
`11.713`という最大深度飽和は再現しなかった。深度範囲は全点で2～12を保っており、
候補ごとの入力依存性は残っている。

40,000更新ではhalt LR `1e-5` armに対し、lossは`+0.000472`、marginは
`-0.000501`とほぼ同等、top-1は`+0.015625`、NDCGは`+0.000756`、pairwiseは
`+0.000780`、平均深度は`+0.047`だった。品質を維持したまま極端な最大深度飽和を
回避したため、現時点ではdense WD `3e-4`を長期追跡の第一候補に昇格する。ただし
35,000更新から40,000更新への深度低下は残るため、「平均深度が完全に安定した」とは
まだ結論しない。次はこのcheckpointを100,000更新まで延長して判断する。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd3e4_u40000_20260723_wd2
checkpoint: checkpoint_000040000.jls
sha256: fdf372e068aca5c37a1915d880dbc2e3a0ca0be4c4bb9d954f9db3ed73bd17bc
teacher states: 160,000
```

## dense weight decay 3e-4の50,000更新基準

後半のdense LRを調整する共通分岐点として、同じarmをLR `2e-4`のまま50,000更新まで
延長した。

| 更新 | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40,000 | 2.645313 | 0.671875 | 0.988018 | 0.881705 | 0.134168 | 2.178 | 2～12 | - |
| 45,000 | 2.622360 | 0.718750 | 0.988368 | 0.884885 | 0.125085 | 2.131 | 2～12 | 15.191 |
| 50,000 | 2.632145 | 0.734375 | 0.987706 | 0.884073 | 0.131590 | 4.010 | 3～12 | 15.545 |

45,000更新ではloss、top-1、NDCGが同時に改善した。50,000更新ではtop-1と平均深度が
さらに上昇した一方、lossは`+0.009785`、NDCGは`-0.000662`、pairwiseは
`-0.000811`反落した。深度は最小3、最大12となり、入力依存の計算量は明確に残って
いる。完全な頭打ちではないが、連続順位品質の反落が始まったため、50,000更新を
後半dense LR半減の分岐点にする。

次のarmはbank、router、halt LRを維持し、attention、FFN、token、register、headの
既存episodic LR scaleだけを`1.0`から`0.5`へ変更する。これにより長期Lookup routingと
halt方策を変えず、denseな短期・視覚・出力経路の更新幅だけを`2e-4`から`1e-4`へ
下げる。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd3e4_u50000_20260723_wd3
checkpoint: checkpoint_000050000.jls
sha256: 13a99d3dea24942e4766aaae340ed3ecf6d13448954e559845f3bd19fbee93de
teacher states: 200,000
```

## 50,000更新以降のepisodic LR半減

50,000更新checkpointから、bank、router、lookup alpha、halt LRを維持し、attention、
FFN、token、register、headに共通で掛かるepisodic LR scaleだけを`0.5`へ下げた。
実効dense LRは`2e-4`から`1e-4`となる。resumeでこのschedule変更だけを許可する
`EVRL_ENABLE_EPISODIC_LR_TRANSITION=1`を追加し、decay開始更新と係数以外が変わる場合は
拒否する。

50,001更新目でscale `0.5`を実際に適用したreal-teacher serial／barrierless
bounds-check smokeは合格した。

| 項目 | 結果 |
|---|---:|
| 出力最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw task VJP最大絶対差 | 0 |
| parameter gradient最大絶対差 | 2.15694e-6 |
| parameter gradient relative L2 | 6.38261e-7 |
| optimizer後parameter/state最大絶対差 | 2.16067e-7 |
| 離散経路、probe、RNG、sampler、optimizer clock | すべて一致 |
| 判定 | 合格 |

最初の5,000更新では次の結果を得た。

| 更新 | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50,000 | 2.632145 | 0.734375 | 0.987706 | 0.884073 | 0.131590 | 4.010 | 3～12 | - |
| 55,000 | 2.612962 | 0.718750 | 0.988942 | 0.890677 | 0.132410 | 4.287 | 2～6 | 14.776 |

LR半減後はloss、NDCG、pairwise、marginが改善し、平均深度も最大側へ飽和せず中間域を
維持した。top-1だけ`-0.015625`だが、固定128状態では2状態分の差である。品質と深度の
目的には合う一方、5,000更新区間の集計速度が下限15を`0.224 updates/s`下回ったため、
学習は55,000更新checkpoint保存後に自動停止した。

LR係数自体はforward/backwardの演算量を変えず、最終更新の学習時間は約0.0605秒だった。
5,000更新集計には新しいsourceの冷間JITが含まれるため、速度条件を緩めるのではなく、
同じ55,000更新checkpointから100更新warmup後の1,000更新benchmarkを行い、定常速度が
15 updates/s以上かを別途判定する。合格するまでは本学習を再開しない。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_half50k_wd3e4_u70000_20260723_lr1
last checkpoint: checkpoint_000055000.jls
sha256: 721393a555d65477e6c475e471041e509a32f4f932671628f3f7f56f59743e69
teacher states: 220,000
```

### LR半減armの定常速度確認

55,000更新checkpointから100更新をJIT warmupとして実行し、その後の1,000更新だけを
測定した。benchmark更新とoptimizer stateは保存していない。

| 項目 | 結果 |
|---|---:|
| warmup | 100更新 |
| 測定更新 | 1,000 |
| 測定実時間 | 63.0281秒 |
| updates/s | 15.8659 |
| states/s | 63.4638 |
| 平均CPU使用率 | 49.1559% |
| candidate中CPU使用率 | 49.4211% |
| 最低15 updates/s | 合格 |

最初の100測定更新だけでは`14.609 updates/s`だったが、200更新時点で`15.141`、
1,000更新時点で`15.866`へ収束した。したがってLR半減で計算量が増えたのではなく、
新しいsourceを読み込んだ直後のJITが5,000更新区間へ残ったことが速度警告の原因である。
以後の本学習では速度条件を15のまま維持し、checkpoint区間1回ではなく10,000更新以上の
区間で判定して冷間起動の偏りを除く。

### 65,000更新までの本学習再開結果

benchmark更新を破棄し、55,000更新checkpointから本学習を再開した。速度判定は
10,000更新を完了する65,000更新境界まで猶予し、下限自体は15のままとした。

| 更新 | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 55,000 | 2.612962 | 0.718750 | 0.988942 | 0.890677 | 0.132410 | 4.287 | 2～6 | - |
| 60,000 | 2.607455 | 0.703125 | 0.988899 | 0.889694 | 0.147984 | 2.113 | 2～7 | 15.206 |
| 65,000 | 2.594471 | 0.757812 | 0.990265 | 0.896571 | 0.142645 | 2.888 | 2～12 | 14.625 |

品質は65,000更新で本試行の最高水準へ到達した。55,000更新比でloss `-0.018491`、
top-1 `+0.039062`、NDCG `+0.001323`、pairwise `+0.005894`、margin
`+0.010235`である。平均深度は`4.287 -> 2.113 -> 2.888`と変動したが、2または12への
固定ではなく入力依存範囲を維持した。

一方、55,000～65,000更新の実データ10,000更新は683.768秒、`14.625 updates/s`で、
定常benchmarkの`15.866`を再現しなかった。実データ系列ではcandidate数と実現再帰深度が
変化するため、短いbenchmarkだけを採用根拠にはできない。65,000更新checkpoint保存後に
自動停止し、速度条件未達のためLR半減armの本学習継続は保留する。

LR係数は演算量を変えないため、次は同じ50,000更新checkpointからscale `1.0`の対照を
65,000更新まで進める。対照も15未満なら区間固有のworkloadまたは実行環境が原因、対照だけ
15以上ならLR半減によって形成された再帰経路の計算量増加が原因と判定する。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_half50k_wd3e4_u70000_20260723_lr2
last checkpoint: checkpoint_000065000.jls
sha256: 672d73a6a5cf1c40f1fa80fd10fd28b34bdd09e3465dcaf4c24602c9181e53f2
teacher states: 260,000
```

## 50,000更新以降のepisodic LR維持対照

LR半減による品質向上と速度低下を分離するため、同じ50,000更新checkpointから
episodic LR scale `1.0`を維持した対照を65,000更新まで進めた。optimizer、sampler、
RNG、teacher順序、Lookup bank、router、halt LR、loss、schedulerは半減armと同一である。

初回起動ではLookup geometryの環境変数を明示しなかったため、checkpointの
`13 tables x 4096 rows x top-3`とlive既定値の不一致を検出し、更新前に安全停止した。
保存済みtopologyから`DSRL_CARRIER_DIM=128`、`DSRL_TABLES_PER_BLOCK=13`、
`DSRL_WTA_CHOICES=16`、`DSRL_ROWS_PER_TABLE_LOOKUP=3`、`EVRL_REGISTERS=4`、
`EVRL_FFN_DIM=128`を復元して再起動した。失敗した起動ではoptimizerもsamplerも進んで
おらず、対照結果には混入していない。

| 更新 | arm | held loss | held top-1 | held NDCG | held pairwise | held margin | held平均深度 | 深度範囲 | 累積区間updates/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50,000 | 共通起点 | 2.632145 | 0.734375 | 0.987706 | 0.884073 | 0.131590 | 4.010 | 3～12 | - |
| 55,000 | scale 1.0 | 2.628409 | 0.695312 | 0.989086 | 0.887796 | 0.127374 | 3.527 | 2～5 | 17.969 |
| 60,000 | scale 1.0 | 2.611488 | 0.703125 | 0.989321 | 0.888707 | 0.132685 | 2.055 | 2～7 | 16.326 |
| 65,000 | scale 1.0 | 2.605426 | 0.742188 | 0.990088 | 0.893202 | 0.140937 | 3.217 | 2～12 | 17.122 |
| 65,000 | scale 0.5 | 2.594471 | 0.757812 | 0.990265 | 0.896571 | 0.142645 | 2.888 | 2～12 | 14.625 |

scale 1.0対照の15,000更新は876.050秒、`17.122 updates/s`、`68.489 states/s`、
CPU平均`53.759%`、candidate中`54.478%`で完了した。速度下限15を余裕を持って満たす。

65kのscale 0.5は対照比でloss `-0.010955`、top-1 `+0.015625`、NDCG
`+0.000177`、pairwise `+0.003369`、margin `+0.001708`と全品質指標が高い。一方、
速度は`-2.497 updates/s`で対照の`85.4%`に留まり、設定済み下限を満たさない。
LR値はforward/backwardの演算量を直接変えないため、差は学習されたactive trajectory、
実現深度、candidate workloadと長区間の実行揺らぎを含む結果である。

scale 1.0の65kは50k比でloss `-0.026719`、top-1 `+0.007812`、NDCG
`+0.002382`、pairwise `+0.009129`、margin `+0.009347`と全主要指標を改善した。
したがって、品質を改善しながら速度条件も満たす現行採用checkpointを50kから
scale 1.0の65kへ更新する。scale 0.5の65kは品質上限の参考として保持する。

```text
run: evrl_boundsfix_p2_c0_halt5e5_lr2e4_full50k_wd3e4_u65000_20260723_ctrl1
checkpoint: checkpoint_000065000.jls
sha256: b4675bbe17a2963e793ec6f9ff6f0300ec0e13c8f0e12974c88ef9bce20d99e8
teacher states: 260,000
status: complete
```
