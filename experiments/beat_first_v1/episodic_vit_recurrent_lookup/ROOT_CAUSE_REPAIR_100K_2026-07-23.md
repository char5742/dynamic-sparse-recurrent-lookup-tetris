# EVRL根本修正と100,000更新の結果

## 結論

Lookup routingのcollapse、register間で共有されていた長期記憶取得、読取り専用だった
episodic memory、haltingの不完全な信用割当を同時に修正し、実teacherで100,000更新を
完了した。今回の結果から、4項目はいずれも実装上だけでなく学習後の数値でも機能したと
判定する。

- Lookup row-load Giniは旧構成の`0.966～0.970`から`0.187～0.193`へ低下した。
- Lookup row coverageは旧構成の`26.9～30.1%`から`99.7～99.9%`へ上昇した。
- 同じ候補内でもregister間のLookup addressは`99.90%`が異なった。
- register間のcross-attention argmaxは`89.72%`が異なった。
- 書込み経路の学習済みscaleは`0.2172`、観測write RMSは`0.3844`であり、working
  memoryは恒等経路へ消えていない。
- training-only固定128状態では再帰深度が`3～12`に分布し、平均`4.880`だった。
- 100,000更新を`11.332 updates/s`で完走し、許容下限`10 updates/s`を守った。
- GCは全測定wall timeの`0.667%`であり、今回の速度低下の主因ではない。

validationとsealed seedには触れていない。このため、本報告は学習可能性、構造上の
問題解消、training split内の診断を示すものであり、独立汎化性能やPreAct超えの最終証明
ではない。

## 修正内容

### 1. Lookup collapseの解消

各state内で、block、table、digit、choiceごとのhard WTA選択頻度を集計する。補助損失は
hard頻度とsoft routing確率の積に相当する勾配として実装し、選択済みchoiceだけでなく
全choiceのrouter logitへ信用を返す。

forward、bank backward、optimizer更新は従来どおり選択rowだけを処理する。追加したのは
小さなrouter確率への勾配であり、全bank rowをdenseに読む処理ではない。学習時の重みは
`EVRL_ROUTE_BALANCE_WEIGHT=0.05`とした。

### 2. register別の長期記憶取得

旧構成は4 registerを平均して1本のcarrierを作り、3 blockのLookup結果を全registerへ
broadcastしていた。これを廃止し、各registerが共有bankに対して独立に3 blockをrouting
する構成へ変更した。

bank parameterは共有したままなのでモデル容量はほぼ増えない。一方、address、選択row、
residual、backward creditはregisterごとに独立する。1 recurrent stepのLookup micro-callは
`3`から`3 x 4 = 12`へ増えた。

### 3. 書込み可能なworking memory

各stepのcross-attentionで読んだ283 tokenへのsupportを再利用し、更新後registerから
episodic tokenへreverse cross-attention writeを行う。書込みは次のとおりである。

```text
memory[t+1] = memory[t] + alpha_write * W_o * weighted(W_v * registers[t+1])
```

次stepは更新後memoryを読む。VJPはwrite value、output projection、scale、register、
cross-attention weightまで戻る。入力固有memoryはこれにより読取り専用encoder cacheでは
なく、再帰中に更新されるworking memoryになった。

### 4. halting信用割当

旧probe modeではone-step probe BCEが有効なとき、trajectory policy gradientを完全に
置換していた。その結果、probeされなかったstopと、それ以前のcontinue判断にはhalting
gradientがなかった。

修正後は全stochastic stop/continueへpolicy gradientとentropy gradientを常に与え、
少数one-step probeのBCEを対象candidateの最終stopへ加算する。probeは物理的疎性を
維持した補助教師であり、trajectory全体の信用割当を置換しない。

## 正当性確認

### serial対barrierless smoke

同じreal-teacher 4状態、candidate上限8、同じcheckpointで比較した。

| 項目 | 差 |
|---|---:|
| 出力の最大絶対差 | `0` |
| lossの最大絶対差 | `0` |
| raw VJPの最大絶対差 | `0` |
| worker gradientの最大絶対差 | `6.873e-7` |
| worker gradientの相対L2差 | `2.822e-7` |
| parameter gradientの最大絶対差 | `3.912e-7` |
| parameter gradientの相対L2差 | `2.873e-7` |
| optimizer後parameterの最大絶対差 | `3.912e-8` |
| optimizer後parameterの相対L2差 | `3.916e-10` |
| optimizer telemetryの最大絶対差 | `3.705e-8` |

candidate seed、実現深度、hard halting、token edge、Lookup row ID、probe target、active
token、sparse row event、optimizer clock、RNG stateは完全一致した。

### 固定batch過学習

update 1,000 checkpointから同じreal-teacher 4状態を100回反復した。hard halting、probe、
task loss、optimizerは本学習と同じである。

| 指標 | 最初10更新 | 最後10更新 | 変化 |
|---|---:|---:|---:|
| 平均loss | `6.2043` | `3.5942` | `-42.07%` |
| 平均深度 | `2.998` | `3.004` | 維持 |

旧構成は同じ試験でloss低下が`4.04%`に留まった。新構成は少数状態を明確に記憶でき、
実用的な信用割当が復元された。

## 100,000更新

### 条件

- run ID：`evrl_rootfix_register_memory_halt_u100000_20260723`
- 初期状態：from scratch
- teacher state batch：4
- 消費teacher state：400,000
- model parameter：20,585,982
- Lookup：3 block、13 table/block、4,096 row/table、top-3
- register：4、model dim 128
- routing temperature：`1.0 -> 0.25`、最初10,000更新でanneal
- balance weight：`0.05`
- halting：最小2、最大12、warmup 1,000更新、4 probes/state
- 学習率：bank `2e-4`、router `4e-4`、その他 `2e-4`、halt `5e-5`
- dense weight decay：`1e-4`、bank weight decay：0
- scheduler：20 worker、barrierless、pinningなし、chunk 8、BLAS 1 thread
- benchmark-only training split。validation／sealed seedは未使用

### 完了結果

| 項目 | 値 |
|---|---:|
| 更新数 | 100,000 |
| 実時間（scheduler全体） | 8,959.066秒 |
| 実学習時間 | 8,824.605秒 |
| 学習updates/s | 11.332 |
| scheduler全体updates/s | 11.162 |
| 平均CPU使用率 | 54.27% |
| executor CPU使用率 | 61.53% |
| allocation/update | 7.355 MB |
| GC時間 | 59.792秒 |
| GC占有率 | 0.667% |
| 最終batch loss | 2.94039 |
| 累積平均再帰深度 | 2.962 |

速度は旧pooled-register構成の全区間`16.463 updates/s`より低い。register別Lookupを
4本実行し、working-memory writeも追加したためである。新構成は旧速度の`68.8%`だが、
ユーザーが指定した下限`10 updates/s`を上回った。GC占有率は1%未満であり、残る速度差は
主として追加した有効計算とqueue tailである。

## 最終checkpoint診断

### Lookup利用

| block | coverage | row-load Gini | 最大row load |
|---:|---:|---:|---:|
| 1 | 99.8648% | 0.1906 | 9,885,356 |
| 2 | 99.8761% | 0.1873 | 10,598,935 |
| 3 | 99.7183% | 0.1927 | 11,263,138 |

旧構成のcoverageは30.14%、26.95%、27.17%、Giniは0.9666、0.9701、0.9665
だった。従って「一つのrowが約99.5%選ばれる」collapseは解消した。ただし完全均等化は
目的ではなく、学習済みhot rowは残している。

### register別取得

training split内の1状態、34候補で全stepを調べた。

| 指標 | 値 |
|---|---:|
| register pair間のLookup address相違率 | 99.9038% |
| cross-attention平均total variation | 0.5112 |
| cross-attention argmax相違率 | 89.7186% |
| 最終register pair RMS距離 | 1.9788 |

共有bankと共有Q/K/Vを使いながら、各registerは異なる入力tokenと異なる長期記憶rowを
取得している。

### working memory

| 指標 | 値 |
|---|---:|
| 学習済みwrite residual scale | 0.21723 |
| 実trajectoryのwrite RMS | 0.38439 |
| write value weight norm | 7.8772 |
| write output weight norm | 7.7831 |

write scaleには0.1の下限があるが、学習後は0.217まで上がり、実際のtoken更新量も0では
ない。次stepへ渡るmemoryは明確に書き換えられている。

### halting

training split内の固定128状態で測定した。

| 指標 | 値 |
|---|---:|
| 平均深度 | 4.8798 |
| 最小深度 | 3 |
| 最大深度 | 12 |
| halt weight norm | 0.06942 |
| halt bias | -0.01490 |

100,000更新の最終training batchでも深度は2～10、probe教師はcontinue 7件、stop 9件に
分かれた。haltingは最短または最大深度へ一様collapseしていない。

### training-only固定128状態

| loss | top-1 | NDCG | pairwise | margin | 平均深度 |
|---:|---:|---:|---:|---:|---:|
| 2.55998 | 0.671875 | 0.991213 | 0.908830 | 0.137023 | 4.8798 |

これはtraining split内の診断値である。validationを見ないという今回の条件を守ったため、
旧held panelやPreActとの差は本報告では計算しない。

## 保存物

- checkpoint：`D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_rootfix_register_memory_halt_u100000_20260723\checkpoints\checkpoint_000100000.jls`
- checkpoint SHA-256：`3def11e92a8af52267f178e25bb59881731a53278dffe1d53dd98fb599b38f61`
- checkpoint size：253,765,189 bytes
- summary：同run directoryの`summary.json`

## 残る境界

4つの根本問題は学習経路と観測値の両方で解消した。ただし、今回の100,000更新は
benchmark-onlyであり、独立validationを評価していない。次に性能比較を行う場合は、
今回触れなかった評価panelを事前固定し、旧EVRLとPreActを同一teacher state予算または
同一wall-clock予算で比較する必要がある。
