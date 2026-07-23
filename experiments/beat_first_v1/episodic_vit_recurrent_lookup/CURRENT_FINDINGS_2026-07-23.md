# EVRL研究の現時点での判明事項

## 2026-07-24追補：現行固定K=64モデルを100,000更新

現行sourceは1段の共有LookupFFN、depthwise視覚経路、4 registerの再帰working
memoryに加え、register別learned WTA/hashで283 tokenから固定64 tokenを直接取得する。
正確なmulti-head cross-attentionとworking-memory writeは選択supportだけで実行し、
全283 tokenの正規化平均を安価な常設経路として残す。総parameter数は`6,954,877`である。

router LR `4e-4`を採用し、初期化からの学習系列を100,000更新、400,000 teacher stateまで
継続した。10k以降の90,000更新は7,310.216秒、最終記録`12.3088 updates/s`、
CPU平均`50.94%`、candidate中`53.36%`だった。steady benchmarkは
`9.067 MB/update`、GC占有率`0.83%`である。

同一training-only固定128状態で10k刻みを再評価した結果、100kがloss`2.602389`、
pairwise`0.903184`で最良、top-1`0.617188`も同率最良だった。NDCGは90kの
`0.989632`が最良で、100kは`0.989165`だった。平均深度は80kで`4.133`へ広がった後、
100kで`3.026`へ戻ったため、halting方策は入力依存性を持つが安定収束していない。

PreAct 12.75kとの差は、loss`+0.051484`、top-1`-0.171875`、
NDCG`-0.005303`、pairwise`-0.027019`、margin`+0.004736`である。
固定K64版は約7.84倍のteacher stateを使っても主要品質でPreActへ届かず、精度超えは
未達だった。一方、記録済みPreAct`0.4894 updates/s`に対し学習throughputは約25.2倍
だった。validationとsealed seedは使用していない。

完全な構成、正当性、全10 checkpoint、SHA-256、PreAct境界は
[`FIXED_K64_EPISODIC_ROUTING_2026-07-24.md`](FIXED_K64_EPISODIC_ROUTING_2026-07-24.md)
に記録した。

## 履歴：固定K導入前のsource geometry変更

固定K routing導入前に、再帰step内部のLookupFFNを3段から1段へ縮約し、24x10セル
working memoryへ共有3x3 depthwise convolutionを追加した。当時のparameter数は
`20,585,982 -> 6,954,621`、全4 registerのLookup呼出しはstepあたり`12 -> 4`である。
manual VJP有限差分とreal-teacherのserial／barrierless一致smokeは合格した。

直後に記録した以下の100,000更新結果は、この変更直前の3-block geometryに対する
確定済み研究結果であり、1-block固定K64 geometryの結果ではない。旧checkpointを
新geometryへ暗黙変換していない。詳細は
[`SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md`](SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md)
を参照。

## 最新追補：現行最終式をフルスクラッチで100,000更新

候補固有の各step 1-step信用割当と有界halting式を、既存checkpointからの短期armだけで
なく、初期値から共同学習した。親checkpointなしで100,000更新、400,000 teacher stateを
正常完了した。20 worker、pinningなし、barrierless chunk 8で`10.651 updates/s`、
CPU平均`54.255%`、candidate中`57.023%`、allocation`8.383 MB/update`、GC比率
`0.665%`だった。最低速度10 updates/sは維持した。

10k刻みの全checkpointを同じtraining-only固定128状態で追加学習なしに再評価した。
100kがloss`2.543041`、NDCG`0.993363`、pairwise`0.921549`で最良となり、
top-1`0.703125`、margin`0.123395`、平均深度`3.863`、深度範囲3～6を得た。
90kはtop-1`0.718750`で最高だったが、連続順位品質を優先して100kを主採用点とする。

100kの深度は3、4、5、6へそれぞれ3,314、330、847、866 candidateが分布し、入力依存
停止は維持した。一方、50～100kの平均深度は`4.891 -> 3.012 -> 3.304 -> 4.496 ->
3.026 -> 3.863`と動いたため、従来の両端崩壊は抑えたが平均深度の完全収束とは判定しない。

同じ固定パネルの旧90k主採用点に対し、loss`-0.008660`、top-1`+0.031250`、
NDCG`+0.001384`、pairwise`+0.008074`、margin`+0.006889`と全順位品質指標を改善した。
現行主採用checkpointのSHA-256は
`2eb9a64293eb2c592830d9c25ea52969950e6440a97415a43ec093d0a77c8766`
である。validationとsealed seedは使用していない。

完全な条件、全10 checkpoint、速度、深度分布、PreActとの境界は
[`HALTING_FULLSCRATCH_100K_2026-07-23.md`](HALTING_FULLSCRATCH_100K_2026-07-23.md)
に記録した。

## 最新追補：haltingを収束させた

根本修正版100kの10k刻み再評価では、平均深度が60～100kで
`2.820 -> 4.273 -> 2.000 -> 3.027 -> 4.880`と振動した。原因は、同じterminal
lossをtrajectory中の全stop／continueへ与える高分散な信用割当と、learned halt logitが
停止境界を無制限に移動できたことだった。

最終実装では、少数の1-step probe候補について各再帰stepのQだけを差し替え、同じ
ListNetとmarginから候補固有の`L_stop - L_continue`教師を作る。probe済み
trajectoryではterminal REINFORCEを重ねない。停止式は次のように有界化した。

```text
halt_logit = 0.5 * (step - 4) + 0.75 * tanh(learned_logit)
```

halt LRは60k以降`1e-6`、entropyは20kまでに0とする。同じ60k checkpointから20k
更新した結果、65/70/75/80kのtraining-only固定128状態の平均深度は
`3.001、3.066、3.006、3.036`となった。旧方式の同区間振幅`2.127`に対して
`0.065`、96.9%減である。NDCGは`0.991098 -> 0.991465`、pairwiseは
`0.906943 -> 0.911968`へ改善した。

20k更新速度は`11.184 updates/s`、GC占有率は`0.763%`だった。最終式で既存90kを
再評価するとloss`2.551701`、top-1`0.671875`、NDCG`0.991979`、pairwise
`0.913475`、平均深度`4.004`、範囲4～5だった。品質最良のため、主採用checkpointは
90k、SHA-256
`3ea19a64fd72521c1e679c53b348525194214db3073e0937e8e515c864e28c71`
を維持する。

validationとsealed seedは使用していない。完全な原因、失敗arm、数値一致smoke、全推移は
[`HALTING_CONVERGENCE_REPAIR_2026-07-23.md`](HALTING_CONVERGENCE_REPAIR_2026-07-23.md)
に記録した。

## 最新追補：4つの根本問題を修正した100,000更新

本書の従来の性能表は、長期記憶をregister間でpoolし、episodic memoryが読取り専用で、
Lookup collapseとhalting信用割当が未解決だった構成の履歴である。その後、次を修正した。

1. state-localなhard頻度とsoft確率によるLookup anti-collapse credit
2. 4 registerそれぞれの独立した長期記憶routing
3. 更新後registerから283 tokenへ戻す書込み可能なworking memory
4. 全trajectoryのpolicy gradientに少数one-step probeを加算するhalting信用割当

新構成はfrom-scratchで100,000更新、400,000 teacher stateを完走した。Lookup Giniは
`0.187～0.193`、coverageは`99.7～99.9%`、register間address相違率は`99.90%`、
working-memory write RMSは`0.384`となった。速度は`11.332 updates/s`、GC占有率は
`0.667%`で、今回許可された最低速度`10 updates/s`を上回った。

validationとsealed seedは使用していない。詳細、正当性smoke、固定batch過学習、最終
checkpointは[`ROOT_CAUSE_REPAIR_100K_2026-07-23.md`](ROOT_CAUSE_REPAIR_100K_2026-07-23.md)
に記録した。以下は改修前構成の比較履歴として保持する。

## 改修前構成の結論

現行EVRLは、入力内の短期関係形成、LookupFFNによる長期記憶選択、入力依存の再帰深度を一つのCPU学習系へ統合できている。8近傍spatial backwardの範囲外書込みを修正した後は、loss、top-1、NDCG、pairwise、marginが学習とともに改善し、hard haltingも2～12 stepを入力ごとに使い分けるようになった。

現在の速度条件付き採用構成は、state batch 4、dense LR `2e-4`、dense weight decay `3e-4`、halt LR `5e-5`、2 probes/stateである。この構成を100,000更新まで完走した。連続順位指標の均衡が最もよい採用checkpointは95,000更新で、top-1 `0.742188`、NDCG `0.991278`、pairwise `0.901964`、margin `0.157540`、平均深度 `2.205`、65,000更新以降の区間速度 `19.392 updates/s`を記録した。

50,000更新後にepisodic dense LRだけを半減した試行は、65,000更新でtop-1 `0.757812`、NDCG `0.990265`まで伸びた。しかし実データ10,000更新の速度が`14.625 updates/s`となり、最低条件の`15 updates/s`を満たさなかった。この65,000更新checkpointは品質上の参考として保持するが、現行採用構成にはしない。

PreActに対する結論は予算軸で異なる。同じteacher state数ではPreActが明確に優位であり、sample efficiencyのPreAct超えは実証できていない。一方、既存ログからほぼ同じ学習実時間を比較するとEVRLが同一panel上の全主要順位指標で優位だった。したがって現時点で実証できたのはwall-clock throughput上の優位であり、最終的な独立汎化性能のPreAct超えではない。

## 改修前アーキテクチャ

各候補はPreActと同じ盤面、候補、差分、NEXT/HOLD、`aux37`だけから独立に評価する。teacher Q値と順位は教師信号にのみ使う。

- 総parameter数：`20,577,789`（約20.58M）
- 入力記憶：240 cell token、6 NEXT/HOLD token、37 aux token、合計283 token
- 視覚経路：dilation `1, 2, 4, 8, 16`の5段depthwise/pointwise残差経路
- 視覚経路の受容野：`63 x 63`。`24 x 10`盤面全体を覆う
- 視覚経路の追加parameter：565
- register数：4
- model dim：128
- attention dim：32、4 heads
- register cross-attention：全283 tokenを直接評価
- cell間通信：8近傍のlearned Q/K/V/Oと相対位置bias
- LookupFFN：共有3 block、各block 13 table、各table 4,096 row、各tableからtop-3 rowを選択
- Lookup bank parameter：`20,451,739`
- 1 macro stepの選択row数：117
- hard halting：最小2、最大12 step
- 停止教師：candidate-localな少数1-step probe

入力tokenを`283 -> 64 -> 16`へ絞る旧hard routingは撤去済みである。短期記憶への入力経路は全token cross-attentionで確保し、物理的疎性は主にLookupFFNのparameter row選択、local spatial edge、active candidate、hard halting、active-only backward、sparse optimizerで維持する。

## 修正した学習阻害要因

8近傍spatial attentionのbackwardで、局所edge数8に対して`BackwardScratch.dweights`が長さ4しかなく、範囲外書込みが起きていた。scratch容量を`LOCAL_SPATIAL_NEIGHBORS = 8`に合わせて修正した。モデル構造、入力、teacher、loss、optimizer semantics、hard halting、checkpoint形式は変更していない。

修正後のreal-teacher serial／barrierless smokeは次の条件で合格した。

- 出力、loss、raw VJP：完全一致
- parameter gradient：許容誤差`1e-6`級
- optimizer更新後parameter：許容誤差`1e-7`級
- routing選択、RNG state、optimizer clock：完全一致

この不具合を含む旧100,000更新試行は研究履歴として残すが、現行モデルの性能根拠には使わない。

## 実行系の判明事項

最速の実行方式は、20 worker、pinningなし、barrierless global candidate queue、chunk size 8である。BLAS内部threadは1とする。

state batchを4から8へ増やすとCPU使用率は上がったが、実効throughputは改善しなかった。

| state batch | updates/s | states/s | CPU平均 | allocation/update | 判定 |
|---:|---:|---:|---:|---:|---|
| 4 | 15.505 | 62.020 | 49.64% | 7.739 MB | 採用 |
| 8 | 7.529 | 60.233 | 56.99% | 15.426 MB | 不採用 |

batch 8は最低`15 updates/s`を満たさず、states/sもbatch 4を下回った。したがって現行モデルではbatch 4を維持する。

Windows CPU Setsによる全コア固定は過去のscheduler比較で遅く、pinningなしを採用している。性能改善はCPU使用率だけでなく、実データ区間のupdates/sで判定する。

## 修正後の学習推移

修正後scratch基準は、batch 4、dense LR `2e-4`、dense WD `1e-4`、halt LR `5e-5`、halting cost 0、2 probes/state、5,000更新のrandom-depth warmupで開始した。

| 更新数 | loss | top-1 | NDCG | margin | 平均深度 | updates/s |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 2.867476 | 0.507812 | 0.978324 | 0.099650 | 2.000 | － |
| 10,000 | 2.810665 | 0.531250 | 0.979387 | 0.088824 | 2.107 | 15.506 |
| 15,000 | 2.758752 | 0.523438 | 0.981218 | 0.104578 | 2.396 | 15.343 |
| 20,000 | 2.752324 | 0.578125 | 0.982200 | 0.103318 | 3.556 | 15.592 |

lossとNDCGは全評価点で改善し、平均深度も2.00から3.56まで増えた。修正前に疑われた「入力を無視して一様予測へ停滞する状態」は解消した。

## halt LRの比較

halt LRだけを下げても、再帰深度の振動は解消しなかった。

| halt LR | 更新数 | loss | top-1 | NDCG | margin | 平均深度 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `5e-5` | 25,000 | 2.710472 | 0.578125 | 0.982187 | 0.122002 | 2.001 | 基準 |
| `1e-5` | 30,000 | 2.691209 | 0.632812 | 0.983490 | 0.116039 | 2.647 | 継続観測 |
| `1e-5` | 35,000 | 2.671507 | 0.671875 | 0.985029 | 0.132433 | 11.713 | 最大深度側へ飽和 |
| `1e-5` | 40,000 | 2.644841 | 0.656250 | 0.987263 | 0.134669 | 2.131 | 深度が反転 |
| `1e-6` | 30,000 | 2.692056 | 0.632812 | 0.983332 | 0.120413 | 2.709 | 明確な優位なし |

`1e-5`は品質を改善したが、平均深度が`2.647 -> 11.713 -> 2.131`と両端を往復した。`1e-6`も短区間では安定化を証明できなかった。halt LR縮小だけを解決策には採用しない。

## dense weight decayの比較

同じ20,000更新checkpointからdense WD `1e-4`と`3e-4`を比較した。WD変更を許可する明示的checkpoint遷移を追加し、serial／barrierless smoke合格後に実行した。

| dense WD | 更新数 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | 30,000 | 2.692367 | 0.632812 | 0.983506 | 0.866969 | 0.118709 | 3.372 | 15.120 |
| `3e-4` | 30,000 | 2.689768 | 0.625000 | 0.983862 | 0.867854 | 0.117013 | 2.110 | 15.532 |
| `3e-4` | 35,000 | 2.664932 | 0.695312 | 0.985194 | 0.875882 | 0.141134 | 3.147 | 15.241 |
| `3e-4` | 40,000 | 2.645313 | 0.671875 | 0.988018 | 0.881705 | 0.134168 | 2.178 | 15.557 |
| `3e-4` | 45,000 | 2.622360 | 0.718750 | 0.988368 | 0.884885 | 0.125085 | 2.131 | 15.191 |
| `3e-4` | 50,000 | 2.632145 | 0.734375 | 0.987706 | 0.884073 | 0.131590 | 4.010 | 15.545 |

WD `3e-4`は、15 updates/s以上を維持しながら品質を改善し、halt LR `1e-5`で発生した最大深度側への長時間飽和も回避した。45,000更新はloss、NDCG、pairwiseの均衡がよく、50,000更新はtop-1と入力依存深度が最良だった。後続試行の共通分岐点には50,000更新を使い、最終的な速度条件付き採用点は下記の65,000更新対照で更新した。

## 50,000更新後のepisodic LR半減

50,001更新目から、attention、FFN、token、register、headの既存episodic LR scaleだけを0.5へ下げた。Lookup bank、router、lookup alpha、halt LRは変更していない。

| 更新数 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 区間速度 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 55,000 | 2.612962 | 0.718750 | 0.988942 | 0.890677 | 0.132410 | 4.287 | 15.866* | 短期定常benchmark合格 |
| 60,000 | 2.607455 | 0.703125 | 0.988899 | 0.889694 | 0.147984 | 2.113 | 15.206 | 合格 |
| 65,000 | 2.594471 | 0.757812 | 0.990265 | 0.896571 | 0.142645 | 2.888 | 14.625 | 速度条件不合格 |

`*` 55,000更新の値は100更新warmup後の1,000更新benchmarkである。実データ10,000更新で測ると65,000更新までの平均は`14.625 updates/s`だった。

この試行は現時点で最高の品質を得たが、速度条件を満たさない。短いbenchmarkだけで採用せず、長い実データ区間を最終判定に使う。

同じ50,000更新checkpointからepisodic LR scale `1.0`を維持した対照は、65,000更新でloss `2.605426`、top-1 `0.742188`、NDCG `0.990088`、pairwise `0.893202`、margin `0.140937`、平均深度`3.217`に到達した。15,000更新の実時間は876.050秒、`17.122 updates/s`、`68.489 states/s`、CPU平均`53.759%`、candidate中`54.478%`であり、速度条件に合格した。

65,000更新のLR半減armは対照比でloss `-0.010955`、top-1 `+0.015625`、NDCG `+0.000177`、pairwise `+0.003369`、margin `+0.001708`と高品質だった。一方、速度は`14.625`対`17.122 updates/s`で、対照の`85.4%`に留まった。したがってLR半減は品質上限の参考、scale 1.0は速度条件付き採用点とする。

## 現在の採用checkpoint

- run：`evrl_boundsfix_p2_c0_halt5e5_lr2e4_full65k_wd3e4_u100000_20260723_long1`
- 更新数：95,000
- teacher state数：380,000
- SHA-256：`622753d65e0502edd09ed4ef10ecf3270b90e7a92e7d9b2d44cb6d71d9350893`
- 採用理由：65k比でloss、NDCG、pairwise、marginをすべて改善し、top-1を維持し、30,000更新の実データ区間で`19.392 updates/s`を満たすため

参考用の65,000更新LR半減checkpointのSHA-256は`672d73a6a5cf1c40f1fa80fd10fd28b34bdd09e3465dcaf4c24602c9181e53f2`である。これは速度失格のため採用checkpointではない。

100,000更新checkpointのSHA-256は`6ebb042a3678d9445ed76499730be15e63527b716c7c77322b2e8fca00cc4b45`である。100kはloss `2.601427`、top-1 `0.726562`、NDCG `0.989715`、pairwise `0.897215`となり、95kから反落したため最終更新という理由だけでは採用しない。

## 65,000～100,000更新の推移

episodic LR scale `1.0`を維持し、モデル、入力、teacher、loss、optimizer、hard haltingを変えずに65kから100kまで継続した。

| 更新数 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 | 区間updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,000 | 2.605426 | 0.742188 | 0.990088 | 0.893202 | 0.140937 | 3.217 | 2～12 | 17.122 |
| 70,000 | 2.610522 | 0.687500 | 0.989263 | 0.891757 | 0.156633 | 2.162 | 2～12 | 18.632 |
| 75,000 | 2.598116 | 0.695312 | 0.989688 | 0.896499 | 0.134000 | 2.667 | 2～12 | 19.092 |
| 80,000 | 2.600593 | 0.718750 | 0.990099 | 0.897866 | 0.122390 | 2.120 | 2～12 | 19.262 |
| 85,000 | 2.593664 | 0.750000 | 0.990198 | 0.897315 | 0.140141 | 2.013 | 2～7 | 19.370 |
| 90,000 | 2.586097 | 0.718750 | 0.990796 | 0.902023 | 0.148828 | 3.020 | 2～12 | 19.374 |
| 95,000 | 2.585896 | 0.742188 | 0.991278 | 0.901964 | 0.157540 | 2.205 | 2～4 | 19.392 |
| 100,000 | 2.601427 | 0.726562 | 0.989715 | 0.897215 | 0.150161 | 2.051 | 2～12 | 19.402 |

85kはtop-1、90kはpairwise、95kはloss、NDCG、marginが最良だった。100kでは複数指標が同時に反落したため、90～100kは単調改善ではなく振動を伴う頭打ち域と判定する。平均深度も`2.013～3.020`を往復しており、入力依存停止は残るが、深度が安定して増える学習にはなっていない。

この改修前構成ではLookup row利用が未解決だった。100k時点の各blockのrow-load Giniは`0.966～0.970`と高く、容量利用は強く偏っていた。後続の根本修正後100kではGiniが`0.187～0.193`へ低下し、このcollapseは解消した。

## PreActとの関係

既存PreActの単一checkpoint品質基準には12,750更新、51,000 teacher stateを使う。この点はtop-1 `0.789062`、NDCG `0.994468`、pairwise `0.930203`、margin `0.116617`、loss `2.550905`だった。EVRL 95kは380,000 teacher stateを使っているため、最終品質同士を同一sample予算の勝敗として扱うことはできない。

sample efficiencyではPreActが優位である。PreActは48,000 teacher stateの12kでtop-1 `0.789062`、NDCG `0.993292`、loss `2.563784`に達した。EVRLは40,000 statesの10kでtop-1 `0.531250`、NDCG `0.979387`、loss `2.810665`、60,000 statesの15kでもtop-1 `0.523438`、NDCG `0.981218`、loss `2.758752`だった。

wall-clockでは逆になる。EVRL 95kの累積学習時間は5,817秒だった。PreAct 3kは記録速度`0.4894 updates/s`から約6,130秒で、EVRLより約5.4%長い。このほぼ同時間比較では、EVRL 95k対PreAct 3kの順に、top-1は`0.742188`対`0.703125`、NDCGは`0.991278`対`0.989252`、pairwiseは`0.901964`対`0.893250`、lossは`2.585896`対`2.715674`だった。EVRLは大きいモデルと多いteacher state処理によって、同時間内の品質で上回った。

最高品質同士ではPreAct 12.75kがEVRL 95kよりtop-1 `+0.046875`、NDCG `+0.003189`、pairwise `+0.028240`、loss `-0.034991`と高い。EVRLはmarginだけ`+0.040924`高かった。したがって、速度優位は確認できたが、sample efficiencyと到達品質のPreAct超えは未達である。

したがって現在の正確な結論は次の通りである。

1. 修正後EVRLは学習可能であり、動的再帰も獲得している。
2. dense WD `3e-4`、episodic LR scale `1.0`の95,000更新が、現在の品質・速度条件を満たす採用点である。
3. episodic LR半減は同一65k比較では品質を伸ばしたが、長区間速度を悪化させた。
4. 同一teacher state数と到達品質ではPreActが優位、ほぼ同一wall-clockではEVRLが優位だった。
5. 改修前構成の90～100kは振動を伴う頭打ち域であり、そこで特定したrouting collapseと深度信用割当は後続の根本修正で改善した。

## 評価上の境界

- sealed game seedには触れていない。
- チューニングには固定training teacher panel 64状態と固定validation teacher panel 128状態を繰り返し使っている。
- validation panelを反復観測しているため、上記の改善は学習挙動と同一panel上の比較証拠ではあるが、独立した汎化性能の最終証明ではない。最終比較には未使用の評価seedまたは複数の新規固定seedが必要である。
- 旧100,000更新試行は範囲外書込みを含むため、修正後モデルの性能根拠から除外する。

詳細な各試行の推移、checkpoint遷移、smoke結果は[`TRAINING_STABILITY_TUNING_2026-07-23.md`](TRAINING_STABILITY_TUNING_2026-07-23.md)に記録している。
