# 単一Lookup＋DWConv版のボトルネック実測

## 目的

LookupFFNを3 blockから1 blockへ縮約したにもかかわらず、短時間smokeの速度が
`12.949 updates/s`に留まり、単純な3倍や2倍へ達しなかった理由を、保存なしの
phase別benchmarkで特定する。

新しい120更新checkpointから再開し、100更新をwarmup、続く100更新だけを計測した。
benchmark更新はcheckpointへ保存していない。validation evaluationとsealed seedは
使用していない。

## 実測条件

- single Lookup block
- recurrent 3x3 depthwise convolution
- 4 register
- 全283 token exact cross-attention
- read/write episodic working memory
- barrierless global candidate queue
- 20 worker、pinningなし、chunk 8
- BLAS 1 thread
- allocation sampling rate 1%

## 全体結果

| 指標 | 結果 |
|---|---:|
| 計測更新 | 100 |
| training時間 | 6.767秒 |
| updates/s | 14.778 |
| states/s | 59.113 |
| candidates/s | 2,633.190 |
| recurrent steps/s | 8,467.056 |
| 平均CPU使用率 | 65.636% |
| candidate中CPU使用率 | 67.024% |
| allocation | 6.620 MB/update |
| GC時間 | 0.0447秒 |
| GC／overall wall | 0.648% |

20更新warmupの最初のsmokeは`12.949 updates/s`だった。100更新warmup後は
`14.778 updates/s`であり、冷間compileと初期cacheの影響が残っていた。

## phase別内訳

20 workerのaggregate wallを20で割ったcapacity-normalized worker時間では、主要部分は
次のようになった。この値は各workerの稼働・待機時間を合計した診断値であり、逐次的な
実wallの内訳ではない。

| phase | 100更新のcapacity-normalized時間 | 主要worker phase内の割合 |
|---|---:|---:|
| recurrent forward | 1.841秒 | 30.6% |
| backward | 2.485秒 | 41.3% |
| queue wait | 1.582秒 | 26.3% |
| prepare | 0.053秒 | 0.9% |
| loss VJP | 0.041秒 | 0.7% |
| gradient reduce | 0.009秒 | 0.1% |
| optimizer | 0.010秒 | 0.2% |

executor CPU時間では、forwardが32.219秒、backwardが44.922秒であり、この2 phaseだけで
executor CPU時間の約94%を占めた。optimizerとGCはボトルネックではない。

allocation samplingでは、backwardが推定3.176 MB/update、forwardが
0.478 MB/updateだった。全体allocationは6.620 MB/updateだが、GC比率は0.65%なので、
現時点の速度を支配しているのはGCではなく実演算とworker tailである。

## 何の演算が大きいか

一つのcandidate・一つのrecurrent stepについて、主要なforward scalar MACを
production geometryから概算すると次のようになる。

| 経路 | 概算scalar MAC/step |
|---|---:|
| 全283 token cross read | 2,423,552 |
| registerから283 tokenへのworking-memory write | 1,211,776 |
| recurrent 3x3 DWConv | 276,480 |
| local-8 spatial attentionと構造化Q/K/V/O | 約232,960 |
| register SwiGLU | 196,608 |
| register self-attention | 約69,632 |
| 1 block Lookup selected-row gather | 19,968 |

Lookupの値は、13 table x top-3 row x 128 value x 4 registerである。WTA routing、
BH4、softmax、prefetchの費用は別に存在するが、bank全体6.8M parameterを積和している
わけではない。

これに対し、全token cross readは各stepで283 tokenのK/Vを投影し、reverse writeは
283 token全てへ出力投影する。両者だけで約3.64M MAC/stepになり、selected-row gatherの
約182倍である。さらにbackwardでは、入力勾配とparameter勾配のため同じtoken領域を
複数回走査する。

したがって、総parameter数を66.22%削減してもwall-clockが同率で減らない。削除した
bank parameterは旧版でも未選択時にはforward、backward、optimizerのいずれも実行
されていなかったためである。今回のparameter削減が直接効くのはRAM、checkpoint、
cache footprint、active routing回数であり、全step演算の3分の2ではない。

## 旧3-blockとの実測差

旧3-block full-scratch 100kの保存済み詳細統計と比較した。run長と学習trajectoryが
異なるため厳密なpaired benchmarkではないが、recurrent stepあたりへ正規化することで
傾向を確認できる。

| 指標 | 旧3-block | 新1-block＋DWConv | 変化 |
|---|---:|---:|---:|
| updates/s | 10.668 | 14.778 | 1.385倍 |
| candidates/s | 1,863.062 | 2,633.190 | 1.413倍 |
| recurrent steps/s | 5,740.915 | 8,467.056 | 1.475倍 |
| forward CPU µs/step | 637.6 | 562.3 | -11.8% |
| backward CPU µs/step | 1,000.5 | 784.1 | -21.6% |
| queue CPU µs/step | 74.1 | 59.5 | -19.8% |

速度向上は存在し、特にbackwardとstep throughputへ効いている。しかしLookup以外の
full-token read/write、spatial path、SwiGLU、およびそのVJPが残るため2倍には届かない。

## 次のボトルネック

優先順は次のとおりである。

1. 全283 tokenのK/V projectionとreverse-write output projection
2. それらを含むcandidate-local backward
3. 可変再帰深度とcandidate数によるbackward tail、queue wait
4. recurrent DWConvとlocal spatial attentionのcell-memory走査
5. Lookup routingとselected-row gather
6. gradient reduce、optimizer、GC

数学モデルを維持したまま改善するなら、最初にcross K/Vの融合、working-memory write
projectionとのmemory scan共有、cross/read-write VJPの融合を検討する。次にbackward
tailだけをより細かいjobへ分割する。入力token routingを再導入したり、283 tokenを
削ったりしなくても、同じdense semanticsのままprojectionとmemory scanを融合する余地が
ある。

## 結論

Lookup削減後の定常速度は短時間初報の`12.949`ではなく`14.778 updates/s`で、旧3-block
長期runの`10.668`に対して約1.39倍だった。2倍へ届かない主因は、Lookupがもともと
active-onlyで安価だった一方、全token read/write attentionとそのbackwardが残り、
追加DWConvも毎step全240セルを処理するためである。CPU使用率を67%に留めるqueue tailも
第二の制限になっている。
