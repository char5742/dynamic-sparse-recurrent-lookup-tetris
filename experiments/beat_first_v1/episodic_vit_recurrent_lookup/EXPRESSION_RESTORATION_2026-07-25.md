# 高速executor上での表現力段階復元

## 目的

2026-07-24の速度優先構成は、2 register、attention 16／1 head、SwiGLU 32、
`depthwise_only`まで縮小することで、動的haltingを維持したまま100,000更新を
`27.732 updates/s`で完走した。一方、固定training-onlyパネルの100,000更新品質は
loss `2.734835`、top-1 `0.585938`、NDCG `0.980287`、pairwise `0.863809`、
margin `0.079808`であり、縮小前の固定K64構成より低かった。

本試行では高速executor、入力、teacher、task loss、LookupFFN、hard halting、
active-only backward、sparse optimizerを固定し、次の順で表現力を戻した。

1. learned spatial attention
2. register数
3. register attentionの次元とhead数
4. SwiGLU幅

採用条件は、学習済み状態のsteady測定で20 updates/s以上を維持し、速度優先構成より
順位品質を回復することとした。固定深度は速度上限の参考測定にだけ使用し、採用モデルは
全て`fixed_depth=0`のsampled hard haltingである。

## 固定した条件

- 入力：PreActと同じboard、candidate、difference、NEXT/HOLD、`aux37`
- teacher：既存real-teacher training split
- state batch：4
- LookupFFN：1 block、13 tables、4,096 rows/table、top-3 active rows
- episodic support：registerごとに固定K=64
- visual stem：5段dilated depthwise／pointwise residual
- register projection：structured
- scheduler：barrierless、pinningなし、forward chunk 8、backward chunk 1
- Julia：20 native workers、BLAS 1 thread
- halting：2～12 step、候補固有の1-step probe付きsampled hard hazard
- validation／sealed seed：未使用

## 段階速度試験

最初に固定深度2でstep単価を測り、次に実際のdynamic haltingで20 warmup＋100更新を
測定した。固定深度値は採用性能ではない。

### 固定深度2の参考値

| learned spatial | registers | attention | SwiGLU | updates/s |
|---|---:|---:|---:|---:|
| 有効 | 2 | 16／1 head | 32 | 36.422 |
| 有効 | 4 | 16／1 head | 32 | 29.597 |
| 有効 | 4 | 32／4 heads | 32 | 28.488 |
| 有効 | 4 | 32／4 heads | 64 | 27.340 |
| 有効 | 4 | 32／4 heads | 128 | 23.902 |

固定深度では全て20を超えたが、これは入力ごとの再帰深度差とbackward tailを含まない。

### dynamic haltingの実測

| registers | attention | SwiGLU | updates/s | 平均深度 | 判定 |
|---:|---:|---:|---:|---:|---|
| 2 | 16／1 head | 32 | 24.247 | 3.335 | 合格 |
| 3 | 16／1 head | 32 | 21.150 | 3.268 | 合格 |
| 4 | 16／1 head | 32 | 19.630 | — | 不合格 |
| 4 | 32／4 heads | 32 | 16.943 | — | 不合格 |
| 3 | 16／1 head | 64 | 22.202 | 3.240 | 合格 |
| 3 | 16／1 head | 128 | 21.107 | 3.296 | 合格 |
| 2 | 32／4 heads | 64 | 19.326 | — | 不合格 |
| 2 | 32／4 heads | 128 | 18.077 | — | 不合格 |

4 registerはattentionを16／1 headに留めても速度下限を割った。attention
32／4 headsも2 registerで20を割った。従って、当初候補の4 register、
attention 32／4 heads、SwiGLU 64～128を全て同時に戻すことは、現在のexecutorでは
速度条件と両立しない。

3 register、attention 16／1 headではSwiGLU 64と128の両方が短時間測定を通過した。
両者を10,000更新まで学習すると、FFN64はFFN128よりlossとmarginが僅かに良く、
速度も高かったため、最小のFFN64を長期候補に選んだ。

## 採用した構成

| 項目 | 速度優先100k | 表現力復元100k |
|---|---:|---:|
| learned spatial attention | 無効 | **有効** |
| registers | 2 | **3** |
| attention | 16／1 head | 16／1 head |
| SwiGLU | 32 | **64** |
| recurrent Lookup blocks | 1 | 1 |
| episodic support/register | 64 | 64 |
| 総parameter数 | 6,897,248 | **6,909,665** |

復元構成では、共有Q/K/V/Oと相対位置biasを持つ物理的local-8 learned spatial
attentionを再び使用する。3個のregisterはそれぞれ別のepisodic supportとLookup rowを
選び、同じsupportへworking-memory writeを行う。全283 token平均の安全経路、
単一Lookup block、dynamic haltingは維持する。

## 数値一致smoke

10,000更新checkpointから、同じreal-teacher 4状態についてserial oracleと
barrierless executorを1更新比較した。

- 出力、loss、raw VJP：完全一致
- candidate seed、hard halting、選択token edge、Lookup row、probe教師：完全一致
- parameter gradient：最大絶対差`6.9514e-6`、相対L2`7.2736e-6`
- optimizer後parameter state：最大絶対差`6.9477e-7`、相対L2`1.5562e-8`
- optimizer clock、RNG、sampler、sparse row clock：一致

smokeは合格し、モデルの数学、RNG順序、active-only更新、checkpoint互換性を
変えていないことを確認した。

## 100,000更新

スクラッチから10,000更新を行い、そのcheckpointから構造とoptimizer stateを変えずに
100,000更新まで継続した。合計400,000 teacher stateを消費した。

- 10k→100k学習時間：`4,522.717秒`
- 10k→100k平均速度：`19.900 updates/s`
- 全体CPU使用率：`53.879%`
- candidate処理中CPU使用率：`56.861%`
- 100k checkpoint SHA-256：
  `324492322bf441c9c3b69767e8d997545d0a4b5383e50b0fd431ae12ad9b456f`

長期区間平均は20を`0.100 updates/s`下回った。コンパイルを除いた100k
checkpointの20 warmup＋100更新steady測定は次のとおりである。

- `20.301 updates/s`
- 平均CPU使用率`55.289%`
- candidate処理中CPU使用率`57.533%`
- 平均深度`2.959`、範囲2～7
- allocation `12.849 MB/update`
- GC占有率`1.178%`

従って学習済み定常状態は20を超えるが、90,000更新の長期区間平均まで厳密に
20以上だったとは主張しない。

## 10k刻みの推移

training splitから固定した同じ128状態を一括評価した。パネル行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.926994 | 0.492188 | 0.970763 | 0.821365 | 0.077335 | 3.329 | 3～5 |
| 20,000 | 2.861632 | 0.468750 | 0.976012 | 0.831129 | 0.060971 | 3.007 | 3～5 |
| 30,000 | 2.796405 | 0.476562 | 0.977364 | 0.838743 | 0.067473 | 3.555 | 3～6 |
| 40,000 | 2.786019 | 0.484375 | 0.976802 | 0.842703 | 0.075003 | 3.000 | 3～3 |
| 50,000 | 2.776928 | 0.500000 | 0.977742 | 0.850335 | 0.082441 | 3.011 | 3～6 |
| 60,000 | 2.753035 | 0.507812 | 0.980581 | 0.854320 | 0.072993 | 3.001 | 3～5 |
| 70,000 | 2.739559 | 0.492188 | 0.984016 | 0.863892 | 0.078314 | 3.255 | 3～6 |
| 80,000 | 2.709284 | 0.531250 | 0.984102 | 0.868035 | 0.077787 | 3.014 | 3～6 |
| 90,000 | 2.687116 | 0.554688 | **0.985523** | **0.875144** | 0.090514 | 3.025 | 3～5 |
| 100,000 | **2.686149** | **0.578125** | 0.984347 | 0.874771 | **0.097056** | 3.023 | 3～6 |

loss、top-1、marginは100kが最良、NDCGとpairwiseは90kが最良だった。90kから
100kでtop-1とmarginは改善し、連続順位品質は僅かに反落した。用途をtop-1に限定せず、
100kを主checkpoint、90kを連続順位品質checkpointとして併記する。

## 速度優先構成との比較

| 指標 | 速度優先100k | 表現力復元100k | 差 |
|---|---:|---:|---:|
| loss | 2.734835 | **2.686149** | **-0.048686** |
| top-1 | **0.585938** | 0.578125 | -0.007813 |
| NDCG | 0.980287 | **0.984347** | **+0.004060** |
| pairwise | 0.863809 | **0.874771** | **+0.010962** |
| margin | 0.079808 | **0.097056** | **+0.017248** |
| steady updates/s | 27.73 | 20.30 | -7.43 |

top-1は固定128状態中1件分低いが、loss、NDCG、pairwise、marginは全て改善した。
従ってlearned spatial attention、追加register、SwiGLU幅の復元は、速度優先化で失った
順位表現力を部分的に回復した。ただし縮小前4-register構成の100k
（loss`2.602389`、top-1`0.617188`、NDCG`0.989165`、pairwise`0.903184`、
margin`0.121353`）までは戻っていない。

## PreActとの差

既存PreAct最高品質点は12,750更新、51,000 teacher stateである。

| 指標 | 表現力復元100k | PreAct 12.75k | EVRL－PreAct |
|---|---:|---:|---:|
| loss | 2.686149 | 2.550905 | +0.135244 |
| top-1 | 0.578125 | 0.789062 | -0.210937 |
| NDCG | 0.984347 | 0.994468 | -0.010121 |
| pairwise | 0.874771 | 0.930203 | -0.055432 |
| margin | 0.097056 | 0.116617 | -0.019561 |

PreActの記録済み学習速度`0.4894 updates/s`に対し、EVRLのsteady速度は約41.5倍である。
ただしEVRLは400,000 teacher stateを使っており、sample efficiencyと到達品質の
PreAct超えは実証していない。

## artifact

主checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_expression_restore_r3_a16h1_f64_dynamic_u010000_u100000_20260725_r2\
checkpoints\checkpoint_000100000.jls
```

主checkpoint SHA-256：

```text
324492322bf441c9c3b69767e8d997545d0a4b5383e50b0fd431ae12ad9b456f
```

全checkpointの機械可読評価：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_expression_restore_r3_a16h1_f64_dynamic_u010000_u100000_20260725_r2\
halting_eval_fixed128_all.json
```

## 結論

現在の20 updates/s制約では、表現力復元の実測可能な境界は
**learned spatial attention＋3 registers＋attention 16／1 head＋SwiGLU64**だった。
4 registersとattention 32／4 headsは実装済みだが、dynamic halting下で速度条件を
満たさない。

採用候補は速度優先100kよりtop-1が1件分低い一方、lossと全ての連続順位指標を改善し、
100k学習済み状態で`20.301 updates/s`を確認した。従って「精度を完全に戻した」とは
結論せず、**速度下限を守った部分的な表現力回復**として採用する。次に4 registerと
32／4 headsを戻すには、モデルを再び削るのではなく、register read/writeと
spatial-attention VJPのexecutorをさらに高速化して速度余裕を作る必要がある。
