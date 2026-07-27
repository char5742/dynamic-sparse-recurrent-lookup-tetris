# 幅・深さバランス調整と100,000更新

## 結論

速度下限を`15 updates/s`へ緩和し、STE／本体勾配分離修正後のEVRLについて、
register数、attention幅・head数、SwiGLU幅、Lookup段数を比較した。

広げた候補のうち、数値一致、長期安定性、速度下限をすべて満たした最良構成は次である。

```text
registers                  3
attention dim / heads     24 / 3
SwiGLU width              64
Lookup blocks / step       1
model dim                128
episodic candidates / K 128 / 64
recurrent depth          2--12（hard dynamic halting）
```

この構成は6,919,987 parameterで、スクラッチから100,000更新、400,000 teacher
stateを学習した。累積学習時間は`5,017.156秒`、累積速度は
`19.932 updates/s`であり、許容下限15を満たした。30kから100kの区間は
`3,425.291秒`、`20.430 updates/s`だった。

ただし、**全候補を含む性能最適点は従来の細い2-register構成のままである**。
今回の3-register構成は100kでpairwiseとmarginを僅かに改善したが、lossとtop-1は
悪化し、速度も約20%低下した。したがって「使える最大幅」と「総合的な最適幅」を
区別し、productionのPareto基準は2-register、attention 16／1 head、SwiGLU32、
1段Lookupを維持する。

## 不変条件

今回変更したのはモデル容量に関する定数だけである。次は変更していない。

- board／candidate／difference／NEXT・HOLD／aux37という入力
- real-teacher、候補独立評価、teacher Qと順位の使い方
- ListNet、old-Q、margin、death、quantile、geometryからなるtask loss
- K64の物理的episodic read/writeとLookupのactive-only backward
- sparse optimizerとoptimizer stateの意味
- learned 8近傍spatial attention、depthwise視覚経路
- 最小2、最大12のhard dynamic haltingとstateあたり2候補の1-step probe
- barrierless、pinningなし、20 worker、BLAS 1 thread

validation row、game validation、sealed seedは使用していない。

## 短時間の速度選別

同じreal-teacher条件で各候補を120更新し、warmup後100更新の速度を測定した。

| registers | attention | SwiGLU | Lookup段数 | parameter数 | updates/s | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 24 / 3 heads | 64 | 1 | 6,919,987 | 21.077 | 長期比較へ |
| 4 | 24 / 3 heads | 64 | 1 | 6,920,116 | 19.159 | register追加の費用に対する優位なし |
| 3 | 32 / 4 heads | 96 | 1 | 6,942,460 | 19.345 | 長期比較へ |
| 4 | 32 / 4 heads | 96 | 1 | 6,942,589 | 18.264 | register追加の費用に対する優位なし |
| 3 | 24 / 3 heads | 64 | 2 | 13,736,244 | 18.159 | 深さ比較へ |
| 3 | 48 / 6 heads | 128 | 1 | 6,975,310 | 16.620 | 長期実行が不安定 |

最後の48幅候補は速度自体は15を上回ったが、長期起動の一回目でJulia／LLVM JITの
`EXCEPTION_ACCESS_VIOLATION`、再試行で進行停止を確認した。checkpointは作らず、
安定性条件により不採用とした。

## 10,000更新の同一パネル比較

評価パネルはtraining-only固定128状態、5,357候補である。パネル行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。

| 構成 | updates/s | loss | top-1 | NDCG | pairwise | margin |
|---|---:|---:|---:|---:|---:|---:|
| 既存 2R・16/1・F32・L1 | 25.230 | **2.933468** | 0.476562 | **0.976198** | **0.829715** | 0.052012 |
| 3R・24/3・F64・L1 | 19.480 | 2.942860 | **0.492188** | 0.972950 | 0.823317 | 0.062148 |
| 3R・32/4・F96・L1 | 18.367 | 2.950592 | **0.492188** | 0.973327 | 0.824351 | **0.063852** |
| 4R・32/4・F96・L1 | 18.264（短期） | 2.971697 | 0.468750 | 0.973480 | 0.824569 | 0.054930 |
| 3R・24/3・F64・L2 | 16.466 | 2.942881 | 0.445312 | 0.974652 | 0.826242 | 0.061277 |

同じ3R・24/3・F64でLookupを1段から2段へ増やしても、10kのlossは
`2.942860 -> 2.942881`で改善せず、top-1は`0.492188 -> 0.445312`へ低下した。
一方、parameter数はほぼ2倍、速度は`19.480 -> 16.466 updates/s`となった。
再帰block内部のLookupを深くする利益は、この条件では確認できない。

## 30,000更新での幅比較

1段Lookupの有力2候補だけを、各10k checkpointから30kまで同じoptimizer stateで
継続した。

| 構成 | 更新 | loss | top-1 | NDCG | pairwise | margin |
|---|---:|---:|---:|---:|---:|---:|
| 3R・24/3・F64 | 10k | 2.942860 | 0.492188 | 0.972950 | 0.823317 | 0.062148 |
|  | 20k | 2.881655 | 0.453125 | 0.975133 | 0.833926 | 0.056795 |
|  | 30k | 2.832357 | 0.484375 | 0.978921 | 0.845941 | 0.045416 |
| 3R・32/4・F96 | 10k | 2.950592 | 0.492188 | 0.973327 | 0.824351 | 0.063852 |
|  | 20k | 2.873987 | 0.468750 | 0.976904 | 0.835695 | 0.049107 |
|  | 30k | **2.816099** | **0.507812** | **0.979542** | **0.849522** | **0.047794** |

32幅候補は30k品質で24幅を上回った。しかしserial／barrierless smokeでは
gradient最大絶対差`7.85223e-4`、global relative L2`2.04038e-4`となり、
既存の許容値を超えた。大きかった箇所は`cell_bias`、token router、cross Q/Kである。
同じ誤ったVJPを共有する可能性を見逃さないため、この候補は品質が良くても不採用とした。

24幅候補は同じ30k smokeに合格したため、100kへ延長した。

## 採用候補の100,000更新推移

| 更新 | loss | top-1 | NDCG | pairwise | margin | 決定論的平均深度 |
|---:|---:|---:|---:|---:|---:|---:|
| 30k | 2.832357 | 0.484375 | 0.978921 | 0.845941 | 0.045416 | 3.000 |
| 40k | 2.792888 | 0.453125 | 0.974854 | 0.846913 | 0.069418 | 3.000 |
| 50k | 2.786536 | 0.539062 | 0.977394 | 0.850948 | 0.057814 | 3.000 |
| 60k | 2.738406 | 0.570312 | 0.984204 | 0.866202 | 0.050970 | 3.000 |
| 70k | 2.745030 | 0.539062 | 0.985134 | 0.871194 | 0.049470 | 3.000 |
| 80k | 2.743188 | 0.562500 | **0.985601** | **0.873287** | 0.059216 | 3.000 |
| 90k | 2.693041 | **0.578125** | 0.983475 | 0.870514 | 0.070898 | 3.000 |
| 100k | **2.679211** | **0.578125** | 0.983389 | 0.872891 | **0.085935** | 3.000 |

lossは100kまで低下し、top-1とmarginも最良または同率最良になった。NDCGとpairwiseは
80kが最良である。100kは離散top-1、loss、marginを優先する主比較点、80kは連続順位
品質の参考点とする。

学習時の最終batchは深度2～6へ分布したが、決定論的な固定パネルでは全checkpoint、
全5,357候補が深度3で停止した。halting interfaceは動的なまま学習したものの、
このパネルでは入力依存の決定論的深度を獲得していない。

## 既存の細いEVRLとの比較

| 指標 | 2R・16/1・F32・L1 100k | 3R・24/3・F64・L1 100k | 3R - 2R |
|---|---:|---:|---:|
| parameter数 | 6,897,248 | 6,919,987 | +22,739 |
| updates/s | 25.230 | 19.932 | -5.299 |
| loss | **2.669060** | 2.679211 | +0.010151 |
| top-1 | **0.593750** | 0.578125 | -0.015625 |
| NDCG | **0.985005** | 0.983389 | -0.001616 |
| pairwise | 0.871259 | **0.872891** | +0.001632 |
| margin | 0.085812 | **0.085935** | +0.000124 |

parameter増加は僅かだが、register read/write、cross-attention、SwiGLUのactive計算が
増えるため、速度はparameter数から予想される以上に低下した。その対価として得たのは
pairwiseとmarginのごく小さな改善だけであり、top-1は固定128状態で2件分低い。
従って現時点の幅最適点は細い2R構成である。

## PreActとの差

同じtraining-only固定パネルで記録済みのPreAct 12,750更新・51,000 teacher stateは、
loss`2.550905`、top-1`0.789062`、NDCG`0.994468`、
pairwise`0.930203`、margin`0.116617`である。

| 指標 | 3R幅拡張EVRL 100k | PreAct | EVRL - PreAct |
|---|---:|---:|---:|
| loss | 2.679211 | 2.550905 | +0.128306 |
| top-1 | 0.578125 | 0.789062 | -0.210938 |
| NDCG | 0.983389 | 0.994468 | -0.011079 |
| pairwise | 0.872891 | 0.930203 | -0.057312 |
| margin | 0.085935 | 0.116617 | -0.030682 |

幅を増やすだけではPreActとの差を縮められなかった。EVRLは400,000 teacher state、
PreActは51,000 stateであり、sample efficiencyも依然としてPreActが上である。

## 速度・allocation・GC

100k採用checkpointから同じ構成で120更新を詳細測定した。

| 項目 | 結果 |
|---|---:|
| steady updates/s | 20.924 |
| 平均CPU使用率 | 55.533% |
| candidate処理中CPU使用率 | 58.618% |
| allocation | 9.820 MB/update |
| executor allocation | 7.367 MB/update |
| data pack allocation | 2.319 MB/update |
| gradient reduce allocation | 79.84 KB/update |
| optimizer allocation | 29.95 KB/update |
| GC時間比 | 0.633% |

GCは速度の主因ではない。worker aggregateではrecurrent forward`26.355秒`、
backward`36.157秒`、queue wait`16.374秒`であり、引き続きbackwardとtailの
負荷不均衡が主要な実行時間を占める。

## 数値一致

100k checkpointに対して、同じreal-teacher 4状態、最大80候補で
serial／barrierless smokeを再実行した。

- 出力、loss、raw VJP：完全一致
- hard halting、selected token edge、Lookup row ID、probe、RNG、
  sampler、optimizer clock：全て完全一致
- worker gradient：最大絶対差`5.37634e-5`、relative L2`5.39971e-6`
- parameter gradient：最大絶対差`1.78218e-5`、relative L2`5.39971e-6`
- optimizer後parameter：最大絶対差`1.50010e-4`、relative L2`3.41137e-6`
- tolerance violation：全項目0

30k時点では全trajectory回帰も再実行し、task-only有限差分3/3、RMSNorm有限差分
8/8、STE／本体勾配分離643/643に合格した。router-only更新ではdistillation lossが
`2.725995 -> 2.725568`へ低下した。

## 最終判断

1. 再帰step内のLookup深度は1段が最適である。
2. strict correctnessを満たす拡幅上限は3R・24/3・F64である。
3. その拡幅上限は約20 updates/sで安定学習できるが、旧2R構成を総合品質で超えない。
4. 現在のPareto最適は2R・16/1・F32・L1である。
5. 32幅候補は品質上有望でも数値一致未達なので、reduce順序またはscratch境界を
   修正しない限り採用しない。
6. 幅・深さの追加探索を続けるより、haltingの候補固有信用とbackward tailを改善する
   方が、品質と速度の双方に対して次の優先課題である。

## 成果物

- 100k checkpoint：
  `D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\`
  `tune_r3_a24h3_f64_l1_u30k_u100k_20260727\checkpoints\checkpoint_000100000.jls`
- checkpoint SHA-256：
  `38efe79fb7a5b843fe1044342637ace7d64095463f08f9e4a6814304cef80035`
- 10k刻み評価JSON SHA-256：
  `3af34210efd814b1da94db9efaa5f524468a40b9a6a8d214b72bed3c5c6a2f62`
- 詳細速度summary SHA-256：
  `e478765140d2987a2dbc8860d62443a02d7414910f3d040f0d2c3e63853712f1`

