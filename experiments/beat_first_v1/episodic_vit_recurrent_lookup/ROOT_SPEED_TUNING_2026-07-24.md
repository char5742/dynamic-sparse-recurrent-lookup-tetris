# K幅変更後の根本速度改善

## 目的

固定Kの値だけを狭めても速度が十分に伸びなかったため、Lookup bank容量と
active-only更新を保ったまま、再帰step内の実演算を削減した。採用条件は
定常`40 updates/s`を目標、動的halting時も`20 updates/s`を下限とした。

benchmarkはすべて実teacher training splitだけを使用した。checkpointは保存せず、
validationとsealed seedを評価へ使用していない。

## 根本修正

速度改善はKの追加縮小ではなく、次の経路へ行った。

1. spatial Q/K/V射影を一回のcell走査へ融合
2. register cross/self attentionの射影を、学習可能なblock構造射影へ変更可能にした
3. recurrent spatial updateを、local attention＋DWConvとDWConv単独から選べるようにした
4. working-memory read/writeの中間表現をfactorizeし、重複した全幅変換を除いた
5. register数、attention幅、head数、SwiGLU幅を同一構造の範囲で縮小した
6. forwardのqueue chunk 8は維持し、重いbackward tailだけchunk 1へ細分化した
7. worker完了順に依存していたscalar勾配を候補別bufferへ保存し、更新境界で
   canonical state/candidate順に再加算した

LookupFFNは引き続き1再帰stepにつき1 block、13 table、各table top-3 rowである。
bankは6,815,744 parameterを保持し、parameter routing、active-only backward、
sparse optimizerの意味は変更していない。

## 同一条件の速度試行

各行は4 state/update、20 Julia worker、BLAS 1 thread、barrierless、pinningなし、
forward chunk 8、固定深度2の100更新benchmarkである。`定常updates/s`はcompileと
warmup後のtraining区間、`計測付きupdates/s`はphase/allocation計測を含む100更新全体を
表す。

| 構成 | parameter | 定常updates/s | 計測付きupdates/s | 判定 |
|---|---:|---:|---:|---|
| 2 register、attention 32、4 head、FFN 128 | 6,954,619 | 34.237 | 32.424 | 不採用 |
| 2 register、attention 32、4 head、FFN 64 | 6,930,043 | 36.833 | 34.896 | 不採用 |
| 2 register、attention 32、2 head、FFN 64 | 6,930,025 | 37.251 | 34.338 | 不採用 |
| 2 register、attention 32、2 head、FFN 32 | 6,917,737 | 38.262 | 35.252 | 不採用 |
| 2 register、attention 16、2 head、FFN 32 | 6,897,257 | 39.517 | 35.426 | 次点 |
| 2 register、attention 16、1 head、FFN 32 | 6,897,248 | 42.060 | 37.673 | 採用候補 |
| 上記＋backward chunk 1 | 6,897,248 | 40.523 | 38.205 | 計測付き採用 |
| 上記＋計測instrumentationなし | 6,897,248 | **41.545** | ― | production採用 |
| 10 table、18 WTA choices | 7,546,464 | 41.522 | 38.260 | 容量増に速度利得なし |

`MODEL_DIM=64`は計測付き`41.438 updates/s`を得たが、serial/barrierlessの
gradientおよびoptimizer state一致に失敗したため不採用とした。速度だけを理由に
数値不一致の構成を採用していない。

旧120k checkpointの幅を維持した実行モード比較では、dense register projection＋
DWConv単独が`18.442 updates/s`、structured projection＋DWConv単独が
`20.297 updates/s`だった。この結果から、Kよりもcross/self projectionと
register/FFN幅が支配的であることを確認した。

## 採用候補

```text
model/carrier dim             128
registers                       2
attention dim                  16
attention heads                 1
register projection    structured
SwiGLU FFN dim                 32
spatial recurrent path  depthwise_only
episodic support K             64 / register
Lookup blocks                   1 / recurrent step
Lookup tables                  13
rows selected per table         3
WTA choices                    16
forward queue chunk             8
backward queue chunk            1
Julia workers                  20
CPU Sets                  disabled
BLAS threads                    1
total parameters        6,897,248
```

固定深度2のproduction相当100更新は`41.545 updates/s`だった。phase/allocation計測を
有効にした厳しい条件では`38.205 updates/s`、全体CPU使用率`58.860%`、
candidate executor CPU使用率`63.238%`、allocation`5.433 MiB/update`、
GC時間比`4.756%`だった。

## 動的再帰

同じcheckpointから固定深度を解除し、sampled hard haltingを実際に有効化した。
平均深度は`2.979`、範囲は2～6、定常速度は`27.614 updates/s`だった。

固定深度2に対する速度比は`0.665`で、`2 / 2.979 = 0.671`という実行step数比とほぼ
一致する。したがって動的版の低下は新しいserial bottleneckではなく、実際に約1.49倍の
再帰stepを実行した結果である。動的版も下限`20 updates/s`を満たす。

速度のためにhalt確率を強制的に上げたり、平均深度を2へ固定したりはしない。固定深度2は
再帰stepあたりの速度上限を測る比較専用とし、本学習では最初の更新からtask lossと
1-step probeによって深度を学習させる。

この後、学習開始方法も実teacherで比較した。深度2～6のrandom-depth warmupを
1,000更新行った試行は平均深度約4.0、`17.368 updates/s`となり、下限20を割ったため
不採用とした。保存されたcheckpointから本学習を継続していない。

代わりに、warmupを置かず最初の更新からsampled hard haltingとstateあたり4候補の
1-step probeを有効化した。5,000更新pilotは次の結果となった。

| 指標 | 結果 |
|---|---:|
| 更新数 | 5,000 |
| 処理state数 | 20,000 |
| updates/s | **25.866** |
| states/s | 103.464 |
| recurrent steps/s | 14,161.317 |
| 平均CPU使用率 | 52.396% |
| candidate中CPU使用率 | 54.778% |
| update 5,000の平均深度 | 2.975 |
| update 5,000の深度範囲 | 2～8 |
| 最初10観測点の平均loss | 4.0176 |
| 最後10観測点の平均loss | 3.2603 |
| loss低下率 | 18.85% |

100更新ごとの観測値はteacher batchが異なるため単調ではないが、前後平均には明確な
学習信号がある。固定training panel 128状態の5,000更新評価は次だった。

| 指標 | 結果 |
|---|---:|
| composite loss | 3.023583 |
| top-1 | 0.4765625 |
| NDCG | 0.968138 |
| pairwise | 0.808480 |
| margin | 0.050655 |
| 平均深度 | 3.01195 |
| 深度範囲 | 3～6 |

5,000更新checkpointを使い、動的haltingと4-probeを有効にしたまま追加100更新を
allocation/GC計測した。定常`28.551 updates/s`、計測込み`27.451 updates/s`、
allocation`11.516 MiB/update`、GC時間比`1.656%`だった。全体CPU使用率は
`53.015%`、candidate executorは`58.901%`である。probeの追加計算を含めても下限20を
十分に上回った。

pilot checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_root_dynamic_from_scratch_pilot_u000000_u005000_20260724\
checkpoints\checkpoint_000005000.jls
```

SHA-256：

```text
86aa6b71e951a6df8498b33eec1ad6aca09e1fb599afb861f87b20aa432153f9
```

## 数値一致

採用候補のserial/barrierless smokeは次を確認した。

- 出力、loss、raw VJP：完全一致
- routing、hard halting、RNG state：完全一致
- worker witness：最大絶対誤差`2.992e-5`、相対L2`9.964e-7`
- parameter gradient：最大絶対誤差`1.244e-6`、相対L2`9.965e-7`
- optimizer後parameter：最大絶対誤差`1.243e-7`、相対L2`2.199e-9`
- optimizer clock、token event count、sparse row state：一致
- 構造・VJP回帰テスト：`551 / 551`合格

smoke用checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_root_40ups_candidate_smoke_seed_u000000_u000001_20260724\
checkpoints\checkpoint_000000001.jls
```

SHA-256：

```text
b9a62c6f93a21586d01c19aa2e500f7f4796fc0fa96833190d4f3cba96070eb9
```

## 評価境界

今回の速度benchmarkと正当性smokeはtraining splitだけで完結している。
`MODEL_DIM=64`の初期試行時に、benchmark-onlyを付け忘れた1更新runがvalidation panelを
一度だけ構築した。これは手順違反であり、そのrun、metric、checkpointは全て不採用とした。
それ以後の比較ではvalidationを構築していない。sealed seedは一度も使用していない。

## 結論

Kをさらに狭めるのではなく、register workspaceとprojection、FFN幅、backward tailを
修正することで、bank容量を維持したまま定常`41.545 updates/s`へ到達した。
sampled hard haltingを有効にした場合も`27.614 updates/s`で下限を満たす。

ただし固定深度2は実装上限の比較専用であり、採用モデルではない。その後の100,000更新
評価で旧haltingがdeterministic深度3へcollapseしたため、候補状態とhalt weightの
正規化類似度へ変更し、residual scaleを0.55へ調整した。新条件は20,000更新時点で
深度3、5、6、平均3.409へ分布し、`27.243 updates/s`を維持した。採用対象はこの
`fixed_depth=0`の動的版だけである。詳細は
[`DYNAMIC_HALTING_NORMALIZATION_2026-07-24.md`](DYNAMIC_HALTING_NORMALIZATION_2026-07-24.md)
に記録した。

この段階で速度探索は終了する。縮小したworkspaceの最終品質は未確定だが、
dynamic-from-scratch 5,000更新でloss、top-1、NDCG、marginとhalting挙動を確認し、
短期学習ゲートは通過した。本学習はpilot checkpointからの継続ではなく、
同じdynamic-from-scratch条件を維持して100kまで継続できる。
