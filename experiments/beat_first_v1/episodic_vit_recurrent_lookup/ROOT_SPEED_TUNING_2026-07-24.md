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
scratch学習初期のwarmupに限定し、その後はtask lossと1-step probeから深度を学習させる。

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

この段階で速度探索は終了する。ただし、縮小したworkspaceの最終品質は未確定である。
本学習前にfixed-batch overfitと短い実teacher学習で学習信号を確認し、品質が維持される
場合だけ100k学習へ進む。
