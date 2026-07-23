# 単一Lookup＋反復Depthwise Convへの移行

## 目的

旧EVRLは一つの再帰stepの内部で、各registerを3個のLookupFFN blockへ直列に通していた。
しかし外側の共有block自体がhard halting付きで反復されるため、同じstep内でLookup深度を
さらに3段持つと、長期記憶参照回数とparameter容量を増やす一方、空間working memoryの
更新には寄与しない。

そこで、入力、teacher、loss、register数、full-token cross-attention、SwiGLU、
register別routing、working-memory write、hard halting、1-step probe、active-only
backward、sparse optimizerは維持し、次の一軸だけを変更した。

```text
旧:  recurrent spatial update -> cross -> self -> SwiGLU
     -> LookupFFN 1 -> LookupFFN 2 -> LookupFFN 3 -> write -> halt

新:  recurrent 3x3 DWConv + spatial update -> cross -> self -> SwiGLU
     -> LookupFFN 1 -> write -> halt
```

## 実装

`DynamicSparseRecurrentLookup.jl`のblock数を`1:3`の実行時geometryとして一般化し、
EVRLはfresh Julia processで必ず`DSRL_BLOCKS=1`を選ぶ。旧3-block modelを使う他の
実験の既定値は3のままなので、共有moduleの従来経路は残している。

新しいdepthwise convolutionは4個のregisterに対してではなく、24x10、240セルの
入力固有working memoryへ適用する。kernelはchannelごとに独立しており、物理配置を
SIMD連続走査に合わせた`128 x 3 x 3`、論理形状をchannelごとの3x3とする。
RMS-normalized cell memoryを読み、learned residual scaleを介して加算する。paddingは
盤面外を読まないzero-padding相当で、denseな240x240接続やmaskを作らない。

同じkernelを全recurrent stepで共有するため、外側の再帰回数が増えるほど受容野が1セル
ずつ広がる。したがって追加の空間変換深度は、内部Lookupの直列化ではなく、実際の思考
回数と一致する。

forward、manual VJP、worker-local gradient、barrierless reduce、dense AdamW state、
parameter topologyへ、次の2 parameterを追加した。

```text
recurrent_depthwise              3 x 3 x 128 = 1,152
recurrent_depthwise_scale_logit                  = 1
```

CPU kernelはchannelを先頭次元に置き、各盤面edgeについて128 channelを連続走査する。
forwardとVJPのchannel loopをSIMD化し、workerごとの不連続gatherを作らない。

## geometryと計算量

production geometryでの変化は次のとおりである。

| 項目 | 旧3段Lookup | 新1段Lookup＋DWConv |
|---|---:|---:|
| 総parameter数 | 20,585,982 | 6,954,621 |
| Lookup bank parameter | 20,447,232 | 6,815,744 |
| Lookup block数 | 3 | 1 |
| registerあたりLookup呼出し／step | 3 | 1 |
| 全4 registerのLookup呼出し／step | 12 | 4 |
| recurrent DWConv parameter | 0 | 1,153 |
| recurrent DWConv scalar MAC／step | 0 | 276,480 |

総parameterは13,631,361、66.22%減少した。Lookup呼出しはstepあたり3分の1になり、
代わりに空間working memoryを直接更新する有界な局所演算を追加した。

旧3-block checkpointは新geometryへ暗黙に読み替えない。どのbankを残すかという別の
学習上の判断を混入させないためであり、新構成の本学習はscratchから行う。

## 数値検証

`test_single_lookup_recurrent_depthwise.jl`で次を検証し、8/8 testが合格した。

- EVRLのLookup block数が1である
- register数4に対してlong-memory micro-callが4である
- recurrent DWConv kernelとscaleのmanual VJPが中央有限差分と一致する
- cell-memory入力cotangentが有限である
- DWConv kernelへ非零勾配が流れる
- 共有Lookup moduleの既定3-block初期化も引き続き成功する

次にreal-teacherを使い、barrierless、20 worker、pinningなし、chunk 8、
BLAS 1 threadで、20更新warmup後に100更新を測定した。validation evaluationと
sealed seedは使用していない。

| 項目 | 結果 |
|---|---:|
| scratch更新数 | 120 |
| 処理teacher state | 480 |
| 計測更新数 | 100 |
| 計測training時間 | 7.723秒 |
| updates/s | 12.949 |
| 平均CPU使用率 | 66.707% |
| candidate処理中CPU使用率 | 68.352% |
| 最終batch loss | 4.262212 |
| 最終batch平均深度 | 4.140 |
| 最終batch深度範囲 | 2～6 |

同じ120更新smokeで、channelが非連続になる素朴なkernel配置は`10.107 updates/s`だった。
数学モデルを変えずchannel連続SIMD配置へ直した結果、速度は1.281倍になった。

120更新は実装smokeであり、学習品質や収束性能の比較には使わない。旧3-block
full-scratch 100kの長区間速度は10.651 updates/sだったが、学習初期のtrajectory、
平均深度、candidate系列、計測長が違うため、今回の短時間値との差をarchitectureの
速度差とは断定しない。

保存checkpoint:

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_single_lookup_dwconv_smoke_u120_20260723_r2\checkpoints\
checkpoint_000000120.jls
```

SHA-256:

```text
dd3fa0cc57794febc61548be90895d14a6f3dd7b074f5c241e64855526edebcc
```

このcheckpointから同一real-teacher 4状態を使い、serial oracleとproduction
barrierless executorを照合した。

| 比較 | 結果 |
|---|---:|
| output最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw VJP最大絶対差 | 0 |
| worker gradient最大絶対差 | `4.41e-6` |
| reduce後parameter gradient最大絶対差 | `1.10e-6` |
| optimizer後parameter/state最大絶対差 | `1.10e-7` |
| routing、halting、RNG、optimizer clock | 全てexact |

したがって、単一Lookup化とDWConv追加後も、serialとbarrierlessでモデル意味論、
active経路、gradient、optimizer stateは一致している。

## 判定

3段Lookupを1段へ縮約し、反復セルworking memoryへ共有DWConvを入れる実装は完了した。
新しいmanual backwardとproduction barrierless経路も合格している。120更新のlossは
収束評価には短すぎるため、この時点では旧100k modelとの性能優劣を主張しない。次の
比較は新geometryをscratchから同一teacher予算まで学習し、10k刻みcheckpointを同じ
training-only panelで観測して行う。
