# 動的haltingの単一深度collapse修正

## 目的

速度調整後のEVRLは学習時にsampled hard haltingを実行していたが、100,000更新
checkpointをdeterministicに評価すると、固定training-only 128状態の全5,357候補が
深度3で停止した。これは物理実行が確率的だっただけで、推論時の計算深度が入力依存に
なっていない。

固定深度2を採用することも、別の単一深度へ移すことも認めず、モデル本体、入力、教師、
task loss、Lookup routing、active-only backward、sparse optimizerを変えずに、
haltingだけを入力依存へ修正した。

validationとsealed seedは使用していない。評価はすべて同じtraining-only固定128状態
（行SHA-256
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`）
で行った。

## 原因

旧式は候補状態の投影へ共有biasを加え、`tanh`した値を深度priorへ足していた。

```text
halt_logit =
    0.5 * (step - 4)
    + 0.75 * tanh(shared_bias + dot(halt_weight, candidate_state))
```

100,000更新時点では共有biasが`1.4146`となり、全候補を同じ閾値側へ押した。共有biasを
単に中心化する最初の修正では、旧checkpointの全候補が深度7へ移っただけだった。
候補投影自体も同符号で`tanh`領域へ集中していたためである。

## 修正

候補状態とhalt weightのcosine類似度を停止証拠とした。

```text
cosine =
    dot(halt_weight, candidate_state)
    / (norm(halt_weight) * norm(candidate_state))

gain = 0.5 + sigmoid(halt_bias)

halt_logit =
    0.5 * (step - 4)
    + 0.55 * gain * cosine
```

共有scalarは`0.5～1.5`の正のgainだけを制御し、全候補の停止閾値を直接移動できない。
停止時刻を前後させる符号と大きさは候補固有の正規化証拠だけから得る。weight/state norm
を含むVJPを解析的に実装し、有限差分で検証した。

barrierless backwardでは、候補ごとのhalt weight/bias寄与をworker-localに保存し、
更新境界でstate、candidate、逆step順に再加算する。worker完了順によるhalt optimizer
stateの変化は許していない。

## 係数試行

モデルは全試行で次の同一構成である。

```text
parameter                     6,897,248
model/carrier dim                   128
registers                             2
attention dim / heads              16 / 1
register projection            structured
spatial recurrent path       depthwise_only
SwiGLU FFN dim                       32
episodic support K                   64
Lookup blocks                         1
Lookup tables / WTA / rows       13 / 16 / 3
fixed depth                            0
one-step probes                 4 / state
workers                               20
queue chunks                 forward 8 / backward 1
CPU Sets                         disabled
```

### residual scale 0.75

スクラッチ5,000更新は`26.272 updates/s`で完走した。しかしdeterministic評価では
深度3が5,225、深度4が60、深度5が72であり、97.5%が深度3へ集中した。

| 指標 | 5,000更新 |
|---|---:|
| loss | 3.192192 |
| top-1 | 0.453125 |
| NDCG | 0.968105 |
| pairwise | 0.805602 |
| margin | 0.038243 |
| 平均深度 | 3.038 |
| 深度範囲 | 3～5 |

評価専用に全候補をstep 4まで進めると、step 4の正規化停止証拠は
`-0.201～0.435`へ分布していた。したがって入力情報が消えているのではなく、step 3の
閾値が低すぎて候補差を見る前に停止していた。

### residual scale 0.55

係数だけを`0.75→0.55`へ下げ、再びスクラッチから5,000更新した。

| 指標 | 5,000更新 |
|---|---:|
| training updates/s | 26.087 |
| training seconds | 191.668 |
| 全体CPU使用率 | 52.231% |
| candidate中CPU使用率 | 54.597% |
| loss | 3.013760 |
| top-1 | 0.4765625 |
| NDCG | 0.967662 |
| pairwise | 0.811115 |
| margin | 0.050864 |
| 平均深度 | 3.389 |
| 深度範囲 | 3～5 |

deterministic深度内訳は次のとおりである。

| 深度 | 候補数 | 割合 |
|---:|---:|---:|
| 3 | 3,317 | 61.92% |
| 4 | 1,998 | 37.30% |
| 5 | 42 | 0.78% |

速度下限20を維持しながら、単一深度collapseを解消した。scale 0.75と比べてloss、
top-1、pairwise、marginも改善したため、0.55を採用した。

## 20,000更新までの継続確認

5,000更新checkpointからoptimizer、RNG、samplerをそのまま継承し、固定深度を使わず
20,000更新まで継続した。追加15,000更新の実学習時間は`550.602秒`、
`27.243 updates/s`だった。全体CPU使用率は`53.179%`、candidate中CPU使用率は
`56.477%`である。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.013760 | 0.4765625 | 0.967662 | 0.811115 | 0.050864 | 3.389 | 3～5 |
| 10,000 | 2.939392 | 0.4609375 | 0.971329 | 0.821058 | 0.072039 | 3.014 | 3～5 |
| 20,000 | 2.903470 | 0.5000000 | 0.976220 | 0.831282 | 0.062859 | 3.409 | 3～6 |

20,000更新のdeterministic深度内訳は、深度3が4,388、深度5が715、深度6が254
だった。中間の深度4を使わないこと自体は固定深度ではなく、候補状態から計算した連続な
停止証拠がstepごとのhazard閾値を越えた結果である。学習時の最終batchも深度2～8へ
分布した。

loss、NDCG、pairwiseは5k、10k、20kを通して改善し、top-1も20kで最高となった。
deterministic評価も複数深度を維持したため、単に学習時の乱数で可変に見えている状態
ではない。

## 数値一致

採用した5,000更新checkpointで、serialとbarrierlessの同一real-teacher batchを比較した。

- 出力、loss、raw VJP：完全一致
- candidate seed、hard halting、token edge、Lookup row、probe教師：完全一致
- parameter gradient：最大絶対差`3.9861e-6`、相対L2`1.8754e-6`
- optimizer後parameter/state：最大絶対差`3.9861e-7`、相対L2`4.0382e-9`
- optimizer clock、RNG、sampler、sparse row clock：一致
- 構造・VJP回帰テスト：最終速度構成`354 / 354`合格

## checkpoint

20,000更新：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_normalized_dynamic_halt_s055_u005000_u020000_20260724\
checkpoints\checkpoint_000020000.jls
```

SHA-256：

```text
b6865c7b131755f0f4d253b206c36b2136cb8c0212abe80f6b4b40909e9f14b0
```

## 結論

固定深度2は採用しない。採用条件は`fixed_depth=0`、候補別sampled hard halting、
deterministic評価で複数深度、1-step probe付きactive-only実行である。

候補投影をnorm非依存のcosine証拠へ変更し、residual scaleを0.55へ調整したことで、
速度下限を守りながら、入力依存の深度3～6と順位品質の継続改善を20,000更新まで確認した。
固定深度2の速度値は再帰step単価の参考上限に限り、production学習性能には用いない。
