# 3段Lookup復元アブレーション

## 目的

高速化のため1段へ縮約した再帰step内のLookupFFNを3段へ戻し、現行の表現力復元構成で
どこまで精度が回復するかを測定した。変更点は`DSRL_BLOCKS=1 -> 3`だけである。

次の要素は1段Lookup対照から変更していない。

- 入力：board、candidate、difference、NEXT/HOLD、`aux37`
- real-teacher、candidate独立評価、task loss
- learned local spatial attentionと再帰DWConv
- 3 registers、attention 16／1 head、SwiGLU 64
- registerごとの固定K64 episodic read/write
- Lookup table数13、4,096 rows/table、tableごとのtop-3 active row
- candidate固有1-step probe付きsampled hard halting
- active-only backward、sparse optimizer、optimizer係数
- barrierless executor、pinningなし、forward chunk 8、backward chunk 1

validation rowとsealed game seedは構築も参照もしていない。

## 実装

`EpisodicViTRecurrentLookup.jl`が`DSRL_BLOCKS=1`を強制していた箇所を撤去し、
`DynamicSparseRecurrentLookup.jl`の既存の1～3 block実装を環境変数で選択できるように
した。旧3段source全体を戻したのではなく、現在のK64 episodic routing、3-register、
attention 16／1 head、SwiGLU 64を保ったままLookup block数だけを増やした。

| 項目 | 1段Lookup対照 | 3段Lookup |
|---|---:|---:|
| 総parameter数 | 6,909,665 | 20,542,179 |
| Lookup bank parameter | 6,815,744 | 20,447,232 |
| Lookup micro-call／step | 3 | 9 |
| active row／step | 39 | 117 |

1段専用testには`DSRL_BLOCKS=1`を明示し、新たに3段のforward、backward、各blockの
sparse bank gradientを検証する回帰testを追加した。

- 3段Lookup回帰：11/11合格
- 1段Lookup＋recurrent DWConv回帰：8/8合格
- fixed-K64 episodic read/write回帰：614/614合格

## 数値一致

10,000更新checkpointをserial oracleと20-worker barrierless executorへ独立復元し、
同じreal-teacher 4状態を1更新した。

- 出力、loss、raw VJP：完全一致
- candidate RNG、hard halting、token edge、全3 blockのLookup row：完全一致
- parameter gradient：最大絶対差`2.4453e-5`、相対L2`9.9323e-6`
- optimizer後parameter state：最大絶対差`2.4447e-6`、相対L2`1.2316e-8`
- 3 bankのglobal clock、sparse row clock、sampler、halt RNG：一致

smokeは合格した。block数の変更によってbarrierless executorの数学やoptimizer
semanticsが変わっていない。

## 100,000更新

3段Lookupを初期値から10,000更新し、そのcheckpointからoptimizer、sampler、
halting RNGを完全復元して100,000更新まで継続した。既存1段checkpointへ追加blockを
途中挿入していないため、3 blockは最初から共同学習されている。

- teacher state：400,000
- 10k→100k学習本体：`6,676.473秒`
- 長期平均：`13.480 updates/s`
- 長期CPU平均：`47.773%`
- 長期candidate処理中CPU：`55.058%`
- 100k checkpoint SHA-256：
  `fc8f7e7e389d721dc02821fef45733394d6fe96bcfff8d14e98570e37222fa48`

区間速度は30k付近で約`9.15 updates/s`まで下がった後、80k～90kでは
約`19.61 updates/s`まで回復した。これはblock計算量だけでなく、学習されたhard
haltingの実現depthが区間ごとに変化したためである。

100k checkpointからcheckpointを保存しない20 warmup＋100更新を行った定常測定は
次の通りだった。

- `22.341 updates/s`
- 全体CPU`67.301%`
- executor CPU`71.887%`
- allocation`8.815 MB/update`
- GC`0.0360秒 / 4.4761秒 = 0.804%`
- sampled平均深度`2.973`、範囲2～6

短い定常batchは1段Lookupの記録済み`20.301 updates/s`を上回ったが、batchとdepthの
組合せに依存する。90,000更新の長期平均は1段`19.900`に対して3段`13.480`なので、
3段の方が高速とは判定しない。

## 10,000更新刻みの同一パネル評価

training splitから固定した同じ128状態を使用した。row SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.911543 | 0.492188 | 0.975060 | 0.829925 | 0.075851 | 3.000 | 3～3 |
| 20,000 | 2.854337 | 0.476562 | 0.978577 | 0.843444 | 0.069415 | 3.000 | 3～3 |
| 30,000 | 2.772898 | 0.500000 | 0.980116 | 0.851496 | 0.061061 | 3.000 | 3～3 |
| 40,000 | 2.749934 | 0.515625 | 0.980351 | 0.855341 | 0.074019 | 3.000 | 3～3 |
| 50,000 | 2.759187 | 0.500000 | 0.980457 | 0.860272 | 0.064833 | 3.000 | 3～3 |
| 60,000 | 2.710621 | 0.539062 | 0.982510 | 0.863445 | 0.071186 | 3.000 | 3～3 |
| 70,000 | 2.715300 | 0.554688 | 0.984797 | 0.870018 | 0.061892 | 3.000 | 3～3 |
| 80,000 | 2.716822 | 0.554688 | 0.985172 | 0.874478 | 0.069064 | 3.000 | 3～3 |
| 90,000 | 2.671964 | 0.585938 | 0.984123 | 0.875204 | 0.069980 | 3.000 | 3～3 |
| 100,000 | **2.665670** | **0.609375** | **0.986058** | **0.880499** | **0.079165** | 3.000 | 3～3 |

loss、top-1、NDCG、pairwise、marginはいずれも100kがこの試行の最良点だった。
一方、deterministic evaluationでは10kから100kまで全candidateが深度3であり、
入力依存の停止深度は固定パネル上で収束していない。学習時のsampled depthは2～6へ
分布するが、評価時の有効計算量は実質的に一定だった。

## 1段Lookupとの厳密比較

現行表現力復元1段Lookupの100kと、同じ固定128状態で比較した。

| 指標 | 1段Lookup 100k | 3段Lookup 100k | 3段－1段 |
|---|---:|---:|---:|
| loss | 2.686149 | **2.665670** | **-0.020479** |
| top-1 | 0.578125 | **0.609375** | **+0.031250** |
| NDCG | 0.984347 | **0.986058** | **+0.001711** |
| pairwise | 0.874771 | **0.880499** | **+0.005728** |
| margin | **0.097056** | 0.079165 | **-0.017891** |
| 長期updates/s | **19.900** | 13.480 | -6.420 |

3段化はtop-1を128状態中4件、loss、NDCG、pairwiseを改善した。従って、
再帰step内部の追加Lookupには表現力上の効果がある。しかしteacher上位2候補のscore差を
表すmarginは明確に悪化し、速度は長期平均で32.3%低下した。全指標での無条件な改善では
ない。

## 旧DSRLN、0.8超モデル、PreActとの関係

3段Lookupへ戻しただけでは、過去の高性能構成までは戻らなかった。

- 旧3段・4-register・attention 32／4 heads・SwiGLU128相当の動的構成100k：
  top-1`0.703125`
- dense WD`3e-4`を用いた旧固定深度Trial 3：
  top-1`0.804688`
- PreAct 12.75k：
  top-1`0.789062`
- 今回の3段・3-register・attention 16／1 head・SwiGLU64：
  top-1`0.609375`

現行1段の`0.578125`から旧固定深度Trial 3の`0.804688`までのgapは`0.226563`である。
3段化で回復した`0.031250`はその約13.8%に留まる。従って過去からの性能低下の主因は
Lookup段数だけではない。4から3へのregister削減、attention 32／4 headsから16／1 head
への縮小、SwiGLU 128から64への縮小、全283 token readから固定K64 supportへの変更、
固定深度からhard haltingへの変更が残りの差を構成する。

また、top-1`0.804688`は固定深度かつdense WD`3e-4`のtuning panel winnerであり、
動的3段DSRLNの結果ではない。今回の結果と同じ意味で「3段Lookupの効果」と解釈しては
ならない。

## artifact

主checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_expression_restore_3lookup_r3_a16h1_f64_dynamic_u010000_u100000_20260725\
checkpoints\checkpoint_000100000.jls
```

全checkpoint評価：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_expression_restore_3lookup_r3_a16h1_f64_dynamic_u010000_u100000_20260725\
halting_eval_fixed128_all.json
```

## 結論

3段Lookupへの復元は有効だった。現在の高速構成に対し、top-1を`+0.031250`、
lossを`-0.020479`、NDCGを`+0.001711`、pairwiseを`+0.005728`改善した。

ただし、marginは`-0.017891`、長期速度は`19.900 -> 13.480 updates/s`となった。
さらに過去のtop-1`0.804688`との差のうち回復できたのは約13.8%である。従って、
「1段Lookup化が性能低下の全原因だった」という仮説は棄却する。3段Lookupは部分的な
精度回復策として採用可能だが、旧性能へ戻すにはregister、attention、SwiGLU、
episodic supportとhalting信用割当の残りの縮小差分も扱う必要がある。
