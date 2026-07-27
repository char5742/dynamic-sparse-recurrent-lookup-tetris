# SWSNN barrierless arena・GC排除・CPU tuning

## 結論

scaled SWSNNの学習経路を、候補chunkごとのZygote tapeから、DSRLN/EVRL型の
固定arena＋解析VJP＋worker-local勾配へ置換した。採用構成は20物理coreへCPU Setを
固定し、state batch 8を候補単位のisbits MPMC jobとして常駐workerへ流す。

production 100,000更新では800,000 teacher state、34,929,701 candidateを
`10,281.140675秒`で処理し、`77.812378 states/s`、全20 core平均CPU使用率
`87.618880%`だった。全100,000 hot updateのallocationは`0 byte`、
GC停止時間は`0.0秒`である。

候補間ListNetは全候補出力を必要とするため、forward完了後のloss/VJP境界だけは
数学的依存として残る。その前後ではworkerを破棄せず、pack、candidate forward、
candidate backward、parameter shard reduction＋AdamW、構造consolidationを同じ
MPMC worker teamへ流す。

## 実装

- 候補ごとにmembrane、active spike、workspace、query、hidden、hard block maskを
  固定長arenaへ保存
- 1,298 binary railをdatasetからworker-local幾何scratchへ直接pack
- spike STE、workspace routing STE、重み、遅延、gate、leak、threshold、
  feedback、headのVJPを解析実装
- 20個のworker-local固定parameter勾配へlock-free加算
- parameterを4,096要素shardへ分け、勾配reduce、gradient norm、AdamW、
  cache更新を同一jobでin-place実行
- 32 node単位で構造consolidationを並列実行
- queue payloadはisbits、queue、job buffer、gradient、optimizer moment、
  forward/backward scratchを起動時に全確保
- Windows CPU Setsを実行時検出し、8 P-core＋12 E-coreへ20 workerを一対一固定

## tuning結果

同じscaledモデル、teacher_v3、BLAS 1 threadで比較した。

| 経路 | state batch | worker | states/s | 全CPU | allocation/update | GC比 |
|---|---:|---:|---:|---:|---:|---:|
| serial Zygote | 8 | 1 | 1.961956 | 8.062% | 17.453 GB | 21.066% |
| chunked Zygote barrierless | 8 | 8 P-core | 6.345502 | 24.093% | 11.015 GB | 44.022% |
| chunked Zygote barrierless | 8 | 20 | 6.213922 | 38.218% | 11.487 GB | 49.720% |
| fixed arena soak | 8 | 20 | **76.612298** | **85.737691%** | **0 byte** | **0.0%** |

採用経路は8-worker Zygote barrierless比`12.073倍`、serial Zygote比`39.049倍`。
CPU利用率は8-worker Zygote比`3.559倍`である。

100% samplingの`Profile.Allocs`をwarmup後の1 updateへ適用した結果：

```text
sampled_allocations = 0
sampled_bytes       = 0
hot allocation      = 0
hot GC seconds      = 0.0
```

## 数値一致

| 検査 | 結果 |
|---|---:|
| 元の0/1・8機構・逐次scan回帰 | 28 / 28 |
| chunked barrierless loss・勾配・Adam更新 | 13 / 13 |
| tiny arena forward・loss・全勾配・AdamW | 6 / 6 |
| real-teacher pack＋scaled arena勾配 | 13 / 13 |

real-teacher scaled 453,815 parameterで、既存Zygoteとの全parameter勾配の
最大絶対差は再実行を含め`1.049041748046875e-5`以下。入力rail、候補出力、loss、raw VJP、
全parameter勾配、AdamW更新を独立に照合した。

## 長時間soak

`arena_soak_u1000_20260727`：

| 項目 | 値 |
|---|---:|
| update | 1,000 |
| teacher state | 8,000 |
| candidate | 350,542 |
| hot wall | 104.421877秒 |
| hot CPU | 1,790.578125 CPU秒 |
| states/s | 76.612298 |
| updates/s | 9.576537 |
| 平均CPU | 85.737691% |
| hot allocation | 0 byte |
| hot GC | 0.0秒 |
| 常駐メモリ観測 | 約7.1 GB、増加なし |

training-only固定32状態：

| 指標 | 初期 | 1,000 | 差 |
|---|---:|---:|---:|
| composite loss | 5.222568 | 3.129654 | -2.092914 |
| top-1 | 0.031250 | 0.218750 | +0.187500 |
| NDCG | 0.827000 | 0.947801 | +0.120801 |
| pairwise | 0.511972 | 0.783535 | +0.271563 |

1,000 checkpoint：

```text
SHA-256 37e11253c6f1bf2ee732ca61469a2f38ca4b45e2370434a383935926592ccb94
```

50更新checkpointから60更新へのresumeも、parameter、AdamW moment/clock、
sampler permutation/RNG、構造状態を復元して0 allocation・0 GCのまま完了した。

## 100k条件

productionは`100,000 update`、state batch 8、合計`800,000 teacher state`。
10,000更新ごとにparameter、optimizer、sampler、構造状態をcheckpoint化する。
hot updateに1 byteでもallocationが戻る、GCが1回でも入る、非有限loss/gradientが出る、
CPU binding検証が失敗する、またはcheckpoint SHAが一致しない場合はfail closedする。

評価はtraining splitから固定した128状態だけを使う。validation row、
game validation、sealed seedは使用しないため、結果を未見データ汎化とは呼ばない。

## 100k実測

`arena_scaled_u100000_20260727`を、commit `fc3d1c4`、source fingerprint
`8877c948edb5d24fe58224a1c2ea09132532eb2b3daf9c1b9ec5957fd5f951ca`
から実行した。

| 項目 | 値 |
|---|---:|
| update | 100,000 |
| teacher state | 800,000 |
| candidate | 34,929,701 |
| hot wall | 10,281.140675秒 |
| hot CPU | 180,164.406250 CPU秒 |
| states/s | 77.812378 |
| updates/s | 9.726547 |
| 全20 core平均CPU | 87.618880% |
| hot allocation | 0 byte |
| hot GC | 0.0秒 |
| 構造consolidation反転 | 499,091 |
| 初期maskとの差 | 55,536 edge |
| weight L2変化 | 49.339868 |
| continuous delay L2変化 | 6.858212 |
| 常駐メモリ観測 | 約6.64 GiB、増加なし |

旧20-worker Zygote経路の`6.213922 states/s`、CPU`38.218%`に対し、
速度は`12.522倍`、CPU利用率は`2.293倍`。serial Zygoteの
`1.961956 states/s`に対しては`39.661倍`である。

hot wallの主な内訳：

| phase | 秒 | hot wall比 |
|---|---:|---:|
| pack | 41.656076 | 0.405% |
| forward | 4,359.016154 | 42.398% |
| loss | 3.000115 | 0.029% |
| backward | 5,687.345805 | 55.318% |
| optimizer | 182.948298 | 1.779% |
| consolidation | 1.881202 | 0.018% |

training-only固定128状態、5,357 candidate、panel SHA-256
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`：

| 指標 | 初期 | 100,000 | 差 |
|---|---:|---:|---:|
| composite loss | 5.258287 | 2.720394 | -2.537892 |
| top-1 | 0.023438 | 0.523438 | +0.500000 |
| NDCG | 0.825348 | 0.978654 | +0.153307 |
| pairwise | 0.500014 | 0.843108 | +0.343093 |
| action margin | 0.000185 | 0.084973 | +0.084788 |

同じ固定panelでの記録値との比較。ただしteacher state予算はPreAct 51,000、
DSRLN 400,000、SWSNN 800,000で揃っていない。

| モデル | loss | top-1 | NDCG | pairwise |
|---|---:|---:|---:|---:|
| PreAct | 2.550905 | 0.789063 | 0.994468 | 0.930203 |
| DSRLN | 2.679211 | 0.578125 | 0.983389 | 0.872891 |
| SWSNN arena | 2.720394 | 0.523438 | 0.978654 | 0.843108 |

従って、SWSNNは明確に教師順位を学習したが、この条件ではDSRLNとPreActを
上回っていない。またtraining-only評価なので汎化性能も主張しない。

最終artifact：

```text
checkpoint_000100000.jld2
bytes   8917020
SHA-256 78658c4956fb999ede8d68f3d724514a8cfda79b79b376604d7dc07c509847c2

results.json
bytes   6673
SHA-256 61bde563545eb367497b1796e5b367dbd7f2c746e03094ccaba87eb1b781c75b
```

100k学習processは最終checkpoint作成後、in-process 128状態評価の完了前に
終了した。Windows Application logに障害記録はなく、原因は未確定である。
固定arenaを保持したまま従来評価の大きな一時領域を追加するpeak memoryが有力な
仮説なので、最終評価は`evaluate_arena_checkpoint.jl`で別processへ分離した。
checkpoint、training split、固定panelのSHAを照合し、学習済みparameterを変更せずに
上記`results.json`を生成した。
