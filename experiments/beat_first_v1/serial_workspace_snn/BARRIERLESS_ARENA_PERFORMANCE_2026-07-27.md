# SWSNN barrierless arena・GC排除・CPU tuning

## 結論

scaled SWSNNの学習経路を、候補chunkごとのZygote tapeから、DSRLN/EVRL型の
固定arena＋解析VJP＋worker-local勾配へ置換した。採用構成は20物理coreへCPU Setを
固定し、state batch 8を候補単位のisbits MPMC jobとして常駐workerへ流す。

1,000更新soakでは8,000 teacher stateを`104.421877秒`で処理し、
`76.612298 states/s`、全20 core平均CPU使用率`85.737691%`だった。
全1,000 hot updateのallocationは`0 byte`、GC停止時間は`0.0秒`である。

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
