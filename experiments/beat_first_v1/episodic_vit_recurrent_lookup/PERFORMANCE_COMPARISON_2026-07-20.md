# 最終held-teacher性能比較 — 2026-07-20

修正済みのepisodic ViT recurrent Lookupモデルを学習後に評価し、PreActと同じ不変のreal-teacher検証パネルで比較した。結果は、精度面では否定的、CPU throughput面では肯定的だった。疎モデルはランキング品質でPreActを**上回らなかった**一方、CPUネイティブ実装では同じパネルを約12倍高速に評価した。

## 固定した比較条件

- teacher dataset manifest SHA-256: `1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded`
- split seed: `2026071817`
- 固定validation subset seed: `2026072315`
- held teacher state数: 128、row-list SHA-256: `fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`
- 入力は同一の`board / candidate / difference / next_hold / aux37`
- 各candidateを独立に評価し、teacher Qと順位は教師信号としてのみ使用
- 標準化ListNet＋margin objectiveと固定teacher top-2 margin
- 観測width 76、learnerのpadding後width 80
- Julia 20 threads、BLAS 1 thread。game validationおよびsealed seedは未使用

PreAct best checkpointとEVRL update 12,000 checkpointは、どちらも48,000 training statesを消費した。これを同一update・同一state数の主要比較とする。80,000 statesを消費したEVRL update 20,000は、最終学習モデルとして別に記載する。

## 結果

| モデル | Updates | 学習state数 | Parameters | Top-1 | NDCG | Pairwise | Margin | Composite loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PreAct best | 12,000 | 48,000 | 1,481,326 | **0.7891** | **0.99329** | **0.92336** | **0.12332** | **2.56378** | 4.19 |
| EVRL budget-matched | 12,000 | 48,000 | 20,577,480 | 0.3594 | 0.97127 | 0.82028 | 0.04954 | 2.97502 | **53.54** |
| EVRL final | 20,000 | 80,000 | 20,577,480 | 0.3750 | 0.95837 | 0.76881 | 0.08783 | 3.16286 | **50.19** |

同一予算EVRLのPreActとの差は次のとおり。

- top-1: `-0.42969`
- NDCG: `-0.02202`
- pairwise accuracy: `-0.10308`
- action margin: `-0.07378`
- composite loss: `+0.41123`（悪化）
- CPU held-panel throughput: `12.77x`

最終EVRLのPreActとの差は次のとおり。

- top-1: `-0.41406`
- NDCG: `-0.03493`
- pairwise accuracy: `-0.15455`
- action margin: `-0.03550`
- composite loss: `+0.59908`（悪化）
- CPU held-panel throughput: `11.97x`

推論時間は、同じ128-stateパネルを3回通したwarm steady-state測定である。PreActは固定4-state batch、EVRLはcandidate独立の1-state evaluation batchと物理的疎routingを使用した。compile、checkpoint load、dataset loadの時間は含めていない。

## 判明したこと

1. **固定batch過学習の成功はheld generalizationを意味しなかった。** spatial credit修正によりarchitectureは学習可能となり、4つのreal-teacher stateを記憶できたが、held top-1はPreActを大きく下回った。
2. **update数を増やしても品質差は埋まらなかった。** EVRLの12,000から20,000 updateにかけてtop-1は`0.01563`しか上昇せず、NDCGは`0.01290`、pairwise accuracyは`0.05147`低下し、composite lossは`0.18785`増加した。一部の選択actionは鋭くなったが、candidate全体の順位は改善しなかった。
3. **単純な容量不足ではなかった。** EVRLはPreActの`13.89x`のparameterを保持するが、held rankingは大幅に劣った。疎route探索と信用割当が主要な未解決問題として残った。
4. **system上の主張は支持されたが、model品質の主張は支持されなかった。** より大きなparameter storeを持ちながら、EVRLの入力依存疎実行はこのPreAct CPU評価より約12倍高速だった。これはPreActの代替ではなく、有用な速度・精度trade-offを示す結果である。
5. **最終checkpointを「PreAct超え」と表現してはならない。** 物理的に疎なepisodic／parameter lookupは高速かつ学習可能だが、当時のrouterとrecurrent stateでは未見teacher stateに対するPreActの空間ranking品質を再現できなかった、というのが最も強く正当化できる結論である。

## 成果物

- machine-readable result: [`performance_comparison_2026-07-20.json`](performance_comparison_2026-07-20.json)
- 再現可能なevaluator: [`evaluate_teacher_comparison.jl`](evaluate_teacher_comparison.jl)
- PreAct checkpoint SHA-256: `f3e40d7b6bd3ea8aa7930b2178b537bdae37eea76cdbf089c3ba489ac99d057e`
- EVRL update 12,000 SHA-256: `a566d0e63eacddbc3e02a8e789f891fd18328953dd4fc443bdc6ac7009e5d858`
- EVRL final SHA-256: `1fc05d63154fc73e5d60367c2b19d63116a975b0a3a772899b7fd0ca382db28e`

binary checkpointとteacher datasetはcommitしていない。上記hashと固定row-panel hashにより、大きなlocal artifactをrepository内に存在するかのように誤認させず、公開比較を監査可能にしている。
