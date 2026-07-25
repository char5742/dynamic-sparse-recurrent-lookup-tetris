# 単一Lookup・4-register・K128拡幅試験

## 目的

再帰step内のLookupFFNを1段へ戻したうえで、速度優先構成で縮小していた短期作業
記憶と関係表現を広げた。変更した主要項目は次の通りである。

- register数：3から4
- attention：16次元・1 headから32次元・4 heads
- SwiGLU：64から128
- episodic token support：K64からK128
- Lookup block：3段から1段

今回のrunはさらに、旧高速K64構成で明示していた`model_dim=128`を指定せず、
source既定の`model_dim=256`を使用した。このため、これは指定4項目だけの独立
ablationではなく、carrier幅も含む拡幅上限試験である。K128単独効果としては
解釈しない。

入力、real teacher、候補独立評価、task loss、learned local spatial attention、
再帰DWConv、working-memory write、candidate固有1-step probe、sampled hard
halting、active-only backward、sparse optimizerは変更していない。

validation rowとsealed game seedは構築も評価もしていない。

## 構成

| 項目 | 値 |
|---|---:|
| Lookup block | 1 |
| model dim | 256 |
| registers | 4 |
| attention | 32次元・4 heads |
| SwiGLU | 128 |
| episodic support | K128 |
| token数 | 283 |
| Lookup table | 13 |
| rows/table | 4,096 |
| active rows/table | top-3 |
| 総parameter数 | 13,909,501 |
| scheduler | barrierless、pinningなし |
| worker | 20 |
| forward / backward chunk | 8 / 1 |
| BLAS thread | 1 |

K128は全283 tokenのscoreを作ってからmaskする方式ではない。registerごとにlearned
hash/WTAで128 tokenの物理supportを取得し、そのsupport内だけを正確に評価・gather
する。working-memory writeとbackwardも同じ選択supportに限定している。

## 回帰と数値一致

構成回帰は次の通り合格した。

- fixed K128 episodic lookup：1,126 / 1,126
- single Lookup＋recurrent DWConv有限差分：8 / 8

10,000更新checkpointをserial oracleと20-worker barrierless executorへ独立復元し、
同じreal-teacher 4状態を1更新した。

- output、loss、raw VJP：完全一致
- candidate seed、hard halting、selected token edge、Lookup row、probe教師：
  完全一致
- parameter gradient：最大絶対差`6.0424e-6`、相対L2`4.5297e-6`
- optimizer後parameter state：最大絶対差`6.0396e-7`、相対L2`9.6094e-9`
- optimizer clock、sampler、halt RNG、sparse row clock：一致

smokeは合格した。

## 100,000更新

初期値から10,000更新し、そのcheckpointからmodel、optimizer、sampler、halting RNGを
完全復元して100,000更新まで継続した。

- teacher state：400,000
- 初期0→10k：1,219.905秒、`8.197 updates/s`
- 継続10k→100k：10,968.902秒、`8.205 updates/s`
- 累積10k→100k統計を含むsummary：12,188.807秒、`8.204 updates/s`
- 長期CPU平均：`66.664%`
- 長期candidate処理中CPU：`69.012%`
- 100k checkpoint SHA-256：
  `f756861a40a50ceba533e8ed11a675432177ba45ad87b89a673d7e4ef4a45d47`

速度は最低10 updates/sを満たさなかった。

最良70k checkpointから、checkpointを保存しない20 warmup＋100更新を測定した。

- `8.319 updates/s`
- 全体CPU：`66.069%`
- executor CPU：`69.591%`
- allocation：`17.797 MB/update`
- executor allocation：`15.270 MB/update`
- GC：`0.0578秒 / 12.0211秒 = 0.481%`

GCは支配的でない。K128 read/write、4-register attentionとそれらのbackwardによる
実計算量が速度低下の中心である。

## 10,000更新刻みの同一パネル評価

training splitから固定した同じ128状態を使用した。row SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 3.014726 | 0.476562 | 0.971032 | 0.819612 | 0.077264 | 3.890 | 3～5 |
| 20,000 | 2.880347 | 0.445312 | 0.976065 | 0.830087 | 0.056648 | 3.289 | 3～6 |
| 30,000 | 2.800537 | 0.515625 | 0.976981 | 0.843430 | 0.068493 | 3.096 | 3～6 |
| 40,000 | 2.746537 | 0.484375 | 0.978259 | 0.852510 | 0.071519 | 3.192 | 3～6 |
| 50,000 | 2.743214 | 0.476562 | 0.977311 | 0.852983 | 0.074328 | 3.030 | 3～6 |
| 60,000 | 2.699295 | **0.625000** | 0.982130 | 0.860551 | 0.083141 | 3.000 | 3～3 |
| 70,000 | **2.697863** | 0.562500 | **0.983626** | **0.868009** | 0.078324 | 3.426 | 3～6 |
| 80,000 | 2.757669 | 0.546875 | 0.982239 | 0.859801 | 0.071040 | 3.191 | 3～6 |
| 90,000 | 2.777614 | 0.531250 | 0.979106 | 0.851380 | **0.085876** | 3.012 | 3～6 |
| 100,000 | 2.817819 | 0.492188 | 0.976181 | 0.838832 | 0.084838 | 3.004 | 3～4 |

loss、NDCG、pairwiseは70k、top-1は60k、marginは90kが最良だった。70k以降は
lossと連続順位品質が明瞭に反落した。100kまで学習不足だったのではなく、70k付近で
頭打ち後に不安定化した。

70k checkpointを主比較点とする。

```text
SHA-256:
1df101be811a7c1338bbd0456c41fefcc3a5244890aaf3ac9bda8dad71093287
```

## 既存構成との比較

| 指標 | 1段K64 100k | 3段K64 100k | 今回1段K128 70k |
|---|---:|---:|---:|
| parameters | 6,909,665 | 20,542,179 | 13,909,501 |
| loss | **2.686149** | **2.665670** | 2.697863 |
| top-1 | 0.578125 | 0.609375 | 0.562500 |
| NDCG | 0.984347 | **0.986058** | 0.983626 |
| pairwise | 0.874771 | **0.880499** | 0.868009 |
| margin | **0.097056** | 0.079165 | 0.078324 |
| 長期updates/s | **19.900** | 13.480 | 8.204 |

今回の70kは1段K64 100kに対し、loss`+0.011714`、top-1`-0.015625`、
NDCG`-0.000721`、pairwise`-0.006762`、margin`-0.018732`であり、総合品質を
上回らなかった。top-1だけなら今回60kの`0.625000`が既存1段と3段の両方を上回るが、
loss、NDCG、pairwiseは弱く、安定した優位ではない。

PreAct 12.75kとの70k比較は、loss`+0.146958`、top-1`-0.226562`、
NDCG`-0.010842`、pairwise`-0.062194`、margin`-0.038293`である。

過去の旧3段・4-register動的構成top-1`0.703125`、固定深度・dense WD`3e-4`
Trial 3の`0.804688`にも到達しなかった。

## 結論

1 Lookupへ戻し、4 registers、attention 32／4 heads、SwiGLU128、K128を実装・学習
した。数値的一致と動的深度3～6は確認したが、速度は`8.204 updates/s`に低下し、
最良70kでも既存K64の総合順位品質を超えなかった。100kではさらに反落した。

従って、この拡幅上限構成をproduction主採用にはしない。結果は次を示す。

1. 3段Lookupを1段へ戻してworkspaceを広げるだけでは旧性能は回復しない。
2. K128と4-register read/writeの計算増はGCではなくforward/backward本体にある。
3. 容量増加はtop-1の局所的な改善を生むが、連続順位品質と安定収束へ変換されていない。
4. `model_dim=256`も同時に広がっているため、K128単独の因果は主張しない。

## artifact

主比較checkpoint：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_1lookup_r4_a32h4_f128_k128_dynamic_u010000_u100000_20260725\
checkpoints\checkpoint_000070000.jls
```

全checkpoint評価：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_1lookup_r4_a32h4_f128_k128_dynamic_u010000_u100000_20260725\
training_only_fixed128_u010000_u100000.json
```

数値一致：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_1lookup_r4_a32h4_f128_k128_dynamic_u000000_u010000_20260725\
barrierless_correctness_smoke_u010000.json
```
