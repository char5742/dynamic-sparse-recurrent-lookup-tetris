# BP修復後・速度25構成の100,000更新結果

## 結論

STEを本体の連続勾配から分離した修正版を、速度下限`25 updates/s`を満たす構成へ
縮約し、親checkpointなしで100,000更新した。400,000 teacher stateを
`3,962.715秒`（学習区間、約66分3秒）で処理し、速度は
`25.230 updates/s`だった。

training-only固定128状態では、10kから100kにかけてcomposite lossが
`2.933468 -> 2.669060`、top-1が`0.476562 -> 0.593750`、
NDCGが`0.976198 -> 0.985005`、pairwise accuracyが
`0.829715 -> 0.871259`へ改善した。したがって、BP修復後の縮約モデルは
100kまで学習を継続できている。ただし既存PreAct基準には届かず、決定論的推論時の
haltingは全candidateで深度3へ集中した。

## 固定したモデルと実行条件

| 項目 | 値 |
|---|---:|
| 総parameter数 | 6,897,248 |
| model dim | 128 |
| Lookup block / step | 1 |
| table / block | 13 |
| WTA choices | 16 |
| 選択row / table | 3 |
| register数 | 2 |
| attention幅 / head数 | 16 / 1 |
| SwiGLU幅 | 32 |
| episodic候補 / 物理support | 128 / 64 |
| router探索slot | 8 |
| spatial relation | learned Q/K/V/O 8近傍attention + depthwise |
| halting | hard dynamic、最小2・最大12 step |
| state batch | 4 |
| scheduler | barrierless、pinningなし、forward chunk 8、backward chunk 1 |
| Julia worker / BLAS thread | 20 / 1 |

入力、teacher、ListNet・marginを含むtask loss、候補独立評価、active-only
Lookup backward、sparse optimizer、hard-haltingの意味は変更していない。

## STE／本体勾配分離の回帰検査

`test_full_trajectory_vjp.jl`へ次の検査を追加した。

1. router surrogateを無効化したtask-only VJPを、hard support固定の中心有限差分と比較
2. RMSNorm VJPを独立に中心有限差分と比較
3. router-only distillationでrouter勾配だけが非ゼロになることを確認
4. 任意係数`\(\lambda=0.37\)`で
   `g_total = g_task + lambda * g_router`を全parameterについて確認
5. `dnormalized`、`dregister_route`、`dmemory_normalized`を直接検査し、
   surrogateを有効化しても本体cotangentが変わらないことを確認
6. router-only Adam更新で本体parameter・本体moment・Lookup optimizer stateが
   変わらず、固定support distillation lossだけが低下することを確認

結果は次のとおりだった。

| 検査 | 結果 |
|---|---:|
| task-only固定support有限差分 | 3 / 3 pass |
| RMSNorm有限差分 | 8 / 8 pass |
| STE分離・加法性・optimizer隔離 | 891 / 891 pass |
| router distillation loss | 3.689158 -> 3.688441 |

STEはhard forwardの真の微分ではないため、STEそのものを有限差分とは比較していない。
有限差分は本体の連続parameterだけに適用し、各摂動でtop-k token IDとLookup row IDが
基準trajectoryから変化していないことを確認した。

production既定値、20 workerのserial対barrierless smokeも再実行し、出力、loss、
raw VJPは完全一致した。worker勾配の最大絶対差は`9.3460e-5`、
parameter勾配は`2.9430e-7`、optimizer更新後parameterは`2.9802e-8`で、
全離散support、hard halting、probe、RNG、optimizer clockも一致した。

## 100,000更新

実行先：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_bpfix_r2_a16h1_f32_spattn_dynamic_u000000_u100000_20260727
```

| 項目 | 結果 |
|---|---:|
| 更新数 | 100,000 |
| teacher state | 400,000 |
| 学習区間実時間 | 3,962.714624秒 |
| 学習区間updates/s | 25.230179 |
| 学習区間states/s | 100.920717 |
| 全体実時間 | 3,979.795373秒 |
| 全体updates/s | 25.126920 |
| 平均CPU使用率 | 58.036979% |
| candidate処理中CPU使用率 | 61.431287% |
| 最終batch loss | 3.154158 |
| 最終batch sampled平均深度 | 2.983957 |
| 最終batch sampled深度範囲 | 2～6 |

この100k実行は軽量throughput記録であり、phase別allocation／GC profilingは
有効化していない。そのため、このrunについてGC比率を推測値で補わない。

100k checkpoint SHA-256：

```text
0a3e20c3f575a2b56edef68c40f92d6db067d0dc1e2dd50c2cc8ed0a39093d59
```

## 10k刻みの同一パネル評価

評価はtraining-only固定128状態で行った。パネル行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`である。
validation rowとsealed seedは使用していない。全checkpointで同一の5,357候補を
評価した。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | 2.933468 | 0.476562 | 0.976198 | 0.829715 | 0.052012 | 3.000 | 3～3 |
| 20k | 2.893800 | 0.460938 | 0.977958 | 0.836891 | 0.041762 | 3.000 | 3～3 |
| 30k | 2.815438 | 0.476562 | 0.978338 | 0.847071 | 0.054374 | 3.000 | 3～3 |
| 40k | 2.807741 | 0.437500 | 0.972095 | 0.844015 | 0.089095 | 3.000 | 3～3 |
| 50k | 2.786405 | 0.515625 | 0.979780 | 0.854551 | 0.060738 | 3.000 | 3～3 |
| 60k | 2.725451 | 0.539062 | 0.984187 | 0.865997 | 0.053281 | 3.000 | 3～3 |
| 70k | 2.693966 | 0.539062 | 0.984962 | 0.867761 | 0.063282 | 3.000 | 3～3 |
| 80k | 2.724018 | 0.546875 | 0.985305 | 0.870579 | 0.056702 | 3.000 | 3～3 |
| 90k | 2.674581 | 0.570312 | **0.985552** | **0.872200** | 0.068095 | 3.000 | 3～3 |
| 100k | **2.669060** | **0.593750** | 0.985005 | 0.871259 | 0.085812 | 3.000 | 3～3 |

10kから100kの差は、loss`-0.264408`（`9.013%`低下）、
top-1`+0.117188`、NDCG`+0.008807`、pairwise`+0.041544`、
margin`+0.033800`である。100kはlossとtop-1が最良、90kはNDCGとpairwiseが
最良だった。80kに一時的なloss反落はあるが、90k・100kで再び改善しており、
100k時点で完全な頭打ちとは断定できない。

## halting

学習時のsampled trajectoryは最終batchで深度2～6へ分布した。一方、
決定論的な固定パネル推論では、10kから100kまで全5,357候補が深度3で停止した。
halt evidence自体は更新され、step 2の平均値は`1.361979 -> 1.459926`へ変化したが、
停止境界をcandidateごとに跨ぐほどの差は形成されなかった。

したがって、現構成は「hard halting interfaceを保持して学習した動的モデル」ではあるが、
このパネル上で入力依存の決定論的深度を獲得したとはいえない。次の品質上の主な残課題は、
task品質と速度を維持したままcandidate固有のhalting信用差を拡大することである。

## PreActとの差

同じtraining-onlyパネルで記録済みのPreAct 12,750更新・51,000 teacher stateは、
loss`2.550905`、top-1`0.789062`、NDCG`0.994468`、
pairwise`0.930203`、margin`0.116617`である。

| 指標 | EVRL 100k | PreAct | EVRL - PreAct |
|---|---:|---:|---:|
| loss | 2.669060 | 2.550905 | +0.118155 |
| top-1 | 0.593750 | 0.789062 | -0.195312 |
| NDCG | 0.985005 | 0.994468 | -0.009463 |
| pairwise | 0.871259 | 0.930203 | -0.058944 |
| margin | 0.085812 | 0.116617 | -0.030805 |

BP修復と高速化後も、現縮約構成はPreAct超えを達成していない。またEVRLは
400,000 teacher state、PreActは51,000 teacher stateなので、sample efficiencyの差は
さらに大きい。過去のEVRL top-1`0.804688`は別geometry・別調整試験であり、
今回の縮約構成の結果へ混同しない。

## 評価境界

- 追加学習なしで10個のcheckpointを再評価した。
- 入力、teacher、loss、routing、optimizer semanticsは変更していない。
- regression追加時のproduction既定動作は倍率1のままである。
- validation row、game validation、sealed seedには触れていない。
- 評価JSONのSHA-256は
  `1286dfda851f0bc47162d02deb7c0b11dce5c9b1721c6cd6e4d9856cc22e27da`
  である。
