# EVRL研究の現時点での判明事項

## 結論

現行EVRLは、入力内の短期関係形成、LookupFFNによる長期記憶選択、入力依存の再帰深度を一つのCPU学習系へ統合できている。8近傍spatial backwardの範囲外書込みを修正した後は、loss、top-1、NDCG、pairwise、marginが学習とともに改善し、hard haltingも2～12 stepを入力ごとに使い分けるようになった。

現在の速度条件付き採用構成は、state batch 4、dense LR `2e-4`、dense weight decay `3e-4`、halt LR `5e-5`、2 probes/stateである。採用checkpointは50,000更新で、top-1 `0.734375`、NDCG `0.987706`、pairwise `0.884073`、margin `0.131590`、平均深度 `4.010`、学習速度 `15.545 updates/s`を記録した。

50,000更新後にepisodic dense LRだけを半減した試行は、65,000更新でtop-1 `0.757812`、NDCG `0.990265`まで伸びた。しかし実データ10,000更新の速度が`14.625 updates/s`となり、最低条件の`15 updates/s`を満たさなかった。この65,000更新checkpointは品質上の参考として保持するが、現行採用構成にはしない。

PreAct超えはまだ実証できていない。既存PreAct結果は学習状態数が異なり、範囲外書込み修正後のEVRLとの同一予算比較をまだ再実施していないためである。

## 現行アーキテクチャ

各候補はPreActと同じ盤面、候補、差分、NEXT/HOLD、`aux37`だけから独立に評価する。teacher Q値と順位は教師信号にのみ使う。

- 総parameter数：`20,577,789`（約20.58M）
- 入力記憶：240 cell token、6 NEXT/HOLD token、37 aux token、合計283 token
- 視覚経路：dilation `1, 2, 4, 8, 16`の5段depthwise/pointwise残差経路
- 視覚経路の受容野：`63 x 63`。`24 x 10`盤面全体を覆う
- 視覚経路の追加parameter：565
- register数：4
- model dim：128
- attention dim：32、4 heads
- register cross-attention：全283 tokenを直接評価
- cell間通信：8近傍のlearned Q/K/V/Oと相対位置bias
- LookupFFN：共有3 block、各block 13 table、各table 4,096 row、各tableからtop-3 rowを選択
- Lookup bank parameter：`20,451,739`
- 1 macro stepの選択row数：117
- hard halting：最小2、最大12 step
- 停止教師：candidate-localな少数1-step probe

入力tokenを`283 -> 64 -> 16`へ絞る旧hard routingは撤去済みである。短期記憶への入力経路は全token cross-attentionで確保し、物理的疎性は主にLookupFFNのparameter row選択、local spatial edge、active candidate、hard halting、active-only backward、sparse optimizerで維持する。

## 修正した学習阻害要因

8近傍spatial attentionのbackwardで、局所edge数8に対して`BackwardScratch.dweights`が長さ4しかなく、範囲外書込みが起きていた。scratch容量を`LOCAL_SPATIAL_NEIGHBORS = 8`に合わせて修正した。モデル構造、入力、teacher、loss、optimizer semantics、hard halting、checkpoint形式は変更していない。

修正後のreal-teacher serial／barrierless smokeは次の条件で合格した。

- 出力、loss、raw VJP：完全一致
- parameter gradient：許容誤差`1e-6`級
- optimizer更新後parameter：許容誤差`1e-7`級
- routing選択、RNG state、optimizer clock：完全一致

この不具合を含む旧100,000更新試行は研究履歴として残すが、現行モデルの性能根拠には使わない。

## 実行系の判明事項

最速の実行方式は、20 worker、pinningなし、barrierless global candidate queue、chunk size 8である。BLAS内部threadは1とする。

state batchを4から8へ増やすとCPU使用率は上がったが、実効throughputは改善しなかった。

| state batch | updates/s | states/s | CPU平均 | allocation/update | 判定 |
|---:|---:|---:|---:|---:|---|
| 4 | 15.505 | 62.020 | 49.64% | 7.739 MB | 採用 |
| 8 | 7.529 | 60.233 | 56.99% | 15.426 MB | 不採用 |

batch 8は最低`15 updates/s`を満たさず、states/sもbatch 4を下回った。したがって現行モデルではbatch 4を維持する。

Windows CPU Setsによる全コア固定は過去のscheduler比較で遅く、pinningなしを採用している。性能改善はCPU使用率だけでなく、実データ区間のupdates/sで判定する。

## 修正後の学習推移

修正後scratch基準は、batch 4、dense LR `2e-4`、dense WD `1e-4`、halt LR `5e-5`、halting cost 0、2 probes/state、5,000更新のrandom-depth warmupで開始した。

| 更新数 | loss | top-1 | NDCG | margin | 平均深度 | updates/s |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 2.867476 | 0.507812 | 0.978324 | 0.099650 | 2.000 | － |
| 10,000 | 2.810665 | 0.531250 | 0.979387 | 0.088824 | 2.107 | 15.506 |
| 15,000 | 2.758752 | 0.523438 | 0.981218 | 0.104578 | 2.396 | 15.343 |
| 20,000 | 2.752324 | 0.578125 | 0.982200 | 0.103318 | 3.556 | 15.592 |

lossとNDCGは全評価点で改善し、平均深度も2.00から3.56まで増えた。修正前に疑われた「入力を無視して一様予測へ停滞する状態」は解消した。

## halt LRの比較

halt LRだけを下げても、再帰深度の振動は解消しなかった。

| halt LR | 更新数 | loss | top-1 | NDCG | margin | 平均深度 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `5e-5` | 25,000 | 2.710472 | 0.578125 | 0.982187 | 0.122002 | 2.001 | 基準 |
| `1e-5` | 30,000 | 2.691209 | 0.632812 | 0.983490 | 0.116039 | 2.647 | 継続観測 |
| `1e-5` | 35,000 | 2.671507 | 0.671875 | 0.985029 | 0.132433 | 11.713 | 最大深度側へ飽和 |
| `1e-5` | 40,000 | 2.644841 | 0.656250 | 0.987263 | 0.134669 | 2.131 | 深度が反転 |
| `1e-6` | 30,000 | 2.692056 | 0.632812 | 0.983332 | 0.120413 | 2.709 | 明確な優位なし |

`1e-5`は品質を改善したが、平均深度が`2.647 -> 11.713 -> 2.131`と両端を往復した。`1e-6`も短区間では安定化を証明できなかった。halt LR縮小だけを解決策には採用しない。

## dense weight decayの比較

同じ20,000更新checkpointからdense WD `1e-4`と`3e-4`を比較した。WD変更を許可する明示的checkpoint遷移を追加し、serial／barrierless smoke合格後に実行した。

| dense WD | 更新数 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | updates/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | 30,000 | 2.692367 | 0.632812 | 0.983506 | 0.866969 | 0.118709 | 3.372 | 15.120 |
| `3e-4` | 30,000 | 2.689768 | 0.625000 | 0.983862 | 0.867854 | 0.117013 | 2.110 | 15.532 |
| `3e-4` | 35,000 | 2.664932 | 0.695312 | 0.985194 | 0.875882 | 0.141134 | 3.147 | 15.241 |
| `3e-4` | 40,000 | 2.645313 | 0.671875 | 0.988018 | 0.881705 | 0.134168 | 2.178 | 15.557 |
| `3e-4` | 45,000 | 2.622360 | 0.718750 | 0.988368 | 0.884885 | 0.125085 | 2.131 | 15.191 |
| `3e-4` | 50,000 | 2.632145 | 0.734375 | 0.987706 | 0.884073 | 0.131590 | 4.010 | 15.545 |

WD `3e-4`は、15 updates/s以上を維持しながら品質を改善し、halt LR `1e-5`で発生した最大深度側への長時間飽和も回避した。45,000更新はloss、NDCG、pairwiseの均衡がよく、50,000更新はtop-1と入力依存深度が最良だった。後続試行の共通分岐点と採用checkpointには50,000更新を使う。

## 50,000更新後のepisodic LR半減

50,001更新目から、attention、FFN、token、register、headの既存episodic LR scaleだけを0.5へ下げた。Lookup bank、router、lookup alpha、halt LRは変更していない。

| 更新数 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 区間速度 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 55,000 | 2.612962 | 0.718750 | 0.988942 | 0.890677 | 0.132410 | 4.287 | 15.866* | 短期定常benchmark合格 |
| 60,000 | 2.607455 | 0.703125 | 0.988899 | 0.889694 | 0.147984 | 2.113 | 15.206 | 合格 |
| 65,000 | 2.594471 | 0.757812 | 0.990265 | 0.896571 | 0.142645 | 2.888 | 14.625 | 速度条件不合格 |

`*` 55,000更新の値は100更新warmup後の1,000更新benchmarkである。実データ10,000更新で測ると65,000更新までの平均は`14.625 updates/s`だった。

この試行は現時点で最高の品質を得たが、速度条件を満たさない。短いbenchmarkだけで採用せず、長い実データ区間を最終判定に使う。

## 現在の採用checkpoint

- run：`evrl_boundsfix_p2_c0_halt5e5_lr2e4_wd3e4_u50000_20260723_wd3`
- 更新数：50,000
- teacher state数：200,000
- SHA-256：`13a99d3dea24942e4766aaae340ed3ecf6d13448954e559845f3bd19fbee93de`
- 採用理由：品質が改善し、hard haltingが可変深度を維持し、実データ区間で`15 updates/s`以上を満たすため

参考用の65,000更新LR半減checkpointのSHA-256は`672d73a6a5cf1c40f1fa80fd10fd28b34bdd09e3465dcaf4c24602c9181e53f2`である。これは速度失格のため採用checkpointではない。

## PreActとの関係

既存のPreAct記録は、48,000 teacher stateでtop-1 `0.7891`、NDCG `0.99329`、pairwise `0.92336`、margin `0.12332`、loss `2.56378`だった。修正後EVRLの採用checkpointは200,000 teacher stateを使っているため、この二つを同一予算の最終比較として扱うことはできない。

参考差分として、EVRL 50,000更新は旧PreAct記録に対してtop-1 `-0.0547`、NDCG `-0.00558`、pairwise `-0.03929`、loss `+0.06837`、margin `+0.00827`である。ただし学習予算が異なるため、これは研究上の勝敗を示す値ではない。

したがって現在の正確な結論は次の通りである。

1. 修正後EVRLは学習可能であり、動的再帰も獲得している。
2. dense WD `3e-4`の50,000更新が、現在の品質・速度条件を満たす採用点である。
3. episodic LR半減は品質をさらに伸ばすが、長区間速度を悪化させた。
4. PreAct超えは未証明であり、修正後モデルによる同一入力、teacher state数、評価panel、実時間または計算予算での再比較が必要である。

## 評価上の境界

- validationおよびsealed game seedには触れていない。
- チューニングには固定teacher評価panelを使っている。
- 同じpanelを繰り返し観測しているため、上記の改善は学習挙動の証拠ではあるが、独立した汎化性能の最終証明ではない。
- 旧100,000更新試行は範囲外書込みを含むため、修正後モデルの性能根拠から除外する。

詳細な各試行の推移、checkpoint遷移、smoke結果は[`TRAINING_STABILITY_TUNING_2026-07-23.md`](TRAINING_STABILITY_TUNING_2026-07-23.md)に記録している。
