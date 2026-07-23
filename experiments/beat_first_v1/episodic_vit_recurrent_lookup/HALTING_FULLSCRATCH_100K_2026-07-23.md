# 現行EVRLのフルスクラッチ100,000更新

## 目的

候補固有の1-step halting信用割当と有界な停止hazardを含む現行EVRLを、既存checkpointへ
依存せず初期値から100,000更新する。途中の10,000更新刻みcheckpointを同じ
training-only固定パネルで再評価し、最終更新を自動採用せず、品質推移と動的深度を確認する。

この試行ではモデル構造、入力、teacher、損失、optimizer semantics、hard halting、
active-only backward、sparse optimizerを変更していない。validation行とsealed seedは
使用していない。

## 実行条件

- run ID：`evrl_halt_bounded_fullscratch_u100000_20260723`
- 親checkpoint：なし
- 開始更新：0
- 終了更新：100,000
- state batch：4
- 処理teacher state：400,000
- 総parameter数：20,585,982
- 再帰深度：2～12
- register数：4
- model次元：128
- LookupFFN：3共有block、各block 13 table、各table 4,096 row、top-3
- token cross-attention：4 registerから全283 token
- halting probe：4候補/state
- halt LR：`5e-5`から、20k～60kで減衰し、最小`1e-6`
- stopping hazard：

```text
halt_logit = 0.5 * (step - 4) + 0.75 * tanh(learned_logit)
```

- scheduler：barrierless global candidate queue、20 worker、pinningなし、chunk 8
- CPU構成：実測8 P-core＋12 E-core
- BLAS thread：1
- checkpoint間隔：10,000更新
- 最低速度条件：10 updates/s

run設定のsource fingerprintは
`5a3bf5a1767f88a87c842d6b2ff96af7e3236efef02e540fd9df8f515f141170`
である。

## 完走結果

100,000更新を正常完了し、10個のcheckpointを全て保存した。学習本体の計測時間は
9,389.019秒、`10.651 updates/s`、`42.603 states/s`だった。scheduler全体の
計測wall timeは9,532.895秒、`10.490 updates/s`である。最低速度10 updates/sを
割らず、throughput停止は発生しなかった。

| 指標 | 結果 |
|---|---:|
| 学習本体の平均CPU使用率 | 54.255% |
| candidate処理中CPU使用率 | 57.023% |
| allocation | 8.383 MB/update |
| GC時間 | 63.366秒 |
| scheduler wall timeに対するGC比率 | 0.665% |
| 最終batch loss | 2.943657 |
| 最終batch平均深度 | 2.994652 |
| 最終batch深度範囲 | 2～10 |
| 最終halt LR | `1e-6` |

allocationの主因は引き続きbackwardであるが、GC比率は1%未満であり、今回の100k完走を
阻害しなかった。

## 同一パネル再評価

追加学習なしで10k～100kの全checkpointを、training splitから固定した同じ128状態で
再評価した。パネル行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
である。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 | 深度範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | 2.811022 | 0.492188 | 0.979928 | 0.853572 | 0.087983 | 3.066 | 3～6 |
| 20k | 2.644648 | 0.609375 | 0.986693 | 0.879061 | 0.103333 | 3.296 | 3～6 |
| 30k | 2.583828 | 0.625000 | 0.988939 | 0.893631 | 0.127713 | 3.530 | 3～6 |
| 40k | 2.566999 | 0.617188 | 0.990049 | 0.902571 | 0.125465 | 3.149 | 3～6 |
| 50k | 2.573094 | 0.664062 | 0.990681 | 0.903048 | **0.143843** | 4.891 | 3～6 |
| 60k | 2.561678 | 0.671875 | 0.991185 | 0.907722 | 0.128534 | 3.012 | 3～6 |
| 70k | 2.552303 | 0.671875 | 0.992075 | 0.911935 | 0.124916 | 3.304 | 3～6 |
| 80k | 2.571582 | 0.695312 | 0.992686 | 0.917855 | 0.112140 | 4.496 | 3～6 |
| 90k | 2.550394 | **0.718750** | 0.992744 | 0.915829 | 0.118151 | 3.026 | 3～6 |
| **100k** | **2.543041** | 0.703125 | **0.993363** | **0.921549** | 0.123395 | 3.863 | 3～6 |

100kはcomposite loss、NDCG、pairwiseが最良であり、主採用checkpointとする。90kは
top-1が最高なので、離散的な首位一致率を優先する用途の参考点として保持する。50kは
marginのみ最高だった。

100kの決定論的深度分布は次の通りである。

| 深度 | candidate数 | 割合 |
|---:|---:|---:|
| 3 | 3,314 | 61.87% |
| 4 | 330 | 6.16% |
| 5 | 847 | 15.81% |
| 6 | 866 | 16.17% |

全candidateが一つの深度へ崩壊しておらず、入力依存の停止は成立している。一方、50～100kの
平均深度は`4.891 -> 3.012 -> 3.304 -> 4.496 -> 3.026 -> 3.863`と動いた。
従来の2～12両端への崩壊は抑えられたが、checkpoint間の平均深度が数値的に収束したとは
判定しない。

## 既存主採用点との比較

同じ固定training-onlyパネルと最終halting式で再評価済みだった旧90k主採用点と比較する。

| 指標 | 旧90k | 今回100k | 差 |
|---|---:|---:|---:|
| loss | 2.551701 | 2.543041 | -0.008660 |
| top-1 | 0.671875 | 0.703125 | +0.031250 |
| NDCG | 0.991979 | 0.993363 | +0.001384 |
| pairwise | 0.913475 | 0.921549 | +0.008074 |
| margin | 0.116506 | 0.123395 | +0.006889 |
| 平均深度 | 4.004 | 3.863 | -0.141 |

今回のフルスクラッチ100kは、旧主採用点を全ての順位品質指標で上回った。そのため現行
主採用checkpointを今回の100kへ更新する。

## PreActとの境界

既存PreAct 12.75kはtop-1 `0.789062`、NDCG `0.994468`、pairwise `0.930203`、
margin `0.116617`、loss `2.550905`だった。今回100kはlossとmarginでは上回る一方、
top-1、NDCG、pairwiseでは届いていない。また、今回EVRLは400,000 teacher state、
PreActは51,000 teacher stateであり、sample efficiencyは依然としてPreActが高い。

この評価はtraining-only固定パネルであり、独立汎化性能の証明ではない。validationと
sealed seedを使わない条件を守ったため、今回の結果だけからPreAct超えを主張しない。

## checkpoint

主採用100k：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
evrl_halt_bounded_fullscratch_u100000_20260723\checkpoints\
checkpoint_000100000.jls
```

SHA-256：

```text
2eb9a64293eb2c592830d9c25ea52969950e6440a97415a43ec093d0a77c8766
```

top-1参考90k SHA-256：

```text
67b1d2f81d63757e2d1697617b19d4157c9531cddd8096ec3c88686a0e7eb3bc
```

全評価結果はrun内の`halting_convergence_training_panel.json`に保存した。

## 結論

現行EVRLは親checkpointなしで100,000更新を完走し、lossとNDCGは10kから100kまで
長期的に改善した。100kが複数の連続順位指標で最良になったため、旧試行のような
「最終更新より手前が明確に最良」という結果ではない。ただし後半の改善量は小さく、
top-1と平均深度には振動が残る。次の改善対象は単純な更新追加より、品質を維持したまま
halting方策のcheckpoint間モード移動をさらに抑えることである。
