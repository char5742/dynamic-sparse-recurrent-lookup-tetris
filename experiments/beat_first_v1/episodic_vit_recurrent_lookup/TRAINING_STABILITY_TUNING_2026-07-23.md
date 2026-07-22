# EVRL 学習安定化調整（2026-07-23）

## 目的

動的再帰P1で見られたloss、top-1、平均深度の振動を、モデル全体のアーキテクチャを変えずに学習率、weight decay、state batchの調整で改善する。その前提として、同じ実teacher系列を長時間安全に学習できることを確認する。

validation subsetとsealed game seedは、この作業では構築も参照もしていない。

## 調整前に判明したbackwardの範囲外書込み

通常実行では200～400更新後にJuliaのGCが`EXCEPTION_ACCESS_VIOLATION`で停止した。scheduler、worker数、GC worker数、resume checkpointを変えても再現したため、`--check-bounds=yes`で既知P1 checkpointを再生した。

その結果、`_spatial_attention_vjp!`が8近傍attentionのedge 5～8を、長さ4の`BackwardScratch.dweights`へ書き込んでいたことが判明した。

```text
BoundsError: 4-element Vector{Float32} at index [5]
source: EpisodicViTRecurrentLookup.jl:_spatial_attention_vjp!
```

scratch容量はepisodic shortlist、register数、旧spatial shortlistの最大値だけで決められており、`LOCAL_SPATIAL_NEIGHBORS = 8`が含まれていなかった。修正は共有softmax VJP scratchの容量上限へ8近傍数を追加するだけであり、forward、backwardの数式、入力、teacher、loss、hard halting、RNG順序、active-only backward、sparse optimizer、checkpoint形式は変更していない。

## 数値一致smoke

10,000更新checkpointと同じreal-teacher 4状態を使い、single-worker oracleと20-worker barrierlessを`--check-bounds=yes`で比較した。

| 項目 | 結果 |
|---|---:|
| 出力最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw task VJP最大絶対差 | 0 |
| parameter gradient最大絶対差 | 4.59794e-6 |
| parameter gradient relative L2 | 2.04267e-6 |
| optimizer後parameter/state最大絶対差 | 4.60073e-7 |
| 離散経路、halt probe、RNG、sampler、optimizer clock | すべて一致 |
| 判定 | 合格 |

## 1,000更新の長時間preflight

実際に100,000更新を完了したP1の75,000更新checkpointから、checkpointを書き出さず1,000更新を再生した。20 Julia worker、BLAS 1 thread、barrierless、pinningなし、chunk 8を使用した。

| 項目 | 結果 |
|---|---:|
| 実行更新 | 1,000 |
| throughput測定更新 | 990（最初の10更新を除外） |
| 測定実時間 | 63.8505秒 |
| updates/s | 15.5050 |
| 全体CPU使用率 | 49.6387% |
| candidate executor CPU使用率 | 51.3630% |
| allocation | 7.739 MB/update |
| GC時間 | 0.5377秒（測定時間の0.84%） |
| GC／範囲外アクセス停止 | なし |
| 最低速度15 updates/s | 合格 |

## 既存結果の扱い

過去のP1 100,000更新は完走しているが、範囲外書込みを含むbackwardで作成されている。したがって既存checkpointは比較用の履歴として保持する一方、修正後モデルの最終性能根拠にはしない。以後のLR、weight decay、batch調整は修正版からscratch学習し、同一teacher-state予算で比較する。

## 次の調整順序

1. 修正版P1設定をscratchから再実行し、安定性の基準曲線を作る。
2. candidate-local probeは固定し、halt learning rateを一軸で下げて平均深度の振動を比較する。
3. dense learning rateとdense weight decayを一軸ずつ比較する。
4. state batch変更は、serial/barrierlessの数値一致とcheckpoint互換性を確認できた場合だけ採用する。
5. loss、top-1、NDCG、margin、平均深度が同時に安定する構成だけを100,000更新まで継続する。

