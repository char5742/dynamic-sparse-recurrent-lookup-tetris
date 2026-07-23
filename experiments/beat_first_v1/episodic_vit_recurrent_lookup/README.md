# エピソード記憶付きViT再帰LookupFFN

このディレクトリには、実teacherによるテトリス候補順位付けを対象とした、現行の動的疎再帰Lookupネットワーク（Dynamic Sparse Recurrent Lookup Network）を実装している。

## アーキテクチャ

各候補は、PreActベースラインと同じ入力フィールド、すなわち盤面、候補、差分、NEXT/HOLD、`aux37`から独立に評価される。teacher Q値と順位は教師信号としてのみ用いる。

モデルは次の要素で構成される。

1. 位置情報付きセルtoken、NEXT/HOLD token、補助tokenからなるエピソード記憶
2. 生の盤面・候補・差分channelに対する5段のdilated depthwise/pointwise視覚残差経路。受容野は`63 x 63`で、`24 x 10`盤面全体を覆う
3. 共有Q/K/V/O射影と3x3相対位置biasを備えた、物理的に疎なlearned local-8 spatial attentionによって反復更新されるセル記憶
4. 複数の再帰register
5. 各registerから全283個のエピソードtokenへのexact cross-attention。K/V射影は再帰stepごとに1回だけ共有計算する
6. learned register self-attentionとSwiGLU変換
7. 各registerが共有bankから独立にroutingするactive-only LookupFFN長期記憶
8. 更新後registerからepisodic tokenへ書き戻す、入力固有のread/write working memory
9. 残差再帰更新とhard halting interface

scoreを全要素について計算した後にdense maskをかける実装や、CountSketchは使用しない。小規模な`4 x 283`のregister/token score領域は直接評価する一方、LookupFFN長期記憶の行はforward、backward、optimizer更新のすべてで物理的疎性を維持する。

## CPU実行

`barrierless_executor.jl`は複数stateの候補を一つのglobal queueへflattenする。20個のnative workerがchunk 8単位で動的に仕事を取得し、継続候補は再帰step間でcompactされる。BLASのthread数は1である。Windows CPU Setsにも対応しているが、試験機で確認された最速設定はpinningなしである。

`barrierless_postphase.jl`はworker-localなdense勾配とsparse勾配を決定論的にreduceし、標準のglobal gradient clippingを適用し、optimizer clockとcheckpoint互換性を保つ。serial/barrierless smokeでは、出力、loss、gradient、routing選択、RNG state、optimizer telemetry、更新後parameter stateの一致を検証する。

## 主要ファイル

- `EpisodicViTRecurrentLookup.jl`：モデル、sparse attention、再帰、VJP、optimizer semantics
- `teacher_training.jl`：実teacher学習とcheckpoint lifecycle
- `barrierless_executor.jl`：動的candidate実行
- `barrierless_postphase.jl`：決定論的reduceとoptimizer phase
- `barrierless_correctness_smoke.jl`：single-thread oracleとの比較
- `bounded_mpmc_queue.jl`：Windows向けbounded allocation-free MPMC queue
- `windows_cpu_sets.jl`：実行時P/Eコア検出と、任意のCPU Sets割り当て
- `run_teacher_signal.jl`：学習entry point
- `CURRENT_FINDINGS_2026-07-23.md`：修正後EVRLの現行構成、全チューニング結果、採用checkpoint、PreAct比較の現在地
- `TRAINING_STABILITY_TUNING_2026-07-23.md`：8近傍backward scratch修正と、修正後のLR・weight decay・batch安定化記録
- `ROOT_CAUSE_REPAIR_100K_2026-07-23.md`：Lookup collapse、register別取得、working-memory書込み、halting信用割当の根本修正と100,000更新結果
- `ROOTFIX_CHECKPOINT_SWEEP_2026-07-23.md`：根本修正版の10,000更新刻み同一パネル再評価と、90,000更新checkpointの採用根拠

## 検証済みproduction geometry

```text
carrier/model dim          128
Lookup tables per block     13
WTA choices                 16
rows selected per table      3
attention dim               32
attention heads              4
registers                    4
episodic cross support      283
local spatial neighbours     8
SwiGLU FFN dim              128
recurrent depth             2--12
```

hard haltingはcandidate-local 1-step probeによる停止教師とともに有効である。修正後の
20,000更新基準では、held平均深度が`2.00 -> 2.11 -> 2.40 -> 3.56`と増え、
最小2、最大12を入力ごとに使い分けながら、held lossとNDCGを全評価点で改善した。

動的学習には、候補単位の1-step probeを任意で有効にできる。sampled stopのうち有界な個数だけをprobeし、その候補のQだけを置換した後にListNetとmarginを再計算し、`L_stop - L_continue`から最終停止判断を教師あり学習する。この方式は物理的疎性を保つ。現行実装では全stop/continueへtrajectory policy gradientを与え、probe BCEは対象候補の最終stopに加算する。詳細は[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md)と[`ROOT_CAUSE_REPAIR_100K_2026-07-23.md`](ROOT_CAUSE_REPAIR_100K_2026-07-23.md)を参照。

根本修正後の100,000更新では、Lookup row-load Giniが`0.187～0.193`、coverageが`99.7～99.9%`となり、旧routing collapseを解消した。register間のLookup address相違率は`99.90%`、working-memory write RMSは`0.384`、training-only固定128状態の深度範囲は`3～12`だった。学習速度は`11.332 updates/s`、GC占有率は`0.667%`である。

同runの10,000更新刻みcheckpointを追加学習なしで再評価した結果、主採用点は90,000更新
となった。固定128状態でloss `2.551617`、top-1 `0.671875`、NDCG `0.991901`、
pairwise `0.913689`を得て、100,000更新よりmargin以外の総合順位品質が高かった。

完了済みの旧100,000更新probe試験では、stateあたり2候補をprobeした。品質と深度の
釣り合いが最良だった95,000更新checkpointは、top-1 `0.73438`、NDCG `0.991345`、
margin `0.14191`、held平均深度`2.19`に到達した。ただし、この試験は後に判明した
8近傍spatial backwardの範囲外書込みを含むため、研究履歴としてのみ保持し、修正後
モデルの最終性能根拠には使用しない。

正確な数値witnessは[`RESULTS_2026-07-20.md`](RESULTS_2026-07-20.md)、PreActとの最終held-teacher比較は[`PERFORMANCE_COMPARISON_2026-07-20.md`](PERFORMANCE_COMPARISON_2026-07-20.md)を参照。

その後の入力routing ablationでは、従来の`283 -> 64 -> 16`というregister memory bottleneckを撤去した。同じ12,000更新予算で、全token cross-attentionはtop-1を`0.35938`から`0.56250`へ改善した。一方、CPU推論速度は`53.54`から`45.17` states/sへ低下した。詳細は[`TOKEN_ROUTING_ABLATION_2026-07-21.md`](TOKEN_ROUTING_ABLATION_2026-07-21.md)を参照。

現行の視覚拡張はdilation `1,2,4,8,16`を用い、追加parameterは565、candidateあたりの追加scalar MACは135,360にすぎない。実受容野は`63 x 63`に達する。25,000更新、100,000 teacher stateで、top-1 `0.68750`、NDCG `0.98586`、margin `0.13215`、held CPU速度`43.72` states/sを得た。詳細は[`GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md`](GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md)を参照。
