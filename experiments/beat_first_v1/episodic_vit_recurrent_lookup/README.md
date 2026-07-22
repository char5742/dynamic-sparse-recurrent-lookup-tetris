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
7. active-onlyなLookupFFN長期記憶
8. 残差再帰更新とhard halting interface

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
fixed recurrent depth        2
```

hard haltingの実装は維持しているが、20,000更新の表現学習では深度2に固定した。意図した最終評価protocolにおいて空間表現とrouting表現が安定していることを確認した後にのみ、再度有効化する。

動的学習には、候補単位の1-step probeを任意で有効にできる。sampled stopのうち有界な個数だけをprobeし、その候補のQだけを置換した後にListNetとmarginを再計算し、`L_stop - L_continue`から最終停止判断を教師あり学習する。この方式は物理的疎性を保ち、有効化時には従来のstate-wide REINFORCEによる信用割当を置き換える。詳細は[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md)を参照。

完了済みの100,000更新probe試験では、stateあたり2候補をprobeした。品質と深度の釣り合いが最良だった95,000更新checkpointは、top-1 `0.73438`、NDCG `0.991345`、margin `0.14191`、held平均深度`2.19`に到達した。90,000更新では、平均深度`3.02`で試験中最高のtop-1 `0.74219`を記録した。旧state-wide halting試験の最終値を全品質指標で上回り、旧試験で生じた最終深度12への飽和も回避した。ただし学習後半の深度は依然として下限寄りである。この結果は、信用割当の改善には成功したが、適応的深度の獲得は部分的成功にとどまると記録している。

正確な数値witnessは[`RESULTS_2026-07-20.md`](RESULTS_2026-07-20.md)、PreActとの最終held-teacher比較は[`PERFORMANCE_COMPARISON_2026-07-20.md`](PERFORMANCE_COMPARISON_2026-07-20.md)を参照。

その後の入力routing ablationでは、従来の`283 -> 64 -> 16`というregister memory bottleneckを撤去した。同じ12,000更新予算で、全token cross-attentionはtop-1を`0.35938`から`0.56250`へ改善した。一方、CPU推論速度は`53.54`から`45.17` states/sへ低下した。詳細は[`TOKEN_ROUTING_ABLATION_2026-07-21.md`](TOKEN_ROUTING_ABLATION_2026-07-21.md)を参照。

現行の視覚拡張はdilation `1,2,4,8,16`を用い、追加parameterは565、candidateあたりの追加scalar MACは135,360にすぎない。実受容野は`63 x 63`に達する。25,000更新、100,000 teacher stateで、top-1 `0.68750`、NDCG `0.98586`、margin `0.13215`、held CPU速度`43.72` states/sを得た。詳細は[`GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md`](GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md)を参照。
