# エピソード記憶付きViT再帰LookupFFN

このディレクトリには、実teacherによるテトリス候補順位付けを対象とした、現行の動的疎再帰Lookupネットワーク（Dynamic Sparse Recurrent Lookup Network）を実装している。

## 最新の学習結果

2026-07-24の速度構成では、固定深度2を採用せず、haltingを候補状態とhalt weightの
正規化類似度で決める方式へ修正した。residual scale 0.55のスクラッチ学習を100,000更新
まで行い、training-only固定128状態でloss`2.734835`、top-1`0.585938`、
NDCG`0.980287`、pairwise`0.863809`、margin`0.079808`を得た。20,000更新からの
追加80,000更新は`27.936 updates/s`、累積100,000更新は`3,605.982秒`で完走した。
連続順位品質は80k～90kが強く、80kはNDCG`0.984419`、pairwise`0.868362`、90kは
最小loss`2.726930`を得た。90kのdeterministic深度は3～6へ明瞭に分布した一方、
100kでは99.25%が深度3へ再集中した。固定深度にはしていないが、品質と深度多様性の
同時収束は未達である。詳細は
[`DYNAMIC_HALTING_NORMALIZATION_2026-07-24.md`](DYNAMIC_HALTING_NORMALIZATION_2026-07-24.md)
を参照。

候補固有の各step 1-step信用割当と有界halting式を含む現行モデルを、親checkpointなしで
100,000更新した。400,000 teacher stateを処理し、`10.651 updates/s`、GC比率
`0.665%`で完走した。

training-only固定128状態による10k刻み再評価では、100kがloss`2.543041`、
NDCG`0.993363`、pairwise`0.921549`で最良となった。top-1は`0.703125`、
marginは`0.123395`、平均深度は`3.863`、深度範囲は3～6である。90kは
top-1`0.718750`で最高だったが、連続順位品質を優先し100kを主採用checkpointとする。

主採用checkpoint SHA-256：

```text
2eb9a64293eb2c592830d9c25ea52969950e6440a97415a43ec093d0a77c8766
```

validationとsealed seedは使用していない。全推移と評価境界は
[`HALTING_FULLSCRATCH_100K_2026-07-23.md`](HALTING_FULLSCRATCH_100K_2026-07-23.md)
を参照。

上記100,000更新は3段Lookup版の最終学習結果であり、現在のsource geometryとは異なる。
2026-07-23に、再帰step内のLookupを3段から1段へ縮約し、24x10セルworking memoryへ
共有3x3 depthwise convolutionを追加した。さらに2026-07-24に、registerから入力memory
への読出しを固定K=64のlearned WTA/hash routingへ変更した。選択64 token内だけで正確な
attentionを行い、全283 tokenの平均を常設の安全経路として残す。旧checkpointは研究履歴
として保持し、新しい1段Lookup＋固定K64 geometryへ暗黙変換しない。移行の詳細は
[`SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md`](SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md)
および
[`FIXED_K64_EPISODIC_ROUTING_2026-07-24.md`](FIXED_K64_EPISODIC_ROUTING_2026-07-24.md)
を参照。

固定K64版は実teacher 10,000更新、40,000 stateまで完走した。training-only固定128状態で
loss`2.858829`、top-1`0.453125`、NDCG`0.974179`、pairwise`0.834829`、
平均深度`3.061`、深度範囲3～6を得た。1kから10kまでlossと全順位指標が改善し、
最終batchでは深度2～9、probe教師continue 4／stop 4となった。追加9,000更新の速度は
`11.4605 updates/s`で、固定K64版に許容した最低10 updates/sを満たした。

固定supportをK=64、80、88、96、128で比較した結果、K=96は短期NDCGとpairwiseを
わずかに改善したが、5,000更新で`9.90695 updates/s`まで低下した。K=80と88は
順位品質、K=128は速度が基準未達だったため、production既定値はK=64を維持する。
詳細は
[`FIXED_K_SUPPORT_TUNING_2026-07-24.md`](FIXED_K_SUPPORT_TUNING_2026-07-24.md)
を参照。

K幅変更後の根本速度改善では、Lookup bank容量を維持したままregister workspaceと
projection、SwiGLU幅、backward tailを見直した。速度合格候補は6,897,248 parameter、
2 register、attention 16、1 head、FFN 32である。固定深度2のproduction相当計測は
`41.545 updates/s`だったが、これは1 step単価の比較上限であり採用モデルではない。
sampled hard haltingを有効化した実測は平均深度`2.979`で`27.614 updates/s`となり、
動的段階の下限20を満たした。
serial/barrierless smokeと551件の回帰テストも合格している。品質はまだ未確定なので、
この構成を直ちに最終性能checkpointとは扱わない。全試行と評価境界は
[`ROOT_SPEED_TUNING_2026-07-24.md`](ROOT_SPEED_TUNING_2026-07-24.md)
を参照。

同構成をdynamic-from-scratchで5,000更新したpilotは`25.866 updates/s`で完走した。
固定training panel 128状態ではloss`3.023583`、top-1`0.4765625`、
NDCG`0.968138`、margin`0.050655`、平均深度`3.01195`だった。動的haltingと
stateあたり4候補の1-step probeを含む詳細100更新でも`27.451 updates/s`、
GC時間比`1.656%`を確認した。深度2～6のrandom-depth warmupは
`17.368 updates/s`で下限を割ったため不採用とし、最初の更新からhaltingを学習する。

## アーキテクチャ

各候補は、PreActベースラインと同じ入力フィールド、すなわち盤面、候補、差分、NEXT/HOLD、`aux37`から独立に評価される。teacher Q値と順位は教師信号としてのみ用いる。

モデルは次の要素で構成される。

1. 位置情報付きセルtoken、NEXT/HOLD token、補助tokenからなるエピソード記憶
2. 生の盤面・候補・差分channelに対する5段のdilated depthwise/pointwise視覚残差経路。受容野は`63 x 63`で、`24 x 10`盤面全体を覆う
3. 共有3x3 depthwise convolution、および共有Q/K/V/O射影と3x3相対位置biasを備えた物理的に疎なlearned local-8 spatial attentionによって、各再帰stepで更新されるセルworking memory
4. 複数の再帰register
5. 各registerがlearned WTA/hashで283 tokenから固定64 tokenを取得し、そのsupport内だけで行うexact cross-attention。全token平均の常設残差経路により入力切断を防ぐ
6. learned register self-attentionとSwiGLU変換
7. 各registerが単一の共有bankから独立にroutingするactive-only LookupFFN長期記憶。再帰step内でLookupを重ねず、必要な追加深度は外側の再帰が担う
8. 更新後registerから読出しと同じ選択supportへだけ書き戻す、入力固有の物理的に疎なread/write working memory
9. 残差再帰更新とhard halting interface

scoreを全要素について計算した後にdense maskをかける実装や、CountSketchは使用しない。
小型routerは283 tokenの低次元descriptorを走査するが、K/V射影、QK attention、
working-memory write、各VJPは選択された64 token接続に限定する。LookupFFN長期記憶の
行もforward、backward、optimizer更新のすべてで物理的疎性を維持する。

## CPU実行

`barrierless_executor.jl`は複数stateの候補を一つのglobal queueへflattenする。20個のnative workerがchunk 8単位で動的に仕事を取得し、継続候補は再帰step間でcompactされる。BLASのthread数は1である。Windows CPU Setsにも対応しているが、試験機で確認された最速設定はpinningなしである。

`barrierless_postphase.jl`はworker-localなdense勾配とsparse勾配を決定論的にreduceし、標準のglobal gradient clippingを適用し、optimizer clockとcheckpoint互換性を保つ。serial/barrierless smokeでは、出力、loss、gradient、routing選択、RNG state、optimizer telemetry、更新後parameter stateの一致を検証する。

## 主要ファイル

- `EpisodicViTRecurrentLookup.jl`：モデル、sparse attention、再帰、VJP、optimizer semantics
- `teacher_training.jl`：実teacher学習とcheckpoint lifecycle
- `barrierless_executor.jl`：動的candidate実行
- `barrierless_postphase.jl`：決定論的reduceとoptimizer phase
- `barrierless_correctness_smoke.jl`：single-thread oracleとの比較
- `test_single_lookup_recurrent_depthwise.jl`：単一Lookup契約とrecurrent DWConv VJPの有限差分検証
- `test_fixed_k64_episodic_lookup.jl`：固定64 support、同一support書込み、全token平均経路、router勾配の検証
- `evaluate_halting_convergence.jl`：validationを構築しない固定training panelのcheckpoint深度評価
- `bounded_mpmc_queue.jl`：Windows向けbounded allocation-free MPMC queue
- `windows_cpu_sets.jl`：実行時P/Eコア検出と、任意のCPU Sets割り当て
- `run_teacher_signal.jl`：学習entry point
- `CURRENT_FINDINGS_2026-07-23.md`：修正後EVRLの現行構成、全チューニング結果、採用checkpoint、PreAct比較の現在地
- `TRAINING_STABILITY_TUNING_2026-07-23.md`：8近傍backward scratch修正と、修正後のLR・weight decay・batch安定化記録
- `ROOT_CAUSE_REPAIR_100K_2026-07-23.md`：Lookup collapse、register別取得、working-memory書込み、halting信用割当の根本修正と100,000更新結果
- `ROOTFIX_CHECKPOINT_SWEEP_2026-07-23.md`：根本修正版の10,000更新刻み同一パネル再評価と、90,000更新checkpointの採用根拠
- `HALTING_CONVERGENCE_REPAIR_2026-07-23.md`：各再帰判断への候補固有1-step信用割当とhalting収束試験
- `HALTING_FULLSCRATCH_100K_2026-07-23.md`：現行最終式のフルスクラッチ100,000更新、全checkpoint推移、主採用点
- `SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md`：3段Lookupから単一Lookup＋反復DWConvへの移行とsmoke結果
- `SINGLE_LOOKUP_BOTTLENECK_PROFILE_2026-07-23.md`：単一Lookup化後のphase別速度、演算内訳、残存ボトルネック
- `FIXED_K64_EPISODIC_ROUTING_2026-07-24.md`：固定K64の疎いepisodic read/write、数値一致、速度、学習率比較
- `FIXED_K_SUPPORT_TUNING_2026-07-24.md`：固定K=64/80/88/96/128の同一条件比較と採否
- `ROOT_SPEED_TUNING_2026-07-24.md`：K幅変更後の根本速度改善、40 updates/s候補、動的halting速度、数値一致

## 速度合格候補

次のgeometryは速度と数値一致に合格したが、長期品質は未確定である。

```text
carrier/model dim          128
Lookup tables per block     13
WTA choices                 16
rows selected per table      3
Lookup blocks per step        1
attention dim               16
attention heads              1
registers                    2
episodic cross support       64 / register
register projection  structured
spatial recurrent path depthwise_only
SwiGLU FFN dim               32
forward queue chunk           8
backward queue chunk          1
recurrent depth            2--12
```

固定深度2は再帰stepあたりの速度上限を測る比較専用である。本学習では最初の更新から
sampled hard haltingと候補固有1-step probeを用い、速度のために平均深度を固定しない。

## 検証済みproduction geometry

```text
carrier/model dim          128
Lookup tables per block     13
WTA choices                 16
rows selected per table      3
Lookup blocks per step        1
attention dim               32
attention heads              4
registers                    4
episodic cross support       64 / register
local spatial neighbours     8
recurrent DWConv kernel     3x3
SwiGLU FFN dim              128
recurrent depth             2--12
```

hard haltingはcandidate-local 1-step probeによる停止教師とともに有効である。probe対象
候補では、既に保存した各再帰stepのQへその候補だけを差し替え、同じListNetとmarginを
再計算する。これにより、最終stopだけでなく、それ以前の全stochastic continue判断へも
`L_stop - L_continue`に基づく候補固有の教師を与える。追加forwardは従来どおり最終
stop候補の1stepだけで、全深度展開は行わない。

probeされないtrajectoryには縮小したpolicy gradientを残し、probe済みtrajectoryでは
高分散なterminal creditと正確なtrace教師を重ねない。さらにstep 4を中心とする単調な
停止hazard priorへ`0.75*tanh(learned_logit)`の有界な入力依存残差を加え、entropy
annealingとhalt learning-rate decayを併用する。learned headだけでpriorを無制限に
上書きできないため、global biasの小変化による最小／最大深度の往復を抑えられる。詳細は
[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md)、
[`ROOT_CAUSE_REPAIR_100K_2026-07-23.md`](ROOT_CAUSE_REPAIR_100K_2026-07-23.md)、
[`HALTING_CONVERGENCE_REPAIR_2026-07-23.md`](HALTING_CONVERGENCE_REPAIR_2026-07-23.md)
を参照。

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
