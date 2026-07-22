# EVRL動的再帰の有効化試験 — 2026-07-21

## 問い

global visual pathとfull 283-token accessを持つEVRLは、recurrent depthを2に固定して学習していた。本試験では、学習途中からsampled hard haltingを有効化できるか、halt headを初期化し直すべきか、または動的depthを最初から導入する必要があるかを検証した。

model body、prediction head、LookupFFN routing、入力、teacher、ranking loss、candidate独立評価、sparse optimizer、real-teacher samplerは変更していない。game validationとsealed seedも未使用である。

trainerが許可する明示的なcheckpoint semantics変更は、`EVRL_ENABLE_DYNAMIC_HALTING_TRANSITION=1`指定時の「継承した非zero固定depthからsampled hard haltingへ」の1種類だけとした。optimizer、routing、loss hyperparameterは完全一致を必須とした。`EVRL_DYNAMIC_HALT_RESET_PROBABILITY`を指定した場合は、このtransition中にhalt weight、bias、それらのAdam momentだけを初期化する。

## 固定深度の親モデル

親モデルは以前の25,000-update報告後も学習を続け、update 75,000で次のheld-panel結果を得た。

| Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|
| 2.591471 | 0.71875 | 0.990663 | 0.900622 | 0.137788 |

dynamic transitionの各branchは、保存済みupdate 80,000 checkpointから開始した。

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_visual_dw5_rf63_fixed2_u100000_20260721_r1\checkpoints\checkpoint_000080000.jls
sha256: 332728081d195814d7b2bcd17728fce83f92fe8a430cd62b2e936ea39d34e104
consumed real-teacher states: 320,000
```

## 結果

| 分岐 | 学習上の介入 | 最終更新 | Held loss | Top-1 | NDCG | Pairwise | Margin | Held決定論的深度 | 終了時の学習sampled depth | Updates/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 途中から直接導入 | 80kでhard haltingを有効化 | 90k | 2.588522 | 0.72656 | 0.991547 | 0.905774 | 0.141727 | 2 / 2 / 2 | mean 2.25, range 2–5 | 27.48 |
| 途中からcurriculum | 5kだけdepth 2–6、その後hard halting | 90k | 2.586021 | 0.71875 | 0.991611 | 0.906437 | 0.145673 | 2 / 2 / 2 | mean 2.23, range 2–4 | release後28.16 |
| Scratch開始 | 最初の5kをdepth 2–6、その後hard halting | 20k | 2.735993 | 0.53906 | 0.983724 | 0.866709 | 0.091127 | 2 / 2 / 2 | mean 2.29, range 2–5 | 26.16 |

`2 / 2 / 2`はheld-panelのmean／minimum／maximumを示す。curriculum区間自体は平均depth 4.0、19.06 updates/sで動作した。より深いrecurrent stateでbodyを改善したが、hard haltingを解放するとdeterministic held sampleはすべてdepth 2へ戻った。

途中から直接導入したbranchではupdate 85,000と90,000で最高top-1 `0.7265625`を得た。95,000まで続けると`0.7109375`へ低下したため、95,000 checkpointは採用しなかった。

from-scratch branchもランキング課題自体は正常に学習し、top-1は初期`0.21875`から5,000で`0.50781`、20,000で`0.53906`へ上昇した。したがってdeterministic variable depthを獲得できなかった原因を、動的再帰の導入時期だけでは説明できない。

## Halt-head初期化ablation

halt headだけをprobability `0.4`へ初期化すると、deterministic inferenceは即座に最大depth 12まで実行し、held loss `7.0786`、top-1 `0.328`となった。その後数千updateでtraining sampled depthは約2.4–2.5へ戻った。このbranchは停止し、不採用とした。

prediction headの初期化は試していない。既に学習したteacher-Q／ranking mapを破壊するだけで、不足しているrecurrent credit signalを与えないためである。halt-only resetはその仮説をより狭く検証したものだが、不十分だった。

## 診断

当時のpolicy-gradient targetは、1つのstate内にある全candidate trajectoryへ同じ実現state-level ranking lossを与えていた。あるcandidateを追加1step進めたことが競合candidateに対して有利だったかを、そのcandidate自身へ教えていない。そのためcompute priceは早期停止への明確な圧力となる一方、追加stepの利益はnoisyでcandidate list全体に結合されていた。

次の根拠がこの解釈を支持する。

1. deeper-state curriculumはcontinuous ranking指標を改善する。
2. stochastic training trajectoryはdepth 3–5を探索し続ける。
3. deterministic held inferenceは全branchでdepth 2のままである。
4. halt-head resetとfrom-scratch trainingでもcollapseは解消しない。

したがって次の介入対象はprediction layerの初期化や学習済みmodel bodyの置換ではなく、haltingの信用割当である。物理的疎性を保つ方法として、少数candidateだけを追加1step probeし、他candidate scoreを固定したままstate ranking lossの正確な変化を教師にできる。compute priceはこの獲得段階では下げ、halt headが有用／無用な追加計算を分離できてから戻すべきである。

## Checkpoint検証情報

```text
direct dynamic 85k:
  sha256 d9547600f987138f3c5415400743d817b18f07d7776be5d6ece220ca7f5a5749

direct dynamic 90k:
  sha256 7096d3a625671d5a6754768c4af9adb094b671e6ea61b051d9a142e50981729b

curriculum then dynamic 90k:
  sha256 70bf69636753d8da97c2ebb30ffda0a32eaeb7f446ac46f62d5225ad26944735

from-scratch dynamic 20k:
  sha256 e2a96fb65bb8cedf68b484f409d84aa8d219bd0795e7afb4b188bbdecd1a9e31
```

binary checkpointとteacher dataはGitへcommitしていない。
