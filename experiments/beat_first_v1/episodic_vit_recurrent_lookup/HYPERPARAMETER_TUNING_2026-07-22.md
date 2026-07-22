# EVRL固定アーキテクチャのhyperparameter調整 — 2026-07-22

> 対象範囲の訂正：以下の試験1～4は、固定深度でのoptimizerおよび正則化controlである。品質baselineとしては有効だが、動的再帰を調整した試験ではない。再帰に焦点を当てた試験は、別途[`DYNAMIC_RECURRENCE_TUNING_2026-07-22.md`](DYNAMIC_RECURRENCE_TUNING_2026-07-22.md)に記録している。

## 固定条件

本記録のすべての試験で、EVRLアーキテクチャ全体を固定した。

- 20,577,789 parameter
- PreActと同一の入力境界および実teacher dataset
- 283-token full cross-attentionと4個のrecurrent register
- `63 x 63`受容野を持つ5段dilated depthwise/pointwise視覚経路
- learned local-8 spatial Q/K/V/O relation
- 共有LookupFFN micro-layer 3段、blockあたり13 table、tableあたり4,096 row、物理的top-3 active-row更新
- candidate独立評価、ranking objective、sampler seed、split、128-state held panel
- 20-worker barrierless executor、CPU pinningなし、chunk 8
- game validationおよびsealed seedへはアクセスしない

以下の試験で変更したのは、optimizer、weight decay、routing scheduleに関するscalar hyperparameterだけである。再帰深度は2に固定した。採用対象の比較は100,000更新まで実行し、5,000更新ごとにheld metricを記録し、次の試験を始める前にcommitおよびpushした。

## 試験1 — 固定深度baseline完了

既存baselineの正確な80,000更新checkpointから100,000更新まで学習を継続した。後続のscratch開始100,000更新調整試験に対するcontrolであり、hyperparameterは一切変更していない。

### Hyperparameter

```text
bank LR       2e-4        bank weight decay    0
router LR     4e-4        dense weight decay   1e-4
attention LR  2e-4        gradient clip        5
FFN LR        2e-4        route temperature    1.0 -> 0.25 by update 12k
token LR      2e-4        recurrent depth      fixed 2
register LR   2e-4
head LR       2e-4
```

### Held panel推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin | 区間updates/s | 区間CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80,000 | 2.596073 | 0.69531 | 0.990361 | 0.901688 | 0.133506 | — | — |
| 85,000 | 2.591489 | **0.73438** | 0.991055 | 0.905273 | 0.134817 | 29.61 | 69.94% |
| 90,000 | 2.582912 | 0.72656 | 0.991496 | **0.906543** | 0.128577 | 30.76 | 72.46% |
| 95,000 | 2.583290 | 0.70312 | 0.991298 | 0.904486 | 0.140228 | 31.57 | 74.01% |
| 100,000 | **2.581711** | 0.71875 | **0.991801** | 0.906227 | **0.146405** | **31.92** | **74.63%** |

lossとNDCGは100,000更新まで改善を続けた一方、離散的なtop-1は85,000更新で最高値を記録した後に振動した。lossが明確にplateauしたわけではないが、top-1の分散を考えるとstep sizeを増やすのではなく、保守的なlearning-rate試験が妥当である。最初の調整armでは、sparse routing容量を担う高めのrouter LRとbank LRを維持し、dense representation LRだけを`0.75`倍する。これにより、後半の表現更新安定性を分離して評価する。

### 実行時間とwitness

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_baseline_fixed2_u100000_20260722_r1

training time for updates 80k -> 100k: 626.580054 s
updates/s:                         31.919305
states/s:                          127.677221
average CPU:                       74.6275%
candidate CPU:                     77.6558%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_baseline_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: ba8620d7caa3a648c7b73635005e400cf96944de36708bf17027537a1ca7e553
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  e2c4e4d4f2c8a243452dd65a417e3c80c610f6dc83983b31128e0b791ee1f095
```

binary checkpointとteacherデータはGitへcommitしていない。

## 試験2 — dense representation LRを0.75倍

scratchから開始した本試験では、attention、FFN、token/visual、register、output headという5種類のdense representation learning rateだけを`2e-4`から`1.5e-4`へ変更した。bank LRは`2e-4`、router LRは`4e-4`、Lookup residual-alpha LRは`2e-4`、dense weight decayは`1e-4`のままであり、アーキテクチャ、seed、データ、loss、routing schedule、固定深度設定はすべて試験1と同一である。

### Held panel推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.856116 | 0.53906 | 0.978386 | 0.837953 | 0.057290 |
| 10,000 | 2.810584 | 0.51562 | 0.979042 | 0.845513 | 0.072452 |
| 15,000 | 2.813538 | 0.50781 | 0.980807 | 0.850272 | 0.070100 |
| 20,000 | 2.791762 | 0.54688 | 0.981539 | 0.853233 | 0.064467 |
| 25,000 | 2.763789 | 0.53906 | 0.981845 | 0.860536 | 0.089816 |
| 30,000 | 2.747787 | 0.54688 | 0.980351 | 0.858367 | 0.089927 |
| 35,000 | 2.742353 | 0.53125 | 0.981982 | 0.865703 | 0.101573 |
| 40,000 | 2.732583 | 0.52344 | 0.980848 | 0.865115 | 0.114253 |
| 45,000 | 2.710017 | 0.53906 | 0.982928 | 0.871285 | 0.114947 |
| 50,000 | 2.706551 | 0.54688 | 0.981505 | 0.869971 | 0.116825 |
| 55,000 | 2.684257 | 0.59375 | 0.985780 | 0.874209 | 0.103992 |
| 60,000 | 2.672819 | 0.59375 | 0.985924 | 0.876116 | 0.107469 |
| 65,000 | 2.655602 | 0.64062 | 0.986628 | 0.883175 | 0.114273 |
| 70,000 | 2.671004 | 0.58594 | 0.986522 | 0.883175 | 0.115489 |
| 75,000 | 2.652258 | 0.66406 | 0.987261 | 0.888353 | 0.132376 |
| 80,000 | 2.641293 | 0.65625 | 0.987924 | 0.891672 | 0.127088 |
| 85,000 | 2.652183 | 0.66406 | 0.987357 | 0.888313 | 0.144588 |
| 90,000 | 2.617375 | 0.67969 | 0.989205 | 0.893652 | 0.119101 |
| 95,000 | 2.614602 | 0.65625 | 0.989280 | 0.894817 | 0.125980 |
| 100,000 | **2.609035** | **0.69531** | **0.989974** | **0.896181** | 0.123212 |

### 判断

試験2は不採用とする。試験1の最終checkpointと比べて、lossは`+0.027324`、top-1は`-0.023438`、NDCGは`-0.001828`、pairwise accuracyは`-0.010046`、marginは`-0.023193`悪化した。dense LRを下げてもtop-1の振動は解消しなかった。top-1は65kの`0.64062`から70kの`0.58594`へ、85kの`0.66406`から95kの`0.65625`へ低下した。この介入は主に学習を遅らせ、同一100k予算でbaselineに追いつかなかった。

次の試験では全learning rateをbaselineへ戻し、dense weight decayだけを`1e-4`から`3e-4`へ変更する。baselineは100k時点でheld lossとtraining lossに差があるため、初期更新を遅くせず、より強い正則化を検証する。

### 実行時間とwitness

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_dense075_fixed2_u100000_20260722_r1

training time:       3,138.017562 s
updates/s:           31.867253
states/s:            127.469013
average CPU:         74.8478%
candidate CPU:       77.7877%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_dense075_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: 6cac26cd4b7b88ddde9f5f4a941cd8fa5215a2a0d103e43254a989144de9e16e
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  79ec6d082692065f38b704d9c47e9705ce903da18fd33c1ed2e236e3900e1059
```

## 試験3 — dense weight decay 3e-4

scratchから開始した本試験では、試験1のlearning rateをすべて復元し、dense weight decayだけを`1e-4`から`3e-4`へ変更した。Lookup bankのweight decayは0のままである。アーキテクチャ、初期化、sampler、データ提示順、loss、routing schedule、固定深度、executor、held panelは同一である。

### Held panel推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.859876 | 0.53125 | 0.978856 | 0.841653 | 0.068698 |
| 10,000 | 2.791346 | 0.52344 | 0.979932 | 0.850689 | 0.085665 |
| 15,000 | 2.763028 | 0.57812 | 0.982595 | 0.861554 | 0.083727 |
| 20,000 | 2.721600 | 0.60156 | 0.983539 | 0.867600 | 0.082771 |
| 25,000 | 2.706555 | 0.58594 | 0.983643 | 0.870668 | 0.096921 |
| 30,000 | 2.679362 | 0.64062 | 0.984503 | 0.876927 | 0.118946 |
| 35,000 | 2.655379 | 0.71875 | 0.987099 | 0.882533 | 0.115824 |
| 40,000 | 2.676071 | 0.64844 | 0.985641 | 0.878742 | 0.136712 |
| 45,000 | 2.619821 | 0.69531 | 0.988658 | 0.886064 | 0.122701 |
| 50,000 | 2.639295 | 0.64844 | 0.986502 | 0.885248 | 0.135180 |
| 55,000 | 2.615425 | 0.74219 | 0.989078 | 0.889951 | 0.125065 |
| 60,000 | 2.611532 | 0.69531 | 0.988770 | 0.890968 | 0.134419 |
| 65,000 | 2.601829 | 0.73438 | 0.989487 | 0.896681 | 0.127673 |
| 70,000 | 2.618289 | 0.68750 | 0.989011 | 0.894113 | 0.126324 |
| 75,000 | 2.605750 | 0.75000 | 0.989422 | 0.897922 | 0.125464 |
| 80,000 | 2.594070 | 0.72656 | 0.990411 | 0.900268 | 0.119268 |
| 85,000 | 2.598925 | 0.73438 | 0.990028 | 0.900217 | 0.136233 |
| 90,000 | **2.591153** | 0.74219 | **0.990525** | **0.901540** | 0.128988 |
| 95,000 | 2.598547 | 0.71875 | 0.989788 | 0.900674 | **0.144499** |
| 100,000 | 2.607862 | **0.80469** | 0.990137 | 0.898083 | 0.144283 |

### 判断

試験3はtop-1に関して有望だが、すべてのmetricで試験1を置き換える明確な結果ではない。最終top-1 `0.8046875`は試験1を`0.0859375`上回り、同じpanelでのPreAct結果`0.7890625`も`0.015625`上回った。記録済みheld panel条件でPreActのtop-1を超えた最初のEVRL checkpointである。

一方、連続的なranking metricは試験1より弱い。最終lossは`+0.026151`、NDCGは`-0.001664`、pairwise accuracyは`-0.008143`、marginは`-0.002122`である。80kから100kまでの5回の評価を平均すると、試験3のtop-1は`0.74531`で試験1の`0.71563`より高いが、lossは`2.59811`対`2.58709`、NDCGは`0.99018`対`0.99120`で劣る。

したがって`3e-4`は現時点のtop-1 armとして保持するが、普遍的なwinnerとはみなさない。このheld panelは既に調整判断へ利用しているため、PreAct超えは開発上のevidenceであって、sealed generalizationの主張ではない。game validationとsealed seedには触れていない。

次の試験ではdense weight decayを`2e-4`へ補間し、baseline learning rateを維持する。top-1の改善を残しつつ、試験1のloss、NDCG、pairwise accuracyを回復できるかを検証する。

### 実行時間とwitness

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd3e4_fixed2_u100000_20260722_r1

training time:       3,202.446105 s
updates/s:           31.226131
states/s:            124.904522
average CPU:         74.6048%
candidate CPU:       77.3738%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd3e4_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: 51b008ea66041da9cfeb7b005a62e75f6d4f06a0a5a9dde94bc4c47653e51912
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  48480aabeb8a0a8e2762888dfbdb9a90d3b906a1569ae7611295d7f8e96b5440
```

## 試験4 — dense weight decay 2e-4

scratchから開始した本試験では、dense weight decayを試験1 baselineの`1e-4`と試験3 top-1 armの`3e-4`の中間へ設定した。それ以外のparameterおよび実行・評価条件はすべて同一である。

### Held panel推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 |
| 5,000 | 2.851713 | 0.53906 | 0.980043 | 0.840504 | 0.065302 |
| 10,000 | 2.784460 | 0.50000 | 0.979888 | 0.846149 | 0.080127 |
| 15,000 | 2.751044 | 0.56250 | 0.983050 | 0.860493 | 0.079198 |
| 20,000 | 2.713456 | 0.60938 | 0.983188 | 0.865943 | 0.094645 |
| 25,000 | 2.693339 | 0.57031 | 0.983044 | 0.870063 | 0.113391 |
| 30,000 | 2.660621 | 0.64062 | 0.986125 | 0.880302 | 0.116080 |
| 35,000 | 2.673281 | 0.64844 | 0.987352 | 0.883035 | 0.104091 |
| 40,000 | 2.649686 | 0.65625 | 0.987742 | 0.884113 | 0.131746 |
| 45,000 | 2.615869 | 0.66406 | 0.989097 | 0.889713 | 0.122493 |
| 50,000 | 2.623402 | 0.69531 | 0.987678 | 0.889993 | 0.129680 |
| 55,000 | 2.632644 | 0.65625 | 0.988862 | 0.891377 | 0.125231 |
| 60,000 | 2.611868 | 0.69531 | 0.989359 | 0.894629 | 0.136155 |
| 65,000 | 2.597276 | 0.75781 | 0.990263 | 0.899451 | 0.140998 |
| 70,000 | 2.611061 | 0.70312 | 0.990018 | 0.898287 | 0.132355 |
| 75,000 | 2.604669 | 0.71875 | 0.989874 | 0.901820 | 0.142173 |
| 80,000 | 2.604071 | 0.68750 | 0.990316 | 0.902574 | 0.139390 |
| 85,000 | **2.586791** | 0.72656 | 0.990715 | 0.904016 | **0.145932** |
| 90,000 | 2.592309 | 0.66406 | 0.990922 | 0.905285 | 0.133420 |
| 95,000 | 2.594969 | 0.67969 | 0.990804 | 0.904189 | 0.140677 |
| 100,000 | 2.590257 | **0.75781** | **0.991009** | **0.906449** | 0.142959 |

### 判断

試験4は有用な中間armだが、両端の試験を同時に上回る結果ではない。100kでは試験1に対してtop-1を`0.0390625`、pairwise accuracyを`0.000222`改善した一方、lossは`0.008546`、NDCGは`0.000792`、marginは`0.003446`悪化した。最終top-1は試験3より`0.046875`低い。

80k～100kの平均は、top-1 `0.70313`、loss `2.59368`、NDCG `0.99075`、pairwise accuracy `0.90450`である。試験3の後半top-1平均`0.74531`および試験1の連続ranking平均を下回る。この固定アーキテクチャsweepには、現時点で異なる二つのwinnerがある。

- 試験3（`dense WD = 3e-4`）：held top-1が最良
- 試験1（`dense WD = 1e-4`）：held loss/NDCGと後半の連続ranking安定性が最良

試験4は、scalarが単調に改善するのではなく、実在する正則化trade-offを確認した。これ以上同じ128-state panelで調整すると選択biasが増える。科学的に次に有用なのは、新しい許可済みdevelopment panelまたは複数training seedによる再現であり、game validationとsealed seedには引き続き触れない。

### 実行時間とwitness

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd2e4_fixed2_u100000_20260722_r1

training time:       3,078.199980 s
updates/s:           32.486518
states/s:            129.946073
average CPU:         74.6836%
candidate CPU:       77.7083%

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd2e4_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
sha256: ab913b03a0c033341e5613c63010f349382bb3736fb67e06797116be10e7a0f2
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  8add363547a9acf4cd74d18c1d2cb8611c6cedcc5385a596107b341cf794492a
```

## Sweepまとめ

| 試験 | 唯一の変更点 | 最終loss | 最終top-1 | 最終NDCG | 最終pairwise | 最終margin | Updates/s | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | baseline | **2.581711** | 0.71875 | **0.991801** | 0.906227 | **0.146405** | 31.92 | 連続ranking winner |
| 2 | dense representation LR `0.75x` | 2.609035 | 0.69531 | 0.989974 | 0.896181 | 0.123212 | 31.87 | 不採用 |
| 3 | dense WD `3e-4` | 2.607862 | **0.80469** | 0.990137 | 0.898083 | 0.144283 | 31.23 | top-1 winner |
| 4 | dense WD `2e-4` | 2.590257 | 0.75781 | 0.991009 | **0.906449** | 0.142959 | **32.49** | 均衡した中間点 |

この調整sessionでは、新たに320,000更新を実行し、実teacher stateを1,280,000回提示した。内訳は、試験1の完了に20k、試験2～4に各100kである。測定学習時間の合計は10,045.24秒だった。scratchから開始した3試験は、すべて同じmodel seedとデータ提示順から開始した。
