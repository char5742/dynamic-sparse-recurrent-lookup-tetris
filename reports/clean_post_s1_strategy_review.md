# S1 後の独立研究戦略監査

日付: 2026-07-18 JST
範囲: 指定済みレポート、C13 実 ListNet backend 結果、復元旧モデルと現行学習コードの read-only 監査。ゲーム、学習、validation/test、コード変更は実行していない。

## 結論

**次の一手は F だけを許可する。旧 20,787,454 parameter checkpoint の「全モデル継続学習」がこの CPU/NPU 構成で成立するかを判定する、25 分上限の end-to-end feasibility benchmark とする。A--E の学習・label 生成・game screen はまだ許可しない。**

これは一般的な backend 探索ではなく、A の直前に置く一回限りの go/no-go gate である。現時点で実際の旧モデル強度から開始できるのは A だけだが、tracked source に全旧モデル learner はなく、更新後の重みを高速 Actor に戻す経路も測られていない。この二点を解かないまま A を走らせることは、最短経路ではなく未計測の実装賭けになる。

F が通っても A を自動許可しない。結果を固定して再監査する。F が落ちた場合も C、D、Reactant 版 legacy learner、別 architecture を自動許可しない。

## 判断を支配する証拠

### 現在の強度差

同一 100 手・development の直接証拠では、旧モデルはおよそ 5,100--6,200 点である。C13 は 1,100--2,800 点で、三対すべて負、平均差 `-3,766.7`、中央値差 `-4,000` だった。全 episode が完走しているため、これは単なる早期 game-over の差ではない。

S1 top-2 one-step lookahead は seed 5755 の最初の一対で 5,500 対 5,900、差 `-400` となり、事前停止条件で閉じた。同時に candidate evaluation 2.99 倍、logical pass 3.00 倍、inference 2.93 倍を要した。したがって、この exact stronger-action mechanism を教師または system policy として続ける根拠はない。ただし S1 は一対で停止した弱い負の証拠であり、すべての return 改善法の不可能性を示すものではない。

### A は初期強度だけは最良だが、実行可能性が未成立

旧 checkpoint は復元済み `LegacyQNetwork` として読み込め、step 0 で旧 policy そのものである。この初期強度は、旧モデルまで数千点を回復する必要がある B/C/D より time-to-actual-beat 上の大きな利点である。

一方、既存の A に最も近い E002 / `scripts/train_value_head.jl` は全 20.8M parameters の継続学習ではない。OpenVINO で旧表現を固定抽出し、`score_net` だけを one-step Double-DQN で更新する。1,562 updates 後も最良は未学習の episode 0 で、二 seed 平均は 15,700 から episode 8 の 15,000 へ悪化した。これを optimizer 名や n-step だけ替えて再実行することは許可しない。

read-only source audit では次も確認した。

- `src/legacy_model.jl` は全旧 graph と checkpoint modernization を提供するが、全モデル `TrainState` を作る production learner はない。
- `experiments/learning/train_distillation.jl` の n-step/PER/target-network 実装は `CompactCandidateQ` と compact checkpoint schema に固定され、legacy checkpoint を初期値として直接使えない。
- `scripts/export_legacy_openvino_weights.jl` と `tools/legacy_openvino.py` に NPZ/OpenVINO 化の部品はあるが、exporter は元 checkpoint と固定 output を前提にしている。更新 checkpoint の安全な export、compile、数値同等性、Actor refresh 時間は未計測である。
- native Lux 旧モデルは既存 25 手で 985 candidates / 53.206 inference seconds（18.5 candidates/s）だった。学習後 Actor を native Lux のまま回す案は最短経路にならない。
- 64 channel / 6 block の 800,133 parameter FiLM でさえ、記録された batch-16 loss+backward は 21.161 秒だった。この値は load contamination を含むので legacy の正確な予測には使えないが、20.8M 全体の CPU reverse-mode を未計測のまま楽観することもできない。

したがって A は「候補として最有望」ではあるが「今すぐ許可可能」ではない。F はこの差を最短時間で埋める。

### compact / Reactant の速度はこの判断を反転しない

Reactant+EnzymeMLIR の actual C13 fixed74 ListNet screen は、同じ 165,051 parameter compact model、固定 batch 4、固定形状で数値 gate を通り、steady update が Zygote の約 2.00 倍だった。しかし first update は 65.288 秒で、1,000 updates の累積は Reactant 95.850 秒、Zygote 88.970 秒、実測 1,000 内では交差していない。last-500 rate からの break-even `1,193` は投影値である。

さらに未検証なのは changing batch の生成/転送、checkpoint export、validation boundary、別 architecture の lowering である。この結果は長い compact ListNet run の backend 候補であって、legacy 20.8M model、FiLM、online RL の実測 speedup ではない。F で Reactant を同時に試すことも、比較対象を増やして一手制約を破るため禁止する。

## A--F の比較

| 選択肢 | 判断 | time-to-actual-old-beat の監査結果 |
|---|---|---|
| A — old 20.8M 初期値から最新 RL | **候補は保持、直接実行は保留** | step 0 が唯一旧モデル同等。ただし head-only 既往は悪化し、全モデル learner、reverse-mode cost、更新重みの高速 Actor refresh が未成立。まず F が必要。 |
| B — 800k FiLM/PreAct を大規模 DAgger 後 RL | reject | 初期 policy 強度がなく、旧 Q DAgger 段階には teacher ceiling がある。800k systems screen は game 強度を示さず、CPU backward も重い。Reactant の証拠はこの architecture へ外挿できない。 |
| C — compact C13 n-step Double DQN/return | reject | 実装は近いが開始点が旧より平均約 3,767 点低い。現 dataset の reward/transition は旧 policy の selected action にしかなく、未選択 action の反実仮想 improvement signal がない。既存 C02 を長くするだけでは teacher ceiling を十分に破れない。 |
| D — Monte Carlo / stronger-return labels | reject for next move | ceiling を破る方向自体は正しいが、唯一直接試した S1 target は負かつ約 3 倍推論。全候補 MC は旧 Actor cost を枝数・horizon 分だけ増やし、既存の bounded 生成器も positive target evidence もない。 |
| E — その他 | reject | 既存証拠に、旧強度で開始しながら A の未計測点を回避する具体的 learner はない。新規案を作るより A の feasibility を一回で判定する方が短い。 |
| F — implementation/throughput feasibility | **唯一許可** | 初期強度という A の利点を残したまま、最大の未知である full update と Actor refresh を 25 分で go/no-go 判定できる。 |

## 唯一許可する F benchmark

### 仮説

**復元旧 checkpoint の全 parameters を使う一状態単位の n-step TD update と、更新重みの OpenVINO Actor refresh は、数値的に正しく、1,000 actor/update steps を 30 分以内に行える実測 rate と memory に収まる。**

benchmark は性能測定だけで、更新 parameter は終了時に破棄する。checkpoint を昇格せず、game score を測らず、policy 学習の結果として扱わない。

### 固定入力と実行条件

- checkpoint: `1313/mainmodel copy 3.jld2`、SHA-256 `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`。
- model: tracked `LegacyQNetwork`、20,787,454 parameters **すべて**を optimizer state に含める。旧表現や head だけの proxy への置換は禁止。
- state: checkpoint の running statistics を `Lux.testmode` で固定する。BatchNorm running state の更新を benchmark 途中で許さない。
- data: `D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2`、SHA-256 `e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d` の既存 training rows だけを read-only 使用する。
- rows: `[1, 251, 501, 751, 1001, 1251]` をこの順に一回ずつ使用し、実行前に episode IDs 1--6、metadata seeds 5742--5747 であることを assert する。各 row は padding せず、その `action_count` の完全 candidate set と stable order を保つ。
- target: dataset の `/600` reward、`n=3`、`gamma=0.997`、同 trajectory の stored old-Q bootstrap から、selected action の frozen TD target を事前計算する。benchmark 中の target 再推定、label tuning、ListNet 混合は禁止。
- loss/update: selected action の Huber loss、AdamW、learning rate `1e-5`、betas `(0.9, 0.999)`、weight decay `1e-4`。最初の differentiation/optimizer specialization を compile time として分離し、その後六 rows の complete forward + backward + optimizer update を同期計測する。
- backend: Julia 1.12.6 + Lux 1.31.4 + Zygote 0.7.11 のみ。thread 数、BLAS thread 数、空き memory、同時 Julia/Python/OpenVINO process を記録し、他の heavy job がある状態では開始しない。
- Actor refresh: 六更新後の temporary parameter/state を一度だけ別出力へ NPZ 化し、OpenVINO CPU と NPU を compile する。固定した一つの training row で Lux との最大絶対誤差、export seconds、compile seconds、inference seconds を同期計測する。元 checkpoint、tracked artifact、既存 NPZ を上書きしない。temporary outputs は非昇格である。
- provenance: 実行前 freeze に source commit、Manifest hash、checkpoint/dataset/harness hash、完全 command、output paths、上記定数、process/thread 状態を記録する。一度だけ実行し、失敗後の backend、batch、row、optimizer、thread tuning はしない。

### 成功指標

以下をすべて満たす場合だけ **F-feasible** とする。

1. zero-update Lux output が stored old-Q と有限に一致し、既知の NPU tolerance `1e-2` を超えない。
2. 六回すべての loss、gradient、parameter、optimizer state が finite で、六回後に少なくとも一つの parameter が変化する。
3. warm complete-update median、first specialization、peak working set、NPZ export、CPU/NPU compile と同期 inference が欠落なく記録される。peak working set は 8 GiB 以下とする。
4. temporary updated weights の Lux 対 OpenVINO 最大絶対誤差が CPU `1e-4` 以下、NPU `1e-2` 以下である。
5. 次の保守的投影が `1,800` 秒以下である。

   `T1000 = measured first-specialization + 1000 * warm-median-update + 1000 * 0.411 + 4 * measured Actor-refresh`

   `0.411 s/step` は S1 の同一 baseline 100 手 wall 41.066 秒から固定した既存 Actor/game cost であり、benchmark 結果から都合よく変更しない。`Actor-refresh` は NPZ export + NPU compile + equivalence synchronization の実測和とする。四回 refresh は 250 updates ごとの最低限の online-policy 更新を表す。

この gate は A が勝つことを証明しない。少なくとも「30 分の A run が learner/Actor plumbing だけで破綻しない」ことを要求する必要条件である。

### 時間上限と即時中止

hard wall limit は **25 分**。次のいずれかで即時終了し、再試行しない。

- checkpoint、dataset、source、Manifest、row role、command、固定定数の不一致。
- validation rows 1501--2000、validation seeds 8001--8008、test seeds 91001--91032、または development game seed をロード・実行しようとした場合。
- game process、score evaluation、teacher generation、persistent training artifact の生成が始まった場合。
- non-finite output/loss/gradient/update、candidate order/schema 不一致、zero-update tolerance failure。
- first full specialization/update が 300 秒を超える、単一 warm update が 120 秒を超える、peak working set が 8 GiB を超える、または OS の paging が継続する場合。
- updated-weight export/compile/equivalence が一回で成立しない場合。
- 必須 accounting が欠ける、または 25 分に達する場合。

失敗時は観測済み数値と停止理由を保存するが、row、backend、shape、parameter subset を変更した救済 run は行わない。

## 証拠の限界

- C13 の負の強度証拠は三 seed、S1 は停止規則により一 seed だけである。差の大きさと方向は branch pruning に十分だが、algorithm family 全体の不可能性を主張しない。
- F の 1,000-step 投影は throughput gate であり、sample efficiency や score improvement の予測ではない。
- 旧 full-model の optimizer dynamics、BatchNorm を固定したままの学習可能性、catastrophic forgetting は F では評価しない。これらは F 通過後に A を設計する際の別 gate である。
- validation/test は本監査でも F でも使用しない。未使用 development 5756--5757 も F には不要であり、温存する。
