# F 後の独立研究戦略監査

日付: 2026-07-18 JST  
基準 commit: `a27dff8`  
範囲: 指定レポート、F harness、`D:\tetris-paper-plus\runs\legacy_full_feasibility_F_c7b2cf2` の read-only 監査。ゲーム、学習、benchmark、Julia/Python は実行していない。

## 結論

**次の一手は P1 だけを許可する。旧 20,787,454-parameter checkpoint をそのまま初期値にし、board tower の最後の二 residual block と最終 projection/normalization、および score head だけを、固定 3-step TD と old-Q anchoring で 300 update する一回限りの保守的 partial fine-tuning pilot である。**

全モデル benchmark、F の validator 修正再実行、全モデル learner、medium 新 architecture、Reactant integration は許可しない。P1 は 2,949,508 parameters だけを optimizer に入れる別の学習仮説であり、F の救済実行ではない。未使用 development seed 5756--5757 までの gate とし、validation seed 8001--8008 と sealed test seed 91001--91032 はロードも実行もしない。

P1 が通っても「統計的に旧モデルを超えた」とは判定しない。統計的判定は、候補を完全 freeze した後に別承認される 32 sealed paired test と、既定の paired mean-difference bootstrap 95% CI lower bound `> 0` だけが行える。P1 pass はその freeze review に進める development evidence に限る。

## F から確定したこと、していないこと

F の形式判定は **F-infeasible** であり、global one-shot marker は消費済みである。六 warm update、更新 weight、OpenVINO CPU/NPU refresh、正しい process-tree peak memory、`T1000` は得られなかった。

一方、停止を「Zygote が 20.8M parameters で timeout した」または「実 parameter gradient が欠けた」と読むのも誤りである。

- optimizer setup と最初の complete-update probe の合計は `123.876928 s` で、300 秒 gate 内に返った。setup と update の個別値、steady rate は保存されていない。
- control flow は zero-update gate を通過して specialization に進んだ。partial JSON の上書きにより個別誤差が失われ、finalizer の `zero-update tolerance failed` は欠落 field を failure に変えた accounting artifact である。
- 停止点は返却 gradient の custom tree scan だった。parameter tree の空 `NamedTuple()` に対して gradient 側の `nothing` を失敗にする実装であり、最初の静的候補は 0-parameter pooling subtree `parameters.board_net.resblocks.layer_1.layer_5.layer_1` である。これは false positive が可能なことを示すが、停止 gradient が保存されなかったため、actual first mismatch と array-leaf coverage は不明である。
- `monitor.json` の peak working set `49,520,640 bytes` は直接 PID のみで process tree を集計しておらず、8 GiB gate の pass 証拠ではない。
- `Lux.testmode` を AD 内で使う警告が出た。固定 running state は意図どおりだが、parameter gradient の数値正しさは別に検証されていない。

したがって F は full-model path を進める必要条件を満たさず、同時に partial AD を否定する証拠でもない。P1 は trainable tree を約 14.2% に狭め、空 subtree を optimizer/gradient contract から除き、凍結 trunk の強度を保持したままこの残った可能性を試す。

## 唯一許可する P1

### 仮説と因果機構

**仮説:** 旧 checkpoint の大部分を bitwise 固定したまま、最終二 board residual block と head にだけ小さい n-step TD correction を与え、同じ state の old-Q へ trust-region anchor を置けば、head-only E002 の catastrophic drift を避けながら旧 policy の少数の価値誤差を修正し、未使用 development paired score を改善できる。

因果機構は三点に限定する。

1. step 0 は旧 policy と同一なので、compact student のように数千点の差を取り戻す必要がない。
2. head だけでなく高位の spatial residual features も更新し、E002 の固定表現 bottleneck を外す。
3. selected action の実 reward に基づく 3-step TD は old-Q imitation ceiling を越える correction signal を与え、同じ candidate chunk 全体への pointwise old-Q anchor は非選択 action と policy ordering の無制限な drift を抑える。

これは勝利を保証しない。固定 old-Q bootstrap と behavior data による保守的な一回の fitted-TD pilotであり、まず「旧強度を壊さず少数の有益な flip が生じるか」だけを問う。

### exact implementation change

F directory と global marker は変更・再利用しない。新しい `experiments/legacy_partial_tail_td/` contract と fresh output namespace を作る。

trainable parameter path は次だけに固定する。

- `board_net.resblocks.layer_29`（residual block 15 の parameterized skip branch）
- `board_net.resblocks.layer_31`（residual block 16 の parameterized skip branch）
- `board_net.conv2`
- `board_net.norm2`
- `score_net`

`layer_30` と `layer_32` の activation は tail forward に含めるが parameter はない。source layer dimensions からの trainable count は `2 * 1,214,272 + 2,307 + 518,657 = 2,949,508`。残る `17,837,946` parameters、queue mixer、前 14 residual blocks、全 running state は frozen とする。AdamW moment array elements は正確に `5,899,016` を要求する。

forward は旧 graph と同じ exact historical chunking（連続 16 candidates と actual-length tail）を保つ。frozen prefix を AD closure の外で計算し、tail parameter tuple だけを differentiate する。更新後は trainable subtree を original full parameter tree に merge し、元 checkpoint や既存 weight artifact を上書きせず fresh NPZ/OpenVINO artifact を作る。

gradient validator は次の contract に置換する。

- trainable tree 内の **actual array leaf** ごとに path、shape、finite、element count を検査する。
- parameter 側に array leaf がない subtree は gradient の `nothing` または空 tuple を許す。
- 最初の mismatch path と gradient inventory を failure artifact にも保存する。
- update 前後で全 frozen array の SHA-256 が不変であることを検査する。
- `Lux.testmode` 警告を黙殺せず、`score_net.layer_3.bias`、`score_net.layer_1.weight`、`board_net.resblocks.layer_31` の conv weight 各一座標について central finite difference と AD を比較し、各々 `abs_error <= 1e-3 + 0.02 * max(abs(fd), abs(ad))` を必須とする。running state の最大変化は 0 とする。

### 固定 data、objective、optimizer

- initializer: `1313/mainmodel copy 3.jld2`、SHA-256 `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`。
- dataset: `teacher_dev_5742_5749_2000.jld2`、SHA-256 `e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d`。
- training role: rows 1--1500、episodes 1--6 / seeds 5742--5747 のうち同 trajectory に三 successor がある rows だけ。data-order RNG は `Xoshiro(0x1313_2026)`、without replacement の最初の 300 rows を実行前 freeze に列挙する。環境 rollout は生成しない。
- target: `/600` reward、`n=3`、`gamma=0.997`。terminal 前は reward を truncate し bootstrap なし、それ以外は row `t+3` の stored old-policy selected Q を使う。target は実行前に固定し、更新 model から再推定しない。
- loss for one row: `Huber(q_theta[selected], y3) + mean(Huber(q_theta[a], q_old[a]) for a in exact selected-action chunk)`。両 Huber の `delta=1`、anchor weight は `1.0`。selected chunk 以外への padding、ListNet、DAgger label、auxiliary loss は加えない。
- optimizer: Julia 1.12.6 + Lux 1.31.4 + Zygote 0.7.11、AdamW `lr=1e-5`、betas `(0.9,0.999)`、weight decay `1e-4`、300 updates、sweep/rollback/checkpoint selection なし。

Reactant は使わない。実 C13 fixed74 repeated-batch で steady update は Zygote の `1.999x` だったが、compile-inclusive crossover は 1,000 updates 内に観測されず、last-500 rate からの約 `1,193` は投影である。P1 は 300 updates、legacy tail、可変 actual-length chunk、merge/export を含み、その 2x を外挿できない。

### seed allowance と評価順

1. training は上記 5742--5747 の stored rows のみ。
2. offline safety gate は既存 split rows 1501--2000、seeds 5748--5749 の stored candidate lists のみ。checkpoint 選択には使わず、固定 update-300 candidate を accept/reject する。
3. offline gate 通過後だけ、最後の未使用 development seeds `5756`, `5757` をこの順で一回ずつ paired 実行する。各 pair は candidate と canonical old OpenVINO baseline、NEXT=5、HOLD、stable order、100 pieces、one logical full-candidate score per decision、zero lookahead とする。
4. 5756 が非正差なら直ちに停止し 5757 は温存する。5750--5755 の再利用、validation 8001--8008、test 91001--91032 は禁止する。

### 成功指標

以下をすべて満たす場合だけ **P1-development-pass** とする。

1. split-tail step-0 Lux output が original full Lux と最大絶対誤差 `<=1e-6`、stored old-Q と `<=1e-2` で一致する。
2. 300 updates が finite。gradient array elements は毎回 `2,949,508`、optimizer moment elements は `5,899,016`、finite-difference gate を通り、少なくとも一つの trainable array が変化し、全 frozen parameter hash と running state は不変である。
3. final merged weight の Lux 対 fresh OpenVINO 最大絶対誤差が CPU `<=1e-4`、NPU `<=1e-2`。完全な export/compile/inference/accounting を保存する。
4. rows 1501--2000 の old-policy top-1 agreement が `>=0.95`、全 Q が finite。失敗時に earlier checkpoint へ戻さない。
5. development の candidate と baseline が両 seed で 100 pieces 完走し、candidate の paired difference が **2/2 で strictly positive**、paired mean が `>=+500`、paired median が positive。全 candidate evaluations、logical/physical calls、generation/inference/wall time を記録する。

この二 seed gate は統計的 model-beat 証拠ではない。pass 後も checkpoint/config/source を freeze する別レビューまで sealed test を許可しない。

### time cap、compute projection、即時停止

hard wall は **35 分**。最初の update は optimizer setup を含め `<=180 s`、次の六 update の warm median は `<=4.5 s`、どの warm update も `<=15 s`、process-tree peak working set は `<=8 GiB` とする。

予算の条件付き上限は次で管理する。

`T_P1 = C_partial + 300 * W_partial + V_offline + R_export + 400 * 0.411`

最後の項は candidate/baseline × 2 seeds × 100 steps の既存 old-policy wall rate である。F の full setup+probe `123.876928 s` を compile reserve、`W_partial=4.5 s`、`V_offline=100 s`、`R_export=120 s` と置く保守的な計画値は `1,858.3 s`（31.0 分）で、35 分まで約 242 秒を残す。ただし F は partial warm rate も updated export も測っていないため、この値は観測予測ではなく stop budget である。六 warm update 後、実測値を用いた残時間投影が 2,100 秒を超えれば学習を続けない。

次のいずれかで即時停止し、row、subset、learning rate、anchor weight、update count、backend、seed を変えた rescue run はしない。

- source/checkpoint/dataset/hash/row-role/command mismatch、または F marker/harness の迂回。
- full 20.8M parameter tree を optimizer/AD に入れようとした場合。
- array-leaf gradient 欠落、finite-difference failure、non-finite、frozen hash/state change、memory/time gate failure。
- final export/CPU/NPU equivalence failure、offline top-1 agreement `<0.95`、必須 accounting 欠落。
- game-over、または最初の completed pair の差 `<=0`。
- development result を見た後の checkpoint rollback、追加 update、hyperparameter adjustment、seed substitution。
- validation/test seed のロードまたは実行。

## 代替案の判定

### conservative partial fine-tuning

**採用。** 旧 checkpoint と step-0 policy を保持でき、E002 より表現自由度があり、old-Q anchor で E002 の観測済み悪化に直接対処する。F が残した最大の不確実性を、full-model rescue ではなく約 14.2% の明示 subset と一つの strength pilot で同時に判定できる。失敗しても 35 分で枝を閉じられる。

### medium 新 architecture + teacher pretraining + end-to-end RL

**次には採用しない。** C13 は 165k student で old model に三 seed 平均 `-3,766.7`、S1 は最初の pair `-400` で停止した。medium FiLM は容量を増やせるが、800,133-parameter 64x6 systems screen の batch-16 loss+backward は汚染込み `21.161 s` で、game strength は未測定である。teacher pretraining、covariate-shift repair、旧強度までの回復、さらに teacher ceiling を越える RL がすべて必要で、強い初期 checkpoint を直接保つ P1 より経路が長い。Reactant の compact fixed74 結果もこの architecture/online pipeline へは移せない。

### 新しい full-model benchmark / training preflight

**禁止。** gradient validator を直す、gradient を保存する、別 row、別 timeout、別 backend、warm update/export だけを測る、といった 20.8M 全 parameter benchmark は、名称や contract を変えても F の未完了 success fields を埋める rescue rerun である。global one-shot の意味を失わせるため不正である。full-model learner の前に一 update を再測定する行為も同じである。将来まったく別の科学的仮説が独立に承認されない限り、現 evidence から full-model AD を再開しない。

## 最終判断

最短の evidence-backed route は、旧モデル同等から始める利点を捨てず、F の全モデル未知を避ける **P1 partial-tail anchored TD** である。P1 以外の実装・benchmark・training は同時に進めない。sealed test は候補 freeze 後の別判断まで閉じたままとする。
