# P1 parse failure 後の独立戦略監査

日付: 2026-07-18 JST  
基準 commit: `6c649056fc247e6dcde64b7c8690218dadbac88b`  
範囲: tracked source、既存レポート、および
`D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1_6c649056_pwsh` の
read-only 監査。モデル/checkpoint をロードせず、学習、ゲーム、OpenVINO、
validation/test seed の使用は行っていない。

## 結論

P1 は科学的仮説を一度も実行せず、`select_rows.jl:53` の Julia 1.12
parse error で終了した。しかし global one-shot marker は消費済みであり、
**構文を直した P1、名前や output directory を変えた P1、P1 の途中からの再開は
すべて rescue retry として禁止する。** 結果が得られなかったことは再試行権を
復活させない。

次の一手として許可するのは **Q1: frozen-old additive residual critic** の
一回限りの offline 学習だけである。旧 20.8M model は完全に凍結し、その旧 Q に
165k compact correction network の出力を加える。correction の最終層を zero
initialize するため update 0 の combined policy は旧 policy と厳密に同じである。
旧 checkpoint 自体を微分・更新する P1 とは別の parameterization、optimizer tree、
checkpoint schema、推論 graph、科学的仮説であり、P1 の救済ではない。

Q1 では training/offline dataset だけを使い、ゲームを行わない。pass しても
`Q1-offline-promoted` にすぎず、旧モデル超え、development/validation pass、G1--G3
のいずれも主張しない。候補 bytes を freeze した後の別レビューなしに validation
seed `8001--8008` または sealed test seed `91001--91032` を使ってはならない。

## P1 terminal artifact の監査

### 確認した停止系列

`monitor.json` は terminal かつ `complete=true` で、実行された phase は次の三つ
だけである。

| phase | wall | exit | 結果 |
|---|---:|---:|---|
| eligibility | 0.451 s | 0 | training rows 1--1500 の eligibility JSON を作成 |
| select_rows | 1.769 s | 1 | Julia parse error |
| finalize_assessment | 0.227 s | 0 | fail-closed assessment を作成 |

`select_rows.stderr.log` の最初の実エラーは正確に次である。

```text
ERROR: LoadError: ParseError:
# Error @ ...\experiments\legacy_partial_tail_td\select_rows.jl:53:36
abspath(PROGRAM_FILE) == @__FILE__ && main()
#                                  └┘ ── invalid identifier
```

停止理由は `phase select_rows failed: process exit 1`、wrapper status は
`wrapper-failed`、最終 status は `P1-development-fail` である。
`extract_training`、`train_partial`、`verify_openvino`、`extract_offline`、
`offline_gate`、`evaluate_development` はすべて「prior required phase failed; no
rescue」として skip された。

### 何が読まれ、何が読まれなかったか

eligibility phase は immutable dataset
`teacher_dev_5742_5749_2000.jld2` の training rows 1--1500 の episode/step/terminal
metadata を読んだ。`select_rows.jl` は eligibility JSON しか入力に取らず、parse
時点で top-level 実行に入っていない。

次の artifact はすべて存在しないことを確認した: `row_freeze.json`、
`training.npz`、`training_extraction.json`、`train_partial.started.json`、
`training_phase.json`、`candidate_weights.npz`、`final_reference.npz`、
`openvino_gate.json`、`offline.npz`、`offline_gate.json`、`development.json`。
したがって checkpoint はロードされず、optimizer setup と update は 0 回、
candidate export/OpenVINO/game は未開始である。assessment も
`validation_seed_used=false`、`sealed_test_seed_used=false`、eligibility artifact も
`validation_or_test_seed_loaded=false` を記録している。

### one-shot 消費

global marker
`D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json` は存在し、
source/authorized commit `6c649056...`、parent `c9ab1a9...`、実 output directory、
`retry_prohibited=true` を固定している。`wrapper_result.json` も同じ marker を指し、
`retry_prohibited=true`、`final_result_is_authoritative_terminal_artifact=true` である。
よって以下は明示的に禁止する。

- line 53 だけを直した同一 P1 の実行
- P1 と同じ partial-tail parameter subset/objective/data order を別名で実行
- marker、start gate、output namespace、commit authorization を迂回する実行
- failed output に row freeze 等を後付けして phase 2 から続けること
- P1 の 300 update を backend、seed、row、更新数、anchor だけ変えて実行すること

構文修正は将来の一般 test coverage には入れてよいが、P1 の scientific run を
復活させる許可にはならない。

## 現 evidence と選択肢

| 選択肢 | 判定 | 情報利得と最短経路の比較 |
|---|---|---|
| 構文修正済み P1 / partial old-tail TD | **禁止** | 一回限り契約の実質的 rescue。P1 marker の意味を失わせる。 |
| F full-old learner または F validator/backend retry | **禁止** | F marker 消費済み。20.8M parameter AD の未完 fields を埋める rescue になる。 |
| C13 round 2 / plain compact old-Q clone | 棄却 | C13 は同条件三 seed すべて負、平均 `-3,766.7`。同じ teacher ceiling と弱い初期 policy を再度背負う。 |
| S1/top-k lookahead の微調整 | 棄却 | S1 は最初の preregistered pair で `-400`、約 3 倍の evaluation cost。parameter sweep は未使用 seed 不足と tuning exposure を増やす。 |
| 800k medium FiLM を scratch/old-Q だけで学習 | 保留 | 容量は増えるが update 0 が弱く、teacher match を回復してから teacher を越える二段階になる。64x6 backward も 21 s の汚染参考値しかない。 |
| Reactant integration だけの追加 benchmark | 保留 | fixed-shape 長時間では有望だが strength signal を一切増やさない。既に crossover 711 を観測済みで、今の最大不確実性ではない。 |
| **Q1 frozen-old additive residual** | **唯一許可** | update 0 で旧強度を保持し、旧 Q の selected-action Bellman error だけを小型 end-to-end network で学べる。旧 model AD を回避し、数分で「保守的 correction が offline held-out へ一般化するか」を直接判定する。 |

Q1 の情報利得は、P1 が答えるはずだった「旧 policy を壊さず return signal を
利用できるか」を、旧 checkpoint の parameter を一つも更新しない別モデルで問える
点にある。失敗すれば、既存 offline transition だけの conservative correction 枝を
閉じ、新しい action-improvement data または online RL が必要だと判断できる。

## 唯一許可する Q1

### 仮説と改善機構

**仮説:** 旧 Q を固定 base とし、C13 で学習済みの compact feature extractor から
zero-initialized additive correction を学習すれば、update 0 の旧 policy を厳密に
維持したまま、old-policy と DAgger behavior trajectory に観測された 3-step
Bellman residual を一般化し、held-out offline rows で少数の有益候補だけを
rerank できる。

combined score は候補 `a` ごとに

```text
q_Q1(s,a) = q_old(s,a) + r_theta(s,a)
```

とする。`q_old` は完全に stop-gradient/frozen で、`r_theta` の最終 scalar Dense
weight/bias を 0 にする。compact backbone は C13 update-250 checkpoint から初期化し、
最終層以外を含め end-to-end で更新する。旧 model parameter、running state、canonical
NPZ/IR/checkpoint は変更しない。

これは search を増やす方式ではない。後の評価では旧 agent と同じ candidate set、
NEXT、HOLD、logical full-candidate call count を使う composite value model として扱い、
追加 residual inference の wall time、physical request、parameter count を別記する。

### immutable input と data role

- base old checkpoint SHA-256:
  `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`
- canonical old OpenVINO weight NPZ SHA-256:
  `2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba`
- compact initializer:
  `C13_round1_preregistered500_warm_c11b_best.jld2`, SHA-256
  `1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270`
- aggregate dataset:
  `teacher_plus_dagger_c13_round1.jld2`, SHA-256
  `4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba`
- training role: aggregate rows 1--2160 / episode ids 1--12 only。
- offline gate: unchanged aggregate rows 2161--2660 / episode ids 13--14 only。
- validation 8001--8008、test 91001--91032、既使用 game seed 5750--5757 は
  load も run もしない。

実行前 preflight は同 episode で `t..t+3` が連続し、`t..t+2` の途中 terminal が
ない exact eligible row set を列挙する。eligibility、全 2,000 minibatch の row order、
その SHA-256 を start gate 前に freeze する。sampling RNG は
`Xoshiro(0x5131_2026)`、batch は 4 states、candidate axis は mask 付き固定 74。
eligible set の permutation を決定論的に繰り返し、epoch 境界で同 RNG を使って
reshuffle する。row の追加、除外、priority sampling、result 後の order 変更は禁止する。

### target、loss、optimizer

各 selected transition について、stored score delta `/600`、`gamma=0.997`、`n=3`
の return と、非 terminal のとき row `t+3` の **全 stored old Q の max** を使って
`y3` を一度だけ固定する。DAgger row の selected action を bootstrap action と
みなしてはならない。

batch loss は次だけとする。

```text
mean Huber(q_old[selected] + r_theta[selected], y3; delta=1)
+ 1.0 * mean Huber(r_theta[valid actions], 0; delta=1)
```

optimizer は AdamW、learning rate `3e-4`、betas `(0.9,0.999)`、weight decay
`1e-4`、global gradient norm clip `1.0`、exactly 2,000 updates。途中 snapshot は
0/500/1000/2000 を診断用に保存してよいが、scientific candidate は update 2000
だけであり、offline result による rollback/best-checkpoint selection は禁止する。

backend は **Julia 1.12.6 + Lux 1.31.4 + Zygote 0.7.11** に固定する。native
Enzyme は actual learner で warm 1.73 倍遅く、allocation も多く、100-update AdamW
trajectory gate を失敗したため採用しない。Reactant+EnzymeMLIR は固定 shape で
711 update 後に cumulative crossing し 1,000 update で 1.199 倍だったが、sampling、
packing、checkpoint/offline validation を含む integration gate が未通過である。
Q1 で model/objective と backend を同時に変えて原因を混ぜない。AD 改善は無視した
のではなく、実測 gate に従って native winner を選ぶ。

### 数値・時間 gate

一つでも失敗すれば即時停止し、再実行、checkpoint rollback、learning-rate/anchor/
update/backend/seed 変更をしない。

1. update 0 で `r_theta` の全 valid output が bitwise zero、combined Q と stored old Q
   の最大絶対誤差 `0`、top-1 agreement `1.0`。
2. model parameter 数、trainable array path、gradient element 数を毎 update 固定し、
   全 loss/gradient/parameter を finite とする。旧 input artifact の SHA-256 は終了時も
   不変でなければならない。
3. first complete update `<=60 s`、update 6--25 の warm median `<=0.25 s`、単一 warm
   update `<=1.0 s`、process-tree peak working set `<=4 GiB`。
4. hard wall **12 分**。実測 warm rate と残 phase から 12 分を超える投影になれば
   2,000 update を続けない。
5. fresh correction checkpoint、combined-reference NPZ、source/Manifest/config/hash、
   stdout/stderr、終了理由を fresh Q1 namespace に保存する。既存 artifact を上書き
   しない。

### offline promotion gate

update-2000 の固定 candidate に対し rows 2161--2660 だけを一度評価し、次をすべて
満たす場合だけ `Q1-offline-promoted` とする。

1. held-out selected-action 3-step Huber が zero-residual/update-0 比で **15%以上低下**。
2. target residual `y3-q_old[selected]` と予測 residual の Pearson correlation が
   `>=0.20`、かつ符号一致率が `>=0.60`（絶対 target residual `>=0.1` の rows）。
3. combined top-1 と old top-1 の agreement が **`>=0.95` かつ `<0.995`**。旧 policy
   を大きく壊さず、少なくとも三つの held-out state で実際に action が変わることを
   要求する。
4. all-candidate residual RMS `<=0.25`、全出力 finite。
5. update-2000 Lux correction と fresh exported OpenVINO CPU correction の最大絶対
   誤差 `<=1e-4`。NPU はこの offline gate の必須条件にせず、使った場合だけ
   `<=1e-2` を要求する。

この gate は action change が有益だと証明しない。pass 後は candidate bytes、combined
model contract、old base、source、Manifest を freeze して独立レビューを行う。そこで
初めて、未使用 validation seeds の一回限り paired screen を許可するか判断する。

## P1 harness から再利用できるもの、できないもの

### 再利用してよい一般部品

- source fingerprint と immutable-input hash の検証方式
- clean HEAD / exact parent binding、start gate、fresh-output 拒否、atomic marker の方式
- Windows Job Object による whole-process-tree kill、working/private bytes、phase wall の監視
- fail-closed assessment、monitor 完了後の authoritative final-result publication order
- stdout/stderr、phase-start、skipped-phase、終了理由を残す artifact schema
- dataset の HDF5/JLD2 hyperslab 抽出、score-delta reward の再計算検査、candidate mask
- historical candidate order/chunk semantics、および Lux/OpenVINO 数値同等性の検査思想

これらを Q1 固有 namespace と Q1 固有 contract に実装し、synthetic/preflight test で
checkpoint/dataset を開かず検査してから一回だけ実行するのは rescue ではない。

### 再利用してはいけないもの

- P1 global marker、start gate、output directory、freeze/row-freeze/final-result
- `select_rows.jl` の構文修正後の実行、または failed eligibility artifact からの再開
- `train_partial.jl`、partial-tail parameter paths、2,949,508-parameter optimizer tree、
  merge/export schema、P1 update order digest
- P1 の development seeds 5756/5757 と paired game gate
- P1 の authorization commit/report を Q1 実行許可として流用すること
- P1 phase 名や contract をコピーして、実質的に同じ旧-checkpoint partial update を
  別 experiment と称すること

## 最終判断

P1 は「悪い学習結果」ではなく「学習前の terminal harness failure」である。ただし
one-shot discipline は成功時だけでなく失敗時にも守る必要がある。最短の正当な次の
経路は、旧モデルの強度を update 0 で保持しながら、旧モデルを微分しない Q1 additive
residual である。Q1 だけを実装・事前監査・一回実行し、他の architecture、backend、
lookahead、旧-model fine-tune を並行して走らせてはならない。
