# Q1 memory terminal 後の独立メタ戦略監査

日付: 2026-07-18 JST  
監査対象 commit: `09de08ef26ae9ef129305887e92331aede02ad5c`  
Q1 output: `D:\tetris-paper-plus\runs\frozen_old_additive_residual_Q1_09de08e`

この監査では tracked source、既存 report、Q1 の JSON/log artifact だけを読んだ。
checkpoint、dataset、OpenVINO、ゲーム、development/validation/test seed はロードも実行も
していない。変更は本 report だけである。

## 結論

Q1 は correction 学習の成否を一度も測っていない。training extraction は完了したが、
`train_correction` process が optimizer update を記録する前に、一回限りの
process-tree private-bytes gate を越えて terminal になった。したがって、Q1 から
「additive residual が弱い」「Zygote が失敗した」「4 GiB なら学習不能」のいずれも
結論してはならない。一方、one-shot marker は消費され、authoritative status は
`Q1-offline-rejected` である。**4 GiB 閾値だけを緩める、同じ Q1 parameterization を
別名で走らせる、training extraction から再開する、backend だけを替える、という
rescue は禁止する。**

次の実験として一つだけ許可するのは、**R1: online counterfactual top-2 safety gate**
である。canonical old policy が実際に訪れる新しい training-only state で、old top-1
と top-2 を同一 future-piece stream から 6-piece rollout し、反実仮想 advantage を
直接作る。旧 model は凍結し、deploy 時は old top-1 を既定値とする。episode-cluster
bootstrap ridge の予測下限が正の場合だけ top-2 へ一段だけ切り替える。

これは Q1 の再実行ではない。Q1 は既存 trajectory の selected action に対する
165k neural residual critic を Zygote で学ぶ仮説だった。R1 は新規 online branch data、
未選択 action の直接比較、100 未満の固定 feature と解析的 ridge ensemble、top-1/top-2
の離散 gate であり、旧 checkpoint、C13 initializer、Q1 training/offline NPZ、Q1 loss、
optimizer、AD、checkpoint schema を一切使わない。

## Q1 terminal evidence

### 実際に完了した phase

`monitor.json` は `complete=true`、`stop_reason="failed"`、総 wall
`16.2923741 s` である。

| phase | exit | wall | peak private bytes | peak working set | 結果 |
|---|---:|---:|---:|---:|---|
| `extract_training` | 0 | 2.4170851 s | 722,784,256 | 105,320,448 | `training.npz` と extraction manifest を作成 |
| `train_correction` | 1 | 13.3527107 s | **4,506,931,200** | 944,803,840 | `4 GiB` private-bytes gate で kill |
| `finalize_assessment` | 0 | 0.4674739 s | 21,573,632 | 37,650,432 | fail-closed finalization |

`verify_openvino`、`extract_offline`、`offline_gate` はすべて
`prior required phase failed; no rescue` として skip された。`training_phase.json`、
checkpoint、candidate weights、update ledger は存在しない。従って complete optimizer
update 数、loss、gradient、steady update/s、実学習中の resident memory は未知である。
private bytes と working set の乖離は予約/commit を含む可能性があるが、事後に contract
を読み替える理由にはならない。

### scope と one-shot

global marker
`D:\tetris-paper-plus\runs\frozen_old_additive_residual_Q1.started.json` は、source と
authorized hardening commit `09de08e...`、実 output、`retry_prohibited=true`、
`rescue_prohibited=true` を固定している。`final_result.json` は
`status="Q1-offline-rejected"`、`validation_seed_used=false`、
`sealed_test_seed_used=false` である。

重要な hash は次である。

| artifact | SHA-256 |
|---|---|
| `started.json` | `6b91537e2469195d41878023517746463155a9f494d243b457682011efd9f0bc` |
| `training_extraction.json` | `eb2e2af594ed22b0b407a36f2ad0cf2670f86ccaed0e7a98f322904e3ec23ded` |
| `monitor.json` | `ed745d328799aa939dbcb2dd19fb06adeccf353796e414f12433e0366e3e9b85` |
| `final_result.json` | `56c57c727fb0156aa5de4118e95e922fec4ef2ce34c8f636f0894fd03947704c` |

assessment は `initializer_exposed_to_offline_rows=true`、
`offline_is_held_out_generalization=false` も明記する。C13 initializer は既存 rows
2161--2660 に既に触れているため、この split を今後も candidate/threshold 選択に使う
のは弱い循環 evidence になる。R1 は既存 aggregate の training/offline 両方を使わない。

## 現 evidence のメタ解釈

- C13 は高速な 165k student を作ったが、同一予算の三 development seed で old より
  `[-4300,-4000,-3000]`、平均 `-3766.7`。old-Q imitation を増やすだけでは teacher
  ceiling と distribution shift を同時に越えられない。
- S1 の ungated top-2 one-step Bellman system は最初の preregistered pair で `-400`、
  約 3 倍の model evaluations を使った。exact S1 を search depth/margin だけ替えて
  続ける根拠はない。ただし一例の負は、common-random-number multi-step return による
  **高精度の選択的** override を否定しない。
- F、P1、Q1 はそれぞれ one-shot harness/feasibility terminal で、旧 model AD、partial
  fine-tune、neural residual の科学的 strength を測れていない。しかし marker discipline
  のため同じ parameterization を修理して再実行することはできない。
- Native Enzyme は actual learner で native Zygote より warm `1.73x` 遅く、allocation
  も多く、AdamW trajectory gate を通らなかった。Reactant+EnzymeMLIR は persistent
  fixed-shape では update 711 付近で累積逆転したが、changing-batch production gate は
  未通過である。Q1 terminal は AD 実行前なので AD 比較を更新しない。
- 現在欠けている最大の情報は「old が選ばなかった action のうち、同じ future の下で
  実際に return を改善する action を、deploy 時に安価かつ高 precision で識別できるか」
  である。既存 selected-action TD rows からはこの反実仮想を同定できない。

## 次候補の比較

| 候補 | 判定 | 情報利得 / 最短性 |
|---|---|---|
| Q1 の memory 上限変更、batch/threads/backend 変更、同 residual の別名実行 | **禁止** | Q1 の同 parameterization を救済するだけで one-shot の意味を失う。 |
| P1/F の validator 修正または旧 model fine-tune | **禁止** | 消費済み parameterization の rescue。旧 model backward の時間・memory リスクも最大。 |
| C13 round 2、medium model の old-Q/DAgger distillation | 棄却 | 強い初期 policy を一度失い、teacher を再現してから越える二段階。old より平均約3.8k低い現 evidence では最短でない。 |
| 800k FiLM + online RL | 保留 | 将来の model-only 本命になり得るが、initial teacher match、online replay、CPU backward、export を同時に解く必要がある。今回の一手として情報変数が多過ぎる。 |
| deeper search をそのまま final agent にする | 棄却 | S1 の負 evidence と約3倍 costがある。さらに depth を増やすと score と計算量を分離できない。 |
| Reactant production integration だけ | 保留 | strength signalを増やさない。長い fixed-shape learner が決まった後の加速 gateである。 |
| **R1 online counterfactual top-2 safety gate** | **唯一許可** | old policy を update 0 だけでなく常時 fallback として保持し、新規未選択-action return を直接測る。fit は解析的で低 memory、deploy の追加 search はゼロ。失敗時も「online improvement labels が不足/予測不能」という次の分岐情報を短時間で得る。 |

## 唯一許可する R1

### 科学仮説

**仮説:** canonical old policy の top-2 候補には、old-Q の局所誤差により top-1 より
6-piece return が高い例が少数存在する。その advantage は old-Q margin、両候補の
board-safety差、score/REN/B2B/T-spin差、現在の HOLD/NEXT から低次元に予測できる。
episode 単位の bootstrap 予測下限が正の state だけ top-2 へ切り替えれば、old policy
を大部分維持しながら paired game score を上げられる。

改善機構は **counterfactual action improvement** であって imitation ではない。
old-Q と old action を教師として複製せず、同一状態・同一 future に対する二 action の
return 差を教師にする。deploy 時に不確実、非 finite、feature-schema 不一致、候補二個
未満のいずれかなら必ず old top-1 へ戻る。

### policy parameterization

1. canonical old model/OpenVINO policy は bitwise read-only とする。candidate set、stable
   order、candidate chunk 16 + actual tail、NEXT=5、HOLD、tie-break は baseline contract
   から変更しない。
2. old-Q が選ぶ top-1 を `a1`、stable order で二番目を `a2` とする。R1 が選べるのは
   `a1` または `a2` だけであり、search tree は deploy 時に展開しない。
3. feature は実装前に順序を固定する。old-Q の raw/within-state standardized top-1,
   top-2, gap、valid action count、`a2-a1` の immediate score、cleared-line、holes、
   covered cells、aggregate/max height、bumpiness、well sum、row/column transition、REN、
   B2B、T-spin の差、current-board の同 safety summary、HOLD/NEXT6 の 42 one-hot とする。
   feature の追加削除や変換 sweep はしない。train mean/std だけで標準化し、constant
   feature は zero にする。全係数を含め 100 parameters 未満を要求する。
4. train episode を cluster として replacement samplingする 256 個の ridge model を
   解析的に解く。target は後述の clipped advantage、ridge `lambda=1.0`、bootstrap RNG
   `Xoshiro(0x5231_2026)` に固定する。予測の 10th percentile `L(x)` が **`>0.05`**
   のときだけ `a2`、それ以外は `a1` とする。threshold、lambda、quantile、feature、
   ensemble 数の sweep、calibration result 後の変更は禁止する。

これは neural residual ではなく、old action から最大一順位だけ離れる bounded policy
gate である。R1 が通っても「旧 neural model 自体の改善」とは呼ばず、旧 model +
analytic gate の **system improvement candidate** と報告する。

### 新規 online data と seed role

既存 C10/C13/Q1 dataset は一切入力にしない。次の seed はこの report 以後、恒久的に
training-only とし、development/validation/test evidence に再利用しない。

- ridge train: `73001:73012`（12 episodes）
- gate calibration: `73101:73106`（6 episodes）
- sampling state: canonical old-policy trajectory の piece
  `10,20,...,240`。branch rollout は本 trajectory に書き戻さない。
- minimum: train 240 states、calibration 120 states。候補二個未満、branch construction
  failure、非 finite Q/return は manifest に記録してその state を除くが、minimum 未満
  なら追加 seed を足さず reject する。

各 state で `a1`,`a2` から state と RNG を独立 copy し、**root を含む6 pieces** を
同じ future-piece stream で進める。root 後は canonical old policy を用いる。return は

```text
G6(a) = sum(k=0:5, gamma^k * score_delta_k / 600)
        + gamma^6 * max_old_Q(s6),  gamma=0.997
A6 = G6(a2) - G6(a1)
```

とする。terminal branch の bootstrap は 0。両 branch の future piece/RNG digest が
一致しなければ全 experiment を失敗にする。fit target は
`clamp(A6,-2,2)`、calibration の scientific metrics は unclipped `A6` を用いる。

state/feature/return/branch-digest/order を固定 numeric table として保存し、train と
calibration の source seed を別 artifact/hash にする。validation `8001:8008`、sealed
test `91001:91032`、既使用 development `5742:5755` は load も run もしない。

### data/fit preflight と停止条件

実行前に hypothesis、source/Manifest、old checkpoint/NPZ/IR、feature schema、seed role、
全 output path、phase command を freeze し、fresh global one-shot marker を使う。既存
artifact の上書きと partial phase resume は禁止する。

次を一つでも満たさなければ calibration へ進まない。

1. copied state を同じ root action で二度 rollout した synthetic/training-only probe の
   score、piece、RNG、selected-action digest が一致する。
2. branch ごとに canonical candidate/chunk/tie semantics を保ち、全 Q/feature/return が
   finite。old checkpoint と OpenVINO artifact の hash が開始前後で不変。
3. data generator の process-tree peak private bytes `<=4 GiB`、working set `<=2 GiB`。
   analytic fit は private bytes `<=1 GiB`。Q1 と異なる parameterization であり、Q1 の
   4 GiB gate を緩めた実験ではない。
4. 最初の32 counterfactual statesからの data-generation total projection `<=55 min`。
   data + fit + calibration hard wall `65 min`、後述 development screen 込み総 hard wall
   `80 min`。超過投影、non-finite、driver reset、process leak で即時停止する。
5. train positive fraction (`A6>0`) が `0.02--0.40`。範囲外なら識別 gate の情報が
   不足または top-2 teacher が非選択的なので、class weightingやcandidate幅を変えず
   reject する。

### calibration promotion gate

train だけで係数、標準化、bootstrap ensemble を確定し、calibration は一回だけ
pass/reject に使う。calibration を見て threshold、lambda、feature、horizon、seed、
ensemble を変更しない。次をすべて要求する。

1. override は calibration state の `1%--15%`、かつ最低12 states、4/6以上のepisode
   に分布する。
2. override 中の `A6>0` precision `>=0.70`。episode-cluster bootstrap 2,000回、固定
   RNG `0x5231_73106` の one-sided 90% lower boundが `>0.50`。
3. override の mean unclipped `A6 >=0.10`、同 cluster-bootstrap one-sided 90% lower
   boundが `>0`。
4. R1 が選んだ `a2` branch が horizon内terminalで、同じ state の `a1` が生存する例
   は 0。全 fallback path で old top-1 と完全一致する。
5. feature/fit/decision の独立 scalar reference と production evaluator が全 calibration
   rowsで同じ decisionを返す。追加 inference median `<=0.10 ms/state`、係数と artifact
   は finite。

pass は `R1-calibration-promoted` にすぎず、game strength、G1--G3、旧 model beat の
証拠ではない。fail 時は threshold調整、top-3化、horizon変更、seed追加をせずこの
exact branchを閉じる。

### 一回限りの development screen

calibration pass 時だけ係数、feature order、old base、source/Manifest、data hashes、
decision reference を freeze し、これまで実行されていない development seed `5756`,
`5757` をこの順に用いる。各 pair は R1 と canonical old baseline、250 pieces、NEXT5、
HOLD、stable candidate order、同じ candidate set、one old full-candidate logical pass per
decision、zero deploy-time lookahead とする。old networkの candidate evaluationsと
physical NPU/CPU-tail requestsは同一で、gateのscalar CPU callとwallを別計上する。

最初の pair differenceが `<=0` なら直ちに停止して5757を使わない。promotion は

- 2/2 paired differences `>0`
- paired mean improvement `>=500`
- completion non-regression
- gateを含むmedian decision wall overhead `<=5%`
- score、pieces、override positions、係数/hash、network/candidate/physical request、
  generation/inference/wallを完全記録

のすべてを要求する。最高点が18,400以上ならG1候補として記録してよいが、二seedの
passもG2またはmodel-only improvementではない。statusは
`R1-development-promoted-system-candidate` に限定する。

R1 pass後も validation/testを自動実行しない。独立freeze review後に一回限りの
validation `8001:8008`を事前登録し、それを通った固定candidateだけが sealed 32 seeds
へ進める。

## AD と CPU memory に関する採否

R1 の fit は small dense linear algebra の解析解で、Lux、Zygote、Enzyme、Reactant の
いずれも使わない。これはAD改善を無視した判断ではない。

- native Enzymeは二度のactual benchmarkでnative Zygoteに負け、現在pinsで再試行する
  情報利得がない。
- Reactant+EnzymeMLIRの利得は、固定shapeで少なくとも約711 updatesを継続するlearner
  でcompile costを償却した場合に限る。R1にoptimizer updateはなく、適用対象がない。
- Q1はAD phaseのloss/updateを残していないため、Q1 memory terminalをbackend選択の
  evidenceに流用しない。
- R1のstreaming numeric rows、100未満features、256個の小ridge solvesは、20.8M旧
  gradient treeや165k candidate tensorを保持しない。Q1で問題化したprivate-bytesを
  科学的仮説ごと避ける。

後続でmedium neural criticを選ぶ場合の既定は、短い/dynamic pilotではZygote、
1,000 updates以上の固定shape production候補ではchanging-batch/transfer/checkpoint込み
1.15x gateを通したReactant+EnzymeMLIRである。R1の結果を見る前にその後続を実行しない。

## 最終判断

Q1はterminal failureであり、residual学習の反証ではないが、契約上閉じている。現時点
で旧modelを最短に越えるために必要なのは、同じoffline selected-action residualを
再試行することではなく、未選択actionの改善returnを新しく得ることである。

従って次は **R1 online counterfactual top-2 safety gateだけ**を、fresh training-only
data、解析的fit、固定calibration、一回限りの残りdevelopment pairの順で行う。成功時も
system candidateとmodel improvementを分離し、失敗時はこのtop-2/horizon6/gateを閉じる。
sealed seedは使わない。
