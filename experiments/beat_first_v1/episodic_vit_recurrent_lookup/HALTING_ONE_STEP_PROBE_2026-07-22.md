# 候補単位1-step halting probe — 2026-07-22

> **2026-07-23追補**：本書はprobe-only信用割当を試した時点の履歴である。その後、
> probeが有効な場合にtrajectory policy gradientを完全に置換すると、未probe stopと
> それ以前のcontinue判断へhalting gradientが届かないことが判明した。現行実装は全
> stochastic stop/continueへpolicy gradientとentropy gradientを与え、少数probe BCEを
> 対象candidateの最終stopへ加算する。最新結果は
> [`ROOT_CAUSE_REPAIR_100K_2026-07-23.md`](ROOT_CAUSE_REPAIR_100K_2026-07-23.md)
> を参照すること。

## 判断

probe mode有効時には、state全体へ同じ符号を与えるREINFORCE型halting更新を、物理的疎性を保ったcandidate-local教師へ置き換える。アーキテクチャ、20,577,789 parameter、入力・teacher contract、通常のtask loss、hard routing、active-only backward、sparse optimizer、checkpoint形式、candidate RNG順序、candidate独立評価は変更しない。

stateごとにsampled stopしたcandidateから最大`P`個を選び、次の処理を行う。

1. 追加の再帰stepを正確に1回だけ実行する
2. そのcandidateのscalar Qだけを`t+1`のQへ置換する
3. 同じListNetとmarginによるranking値を再計算する
4. `delta = L_stop - L_continue`と定義する
5. `delta > c`ならcontinue、それ以外ならstopを最終halt logitの教師とする

probeしなかったcandidateにはhalting gradientを与えない。task backwardの前に元のscore matrixを復元するため、task lossとtask VJPはprobeに依存しない。probe forwardはroute-usage counterを更新せず、backward用probe trajectoryも作成しない。

実行時設定は次のとおり。

```text
EVRL_HALT_PROBES_PER_STATE   stateごとにprobeする停止candidate数
EVRL_HALT_PROBE_WEIGHT       candidate-local halt targetに対するBCE weight
EVRL_COMPUTE_PRICE           probe modeでのcontinue閾値c
EVRL_ENABLE_HALT_PROBE_TRANSITION=1
                             旧checkpointからの明示的transition gate
```

probe数のdefaultは0であり、旧checkpointの挙動を維持する。version 1 checkpointにprobe fieldが存在しない場合は、probe数0、weight 1としてnormalizeする。checkpoint serialization形式は変更していない。

## 1-step primitiveのwitness

強制深度3のtrajectoryに`probe_one_step!`を適用した結果を、同じmodel・入力を通常の強制深度4でforwardした結果と比較した。出力差の最大絶対値は`2.3841858e-7`で、許容値`2e-6`を下回った。

## 実teacherによるserial/barrierless正当性smoke

productionと同じ20-worker smokeでは、現行20,577,789-parameter R2の10,000更新checkpoint、4つのtraining state、その全valid candidate（34、53、51、68個）、stateあたり2 probeを使用した。validation rowおよびsealed game seedは構築も参照もしていない。

結果：**合格**。

```text
output max abs                         0
loss max abs                           0
raw task VJP max abs                   0
probe target exact                     true
probe delta exact                      true
halt RNG state / next values exact     true / true
sampler state / next rows exact        true / true
parameter gradient max abs             4.0382147e-6
parameter gradient relative L2         1.7158563e-6
post-optimizer parameter max abs        4.0419400e-7
optimizer clocks                       exact at update 10,001
```

checkpoint witness：

```text
D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\
  evrl_recurrence_cp0_haltlr1e5_warmup5k_u100000_20260722_r2\checkpoints\
  checkpoint_000010000.jls
sha256 3bd4140707a10cd63781bd39c65d21255ae8dbaa0ea022c78ab501b3f014041b
```

## 100更新throughput preflight

同じcheckpointから、warmup 10更新と測定100更新の実teacher学習を実施した。stateあたり2 probe、barrierless scheduling、pinningなし、chunk 8、Julia worker 20、BLAS thread 1である。benchmark更新のcheckpointは保存していない。

```text
measured updates             100
updates/s                    20.6586
states/s                     82.6343
candidate CPU                60.5010%
overall CPU                  60.1197%
last update probes           8
continue / stop targets      2 / 6
last mean delta             -6.0588e-5
minimum speed criterion      15 updates/s
result                       pass
```

このpreflightがprobeなしのR1全体より遅いのは、1更新あたり最大8回のexact追加再帰stepを実行し、4つの小さなstate-local ranking lossを再計算するためである。それでも明示的に許容された下限15 updates/sを上回った。これは品質評価ではない。benchmark-only modeでは新しいheld評価を行わず、checkpointも書き出していない。

## 試験状況

candidate-localな信用割当への訂正が指定された時点で、scalar halt-LR試験R2は、既に完了していた10,000更新境界で停止した。上記smokeの変更不能な親checkpointとしてのみ保持し、100k完了済み調整結果とはみなさない。次のfull trialでは、完了済みR1 policyへ後付けせず、動的学習scheduleの開始時点からprobeを有効にする。

## 試験P1 — 100,000更新probe学習完了

P1ではprobe-aware policyをscratchから学習した。全アーキテクチャ、20,577,789 parameter、実teacher提示順、optimizer、task loss、hard routing、executorを固定した。再帰R1から変更したhalting信用割当は、上記candidate-local probe targetだけである。

```text
probe candidates/state       2
probe BCE weight             1
continue threshold c         0
halt learning rate           5e-5
random-depth warmup          updates 1--5,000, depth 2--6
learned recurrent range      2--12
teacher states/update        4
total updates/states         100,000 / 400,000
```

orchestration commandは75,000更新checkpointを完全に書き出した後、hostの2時間timeoutへ到達した。model errorは発生していない。そのcheckpointからoptimizer、sampler、halt RNG、route-usage stateを正確に復元して学習を再開し、100,000更新まで完了した。resume runの初期評価は75,000更新時のmetricを完全に再現した。

### Held panelと深度推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin | 学習深度 | Held深度 | Held範囲 | Probe continue/stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 | 2.000 | 2.000 | 2--2 | -- |
| 5,000 | 2.840448 | 0.53125 | 0.979206 | 0.839419 | 0.105218 | 2.000 | 2.000 | 2--2 | warmup |
| 10,000 | 2.816590 | 0.52344 | 0.978723 | 0.843999 | 0.103889 | 3.607 | 3.538 | 3--12 | 4 / 4 |
| 15,000 | 2.774000 | 0.50000 | 0.980272 | 0.852634 | 0.097204 | 3.048 | 3.006 | 2--12 | 3 / 5 |
| 20,000 | 2.762955 | 0.59375 | 0.983187 | 0.863068 | 0.089299 | 5.121 | 4.886 | 3--7 | 8 / 0 |
| 25,000 | 2.723159 | 0.53125 | 0.981245 | 0.864550 | 0.113911 | 2.027 | 2.055 | 2--12 | 4 / 4 |
| 30,000 | 2.687534 | 0.58594 | 0.984653 | 0.870740 | 0.113859 | 2.025 | 2.060 | 2--12 | 6 / 2 |
| 35,000 | 2.678400 | 0.62500 | 0.984968 | 0.873602 | 0.126336 | 5.014 | 4.940 | 4--12 | 4 / 4 |
| 40,000 | 2.663871 | 0.61719 | 0.985075 | 0.876878 | 0.140843 | 2.068 | 2.105 | 2--12 | 4 / 4 |
| 45,000 | 2.654798 | 0.61719 | 0.985994 | 0.883263 | 0.119168 | 2.371 | 2.490 | 2--12 | 5 / 3 |
| 50,000 | 2.643391 | 0.61719 | 0.985995 | 0.884732 | 0.142789 | 5.307 | 5.347 | 3--12 | 5 / 3 |
| 55,000 | 2.636955 | 0.63281 | 0.988076 | 0.887374 | 0.125715 | 2.059 | 2.096 | 2--12 | 4 / 4 |
| 60,000 | 2.647861 | 0.67188 | 0.988406 | 0.886657 | 0.120213 | 2.076 | 2.122 | 2--12 | 4 / 4 |
| 65,000 | 2.622504 | 0.67188 | 0.989416 | 0.894942 | 0.140595 | 3.095 | 3.076 | 2--4 | 3 / 5 |
| 70,000 | 2.626701 | 0.65625 | 0.988544 | 0.892699 | 0.145571 | 2.203 | 2.270 | 2--12 | 5 / 3 |
| 75,000 | 2.614308 | 0.67188 | 0.989111 | 0.897401 | 0.132339 | 3.460 | 3.399 | 2--6 | resume boundary |
| 80,000 | 2.604949 | 0.69531 | 0.990553 | 0.899782 | 0.123521 | 2.044 | 2.042 | 2--12 | 4 / 4 |
| 85,000 | 2.613532 | 0.70312 | 0.988633 | 0.896584 | 0.133625 | 2.433 | 2.529 | 2--12 | 3 / 5 |
| 90,000 | 2.592303 | **0.74219** | 0.991142 | 0.902739 | 0.119525 | 3.021 | 3.021 | 2--12 | 4 / 4 |
| 95,000 | **2.587874** | 0.73438 | **0.991345** | **0.904401** | 0.141909 | 2.205 | 2.194 | 2--12 | 3 / 5 |
| 100,000 | 2.605494 | 0.71094 | 0.990369 | 0.902199 | **0.151411** | 2.030 | 2.011 | 2--12 | 5 / 3 |

continue/stop列は各reporting boundaryにおける単一updateのtelemetryであり、直前5,000更新のpopulation countではない。それでもこの有界なwitnessから、candidateごとに両方向のtargetが生成されていることが分かる。すべてのcandidateへ一つのstate-wide advantage符号を強制する状態ではなくなった。

### 判断

P1は信用割当として成功、動的深度としては部分的成功である。品質と深度の釣り合いが最良なのは95,000更新checkpointである。P1中最低のheld composite lossと最高のheld NDCGを持ち、top-1 `0.73438`、pairwise `0.90440`、margin `0.14191`、平均深度`2.19`を維持する。90,000更新はP1のtop-1最高値（`0.74219`、平均深度`3.02`）、100,000更新はmargin最高値（`0.15141`）である。

不採用のstate-wide R1最終checkpointと比較すると、P1最終値はlossを`0.004945`、top-1を`0.007812`、NDCGを`0.000565`、pairwise accuracyを`0.002944`、marginを`0.008825`改善した。さらに重要なのは、R1のほぼ完全な深度12飽和（平均`11.91`）で終了せず、P1は`2.01`で終了し、途中の複数checkpointでheld平均深度3～5を示した点である。

ただし、これは理想的な適応計算を証明するものではない。P1には依然として振動があり、選択checkpointと最終checkpointは下限深度寄りである。固定深度試験1の最終controlと比べると、P1の95,000更新はloss `+0.006163`、top-1 `+0.015625`、NDCG `-0.000456`、pairwise `-0.001826`、margin `-0.004496`である。したがってcandidate-local halting信用割当は改善したが、固定深度に対する全metricでの明確な勝利ではない。

記録済みPreAct panel結果と比べると、P1の95,000更新はtop-1で`0.054688`、NDCGで`0.001945`、pairwise accuracyで`0.018959`低い。一方、marginは`0.018589`高く、composite lossは`0.024094`高い。このheld panelは開発判断に用いたもので、sealed generalization setではない。

### 実行時間と成果物

```text
aggregate training time       5,015.947635 s
aggregate updates/s           19.936412
aggregate average CPU         59.8515%
aggregate candidate CPU       60.9202%
resume 75k--100k time         1,298.462386 s
resume updates/s              19.253542

balanced checkpoint:
  ...\evrl_haltprobe_p2_c0_w1_warmup5k_u100000_20260722_p1_resume2\
  checkpoints\checkpoint_000095000.jls
  sha256 33de0fce4f1e7b6b734069bebe00a3eb822635a4e1d3131619f84f8c206cef02

final checkpoint:
  ...\evrl_haltprobe_p2_c0_w1_warmup5k_u100000_20260722_p1_resume2\
  checkpoints\checkpoint_000100000.jls
  bytes 253,691,534
  sha256 224b8f0adb7c41b3c5b9830a1d06b2ffe8ab00266644db5c29a1c29c07e26458

first-segment metrics sha256:
  9ba8da7da37fa637dc166b76410612faa7b4d5f2ee55e4a06a5f2059919bcb9c
resume metrics sha256:
  1ef8b116b594d55272388f12ef3573f19cbdefba078e1b927a1ef1eb8ed55321
resume summary sha256:
  55fede26a83af61d4d45fdc994f58f70e4c1de0b34da62a7ab546e428d6937ab
```

binary checkpointとteacherデータはlocalに保持し、Gitへcommitしていない。
