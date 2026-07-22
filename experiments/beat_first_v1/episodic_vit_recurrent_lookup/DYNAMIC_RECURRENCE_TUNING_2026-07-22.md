# EVRL動的再帰調整 — 2026-07-22

## 訂正と目的

直前に実施した100,000更新のsweepは、再帰深度を2に固定したままdense learning rateとweight decayを変更したものだった。これらは固定深度controlとして有効だが、この段階で求められていた動的再帰機構の調整にはなっていない。本記録ではその対象範囲を訂正する。ここに記載する試験はすべて、5,000更新のrandom-depth curriculum後にsampled hard haltingを最初から使用し、haltingに関するscalarだけを一度に一つ変更する。

アーキテクチャ、20,577,789 parameter、入力contract、実teacherデータと提示順、ranking loss、LookupFFN routing、active-only backward、sparse optimizer、20-worker barrierless executor、128-state held panel、初期化seedは固定した。game validationとsealed seedには触れていない。

成功条件は、最終ranking scoreが良いことだけではない。決定論的held深度が下限または上限に飽和せず、入力依存性を維持すること、sampled training深度にも同じ定性的挙動が現れること、品質と計算量の関係が固定深度controlに対して競争力を保つことを要求する。

## 試験R1 — compute priceを0へ変更

### 単独変更点

従来の動的設定では`compute_price = 0.02`だった。R1ではこの値だけを0へ変更し、次の条件を維持した。

```text
warmup updates       5,000 (uniform random depth 2--6)
fixed depth          disabled
initial halt prob.   0.5
halt LR              5e-5
policy weight        0.05
entropy weight       0.001
dense weight decay   1e-4
bank/router LR       2e-4 / 4e-4
recurrent range      2--12
```

これは、過去に観測した深度2へのcollapseが、追加計算に対する明示的な価格だけで発生したという限定的仮説を検証する試験である。

### Held panelと深度推移

| 更新 | Loss | Top-1 | NDCG | Pairwise | Margin | 学習深度 | Held深度 | Held範囲 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.395339 | 0.21875 | 0.860435 | 0.546154 | 0.040159 | 2.000 | 2.000 | 2--2 |
| 5,000 | 2.857160 | 0.51562 | 0.979035 | 0.840006 | 0.104091 | 2.000 | 2.000 | 2--2 |
| 10,000 | 2.804495 | 0.53125 | 0.979766 | 0.850205 | 0.095223 | 2.126 | 2.138 | 2--8 |
| 15,000 | 2.770932 | 0.53906 | 0.981487 | 0.855166 | 0.088338 | 6.072 | 5.193 | 2--12 |
| 20,000 | 2.735852 | 0.57031 | 0.983592 | 0.863802 | 0.082615 | 2.006 | 2.027 | 2--12 |
| 25,000 | 2.721970 | 0.53906 | 0.981148 | 0.864647 | 0.111630 | 2.001 | 2.005 | 2--4 |
| 30,000 | 2.690626 | 0.55469 | 0.983931 | 0.866516 | 0.107203 | 2.004 | 2.019 | 2--8 |
| 35,000 | 2.691957 | 0.63281 | 0.984052 | 0.874513 | 0.115638 | 2.475 | 2.149 | 2--10 |
| 40,000 | 2.669604 | 0.62500 | 0.986060 | 0.881214 | 0.128812 | 2.052 | 2.116 | 2--12 |
| 45,000 | 2.640631 | 0.61719 | 0.986034 | 0.882523 | 0.136873 | 2.002 | 2.018 | 2--12 |
| 50,000 | 2.644805 | 0.65625 | 0.987395 | 0.885983 | 0.126937 | 2.036 | 2.051 | 2--12 |
| 55,000 | 2.638223 | 0.65625 | 0.988389 | 0.889363 | 0.116839 | 2.134 | 2.209 | 2--12 |
| 60,000 | 2.631392 | 0.64062 | 0.986514 | 0.886535 | 0.141391 | 12.000 | 11.998 | 11--12 |
| 65,000 | 2.622564 | 0.67188 | 0.988940 | 0.892556 | 0.130395 | 12.000 | 12.000 | 12--12 |
| 70,000 | 2.614607 | 0.65625 | 0.989047 | 0.895548 | 0.132581 | 11.939 | 11.947 | 2--12 |
| 75,000 | 2.609085 | 0.69531 | 0.989456 | 0.897448 | 0.133078 | 2.033 | 2.052 | 2--12 |
| 80,000 | 2.596597 | 0.69531 | 0.990671 | 0.899239 | 0.131604 | 2.004 | 2.012 | 2--6 |
| 85,000 | 2.607638 | 0.68750 | 0.989438 | 0.897030 | 0.136655 | 2.000 | 2.000 | 2--2 |
| 90,000 | 2.595474 | 0.70312 | 0.990179 | 0.899312 | 0.139997 | 2.331 | 2.450 | 2--12 |
| 95,000 | 2.597807 | 0.70312 | 0.990683 | 0.898203 | 0.146593 | 2.017 | 2.031 | 2--9 |
| 100,000 | 2.610439 | 0.70312 | 0.989804 | 0.899255 | 0.142586 | 11.955 | 11.913 | 2--12 |

決定的な結果は最終checkpointだけでなく、深度の軌跡にある。held平均深度は15kの`5.19`から20kの`2.03`へ移動し、60kから70kまではほぼ12に飽和し、75kには`2.05`へ戻り、最終的に`11.91`となった。学習時深度も同じ極値をたどった。これは有用な入力依存の計算配分ではなく、深度の両端を高分散で往復するpolicyである。

### 品質と速度に関する判断

R1は動的再帰設定として不採用とする。100kでloss `2.610439`、top-1 `0.703125`、NDCG `0.989804`、pairwise `0.899255`、margin `0.142586`を記録した。固定深度の試験1 controlと比較すると、lossは`+0.028728`、top-1は`-0.015625`、NDCGは`-0.001997`、pairwise accuracyは`-0.006972`悪化した。記録済みPreAct top-1よりも`-0.085938`低い。

100,000更新の実時間は`4,035.570673 s`で、`24.779643 updates/s`だった。平均CPU使用率は`78.9237%`、candidate区間CPU使用率は`81.8531%`である。固定深度controlは`31.9193 updates/s`だったため、不安定な深い飽和によりupdate throughputが約22.4%低下したにもかかわらず、held品質は向上しなかった。

compute priceを0にした結果、priceだけが原因という限定的仮説は否定された。境界間を大きく往復する挙動は、halt-policy更新が過度に強いか、noiseが大きいことを示している。次の一軸試験では、R1のその他の条件を保ったままhalt LRだけを`5e-5`から`1e-5`へ下げる。dense LR、weight decay、architecture、loss、routing parameterは変更しない。

### 実行witness

```text
run:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_recurrence_cp0_warmup5k_u100000_20260722_r1

checkpoint:
  D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_recurrence_cp0_warmup5k_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls
bytes:   253,690,013
sha256:  e39732e972ba32e7b35c4962fd26282a66e7d2cbeeafc356d7341013346589ef
updates: 100,000
consumed real-teacher states: 400,000

metrics.jsonl sha256:
  ad133aeec3d5578c2fa57aa2c674509c4dd2acb55c049ab7c5ab9a43d69dd7ca

summary.json sha256:
  82b021d40bbfeca481042dbfebdf9135cbdf26531fac78923617977b4b16986f
```

binary checkpointとteacherデータはGitへcommitしていない。

## 試験R2 — 信用割当の訂正後に中止

R2ではhalt learning rateだけを`5e-5`から`1e-5`へ下げ、10,000更新まで到達してcheckpointを保存したが、100,000更新までは継続しなかった。scalar調整ではstate-wide REINFORCE targetが残り、candidate単位のvalue-of-computationに対する信用割当を解決できないため試験を中止した。10,000更新checkpointは、完了済み調整結果としてではなく、1-step probe correctness smokeの変更不能な親checkpointとして保持する。

代替策は、有界なcandidate-local 1-step probeである。停止したcandidate一つのQだけをexact next-step Qへ置換し、ListNetとmarginを再計算し、`L_stop - L_continue > c`のときにcontinueを教師とする。正当性と速度preflightは[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md)へ記録した。

## 試験P1 — candidate-local 1-step probe

P1は、stateあたり2つの停止candidateをprobeし、`c = 0`、probe BCE weight 1、halt LR `5e-5`、R1と同じ5,000更新random-depth curriculumで100,000更新を完了した。モデル、データ、task loss、optimizer、routing、seed、held panel、executor条件はすべて固定した。

品質と深度の釣り合いが最良だったのは95,000更新checkpointで、loss `2.587874`、top-1 `0.734375`、NDCG `0.991345`、pairwise `0.904401`、margin `0.141909`、held平均深度`2.194`だった。top-1の最高値は90,000更新の`0.742188`で、平均深度は`3.021`だった。最終100,000更新では、loss `2.605494`、top-1 `0.710938`、NDCG `0.990369`、pairwise `0.902199`、margin `0.151411`、held平均深度`2.011`となった。

R1と異なり、P1では同一reporting batch内にcontinue targetとstop targetの両方が現れ、最終状態も深度12へ飽和しなかった。最終品質指標はすべてR1を上回ったが、依然として振動があり、最終深度は下限に近い。この方式は正しいsparse halting信用割当機構として採用するが、現在のhalt policyが理想的な入力依存計算へ到達した証明とはみなさない。

5,000更新ごとの完全な推移、exact serial/barrierless smoke、throughput、checkpoint hash、PreActおよび固定深度との比較は[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](HALTING_ONE_STEP_PROBE_2026-07-22.md)に記録している。
