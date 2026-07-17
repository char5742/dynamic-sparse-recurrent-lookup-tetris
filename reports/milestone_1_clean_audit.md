# Milestone 1 clean audit

監査日: 2026-07-17

## 監査範囲と結論

この監査は、現在のワークスペースに存在する `reports/g0_audit.md`、
`reports/experiment_ledger.md`、`configs/evaluation_protocol.toml`、
`experiments/`、`scripts/`、`runs/` と、それらが直接参照する評価実装・成果物だけを
根拠とした。過去の会話、担当者の説明、保存されていない標準出力は証拠に含めていない。
実験は実行していない。

**結論は「基準モデルの復元は有用な段階まで進んだが、公平な同一条件で統計的に
上回った証拠はまだない」である。** 現時点で計算資源を最優先で投じる先は新しい
ハイパーパラメータ探索ではなく、基準モデルの意味論とG2判定器の固定である。
その後に限り、唯一の正の兆候である1手Bellman再順位付けを、追加先読みなしの
1回推論モデルへ蒸留する実験に集中するのが最短である。

## 1. 証拠上の進行度

| 項目 | 独立判定 | 現在の証拠 |
|---|---|---|
| G0: 旧モデル復元 | **条件付き達成** | チェックポイントのハッシュ、モデル構造、エンジン、スコア規則が記録され、NPUで開発6 seedの250手完走結果がある。ただし、正準Luxモデルと実運用NPU方策の行動一致は未証明である。 |
| G1: 18,400以上 | **未達** | 現存する完全な250手JSONで18,400以上はない。旧NPUの最大は16,200。台帳上のlookahead 17,000はG1未達であり、その17,000の生JSONも現存しない。 |
| G2: 統計的超過 | **未着手** | 32 test seedのペア評価はなく、適格な1回推論候補もない。test seedを実行した成果物は見当たらない点は良い。 |
| モデル候補 | **有効候補なし** | 模倣、残差模倣、価値head TDはいずれも基準未満または学習前が最良。architecture、AD backend、compact learningは測定結果がない。 |
| システム候補 | **弱い仮説のみ** | E005は台帳上2 seedで差 `[+1100, 0]`、平均+550だが、追加候補評価と推論時間を約2倍以上使うためmodel-onlyではない。さらに+1100側の完全な生成果物がない。 |

現存する旧NPU/NEXT=5の6本 (`5742:5747`) は全て250手完走し、スコアは
`[15900, 15600, 15200, 14000, 16200, 15100]`、平均 **15,333.3**、中央値
**15,400**、範囲 **14,000–16,200**、候補評価合計 **65,171** である。
これは開発基準として有用だが、6 seedの分布だけで旧モデルの裾や最大値を復元したとは
いえない。

失敗枝の実測は一貫している。

- `imitation_v1_summary.json`: 5 episode平均1,040、最大2,400。
- `residual_imitation_v1_summary.json`: 学習前平均7,520が、step 50で6,380、
  step 800で3,040へ悪化。保存上の最良はstep 0。
- `value_head_training.json`: episode 0の平均15,700が最良で、episode 4は15,550、
  episode 8は15,000。保存上の最良は未学習head。
- NEXT4/5 ensemble: seed 5742で14,200、同seedのNEXT5基準15,900に対し
  **-1,700**、かつ推論を2回行う。

## 2. 最大の盲点、リーク、不公平

### P0: 現在のG2判定器は評価契約を強制しない

`scripts/compare_paired_evaluations.jl` が検査するのは両JSONのseed集合が同じことだけで、
次を検査しない。

- seedが正確に `91001:91032` の32本であること
- 全episodeが250手契約であること
- checkpoint、architecture、NEXT=5、候補順、tie-break、推論batch、backendが同じ
  凍結条件に従うこと
- model-onlyの `lookahead_expansions=0` と推論予算を守ること
- `network_calls`、候補評価数、時間指標が全て存在すること

入力に `steps` がなければ250と仮定し、候補数や推論時間の欠損も許す。したがって任意の
2開発seedでも `g2_location_success=true` を出せる。現状の成功フラグはG2証拠として
使用してはならない。

### P0: 実験状態が版管理されておらず、台帳の必須来歴を満たさない

監査時点でGitリポジトリにcommitがなく、対象ファイルは全てuntrackedだった。
`experiment_ledger.md` が新規実験に要求する source hash、Manifest hash、完全config、
checkpoint/log pathを満たすrunはない。多くのJSONはcheckpoint hash、コードhash、
Manifest hash、実際のnetwork request数を記録していない。この状態では、同じ名前の
成果物を同じコードから再現したと証明できず、昇格対象にできない。

### P0: 旧モデルの方策意味論にbatch/backend依存が残る

旧LayerNormは候補batch軸を含めて正規化する。そのため候補順だけでなく、
**batch size、batch境界、短い末尾batchの処理backend** がモデルそのものの一部である。
`LegacyOpenVINOInference` は16件の完全batchをNPU/GPU、末尾を動的CPUで処理する
ハイブリッド方策だが、評価契約はbatch size 16とCPU tailを凍結していない。
また `network_calls_per_decision=1` は論理的な全候補採点を意味するのか、実際の
OpenVINO requestを意味するのか不明で、runsは `network_calls` を記録していない。

数値同値性の成果物はランダムな **batch 8を1個** 比較したものだけである。NPUの
最大絶対誤差は約6.8e-4だが、argmax一致率、最小Q margin、実ゲーム軌跡一致はない。
実際、同じ旧headでも、直接NPU評価はseed 5742/5743で `[15900,15600]`、NPU featureを
分離して旧Lux headへ入れたE002 episode 0は `[15400,16000]` となり、実装経路で方策が
変わり得ることが確認できる。OpenVINOは高速な実装候補ではあるが、正準旧モデルと
同一方策であるという証拠にはまだ不足する。

### P1: E005の中心主張を現在の生成果物から再計算できない

台帳はseed 5742のtop-2/blend 0.5を17,000とするが、その250手・marginなしのJSONは
存在しない。現存する `openvino_lookahead_k2_b0.5_seed5742.json` は名前に反して
`max_steps=100`、`q_margin_threshold=0.05`、score 5,200であり別実験である。
seed 5743のmarginなし結果は、保存失敗後に作られたことを明記したJSONで、コードhashや
元ログを持たない。空の同名JSONも残る。従って `[+1100,0]` は探索仮説としては使えても、
昇格証拠には使えない。

### P1: seed分割が複数あり、同じ「development」の意味になっていない

凍結protocolのdevelopmentは `5742:5757` だが、模倣のモデル選択は `4001:4005`、
value-headの学習は `7001...`、compact learningは `71001:71008` をdevelopmentと呼ぶ。
学習用seedを別に持つこと自体は正しいが、モデル選択に使う内部holdout、protocolの
development、validationを明確に分離しないと、異なる枝の数字を比較できない。
特に `4001:4005` は複数snapshotから最良を選ぶために使われている。

### P1: testは未使用だが、真にsealedではなく、主判定も二重である

test seedは設定ファイルに平文で列挙され、実行禁止は運用上の注意書きだけである。
現時点で漏洩実行の証拠はないが、凍結前に誰でも参照・反復できる。
また成功規則の「paired mean **or** median」は二つの主検定のORであり、多重性調整がない。
主統計を一つに事前固定するか、両方を使うなら補正が必要である。

### P1: 現在のcompact offline TDには旧方策を超える直接の教師信号がない

teacher datasetは全候補の旧Qを保存するが、報酬とn-step遷移は旧方策が選んだactionに
しかない。TD更新も `selected_batch`、すなわち旧方策の選択actionだけを学習する。
これは旧方策の圧縮・模倣には有効だが、未選択actionが実際には良かったという反実仮想
信号を持たず、単独で旧モデルを超える機序が弱い。40手で切ったepisode末尾もterminalで
ないまま部分returnとして扱われる。C01はインフラsmokeとして有用だが、現状のC02を
大規模化する根拠にはならない。

### P2: G1の18,400は統計的超過の証拠ではない

18,300はseedなしの過去最大であり、現在は候補生成RNGを修復した別の実行契約である。
さらに新手法側の試行回数を揃える規則がない。18,400の単発達成はデモとしてはよいが、
旧モデルを公平に超えた主張は必ずG2のペア差で行うべきである。

## 3. 即時停止または保留すべき枝

### 終了

- E003 perfect-clear beamの単純な幅拡大。
- E004 NEXT4/5 ensemble。悪化し、model-only推論予算も超える。
- E006 margin 0.05 gating、およびseed 5742での追加margin/blend/top-k sweep。
- 現行 `imitation_v1` と `residual_imitation_v1` の継続学習。
- 現行E002 value-head TDの同じデータ生成・損失・ハイパーパラメータでの延長。
- top-k 3/4、高blendのlookahead枝。部分episodeでも計算量に見合う優位がない。
- test seedの実行。候補と評価器が凍結されるまで0本を維持する。

### 保留

- E005はmodel-only候補としては終了する。システム研究として残す場合も、現在の成果物を
  使った追加sweepは止め、評価器修復後の事前登録ペア確認を1回だけ行う。
- AD backend比較と大規模architecture benchmarkは、学習信号が基準を上回るまで保留する。
  今はlearner throughputより候補品質と評価同一性が律速である。
- C02 offline TDの大規模化は保留する。C00/C01の短いsmokeは実装健全性確認に限り許すが、
  旧Q模倣精度だけを性能昇格理由にしない。

## 4. 最短の次の1〜3実験

### N1 — 正準baseline・評価器conformance（最優先、性能探索前）

実ゲームの固定状態列で、Lux FP32 CPU、OpenVINO CPU、NPU batch 16 + CPU tailの
候補ベクトル、candidate hash、argmax、top-2 marginを比較する。完全batchと短いtailを
必ず含め、同じ初期状態から選択actionと軌跡も照合する。同時にG2判定器へ32 test seed、
250手、凍結checkpoint/config hash、NEXT、予算、必須metricsの強制検査を入れる。

成功条件は、(a) 正準実装を一つ宣言できる、(b) 採用acceleratorが監査軌跡で100%同じ
actionを選ぶ、または不一致なら正準Luxを旧baselineとして使いaccelerator結果を別物と
明記する、(c) 不完全なrunからG2成功フラグが出ない、の全てである。失敗時は新規学習を
止めてここを修正する。

### N2 — top-2 Bellman教師の1回推論蒸留smoke

現在の正の機序は、追加計算を使うtop-2 Bellman再順位付けだけである。旧方策の全Qを
そのまま模倣するのではなく、学習専用trajectory上でtop-2の1手先target/選択を生成し、
凍結旧表現への小さなresidual head、または1つに絞ったcompact/FiLMモデルへlistwiseに
蒸留する。実行時はlookaheadなし、全候補への論理的な採点1回だけにする。

30〜60分smokeの成功条件を事前に、(a) 内部holdout状態で再順位付け教師へのtop-1一致が
旧argmaxより明確に改善、(b) 未学習よりlistwise lossが低下、(c) model-only evaluatorで
有限かつ決定的、(d) parameter/backend/throughputが記録済み、とする。満たさなければ
この機序への大規模データ生成を止める。

### N3 — 未使用developmentからvalidationへの単一候補ゲート

N2を通った**1 architecture・1 checkpoint**だけを凍結し、まず未評価のprotocol
development seed `5748:5757` で正準旧baselineとペア評価する。次へ進む目安は、平均差
+500以上、中央値差>0、10本中7本以上が非負、completion rate非悪化である。これは探索用
gateでありG2主張にはしない。

通過した1候補だけをvalidation `8001:8008` で1回評価する。validationでも事前固定した
主統計のpaired 95% CI下限>0、completion rate非悪化を要求する。失敗したらtestを開かず、
結果を受けた同validationへの反復調整もしない。別候補を試す場合は、validation再利用に
よる選択バイアスを明記し、より厳しい最終判定を採用する。

## 5. 昇格条件

候補がtestへ昇格する前に、次を全て満たす必要がある。

1. **来歴固定**: commitまたは内容ハッシュでsource、Manifest、protocol、checkpoint、
   evaluatorを固定し、runに完全config、seed、backend、parameter count、終了理由を保存する。
2. **正準旧baseline固定**: batch size、候補chunk境界、CPU tail、dtype、tie-breakを含む
   旧モデルの方策意味論を固定する。acceleratorを使うならaction同値性を証明する。
3. **model-only同一条件**: 同じエンジン、スコア、250手、Xoshiro seed、HOLD、NEXT=5、
   stable candidate order、候補集合を使い、lookahead=0とする。「1 network call」の定義を
   論理採点passと実backend requestに分け、両方を記録する。
4. **単一候補凍結**: test前にarchitecture、checkpoint、NEXT、推論budget、backendを固定し、
   validationを通過した候補を差し替えない。
5. **一回限りのG2**: 正確な32 test seedを旧baselineとcandidateでペア評価し、欠損runを
   許さない。主統計は平均差または中央値差の**どちらか一つ**を事前固定し、固定seedの
   paired bootstrap 95% CI下限>0を要求する。OR規則を残すなら多重性補正を行う。
6. **完全報告**: mean、median、max、p10/p25/p75/p90、completion rate、全paired差、CI、
   candidate evaluations、論理/物理network calls、generation/inference/wall timeを報告する。
   最大値やG1は副次指標に留める。
7. **system候補の分離**: lookaheadやensembleはmodel-onlyに混ぜず、同じ計算予算を旧側にも
   与えた別比較、または追加計算量を明示したsystem結果としてのみ昇格する。

計算配分は、まずN1へ必要量を全投入し、通過後はN2/N3へ集中する。現在の失敗枝、AD速度
最適化、追加architecture列挙、test seedには計算を配分しない。これが「旧モデルを公平な
同一条件で統計的に超える」までの最短経路である。
