# Milestone 2 clean review

監査日: 2026-07-17

## 監査範囲と結論

このレビューは、現在のワークスペースにある `reports/`、
`configs/evaluation_protocol.toml`、評価・G2判定用の `scripts/`、
`experiments/` と、それらが参照する現存run・成果物だけを根拠とした。過去の会話は
根拠にせず、ゲーム、学習、長時間benchmarkは実行していない。

**結論は「短い開発用判定だけは続行可、長い学習・validation・testへの昇格は不可」
である。** 前回監査後、32 test seedの強制、主統計のpaired meanへの一本化、
論理model passと物理backend requestの区別、旧方策の実状態conformanceが追加され、
評価設計は明確に改善した。一方、正準baselineはprotocol上まだ `pending`、G2 exporterは
未実装、hashは実体と照合されず、全ファイルはGit上untrackedである。従って、今G2を
実行しても比較の来歴を固定できない。

学習面では、74-actionへ修正したcompact listwise学習が再現し、lr=1e-3で500 held
development statesの旧教師top-1一致が **37.0%** まで上がった。これは有効な
「旧モデル圧縮の学習信号」だが、教師は旧Qそのものなので、旧モデル超過の教師信号では
ない。現在のcompact checkpointにはゲーム結果がなく、Bellman residual gateは唯一の
100手評価で旧モデルより **-1,000** だった。公平な複数seedで旧モデルを上回った候補は
まだ存在しない。

## 現在の証拠の独立判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| 旧モデル復元 | **実用段階、未凍結** | NPUの開発6 seedは全て250手完走し、平均15,333.3。Lux軌跡16 decisionではOpenVINO CPU/NPUとも16/16 argmax一致。ただし1 seed・16手だけで、protocolの正準backendはまだpending。 |
| G2判定器 | **大幅改善、未完成** | 正確な32 test seed、250手上限、主統計、model-only budget、必須episode fieldを検査する。旧汎用比較器はG2認定をしなくなった。一方、実hash照合とexporterがない。 |
| compact learning | **offline signalあり、性能証拠なし** | C10bのlr=3e-4はtop-1 .222、MRR .366、相関 .488。C11aのlr=1e-3は同一split・同一300 updateでtop-1 .370、MRR .527、相関 .657。250手どころか現checkpointのゲーム評価は0本。 |
| Bellman residual gate | **弱い信号、現状不合格** | 180 teacher statesで29 flip、held balanced accuracy .500→.706。未校正gateはseed 5750/100手で4,800、旧モデル5,800に対し-1,000。 |
| architecture / AD | **systems判断のみ完了** | Zygote採用、Enzyme/Reactant棄却は時間対効果上妥当。FiLM推奨はgame strengthではなく、かつ一部timingはteacher生成と競合。 |
| test seed | **未使用を維持** | 現存run・experiment recordにtest scoreはなく、`91001:91032` はprotocolとtest fixtureにのみ現れる。 |
| G1 / G2 | **未達 / 未着手** | 現存する新モデルの18,400以上の適格runも、32 seed paired testもない。 |

## 再現性の評価

良い点は、旧checkpointとteacher datasetにSHA-256が報告され、Julia/Lux/OpenVINOの
version、固定seed、action数、更新数、backend、時間が以前より多く残るようになったこと、
また `scripts/source_fingerprint.jl` がsource treeを内容hash化できることである。
C10bとC11aは同じdataset、split、初期化既定値、300 updatesでlrだけを比較しており、
offline比較自体は素直である。

ただし、現状のlearning ledgerは昇格用来歴として不十分である。

- 全worktreeがGit上untrackedで、commitによる固定点がない。
- C11aのJSONLは `LEARNING_SEED`、state batch、TD batch、validation episode数、
  dataset/checkpoint/source/Manifest SHA-256、実行commandを記録しない。中央の
  `reports/experiment_ledger.md` と `reports/learning.md` はC11a生成前の状態で、最新の
  lr選択結果が中央台帳へ昇格されていない。
- teacher datasetの隣接JSONはcheckpointのpathとbyte数を持つがcheckpoint hash、
  generator/source/Manifest hashを持たない。dataset hashは文章reportにあるが、学習runと
  機械的に結合されていない。
- G2 validatorの `checkpoint_sha256`、`config_sha256`、`source_sha256`、
  `manifest_sha256` は64桁小文字かだけを検査し、対象fileの実hashと照合しない。
  現在の合成testは `aaaa...` 等の架空hashでも `eligible/pass` になることを実際に示す。
- `evaluation_freeze_id` も、baseline/candidateで同じ非空文字列ならよく、凍結registryや
  manifestとの結合がない。

従って、現在の結果は研究ノートとしては追跡可能だが、promotion-gradeの再現性はない。
source fingerprint、protocol、evaluator、checkpoint、datasetを一つのrun manifestへ
実hashで結合し、そのmanifest自体を保存してから次の性能主張を行う必要がある。

## 比較の公平性の評価

### 改善した点

- development / validation / testが明示され、testは候補凍結前に実行しない。
- G2主統計はpaired mean difference一つに固定され、mean/medianのOR判定は解消した。
- model-onlyは同じ候補生成、lookahead 0、1 logical pass/decisionと定義された。
- G2 validatorは正確な32 test seed、重複、短縮episode、欠損時間・call数を拒否する。
- 汎用paired比較scriptは記述統計専用となり、G2成功flagを出さない。

### 最も危険な未解決点（P0）

**評価契約が「正準policyの実体」と「提出JSON」をまだ暗号学的・機械的に結合して
いない。** これは現在最大の危険である。

1. `reports/baseline_conformance.md` は正準policyをLux FP32 CPU、chunk 16、actual-size
   tailと宣言する一方、protocolは
   `canonical_backend = "pending real-state Lux/OpenVINO action-conformance audit"` のままで
   ある。reportと唯一の「immutable contract」が一致していない。
2. protocolの `protocol_version` は1.0.0のままだが、前回監査が記録する旧G2規則から
   primary mean固定などの実質変更が入っている。修正内容は妥当でも、versioned freezeは
   証明されていない。validatorは検証時点の可変なon-disk protocolをhashするだけである。
3. validatorはruntime fieldを非空文字列として要求するだけで、baselineのbackend、dtype、
   batch 16、CPU tail、tie-breakをprotocolの正準値と照合しない。candidate側のruntimeが
   baselineと異なること自体は許されるが、事前凍結値との一致は必要である。
4. 16 decisionのconformanceは有用なsmokeだが、NPU最大Q誤差は4.995e-3、Luxの最小
   top-2 marginは0である。NPUをLuxと一般に同一policyとするには弱い。NPU hybrid自体を
   operational canonical baselineと宣言するか、near-tieを重点化した広いaction監査を
   行う必要がある。
5. G2 schemaへ書き出すexporterがなく、既存baseline、compact、gateのJSONはfield名も
   揃わない。特に既存runはhash、logical/physical call、candidate/evaluator configの
   全てを同時には持たない。
6. validatorはprotocolが要求するmean/median/max/p10/p25/p75/p90、completion rate、
   inference timeの完全な比較表を出さず、paired mean/medianだけでpassを返す。入力時間が
   finiteかは見るが、必須報告を一つの認定成果物に閉じていない。

このP0を解消せずtestを開くと、後からbackend、source、config、policy semanticsのどれを
評価したかを復元できない。test seedが平文であることより、今はこの来歴の穴の方が危険で
ある。

## 実験選別の評価

時間制限、stop reason、失敗結果を台帳に残し、Enzyme、Reactant、NEXT ensemble、旧模倣、
旧value-head TDなどを棄却した判断は概ね健全である。C10の2000-action bugを隠さず、
74-actionで同じoffline指標を再現した点も良い。

一方、現在の二つの学習枝には目的とのずれがある。

### Compact listwise

C11aの37% top-1はrandom 2.55%からの明瞭な改善であり、pipeline健全性の証拠である。
しかしteacher targetは旧モデルの全候補Qなので、学習を延ばして理想的に成功すると旧方策へ
近づく。近似誤差や一般化で偶然旧モデルを上回る可能性はあるが、それを狙う教師信号はない。
したがって、同じold-Q listwiseを3000 updatesへ延長する前にgame gateが必要である。
FiLM ablationを同じ500-state holdoutで追加選別すると、offline holdoutへの適応だけが進む。
seeds 5748–5749は既にlr選択へ使ったdevelopmentであり、独立validationではない。

### Bellman residual gate

Bellman branchは旧Q以外の選好を含む点で目的に近いが、根拠はまだ非常に細い。教師datasetは
3 training seeds×60 statesだけで、教師trajectory自体のpaired baseline優位を記録して
いない。未校正gateはfresh developmentの唯一の100手で-1,000だった。校正threshold
0.27216053はheld episode 60 states上で多数のthresholdから最大proxy gainを選び、選択した
動作は1件、推定mean teacher gainは0.0004305に過ぎない。これは過選択に弱く、性能改善の
証拠ではない。校正版をfresh seedで一度確認する価値だけが残る。

## 今すぐ止める枝と保留する枝

### 終了

- standalone Enzyme、Reactant+EnzymeMLIRの追加benchmark・互換性追跡。
- ConvNeXt、64-channel architectureの追加systems benchmark。game signalが先である。
- E003/E004/E006、旧 `imitation_v1`、旧 `residual_imitation_v1`、同じE002 value-head TD。
- 未校正Bellman gate、および同じ60-state holdoutを使うthreshold/hidden/lr sweep。
- lookahead/ensembleをmodel-only改善として扱うこと。system研究なら別予算でのみ残す。
- corrected dataset上のC02 offline TD開始。旧方策が選んだactionのtransitionしかなく、
  未選択actionを改善する反実仮想信号がない。
- test seedの実行。

### 凍結して短い判定だけ許可

- **C11a**: checkpointを変えず、長い3000-update runとFiLM ablationを止め、まず短い
  paired game screenを行う。通らなければpure old-Q distillationを「超過枝」として終了する。
- **校正済みBellman gate**: threshold 0.27216053を変えず、fresh developmentで一回だけ
  確認する。無flip、score非改善、または大きなregressionならbranchを閉じる。

## 次の短い判断点

### D0 — 計算前のfreeze gate

長い学習の前に、protocolを新versionとして固定し、次を満たすまでperformance runを増やさ
ない。

1. 正準baselineをLux FP32かNPU+CPU-tailのどちらか一つとしてprotocolへ明記する。
2. 同一evaluator/engine hashと、モデル固有checkpoint/config hashを分離してrun manifestへ
   実fileから自動記録する。
3. G2 exporterを一つ作り、既存評価関数からschema fieldを直接生成する。
4. 架空hash、未登録freeze ID、baseline runtime不一致、改変protocolを拒否するnegative testを
   追加する。
5. validator出力にprotocol必須の全score分位点、completion、paired差、論理/物理call、
   inference/wall timeを含める。

これはゲームを必要としない短い判断点である。失敗中はvalidation/testだけでなく、候補の
長時間学習も止める。

### D1 — 3 seed・100手の事前固定paired screen

D0通過後、現在の記録で未使用のdevelopment `5751:5753` を使い、同じ100手上限で
canonical baseline、凍結C11a、凍結済み校正Bellman gateを一度だけpaired評価する。
checkpoint、threshold、NEXT=5、候補順、seed、hash、logical/physical requestsを実行前に
manifestへ固定し、結果を見て再調整しない。

これは統計的主張ではなく枝刈りである。goal trackへ残す最低条件を、全episode完走、
paired平均差>0、paired中央値差>0、3 seed中2 seed以上が非負、とする。Bellman gateはさらに
少なくとも1回の実flipが必要で、無flipのscore tieを成功と数えない。通過候補が0なら両枝を
止め、旧Q模倣の延長ではなく、旧方策を上回るlabel/returnを持つ学習機序を再設計する。

通過候補が1つだけなら、その一つだけを残りのfresh development `5754:5757` の250手へ進め、
そこで事前登録済みBG02相当のmean +500、median positive、70%以上non-negative、completion
非悪化を要求する。その後にarchitecture/checkpointを完全凍結し、validation `8001:8008`
を一回だけ使う。test 32 seedはvalidation通過後も、G2 exporterとmanifestのdry-runが通る
まで実行しない。

## 最終提言

今の研究は、失敗枝を正しく捨て、offline learnerを高速化し、G2判定器の明白な穴をかなり
塞いだ点で前進している。しかし、現在一番良い数字は「旧教師への37%一致」であって、
「旧モデルより強い」証拠ではない。今は学習量を増やす局面ではなく、評価freezeを完成し、
凍結済み二候補をfresh 3 seedで安価に落とす局面である。

**したがって、現状のまま長い学習を進めてはいけない。D0の来歴固定とD1の短いpaired
screenだけを許可し、通過した単一候補にのみ次の計算を配分する。**
