# Serial Workspace SNN — beat-first第3モデル

PreAct-ECA、Dynamic Sparse Recurrent Lookup Network (DSRLN/EVRL) に続く、
テトリス実teacher候補順位付け用の第3モデルである。8ノードの
`serial_synaptic_printer_poc/v2_minimal_cognitive_graph`を、既存2モデルと同じ
候補入力・22出力・教師損失へ接続した。

この実験の主張は「8機構を実装した学習可能な第3モデル」であり、AGI、ゲーム強度、
PreAct超えを事前に主張するものではない。評価はreal-teacherのtraining splitだけを
使い、validation row、game validation、sealed seedを使わない。

## scaled構成

| 項目 | 値 |
|---|---:|
| 連続状態ノード | 4,608 |
| block | 96 |
| node / block | 48 |
| 候補シナプス | 110,592 |
| fanout / node | 24 |
| 初期ON密度 | 約50% |
| global workspace容量 | 8 block |
| 周回数 | 4 |
| 0/1 sensory rail | 1,298 |
| 学習parameter | 453,815 |
| 出力 | Q 1 + death 1 + quantile 16 + geometry 4 |

盤面240 bit、候補盤面240 bit、差分の正負dual rail 480 bit、NEXT/HOLD 42 bit、
37補助特徴の8段thermometer 296 bitを連結する。したがってモデル境界へ入る値は
すべて厳密な0/1である。補助特徴の精度は8段へ量子化される。

## 8機構の対応

| 必須機構 | 実装 |
|---|---|
| グラフ構造＝意味・知識 | 固定edge identityと、学習される各edgeのON/OFF・重み・遅延 |
| 現在の発火経路＝思考 | 各cycleの`(source, destination, relation)`列をtraceへ保存 |
| ノード内部状態＝連続的な概念状態 | 各LIF-like nodeの`Float32` membrane |
| active block＝global workspace | content WTAで96 blockから毎cycle 8 blockを選択 |
| 発火先の選択＝attentionの代替 | token間softmax attentionを使わず、query・membrane・keyによるhard top-k routing |
| シナプスON/OFF変更＝構造的学習 | gateのSTE学習と、重み×gate evidenceによる周期的な離散consolidation |
| 重み・遅延変更＝連続的学習 | 各edgeのweightと0～1 cycleの連続delayをAdamWで更新 |
| 周回走査＝時間発展 | edge tape全体を4周し、leak・spike・reset・遅延配送・workspace feedbackを更新 |

通常学習は、inboxへ加算してからnodeを更新するという同じ意味を保ったままedgeを
まとめて走査する。監査用`serial_synapse_scan`は全候補シナプスを1本ずつ順に検査する。
テストは両実装の数値一致を確認する。

## ファイル

- `SerialWorkspaceSNN.jl`：モデル、0/1 encoder、逐次／学習用走査、構造固定化、思考trace
- `train_teacher.jl`：teacher_v3読込、共通損失、AdamW、training-only評価、checkpoint
- `ArenaWorkspaceTraining.jl`：固定candidate arena、解析VJP、worker-local勾配、
  isbits MPMC queue、parallel in-place AdamW、Windows CPU Set固定
- `train_arena_100k.jl`：GC-free hot loop、10k刻みcheckpoint、SHA検証付きresume、
  100,000更新production driver
- `benchmark_arena.jl` / `profile_arena_allocations.jl`：CPU・phase・allocation・GC実測
- `test_arena_training.jl` / `test_arena_real_batch.jl`：Zygote参照との
  loss・raw VJP・全parameter勾配・AdamW更新一致
- `evaluate_training_panel.jl`：既存2モデルと同じ固定128状態panelの再評価
- `runtests.jl`：入出力契約、8機構の因果性、勾配、逐次一致
- `RESULTS_2026-07-27.md`：scaled初回300更新の条件、数値、比較境界
- `BARRIERLESS_ARENA_PERFORMANCE_2026-07-27.md`：GC排除、CPU tuning、
  1,000更新soak、100k実行条件
- `trained/<run-id>/results.json`：前後評価、連続／構造学習witness、速度、artifact hash
- `trained/<run-id>/checkpoint_final.jld2`：parameter、optimizer、sampler、設定

## 実行

```powershell
julia --project=. experiments/beat_first_v1/serial_workspace_snn/runtests.jl
```

```powershell
$env:SWSNN_PRESET = "scaled"
$env:SWSNN_MAX_UPDATES = "300"
$env:SWSNN_EVAL_STATES = "32"
$env:SWSNN_STRUCTURAL_INTERVAL = "25"
$env:SWSNN_RUN_ID = "scaled_teacher_u300"
julia --project=. experiments/beat_first_v1/serial_workspace_snn/train_teacher.jl
```

既存2モデルと同じtraining-only固定128状態panelで比較する場合：

```powershell
$env:SWSNN_RUN_DIR = "experiments/beat_first_v1/serial_workspace_snn/trained/scaled_teacher_u300"
julia --project=. experiments/beat_first_v1/serial_workspace_snn/evaluate_training_panel.jl
```

GC-free barrierless arenaで100,000更新する場合：

```powershell
$env:SWSNN_MAX_UPDATES = "100000"
$env:SWSNN_STATE_BATCH = "8"
$env:SWSNN_ACTIVE_WORKERS = "20"
$env:SWSNN_CPUSET_MODE = "all"
$env:SWSNN_CHECKPOINT_INTERVAL = "10000"
$env:SWSNN_RUN_ID = "arena_scaled_u100000"
julia --threads=20,0 --project=. experiments/beat_first_v1/serial_workspace_snn/train_arena_100k.jl
```

`SWSNN_RESUME_CHECKPOINT`と`SWSNN_RESUME_SHA256`を同時に指定すると、
parameter、AdamW moment/clock、sampler permutation/RNG、構造学習状態、累積性能計数を
検証して再開する。`SWSNN_MAX_HOT_ALLOCATION_BYTES=0`では、1更新でもhot loopに
allocationが戻れば学習を停止する。

`SWSNN_PRESET`は`tiny`、`small`、`scaled`、`large`を受け付ける。`large`は
8,192 node、262,144候補シナプスであり、まず`scaled`の収束とCPU効率を確認してから
長時間学習へ進むための構成である。

## 検証済み境界

- 0/1 rail、22出力、有限値
- capacity固定のhard global workspace
- 入力依存の発火経路
- 非二値の連続membrane
- gate全OFF ablationで出力が変化
- weight、delay、gateの非ゼロend-to-end gradient
- ONとOFFを同時に変更する構造consolidation
- 1周と4周で出力が変化
- 逐次edge scanとベクトル化edge scanの一致
- 実teacherの読込、更新、評価、checkpoint再現情報の保存
- real-teacher scaledでarenaとZygoteの全parameter勾配差 `1.049042e-5`以下
- 100% allocation samplingでhot update `0 allocation / 0 byte`
- 1,000更新soakでhot allocation `0 byte`、hot GC `0.0秒`

結果の解釈では、training-only固定panelの改善を未見データ汎化と呼ばない。また、
parameter数だけでPreActやDSRLNと同等とはみなさず、teacher state予算とpanelを揃えた
比較を別々に報告する。
