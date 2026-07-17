# Baseline conformance and G2 gate

監査日: 2026-07-17

## 結論

旧モデルの正準方策を **Lux FP32 CPU、stable candidate order、16候補chunk、
末尾は実候補数のshort batch** と固定した。開発seed 5742の正準Lux軌跡16 decisionで、
OpenVINO CPUと「NPUの完全16batch + CPUのshort tail」はともに **16/16で同じargmax**
を選んだ。従って、この監査範囲ではNPU経路を旧baselineの高速実装として扱える。

ただし、これは1開発seed・16手のconformanceであり、全250手・全seedに対する数学的な
同値証明ではない。NPU量子化誤差がtop-2 marginを上回る局面は今後もあり得るため、
baseline本評価では候補hash、選択hash、backendをrunへ必ず保存する。

## 実測

成果物: `artifacts/baseline_conformance/seed5742.json`

| 項目 | 結果 |
|---|---:|
| 開発seed | 5742 |
| 正準軌跡decision | 16 |
| 候補数範囲 | 18–57 |
| 完全16batch | 全decisionで実行 |
| short CPU tail | 全decisionで実行（tail 2–13） |
| OpenVINO CPU argmax一致 | 16/16 |
| OpenVINO NPU+CPU tail argmax一致 | 16/16 |
| CPU vs Lux 全Q最大絶対誤差 | 8.106232e-6 |
| NPU vs Lux 全Q最大絶対誤差 | 4.995346e-3 |
| Luxの最小top-2 margin | 0.0 |

decision 12はLuxのtop-2が同値（Float32差0）だったが、stable candidate orderの
first-argmax規則により3経路とも候補index 3を選んだ。各decisionについて、state
SHA-256、順序付きcandidate SHA-256列、全Q値とQ-vector SHA-256、3経路のargmax、
top-2 margin、選択候補hashをJSONへ保存した。

この実行中は別processがNPUでteacher生成を行っていた。従ってJSON内の推論時間は
**contaminatedであり速度比較には使用不可**。数値・argmax監査だけを採用する。

再実行:

```powershell
$env:CONFORMANCE_DECISIONS='16'
$env:CONFORMANCE_SEED='5742'
julia --project=. --threads=20 scripts\audit_baseline_conformance.jl
```

## G2 validator

`scripts/validate_g2_submission.jl` をG2の唯一の認定入口として追加した。次を全て満たさない
submissionは `eligibility="ineligible"` となり、`g2_decision` 自体を出力しない。

- protocol上の正確なsealed test 32 seedだけを1回ずつ含む
- episode limit 250、NEXT=5、HOLD、stable candidate order
- lookahead 0、全候補への論理network passは1 decisionあたり1回
- baseline/candidateの凍結budgetと事前指定primary statisticが一致
- checkpoint/config/source/Manifestの64桁SHA-256とprotocol実体のSHA-256
- backend、dtype、batch size、tail backend、tie-break
- 各episodeの候補評価数、論理／物理network calls、generation/inference/wall time
- 250手未満なら `game_over=true`、論理call数は実decision数と一致

契約を通過した場合のみ、事前固定したmeanまたはmedianのpaired bootstrap 95% CIを計算し、
下限が0より大きいときだけ `g2_decision="pass"` を返す。mean/medianのOR判定は行わない。

汎用の `scripts/compare_paired_evaluations.jl` から危険な
`g2_location_success` を削除した。現在は記述統計だけを返し、G2を認定しないことを明記する。

合成fixtureのみのテスト結果:

```text
candidate simulation preserves root RNG and replay: 6/6 pass
G2 validator rejects incomplete and development-seed submissions: 8/8 pass
```

testでは実ゲームを起動しておらず、sealed test seedのscore評価は0件のままである。

## 残る不足

- G2用のbaseline/candidate exporterは新schemaに合わせて別途実装が必要。
- 現在のcheckpoint/config/source/Manifest hashは形式と存在を要求するが、source hashを
  どのtree範囲から作るかはfreeze時に一つへ固定する必要がある。
- NPU単独の正確なlatencyはteacher生成終了後に短い独占benchmarkで再測定する。
- 最終G2前には、凍結candidateと同じevaluatorでbaselineも32 seed分生成する必要がある。
  validatorはその時点まで成功判定を出さない。
