# F: 旧 20.8M 全モデル継続学習 feasibility

日付: 2026-07-18 JST

source commit: `c7b2cf28a07b92504ff8827799f01034419295f7`

判定: **F-infeasible（事前登録した一回限りの gate として）**

## 結論

唯一許可された Lux 1.31.4 + Zygote 0.7.11 の F 実行は、六 warm update、更新重みの export、OpenVINO CPU/NPU 同等性へ到達せず終了した。したがって F の全成功条件は満たさず、A の実行を許可する証拠にはしない。global one-shot marker は作成済みで、この条件の救済再実行は行わない。

ただし、この停止を「Zygote が 20.8M parameter では遅すぎた」または「実パラメータの勾配を生成できなかった」と解釈してはいけない。optimizer setup と、最初の specialization probe complete-update の合計は `123.876928 s` で、300 秒制限内だった。停止したのはその返り値を走査する独自の勾配木完全性検査である。

## 観測された進行

| 段階 | 結果 |
|---|---|
| freeze / hash / native argv | pass |
| training-only 6 rows の抽出 | pass |
| checkpoint load / 20,787,454 parameters | pass |
| zero-update old-Q 照合 | pass した後に次段階へ進行。ただし partial JSON の上書きにより個別誤差は最終 artifact に残らなかった |
| optimizer state | 事前検査上、全 parameter 数の 2 倍の array elements を持つ状態を構築 |
| optimizer setup + 最初の complete-update probe | 合計 `123.876928 s` で返却、300 秒上限内 |
| 勾配走査 | `specialization gradient has a missing/nothing/mismatched parameter leaf` で停止 |
| 六 warm update | 未実行 |
| NPZ export / OpenVINO refresh | 未実行 |
| game / score / validation / test | 未実行 |

全 wrapper wall は `161.0591958 s`、Julia phase は `160.0602948 s` だった。temporary weight は生成・昇格されていない。

## 勾配木エラーの read-only 診断

実行後に AD や model forward を再実行せず、固定 source graph と検査関数だけを監査した。

`LegacyQNetwork` の parameter tree には、parameter を持たない pooling / activation layer、`NoOpLayer` である `board_encoder`、`ren_encoder`、`btb_encoder`、`tspin_encoder`、`LegacyPositionalEncoding` が空の `NamedTuple()` として含まれる。一方 `gradient_covers_parameters` は parameter 側が空の `NamedTuple()` でも、gradient 側が `NamedTuple` であることを要求する。Zygote が未使用・空 subtree を `nothing` と返す場合、この検査は trainable array が一つも欠けていなくても false になる。

checkpoint の parameter tree だけを read-only 走査したとき、最初の空 subtree は `parameters.board_net.resblocks.layer_1.layer_5.layer_1` だった。source graph 上、これは最初の residual block の squeeze-excitation gate 内にある 0-parameter pooling layerに対応し、validator false positive が静的に成立する一つの plausible candidate である。停止した gradient object 自体は保存されなかったため、実際の first failing path と actual array-leaf coverage は特定不能である。したがって今回の error は、実 trainable leaf の欠落を証明しない。

また LuxLib は `Lux.testmode` の state を AD 内で使ったことについて警告を出した。F の契約は BatchNorm running state の固定を要求していたため test mode 自体は意図的だが、将来別契約を設計する場合は、state 固定と勾配数値一致を独立に検証する必要がある。

## artifact accounting の注意

`final_result.json` の failure list にある次の記述は、観測事実ではなく、途中終了時の必須 field 欠落を一律 failure 文へ変換した結果である。

- `zero-update tolerance failed`: control flow 上は zero-update gate を pass している。正確には「最終 JSON に誤差値が残っていない」。
- `first specialization exceeded 300 seconds`: 実測は `123.876928 s` で上限内。正確には「`specialization_passed` record がない」。

同様に `monitor.json` の peak working set `49,520,640 bytes` は `Start-Process` が返した直接 PID だけの値で、process tree を集計していない。WindowsApps alias 経由の実行であることも踏まえると、この値では全 Julia process tree の 8 GiB memory gate を certify できない。

## 判定と意味

- **F の形式判定:** infeasible。六 update、warm median、正しい peak memory、export、CPU/NPU 同等性、`T1000` が欠落したため。
- **Zygote の速度判定:** optimizer setup + 最初の complete-update probe は合計 123.9 秒。specialization の即時停止上限 300 秒は通過したが、各内訳と六 warm update がないため定常速度は不明。
- **Zygote の勾配判定:** inconclusive。actual first failing path と array-leaf coverage は保存・特定されていない。0-parameter subtree による validator false positive は静的に成立する候補の一つである。
- **Enzyme との比較:** この F は Zygote 単独 gate であり、Enzyme/Reactant は実行していない。E021 の compact model 結果を legacy 20.8M model へ外挿しない。
- **次の権限:** F 失敗時の事前指示どおり、A、Reactant legacy learner、別 architecture、救済 F を自動実行しない。次の実験はクリーン再監査で別契約として選ぶ。

## Provenance

- output directory: `D:\tetris-paper-plus\runs\legacy_full_feasibility_F_c7b2cf2`
- global one-shot marker: `D:\tetris-paper-plus\runs\legacy_full_feasibility_F.started.json`
- source fingerprint SHA-256: `c5f379ff33e1a2c7d97c98745eb92181de5c53136970d0e30c046cbc74a97754`
- harness SHA-256: `366756d0c7d4a74d5140de1c537b8e5622de35049939a64de1f82f979fa04b58`
- `freeze.json`: `6aceb3861fda1ee827c90c8e9323cb892490abf3648ae22cbf800714aa60c51d`
- `final_result.json`: `90000b364c90347069a2aefe08b5fbad5b4bd5d828add0289ca63b9f1cb17e18`
- `assessment.json`: `ade3eef329e2e3bad9409cbb904897674ccc5911ac72ec6abc65cd2ebe17bb9a`
- `monitor.json`: `b90ac5ef7177483891ac3692830df6a7103e41289927875749b4dd590d28af63`
- `julia_phase.json`: `e1dbd5c715c1c5369629983f79f8d0be27e0e861d38ed90751b8cc4b75258852`
- `julia.stderr.log`: `415164da1cf84e374bbeb23cc9a3992af88d50d9d956b427dc2a00df603b3eeb`
- checkpoint SHA-256: `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`
- dataset SHA-256: `e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d`
