# Minimal Cognitive Graph v2

前段のone-shot prototype PoCを置き換えるのではなく、その上に構築した
8項目完全実装用の最小モデルです。

課題は次の文脈依存bit学習です。

```text
answer = context XOR bit
```

入力、内部イベント、回答は0/1です。学習対象は4つの組合せだけです。

## 制約下での最小サイズ

論理ゼロも明示的なイベントにするdual-rail制約の下では、

- context rail: 2 nodes
- data rail: 2 nodes
- continuous global-workspace state: 2 nodes
- answer rail: 2 nodes

の合計8 nodesが必要です。

シナプスは、

- contextからworkspace: 2
- workspaceの再帰保持: 2
- 各`(context, bit)`のplastic mapping slot: 4

の合計8 synapsesです。追加のhidden layer、prototype table、dense matrixは
ありません。学習する連続値は4 weightと4 delayの8値です。

## 8項目の実装

| 要求 | v2での実装 |
|---|---|
| グラフ構造＝意味・知識 | 各memory block内のenabled mapping edgeと接続先がbit規則を保持 |
| 現在の発火経路＝思考 | context→workspace→recurrent workspace→選択memory→answerをtrace化 |
| ノード内部状態＝連続的概念状態 | workspace nodeがFloat32状態をleak・再帰入力で時間発展 |
| active block＝global workspace | capacity 1のworkspaceがmemory blockを一つだけadmit |
| 発火先選択＝attention代替 | workspace状態とblock keyのtop-1 content scoreで動的選択 |
| シナプスON/OFF＝構造的学習 | dormant mapping slotをON、教師訂正時はOFF→retarget→ON |
| weight・delay＝連続的学習 | local eligibilityとteacher third factorで両方をFloat32更新 |
| 周回走査＝時間発展 | 全edge tapeを毎cycle逐次走査し、delay queueと再帰で回答まで周回 |

テストでは存在確認だけでなく、learned edgeの接続先変更、recurrent edgeの無効化、
context切替などの因果的ablationを行います。

## 実行

リポジトリルートから実行します。

```powershell
julia --project=. experiments/serial_synaptic_printer_poc/v2_minimal_cognitive_graph/train.jl
julia --project=. experiments/serial_synaptic_printer_poc/v2_minimal_cognitive_graph/runtests.jl
```

学習済みモデルへ質問します。

```powershell
julia --project=. experiments/serial_synaptic_printer_poc/v2_minimal_cognitive_graph/demo.jl 1 0
```

`1 XOR 0`なので期待回答は`1`です。

学習処理は次を生成します。

- `trained/minimal_cognitive_graph.mcg`
- `trained/training_trace.tsv`
- `trained/inference_trace.tsv`

## 検証結果

Julia 1.12.6で4 epoch学習した結果です。

```text
学習前                 0/4回答
学習後                 4/4正解
model                   8 nodes / 5 blocks / 8 synapses
plastic state           4 structural gates / 4 weights / 4 delays
structural learning     4 OFF→ON
continuous updates      32
回答まで                全問5 cycles
一問のedge精査          5 cycles × 8 edges = 40
```

各plastic edgeは最終的に`weight=0.93185127`、
`delay=1.18173`を獲得しました。`context=1, bit=0`では、

```text
context1_to_workspace1
→ workspace1_recurrent
→ memory1_bit0_mapping
→ answer=1
```

を通り、workspace状態は`1.0 → 0.75 → 0.72499996 → 0.72249997`
と連続値で時間発展します。recurrent edgeを無効化するとcycle 3で
contextがattention閾値を下回り、回答不能になることもテストしています。

8項目の機構テスト、因果的ablation、XNOR再学習、決定論的保存を含む
全71テストが合格しています。学習済みモデルはbyte単位で再生成でき、
SHA-256は次です。

```text
e8d92b51912648be8c864ea1fd414206c63be7cfc978c3c02c8a7f4429c75bc9
```

## 完全実装という表現の範囲

ここでの「完全」は、上記8機構がすべて実行系に存在し、bit課題の正答経路で
実際に作動し、個別の因果的テストを通るという意味です。AGI、自然言語の意味理解、
大規模スケーリングを達成したという意味ではありません。
