# Serial Synaptic Printer PoC

CPUを走査ヘッド、シナプス列を紙面、0/1スパイクをインクとして扱う
最小の学習済みSNN概念実証です。

> 8項目（連続状態、global workspace、attention代替、構造・重み・delay学習、
> 再帰時間発展を含む）の最小実装は
> [`v2_minimal_cognitive_graph`](v2_minimal_cognitive_graph/README.md)です。
> 以下は最初のone-shot associative memoryであるv1の説明です。

## 検証すること

- 入力と出力は0/1だけである。
- CPUは行列積を使わず、シナプス配列を先頭から1本ずつ精査する。
- 教師表からグラフ構造を形成し、学習後に望む回答を返す。
- 学習済みグラフを保存し、別プロセスで再読込して同じ回答を返す。
- 推論時には浮動小数点演算や誤差逆伝播を使わない。

課題は3入力2出力の1-bit full adderです。

```text
input:  a, b, carry-in
answer: sum, carry-out
```

全8通りの教師回答は[`training_data.tsv`](training_data.tsv)にあり、
ネットワーク実装は加算規則を知りません。

## ネットワーク

ゼロを「イベントなし」にしないため、各ビットをdual railで表します。

```text
input bit 0 -> input[i]=0 neuron spikes
input bit 1 -> input[i]=1 neuron spikes
```

一回の教師例からexact-match prototype neuronを一つ生成します。
prototypeには、現在発火している入力railからだけunit-weight synapseを
接続します。閾値は入力ビット数なので、全ビットが一致するときだけ発火します。

続いてprototypeから、教師が指定した各出力railへunit-weight synapseを
接続します。

```text
dual-rail inputs
        |
        | serial scan pass 1
        v
exact-match prototype
        |
        | serial scan pass 2
        v
dual-rail answers
```

知識は別のlookup tableではなく、生成されたノードとシナプスそのものに
保存されます。推論は二つのphase barrierを持つため、各phase内で
シナプス順を逆転しても回答は変わりません。

## 実行

リポジトリルートから学習、保存、全問検証を実行します。

```powershell
julia --project=. experiments/serial_synaptic_printer_poc/train_full_adder.jl
```

学習済みモデルへ個別に質問します。

```powershell
julia --project=. experiments/serial_synaptic_printer_poc/demo.jl 1 0 1
```

この例の期待回答は`sum=0, carry-out=1`、すなわち`01`です。

テストを実行します。

```powershell
julia --project=. experiments/serial_synaptic_printer_poc/runtests.jl
```

学習スクリプトは次を生成します。

- `trained/full_adder_snn.ssg`: 決定論的なテキスト形式の学習済みSNN
- `trained/full_adder_edges.tsv`: 人間が確認できるシナプス列

## 検証結果

Julia 1.12.6で実行した結果です。

```text
学習前             0/8回答
one-shot学習       8 prototype生成
学習後             8/8正解
graph              18 nodes / 40 synapses
一問の逐次精査     80 edge inspections
実際のevent配送    14 deliveries
保存・再読込       8/8正解
```

`101`への別プロセス推論は、期待どおり`01`
（`sum=0, carry-out=1`）を返しました。

学習済みモデルは同じ入力からbyte単位で再生成でき、SHA-256は
`e1bc3efe4620898427b0abe0b981860b3cefe13a98e2d3d30e56af9543117b1b`
です。テストはfull-adder全問、別規則のXNOR、edge順反転、
決定論的保存、再読込、矛盾教師の拒否を検証します。

## このPoCがまだ証明しないこと

これはone-shot associative memoryの概念実証です。教師表を正確に記憶しますが、
未提示入力への一般化、概念の共有、長期的信用割当はまだありません。
prototype数とシナプス数も記憶したパターン数に比例します。

また、意図した「全シナプス逐次精査」をそのまま実装しているため、一問ごとに
全edge tapeを二回走査します。次段階では、

1. active blockだけを走査するタイル化
2. prototype間で共有できる部分概念
3. 連続的なnode stateとdelta event
4. third-factorによる局所的な重み・構造更新
5. 未知タスクを含むスケーリング試験

を追加する必要があります。
