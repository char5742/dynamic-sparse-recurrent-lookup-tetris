# 動的疎再帰ルックアップネットワーク：研究・実験軌跡

最終更新: 2026-07-21

この文書は、CPUネイティブな動的ニューラルネットワークを構築する
研究について、着想、設計判断、失敗、修正、検証結果を時系列で保存する。
成功結果だけでなく、棄却した方式と未解決事項も記録する。

## 1. 研究の出発点

着想は、脳の計算原理に見られる次の性質をニューラルネットワークへ
導入することだった。

- 巨大な記憶容量のうち、その瞬間に必要な一部だけが活動する。
- 同じ回路を、問題の難しさに応じた回数だけ反復使用する。
- 活動した経路だけが学習更新を受ける。
- 現在入力の要素間に、入力固有の関係を形成する。

この構想を計算グラフの三軸で表す。

1. `G_rel(x,t)`: 入力内で誰が誰を見るか。
2. `G_param(x,t)`: どの長期記憶・parameterを使用するか。
3. `G_time(x)`: 何回思考を反復するか。

Transformerは主として1を入力依存にする。提案モデルは、1、2、3をすべて
入力依存にし、さらにforward、backward、optimizerの実計算量まで選択経路へ
限定することを目標とする。

## 2. 中心アーキテクチャ

仮称は **Dynamic Sparse Recurrent Lookup Network (DSRLN)**、日本語では
「動的疎再帰ルックアップネットワーク」とする。

### 長期記憶

LookupFFN bankを、全入力で共有される学習可能な長期記憶として使う。
各入力・各反復は、全bankを密に計算せず、learned routeが選んだ少数行だけを
gatherする。選択されなかった行にはbackwardもoptimizer更新も発生しない。

### 短期作業記憶

盤面cell、candidate、difference、next/hold、`aux37`を位置付きtokenとして
保持する。少数のregisterを再帰的な作業状態とし、入力固有token memoryから
必要な情報を読む。

### 入力内関係

全token間のdense QK行列は作らない。盤面cellは物理的なlocal-8 edgeと
learned sparse edgeで接続し、共有Q/K/V/Oと相対位置biasを用いて更新する。
registerからepisodic memoryへのcross-attentionも、有界候補だけを正確に
score・softmax・gatherする。

### 再帰と思考時間

同じblockを反復間で共有する。hard-halting interfaceは、必要なcandidateだけを
次stepのactive queueへ戻す。今回の20,000更新runでは、表現学習の安定性を
先に確認するため深度を2へ固定した。

### CPU実行

SLIDEの固定LSHやニューロン層は採用しない。参考にするのは、使われない
parameterへ触れず、不規則な疎実行をCPUのwork queue、SIMD、cache、RAMへ
直接対応させるシステム思想だけである。

## 3. 実験軌跡

### Phase A — CPU疎実行とLookupFFN基盤

初期研究では、疎なparameter routing、active-only backward、sparse optimizer、
CPU上のWTA/Lookup実行を構築した。1層、3層、active-width k64/k128/k256、
collision-prefilter、Mongoose系routeなどを段階的に比較した。

この段階で得た重要な知見は次の通り。

- 保存parameter数とactive computeは大きく分離できる。
- FLOP削減だけではwall-clock短縮を保証しない。
- random gather、cache、route union、gradient reduceが実時間を決める。
- hard routeのcollapseと未選択経路への信用割当が主要な学習課題になる。
- 成功runだけでなく、route cap超過や事前登録gate不成立もfail-closedで保存する。

詳細な個別runは
[`experiments/beat_first_v1/EXPERIMENT_LEDGER.md`](experiments/beat_first_v1/EXPERIMENT_LEDGER.md)
に記録している。

### Phase B — LookupFFN再帰とhard halting

TRM、PonderNet、Universal Transformerに対応する要素を検討し、同じLookup blockを
反復再利用する構造へ進んだ。PonderNetのように学習時に全深度を展開すると
CPU動的学習の利点が失われるため、実sampled trajectoryだけを実行・保存する
hard haltingを採用した。

目標計算量は、概念的には次式で表せる。

```text
C(x) ~= T(x) * L * (C_route + 2 * H * K * d)
```

- 容量はtable rows `S` に比例する。
- active computeは選択行数 `K` に比例する。
- 思考時間は入力依存深度 `T(x)` に比例する。

### Phase C — Transformerとの比較から判明した不足

単一carrierと長期Lookup memoryだけでは、現在入力の要素同士を直接結び付ける
機構が不足していた。Transformerの本質は、単にAttention層を持つことではなく、
入力ごとに有効な情報接続グラフを形成できる点にある。

そこで、長期parameter memoryとは別に、入力固有episodic memoryと複数registerを
導入する方針へ移った。

### Phase D — Dense cross-attention案の棄却

初期ViT案では、283 tokenすべてとのdense cross-attentionを検討・実装した。
しかし、全QK scoreを計算した後にtop-k maskを掛ける方式は、論理的に疎でも
物理計算は密であり、CPU特化という研究目的に反するため棄却した。

採用条件を次のように固定した。

- 全token score計算を禁止する。
- dense maskを禁止する。
- learned hash/WTAで有界候補を直接取得する。
- 候補tokenだけを正確にscore・softmax・gatherする。
- 選択edgeだけをbackwardする。
- Lookup rowだけをsparse optimizerで更新する。

### Phase E — barrierless scheduler

Julia学習が約15% CPUしか使っていなかったため、数学モデルを変えず実行schedulerを
改善した。

比較した構成は次の通り。

| 構成 | updates/s | 判定 |
|---|---:|---|
| 現行static | 23.19 | 基準 |
| barrierless、pinningなし、chunk 8 | 27.35 | 当時最速 |
| 全core CPU Sets固定 | 18.71 | 不採用 |
| P-coreのみ8 worker | 26.67 | 次点 |

Windowsでは論理CPU番号を固定せず、必要時にはCPU Sets APIからP/E分類する。
ただし実測ではpinningなしが最速だった。最終schedulerは、全stateのcandidateを
flattenし、20 workerが同じglobal queueからchunk 8で取得する。P/Eへ固定担当数を
与えず、速いworkerが自然に多く処理する。

### Phase F — schedulerだけでは解決しなかったallocation

初期barrierless調整後もspeedupは1.18倍に留まった。

- 134 MB allocation/update
- GC時間 約46.6%
- backward 約61%
- queue wait 約26%

queue単体試験は合格しており、主因はqueue correctnessではなかった。候補backwardの
Dict、一時Vector/Matrix、gradient bufferをworker-local persistent scratchへ移した。

### Phase G — 最初の固定batch過学習失敗

update 1,000 checkpointから同じreal-teacher 4状態を100回反復したところ、
loss低下は約4.04%だけだった。

```text
最初10更新 loss平均  3.8233
最後10更新 loss平均  3.6688
最終loss              3.6416
ListNet               3.8037 -> 3.6496
old-Q                 0.06548 -> 0.06482
margin                0.000894 -> 0.000971
平均再帰深度          2.206 -> 2.192
```

同じ4状態すら十分に記憶できないため、データ多様性や通常lossの揺れでは説明できず、
学習系の構造的失敗と判定した。

### Phase H — 過学習失敗の原因

調査により、構想全体ではなく、実装が「関係形成前にhard routingで情報を捨てる」
構造になっていたことが判明した。

1. cell tokenは独立生成後に固定され、cell同士で局所情報を伝播・更新しなかった。
2. learned Q/K/V/Oが実際の主要経路で使われていなかった。
3. registerは283 tokenからhard routeされた少数tokenだけを読んだ。
4. 相対位置表現がなく、空間relationは選択後の少数token内に限られた。
5. cross-attention residual scaleが`0.03 -> 0.00255`へ縮み、入力経路が消えた。
6. 未選択tokenへ勾配がなく、初期routeの誤りを修復できなかった。
7. Lookup各blockは単一rowを約99.5%選択し、長期記憶がcollapseした。
8. haltingは平均約2.26 stepのまま変化しなかった。

### Phase I — 空間・信用割当の修正

次の修正を行った。

- 各cellをlocal-8近傍へ物理的に接続する。
- cell memory自身を各再帰stepで更新する。
- spatial attentionへ共有learned Q/K/V/Oを導入する。
- 3x3相対位置biasを導入する。
- candidate tokenをcross-attentionの必須supportへ含める。
- cross residual scaleに正の下限を設け、入力経路の消失を防ぐ。
- memory BPTTとtokenization VJPを実装する。
- Lookupはallocation-freeなexact top-3を使用する。
- 表現学習を確認するまでhaltingを深度2へ固定する。

240 cell x 9近傍ならedgeは約2,160であり、dense 57,600 edgeを避けながら
CNN相当の局所伝播と入力依存attentionを両立できる。

### Phase J — 修正後の固定batch gate

同じreal-teacher 4状態に対する過学習gateは、修正後に明確に合格した。

```text
ListNet  3.54684  -> 0.99465
KL       2.56241  -> 0.010223
old-Q    1.88590  -> 0.021857
top-1    0        -> 1
NDCG     0.70592  -> 1
```

これにより、少なくとも現行表現・backward・optimizerがreal teacher信号を
実用的な強さで学習できることを確認した。

### Phase K — serial/barrierless厳密一致

更新2,000 checkpointから、serial oracleと20-worker barrierlessを独立restoreし、
同じ4 training rowで1更新比較した。

| 比較対象 | 最大絶対差 | relative L2 |
|---|---:|---:|
| output | 0 | 0 |
| loss | 0 | 0 |
| raw VJP | 0 | 0 |
| parameter gradient | 1.84774399e-6 | 1.64699601e-6 |
| optimizer後parameter | 1.84401870e-7 | 2.14590685e-9 |

candidate seed、深度、halting、token edge、Lookup row、active mask、sparse event、
optimizer clock、RNG、sampler stateは完全一致した。

この過程で、barrierless postphaseのdense parameter registryから、新規Attention系
14 parameterが漏れていた重大な実装バグも発見・修正した。

### Phase L — postphase allocation修正

正しい34-parameter registryを有効にすると、
`Dict{Symbol,Array{Float32}}`による配列rankの型消去が露呈した。scalar SIMD loopで
boxingが発生し、特にcanonical `cell_bias` replayのBool reductionだけで
約72 MB/updateを割り当てていた。

配列rankを具体化するtyped helperとsingle-pass replayへ変更した。

| 指標 | 修正前 | 修正後 |
|---|---:|---:|
| scheduler throughput | 4.24 updates/s | 19.81 updates/s |
| allocation | 156.6 MB/update | 5.62 MB/update |
| executor allocation | 134.5 MB/update | 3.19 MB/update |
| 測定GC時間 | 有意 | 0 s |

修正後に厳密smokeを再実行し、上記一致を確認した。

### Phase M — 本学習20,000更新

更新2,000 checkpointから、同一model・optimizer・sampler状態をrestoreし、
絶対target 20,000までtraining-onlyで継続した。

```text
追加更新                    18,000
完了時consumed states       80,000
resume segment実時間        862.323 s
resume segment throughput   20.8738 updates/s
candidates/s                3,644.27
recurrent steps/s           7,288.54
最終step composite loss     3.46772
最終step ListNet            3.40899
最終step old-Q              0.186227
最終step margin             0.0546544
```

最終checkpoint:

```text
update  20000
bytes   253663125
sha256  1fc05d63154fc73e5d60367c2b19d63116a975b0a3a772899b7fd0ca382db28e
```

binary checkpointとteacher datasetはGitHubへcommitしない。公開記録にはサイズと
SHA-256だけを残す。

### Phase N — held teacher性能評価とPreAct比較

別途許可された評価として、game validation／sealed seedを開かず、PreActと同じ
real-teacher validation splitから固定128 statesを再構成した。split seedは
`2026071817`、subset seedは`2026072315`、row-list SHA-256は
`fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25`である。

同一12,000 updates／48,000 consumed statesでは次の結果になった。

| model | top-1 | NDCG | pairwise | composite loss | CPU states/s |
|---|---:|---:|---:|---:|---:|
| PreAct best | 0.78906 | 0.99329 | 0.92336 | 2.56378 | 4.19 |
| EVRL update 12,000 | 0.35938 | 0.97127 | 0.82028 | 2.97502 | 53.54 |

EVRL最終update 20,000／80,000 statesはtop-1 `0.37500`、NDCG
`0.95837`、pairwise `0.76881`、composite loss `3.16286`、CPU
`50.19 states/s`だった。PreActとの差はtop-1 `-0.41406`、NDCG
`-0.03493`であり、精度面のPreAct超えは否定された。一方、同じheld panelの
steady-state CPU推論は`11.97x`高速だった。

update 12,000から20,000へ追加学習するとtop-1は`+0.01563`だけ上昇したが、
NDCGは`-0.01290`、pairwiseは`-0.05147`、lossは`+0.18785`悪化した。
固定4-state過学習gateはtrainabilityを示したが、未知stateでのroute discoveryと
全候補順位の信用割当を保証しなかった。20.58M parameters（PreActの13.89倍）を
持つため、単純な容量不足でもない。現時点の結論は「物理的疎実行は高速だが、
current router／recurrent representationのheld ranking品質は不十分」である。

## 4. 現在確定していること

### 確認済み

- spatial episodic + Lookup再帰モデルは固定batchを明確に過学習できる。
- serialとbarrierlessは許容誤差内で同じ数学的更新を行う。
- active-only sparse backwardとsparse optimizer clockは維持されている。
- warm benchmarkと本学習segmentは15 updates/sの受入条件を超える。
- 20,000更新をcheckpoint互換性を保って完走した。
- 同一48,000 states比較ではPreActがtop-1とNDCGで大幅に上回る。
- EVRL最終modelはheld CPU推論でPreActの11.97倍のstates/sを達成する。

### まだ証明していない

- 同一wall-clockまたはhardware-counterで正規化した計算予算での比較。
- learned hard haltingを再導入した場合の精度・速度Pareto改善。
- route collapseを抑えながら長期bank容量を十分利用できる最終設定。
- game validationおよびsealed seedでの強さ。

validation `8001:8008` とsealed seed `91001:91032`は未使用のままである。

## 5. 次の研究段階

1. held gapの原因をroute recall、token-edge credit、long-memory row利用率へ分解する。
2. 同一wall-clock／hardware-counter予算でPreActと比較する。
3. routing entropy、unique row利用率、popular-row集中を測る。
4. held rankingを回復するまではhard haltingを再導入しない。
5. input-dependent relation、parameter path、thinking timeの三軸が、実計算量でも
   動的であることをablationで示す。
6. checkpointを変更せず、CPU wall-clock、RAM、allocation、GC、tail latencyを
   Pareto評価する。

## 6. 実装・記録への入口

- 現行モデル:
  [`experiments/beat_first_v1/episodic_vit_recurrent_lookup`](experiments/beat_first_v1/episodic_vit_recurrent_lookup)
- 数値結果:
  [`RESULTS_2026-07-20.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/RESULTS_2026-07-20.md)
- PreAct性能比較:
  [`PERFORMANCE_COMPARISON_2026-07-20.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/PERFORMANCE_COMPARISON_2026-07-20.md)
- 全実験ledger:
  [`EXPERIMENT_LEDGER.md`](experiments/beat_first_v1/EXPERIMENT_LEDGER.md)
- 旧DSRLN:
  [`dynamic_sparse_recurrent_lookup`](experiments/beat_first_v1/dynamic_sparse_recurrent_lookup)
- 旧Lookup/SLIDE CPU基盤:
  [`residual_lookup_slide`](experiments/beat_first_v1/residual_lookup_slide)
- 3層疎modelとrouting派生:
  [`sparse_dynamic_3layer`](experiments/beat_first_v1/sparse_dynamic_3layer)
- barrierless executor:
  [`barrierless_executor.jl`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/barrierless_executor.jl)
- correctness smoke:
  [`barrierless_correctness_smoke.jl`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/barrierless_correctness_smoke.jl)

## 15. 2026-07-21 — EVRL input-token routing ablation

The register-to-episodic-memory router was removed.  The old path retrieved 64
of 283 tokens and retained an exact top-16 shortlist; the replacement projects
all 283 tokens once per recurrent step and lets all four registers perform
exact cross-attention over the shared K/V memory.  LookupFFN parameter routing,
local-8 spatial attention, recurrence, loss, optimizer, input contract, and
candidate-independent evaluation were unchanged.

Serial/barrierless real-teacher correctness passed.  At the equal
12,000-update / 48,000-state budget, held-panel top-1 improved from `0.35938`
to `0.56250` and NDCG from `0.97127` to `0.97867`.  CPU inference changed from
`53.54` to `45.17` states/s; training completed at `23.37 updates/s`.  The
full-token model remains below PreAct (`0.78906` top-1), but is `10.77x` faster
on the same 128-state CPU panel.  Exact conditions and hashes are recorded in
[`experiments/beat_first_v1/episodic_vit_recurrent_lookup/TOKEN_ROUTING_ABLATION_2026-07-21.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/TOKEN_ROUTING_ABLATION_2026-07-21.md).

## 16. 2026-07-21 — 全盤面受容野を持つ軽量visual path

単一3x3 depthwise filterでは受容野が局所一点に留まるため不採用とした。
raw board/candidate/differenceの3 channelへdilation `1,2,4,8,16`の
depthwise 3x3、SiLU、3x3 channel pointwise residualを5段適用し、最後に
3 -> 128 pointwise projectionを既存cell tokenへ加える構成へ変更した。
理論受容野は`63 x 63`で24x10盤面全体を包含し、左上入力から右下tokenへの
非zero応答も実測した。

追加costは565 parameters、135,360 scalar MAC/candidateである。serial対20-worker
barrierless smokeはoutput/loss/raw VJP完全一致、worker gradient最大差
`5.62e-6`、optimizer後state最大差`2.09e-7`で合格した。100-update warm
benchmarkは`31.58 updates/s`、candidate CPU `76.98%`だった。

12,000 updatesではheld top-1 `0.50000`、NDCG `0.97936`で、旧full-token
12kのtop-1 `0.56250`を下回るmixed resultだった。しかし同じmodelを100,000
teacher statesに相当する25,000 updatesまで継続すると、top-1 `0.68750`、
NDCG `0.98586`、pairwise `0.87751`、margin `0.13215`、loss `2.66337`へ改善した。
したがって12k時点は頭打ちではなかった。一方、no-visual 25k対照をまだ実行して
いないため、この改善全量をvisual pathの効果とは主張しない。PreActとの差は
top-1 `-0.10156`、NDCG `-0.00743`で残るが、held CPU推論は`10.43x`速い。

詳細とcheckpoint hashは
[`GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/GLOBAL_VISUAL_RECEPTIVE_FIELD_2026-07-21.md)
に記録した。

## 17. 2026-07-21 — 動的再帰の再導入試験

固定深度2で学習したupdate 80,000 checkpointから、(1) hard haltingの直接導入、
(2) 深度2–6を5,000更新学習してからの導入、(3) halt headだけの初期化を比較した。
さらに同じ最終architectureを、最初の5,000更新から深度2–6で学習する
from-scratch動的modelも20,000更新まで実行した。

直接導入はupdate 90,000でtop-1 `0.72656`、NDCG `0.99155`、深度curriculum版は
top-1 `0.71875`、NDCG `0.99161`を得た。from-scratch版もtop-1を初期
`0.21875`から`0.53906`へ改善した。しかし全方式でheld deterministic depthは
全candidate一律2だった。halt headをprobability 0.4へ初期化すると一度は最大深度
12へ崩れ、学習後は再び約2.4 stepへ戻った。

したがって「途中導入だけ」が原因ではない。現行REINFORCEはstate-wide ranking lossを
全candidate trajectoryへ同じ値で与えるため、candidate固有の追加1stepの価値を
識別できない。prediction最終層を消すより、少数candidateのone-step probeで
`loss(t) - loss(t+1)`を直接教えるhalting信用割当が次の最小修正である。

全条件、速度、checkpoint witnessは
[`DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/DYNAMIC_RECURRENCE_ACTIVATION_2026-07-21.md)
に記録した。
