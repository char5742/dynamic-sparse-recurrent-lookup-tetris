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

## 18. 2026-07-22 — 固定architectureの100k hyperparameter tuning

architecture・入力・教師・loss構成・sampler seed・held panelを固定し、scalar
optimizer／weight decay／routing schedule／halting係数だけを一軸ずつ変更する
100,000-update tuningを開始した。各試行の終了後に結果を記録・pushしてから次を
開始する。

Trial 1として既存固定深度baselineを80,000から100,000へ完走した。最終heldは
loss `2.581711`、top-1 `0.71875`、NDCG `0.991801`、pairwise `0.906227`、
margin `0.146405`。top-1最高値は85,000時点の`0.734375`だった。80k以降の
学習部は`626.58 s`、`31.92 updates/s`である。

全試行ledgerとcheckpoint witnessは
[`HYPERPARAMETER_TUNING_2026-07-22.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/HYPERPARAMETER_TUNING_2026-07-22.md)
へ逐次追記する。

Trial 2ではattention／FFN／token／register／head LRだけを`2e-4`から
`1.5e-4`へ下げ、同一seedでfrom-scratch 100,000更新を実施した。最終heldは
loss `2.609035`、top-1 `0.69531`、NDCG `0.989974`、pairwise `0.896181`、
margin `0.123212`で、Trial 1より全指標が悪化した。低LRでもtop-1振動は残り、
単純なdense更新振幅が主因ではなかったため不採用とした。

Trial 3ではbaseline LRを復元し、dense weight decayだけを`1e-4`から`3e-4`へ
変更した。100,000更新の最終held top-1は`0.8046875`となり、Trial 1を
`+0.0859375`、同panelのPreAct `0.7890625`を`+0.015625`上回った。一方で
loss `2.607862`、NDCG `0.990137`、pairwise `0.898083`はTrial 1より悪く、
top-1に限定された改善である。tuningに使用したheld panel上の結果であり、sealed
generalizationの証明ではない。game validation／sealed seedは未使用である。

Trial 4ではdense weight decayを中間の`2e-4`にして100,000更新した。最終heldは
loss `2.590257`、top-1 `0.75781`、NDCG `0.991009`、pairwise `0.906449`、
margin `0.142959`。Trial 1よりtop-1／pairwiseは高いがloss／NDCG／marginは低く、
Trial 3のtop-1にも届かなかった。結果として、`3e-4`がheld top-1 winner、
`1e-4` baselineがcontinuous-ranking winner、`2e-4`が中間Pareto armとなった。

## 19. 2026-07-22 — 動的再帰tuningへの軌道修正

直前のTrials 1--4はすべてdepth 2固定であり、optimizer／weight decay対照としては
有効だが、依頼された動的リカレント調整ではなかった。このscope誤りを明示的に訂正し、
from-scratch sampled hard halting、最初の5,000更新をdepth 2--6 curriculumとする
100,000-update試行へ戻した。

Recurrence Trial R1では、他条件を固定してcompute priceだけを`0.02`から`0`へ変更した。
最終heldはloss `2.610439`、top-1 `0.703125`、NDCG `0.989804`、pairwise
`0.899255`、margin `0.142586`。実時間`4,035.57 s`、`24.78 updates/s`、CPU平均
`78.92%`、candidate中`81.85%`だった。

しかし深度は安定しなかった。held meanは15kで`5.19`、20kで`2.03`、60--70kで
ほぼ`12`、75kで`2.05`、100kで再び`11.91`となった。training sampled depthも
同じ両端飽和を示した。したがってcompute priceを外すだけでは入力依存の思考時間を
獲得できず、halt policyの更新幅または分散が支配的である。R1は不採用とし、次試行は
他条件を固定したままhalt LRだけを`5e-5`から`1e-5`へ下げる。

曲線、比較、checkpoint witnessは
[`DYNAMIC_RECURRENCE_TUNING_2026-07-22.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/DYNAMIC_RECURRENCE_TUNING_2026-07-22.md)
に記録した。

## 20. 2026-07-22 — candidate-local 1-step halting probe

halt LRを下げるだけのR2は保存済みupdate 10,000境界で停止した。問題はscalarの
大きさより、同じstate-wide ranking lossを全candidateの停止actionへ配る信用割当
そのものだったためである。

probe modeでは各stateの停止候補を最大2件だけ正確に追加1step進め、そのcandidateの
Qだけを差し替えたListNet＋marginを再計算する。`L_stop-L_continue > c`なら
continue、それ以外はstopを最終halt logitへBCE教師として与える。未probe candidate
にはhalt gradientを流さず、task score／task loss／task VJPはprobe前の値を保持する。
全深度PonderNet展開は行わない。

1-step primitiveは通常のd+1 forwardと最大差`2.38e-7`で一致した。現行R2
update-10,000 checkpointを使うreal-teacher serial/barrierless smokeではoutput、loss、
raw VJP、probe target／deltaが完全一致し、parameter gradient最大差`4.04e-6`、
optimizer後parameter最大差`4.04e-7`で合格した。100 measured updatesは
`20.66 updates/s`で、最低15 updates/sを維持した。benchmark更新は保存していない。

設計、正当性witness、速度preflightは
[`HALTING_ONE_STEP_PROBE_2026-07-22.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/HALTING_ONE_STEP_PROBE_2026-07-22.md)
に記録した。

続くfrom-scratch P1は2 probes/state、`c=0`、probe weight 1、halt LR
`5e-5`のまま100,000更新（400,000 teacher-state draws）を完走した。update
95,000がbalanced winnerで、loss `2.587874`、top-1 `0.734375`、NDCG
`0.991345`、pairwise `0.904401`、margin `0.141909`、held mean depth
`2.194`。top-1最高値は90,000の`0.742188`（mean depth `3.021`）、最終
100,000はloss `2.605494`、top-1 `0.710938`、NDCG `0.990369`、margin
`0.151411`、mean depth `2.011`だった。

state-wide R1最終に対してP1最終はloss、top-1、NDCG、pairwise、marginの
全指標を改善し、R1のheld mean depth `11.91`という最大深度飽和を解消した。
同一batch内でもcontinue/stop probe教師が混在したため、candidate固有の信用割当は
成立した。一方で深度は依然2--5付近を振動し、最終は最小深度寄りである。したがって
「少数1-step probeは旧REINFORCEより正しい」は支持されるが、「理想的な入力依存
computeを獲得した」とはまだ結論しない。aggregate speedは`19.936 updates/s`、
CPU平均`59.85%`、candidate中`60.92%`で、許容下限15 updates/sを維持した。

## 21. 2026-07-23 — spatial backward範囲外書込みの修正

学習安定化調整の前提確認中、8近傍spatial attention backwardが長さ4の
scratchへedge 5～8を書き込む範囲外アクセスを特定した。通常実行ではheapを徐々に
破壊し、200～400更新後のGC access violationとして現れていた。

`BackwardScratch.dweights`を全attention supportの最大値で確保するよう修正した。
数式、入力、teacher、loss、hard halting、RNG順序、active-only backward、
sparse optimizer、checkpoint形式は不変である。

bounds-check付きserial／20-worker smokeは合格し、出力、loss、raw VJPは完全一致、
parameter gradient最大差`4.60e-6`、optimizer後state最大差`4.60e-7`だった。
P1 75,000更新checkpointからの1,000更新preflightも完走し、測定990更新で
`15.505 updates/s`、allocation `7.739 MB/update`、GC時間比`0.84%`となった。

過去checkpointは履歴として保持するが、修正後の最終性能根拠には使わず、LR、
weight decay、state batchの比較は修正版からscratchでやり直す。

## 22. 2026-07-23 — batch比較と修正後20,000更新基準

state batch 8へ実行workspaceとqueue capacityを一般化し、8状態・全328候補の
serial／barrierless bounds-check smokeを実施した。出力、loss、raw VJPは完全一致、
parameter gradient最大差`3.51e-6`、optimizer後state最大差`3.51e-7`で合格した。

一方、100更新benchmarkはbatch 4の`15.505 updates/s`、`62.020 states/s`に対し、
batch 8が`7.529 updates/s`、`60.233 states/s`だった。CPU使用率だけは上がったが
実効state throughputは改善せず、最低15 updates/sを満たさないためbatch 8を不採用とした。

batch 4、dense LR `2e-4`、dense WD `1e-4`、halt LR `5e-5`、2 probes/stateの
修正後scratch試行を20,000更新まで実施した。held lossは`2.867476 -> 2.752324`、
top-1は`0.507812 -> 0.578125`、NDCGは`0.978324 -> 0.982200`へ改善した。
平均深度も`2.000 -> 2.107 -> 2.396 -> 3.556`と単調に獲得され、区間速度は
15.34～15.59 updates/sだった。修正後は現行LR／WDを安定基準とし、50,000更新以降の
episodic dense LRだけを0.5倍にする長期試行へ昇格する。

詳細は
[`TRAINING_STABILITY_TUNING_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/TRAINING_STABILITY_TUNING_2026-07-23.md)
に記録した。

20,000更新から現行halt LR `5e-5`を5,000更新延長すると、held平均深度が
`3.5563`から`2.0005`へ急落した。同じ20,000更新checkpointからhalt LRだけを
`1e-5`へ変更したpaired armでは、25,000更新の平均深度が`2.0844`、loss
`2.710228`、top-1 `0.578125`、margin `0.122518`となった。対照よりloss、margin、
深度、速度はわずかに改善し、NDCGだけ`0.000241`低下した。短区間では決着しないため、
`1e-5` armを50,000更新まで延長する。

延長結果では品質は改善し、40,000更新でheld loss `2.644841`、top-1
`0.656250`、NDCG `0.987263`、margin `0.134669`へ到達した。一方、平均深度は
25k `2.084`、30k `2.647`、35k `11.713`、40k `2.131`と両端間を反転した。
したがってhalt LR `1e-5`は品質armとして保持するが、深度安定化armとしては不採用とした。

同じ20,000更新checkpointからhalt LRだけを`1e-6`へ下げた比較も30,000更新まで
実施した。25kはloss `2.713433`、top-1 `0.570312`、平均深度`2.086`、30kは
loss `2.692056`、top-1 `0.632812`、NDCG `0.983332`、margin `0.120413`、
平均深度`2.709`だった。区間速度は`15.636 updates/s`で許容下限を維持した。

ただし`1e-5`も30kでは平均深度`2.647`だった後、35kで最大深度側へ飽和している。
したがって`1e-6`の短区間結果も安定化成功とは判定せず、halt LRの縮小だけでは
深度振動を解消できないという結論にした。batch 8も実効state throughputを改善しない
ため不採用のままとし、次はbatch 4でdense weight decayだけを比較する。

dense weight decayは、同じ20,000更新checkpointとhalt LR `5e-5`を使い、`1e-4`と
`3e-4`を30,000更新までpaired比較した。dense WDだけのresume変更を明示的に許可する
遷移を追加し、real-teacher serial／barrierless smokeは出力、loss、raw VJP完全一致、
parameter gradient最大差`1.53e-6`、optimizer後state最大差`1.53e-7`で合格した。

30kでは`1e-4`がloss `2.692367`、top-1 `0.632812`、NDCG `0.983506`、margin
`0.118709`、平均深度`3.372`。`3e-4`はloss `2.689768`、top-1 `0.625000`、
NDCG `0.983862`、margin `0.117013`、平均深度`2.110`だった。`3e-4`はloss、
NDCG、pairwiseを僅かに改善し、25kから30kの平均深度を`2.113 -> 2.110`に保ったが、
top-1とmarginは僅かに低下した。短区間の安定性armとして保持し、40kまで延長して
長期の品質と入力依存深度が残るかを確認する。

WD `3e-4` armを40kまで延長すると、35kはloss `2.664932`、top-1 `0.695312`、
NDCG `0.985194`、margin `0.141134`、平均深度`3.147`、40kはloss `2.645313`、
top-1 `0.671875`、NDCG `0.988018`、margin `0.134168`、平均深度`2.178`となった。
loss、NDCG、pairwiseは全評価点で改善し、15.24～15.56 updates/sを維持した。

平均深度の変動は残るが、halt LR `1e-5` armが35kで示した`11.713`という最大深度
飽和は回避した。40k品質もhalt LR `1e-5` armと同等以上のPareto点だったため、
dense WD `3e-4`を長期追跡の第一候補へ昇格し、100kまで延長する。

同じarmを50kまで進めると、45kはloss `2.622360`、top-1 `0.718750`、NDCG
`0.988368`、平均深度`2.131`、50kはloss `2.632145`、top-1 `0.734375`、
NDCG `0.987706`、平均深度`4.010`となった。top-1と入力依存深度は伸びたが、
continuous-ranking指標は45kから小幅に反落した。

50k checkpointを後半LR調整の共通分岐点とし、bank、router、halt LRは維持したまま、
attention、FFN、token、register、headに掛かる既存episodic LR scaleだけを0.5へ下げる。
これによりアーキテクチャとtask lossを変えず、dense経路の後半更新幅だけを比較する。

episodic LR scheduleの一軸遷移を実装し、50,001更新目からscale `0.5`を適用した
serial／barrierless smokeは出力、loss、raw VJP完全一致、parameter gradient最大差
`2.16e-6`、optimizer後state最大差`2.16e-7`で合格した。

55kではloss `2.612962`、top-1 `0.718750`、NDCG `0.988942`、pairwise
`0.890677`、margin `0.132410`、平均深度`4.287`（範囲2～6）となった。品質と深度は
有望だが、冷間JITを含む5k区間速度が`14.776 updates/s`で下限15を僅かに下回り、
checkpoint保存後に自動停止した。速度条件は緩めず、同じcheckpointから100更新warmup後の
1,000更新benchmarkで定常速度を確認してから採否を決める。

定常benchmarkは1,000測定更新を63.0281秒、`15.8659 updates/s`で完了し、速度条件に
合格した。states/sは`63.4638`、CPU平均`49.16%`、candidate中`49.42%`だった。
benchmark更新は保存していない。LR半減armを採用候補として55k checkpointから再開し、
速度判定区間を10,000更新以上にして冷間JITの偏りを除く。

本学習を55kから再開した結果、65kはloss `2.594471`、top-1 `0.757812`、NDCG
`0.990265`、pairwise `0.896571`、margin `0.142645`、平均深度`2.888`に到達した。
品質は本試行の最高水準だが、実データ10,000更新の速度は`14.625 updates/s`で下限15を
満たさず、65k checkpoint保存後に自動停止した。

短い定常benchmarkと長い実データ区間の差を無視せず、LR半減armは一旦保留する。同じ
50k checkpointからscale 1.0の対照を65kまで進め、区間workloadと学習率による経路変化を
分離してから採否を決める。

## 23. 2026-07-23 — 現時点の判明事項を統合

8近傍spatial backwardの範囲外書込みを修正した後の試行だけを現行性能根拠として整理した。
修正後EVRLはloss、top-1、NDCG、pairwise、marginを改善し、2～12 stepの入力依存再帰も
獲得している。

速度条件付きの現行採用点は、batch 4、dense LR `2e-4`、dense WD `3e-4`、halt LR
`5e-5`、2 probes/stateの50,000更新checkpointである。top-1 `0.734375`、NDCG
`0.987706`、pairwise `0.884073`、margin `0.131590`、平均深度`4.010`、速度
`15.545 updates/s`を記録した。

50k以降にepisodic dense LRだけを半減した試行は、65kでtop-1 `0.757812`、NDCG
`0.990265`まで伸びたが、実データ10,000更新では`14.625 updates/s`となり、最低速度
`15 updates/s`を満たさなかった。このcheckpointは品質参考として保持し、採用点にはしない。

PreActの既存結果とは学習teacher state数が異なるため、現時点ではPreAct超えを主張しない。
修正後EVRLの同一予算比較が必要である。構成、修正内容、全比較表、checkpoint hash、評価上の
境界は
[`CURRENT_FINDINGS_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/CURRENT_FINDINGS_2026-07-23.md)
へ統合した。

## 24. 2026-07-23 — 50k以降のLR維持対照を完了

同じ50,000更新checkpointからepisodic LR scale `1.0`を維持した対照を65,000更新まで
実行した。65kはloss `2.605426`、top-1 `0.742188`、NDCG `0.990088`、pairwise
`0.893202`、margin `0.140937`、平均深度`3.217`（範囲2～12）へ到達した。

15,000更新の実時間は876.050秒、`17.122 updates/s`、`68.489 states/s`、CPU平均
`53.759%`、candidate中`54.478%`で、速度条件に合格した。50k比でも全主要品質指標を
改善したため、速度条件付き採用checkpointをこの65k対照へ更新した。

LR半減armの65kは対照よりloss、top-1、NDCG、pairwise、marginのすべてで高かったが、
`14.625 updates/s`で下限15を満たさなかった。速度は対照の`85.4%`である。したがって、
LR半減armは品質上限の参考、scale 1.0対照は採用点として分離して保持する。

詳細とcheckpoint SHAは
[`TRAINING_STABILITY_TUNING_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/TRAINING_STABILITY_TUNING_2026-07-23.md)
および
[`CURRENT_FINDINGS_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/CURRENT_FINDINGS_2026-07-23.md)
へ反映した。

## 25. 2026-07-23 — 100k完走と予算別PreAct比較

速度条件に合格した65k checkpointからアーキテクチャと学習条件を変えず、100,000更新まで
継続した。35,000更新を1,803.908秒、`19.402 updates/s`、`77.609 states/s`で完了し、
最低15 updates/sを維持した。

85kはtop-1 `0.750000`、90kはpairwise `0.902023`、95kはloss `2.585896`、
NDCG `0.991278`、margin `0.157540`でそれぞれ最良だった。100kではloss
`2.601427`、top-1 `0.726562`、NDCG `0.989715`、pairwise `0.897215`へ反落した。
したがって、90～100kを振動を伴う頭打ち域と判定し、連続順位指標の均衡がよい95kを
速度条件付き採用checkpointとした。SHA-256は
`622753d65e0502edd09ed4ef10ecf3270b90e7a92e7d9b2d44cb6d71d9350893`である。

PreAct比較は予算軸を分離した。同じteacher state数ではPreAct 12kがEVRL 10k～15kを
明確に上回り、sample efficiencyの優位はPreActにある。一方、EVRL 95kの累積5,817秒に
近いPreAct 3kの約6,130秒を比較すると、EVRLがtop-1、NDCG、pairwise、lossで上回った。
最高到達品質ではPreAct 12.75kがEVRL 95kより高い。従って実証できた主張は、
「EVRLは同じwall-clock内の固定panel品質で上回るが、同じteacher state数と最高到達品質では
まだPreActを超えていない」である。

hard haltingは可変深度を維持したが、held平均深度は2.01～3.02を往復し、安定増加は
示さなかった。100k時点のLookup row-load Giniも`0.966～0.970`と高い。
次の改善対象は更新数の単純追加ではなく、Lookup routing collapseと停止信用割当である。

## 26. 2026-07-23 — Lookup・register・working memory・haltingの根本修正

前節で特定した問題を局所的な係数調整ではなく、信用割当とmemory dataflowから修正した。

- Lookup routerへstate-local hard頻度とsoft確率を組み合わせたanti-collapse勾配を追加した。
- 平均registerから1本だけ長期記憶を引く経路を廃止し、4 registerが共有bankを独立routing
  するようにした。
- cross-attentionのread supportを再利用し、更新後registerを283 tokenへ書き戻す
  working-memory writeを追加した。
- one-step probeがpolicy gradientを置換していた実装を修正し、全trajectoryの
  stop/continue creditへprobe BCEを加算するようにした。

real-teacher serial／barrierless smokeは出力、loss、raw VJPが完全一致し、parameter
gradient最大差`3.91e-7`、optimizer後parameter最大差`3.91e-8`で合格した。同じ4状態を
100回反復する過学習試験では、最初10更新から最後10更新へのloss低下が旧構成の`4.04%`
から`42.07%`へ改善した。

from-scratch 100,000更新は8,959.066秒で完了した。学習速度`11.332 updates/s`、
allocation`7.355 MB/update`、GC`59.792秒`、GC占有率`0.667%`だった。許容下限
`10 updates/s`を維持した。

100k累積のLookup coverageは各block`99.7～99.9%`、row-load Giniは`0.187～0.193`
だった。改修前のcoverage`26.9～30.1%`、Gini`0.966～0.970`から大幅に改善した。
training split内の実trajectoryでは、register間Lookup address相違率`99.90%`、
cross-attention argmax相違率`89.72%`、working-memory write RMS`0.384`を確認した。

training-only固定128状態はloss`2.55998`、top-1`0.671875`、NDCG`0.991213`、
pairwise`0.908830`、margin`0.137023`、平均深度`4.880`、深度範囲`3～12`だった。
validationとsealed seedは使用していないため、これは独立汎化性能の主張ではない。

100k checkpointのSHA-256は
`3def11e92a8af52267f178e25bb59881731a53278dffe1d53dd98fb599b38f61`である。完全な条件と
診断値は
[`ROOT_CAUSE_REPAIR_100K_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/ROOT_CAUSE_REPAIR_100K_2026-07-23.md)
へ記録した。

## 27. 2026-07-23 — 根本修正版の10k刻みcheckpointを同一パネルで再選択

追加学習を行わず、根本修正版runの10,000～100,000更新checkpointを、training splitから
固定した同じ128状態で再評価した。パネル行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`であり、
validationとsealed seedは使用していない。

90,000更新はcomposite loss`2.551617`で最小、top-1`0.671875`で最高同率、
pairwise`0.913689`で単独最高だった。NDCGも`0.991901`で、単独最高の70,000更新
`0.992014`との差は`0.000113`に留まった。100,000更新はmargin`0.137023`で最高だったが、
loss`2.559976`、NDCG`0.991213`、pairwise`0.908830`へ反落した。

この結果から、完走時点の100,000更新を自動採用せず、総合順位品質が最良の90,000更新を
主採用checkpointとした。SHA-256は
`3ea19a64fd72521c1e679c53b348525194214db3073e0937e8e515c864e28c71`である。

推移は三段階に分かれた。10～30kは平均深度約2のままlossと全順位指標が急改善した。
30～60kは緩やかな改善とtop-1停滞、60～70kは平均深度`2.820 -> 4.273`を伴う第2の
改善だった。70～100kでは平均深度が`4.273 -> 2.000 -> 3.027 -> 4.880`と振動し、
90kで総合最良に達した後、100kはmarginだけを伸ばしてloss、NDCG、pairwiseが反落した。
一方、Lookup平均Giniは`0.426669 -> 0.190199`へ全期間で単調低下し、coverageは
`85.15% -> 99.82%`へ単調上昇した。後半の振動はLookup collapse再発ではなく、task loss
内の複数目的とhalting方策の相互作用による可能性が高い。

全checkpointの結果と採用規則は
[`ROOTFIX_CHECKPOINT_SWEEP_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/ROOTFIX_CHECKPOINT_SWEEP_2026-07-23.md)
へ記録した。

## 28. 2026-07-23 — halting信用割当を収束させた

根本修正版の60～100kで平均深度が`2.820 -> 4.273 -> 2.000 -> 3.027 ->
4.880`と振動した原因を、trajectory全体へ同じterminal lossを与える高分散な信用割当と
特定した。悪いtrajectoryでは途中のcontinueと最後のstopが同時に抑制され、halt logitが
0付近にある多数候補がglobal biasの小変化だけで一斉に反転していた。

既存の物理的に疎な1-step probeを拡張し、probe候補について既に保存済みの各step Qだけを
差し替え、同じListNetとmargin lossを再計算した。これにより最終stopだけでなく、先行する
全stochastic continueへ候補固有の`L_stop - L_continue`教師を与えた。probe済み
trajectoryでは高分散なterminal REINFORCEを重ねず、未probe trajectoryだけ0.1倍の
fallbackを残した。

最初にstep 4中心の単調hazard priorを加えたが、無制限のlearned residualがpriorを
上書きし、20k更新後も平均深度が`3.182 -> 3.413 -> 2.006 -> 4.133`と振動した。
このarmは不採用とした。

最終版では
`halt_logit = 0.5 * (step - 4) + 0.75 * tanh(learned_logit)`とし、halt LRを
60k以降`1e-6`へ収束させた。real-teacher serial／barrierless smokeは出力、loss、
raw VJPが完全一致し、parameter gradient最大差`1.47e-6`、optimizer後parameter
最大差`1.94e-5`で合格した。

同じ60k checkpointから20k更新すると、固定training-only 128状態の平均深度は
65/70/75/80kで`3.001、3.066、3.006、3.036`となった。振幅`0.065`は不採用armの
`2.127`から96.9%減少した。決定論的深度は3～6、学習時の確率的深度は最終batchで
2～9だった。NDCGは`0.991098 -> 0.991465`、pairwiseは
`0.906943 -> 0.911968`へ改善した。

20k更新は1,788.273秒、`11.184 updates/s`で、allocationは`8.620 MB/update`、
GC占有率は`0.763%`だった。速度下限10 updates/sを維持した。

最終halting式で既存90kも再評価し、loss`2.551701`、top-1`0.671875`、NDCG
`0.991979`、pairwise`0.913475`、平均深度`4.004`、範囲4～5を得た。収束arm75kより
総合品質が高いため、主採用checkpointは90k、SHA-256
`3ea19a64fd72521c1e679c53b348525194214db3073e0937e8e515c864e28c71`
を維持した。

詳細は
[`HALTING_CONVERGENCE_REPAIR_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/HALTING_CONVERGENCE_REPAIR_2026-07-23.md)
へ記録した。validationとsealed seedは使用していない。

## 29. 2026-07-23 — 現行最終式をフルスクラッチで100k学習

候補固有の各step 1-step信用割当と有界halting式を、既存60k checkpointからの短期arm
だけでなく、初期値から共同学習した。親checkpointなしで100,000更新、400,000 teacher
stateを正常完了した。学習本体は9,389.019秒、`10.651 updates/s`、CPU平均
`54.255%`、candidate中`57.023%`、allocation`8.383 MB/update`、GC比率
`0.665%`だった。最低速度10 updates/sを維持した。

10k～100kの全checkpointを、SHA-256
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
の同じtraining-only固定128状態で再評価した。100kがloss`2.543041`、
NDCG`0.993363`、pairwise`0.921549`で最良、top-1`0.703125`、
margin`0.123395`だった。90kはtop-1`0.718750`で最高だった。連続順位品質を優先し、
100kを新しい主採用checkpointとした。SHA-256は
`2eb9a64293eb2c592830d9c25ea52969950e6440a97415a43ec093d0a77c8766`
である。

100kの決定論的深度は3～6で、深度3、4、5、6へ3,314、330、847、866 candidateが
分布した。入力依存停止は成立した一方、50～100kの平均深度は
`4.891 -> 3.012 -> 3.304 -> 4.496 -> 3.026 -> 3.863`と動いた。従来の最小／最大
深度への崩壊は抑えたが、checkpoint間の平均深度は未収束である。

同じ固定パネルの旧90k主採用点より、loss`0.008660`、top-1`0.031250`、
NDCG`0.001384`、pairwise`0.008074`、margin`0.006889`改善した。既存PreAct
12.75kに対してはlossとmarginのみ上回り、top-1、NDCG、pairwiseとsample efficiencyは
まだ届いていない。validationとsealed seedは使用していないため、独立汎化性能の主張では
ない。

詳細は
[`HALTING_FULLSCRATCH_100K_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/HALTING_FULLSCRATCH_100K_2026-07-23.md)
へ記録した。

## 30. 2026-07-23 — 再帰step内を単一Lookup＋セルDWConvへ縮約

外側で同じblockを入力依存回数だけ再帰するにもかかわらず、旧EVRLは各step・各registerで
LookupFFNを3段直列に実行していた。内部Lookup深度を廃止し、EVRLのLookup blockを1個へ
縮約した。代わりに、入力固有の24x10セルworking memoryへ共有3x3 depthwise
convolutionを各再帰stepで適用する。register数、全283 token cross-attention、
SwiGLU、register別long-memory routing、working-memory書込み、hard halting、
1-step probe、active-only backward、sparse optimizerは変更していない。

production geometryの総parameter数は`20,585,982 -> 6,954,621`となり、
13,631,361、66.22%減少した。全4 registerのLookup micro-callはstepあたり
`12 -> 4`、追加DWConvは1,153 parameter、276,480 scalar MAC/stepである。同じ
DWConv kernelをstep間共有するため、物理的な受容野は実行した思考回数に応じて広がる。

DWConv kernelとresidual scaleのmanual VJPを有限差分で検証し、8/8 testが合格した。
channel連続SIMD kernelへしたreal-teacher scratchは120更新を完了し、20更新warmup後の
100更新で7.723秒、`12.949 updates/s`、CPU平均`66.707%`、candidate中
`68.352%`だった。素朴な非連続kernel配置の同じsmokeは`10.107 updates/s`であり、
数学モデルを変えないmemory layout修正で1.281倍になった。

同checkpointからserialとbarrierlessを同一4状態で比較し、output、loss、raw VJPは
完全一致した。parameter gradient最大差は`1.10e-6`、optimizer後state最大差は
`1.10e-7`で、routing、halting、RNG、optimizer clockも全てexactだった。

旧3-block 100k checkpointは新geometryへ暗黙変換せず研究履歴として保持する。今回の
120更新は実装確認であり性能評価には使わない。新構成の収束性能はscratch本学習後に
別途比較する。validation評価とsealed seedは使用していない。

詳細は
[`SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/SINGLE_LOOKUP_RECURRENT_DWCONV_2026-07-23.md)
へ記録した。

## 31. 2026-07-23 — 単一Lookup化後の残存ボトルネックを測定

新しい120更新checkpointから、checkpointを保存しない100更新warmupと100更新詳細
benchmarkを実行した。定常速度は`14.778 updates/s`、`59.113 states/s`、
`2,633.190 candidates/s`、`8,467.056 recurrent steps/s`だった。CPU平均
`65.636%`、candidate中`67.024%`、allocation`6.620 MB/update`、GC比率
`0.648%`である。

executor CPU時間の約94%をforwardとbackwardが占めた。capacity-normalized worker
時間ではbackward 41.3%、forward 30.6%、queue wait 26.3%で、gradient reduce、
optimizer、GCはいずれも支配的ではなかった。

production geometryの概算では、全283 token cross readが約2.42M MAC/step、
working-memory reverse writeが約1.21M MAC/stepなのに対し、単一Lookupのselected-row
gatherは約0.020M MAC/stepである。旧3-blockでも全bankを計算せず、activeな39行だけを
読んでいたため、総parameterを3分の1へ減らしてもstep全体は3分の1にならない。

旧3-block 100kの保存統計と比較すると、updates/sは`10.668 -> 14.778`で1.385倍、
candidates/sは1.413倍、recurrent steps/sは1.475倍だった。stepあたりforward CPUは
11.8%、backward CPUは21.6%減った。速度向上は確認できたが、全token read/write、
そのVJP、追加DWConv、可変深度によるworker tailが残るため2倍には届かなかった。

詳細は
[`SINGLE_LOOKUP_BOTTLENECK_PROFILE_2026-07-23.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/SINGLE_LOOKUP_BOTTLENECK_PROFILE_2026-07-23.md)
へ記録した。
