# エピソード記憶routerの信用割当修正

## 結論

LookupFFNを3段から1段へ戻したことだけが精度低下の原因ではなかった。主要な連続経路の
手書きVJPは概ね正しかった一方、入力token routerの代理勾配が二つの問題を持っていた。

1. 既に選ばれたK64 tokenにしかrouter勾配がなく、初期の選択漏れを修復できない。
2. hard routing用の代理勾配がrouter parameterだけでなくtoken表現とregister状態へも
   逆流し、本来のtask gradientを汚していた。

後者は全trajectory有限差分監査で直接確認した。router代理勾配を本体表現へ逆流させた
状態では、cross residual scaleなど複数の連続parameterで有限差分と手書きVJPが一致
しなかった。router入力をstop-gradientにし、代理勾配の所有者をrouter parameterだけに
限定すると、同じ監査は全項目で合格した。

## 修正内容

高価なcross-attentionを全283 tokenへ戻してはいない。物理的なK64 read/writeを維持した
まま、routerだけを次の二段構成へ変更した。

1. learned hash/WTAでregisterごとに最大128 tokenの有界候補を取得する。
2. 8次元の小型router空間で候補だけを正確に再順位付けし、K64だけにK/V射影、
   multi-head QK、weighted gather、working-memory writeを行う。

学習時はK64のうち8 tokenを候補pool内の決定論的probeに割り当てる。選択K64で得た
exact attention massを128候補上のrouter softmaxへ蒸留し、未選択候補にも対照的な
router勾配を与える。probeで有用だったtokenは以後のtop-Kへ昇格できる。

重要なのは、この蒸留勾配とhard-support STEをtoken表現・register状態へは流さない点で
ある。本体にはtask lossの正確な連続VJPだけを流し、router行列だけがrouting surrogateを
受け取る。

今回の主要構成は次の通りである。

| 項目 | 値 |
|---|---:|
| Lookup block | 1 |
| model dim | 128 |
| register | 4 |
| attention | 32次元、4 head |
| SwiGLU | 128 |
| hash candidate pool | 128 |
| physical episodic support | 64 |
| 学習時exploration slot | 8 |
| router distillation weight | 0.1 |
| 総parameter数 | 6,954,877 |

## 全trajectory VJP監査

`test_full_trajectory_vjp.jl`を追加し、固定2-step trajectoryの離散token supportと
Lookup rowを固定したまま、出力との内積に対する中心有限差分を手書きVJPと比較した。
token/register routerはhard selection用代理勾配なので、有限差分対象から意図的に除外
した。

監査対象には次を含む。

- tokenizationとglobal visual stem
- learned local spatial Q/K/V/Oと相対位置
- recurrent depthwise
- cross-attention readとworking-memory write
- register self-attention
- SwiGLU
- register別Lookup gate
- Lookup head、bias、BH4、alpha、active bank row

修正前はrouter surrogateが本体へ混入し、例えばcross residual scaleの相対差が約48%に
達した。stop-gradient修正後は監査testが`2/2`合格し、40 parameter familyすべてが設定
した絶対・相対誤差条件内へ入った。headとbiasは約`1.4e-6`、主要attention／FFN行列は
概ね1%未満、routing境界に近く小さいepsilonしか使えない項目も約5%未満だった。

## 回帰とserial／barrierless一致

回帰testは次の通り合格した。

- fixed-K64 episodic read/write/router：`633/633`
- 1段Lookup＋recurrent DWConv：`8/8`
- 3段Lookup互換性：`11/11`
- 全trajectory有限差分：`2/2`

新規checkpointの次の同一real-teacher 4状態について、canonical serialと20-worker
barrierlessを1更新比較した。

| 項目 | 結果 |
|---|---:|
| output最大絶対差 | 0 |
| loss最大絶対差 | 0 |
| raw output VJP最大絶対差 | 0 |
| worker gradient相対L2 | `3.491e-7` |
| reduced parameter gradient相対L2 | `3.493e-7` |
| optimizer後parameter/state相対L2 | `1.259e-9` |

candidate seed、実現深度、hard halting、選択token edge、Lookup row、probe教師、
active token、sparse row event、optimizer clock、sampler、halt RNGも完全一致した。

## 同一4状態の記憶試験

validation／sealed seedは使用せず、real-teacher training row 1～4だけを100回反復
した。checkpointは保存していない。

### 固定3-step診断

| 指標 | 開始 | 100更新後 |
|---|---:|---:|
| 最初／最後10更新の平均loss | 4.38775 | 2.84074 |
| loss低下率 |  | **35.26%** |
| top-1 | 0 | 0.5 |
| NDCG | 0.82805 | 0.98386 |
| pairwise | 0.57390 | 0.85212 |
| old-Q loss | 3.72688 | 0.02156 |

### hard halting有効

固定深度をproductionへ採用してはいない。同じ4状態についてwarmupなしのsampled hard
haltingと1-step probeを有効にして、別の初期値から100更新した。

| 指標 | 開始 | 100更新後 |
|---|---:|---:|
| 最初／最後10更新の平均loss | 6.45022 | 3.04804 |
| loss低下率 |  | **52.75%** |
| top-1 | 0 | 1.0 |
| NDCG | 0.84905 | 0.98016 |
| old-Q loss | 3.23573 | 0.08198 |
| 評価平均深度 | 4.000 | 4.011 |
| 評価深度範囲 | 4～4 | 4～5 |

旧破綻試験では同一4状態100更新のloss低下が4.04%だった。今回の35.26～52.75%は、
現行1段Lookupモデルが少数状態を実用的な強さで記憶できること、またhard haltingを
有効にしても学習経路が切れていないことを示す。ただし同一4状態の結果は汎化性能を
意味しない。

## 速度、allocation、GC

20 Julia worker、BLAS 1、barrierless、pinningなし、forward chunk 8、backward chunk 1、
hard halting、2 probes/stateで、20 warmup＋100更新を測定した。

| 指標 | 値 |
|---|---:|
| updates/s | **14.953** |
| 平均CPU | 64.25% |
| candidate処理中CPU | 67.18% |
| allocation/update | 12.390 MB |
| executor allocation/update | 9.715 MB |
| GC時間 | 0.0456秒 / 6.6876秒 |
| GC占有率 | **0.681%** |
| sampled平均深度（最終batch） | 3.229 |
| 深度範囲（最終batch） | 2～8 |

GCはボトルネックではない。速度低下は128候補の再順位付け・router distillationと、
既存のforward/backward本体による。高価なepisodic K/V・QK・writeはK64のままであり、
全283 token dense attentionには戻していない。今回許容された10 updates/sは満たすが、
以前の20 updates/s目標には届いていない。

## 判定

「3段Lookupがないから全ての性能が失われた」という仮説は棄却する。3段化には小さな
表現力改善があるが、今回の監査ではそれ以前にrouter信用割当が本体task gradientを
汚していたことが判明した。

今回の修正により、

- 本体の連続VJPは全trajectory有限差分と一致
- 未選択候補へrouter固有の対照的信用が到達
- 物理K64 read/writeを維持
- 1段Lookupでも固定batch記憶試験に明確に合格
- hard halting時にも深度4～5と学習改善を両立
- serial／barrierlessとoptimizer stateが一致

した。

次の品質判定は、追加の大規模変更をせず、この構成を初期値から短期teacher streamで
学習し、既存と同一のtraining-only 128状態panelで比較してから行う。今回の作業では
validation rowとsealed game seedには触れていない。
