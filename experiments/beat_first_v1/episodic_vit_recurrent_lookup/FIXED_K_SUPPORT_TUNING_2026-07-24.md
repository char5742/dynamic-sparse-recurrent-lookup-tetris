# 固定エピソードsupport Kの調整

## 目的

単一Lookup＋再帰depthwise版の全体構造を変えず、各registerが283 tokenから読む
固定support数だけを調整した。Kは1 run中に変化させず、学習時と推論時で同じ値を使う。
curriculum、annealing、dense score後のmask、追加routerは導入していない。

比較した値は`64 / 80 / 88 / 96 / 128`である。入力、teacher、loss、optimizer、
hard halting、probe、state batch、seed、schedulerは固定した。全試行は実teacherを使い、
評価は同じtraining-only固定128状態だけで行った。評価行SHA-256は
`c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c`
であり、validation rowとsealed seedには触れていない。

## 実装

`EVRL_EPISODIC_SUPPORT`をrun開始時に一度だけ読み、`16:283`の固定値として
scratch、trajectory、forward、backward、topologyへ一貫して反映するようにした。
未指定時は従来どおりK=64である。したがって既存production設定の挙動は変わらない。

構造試験はK=64、96、128でそれぞれ25項目すべて合格した。単一Lookup＋再帰
depthwiseの回帰試験も8項目すべて合格した。

K=128では実teacher checkpointを用いてserialとbarrierlessを比較し、次を確認した。

| 項目 | 結果 |
|---|---:|
| 出力の最大絶対差 | 0 |
| lossの最大絶対差 | 0 |
| raw VJPの最大絶対差 | 0 |
| parameter gradientの相対L2差 | 1.0068e-6 |
| optimizer更新後stateの相対L2差 | 2.184e-9 |

hard halting、選択token、Lookup row、RNG、optimizer clockも一致した。

## 1,000更新比較

共通条件はparameter数`6,954,877`、state batch 4、router LR`4e-4`、その他の
dense LR`2e-4`、dense WD`3e-4`、halt LR`5e-5`、warmup 5,000更新、
2 probe/state、compute price 0、barrierless、pinningなし、chunk 8である。

| K | loss | top-1 | NDCG | pairwise | margin | updates/s | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 3.276297 | **0.421875** | 0.955092 | 0.773679 | **0.110220** | **11.43** | 基準 |
| 80 | **3.252699** | 0.367188 | 0.949633 | 0.763625 | 0.105456 | 8.642 | 速度・順位とも不採用 |
| 88 | 3.448901 | 0.359375 | 0.948478 | 0.745965 | 0.057814 | 10.672 | 速度合格、品質不採用 |
| 96 | 3.263806 | 0.312500 | **0.956662** | **0.775232** | 0.071520 | 10.424 | 長期確認へ |
| 128 | 3.246717 | **0.421875** | 0.952711 | 0.765488 | 0.065297 | 9.612 | 速度不採用 |

K=96だけはK=64に対してNDCG`+0.001570`、pairwise`+0.001553`だったため、
追加学習した。K=80と88は連続順位品質がK=64を下回り、K=128は最低速度
10 updates/sを満たさなかった。

## 実行性能

K=80以降の完全統制runでは、20 Julia worker、BLAS 1、同じschedulerを使用した。

| K | 全体CPU | candidate中CPU | allocation/update | GC占有率 | active Lookup要素 |
|---:|---:|---:|---:|---:|---:|
| 80 | 46.31% | 47.52% | 6.170 MB | 0.428% | 587,648 |
| 88 | 59.10% | 61.57% | 6.170 MB | 0.512% | 535,680 |
| 96 | 58.64% | 60.91% | 6.170 MB | 0.565% | 526,976 |
| 128 | 58.41% | 60.46% | 6.175 MB | 0.467% | 515,712 |

速度はKだけの単調関数ではなかった。Kがroutingと予測を変えるため、同じteacher
batchでも選択されるLookup rowの和集合、backward、sparse optimizer量が変化する。
したがってattentionのQK数だけから実時間を推定せず、必ずrun全体で測定した。

## K=96の5,000更新確認

K=96は1,000更新から5,000更新まで継続した。

| 更新 | loss | top-1 | NDCG | pairwise | margin | 平均深度 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 3.263806 | 0.312500 | 0.956662 | 0.775232 | 0.071520 | 3.000 |
| 2,000 | 3.047027 | 0.414063 | 0.966922 | 0.793464 | 0.079964 | 3.000 |
| 3,000 | 3.012472 | 0.414063 | 0.972066 | 0.812123 | 0.084012 | 3.000 |
| 4,000 | 3.014505 | 0.445313 | 0.971186 | 0.819159 | 0.091015 | 3.000 |
| 5,000 | 2.931962 | 0.406250 | 0.973400 | 0.826625 | 0.082331 | 3.000 |

順位学習信号は継続したが、4,500～5,000更新のsteady区間は
`9.90695 updates/s`、全体CPU`56.25%`、candidate中CPU`59.31%`だった。
最低10 updates/sを下回ったため、5,000更新checkpointを保存して自動停止した。

checkpoint SHA-256：

```text
c7521d691210f9b71e5e99ff6851aa404acfd2c701248ec3c8a893b0885bba06
```

## 除外した起動

K=80の調整中、起動環境でDSRL形状またはEVRLのregister/FFN形状を明示しなかった
2 runを検出した。前者は76 table・343 row・256次元、後者は8 register・FFN 256であり、
K以外も変化していた。configのtopology照合で発見し、精度比較から完全に除外した。
上表のK=80は、13 table・4096 row・128次元・top3・4 register・FFN 128を明示した
3回目の完全統制runだけを使っている。

## 結論

production設定は固定K=64を維持する。

- K=64は1,000更新のtop-1とmarginが最良で、11 updates/s以上を満たした。
- K=96は短期NDCGとpairwiseをわずかに改善したが、5,000更新で速度条件を下回った。
- K=80と88はK=64より順位品質が低かった。
- K=128は速度と連続順位品質の両方で採用根拠を得られなかった。

Kの拡大だけではPreActとの差を解消できない。後続の学習率調整や追加学習は、
検証済みK=64の100,000更新checkpointを基準に行う。
