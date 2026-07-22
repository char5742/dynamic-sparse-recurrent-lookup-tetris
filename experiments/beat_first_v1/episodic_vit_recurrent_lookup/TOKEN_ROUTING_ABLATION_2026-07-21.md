# EVRL入力token routing ablation — 2026-07-21

## 判断

registerからepisodic memoryへの経路で、283 tokenをlearned hash/WTAにより64候補へ絞り、さらに正確なtop-16 shortlistを作る処理を廃止した。各registerは283個すべての入力tokenへattentionする。

変更範囲はepisodic input accessだけに限定した。PreActと完全に同じinput contract、物理local-8 cell attention、register self-attention、SwiGLU、LookupFFN parameter routing、active-only sparse bank update、共有recurrence、loss、optimizer semantics、比較用の固定depth 2は変更していない。

実装では各recurrent stepにつき、token memory全体を共有`32 x 283` K matrixと共有`32 x 283` V matrixへ一度だけ射影する。4つのregister queryは、その共有row上で正確な4-head softmaxを計算する。token candidate table、top-k support、dense mask、routing STEは存在しない。このため全episodic tokenにK/Vを介したtask gradientの直接経路がある。

## 動機

旧token routerは、registerによる入力参照をCPU上で物理的に疎にするため導入された。しかし固定283 tokenという現行規模では、その情報損失を正当化する比較結果がなかった。cell tokenizationとrecurrent spatial updateのcostは既に全token分を支払っているのに、最後のregister入力経路だけがtokenを破棄し、未選択位置への信用割当を遮断していた。本ablationはこの制限を直接検証する。

## 正当性の検証結果

canonical serial candidate state machineと20-worker barrierless executorを、同じ新checkpointから独立に復元した。全parameter／optimizer比較を短時間で行うため、4件のreal-teacher training statesについて各state最大4 candidatesに限定した。

| 確認項目 | 最大絶対差 | Relative L2 |
|---|---:|---:|
| Output | 0 | 0 |
| Loss | 0 | 0 |
| Raw output VJP | 0 | 0 |
| Worker gradient | 1.31130219e-6 | 5.69284813e-8 |
| Reduced parameter gradient | 5.21540642e-8 | 5.78742739e-8 |
| Optimizer後parameter/state | 1.49011612e-8 | 1.75511969e-10 |

candidate RNG、depth、hard halting、full-token support shape、Lookup row ID、active row、usage、optimizer clock、sparse row clock、sampler state、update後RNG stateは完全一致した。

## 学習

full-token modelを新規初期化し、同一予算のrouted modelおよびPreAct baselineと同じ12,000 updates／48,000 teacher statesまで学習した。

| 項目 | 値 |
|---|---:|
| Parameters | 20,577,224 |
| Updates | 12,000 |
| 消費state数 | 48,000 |
| Training throughput | 23.3685 updates/s |
| Candidates/s | 4,077.17 |
| 最終composite loss | 3.13755 |
| 最終ListNet loss | 3.00577 |
| 最終old-Q loss | 0.465372 |
| 最終margin loss | 0.0345595 |

最終checkpoint:

```text
path: D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_fixed2_u12000_20260721_r1\checkpoints\checkpoint_000012000.jls
bytes: 253659317
sha256: 80cc8264a03facf5ff4d0c13cde205b0763012281254362b2f15521c262a4f1c
```

binary checkpointは通常のGitHub object上限を超えるためcommitしていない。

## Held-teacher結果

以前と完全に同じ128-state panelを再利用した。

```text
dataset manifest: 1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded
split seed:        2026071817
subset seed:       2026072315
row-list SHA-256:  fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25
```

teacher Qと順位は教師信号としてのみ使用し、各candidateを独立に評価した。game validationとsealed seedは未使用である。

| 12k updates／48k states時点のモデル | Top-1 | NDCG | Pairwise | Margin | Composite loss | CPU states/s |
|---|---:|---:|---:|---:|---:|---:|
| PreAct-ECA | 0.78906 | 0.99329 | 0.92336 | 0.12332 | 2.56378 | 4.1925 |
| Routed EVRL (64 -> 16) | 0.35938 | 0.97127 | 0.82028 | 0.04954 | 2.97502 | 53.5423 |
| Full-token EVRL (283) | **0.56250** | **0.97867** | **0.84152** | **0.07205** | **2.80916** | **45.1745** |

full-token EVRLのrouted EVRLに対する差は次のとおり。

- top-1: `+0.203125`
- NDCG: `+0.0074061`
- pairwise accuracy: `+0.0212389`
- action margin: `+0.0225019`
- composite loss: `-0.1658505`
- inference throughput: `0.843715x`（15.63%低下）

full-token EVRLはPreActよりtop-1で`0.2265625`、NDCGで`0.0146179`低いが、このCPU held-panel測定では`10.7749x`高速だった。

## 結論

283 token規模では、hard input-token routingは必要な効率化ではなく有害なbottleneckだった。撤去により、控えめな推論costと引き換えにランキング品質が大幅に回復し、学習throughputの低下は観測されなかった。LookupFFN parameter sparsityは維持されているため、この結果はdynamic long-memory routing自体を否定しない。否定するのは、この規模で既にtoken化された短期episodic memoryを破棄する設計である。

machine-readableな根拠は[`token_routing_ablation_2026-07-21.json`](token_routing_ablation_2026-07-21.json)に保存した。
