# CPU-native dynamic sparse recurrent lookup research

This repository contains a Tetris teacher-learning research program built
around a CPU-native, dynamically executed neural architecture.  The current
model combines:

- learned sparse LookupFFN banks as long-term parameter memory;
- sparse episodic attention over board, candidate, next/hold, and auxiliary
  tokens;
- recurrent multi-register working memory;
- physically sparse forward, backward, and optimizer updates;
- hard, input-dependent routing and a recurrent halting interface; and
- a barrierless Windows scheduler for heterogeneous Intel P/E cores.

The central implementation is
[`experiments/beat_first_v1/episodic_vit_recurrent_lookup`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/README.md).
An independent third-model line, which scales the serial synaptic-printer
concept into a supervised Tetris graph SNN with a hard global workspace, is in
[`experiments/beat_first_v1/serial_workspace_snn`](experiments/beat_first_v1/serial_workspace_snn/README.md).
Earlier sparse models, routing experiments, CPU scheduler work, comparison
harnesses, and failed experiments are retained under
[`experiments/beat_first_v1`](experiments/beat_first_v1).

The complete Japanese research narrative, including rejected designs and
failed overfit tests, is preserved in
[`RESEARCH_TRAJECTORY.md`](RESEARCH_TRAJECTORY.md).

## 最新の検証結果

2026-07-27に速度下限を`15 updates/s`へ緩和し、register数、attention幅・head数、
SwiGLU幅、Lookup段数を比較した。数値一致と安定性を満たした拡幅上限は
3 registers、attention 24／3 heads、SwiGLU64、1段Lookupである。

- parameter数：6,919,987
- スクラッチ100,000更新：400,000 teacher state
- 累積学習時間：5,017.156秒、19.932 updates/s
- steady：20.924 updates/s
- allocation：9.820 MB/update、GC比率0.633%
- 固定training-only 128状態の100k：loss 2.679211、top-1 0.578125、
  NDCG 0.983389、pairwise 0.872891、margin 0.085935
- serial／barrierless最終smoke：合格
- validation、game validation、sealed seed：未使用
- 100k SHA-256：
  `38efe79fb7a5b843fe1044342637ace7d64095463f08f9e4a6814304cef80035`

この拡幅候補はpairwiseとmarginを僅かに改善したが、従来の2-register、
attention 16／1 head、SwiGLU32構成よりlossとtop-1が悪く、約20%遅かった。
2段Lookupもparameter数をほぼ倍増させながら10k品質を改善しなかった。したがって、
拡幅可能な上限は確認したものの、現時点の幅・深さPareto最適は従来の細い1段構成である。
全候補、30k選別、100k推移、PreAct差、GC、数値一致は
[`WIDTH_DEPTH_BALANCE_TUNING_100K_2026-07-27.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/WIDTH_DEPTH_BALANCE_TUNING_100K_2026-07-27.md)
に記録した。

## Repository policy

Teacher datasets, run directories, and binary checkpoints are intentionally
excluded.  They are large, machine-local artifacts and are not required to
review the implementation.  Published result records include artifact sizes
and hashes so local artifacts can be verified independently.

## Environment

The project is developed on Windows with Julia 1.12.6.  Instantiate the root
environment with:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

The experiment-specific entry points and environment variables are documented
next to each experiment.  BLAS must remain single-threaded when the native
candidate scheduler is enabled.
