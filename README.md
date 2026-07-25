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
Earlier sparse models, routing experiments, CPU scheduler work, comparison
harnesses, and failed experiments are retained under
[`experiments/beat_first_v1`](experiments/beat_first_v1).

The complete Japanese research narrative, including rejected designs and
failed overfit tests, is preserved in
[`RESEARCH_TRAJECTORY.md`](RESEARCH_TRAJECTORY.md).

## 最新の検証結果

2026-07-25に、再帰step内のLookupFFNを1段へ戻し、4 registers、attention
32／4 heads、SwiGLU128、固定K128 episodic read/write、model dim 256へ拡幅した。

- parameter数：13,909,501
- serial／barrierless correctness smoke：合格
- 10,000→100,000更新の長期平均：8.204 updates/s
- 70k checkpointの短期steady測定：8.319 updates/s
- steady allocation：17.797 MB/update、GC比率0.481%
- 固定training-only 128状態の最良70k：loss 2.697863、top-1 0.562500、
  NDCG 0.983626、pairwise 0.868009、margin 0.078324、平均深度3.426
- top-1最高60k：0.625000
- game validationとsealed seed：未使用
- 主比較70k SHA-256：
  `1df101be811a7c1338bbd0456c41fefcc3a5244890aaf3ac9bda8dad71093287`

70k以降はlossと連続順位品質が反落し、100kはloss 2.817819、top-1 0.492188まで
悪化した。最良70kも既存1段K64 100kと3段K64 100kの総合品質を超えず、速度下限10も
満たさなかった。これはK128だけでなくmodel dim 128から256への復元も含む拡幅上限
試験であり、K単独効果とは解釈しない。全推移、旧構成、PreAct、速度、GC、数値一致は
[`WIDENED_SINGLE_LOOKUP_K128_2026-07-25.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/WIDENED_SINGLE_LOOKUP_K128_2026-07-25.md)
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
