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

2026-07-25に、learned local spatial attention、3 registers、attention
16／1 head、SwiGLU64、固定K64 episodic read/writeを保ち、再帰step内の
LookupFFNだけを1段から3段へ戻した。

- parameter数：20,542,179
- serial／barrierless correctness smoke：合格
- 10,000→100,000更新の長期平均：13.480 updates/s
- 100k checkpointの短期steady測定：22.341 updates/s
- steady allocation：8.815 MB/update、GC比率0.804%
- 固定training-only 128状態の100k：loss 2.665670、top-1 0.609375、
  NDCG 0.986058、pairwise 0.880499、margin 0.079165、評価深度3
- 1段Lookup 100k比：loss -0.020479、top-1 +0.031250、
  NDCG +0.001711、pairwise +0.005728、margin -0.017891
- game validationとsealed seed：未使用
- 最終checkpoint SHA-256：
  `fc8f7e7e389d721dc02821fef45733394d6fe96bcfff8d14e98570e37222fa48`

3段化は順位品質を部分的に戻したが、過去の固定深度tuning winner
top-1 0.804688との差の約13.8%を埋めたに留まる。marginと長期速度は悪化したため、
Lookup段数だけが過去からの性能低下原因ではない。全10k checkpoint推移、旧DSRLN、
PreAct、速度、GC、数値一致の境界は
[`THREE_LOOKUP_RESTORATION_2026-07-25.md`](experiments/beat_first_v1/episodic_vit_recurrent_lookup/THREE_LOOKUP_RESTORATION_2026-07-25.md)
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
