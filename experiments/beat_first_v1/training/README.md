# Beat-first training

This directory is the short critical-path trainer for the three architectures
in `../models`. It does not read validation or sealed game seeds.

Stage 1 runs fixed-width (74 candidates) end-to-end teacher pretraining with:

- masked standardized ListNet (primary),
- old-Q Huber regression,
- teacher top-1/top-2 margin regression,
- selected-action death BCE,
- a small quantile-to-teacher initialization loss.

The teacher file has exact all-candidate line-clear/geometry targets derivable
from board plus placement, but only selected-action death observations. The
packer records all of them; the frozen model interface currently consumes the
geometry as its 37 engineered input features and exposes only Q, death, and
quantile heads.

Successive halving is a single process and a single driver: all three models
receive the same stage-1 row schedule, the best two receive stage 2, and the
best one receives stage 3. Promotion is lexicographic held-out teacher
top-1/NDCG/pairwise accuracy/old-Q Huber. Game-score promotion remains a
separate evaluation step.

Native fallback example:

```powershell
$env:BEAT_BACKEND='native'
$env:BEAT_HALVING_UPDATES='100,200,500'
julia --project=. experiments/beat_first_v1/training/train_supervised.jl
```

For Reactant, set `BEAT_BACKEND_SOURCE` and `BEAT_BACKEND_MODULE` to the fixed
shape backend. Both paths share `supervised_objective` exactly.

`rl_stage2.jl` contains the next-stage PER, n-step, EMA, teacher-decay, and
QR-DQN primitives. It is intentionally not launched until the teacher champion
passes fixed development games.
