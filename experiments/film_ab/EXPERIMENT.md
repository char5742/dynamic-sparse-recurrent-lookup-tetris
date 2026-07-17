# F01: controlled NEXT/HOLD FiLM A/B

- Hypothesis: conditioning the residual board feature extractor on the six
  HOLD/NEXT tokens improves complete-candidate teacher-policy ranking over the
  current head-concatenation-only `CompactCandidateQ`.
- Mechanism: the same 64-dimensional queue embedding produces per-channel
  scale and shift in the residual block, allowing board patterns to change
  meaning as a function of future pieces.
- Change: add one zero-initialized `64 -> 16` FiLM projection to the otherwise
  identical 8-channel, one-block, two-spatial-channel compact model.
- Controls: common shared initialization, exactly equal update-zero output,
  same precomputed row batches, dataset, fixed 74 candidate width, standardized
  masked ListNet CE, seed `517812839`, AdamW, learning rate `1e-3`, 300 updates.
  Both teacher and standardized student logits use temperature `0.25`, matching
  the learning track's current strongest offline screen.
- Success: FiLM development top-1 and MRR exceed plain, listwise CE is lower,
  and the gain is material enough to justify later development-game evaluation.
- Time limit: 5 minutes total wall time for both 300-update fits after Julia
  compilation; stop either fit on NaN/Inf.
- Rejection: no improvement in development top-1/MRR, numerical instability,
  >25% parameter growth, or a wall-time penalty disproportionate to ranking
  gain.
- Leakage: only teacher episodes/seeds 5742--5749 are loaded. Validation and
  sealed test game seeds are not executed by this experiment.

Run only after the root agent assigns an uncontaminated CPU window:

```powershell
julia --project=. --threads=20 experiments\film_ab\train_ab.jl
```
