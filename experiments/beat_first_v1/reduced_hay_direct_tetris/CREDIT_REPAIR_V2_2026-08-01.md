# Reduced Hay v2 credit repair, 2026-08-01

## Scope

This repair followed the failed 100,000-update run whose deterministic
128-state validation result was excess loss `0.879047`, top-1 `0.195313`,
NDCG `0.922203`, and pairwise accuracy `0.700626`.

The failure was not treated as a request for another long run.  The critical
path was limited to routing saturation, recurrent-gradient interference,
branch-placement churn, and the validity of the exact reference comparison.

## Root repairs

1. Reduced Hay v2 routing now applies a monotone soft bound after score RMS
   normalization.  A worst-case 96-block score outlier has maximum training
   probability `0.103333`, rather than approaching one.  The ordered
   Plackett-Luce eligibility includes the bound derivative and passes finite
   differences with maximum error `5.84e-10`.  The default Point-SNN policy
   remains unbounded; the new bound is enabled only by Reduced Hay v2.
2. Workspace routing stays at temperature `1.0`; entropy and load controls are
   `4.0`, `0.85`, and `0.10`.  The failed `0.10` temperature schedule is no
   longer the production default or guard restart configuration.
3. Each recurrent parameter family is clipped independently to norm `5`.
   A large branch-bias adjoint can no longer scale down key/query, edge,
   delay, leak, or workspace-decay updates.
4. Branch relocation now requires utility gain above `0.25`, scans one of 64
   round-robin source partitions per consolidation event, and clears the
   moved edge's stale branch utility and optimizer moments.
5. The functional exact model accepts the arena trainer's current per-edge
   branch map.  A post-consolidation comparison no longer silently uses the
   initial relation-only placement.
6. The fixed-panel evaluator has explicit `edge_off`, `plateau_off`, and
   `apical_off` modes.  Resume into a new output directory now writes trace
   and checkpoint-manifest headers correctly.

## Verification

### Correctness and CPU hot path

- Reduced Hay v2 arena tests: `109 / 109` pass.
- Local versus exact tiny-model gradient cosine after 128 feedback warm-up:
  recurrent `0.877966`, credit-core `0.875055`, edge `0.956013`, supervised
  head `1.000000`, feedback `0.996301`.
- Workspace key cosine: `0.947387`.
- Workspace decay cosine: `1.000000` with matching sign.
- Production benchmark, batch 8, width 80, 20 workers:
  `97.361 states/s`, CPU utilization `81.8%`, hot allocation `0 byte`,
  GC `0 s`.

### Five-thousand-update mechanism check

The run resumed from the 1,000-update scratch checkpoint and reached 5,000:

`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\credit_repair_v2_resume_u5000_20260801`

During updates 1,001--5,000:

- mean routing entropy: `0.963337`;
- minimum routing entropy: `0.958090`;
- raw total gradient median / p99 / maximum: `4.294 / 10.870 / 18.901`;
- branch moves: `228`;
- hot allocation and GC: zero.

The old 100k trace had raw gradient maximum about `1.25e9` and branch moves
at roughly 75 times the repaired rate.

On the same deterministic 128-state validation panel at update 5,000:

| Mode | Excess loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| full | `1.118375` | `0.187500` | `0.905343` | `0.663640` |
| edge off | `1.160323` | `0.171875` | `0.913435` | `0.687111` |
| plateau off | `1.202600` | `0.171875` | `0.906311` | `0.661946` |
| apical off | `1.205261` | `0.179688` | `0.910234` | `0.674062` |

The recurrent edge, plateau, and apical mechanisms all improve the primary
excess-loss objective at 5k.  This is a mechanism check, not a claim that 5k
held quality is competitive; the held excess loss itself remains high.

### Eight-state memorization gate

The 2,000-update fixed-eight-state run is:

`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\credit_repair_v2_overfit8_u2000_20260801`

Deterministic evaluation of the exact stored eight states measured excess
loss `0.328426`, top-1 `0.875`, NDCG `0.973939`, and pairwise accuracy
`0.772535`.  This passes the required excess-loss-below-`0.5` gate.

## Decision boundary

### One-hundred-thousand-update scratch result

The repaired scratch run completed all 100,000 updates and 800,000 teacher
states without a watchdog restart:

`D:\tetris-paper-plus\runs\reduced_hay_v2_arena\credit_repair_v2_full_scratch_u100000_20260801_201223`

The final checkpoint SHA-256 is
`cb955f3b3512f886919acb7f3d3447aeb8b9cfb4ef137770909111a2b848cf7e`.
Total wall time was `8,649.26 s` (`11.562 updates/s`), total GC time was zero,
and the mean trace throughput after the first 100 updates was
`99.891 states/s`.

The 25,000-update mean training losses were `3.432881`, `3.268220`,
`3.241695`, and `3.234556`.  Raw gradient norm p99 / maximum were
`10.400 / 146.077`; unlike the failed run, there was no late gradient
explosion.  Mean routing entropy remained between `0.9535` and `0.9605`.

On the same deterministic 128-state validation panel:

| Run / mode | Excess loss | Top-1 | NDCG | Pairwise |
|---|---:|---:|---:|---:|
| failed 100k | `0.879047` | `0.195313` | `0.922203` | `0.700626` |
| repaired 100k, full | `0.642439` | `0.281250` | `0.951146` | `0.772012` |
| repaired 100k, edge off | `0.642076` | `0.281250` | `0.950553` | `0.773088` |
| repaired 100k, plateau off | `0.667732` | `0.296875` | `0.950369` | `0.768990` |
| repaired 100k, apical off | `0.863670` | `0.171875` | `0.921881` | `0.734862` |

The repaired run clearly improves the failed checkpoint on every primary
held ranking metric.  Apical modulation is causally important at 100k and
the plateau improves the primary excess-loss objective.  The recurrent edge
ablation is effectively tied with the full model at 128 states, so the 5k
edge contribution did not remain established at 100k.  This is a remaining
architecture or utilization issue, not evidence of the former catastrophic
credit-assignment failure.

The run is held-teacher ranking evidence only.  It does not establish gameplay
strength or equal-wall-clock superiority over Point-SNN, DSRLN, PreAct, or a
GRU baseline.
