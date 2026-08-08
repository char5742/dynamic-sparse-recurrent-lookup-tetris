# Canonical Reduced-Hay status

Updated: 2026-08-09 JST

## Main branch

`main` is intentionally kept at the last fully green production state.

- Hard-event source credit, Cell/Local/Graph: `b5b77bb`
- Hard-event Training/Barrierless integration: `d694842`
- Canonical teacher-v3 Dataset I/O: `c611bd8`

Verified evidence:

- Stage A Cell/Local/Graph normal and bounds: exit 0
- Stage B full G1 normal and bounds: exit 0
- Two-state Training serial versus 1/2/4 workers: 83/83
- Warmed full update: 0 allocated bytes, 0 GC
- Dataset I/O normal and bounds: 73/73
- Real teacher-v3 load: format 3, 518 parts, 110,366 states,
  100,243 train, 10,123 validation, 4,817,295 candidates

## Checkpoint schema2 handoff

The schema2 implementation is saved, but is not merged because its final full
G1 gate is red:

- Branch: `codex/canonical-checkpoint-v2-20260808-231653`
- Latest WIP commit: `3ddec475279d2cc3e173cb64e53f96d8ffbcf88e`
- Checkpoint normal/bounds: 38/38
- Training normal/bounds: 125/125
- Root normal/bounds: 202/202
- Full G1: 91/94; the only failures are the three production-root closure
  expectations that have not yet admitted `Sampler.jl` / `ReducedHayCPUSampler`

Next owner should update only the three expected closure/public-surface sets in
`test_canonical_integration.jl`, rerun full G1 normal and bounds, and merge the
WIP branch into `main` only when both exit 0.

## Promotion order

Do not start the 100k run before all prior gates pass:

1. Merge green Checkpoint schema2/resume.
2. Implement and run the canonical benchmark.
3. Prove at least 20 updates/s under the frozen production contract.
4. Complete the 1k gate.
5. Complete the 10k gate.
6. Decide whether the evidence permits the 100k run.

The 100k run has not started.
