# D0 evaluation freeze and G2 provenance chain

Status: implementation complete; synthetic conformance tests pass. No test-seed
game was executed. The fixtures use the 32 frozen seed *identifiers* only as
synthetic JSON keys.

## Outcome

The previous G2 validator accepted any syntactically valid 64-character hash
and any shared nonempty freeze ID. D0 now makes the provenance chain explicit:

```text
protocol v1.1.0 + current source fingerprint + Manifest
                 + baseline config/checkpoint
                 + candidate config/checkpoint
                              |
                              v
             evaluation-freeze-v1 registry
                              |
             registry file SHA-256 + freeze ID
                              |
             role-specific G2 submissions
                              |
                              v
                g2-validation-v2 report
```

`evaluation_freeze_id` is the SHA-256 of stable canonical material (protocol,
source, Manifest, and both role records). It does not include itself or the
registry file bytes, so it is non-circular. Each submission separately records
the SHA-256 of the completed registry file. The registry should live under a
run/artifact directory, not under the source-fingerprint roots.

## Implemented commands

Create a registry only after both role configs and checkpoints are final:

```powershell
julia --project=. scripts\create_evaluation_freeze.jl `
  configs\baseline_openvino_npu.toml `
  configs\candidate_frozen.toml `
  D:\tetris-paper-plus\runs\g2\evaluation_freeze.json
```

The creator verifies the role, config and checkpoint files, declared checkpoint
hashes, protocol baseline checkpoint, protocol model-only budget, current
source fingerprint and Manifest. It stores exact runtime and budget tables for
each role.

Export one aggregate JSON or 32 individual episode JSON files. The exporter
requires an actual physical-backend request count; it does not fabricate that
count from aggregate candidates:

```powershell
julia --project=. scripts\export_g2_submission.jl `
  D:\tetris-paper-plus\runs\g2\evaluation_freeze.json `
  baseline `
  D:\tetris-paper-plus\runs\g2\baseline_submission.json `
  D:\tetris-paper-plus\runs\g2\baseline_episodes.json

julia --project=. scripts\export_g2_submission.jl `
  D:\tetris-paper-plus\runs\g2\evaluation_freeze.json `
  candidate `
  D:\tetris-paper-plus\runs\g2\candidate_submission.json `
  D:\tetris-paper-plus\runs\g2\candidate_episode_*.json
```

Validate the frozen pair:

```powershell
julia --project=. scripts\validate_g2_submission.jl `
  D:\tetris-paper-plus\runs\g2\baseline_submission.json `
  D:\tetris-paper-plus\runs\g2\candidate_submission.json `
  D:\tetris-paper-plus\runs\g2\evaluation_freeze.json `
  D:\tetris-paper-plus\runs\g2\g2_validation.json
```

## Enforcement

Before statistics are computed, validation now rejects:

- an absent registry or an unregistered/mutated freeze ID;
- a registry whose protocol, source tree or Manifest no longer matches disk;
- a registry with a noncanonical ID;
- modified/missing config or checkpoint files;
- config/checkpoint paths or hashes inconsistent with the registry;
- baseline checkpoint inconsistent with protocol v1.1.0;
- submission hashes, paths, role, runtime or budget inconsistent with the
  registered role;
- modified/missing episode source JSON files;
- incomplete, duplicate or non-test seed sets and all prior episode/budget
  contract failures.

An eligible result now contains, for both roles, mean/median/max/p10/p25/p75/p90
score, completion rate, candidate evaluation counts, logical and physical call
counts, generation/inference/wall-time totals and representative values. It
also contains all 32 paired rows, paired mean/median bootstrap intervals,
win/tie/loss counts and completion-rate difference.

## Test evidence

Command:

```powershell
julia --project=. --threads=4 test\runtests.jl
```

Final full run after implementation and documentation updates:

```text
candidate simulation preserves root RNG and replay: 6/6 passed
registered freeze/export/G2 validation chain:        25/25 passed
```

The synthetic chain proves a valid +100 paired fixture is exported and accepted
with all 32 paired rows and summary metrics. Negative fixtures prove rejection
of no registry, unregistered freeze ID, fake submission hash, baseline runtime
mismatch, modified checkpoint, modified config and modified episode-source
file. The final run completed after the report and scripts were present in the
source-fingerprint roots.

## Remaining gate

Infrastructure D0 is complete, but no real candidate config/checkpoint has been
frozen. Creating the real registry is a deliberate one-way experiment decision
and should occur only after D1 selects the single candidate. Validation and test
game seeds remain unopened until that point.
