# P1: legacy partial-tail TD one-shot terminal record

Date: 2026-07-18 JST

Source commit: `6c649056fc247e6dcde64b7c8690218dadbac88b`

Decision: **P1-development-fail**

## Conclusion

The one authorized P1 execution terminated before row selection completed. The
terminal cause was a Julia parse error at
`experiments/legacy_partial_tail_td/select_rows.jl:53`; the expression
`abspath(PROGRAM_FILE) == @__FILE__ && main()` was rejected with
`invalid identifier`. The global one-shot marker was consumed, so this P1
contract must not be retried or rescued.

This is an infrastructure failure, not evidence for or against the scientific
hypothesis. No training, optimizer update, OpenVINO inference, game, development
evaluation, validation, or sealed test ran.

The run source predates the separately committed post-P1 Q1 strategy review
(`cf63479c429a81d4d6fd3ade74af2585f327d3a7`). No AD or Q1 result is part of
this P1 record.

## Observed phase accounting

| Phase | Seconds | Exit | Observation |
|---|---:|---:|---|
| `eligibility` | 0.4511315 | 0 | Eligibility artifact was written |
| `select_rows` | 1.7693951 | 1 | Parse error at `select_rows.jl:53` |
| `finalize_assessment` | 0.2269213 | 0 | Terminal failure artifacts were written |

The authoritative monitor recorded total wall time `2.5508624 s`, terminal
stop reason `phase select_rows failed: process exit 1`, and a complete terminal
monitor. The final status is `P1-development-fail`.

The following required phases were skipped after the failure:

- `extract_training`
- `train_partial`
- `verify_openvino`
- `extract_offline`
- `offline_gate`
- `evaluate_development`

Consequently, training data extraction, optimizer construction and updates,
candidate checkpoint export, OpenVINO CPU/NPU checks, game execution, and all
development/validation/test seed use were absent.

## Static scope of the parse defect

Read-only source search found the same terminal expression in three additional
entrypoints that were not executed:

- `experiments/legacy_partial_tail_td/train_partial.jl:566`
- `experiments/legacy_partial_tail_td/offline_gate.jl:99`
- `experiments/legacy_partial_tail_td/evaluate_development.jl:196`

These are recorded only as three latent same-form entrypoint defects. Their
runtime behavior was not observed in P1 and is not promoted to an execution
result.

## Immutable-input handling

The historical checkpoint was not deserialized for this terminal report. Only
filesystem byte count and SHA-256 were checked:

- path: `C:\Users\fshuu\Documents\tetris\1313\mainmodel copy 3.jld2`
- bytes: `83,460,093`
- SHA-256: `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`

The observed hash matches the immutable-input record in `freeze.json`. This
report did not execute a model, load the dataset, invoke OpenVINO, run the game,
or consume any development, validation, or test seed.

## Decision and scientific meaning

- Formal result: **P1-development-fail**.
- Retry authority: none; the global marker is consumed and records
  `retry_prohibited: true`.
- Scientific hypothesis: **untested**. There is no score, loss, update, or model
  strength evidence.
- G1/G2/G3 implication: none.
- Candidate checkpoint: none was produced.
- AD/Q1 implication: none; those are outside this run and this report.

## Provenance

- output directory:
  `D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1_6c649056_pwsh`
- global one-shot marker:
  `D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json`
- source commit:
  `6c649056fc247e6dcde64b7c8690218dadbac88b`
- source SHA-256:
  `443397f4ae14af06f5e35b15b59a1def7cd08d7f11b97176457d8636c0f82cba`
- manifest SHA-256:
  `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`
- harness SHA-256:
  `b8987e74d43912949b40a275ff1c43db819094697a0a8f0689b46a7406a98041`
- `final_result.json` SHA-256:
  `11554c7b4d19f56e94da66a5398db25c688c46ba51bee23511a22a859f98b22b`
