# Q1: frozen-old additive residual one-shot terminal record

Date: 2026-07-18 JST

Source commit: `09de08ef26ae9ef129305887e92331aede02ad5c`

Decision: **Q1-offline-rejected**

## Conclusion

The one authorized Q1 execution completed its frozen-order and training-data
extraction stages, then the `train_correction` process tree crossed the
preregistered 4 GiB private-memory limit before it emitted an update-0 or
training-phase artifact. The wrapper terminated that phase and skipped
OpenVINO verification and the reused-development offline gate. The global
one-shot marker was consumed, so this exact Q1 contract must not be retried,
resumed, or rescued.

This is a resource-gate termination, not evidence for or against the scientific
hypothesis. The proposed 165,051-parameter additive correction was not shown to
complete initialization, optimizer setup, or any update. No correction weights
or candidate checkpoint were produced. No game, game seed, validation seed,
sealed test seed, or OpenVINO inference was used.

## Observed phase accounting

| Phase | Seconds | Exit | Observation |
|---|---:|---:|---|
| `extract_training` | 2.4170851 | 0 | Extracted 2,124 eligible rows and wrote `training.npz` |
| `train_correction` | 13.3527107 | 1 | Process-tree private memory exceeded 4 GiB |
| `finalize_assessment` | 0.4674739 | 0 | Terminal assessment artifacts were written |

The terminal monitor recorded total wall time `16.2923741 s`, peak process-tree
private bytes `4,506,931,200` (4.198 GiB), and peak process-tree working-set
bytes `944,803,840` (0.880 GiB). Its terminal failure was
`phase train_correction failed: process-tree memory exceeded 4 GiB`.

The following phases were skipped after the failure:

- `verify_openvino`
- `extract_offline`
- `offline_gate`

The terminal assessment therefore found the training phase, OpenVINO gate,
offline extraction, and offline gate missing or malformed, and assigned
`Q1-offline-rejected`.

## Memory interpretation and evidence boundary

On Windows, the monitored private-byte value represents privately committed
virtual memory; it is not resident RAM. The working set is the separately
observed resident-memory measure. Both values are process-tree sums, and the
harness did not save per-PID memory samples. The artifacts therefore establish
that the preregistered private-commit limit was crossed, but do not identify
which process or allocation caused it and do not prove duplicated model memory.

The run directory contains no `training_phase`, `training_failure`, progress,
update-0 snapshot, optimizer/update, checkpoint, or correction-weight artifact.
Because the first update-0 milestone was never emitted, it is not established
that model initialization or optimizer setup completed. It would be incorrect
to infer the exact allocation site from the monitor alone.

## Completed extraction and frozen scope

The completed extraction read aggregate rows 1:2160 (episodes 1:12), found the
preregistered 2,124 eligible training targets, and did not load validation/test
or game seeds. Its target used the raw stored old-Q maximum at `t+3`; DAgger
behavior bootstrap was disabled. The initializer was already exposed to the
separate offline rows, so their frozen role remained only
`reused_development_guard`, not held-out generalization.

Frozen run constants included 2,000 Zygote updates, state batch 4, 74 candidate
actions, a 720 s wall limit, and a 4,294,967,296-byte process-tree limit. The
source tree and repository were clean at freeze time.

## Decision and scientific meaning

- Formal result: **Q1-offline-rejected**.
- Retry/rescue authority: none; the global one-shot marker is consumed and both
  retry and rescue were preregistered as prohibited.
- Scientific hypothesis: **untested**. No optimizer update, learned weights,
  loss trajectory, offline safety result, or game-strength result exists.
- Model/G1/G2/G3 implication: none.
- Candidate checkpoint: none was produced or frozen.
- OpenVINO implication: none; verification did not run.
- Seed use: no game, validation, or sealed test seed was used.

## Provenance

- output directory:
  `D:\tetris-paper-plus\runs\frozen_old_additive_residual_Q1_09de08e`
- global one-shot marker:
  `D:\tetris-paper-plus\runs\frozen_old_additive_residual_Q1.started.json`
- source commit:
  `09de08ef26ae9ef129305887e92331aede02ad5c`
- authorized base commit:
  `8d784985f300598d2a05ed4402902ae86dfb4908`
- actual parent commit:
  `e838ce2b363962f406bd54e07358c476ffb54687`
- source SHA-256:
  `a4f8178c33bc11a2915112ca8cda2350675be948bb6f69edb5c527356d14245f`
- source-fingerprint file SHA-256:
  `653b373c8923b39ccbd3097c3d967865c5d3d97f1298ccc84cb7f9ba1259a56c`
- Manifest SHA-256:
  `2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09`
- harness SHA-256:
  `d0cab6afd1c49a58cbf631f4bc30d3466210209b07dc966242370287cd9b6536`
- `freeze.json` SHA-256:
  `2d2663e348610a7a28b6fd72b12da251175aec8ebcab41550ad99cb23d035ce1`
- `started.json` and global marker SHA-256:
  `6b91537e2469195d41878023517746463155a9f494d243b457682011efd9f0bc`
- `training_extraction.json` SHA-256:
  `eb2e2af594ed22b0b407a36f2ad0cf2670f86ccaed0e7a98f322904e3ec23ded`
- `training.npz` SHA-256:
  `ad5e1476b5f76b5f8197f167cbf008e93adce9f18cdb7bec69c05f5611a15e7b`
- `monitor.json` SHA-256:
  `ed745d328799aa939dbcb2dd19fb06adeccf353796e414f12433e0366e3e9b85`
- `final_result.json` SHA-256:
  `56c57c727fb0156aa5de4118e95e922fec4ef2ce34c8f636f0894fd03947704c`
