# G0 historical-model audit

Frozen on 2026-07-17. The source directory `1313/` is treated as input-only and
is excluded from Git. Reproduction code and generated artifacts live outside it.

## Immutable input inventory

| File | Bytes | SHA-256 |
|---|---:|---|
| `log.csv` | 82,607 | `E841BC91DCCFADC2A2C26A7D2C2E5E5AFACB9269DD8A42FAFC4115496E4AF4DC` |
| `mainmodel copy 3.jld2` | 83,460,093 | `7B0F78EDD0867D468C376F1B5375BB9A4D2195FA0FA5F76F94924723B26ADFC1` |
| `NEXT性能変化.xlsx` | 362,080 | `C589FCC8780CC47F2511F6C646BAB3933C9B873C841DBA0A89B5F7C947F9F8C2` |
| `plot.svg` | 70,911 | `F9E475F3B6A063BD0A5F4D6EA74A918ED29C1C5D77B11BCD5261F00C9065F8BA` |

## Reconstructed model

- Checkpoint: `mainmodel copy 3.jld2`
- Parameters: 20,787,454 (approximately 79.3 MiB of FP32 parameters)
- Board tower: 256 channels and 16 residual SE blocks
- Queue path: six 128-dimensional token embeddings followed by an MLP-Mixer
- Head: flattened 24x10 board representation plus queue/scalar features,
  `249 -> 1024 -> 256 -> 1`
- The historical LayerNorm spans feature, token, and candidate-set dimensions.
  Evaluation therefore preserves the original candidate batching behavior.

## Frozen environment and score rules

The complete contract is `configs/evaluation_protocol.toml`. In summary: 24x10
internal board (20 visible rows), HOLD enabled, stable candidate order, Xoshiro
RNG, and a limit of 250 placed pieces. The score is produced by
`vendor/Tetris/src/game.jl:add_score!`: line clears, T-spins, B2B, REN, and a
1,000-point perfect-clear bonus; hard/soft-drop distance is not scored.

Candidate simulation was repaired to copy the root RNG rather than mutate it.
Candidate ordering is explicitly stable. The historical rotation, direction,
and hard-drop behavior is retained from the 2024 engine snapshot. These repairs
are isolated in the reproduction engine and covered by deterministic tests.

## What the historical numbers mean

| Number | Traceable meaning |
|---:|---|
| 16,900 | User-reported historical achievement. No seed-bearing raw record has yet been identified. |
| 17,500 | Maximum score present in the recovered training `log.csv`. |
| 18,300 | Maximum raw cell in `NEXT性能変化.xlsx`, sheet/series `性能変化large256_15`, under NEXT=4. The workbook does not retain the execution seed. |

The 18,300 workbook value is valid evidence of an observed historical score but
cannot be exactly replayed by seed. G1 is therefore frozen at 18,400 under the
same 250-piece score contract. G2 does not use any of these maxima; it uses the
sealed paired seed protocol.

## Direct checkpoint reproduction

OpenVINO 2026.2.1 reproduces the restored Lux graph with maximum absolute error
of about `5.1e-6` on CPU and `6.9e-4` on NPU. Direct fixed-budget NPU episodes:

| Seed | NEXT | Score | Pieces | Candidate evaluations | Inference seconds |
|---:|---:|---:|---:|---:|---:|
| 5742 | 5 | 15,900 | 250 | 10,353 | 58.35 |
| 5743 | 5 | 15,600 | 250 | 10,964 | 69.05 |
| 5744 | 5 | 15,200 | 250 | 10,998 | 173.46 (shared NPU) |
| 5745 | 5 | 14,000 | 250 | 10,890 | 172.48 (shared NPU) |
| 5746 | 5 | 16,200 | 250 | 10,782 | 169.98 (shared NPU) |
| 5747 | 5 | 15,100 | 250 | 11,184 | 174.26 (shared NPU) |

The four latter runs shared one NPU and are score-valid but not used for latency
claims. The restored score range agrees with the historical distribution, so G0
is complete. This does not itself prove G1, G2, or G3.
