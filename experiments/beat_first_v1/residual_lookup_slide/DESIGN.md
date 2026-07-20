# Residual Lookup-SLIDE R0

This directory is a new, isolated CPU-model prototype. It does not modify or
inherit the trainable router in `sparse_dynamic_3layer`, and it is not the
NNUE/difference-update Track B. NNUE remains a separate inference-reuse idea.
No score or model-strength claim follows from this static implementation.

## Frozen topology

The independent-candidate boundary is the existing sparse adapter's
`raw496`, `context22`, and flattened `NEXT/HOLD42`. A fixed signed CountSketch
maps `raw496 -> 192`, then context and NEXT/HOLD are copied to form a
256-dimensional carrier:

```text
[CountSketch192(raw496), context22, NEXT/HOLD42] = x0(256)
```

Three identical residual lookup blocks follow. In each block:

1. fixed RMSNorm;
2. fixed layer-seeded signed normalized FHT256;
3. 76 independent tables, each addressed by three independent seven-way WTA
   digits (`7^3 = 343` rows);
4. one direct 256-value row from every table;
5. `y = sum(selected rows) / sqrt(76)`;
6. `x <- x + alpha*y`, where `alpha = 0.1*tanh(a)` and is initialized to
   exactly `0.01`. The frozen Float32 logit is stored by bit pattern, and model
   initialization fails closed unless Julia evaluates it to exactly `0.01f0`.

There is no learned router, STE, shortlist, reranker, rehash, dense expert, or
fallback bank scan. Strict first-maximum tie breaking and immutable hash seeds
make every address deterministic. The final dense head is `256 -> 22`.

The bank is physically `256 x (343*76)` per block. Columns are table-major:

```text
flat_column = (table - 1) * 343 + address
```

Thus every selected row is contiguous in Julia's column-major storage.

## Exact parameter accounting

| Component | Parameters |
|---|---:|
| 3 x 76 x 343 x 256 lookup values | 20,020,224 |
| head weights (`22 x 256`) | 5,632 |
| head bias | 22 |
| three residual logits | 3 |
| **total** | **20,025,881** |

One candidate touches exactly 228 lookup rows (58,368 bank values), all 5,654
head parameters, and three residual logits: 64,025 active trainable
parameters. Routing performs 4,104 WTA comparisons and 6,144 FHT add/sub
operations. These are reported separately because table accumulation is not a
dense dot-product MAC.

## Manual selected-only reverse pass

Hard WTA addresses have zero derivative. The tape stores only 228 addresses,
their flat columns, three block inputs/lookup sums, and the final carrier.
`vjp_selected_parameters` returns each block's gradients as a `256 x 76`
matrix aligned with the 76 selected flat columns. It never allocates a dense
bank mask or a 20M-element bank gradient.

For a block `x_out = x_in + alpha*y`, the reverse pass is exact on the frozen
hard route:

```text
d(selected row) = alpha * dx_out / sqrt(76)
d(a) = dot(dx_out, y) * 0.1 * (1 - tanh(a)^2)
dx_in = dx_out
```

The dense head and fixed CountSketch transpose are also implemented manually.
The companion sparse optimizer consumes `(columns, dbanks)` and is responsible
for updating only those rows.

## Route and usage telemetry

Every forward result reports the three sets of table addresses/flat columns,
selected row/value counts, active parameter count, comparison count, and FHT
work. Optional `RouteUsage` records a `343 x tables x 3` counter cube outside
the model. Its summary reports per-layer selected count, occupied table-row
coverage, maximum load, and mean normalized per-table entropy. Telemetry is
not trainable and is not included in parameter accounting.

## Integration boundary

The intended adapter reuses the already frozen 496/22/42 feature split. Model
output row order remains the current 22-output supervised/RL contract. A
checkpoint must bind the topology, immutable router seeds, model arrays,
sparse optimizer state, dense optimizer state, trainer state, and RNG.

The first bounded runtime gate, after the current sole-Julia workload ends, is:

1. run `test_model.jl` and the optimizer/checkpoint tests;
2. finite-difference the head, one selected lookup value, one alpha logit, and
   one raw CountSketch coordinate while asserting routes do not change;
3. verify exact checkpoint continuation across one selected-only update;
4. benchmark candidate forward/VJP at one and many candidates, reporting
   selected rows, bytes touched, route entropy, allocation, and wall time;
5. compare a selected-only update against an independent tiny dense reference;
6. only then connect the existing teacher loss and start a bounded signal run.

### GO / NO-GO

GO requires exact accounting, deterministic addresses, numerical VJP
agreement, active-only optimizer writes, exact checkpoint continuation, and
measured throughput that justifies a 20M lookup bank. Any full-bank scan in a
candidate forward/update, route-dependent non-determinism, silent address
change during the numerical witness, or weak teacher signal is NO-GO. The
prototype must not displace the current convergence runs before these gates.
