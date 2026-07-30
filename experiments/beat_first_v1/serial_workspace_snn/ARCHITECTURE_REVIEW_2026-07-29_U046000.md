# DECOLLE block-local SNN review at update 46,000

## Stop state

- Run: `arena_v3_decolle_scratch_u100000_20260729_r2`
- Training was manually stopped at the user's request for architecture review.
- Final trace update: 46,000.
- Final durable checkpoint: update 40,000.
- Controller PID 23788 and Julia PID 21980 were stopped.
- No related run processes remained after the stop.
- The ignored run directory contains `MANUAL_STOP_NOTICE.json`; controller-managed files and checkpoints were not rewritten.

## Durable checkpoint

- Run-relative path: `checkpoints/checkpoint_000040000.jld2`
- Bytes: 9,326,716
- SHA-256: `49fbcf147674c167a7f1404e110394d2953a3fbdb9a2b68f48b789e200f5a1e2`
- The file size and SHA-256 match the run's `checkpoint_manifest.jsonl`.

The `trained/` directory is intentionally ignored by Git, so this document records the checkpoint identity without adding the binary checkpoint to the repository.

## Observed learning curve

| Update | Window loss | Window ListNet KL | Firing rate | Routing entropy | Utility nonzero |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 4.763609 | 1.415915 | 0.006501 | 0.6345 | 0.826 |
| 10,000 | 3.707944 | 1.087181 | 0.018410 | 0.5618 | 0.934 |
| 20,000 | 3.538907 | 0.922171 | 0.023783 | 0.4567 | 0.981 |
| 30,000 | 3.478497 | 0.854847 | 0.026993 | 0.3631 | 0.993 |
| 40,000 | 3.452997 | 0.816038 | 0.026371 | 0.4492 | 0.998 |
| 46,000 | 3.422558 | 0.795756 | 0.027953 | 0.4837 | 0.999 |

From 1,000 to 46,000 updates, window loss improved by 1.341051 and ListNet KL improved by 0.620159. Improvement slowed substantially after 20,000 updates:

| Segment | Loss improvement per 1k | KL improvement per 1k |
|---|---:|---:|
| 1k-10k | 0.11730 | 0.03653 |
| 10k-20k | 0.01690 | 0.01650 |
| 20k-30k | 0.00604 | 0.00673 |
| 30k-40k | 0.00255 | 0.00388 |
| 40k-46k | 0.00507 | 0.00338 |

The last segment still improved, so the run was not mathematically flat. It was, however, in a strongly diminishing-return regime.

## Internal-state interpretation

- Hot execution allocation stayed at 0 bytes and measured hot GC stayed at 0 seconds.
- CPU utilization at update 46,000 was 76.89%.
- Firing rate rose from 0.0065 at 1k to 0.0280 at 46k without a dead-network collapse.
- Routing entropy remained nonzero and recovered from 0.363 at 30k to 0.484 at 46k; routing did not collapse to one deterministic block set.
- Utility nonzero fraction reached 99.92%, showing that the block-local third factor and eligibility produced responsibility signals across nearly all synapses.
- Structural flips remained exactly 0 through update 46,000. Utility learning was active, but the fixed-fanout 1-node/1-swap criterion never admitted a topology change.
- Local Q loss improved strongly early, from 3.875 at 1k to 1.612 at 10k, then rose to 2.070 at 46k.
- Local death loss settled near 0.59 rather than continuing to improve.
- Local quantile loss was 2.558 at 46k and remained noisy.
- Local geometry loss rose from 0.112 at 1k to 0.363 at 46k. These are sampled-batch values rather than a fixed validation panel, but they show that the four local predictors were not jointly and monotonically improving in the production stream.

## Conclusion

The evidence supports a plateau of the effective trained architecture, not a proven theoretical capacity ceiling of the intended architecture.

The strongest limiting observations are:

1. topology never changed despite mature utility;
2. block-local auxiliary predictors stopped improving jointly;
3. ListNet gains per 1k fell by roughly an order of magnitude after the early phase;
4. routing and firing stayed alive, so the plateau is not explained by a simple silent-network or routing-collapse failure.

The next direction should change the representation, credit, or topology mechanism rather than merely extend this same run to 100k.

No final 100k evaluation or `verified_complete` claim is valid for this manually stopped run.
