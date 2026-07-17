# Independent AD backend retry (2026-07-18)

This is a repo-local, independent retry. It does not modify the earlier
`experiments/ad_backend` harness or the external `D:\tetris-paper-plus\ad_retry`
artifacts.

## Fixed workload

- tracked `CompactCandidateQ(channels=8, blocks=1, spatial_channels=2)`;
- tracked standardized, masked ListNet objective;
- C11b warm checkpoint and aggregate C13 dataset, both read-only;
- fixed training rows `[1, 750, 1501, 2160]`, cycled for scaling shapes;
- action width 74, Float32, temperature 1;
- AdamW at `1e-3`, betas `(0.9, 0.999)`, weight decay `1e-4`;
- Julia 20 threads and BLAS 10 threads unless a result says otherwise.

No game, game score, validation row, sealed evaluation seed, or held-out test
seed is used.

## Paths compared

1. Native `Zygote.withgradient`, followed by `Optimisers.update!`.
2. Native `Enzyme.autodiff(ReverseWithPrimal, ...)`, with an `Active` scalar
   return, `Duplicated` parameters, `Const` model/state/data/objective, one
   preallocated parameter shadow zeroed and reused, and `Optimisers.update!`.
3. Lux's native `AutoEnzyme` training API. This is the explicitly
   mutation/caching-friendly alternative: Lux caches and zeroes the parameter
   shadow and applies the optimizer in place. It is retained only if it is
   materially different from direct Enzyme.
4. Reactant CPU + EnzymeMLIR through persistent Lux `single_train_step!`, with
   `return_gradients=Val(false)`, `sync=true`, and the returned `TrainState`
   retained. The timed region includes forward, loss, backward, and AdamW.

The native direct Enzyme path returns primal loss and derivative from one
differentiated execution; it does not time a separate loss forward.

## Measurements and stop rules

- Numerical gate: initial loss, native gradient cosine/max error/relative L2,
  and five-update loss/parameter agreement.
- Fresh process per timed backend and shape. Update 1 is compile-inclusive;
  steady statistics exclude updates 1--5.
- Historical fixed state batch 4: up to 1,000 updates for Zygote and Reactant,
  and at least 100 for native Enzyme unless it is already decisively slower.
- Scaling shapes 16/32/64/128 are attempted with strict wall/RAM timeboxing.
- A failure or unsupported shape is evidence and remains in `artifacts/` and
  `FAILURES.md`; it is not silently discarded.
- Production adoption requires numerical equivalence and at least 1.15x
  compile-inclusive end-to-end learner speed on the intended update horizon.
