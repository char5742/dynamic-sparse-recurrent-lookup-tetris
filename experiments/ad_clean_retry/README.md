# AD clean retry

This directory is a self-contained CPU microbenchmark of one fixed-shape update on an
82,240-parameter dense Lux MLP: Lux forward, scalar MSE loss, reverse-mode gradient,
and one AdamW update. It is not the actual Tetris learner. Each backend runs in a fresh
Julia process. The loss is returned by the differentiated forward; no backend performs
a second forward pass. Reactant timing includes an explicit completion barrier.

The environment is resolved by Julia 1.12 and recorded in `Manifest.toml` and
`artifacts/environment.json`. All generated evidence stays under `artifacts/`.

Example native run:

```powershell
$env:AD_CLEAN_BACKEND='zygote'
$env:AD_CLEAN_BATCH='16'
$env:AD_CLEAN_STEPS='1000'
julia --threads=20 --project=. benchmark_native.jl
```

Example Reactant run:

```powershell
$env:AD_CLEAN_BATCH='16'
$env:AD_CLEAN_STEPS='1000'
julia --threads=20 --project=. benchmark_reactant.jl
```

The first update is compile-inclusive. Steady throughput excludes updates 1-10.

`run_all.ps1` reproduces the 12 performance cases in separate processes. It applies
a 300-second native timeout and a 600-second Reactant timeout by default, redirects
stdout/stderr per case, and refuses to continue after a failed case. The checked-in
`Manifest.toml` pins the measured environment. `RESULT.md` is the human-readable
decision record and `artifacts/summary.json` is its machine-readable input.
