"""
Canonical correctness entrypoint for the new Reduced-Hay direct-Tetris
mainline.

This currently dispatches the reference reverse-mode BPTT trainer. It is not
yet the production barrierless analytic-VJP trainer; that distinction is
intentional and recorded in README.md.
"""

include(joinpath(@__DIR__, "train_direct_smoke.jl"))
main(ARGS)
