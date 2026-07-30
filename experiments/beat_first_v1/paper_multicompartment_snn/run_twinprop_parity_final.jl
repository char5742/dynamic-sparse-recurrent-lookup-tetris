"""
Canonical CLI entry for HD-SWSNN-TwinProp XOR/parity reproduction.

The implementation lives in `run_twinprop_parity.jl`; keeping this explicit
Final entry prevents callers from loading the pre-overlay draft module.
"""

include(joinpath(@__DIR__, "run_twinprop_parity.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

