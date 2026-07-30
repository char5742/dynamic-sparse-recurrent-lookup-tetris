# Canonical beat_first_v1 loader for the content-addressed final.v2
# development-scale teacher gate.

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinal.jl"))

Base.include(
    DevelopmentScaleChainGate,
    joinpath(@__DIR__, "DevelopmentScaleChainGateFinalV2Overlay.jl"),
)
