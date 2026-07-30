# Canonical dynamic-metadata, content-addressed final.v2 development gate.

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinalV2.jl"))

Base.include(
    DevelopmentScaleChainGate,
    joinpath(
        @__DIR__,
        "DevelopmentScaleChainGateDynamicIdentityOverlay.jl",
    ),
)
