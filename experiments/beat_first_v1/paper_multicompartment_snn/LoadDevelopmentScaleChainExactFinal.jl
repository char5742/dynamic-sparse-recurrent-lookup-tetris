# Canonical four-stage development-chain verifier.

include(
    joinpath(
        @__DIR__,
        "LoadDevelopmentScaleChainGateDev1500Canonical.jl",
    ),
)

Base.include(
    DevelopmentScaleChainGate,
    joinpath(@__DIR__, "DevelopmentScaleChainExactStagesOverlay.jl"),
)
