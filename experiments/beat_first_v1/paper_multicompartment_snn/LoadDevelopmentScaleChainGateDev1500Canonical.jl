# Canonical development gate for the official NeuronIO-compatible 1500 ms
# final.v2 Hay/NEURON artifact.

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinalDynamic.jl"))

Base.include(
    DevelopmentScaleChainGate,
    joinpath(
        @__DIR__,
        "DevelopmentScaleChainGateDev1500CanonicalOverlay.jl",
    ),
)
