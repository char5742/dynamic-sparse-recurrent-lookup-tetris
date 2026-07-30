include(joinpath(@__DIR__, "PaperELMTwinOfficialV2FinalBase.jl"))

@eval PaperELMTwinOfficialV2Final begin
    include(joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2FinalDifferentiable.jl",
    ))
end
