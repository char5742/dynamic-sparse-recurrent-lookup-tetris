include(joinpath(@__DIR__, "PaperELMTwinOfficialV2ReleaseBase.jl"))

@eval PaperELMTwinOfficialV2Release begin
    include(joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2ReleaseHardening.jl",
    ))
end

