# ACL-safe refinement of the profiled canonical loader.

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2ProfiledCanonical.jl",
))

if !isdefined(
    OFFICIAL_ELM_PROFILED_CANONICAL,
    :_PROFILED_ACTIVATION_DISPATCH_V2,
)
    Base.include(
        OFFICIAL_ELM_PROFILED_CANONICAL,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2ActivationProfilesHotfixV2.jl",
        ),
    )
    @eval OFFICIAL_ELM_PROFILED_CANONICAL const _PROFILED_ACTIVATION_DISPATCH_V2 =
        true
end

const PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V2 =
    OFFICIAL_ELM_PROFILED_CANONICAL
