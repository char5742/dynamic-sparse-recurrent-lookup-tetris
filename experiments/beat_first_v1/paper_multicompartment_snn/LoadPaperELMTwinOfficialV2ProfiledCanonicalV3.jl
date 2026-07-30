# Final ACL-safe profiled loader.  This intentionally bypasses the discarded
# V2 loader guard and installs the RNG dispatch refinement once per process.

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2ProfiledCanonical.jl",
))

Base.include(
    OFFICIAL_ELM_PROFILED_CANONICAL,
    joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2ActivationProfilesHotfixV2.jl",
    ),
)

const PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V3 =
    OFFICIAL_ELM_PROFILED_CANONICAL
