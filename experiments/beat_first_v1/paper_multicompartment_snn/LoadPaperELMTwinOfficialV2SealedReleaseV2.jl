# Canonical sealed release loader for the profiled SiLU Paper ELM v2 twin.
# The sealed module itself loads the V3 profiled canonical builder, including
# the activation-profile RNG hotfix, before exposing its promotion boundary.

if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ),
    )
end

if !isdefined(Main, :PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CANONICAL)
    const PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CANONICAL =
        Main.PaperELMTwinOfficialV2SealedReleaseV2
end

PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CANONICAL