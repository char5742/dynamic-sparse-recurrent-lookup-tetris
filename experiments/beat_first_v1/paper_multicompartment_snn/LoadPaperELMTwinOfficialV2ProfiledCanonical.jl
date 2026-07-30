# Canonical Final preprocessing/integrity plus configurable Spieler-v2
# activation and explicit checkpoint-vs-paper provenance profiles.

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2FinalCanonical.jl",
))

const OFFICIAL_ELM_PROFILED_CANONICAL =
    Main.PAPER_ELM_OFFICIAL_V2_FINAL_CANONICAL

if !isdefined(
    OFFICIAL_ELM_PROFILED_CANONICAL,
    :ProfiledOfficialPaperELMTwin,
)
    Base.include(
        OFFICIAL_ELM_PROFILED_CANONICAL,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2ActivationProfiles.jl",
        ),
    )
end

const PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL =
    OFFICIAL_ELM_PROFILED_CANONICAL
