# Canonical strict-attestation plus differentiable-kernel loader.

if !isdefined(Main, :PaperELMTwinOfficialV2Final)
    include(joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2Final.jl",
    ))
end

const OFFICIAL_ELM_FINAL_CANONICAL =
    Main.PaperELMTwinOfficialV2Final

if !isdefined(
    OFFICIAL_ELM_FINAL_CANONICAL,
    :twin_forward_after_preflight,
)
    Base.include(
        OFFICIAL_ELM_FINAL_CANONICAL,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2FinalDifferentiable.jl",
        ),
    )
end

OFFICIAL_ELM_FINAL_CANONICAL.OFFICIAL_ELM_INPUT_DIM == 1_278 ||
    error("canonical official ELM input dimension differs")
OFFICIAL_ELM_FINAL_CANONICAL.OFFICIAL_DENDRITIC_LOCATIONS == 639 ||
    error("canonical official ELM dendritic location count differs")

const PAPER_ELM_OFFICIAL_V2_FINAL_CANONICAL =
    OFFICIAL_ELM_FINAL_CANONICAL
