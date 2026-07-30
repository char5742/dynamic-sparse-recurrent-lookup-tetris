# Canonical official ELM release chain:
# strict independently recomputed attestation -> gradient-safe execution.

if !isdefined(
    Main,
    :PaperELMTwinOfficialV2ReleaseExecution,
)
    include(joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2ReleaseExecution.jl",
    ))
end

const PAPER_ELM_OFFICIAL_V2_RELEASE_CANONICAL =
    Main.PaperELMTwinOfficialV2ReleaseExecution
const PAPER_ELM_OFFICIAL_V2_RELEASE_ATTESTATION =
    PAPER_ELM_OFFICIAL_V2_RELEASE_CANONICAL.Release
const PAPER_ELM_OFFICIAL_V2_NUMERICAL_KERNEL =
    PAPER_ELM_OFFICIAL_V2_RELEASE_CANONICAL.ELM

PAPER_ELM_OFFICIAL_V2_NUMERICAL_KERNEL.OFFICIAL_ELM_INPUT_DIM ==
    1_278 ||
    error("canonical official release ELM input dimension differs")
PAPER_ELM_OFFICIAL_V2_NUMERICAL_KERNEL.OFFICIAL_DENDRITIC_LOCATIONS ==
    639 ||
    error("canonical official release dendritic locations differ")
