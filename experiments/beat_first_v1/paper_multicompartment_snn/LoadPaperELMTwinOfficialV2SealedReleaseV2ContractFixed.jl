# Canonical loader for the corrected sealed V2 held-out measurement contract.
# It preserves the exact V2 artifact/type boundary and patches only the
# evaluator and attestation constructor inside the existing sealed module.

if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ),
    )
end

const _PAPER_ELM_SEALED_V2_CONTRACT_MODULE =
    Main.PaperELMTwinOfficialV2SealedReleaseV2

if !isdefined(
    _PAPER_ELM_SEALED_V2_CONTRACT_MODULE,
    :SEALED_V2_CONTRACT_FIX_APPLIED,
)
    Base.include(
        _PAPER_ELM_SEALED_V2_CONTRACT_MODULE,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2ContractFix.jl",
        ),
    )
end

if !isdefined(
    Main,
    :PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED,
)
    const PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED =
        _PAPER_ELM_SEALED_V2_CONTRACT_MODULE
end

PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED
