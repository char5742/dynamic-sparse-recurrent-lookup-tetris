# Canonical sealed V2 loader with the corrected held-out metric contract.

if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ),
    )
end

const _PAPER_ELM_SEALED_V2_CONTRACT_MODULE_V2 =
    Main.PaperELMTwinOfficialV2SealedReleaseV2

if !isdefined(
    _PAPER_ELM_SEALED_V2_CONTRACT_MODULE_V2,
    :SEALED_V2_CONTRACT_FIX_V2_APPLIED,
)
    Base.include(
        _PAPER_ELM_SEALED_V2_CONTRACT_MODULE_V2,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2ContractFixV2.jl",
        ),
    )
end

if !isdefined(
    Main,
    :PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2,
)
    const PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2 =
        _PAPER_ELM_SEALED_V2_CONTRACT_MODULE_V2
end

PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2
