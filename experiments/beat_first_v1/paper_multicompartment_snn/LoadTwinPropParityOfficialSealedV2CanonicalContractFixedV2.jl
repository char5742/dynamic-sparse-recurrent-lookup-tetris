"""
Canonical parity loader for the contract-fixed sealed V2 evaluator.

The parity implementation bytes remain pinned. Only its independently reviewed
evaluator trust anchor is substituted with the SHA-256 of the executable
contract-fix overlay. The overlay is loaded first, so raw-shard verification
recomputes the same corrected attestation.
"""

using SHA

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
))

let
    parity_source_path = joinpath(
        @__DIR__,
        "TwinPropParityOfficialSealedV2Canonical.jl",
    )
    expected_parity_sha256 =
        "952d10f4846c4b5e5830fb81d4813bdb7ecc1da3459be35863f4385012612ff6"
    bytes2hex(SHA.sha256(read(parity_source_path))) ==
        expected_parity_sha256 ||
        error("sealed V2 parity canonical source SHA-256 differs")
    source = read(parity_source_path, String)

    original_evaluator_sha256 =
        "9f77b11759e6aa1bfbdaa1345b6e38d8bf54d58a8d7b30f861a7d6ac6fcc61c6"
    corrected_evaluator_sha256 =
        Main.PaperELMTwinOfficialV2SealedReleaseV2.
            corrected_evaluator_source_sha256_v2()
    length(findall(original_evaluator_sha256, source)) == 1 ||
        error("sealed V2 evaluator binding token differs")
    patched = replace(
        source,
        original_evaluator_sha256 => corrected_evaluator_sha256;
        count=1,
    )
    Base.include_string(
        Main,
        patched,
        parity_source_path * "#contract-fixed-v2-evaluator",
    )
end

if !isdefined(
    Main,
    :TWINPROP_PARITY_OFFICIAL_SEALED_V2_CONTRACT_FIXED_CANONICAL,
)
    const TWINPROP_PARITY_OFFICIAL_SEALED_V2_CONTRACT_FIXED_CANONICAL =
        Main.TwinPropParityOfficial
end

TWINPROP_PARITY_OFFICIAL_SEALED_V2_CONTRACT_FIXED_CANONICAL
