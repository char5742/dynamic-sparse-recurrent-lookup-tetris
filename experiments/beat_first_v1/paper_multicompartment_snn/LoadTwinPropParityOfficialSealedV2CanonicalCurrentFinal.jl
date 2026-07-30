"""
Final exact-current-V2 parity loader.

The parity implementation itself is SHA-256 fixed.  Its evaluator digest is
bound at load time to the exact current
`PaperELMTwinOfficialV2SealedReleaseV2.jl` bytes.  The sealed verifier then
recomputes the complete attestation from raw shards with those same bytes, so
an artifact made by any prior evaluator cannot pass.  The activation/profile,
model, differentiable, core, and teacher-contract sources remain independently
pinned in the parity implementation.
"""

using SHA

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
    evaluator_source_path = joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2SealedReleaseV2.jl",
    )
    current_evaluator_sha256 =
        bytes2hex(SHA.sha256(read(evaluator_source_path)))
    length(findall(original_evaluator_sha256, source)) == 1 ||
        error("sealed V2 evaluator binding token differs")
    patched = replace(
        source,
        original_evaluator_sha256 => current_evaluator_sha256;
        count=1,
    )
    Base.include_string(
        Main,
        patched,
        parity_source_path * "#exact-current-v2-evaluator",
    )
end

if !isdefined(
    Main,
    :TWINPROP_PARITY_OFFICIAL_SEALED_V2_CURRENT_CANONICAL,
)
    const TWINPROP_PARITY_OFFICIAL_SEALED_V2_CURRENT_CANONICAL =
        Main.TwinPropParityOfficial
end

TWINPROP_PARITY_OFFICIAL_SEALED_V2_CURRENT_CANONICAL
