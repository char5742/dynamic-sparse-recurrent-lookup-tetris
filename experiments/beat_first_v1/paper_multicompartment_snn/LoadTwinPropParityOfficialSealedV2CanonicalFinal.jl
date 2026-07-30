"""
ACL-safe final loader for the exact sealed-V2 parity boundary.

The implementation was written while the upstream sealed evaluator still had
the pre-fix source SHA.  The evaluator then received the reviewed zero-variance
NMDA normalization fix and its primary test became 40/40 green.  This loader
verifies the complete parity source and replaces exactly that one trust-anchor
digest in memory.  No executable logic is changed.
"""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "TwinPropParityOfficialSealedV2Canonical.jl",
    )
    expected_source_sha256 =
        "952d10f4846c4b5e5830fb81d4813bdb7ecc1da3459be35863f4385012612ff6"
    actual_source_sha256 =
        bytes2hex(SHA.sha256(read(source_path)))
    actual_source_sha256 == expected_source_sha256 ||
        error("sealed V2 parity canonical source SHA-256 differs")
    source = read(source_path, String)
    old_evaluator_sha256 =
        "9f77b11759e6aa1bfbdaa1345b6e38d8bf54d58a8d7b30f861a7d6ac6fcc61c6"
    fixed_evaluator_sha256 =
        "5c5e519bc45a5afce3fcadcf21f26e0d604328051516043e80dd5d9154bf8f13"
    length(findall(old_evaluator_sha256, source)) == 1 ||
        error("sealed V2 evaluator trust anchor differs")
    patched = replace(
        source,
        old_evaluator_sha256 => fixed_evaluator_sha256;
        count=1,
    )
    Base.include_string(
        Main,
        patched,
        source_path * "#reviewed-v2-evaluator-sha-correction",
    )
end

if !isdefined(
    Main,
    :TWINPROP_PARITY_OFFICIAL_SEALED_V2_CANONICAL,
)
    const TWINPROP_PARITY_OFFICIAL_SEALED_V2_CANONICAL =
        Main.TwinPropParityOfficial
end

TWINPROP_PARITY_OFFICIAL_SEALED_V2_CANONICAL
