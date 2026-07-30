"""Run the SHA-verified runner tests against the final exact-V2 entry."""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "test_run_twinprop_parity_sealed_v2_canonical.jl",
    )
    expected_sha256 =
        "4f00400b55bbc225b9bd09cef94df5482f2d9c908cb02a2791b8b2071a71bbe5"
    bytes2hex(SHA.sha256(read(source_path))) == expected_sha256 ||
        error("sealed V2 runner test source SHA-256 differs")
    source = read(source_path, String)
    replacements = (
        "run_twinprop_parity_sealed_v2_canonical.jl" =>
            "run_twinprop_parity_sealed_v2_final.jl",
        "TwinPropParityOfficialSealedV2Canonical.jl" =>
            "LoadTwinPropParityOfficialSealedV2CanonicalFinal.jl",
        "exact-v2-canonical-transform" =>
            "final-exact-v2-transform",
    )
    for (old, new) in replacements
        length(findall(old, source)) == 1 ||
            error("sealed V2 runner test token `$old` differs")
        source = replace(source, old => new; count=1)
    end
    Base.include_string(
        Main,
        source,
        source_path * "#reviewed-final-runner",
    )
end
