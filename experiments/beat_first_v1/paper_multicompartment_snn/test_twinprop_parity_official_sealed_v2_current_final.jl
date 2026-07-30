"""Run the full V2 parity boundary test against the exact-current loader."""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "test_twinprop_parity_official_sealed_v2_canonical.jl",
    )
    expected_sha256 =
        "0ba1169a68731b5f7b12c5a9a26dad5c2656b10a583d3e0d01b26e7446e4382a"
    bytes2hex(SHA.sha256(read(source_path))) == expected_sha256 ||
        error("sealed V2 parity test source SHA-256 differs")
    source = read(source_path, String)
    old_include = """
        "TwinPropParityOfficialSealedV2Canonical.jl",
    ))
    """
    current_include = """
        "LoadTwinPropParityOfficialSealedV2CanonicalCurrentFinal.jl",
    ))
    """
    length(findall(old_include, source)) == 1 ||
        error("sealed V2 parity test include block differs")
    patched = replace(source, old_include => current_include; count=1)
    Base.include_string(
        Main,
        patched,
        source_path * "#exact-current-v2-test",
    )
end
