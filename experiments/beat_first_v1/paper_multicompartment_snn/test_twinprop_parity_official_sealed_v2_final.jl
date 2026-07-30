"""Run the SHA-verified V2 parity tests through the reviewed final loader."""

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
    old_loader = "TwinPropParityOfficialSealedV2Canonical.jl"
    final_loader =
        "LoadTwinPropParityOfficialSealedV2CanonicalFinal.jl"
    length(findall(old_loader, source)) == 1 ||
        error("sealed V2 parity test loader reference differs")
    patched = replace(source, old_loader => final_loader; count=1)
    Base.include_string(
        Main,
        patched,
        source_path * "#reviewed-final-loader",
    )
end
