"""
Executable exact-V2 loader for the sealed TwinProp XOR/parity runner.

The audited runner body is immutable and SHA-256 checked.  This loader makes
only three one-count transformations in memory:

1. canonical V1 parity include -> canonical V2 parity include,
2. both V1 sealed-release references -> exact V2 references,
3. removal of the Julia-1.12-incompatible footer.

The verified body supports only d=2/d=4 and only `variant=:full`.
"""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "run_twinprop_parity_sealed_final.jl",
    )
    expected_sha256 =
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    actual_sha256 = bytes2hex(SHA.sha256(read(source_path)))
    actual_sha256 == expected_sha256 ||
        error("audited sealed parity runner source SHA-256 differs")
    source = read(source_path, String)

    old_include = "TwinPropParityOfficialSealedCanonical.jl"
    new_include = "TwinPropParityOfficialSealedV2Canonical.jl"
    length(findall(old_include, source)) == 1 ||
        error("sealed parity runner canonical include differs")
    source = replace(source, old_include => new_include; count=1)

    old_release = "SealedELMRelease"
    new_release = "SealedELMReleaseV2"
    length(findall(old_release, source)) == 2 ||
        error("sealed parity runner release references differ")
    source = replace(source, old_release => new_release)

    rejected_footer =
        "abspath(PROGRAM_FILE) == @__FILE__ && main()\n"
    length(findall(rejected_footer, source)) == 1 ||
        error("sealed parity runner footer differs")
    source = replace(source, rejected_footer => ""; count=1)

    # No ablation token is permitted in the production parity body.
    any(
        occursin(token, source)
        for token in (":passive", ":no_nmda", ":soma_only")
    ) && error("sealed V2 parity runner contains an ablation path")
    length(findall("variant=:full", source)) == 2 ||
        error("sealed V2 parity runner full-cell binding differs")

    Base.include_string(
        @__MODULE__,
        source,
        source_path * "#exact-v2-canonical-transform",
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
