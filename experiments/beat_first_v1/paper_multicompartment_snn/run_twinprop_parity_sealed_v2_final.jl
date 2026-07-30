"""
Final production entry for full-cell XOR and 4-bit parity on sealed V2.

The SHA-verified audited body is transformed only to select the reviewed V2
canonical loader/type and to remove its incompatible footer.  It contains no
ablation path and detailed NEURON replay remains the sole reproduction score.
"""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "run_twinprop_parity_sealed_final.jl",
    )
    expected_sha256 =
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    bytes2hex(SHA.sha256(read(source_path))) == expected_sha256 ||
        error("audited sealed parity runner source SHA-256 differs")
    source = read(source_path, String)

    old_include = "TwinPropParityOfficialSealedCanonical.jl"
    final_include =
        "LoadTwinPropParityOfficialSealedV2CanonicalFinal.jl"
    length(findall(old_include, source)) == 1 ||
        error("sealed parity runner canonical include differs")
    source = replace(source, old_include => final_include; count=1)

    length(findall("SealedELMRelease", source)) == 2 ||
        error("sealed parity runner release references differ")
    source = replace(
        source,
        "SealedELMRelease" => "SealedELMReleaseV2",
    )

    footer = "abspath(PROGRAM_FILE) == @__FILE__ && main()\n"
    length(findall(footer, source)) == 1 ||
        error("sealed parity runner footer differs")
    source = replace(source, footer => ""; count=1)

    any(
        occursin(token, source)
        for token in (":passive", ":no_nmda", ":soma_only")
    ) && error("final sealed V2 runner contains an ablation path")
    length(findall("variant=:full", source)) == 2 ||
        error("final sealed V2 runner full-cell binding differs")

    Base.include_string(
        @__MODULE__,
        source,
        source_path * "#final-exact-v2-transform",
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
