"""
Production XOR/parity entry for the contract-fixed sealed V2 twin.

The audited runner body is byte-pinned.  This entry changes only its canonical
loader and V2 module alias, so parity training consumes the same corrected
sealed artifact emitted by
`finalize_paper_elm_twin_official_full_contract_fixed_v2.jl`.
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
    corrected_include =
        "LoadTwinPropParityOfficialSealedV2CanonicalContractFixedV2.jl"
    length(findall(old_include, source)) == 1 ||
        error("sealed parity runner canonical include differs")
    source = replace(
        source,
        old_include => corrected_include;
        count=1,
    )

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
    ) && error("contract-fixed V2 runner contains an ablation path")
    length(findall("variant=:full", source)) == 2 ||
        error("contract-fixed V2 runner full-cell binding differs")

    Base.include_string(
        @__MODULE__,
        source,
        source_path * "#contract-fixed-v2-transform",
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
