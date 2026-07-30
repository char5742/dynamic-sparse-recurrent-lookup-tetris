using Test
using JSON3
using SHA

include(joinpath(@__DIR__, "OfficialTeacherContract.jl"))
using .OfficialTeacherContract

const SMOKE_MANIFEST_V2 = get(
    ENV,
    "HD_SWSNN_OFFICIAL_SMOKE_MANIFEST",
    raw"C:\tmp\hd_swsnn_twinprop_neuron_smoke\manifest.json",
)
const EXPECTED_SMOKE_DIGEST_V2 =
    "8c2b158314ec082d6ab43795d685bbf9951d4527b26ebc8fb6606ad1404ec1ae"

sha256_text_v2(text) = bytes2hex(SHA.sha256(codeunits(text)))

function synthetic_manifest_v2(; explicit::Bool=true)
    canonical =
        "{\"config\":{\"celsius\":34.0,\"label\":\"a b\"}," *
        "\"model_name\":\"HD-SWSNN-TwinProp\"," *
        "\"schema_name\":\"hd_swsnn_twinprop.neuron_teacher.v1\"}"
    digest = sha256_text_v2(canonical)
    contract =
        canonical[begin:prevind(canonical, lastindex(canonical))] *
        ",\"teacher_contract_sha256\":\"$digest\"}"
    fields = String[
        "\"decoy\":" *
        String(JSON3.write("quoted \"teacher_contract\": { text")),
        "\"nested\":{\"teacher_contract\":{\"ignored\":true}}",
        "\"teacher_contract\":$contract",
    ]
    explicit && push!(
        fields,
        "\"teacher_contract_canonical_json\":" *
        String(JSON3.write(canonical)),
    )
    push!(fields, "\"teacher_contract_sha256\":\"$digest\"")
    return "{" * join(fields, ",") * "}", canonical, digest
end

@testset "official smoke preserves Python numeric lexemes" begin
    @test isfile(SMOKE_MANIFEST_V2)
    smoke = read(SMOKE_MANIFEST_V2, String)
    result = verify_teacher_contract_file(SMOKE_MANIFEST_V2)
    @test result.digest == EXPECTED_SMOKE_DIGEST_V2
    @test result.canonical_source == :raw_contract_lexemes
    @test occursin("\"celsius\":34.0", result.canonical_json)
    @test !occursin("\"celsius\":34,", result.canonical_json)
    @test sha256_text_v2(result.canonical_json) ==
          EXPECTED_SMOKE_DIGEST_V2

    contract =
        extract_top_level_json_value(smoke, "teacher_contract")
    @test occursin("\"celsius\": 34.0", contract)
    formatted = replace(
        smoke,
        "\"teacher_contract\": {" =>
            "\"teacher_contract\"  : \r\n\t {";
        count=1,
    )
    @test verify_teacher_contract_manifest(formatted).digest ==
          EXPECTED_SMOKE_DIGEST_V2

    changed = replace(
        contract,
        "\"celsius\": 34.0" => "\"celsius\": 34";
        count=1,
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(smoke, contract => changed; count=1),
    )
end

@testset "explicit canonical is preferred but cannot mask corruption" begin
    manifest, canonical, digest = synthetic_manifest_v2()
    result = verify_teacher_contract_manifest(manifest)
    @test result.digest == digest
    @test result.canonical_json == canonical
    @test result.canonical_source == :explicit_manifest_field
    @test occursin(
        "\"teacher_contract_sha256\"",
        result.contract_json,
    )

    fallback, _, _ = synthetic_manifest_v2(; explicit=false)
    @test verify_teacher_contract_manifest(fallback).canonical_source ==
          :raw_contract_lexemes

    contract =
        extract_top_level_json_value(manifest, "teacher_contract")
    changed_contract = replace(
        contract,
        "\"celsius\":34.0" => "\"celsius\":34";
        count=1,
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, contract => changed_contract; count=1),
    )

    explicit_json = String(JSON3.write(canonical))
    changed_explicit = String(
        JSON3.write(replace(canonical, "34.0" => "35.0"; count=1)),
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, explicit_json => changed_explicit; count=1),
    )
    spaced_explicit = String(
        JSON3.write(replace(canonical, "{\"config\"" => "{ \"config\"")),
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, explicit_json => spaced_explicit; count=1),
    )
end

@testset "digest and structural corruption fail closed" begin
    manifest, canonical, digest = synthetic_manifest_v2()
    zero_digest = repeat("0", 64)
    contract =
        extract_top_level_json_value(manifest, "teacher_contract")

    # Corrupt only the final top-level declaration.
    suffix = "\"teacher_contract_sha256\":\"$digest\"}"
    @test endswith(manifest, suffix)
    prefix_last =
        prevind(manifest, lastindex(manifest), length(suffix))
    top_corrupt =
        manifest[begin:prefix_last] *
        "\"teacher_contract_sha256\":\"$zero_digest\"}"
    @test_throws ErrorException verify_teacher_contract_manifest(top_corrupt)

    nested_corrupt = replace(
        contract,
        digest => zero_digest;
        count=1,
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, contract => nested_corrupt; count=1),
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, digest => uppercase(digest)),
    )

    duplicate =
        "{\"teacher_contract\":" *
        contract *
        "," *
        manifest[nextind(manifest, firstindex(manifest)):end]
    @test_throws ErrorException verify_teacher_contract_manifest(duplicate)

    reordered =
        "{\"schema_name\":\"hd_swsnn_twinprop.neuron_teacher.v1\"," *
        "\"config\":{\"celsius\":34.0,\"label\":\"a b\"}," *
        "\"model_name\":\"HD-SWSNN-TwinProp\"," *
        "\"teacher_contract_sha256\":\"$digest\"}"
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, contract => reordered; count=1),
    )

    string_changed = replace(
        contract,
        "\"a b\"" => "\"ab\"";
        count=1,
    )
    @test_throws ErrorException verify_teacher_contract_manifest(
        replace(manifest, contract => string_changed; count=1),
    )

    @test compact_json_lexemes(
        " { \n \"x\" : \"a b\\\\\\\" c\", \"n\" : 34.0 } ",
    ) == "{\"x\":\"a b\\\\\\\" c\",\"n\":34.0}"
    @test sha256_text_v2(canonical) == digest
end
