module DistillationDatasetBridgeOfficial1278SealedContractFixedV2Final

"""
Critical-path adapter from the corrected sealed V2 artifact to the canonical
official-1278 bit-exact writer.

Extraction validates the sealed container, corrected evaluator identity, fixed
gate, frozen hashes, and source-manifest identity without deserializing teacher
shards.  The existing final-V2 writer remains the sole source verifier and
post-write bit-exact promotion authority.
"""

using JSON3
using SHA

Base.include(
    Main,
    joinpath(
        @__DIR__,
        "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
    ),
)
Base.include(
    Main,
    joinpath(
        @__DIR__,
        "prepare_distillation_dataset_official1278_sealed_final_v2.jl",
    ),
)

const Sealed =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2
const Twin = Sealed.Twin
const Writer =
    Main.DistillationDatasetBridgeOfficial1278SealedFinalV2

Writer.Sealed === Sealed ||
    error("official-1278 writer is not bound to corrected sealed V2")
Writer.Bridge.Sealed === Sealed ||
    error("official-1278 base bridge is not bound to corrected sealed V2")

export SEALED_RELEASE_SCHEMA,
    OFFICIAL_INPUT_DIM,
    extract_attested_frozen,
    prepare_distillation_dataset_from_sealed,
    main

const SEALED_RELEASE_SCHEMA = Sealed.SEALED_RELEASE_SCHEMA
const OFFICIAL_INPUT_DIM = Writer.OFFICIAL_INPUT_DIM

function _file_sha256(path::AbstractString)
    open(path, "r") do stream
        return bytes2hex(SHA.sha256(stream))
    end
end

function _validate_attested_bundle(
    bundle::Sealed.SealedOfficialELMRelease;
    require_production::Bool,
)
    payload = bundle.attestation.payload
    payload.schema == Sealed.SEALED_RELEASE_SCHEMA ||
        error("sealed release schema is not corrected final V2")
    payload.artifact_kind == Sealed.SEALED_RELEASE_ARTIFACT_KIND ||
        error("sealed release artifact kind differs")
    Sealed.canonical_sha256(payload) ==
        bundle.attestation.attestation_sha256 ||
        error("sealed attestation payload digest mismatch")
    payload.evaluator.source_sha256 ==
        Sealed.corrected_evaluator_source_sha256_v2() ||
        error("sealed release does not pin the corrected evaluator")
    payload.outcome.gate_passed === true ||
        error("sealed corrected V2 gate failed")
    payload.outcome.caller_metrics_accepted === false ||
        error("sealed release accepted caller metrics")
    payload.outcome.caller_targets_accepted === false ||
        error("sealed release accepted caller targets")
    payload.outcome.caller_manifest_digest_accepted === false ||
        error("sealed release accepted caller manifest identity")
    if require_production
        payload.outcome.promotable_production === true ||
            error("sealed release is not production-promotable")
    end

    frozen = bundle.frozen
    Twin.assert_frozen_official_elm_unchanged(frozen)
    payload.model.parameter_sha256 == frozen.parameter_sha256 ||
        error("sealed parameter hash differs from embedded frozen twin")
    payload.model.base_artifact_sha256 == frozen.artifact_sha256 ||
        error("sealed artifact hash differs from embedded frozen twin")
    return bundle
end

function _validate_manifest_identity(
    bundle::Sealed.SealedOfficialELMRelease,
    dataset_root::AbstractString,
)
    root = abspath(dataset_root)
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("teacher dataset has no manifest.json: $root")
    payload = bundle.attestation.payload
    _file_sha256(manifest_path) ==
        payload.teacher.manifest_sha256 ||
        error("sealed release/teacher manifest SHA-256 mismatch")
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.teacher_contract_sha256) ==
        payload.teacher.teacher_contract_sha256 ||
        error("sealed release/teacher contract SHA-256 mismatch")
    String(manifest.completion_state) == "complete" ||
        error("teacher manifest is incomplete")
    return manifest_path
end

"""
Extract the exact embedded frozen twin without opening any teacher shard.

This is an attestation-preserving format adapter, not an independent release
promotion.  The downstream final-V2 writer re-verifies source shards, the fixed
gate, and every cached primary target before publishing a bit-exact dataset.
"""
function extract_attested_frozen(
    sealed_artifact_path::AbstractString,
    frozen_output_path::AbstractString,
    dataset_root::AbstractString;
    require_production::Bool=true,
)
    sealed_path = abspath(sealed_artifact_path)
    output_path = abspath(frozen_output_path)
    sealed_path == output_path &&
        error("sealed input and bare-frozen output must differ")

    bundle = Sealed._load(sealed_path)
    _validate_attested_bundle(bundle; require_production)
    manifest_path = _validate_manifest_identity(bundle, dataset_root)

    parent = dirname(output_path)
    isdir(parent) || mkpath(parent)
    temporary = tempname(parent) * ".jld2"
    try
        Twin.save_frozen_official_elm(temporary, bundle.frozen)
        staged = Twin.load_frozen_official_elm(temporary)
        staged.parameter_sha256 == bundle.frozen.parameter_sha256 ||
            error("staged frozen parameter SHA-256 differs")
        staged.artifact_sha256 == bundle.frozen.artifact_sha256 ||
            error("staged frozen artifact SHA-256 differs")
        mv(temporary, output_path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end

    roundtrip = Twin.load_frozen_official_elm(output_path)
    roundtrip.parameter_sha256 == bundle.frozen.parameter_sha256 ||
        error("written frozen parameter SHA-256 differs")
    roundtrip.artifact_sha256 == bundle.frozen.artifact_sha256 ||
        error("written frozen artifact SHA-256 differs")
    return (;
        sealed_artifact_path=sealed_path,
        frozen_artifact_path=output_path,
        frozen_file_sha256=_file_sha256(output_path),
        parameter_sha256=roundtrip.parameter_sha256,
        artifact_sha256=roundtrip.artifact_sha256,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
        corrected_evaluator_source_sha256=
            Sealed.corrected_evaluator_source_sha256_v2(),
        teacher_manifest_path=manifest_path,
        teacher_manifest_sha256=
            bundle.attestation.payload.teacher.manifest_sha256,
        teacher_shards_opened=false,
    )
end

function _split_wrapper_arguments(arguments)
    sealed_artifact = strip(get(
        ENV,
        "HD_TWINPROP_SEALED_ARTIFACT",
        "",
    ))
    passthrough = String[]
    index = 1
    while index <= length(arguments)
        argument = String(arguments[index])
        startswith(argument, "--") ||
            error("unexpected positional argument: $argument")
        index < length(arguments) ||
            error("missing value for $argument")
        value = String(arguments[index + 1])
        if argument == "--sealed-artifact"
            sealed_artifact = value
        else
            push!(passthrough, argument, value)
        end
        index += 2
    end
    isempty(sealed_artifact) &&
        error(
            "--sealed-artifact or " *
            "HD_TWINPROP_SEALED_ARTIFACT is required",
        )
    return abspath(sealed_artifact), passthrough
end

function prepare_distillation_dataset_from_sealed(arguments=ARGS)
    sealed_artifact, passthrough =
        _split_wrapper_arguments(arguments)
    config = Writer.V6._parse_arguments(passthrough)
    extraction = extract_attested_frozen(
        sealed_artifact,
        config.frozen_twin_path,
        config.dataset_path;
        require_production=config.require_full_public_counts,
    )
    report = Writer.prepare_distillation_dataset_release(config)
    return merge(
        report,
        (;
            source_sealed_artifact=sealed_artifact,
            extracted_frozen=extraction,
            corrected_v2_bridge=true,
        ),
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_from_sealed(arguments)
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeOfficial1278SealedContractFixedV2Final

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeOfficial1278SealedContractFixedV2Final.main()
end
