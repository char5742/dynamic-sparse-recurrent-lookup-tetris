module FinalizePublicSpielerM100NMDA4Heldout

using JLD2
using SHA

if !isdefined(Main, :PublicSpielerM100NMDA4SealedPath)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PublicSpielerM100NMDA4SealedPath.jl",
        ),
    )
end
const Path = Main.PublicSpielerM100NMDA4SealedPath
const Evaluator = Path.Evaluator
const Twin = Path.Twin

const HELDOUT_SCHEMA =
    "hd_swsnn.spieler_public_m100_nmda4.heldout.final.v1"
const HELDOUT_ARTIFACT_KIND =
    "SealedPublicSpielerM100NMDA4HeldoutV1"
const HELDOUT_FORMAT_VERSION = 1

struct PublicM100NMDA4HeldoutRelease{F,P}
    frozen::F
    payload::P
    attestation_sha256::String
end

_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _claim_once(path, validation_sha256)
    claim = abspath(String(path)) * ".heldout_once.claim"
    ispath(claim) &&
        error("held-out finalization was already claimed: $claim")
    parent = dirname(claim)
    isdir(parent) || mkpath(parent)
    io = Base.Filesystem.open(
        claim,
        Base.JL_O_CREAT | Base.JL_O_EXCL | Base.JL_O_WRONLY,
        0o600,
    )
    try
        println(io, "schema=heldout-once-claim-v1")
        println(io, "validation_attestation_sha256=$validation_sha256")
        println(io, "state=claimed-before-heldout-target-decode")
    finally
        close(io)
    end
    return claim
end

function _verify_same_dataset!(dataset, validation)
    payload = validation.payload
    dataset.manifest_sha256 == payload.teacher.manifest_sha256 ||
        error("finalizer manifest differs from validation")
    dataset.source_dataset_sha256 ==
        payload.teacher.source_dataset_sha256 ||
        error("finalizer dataset differs from validation")
    dataset.teacher_contract_sha256 ==
        payload.teacher.teacher_contract_sha256 ||
        error("finalizer teacher contract differs from validation")
    Evaluator.canonical_sha256(dataset.fit_ids) ==
        payload.split.fit_ids_sha256 ||
        error("finalizer fit split differs from validation")
    Evaluator.canonical_sha256(dataset.validation_ids) ==
        payload.split.validation_ids_sha256 ||
        error("finalizer validation split differs")
    Evaluator.canonical_sha256(dataset.heldout_ids) ==
        payload.split.heldout_ids_sha256 ||
        error("finalizer held-out split differs")
    return nothing
end

function finalize_heldout(
    validation_artifact_path,
    manifest_path,
    shard_directory,
    output_path;
    scratch_root=nothing,
)
    validation =
        Path.load_validation_qualified_candidate(validation_artifact_path)
    destination = abspath(String(output_path))
    ispath(destination) &&
        error("refusing to overwrite held-out release: $destination")
    dataset = Evaluator._verify_manifest_and_shards(
        manifest_path,
        shard_directory,
    )
    _verify_same_dataset!(dataset, validation)

    # Persistent, fail-closed claim is written before the first held-out target
    # decode. A crash cannot silently turn this into another selection pass.
    claim = _claim_once(destination, validation.attestation_sha256)
    audit = Evaluator._HeldoutEvaluationAudit()
    evaluation = Evaluator._with_scratch(scratch_root) do scratch
        Evaluator._evaluate(
            dataset,
            validation.frozen,
            scratch,
            audit,
        )
    end
    audit.metric_evaluations_after_selection == 1 ||
        error("held-out evaluator did not run exactly once")
    gate = Evaluator._gate(evaluation.metrics)
    evaluator = Evaluator._contract_fixed_evaluator_v2((;
        split_role="heldout",
        gate_thresholds_unchanged=true,
        minimum_spike_auroc=Evaluator.MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=Evaluator.MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_regional_nmda_normalized_rmse=
            Evaluator.MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
    ))
    payload = (;
        schema=HELDOUT_SCHEMA,
        artifact_kind=HELDOUT_ARTIFACT_KIND,
        profile=String(Path.PUBLIC_M100_NMDA4_PROFILE),
        validation=(;
            artifact_file_sha256=
                _file_sha256(validation_artifact_path),
            attestation_sha256=validation.attestation_sha256,
            validation_passed=true,
        ),
        teacher=validation.payload.teacher,
        split=validation.payload.split,
        model=validation.payload.model,
        evaluator,
        heldout_metrics=evaluation.metrics,
        heldout_gate=gate,
        outcome=(;
            heldout_evaluated_after_validation=true,
            heldout_metric_evaluations_after_selection=
                audit.metric_evaluations_after_selection,
            gate_passed=gate.passed,
            paper_scale_claimed=false,
            paper_m1000_release_claimed=false,
            promotable_as_paper_m1000_release=false,
        ),
        one_shot=(;
            claim_file=basename(claim),
            claim_written_before_target_decode=true,
            validation_was_required=true,
        ),
    )
    digest = Evaluator.canonical_sha256(payload)
    release = PublicM100NMDA4HeldoutRelease(
        validation.frozen,
        payload,
        digest,
    )
    parent = dirname(destination)
    isdir(parent) || mkpath(parent)
    jldsave(
        destination;
        artifact_kind=HELDOUT_ARTIFACT_KIND,
        format_version=HELDOUT_FORMAT_VERSION,
        release,
    )
    open(claim, "a") do io
        println(io, "state=sealed")
        println(io, "heldout_attestation_sha256=$digest")
        println(io, "artifact_file_sha256=$(_file_sha256(destination))")
    end
    return release
end

function _parse_args(args)
    values = Dict{String,String}()
    index = 1
    while index <= length(args)
        startswith(args[index], "--") ||
            error("unexpected argument: $(args[index])")
        index < length(args) ||
            error("missing value for $(args[index])")
        values[args[index][3:end]] = args[index + 1]
        index += 2
    end
    for name in ("validation", "manifest", "shards", "output")
        haskey(values, name) || error("--$name is required")
    end
    return values
end

function main(args=ARGS)
    values = _parse_args(args)
    release = finalize_heldout(
        values["validation"],
        values["manifest"],
        values["shards"],
        values["output"];
        scratch_root=get(values, "scratch-root", nothing),
    )
    println("heldout_attestation_sha256=" *
        release.attestation_sha256)
    println("gate_passed=" *
        string(release.payload.outcome.gate_passed))
    return release
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    FinalizePublicSpielerM100NMDA4Heldout.main(ARGS)
end
