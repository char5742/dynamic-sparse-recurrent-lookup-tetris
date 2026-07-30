module HDSWSNNTwinPropReleaseGate

using Dates
using JSON3

include(joinpath(@__DIR__, "validate_hd_swsnn_twinprop_final.jl"))
using .HDSWSNNTwinPropFinalValidation

const Core = HDSWSNNTwinPropFinalValidation
const Legacy = Core.PaperMechanismValidation

export MODEL_FAMILY,
    VALIDATION_SCHEMA,
    ValidationOptions,
    aggregate_validation,
    artifact_chain_summary,
    digital_twin_summary,
    distilled_fidelity_summary,
    main,
    parse_options,
    training_summary

const MODEL_FAMILY = Core.MODEL_FAMILY
const VALIDATION_SCHEMA = "paper-mechanism-validation-release-v1"
const ValidationOptions = Core.ValidationOptions
const parse_options = Core.parse_options

function _valid_sha256(value)
    value isa AbstractString || return false
    return occursin(r"^[0-9a-fA-F]{64}$", strip(String(value)))
end

function _state_count(path::AbstractString)
    loaded = Core._load_report_and_artifact(
        path,
        (:output, :artifact_path, :checkpoint_path),
    )
    parameters = Core._first_property(
        loaded.artifact_payload,
        (:parameters,),
        nothing,
    )
    config = Core._first_property(
        loaded.artifact_payload,
        (:config,),
        nothing,
    )
    explicit = Core._first_number(
        (loaded.report_payload, loaded.artifact_payload),
        (:state_count, :state_dim, :internal_state_dim),
    )
    if explicit !== nothing
        rounded = round(Int, explicit)
        return explicit == rounded ? rounded : nothing
    end
    initial_state = Core._first_property(
        parameters,
        (:initial_state,),
        nothing,
    )
    initial_state isa AbstractArray && return length(initial_state)
    semantics = Core._first_property(
        config,
        (:state_semantics,),
        nothing,
    )
    semantics isa Union{Tuple,AbstractArray} &&
        return length(semantics)
    return nothing
end

function digital_twin_summary(path::Union{Nothing,AbstractString})
    summary = Core.digital_twin_summary(path)
    summary.available || return merge(summary, (
        release_hashes_valid=false,
        passed=false,
    ))
    hashes_valid = all(_valid_sha256, (
        summary.artifact_source.sha256,
        summary.hashes.semantic_artifact_sha256,
        summary.hashes.parameter_sha256,
        summary.lineage.cell_mechanism_sha256,
        summary.lineage.morphology_sha256,
        summary.lineage.digital_twin_hash,
    ))
    return merge(summary, (
        release_hashes_valid=hashes_valid,
        passed=summary.passed && hashes_valid,
    ))
end

function distilled_fidelity_summary(
    path::Union{Nothing,AbstractString},
)
    summary = Core.distilled_fidelity_summary(path)
    if !summary.available || path === nothing
        return merge(summary, (
            state_count=nothing,
            release_hashes_valid=false,
            release_state_count_is_11=false,
            passed=false,
        ))
    end
    state_count = _state_count(path)
    hashes_valid = all(_valid_sha256, (
        summary.artifact_source.sha256,
        summary.hashes.parameter_sha256,
        summary.lineage.teacher_sha256,
        summary.lineage.digital_twin_hash,
        summary.lineage.cell_mechanism_sha256,
        summary.lineage.morphology_sha256,
    ))
    state_count_is_11 = state_count == 11
    return merge(summary, (
        state_count,
        release_hashes_valid=hashes_valid,
        release_state_count_is_11=state_count_is_11,
        passed=summary.passed && hashes_valid && state_count_is_11,
    ))
end

function training_summary(path::Union{Nothing,AbstractString})
    summary = Core.training_summary(path)
    summary.available || return merge(summary, (
        release_hashes_valid=false,
        passed=false,
    ))
    audit = summary.frozen_internal_audit
    hashes_valid = all(_valid_sha256, (
        summary.source.sha256,
        summary.lineage.digital_twin_hash,
        summary.lineage.distilled_artifact_hash,
        summary.lineage.internal_parameter_sha256,
        summary.lineage.cell_mechanism_sha256,
        summary.lineage.morphology_sha256,
        audit.initial_artifact_sha256,
        audit.final_artifact_sha256,
        audit.initial_parameter_sha256,
        audit.final_parameter_sha256,
        audit.distilled_artifact_hash_before,
        audit.distilled_artifact_hash_after,
    ))
    return merge(summary, (
        release_hashes_valid=hashes_valid,
        passed=summary.passed && hashes_valid,
    ))
end

function artifact_chain_summary(twin, distilled, training)
    chain = Core.artifact_chain_summary(twin, distilled, training)
    release_hashes_valid =
        getproperty(twin, :release_hashes_valid) &&
        getproperty(distilled, :release_hashes_valid) &&
        getproperty(training, :release_hashes_valid)
    return merge(chain, (
        release_hashes_valid,
        passed=chain.passed && release_hashes_valid,
    ))
end

function aggregate_validation(options::ValidationOptions)
    kernel = options.run_kernel_contract ?
        Legacy.canonical_kernel_contract() :
        (
            available=false,
            passed=false,
            reason="kernel contract disabled by CLI",
        )
    twin = digital_twin_summary(options.twin_path)
    parity = Legacy.parity_summary(options.parity_path)
    trajectory = Legacy.trajectory_summary(
        options.trajectory_paths,
        options.dimensions,
    )
    distilled = distilled_fidelity_summary(options.distilled_path)
    training = training_summary(options.training_path)
    artifact_chain = artifact_chain_summary(twin, distilled, training)
    required_gates = (
        canonical_detailed_kernel=kernel.passed,
        digital_twin_heldout_fidelity=twin.passed,
        parity_and_retrained_ablation=parity.passed,
        high_rank_voltage_and_nmda_scaling=trajectory.passed,
        distilled11_trajectory_fidelity=distilled.passed,
        distilled_state_count_exactly_11=
            distilled.release_state_count_is_11,
        final_scratch_10k_training=training.passed,
        artifact_lineage_chain=artifact_chain.passed,
        all_lineage_hashes_are_sha256=
            artifact_chain.release_hashes_valid,
        frozen_internal_max_delta_zero=
            training.available &&
            training.gates.internal_max_delta_zero,
    )
    return (
        schema=VALIDATION_SCHEMA,
        model_family=MODEL_FAMILY,
        definition=(
            "detailed Hay multicompartment mechanism -> multi-target " *
            "digital twin -> exact 11-state distillation -> frozen " *
            "internal cell during HD-SWSNN task training"
        ),
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        disclosure=(
            "Release gate for the repository reconstruction; identity " *
            "with unpublished author code is not claimed."
        ),
        options=(
            parity_path=options.parity_path,
            twin_path=options.twin_path,
            trajectory_paths=options.trajectory_paths,
            distilled_path=options.distilled_path,
            training_path=options.training_path,
            dimensions=options.dimensions,
            strict=options.strict,
        ),
        kernel_contract=kernel,
        digital_twin=twin,
        parity,
        detailed_trajectory=trajectory,
        distilled11_fidelity=distilled,
        final_training=training,
        artifact_chain,
        required_gates,
        passed=all(values(required_gates)),
    )
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    if options === :help
        println(Core._usage())
        return nothing
    end
    result = aggregate_validation(options)
    mkpath(dirname(options.output_path))
    open(options.output_path, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "HD-SWSNN-TwinProp release gate passed=$(result.passed) " *
        "output=$(options.output_path)",
    )
    if options.strict && !result.passed
        error("HD-SWSNN-TwinProp release gates failed")
    end
    return result
end

end # module HDSWSNNTwinPropReleaseGate

if abspath(PROGRAM_FILE) == @__FILE__
    HDSWSNNTwinPropReleaseGate.main()
end
