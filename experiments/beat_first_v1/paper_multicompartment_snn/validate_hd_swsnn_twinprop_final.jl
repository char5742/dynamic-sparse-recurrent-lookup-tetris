module HDSWSNNTwinPropFinalValidation

using Dates
using JSON3

include(joinpath(@__DIR__, "validate_paper_mechanisms.jl"))
using .PaperMechanismValidation

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

const MODEL_FAMILY = "HD-SWSNN-TwinProp"
const VALIDATION_SCHEMA = "paper-mechanism-validation-v2"
const FINAL_TRAINING_SCHEMA = "hd-swsnn-twinprop-result-final-v1"
const MINIMUM_SCRATCH_UPDATES = 10_000

struct ValidationOptions
    parity_path::Union{Nothing,String}
    twin_path::Union{Nothing,String}
    trajectory_paths::Vector{String}
    distilled_path::Union{Nothing,String}
    training_path::String
    output_path::String
    dimensions::Vector{Int}
    strict::Bool
    run_kernel_contract::Bool
end

function _usage()
    return """
    Usage:
      julia --project=. paper_multicompartment_snn/validate_hd_swsnn_twinprop_final.jl [options]

    Required:
      --training PATH       Final HD-SWSNN-TwinProp scratch-training result.

    Evidence artifacts:
      --parity PATH         TwinProp parity JSON/JLD2 result.
      --twin PATH           Frozen digital-twin JLD2 or its JSON report.
      --trajectory PATHS    Semicolon-separated detailed trajectory artifacts.
      --distilled PATH      Frozen 11-state JLD2 or its JSON report.

    Other:
      --output PATH         Aggregate validation JSON.
      --dimensions 2,4,6    Trajectory dimensions lacking explicit metadata.
      --strict              Exit nonzero unless every required gate passes.
      --no-kernel-contract  Skip execution of the detailed-kernel contract.
      --help                Print this message.
    """
end

function parse_options(arguments=ARGS)
    values = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") ||
            throw(ArgumentError("unexpected positional argument: $argument"))
        name = argument[3:end]
        if name in ("strict", "no-kernel-contract", "help")
            push!(flags, name)
            index += 1
            continue
        end
        index < length(arguments) ||
            throw(ArgumentError("missing value for $argument"))
        haskey(values, name) &&
            throw(ArgumentError("option repeated: $argument"))
        values[name] = arguments[index + 1]
        index += 2
    end
    "help" in flags && return :help

    allowed = Set((
        "parity",
        "twin",
        "trajectory",
        "distilled",
        "training",
        "output",
        "dimensions",
    ))
    unknown = sort(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) ||
        throw(ArgumentError("unknown option(s): $(join(unknown, ", "))"))

    nullable(name) = begin
        value = strip(get(values, name, ""))
        isempty(value) ? nothing : abspath(value)
    end
    training = nullable("training")
    training === nothing &&
        throw(ArgumentError("--training PATH is required"))
    dimensions = parse.(
        Int,
        filter(!isempty, strip.(split(
            get(values, "dimensions", "2,4,6"),
            ',',
        ))),
    )
    isempty(dimensions) &&
        throw(ArgumentError("--dimensions must not be empty"))
    all(>(0), dimensions) ||
        throw(ArgumentError("--dimensions must be positive"))
    length(unique(dimensions)) == length(dimensions) ||
        throw(ArgumentError("--dimensions must not contain duplicates"))
    trajectory_value = strip(get(values, "trajectory", ""))
    trajectories = isempty(trajectory_value) ?
        String[] :
        abspath.(filter(
            !isempty,
            strip.(split(trajectory_value, ';')),
        ))
    output = abspath(get(
        values,
        "output",
        joinpath(
            @__DIR__,
            "results",
            "hd_swsnn_twinprop_final_validation.json",
        ),
    ))
    return ValidationOptions(
        nullable("parity"),
        nullable("twin"),
        trajectories,
        nullable("distilled"),
        training,
        output,
        dimensions,
        "strict" in flags,
        !("no-kernel-contract" in flags),
    )
end

@inline function _property(value, name::Symbol, default=nothing)
    value === nothing && return default
    if value isa AbstractDict
        haskey(value, name) && return value[name]
        text = String(name)
        haskey(value, text) && return value[text]
        return default
    end
    hasproperty(value, name) && return getproperty(value, name)
    return default
end

function _first_property(value, names, default=nothing)
    for name in names
        found = _property(value, Symbol(name), nothing)
        found === nothing || return found
    end
    return default
end

function _first_from(containers, names, default=nothing)
    for container in containers
        found = _first_property(container, names, nothing)
        found === nothing || return found
    end
    return default
end

function _unwrap(payload)
    wrapped = _first_property(payload, (:payload,), nothing)
    return wrapped === nothing ? payload : wrapped
end

function _nonempty_string(value)
    value === nothing && return nothing
    value isa Union{AbstractString,Symbol} || return nothing
    text = strip(String(value))
    return isempty(text) ? nothing : text
end

function _first_string(containers, names)
    value = _first_from(containers, names, nothing)
    return _nonempty_string(value)
end

function _finite_number(value)
    value isa Real || return nothing
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

function _first_number(containers, names)
    return _finite_number(_first_from(containers, names, nothing))
end

function _first_bool(containers, names)
    value = _first_from(containers, names, nothing)
    value isa Bool && return value
    value isa Integer && return value != 0
    return nothing
end

@inline _normalize_hash(value) = begin
    text = _nonempty_string(value)
    text === nothing ? nothing : lowercase(text)
end

@inline function _same_hash(left, right)
    normalized_left = _normalize_hash(left)
    normalized_right = _normalize_hash(right)
    return normalized_left !== nothing &&
        normalized_right !== nothing &&
        normalized_left == normalized_right
end

function _resolve_reference(report_path, payload, names)
    reference = _first_property(payload, names, nothing)
    reference = _nonempty_string(reference)
    reference === nothing && return nothing
    return isabspath(reference) ?
        normpath(reference) :
        normpath(joinpath(dirname(report_path), reference))
end

function _load_report_and_artifact(path, reference_names)
    report = PaperMechanismValidation.load_artifact(path)
    report_payload = _unwrap(report.payload)
    reference = lowercase(splitext(report.path)[2]) == ".json" ?
        _resolve_reference(report.path, report_payload, reference_names) :
        nothing
    artifact = reference === nothing ?
        report :
        PaperMechanismValidation.load_artifact(reference)
    return (; report, report_payload, artifact,
        artifact_payload=_unwrap(artifact.payload))
end

function _source_record(source)
    return (path=source.path, sha256=source.sha256)
end

function _metric(container, names)
    return PaperMechanismValidation._float_metric(container, names)
end

function _metric_container(containers)
    for container in containers
        candidate = _first_property(
            container,
            (:held_out_test, :heldout_test, :heldout, :test, :metrics),
            nothing,
        )
        candidate === nothing || return candidate
    end
    return first(containers)
end

function digital_twin_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="digital-twin artifact was not supplied",
    )
    loaded = _load_report_and_artifact(
        path,
        (:checkpoint_path, :artifact_path, :output),
    )
    frozen = _first_property(
        loaded.artifact_payload,
        (:frozen,),
        nothing,
    )
    metadata = _first_property(frozen, (:metadata,), nothing)
    integrity = _first_from(
        (
            loaded.report_payload,
            loaded.artifact_payload,
            frozen,
        ),
        (:frozen_integrity, :integrity),
        nothing,
    )
    containers = (
        loaded.report_payload,
        loaded.artifact_payload,
        frozen,
        metadata,
    )
    metrics = _metric_container((
        loaded.report_payload,
        metadata,
        loaded.artifact_payload,
    ))
    voltage_rmse = _metric(
        metrics,
        (:soma_voltage_rmse_mv, :voltage_rmse_mv, :voltage_rmse),
    )
    spike_auroc = _metric(metrics, (:spike_auroc,))
    spike_f1 = _metric(metrics, (:spike_f1,))
    nmda_rmse = _metric(metrics, (:nmda_rmse, :nmda_rmse_by_region))
    nmda_correlation = _metric(
        metrics,
        (:nmda_correlation, :nmda_correlation_by_region),
    )
    mechanism_hash = _first_string(
        containers,
        (:cell_mechanism_sha256, :detailed_kernel_hash),
    )
    morphology_hash = _first_string(
        containers,
        (:morphology_sha256, :morphology_hash),
    )
    semantic_artifact_hash = _first_string(
        containers,
        (:digital_twin_hash, :frozen_artifact_sha256, :artifact_sha256),
    )
    parameter_hash = _first_string(
        containers,
        (:frozen_parameter_sha256, :parameter_sha256),
    )
    frozen_flag = _first_bool(
        (loaded.report_payload, metadata, integrity),
        (:frozen_internal, :frozen),
    )
    integrity_frozen = _first_bool((integrity,), (:frozen, :passed))
    maximum_delta = _first_number(
        (integrity,),
        (:max_delta, :internal_max_delta),
    )
    integrity_parameter_hash = _first_string(
        (integrity,),
        (:parameter_sha256,),
    )
    integrity_artifact_hash = _first_string(
        (integrity,),
        (:artifact_sha256,),
    )
    frozen_integrity_passed =
        frozen_flag === true &&
        integrity_frozen === true &&
        maximum_delta == 0.0 &&
        _same_hash(parameter_hash, integrity_parameter_hash) &&
        _same_hash(semantic_artifact_hash, integrity_artifact_hash)
    metrics_passed =
        voltage_rmse !== nothing &&
        spike_auroc !== nothing &&
        nmda_rmse !== nothing &&
        voltage_rmse <= 5.0 &&
        spike_auroc >= 0.985 &&
        isfinite(nmda_rmse)
    lineage_present =
        mechanism_hash !== nothing &&
        morphology_hash !== nothing &&
        semantic_artifact_hash !== nothing &&
        parameter_hash !== nothing
    return (
        available=true,
        passed=metrics_passed && frozen_integrity_passed && lineage_present,
        report_source=_source_record(loaded.report),
        artifact_source=_source_record(loaded.artifact),
        report_references_artifact=
            loaded.report.path != loaded.artifact.path,
        hashes=(
            artifact_file_sha256=loaded.artifact.sha256,
            semantic_artifact_sha256=semantic_artifact_hash,
            parameter_sha256=parameter_hash,
        ),
        lineage=(
            cell_mechanism_sha256=mechanism_hash,
            morphology_sha256=morphology_hash,
            digital_twin_hash=semantic_artifact_hash,
        ),
        frozen_integrity=(
            passed=frozen_integrity_passed,
            frozen=frozen_flag,
            integrity_frozen,
            internal_max_delta=maximum_delta,
            parameter_sha256=integrity_parameter_hash,
            artifact_sha256=integrity_artifact_hash,
        ),
        metrics=(
            soma_voltage_rmse_mv=voltage_rmse,
            spike_auroc,
            spike_f1,
            nmda_rmse,
            nmda_correlation,
        ),
        thresholds=(
            spike_auroc_minimum=0.985,
            soma_voltage_rmse_mv_maximum=5.0,
            nmda_rmse_must_be_finite=true,
        ),
    )
end

function distilled_fidelity_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="distilled-11 artifact was not supplied",
    )
    loaded = _load_report_and_artifact(
        path,
        (:output, :artifact_path, :checkpoint_path),
    )
    provenance = _first_property(
        loaded.report_payload,
        (:provenance, :lineage),
        nothing,
    )
    artifact_provenance = _first_property(
        loaded.artifact_payload,
        (:provenance, :lineage),
        nothing,
    )
    containers = (
        loaded.report_payload,
        loaded.artifact_payload,
        provenance,
        artifact_provenance,
    )
    metrics_root = _first_from(
        (loaded.report_payload, loaded.artifact_payload),
        (:metrics,),
        loaded.artifact_payload,
    )
    test_metrics = _first_property(metrics_root, (:test,), metrics_root)
    voltage_rmse = _metric(
        test_metrics,
        (:soma_voltage_rmse_mv, :voltage_rmse_mv),
    )
    voltage_correlation = _metric(
        test_metrics,
        (:soma_voltage_correlation, :voltage_correlation),
    )
    spike_auroc = _metric(test_metrics, (:spike_auroc,))
    spike_f1 = _metric(test_metrics, (:spike_f1,))
    nmda_rmse = _metric(
        test_metrics,
        (:nmda_rmse_by_region, :nmda_rmse),
    )
    nmda_correlation = _metric(
        test_metrics,
        (:nmda_correlation_by_region, :nmda_correlation),
    )
    dendritic_rmse = _metric(
        test_metrics,
        (:dendritic_voltage_rmse_mv,),
    )
    calcium_auroc = _metric(test_metrics, (:calcium_event_auroc,))
    free_rollout_horizon = _metric(
        test_metrics,
        (:free_rollout_horizon,),
    )
    parameter_hash = _first_string(containers, (:parameter_sha256,))
    teacher_hash = _first_string(containers, (:teacher_sha256,))
    digital_twin_hash = _first_string(
        containers,
        (:digital_twin_hash, :digital_twin_sha256),
    )
    mechanism_hash = _first_string(
        containers,
        (:cell_mechanism_sha256, :detailed_kernel_hash),
    )
    morphology_hash = _first_string(
        containers,
        (:morphology_sha256, :morphology_hash),
    )
    frozen_internal = _first_bool(containers, (:frozen_internal,))
    claimed_artifact_hash = _first_string(
        (loaded.report_payload,),
        (:artifact_sha256,),
    )
    report_hash_passed = claimed_artifact_hash === nothing ||
        _same_hash(claimed_artifact_hash, loaded.artifact.sha256)
    gate = _first_from(
        (loaded.report_payload, loaded.artifact_payload),
        (:gate,),
        nothing,
    )
    artifact_gate_passed = _first_bool((gate,), (:passed,))
    metrics_passed =
        voltage_rmse !== nothing &&
        spike_auroc !== nothing &&
        nmda_correlation !== nothing &&
        dendritic_rmse !== nothing &&
        voltage_rmse <= 5.0 &&
        spike_auroc >= 0.95 &&
        nmda_correlation >= 0.80 &&
        dendritic_rmse <= 10.0
    lineage_present =
        parameter_hash !== nothing &&
        teacher_hash !== nothing &&
        digital_twin_hash !== nothing &&
        mechanism_hash !== nothing &&
        morphology_hash !== nothing
    frozen_integrity_passed =
        frozen_internal === true &&
        report_hash_passed &&
        artifact_gate_passed === true &&
        parameter_hash !== nothing
    return (
        available=true,
        passed=metrics_passed &&
            lineage_present &&
            frozen_integrity_passed,
        report_source=_source_record(loaded.report),
        artifact_source=_source_record(loaded.artifact),
        report_references_artifact=
            loaded.report.path != loaded.artifact.path,
        hashes=(
            artifact_file_sha256=loaded.artifact.sha256,
            claimed_artifact_sha256=claimed_artifact_hash,
            parameter_sha256=parameter_hash,
        ),
        lineage=(
            teacher_sha256=teacher_hash,
            digital_twin_hash,
            cell_mechanism_sha256=mechanism_hash,
            morphology_sha256=morphology_hash,
        ),
        frozen_integrity=(
            passed=frozen_integrity_passed,
            frozen_internal,
            report_artifact_hash_matches=report_hash_passed,
            artifact_gate_passed,
        ),
        metrics=(
            soma_voltage_rmse_mv=voltage_rmse,
            soma_voltage_correlation=voltage_correlation,
            spike_auroc,
            spike_f1,
            nmda_rmse,
            nmda_correlation,
            dendritic_voltage_rmse_mv=dendritic_rmse,
            calcium_event_auroc=calcium_auroc,
            free_rollout_horizon,
        ),
        thresholds=(
            soma_voltage_rmse_mv_maximum=5.0,
            spike_auroc_minimum=0.95,
            nmda_correlation_minimum=0.80,
            dendritic_voltage_rmse_mv_maximum=10.0,
        ),
    )
end

function _record_check!(checks, name, passed; observed=nothing, expected=nothing)
    push!(checks, (
        name=String(name),
        passed=Bool(passed),
        observed,
        expected,
    ))
    return Bool(passed)
end

function training_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="final training result was not supplied",
    )
    artifact = PaperMechanismValidation.load_artifact(path)
    payload = _unwrap(artifact.payload)
    run_config = _first_property(payload, (:run_config,), nothing)
    audit = _first_property(
        payload,
        (:frozen_internal_audit,),
        nothing,
    )
    checks = NamedTuple[]
    schema = _first_string((payload,), (:schema,))
    model_family = _first_string(
        (payload, run_config),
        (:model_family,),
    )
    completed = _first_bool((payload,), (:completed,))
    updates = _first_number((payload,), (:updates,))
    start_mode = _first_string((run_config,), (:start_mode,))
    target_updates = _first_number((run_config,), (:target_updates,))
    cell_mode = _first_string(
        (run_config, audit),
        (:cell_mode,),
    )
    frozen_internal = _first_bool(
        (run_config, audit),
        (:frozen_internal,),
    )
    audit_passed = _first_bool((audit,), (:passed,))
    maximum_delta = _first_number(
        (audit,),
        (:internal_max_delta, :max_delta),
    )
    initial_artifact_hash = _first_string(
        (audit,),
        (:initial_artifact_sha256, :initial_sha256),
    )
    final_artifact_hash = _first_string(
        (audit,),
        (:final_artifact_sha256, :final_sha256),
    )
    initial_parameter_hash = _first_string(
        (audit,),
        (:initial_parameter_sha256,),
    )
    final_parameter_hash = _first_string(
        (audit,),
        (:final_parameter_sha256,),
    )
    distilled_before = _first_string(
        (audit,),
        (:distilled_artifact_hash_before,),
    )
    distilled_after = _first_string(
        (audit,),
        (:distilled_artifact_hash_after,),
    )
    run_distilled_hash = _first_string(
        (run_config,),
        (:distilled_artifact_hash,),
    )
    run_parameter_hash = _first_string(
        (run_config,),
        (:internal_parameter_sha256,),
    )
    digital_twin_hash = _first_string(
        (run_config,),
        (:digital_twin_hash,),
    )
    mechanism_hash = _first_string(
        (run_config,),
        (:cell_mechanism_sha256,),
    )
    morphology_hash = _first_string(
        (run_config,),
        (:morphology_sha256, :morphology_hash),
    )

    _record_check!(
        checks,
        "final_training_schema",
        schema == FINAL_TRAINING_SCHEMA;
        observed=schema,
        expected=FINAL_TRAINING_SCHEMA,
    )
    _record_check!(
        checks,
        "official_model_family",
        model_family == MODEL_FAMILY;
        observed=model_family,
        expected=MODEL_FAMILY,
    )
    _record_check!(checks, "completed", completed === true)
    _record_check!(
        checks,
        "scratch_start",
        start_mode == "scratch";
        observed=start_mode,
        expected="scratch",
    )
    _record_check!(
        checks,
        "completed_at_least_10k",
        updates !== nothing && updates >= MINIMUM_SCRATCH_UPDATES;
        observed=updates,
        expected=MINIMUM_SCRATCH_UPDATES,
    )
    _record_check!(
        checks,
        "target_at_least_10k",
        target_updates !== nothing &&
            target_updates >= MINIMUM_SCRATCH_UPDATES,
        observed=target_updates,
        expected=MINIMUM_SCRATCH_UPDATES,
    )
    _record_check!(
        checks,
        "target_completed",
        updates !== nothing &&
            target_updates !== nothing &&
            updates >= target_updates,
    )
    _record_check!(
        checks,
        "distilled_frozen_cell_mode",
        cell_mode == "distilled-frozen";
        observed=cell_mode,
        expected="distilled-frozen",
    )
    _record_check!(
        checks,
        "frozen_internal_enabled",
        frozen_internal === true,
    )
    _record_check!(checks, "frozen_audit_passed", audit_passed === true)
    _record_check!(
        checks,
        "internal_max_delta_zero",
        maximum_delta == 0.0;
        observed=maximum_delta,
        expected=0.0,
    )
    _record_check!(
        checks,
        "artifact_hash_initial_final_identical",
        _same_hash(initial_artifact_hash, final_artifact_hash),
    )
    _record_check!(
        checks,
        "parameter_hash_initial_final_identical",
        _same_hash(initial_parameter_hash, final_parameter_hash),
    )
    _record_check!(
        checks,
        "distilled_hash_before_after_identical",
        _same_hash(distilled_before, distilled_after),
    )
    _record_check!(
        checks,
        "audit_artifact_matches_run_config",
        _same_hash(initial_artifact_hash, run_distilled_hash) &&
            _same_hash(distilled_before, run_distilled_hash),
    )
    _record_check!(
        checks,
        "audit_parameter_matches_run_config",
        _same_hash(initial_parameter_hash, run_parameter_hash),
    )
    _record_check!(
        checks,
        "training_lineage_present",
        digital_twin_hash !== nothing &&
            mechanism_hash !== nothing &&
            morphology_hash !== nothing,
    )
    passed = all(check -> check.passed, checks)
    return (
        available=true,
        passed,
        source=_source_record(artifact),
        checks,
        model_family,
        schema,
        completed,
        updates,
        run_config=(
            start_mode,
            target_updates,
            cell_mode,
            frozen_internal,
        ),
        lineage=(
            digital_twin_hash,
            distilled_artifact_hash=run_distilled_hash,
            internal_parameter_sha256=run_parameter_hash,
            cell_mechanism_sha256=mechanism_hash,
            morphology_sha256=morphology_hash,
        ),
        frozen_internal_audit=(
            passed=audit_passed,
            internal_max_delta=maximum_delta,
            initial_artifact_sha256=initial_artifact_hash,
            final_artifact_sha256=final_artifact_hash,
            initial_parameter_sha256=initial_parameter_hash,
            final_parameter_sha256=final_parameter_hash,
            distilled_artifact_hash_before=distilled_before,
            distilled_artifact_hash_after=distilled_after,
        ),
        gates=(
            scratch_10k=all(check ->
                check.passed,
                filter(check -> check.name in (
                    "completed",
                    "scratch_start",
                    "completed_at_least_10k",
                    "target_at_least_10k",
                    "target_completed",
                ), checks)),
            internal_max_delta_zero=maximum_delta == 0.0,
            frozen_hashes_unchanged=
                _same_hash(initial_artifact_hash, final_artifact_hash) &&
                _same_hash(initial_parameter_hash, final_parameter_hash) &&
                _same_hash(distilled_before, distilled_after),
        ),
    )
end

function artifact_chain_summary(twin, distilled, training)
    detailed_kernel_path = joinpath(@__DIR__, "PaperHayCell.jl")
    detailed_kernel_hash =
        PaperMechanismValidation._sha256_file(detailed_kernel_path)
    checks = NamedTuple[]
    twin_mechanism = twin.available ?
        twin.lineage.cell_mechanism_sha256 : nothing
    distilled_mechanism = distilled.available ?
        distilled.lineage.cell_mechanism_sha256 : nothing
    training_mechanism = training.available ?
        training.lineage.cell_mechanism_sha256 : nothing
    _record_check!(
        checks,
        "detailed_kernel_sha_chain",
        _same_hash(detailed_kernel_hash, twin_mechanism) &&
            _same_hash(detailed_kernel_hash, distilled_mechanism) &&
            _same_hash(detailed_kernel_hash, training_mechanism);
        observed=(
            paper_hay_cell=detailed_kernel_hash,
            twin=twin_mechanism,
            distilled=distilled_mechanism,
            training=training_mechanism,
        ),
    )

    twin_morphology = twin.available ?
        twin.lineage.morphology_sha256 : nothing
    distilled_morphology = distilled.available ?
        distilled.lineage.morphology_sha256 : nothing
    training_morphology = training.available ?
        training.lineage.morphology_sha256 : nothing
    _record_check!(
        checks,
        "morphology_sha_chain",
        _same_hash(twin_morphology, distilled_morphology) &&
            _same_hash(twin_morphology, training_morphology);
        observed=(
            twin=twin_morphology,
            distilled=distilled_morphology,
            training=training_morphology,
        ),
    )

    twin_file_hash = twin.available ?
        twin.artifact_source.sha256 : nothing
    distilled_teacher_hash = distilled.available ?
        distilled.lineage.teacher_sha256 : nothing
    distilled_twin_hash = distilled.available ?
        distilled.lineage.digital_twin_hash : nothing
    training_twin_hash = training.available ?
        training.lineage.digital_twin_hash : nothing
    _record_check!(
        checks,
        "twin_file_to_distill_to_training",
        _same_hash(twin_file_hash, distilled_teacher_hash) &&
            _same_hash(twin_file_hash, distilled_twin_hash) &&
            _same_hash(twin_file_hash, training_twin_hash);
        observed=(
            twin_artifact_file=twin_file_hash,
            distilled_teacher=distilled_teacher_hash,
            distilled_digital_twin=distilled_twin_hash,
            training_digital_twin=training_twin_hash,
        ),
    )

    distilled_file_hash = distilled.available ?
        distilled.artifact_source.sha256 : nothing
    training_distilled_hash = training.available ?
        training.lineage.distilled_artifact_hash : nothing
    _record_check!(
        checks,
        "distilled_file_to_training",
        _same_hash(distilled_file_hash, training_distilled_hash);
        observed=(
            distilled_artifact_file=distilled_file_hash,
            training_distilled_artifact=training_distilled_hash,
        ),
    )

    distilled_parameter_hash = distilled.available ?
        distilled.hashes.parameter_sha256 : nothing
    training_parameter_hash = training.available ?
        training.lineage.internal_parameter_sha256 : nothing
    _record_check!(
        checks,
        "distilled_parameter_to_training_internal",
        _same_hash(distilled_parameter_hash, training_parameter_hash);
        observed=(
            distilled_parameter=distilled_parameter_hash,
            training_internal_parameter=training_parameter_hash,
        ),
    )
    return (
        available=twin.available && distilled.available && training.available,
        passed=all(check -> check.passed, checks),
        checks,
        detailed_kernel_source=(
            path=abspath(detailed_kernel_path),
            sha256=detailed_kernel_hash,
        ),
    )
end

function aggregate_validation(options::ValidationOptions)
    kernel = options.run_kernel_contract ?
        PaperMechanismValidation.canonical_kernel_contract() :
        (
            available=false,
            passed=false,
            reason="kernel contract disabled by CLI",
        )
    twin = digital_twin_summary(options.twin_path)
    parity = PaperMechanismValidation.parity_summary(options.parity_path)
    trajectory = PaperMechanismValidation.trajectory_summary(
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
        final_scratch_10k_training=training.passed,
        artifact_lineage_chain=artifact_chain.passed,
        frozen_internal_max_delta_zero=
            training.available &&
            training.gates.internal_max_delta_zero,
    )
    return (
        schema=VALIDATION_SCHEMA,
        model_family=MODEL_FAMILY,
        definition=(
            "detailed Hay multicompartment mechanism -> multi-target " *
            "digital twin -> distilled 11-state internal cell -> " *
            "frozen internal cell during HD-SWSNN task training"
        ),
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        disclosure=(
            "This validates the repository's paper-mechanism " *
            "reconstruction; unpublished author code identity is not claimed."
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
        println(_usage())
        return nothing
    end
    result = aggregate_validation(options)
    mkpath(dirname(options.output_path))
    open(options.output_path, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "HD-SWSNN-TwinProp validation passed=$(result.passed) " *
        "output=$(options.output_path)",
    )
    if options.strict && !result.passed
        error("HD-SWSNN-TwinProp final validation gates failed")
    end
    return result
end

end # module HDSWSNNTwinPropFinalValidation

if abspath(PROGRAM_FILE) == @__FILE__
    HDSWSNNTwinPropFinalValidation.main()
end
