module PaperMechanismValidation

using Dates
using JLD2
using JSON3
using LinearAlgebra
using SHA
using Statistics

include(joinpath(@__DIR__, "PaperHayCell.jl"))
using .PaperHayCell

export VALIDATION_SCHEMA,
    ValidationOptions,
    aggregate_validation,
    canonical_kernel_contract,
    digital_twin_summary,
    distilled_fidelity_summary,
    load_artifact,
    main,
    nmda_recruitment_metrics,
    parity_summary,
    parse_options,
    trajectory_summary,
    voltage_pca_metrics

const VALIDATION_SCHEMA = "paper-mechanism-validation-v1"
const PAPER_REFERENCE = (
    full_4bit=0.994,
    passive_4bit=0.781,
    soma_only_4bit=0.769,
    no_nmda_4bit=0.738,
    twin_spike_auroc=0.98576,
)
const DEFAULT_DIMENSIONS = Int[2, 4, 6]

struct ValidationOptions
    parity_path::Union{Nothing,String}
    twin_path::Union{Nothing,String}
    trajectory_paths::Vector{String}
    distilled_path::Union{Nothing,String}
    output_path::String
    dimensions::Vector{Int}
    strict::Bool
    run_kernel_contract::Bool
end

function _usage()
    return """
    Usage:
      julia --project=. paper_multicompartment_snn/validate_paper_mechanisms.jl [options]

    Options:
      --parity PATH         TwinPropParity JSON/JLD2 result.
      --twin PATH           Digital-twin held-out metrics JSON/JLD2 result.
      --trajectory PATHS    Semicolon-separated detailed trajectory shards or
                            a directory/manifest. Expected arrays have shape
                            compartment x time x sample.
      --distilled PATH      Frozen 11-state distillation JLD2/metrics JSON.
      --output PATH         Aggregate JSON output.
      --dimensions 2,4,6    Dimensions assigned to trajectory paths that do
                            not carry an explicit dimension.
      --strict              Exit nonzero unless every required gate passes.
      --no-kernel-contract  Do not execute the canonical detailed kernel.
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

    nullable(name) = begin
        value = strip(get(values, name, ""))
        isempty(value) ? nothing : abspath(value)
    end
    trajectory_value = strip(get(values, "trajectory", ""))
    trajectory_paths = isempty(trajectory_value) ?
        String[] :
        abspath.(filter(!isempty, strip.(split(trajectory_value, ';'))))
    output = abspath(get(
        values,
        "output",
        joinpath(
            @__DIR__,
            "results",
            "paper_mechanism_validation.json",
        ),
    ))
    return ValidationOptions(
        nullable("parity"),
        nullable("twin"),
        trajectory_paths,
        nullable("distilled"),
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

function _children(value)
    if value isa AbstractDict
        return collect(values(value))
    elseif value isa NamedTuple
        return collect(values(value))
    elseif value isa JSON3.Object
        return [getproperty(value, name) for name in propertynames(value)]
    end
    return Any[]
end

function _find_recursive(value, names; depth=0, maximum_depth=5)
    depth > maximum_depth && return nothing
    found = _first_property(value, names, nothing)
    found === nothing || return found
    for child in _children(value)
        nested = _find_recursive(
            child,
            names;
            depth=depth + 1,
            maximum_depth,
        )
        nested === nothing || return nested
    end
    return nothing
end

@inline function _finite_float(value)
    value isa Real || return nothing
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

function _float_metric(value, names)
    found = _find_recursive(value, names)
    scalar = _finite_float(found)
    scalar === nothing || return scalar
    if found isa AbstractArray
        finite = Float64[
            Float64(item)
            for item in found
            if item isa Real && isfinite(Float64(item))
        ]
        isempty(finite) || return mean(finite)
    end
    return nothing
end

function _bool_metric(value, names)
    found = _find_recursive(value, names)
    found isa Bool && return found
    found isa Integer && return found != 0
    return nothing
end

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function load_artifact(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("artifact does not exist: $source")
    extension = lowercase(splitext(source)[2])
    payload = if extension == ".json"
        JSON3.read(read(source, String))
    elseif extension in (".jld2", ".jld")
        JLD2.load(source)
    else
        error("unsupported artifact extension $extension: $source")
    end
    return (
        path=source,
        sha256=_sha256_file(source),
        payload,
    )
end

function _record_check!(checks, name, passed; observed=nothing, expected=nothing)
    push!(
        checks,
        (
            name=String(name),
            passed=Bool(passed),
            observed,
            expected,
        ),
    )
    return passed
end

function _active_channel_total(parameters)
    total = 0.0
    for name in (
        :gbar_nat,
        :gbar_nap,
        :gbar_kp,
        :gbar_kt,
        :gbar_skv3,
        :gbar_im,
        :gbar_ih,
        :gbar_cahva,
        :gbar_calva,
        :gbar_skca,
    )
        total += sum(abs, getproperty(parameters, name))
    end
    return total
end

function _runtime_ablation_probe(tree, mode::Symbol)
    parameters = HayParameters(tree; ablation=mode)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    target = Int(first(tree.apical_hot_zone))
    add_synaptic_event!(
        drive,
        target;
        ampa=0.10f0,
        nmda=0.10f0,
    )
    nmda_inward_sum = 0.0
    active_current_sum = 0.0
    calcium_events = 0
    soma_spikes = 0
    soma_peak = -Inf
    for time in 1:50
        time == 1 || reset_drive!(drive)
        soma_spikes += hay_cell_step!(
            state,
            drive,
            diagnostics,
            tree,
            parameters,
        ) > 0.5f0
        nmda_inward_sum += sum(value -> max(-Float64(value), 0.0),
            diagnostics.nmda_current)
        active_current_sum += sum(abs, diagnostics.nat_current)
        active_current_sum += sum(abs, diagnostics.cahva_current)
        active_current_sum += sum(abs, diagnostics.calva_current)
        calcium_events += count(>(0.5f0), state.local_ca_event)
        soma_peak = max(soma_peak, Float64(state.voltage_mv[Int(tree.soma)]))
    end
    return (
        mode=String(mode),
        nmda_inward_sum,
        active_current_sum,
        calcium_events,
        soma_spikes,
        soma_peak_mv=soma_peak,
        finite=all(isfinite, state.voltage_mv) &&
            all(isfinite, state.intracellular_calcium),
    )
end

"""
Run the non-negotiable detailed-kernel contract directly against the canonical
`PaperHayCell` implementation. This does not infer paper fidelity from a
training log.
"""
function canonical_kernel_contract()
    checks = NamedTuple[]
    tree = paper_hay_tree()
    count = compartment_count(tree)
    full = HayParameters(tree; ablation=:full)
    passive = HayParameters(tree; ablation=:passive)
    no_nmda = HayParameters(tree; ablation=:no_nmda)
    soma_only = HayParameters(tree; ablation=:soma_only)

    _record_check!(
        checks,
        "multicompartment_tree",
        count >= 16;
        observed=count,
        expected="at least 16 cable-coupled compartments",
    )
    parent_order = tree.parent[1] == 0 && all(
        index -> 1 <= tree.parent[index] < index,
        2:count,
    )
    _record_check!(checks, "parent_before_child_tree", parent_order)
    regions = Set(tree.region)
    _record_check!(
        checks,
        "basal_apical_morphology",
        all(region -> region in regions, (
            SOMA,
            BASAL,
            APICAL_TRUNK,
            APICAL_TUFT,
        )),
    )
    _record_check!(
        checks,
        "apical_calcium_hot_zone",
        !isempty(tree.apical_hot_zone) &&
            all(index ->
                full.gbar_cahva[index] > 0 &&
                full.gbar_calva[index] > 0,
                Int.(tree.apical_hot_zone)),
    )
    _record_check!(
        checks,
        "axial_cable_coupling",
        all(>(0.0f0), @view(full.axial_conductance_ns[2:end])),
    )
    _record_check!(
        checks,
        "all_hay_channel_families_active",
        _active_channel_total(full) > 0.0,
    )
    _record_check!(
        checks,
        "passive_removes_voltage_gated_channels",
        _active_channel_total(passive) == 0.0,
    )
    _record_check!(
        checks,
        "no_nmda_retains_voltage_gated_channels",
        _active_channel_total(no_nmda) > 0.0 &&
            _active_channel_total(no_nmda) ≈
                _active_channel_total(full),
    )
    _record_check!(
        checks,
        "no_nmda_zero_maximum_conductance",
        no_nmda.nmda_max_ns == 0.0f0,
    )
    _record_check!(
        checks,
        "passive_retains_nmda",
        passive.nmda_max_ns == full.nmda_max_ns == 0.30f0,
    )
    soma = Int(tree.soma)
    soma_only_contract =
        all(iszero, @view(soma_only.axial_conductance_ns[2:end])) &&
        all(iszero, @view(soma_only.g_pas[2:end])) &&
        soma_only.gbar_nat[soma] > 0.0f0 &&
        all(iszero, @view(soma_only.gbar_nat[2:end]))
    _record_check!(
        checks,
        "soma_only_collapses_dendritic_dynamics",
        soma_only_contract,
    )
    _record_check!(
        checks,
        "paper_receptor_maxima",
        full.ampa_max_ns == 0.40f0 &&
            full.nmda_max_ns == 0.30f0 &&
            full.gaba_max_ns == 0.70f0;
        observed=(
            ampa=full.ampa_max_ns,
            nmda=full.nmda_max_ns,
            gaba=full.gaba_max_ns,
        ),
        expected=(ampa=0.40, nmda=0.30, gaba=0.70),
    )
    _record_check!(
        checks,
        "paper_time_resolution",
        full.outer_dt_ms == 1.0f0 &&
            full.substeps >= 1 &&
            full.substeps * full.substep_dt_ms ≈ full.outer_dt_ms,
    )
    block_negative = nmda_magnesium_block(-80.0f0)
    block_depolarized = nmda_magnesium_block(-20.0f0)
    _record_check!(
        checks,
        "voltage_dependent_nmda_magnesium_block",
        0.0f0 < block_negative < block_depolarized < 1.0f0;
        observed=(at_minus_80=block_negative, at_minus_20=block_depolarized),
    )

    negative_rejected = false
    drive = HaySynapticDrive(tree)
    try
        add_synaptic_event!(drive, soma; ampa=-0.1f0)
    catch error
        negative_rejected = error isa ArgumentError
    end
    _record_check!(
        checks,
        "nonnegative_conductance_events",
        negative_rejected,
    )

    probes = Dict(
        mode => _runtime_ablation_probe(tree, mode)
        for mode in (:full, :passive, :no_nmda, :soma_only)
    )
    _record_check!(
        checks,
        "full_runtime_nmda_current",
        probes[:full].nmda_inward_sum > 0.0,
    )
    _record_check!(
        checks,
        "passive_runtime_retains_nmda_current",
        probes[:passive].nmda_inward_sum > 0.0,
    )
    _record_check!(
        checks,
        "no_nmda_runtime_current_is_zero",
        probes[:no_nmda].nmda_inward_sum == 0.0,
    )
    _record_check!(
        checks,
        "passive_runtime_active_current_is_zero",
        probes[:passive].active_current_sum == 0.0,
    )
    _record_check!(
        checks,
        "all_ablation_probes_finite",
        all(probe -> probe.finite, values(probes)),
    )

    return (
        available=true,
        passed=all(check -> check.passed, checks),
        compartment_count=count,
        checks,
        probes=collect(values(probes)),
        readout_contract=(
            exported_event="HayState.soma_spike",
            internal_only=(
                "voltage_mv",
                "intracellular_calcium",
                "local_ca_event",
            ),
            note="task artifacts must independently attest soma_spike_only",
        ),
    )
end

function _metric_payload(artifact)
    payload = artifact.payload
    wrapped = _first_property(payload, (:payload,), nothing)
    return wrapped === nothing ? payload : wrapped
end

function digital_twin_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="digital-twin artifact was not supplied",
    )
    artifact = load_artifact(path)
    payload = _metric_payload(artifact)
    metrics = _find_recursive(payload, (:heldout, :test, :metrics))
    metrics === nothing && (metrics = payload)
    voltage_rmse = _float_metric(
        metrics,
        (:soma_voltage_rmse_mv, :voltage_rmse_mv, :voltage_rmse),
    )
    spike_auroc = _float_metric(metrics, (:spike_auroc,))
    spike_f1 = _float_metric(metrics, (:spike_f1,))
    nmda_rmse = _float_metric(
        metrics,
        (:nmda_rmse, :nmda_rmse_by_region),
    )
    nmda_correlation = _float_metric(
        metrics,
        (:nmda_correlation, :nmda_correlation_by_region),
    )
    required_present =
        voltage_rmse !== nothing &&
        spike_auroc !== nothing &&
        nmda_rmse !== nothing
    passed = required_present &&
        spike_auroc >= 0.985 &&
        voltage_rmse <= 5.0 &&
        isfinite(nmda_rmse)
    return (
        available=true,
        passed,
        source=(path=artifact.path, sha256=artifact.sha256),
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
        paper_reference=(spike_auroc=PAPER_REFERENCE.twin_spike_auroc,),
    )
end

function distilled_fidelity_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="distilled-11 artifact was not supplied",
    )
    artifact = load_artifact(path)
    payload = _metric_payload(artifact)
    metrics_root = _first_property(payload, (:metrics,), payload)
    test_metrics = _find_recursive(metrics_root, (:test,))
    test_metrics === nothing && (test_metrics = metrics_root)
    voltage_rmse = _float_metric(
        test_metrics,
        (:soma_voltage_rmse_mv, :voltage_rmse_mv),
    )
    voltage_correlation = _float_metric(
        test_metrics,
        (:soma_voltage_correlation, :voltage_correlation),
    )
    spike_auroc = _float_metric(test_metrics, (:spike_auroc,))
    spike_f1 = _float_metric(test_metrics, (:spike_f1,))
    nmda_rmse = _float_metric(
        test_metrics,
        (:nmda_rmse_by_region, :nmda_rmse),
    )
    nmda_correlation = _float_metric(
        test_metrics,
        (:nmda_correlation_by_region, :nmda_correlation),
    )
    dendritic_rmse = _float_metric(
        test_metrics,
        (:dendritic_voltage_rmse_mv,),
    )
    calcium_auroc = _float_metric(
        test_metrics,
        (:calcium_event_auroc,),
    )
    free_rollout_horizon = _float_metric(
        test_metrics,
        (:free_rollout_horizon,),
    )
    required_present =
        voltage_rmse !== nothing &&
        spike_auroc !== nothing &&
        nmda_correlation !== nothing &&
        dendritic_rmse !== nothing
    passed = required_present &&
        voltage_rmse <= 5.0 &&
        spike_auroc >= 0.95 &&
        nmda_correlation >= 0.80 &&
        dendritic_rmse <= 10.0
    return (
        available=true,
        passed,
        source=(path=artifact.path, sha256=artifact.sha256),
        parameter_sha256=_first_property(
            payload,
            (:parameter_sha256,),
            nothing,
        ),
        teacher_sha256=_first_property(
            payload,
            (:teacher_sha256,),
            nothing,
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

function _task_array(payload)
    tasks = _find_recursive(payload, (:tasks,))
    tasks isa AbstractArray && return collect(tasks)
    return Any[]
end

function _task_variant(task)
    value = _first_property(task, (:variant, :ablation), "full")
    return lowercase(String(value))
end

function _task_dimension(task)
    value = _first_property(task, (:dimension, :bits, :d), nothing)
    value isa Integer && return Int(value)
    value isa Real && return round(Int, value)
    value isa AbstractString && return parse(Int, value)
    return nothing
end

function _task_accuracy(task)
    for name in (
        :transfer_jitter_accuracy,
        :jitter_accuracy,
        :transfer_clean_accuracy,
        :clean_accuracy,
        :test_accuracy,
        :accuracy,
    )
        value = _finite_float(_property(task, name, nothing))
        value === nothing || return value
    end
    return nothing
end

function _median_accuracy(tasks, dimension, variant)
    values = Float64[]
    for task in tasks
        _task_dimension(task) == dimension || continue
        _task_variant(task) == variant || continue
        accuracy = _task_accuracy(task)
        accuracy === nothing || push!(values, accuracy)
    end
    return isempty(values) ? nothing : median(values)
end

function _constraint_attestation(tasks, names)
    observed = Bool[]
    for task in tasks
        constraints = _property(task, :constraints, nothing)
        value = _bool_metric(constraints, names)
        value === nothing || push!(observed, value)
    end
    return isempty(observed) ? nothing : all(observed)
end

function parity_summary(path::Union{Nothing,AbstractString})
    path === nothing && return (
        available=false,
        passed=false,
        reason="parity artifact was not supplied",
    )
    artifact = load_artifact(path)
    payload = _metric_payload(artifact)
    tasks = _task_array(payload)
    isempty(tasks) && return (
        available=true,
        passed=false,
        source=(path=artifact.path, sha256=artifact.sha256),
        reason="artifact has no tasks array",
    )

    variants = ("full", "passive", "no_nmda", "soma_only")
    dimensions = sort(unique(filter(!isnothing, _task_dimension.(tasks))))
    aggregate = NamedTuple[]
    for dimension in dimensions, variant in variants
        accuracy = _median_accuracy(tasks, dimension, variant)
        accuracy === nothing && continue
        count = sum(task ->
            _task_dimension(task) == dimension &&
            _task_variant(task) == variant &&
            _task_accuracy(task) !== nothing,
            tasks)
        push!(aggregate, (; dimension, variant, median_accuracy=accuracy, count))
    end

    full_2 = _median_accuracy(tasks, 2, "full")
    full_4 = _median_accuracy(tasks, 4, "full")
    full_6 = _median_accuracy(tasks, 6, "full")
    full_gates = (
        d2=full_2 !== nothing && full_2 >= 0.98,
        d4=full_4 !== nothing && full_4 >= 0.985,
        d6=full_6 === nothing ? nothing : full_6 >= 0.95,
    )
    ablation_gaps = NamedTuple[]
    for variant in ("passive", "no_nmda", "soma_only")
        accuracy = _median_accuracy(tasks, 4, variant)
        gap = full_4 === nothing || accuracy === nothing ?
            nothing : full_4 - accuracy
        push!(
            ablation_gaps,
            (
                variant,
                full_accuracy=full_4,
                ablated_accuracy=accuracy,
                gap,
                passed=gap !== nothing && gap >= 0.15,
            ),
        )
    end
    soma_spike_only = _constraint_attestation(
        tasks,
        (:soma_spike_only, :decision_window_soma_spike_only),
    )
    nonnegative_conductance = _constraint_attestation(
        tasks,
        (:nonnegative_conductance, :conductance_bounds_satisfied),
    )
    location_constraints = _constraint_attestation(
        tasks,
        (:location_constraints_satisfied, :contact_density_satisfied),
    )
    dimension_gate =
        full_gates.d2 &&
        full_gates.d4 &&
        (full_gates.d6 === nothing || full_gates.d6)
    constraint_gate =
        soma_spike_only === true &&
        nonnegative_conductance !== false &&
        location_constraints !== false
    ablation_gate = all(item -> item.passed, ablation_gaps)
    return (
        available=true,
        passed=dimension_gate && ablation_gate && constraint_gate,
        source=(path=artifact.path, sha256=artifact.sha256),
        schema=_first_property(payload, (:schema,), nothing),
        task_count=length(tasks),
        aggregate,
        full_gates,
        ablation_gaps,
        constraints=(
            soma_spike_only,
            nonnegative_conductance,
            location_constraints,
        ),
        thresholds=(
            full_d2_minimum=0.98,
            full_d4_minimum=0.985,
            full_d6_minimum=0.95,
            full_minus_each_4bit_ablation_minimum=0.15,
        ),
        paper_reference=(
            full_4bit=PAPER_REFERENCE.full_4bit,
            passive_4bit=PAPER_REFERENCE.passive_4bit,
            soma_only_4bit=PAPER_REFERENCE.soma_only_4bit,
            no_nmda_4bit=PAPER_REFERENCE.no_nmda_4bit,
        ),
    )
end

function _numeric_array(value)
    value isa AbstractArray{<:Real} && return Float64.(value)
    value isa AbstractArray || return nothing
    isempty(value) && return Array{Float64}(undef, 0)
    child_arrays = [_numeric_array(child) for child in value]
    any(isnothing, child_arrays) && return nothing
    child_shape = size(first(child_arrays))
    all(child -> size(child) == child_shape, child_arrays) || return nothing
    result = Array{Float64}(undef, length(value), child_shape...)
    for index in eachindex(value)
        if isempty(child_shape)
            result[index] = Float64(value[index])
        else
            selectdim(result, 1, index) .= child_arrays[index]
        end
    end
    return result
end

function _observation_matrix(
    values;
    compartment_axis::Int=1,
    exclude_indices=Int[],
)
    array = _numeric_array(values)
    array === nothing &&
        throw(ArgumentError("trajectory is not a rectangular numeric array"))
    ndims(array) >= 2 ||
        throw(ArgumentError("trajectory must have at least two dimensions"))
    1 <= compartment_axis <= ndims(array) ||
        throw(ArgumentError("invalid compartment axis"))
    order = vcat(
        compartment_axis,
        filter(!=(compartment_axis), collect(1:ndims(array))),
    )
    permuted = permutedims(array, order)
    flattened = reshape(permuted, size(permuted, 1), :)
    keep = setdiff(collect(axes(flattened, 1)), exclude_indices)
    isempty(keep) && throw(ArgumentError("no dendritic compartments remain"))
    return Matrix(transpose(@view(flattened[keep, :])))
end

"""
PCA over all observations (time x trial) and all dendritic compartments.
Rows are observations and columns are compartments after conversion.
"""
function voltage_pca_metrics(
    voltage;
    compartment_axis::Int=1,
    exclude_indices=Int[],
)
    matrix = _observation_matrix(
        voltage;
        compartment_axis,
        exclude_indices,
    )
    observations, compartments = size(matrix)
    observations >= 2 ||
        throw(ArgumentError("PCA requires at least two observations"))
    matrix .-= mean(matrix; dims=1)
    covariance = Symmetric(transpose(matrix) * matrix / observations)
    eigenvalues = sort!(
        max.(eigvals(covariance), 0.0);
        rev=true,
    )
    total = sum(eigenvalues)
    if total <= eps(Float64)
        return (
            observations,
            compartments,
            total_variance=total,
            k90=0,
            participation_rank=0.0,
            entropy_rank=0.0,
            pc1_fraction=0.0,
            eigenvalues,
        )
    end
    cumulative = cumsum(eigenvalues) ./ total
    k90 = something(findfirst(>=(0.90), cumulative), length(eigenvalues))
    participation = total^2 / sum(abs2, eigenvalues)
    probability = eigenvalues ./ total
    entropy = -sum(
        value > 0.0 ? value * log(value) : 0.0
        for value in probability
    )
    return (
        observations,
        compartments,
        total_variance=total,
        k90,
        participation_rank=participation,
        entropy_rank=exp(entropy),
        pc1_fraction=first(eigenvalues) / total,
        eigenvalues,
    )
end

function nmda_recruitment_metrics(
    current;
    compartment_axis::Int=1,
    exclude_indices=Int[],
)
    matrix = _observation_matrix(
        current;
        compartment_axis,
        exclude_indices,
    )
    inward = max.(-matrix, 0.0)
    per_compartment = vec(mean(inward; dims=1))
    mean_inward = mean(per_compartment)
    maximum_inward = maximum(per_compartment)
    threshold = max(1.0e-12, 0.05 * maximum_inward)
    recruited = count(>(threshold), per_compartment)
    total = sum(per_compartment)
    spatial_entropy = if total <= eps(Float64)
        0.0
    else
        probability = per_compartment ./ total
        -sum(value > 0.0 ? value * log(value) : 0.0 for value in probability)
    end
    normalized_entropy = length(per_compartment) <= 1 ?
        0.0 : spatial_entropy / log(length(per_compartment))
    return (
        observations=size(matrix, 1),
        compartments=size(matrix, 2),
        mean_inward_current=mean_inward,
        maximum_compartment_mean=maximum_inward,
        recruitment_threshold=threshold,
        recruited_compartments=recruited,
        recruited_fraction=recruited / length(per_compartment),
        spatial_entropy_normalized=normalized_entropy,
        per_compartment_mean=per_compartment,
    )
end

function _manifest_paths(path, payload)
    shards = _first_property(payload, (:shards, :files, :paths), nothing)
    shards isa AbstractArray || return String[]
    base = dirname(path)
    result = String[]
    for shard in shards
        candidate = shard isa AbstractString ?
            String(shard) :
            String(_first_property(shard, (:path, :file), ""))
        isempty(candidate) && continue
        push!(result, isabspath(candidate) ? candidate : joinpath(base, candidate))
    end
    return result
end

function _expand_trajectory_paths(paths)
    expanded = String[]
    for path in paths
        if isdir(path)
            append!(
                expanded,
                sort(filter(file ->
                    lowercase(splitext(file)[2]) in (".jld2", ".jld"),
                    readdir(path; join=true))),
            )
        elseif isfile(path) && lowercase(splitext(path)[2]) == ".json"
            artifact = load_artifact(path)
            manifest_paths = _manifest_paths(path, artifact.payload)
            if isempty(manifest_paths)
                push!(expanded, path)
            else
                append!(expanded, manifest_paths)
            end
        else
            push!(expanded, path)
        end
    end
    return expanded
end

function _trajectory_payloads(payload)
    wrapped = _first_property(payload, (:payload,), nothing)
    wrapped === nothing || (payload = wrapped)
    records = _first_property(payload, (:trajectories, :tasks), nothing)
    records isa AbstractArray && return collect(records)
    return Any[payload]
end

function _trajectory_dimension(record, fallback)
    dimension = _first_property(record, (:dimension, :bits, :d), fallback)
    dimension isa Integer && return Int(dimension)
    dimension isa Real && return round(Int, dimension)
    dimension isa AbstractString && return parse(Int, dimension)
    return fallback
end

function _trajectory_variant(record)
    return lowercase(String(_first_property(
        record,
        (:variant, :ablation),
        "full",
    )))
end

function _trajectory_arrays(record)
    voltage = _first_property(
        record,
        (
            :target_compartment_voltage,
            :compartment_voltage_mv,
            :dendritic_voltage_mv,
        ),
        nothing,
    )
    nmda = _first_property(
        record,
        (
            :target_compartment_nmda,
            :compartment_nmda_current,
            :nmda_current,
        ),
        nothing,
    )
    regions = _first_property(
        record,
        (:compartment_region, :regions),
        nothing,
    )
    return (; voltage, nmda, regions)
end

function _median_field(records, dimension, name)
    values = Float64[]
    for record in records
        record.dimension == dimension || continue
        value = getproperty(record, name)
        value isa Real && isfinite(Float64(value)) &&
            push!(values, Float64(value))
    end
    return isempty(values) ? nothing : median(values)
end

function _strictly_increasing(values)
    any(isnothing, values) && return false
    return all(index -> values[index] < values[index + 1],
        1:(length(values) - 1))
end

function trajectory_summary(paths, dimensions=DEFAULT_DIMENSIONS)
    isempty(paths) && return (
        available=false,
        passed=false,
        reason="detailed trajectory artifact was not supplied",
    )
    sources = NamedTuple[]
    records = NamedTuple[]
    expanded = _expand_trajectory_paths(paths)
    for (path_index, path) in enumerate(expanded)
        artifact = load_artifact(path)
        push!(sources, (path=artifact.path, sha256=artifact.sha256))
        fallback = path_index <= length(dimensions) ?
            dimensions[path_index] : nothing
        for record in _trajectory_payloads(artifact.payload)
            arrays = _trajectory_arrays(record)
            arrays.voltage === nothing && continue
            arrays.nmda === nothing && continue
            dimension = _trajectory_dimension(record, fallback)
            variant = _trajectory_variant(record)
            regions = arrays.regions
            exclude = if regions isa AbstractVector &&
                         length(regions) == size(_numeric_array(arrays.voltage), 1)
                findall(value -> UInt8(value) == SOMA, regions)
            else
                Int[1]
            end
            pca = voltage_pca_metrics(
                arrays.voltage;
                exclude_indices=exclude,
            )
            nmda = nmda_recruitment_metrics(
                arrays.nmda;
                exclude_indices=exclude,
            )
            push!(
                records,
                (
                    dimension,
                    variant,
                    source=artifact.path,
                    observations=pca.observations,
                    compartments=pca.compartments,
                    k90=pca.k90,
                    participation_rank=pca.participation_rank,
                    entropy_rank=pca.entropy_rank,
                    pc1_fraction=pca.pc1_fraction,
                    total_variance=pca.total_variance,
                    mean_nmda_inward=nmda.mean_inward_current,
                    nmda_recruited_fraction=nmda.recruited_fraction,
                    nmda_spatial_entropy=nmda.spatial_entropy_normalized,
                ),
            )
        end
    end
    full_records = filter(record -> record.variant == "full", records)
    required = filter(dimension -> dimension in dimensions, (2, 4, 6))
    k90 = [_median_field(full_records, dimension, :k90) for dimension in required]
    rank = [
        _median_field(full_records, dimension, :participation_rank)
        for dimension in required
    ]
    nmda = [
        _median_field(full_records, dimension, :mean_nmda_inward)
        for dimension in required
    ]
    recruited = [
        _median_field(full_records, dimension, :nmda_recruited_fraction)
        for dimension in required
    ]
    rank_monotonic = length(required) >= 2 && _strictly_increasing(rank)
    k90_monotonic = length(required) >= 2 && _strictly_increasing(k90)
    nmda_monotonic = length(required) >= 2 && _strictly_increasing(nmda)
    rank_ratio = length(rank) >= 2 &&
        first(rank) !== nothing &&
        last(rank) !== nothing &&
        first(rank) > 0 ?
        last(rank) / first(rank) : nothing
    nmda_ratio = length(nmda) >= 2 &&
        first(nmda) !== nothing &&
        last(nmda) !== nothing &&
        first(nmda) > 0 ?
        last(nmda) / first(nmda) : nothing
    recruitment_gain = length(recruited) >= 2 &&
        first(recruited) !== nothing &&
        last(recruited) !== nothing ?
        last(recruited) - first(recruited) : nothing
    passed =
        !isempty(records) &&
        rank_monotonic &&
        k90_monotonic &&
        nmda_monotonic &&
        rank_ratio !== nothing &&
        rank_ratio >= 1.20 &&
        nmda_ratio !== nothing &&
        nmda_ratio >= 1.10
    return (
        available=!isempty(records),
        passed,
        sources,
        records,
        dimension_summary=(
            dimensions=required,
            k90,
            participation_rank=rank,
            mean_nmda_inward=nmda,
            nmda_recruited_fraction=recruited,
        ),
        gates=(
            k90_strictly_increasing=k90_monotonic,
            participation_rank_strictly_increasing=rank_monotonic,
            nmda_recruitment_strictly_increasing=nmda_monotonic,
            participation_rank_last_over_first=rank_ratio,
            nmda_last_over_first=nmda_ratio,
            recruited_fraction_last_minus_first=recruitment_gain,
        ),
        thresholds=(
            participation_rank_last_over_first_minimum=1.20,
            nmda_last_over_first_minimum=1.10,
        ),
        method=(
            voltage="PCA of centered dendritic compartment voltage over every time x trial observation; soma excluded",
            nmda="mean inward max(-I_NMDA,0) per dendritic compartment over every time x trial observation",
        ),
    )
end

function aggregate_validation(options::ValidationOptions)
    kernel = options.run_kernel_contract ?
        canonical_kernel_contract() :
        (
            available=false,
            passed=false,
            reason="kernel contract disabled by CLI",
        )
    twin = digital_twin_summary(options.twin_path)
    parity = parity_summary(options.parity_path)
    trajectory = trajectory_summary(
        options.trajectory_paths,
        options.dimensions,
    )
    distilled = distilled_fidelity_summary(options.distilled_path)
    required_gates = (
        canonical_detailed_kernel=kernel.passed,
        digital_twin_heldout_fidelity=twin.passed,
        parity_and_retrained_ablation=parity.passed,
        high_rank_voltage_and_nmda_scaling=trajectory.passed,
        distilled11_trajectory_fidelity=distilled.passed,
    )
    return (
        schema=VALIDATION_SCHEMA,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        paper_reference=(
            source="Aizenbud et al. 2026 bioRxiv 10.64898/2026.06.08.730984v1",
            values=PAPER_REFERENCE,
            note="reference values are never substituted for measured artifacts",
        ),
        options=(
            parity_path=options.parity_path,
            twin_path=options.twin_path,
            trajectory_paths=options.trajectory_paths,
            distilled_path=options.distilled_path,
            dimensions=options.dimensions,
            strict=options.strict,
        ),
        kernel_contract=kernel,
        digital_twin=twin,
        parity,
        detailed_trajectory=trajectory,
        distilled11_fidelity=distilled,
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
        "paper mechanism validation passed=$(result.passed) " *
        "output=$(options.output_path)",
    )
    if options.strict && !result.passed
        error("paper-mechanism validation gates failed")
    end
    return result
end

end # module PaperMechanismValidation

if abspath(PROGRAM_FILE) == @__FILE__
    PaperMechanismValidation.main()
end
