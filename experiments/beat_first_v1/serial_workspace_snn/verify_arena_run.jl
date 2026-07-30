using Dates
using JLD2
using JSON3
using Random
using SHA

include(joinpath(@__DIR__, "train_arena_100k.jl"))

const VERIFICATION_FORMAT = "serial-workspace-snn-arena-run-verification"
const VERIFICATION_VERSION = 2
const REQUIRED_CHECKPOINT_FORMAT =
    "serial-workspace-snn-arena-checkpoint"
const REQUIRED_CHECKPOINT_VERSION = 3
const REQUIRED_EXPERIMENT_ID = "serial_workspace_snn_arena_v3"
const REQUIRED_MODEL_PRESET = "scaled_v2"
const CHECKPOINT_PATTERN = r"^checkpoint_(\d{9})\.jld2$"
const REQUIRED_PROJECT_PATH = realpath(joinpath(@__DIR__, ".."))
const REQUIRED_VERIFIER_BLAS_THREADS = 1
const REQUIRED_VERIFIER_DEFAULT_THREADS = 1
const REQUIRED_VERIFIER_INTERACTIVE_THREADS = 0
const CHECKPOINT_PAYLOAD_FIELDS = (
    :format,
    :version,
    :component_loss_alias_contract,
    :checkpoint_kind,
    :update,
    :parent_checkpoint,
    :dataset_content_sha256,
    :dataset_integrity,
    :runtime_provenance,
    :parameters,
    :optimizer,
    :trainer_state,
    :total_structural_flips,
    :synapse_utility,
    :utility_updates,
    :sampler_state,
    :initial_parameters,
    :config,
    :initial_metrics,
    :progress,
    :persistent_team_warmup,
    :segment_state,
    :last_training_dynamics,
    :finalization,
)
const CHECKPOINT_STATE_FIELDS = (
    :component_loss_alias_contract,
    :dataset_content_sha256,
    :dataset_integrity,
    :runtime_provenance,
    :parameters,
    :optimizer,
    :trainer_state,
    :total_structural_flips,
    :synapse_utility,
    :utility_updates,
    :sampler_state,
    :initial_parameters,
    :config,
    :initial_metrics,
    :progress,
    :persistent_team_warmup,
    :segment_state,
    :last_training_dynamics,
)
const CHECKPOINT_REFERENCE_FIELDS = (
    :kind,
    :path,
    :bytes,
    :sha256,
    :update,
)
const LAUNCH_BINDING_FIELDS = (:path, :sha256)
const CHECKPOINT_OPTIMIZER_FIELDS = (
    :first_moment,
    :second_moment,
    :learning_rate,
    :beta1,
    :beta2,
    :beta1_power,
    :beta2_power,
    :epsilon,
    :weight_decay,
    :step,
)
const CHECKPOINT_TRAINER_STATE_FIELDS = (
    :last_loss,
    :last_gradient_norm,
    :structure_weight,
)
const COMPONENT_LOSS_NAMES = (
    :old_q_loss,
    :q_huber_loss,
    :margin_loss,
    :raw_top_gap_loss,
    :death_loss,
    :quantile_teacher_loss,
    :geometry_loss,
    :line_clear_loss,
    :max_height_loss,
    :holes_loss,
    :cavities_loss,
    :structure_loss,
)
const COMPONENT_LOSS_WINDOW_FIELDS = Tuple(
    Symbol("window_", String(name), "_sum")
    for name in COMPONENT_LOSS_NAMES
)
const COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS = Tuple(
    Symbol("active_window_", String(name), "_sum")
    for name in COMPONENT_LOSS_NAMES
)
const CHECKPOINT_COMPONENT_LOSS_FIELDS = (
    COMPONENT_LOSS_WINDOW_FIELDS...,
    COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS...,
    COMPONENT_LOSS_NAMES...,
)
const COMPONENT_LOSS_ALIAS_CONTRACT = (;
    schema_version=1,
    q_huber_loss=(;
        alias_of="old_q_loss",
        identity="bit_exact",
    ),
    raw_top_gap_loss=(;
        alias_of="margin_loss",
        identity="bit_exact",
    ),
)
const CHECKPOINT_LOSS_FIELDS = (
    :composite_loss,
    :listnet_loss,
    :teacher_entropy,
    :listnet_kl,
    :old_q_loss,
    :q_huber_loss,
    :margin_loss,
    :raw_top_gap_loss,
    :death_loss,
    :quantile_teacher_loss,
    :geometry_loss,
    :line_clear_loss,
    :max_height_loss,
    :holes_loss,
    :cavities_loss,
    :structure_loss,
    :gate_density,
    :valid_candidates,
)
const CHECKPOINT_SAMPLER_FIELDS = (
    :format_version,
    :source_rows,
    :permutation,
    :cursor,
    :completed_epochs,
    :rng,
)
const CHECKPOINT_PROGRESS_FIELDS = (
    :updates,
    :teacher_states,
    :candidates,
    :hot_wall_seconds,
    :hot_cpu_seconds,
    :hot_allocation_bytes,
    :hot_gc_seconds,
    :pack_seconds,
    :forward_seconds,
    :loss_seconds,
    :shadow_seconds,
    :backward_seconds,
    :optimizer_seconds,
    :consolidation_seconds,
    :window_updates,
    :window_composite_loss,
    :window_listnet_ce,
    :window_teacher_entropy,
    :window_listnet_kl,
    :window_composite_excess,
    :completed_component_loss_window_updates,
    :telemetry_schema_version,
    :component_loss_alias_contract,
    :component_losses,
)
const CHECKPOINT_WARMUP_FIELDS = (
    :isolation_verified,
    :warmup_optimizer_step,
    :warmup_loss,
    :queue_length,
    :remaining,
    :failure_worker,
)
const CHECKPOINT_SEGMENT_FIELDS = (
    :start_update,
    :updates,
    :overall_seconds,
)
const CHECKPOINT_DYNAMICS_FIELDS = (
    :schema_version,
    :firing_rate,
    :workspace_route_entropy,
    :workspace_exploitation_entropy,
    :hard_route_load_entropy,
    :hard_route_effective_blocks,
    :hard_route_top8_share,
    :route_probability_mass_error,
    :route_probability_max_mass_error,
    :workspace_rms,
    :gate_density,
    :utility_mean,
    :utility_nonzero_fraction,
    :head_pre_rms,
    :hidden_inv_rms_mean,
    :hidden_inv_rms_min,
    :hidden_inv_rms_max,
    :hidden_tanh_derivative_mean,
    :route_selection_gap,
    :route_score_rms,
    :hard_mask_unique_fraction,
    :hard_mask_cycle_churn,
    :entropy_floor_violation_fraction,
    :utility_swap_gap,
    :consolidation_scheduled,
    :consolidation_actual,
    :net_mask_flips,
    :gate_probability_mean,
    :gate_derivative_mean,
    :delay_mean,
    :delay_derivative_mean,
    :leak_mean,
    :leak_derivative_mean,
    :threshold_mean,
    :threshold_derivative_mean,
    :workspace_decay,
    :workspace_decay_derivative,
    :membrane_threshold_margin_mean,
    :membrane_threshold_margin_rms,
    :surrogate_sensitivity_mean,
    :surrogate_sensitivity_rms,
    :eligibility_rms,
    :local_q_loss,
    :local_death_loss,
    :local_quantile_loss,
    :local_geometry_loss,
)
const CHECKPOINT_DYNAMICS_BOOL_FIELDS = (
    :consolidation_scheduled,
    :consolidation_actual,
)
const CHECKPOINT_DYNAMICS_INT_FIELDS = (
    :schema_version,
    :net_mask_flips,
)
const CHECKPOINT_DYNAMICS_FLOAT_FIELDS = Tuple(
    property for property in CHECKPOINT_DYNAMICS_FIELDS
    if !(property in CHECKPOINT_DYNAMICS_BOOL_FIELDS) &&
       !(property in CHECKPOINT_DYNAMICS_INT_FIELDS)
)
const CHECKPOINT_FINALIZATION_FIELDS = (
    :status,
    :finalized_at,
    :optimizer_steps_after_target,
    :expected_results_path,
    :expected_manifest_path,
    :team_teardown,
    :training_checkpoint,
    :final_metrics,
    :component_loss_alias_contract,
    :completed_component_loss_window_updates,
    :component_loss_telemetry,
)
const REQUIRED_TRACE_COLUMNS = (
    :trace_schema_version,
    :update,
    :teacher_states,
    :loss,
    :listnet_ce,
    :teacher_entropy,
    :listnet_kl,
    :composite_excess,
    :window_updates,
    :window_loss,
    :window_listnet_ce,
    :window_teacher_entropy,
    :window_listnet_kl,
    :window_composite_excess,
    :component_loss_alias_schema_version,
    :q_huber_loss_alias_of,
    :raw_top_gap_loss_alias_of,
    :component_loss_alias_identity,
    COMPONENT_LOSS_NAMES...,
    COMPONENT_LOSS_WINDOW_FIELDS...,
    :gradient_norm,
    :enabled_synapses,
    :structural_flips_total,
    :training_dynamics_schema_version,
    Tuple(
        property for property in CHECKPOINT_DYNAMICS_FIELDS
        if property != :schema_version
    )...,
    :states_per_second,
    :cpu_percent,
    :hot_allocation_bytes,
    :hot_gc_seconds,
    :shadow_seconds,
)
const TRACE_INTEGER_FIELDS = (
    :trace_schema_version,
    :update,
    :teacher_states,
    :window_updates,
    :enabled_synapses,
    :structural_flips_total,
    :training_dynamics_schema_version,
    :net_mask_flips,
    :hot_allocation_bytes,
    :component_loss_alias_schema_version,
)
const TRACE_BOOL_FIELDS = CHECKPOINT_DYNAMICS_BOOL_FIELDS
const TRACE_STRING_FIELDS = (
    :q_huber_loss_alias_of,
    :raw_top_gap_loss_alias_of,
    :component_loss_alias_identity,
)
const TRACE_FLOAT_FIELDS = Tuple(
    property for property in REQUIRED_TRACE_COLUMNS
    if !(property in TRACE_INTEGER_FIELDS) &&
       !(property in TRACE_BOOL_FIELDS) &&
       !(property in TRACE_STRING_FIELDS)
)
const TRACE_COMPONENT_LOSS_FIELDS = (
    COMPONENT_LOSS_NAMES...,
    COMPONENT_LOSS_WINDOW_FIELDS...,
)
const VERIFICATION_TRAINING_CHECKPOINT_FIELDS = (
    :update,
    :path,
    :bytes,
    :sha256,
    :payload_format,
    :payload_version,
    :checkpoint_kind,
    :structural_flips,
    :utility_updates,
    :hard_gate_budget,
    :initial_parameters_sha256,
)
const ANCESTRY_CONFIG_INVARIANT_FIELDS = (
    :experiment_id,
    :checkpoint_schema,
    :production_contract,
    :production_contract_sha256,
    :model_preset,
    :model,
    :parameter_count,
    :maximum_updates,
    :state_batch,
    :target_teacher_states,
    :candidate_width,
    :learning_rate,
    :weight_decay,
    :structure_weight,
    :optimizer,
    :learning_mode,
    :eprop_reducers,
    :structural_interval,
    :checkpoint_interval,
    :log_interval,
    :evaluation_states,
    :maximum_hot_allocation_bytes,
    :dataset_path,
    :dataset_content_sha256,
    :dataset_integrity,
    :training_rows,
    :training_rows_sha256,
    :training_eval_states,
    :training_panel_rows_sha256,
    :model_seed,
    :sampler_seed,
    :routing_seed,
    :source_fingerprint,
    :runtime_provenance,
    :representation,
    :workspace_retention,
    :spiking,
    :eprop,
    :routing,
    :executor,
)

function enforce_verifier_runtime!()
    hermetic_runtime = hermetic_runtime_options()
    startup_file_option = Int(Base.JLOptions().startupfile)
    history_file_option = Int(Base.JLOptions().historyfile)
    startup_file_option == 2 || error(
        "verifier requires --startup-file=no: " *
        "observed Julia option=$startup_file_option",
    )
    history_file_option == 0 || error(
        "verifier requires --history-file=no: " *
        "observed Julia option=$history_file_option",
    )
    julia_threads = Threads.nthreads(:default)
    interactive_threads = Threads.nthreads(:interactive)
    julia_threads == REQUIRED_VERIFIER_DEFAULT_THREADS || error(
        "verifier requires one default Julia thread: " *
        "observed=$julia_threads",
    )
    interactive_threads == REQUIRED_VERIFIER_INTERACTIVE_THREADS || error(
        "verifier requires zero interactive Julia threads: " *
        "observed=$interactive_threads",
    )
    active_project_file = Base.active_project()
    active_project_file === nothing &&
        error("verifier has no active Julia project")
    required_project_file =
        realpath(joinpath(REQUIRED_PROJECT_PATH, "Project.toml"))
    require_equal(
        normalized_path(realpath(active_project_file)),
        normalized_path(required_project_file),
        "active verifier Project.toml",
    )
    BLAS.set_num_threads(REQUIRED_VERIFIER_BLAS_THREADS)
    actual_blas_threads = BLAS.get_num_threads()
    actual_blas_threads == REQUIRED_VERIFIER_BLAS_THREADS || error(
        "verifier could not enforce one BLAS thread: " *
        "observed=$actual_blas_threads",
    )
    required_manifest_file =
        realpath(joinpath(REQUIRED_PROJECT_PATH, "Manifest.toml"))
    julia_executable_path = realpath(joinpath(
        Sys.BINDIR,
        Base.julia_exename(),
    ))
    return (;
        julia_threads,
        required_julia_threads=REQUIRED_VERIFIER_DEFAULT_THREADS,
        interactive_threads,
        required_interactive_threads=
            REQUIRED_VERIFIER_INTERACTIVE_THREADS,
        blas_threads=actual_blas_threads,
        required_blas_threads=REQUIRED_VERIFIER_BLAS_THREADS,
        blas_contract_verified=true,
        startup_file_option,
        history_file_option,
        startup_file_disabled=true,
        history_file_disabled=true,
        active_project_file=realpath(active_project_file),
        active_project_sha256=file_sha256(active_project_file),
        required_project_file,
        required_project_sha256=file_sha256(required_project_file),
        required_manifest_file,
        required_manifest_sha256=file_sha256(required_manifest_file),
        active_project_path=realpath(dirname(active_project_file)),
        required_project_path=REQUIRED_PROJECT_PATH,
        project_contract_verified=true,
        required_runtime_arguments=collect(
            REQUIRED_JULIA_RUNTIME_ARGUMENTS,
        ),
        julia_project_option=
            hermetic_runtime.julia_project_option,
        julia_project_option_path=
            hermetic_runtime.julia_project_option_path,
        julia_executable_path,
        julia_executable_sha256=file_sha256(julia_executable_path),
        julia_version=string(VERSION),
    )
end

function parse_verification_arguments(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if startswith(argument, "--") && occursin('=', argument)
            key, value = split(argument[3:end], '='; limit=2)
            options[key] = value
            index += 1
        elseif startswith(argument, "--")
            index == length(arguments) &&
                error("missing value for verification argument $argument")
            options[argument[3:end]] = arguments[index + 1]
            index += 2
        else
            error("unexpected positional verification argument: $argument")
        end
    end
    haskey(options, "run-dir") || error("set --run-dir")
    haskey(options, "expected-updates") ||
        error("set --expected-updates")
    haskey(options, "expected-run-id") ||
        error("set --expected-run-id")
    haskey(options, "expected-start-mode") ||
        error("set --expected-start-mode")
    haskey(options, "launch-manifest") ||
        error("set --launch-manifest")
    haskey(options, "launch-manifest-sha256") ||
        error("set --launch-manifest-sha256")
    expected_updates = parse(Int, options["expected-updates"])
    expected_updates >= 1 || error("--expected-updates must be positive")
    expected_run_id = options["expected-run-id"]
    occursin(r"^[A-Za-z0-9_.-]+$", expected_run_id) ||
        error("--expected-run-id is unsafe")
    expected_start_mode = options["expected-start-mode"]
    expected_start_mode in ("scratch", "resume", "finalize-only") ||
        error("--expected-start-mode is invalid")
    run_dir = abspath(options["run-dir"])
    launch_manifest_path = abspath(options["launch-manifest"])
    launch_manifest_sha256 =
        lowercase(options["launch-manifest-sha256"])
    occursin(r"^[0-9a-f]{64}$", launch_manifest_sha256) ||
        error("--launch-manifest-sha256 must be a SHA-256 digest")

    parent_keys = (
        "parent-checkpoint",
        "parent-sha256",
        "parent-update",
    )
    parent_count = count(key -> haskey(options, key), parent_keys)
    parent_count in (0, length(parent_keys)) ||
        error("parent checkpoint path, SHA-256, and update are atomic")
    parent_checkpoint = if iszero(parent_count)
        nothing
    else
        parent_sha256 = lowercase(options["parent-sha256"])
        occursin(r"^[0-9a-f]{64}$", parent_sha256) ||
            error("--parent-sha256 must be a SHA-256 digest")
        parent_update = parse(Int, options["parent-update"])
        parent_update >= 0 || error("--parent-update must be nonnegative")
        (;
            path=abspath(options["parent-checkpoint"]),
            sha256=parent_sha256,
            update=parent_update,
        )
    end
    if expected_start_mode == "scratch"
        parent_checkpoint === nothing ||
            error("scratch verification cannot accept a parent checkpoint")
    else
        parent_checkpoint !== nothing ||
            error("$expected_start_mode verification requires a parent checkpoint")
    end
    output_path = abspath(get(
        options,
        "verification-output",
        joinpath(run_dir, "verification.json"),
    ))
    return (;
        run_dir,
        expected_updates,
        expected_run_id,
        expected_start_mode,
        launch_manifest_path,
        launch_manifest_sha256,
        parent_checkpoint,
        output_path,
    )
end

function atomic_verification_json(path, value)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    isfile(temporary) && rm(temporary; force=true)
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
        end
        mv(temporary, destination; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return destination
end

file_sha256(path) = bytes2hex(open(sha256, path))

function required_property(value, name::Symbol, location::AbstractString)
    hasproperty(value, name) ||
        error("$location is missing property $(String(name))")
    return getproperty(value, name)
end

function require_equal(actual, expected, location::AbstractString)
    actual == expected ||
        error("$location differs: observed=$(repr(actual)) expected=$(repr(expected))")
    return actual
end

function verify_finite(value, location::AbstractString)
    if value isa AbstractFloat
        isfinite(value) || error("$location is not finite: $value")
    elseif value isa Number || value isa AbstractString ||
           value isa Symbol || value isa Bool || value === nothing
        return nothing
    elseif value isa NamedTuple || value isa AbstractDict ||
           value isa JSON3.Object
        for (key, child) in pairs(value)
            verify_finite(child, "$location.$key")
        end
    elseif value isa Tuple || value isa AbstractArray ||
           value isa JSON3.Array
        for (index, child) in pairs(value)
            verify_finite(child, "$location[$index]")
        end
    end
    return nothing
end

function verify_trace_training_dynamics(dynamics, location::AbstractString)
    require_exact_properties(
        dynamics,
        CHECKPOINT_DYNAMICS_FIELDS,
        location,
    )
    schema_version = required_property(
        dynamics,
        :schema_version,
        location,
    )
    schema_version isa Int ||
        error("$location.schema_version is not Int")
    require_equal(schema_version, 4, "$location schema version")
    for property in CHECKPOINT_DYNAMICS_BOOL_FIELDS
        required_property(dynamics, property, location) isa Bool ||
            error("$location.$(String(property)) is not Bool")
    end
    net_mask_flips = required_property(
        dynamics,
        :net_mask_flips,
        location,
    )
    net_mask_flips isa Int ||
        error("$location.net_mask_flips is not Int")
    net_mask_flips >= 0 ||
        error("$location.net_mask_flips is negative")
    Bool(dynamics.consolidation_actual) &&
        !Bool(dynamics.consolidation_scheduled) && error(
            "$location reports actual consolidation when none was scheduled",
        )
    signed_fields = Set((
        :route_selection_gap,
        :utility_swap_gap,
        :membrane_threshold_margin_mean,
    ))
    for property in CHECKPOINT_DYNAMICS_FLOAT_FIELDS
        value = required_property(dynamics, property, location)
        value isa Float64 ||
            error("$location.$(String(property)) is not Float64")
        isfinite(value) ||
            error("$location.$(String(property)) is not finite")
        property in signed_fields || value >= 0.0 ||
            error("$location.$(String(property)) is negative")
    end
    for property in (
        :firing_rate,
        :workspace_route_entropy,
        :workspace_exploitation_entropy,
        :hard_route_load_entropy,
        :hard_route_top8_share,
        :gate_density,
        :utility_nonzero_fraction,
        :hidden_tanh_derivative_mean,
        :hard_mask_unique_fraction,
        :hard_mask_cycle_churn,
        :entropy_floor_violation_fraction,
        :gate_probability_mean,
        :delay_mean,
        :workspace_decay,
    )
        value = Float64(required_property(dynamics, property, location))
        0.0 <= value <= 1.0 ||
            error("$location.$(String(property)) is outside [0, 1]")
    end
    Float64(dynamics.workspace_rms) <= 1.0 ||
        error("$location.workspace_rms exceeds tanh range")
    Float64(dynamics.hard_route_effective_blocks) >= 1.0 ||
        error("$location.hard_route_effective_blocks is below one")
    Float64(dynamics.hidden_inv_rms_min) <=
        Float64(dynamics.hidden_inv_rms_mean) <=
        Float64(dynamics.hidden_inv_rms_max) || error(
            "$location hidden inverse-RMS ordering is invalid",
        )
    0.0 <= Float64(dynamics.gate_derivative_mean) <= 0.25 ||
        error("$location.gate_derivative_mean is outside sigmoid range")
    0.0 <= Float64(dynamics.delay_derivative_mean) <= 0.25 ||
        error("$location.delay_derivative_mean is outside sigmoid range")
    0.45 <= Float64(dynamics.leak_mean) <= 0.95 ||
        error("$location.leak_mean is outside transformed range")
    0.0 <= Float64(dynamics.leak_derivative_mean) <= 0.125 ||
        error("$location.leak_derivative_mean is outside transformed range")
    0.25 <= Float64(dynamics.threshold_mean) <= 1.0 ||
        error("$location.threshold_mean is outside transformed range")
    0.0 <= Float64(dynamics.threshold_derivative_mean) <= 0.1875 ||
        error(
            "$location.threshold_derivative_mean is outside transformed range",
        )
    0.0 <= Float64(dynamics.workspace_decay_derivative) <= 0.25 ||
        error(
            "$location.workspace_decay_derivative is outside generic range",
        )
    Float64(dynamics.membrane_threshold_margin_rms) + 1.0e-12 >=
        abs(Float64(dynamics.membrane_threshold_margin_mean)) || error(
            "$location membrane margin RMS is below absolute mean",
        )
    Float64(dynamics.surrogate_sensitivity_rms) + 1.0e-12 >=
        Float64(dynamics.surrogate_sensitivity_mean) || error(
            "$location surrogate RMS is below its mean",
        )
    return dynamics
end

function parse_trace(path)
    isfile(path) || error("training trace does not exist: $path")
    lines = readlines(path)
    length(lines) >= 2 || error("training trace is incomplete: $path")
    header_strings = String.(split(first(lines), '\t'; keepempty=true))
    header_strings[1] = replace(header_strings[1], '\ufeff' => "")
    header = Tuple(Symbol.(header_strings))
    header == REQUIRED_TRACE_COLUMNS || error(
        "training trace schema differs: observed=$(header_strings) " *
        "expected=$(String.(REQUIRED_TRACE_COLUMNS))",
    )
    length(unique(header)) == length(header) ||
        error("training trace contains duplicate columns")

    column_index = Dict(name => index for (index, name) in pairs(header))
    updates = Int[]
    teacher_states = Int[]
    parsed_records = NamedTuple[]
    last_record = nothing
    for (offset, line) in enumerate(@view(lines[2:end]))
        line_number = offset + 1
        isempty(strip(line)) &&
            error("training trace has a blank record at line $line_number")
        fields = split(line, '\t'; keepempty=true)
        length(fields) == length(header) || error(
            "training trace column count differs at line $line_number",
        )
        values = Dict{Symbol,Any}()
        for property in TRACE_INTEGER_FIELDS
            text = fields[column_index[property]]
            values[property] = property == :hot_allocation_bytes ?
                parse(Int128, text) : parse(Int, text)
        end
        for property in TRACE_BOOL_FIELDS
            text = fields[column_index[property]]
            values[property] = if text == "true"
                true
            elseif text == "false"
                false
            else
                error(
                    "training trace $(String(property)) is not canonical " *
                    "Bool text at line $line_number",
                )
            end
        end
        for property in TRACE_STRING_FIELDS
            values[property] = String(fields[column_index[property]])
        end
        for property in TRACE_FLOAT_FIELDS
            number = parse(Float64, fields[column_index[property]])
            isfinite(number) || error(
                "training trace $(String(property)) is not finite at " *
                "line $line_number",
            )
            values[property] = number
        end

        update = values[:update]::Int
        states = values[:teacher_states]::Int
        require_equal(
            values[:trace_schema_version],
            3,
            "training trace schema version at update $update",
        )
        require_equal(
            values[:training_dynamics_schema_version],
            4,
            "training trace dynamics schema version at update $update",
        )
        require_equal(
            values[:component_loss_alias_schema_version],
            1,
            "training trace component loss alias schema at update $update",
        )
        require_equal(
            values[:q_huber_loss_alias_of],
            "old_q_loss",
            "training trace q_huber_loss alias at update $update",
        )
        require_equal(
            values[:raw_top_gap_loss_alias_of],
            "margin_loss",
            "training trace raw_top_gap_loss alias at update $update",
        )
        require_equal(
            values[:component_loss_alias_identity],
            "bit_exact",
            "training trace component loss alias identity at update $update",
        )
        for property in (
            :update,
            :teacher_states,
            :window_updates,
            :enabled_synapses,
            :structural_flips_total,
            :net_mask_flips,
            :hot_allocation_bytes,
        )
            values[property] >= 0 || error(
                "training trace $(String(property)) is negative at " *
                "update $update",
            )
        end
        update > 0 ||
            error("training trace contains a nonpositive update")
        states > 0 ||
            error("training trace contains nonpositive teacher states")
        values[:enabled_synapses] > 0 || error(
            "training trace enabled-synapse count is nonpositive at " *
            "update $update",
        )
        allocation = values[:hot_allocation_bytes]::Int128
        allocation == 0 || error(
            "training trace hot allocation is nonzero at update $update",
        )
        gc_seconds = values[:hot_gc_seconds]::Float64
        gc_seconds == 0.0 || error(
            "training trace hot GC time is nonzero at update $update",
        )
        for property in TRACE_COMPONENT_LOSS_FIELDS
            values[property] >= 0.0 || error(
                "training trace $(String(property)) is negative at " *
                "update $update",
            )
        end
        require_approx(
            values[:geometry_loss],
            (
                values[:line_clear_loss] +
                values[:max_height_loss] +
                values[:holes_loss] +
                values[:cavities_loss]
            ) / 4.0,
            "training trace point geometry decomposition at update $update";
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        require_approx(
            values[:window_geometry_loss_sum],
            (
                values[:window_line_clear_loss_sum] +
                values[:window_max_height_loss_sum] +
                values[:window_holes_loss_sum] +
                values[:window_cavities_loss_sum]
            ) / 4.0,
            "training trace window geometry decomposition at update $update";
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        component_loss_telemetry =
            NamedTuple{TRACE_COMPONENT_LOSS_FIELDS}(Tuple(
                values[property] for property in TRACE_COMPONENT_LOSS_FIELDS
            ))
        for (alias, canonical) in (
            (:q_huber_loss, :old_q_loss),
            (:raw_top_gap_loss, :margin_loss),
            (:window_q_huber_loss_sum, :window_old_q_loss_sum),
            (:window_raw_top_gap_loss_sum, :window_margin_loss_sum),
        )
            require_bit_exact_alias(
                component_loss_telemetry,
                alias,
                canonical,
                "training trace component loss telemetry at update $update",
            )
        end
        training_dynamics =
            NamedTuple{CHECKPOINT_DYNAMICS_FIELDS}(Tuple(
                property == :schema_version ?
                    values[:training_dynamics_schema_version] :
                    values[property]
                for property in CHECKPOINT_DYNAMICS_FIELDS
            ))
        verify_trace_training_dynamics(
            training_dynamics,
            "training trace dynamics at update $update",
        )

        !isempty(updates) && update <= last(updates) && error(
            "training trace updates are not strictly increasing at $update",
        )
        !isempty(teacher_states) && states <= last(teacher_states) && error(
            "training trace teacher states are not strictly increasing at " *
            "update $update",
        )
        push!(updates, update)
        push!(teacher_states, states)
        last_record = Dict(
            header_strings[index] => fields[index]
            for index in eachindex(header_strings)
        )
        push!(parsed_records, (;
            update,
            teacher_states=states,
            loss=values[:loss]::Float64,
            gradient_norm=values[:gradient_norm]::Float64,
            hot_allocation_bytes=allocation,
            hot_gc_seconds=gc_seconds,
            window_updates=values[:window_updates]::Int,
            structural_flips_total=
                values[:structural_flips_total]::Int,
            component_loss_telemetry,
            training_dynamics,
        ))
    end
    for index in 2:length(parsed_records)
        previous = parsed_records[index - 1]
        current = parsed_records[index]
        current.structural_flips_total >=
            previous.structural_flips_total || error(
                "training trace structural flips decreased at update " *
                "$(current.update)",
            )
        current.hot_allocation_bytes >=
            previous.hot_allocation_bytes || error(
                "training trace cumulative hot allocation decreased at " *
                "update $(current.update)",
            )
    end
    return (;
        records=length(updates),
        first_update=first(updates),
        last_update=last(updates),
        last_teacher_states=last(teacher_states),
        last_record,
        updates,
        teacher_states,
        parsed_records,
        bytes=filesize(path),
        sha256=file_sha256(path),
    )
end

function checkpoint_files(
    checkpoint_dir,
    expected_finalization_update;
    require_training=true,
    allow_finalization=true,
)
    isdir(checkpoint_dir) ||
        error("checkpoint directory does not exist: $checkpoint_dir")
    require_equal(
        normalized_path(realpath(checkpoint_dir)),
        normalized_path(checkpoint_dir),
        "canonical checkpoint directory",
    )
    artifacts = NamedTuple[]
    expected_finalization_name =
        "finalization_checkpoint_" *
        lpad(string(expected_finalization_update), 9, '0') *
        ".jld2"
    for path in readdir(checkpoint_dir; join=true)
        islink(path) && error(
            "checkpoint directory contains a symbolic-link entry: $path",
        )
        isfile(path) || error(
            "checkpoint directory contains a non-file entry: $path",
        )
        require_equal(
            normalized_path(realpath(path)),
            normalized_path(path),
            "canonical checkpoint entry",
        )
        name = basename(path)
        if name == expected_finalization_name
            allow_finalization || error(
                "checkpoint directory unexpectedly contains a " *
                "finalization checkpoint: $path",
            )
            continue
        end
        matched = match(CHECKPOINT_PATTERN, name)
        matched === nothing && error(
            "checkpoint directory contains an unexpected regular file: " *
            path,
        )
        push!(artifacts, (;
            update=parse(Int, only(matched.captures)),
            path=abspath(path),
        ))
    end
    require_training && isempty(artifacts) &&
        error("no arena training checkpoints were found")
    sort!(artifacts; by=artifact -> artifact.update)
    updates = [artifact.update for artifact in artifacts]
    length(unique(updates)) == length(updates) ||
        error("checkpoint updates are duplicated")
    return artifacts
end

function verify_checkpoint_manifest_set(
    artifacts,
    manifest,
    expected_last_update,
    location,
)
    artifact_updates = [artifact.update for artifact in artifacts]
    manifest_updates = sort!(collect(keys(manifest)))
    manifest_updates == artifact_updates || error(
        "$location update set differs from live checkpoint files: " *
        "manifest=$manifest_updates files=$artifact_updates",
    )
    isempty(artifact_updates) &&
        error("$location contains no training checkpoints")
    last(artifact_updates) == expected_last_update || error(
        "$location last update differs: observed=$(last(artifact_updates)) " *
        "expected=$expected_last_update",
    )
    for artifact in artifacts
        record = manifest[artifact.update]
        require_equal(
            record.kind,
            "training",
            "$location update $(artifact.update) kind",
        )
        require_equal(
            normalized_path(record.path),
            normalized_path(artifact.path),
            "$location update $(artifact.update) path",
        )
        require_equal(
            record.bytes,
            filesize(artifact.path),
            "$location update $(artifact.update) byte size",
        )
        require_equal(
            record.sha256,
            file_sha256(artifact.path),
            "$location update $(artifact.update) SHA-256",
        )
    end
    return true
end

function checkpoint_manifest(path)
    isfile(path) || error("checkpoint manifest does not exist: $path")
    records = Dict{Int,NamedTuple}()
    open(path, "r") do io
        for (line_number, line) in enumerate(eachline(io))
            isempty(strip(line)) && error(
                "checkpoint manifest contains a blank line at line " *
                "$line_number",
            )
            record = JSON3.read(line)
            update = Int(required_property(
                record,
                :update,
                "checkpoint manifest line $line_number",
            ))
            haskey(records, update) &&
                error("checkpoint manifest duplicates update $update")
            records[update] = (;
                kind=String(required_property(
                    record,
                    :kind,
                    "checkpoint manifest line $line_number",
                )),
                path=abspath(String(required_property(
                    record,
                    :path,
                    "checkpoint manifest line $line_number",
                ))),
                sha256=lowercase(String(required_property(
                    record,
                    :sha256,
                    "checkpoint manifest line $line_number",
                ))),
                bytes=Int(required_property(
                    record,
                    :bytes,
                    "checkpoint manifest line $line_number",
                )),
            )
        end
    end
    return records
end

function normalized_path(path)
    return lowercase(normpath(abspath(String(path))))
end

function require_approx(
    actual,
    expected,
    location::AbstractString;
    atol=1.0e-7,
    rtol=1.0e-6,
)
    observed = Float64(actual)
    target = Float64(expected)
    isfinite(observed) || error("$location is not finite")
    isapprox(observed, target; atol, rtol) || error(
        "$location differs: observed=$observed expected=$target",
    )
    return observed
end

function require_sha256(value, location::AbstractString)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$location is not a canonical SHA-256 digest")
    return digest
end

function require_json_uint64(value, expected::UInt64, location::AbstractString)
    if value isa Integer
        value >= 0 || error("$location is negative")
        UInt64(value) == expected || error(
            "$location differs: observed=$(repr(value)) expected=$expected",
        )
    elseif value isa AbstractFloat
        observed = Float64(value)
        isfinite(observed) || error("$location is not finite")
        isinteger(observed) || error("$location is not an integer")
        observed == Float64(expected) || error(
            "$location differs: observed=$(repr(observed)) " *
            "expected=$(repr(Float64(expected)))",
        )
    else
        error("$location is not a JSON number")
    end
    return expected
end

function semantic_equal(left, right)
    if left isa Number && right isa Number
        return if left isa AbstractFloat || right isa AbstractFloat
            isfinite(Float64(left)) && isfinite(Float64(right)) &&
                isapprox(
                    Float64(left),
                    Float64(right);
                    atol=1.0e-9,
                    rtol=1.0e-7,
                )
        else
            left == right
        end
    elseif (left isa AbstractString || left isa Symbol) &&
           (right isa AbstractString || right isa Symbol)
        return String(left) == String(right)
    elseif (left isa NamedTuple || left isa AbstractDict ||
            left isa JSON3.Object) &&
           (right isa NamedTuple || right isa AbstractDict ||
            right isa JSON3.Object)
        left_keys = sort!(String.(collect(keys(left))))
        right_keys = sort!(String.(collect(keys(right))))
        left_keys == right_keys || return false
        for key_string in left_keys
            key = Symbol(key_string)
            left_value = hasproperty(left, key) ?
                getproperty(left, key) : left[key_string]
            right_value = hasproperty(right, key) ?
                getproperty(right, key) : right[key_string]
            semantic_equal(left_value, right_value) || return false
        end
        return true
    elseif (left isa Tuple || left isa AbstractArray ||
            left isa JSON3.Array) &&
           (right isa Tuple || right isa AbstractArray ||
            right isa JSON3.Array)
        length(left) == length(right) || return false
        return all(
            semantic_equal(left[index], right[index])
            for index in eachindex(left)
        )
    end
    return left === nothing && right === nothing || left == right
end

function require_semantic_equal(left, right, location::AbstractString)
    semantic_equal(left, right) || error("$location differs")
    return left
end

function require_exact_properties(
    value,
    expected,
    location::AbstractString,
)
    observed_names = sort!(String.(collect(propertynames(value))))
    expected_names = sort!(String.(collect(expected)))
    observed_names == expected_names || error(
        "$location property set differs: observed=$observed_names " *
        "expected=$expected_names",
    )
    return value
end

function require_integer_value(value, location::AbstractString)
    value isa Integer && !(value isa Bool) ||
        error("$location is not an integer: $(repr(value))")
    return value
end

function require_finite_real(value, location::AbstractString)
    value isa Real && !(value isa Bool) ||
        error("$location is not a real number: $(repr(value))")
    observed = Float64(value)
    isfinite(observed) ||
        error("$location is not finite: $(repr(value))")
    return observed
end

function exact_state_equal(left, right)
    typeof(left) === typeof(right) || return false
    if left isa Number || left isa AbstractString ||
       left isa Symbol || left isa Bool || left === nothing
        return isequal(left, right)
    elseif left isa NamedTuple
        propertynames(left) == propertynames(right) || return false
        return all(
            exact_state_equal(
                getproperty(left, name),
                getproperty(right, name),
            )
            for name in propertynames(left)
        )
    elseif left isa AbstractDict
        keys(left) == keys(right) || return false
        return all(
            exact_state_equal(left[key], right[key])
            for key in keys(left)
        )
    elseif left isa AbstractArray
        axes(left) == axes(right) || return false
        eltype(left) === eltype(right) || return false
        return all(
            exact_state_equal(left[index], right[index])
            for index in eachindex(left, right)
        )
    elseif left isa Tuple
        length(left) == length(right) || return false
        return all(
            exact_state_equal(left[index], right[index])
            for index in eachindex(left)
        )
    elseif isstructtype(typeof(left))
        fieldcount(typeof(left)) == fieldcount(typeof(right)) ||
            return false
        return all(
            exact_state_equal(
                getfield(left, index),
                getfield(right, index),
            )
            for index in 1:fieldcount(typeof(left))
        )
    end
    return isequal(left, right)
end

function require_exact_state_equal(
    left,
    right,
    location::AbstractString,
)
    exact_state_equal(left, right) ||
        error("$location state differs byte-semantically")
    return left
end

function verify_checkpoint_reference_schema(
    reference,
    location::AbstractString;
    expected_kind=nothing,
)
    require_exact_properties(
        reference,
        CHECKPOINT_REFERENCE_FIELDS,
        location,
    )
    kind = String(required_property(reference, :kind, location))
    expected_kind === nothing || require_equal(
        kind,
        String(expected_kind),
        "$location kind",
    )
    path = String(required_property(reference, :path, location))
    isempty(strip(path)) && error("$location path is blank")
    isabspath(path) || error("$location path is not absolute: $path")
    bytes = require_integer_value(
        required_property(reference, :bytes, location),
        "$location byte size",
    )
    bytes >= 0 || error("$location byte size is negative")
    update = require_integer_value(
        required_property(reference, :update, location),
        "$location update",
    )
    update >= 0 || error("$location update is negative")
    digest = require_sha256(
        required_property(reference, :sha256, location),
        "$location SHA-256",
    )
    return (;
        kind,
        path=abspath(path),
        bytes=Int(bytes),
        sha256=digest,
        update=Int(update),
    )
end

function verify_verification_checkpoint_report(
    report,
    location::AbstractString;
    expected_kind=nothing,
)
    require_exact_properties(
        report,
        VERIFICATION_TRAINING_CHECKPOINT_FIELDS,
        location,
    )
    kind_value = required_property(report, :checkpoint_kind, location)
    kind_value isa AbstractString ||
        error("$location.checkpoint_kind is not a string")
    kind = String(kind_value)
    kind in ("training", "finalization", "residual") || error(
        "$location.checkpoint_kind is invalid: $kind",
    )
    expected_kind === nothing || require_equal(
        kind,
        String(expected_kind),
        "$location checkpoint kind",
    )
    update = required_property(report, :update, location)
    update isa Int || error("$location.update is not Int")
    update >= 0 || error("$location.update is negative")
    path = required_property(report, :path, location)
    path isa AbstractString || error("$location.path is not a string")
    isabspath(String(path)) || error("$location.path is not absolute")
    bytes = required_property(report, :bytes, location)
    bytes isa Int || error("$location.bytes is not Int")
    bytes >= 1 || error("$location.bytes is not positive")
    require_sha256(
        required_property(report, :sha256, location),
        "$location SHA-256",
    )
    require_equal(
        String(required_property(
            report,
            :payload_format,
            location,
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "$location payload format",
    )
    payload_version = required_property(
        report,
        :payload_version,
        location,
    )
    payload_version isa Int ||
        error("$location.payload_version is not Int")
    require_equal(
        payload_version,
        REQUIRED_CHECKPOINT_VERSION,
        "$location payload version",
    )
    for property in (:structural_flips, :utility_updates)
        value = required_property(report, property, location)
        value isa Int ||
            error("$location.$(String(property)) is not Int")
        value >= 0 ||
            error("$location.$(String(property)) is negative")
    end
    require_sha256(
        required_property(
            report,
            :initial_parameters_sha256,
            location,
        ),
        "$location initial-parameter SHA-256",
    )
    verify_finite(
        required_property(report, :hard_gate_budget, location),
        "$location hard gate budget",
    )
    return report
end

function verify_component_loss_alias_contract(
    contract,
    location::AbstractString,
)
    require_exact_properties(
        contract,
        propertynames(COMPONENT_LOSS_ALIAS_CONTRACT),
        location,
    )
    schema_version = required_property(
        contract,
        :schema_version,
        location,
    )
    schema_version isa Int ||
        error("$location.schema_version is not Int")
    require_equal(schema_version, 1, "$location schema version")
    for property in (:q_huber_loss, :raw_top_gap_loss)
        declaration = required_property(contract, property, location)
        expected = getproperty(COMPONENT_LOSS_ALIAS_CONTRACT, property)
        require_exact_properties(
            declaration,
            propertynames(expected),
            "$location.$(String(property))",
        )
        for child in (:alias_of, :identity)
            value = required_property(
                declaration,
                child,
                "$location.$(String(property))",
            )
            value isa AbstractString || error(
                "$location.$(String(property)).$(String(child)) is not a string",
            )
            require_equal(
                String(value),
                getproperty(expected, child),
                "$location.$(String(property)).$(String(child))",
            )
        end
    end
    return contract
end

function require_bit_exact_alias(
    telemetry,
    alias::Symbol,
    canonical::Symbol,
    location::AbstractString,
)
    alias_value = required_property(telemetry, alias, location)
    canonical_value = required_property(telemetry, canonical, location)
    isequal(alias_value, canonical_value) || error(
        "$location.$(String(alias)) is not bit-exact with " *
        "$(String(canonical))",
    )
    return alias_value
end

function verify_component_loss_telemetry(
    telemetry,
    location::AbstractString;
    last_loss=nothing,
)
    require_exact_properties(
        telemetry,
        CHECKPOINT_COMPONENT_LOSS_FIELDS,
        location,
    )
    for property in CHECKPOINT_COMPONENT_LOSS_FIELDS
        value = required_property(telemetry, property, location)
        value isa Float64 || error(
            "$location.$(String(property)) is not Float64",
        )
        isfinite(value) ||
            error("$location.$(String(property)) is not finite")
        value >= 0.0 ||
            error("$location.$(String(property)) is negative")
    end
    for (alias, canonical) in (
        (:q_huber_loss, :old_q_loss),
        (:raw_top_gap_loss, :margin_loss),
        (:window_q_huber_loss_sum, :window_old_q_loss_sum),
        (:window_raw_top_gap_loss_sum, :window_margin_loss_sum),
        (
            :active_window_q_huber_loss_sum,
            :active_window_old_q_loss_sum,
        ),
        (
            :active_window_raw_top_gap_loss_sum,
            :active_window_margin_loss_sum,
        ),
    )
        require_bit_exact_alias(
            telemetry,
            alias,
            canonical,
            location,
        )
    end
    if last_loss !== nothing
        for property in COMPONENT_LOSS_NAMES
            require_approx(
                required_property(telemetry, property, location),
                required_property(
                    last_loss,
                    property,
                    "$location trainer last loss",
                ),
                "$location point $(String(property))";
                atol=1.0e-8,
                rtol=1.0e-7,
            )
        end
    end
    require_approx(
        required_property(telemetry, :geometry_loss, location),
        (
            Float64(telemetry.line_clear_loss) +
            Float64(telemetry.max_height_loss) +
            Float64(telemetry.holes_loss) +
            Float64(telemetry.cavities_loss)
        ) / 4.0,
        "$location geometry decomposition";
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    return telemetry
end

function verify_finalization_record_schema(record, location::AbstractString)
    require_exact_properties(
        record,
        CHECKPOINT_FINALIZATION_FIELDS,
        location,
    )
    require_equal(
        String(required_property(record, :status, location)),
        "finalization_checkpoint_complete",
        "$location status",
    )
    finalized_at = String(required_property(
        record,
        :finalized_at,
        location,
    ))
    isempty(strip(finalized_at)) &&
        error("$location finalization timestamp is blank")
    require_equal(
        require_integer_value(
            required_property(
                record,
                :optimizer_steps_after_target,
                location,
            ),
            "$location optimizer steps after target",
        ),
        0,
        "$location optimizer steps after target",
    )
    for property in (
        :expected_results_path,
        :expected_manifest_path,
    )
        path = String(required_property(record, property, location))
        isempty(strip(path)) &&
            error("$location $(String(property)) is blank")
        isabspath(path) ||
            error("$location $(String(property)) is not absolute")
    end
    verify_checkpoint_reference_schema(
        required_property(record, :training_checkpoint, location),
        "$location training checkpoint";
        expected_kind="training",
    )
    verify_component_loss_alias_contract(
        required_property(
            record,
            :component_loss_alias_contract,
            location,
        ),
        "$location component loss alias contract",
    )
    completed_component_window_updates = required_property(
        record,
        :completed_component_loss_window_updates,
        location,
    )
    completed_component_window_updates isa Int || error(
        "$location.completed_component_loss_window_updates is not Int",
    )
    completed_component_window_updates >= 0 || error(
        "$location.completed_component_loss_window_updates is negative",
    )
    verify_component_loss_telemetry(
        required_property(
            record,
            :component_loss_telemetry,
            location,
        ),
        "$location component loss telemetry",
    )
    verify_finite(
        required_property(record, :final_metrics, location),
        "$location final metrics",
    )
    return record
end

function verify_file_artifact_reference(
    reference,
    expected_path,
    expected_kind,
    expected_update,
    location::AbstractString,
)
    path = abspath(String(required_property(
        reference,
        :path,
        location,
    )))
    require_equal(
        normalized_path(path),
        normalized_path(expected_path),
        "$location path",
    )
    isfile(path) || error("$location file does not exist: $path")
    require_equal(
        String(required_property(reference, :kind, location)),
        expected_kind,
        "$location kind",
    )
    require_equal(
        Int(required_property(reference, :update, location)),
        expected_update,
        "$location update",
    )
    require_equal(
        Int(required_property(reference, :bytes, location)),
        filesize(path),
        "$location byte size",
    )
    digest = file_sha256(path)
    require_equal(
        require_sha256(
            required_property(reference, :sha256, location),
            "$location SHA-256",
        ),
        digest,
        "$location SHA-256",
    )
    return (;
        kind=expected_kind,
        path,
        bytes=filesize(path),
        sha256=digest,
        update=expected_update,
    )
end

function verify_launch_manifest(parsed)
    launch_manifest_path = parsed.launch_manifest_path
    isfile(launch_manifest_path) ||
        error("launch manifest does not exist: $launch_manifest_path")
    launch_manifest_digest = file_sha256(launch_manifest_path)
    require_equal(
        launch_manifest_digest,
        parsed.launch_manifest_sha256,
        "launch manifest SHA-256",
    )
    launch = JSON3.read(read(launch_manifest_path, String))
    require_equal(
        String(required_property(launch, :format, "launch manifest")),
        "serial-workspace-snn-arena-run-launch",
        "launch manifest format",
    )
    require_equal(
        Int(required_property(launch, :version, "launch manifest")),
        2,
        "launch manifest version",
    )
    require_equal(
        String(required_property(launch, :run_id, "launch manifest")),
        parsed.expected_run_id,
        "launch manifest run ID",
    )
    require_equal(
        normalized_path(required_property(
            launch,
            :run_directory,
            "launch manifest",
        )),
        normalized_path(parsed.run_dir),
        "launch manifest run directory",
    )
    require_equal(
        Int(required_property(
            launch,
            :expected_updates,
            "launch manifest",
        )),
        parsed.expected_updates,
        "launch manifest expected updates",
    )
    require_equal(
        String(required_property(
            launch,
            :start_mode,
            "launch manifest",
        )),
        parsed.expected_start_mode,
        "launch manifest start mode",
    )
    active_project = Base.active_project()
    active_project === nothing &&
        error("verifier has no active Julia project")
    require_equal(
        normalized_path(realpath(active_project)),
        normalized_path(realpath(joinpath(
            REQUIRED_PROJECT_PATH,
            "Project.toml",
        ))),
        "active verifier Project.toml",
    )
    require_equal(
        normalized_path(required_property(
            launch,
            :project_path,
            "launch manifest",
        )),
        normalized_path(REQUIRED_PROJECT_PATH),
        "launch manifest canonical project path",
    )
    runtime_arguments = [
        String(argument) for argument in required_property(
            launch,
            :julia_runtime_arguments,
            "launch manifest",
        )
    ]
    require_equal(
        runtime_arguments,
        collect(REQUIRED_JULIA_RUNTIME_ARGUMENTS),
        "launch manifest Julia runtime arguments",
    )
    expected_code_paths = (
        controller=joinpath(
            @__DIR__,
            "run_arena_100k_controller.ps1",
        ),
        training=joinpath(@__DIR__, "train_arena_100k.jl"),
        verifier=@__FILE__,
    )
    code_artifacts = required_property(
        launch,
        :code_artifacts,
        "launch manifest",
    )
    verified_code_artifacts = NamedTuple[]
    for (name, expected_path) in pairs(expected_code_paths)
        artifact = required_property(
            code_artifacts,
            name,
            "launch code artifacts",
        )
        artifact_path = abspath(String(required_property(
            artifact,
            :path,
            "launch code artifact $(String(name))",
        )))
        require_equal(
            normalized_path(artifact_path),
            normalized_path(expected_path),
            "launch code artifact $(String(name)) path",
        )
        isfile(artifact_path) ||
            error("launch code artifact does not exist: $artifact_path")
        require_equal(
            Int(required_property(
                artifact,
                :bytes,
                "launch code artifact $(String(name))",
            )),
            filesize(artifact_path),
            "launch code artifact $(String(name)) byte size",
        )
        artifact_digest = file_sha256(artifact_path)
        require_equal(
            require_sha256(required_property(
                artifact,
                :sha256,
                "launch code artifact $(String(name))",
            ), "launch code artifact $(String(name)) SHA-256"),
            artifact_digest,
            "launch code artifact $(String(name)) SHA-256",
        )
        push!(verified_code_artifacts, (;
            name=String(name),
            path=artifact_path,
            bytes=filesize(artifact_path),
            sha256=artifact_digest,
        ))
    end
    for (property, expected_path) in (
        :controller_script => expected_code_paths.controller,
        :training_script => expected_code_paths.training,
        :verifier_script => expected_code_paths.verifier,
    )
        require_equal(
            normalized_path(required_property(
                launch,
                property,
                "launch manifest",
            )),
            normalized_path(expected_path),
            "launch manifest $(String(property))",
        )
    end

    contract = required_property(
        launch,
        :expected_contract,
        "launch manifest",
    )
    require_equal(
        String(required_property(
            contract,
            :experiment_id,
            "launch expected contract",
        )),
        REQUIRED_EXPERIMENT_ID,
        "launch expected experiment ID",
    )
    require_equal(
        String(required_property(
            contract,
            :checkpoint_format,
            "launch expected contract",
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "launch expected checkpoint format",
    )
    require_equal(
        Int(required_property(
            contract,
            :checkpoint_version,
            "launch expected contract",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "launch expected checkpoint version",
    )
    require_equal(
        String(required_property(
            contract,
            :learning_mode,
            "launch expected contract",
        )),
        "local_hybrid",
        "launch expected learning mode",
    )
    require_equal(
        String(required_property(
            contract,
            :model_preset,
            "launch expected contract",
        )),
        REQUIRED_MODEL_PRESET,
        "launch expected model preset",
    )
    require_equal(
        String(required_property(
            contract,
            :start_mode,
            "launch expected contract",
        )),
        parsed.expected_start_mode,
        "launch expected start mode",
    )
    require_equal(
        Bool(required_property(
            contract,
            :scratch,
            "launch expected contract",
        )),
        parsed.expected_start_mode == "scratch",
        "launch expected scratch flag",
    )
    require_equal(
        normalized_path(required_property(
            contract,
            :canonical_project_path,
            "launch expected contract",
        )),
        normalized_path(REQUIRED_PROJECT_PATH),
        "launch expected canonical project path",
    )
    require_equal(
        Bool(required_property(
            contract,
            :startup_file,
            "launch expected contract",
        )),
        false,
        "launch expected startup-file flag",
    )
    require_equal(
        Bool(required_property(
            contract,
            :history_file,
            "launch expected contract",
        )),
        false,
        "launch expected history-file flag",
    )
    for property in (
        :require_dataset_content_sha256,
        :require_dataset_integrity,
        :require_runtime_provenance,
        :require_source_fingerprint,
    )
        require_equal(
            Bool(required_property(
                contract,
                property,
                "launch expected contract",
            )),
            true,
            "launch expected $(String(property)) flag",
        )
    end
    require_equal(
        Int(required_property(
            launch,
            :julia_threads,
            "launch manifest",
        )),
        Int(required_property(
            contract,
            :active_workers,
            "launch expected contract",
        )),
        "launch Julia thread count",
    )

    expected_environment = (
        SWSNN_LEARNING_MODE="local_hybrid",
        SWSNN_MODEL_PRESET="scaled_v2",
        SWSNN_STATE_BATCH="8",
        SWSNN_ACTIVE_WORKERS=string(Int(required_property(
            contract,
            :active_workers,
            "launch expected contract",
        ))),
        SWSNN_EPROP_REDUCERS=string(Int(required_property(
            contract,
            :eprop_reducers,
            "launch expected contract",
        ))),
        SWSNN_CPUSET_MODE="all",
        SWSNN_LEARNING_RATE="0.0005",
        SWSNN_WEIGHT_DECAY="0.00001",
        SWSNN_STRUCTURE_WEIGHT="0.01",
        SWSNN_STRUCTURAL_INTERVAL="25",
        SWSNN_CHECKPOINT_INTERVAL=string(Int(required_property(
            contract,
            :checkpoint_interval,
            "launch expected contract",
        ))),
        SWSNN_LOG_INTERVAL=string(Int(required_property(
            contract,
            :log_interval,
            "launch expected contract",
        ))),
        SWSNN_EVAL_STATES=string(Int(required_property(
            contract,
            :evaluation_states,
            "launch expected contract",
        ))),
        SWSNN_UTILITY_DECAY="0.99",
        SWSNN_UTILITY_CONNECTION_COST="0.000001",
        SWSNN_UTILITY_KEEP_FRACTION="0.5",
        SWSNN_UTILITY_TURNOVER_PERIOD="128",
        SWSNN_MAX_HOT_ALLOCATION_BYTES="0",
        SWSNN_MAX_UPDATES=string(parsed.expected_updates),
        SWSNN_RUN_ID=parsed.expected_run_id,
        SWSNN_START_MODE=replace(parsed.expected_start_mode, "-" => "_"),
        SWSNN_SCRATCH=
            parsed.expected_start_mode == "scratch" ? "true" : "false",
        SWSNN_DATASET=String(required_property(
            contract,
            :dataset_path,
            "launch expected contract",
        )),
    )
    environment =
        required_property(launch, :environment, "launch manifest")
    for (name, expected_value) in pairs(expected_environment)
        require_equal(
            String(required_property(
                environment,
                name,
                "launch environment",
            )),
            expected_value,
            "launch environment $(String(name))",
        )
    end

    launch_parent = required_property(
        launch,
        :parent_checkpoint,
        "launch manifest",
    )
    if parsed.parent_checkpoint === nothing
        launch_parent === nothing ||
            error("scratch launch manifest unexpectedly has a parent")
        for property in (:SWSNN_RESUME_CHECKPOINT, :SWSNN_RESUME_SHA256)
            hasproperty(environment, property) && error(
                "scratch launch environment contains $(String(property))",
            )
        end
    else
        launch_parent === nothing &&
            error("launch manifest is missing its parent checkpoint")
        require_equal(
            normalized_path(required_property(
                launch_parent,
                :path,
                "launch parent checkpoint",
            )),
            normalized_path(parsed.parent_checkpoint.path),
            "launch parent checkpoint path",
        )
        require_equal(
            require_sha256(required_property(
                launch_parent,
                :sha256,
                "launch parent checkpoint",
            ), "launch parent checkpoint SHA-256"),
            parsed.parent_checkpoint.sha256,
            "launch parent checkpoint SHA-256",
        )
        require_equal(
            Int(required_property(
                launch_parent,
                :update,
                "launch parent checkpoint",
            )),
            parsed.parent_checkpoint.update,
            "launch parent checkpoint update",
        )
        require_equal(
            normalized_path(required_property(
                environment,
                :SWSNN_RESUME_CHECKPOINT,
                "launch environment",
            )),
            normalized_path(parsed.parent_checkpoint.path),
            "launch resume checkpoint environment path",
        )
        require_equal(
            lowercase(String(required_property(
                environment,
                :SWSNN_RESUME_SHA256,
                "launch environment",
            ))),
            parsed.parent_checkpoint.sha256,
            "launch resume checkpoint environment SHA-256",
        )
    end
    return (;
        path=launch_manifest_path,
        bytes=filesize(launch_manifest_path),
        sha256=launch_manifest_digest,
        document=launch,
        contract,
        code_artifacts=verified_code_artifacts,
    )
end

function verify_launch_binding_record(
    binding,
    expected_path,
    expected_sha256,
    location::AbstractString,
)
    require_exact_properties(
        binding,
        LAUNCH_BINDING_FIELDS,
        location,
    )
    path = abspath(String(required_property(binding, :path, location)))
    require_equal(
        normalized_path(path),
        normalized_path(expected_path),
        "$location path",
    )
    isfile(path) || error("$location file does not exist: $path")
    islink(path) &&
        error("$location is a symbolic link: $path")
    require_equal(
        normalized_path(realpath(path)),
        normalized_path(path),
        "$location canonical path",
    )
    digest = file_sha256(path)
    require_equal(
        require_sha256(
            required_property(binding, :sha256, location),
            "$location SHA-256",
        ),
        digest,
        "$location live SHA-256",
    )
    require_equal(
        digest,
        require_sha256(expected_sha256, "$location expected SHA-256"),
        "$location pinned SHA-256",
    )
    return (; path, sha256=digest)
end

function verify_segment_launch_binding(
    config,
    run_dir,
    expected_parent,
    location::AbstractString,
)
    run_id = String(required_property(
        config,
        :run_id,
        "$location config",
    ))
    output_root = dirname(run_dir)
    expected_path = joinpath(
        output_root,
        "_controllers",
        run_id,
        "launch_manifest.json",
    )
    binding = required_property(
        config,
        :launch_binding,
        "$location config",
    )
    binding_path = abspath(String(required_property(
        binding,
        :path,
        "$location launch binding",
    )))
    binding_sha256 = require_sha256(
        required_property(
            binding,
            :sha256,
            "$location launch binding",
        ),
        "$location launch binding SHA-256",
    )
    verified_binding = verify_launch_binding_record(
        binding,
        expected_path,
        binding_sha256,
        "$location launch binding",
    )
    launch_document = JSON3.read(read(verified_binding.path, String))
    launch_parent = required_property(
        launch_document,
        :parent_checkpoint,
        "$location launch manifest",
    )
    parsed_parent = if launch_parent === nothing
        nothing
    else
        (;
            path=abspath(String(required_property(
                launch_parent,
                :path,
                "$location launch parent",
            ))),
            sha256=require_sha256(required_property(
                launch_parent,
                :sha256,
                "$location launch parent",
            ), "$location launch parent SHA-256"),
            update=Int(require_integer_value(
                required_property(
                    launch_parent,
                    :update,
                    "$location launch parent",
                ),
                "$location launch parent update",
            )),
        )
    end
    start_mode = replace(
        String(required_property(
            config,
            :start_mode,
            "$location config",
        )),
        "_" => "-",
    )
    parsed = (;
        launch_manifest_path=verified_binding.path,
        launch_manifest_sha256=verified_binding.sha256,
        expected_run_id=run_id,
        run_dir=abspath(run_dir),
        expected_updates=Int(required_property(
            config,
            :maximum_updates,
            "$location config",
        )),
        expected_start_mode=start_mode,
        parent_checkpoint=parsed_parent,
    )
    launch = verify_launch_manifest(parsed)
    config_path = joinpath(run_dir, "config.json")
    isfile(config_path) ||
        error("$location config.json does not exist: $config_path")
    islink(config_path) &&
        error("$location config.json is a symbolic link")
    require_equal(
        normalized_path(realpath(config_path)),
        normalized_path(config_path),
        "$location canonical config.json path",
    )
    config_document = JSON3.read(read(config_path, String))
    require_semantic_equal(
        required_property(
            config_document,
            :config,
            "$location config.json",
        ),
        config,
        "$location checkpoint config versus config.json",
    )
    require_semantic_equal(
        required_property(
            config_document,
            :parent_checkpoint,
            "$location config.json",
        ),
        expected_parent,
        "$location checkpoint parent versus config.json",
    )
    return launch
end

function verify_production_config(config, parsed, launch)
    verify_launch_binding_record(
        required_property(config, :launch_binding, "run config"),
        launch.path,
        launch.sha256,
        "run launch binding",
    )
    require_equal(
        String(required_property(
            config,
            :experiment_id,
            "run config",
        )),
        REQUIRED_EXPERIMENT_ID,
        "run experiment ID",
    )
    schema = required_property(config, :checkpoint_schema, "run config")
    require_equal(
        String(required_property(schema, :format, "checkpoint schema")),
        REQUIRED_CHECKPOINT_FORMAT,
        "configured checkpoint format",
    )
    require_equal(
        Int(required_property(schema, :version, "checkpoint schema")),
        REQUIRED_CHECKPOINT_VERSION,
        "configured checkpoint version",
    )
    production_contract = required_property(
        config,
        :production_contract,
        "run config",
    )
    require_equal(
        Int(required_property(
            production_contract,
            :version,
            "production contract",
        )),
        Int(PRODUCTION_CONTRACT_VERSION),
        "production contract version",
    )
    require_equal(
        String(required_property(
            production_contract,
            :experiment_id,
            "production contract",
        )),
        REQUIRED_EXPERIMENT_ID,
        "production contract experiment ID",
    )
    require_sha256(
        required_property(
            config,
            :production_contract_sha256,
            "run config",
        ),
        "production contract SHA-256",
    )
    start_mode = replace(
        String(required_property(config, :start_mode, "run config")),
        "_" => "-",
    )
    require_equal(
        start_mode,
        parsed.expected_start_mode,
        "configured start mode",
    )
    require_equal(
        Bool(required_property(config, :scratch, "run config")),
        start_mode == "scratch",
        "configured scratch flag",
    )
    require_equal(
        String(required_property(config, :run_id, "run config")),
        parsed.expected_run_id,
        "configured run ID",
    )
    require_equal(
        Int(required_property(
            config,
            :maximum_updates,
            "run config",
        )),
        parsed.expected_updates,
        "configured maximum updates",
    )
    require_equal(
        String(required_property(config, :learning_mode, "run config")),
        "local_hybrid",
        "configured learning mode",
    )
    require_equal(
        String(required_property(config, :model_preset, "run config")),
        REQUIRED_MODEL_PRESET,
        "configured model preset",
    )

    contract = launch.contract
    for property in (
        :state_batch,
        :active_workers,
        :eprop_reducers,
        :structural_interval,
        :checkpoint_interval,
        :log_interval,
    )
        require_equal(
            Int(required_property(config, property, "run config")),
            Int(required_property(
                contract,
                property,
                "launch expected contract",
            )),
            "configured $(String(property))",
        )
    end
    require_equal(
        Int(required_property(
            config,
            :training_eval_states,
            "run config",
        )),
        Int(required_property(
            contract,
            :evaluation_states,
            "launch expected contract",
        )),
        "configured evaluation-state count",
    )
    require_equal(
        String(required_property(config, :cpuset_mode, "run config")),
        "all",
        "configured CPU-set mode",
    )
    require_equal(
        Int(required_property(config, :julia_threads, "run config")),
        Int(required_property(
            contract,
            :active_workers,
            "launch expected contract",
        )),
        "configured Julia thread count",
    )
    require_equal(
        Int(required_property(config, :blas_threads, "run config")),
        1,
        "configured BLAS thread count",
    )
    require_equal(
        Int(required_property(
            config,
            :target_teacher_states,
            "run config",
        )),
        parsed.expected_updates *
        Int(required_property(config, :state_batch, "run config")),
        "configured target teacher states",
    )
    require_approx(
        required_property(config, :learning_rate, "run config"),
        5.0e-4,
        "configured learning rate",
    )
    require_approx(
        required_property(config, :weight_decay, "run config"),
        1.0e-5,
        "configured weight decay",
    )
    require_approx(
        required_property(config, :structure_weight, "run config"),
        1.0e-2,
        "configured structure weight",
    )
    require_equal(
        Int(required_property(
            config,
            :maximum_hot_allocation_bytes,
            "run config",
        )),
        0,
        "configured maximum hot allocation",
    )

    expected_model = (
        blocks=96,
        nodes=4608,
        candidate_synapses=110592,
        enabled_synapses=55296,
        fanout=24,
        cycles=4,
        workspace_capacity=8,
        input_rails=1298,
    )
    model = required_property(config, :model, "run config")
    for (property, expected_value) in pairs(expected_model)
        require_equal(
            Int(required_property(model, property, "configured model")),
            expected_value,
            "configured model $(String(property))",
        )
    end
    require_equal(
        Int(required_property(config, :parameter_count, "run config")),
        444599,
        "configured parameter count",
    )
    require_equal(
        Int(required_property(config, :candidate_width, "run config")),
        80,
        "configured candidate width",
    )
    representation =
        required_property(config, :representation, "run config")
    for (property, expected_value) in (
        :query_role => "routing_only",
        :head_features => "workspace_plus_hard_selected_pool",
        :query_normalization => "candidate_local_rmsnorm_tanh",
        :hidden_normalization => "candidate_local_rmsnorm_tanh",
    )
        require_equal(
            String(required_property(
                representation,
                property,
                "configured representation",
            )),
            expected_value,
            "configured representation $(String(property))",
        )
    end
    for (property, expected_value) in (
        :rms_epsilon => 1.0e-4,
        :query_norm_scale => 0.5,
        :hidden_norm_scale => 0.75,
    )
        require_approx(
            required_property(
                representation,
                property,
                "configured representation",
            ),
            expected_value,
            "configured representation $(String(property))",
        )
    end
    retention = required_property(
        config,
        :workspace_retention,
        "run config",
    )
    require_equal(
        String(required_property(
            retention,
            :parameterization,
            "configured workspace retention",
        )),
        "bounded_sigmoid",
        "configured workspace-retention parameterization",
    )
    require_approx(
        required_property(
            retention,
            :minimum,
            "configured workspace retention",
        ),
        0.60,
        "configured workspace-retention minimum",
    )
    require_approx(
        required_property(
            retention,
            :maximum,
            "configured workspace retention",
        ),
        0.95,
        "configured workspace-retention maximum",
    )
    require_approx(
        required_property(
            required_property(config, :spiking, "run config"),
            :spike_temperature,
            "configured spiking",
        ),
        0.20,
        "configured spike temperature",
    )
    require_json_uint64(
        required_property(config, :model_seed, "run config"),
        UInt64(MODEL_SEED),
        "configured model seed",
    )
    require_json_uint64(
        required_property(config, :sampler_seed, "run config"),
        UInt64(SAMPLER_SEED),
        "configured sampler seed",
    )
    require_json_uint64(
        required_property(config, :routing_seed, "run config"),
        UInt64(ROUTING_SEED),
        "configured routing seed",
    )

    eprop = required_property(config, :eprop, "run config")
    Bool(required_property(eprop, :enabled, "configured e-prop")) ||
        error("configured e-prop is disabled")
    expected_eprop = (
        trace_decay_scale=1.0,
        feedback_seed=UInt64(0x4550524f50534844),
        feedback_scale=1.0,
        feedback_mode="block_local",
        eligibility_mode="membrane",
        error_signal_mode="full_raw",
        edge_parameter_mode="weight_gate_delay",
        node_parameter_mode="full_state",
        routing_parameter_mode="three_factor",
        signal_schedule="all_cycles",
        third_factor_mode="aligned",
        time_order="forward",
        routing_entropy_weight=0.002,
        routing_entropy_floor=0.70,
        routing_load_weight=0.002,
    )
    for (property, expected_value) in pairs(expected_eprop)
        observed = required_property(eprop, property, "configured e-prop")
        if expected_value isa Integer
            require_json_uint64(
                observed,
                UInt64(expected_value),
                "configured e-prop $(String(property))",
            )
        elseif expected_value isa Number
            require_approx(
                observed,
                expected_value,
                "configured e-prop $(String(property))",
            )
        else
            require_equal(
                String(observed),
                expected_value,
                "configured e-prop $(String(property))",
            )
        end
    end

    routing = required_property(config, :routing, "run config")
    require_json_uint64(
        required_property(
            routing,
            :routing_seed,
            "configured routing",
        ),
        UInt64(ROUTING_SEED),
        "configured routing seed",
    )
    expected_routing = (
        inference_selection="deterministic_hard_top_k",
        training_selection=
            "stochastic_hard_top_k_without_replacement",
        parameter_update=
            "ordered_plackett_luce_score_eligibility_three_factor",
        learning_signal_semantics="supervised_reward_surrogate",
        reward_source=
            "candidate_centered_listnet_and_auxiliary_loss_advantage",
    )
    for (property, expected_value) in pairs(expected_routing)
        require_equal(
            String(required_property(
                routing,
                property,
                "configured routing",
            )),
            expected_value,
            "configured routing $(String(property))",
        )
    end
    for (property, expected_value) in (
        :route_probability_mass => 1.0,
        :route_temperature => 0.35,
        :exploration_probability => 0.05,
        :score_normalization_epsilon => 1.0e-4,
        :entropy_weight => 0.002,
        :entropy_floor => 0.70,
        :load_balance_weight => 0.002,
    )
        require_approx(
            required_property(
                routing,
                property,
                "configured routing",
            ),
            expected_value,
            "configured routing $(String(property))",
        )
    end

    executor = required_property(config, :executor, "run config")
    for (property, expected_value) in (
        :fixed_candidate_arenas => true,
        :analytic_vjp => false,
        :supervised_head_vjp => true,
        :worker_local_gradients => true,
        :mpmc_isbits_jobs => true,
        :parallel_in_place_adamw => true,
        :gc_disabled_inside_hot_training => true,
    )
        require_equal(
            Bool(required_property(
                executor,
                property,
                "configured executor",
            )),
            expected_value,
            "configured executor $(String(property))",
        )
    end
    require_equal(
        String(required_property(
            executor,
            :recurrent_credit_assignment,
            "configured executor",
        )),
        "eprop_decolle_block_local_three_factor",
        "configured recurrent credit assignment",
    )
    require_equal(
        Int(required_property(
            executor,
            :eprop_reducers,
            "configured executor",
        )),
        Int(required_property(config, :eprop_reducers, "run config")),
        "executor e-prop reducer count",
    )
    structural = required_property(
        executor,
        :structural_learning,
        "configured executor",
    )
    require_equal(
        String(required_property(
            structural,
            :mode,
            "configured structural learner",
        )),
        "utility",
        "configured structural learner mode",
    )
    for (property, expected_value) in (
        :utility_decay => 0.99,
        :utility_connection_cost => 1.0e-6,
        :utility_keep_fraction => 0.5,
    )
        require_approx(
            required_property(
                structural,
                property,
                "configured structural learner",
            ),
            expected_value,
            "configured structural learner $(String(property))",
        )
    end
    require_equal(
        Int(required_property(
            structural,
            :utility_turnover_period,
            "configured structural learner",
        )),
        128,
        "configured structural learner turnover period",
    )
    require_equal(
        String(required_property(
            structural,
            :responsibility,
            "configured structural learner",
        )),
        "normalized_abs_block_signal_times_eligibility",
        "configured structural learner responsibility",
    )

    source_digest = require_sha256(
        required_property(config, :source_fingerprint, "run config"),
        "configured source fingerprint",
    )
    require_equal(
        source_digest,
        source_fingerprint(),
        "configured source fingerprint versus verifier source",
    )
    require_sha256(
        required_property(
            config,
            :dataset_content_sha256,
            "run config",
        ),
        "configured dataset content SHA-256",
    )
    required_property(config, :dataset_integrity, "run config")
    required_property(config, :runtime_provenance, "run config")
    duplicate_contract_fields = (
        :model_preset,
        :model,
        :parameter_count,
        :maximum_updates,
        :state_batch,
        :candidate_width,
        :active_workers,
        :eprop_reducers,
        :cpuset_mode,
        :julia_threads,
        :blas_threads,
        :learning_mode,
        :structural_interval,
        :checkpoint_interval,
        :log_interval,
        :maximum_hot_allocation_bytes,
        :dataset_content_sha256,
        :dataset_integrity,
        :training_rows,
        :training_rows_sha256,
        :training_panel_rows_sha256,
        :model_seed,
        :sampler_seed,
        :representation,
        :workspace_retention,
        :spiking,
        :eprop,
        :routing,
        :executor,
        :runtime_provenance,
    )
    for property in duplicate_contract_fields
        require_semantic_equal(
            required_property(
                production_contract,
                property,
                "production contract",
            ),
            required_property(config, property, "run config"),
            "production contract $(String(property))",
        )
    end
    require_equal(
        Int(required_property(
            production_contract,
            :evaluation_states,
            "production contract",
        )),
        Int(required_property(
            config,
            :training_eval_states,
            "run config",
        )),
        "production contract evaluation states",
    )
    require_semantic_equal(
        required_property(
            production_contract,
            :optimizer,
            "production contract",
        ),
        required_property(config, :optimizer, "run config"),
        "production contract optimizer",
    )
    if parsed.expected_updates == 100_000 &&
       parsed.expected_start_mode == "scratch"
        Bool(required_property(
            config,
            :production_target_match,
            "run config",
        )) || error("100k scratch run does not match production target")
    end
    return config
end

function verify_bound_inputs(config, launch)
    expected_runtime = runtime_provenance(String(
        required_property(config, :source_fingerprint, "run config"),
    ))
    observed_runtime = required_property(
        config,
        :runtime_provenance,
        "run config",
    )
    require_semantic_equal(
        observed_runtime,
        expected_runtime,
        "runtime provenance versus current verifier runtime",
    )
    launch_document = launch.document
    require_equal(
        normalized_path(required_property(
            observed_runtime,
            :julia_executable_path,
            "runtime provenance",
        )),
        normalized_path(required_property(
            launch_document,
            :julia_executable,
            "launch manifest",
        )),
        "runtime Julia executable versus launch",
    )
    configured_project = required_property(
        observed_runtime,
        :project_toml_path,
        "runtime provenance",
    )
    require_equal(
        normalized_path(dirname(String(configured_project))),
        normalized_path(required_property(
            launch_document,
            :project_path,
            "launch manifest",
        )),
        "runtime project versus launch",
    )

    dataset_path = abspath(String(required_property(
        config,
        :dataset_path,
        "run config",
    )))
    require_equal(
        normalized_path(dataset_path),
        normalized_path(required_property(
            launch.contract,
            :dataset_path,
            "launch expected contract",
        )),
        "configured dataset path versus launch",
    )
    preflight = dataset_binding_preflight(dataset_path)
    evaluation_states = Int(required_property(
        config,
        :training_eval_states,
        "run config",
    ))
    state_batch =
        Int(required_property(config, :state_batch, "run config"))
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(state_batch, evaluation_states),
    )
    dataset_content_sha256, dataset_integrity =
        bind_loaded_dataset(dataset_path, dataset, preflight)
    require_equal(
        String(required_property(
            config,
            :dataset_content_sha256,
            "run config",
        )),
        dataset_content_sha256,
        "dataset content SHA-256",
    )
    require_semantic_equal(
        required_property(config, :dataset_integrity, "run config"),
        dataset_integrity,
        "dataset integrity evidence",
    )
    Bool(required_property(
        config,
        :validation_rows_used,
        "run config",
    )) && error("production training config used validation rows")
    Bool(required_property(
        config,
        :game_validation_used,
        "run config",
    )) && error("production training config used game validation")
    Bool(required_property(
        config,
        :sealed_seeds_used,
        "run config",
    )) && error("production training config used sealed seeds")

    training_rows = training_rows_only(dataset)
    require_equal(
        length(training_rows),
        Int(required_property(config, :training_rows, "run config")),
        "configured training-row count",
    )
    require_equal(
        bytes2hex(sha256(reinterpret(UInt8, training_rows))),
        String(required_property(
            config,
            :training_rows_sha256,
            "run config",
        )),
        "configured training-row SHA-256",
    )
    panel_rows = fixed_training_panel(training_rows, evaluation_states)
    require_equal(
        length(panel_rows),
        evaluation_states,
        "fixed training panel size",
    )
    require_equal(
        bytes2hex(sha256(reinterpret(UInt8, panel_rows))),
        String(required_property(
            config,
            :training_panel_rows_sha256,
            "run config",
        )),
        "fixed training panel SHA-256",
    )
    return (; dataset, training_rows, panel_rows)
end

function verify_array_registry(
    registry,
    expected_parameters,
    location::AbstractString,
)
    keys(registry) == ArenaWorkspaceTraining.PARAMETER_FIELDS || error(
        "$location parameter registry differs",
    )
    for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
        values = required_property(registry, name, location)
        expected = required_property(
            expected_parameters,
            name,
            "expected parameter registry",
        )
        size(values) == size(expected) || error(
            "$location.$(String(name)) shape differs: " *
            "observed=$(size(values)) expected=$(size(expected))",
        )
        all(isfinite, values) || error(
            "$location.$(String(name)) contains a non-finite value",
        )
    end
    return registry
end

function verify_hard_gate_budget(parameters, model_config, location)
    logits = parameters.gate_logits
    nodes = Int(required_property(
        model_config,
        :nodes,
        "configured model",
    ))
    enabled = Int(required_property(
        model_config,
        :enabled_synapses,
        "configured model",
    ))
    size(logits, 1) == nodes ||
        error("$location gate-logit node count differs")
    enabled % nodes == 0 ||
        error("configured enabled-synapse budget is not per-node integral")
    expected_per_node = enabled ÷ nodes
    total_enabled = 0
    @inbounds for node in axes(logits, 1)
        node_enabled = 0
        for relation in axes(logits, 2)
            node_enabled += logits[node, relation] >= 0.0f0
        end
        node_enabled == expected_per_node || error(
            "$location node $node hard-gate budget differs: " *
            "observed=$node_enabled expected=$expected_per_node",
        )
        total_enabled += node_enabled
    end
    total_enabled == enabled ||
        error("$location total hard-gate budget differs")
    return (; expected_per_node, total_enabled)
end

function deterministic_sampler_snapshot(
    training_rows,
    update::Integer,
    state_batch::Integer,
)
    sampler = EpochSampler(training_rows, Xoshiro(SAMPLER_SEED))
    states = Int(update) * Int(state_batch)
    states > 0 && next_batch!(sampler, states)
    return sampler_snapshot(sampler)
end

function verify_checkpoint_payload_core(
    payload,
    expected_parameters,
    training_rows,
    state_batch::Integer;
    location::AbstractString,
    expected_update::Integer,
    expected_kind::AbstractString,
    expected_config=nothing,
    expected_sampler_snapshot=nothing,
    expected_initial_metrics=nothing,
    expected_parent=nothing,
    expected_parent_is_set::Bool=false,
)
    require_exact_properties(
        payload,
        CHECKPOINT_PAYLOAD_FIELDS,
        location,
    )
    require_equal(
        String(required_property(payload, :format, location)),
        REQUIRED_CHECKPOINT_FORMAT,
        "$location format",
    )
    require_equal(
        Int(require_integer_value(
            required_property(payload, :version, location),
            "$location version",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "$location version",
    )
    payload_alias_contract = required_property(
        payload,
        :component_loss_alias_contract,
        location,
    )
    verify_component_loss_alias_contract(
        payload_alias_contract,
        "$location component loss alias contract",
    )
    kind = String(required_property(
        payload,
        :checkpoint_kind,
        location,
    ))
    require_equal(kind, String(expected_kind), "$location kind")
    update = Int(require_integer_value(
        required_property(payload, :update, location),
        "$location update",
    ))
    require_equal(update, Int(expected_update), "$location update")
    update >= 0 || error("$location update is negative")
    state_batch = Int(state_batch)
    state_batch >= 1 || error("$location state batch is not positive")

    config = required_property(payload, :config, location)
    expected_config === nothing || require_semantic_equal(
        config,
        expected_config,
        "$location config",
    )
    require_equal(
        Int(require_integer_value(
            required_property(config, :state_batch, "$location config"),
            "$location configured state batch",
        )),
        state_batch,
        "$location configured state batch",
    )
    schema = required_property(
        config,
        :checkpoint_schema,
        "$location config",
    )
    require_equal(
        String(required_property(
            schema,
            :format,
            "$location checkpoint schema",
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "$location configured checkpoint format",
    )
    require_equal(
        Int(require_integer_value(
            required_property(
                schema,
                :version,
                "$location checkpoint schema",
            ),
            "$location configured checkpoint version",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "$location configured checkpoint version",
    )
    dataset_digest = require_sha256(
        required_property(payload, :dataset_content_sha256, location),
        "$location dataset content SHA-256",
    )
    require_equal(
        dataset_digest,
        require_sha256(required_property(
            config,
            :dataset_content_sha256,
            "$location config",
        ), "$location configured dataset content SHA-256"),
        "$location dataset content binding",
    )
    for property in (:dataset_integrity, :runtime_provenance)
        require_semantic_equal(
            required_property(payload, property, location),
            required_property(config, property, "$location config"),
            "$location $(String(property)) binding",
        )
    end

    parameters = required_property(payload, :parameters, location)
    initial_parameters = required_property(
        payload,
        :initial_parameters,
        location,
    )
    verify_array_registry(
        parameters,
        expected_parameters,
        "$location parameters",
    )
    verify_array_registry(
        initial_parameters,
        expected_parameters,
        "$location initial parameters",
    )
    require_exact_state_equal(
        initial_parameters,
        expected_parameters,
        "$location deterministic initial parameters",
    )
    gate_budget = verify_hard_gate_budget(
        parameters,
        required_property(config, :model, "$location config"),
        location,
    )

    optimizer = required_property(payload, :optimizer, location)
    require_exact_properties(
        optimizer,
        CHECKPOINT_OPTIMIZER_FIELDS,
        "$location optimizer",
    )
    first_moment = required_property(
        optimizer,
        :first_moment,
        "$location optimizer",
    )
    second_moment = required_property(
        optimizer,
        :second_moment,
        "$location optimizer",
    )
    verify_array_registry(
        first_moment,
        expected_parameters,
        "$location optimizer first moment",
    )
    verify_array_registry(
        second_moment,
        expected_parameters,
        "$location optimizer second moment",
    )
    optimizer_step = Int(require_integer_value(
        required_property(optimizer, :step, "$location optimizer"),
        "$location optimizer step",
    ))
    require_equal(
        optimizer_step,
        update,
        "$location optimizer step",
    )
    optimizer_config = required_property(
        config,
        :optimizer,
        "$location config",
    )
    for property in (
        :learning_rate,
        :beta1,
        :beta2,
        :epsilon,
        :weight_decay,
    )
        value = require_finite_real(
            required_property(optimizer, property, "$location optimizer"),
            "$location optimizer.$(String(property))",
        )
        require_semantic_equal(
            required_property(optimizer, property, "$location optimizer"),
            required_property(
                optimizer_config,
                property,
                "$location configured optimizer",
            ),
            "$location optimizer.$(String(property)) versus config",
        )
        property in (:learning_rate, :epsilon, :weight_decay) &&
            value < 0.0 && error(
                "$location optimizer.$(String(property)) is negative",
            )
    end
    for property in (:beta1_power, :beta2_power)
        value = require_finite_real(
            required_property(optimizer, property, "$location optimizer"),
            "$location optimizer.$(String(property))",
        )
        0.0 <= value <= 1.0 || error(
            "$location optimizer.$(String(property)) is outside [0, 1]",
        )
    end
    for property in (:beta1, :beta2)
        value = Float64(required_property(
            optimizer,
            property,
            "$location optimizer",
        ))
        0.0 <= value < 1.0 || error(
            "$location optimizer.$(String(property)) is outside [0, 1)",
        )
    end

    trainer_state = required_property(
        payload,
        :trainer_state,
        location,
    )
    require_exact_properties(
        trainer_state,
        CHECKPOINT_TRAINER_STATE_FIELDS,
        "$location trainer state",
    )
    last_loss = required_property(
        trainer_state,
        :last_loss,
        "$location trainer state",
    )
    require_exact_properties(
        last_loss,
        CHECKPOINT_LOSS_FIELDS,
        "$location trainer last loss",
    )
    valid_candidates = Int(require_integer_value(
        required_property(
            last_loss,
            :valid_candidates,
            "$location trainer last loss",
        ),
        "$location trainer last-loss valid candidate count",
    ))
    valid_candidates >= 0 ||
        error("$location trainer valid-candidate count is negative")
    loss_float_fields = filter(
        !isequal(:valid_candidates),
        CHECKPOINT_LOSS_FIELDS,
    )
    if update == 0
        for property in loss_float_fields
            value = required_property(
                last_loss,
                property,
                "$location trainer last loss",
            )
            value isa Float32 && isequal(value, 0.0f0) || error(
                "$location checkpoint-zero last loss " *
                "$(String(property)) is not Float32 zero",
            )
        end
        valid_candidates == 0 ||
            error("$location checkpoint-zero valid candidates are nonzero")
        gradient_norm = required_property(
            trainer_state,
            :last_gradient_norm,
            "$location trainer state",
        )
        gradient_norm isa AbstractFloat && isnan(gradient_norm) || error(
            "$location checkpoint-zero last gradient norm is not NaN",
        )
    else
        for property in loss_float_fields
            value = required_property(
                last_loss,
                property,
                "$location trainer last loss",
            )
            value isa Float32 || error(
                "$location trainer last loss.$(String(property)) " *
                "is not Float32",
            )
            require_finite_real(
                value,
                "$location trainer last loss.$(String(property))",
            )
            Float64(value) >= 0.0 || error(
                "$location trainer last loss.$(String(property)) " *
                "is negative",
            )
        end
        valid_candidates > 0 ||
            error("$location trained checkpoint has no valid candidates")
        require_finite_real(
            required_property(
                trainer_state,
                :last_gradient_norm,
                "$location trainer state",
            ),
            "$location trainer last gradient norm",
        )
    end
    require_bit_exact_alias(
        last_loss,
        :q_huber_loss,
        :old_q_loss,
        "$location trainer last loss",
    )
    require_bit_exact_alias(
        last_loss,
        :raw_top_gap_loss,
        :margin_loss,
        "$location trainer last loss",
    )
    structure_weight = required_property(
        trainer_state,
        :structure_weight,
        "$location trainer state",
    )
    require_finite_real(
        structure_weight,
        "$location trainer structure weight",
    )
    require_semantic_equal(
        structure_weight,
        required_property(
            config,
            :structure_weight,
            "$location config",
        ),
        "$location trainer structure weight versus config",
    )

    utility = required_property(payload, :synapse_utility, location)
    utility isa AbstractArray ||
        error("$location synapse utility is not an array")
    size(utility) == size(parameters.synapse_weight) ||
        error("$location synapse utility shape differs")
    all(isfinite, utility) ||
        error("$location synapse utility contains non-finite values")
    all(>=(zero(eltype(utility))), utility) ||
        error("$location synapse utility contains negative values")
    utility_updates = Int(require_integer_value(
        required_property(payload, :utility_updates, location),
        "$location utility update count",
    ))
    require_equal(
        utility_updates,
        update,
        "$location utility update count",
    )
    structural_flips = Int(require_integer_value(
        required_property(payload, :total_structural_flips, location),
        "$location structural flip count",
    ))
    structural_flips >= 0 ||
        error("$location structural flip count is negative")

    sampler_state = required_property(payload, :sampler_state, location)
    require_exact_properties(
        sampler_state,
        CHECKPOINT_SAMPLER_FIELDS,
        "$location sampler state",
    )
    restored_sampler = restore_sampler(training_rows, sampler_state)
    require_equal(
        sampler_consumed_states(restored_sampler),
        update * state_batch,
        "$location sampler consumed-state count",
    )
    sampler_expectation = expected_sampler_snapshot === nothing ?
        deterministic_sampler_snapshot(
            training_rows,
            update,
            state_batch,
        ) : expected_sampler_snapshot
    require_exact_state_equal(
        sampler_state,
        sampler_expectation,
        "$location deterministic sampler state",
    )

    initial_metrics = required_property(
        payload,
        :initial_metrics,
        location,
    )
    verify_finite(initial_metrics, "$location initial metrics")
    expected_initial_metrics === nothing || require_semantic_equal(
        initial_metrics,
        expected_initial_metrics,
        "$location initial metrics",
    )

    progress = required_property(payload, :progress, location)
    require_exact_properties(
        progress,
        CHECKPOINT_PROGRESS_FIELDS,
        "$location progress",
    )
    for property in (
        :updates,
        :teacher_states,
        :candidates,
        :window_updates,
        :completed_component_loss_window_updates,
    )
        value = required_property(progress, property, "$location progress")
        value isa Int || error(
            "$location progress.$(String(property)) is not Int",
        )
        value = require_integer_value(
            value,
            "$location progress.$(String(property))",
        )
        value >= 0 || error(
            "$location progress.$(String(property)) is negative",
        )
    end
    hot_allocation_bytes = required_property(
        progress,
        :hot_allocation_bytes,
        "$location progress",
    )
    hot_allocation_bytes isa Int128 || error(
        "$location progress.hot_allocation_bytes is not Int128",
    )
    hot_allocation_bytes >= 0 ||
        error("$location progress.hot_allocation_bytes is negative")
    require_equal(
        Int(required_property(progress, :updates, "$location progress")),
        update,
        "$location progress update",
    )
    require_equal(
        Int(required_property(
            progress,
            :teacher_states,
            "$location progress",
        )),
        update * state_batch,
        "$location progress teacher-state count",
    )
    active_component_window_updates = Int(required_property(
        progress,
        :window_updates,
        "$location progress",
    ))
    completed_component_window_updates = Int(required_property(
        progress,
        :completed_component_loss_window_updates,
        "$location progress",
    ))
    active_component_window_updates <= update || error(
        "$location active component-loss window exceeds total updates",
    )
    completed_component_window_updates <= update || error(
        "$location completed component-loss window exceeds total updates",
    )
    telemetry_schema_version = required_property(
        progress,
        :telemetry_schema_version,
        "$location progress",
    )
    telemetry_schema_version isa Int || error(
        "$location progress.telemetry_schema_version is not Int",
    )
    require_equal(
        telemetry_schema_version,
        3,
        "$location progress telemetry schema version",
    )
    progress_alias_contract = required_property(
        progress,
        :component_loss_alias_contract,
        "$location progress",
    )
    verify_component_loss_alias_contract(
        progress_alias_contract,
        "$location progress component loss alias contract",
    )
    require_exact_state_equal(
        progress_alias_contract,
        payload_alias_contract,
        "$location payload/progress component loss alias contract",
    )
    for property in (
        :hot_wall_seconds,
        :hot_cpu_seconds,
        :hot_gc_seconds,
        :pack_seconds,
        :forward_seconds,
        :loss_seconds,
        :shadow_seconds,
        :backward_seconds,
        :optimizer_seconds,
        :consolidation_seconds,
        :window_composite_loss,
        :window_listnet_ce,
        :window_teacher_entropy,
        :window_listnet_kl,
        :window_composite_excess,
    )
        value = required_property(progress, property, "$location progress")
        value isa Float64 || error(
            "$location progress.$(String(property)) is not Float64",
        )
        require_finite_real(
            value,
            "$location progress.$(String(property))",
        )
    end
    component_losses = required_property(
        progress,
        :component_losses,
        "$location progress",
    )
    verify_component_loss_telemetry(
        component_losses,
        "$location progress component losses";
        last_loss=update > 0 ? last_loss : nothing,
    )
    if active_component_window_updates == 0
        for property in COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS
            require_equal(
                required_property(
                    component_losses,
                    property,
                    "$location progress component losses",
                ),
                0.0,
                "$location zero-count active component window " *
                String(property),
            )
        end
    end
    if completed_component_window_updates == 0
        for property in COMPONENT_LOSS_WINDOW_FIELDS
            require_equal(
                required_property(
                    component_losses,
                    property,
                    "$location progress component losses",
                ),
                0.0,
                "$location zero-count published component window " *
                String(property),
            )
        end
    end

    warmup = required_property(
        payload,
        :persistent_team_warmup,
        location,
    )
    if update == 0
        warmup === nothing || error(
            "$location checkpoint-zero warmup state is not nothing",
        )
    else
        warmup === nothing &&
            error("$location trained checkpoint has no warmup state")
        require_exact_properties(
            warmup,
            CHECKPOINT_WARMUP_FIELDS,
            "$location persistent-team warmup",
        )
        Bool(required_property(
            warmup,
            :isolation_verified,
            "$location persistent-team warmup",
        )) || error("$location warmup isolation is not verified")
        require_equal(
            Int(require_integer_value(
                required_property(
                    warmup,
                    :warmup_optimizer_step,
                    "$location persistent-team warmup",
                ),
                "$location warmup optimizer step",
            )),
            1,
            "$location warmup optimizer step",
        )
        require_finite_real(
            required_property(
                warmup,
                :warmup_loss,
                "$location persistent-team warmup",
            ),
            "$location warmup loss",
        )
        for property in (:queue_length, :remaining, :failure_worker)
            require_equal(
                Int(require_integer_value(
                    required_property(
                        warmup,
                        property,
                        "$location persistent-team warmup",
                    ),
                    "$location warmup $(String(property))",
                )),
                0,
                "$location warmup $(String(property))",
            )
        end
    end

    segment_state = required_property(payload, :segment_state, location)
    require_exact_properties(
        segment_state,
        CHECKPOINT_SEGMENT_FIELDS,
        "$location segment state",
    )
    segment_start = Int(require_integer_value(
        required_property(
            segment_state,
            :start_update,
            "$location segment state",
        ),
        "$location segment start update",
    ))
    segment_updates = Int(require_integer_value(
        required_property(
            segment_state,
            :updates,
            "$location segment state",
        ),
        "$location segment update count",
    ))
    segment_start >= 0 ||
        error("$location segment start update is negative")
    segment_updates >= 0 ||
        error("$location segment update count is negative")
    require_equal(
        segment_start + segment_updates,
        update,
        "$location segment end update",
    )
    segment_seconds = require_finite_real(
        required_property(
            segment_state,
            :overall_seconds,
            "$location segment state",
        ),
        "$location segment overall seconds",
    )
    segment_seconds >= 0.0 ||
        error("$location segment overall seconds is negative")

    dynamics = required_property(
        payload,
        :last_training_dynamics,
        location,
    )
    verify_trace_training_dynamics(
        dynamics,
        "$location last training dynamics",
    )
    require_exact_properties(
        dynamics,
        CHECKPOINT_DYNAMICS_FIELDS,
        "$location last training dynamics",
    )
    require_equal(
        required_property(
            dynamics,
            :schema_version,
            "$location last training dynamics",
        ),
        4,
        "$location dynamics schema version",
    )
    for property in CHECKPOINT_DYNAMICS_BOOL_FIELDS
        required_property(
            dynamics,
            property,
            "$location last training dynamics",
        ) isa Bool || error(
            "$location dynamics.$(String(property)) is not Bool",
        )
    end
    net_mask_flips = required_property(
        dynamics,
        :net_mask_flips,
        "$location last training dynamics",
    )
    net_mask_flips isa Int ||
        error("$location dynamics.net_mask_flips is not Int")
    0 <= net_mask_flips <= length(parameters.gate_logits) || error(
        "$location dynamics.net_mask_flips is outside the mask size",
    )
    Bool(dynamics.consolidation_actual) &&
        !Bool(dynamics.consolidation_scheduled) && error(
            "$location reports actual consolidation when none was scheduled",
        )
    for property in CHECKPOINT_DYNAMICS_FLOAT_FIELDS
        value = required_property(
            dynamics,
            property,
            "$location last training dynamics",
        )
        value isa Float64 || error(
            "$location dynamics.$(String(property)) is not Float64",
        )
        require_finite_real(
            value,
            "$location dynamics.$(String(property))",
        )
    end
    for property in (
        :firing_rate,
        :workspace_route_entropy,
        :workspace_exploitation_entropy,
        :hard_route_load_entropy,
        :hard_route_top8_share,
        :gate_density,
        :utility_nonzero_fraction,
        :hidden_tanh_derivative_mean,
        :hard_mask_unique_fraction,
        :hard_mask_cycle_churn,
        :entropy_floor_violation_fraction,
        :gate_probability_mean,
        :delay_mean,
    )
        value = Float64(required_property(
            dynamics,
            property,
            "$location last training dynamics",
        ))
        0.0 <= value <= 1.0 || error(
            "$location dynamics.$(String(property)) is outside [0, 1]",
        )
    end
    for property in (
        :hard_route_effective_blocks,
        :route_probability_mass_error,
        :route_probability_max_mass_error,
        :workspace_rms,
        :utility_mean,
        :head_pre_rms,
        :hidden_inv_rms_mean,
        :hidden_inv_rms_min,
        :hidden_inv_rms_max,
        :route_score_rms,
        :gate_derivative_mean,
        :delay_derivative_mean,
        :leak_derivative_mean,
        :threshold_derivative_mean,
        :workspace_decay_derivative,
        :membrane_threshold_margin_rms,
        :surrogate_sensitivity_mean,
        :surrogate_sensitivity_rms,
        :eligibility_rms,
    )
        Float64(required_property(
            dynamics,
            property,
            "$location last training dynamics",
        )) >= 0.0 || error(
            "$location dynamics.$(String(property)) is negative",
        )
    end
    Float64(dynamics.workspace_rms) <= 1.0 || error(
        "$location dynamics.workspace_rms exceeds tanh range",
    )
    1.0 <= Float64(dynamics.hard_route_effective_blocks) <=
        Float64(required_property(
            required_property(config, :model, "$location config"),
            :blocks,
            "$location configured model",
        )) || error(
            "$location dynamics hard-route effective block count is invalid",
        )
    Float64(dynamics.hidden_inv_rms_min) <=
        Float64(dynamics.hidden_inv_rms_mean) <=
        Float64(dynamics.hidden_inv_rms_max) || error(
            "$location hidden inverse-RMS ordering is invalid",
        )
    0.0 <= Float64(dynamics.gate_derivative_mean) <= 0.25 || error(
        "$location gate derivative mean is outside sigmoid range",
    )
    0.0 <= Float64(dynamics.delay_derivative_mean) <= 0.25 || error(
        "$location delay derivative mean is outside sigmoid range",
    )
    0.45 <= Float64(dynamics.leak_mean) <= 0.95 || error(
        "$location leak mean is outside transformed range",
    )
    0.0 <= Float64(dynamics.leak_derivative_mean) <= 0.125 || error(
        "$location leak derivative mean is outside transformed range",
    )
    0.25 <= Float64(dynamics.threshold_mean) <= 1.0 || error(
        "$location threshold mean is outside transformed range",
    )
    0.0 <= Float64(dynamics.threshold_derivative_mean) <= 0.1875 || error(
        "$location threshold derivative mean is outside transformed range",
    )
    workspace_retention = required_property(
        config,
        :workspace_retention,
        "$location config",
    )
    workspace_minimum = Float64(required_property(
        workspace_retention,
        :minimum,
        "$location workspace retention config",
    ))
    workspace_maximum = Float64(required_property(
        workspace_retention,
        :maximum,
        "$location workspace retention config",
    ))
    workspace_minimum <= Float64(dynamics.workspace_decay) <=
        workspace_maximum || error(
            "$location workspace decay is outside configured range",
        )
    0.0 <= Float64(dynamics.workspace_decay_derivative) <=
        (workspace_maximum - workspace_minimum) / 4.0 || error(
            "$location workspace decay derivative is outside range",
        )
    Float64(dynamics.membrane_threshold_margin_rms) + 1.0e-12 >=
        abs(Float64(dynamics.membrane_threshold_margin_mean)) || error(
            "$location membrane margin RMS is below absolute mean",
        )
    Float64(dynamics.surrogate_sensitivity_rms) + 1.0e-12 >=
        Float64(dynamics.surrogate_sensitivity_mean) || error(
            "$location surrogate RMS is below its mean",
        )
    transformed_mean(values, transform) = begin
        total = 0.0
        for value in values
            total += Float64(transform(Float32(value)))
        end
        total / length(values)
    end
    float32_transformed_mean(values, transform) = begin
        total = 0.0f0
        for value in values
            total += transform(Float32(value))
        end
        Float64(total / Float32(length(values)))
    end
    probability(value) = sigmoid(value)
    probability_derivative(value) = begin
        p = sigmoid(value)
        p * (1.0f0 - p)
    end
    transformed_expectations = (;
        gate_probability_mean=transformed_mean(
            parameters.gate_logits,
            probability,
        ),
        gate_derivative_mean=transformed_mean(
            parameters.gate_logits,
            probability_derivative,
        ),
        delay_mean=transformed_mean(
            parameters.delay_logits,
            probability,
        ),
        delay_derivative_mean=transformed_mean(
            parameters.delay_logits,
            probability_derivative,
        ),
        leak_mean=transformed_mean(
            parameters.leak_logits,
            value -> 0.45f0 + 0.50f0 * sigmoid(value),
        ),
        leak_derivative_mean=transformed_mean(
            parameters.leak_logits,
            value -> begin
                p = sigmoid(value)
                0.50f0 * p * (1.0f0 - p)
            end,
        ),
        threshold_mean=transformed_mean(
            parameters.threshold_logits,
            value -> 0.25f0 + 0.75f0 * sigmoid(value),
        ),
        threshold_derivative_mean=transformed_mean(
            parameters.threshold_logits,
            value -> begin
                p = sigmoid(value)
                0.75f0 * p * (1.0f0 - p)
            end,
        ),
        workspace_decay=Float64(
            SerialWorkspaceSNN.bounded_workspace_decay(
                parameters.workspace_decay_logit[1],
            ),
        ),
        workspace_decay_derivative=Float64(
            SerialWorkspaceSNN.bounded_workspace_decay_derivative(
                parameters.workspace_decay_logit[1],
            ),
        ),
    )
    for property in propertynames(transformed_expectations)
        require_approx(
            required_property(
                dynamics,
                property,
                "$location last training dynamics",
            ),
            getproperty(transformed_expectations, property),
            "$location transformed diagnostic $(String(property))";
            atol=1.0e-8,
            rtol=1.0e-7,
        )
    end
    require_approx(
        required_property(
            dynamics,
            :gate_density,
            "$location last training dynamics",
        ),
        float32_transformed_mean(parameters.gate_logits, sigmoid),
        "$location dynamics gate density versus parameters";
        atol=1.0e-8,
        rtol=1.0e-7,
    )
    require_approx(
        required_property(
            dynamics,
            :utility_mean,
            "$location last training dynamics",
        ),
        sum(Float64, utility) / length(utility),
        "$location dynamics utility mean versus state";
        atol=1.0e-8,
        rtol=1.0e-7,
    )
    require_approx(
        required_property(
            dynamics,
            :utility_nonzero_fraction,
            "$location last training dynamics",
        ),
        count(>(zero(eltype(utility))), utility) / length(utility),
        "$location dynamics utility support versus state";
        atol=1.0e-8,
        rtol=1.0e-7,
    )

    parent = required_property(payload, :parent_checkpoint, location)
    expected_parent_is_set && require_semantic_equal(
        parent,
        expected_parent,
        "$location parent checkpoint",
    )
    start_mode = replace(
        String(required_property(config, :start_mode, "$location config")),
        "_" => "-",
    )
    if kind == "training"
        required_property(payload, :finalization, location) === nothing ||
            error("$location training checkpoint has finalization state")
        if start_mode == "scratch"
            parent === nothing ||
                error("$location scratch checkpoint has a parent")
            segment_start == 0 ||
                error("$location scratch segment does not start at zero")
        elseif start_mode == "resume"
            parent === nothing &&
                error("$location resume checkpoint has no parent")
            parent_reference = verify_checkpoint_reference_schema(
                parent,
                "$location parent checkpoint";
                expected_kind="training",
            )
            parent_reference.update < update || error(
                "$location resume parent update is not earlier",
            )
            require_equal(
                segment_start,
                parent_reference.update,
                "$location resume segment start",
            )
        else
            error("$location training checkpoint start mode is invalid")
        end
    elseif kind == "finalization"
        parent === nothing &&
            error("$location finalization checkpoint has no parent")
        parent_reference = verify_checkpoint_reference_schema(
            parent,
            "$location parent checkpoint";
            expected_kind="training",
        )
        require_equal(
            parent_reference.update,
            update,
            "$location finalization parent update",
        )
        finalization = required_property(payload, :finalization, location)
        finalization === nothing &&
            error("$location finalization record is missing")
        verify_finalization_record_schema(
            finalization,
            "$location finalization record",
        )
        require_exact_state_equal(
            required_property(
                finalization,
                :component_loss_alias_contract,
                "$location finalization record",
            ),
            payload_alias_contract,
            "$location finalization/payload component loss alias contract",
        )
        require_exact_state_equal(
            required_property(
                finalization,
                :completed_component_loss_window_updates,
                "$location finalization record",
            ),
            completed_component_window_updates,
            "$location finalization/payload completed component window count",
        )
        require_exact_state_equal(
            required_property(
                finalization,
                :training_checkpoint,
                "$location finalization record",
            ),
            parent,
            "$location finalization parent references",
        )
    else
        error("$location checkpoint kind is invalid: $kind")
    end

    if update == 0
        for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
            require_exact_state_equal(
                getproperty(parameters, name),
                getproperty(initial_parameters, name),
                "$location checkpoint-zero $(String(name))",
            )
            all(iszero, getproperty(first_moment, name)) ||
                error(
                    "$location checkpoint-zero first moment " *
                    "$(String(name)) is nonzero",
                )
            all(iszero, getproperty(second_moment, name)) ||
                error(
                    "$location checkpoint-zero second moment " *
                    "$(String(name)) is nonzero",
                )
        end
        all(iszero, utility) ||
            error("$location checkpoint-zero utility is nonzero")
        structural_flips == 0 ||
            error("$location checkpoint-zero structural flips are nonzero")
        for property in filter(
            property -> !(
                property in (
                    :component_losses,
                    :telemetry_schema_version,
                    :component_loss_alias_contract,
                )
            ),
            CHECKPOINT_PROGRESS_FIELDS,
        )
            value = required_property(
                progress,
                property,
                "$location checkpoint-zero progress",
            )
            require_equal(
                value,
                zero(value),
                "$location checkpoint-zero progress $(String(property))",
            )
        end
        for property in CHECKPOINT_COMPONENT_LOSS_FIELDS
            require_equal(
                required_property(
                    component_losses,
                    property,
                    "$location checkpoint-zero component losses",
                ),
                0.0,
                "$location checkpoint-zero component loss " *
                String(property),
            )
        end
    end
    return (;
        kind,
        update,
        config,
        parameters,
        initial_parameters,
        optimizer,
        trainer_state,
        synapse_utility=utility,
        utility_updates,
        total_structural_flips=structural_flips,
        sampler_state,
        initial_metrics,
        progress,
        persistent_team_warmup=warmup,
        segment_state,
        last_training_dynamics=dynamics,
        hard_gate_budget=gate_budget,
    )
end

function verify_checkpoint_state_equivalence(
    training_payload,
    finalization_payload,
    location::AbstractString,
)
    for property in (:format, :version, :update)
        require_exact_state_equal(
            required_property(
                training_payload,
                property,
                "$location training payload",
            ),
            required_property(
                finalization_payload,
                property,
                "$location finalization payload",
            ),
            "$location $(String(property))",
        )
    end
    for property in CHECKPOINT_STATE_FIELDS
        require_exact_state_equal(
            required_property(
                training_payload,
                property,
                "$location training payload",
            ),
            required_property(
                finalization_payload,
                property,
                "$location finalization payload",
            ),
            "$location $(String(property))",
        )
    end
    return true
end

function verify_checkpoint(
    artifact,
    results,
    expected_updates,
    state_batch,
    manifest,
    canonical_config,
    expected_parameters,
    training_rows,
    expected_sampler_snapshot,
)
    update = artifact.update
    path = artifact.path
    bytes = filesize(path)
    digest = file_sha256(path)
    if manifest !== nothing
        haskey(manifest, update) ||
            error("checkpoint manifest is missing update $update")
        require_equal(
            manifest[update].kind,
            "training",
            "checkpoint $update manifest kind",
        )
        require_equal(
            manifest[update].sha256,
            digest,
            "checkpoint $update manifest SHA-256",
        )
        require_equal(
            manifest[update].bytes,
            bytes,
            "checkpoint $update manifest byte size",
        )
        require_equal(
            normalized_path(manifest[update].path),
            normalized_path(path),
            "checkpoint $update manifest path",
        )
    end

    file = JLD2.load(path)
    haskey(file, "payload") ||
        error("checkpoint $update has no payload")
    payload = file["payload"]
    core = verify_checkpoint_payload_core(
        payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="checkpoint $update payload",
        expected_update=update,
        expected_kind="training",
        expected_config=canonical_config,
        expected_sampler_snapshot,
        expected_initial_metrics=required_property(
            results,
            :initial,
            "results",
        ),
        expected_parent=required_property(
            results,
            :parent_checkpoint,
            "results",
        ),
        expected_parent_is_set=true,
    )
    require_equal(
        String(required_property(
            payload,
            :format,
            "checkpoint $update payload",
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "checkpoint $update payload format",
    )
    require_equal(
        Int(required_property(
            payload,
            :version,
            "checkpoint $update payload",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "checkpoint $update payload version",
    )
    require_equal(
        String(required_property(
            payload,
            :checkpoint_kind,
            "checkpoint $update payload",
        )),
        "training",
        "checkpoint $update kind",
    )
    require_semantic_equal(
        required_property(
            payload,
            :parent_checkpoint,
            "checkpoint $update payload",
        ),
        required_property(results, :parent_checkpoint, "results"),
        "checkpoint $update parent checkpoint",
    )
    require_equal(
        Int(required_property(payload, :update, "checkpoint $update payload")),
        update,
        "checkpoint $update payload update",
    )
    optimizer = required_property(
        payload,
        :optimizer,
        "checkpoint $update payload",
    )
    require_equal(
        Int(required_property(
            optimizer,
            :step,
            "checkpoint $update optimizer",
        )),
        update,
        "checkpoint $update optimizer step",
    )
    for property in (
        :learning_rate,
        :beta1,
        :beta2,
        :beta1_power,
        :beta2_power,
        :epsilon,
        :weight_decay,
    )
        value = Float64(required_property(
            optimizer,
            property,
            "checkpoint $update optimizer",
        ))
        isfinite(value) ||
            error("checkpoint $update optimizer.$property is not finite")
    end
    progress = required_property(
        payload,
        :progress,
        "checkpoint $update payload",
    )
    require_equal(
        Int(required_property(
            progress,
            :updates,
            "checkpoint $update progress",
        )),
        update,
        "checkpoint $update progress update",
    )
    require_equal(
        Int(required_property(
            progress,
            :teacher_states,
            "checkpoint $update progress",
        )),
        update * state_batch,
        "checkpoint $update teacher-state count",
    )
    config = required_property(payload, :config, "checkpoint $update payload")
    require_semantic_equal(
        config,
        canonical_config,
        "checkpoint $update config versus config.json",
    )
    checkpoint_contract = required_property(
        config,
        :production_contract,
        "checkpoint $update config",
    )
    checkpoint_contract_digest =
        production_contract_sha256(checkpoint_contract)
    embedded_contract_digest = require_sha256(
        required_property(
            config,
            :production_contract_sha256,
            "checkpoint $update config",
        ),
        "checkpoint $update production contract SHA-256",
    )
    require_equal(
        checkpoint_contract_digest,
        embedded_contract_digest,
        "checkpoint $update production contract SHA-256",
    )
    require_equal(
        embedded_contract_digest,
        require_sha256(
            required_property(
                canonical_config,
                :production_contract_sha256,
                "run config",
            ),
            "run production contract SHA-256",
        ),
        "checkpoint $update/run production contract SHA-256",
    )
    require_equal(
        UInt64(required_property(
            config,
            :model_seed,
            "checkpoint $update config",
        )),
        UInt64(MODEL_SEED),
        "checkpoint $update model seed",
    )
    require_equal(
        UInt64(required_property(
            config,
            :sampler_seed,
            "checkpoint $update config",
        )),
        UInt64(SAMPLER_SEED),
        "checkpoint $update sampler seed",
    )
    require_equal(
        UInt64(required_property(
            config,
            :routing_seed,
            "checkpoint $update config",
        )),
        UInt64(ROUTING_SEED),
        "checkpoint $update routing seed",
    )
    checkpoint_eprop =
        required_property(config, :eprop, "checkpoint $update config")
    require_equal(
        UInt64(required_property(
            checkpoint_eprop,
            :feedback_seed,
            "checkpoint $update e-prop config",
        )),
        UInt64(0x4550524f50534844),
        "checkpoint $update e-prop feedback seed",
    )
    checkpoint_routing =
        required_property(config, :routing, "checkpoint $update config")
    require_equal(
        UInt64(required_property(
            checkpoint_routing,
            :routing_seed,
            "checkpoint $update routing config",
        )),
        UInt64(ROUTING_SEED),
        "checkpoint $update nested routing seed",
    )
    require_equal(
        Int(required_property(
            config,
            :maximum_updates,
            "checkpoint $update config",
        )),
        expected_updates,
        "checkpoint $update configured maximum updates",
    )
    require_equal(
        String(required_property(config, :run_id, "checkpoint $update config")),
        String(required_property(results.config, :run_id, "results config")),
        "checkpoint $update run ID",
    )
    require_equal(
        String(required_property(
            config,
            :source_fingerprint,
            "checkpoint $update config",
        )),
        String(required_property(
            results.config,
            :source_fingerprint,
            "results config",
        )),
        "checkpoint $update source fingerprint",
    )
    require_equal(
        String(required_property(
            config,
            :training_rows_sha256,
            "checkpoint $update config",
        )),
        String(required_property(
            results.config,
            :training_rows_sha256,
            "results config",
        )),
        "checkpoint $update training-row SHA-256",
    )

    for property in (
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
    )
        require_semantic_equal(
            required_property(
                payload,
                property,
                "checkpoint $update payload",
            ),
            required_property(
                canonical_config,
                property,
                "run config",
            ),
            "checkpoint $update $(String(property))",
        )
    end
    require_semantic_equal(
        required_property(
            payload,
            :initial_metrics,
            "checkpoint $update payload",
        ),
        required_property(results, :initial, "results"),
        "checkpoint $update initial metrics",
    )

    parameters = required_property(
        payload,
        :parameters,
        "checkpoint $update payload",
    )
    verify_array_registry(
        parameters,
        expected_parameters,
        "checkpoint $update parameters",
    )
    initial_parameters = required_property(
        payload,
        :initial_parameters,
        "checkpoint $update payload",
    )
    verify_array_registry(
        initial_parameters,
        expected_parameters,
        "checkpoint $update initial parameters",
    )
    first_moment = required_property(
        optimizer,
        :first_moment,
        "checkpoint $update optimizer",
    )
    second_moment = required_property(
        optimizer,
        :second_moment,
        "checkpoint $update optimizer",
    )
    verify_array_registry(
        first_moment,
        expected_parameters,
        "checkpoint $update first moment",
    )
    verify_array_registry(
        second_moment,
        expected_parameters,
        "checkpoint $update second moment",
    )
    utility = required_property(
        payload,
        :synapse_utility,
        "checkpoint $update payload",
    )
    size(utility) == size(parameters.synapse_weight) ||
        error("checkpoint $update utility shape differs")
    all(isfinite, utility) ||
        error("checkpoint $update utility contains a non-finite value")
    all(>=(0.0f0), utility) ||
        error("checkpoint $update utility contains a negative value")
    require_equal(
        Int(required_property(
            payload,
            :utility_updates,
            "checkpoint $update payload",
        )),
        update,
        "checkpoint $update utility update count",
    )
    structural_flips = Int(required_property(
        payload,
        :total_structural_flips,
        "checkpoint $update payload",
    ))
    structural_flips >= 0 ||
        error("checkpoint $update structural flip count is negative")
    gate_budget = verify_hard_gate_budget(
        parameters,
        canonical_config.model,
        "checkpoint $update",
    )
    sampler_state = required_property(
        payload,
        :sampler_state,
        "checkpoint $update payload",
    )
    restored_sampler = restore_sampler(training_rows, sampler_state)
    length(restored_sampler.source_rows) == length(training_rows) ||
        error("checkpoint $update sampler row count differs")
    require_equal(
        sampler_consumed_states(restored_sampler),
        update * state_batch,
        "checkpoint $update sampler consumed-state count",
    )
    require_semantic_equal(
        sampler_state,
        expected_sampler_snapshot,
        "checkpoint $update deterministic sampler state",
    )
    verify_finite(progress, "checkpoint $update progress")

    if update == 0
        for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
            isequal(
                getproperty(parameters, name),
                getproperty(initial_parameters, name),
            ) || error(
                "checkpoint zero $(String(name)) differs from initial parameters",
            )
            all(iszero, getproperty(first_moment, name)) || error(
                "checkpoint zero first moment $(String(name)) is nonzero",
            )
            all(iszero, getproperty(second_moment, name)) || error(
                "checkpoint zero second moment $(String(name)) is nonzero",
            )
        end
        all(iszero, utility) ||
            error("checkpoint zero utility is nonzero")
        structural_flips == 0 ||
            error("checkpoint zero structural flips are nonzero")
        require_approx(
            required_property(
                optimizer,
                :beta1_power,
                "checkpoint zero optimizer",
            ),
            required_property(
                optimizer,
                :beta1,
                "checkpoint zero optimizer",
            ),
            "checkpoint zero beta1 clock",
        )
        require_approx(
            required_property(
                optimizer,
                :beta2_power,
                "checkpoint zero optimizer",
            ),
            required_property(
                optimizer,
                :beta2,
                "checkpoint zero optimizer",
            ),
            "checkpoint zero beta2 clock",
        )
        for property in (
            :updates,
            :teacher_states,
            :candidates,
            :hot_wall_seconds,
            :hot_cpu_seconds,
            :hot_allocation_bytes,
            :hot_gc_seconds,
        )
            require_equal(
                required_property(
                    progress,
                    property,
                    "checkpoint zero progress",
                ),
                zero(required_property(
                    progress,
                    property,
                    "checkpoint zero progress",
                )),
                "checkpoint zero progress $(String(property))",
            )
        end
    end
    initial_parameters_sha256 =
        array_registry_sha256(initial_parameters)
    expected_initial_sha256 =
        array_registry_sha256(expected_parameters)
    require_equal(
        initial_parameters_sha256,
        expected_initial_sha256,
        "checkpoint $update initial parameters versus deterministic model seed",
    )

    return (;
        update,
        path,
        bytes,
        sha256=digest,
        payload_format=REQUIRED_CHECKPOINT_FORMAT,
        payload_version=REQUIRED_CHECKPOINT_VERSION,
        checkpoint_kind=core.kind,
        structural_flips,
        utility_updates=update,
        hard_gate_budget=core.hard_gate_budget,
        initial_parameters_sha256,
        payload,
    )
end

function expected_checkpoint_updates(results, expected_updates)
    interval = Int(required_property(
        results.config,
        :checkpoint_interval,
        "results config",
    ))
    interval >= 1 || error("checkpoint interval must be positive")
    segment_updates = hasproperty(results.throughput, :segment_updates) ?
        Int(results.throughput.segment_updates) : expected_updates
    1 <= segment_updates <= expected_updates ||
        error("results segment update count is out of range")
    segment_start = expected_updates - segment_updates
    updates = Int[]
    for update in (segment_start + 1):expected_updates
        update % interval == 0 && push!(updates, update)
    end
    expected_updates in updates || push!(updates, expected_updates)
    sort!(unique!(updates))
    return (; interval, segment_start, updates)
end

function expected_parent_checkpoint_updates(
    parent_config,
    parent_payload,
    expected_update,
)
    interval = Int(required_property(
        parent_config,
        :checkpoint_interval,
        "parent config",
    ))
    interval >= 1 ||
        error("parent checkpoint interval must be positive")
    start_mode = replace(
        String(required_property(
            parent_config,
            :start_mode,
            "parent config",
        )),
        "_" => "-",
    )
    segment_start = if start_mode == "scratch"
        required_property(
            parent_payload,
            :parent_checkpoint,
            "parent payload",
        ) === nothing || error(
            "scratch parent payload unexpectedly has a parent checkpoint",
        )
        0
    elseif start_mode == "resume"
        parent_reference = required_property(
            parent_payload,
            :parent_checkpoint,
            "parent payload",
        )
        parent_reference === nothing &&
            error("resume parent payload has no parent checkpoint")
        Int(required_property(
            parent_reference,
            :update,
            "parent payload parent checkpoint",
        ))
    else
        error(
            "a finalize-only recovery parent must come from a scratch " *
            "or resume training segment",
        )
    end
    0 <= segment_start < expected_update || error(
        "parent segment start is outside the target interval",
    )
    updates = Int[]
    start_mode == "scratch" && push!(updates, 0)
    for update in (segment_start + 1):expected_update
        update % interval == 0 && push!(updates, update)
    end
    expected_update in updates || push!(updates, expected_update)
    sort!(unique!(updates))
    return (; interval, segment_start, updates, start_mode)
end

function verify_residual_finalization_checkpoint(
    path,
    training_checkpoint,
    parent_config,
    expected_parameters,
    training_rows,
    state_batch,
    expected_update,
)
    isfile(path) ||
        error("residual finalization checkpoint does not exist: $path")
    file = JLD2.load(path)
    haskey(file, "payload") ||
        error("residual finalization checkpoint has no payload")
    payload = file["payload"]
    training_reference = (;
        kind="training",
        path=training_checkpoint.path,
        bytes=training_checkpoint.bytes,
        sha256=training_checkpoint.sha256,
        update=training_checkpoint.update,
    )
    verify_checkpoint_payload_core(
        payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="residual finalization payload",
        expected_update,
        expected_kind="finalization",
        expected_config=parent_config,
        expected_sampler_snapshot=
            training_checkpoint.payload.sampler_state,
        expected_initial_metrics=
            training_checkpoint.payload.initial_metrics,
        expected_parent=training_reference,
        expected_parent_is_set=true,
    )
    verify_checkpoint_state_equivalence(
        training_checkpoint.payload,
        payload,
        "residual finalization versus training checkpoint",
    )
    require_equal(
        String(required_property(
            payload,
            :format,
            "residual finalization payload",
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "residual finalization format",
    )
    require_equal(
        Int(required_property(
            payload,
            :version,
            "residual finalization payload",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "residual finalization version",
    )
    require_equal(
        String(required_property(
            payload,
            :checkpoint_kind,
            "residual finalization payload",
        )),
        "finalization",
        "residual finalization kind",
    )
    require_equal(
        Int(required_property(
            payload,
            :update,
            "residual finalization payload",
        )),
        expected_update,
        "residual finalization update",
    )
    require_semantic_equal(
        required_property(
            payload,
            :parent_checkpoint,
            "residual finalization payload",
        ),
        training_reference,
        "residual finalization parent checkpoint",
    )
    require_semantic_equal(
        required_property(
            payload,
            :config,
            "residual finalization payload",
        ),
        parent_config,
        "residual finalization config",
    )
    for property in (
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
    )
        require_semantic_equal(
            required_property(
                payload,
                property,
                "residual finalization payload",
            ),
            required_property(
                parent_config,
                property,
                "parent config",
            ),
            "residual finalization $(String(property))",
        )
    end
    training_payload = training_checkpoint.payload
    parameters = required_property(
        payload,
        :parameters,
        "residual finalization payload",
    )
    initial_parameters = required_property(
        payload,
        :initial_parameters,
        "residual finalization payload",
    )
    optimizer = required_property(
        payload,
        :optimizer,
        "residual finalization payload",
    )
    verify_array_registry(
        parameters,
        expected_parameters,
        "residual finalization parameters",
    )
    verify_array_registry(
        initial_parameters,
        expected_parameters,
        "residual finalization initial parameters",
    )
    require_equal(
        Int(required_property(
            optimizer,
            :step,
            "residual finalization optimizer",
        )),
        expected_update,
        "residual finalization optimizer step",
    )
    for property in (
        :learning_rate,
        :beta1,
        :beta2,
        :beta1_power,
        :beta2_power,
        :epsilon,
        :weight_decay,
    )
        value = Float64(required_property(
            optimizer,
            property,
            "residual finalization optimizer",
        ))
        isfinite(value) || error(
            "residual finalization optimizer.$property is not finite",
        )
    end
    verify_array_registry(
        required_property(
            optimizer,
            :first_moment,
            "residual finalization optimizer",
        ),
        expected_parameters,
        "residual finalization first moment",
    )
    verify_array_registry(
        required_property(
            optimizer,
            :second_moment,
            "residual finalization optimizer",
        ),
        expected_parameters,
        "residual finalization second moment",
    )
    for (location, left, right) in (
        (
            "parameters",
            array_registry_sha256(parameters),
            array_registry_sha256(training_payload.parameters),
        ),
        (
            "initial parameters",
            array_registry_sha256(initial_parameters),
            array_registry_sha256(training_payload.initial_parameters),
        ),
        (
            "first moment",
            array_registry_sha256(optimizer.first_moment),
            array_registry_sha256(
                training_payload.optimizer.first_moment,
            ),
        ),
        (
            "second moment",
            array_registry_sha256(optimizer.second_moment),
            array_registry_sha256(
                training_payload.optimizer.second_moment,
            ),
        ),
    )
        require_equal(
            left,
            right,
            "residual finalization $location state",
        )
    end
    utility = required_property(
        payload,
        :synapse_utility,
        "residual finalization payload",
    )
    require_equal(
        bytes2hex(sha256(reinterpret(UInt8, vec(utility)))),
        bytes2hex(sha256(reinterpret(
            UInt8,
            vec(training_payload.synapse_utility),
        ))),
        "residual finalization utility state",
    )
    require_equal(
        Int(required_property(
            payload,
            :utility_updates,
            "residual finalization payload",
        )),
        expected_update,
        "residual finalization utility update count",
    )
    require_equal(
        Int(required_property(
            payload,
            :total_structural_flips,
            "residual finalization payload",
        )),
        Int(training_payload.total_structural_flips),
        "residual finalization structural flip state",
    )
    for property in (:sampler_state, :progress, :initial_metrics)
        require_semantic_equal(
            required_property(
                payload,
                property,
                "residual finalization payload",
            ),
            required_property(
                training_payload,
                property,
                "parent training payload",
            ),
            "residual finalization $(String(property)) state",
        )
    end
    record = required_property(
        payload,
        :finalization,
        "residual finalization payload",
    )
    require_equal(
        String(required_property(
            record,
            :status,
            "residual finalization record",
        )),
        "finalization_checkpoint_complete",
        "residual finalization status",
    )
    require_equal(
        Int(required_property(
            record,
            :optimizer_steps_after_target,
            "residual finalization record",
        )),
        0,
        "residual finalization optimizer steps after target",
    )
    require_semantic_equal(
        required_property(
            record,
            :training_checkpoint,
            "residual finalization record",
        ),
        training_reference,
        "residual finalization record training checkpoint",
    )
    require_exact_state_equal(
        required_property(
            record,
            :component_loss_telemetry,
            "residual finalization record",
        ),
        required_property(
            required_property(
                payload,
                :progress,
                "residual finalization payload",
            ),
            :component_losses,
            "residual finalization progress",
        ),
        "residual finalization component loss telemetry",
    )
    require_exact_state_equal(
        required_property(
            record,
            :completed_component_loss_window_updates,
            "residual finalization record",
        ),
        required_property(
            required_property(
                payload,
                :progress,
                "residual finalization payload",
            ),
            :completed_component_loss_window_updates,
            "residual finalization progress",
        ),
        "residual finalization completed component window count",
    )
    require_exact_state_equal(
        required_property(
            record,
            :component_loss_alias_contract,
            "residual finalization record",
        ),
        required_property(
            payload,
            :component_loss_alias_contract,
            "residual finalization payload",
        ),
        "residual finalization component loss alias contract",
    )
    verify_finite(
        required_property(
            record,
            :final_metrics,
            "residual finalization record",
        ),
        "residual finalization metrics",
    )
    parent_run_dir = dirname(dirname(training_checkpoint.path))
    require_equal(
        normalized_path(required_property(
            record,
            :expected_results_path,
            "residual finalization record",
        )),
        normalized_path(joinpath(parent_run_dir, "results.json")),
        "residual finalization expected results path",
    )
    require_equal(
        normalized_path(required_property(
            record,
            :expected_manifest_path,
            "residual finalization record",
        )),
        normalized_path(joinpath(
            parent_run_dir,
            "finalization_manifest.json",
        )),
        "residual finalization expected manifest path",
    )
    return (;
        present=true,
        verified=true,
        path=abspath(path),
        bytes=filesize(path),
        sha256=file_sha256(path),
        update=expected_update,
        team_teardown_reference=required_property(
            record,
            :team_teardown,
            "residual finalization record",
        ),
        final_metrics=required_property(
            record,
            :final_metrics,
            "residual finalization record",
        ),
    )
end

function verify_metric_snapshot(
    reported,
    recomputed,
    location::AbstractString,
)
    reported_keys = sort!(String.(collect(keys(reported))))
    recomputed_keys = sort!(String.(collect(keys(recomputed))))
    reported_keys == recomputed_keys || error(
        "$location metric fields differ: observed=$reported_keys " *
        "expected=$recomputed_keys",
    )
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for key_string in recomputed_keys
        key = Symbol(key_string)
        observed = required_property(reported, key, location)
        expected = required_property(recomputed, key, "recomputed metrics")
        if expected isa Number
            observed_number = Float64(observed)
            expected_number = Float64(expected)
            isfinite(observed_number) ||
                error("$location.$key_string is not finite")
            isfinite(expected_number) ||
                error("recomputed $location.$key_string is not finite")
            absolute_error = abs(observed_number - expected_number)
            relative_error = absolute_error /
                max(abs(expected_number), 1.0e-12)
            maximum_absolute_error =
                max(maximum_absolute_error, absolute_error)
            maximum_relative_error =
                max(maximum_relative_error, relative_error)
            isapprox(
                observed_number,
                expected_number;
                atol=1.0e-8,
                rtol=1.0e-7,
            ) || error(
                "$location.$key_string differs from fixed-panel " *
                "recomputation: observed=$observed_number " *
                "expected=$expected_number",
            )
        else
            require_semantic_equal(
                observed,
                expected,
                "$location.$key_string",
            )
        end
    end
    return (; maximum_absolute_error, maximum_relative_error)
end

function stable_sigmoid(value)
    x = Float64(value)
    return x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
end

function verify_metrics_and_witness(
    results,
    final_payload,
    model,
    states,
    bound_inputs,
    trace,
    expected_updates,
    state_batch,
)
    eval_batch = allocate_host_batch(1; max_candidates=80)
    recomputed_initial = evaluate(
        model,
        final_payload.initial_parameters,
        states,
        bound_inputs.dataset,
        bound_inputs.panel_rows,
        eval_batch,
    )
    recomputed_final = evaluate(
        model,
        final_payload.parameters,
        states,
        bound_inputs.dataset,
        bound_inputs.panel_rows,
        eval_batch,
    )
    initial_error = verify_metric_snapshot(
        results.initial,
        recomputed_initial,
        "results.initial",
    )
    final_error = verify_metric_snapshot(
        results.final,
        recomputed_final,
        "results.final",
    )
    require_semantic_equal(
        final_payload.initial_metrics,
        results.initial,
        "final checkpoint initial metrics versus results",
    )

    initial_objective = reducible_objective_metrics(recomputed_initial)
    final_objective = reducible_objective_metrics(recomputed_final)
    reducible = required_property(
        results,
        :reducible_objective,
        "results",
    )
    require_semantic_equal(
        required_property(reducible, :initial, "results reducible objective"),
        initial_objective,
        "results initial reducible objective",
    )
    require_semantic_equal(
        required_property(reducible, :final, "results reducible objective"),
        final_objective,
        "results final reducible objective",
    )

    expected_deltas = (;
        composite_loss=
            recomputed_final.composite_loss -
            recomputed_initial.composite_loss,
        listnet_ce=
            final_objective.listnet_ce -
            initial_objective.listnet_ce,
        teacher_entropy=
            final_objective.teacher_entropy -
            initial_objective.teacher_entropy,
        listnet_kl=
            final_objective.listnet_kl -
            initial_objective.listnet_kl,
        composite_excess=
            final_objective.composite_excess -
            initial_objective.composite_excess,
        top1_agreement=
            recomputed_final.top1_agreement -
            recomputed_initial.top1_agreement,
        ndcg=recomputed_final.ndcg - recomputed_initial.ndcg,
        pairwise_accuracy=
            recomputed_final.pairwise_accuracy -
            recomputed_initial.pairwise_accuracy,
    )
    require_semantic_equal(
        results.deltas,
        expected_deltas,
        "results deltas versus fixed-panel recomputation",
    )

    witness = results.learning_witness
    require_equal(
        Int(required_property(witness, :final_update, "learning witness")),
        expected_updates,
        "learning witness final update",
    )
    require_equal(
        Int(required_property(
            witness,
            :consumed_teacher_states,
            "learning witness",
        )),
        expected_updates * state_batch,
        "learning witness consumed teacher states",
    )
    last_trace = last(trace.parsed_records)
    final_component_losses = required_property(
        required_property(
            final_payload,
            :progress,
            "final checkpoint",
        ),
        :component_losses,
        "final checkpoint progress",
    )
    require_equal(
        last_trace.window_updates,
        Int(required_property(
            required_property(
                final_payload,
                :progress,
                "final checkpoint",
            ),
            :completed_component_loss_window_updates,
            "final checkpoint progress",
        )),
        "final trace/checkpoint completed component window count",
    )
    for property in (
        COMPONENT_LOSS_NAMES...,
        COMPONENT_LOSS_WINDOW_FIELDS...,
    )
        require_equal(
            required_property(
                last_trace.component_loss_telemetry,
                property,
                "final training trace component losses",
            ),
            required_property(
                final_component_losses,
                property,
                "final checkpoint component losses",
            ),
            "final trace/checkpoint component loss $(String(property))",
        )
    end
    for property in COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS
        require_equal(
            required_property(
                final_component_losses,
                property,
                "final checkpoint component losses",
            ),
            0.0,
            "final checkpoint active component window $(String(property))",
        )
    end
    require_exact_state_equal(
        last_trace.training_dynamics,
        required_property(
            final_payload,
            :last_training_dynamics,
            "final checkpoint",
        ),
        "final trace/checkpoint training dynamics",
    )
    require_approx(
        required_property(witness, :last_batch_loss, "learning witness"),
        last_trace.loss,
        "learning witness last loss";
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    require_approx(
        required_property(
            witness,
            :last_gradient_norm,
            "learning witness",
        ),
        last_trace.gradient_norm,
        "learning witness last gradient norm";
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    weight_delta = sqrt(sum(
        abs2,
        Float64.(
            final_payload.parameters.synapse_weight .-
            final_payload.initial_parameters.synapse_weight
        ),
    ))
    delay_delta = sqrt(sum(
        (
            stable_sigmoid(current) - stable_sigmoid(initial)
        )^2
        for (current, initial) in zip(
            final_payload.parameters.delay_logits,
            final_payload.initial_parameters.delay_logits,
        )
    ))
    mask_flips = count(
        (final_payload.parameters.gate_logits .>= 0.0f0) .!=
        (final_payload.initial_parameters.gate_logits .>= 0.0f0),
    )
    require_approx(
        required_property(
            witness,
            :synapse_weight_l2_delta,
            "learning witness",
        ),
        weight_delta,
        "learning witness synapse-weight delta";
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    require_approx(
        required_property(
            witness,
            :continuous_delay_l2_delta,
            "learning witness",
        ),
        delay_delta,
        "learning witness continuous-delay delta";
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    require_equal(
        Int(required_property(
            witness,
            :structural_consolidation_flips,
            "learning witness",
        )),
        Int(final_payload.total_structural_flips),
        "learning witness structural flips",
    )
    last_trace.structural_flips_total === nothing && error(
        "trace final structural flip count is missing",
    )
    require_equal(
        last_trace.structural_flips_total,
        Int(final_payload.total_structural_flips),
        "trace final structural flips",
    )
    require_equal(
        Int(required_property(
            witness,
            :final_mask_flips_from_initial,
            "learning witness",
        )),
        mask_flips,
        "learning witness mask flips",
    )
    return (;
        verified=true,
        panel_states=length(bound_inputs.panel_rows),
        panel_rows_sha256=bytes2hex(sha256(reinterpret(
            UInt8,
            bound_inputs.panel_rows,
        ))),
        initial_error,
        final_error,
        recomputed_initial,
        recomputed_final,
    )
end

function verify_ancestry_config_invariants(
    ancestor_config,
    child_config,
    location::AbstractString,
)
    require_exact_properties(
        ancestor_config,
        propertynames(child_config),
        "$location config schema",
    )
    allowed_to_differ = Set((
        :run_id,
        :start_mode,
        :scratch,
        :production_target_match,
        :launch_binding,
    ))
    for property in propertynames(child_config)
        property in allowed_to_differ && continue
        require_semantic_equal(
            required_property(
                ancestor_config,
                property,
                "$location ancestor config",
            ),
            required_property(
                child_config,
                property,
                "$location child config",
            ),
            "$location ancestor/child $(String(property))",
        )
    end
    for property in ANCESTRY_CONFIG_INVARIANT_FIELDS
        required_property(
            ancestor_config,
            property,
            "$location ancestor config",
        )
        required_property(
            child_config,
            property,
            "$location child config",
        )
    end
    start_mode = replace(
        String(required_property(
            ancestor_config,
            :start_mode,
            "$location ancestor config",
        )),
        "_" => "-",
    )
    start_mode in ("scratch", "resume") || error(
        "$location ancestor start mode is not scratch or resume",
    )
    require_equal(
        Bool(required_property(
            ancestor_config,
            :scratch,
            "$location ancestor config",
        )),
        start_mode == "scratch",
        "$location ancestor scratch flag",
    )
    run_id = String(required_property(
        ancestor_config,
        :run_id,
        "$location ancestor config",
    ))
    isempty(strip(run_id)) && error("$location ancestor run ID is blank")
    contract = required_property(
        ancestor_config,
        :production_contract,
        "$location ancestor config",
    )
    require_equal(
        production_contract_sha256(contract),
        require_sha256(required_property(
            ancestor_config,
            :production_contract_sha256,
            "$location ancestor config",
        ), "$location ancestor production contract SHA-256"),
        "$location ancestor production contract digest",
    )
    production_target = required_property(
        ancestor_config,
        :production_target,
        "$location ancestor config",
    )
    expected_target_match =
        String(required_property(
            ancestor_config,
            :model_preset,
            "$location ancestor config",
        )) == String(required_property(
            production_target,
            :model_preset,
            "$location production target",
        )) &&
        start_mode == replace(
            String(required_property(
                production_target,
                :start_mode,
                "$location production target",
            )),
            "_" => "-",
        ) &&
        String(required_property(
            ancestor_config,
            :learning_mode,
            "$location ancestor config",
        )) == String(required_property(
            production_target,
            :learning_mode,
            "$location production target",
        )) &&
        Int(required_property(
            ancestor_config,
            :maximum_updates,
            "$location ancestor config",
        )) == Int(required_property(
            production_target,
            :maximum_updates,
            "$location production target",
        ))
    require_equal(
        Bool(required_property(
            ancestor_config,
            :production_target_match,
            "$location ancestor config",
        )),
        expected_target_match,
        "$location ancestor production-target match",
    )
    return start_mode
end

function verify_checkpoint_lineage_segment(
    reference,
    child_config,
    expected_parameters,
    training_rows,
    state_batch,
    expected_initial_metrics,
    visited_paths::Set{String},
    visited_digests::Set{String};
    allow_residual_finalization::Bool=false,
    depth::Int=1,
)
    location = "resume ancestry depth $depth"
    normalized_reference = verify_checkpoint_reference_schema(
        reference,
        "$location checkpoint reference";
        expected_kind="training",
    )
    canonical_key = normalized_path(normalized_reference.path)
    canonical_key in visited_paths &&
        error("$location contains a checkpoint-path cycle")
    normalized_reference.sha256 in visited_digests &&
        error("$location contains a checkpoint-digest cycle")
    push!(visited_paths, canonical_key)
    push!(visited_digests, normalized_reference.sha256)

    path = normalized_reference.path
    isfile(path) || error("$location checkpoint does not exist: $path")
    islink(path) &&
        error("$location checkpoint is a symbolic link: $path")
    require_equal(
        normalized_path(realpath(path)),
        canonical_key,
        "$location canonical checkpoint path",
    )
    require_equal(
        filesize(path),
        normalized_reference.bytes,
        "$location checkpoint byte size",
    )
    require_equal(
        file_sha256(path),
        normalized_reference.sha256,
        "$location checkpoint SHA-256",
    )
    file = JLD2.load(path)
    haskey(file, "payload") ||
        error("$location checkpoint has no payload")
    payload = file["payload"]
    require_equal(
        Int(required_property(payload, :update, "$location payload")),
        normalized_reference.update,
        "$location payload/reference update",
    )
    segment_config = required_property(
        payload,
        :config,
        "$location payload",
    )
    start_mode = verify_ancestry_config_invariants(
        segment_config,
        child_config,
        location,
    )

    checkpoint_dir = dirname(path)
    run_dir = dirname(checkpoint_dir)
    require_equal(
        lowercase(basename(checkpoint_dir)),
        "checkpoints",
        "$location checkpoint-directory name",
    )
    require_equal(
        lowercase(basename(run_dir)),
        lowercase(String(required_property(
            segment_config,
            :run_id,
            "$location config",
        ))),
        "$location run-directory name",
    )
    artifacts = checkpoint_files(
        checkpoint_dir,
        normalized_reference.update;
        require_training=true,
        allow_finalization=allow_residual_finalization,
    )
    manifest_path = joinpath(run_dir, "checkpoint_manifest.jsonl")
    manifest = checkpoint_manifest(manifest_path)
    verify_checkpoint_manifest_set(
        artifacts,
        manifest,
        normalized_reference.update,
        "$location checkpoint manifest",
    )
    policy = expected_parent_checkpoint_updates(
        segment_config,
        payload,
        normalized_reference.update,
    )
    artifact_updates = [artifact.update for artifact in artifacts]
    require_equal(
        artifact_updates,
        policy.updates,
        "$location checkpoint cadence",
    )
    target_matches = filter(
        artifact -> artifact.update == normalized_reference.update,
        artifacts,
    )
    length(target_matches) == 1 || error(
        "$location checkpoint set does not contain exactly one target",
    )
    require_equal(
        normalized_path(only(target_matches).path),
        canonical_key,
        "$location target checkpoint path",
    )

    segment_parent = required_property(
        payload,
        :parent_checkpoint,
        "$location target payload",
    )
    segment_launch = verify_segment_launch_binding(
        segment_config,
        run_dir,
        segment_parent,
        location,
    )
    previous_flips = -1
    checkpoint_reports = NamedTuple[]
    root_zero_verified = false
    for artifact in artifacts
        artifact_file = JLD2.load(artifact.path)
        haskey(artifact_file, "payload") || error(
            "$location checkpoint $(artifact.update) has no payload",
        )
        artifact_payload = artifact_file["payload"]
        core = verify_checkpoint_payload_core(
            artifact_payload,
            expected_parameters,
            training_rows,
            state_batch;
            location="$location checkpoint $(artifact.update) payload",
            expected_update=artifact.update,
            expected_kind="training",
            expected_config=segment_config,
            expected_sampler_snapshot=deterministic_sampler_snapshot(
                training_rows,
                artifact.update,
                state_batch,
            ),
            expected_initial_metrics,
            expected_parent=segment_parent,
            expected_parent_is_set=true,
        )
        core.total_structural_flips >= previous_flips || error(
            "$location structural flip count decreases at update " *
            "$(artifact.update)",
        )
        previous_flips = core.total_structural_flips
        require_exact_state_equal(
            core.initial_parameters,
            required_property(
                payload,
                :initial_parameters,
                "$location target payload",
            ),
            "$location checkpoint $(artifact.update) initial parameters",
        )
        require_exact_state_equal(
            core.initial_metrics,
            required_property(
                payload,
                :initial_metrics,
                "$location target payload",
            ),
            "$location checkpoint $(artifact.update) initial metrics",
        )
        root_zero_verified |=
            start_mode == "scratch" && artifact.update == 0
        push!(checkpoint_reports, (;
            update=artifact.update,
            path=artifact.path,
            bytes=filesize(artifact.path),
            sha256=file_sha256(artifact.path),
            checkpoint_kind=core.kind,
            segment_start_update=Int(
                core.segment_state.start_update,
            ),
        ))
    end

    ancestor_report = nothing
    if start_mode == "scratch"
        root_zero_verified || error(
            "$location scratch lineage does not contain checkpoint zero",
        )
        segment_parent === nothing || error(
            "$location scratch lineage unexpectedly has a parent",
        )
    else
        segment_parent === nothing &&
            error("$location resume lineage has no parent")
        parent_reference = verify_checkpoint_reference_schema(
            segment_parent,
            "$location parent reference";
            expected_kind="training",
        )
        require_equal(
            parent_reference.update,
            policy.segment_start,
            "$location parent/segment boundary",
        )
        ancestor_report = verify_checkpoint_lineage_segment(
            segment_parent,
            child_config,
            expected_parameters,
            training_rows,
            state_batch,
            expected_initial_metrics,
            visited_paths,
            visited_digests;
            allow_residual_finalization=false,
            depth=depth + 1,
        )
    end
    return (;
        depth,
        path,
        bytes=filesize(path),
        sha256=file_sha256(path),
        update=normalized_reference.update,
        format=REQUIRED_CHECKPOINT_FORMAT,
        version=REQUIRED_CHECKPOINT_VERSION,
        checkpoint_kind="training",
        run_dir,
        start_mode,
        manifest_path=abspath(manifest_path),
        manifest_sha256=file_sha256(manifest_path),
        launch_manifest=(;
            path=segment_launch.path,
            bytes=segment_launch.bytes,
            sha256=segment_launch.sha256,
        ),
        checkpoint_cadence=policy,
        checkpoints=checkpoint_reports,
        ancestor=ancestor_report,
        root_zero_verified=
            start_mode == "scratch" ? true :
            ancestor_report.root_zero_verified,
    )
end

function verify_resume_ancestry(
    parsed,
    reported_parent,
    canonical_config,
    expected_parameters,
    training_rows,
    state_batch,
    expected_initial_metrics,
)
    if parsed.parent_checkpoint === nothing
        reported_parent === nothing ||
            error("scratch run unexpectedly reports a parent checkpoint")
        return nothing
    end
    reported_parent === nothing &&
        error("non-scratch run does not report a parent checkpoint")
    parent = parsed.parent_checkpoint
    isfile(parent.path) ||
        error("parent checkpoint does not exist: $(parent.path)")
    reported_reference = verify_checkpoint_reference_schema(
        reported_parent,
        "reported parent checkpoint";
        expected_kind="training",
    )
    require_equal(
        normalized_path(reported_reference.path),
        normalized_path(parent.path),
        "reported parent checkpoint path",
    )
    require_equal(
        reported_reference.sha256,
        parent.sha256,
        "reported parent checkpoint SHA-256",
    )
    require_equal(
        reported_reference.bytes,
        filesize(parent.path),
        "reported parent checkpoint byte size",
    )
    require_equal(
        reported_reference.update,
        parent.update,
        "reported parent checkpoint update",
    )
    require_equal(
        file_sha256(parent.path),
        parent.sha256,
        "parent checkpoint SHA-256",
    )
    if parsed.expected_start_mode == "resume"
        parent.update < parsed.expected_updates ||
            error("resume parent is not earlier than the target update")
    else
        require_equal(
            parent.update,
            parsed.expected_updates,
            "finalize-only parent update",
        )
    end
    lineage = verify_checkpoint_lineage_segment(
        reported_parent,
        canonical_config,
        expected_parameters,
        training_rows,
        state_batch,
        expected_initial_metrics,
        Set{String}(),
        Set{String}();
        allow_residual_finalization=
            parsed.expected_start_mode == "finalize-only",
    )
    lineage.root_zero_verified ||
        error("resume ancestry did not reach a scratch checkpoint zero")
    return lineage
end

function verify_team_teardown_artifact(
    path,
    config,
    expected_update,
)
    isfile(path) || error("team teardown does not exist: $path")
    payload = JSON3.read(read(path, String))
    require_equal(
        String(required_property(payload, :format, "team teardown")),
        "serial-workspace-snn-team-teardown",
        "team teardown format",
    )
    require_equal(
        Int(required_property(payload, :version, "team teardown")),
        1,
        "team teardown version",
    )
    require_equal(
        Int(required_property(payload, :update, "team teardown")),
        expected_update,
        "team teardown update",
    )
    require_equal(
        String(required_property(
            payload,
            :dataset_content_sha256,
            "team teardown",
        )),
        String(required_property(
            config,
            :dataset_content_sha256,
            "run config",
        )),
        "team teardown dataset binding",
    )
    require_semantic_equal(
        required_property(payload, :runtime_provenance, "team teardown"),
        required_property(config, :runtime_provenance, "run config"),
        "team teardown runtime provenance",
    )
    julia_workers = Int(required_property(
        payload,
        :julia_workers,
        "team teardown",
    ))
    require_equal(
        julia_workers,
        Int(required_property(config, :julia_threads, "run config")),
        "team teardown Julia worker count",
    )
    require_equal(
        Int(required_property(
            payload,
            :active_workers,
            "team teardown",
        )),
        Int(required_property(config, :active_workers, "run config")),
        "team teardown active worker count",
    )
    require_equal(
        String(required_property(
            payload,
            :cpuset_mode,
            "team teardown",
        )),
        "all",
        "team teardown CPU-set mode",
    )
    require_equal(
        Int(required_property(
            payload,
            :booted_workers,
            "team teardown",
        )),
        julia_workers,
        "team teardown booted worker count",
    )
    require_equal(
        Int(required_property(
            payload,
            :ready_workers,
            "team teardown",
        )),
        julia_workers,
        "team teardown ready worker count",
    )
    require_equal(
        Int(required_property(
            payload,
            :shutdown_requested,
            "team teardown",
        )),
        1,
        "team teardown shutdown acknowledgement",
    )
    for property in (:queue_closed, :all_bindings_verified,
                     :all_bindings_released)
        Bool(required_property(payload, property, "team teardown")) ||
            error("team teardown $(String(property)) is false")
    end
    for property in (:queue_length, :remaining)
        require_equal(
            Int(required_property(payload, property, "team teardown")),
            0,
            "team teardown $(String(property))",
        )
    end
    bindings = collect(required_property(
        payload,
        :bindings,
        "team teardown",
    ))
    require_equal(
        length(bindings),
        julia_workers,
        "team teardown binding count",
    )
    worker_slots = Int[]
    thread_ids = Int[]
    cpu_set_ids = Int[]
    for (index, binding) in enumerate(bindings)
        Bool(required_property(binding, :verified, "teardown binding")) ||
            error("team teardown binding $index is not verified")
        Bool(required_property(binding, :released, "teardown binding")) ||
            error("team teardown binding $index is not released")
        push!(worker_slots, Int(required_property(
            binding,
            :worker_slot,
            "teardown binding",
        )))
        push!(thread_ids, Int(required_property(
            binding,
            :julia_thread_id,
            "teardown binding",
        )))
        push!(cpu_set_ids, Int(required_property(
            binding,
            :cpu_set_id,
            "teardown binding",
        )))
    end
    worker_slots == collect(1:julia_workers) ||
        error("team teardown worker slots are not canonical")
    length(unique(thread_ids)) == julia_workers ||
        error("team teardown Julia thread IDs are not unique")
    length(unique(cpu_set_ids)) == julia_workers ||
        error("team teardown CPU-set IDs are not unique")
    return (;
        path=abspath(path),
        bytes=filesize(path),
        sha256=file_sha256(path),
        update=expected_update,
        cpu_set_ids,
        payload,
    )
end

function verify_finalization_artifacts(
    run_dir,
    results_path,
    results,
    canonical_config,
    expected_parameters,
    training_checkpoint,
    teardown,
    training_rows,
    state_batch,
    expected_update,
)
    manifest_path = joinpath(run_dir, "finalization_manifest.json")
    isfile(manifest_path) ||
        error("finalization manifest does not exist: $manifest_path")
    manifest = JSON3.read(read(manifest_path, String))
    require_equal(
        String(required_property(
            manifest,
            :format,
            "finalization manifest",
        )),
        "serial-workspace-snn-finalization-manifest",
        "finalization manifest format",
    )
    require_equal(
        Int(required_property(
            manifest,
            :version,
            "finalization manifest",
        )),
        1,
        "finalization manifest version",
    )
    require_equal(
        Int(required_property(
            manifest,
            :update,
            "finalization manifest",
        )),
        expected_update,
        "finalization manifest update",
    )
    require_equal(
        Int(required_property(
            manifest,
            :optimizer_steps_after_target,
            "finalization manifest",
        )),
        0,
        "finalization optimizer steps after target",
    )
    require_equal(
        String(required_property(
            manifest,
            :dataset_content_sha256,
            "finalization manifest",
        )),
        String(required_property(
            canonical_config,
            :dataset_content_sha256,
            "run config",
        )),
        "finalization dataset binding",
    )
    require_semantic_equal(
        required_property(
            manifest,
            :runtime_provenance,
            "finalization manifest",
        ),
        required_property(
            canonical_config,
            :runtime_provenance,
            "run config",
        ),
        "finalization runtime provenance",
    )
    results_artifact = verify_file_artifact_reference(
        required_property(manifest, :results, "finalization manifest"),
        results_path,
        "results",
        expected_update,
        "finalization results artifact",
    )
    teardown_artifact = verify_file_artifact_reference(
        required_property(
            manifest,
            :team_teardown,
            "finalization manifest",
        ),
        teardown.path,
        "team_teardown",
        expected_update,
        "finalization team teardown artifact",
    )
    require_semantic_equal(
        required_property(
            manifest,
            :training_checkpoint,
            "finalization manifest",
        ),
        required_property(results, :training_checkpoint, "results"),
        "finalization training checkpoint reference",
    )
    training_artifact = verify_file_artifact_reference(
        required_property(
            manifest,
            :training_checkpoint,
            "finalization manifest",
        ),
        training_checkpoint.path,
        "training",
        expected_update,
        "finalization training checkpoint",
    )
    finalization_path = joinpath(
        run_dir,
        "checkpoints",
        "finalization_checkpoint_" *
        lpad(string(expected_update), 9, '0') *
        ".jld2",
    )
    finalization_artifact = verify_file_artifact_reference(
        required_property(
            manifest,
            :finalization_checkpoint,
            "finalization manifest",
        ),
        finalization_path,
        "finalization",
        expected_update,
        "finalization checkpoint artifact",
    )
    require_semantic_equal(
        required_property(results, :checkpoint, "results"),
        required_property(
            manifest,
            :finalization_checkpoint,
            "finalization manifest",
        ),
        "results final checkpoint reference",
    )
    file = JLD2.load(finalization_path)
    haskey(file, "payload") ||
        error("finalization checkpoint has no payload")
    payload = file["payload"]
    verify_checkpoint_payload_core(
        payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="finalization payload",
        expected_update,
        expected_kind="finalization",
        expected_config=canonical_config,
        expected_sampler_snapshot=
            training_checkpoint.payload.sampler_state,
        expected_initial_metrics=required_property(
            results,
            :initial,
            "results",
        ),
        expected_parent=required_property(
            results,
            :training_checkpoint,
            "results",
        ),
        expected_parent_is_set=true,
    )
    verify_checkpoint_state_equivalence(
        training_checkpoint.payload,
        payload,
        "finalization versus training checkpoint",
    )
    require_equal(
        String(required_property(payload, :format, "finalization payload")),
        REQUIRED_CHECKPOINT_FORMAT,
        "finalization checkpoint format",
    )
    require_equal(
        Int(required_property(payload, :version, "finalization payload")),
        REQUIRED_CHECKPOINT_VERSION,
        "finalization checkpoint version",
    )
    require_equal(
        String(required_property(
            payload,
            :checkpoint_kind,
            "finalization payload",
        )),
        "finalization",
        "finalization checkpoint kind",
    )
    require_equal(
        Int(required_property(payload, :update, "finalization payload")),
        expected_update,
        "finalization checkpoint update",
    )
    require_semantic_equal(
        required_property(
            payload,
            :parent_checkpoint,
            "finalization payload",
        ),
        required_property(results, :training_checkpoint, "results"),
        "finalization checkpoint parent",
    )
    require_semantic_equal(
        required_property(payload, :config, "finalization payload"),
        canonical_config,
        "finalization checkpoint config",
    )
    for property in (
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
    )
        require_semantic_equal(
            required_property(
                payload,
                property,
                "finalization payload",
            ),
            required_property(
                canonical_config,
                property,
                "run config",
            ),
            "finalization checkpoint $(String(property))",
        )
    end
    parameters = required_property(
        payload,
        :parameters,
        "finalization payload",
    )
    verify_array_registry(
        parameters,
        expected_parameters,
        "finalization parameters",
    )
    initial_parameters = required_property(
        payload,
        :initial_parameters,
        "finalization payload",
    )
    verify_array_registry(
        initial_parameters,
        expected_parameters,
        "finalization initial parameters",
    )
    require_equal(
        array_registry_sha256(initial_parameters),
        array_registry_sha256(expected_parameters),
        "finalization deterministic initial parameters",
    )
    optimizer =
        required_property(payload, :optimizer, "finalization payload")
    require_equal(
        Int(required_property(
            optimizer,
            :step,
            "finalization optimizer",
        )),
        expected_update,
        "finalization optimizer step",
    )
    verify_array_registry(
        required_property(
            optimizer,
            :first_moment,
            "finalization optimizer",
        ),
        expected_parameters,
        "finalization first moment",
    )
    verify_array_registry(
        required_property(
            optimizer,
            :second_moment,
            "finalization optimizer",
        ),
        expected_parameters,
        "finalization second moment",
    )
    utility = required_property(
        payload,
        :synapse_utility,
        "finalization payload",
    )
    size(utility) == size(parameters.synapse_weight) ||
        error("finalization utility shape differs")
    all(isfinite, utility) ||
        error("finalization utility contains non-finite values")
    all(>=(0.0f0), utility) ||
        error("finalization utility contains negative values")
    require_equal(
        Int(required_property(
            payload,
            :utility_updates,
            "finalization payload",
        )),
        expected_update,
        "finalization utility update count",
    )
    flips = Int(required_property(
        payload,
        :total_structural_flips,
        "finalization payload",
    ))
    flips >= 0 || error("finalization structural flips are negative")
    verify_hard_gate_budget(
        parameters,
        canonical_config.model,
        "finalization checkpoint",
    )
    require_equal(
        Int(required_property(
            required_property(
                payload,
                :progress,
                "finalization payload",
            ),
            :updates,
            "finalization progress",
        )),
        expected_update,
        "finalization progress update",
    )
    reference_sampler = EpochSampler(training_rows, Xoshiro(SAMPLER_SEED))
    next_batch!(reference_sampler, expected_update * state_batch)
    require_semantic_equal(
        required_property(
            payload,
            :sampler_state,
            "finalization payload",
        ),
        sampler_snapshot(reference_sampler),
        "finalization deterministic sampler state",
    )

    training_file = JLD2.load(training_artifact.path)
    haskey(training_file, "payload") ||
        error("finalization training checkpoint has no payload")
    training_payload = training_file["payload"]
    require_equal(
        String(required_property(
            training_payload,
            :format,
            "finalization training payload",
        )),
        REQUIRED_CHECKPOINT_FORMAT,
        "finalization training checkpoint format",
    )
    require_equal(
        Int(required_property(
            training_payload,
            :version,
            "finalization training payload",
        )),
        REQUIRED_CHECKPOINT_VERSION,
        "finalization training checkpoint version",
    )
    require_equal(
        String(required_property(
            training_payload,
            :checkpoint_kind,
            "finalization training payload",
        )),
        "training",
        "finalization training checkpoint kind",
    )
    require_equal(
        Int(required_property(
            training_payload,
            :update,
            "finalization training payload",
        )),
        expected_update,
        "finalization training checkpoint update",
    )
    for (location, left, right) in (
        (
            "parameters",
            array_registry_sha256(parameters),
            array_registry_sha256(training_payload.parameters),
        ),
        (
            "initial parameters",
            array_registry_sha256(initial_parameters),
            array_registry_sha256(training_payload.initial_parameters),
        ),
        (
            "first moment",
            array_registry_sha256(optimizer.first_moment),
            array_registry_sha256(
                training_payload.optimizer.first_moment,
            ),
        ),
        (
            "second moment",
            array_registry_sha256(optimizer.second_moment),
            array_registry_sha256(
                training_payload.optimizer.second_moment,
            ),
        ),
    )
        require_equal(left, right, "finalization $location state")
    end
    require_equal(
        bytes2hex(sha256(reinterpret(UInt8, vec(utility)))),
        bytes2hex(sha256(reinterpret(
            UInt8,
            vec(training_payload.synapse_utility),
        ))),
        "finalization utility state",
    )
    require_semantic_equal(
        payload.sampler_state,
        training_payload.sampler_state,
        "finalization sampler versus training checkpoint",
    )
    require_semantic_equal(
        payload.progress,
        training_payload.progress,
        "finalization progress versus training checkpoint",
    )
    require_equal(
        flips,
        Int(training_payload.total_structural_flips),
        "finalization structural flips versus training checkpoint",
    )
    for property in (
        :production_contract,
        :production_contract_sha256,
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
        :source_fingerprint,
    )
        require_semantic_equal(
            required_property(
                training_payload.config,
                property,
                "finalization training config",
            ),
            required_property(
                canonical_config,
                property,
                "run config",
            ),
            "finalization parent/child $(String(property))",
        )
    end

    record = required_property(
        payload,
        :finalization,
        "finalization payload",
    )
    require_equal(
        String(required_property(record, :status, "finalization record")),
        "finalization_checkpoint_complete",
        "finalization status",
    )
    require_equal(
        Int(required_property(
            record,
            :optimizer_steps_after_target,
            "finalization record",
        )),
        0,
        "finalization record optimizer steps",
    )
    require_equal(
        normalized_path(required_property(
            record,
            :expected_results_path,
            "finalization record",
        )),
        normalized_path(results_path),
        "finalization record expected results path",
    )
    require_equal(
        normalized_path(required_property(
            record,
            :expected_manifest_path,
            "finalization record",
        )),
        normalized_path(manifest_path),
        "finalization record expected manifest path",
    )
    require_semantic_equal(
        required_property(record, :team_teardown, "finalization record"),
        required_property(
            manifest,
            :team_teardown,
            "finalization manifest",
        ),
        "finalization record team teardown",
    )
    require_semantic_equal(
        required_property(
            record,
            :training_checkpoint,
            "finalization record",
        ),
        required_property(
            manifest,
            :training_checkpoint,
            "finalization manifest",
        ),
        "finalization record training checkpoint",
    )
    require_exact_state_equal(
        required_property(
            record,
            :component_loss_telemetry,
            "finalization record",
        ),
        required_property(
            required_property(
                payload,
                :progress,
                "finalization payload",
            ),
            :component_losses,
            "finalization progress",
        ),
        "finalization record component loss telemetry",
    )
    require_exact_state_equal(
        required_property(
            record,
            :completed_component_loss_window_updates,
            "finalization record",
        ),
        required_property(
            required_property(
                payload,
                :progress,
                "finalization payload",
            ),
            :completed_component_loss_window_updates,
            "finalization progress",
        ),
        "finalization record completed component window count",
    )
    require_exact_state_equal(
        required_property(
            record,
            :component_loss_alias_contract,
            "finalization record",
        ),
        required_property(
            payload,
            :component_loss_alias_contract,
            "finalization payload",
        ),
        "finalization record component loss alias contract",
    )
    require_semantic_equal(
        required_property(
            results,
            :component_loss_alias_contract,
            "results",
        ),
        required_property(
            record,
            :component_loss_alias_contract,
            "finalization record",
        ),
        "results/finalization component loss alias contract",
    )
    require_semantic_equal(
        required_property(
            results,
            :component_loss_telemetry,
            "results",
        ),
        required_property(
            record,
            :component_loss_telemetry,
            "finalization record",
        ),
        "results/finalization component loss telemetry",
    )
    require_semantic_equal(
        required_property(
            results,
            :completed_component_loss_window_updates,
            "results",
        ),
        required_property(
            record,
            :completed_component_loss_window_updates,
            "finalization record",
        ),
        "results/finalization completed component window count",
    )
    require_semantic_equal(
        required_property(record, :final_metrics, "finalization record"),
        required_property(results, :final, "results"),
        "finalization record metrics",
    )
    results_finalization = required_property(
        results,
        :finalization,
        "results",
    )
    require_equal(
        String(required_property(
            results_finalization,
            :mode,
            "results finalization",
        )),
        "finalization_checkpoint_then_results_then_manifest",
        "results finalization mode",
    )
    require_equal(
        Int(required_property(
            results_finalization,
            :optimizer_steps_after_target,
            "results finalization",
        )),
        0,
        "results finalization optimizer steps",
    )
    require_semantic_equal(
        required_property(
            results_finalization,
            :checkpoint,
            "results finalization",
        ),
        required_property(results, :checkpoint, "results"),
        "results finalization checkpoint",
    )
    require_semantic_equal(
        required_property(
            results_finalization,
            :training_checkpoint,
            "results finalization",
        ),
        required_property(results, :training_checkpoint, "results"),
        "results finalization training checkpoint",
    )
    require_equal(
        normalized_path(required_property(
            results_finalization,
            :manifest_path,
            "results finalization",
        )),
        normalized_path(manifest_path),
        "results finalization manifest path",
    )
    return (;
        manifest=(;
            path=abspath(manifest_path),
            bytes=filesize(manifest_path),
            sha256=file_sha256(manifest_path),
            update=expected_update,
        ),
        results=results_artifact,
        team_teardown=teardown_artifact,
        training_checkpoint=training_artifact,
        finalization_checkpoint=finalization_artifact,
        payload,
    )
end

function verify_run(parsed)
    verifier_runtime = enforce_verifier_runtime!()
    run_dir = parsed.run_dir
    expected_updates = parsed.expected_updates
    isdir(run_dir) || error("run directory does not exist: $run_dir")
    launch = verify_launch_manifest(parsed)
    config_path = joinpath(run_dir, "config.json")
    isfile(config_path) ||
        error("config.json does not exist: $config_path")
    config_document = JSON3.read(read(config_path, String))
    canonical_config = required_property(
        config_document,
        :config,
        "config.json",
    )
    config_parent = required_property(
        config_document,
        :parent_checkpoint,
        "config.json",
    )
    results_path = joinpath(run_dir, "results.json")
    isfile(results_path) || error("results.json does not exist: $results_path")
    results = JSON3.read(read(results_path, String))
    for property in (
        :config,
        :component_loss_alias_contract,
        :parent_checkpoint,
        :initial,
        :final,
        :reducible_objective,
        :deltas,
        :learning_witness,
        :throughput,
        :checkpoint,
        :training_checkpoint,
        :persistent_team_warmup,
        :component_loss_telemetry,
        :completed_component_loss_window_updates,
        :final_training_dynamics,
        :bindings,
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
        :team_teardown,
        :training_trace,
        :finalization,
    )
        required_property(results, property, "results")
    end
    verify_component_loss_alias_contract(
        required_property(
            results,
            :component_loss_alias_contract,
            "results",
        ),
        "results component loss alias contract",
    )
    results_completed_component_window_updates = required_property(
        results,
        :completed_component_loss_window_updates,
        "results",
    )
    results_completed_component_window_updates isa Int || error(
        "results.completed_component_loss_window_updates is not Int",
    )
    results_completed_component_window_updates >= 0 || error(
        "results.completed_component_loss_window_updates is negative",
    )
    require_semantic_equal(
        results.config,
        canonical_config,
        "results config versus config.json",
    )
    require_semantic_equal(
        results.parent_checkpoint,
        config_parent,
        "results parent checkpoint versus config.json",
    )
    for property in (
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
    )
        require_semantic_equal(
            required_property(results, property, "results"),
            required_property(canonical_config, property, "run config"),
            "results $(String(property))",
        )
    end
    verify_production_config(
        canonical_config,
        parsed,
        launch,
    )
    bound_inputs = verify_bound_inputs(canonical_config, launch)
    model = build_model(:scaled_v2)
    expected_parameters, states = Lux.setup(Xoshiro(MODEL_SEED), model)
    ancestry = verify_resume_ancestry(
        parsed,
        config_parent,
        canonical_config,
        expected_parameters,
        bound_inputs.training_rows,
        Int(required_property(
            canonical_config,
            :state_batch,
            "run config",
        )),
        required_property(results, :initial, "results"),
    )

    run_id = String(required_property(results.config, :run_id, "results config"))
    require_equal(
        lowercase(basename(normpath(run_dir))),
        lowercase(run_id),
        "results run ID versus run-directory name",
    )
    require_equal(run_id, parsed.expected_run_id, "results run ID")
    require_equal(
        Int(required_property(
            results.config,
            :maximum_updates,
            "results config",
        )),
        expected_updates,
        "configured maximum updates",
    )
    state_batch =
        Int(required_property(results.config, :state_batch, "results config"))
    state_batch >= 1 || error("configured state batch must be positive")
    expected_teacher_states = expected_updates * state_batch
    require_equal(
        Int(required_property(
            results.learning_witness,
            :final_update,
            "results learning witness",
        )),
        expected_updates,
        "final learning-witness update",
    )
    require_equal(
        Int(required_property(
            results.learning_witness,
            :consumed_teacher_states,
            "results learning witness",
        )),
        expected_teacher_states,
        "learning-witness teacher-state count",
    )
    require_equal(
        Int(required_property(
            results.throughput,
            :updates,
            "results throughput",
        )),
        expected_updates,
        "throughput update count",
    )
    require_equal(
        Int(required_property(
            results.throughput,
            :teacher_states,
            "results throughput",
        )),
        expected_teacher_states,
        "throughput teacher-state count",
    )

    for property in (:initial, :final, :deltas, :learning_witness, :throughput)
        verify_finite(
            required_property(results, property, "results"),
            "results.$(String(property))",
        )
    end
    throughput = results.throughput
    require_equal(
        Int(required_property(
            throughput,
            :hot_allocation_bytes,
            "results throughput",
        )),
        0,
        "results total hot allocation",
    )
    require_equal(
        Float64(required_property(
            throughput,
            :hot_gc_seconds,
            "results throughput",
        )),
        0.0,
        "results total hot GC time",
    )
    expected_segment_start = if parsed.expected_start_mode == "finalize-only"
        parent_file = JLD2.load(parsed.parent_checkpoint.path)
        parent_payload = parent_file["payload"]
        Int(required_property(
            required_property(
                parent_payload,
                :segment_state,
                "finalize-only parent payload",
            ),
            :start_update,
            "finalize-only parent segment state",
        ))
    else
        ancestry === nothing ? 0 : ancestry.update
    end
    expected_segment_updates = expected_updates - expected_segment_start
    require_equal(
        Int(required_property(
            throughput,
            :segment_updates,
            "results throughput",
        )),
        expected_segment_updates,
        "results segment update count",
    )
    require_equal(
        Int(required_property(
            throughput,
            :optimizer_steps_this_process,
            "results throughput",
        )),
        parsed.expected_start_mode == "finalize-only" ?
            0 : expected_segment_updates,
        "results optimizer steps in this process",
    )

    expected_trace_path = if parsed.expected_start_mode == "finalize-only"
        joinpath(
            dirname(dirname(parsed.parent_checkpoint.path)),
            "training_trace.tsv",
        )
    else
        joinpath(run_dir, "training_trace.tsv")
    end
    trace_artifact = verify_file_artifact_reference(
        results.training_trace,
        expected_trace_path,
        "training_trace",
        expected_updates,
        "results training trace",
    )
    trace_path = trace_artifact.path
    trace = parse_trace(trace_path)
    require_equal(trace.last_update, expected_updates, "trace final update")
    require_equal(
        trace.last_teacher_states,
        expected_teacher_states,
        "trace final teacher-state count",
    )
    log_interval = Int(required_property(
        canonical_config,
        :log_interval,
        "run config",
    ))
    expected_trace_updates = [
        update
        for update in (expected_segment_start + 1):expected_updates
        if update == 1 ||
           update % log_interval == 0 ||
           update == expected_updates
    ]
    require_equal(
        trace.updates,
        expected_trace_updates,
        "training trace update set",
    )
    previous_trace_update = expected_segment_start
    for record in trace.parsed_records
        require_equal(
            record.teacher_states,
            record.update * state_batch,
            "trace teacher states at update $(record.update)",
        )
        require_equal(
            record.window_updates,
            record.update - previous_trace_update,
            "trace window size at update $(record.update)",
        )
        previous_trace_update = record.update
    end

    is_finalize_only = parsed.expected_start_mode == "finalize-only"
    expected = is_finalize_only ?
        nothing : expected_checkpoint_updates(results, expected_updates)
    raw_artifacts = checkpoint_files(
        joinpath(run_dir, "checkpoints"),
        expected_updates;
        require_training=!is_finalize_only,
    )
    is_finalize_only && !isempty(raw_artifacts) && error(
        "finalize-only segment contains arena training checkpoints",
    )
    artifact_updates = [artifact.update for artifact in raw_artifacts]
    if parsed.expected_start_mode == "scratch"
        first(artifact_updates) == 0 ||
            error("scratch run is missing checkpoint zero")
    elseif !is_finalize_only
        0 in artifact_updates && error(
            "a non-scratch segment contains an update-zero checkpoint",
        )
    end
    manifest_path = if is_finalize_only
        joinpath(
            dirname(dirname(parsed.parent_checkpoint.path)),
            "checkpoint_manifest.jsonl",
        )
    else
        joinpath(run_dir, "checkpoint_manifest.jsonl")
    end
    manifest = checkpoint_manifest(manifest_path)
    parent_artifacts = NamedTuple[]
    parent_residual_finalization_path = nothing
    if is_finalize_only
        parent_checkpoint_dir =
            dirname(parsed.parent_checkpoint.path)
        parent_run_dir = dirname(parent_checkpoint_dir)
        for completed_artifact in (
            "results.json",
            "finalization_manifest.json",
        )
            isfile(joinpath(parent_run_dir, completed_artifact)) && error(
                "finalize-only parent unexpectedly contains " *
                completed_artifact,
            )
        end
        parent_residual_finalization_path = joinpath(
            parent_checkpoint_dir,
            "finalization_checkpoint_" *
            lpad(string(expected_updates), 9, '0') *
            ".jld2",
        )
        parent_artifacts = checkpoint_files(
            parent_checkpoint_dir,
            expected_updates;
            require_training=true,
            allow_finalization=true,
        )
        verify_checkpoint_manifest_set(
            parent_artifacts,
            manifest,
            expected_updates,
            "finalize-only parent checkpoint manifest",
        )
        parent_target_artifacts = filter(
            artifact -> artifact.update == expected_updates,
            parent_artifacts,
        )
        length(parent_target_artifacts) == 1 || error(
            "finalize-only parent checkpoint set does not contain exactly " *
            "one target checkpoint",
        )
        parent_target_artifact = only(parent_target_artifacts)
        require_equal(
            normalized_path(parent_target_artifact.path),
            normalized_path(parsed.parent_checkpoint.path),
            "finalize-only parent checkpoint live path",
        )
        require_equal(
            file_sha256(parent_target_artifact.path),
            parsed.parent_checkpoint.sha256,
            "finalize-only parent checkpoint live SHA-256",
        )
        require_equal(
            filesize(parent_target_artifact.path),
            filesize(parsed.parent_checkpoint.path),
            "finalize-only parent checkpoint live byte size",
        )
    else
        nonzero_artifact_updates = filter(!iszero, artifact_updates)
        nonzero_artifact_updates == expected.updates || error(
            "checkpoint updates differ: observed=$artifact_updates expected=$(expected.updates) with optional update 0",
        )
        verify_checkpoint_manifest_set(
            raw_artifacts,
            manifest,
            expected_updates,
            "checkpoint manifest",
        )
    end

    checkpoints = NamedTuple[]
    payload_format = nothing
    payload_version = nothing
    initial_parameters_sha256 = nothing
    reference_sampler = EpochSampler(
        bound_inputs.training_rows,
        Xoshiro(SAMPLER_SEED),
    )
    reference_sampler_update = 0
    for artifact in raw_artifacts
        artifact.update >= reference_sampler_update ||
            error("checkpoint sampler updates are not ordered")
        additional_updates =
            artifact.update - reference_sampler_update
        if additional_updates > 0
            next_batch!(
                reference_sampler,
                additional_updates * state_batch,
            )
        end
        reference_sampler_update = artifact.update
        checkpoint = verify_checkpoint(
            artifact,
            results,
            expected_updates,
            state_batch,
            manifest,
            canonical_config,
            expected_parameters,
            bound_inputs.training_rows,
            sampler_snapshot(reference_sampler),
        )
        payload_format === nothing &&
            (payload_format = checkpoint.payload_format)
        payload_version === nothing &&
            (payload_version = checkpoint.payload_version)
        require_equal(
            checkpoint.payload_format,
            payload_format,
            "checkpoint payload format at update $(checkpoint.update)",
        )
        require_equal(
            checkpoint.payload_version,
            payload_version,
            "checkpoint payload version at update $(checkpoint.update)",
        )
        if !isempty(checkpoints)
            checkpoint.structural_flips >=
                last(checkpoints).structural_flips || error(
                "checkpoint structural flip count decreased at update " *
                "$(checkpoint.update)",
            )
        end
        if initial_parameters_sha256 === nothing
            initial_parameters_sha256 =
                checkpoint.initial_parameters_sha256
        else
            require_equal(
                checkpoint.initial_parameters_sha256,
                initial_parameters_sha256,
                "checkpoint initial-parameter SHA-256 at update " *
                "$(checkpoint.update)",
            )
        end
        push!(checkpoints, checkpoint)
    end

    parent_checkpoint_policy = nothing
    parent_residual_finalization = nothing
    final_checkpoint = if is_finalize_only
        parent_file = JLD2.load(parsed.parent_checkpoint.path)
        haskey(parent_file, "payload") ||
            error("finalize-only parent checkpoint has no payload")
        parent_payload = parent_file["payload"]
        parent_config = required_property(
            parent_payload,
            :config,
            "finalize-only parent payload",
        )
        parent_results = (;
            parent_checkpoint=required_property(
                parent_payload,
                :parent_checkpoint,
                "finalize-only parent payload",
            ),
            config=parent_config,
            initial=required_property(results, :initial, "results"),
        )
        parent_sampler = EpochSampler(
            bound_inputs.training_rows,
            Xoshiro(SAMPLER_SEED),
        )
        next_batch!(
            parent_sampler,
            expected_updates * state_batch,
        )
        verified_parent_checkpoint = verify_checkpoint(
            parent_target_artifact,
            parent_results,
            expected_updates,
            state_batch,
            manifest,
            parent_config,
            expected_parameters,
            bound_inputs.training_rows,
            sampler_snapshot(parent_sampler),
        )
        parent_checkpoint_policy = expected_parent_checkpoint_updates(
            parent_config,
            parent_payload,
            expected_updates,
        )
        parent_live_updates =
            [artifact.update for artifact in parent_artifacts]
        require_equal(
            parent_live_updates,
            parent_checkpoint_policy.updates,
            "finalize-only parent checkpoint cadence",
        )
        parent_residual_finalization = if isfile(
            parent_residual_finalization_path,
        )
            verify_residual_finalization_checkpoint(
                parent_residual_finalization_path,
                verified_parent_checkpoint,
                parent_config,
                expected_parameters,
                bound_inputs.training_rows,
                state_batch,
                expected_updates,
            )
        else
            (;
                present=false,
                verified=true,
                path=abspath(parent_residual_finalization_path),
            )
        end
        verified_parent_checkpoint
    else
        last(checkpoints)
    end
    require_equal(
        final_checkpoint.update,
        expected_updates,
        "final checkpoint update",
    )
    reported_checkpoint = results.training_checkpoint
    require_equal(
        Int(required_property(
            reported_checkpoint,
            :update,
            "results training checkpoint",
        )),
        expected_updates,
        "reported training checkpoint update",
    )
    require_equal(
        lowercase(String(required_property(
            reported_checkpoint,
            :sha256,
            "results training checkpoint",
        ))),
        final_checkpoint.sha256,
        "reported training checkpoint SHA-256",
    )
    require_equal(
        Int(required_property(
            reported_checkpoint,
            :bytes,
            "results training checkpoint",
        )),
        final_checkpoint.bytes,
        "reported training checkpoint byte size",
    )
    require_equal(
        normalized_path(required_property(
            reported_checkpoint,
            :path,
            "results training checkpoint",
        )),
        normalized_path(final_checkpoint.path),
        "reported training checkpoint path",
    )
    teardown_path = if is_finalize_only
        joinpath(
            dirname(dirname(parsed.parent_checkpoint.path)),
            "team_teardown.json",
        )
    else
        joinpath(run_dir, "team_teardown.json")
    end
    teardown = verify_team_teardown_artifact(
        teardown_path,
        canonical_config,
        expected_updates,
    )
    if is_finalize_only && parent_residual_finalization.present
        require_semantic_equal(
            parent_residual_finalization.team_teardown_reference,
            (;
                kind="team_teardown",
                path=teardown.path,
                bytes=teardown.bytes,
                sha256=teardown.sha256,
                update=teardown.update,
            ),
            "residual finalization team teardown reference",
        )
        require_semantic_equal(
            parent_residual_finalization.final_metrics,
            required_property(results, :final, "results"),
            "residual finalization fixed-panel metrics",
        )
    end
    require_semantic_equal(
        results.team_teardown,
        (;
            kind="team_teardown",
            path=teardown.path,
            bytes=teardown.bytes,
            sha256=teardown.sha256,
            update=teardown.update,
        ),
        "results team teardown artifact",
    )
    finalization = verify_finalization_artifacts(
        run_dir,
        results_path,
        results,
        canonical_config,
        expected_parameters,
        final_checkpoint,
        teardown,
        bound_inputs.training_rows,
        state_batch,
        expected_updates,
    )
    final_payload = finalization.payload
    require_semantic_equal(
        required_property(
            results,
            :component_loss_alias_contract,
            "results",
        ),
        required_property(
            final_payload,
            :component_loss_alias_contract,
            "final checkpoint",
        ),
        "results component loss alias contract versus final checkpoint",
    )
    require_semantic_equal(
        required_property(
            results,
            :component_loss_telemetry,
            "results",
        ),
        required_property(
            final_payload.progress,
            :component_losses,
            "final checkpoint progress",
        ),
        "results component loss telemetry versus final checkpoint",
    )
    require_semantic_equal(
        required_property(
            results,
            :completed_component_loss_window_updates,
            "results",
        ),
        required_property(
            final_payload.progress,
            :completed_component_loss_window_updates,
            "final checkpoint progress",
        ),
        "results completed component window count versus final checkpoint",
    )
    require_semantic_equal(
        required_property(
            results,
            :final_training_dynamics,
            "results",
        ),
        required_property(
            final_payload,
            :last_training_dynamics,
            "final checkpoint",
        ),
        "results final training dynamics versus final checkpoint",
    )
    metric_verification = verify_metrics_and_witness(
        results,
        final_payload,
        model,
        states,
        bound_inputs,
        trace,
        expected_updates,
        state_batch,
    )

    final_progress = final_payload.progress
    for property in (
        :updates,
        :teacher_states,
        :candidates,
        :hot_allocation_bytes,
    )
        require_equal(
            Int128(required_property(
                throughput,
                property,
                "results throughput",
            )),
            Int128(required_property(
                final_progress,
                property,
                "final checkpoint progress",
            )),
            "results throughput $(String(property))",
        )
    end
    for property in (
        :hot_wall_seconds,
        :hot_cpu_seconds,
        :hot_gc_seconds,
        :pack_seconds,
        :forward_seconds,
        :loss_seconds,
        :backward_seconds,
        :optimizer_seconds,
        :consolidation_seconds,
    )
        require_approx(
            required_property(
                throughput,
                property,
                "results throughput",
            ),
            required_property(
                final_progress,
                property,
                "final checkpoint progress",
            ),
            "results throughput $(String(property))";
            atol=1.0e-9,
            rtol=1.0e-7,
        )
    end
    warmup = results.persistent_team_warmup
    Bool(required_property(
        warmup,
        :isolation_verified,
        "persistent-team warmup",
    )) || error("persistent-team warmup isolation is not verified")
    for property in (:queue_length, :remaining, :failure_worker)
        require_equal(
            Int(required_property(
                warmup,
                property,
                "persistent-team warmup",
            )),
            0,
            "persistent-team warmup $(String(property))",
        )
    end
    bindings = results.bindings
    Bool(required_property(bindings, :verified, "results bindings")) ||
        error("CPU-set bindings were not verified")
    cpu_set_ids = collect(required_property(
        bindings,
        :cpu_set_ids,
        "results bindings",
    ))
    require_equal(
        length(cpu_set_ids),
        Int(required_property(
            canonical_config,
            :active_workers,
            "run config",
        )),
        "CPU-set binding count",
    )
    length(unique(Int.(cpu_set_ids))) == length(cpu_set_ids) ||
        error("CPU-set bindings are not unique")
    require_equal(
        Int.(cpu_set_ids),
        teardown.cpu_set_ids[1:length(cpu_set_ids)],
        "results CPU-set bindings versus teardown",
    )

    checkpoint_reports = [
        verify_verification_checkpoint_report(
            (;
                checkpoint.update,
                checkpoint.path,
                checkpoint.bytes,
                checkpoint.sha256,
                checkpoint.payload_format,
                checkpoint.payload_version,
                checkpoint_kind=checkpoint.checkpoint_kind,
                checkpoint.structural_flips,
                checkpoint.utility_updates,
                checkpoint.hard_gate_budget,
                checkpoint.initial_parameters_sha256,
            ),
            "verification checkpoint report at update " *
            string(checkpoint.update);
            expected_kind=checkpoint.checkpoint_kind,
        )
        for checkpoint in checkpoints
    ]
    training_checkpoint_report = verify_verification_checkpoint_report(
        (;
            update=final_checkpoint.update,
            path=final_checkpoint.path,
            bytes=final_checkpoint.bytes,
            sha256=final_checkpoint.sha256,
            payload_format=final_checkpoint.payload_format,
            payload_version=final_checkpoint.payload_version,
            checkpoint_kind=final_checkpoint.checkpoint_kind,
            structural_flips=final_checkpoint.structural_flips,
            utility_updates=final_checkpoint.utility_updates,
            hard_gate_budget=final_checkpoint.hard_gate_budget,
            initial_parameters_sha256=
                final_checkpoint.initial_parameters_sha256,
        ),
        "verification target training checkpoint report";
        expected_kind="training",
    )
    if is_finalize_only
        checkpoint_reports = [training_checkpoint_report]
    end
    final_checkpoint_report = merge(
        finalization.finalization_checkpoint,
        (;
            payload_format=REQUIRED_CHECKPOINT_FORMAT,
            payload_version=REQUIRED_CHECKPOINT_VERSION,
            checkpoint_kind="finalization",
        ),
    )

    return (;
        format=VERIFICATION_FORMAT,
        version=VERIFICATION_VERSION,
        status="verified_complete",
        verified=true,
        verified_at=string(now()),
        run_dir,
        run_id,
        expected_updates,
        expected_teacher_states,
        metrics_verified=true,
        verifier_runtime,
        config=(;
            path=abspath(config_path),
            bytes=filesize(config_path),
            sha256=file_sha256(config_path),
        ),
        launch_manifest=(;
            path=launch.path,
            bytes=launch.bytes,
            sha256=launch.sha256,
            code_artifacts=launch.code_artifacts,
        ),
        parent_checkpoint=ancestry,
        parent_checkpoint_policy,
        parent_residual_finalization,
        team_teardown=(;
            path=teardown.path,
            bytes=teardown.bytes,
            sha256=teardown.sha256,
            update=teardown.update,
        ),
        finalization_manifest=finalization.manifest,
        finalization_training_checkpoint=training_checkpoint_report,
        results=(;
            path=abspath(results_path),
            bytes=filesize(results_path),
            sha256=file_sha256(results_path),
            metrics_verified=true,
            fixed_panel_recomputation=metric_verification,
        ),
        trace=merge(trace, (; path=abspath(trace_path))),
        checkpoint_policy=(;
            interval=is_finalize_only ?
                Int(required_property(
                    canonical_config,
                    :checkpoint_interval,
                    "run config",
                )) : expected.interval,
            segment_start_update=expected_segment_start,
            manifest_present=true,
            manifest_path=abspath(manifest_path),
            manifest_bytes=filesize(manifest_path),
            manifest_sha256=file_sha256(manifest_path),
        ),
        checkpoints=checkpoint_reports,
        training_checkpoint=training_checkpoint_report,
        final_checkpoint=final_checkpoint_report,
    )
end

function verification_main(arguments=ARGS)
    verifier_runtime = try
        enforce_verifier_runtime!()
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        return 2
    end
    parsed = try
        parse_verification_arguments(arguments)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        return 2
    end

    try
        report = verify_run(parsed)
        atomic_verification_json(parsed.output_path, report)
        println(JSON3.write(report))
        return 0
    catch exception
        failure = (;
            format=VERIFICATION_FORMAT,
            version=VERIFICATION_VERSION,
            status="verification_failed",
            verified=false,
            verified_at=string(now()),
            run_dir=parsed.run_dir,
            run_id=parsed.expected_run_id,
            expected_start_mode=parsed.expected_start_mode,
            expected_updates=parsed.expected_updates,
            verifier_runtime,
            launch_manifest=(;
                path=parsed.launch_manifest_path,
                expected_sha256=parsed.launch_manifest_sha256,
            ),
            error=sprint(showerror, exception, catch_backtrace()),
        )
        try
            atomic_verification_json(parsed.output_path, failure)
        catch write_exception
            println(
                stderr,
                "could not write verification failure artifact: ",
                sprint(showerror, write_exception, catch_backtrace()),
            )
        end
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        return 1
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    exit(verification_main())
