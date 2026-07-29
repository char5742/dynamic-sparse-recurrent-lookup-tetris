using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore

const ANALYSIS_FORMAT = "serial-workspace-snn-checkpoint-analysis"
const ANALYSIS_VERSION = 2
const ARENA_CHECKPOINT_FORMAT =
    "serial-workspace-snn-arena-checkpoint"
const PRODUCTION_ARENA_CHECKPOINT_VERSION = 3
const LEGACY_ARENA_CHECKPOINT_VERSIONS = Set((1, 2))
const RUN_VERIFICATION_FORMAT =
    "serial-workspace-snn-arena-run-verification"
const RUN_VERIFICATION_VERSION = 2
const SHA256_PATTERN = r"^[0-9a-f]{64}$"
const CHECKPOINT_FILENAME_PATTERN =
    r"^checkpoint_([0-9]{9})\.jld2$"
const FINALIZATION_CHECKPOINT_FILENAME_PATTERN =
    r"^finalization_checkpoint_([0-9]{9})\.jld2$"
const SPLIT_SEED = UInt64(2026071817)
const TRAIN_EVAL_SEED = UInt64(2026071801) + UInt64(0x101)
const VALIDATION_EVAL_SEED = UInt64(2026071801) + UInt64(0x202)
const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_PANEL_STATES = 128
const DEFAULT_ROUTING_EXPLORATION = 0.05f0
const SATURATION_ELIGIBILITY_EDGE_SAMPLES = 1024
const ROUTING_ENTROPY_FLOOR_FALLBACK = 0.70
const TRAINING_TRACE_V3_COLUMNS = (
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
    :window_old_q_loss_sum,
    :window_q_huber_loss_sum,
    :window_margin_loss_sum,
    :window_raw_top_gap_loss_sum,
    :window_death_loss_sum,
    :window_quantile_teacher_loss_sum,
    :window_geometry_loss_sum,
    :window_line_clear_loss_sum,
    :window_max_height_loss_sum,
    :window_holes_loss_sum,
    :window_cavities_loss_sum,
    :window_structure_loss_sum,
    :gradient_norm,
    :enabled_synapses,
    :structural_flips_total,
    :training_dynamics_schema_version,
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
    :states_per_second,
    :cpu_percent,
    :hot_allocation_bytes,
    :hot_gc_seconds,
    :shadow_seconds,
)
const TRAINING_TRACE_V3_INTEGER_COLUMNS = Set((
    :trace_schema_version,
    :training_dynamics_schema_version,
    :update,
    :teacher_states,
    :window_updates,
    :enabled_synapses,
    :structural_flips_total,
    :net_mask_flips,
    :hot_allocation_bytes,
    :component_loss_alias_schema_version,
))
const TRAINING_TRACE_V3_BOOLEAN_COLUMNS = Set((
    :consolidation_scheduled,
    :consolidation_actual,
))
const TRAINING_TRACE_V3_STRING_COLUMNS = Set((
    :q_huber_loss_alias_of,
    :raw_top_gap_loss_alias_of,
    :component_loss_alias_identity,
))
const GAP_BUCKETS = (
    (name="lt_0.01", lower=0.0, upper=0.01),
    (name="0.01_to_0.02", lower=0.01, upper=0.02),
    (name="0.02_to_0.05", lower=0.02, upper=0.05),
    (name="0.05_to_0.10", lower=0.05, upper=0.10),
    (name="0.10_to_0.20", lower=0.10, upper=0.20),
    (name="ge_0.20", lower=0.20, upper=Inf),
)
const ABLATION_NAMES = (
    :full,
    :workspace_off,
    :selected_pool_off,
    :both_off,
    :synapse_off,
    :memory_off,
)
const EXACT_CHECKPOINT_STATE_FIELDS = (
    :parameters,
    :optimizer,
    :trainer_state,
    :sampler_state,
    :initial_parameters,
    :initial_metrics,
    :progress,
    :persistent_team_warmup,
    :segment_state,
    :last_training_dynamics,
    :synapse_utility,
    :utility_updates,
    :total_structural_flips,
)
const LINEAGE_CONFIG_FIELDS = (
    :experiment_id,
    :checkpoint_schema,
    :production_contract,
    :production_contract_sha256,
    :model_preset,
    :model,
    :parameter_count,
    :maximum_updates,
    :state_batch,
    :candidate_width,
    :learning_mode,
    :eprop,
    :routing,
    :executor,
    :training_rows_sha256,
    :training_panel_rows_sha256,
    :dataset_content_sha256,
    :dataset_integrity,
    :runtime_provenance,
    :source_fingerprint,
)

function usage()
    return """
Usage:
  julia --project=experiments/beat_first_v1 \
      experiments/beat_first_v1/serial_workspace_snn/analyze_arena_checkpoint.jl \
      RUN_DIR [PANEL]
  julia --project=experiments/beat_first_v1 \
      experiments/beat_first_v1/serial_workspace_snn/analyze_arena_checkpoint.jl \
      --checkpoint CHECKPOINT --panel PANEL \
      --allow-unverified-checkpoint --checkpoint-sha256 HEX \
      --require-update N
  julia --project=experiments/beat_first_v1 \
      experiments/beat_first_v1/serial_workspace_snn/analyze_arena_checkpoint.jl \
      --run-dir RUN_DIR [--panel both]
  julia --project=experiments/beat_first_v1 \
      experiments/beat_first_v1/serial_workspace_snn/analyze_arena_checkpoint.jl \
      --checkpoint CHECKPOINT --panel fixed|validation|both|ROWS_FILE \
      --allow-unverified-checkpoint --checkpoint-sha256 HEX \
      --require-update N

Options:
  --dataset PATH              Override the checkpoint dataset.
  --states N                  Fixed-training-panel state count (default 128).
  --validation-states N       Validation-panel state count (default --states).
  --output PATH               Atomic JSON output path.
  --checkpoint-sha256 HEX     Require this checkpoint SHA-256.
  --verification-sha256 HEX   Optionally pin verification.json in run mode.
  --require-update N          Fail unless the checkpoint update is N.
  --no-parity-check           Report, but do not enforce, reference-forward parity.
  --allow-unverified-checkpoint
                              Required for direct --checkpoint analysis. Direct
                              analysis also requires --checkpoint-sha256 and
                              --require-update.
  --allow-legacy-provenance   Permit an old checkpoint only for an explicitly
                              provenance-incomplete, non-production report.

Environment fallbacks:
  SWSNN_RUN_DIR, SWSNN_CHECKPOINT, SWSNN_ANALYSIS_PANEL, SWSNN_DATASET,
  SWSNN_ANALYSIS_STATES, SWSNN_ANALYSIS_VALIDATION_STATES,
  SWSNN_ANALYSIS_OUTPUT, SWSNN_CHECKPOINT_SHA256, SWSNN_REQUIRE_UPDATE.
  SWSNN_VERIFICATION_SHA256.
"""
end

function parse_arguments(args)
    values = Dict{String,String}()
    flags = Set{String}()
    positional = String[]
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            println(usage())
            exit()
        elseif argument in (
            "--no-parity-check",
            "--allow-unverified-checkpoint",
            "--allow-legacy-provenance",
        )
            push!(flags, argument[3:end])
            index += 1
        elseif startswith(argument, "--")
            name = argument[3:end]
            name in (
                "run-dir",
                "checkpoint",
                "panel",
                "dataset",
                "states",
                "validation-states",
                "output",
                "checkpoint-sha256",
                "verification-sha256",
                "require-update",
            ) || error("unknown argument --$name\n$(usage())")
            index < length(args) ||
                error("missing value after --$name\n$(usage())")
            values[name] = args[index + 1]
            index += 2
        else
            push!(positional, argument)
            index += 1
        end
    end
    length(positional) <= 2 ||
        error("too many positional arguments\n$(usage())")

    run_dir = get(values, "run-dir", "")
    checkpoint = get(values, "checkpoint", "")
    panel = get(
        values,
        "panel",
        get(ENV, "SWSNN_ANALYSIS_PANEL", "both"),
    )
    if !isempty(positional)
        target = positional[1]
        if isdir(target)
            isempty(run_dir) ||
                error("run directory was supplied more than once")
            run_dir = target
        else
            isempty(checkpoint) ||
                error("checkpoint was supplied more than once")
            checkpoint = target
        end
    end
    if length(positional) == 2
        haskey(values, "panel") &&
            error("panel was supplied more than once")
        panel = positional[2]
    end
    if isempty(strip(run_dir)) && isempty(strip(checkpoint))
        run_dir = get(ENV, "SWSNN_RUN_DIR", "")
        checkpoint = get(ENV, "SWSNN_CHECKPOINT", "")
    end
    isempty(strip(run_dir)) || isempty(strip(checkpoint)) ||
        error("use either a run directory or an explicit checkpoint, not both")
    isempty(strip(run_dir)) && isempty(strip(checkpoint)) &&
        error("supply RUN_DIR or CHECKPOINT\n$(usage())")

    states = parse(
        Int,
        get(
            values,
            "states",
            get(
                ENV,
                "SWSNN_ANALYSIS_STATES",
                string(DEFAULT_PANEL_STATES),
            ),
        ),
    )
    states >= 1 || error("--states must be positive")
    validation_states = parse(
        Int,
        get(
            values,
            "validation-states",
            get(
                ENV,
                "SWSNN_ANALYSIS_VALIDATION_STATES",
                string(states),
            ),
        ),
    )
    validation_states >= 1 ||
        error("--validation-states must be positive")
    required_update_text = get(
        values,
        "require-update",
        get(ENV, "SWSNN_REQUIRE_UPDATE", ""),
    )
    required_update = isempty(strip(required_update_text)) ?
        nothing : parse(Int, required_update_text)
    required_update === nothing || required_update >= 0 ||
        error("--require-update must be non-negative")
    expected_sha256 = lowercase(strip(get(
        values,
        "checkpoint-sha256",
        get(ENV, "SWSNN_CHECKPOINT_SHA256", ""),
    )))
    isempty(expected_sha256) ||
        occursin(SHA256_PATTERN, expected_sha256) ||
        error("--checkpoint-sha256 must be exactly 64 hexadecimal digits")
    expected_verification_sha256 = lowercase(strip(get(
        values,
        "verification-sha256",
        get(ENV, "SWSNN_VERIFICATION_SHA256", ""),
    )))
    isempty(expected_verification_sha256) ||
        occursin(SHA256_PATTERN, expected_verification_sha256) ||
        error(
            "--verification-sha256 must be exactly 64 hexadecimal digits",
        )
    allow_unverified_checkpoint =
        "allow-unverified-checkpoint" in flags
    allow_legacy_provenance =
        "allow-legacy-provenance" in flags
    if !isempty(strip(run_dir))
        allow_unverified_checkpoint && error(
            "--allow-unverified-checkpoint is only valid with an explicit " *
            "--checkpoint",
        )
        allow_legacy_provenance && error(
            "--allow-legacy-provenance is only valid with an explicit " *
            "--checkpoint",
        )
    else
        isempty(expected_verification_sha256) || error(
            "--verification-sha256 is only valid with --run-dir",
        )
        allow_unverified_checkpoint || error(
            "direct checkpoint analysis is unverified; explicitly pass " *
            "--allow-unverified-checkpoint",
        )
        isempty(expected_sha256) && error(
            "direct checkpoint analysis requires --checkpoint-sha256",
        )
        required_update === nothing && error(
            "direct checkpoint analysis requires --require-update",
        )
    end
    return (;
        run_dir=isempty(strip(run_dir)) ? nothing : abspath(run_dir),
        checkpoint=isempty(strip(checkpoint)) ?
            nothing : abspath(checkpoint),
        panel=strip(panel),
        dataset=get(
            values,
            "dataset",
            get(ENV, "SWSNN_DATASET", ""),
        ),
        states,
        validation_states,
        output=get(
            values,
            "output",
            get(ENV, "SWSNN_ANALYSIS_OUTPUT", ""),
        ),
        expected_sha256,
        expected_verification_sha256,
        required_update,
        enforce_parity=!("no-parity-check" in flags),
        allow_unverified_checkpoint,
        allow_legacy_provenance,
    )
end

@inline function property_or(value, name::Symbol, default=nothing)
    if value isa AbstractDict
        haskey(value, name) && return value[name]
        haskey(value, String(name)) && return value[String(name)]
        return default
    end
    return hasproperty(value, name) ? getproperty(value, name) : default
end

function required_property(
    value,
    name::Symbol,
    location::AbstractString,
)
    result = property_or(value, name, nothing)
    result === nothing &&
        error("$location is missing $(String(name))")
    return result
end

file_sha256(path) = bytes2hex(open(sha256, path))

function require_sha256(value, location::AbstractString)
    digest = lowercase(String(value))
    occursin(SHA256_PATTERN, digest) ||
        error("$location is not a canonical SHA-256 digest")
    return digest
end

function normalized_existing_path(path, location::AbstractString)
    candidate = abspath(String(path))
    ispath(candidate) || error("$location does not exist: $candidate")
    resolved = realpath(candidate)
    return Sys.iswindows() ?
        lowercase(normpath(resolved)) : normpath(resolved)
end

normalized_declared_path(path) = Sys.iswindows() ?
    lowercase(normpath(abspath(String(path)))) :
    normpath(abspath(String(path)))

function path_is_within(child, parent)
    child_path = normalized_existing_path(child, "contained path")
    parent_path = normalized_existing_path(parent, "container path")
    relative = relpath(child_path, parent_path)
    isabspath(relative) && return false
    relative == "." && return true
    parts = splitpath(relative)
    return !isempty(parts) && first(parts) != ".."
end

function declared_path_is_within(child, parent)
    child_path = normalized_declared_path(child)
    parent_path = normalized_declared_path(parent)
    relative = try
        relpath(child_path, parent_path)
    catch exception
        exception isa ArgumentError || rethrow()
        return false
    end
    isabspath(relative) && return false
    relative == "." && return true
    parts = splitpath(relative)
    return !isempty(parts) && first(parts) != ".."
end

function normalized_path_through_existing_ancestor(path)
    candidate = abspath(String(path))
    suffix = String[]
    while !ispath(candidate)
        parent = dirname(candidate)
        parent == candidate &&
            error("path has no existing ancestor: $(abspath(String(path)))")
        pushfirst!(suffix, basename(candidate))
        candidate = parent
    end
    resolved = isempty(suffix) ?
        realpath(candidate) :
        joinpath(realpath(candidate), suffix...)
    return Sys.iswindows() ?
        lowercase(normpath(resolved)) : normpath(resolved)
end

function resolved_declared_path_is_within(child, parent)
    child_path = normalized_path_through_existing_ancestor(child)
    parent_path = normalized_existing_path(parent, "container path")
    relative = try
        relpath(child_path, parent_path)
    catch exception
        exception isa ArgumentError || rethrow()
        return false
    end
    isabspath(relative) && return false
    relative == "." && return true
    parts = splitpath(relative)
    return !isempty(parts) && first(parts) != ".."
end

function require_same_existing_path(
    actual,
    expected,
    location::AbstractString,
)
    normalized_existing_path(actual, location) ==
        normalized_existing_path(expected, "$location expected path") ||
        error(
            "$location differs: observed=$(abspath(String(actual))) " *
            "expected=$(abspath(String(expected)))",
        )
    return abspath(String(actual))
end

function read_json_object(path, location::AbstractString)
    isfile(path) || error("$location does not exist: $path")
    value = try
        JSON3.read(read(path, String))
    catch exception
        error("$location is malformed JSON: $(sprint(showerror, exception))")
    end
    value isa JSON3.Object ||
        error("$location must contain a JSON object")
    return value
end

function read_json_object_snapshot(path, location::AbstractString)
    artifact_path = abspath(path)
    isfile(artifact_path) ||
        error("$location does not exist: $artifact_path")
    data = read(artifact_path)
    value = try
        JSON3.read(String(data))
    catch exception
        error("$location is malformed JSON: $(sprint(showerror, exception))")
    end
    value isa JSON3.Object ||
        error("$location must contain a JSON object")
    return (;
        document=value,
        path=artifact_path,
        bytes=length(data),
        sha256=bytes2hex(sha256(data)),
    )
end

# Passing both values through JSON is deliberate. Training writes config.json
# from the same NamedTuple embedded in JLD2; this removes Julia storage-type
# differences (for example Symbol versus JSON string) without weakening any
# recorded value.
canonical_json(value) = JSON3.read(JSON3.write(value))
canonical_json_text(value) = JSON3.write(canonical_json(value))

function require_canonical_equal(
    actual,
    expected,
    location::AbstractString,
)
    canonical_json_text(actual) == canonical_json_text(expected) ||
        error("$location differs")
    return actual
end

function checkpoint_record(
    value,
    location::AbstractString;
    allow_finalization::Bool=false,
)
    update = Int(required_property(value, :update, location))
    update >= 0 || error("$location update must be non-negative")
    path = abspath(String(required_property(value, :path, location)))
    bytes = Int(required_property(value, :bytes, location))
    bytes >= 1 || error("$location byte size must be positive")
    digest = require_sha256(
        required_property(value, :sha256, location),
        "$location SHA-256",
    )
    matched = match(CHECKPOINT_FILENAME_PATTERN, basename(path))
    artifact_kind = "training"
    if matched === nothing && allow_finalization
        matched = match(
            FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
            basename(path),
        )
        artifact_kind = "finalization"
    end
    matched === nothing && error(
        "$location path is not an allowed checkpoint artifact: $path",
    )
    parse(Int, only(matched.captures)) == update ||
        error("$location filename update differs from its update")
    for name in (:kind, :checkpoint_kind)
        declared_kind = property_or(value, name, nothing)
        declared_kind === nothing && continue
        String(declared_kind) == artifact_kind ||
            error(
                "$location $(String(name)) differs from filename-derived " *
                "artifact kind $artifact_kind",
            )
    end
    return (; update, path, bytes, sha256=digest, artifact_kind)
end

function checkpoint_manifest_records_from_bytes(data, location::AbstractString)
    records = Dict{Int,NamedTuple}()
    line_count = 0
    for (line_number, line) in enumerate(eachline(IOBuffer(data)))
        isempty(strip(line)) && error(
            "$location contains a blank line at $line_number",
        )
        line_count += 1
        parsed = try
            JSON3.read(line)
        catch exception
            error(
                "$location line $line_number is malformed: " *
                sprint(showerror, exception),
            )
        end
        parsed isa JSON3.Object ||
            error("$location line $line_number is not an object")
        String(required_property(
            parsed,
            :kind,
            "$location line $line_number",
        )) == "training" ||
            error("$location line $line_number kind is not training")
        record = checkpoint_record(
            parsed,
            "$location line $line_number",
        )
        haskey(records, record.update) &&
            error("$location duplicates update $(record.update)")
        records[record.update] = record
    end
    line_count >= 1 || error("$location is empty")
    return records
end

function checkpoint_manifest_records(path)
    isfile(path) || error("checkpoint manifest does not exist: $path")
    return checkpoint_manifest_records_from_bytes(
        read(path),
        "checkpoint manifest",
    )
end

function checkpoint_manifest_snapshot(path)
    artifact_path = abspath(path)
    isfile(artifact_path) ||
        error("checkpoint manifest does not exist: $artifact_path")
    data = read(artifact_path)
    return (;
        records=checkpoint_manifest_records_from_bytes(
            data,
            "checkpoint manifest",
        ),
        path=artifact_path,
        bytes=length(data),
        sha256=bytes2hex(sha256(data)),
    )
end

function verify_file_record(
    record,
    expected_path,
    location::AbstractString,
)
    require_same_existing_path(record.path, expected_path, "$location path")
    filesize(record.path) == record.bytes ||
        error("$location byte size differs from the live file")
    file_sha256(record.path) == record.sha256 ||
        error("$location SHA-256 differs from the live file")
    return record
end

function file_artifact_record(value, location::AbstractString)
    path = abspath(String(required_property(value, :path, location)))
    isfile(path) || error("$location path does not exist: $path")
    bytes = Int(required_property(value, :bytes, location))
    bytes >= 1 || error("$location byte size must be positive")
    digest = require_sha256(
        required_property(value, :sha256, location),
        "$location SHA-256",
    )
    filesize(path) == bytes ||
        error("$location byte size differs from the live file")
    file_sha256(path) == digest ||
        error("$location SHA-256 differs from the live file")
    return (; path, bytes, sha256=digest)
end

function require_artifact_semantics(
    value,
    location::AbstractString;
    kind::AbstractString,
    update::Int,
)
    String(required_property(value, :kind, location)) == kind ||
        error("$location kind differs from $kind")
    Int(required_property(value, :update, location)) == update ||
        error("$location update differs from $update")
    return value
end

function require_numeric_match(
    actual,
    expected,
    location::AbstractString;
    tolerance::Float64=1.0e-7,
)
    actual_value = Float64(actual)
    expected_value = Float64(expected)
    isfinite(actual_value) && isfinite(expected_value) ||
        error("$location is not finite")
    isapprox(
        actual_value,
        expected_value;
        rtol=tolerance,
        atol=tolerance,
    ) || error(
        "$location differs: observed=$actual_value expected=$expected_value",
    )
    return actual_value
end

function parse_trace_integer(text::AbstractString, location)
    occursin(r"^-?[0-9]+$", text) ||
        error("$location is not a canonical integer")
    value = tryparse(Int, text)
    value === nothing &&
        error("$location is outside the Int range")
    return value
end

function parse_trace_boolean(text::AbstractString, location)
    text == "true" && return true
    text == "false" && return false
    error("$location is not canonical true/false")
end

function expected_trace_updates(
    segment_start::Int,
    terminal_update::Int,
    log_interval::Int,
)
    0 <= segment_start < terminal_update ||
        throw(ArgumentError("trace segment bounds are invalid"))
    log_interval >= 1 ||
        throw(ArgumentError("trace log interval must be positive"))
    updates = Int[]
    for update in (segment_start + 1):terminal_update
        (
            update == 1 ||
            update % log_interval == 0 ||
            update == terminal_update
        ) && push!(updates, update)
    end
    return updates
end

function parse_bound_training_trace(
    artifact;
    segment_start::Int,
    terminal_update::Int,
    log_interval::Int,
    state_batch::Int,
    require_v3::Bool,
)
    state_batch >= 1 ||
        throw(ArgumentError("trace state batch must be positive"))
    bytes = read(artifact.path)
    length(bytes) == artifact.bytes ||
        error("training trace byte size changed before parsing")
    bytes2hex(sha256(bytes)) == artifact.sha256 ||
        error("training trace SHA-256 changed before parsing")
    text = String(bytes)
    occursin('\r', text) &&
        error("training trace contains a noncanonical carriage return")
    endswith(text, '\n') ||
        error("training trace is missing its terminal newline")
    lines = split(text, '\n'; keepempty=true)
    pop!(lines)
    isempty(lines) &&
        error("training trace is empty")
    any(isempty, lines) &&
        error("training trace contains a blank line")
    header_text = split(first(lines), '\t'; keepempty=true)
    any(isempty, header_text) &&
        error("training trace header contains an empty column")
    length(unique(header_text)) == length(header_text) ||
        error("training trace header contains a duplicate column")
    header = Symbol.(header_text)
    header_set = Set(header)
    expected_v3 = Set(TRAINING_TRACE_V3_COLUMNS)
    if header_set != expected_v3
        missing = sort!(String.(collect(setdiff(
            expected_v3,
            header_set,
        ))))
        extra = sort!(String.(collect(setdiff(
            header_set,
            expected_v3,
        ))))
        error(
            "training trace schema is not exact v3; missing=" *
            join(missing, ",") * " extra=" * join(extra, ","),
        )
    end
    if require_v3 && header_set != expected_v3
        error("production analysis requires training trace v3")
    end

    records = Vector{Dict{Symbol,Any}}()
    for (line_index, line) in enumerate(@view(lines[2:end]))
        fields = split(line, '\t'; keepempty=true)
        length(fields) == length(header) ||
            error(
                "training trace row $line_index has " *
                "$(length(fields)) fields; expected $(length(header))",
            )
        any(isempty, fields) &&
            error("training trace row $line_index contains an empty field")
        record = Dict{Symbol,Any}()
        for (column, field) in zip(header, fields)
            location =
                "training trace row $line_index column $column"
            value = if column in TRAINING_TRACE_V3_INTEGER_COLUMNS
                parse_trace_integer(field, location)
            elseif column in TRAINING_TRACE_V3_BOOLEAN_COLUMNS
                parse_trace_boolean(field, location)
            elseif column in TRAINING_TRACE_V3_STRING_COLUMNS
                String(field)
            else
                parsed = tryparse(Float64, field)
                parsed === nothing &&
                    error("$location is not numeric")
                isfinite(parsed) ||
                    error("$location is not finite")
                parsed
            end
            record[column] = value
        end
        record[:trace_schema_version] == 3 ||
            error("training trace row $line_index schema version differs")
        record[:training_dynamics_schema_version] == 3 ||
            error(
                "training trace row $line_index dynamics schema version differs",
            )
        record[:component_loss_alias_schema_version] == 1 ||
            error(
                "training trace row $line_index component alias schema differs",
            )
        record[:q_huber_loss_alias_of] == "old_q_loss" ||
            error(
                "training trace row $line_index q-Huber alias differs",
            )
        record[:raw_top_gap_loss_alias_of] == "margin_loss" ||
            error(
                "training trace row $line_index top-gap alias differs",
            )
        record[:component_loss_alias_identity] == "bit_exact" ||
            error(
                "training trace row $line_index alias identity differs",
            )
        record[:q_huber_loss] == record[:old_q_loss] ||
            error(
                "training trace row $line_index q-Huber alias value differs",
            )
        record[:raw_top_gap_loss] == record[:margin_loss] ||
            error(
                "training trace row $line_index top-gap alias value differs",
            )
        record[:window_q_huber_loss_sum] ==
            record[:window_old_q_loss_sum] ||
            error(
                "training trace row $line_index window q-Huber alias differs",
            )
        record[:window_raw_top_gap_loss_sum] ==
            record[:window_margin_loss_sum] ||
            error(
                "training trace row $line_index window top-gap alias differs",
            )
        record[:update] >= 1 ||
            error("training trace row $line_index update is not positive")
        record[:teacher_states] ==
            record[:update] * state_batch ||
            error(
                "training trace row $line_index teacher-state count differs",
            )
        record[:window_updates] >= 1 ||
            error("training trace row $line_index window is empty")
        record[:enabled_synapses] >= 0 ||
            error("training trace row $line_index enabled count is negative")
        record[:structural_flips_total] >= 0 ||
            error("training trace row $line_index flip count is negative")
        record[:net_mask_flips] >= 0 ||
            error("training trace row $line_index net flips are negative")
        record[:hot_allocation_bytes] >= 0 ||
            error("training trace row $line_index allocations are negative")
        require_v3 && record[:hot_gc_seconds] != 0.0 &&
            error(
                "training trace row $line_index reports GC time inside " *
                "the production hot segment",
            )
        record[:consolidation_actual] &&
            !record[:consolidation_scheduled] &&
            error(
                "training trace row $line_index consolidated without schedule",
            )
        push!(records, record)
    end
    isempty(records) &&
        error("training trace has no data records")
    observed_updates = Int[
        record[:update] for record in records
    ]
    all(>(0), diff(observed_updates)) ||
        error("training trace updates are not strictly increasing")
    length(unique(observed_updates)) == length(observed_updates) ||
        error("training trace contains a duplicate update")
    expected_updates = expected_trace_updates(
        segment_start,
        terminal_update,
        log_interval,
    )
    observed_updates == expected_updates ||
        error(
            "training trace cadence differs: observed=" *
            string(observed_updates) * " expected=" *
            string(expected_updates),
        )
    last(observed_updates) == terminal_update ||
        error("training trace is nonterminal")
    structural_flips = Int[
        record[:structural_flips_total] for record in records
    ]
    issorted(structural_flips) ||
        error("training trace cumulative structural flips decreased")
    hot_allocations = Int[
        record[:hot_allocation_bytes] for record in records
    ]
    issorted(hot_allocations) ||
        error("training trace cumulative hot allocations decreased")
    return (;
        artifact,
        schema_version=3,
        columns=String.(header),
        column_count=length(header),
        rows=length(records),
        segment_start,
        terminal_update,
        log_interval,
        state_batch,
        expected_updates,
        records,
    )
end

function ols_trace_window(updates, values)
    length(updates) == length(values) ||
        throw(DimensionMismatch("trace regression lengths differ"))
    count_value = length(updates)
    count_value >= 1 ||
        throw(ArgumentError("trace regression is empty"))
    x = Float64.(updates)
    y = Float64.(values)
    if count_value == 1
        return (;
            points=1,
            update_first=Int(first(updates)),
            update_last=Int(last(updates)),
            slope_per_update=nothing,
            slope_per_1000_updates=nothing,
            intercept=first(y),
            residual_std=nothing,
            slope_standard_error=nothing,
        )
    end
    x_mean = mean(x)
    y_mean = mean(y)
    centered_x = x .- x_mean
    centered_y = y .- y_mean
    sum_x_square = sum(abs2, centered_x)
    sum_x_square > 0.0 ||
        error("trace regression updates have zero variance")
    slope = dot(centered_x, centered_y) / sum_x_square
    intercept = y_mean - slope * x_mean
    residual = y .- (intercept .+ slope .* x)
    residual_std = count_value > 2 ?
        sqrt(sum(abs2, residual) / (count_value - 2)) :
        nothing
    slope_standard_error = residual_std === nothing ?
        nothing : residual_std / sqrt(sum_x_square)
    return (;
        points=count_value,
        update_first=Int(first(updates)),
        update_last=Int(last(updates)),
        slope_per_update=slope,
        slope_per_1000_updates=1000.0 * slope,
        intercept,
        residual_std,
        slope_standard_error,
    )
end

function trace_metric_trend(updates, values)
    count_value = length(updates)
    count_value == length(values) ||
        throw(DimensionMismatch("trace trend lengths differ"))
    endpoint_width = count_value == 1 ?
        1 : min(10, fld(count_value, 2))
    last50 = max(1, count_value - 49):count_value
    last20 = max(1, count_value - 19):count_value
    return (;
        first_value=Float64(first(values)),
        last_value=Float64(last(values)),
        last_minus_first=
            Float64(last(values)) - Float64(first(values)),
        endpoint_window_width=endpoint_width,
        first_window_mean=mean(Float64.(
            @view values[1:endpoint_width]
        )),
        last_window_mean=mean(Float64.(
            @view values[(end - endpoint_width + 1):end]
        )),
        last_minus_first_window_mean=
            mean(Float64.(@view values[
                (end - endpoint_width + 1):end
            ])) -
            mean(Float64.(@view values[1:endpoint_width])),
        all=ols_trace_window(updates, values),
        last_50=ols_trace_window(
            @view(updates[last50]),
            @view(values[last50]),
        ),
        last_20=ols_trace_window(
            @view(updates[last20]),
            @view(values[last20]),
        ),
    )
end

function training_trace_summary(parsed)
    records = parsed.records
    updates = Int[record[:update] for record in records]
    excluded = union(
        TRAINING_TRACE_V3_BOOLEAN_COLUMNS,
        TRAINING_TRACE_V3_STRING_COLUMNS,
        Set((
            :trace_schema_version,
            :training_dynamics_schema_version,
            :update,
            :teacher_states,
        )),
    )
    trends = Dict{String,Any}()
    for column in TRAINING_TRACE_V3_COLUMNS
        column in excluded && continue
        values = Float64[record[column] for record in records]
        trends[String(column)] = trace_metric_trend(
            updates,
            values,
        )
    end
    window_sum_columns = filter(
        column ->
            startswith(String(column), "window_") &&
            endswith(String(column), "_sum"),
        TRAINING_TRACE_V3_COLUMNS,
    )
    derived = Dict{String,Any}()
    for column in window_sum_columns
        values = Float64[
            record[column] / record[:window_updates]
            for record in records
        ]
        name = replace(String(column), "_sum" => "_per_update")
        derived[name] = trace_metric_trend(updates, values)
    end
    return (;
        artifact=parsed.artifact,
        schema_version=parsed.schema_version,
        columns=parsed.columns,
        column_count=parsed.column_count,
        rows=parsed.rows,
        segment_start=parsed.segment_start,
        terminal_update=parsed.terminal_update,
        log_interval=parsed.log_interval,
        state_batch=parsed.state_batch,
        expected_updates=parsed.expected_updates,
        cadence_verified=true,
        numeric_finiteness_verified=true,
        component_loss_alias_contract=(;
            schema_version=1,
            q_huber_loss=(;
                alias_of="old_q_loss",
                identity="bit_exact",
            ),
            raw_top_gap_loss=(;
                alias_of="margin_loss",
                identity="bit_exact",
            ),
            aliases_are_not_independent_loss_evidence=true,
        ),
        stochastic_scope=(
            "point fields are one stochastic minibatch; window fields are " *
            "means or sums over the preceding logged update window"
        ),
        trends,
        derived_window_component_means=derived,
    )
end

function strict_verified_run_binding(
    run_dir::AbstractString,
    expected_verification_sha256::AbstractString="",
)
    root = abspath(run_dir)
    isdir(root) || error("run directory does not exist: $root")
    verification_path = joinpath(root, "verification.json")
    verification_snapshot = read_json_object_snapshot(
        verification_path,
        "verification.json",
    )
    verification = verification_snapshot.document
    verification_sha256 = verification_snapshot.sha256
    verification_bytes = verification_snapshot.bytes
    isempty(expected_verification_sha256) ||
        verification_sha256 == expected_verification_sha256 ||
        error("verification.json SHA-256 differs from the caller pin")
    String(required_property(
        verification,
        :format,
        "verification.json",
    )) == RUN_VERIFICATION_FORMAT ||
        error("verification.json format is not the arena verifier format")
    Int(required_property(
        verification,
        :version,
        "verification.json",
    )) == RUN_VERIFICATION_VERSION ||
        error(
            "verification.json version must be " *
            string(RUN_VERIFICATION_VERSION),
        )
    Bool(required_property(
        verification,
        :verified,
        "verification.json",
    )) || error("verification.json does not certify the run")
    String(required_property(
        verification,
        :status,
        "verification.json",
    )) == "verified_complete" ||
        error("verification.json status is not verified_complete")
    Bool(required_property(
        verification,
        :metrics_verified,
        "verification.json",
    )) || error("verification.json did not verify fixed-panel metrics")
    verifier_runtime = required_property(
        verification,
        :verifier_runtime,
        "verification.json",
    )
    Int(required_property(
        verifier_runtime,
        :julia_threads,
        "verification verifier runtime",
    )) >= 1 || error("verifier runtime reports no Julia threads")
    Int(required_property(
        verifier_runtime,
        :blas_threads,
        "verification verifier runtime",
    )) == 1 || error("verifier runtime did not use one BLAS thread")
    Int(required_property(
        verifier_runtime,
        :required_blas_threads,
        "verification verifier runtime",
    )) == 1 || error("verifier BLAS-thread contract differs")
    Bool(required_property(
        verifier_runtime,
        :blas_contract_verified,
        "verification verifier runtime",
    )) || error("verifier BLAS-thread contract was not verified")
    Bool(required_property(
        verifier_runtime,
        :startup_file_disabled,
        "verification verifier runtime",
    )) || error("verifier startup file was not disabled")
    Bool(required_property(
        verifier_runtime,
        :history_file_disabled,
        "verification verifier runtime",
    )) || error("verifier history file was not disabled")
    Int(required_property(
        verifier_runtime,
        :startup_file_option,
        "verification verifier runtime",
    )) == 2 || error("verifier startup-file option differs")
    Int(required_property(
        verifier_runtime,
        :history_file_option,
        "verification verifier runtime",
    )) == 0 || error("verifier history-file option differs")
    require_same_existing_path(
        required_property(verification, :run_dir, "verification.json"),
        root,
        "verification run directory",
    )
    run_id = String(required_property(
        verification,
        :run_id,
        "verification.json",
    ))
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) ||
        error("verification run ID is unsafe")
    (
        Sys.iswindows() ?
        lowercase(basename(normpath(root))) == lowercase(run_id) :
        basename(normpath(root)) == run_id
    ) ||
        error("verification run ID differs from the run directory name")
    expected_updates = Int(required_property(
        verification,
        :expected_updates,
        "verification.json",
    ))
    expected_updates >= 1 ||
        error("verification expected update must be positive")
    launch_preview_record = required_property(
        verification,
        :launch_manifest,
        "verification.json",
    )
    launch_preview_path = abspath(String(required_property(
        launch_preview_record,
        :path,
        "verification launch manifest",
    )))
    require_same_existing_path(
        launch_preview_path,
        joinpath(
            dirname(root),
            "_controllers",
            run_id,
            "launch_manifest.json",
        ),
        "controller launch manifest path",
    )
    launch_preview_snapshot = read_json_object_snapshot(
        launch_preview_path,
        "launch manifest",
    )
    launch_preview = launch_preview_snapshot.document
    launch_start_mode = String(required_property(
        launch_preview,
        :start_mode,
        "launch manifest",
    ))
    launch_start_mode in ("scratch", "resume", "finalize-only") ||
        error("unsupported verified start mode $launch_start_mode")
    is_finalize_only = launch_start_mode == "finalize-only"

    checkpoint_dir = joinpath(root, "checkpoints")
    isdir(checkpoint_dir) ||
        error("verified run has no checkpoints directory")
    raw_verified_checkpoints = required_property(
        verification,
        :checkpoints,
        "verification.json",
    )
    raw_verified_checkpoints isa AbstractVector ||
        raw_verified_checkpoints isa JSON3.Array ||
        error("verification checkpoints must be an array")
    isempty(raw_verified_checkpoints) &&
        error("verification checkpoints are empty")
    verified_checkpoints = Dict{Int,NamedTuple}()
    for (index, raw) in enumerate(raw_verified_checkpoints)
        record = checkpoint_record(
            raw,
            "verification checkpoint $index",
        )
        haskey(verified_checkpoints, record.update) &&
            error("verification duplicates checkpoint update $(record.update)")
        if !is_finalize_only
            dirname(normalized_existing_path(
                record.path,
                "verification checkpoint $index",
            )) == normalized_existing_path(
                checkpoint_dir,
                "checkpoint directory",
            ) || error(
                "verification checkpoint $index is outside the run checkpoint " *
                "directory",
            )
        end
        verify_file_record(
            record,
            is_finalize_only ?
                record.path :
                joinpath(checkpoint_dir, basename(record.path)),
            "verification checkpoint $index",
        )
        verified_checkpoints[record.update] = record
    end
    parent_residual_finalization_record = nothing
    parent_residual_finalization_checkpoint = nothing
    parent_residual_expected_results_path = nothing
    parent_residual_expected_manifest_path = nothing
    if is_finalize_only
        length(verified_checkpoints) == 1 ||
            error(
                "finalize-only verification must bind exactly one parent " *
                "training checkpoint",
            )
        haskey(verified_checkpoints, expected_updates) ||
            error(
                "finalize-only verification does not bind the target parent " *
                "training checkpoint",
            )
    end

    final_checkpoint = checkpoint_record(
        required_property(
            verification,
            :final_checkpoint,
            "verification.json",
        ),
        "verification final checkpoint";
        allow_finalization=true,
    )
    final_checkpoint.update == expected_updates ||
        error("verified final checkpoint update differs from expected_updates")
    final_checkpoint.artifact_kind == "finalization" ||
        error("verification final checkpoint is not a finalization artifact")
    dirname(normalized_existing_path(
        final_checkpoint.path,
        "verification final checkpoint",
    )) == normalized_existing_path(
        checkpoint_dir,
        "checkpoint directory",
    ) || error("verification final checkpoint is outside checkpoint directory")
    verify_file_record(
        final_checkpoint,
        final_checkpoint.path,
        "verification final checkpoint",
    )

    # A completed run is immutable for analysis purposes. The allowed set is
    # the verified periodic/training checkpoints plus exactly one verified
    # finalization checkpoint.
    live_checkpoint_entries = readdir(checkpoint_dir; join=true)
    all(isfile, live_checkpoint_entries) ||
        error("checkpoints directory contains a non-file entry")
    expected_entry_paths = Set(
        normalized_declared_path(record.path)
        for record in values(verified_checkpoints)
        if dirname(normalized_existing_path(
            record.path,
            "verified checkpoint",
        )) == normalized_existing_path(
            checkpoint_dir,
            "checkpoint directory",
        )
    )
    push!(
        expected_entry_paths,
        normalized_declared_path(final_checkpoint.path),
    )
    live_entry_paths = Set(
        normalized_declared_path(path)
        for path in live_checkpoint_entries
    )
    length(live_checkpoint_entries) == length(expected_entry_paths) ||
        error("checkpoints directory contains an alias or extra entry")
    live_entry_paths == expected_entry_paths ||
        error("checkpoint directory entry paths differ from verification")
    verified_paths = Set(
        normalized_existing_path(record.path, "verified checkpoint")
        for record in values(verified_checkpoints)
        if dirname(normalized_existing_path(
            record.path,
            "verified checkpoint",
        )) == normalized_existing_path(
            checkpoint_dir,
            "checkpoint directory",
        )
    )
    push!(
        verified_paths,
        normalized_existing_path(
            final_checkpoint.path,
            "verified finalization checkpoint",
        ),
    )
    live_paths = Set(
        normalized_existing_path(path, "live checkpoint entry")
        for path in live_checkpoint_entries
    )
    length(verified_paths) == length(expected_entry_paths) ||
        error("verified checkpoint artifacts contain a resolved-path alias")
    length(live_paths) == length(live_checkpoint_entries) ||
        error("live checkpoint entries contain a resolved-path alias")
    live_paths == verified_paths || error(
        "live checkpoint directory differs from the exact verified artifact " *
        "set (extra, partial, latest, missing, or replaced checkpoint)",
    )
    for name in ("checkpoint_final.jld2", "checkpoint_latest.jld2")
        ispath(joinpath(root, name)) && error(
            "run directory contains an unverified checkpoint alias: $name",
        )
    end

    results_record = required_property(
        verification,
        :results,
        "verification.json",
    )
    Bool(required_property(
        results_record,
        :metrics_verified,
        "verification results",
    )) || error("verification results did not verify metrics")
    fixed_panel_recomputation = required_property(
        results_record,
        :fixed_panel_recomputation,
        "verification results",
    )
    Bool(required_property(
        fixed_panel_recomputation,
        :verified,
        "verification fixed-panel recomputation",
    )) || error("verifier fixed-panel recomputation did not match")
    results_path = abspath(String(required_property(
        results_record,
        :path,
        "verification results",
    )))
    require_same_existing_path(
        results_path,
        joinpath(root, "results.json"),
        "verification results path",
    )
    results_bytes = Int(required_property(
        results_record,
        :bytes,
        "verification results",
    ))
    results_sha256 = require_sha256(
        required_property(
            results_record,
            :sha256,
            "verification results",
        ),
        "verification results SHA-256",
    )
    results_snapshot = read_json_object_snapshot(
        results_path,
        "results.json",
    )
    results_snapshot.bytes == results_bytes ||
        error("results.json byte size differs from verification")
    results_snapshot.sha256 == results_sha256 ||
        error("results.json SHA-256 differs from verification")
    results = results_snapshot.document
    verification_parent = property_or(
        verification,
        :parent_checkpoint,
        nothing,
    )
    results_parent = property_or(
        results,
        :parent_checkpoint,
        nothing,
    )
    parent_checkpoint = nothing
    if launch_start_mode == "scratch"
        verification_parent === nothing ||
            error("scratch verification unexpectedly has a parent")
        results_parent === nothing ||
            error("scratch results unexpectedly have a parent")
    else
        verification_parent === nothing &&
            error("$launch_start_mode verification has no parent checkpoint")
        results_parent === nothing &&
            error("$launch_start_mode results have no parent checkpoint")
        parent_checkpoint = checkpoint_record(
            verification_parent,
            "verification parent checkpoint";
            allow_finalization=true,
        )
        results_parent_checkpoint = checkpoint_record(
            results_parent,
            "results parent checkpoint";
            allow_finalization=true,
        )
        require_canonical_equal(
            results_parent_checkpoint,
            parent_checkpoint,
            "results/verification parent checkpoint",
        )
        verify_file_record(
            parent_checkpoint,
            parent_checkpoint.path,
            "verification parent checkpoint",
        )
        parent_checkpoint.artifact_kind == "training" ||
            error("$launch_start_mode parent checkpoint is not a training artifact")
        parent_run_dir = dirname(dirname(parent_checkpoint.path))
        require_same_existing_path(
            dirname(parent_checkpoint.path),
            joinpath(parent_run_dir, "checkpoints"),
            "$launch_start_mode parent checkpoint directory",
        )
        if is_finalize_only
            require_canonical_equal(
                parent_checkpoint,
                verified_checkpoints[expected_updates],
                "finalize-only parent/verified training checkpoint",
            )
            !path_is_within(parent_run_dir, root) &&
                !path_is_within(root, parent_run_dir) ||
                error(
                    "finalize-only parent and child run directories must be " *
                    "disjoint",
                )
            parent_residual_expected_results_path =
                joinpath(parent_run_dir, "results.json")
            parent_residual_expected_manifest_path =
                joinpath(parent_run_dir, "finalization_manifest.json")
            !ispath(parent_residual_expected_results_path) ||
                error(
                    "finalize-only parent has a results artifact; the parent " *
                    "run is already finalized",
                )
            !ispath(parent_residual_expected_manifest_path) ||
                error(
                    "finalize-only parent has a finalization manifest; the " *
                    "parent run is already finalized or corrupt",
                )
        end
    end
    verification_trace_raw = required_property(
        verification,
        :trace,
        "verification.json",
    )
    Int(required_property(
        verification_trace_raw,
        :last_update,
        "verification training trace",
    )) == expected_updates ||
        error("verification training trace final update differs")
    trace_artifact = file_artifact_record(
        verification_trace_raw,
        "verification training trace",
    )
    verification_teardown_raw = required_property(
        verification,
        :team_teardown,
        "verification.json",
    )
    Int(required_property(
        verification_teardown_raw,
        :update,
        "verification team teardown",
    )) == expected_updates ||
        error("verification team teardown update differs")
    team_teardown_artifact = file_artifact_record(
        verification_teardown_raw,
        "verification team teardown",
    )
    lineage_artifact_root = is_finalize_only ?
        dirname(dirname(parent_checkpoint.path)) : root
    require_same_existing_path(
        trace_artifact.path,
        joinpath(lineage_artifact_root, "training_trace.tsv"),
        "verification training trace path",
    )
    require_same_existing_path(
        team_teardown_artifact.path,
        joinpath(lineage_artifact_root, "team_teardown.json"),
        "verification team teardown path",
    )
    results_trace_raw = require_artifact_semantics(
        required_property(
            results,
            :training_trace,
            "results.json",
        ),
        "results training trace";
        kind="training_trace",
        update=expected_updates,
    )
    results_trace_artifact = file_artifact_record(
        results_trace_raw,
        "results training trace",
    )
    require_canonical_equal(
        results_trace_artifact,
        trace_artifact,
        "results/verification training trace",
    )
    results_teardown_raw = require_artifact_semantics(
        required_property(
            results,
            :team_teardown,
            "results.json",
        ),
        "results team teardown";
        kind="team_teardown",
        update=expected_updates,
    )
    results_teardown_artifact = file_artifact_record(
        results_teardown_raw,
        "results team teardown",
    )
    require_canonical_equal(
        results_teardown_artifact,
        team_teardown_artifact,
        "results/verification team teardown",
    )

    results_checkpoint = checkpoint_record(
        required_property(results, :checkpoint, "results.json"),
        "results checkpoint";
        allow_finalization=true,
    )
    results_checkpoint.update == expected_updates ||
        error("results finalization checkpoint update differs")
    require_canonical_equal(
        results_checkpoint,
        final_checkpoint,
        "results/verification finalization checkpoint",
    )
    verify_file_record(
        results_checkpoint,
        final_checkpoint.path,
        "results finalization checkpoint",
    )
    results_training_checkpoint = checkpoint_record(
        required_property(
            results,
            :training_checkpoint,
            "results.json",
        ),
        "results training checkpoint",
    )
    results_training_checkpoint.update == expected_updates ||
        error("results final training checkpoint update differs")
    haskey(
        verified_checkpoints,
        results_training_checkpoint.update,
    ) ||
        error("results training checkpoint is absent from verification")
    verification_training_checkpoint = checkpoint_record(
        required_property(
            verification,
            :training_checkpoint,
            "verification.json",
        ),
        "verification training checkpoint",
    )
    finalization_training_checkpoint = checkpoint_record(
        required_property(
            verification,
            :finalization_training_checkpoint,
            "verification.json",
        ),
        "verification finalization training checkpoint",
    )
    require_canonical_equal(
        results_training_checkpoint,
        verified_checkpoints[results_training_checkpoint.update],
        "results/checkpoint-list training checkpoint",
    )
    require_canonical_equal(
        verification_training_checkpoint,
        results_training_checkpoint,
        "verification/results training checkpoint",
    )
    require_canonical_equal(
        finalization_training_checkpoint,
        results_training_checkpoint,
        "verification finalization/results training checkpoint",
    )
    verify_file_record(
        results_training_checkpoint,
        verified_checkpoints[results_training_checkpoint.update].path,
        "results final training checkpoint",
    )
    results_finalization = required_property(
        results,
        :finalization,
        "results.json",
    )
    String(required_property(
        results_finalization,
        :mode,
        "results finalization",
    )) == "finalization_checkpoint_then_results_then_manifest" ||
        error("results finalization mode differs")
    Int(required_property(
        results_finalization,
        :optimizer_steps_after_target,
        "results finalization",
    )) == 0 ||
        error("results finalization reports post-target optimizer steps")
    require_canonical_equal(
        checkpoint_record(
            required_property(
                results_finalization,
                :checkpoint,
                "results finalization",
            ),
            "results finalization checkpoint";
            allow_finalization=true,
        ),
        final_checkpoint,
        "results finalization/final checkpoint",
    )
    require_canonical_equal(
        checkpoint_record(
            required_property(
                results_finalization,
                :training_checkpoint,
                "results finalization",
            ),
            "results finalization training checkpoint",
        ),
        results_training_checkpoint,
        "results finalization/training checkpoint",
    )

    policy = required_property(
        verification,
        :checkpoint_policy,
        "verification.json",
    )
    Bool(required_property(
        policy,
        :manifest_present,
        "verification checkpoint policy",
    )) || error("verified production run must have a checkpoint manifest")
    manifest_path = abspath(String(required_property(
        policy,
        :manifest_path,
        "verification checkpoint policy",
    )))
    expected_manifest_path = is_finalize_only ?
        joinpath(
            dirname(dirname(parent_checkpoint.path)),
            "checkpoint_manifest.jsonl",
        ) :
        joinpath(root, "checkpoint_manifest.jsonl")
    require_same_existing_path(
        manifest_path,
        expected_manifest_path,
        "verification checkpoint manifest path",
    )
    manifest_snapshot = checkpoint_manifest_snapshot(manifest_path)
    manifest = manifest_snapshot.records
    manifest_sha256 = manifest_snapshot.sha256
    manifest_bytes = manifest_snapshot.bytes
    require_sha256(
        required_property(
            policy,
            :manifest_sha256,
            "verification checkpoint policy",
        ),
        "verification checkpoint manifest SHA-256",
    ) == manifest_sha256 ||
        error("verification checkpoint manifest SHA-256 differs")
    Int(required_property(
        policy,
        :manifest_bytes,
        "verification checkpoint policy",
    )) == manifest_bytes ||
        error("verification checkpoint manifest byte size differs")
    if is_finalize_only
        haskey(manifest, expected_updates) ||
            error(
                "finalize-only parent checkpoint manifest is missing the " *
                "target update",
            )
        maximum(keys(manifest)) == expected_updates ||
            error(
                "finalize-only parent checkpoint manifest contains a higher " *
                "or non-final target update",
            )
        parent_checkpoint_dir = dirname(parent_checkpoint.path)
        for (update, record) in manifest
            dirname(normalized_existing_path(
                record.path,
                "parent checkpoint manifest record $update",
            )) == normalized_existing_path(
                parent_checkpoint_dir,
                "parent checkpoint directory",
            ) || error(
                "parent checkpoint manifest record $update is outside the " *
                "parent checkpoint directory",
            )
            verify_file_record(
                record,
                record.path,
                "parent checkpoint manifest record $update",
            )
        end
        parent_entries = readdir(parent_checkpoint_dir; join=true)
        all(isfile, parent_entries) ||
            error("parent checkpoint directory contains a non-file entry")
        live_training_entries = String[]
        live_finalization_entries = String[]
        for path in parent_entries
            training_match =
                match(CHECKPOINT_FILENAME_PATTERN, basename(path))
            finalization_match = match(
                FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
                basename(path),
            )
            if training_match !== nothing
                push!(live_training_entries, path)
            elseif finalization_match !== nothing
                parse(Int, only(finalization_match.captures)) ==
                    expected_updates || error(
                    "parent checkpoint directory contains a finalization " *
                    "artifact for the wrong update",
                )
                push!(live_finalization_entries, path)
            else
                error(
                    "parent checkpoint directory contains an unverified " *
                    "extra, partial, latest, or aliased artifact: " *
                    basename(path),
                )
            end
        end
        length(live_finalization_entries) <= 1 ||
            error(
                "parent checkpoint directory contains duplicate finalization " *
                "artifacts",
            )
        if length(live_finalization_entries) == 1
            residual_path = only(live_finalization_entries)
            parent_residual_finalization_record = checkpoint_record(
                (;
                    update=expected_updates,
                    path=residual_path,
                    bytes=filesize(residual_path),
                    sha256=file_sha256(residual_path),
                ),
                "parent residual finalization checkpoint";
                allow_finalization=true,
            )
            parent_residual_finalization_checkpoint =
                load_analysis_checkpoint(
                    residual_path,
                    parent_residual_finalization_record.sha256,
                )
            parent_residual_finalization_checkpoint.production_schema ||
                error(
                    "parent residual finalization checkpoint is not " *
                    "production schema",
                )
            String(
                parent_residual_finalization_checkpoint.checkpoint_kind,
            ) == "finalization" ||
                error(
                    "parent residual finalization payload kind differs",
                )
            parent_residual_finalization_checkpoint.update ==
                expected_updates ||
                error(
                    "parent residual finalization payload update differs",
                )
            require_canonical_equal(
                checkpoint_record(
                    parent_residual_finalization_checkpoint.
                        payload_parent_checkpoint,
                    "parent residual finalization parent checkpoint",
                ),
                manifest[expected_updates],
                "parent residual finalization/manifest training checkpoint",
            )
            residual_finalization =
                parent_residual_finalization_checkpoint.payload_finalization
            residual_finalization === nothing &&
                error(
                    "parent residual finalization checkpoint has no " *
                    "finalization record",
                )
            String(required_property(
                residual_finalization,
                :status,
                "parent residual finalization record",
            )) == "finalization_checkpoint_complete" ||
                error("parent residual finalization record is incomplete")
            Int(required_property(
                residual_finalization,
                :optimizer_steps_after_target,
                "parent residual finalization record",
            )) == 0 ||
                error(
                    "parent residual finalization has post-target optimizer " *
                    "steps",
                )
            require_canonical_equal(
                checkpoint_record(
                    required_property(
                        residual_finalization,
                        :training_checkpoint,
                        "parent residual finalization record",
                    ),
                    "parent residual finalization training checkpoint",
                ),
                manifest[expected_updates],
                "parent residual finalization/manifest training checkpoint",
            )
            normalized_declared_path(required_property(
                residual_finalization,
                :expected_results_path,
                "parent residual finalization record",
            )) == normalized_declared_path(
                parent_residual_expected_results_path,
            ) || error(
                "parent residual finalization expected results path differs",
            )
            normalized_declared_path(required_property(
                residual_finalization,
                :expected_manifest_path,
                "parent residual finalization record",
            )) == normalized_declared_path(
                parent_residual_expected_manifest_path,
            ) || error(
                "parent residual finalization expected manifest path differs",
            )
            !ispath(parent_residual_expected_results_path) ||
                error(
                    "finalize-only parent residual has a results artifact; " *
                    "the parent run is already finalized",
                )
            !ispath(parent_residual_expected_manifest_path) ||
                error(
                    "finalize-only parent residual has a finalization manifest; " *
                    "the parent run is already finalized or corrupt",
                )
            residual_teardown_raw = required_property(
                residual_finalization,
                :team_teardown,
                "parent residual finalization record",
            )
            String(required_property(
                residual_teardown_raw,
                :kind,
                "parent residual finalization team teardown",
            )) == "team_teardown" ||
                error(
                    "parent residual finalization team teardown kind differs",
                )
            Int(required_property(
                residual_teardown_raw,
                :update,
                "parent residual finalization team teardown",
            )) == expected_updates ||
                error(
                    "parent residual finalization team teardown update differs",
                )
            require_canonical_equal(
                file_artifact_record(
                    residual_teardown_raw,
                    "parent residual finalization team teardown",
                ),
                team_teardown_artifact,
                "parent residual finalization/verified team teardown",
            )
            require_canonical_equal(
                required_property(
                    residual_finalization,
                    :final_metrics,
                    "parent residual finalization record",
                ),
                required_property(
                    results,
                    :final,
                    "finalize-only results",
                ),
                "parent residual finalization/results final metrics",
            )
        end
        manifest_declared_paths = Set(
            normalized_declared_path(record.path)
            for record in values(manifest)
        )
        live_training_declared_paths =
            Set(normalized_declared_path.(live_training_entries))
        length(live_training_entries) == length(manifest_declared_paths) ||
            error(
                "parent checkpoint directory contains a training checkpoint " *
                "alias, extra, or omission",
            )
        live_training_declared_paths == manifest_declared_paths ||
            error(
                "parent checkpoint directory training artifact set differs " *
                "from its manifest",
            )
        manifest_real_paths = Set(
            normalized_existing_path(
                record.path,
                "parent manifest checkpoint",
            )
            for record in values(manifest)
        )
        live_training_real_paths = Set(
            normalized_existing_path(path, "parent live training checkpoint")
            for path in live_training_entries
        )
        length(manifest_real_paths) == length(manifest_declared_paths) ||
            error("parent checkpoint manifest contains a resolved-path alias")
        length(live_training_real_paths) ==
            length(live_training_entries) ||
            error(
                "parent live training checkpoints contain a resolved-path " *
                "alias",
            )
        manifest_real_paths == live_training_real_paths ||
            error(
                "parent live training checkpoint set differs from its manifest",
            )
    else
        Set(keys(manifest)) == Set(keys(verified_checkpoints)) ||
            error("checkpoint manifest update set differs from verification")
    end
    for update in keys(verified_checkpoints)
        require_canonical_equal(
            manifest[update],
            verified_checkpoints[update],
            "checkpoint manifest record at update $update",
        )
        verify_file_record(
            manifest[update],
            verified_checkpoints[update].path,
            "checkpoint manifest record at update $update",
        )
    end

    finalization_manifest_raw = required_property(
        verification,
        :finalization_manifest,
        "verification.json",
    )
    Int(required_property(
        finalization_manifest_raw,
        :update,
        "verification finalization manifest",
    )) == expected_updates ||
        error("verification finalization manifest update differs")
    finalization_manifest_artifact = file_artifact_record(
        finalization_manifest_raw,
        "verification finalization manifest",
    )
    require_same_existing_path(
        finalization_manifest_artifact.path,
        joinpath(root, "finalization_manifest.json"),
        "verification finalization manifest path",
    )
    require_same_existing_path(
        required_property(
            results_finalization,
            :manifest_path,
            "results finalization",
        ),
        finalization_manifest_artifact.path,
        "results finalization manifest path",
    )
    finalization_manifest_snapshot = read_json_object_snapshot(
        finalization_manifest_artifact.path,
        "finalization manifest",
    )
    finalization_manifest_snapshot.bytes ==
        finalization_manifest_artifact.bytes ||
        error("finalization manifest snapshot byte size differs")
    finalization_manifest_snapshot.sha256 ==
        finalization_manifest_artifact.sha256 ||
        error("finalization manifest snapshot SHA-256 differs")
    finalization_manifest = finalization_manifest_snapshot.document
    String(required_property(
        finalization_manifest,
        :format,
        "finalization manifest",
    )) == "serial-workspace-snn-finalization-manifest" ||
        error("finalization manifest format differs")
    Int(required_property(
        finalization_manifest,
        :version,
        "finalization manifest",
    )) == 1 || error("finalization manifest version differs")
    Int(required_property(
        finalization_manifest,
        :update,
        "finalization manifest",
    )) == expected_updates ||
        error("finalization manifest update differs")
    manifest_finalization_checkpoint = checkpoint_record(
        required_property(
            finalization_manifest,
            :finalization_checkpoint,
            "finalization manifest",
        ),
        "finalization manifest final checkpoint";
        allow_finalization=true,
    )
    require_canonical_equal(
        manifest_finalization_checkpoint,
        final_checkpoint,
        "finalization manifest/verification final checkpoint",
    )
    manifest_training_checkpoint = checkpoint_record(
        required_property(
            finalization_manifest,
            :training_checkpoint,
            "finalization manifest",
        ),
        "finalization manifest training checkpoint",
    )
    require_canonical_equal(
        manifest_training_checkpoint,
        results_training_checkpoint,
        "finalization manifest/results training checkpoint",
    )
    manifest_results_raw = require_artifact_semantics(
        required_property(
            finalization_manifest,
            :results,
            "finalization manifest",
        ),
        "finalization manifest results";
        kind="results",
        update=expected_updates,
    )
    manifest_results_artifact = file_artifact_record(
        manifest_results_raw,
        "finalization manifest results",
    )
    require_same_existing_path(
        manifest_results_artifact.path,
        results_path,
        "finalization manifest results path",
    )
    manifest_results_artifact.bytes == results_bytes ||
        error("finalization manifest results byte size differs")
    manifest_results_artifact.sha256 == results_sha256 ||
        error("finalization manifest results SHA-256 differs")
    manifest_teardown_raw = require_artifact_semantics(
        required_property(
            finalization_manifest,
            :team_teardown,
            "finalization manifest",
        ),
        "finalization manifest team teardown";
        kind="team_teardown",
        update=expected_updates,
    )
    manifest_teardown_artifact = file_artifact_record(
        manifest_teardown_raw,
        "finalization manifest team teardown",
    )
    require_canonical_equal(
        manifest_teardown_artifact,
        team_teardown_artifact,
        "finalization manifest/verification team teardown",
    )
    Int(required_property(
        finalization_manifest,
        :optimizer_steps_after_target,
        "finalization manifest",
    )) == 0 ||
        error("finalization manifest reports post-target optimizer steps")

    config_path = joinpath(root, "config.json")
    config_snapshot =
        read_json_object_snapshot(config_path, "config.json")
    config_file = config_snapshot.document
    config_sha256 = config_snapshot.sha256
    config_bytes = config_snapshot.bytes
    verified_config_artifact = file_artifact_record(
        required_property(
            verification,
            :config,
            "verification.json",
        ),
        "verification config",
    )
    require_same_existing_path(
        verified_config_artifact.path,
        config_path,
        "verification config path",
    )
    verified_config_artifact.sha256 == config_sha256 ||
        error("verification config SHA-256 differs")
    verified_config_artifact.bytes == config_bytes ||
        error("verification config byte size differs")
    launch_manifest_artifact = file_artifact_record(
        required_property(
            verification,
            :launch_manifest,
            "verification.json",
        ),
        "verification launch manifest",
    )
    require_same_existing_path(
        launch_manifest_artifact.path,
        launch_preview_snapshot.path,
        "verification launch manifest snapshot path",
    )
    launch_manifest_artifact.bytes == launch_preview_snapshot.bytes ||
        error("launch manifest snapshot byte size differs")
    launch_manifest_artifact.sha256 == launch_preview_snapshot.sha256 ||
        error("launch manifest snapshot SHA-256 differs")
    launch_manifest = launch_preview
    String(required_property(
        launch_manifest,
        :format,
        "launch manifest",
    )) == "serial-workspace-snn-arena-run-launch" ||
        error("launch manifest format differs")
    Int(required_property(
        launch_manifest,
        :version,
        "launch manifest",
    )) == 2 || error("launch manifest version differs")
    String(required_property(
        launch_manifest,
        :run_id,
        "launch manifest",
    )) == run_id || error("launch manifest run ID differs")
    require_same_existing_path(
        required_property(
            launch_manifest,
            :run_directory,
            "launch manifest",
        ),
        root,
        "launch manifest run directory",
    )
    Int(required_property(
        launch_manifest,
        :expected_updates,
        "launch manifest",
    )) == expected_updates ||
        error("launch manifest expected update differs")
    String(required_property(
        launch_manifest,
        :start_mode,
        "launch manifest",
    )) == launch_start_mode ||
        error("launch manifest start mode changed")
    launch_parent_raw = property_or(
        launch_manifest,
        :parent_checkpoint,
        nothing,
    )
    launch_environment = required_property(
        launch_manifest,
        :environment,
        "launch manifest",
    )
    if launch_start_mode == "scratch"
        launch_parent_raw === nothing ||
            error("scratch launch manifest unexpectedly has a parent")
        for name in (:SWSNN_RESUME_CHECKPOINT, :SWSNN_RESUME_SHA256)
            property_or(launch_environment, name, nothing) === nothing ||
                error(
                    "scratch launch environment unexpectedly contains " *
                    String(name),
                )
        end
    else
        launch_parent_raw === nothing &&
            error("$launch_start_mode launch manifest has no parent")
        require_same_existing_path(
            required_property(
                launch_parent_raw,
                :path,
                "launch parent checkpoint",
            ),
            parent_checkpoint.path,
            "launch/verification parent checkpoint path",
        )
        require_sha256(
            required_property(
                launch_parent_raw,
                :sha256,
                "launch parent checkpoint",
            ),
            "launch parent checkpoint SHA-256",
        ) == parent_checkpoint.sha256 ||
            error("launch/verification parent checkpoint SHA-256 differs")
        Int(required_property(
            launch_parent_raw,
            :update,
            "launch parent checkpoint",
        )) == parent_checkpoint.update ||
            error("launch/verification parent checkpoint update differs")
        require_same_existing_path(
            required_property(
                launch_environment,
                :SWSNN_RESUME_CHECKPOINT,
                "launch environment",
            ),
            parent_checkpoint.path,
            "launch resume checkpoint environment path",
        )
        require_sha256(
            required_property(
                launch_environment,
                :SWSNN_RESUME_SHA256,
                "launch environment",
            ),
            "launch resume checkpoint environment SHA-256",
        ) == parent_checkpoint.sha256 ||
            error(
                "launch resume checkpoint environment SHA-256 differs",
            )
    end
    launch_code_artifacts = required_property(
        launch_manifest,
        :code_artifacts,
        "launch manifest",
    )
    expected_launch_code_paths = (
        controller=joinpath(@__DIR__, "run_arena_100k_controller.ps1"),
        training=joinpath(@__DIR__, "train_arena_100k.jl"),
        verifier=joinpath(@__DIR__, "verify_arena_run.jl"),
    )
    normalized_launch_code_artifacts = NamedTuple[]
    for (name, expected_path) in pairs(expected_launch_code_paths)
        artifact = file_artifact_record(
            required_property(
                launch_code_artifacts,
                name,
                "launch code artifacts",
            ),
            "launch code artifact $(String(name))",
        )
        require_same_existing_path(
            artifact.path,
            expected_path,
            "launch code artifact $(String(name)) path",
        )
        push!(
            normalized_launch_code_artifacts,
            (;
                name=String(name),
                path=artifact.path,
                bytes=artifact.bytes,
                sha256=artifact.sha256,
            ),
        )
    end
    for (name, expected_path) in (
        :controller_script => expected_launch_code_paths.controller,
        :training_script => expected_launch_code_paths.training,
        :verifier_script => expected_launch_code_paths.verifier,
    )
        require_same_existing_path(
            required_property(
                launch_manifest,
                name,
                "launch manifest",
            ),
            expected_path,
            "launch manifest $(String(name))",
        )
    end
    require_canonical_equal(
        required_property(
            launch_preview_record,
            :code_artifacts,
            "verification launch manifest",
        ),
        normalized_launch_code_artifacts,
        "verification/launch code artifacts",
    )
    file_config = required_property(config_file, :config, "config.json")
    config_parent_raw = property_or(
        config_file,
        :parent_checkpoint,
        nothing,
    )
    if launch_start_mode == "scratch"
        config_parent_raw === nothing ||
            error("scratch config.json unexpectedly has a parent checkpoint")
    else
        config_parent_raw === nothing &&
            error("$launch_start_mode config.json has no parent checkpoint")
        config_parent_checkpoint = checkpoint_record(
            config_parent_raw,
            "config.json parent checkpoint",
        )
        require_canonical_equal(
            config_parent_checkpoint,
            parent_checkpoint,
            "config.json/verification parent checkpoint",
        )
    end
    results_config = required_property(results, :config, "results.json")
    require_canonical_equal(
        results_config,
        file_config,
        "results/config.json configuration binding",
    )
    launch_contract = required_property(
        launch_manifest,
        :expected_contract,
        "launch manifest",
    )
    config_schema = required_property(
        file_config,
        :checkpoint_schema,
        "run config",
    )
    for (name, expected) in (
        :experiment_id => required_property(
            file_config,
            :experiment_id,
            "run config",
        ),
        :checkpoint_format => required_property(
            config_schema,
            :format,
            "run checkpoint schema",
        ),
        :learning_mode => required_property(
            file_config,
            :learning_mode,
            "run config",
        ),
        :model_preset => required_property(
            file_config,
            :model_preset,
            "run config",
        ),
        :start_mode => launch_start_mode,
        :cpuset_mode => required_property(
            file_config,
            :cpuset_mode,
            "run config",
        ),
    )
        String(required_property(
            launch_contract,
            name,
            "launch expected contract",
        )) == String(expected) ||
            error(
                "launch expected contract $(String(name)) differs from config",
            )
    end
    Int(required_property(
        launch_contract,
        :checkpoint_version,
        "launch expected contract",
    )) == Int(required_property(
        config_schema,
        :version,
        "run checkpoint schema",
    )) || error("launch expected checkpoint version differs from config")
    Bool(required_property(
        launch_contract,
        :scratch,
        "launch expected contract",
    )) == (launch_start_mode == "scratch") ||
        error("launch expected scratch/start-mode flags differ")
    Bool(required_property(
        launch_contract,
        :startup_file,
        "launch expected contract",
    )) == false || error("launch expected startup-file flag is enabled")
    Bool(required_property(
        launch_contract,
        :history_file,
        "launch expected contract",
    )) == false || error("launch expected history-file flag is enabled")
    for name in (
        :state_batch,
        :active_workers,
        :eprop_reducers,
        :structural_interval,
        :checkpoint_interval,
        :log_interval,
        :maximum_hot_allocation_bytes,
    )
        Int(required_property(
            launch_contract,
            name,
            "launch expected contract",
        )) == Int(required_property(
            file_config,
            name,
            "run config",
        )) || error(
            "launch expected contract $(String(name)) differs from config",
        )
    end
    Int(required_property(
        launch_contract,
        :evaluation_states,
        "launch expected contract",
    )) == Int(required_property(
        file_config,
        :training_eval_states,
        "run config",
    )) || error("launch expected evaluation-state count differs from config")
    for name in (:learning_rate, :weight_decay, :structure_weight)
        require_numeric_match(
            required_property(
                launch_contract,
                name,
                "launch expected contract",
            ),
            required_property(file_config, name, "run config"),
            "launch expected contract $(String(name))",
        )
    end
    runtime_config = required_property(
        file_config,
        :runtime_provenance,
        "run config",
    )
    launch_project_path = required_property(
        launch_manifest,
        :project_path,
        "launch manifest",
    )
    require_same_existing_path(
        launch_project_path,
        dirname(String(required_property(
            runtime_config,
            :project_toml_path,
            "run runtime provenance",
        ))),
        "launch/runtime project path",
    )
    require_same_existing_path(
        required_property(
            launch_contract,
            :canonical_project_path,
            "launch expected contract",
        ),
        launch_project_path,
        "launch contract/project path",
    )
    require_same_existing_path(
        required_property(
            launch_manifest,
            :julia_executable,
            "launch manifest",
        ),
        required_property(
            runtime_config,
            :julia_executable_path,
            "run runtime provenance",
        ),
        "launch/runtime Julia executable",
    )
    Int(required_property(
        launch_manifest,
        :julia_threads,
        "launch manifest",
    )) == Int(required_property(
        file_config,
        :julia_threads,
        "run config",
    )) || error("launch Julia thread count differs from config")
    String.(
        required_property(
            launch_manifest,
            :julia_runtime_arguments,
            "launch manifest",
        ),
    ) == ["--startup-file=no", "--history-file=no"] ||
        error("launch Julia runtime arguments differ from production contract")

    config_executor = required_property(
        file_config,
        :executor,
        "run config",
    )
    config_eprop = required_property(
        file_config,
        :eprop,
        "run config",
    )
    launch_full_eprop = required_property(
        launch_contract,
        :full_eprop,
        "launch expected contract",
    )
    for (name, expected) in (
        :analytic_vjp => required_property(
            config_executor,
            :analytic_vjp,
            "run executor config",
        ),
        :supervised_head_vjp => required_property(
            config_executor,
            :supervised_head_vjp,
            "run executor config",
        ),
        :recurrent_credit_assignment => required_property(
            config_executor,
            :recurrent_credit_assignment,
            "run executor config",
        ),
        :edge_parameter_mode => required_property(
            config_eprop,
            :edge_parameter_mode,
            "run e-prop config",
        ),
        :node_parameter_mode => required_property(
            config_eprop,
            :node_parameter_mode,
            "run e-prop config",
        ),
        :routing_parameter_mode => required_property(
            config_eprop,
            :routing_parameter_mode,
            "run e-prop config",
        ),
        :third_factor_mode => required_property(
            config_eprop,
            :third_factor_mode,
            "run e-prop config",
        ),
        :time_order => required_property(
            config_eprop,
            :time_order,
            "run e-prop config",
        ),
    )
        require_canonical_equal(
            required_property(
                launch_full_eprop,
                name,
                "launch full e-prop contract",
            ),
            expected,
            "launch/config full e-prop $(String(name))",
        )
    end
    config_routing = required_property(
        file_config,
        :routing,
        "run config",
    )
    launch_routing = required_property(
        launch_contract,
        :routing,
        "launch expected contract",
    )
    for name in (
        :inference_selection,
        :training_selection,
        :parameter_update,
        :exploration_probability,
        :entropy_weight,
        :entropy_floor,
        :load_balance_weight,
    )
        launch_value = required_property(
            launch_routing,
            name,
            "launch routing contract",
        )
        config_value = required_property(
            config_routing,
            name,
            "run routing config",
        )
        if launch_value isa Number && config_value isa Number
            require_numeric_match(
                launch_value,
                config_value,
                "launch/config routing $(String(name))",
            )
        else
            String(launch_value) == String(config_value) ||
                error("launch/config routing $(String(name)) differs")
        end
    end
    config_structural = required_property(
        config_executor,
        :structural_learning,
        "run executor config",
    )
    launch_utility = required_property(
        launch_contract,
        :utility,
        "launch expected contract",
    )
    for (launch_name, config_name) in (
        :mode => :mode,
        :decay => :utility_decay,
        :connection_cost => :utility_connection_cost,
        :keep_fraction => :utility_keep_fraction,
        :turnover_period => :utility_turnover_period,
    )
        launch_value = required_property(
            launch_utility,
            launch_name,
            "launch utility contract",
        )
        config_value = required_property(
            config_structural,
            config_name,
            "run structural config",
        )
        if launch_value isa Number && config_value isa Number
            require_numeric_match(
                launch_value,
                config_value,
                "launch/config utility $(String(launch_name))",
            )
        else
            String(launch_value) == String(config_value) ||
                error(
                    "launch/config utility $(String(launch_name)) differs",
                )
        end
    end
    config_model = required_property(file_config, :model, "run config")
    launch_model = required_property(
        launch_contract,
        :model,
        "launch expected contract",
    )
    for name in (
        :blocks,
        :nodes,
        :candidate_synapses,
        :enabled_synapses,
        :fanout,
        :cycles,
        :workspace_capacity,
        :input_rails,
    )
        Int(required_property(
            launch_model,
            name,
            "launch model contract",
        )) == Int(required_property(
            config_model,
            name,
            "run model config",
        )) || error("launch/config model $(String(name)) differs")
    end
    for name in (:parameter_count, :candidate_width)
        Int(required_property(
            launch_model,
            name,
            "launch model contract",
        )) == Int(required_property(
            file_config,
            name,
            "run config",
        )) || error("launch/config model $(String(name)) differs")
    end
    for name in (
        :require_dataset_content_sha256,
        :require_dataset_integrity,
        :require_runtime_provenance,
        :require_source_fingerprint,
    )
        Bool(required_property(
            launch_contract,
            name,
            "launch expected contract",
        )) || error("launch expected contract disables $(String(name))")
    end
    require_same_existing_path(
        required_property(
            launch_contract,
            :dataset_path,
            "launch expected contract",
        ),
        required_property(file_config, :dataset_path, "run config"),
        "launch/config dataset path",
    )
    launch_output_root = abspath(String(required_property(
        launch_manifest,
        :output_root,
        "launch manifest",
    )))
    require_same_existing_path(
        launch_output_root,
        dirname(root),
        "launch output root/run directory parent",
    )
    require_same_existing_path(
        joinpath(launch_output_root, run_id),
        root,
        "launch output root/run ID",
    )
    launch_environment = required_property(
        launch_manifest,
        :environment,
        "launch manifest",
    )
    for (name, expected) in (
        :SWSNN_LEARNING_MODE => String(required_property(
            file_config,
            :learning_mode,
            "run config",
        )),
        :SWSNN_MODEL_PRESET => String(required_property(
            file_config,
            :model_preset,
            "run config",
        )),
        :SWSNN_CPUSET_MODE => String(required_property(
            file_config,
            :cpuset_mode,
            "run config",
        )),
        :SWSNN_RUN_ID => run_id,
        :SWSNN_START_MODE => replace(launch_start_mode, "-" => "_"),
        :SWSNN_SCRATCH =>
            launch_start_mode == "scratch" ? "true" : "false",
    )
        String(required_property(
            launch_environment,
            name,
            "launch environment",
        )) == expected ||
            error("launch environment $(String(name)) differs from config")
    end
    for (name, expected) in (
        :SWSNN_STATE_BATCH => required_property(
            file_config,
            :state_batch,
            "run config",
        ),
        :SWSNN_ACTIVE_WORKERS => required_property(
            file_config,
            :active_workers,
            "run config",
        ),
        :SWSNN_EPROP_REDUCERS => required_property(
            file_config,
            :eprop_reducers,
            "run config",
        ),
        :SWSNN_STRUCTURAL_INTERVAL => required_property(
            file_config,
            :structural_interval,
            "run config",
        ),
        :SWSNN_CHECKPOINT_INTERVAL => required_property(
            file_config,
            :checkpoint_interval,
            "run config",
        ),
        :SWSNN_LOG_INTERVAL => required_property(
            file_config,
            :log_interval,
            "run config",
        ),
        :SWSNN_EVAL_STATES => required_property(
            file_config,
            :training_eval_states,
            "run config",
        ),
        :SWSNN_MAX_HOT_ALLOCATION_BYTES => required_property(
            file_config,
            :maximum_hot_allocation_bytes,
            "run config",
        ),
        :SWSNN_MAX_UPDATES => expected_updates,
        :SWSNN_UTILITY_TURNOVER_PERIOD => required_property(
            config_structural,
            :utility_turnover_period,
            "run structural config",
        ),
    )
        parse(Int, String(required_property(
            launch_environment,
            name,
            "launch environment",
        ))) == Int(expected) ||
            error("launch environment $(String(name)) differs from config")
    end
    for (name, expected) in (
        :SWSNN_LEARNING_RATE => required_property(
            file_config,
            :learning_rate,
            "run config",
        ),
        :SWSNN_WEIGHT_DECAY => required_property(
            file_config,
            :weight_decay,
            "run config",
        ),
        :SWSNN_STRUCTURE_WEIGHT => required_property(
            file_config,
            :structure_weight,
            "run config",
        ),
        :SWSNN_UTILITY_DECAY => required_property(
            config_structural,
            :utility_decay,
            "run structural config",
        ),
        :SWSNN_UTILITY_CONNECTION_COST => required_property(
            config_structural,
            :utility_connection_cost,
            "run structural config",
        ),
        :SWSNN_UTILITY_KEEP_FRACTION => required_property(
            config_structural,
            :utility_keep_fraction,
            "run structural config",
        ),
    )
        require_numeric_match(
            parse(Float64, String(required_property(
                launch_environment,
                name,
                "launch environment",
            ))),
            expected,
            "launch environment $(String(name))",
        )
    end
    require_same_existing_path(
        required_property(
            launch_environment,
            :SWSNN_DATASET,
            "launch environment",
        ),
        required_property(file_config, :dataset_path, "run config"),
        "launch environment/config dataset path",
    )
    require_same_existing_path(
        required_property(
            launch_environment,
            :SWSNN_OUTPUT,
            "launch environment",
        ),
        launch_output_root,
        "launch environment/output root",
    )
    results_dataset_content_sha256 = require_sha256(
        required_property(
            results,
            :dataset_content_sha256,
            "results.json",
        ),
        "results dataset content SHA-256",
    )
    results_dataset_content_sha256 == require_sha256(
        required_property(
            results_config,
            :dataset_content_sha256,
            "results config",
        ),
        "results config dataset content SHA-256",
    ) || error("results/config dataset content SHA-256 differs")
    require_canonical_equal(
        required_property(
            results,
            :dataset_integrity,
            "results.json",
        ),
        required_property(
            results_config,
            :dataset_integrity,
            "results config",
        ),
        "results/config dataset integrity",
    )
    require_canonical_equal(
        required_property(
            results,
            :runtime_provenance,
            "results.json",
        ),
        required_property(
            results_config,
            :runtime_provenance,
            "results config",
        ),
        "results/config runtime provenance",
    )
    require_sha256(
        required_property(
            finalization_manifest,
            :dataset_content_sha256,
            "finalization manifest",
        ),
        "finalization manifest dataset content SHA-256",
    ) == results_dataset_content_sha256 ||
        error("finalization manifest dataset content SHA-256 differs")
    require_canonical_equal(
        required_property(
            finalization_manifest,
            :runtime_provenance,
            "finalization manifest",
        ),
        required_property(
            results_config,
            :runtime_provenance,
            "results config",
        ),
        "finalization manifest/config runtime provenance",
    )
    String(required_property(
        results_config,
        :run_id,
        "results config",
    )) == run_id || error("results config run ID differs from verification")
    replace(
        String(required_property(
            results_config,
            :start_mode,
            "results config",
        )),
        "_" => "-",
    ) == launch_start_mode ||
        error("results config start mode differs from launch manifest")
    expected_teacher_states = Int(required_property(
        verification,
        :expected_teacher_states,
        "verification.json",
    ))
    state_batch = Int(required_property(
        results_config,
        :state_batch,
        "results config",
    ))
    expected_teacher_states == expected_updates * state_batch ||
        error("verification expected teacher-state count differs")
    Int(required_property(
        fixed_panel_recomputation,
        :panel_states,
        "verification fixed-panel recomputation",
    )) == Int(required_property(
        results_config,
        :training_eval_states,
        "results config",
    )) || error("verification fixed-panel state count differs from config")
    require_sha256(
        required_property(
            fixed_panel_recomputation,
            :panel_rows_sha256,
            "verification fixed-panel recomputation",
        ),
        "verification fixed-panel rows SHA-256",
    ) == lowercase(String(required_property(
        results_config,
        :training_panel_rows_sha256,
        "results config",
    ))) || error("verification fixed-panel rows differ from config")
    checkpoint_interval = Int(required_property(
        policy,
        :interval,
        "verification checkpoint policy",
    ))
    checkpoint_interval >= 1 ||
        error("verification checkpoint interval must be positive")
    checkpoint_interval == Int(required_property(
        results_config,
        :checkpoint_interval,
        "results config",
    )) || error(
        "verification checkpoint interval differs from results config",
    )
    segment_start_update = Int(required_property(
        policy,
        :segment_start_update,
        "verification checkpoint policy",
    ))
    0 <= segment_start_update < expected_updates ||
        error("verification checkpoint segment start is out of range")
    if launch_start_mode == "scratch"
        segment_start_update == 0 ||
            error("scratch verification segment must start at zero")
    elseif launch_start_mode == "resume"
        segment_start_update == parent_checkpoint.update ||
            error(
                "resume verification segment start differs from parent update",
            )
    end
    expected_checkpoint_updates = Int[]
    results_scratch = Bool(required_property(
        results_config,
        :scratch,
        "results config",
    ))
    results_scratch == (launch_start_mode == "scratch") ||
        error("results config scratch/start-mode flags differ")
    results_scratch && push!(expected_checkpoint_updates, 0)
    for update in (segment_start_update + 1):expected_updates
        update % checkpoint_interval == 0 &&
            push!(expected_checkpoint_updates, update)
    end
    expected_updates in expected_checkpoint_updates ||
        push!(expected_checkpoint_updates, expected_updates)
    sort!(unique!(expected_checkpoint_updates))
    observed_checkpoint_updates =
        sort!(collect(keys(verified_checkpoints)))
    if is_finalize_only
        observed_checkpoint_updates == [expected_updates] ||
            error(
                "finalize-only verification checkpoint set must contain " *
                "exactly the target parent checkpoint",
            )
    else
        observed_checkpoint_updates == expected_checkpoint_updates ||
            error(
                "verification checkpoint updates differ from configured policy: " *
                "observed=$observed_checkpoint_updates " *
                "expected=$expected_checkpoint_updates",
            )
    end

    return (;
        run_dir=root,
        run_id,
        launch_start_mode,
        is_finalize_only,
        segment_start_update,
        checkpoint_interval,
        expected_updates,
        checkpoint=final_checkpoint,
        checkpoints=verified_checkpoints,
        training_checkpoint=results_training_checkpoint,
        manifest_path,
        manifest_sha256,
        manifest_bytes,
        manifest_records=manifest,
        parent_residual_finalization_record,
        parent_residual_finalization_checkpoint,
        parent_residual_expected_results_path,
        parent_residual_expected_manifest_path,
        finalization_manifest_artifact,
        finalization_manifest,
        results_path,
        results_sha256,
        results_bytes,
        results,
        trace_artifact,
        team_teardown_artifact,
        parent_checkpoint,
        config_path,
        config_sha256,
        config_bytes,
        config=file_config,
        launch_manifest_artifact,
        launch_manifest,
        launch_output_root,
        launch_code_artifacts=normalized_launch_code_artifacts,
        fixed_panel_recomputation,
        verifier_runtime,
        verification_path=abspath(verification_path),
        verification_sha256,
        verification_bytes,
        verification,
    )
end

function load_analysis_checkpoint(
    path,
    expected_sha256;
    allow_legacy_provenance::Bool=false,
)
    checkpoint_path = abspath(path)
    isfile(checkpoint_path) ||
        error("checkpoint does not exist: $checkpoint_path")
    checkpoint_bytes = read(checkpoint_path)
    actual_sha256 = bytes2hex(sha256(checkpoint_bytes))
    isempty(expected_sha256) ||
        actual_sha256 == expected_sha256 ||
        error(
            "checkpoint SHA-256 differs: expected $expected_sha256, " *
            "got $actual_sha256",
        )
    saved = JLD2.jldopen(checkpoint_bytes, "r") do file
        Dict(
            String(name) => file[name]
            for name in keys(file)
        )
    end
    if haskey(saved, "payload")
        payload = saved["payload"]
        format = property_or(payload, :format, nothing)
        format == ARENA_CHECKPOINT_FORMAT || error(
            "unsupported checkpoint payload format $(repr(format))",
        )
        version = Int(property_or(payload, :version, -1))
        production_schema =
            version == PRODUCTION_ARENA_CHECKPOINT_VERSION
        if !production_schema
            version in LEGACY_ARENA_CHECKPOINT_VERSIONS ||
                error("unsupported arena checkpoint version $version")
            allow_legacy_provenance || error(
                "arena checkpoint version $version is legacy; pass " *
                "--allow-legacy-provenance only for a non-production report",
            )
        end
        parameters = property_or(payload, :parameters)
        parameters === nothing &&
            error("checkpoint payload has no parameters")
        update = Int(property_or(payload, :update, -1))
        update >= 0 || error("checkpoint payload has no valid update")
        optimizer = property_or(payload, :optimizer, nothing)
        optimizer === nothing &&
            error("checkpoint payload has no optimizer state")
        optimizer_step = Int(property_or(optimizer, :step, -1))
        optimizer_step == update || error(
            "checkpoint update $update differs from optimizer step " *
            "$optimizer_step",
        )
        if production_schema
            checkpoint_kind = required_property(
                payload,
                :checkpoint_kind,
                "production checkpoint payload",
            )
            String(checkpoint_kind) in (
                "training",
                "final",
                "finalization",
            ) ||
                error("production checkpoint kind is unsupported")
            payload_dataset_content_sha256 = require_sha256(
                required_property(
                    payload,
                    :dataset_content_sha256,
                    "production checkpoint payload",
                ),
                "payload dataset content SHA-256",
            )
            payload_dataset_integrity = required_property(
                payload,
                :dataset_integrity,
                "production checkpoint payload",
            )
            payload_runtime_provenance = required_property(
                payload,
                :runtime_provenance,
                "production checkpoint payload",
            )
        else
            checkpoint_kind =
                property_or(payload, :checkpoint_kind, nothing)
            payload_dataset_content_sha256 =
                property_or(payload, :dataset_content_sha256, nothing)
            payload_dataset_integrity =
                property_or(payload, :dataset_integrity, nothing)
            payload_runtime_provenance =
                property_or(payload, :runtime_provenance, nothing)
        end
        payload_finalization =
            property_or(payload, :finalization, nothing)
        payload_parent_checkpoint =
            property_or(payload, :parent_checkpoint, nothing)
        return (;
            path=checkpoint_path,
            sha256=actual_sha256,
            bytes=length(checkpoint_bytes),
            parameters,
            states=NamedTuple(),
            config=property_or(payload, :config, nothing),
            optimizer,
            update,
            checkpoint_format=String(format),
            checkpoint_version=version,
            checkpoint_kind,
            payload_dataset_content_sha256,
            payload_dataset_integrity,
            payload_runtime_provenance,
            payload_finalization,
            payload_parent_checkpoint,
            trainer_state=property_or(payload, :trainer_state, nothing),
            sampler_state=property_or(payload, :sampler_state, nothing),
            initial_parameters=
                property_or(payload, :initial_parameters, nothing),
            initial_metrics=property_or(payload, :initial_metrics, nothing),
            progress=property_or(payload, :progress, nothing),
            persistent_team_warmup=
                property_or(payload, :persistent_team_warmup, nothing),
            segment_state=property_or(payload, :segment_state, nothing),
            last_training_dynamics=
                property_or(payload, :last_training_dynamics, nothing),
            synapse_utility=
                property_or(payload, :synapse_utility, nothing),
            utility_updates=
                property_or(payload, :utility_updates, nothing),
            total_structural_flips=
                property_or(payload, :total_structural_flips, nothing),
            schema="arena_payload",
            production_schema,
        )
    end
    allow_legacy_provenance || error(
        "root-ps checkpoints have no production arena provenance; pass " *
        "--allow-legacy-provenance only for a non-production report",
    )
    haskey(saved, "ps") ||
        error("checkpoint contains neither payload nor ps")
    root_update = Int(get(saved, "update", -1))
    root_update >= 0 || error("root-ps checkpoint has no valid update")
    return (;
        path=checkpoint_path,
        sha256=actual_sha256,
        bytes=length(checkpoint_bytes),
        parameters=saved["ps"],
        states=get(saved, "st", NamedTuple()),
        config=get(saved, "config", NamedTuple()),
        optimizer=nothing,
        update=root_update,
        checkpoint_format=nothing,
        checkpoint_version=nothing,
        checkpoint_kind=nothing,
        payload_dataset_content_sha256=nothing,
        payload_dataset_integrity=nothing,
        payload_runtime_provenance=nothing,
        payload_finalization=nothing,
        payload_parent_checkpoint=nothing,
        initial_parameters=nothing,
        synapse_utility=nothing,
        utility_updates=nothing,
        total_structural_flips=nothing,
        schema="root_ps",
        production_schema=false,
    )
end

function validate_parameter_shapes(model, parameters)
    nodes = model.blocks * model.node_dim
    hidden = model.hidden
    expected = (
        input_gain=(nodes,),
        input_bias=(nodes,),
        query_weight=(model.node_dim, SerialWorkspaceSNN.INPUT_RAILS),
        workspace_key=(model.node_dim, model.blocks),
        feedback_gain=(model.node_dim, model.blocks),
        leak_logits=(nodes,),
        threshold_logits=(nodes,),
        synapse_weight=(nodes, model.fanout),
        gate_logits=(nodes, model.fanout),
        delay_logits=(nodes, model.fanout),
        workspace_decay_logit=(1,),
        head_bias=(hidden,),
        output_weight=(22, hidden),
        output_bias=(22,),
    )
    for (name, expected_shape) in pairs(expected)
        hasproperty(parameters, name) ||
            error("checkpoint parameters have no $name")
        actual_shape = size(getproperty(parameters, name))
        actual_shape == expected_shape || error(
            "parameter $name shape $actual_shape differs from " *
            "$expected_shape",
        )
    end
    head_width = size(parameters.head_weight, 2)
    size(parameters.head_weight, 1) == hidden || error(
        "head hidden width $(size(parameters.head_weight, 1)) differs from " *
        "preset hidden width $hidden",
    )
    head_width in (2 * model.node_dim, 3 * model.node_dim) || error(
        "head input width $head_width is incompatible with node_dim " *
        "$(model.node_dim)",
    )
    return model
end

function topology_matches(model, blocks, node_dim, fanout, cycles, workspace_k, hidden)
    return model.blocks == blocks &&
        model.node_dim == node_dim &&
        model.fanout == fanout &&
        model.cycles == cycles &&
        model.workspace_k == workspace_k &&
        model.hidden == hidden
end

function require_checkpoint_state_equal(actual, expected, location)
    for name in EXACT_CHECKPOINT_STATE_FIELDS
        isequal(
            getproperty(actual, name),
            getproperty(expected, name),
        ) || error("$location differs for $(String(name))")
    end
    return true
end

function require_lineage_config_equal(actual, expected, location)
    for name in LINEAGE_CONFIG_FIELDS
        require_canonical_equal(
            required_property(actual, name, "$location actual config"),
            required_property(expected, name, "$location expected config"),
            "$location $(String(name))",
        )
    end
    return true
end

function expected_lineage_manifest_updates(
    owner_config,
    owner_scratch,
    segment_start,
    branch_update,
    manifest,
)
    checkpoint_interval = Int(required_property(
        owner_config,
        :checkpoint_interval,
        "training lineage config",
    ))
    checkpoint_interval >= 1 ||
        error("training lineage checkpoint interval is not positive")
    maximum_updates = Int(required_property(
        owner_config,
        :maximum_updates,
        "training lineage config",
    ))
    maximum_updates >= branch_update ||
        error(
            "training lineage branch update exceeds configured maximum",
        )
    observed_updates = sort!(collect(keys(manifest)))
    isempty(observed_updates) &&
        error("training lineage checkpoint manifest is empty")
    all(update -> 0 <= update <= maximum_updates, observed_updates) ||
        error(
            "training lineage manifest update is outside the configured " *
            "range",
        )
    if owner_scratch
        segment_start == 0 ||
            error("scratch lineage segment does not start at zero")
        first(observed_updates) == 0 ||
            error("scratch lineage manifest does not start at update zero")
    else
        all(update -> update > segment_start, observed_updates) ||
            error(
                "resume lineage manifest contains an update at or before " *
                "its segment start",
            )
    end
    for update in observed_updates
        (
            owner_scratch && update == 0 ||
            update % checkpoint_interval == 0 ||
            update == maximum_updates
        ) || error(
            "training lineage manifest update $update violates checkpoint " *
            "cadence",
        )
    end
    maximum_observed_update = last(observed_updates)
    expected_updates = Int[]
    owner_scratch && push!(expected_updates, 0)
    for update in (segment_start + 1):maximum_observed_update
        update % checkpoint_interval == 0 &&
            push!(expected_updates, update)
    end
    maximum_observed_update == maximum_updates &&
        push!(expected_updates, maximum_updates)
    sort!(unique!(expected_updates))
    observed_updates == expected_updates ||
        error(
            "training lineage checkpoint manifest cadence is incomplete: " *
            "observed=$observed_updates expected=$expected_updates",
        )
    haskey(manifest, branch_update) ||
        error(
            "training lineage manifest does not contain branch update " *
            string(branch_update),
        )
    expected_through_branch = filter(
        update -> update <= branch_update,
        expected_updates,
    )
    observed_through_branch = filter(
        update -> update <= branch_update,
        observed_updates,
    )
    observed_through_branch == expected_through_branch ||
        error(
            "training lineage manifest cadence is incomplete through branch " *
            "update $branch_update",
        )
    return (; checkpoint_interval, maximum_updates, observed_updates)
end

function bind_lineage_checkpoint_directory(
    owner_checkpoint_dir,
    owner_config,
    owner_scratch,
    owner_parent_record,
    segment_start,
    manifest,
    branch_checkpoint,
)
    entries = readdir(owner_checkpoint_dir; join=true)
    all(isfile, entries) ||
        error(
            "training lineage checkpoint directory contains a non-file entry",
        )
    all(path -> !islink(path), entries) ||
        error(
            "training lineage checkpoint directory contains a symbolic-link " *
            "or reparse entry",
        )
    owner_checkpoint_real_path = normalized_existing_path(
        owner_checkpoint_dir,
        "training lineage checkpoint directory",
    )
    for path in entries
        dirname(normalized_existing_path(
            path,
            "training lineage checkpoint-directory entry",
        )) == owner_checkpoint_real_path ||
            error(
                "training lineage checkpoint-directory entry resolves " *
                "outside its owner directory",
            )
    end
    training_entries = Dict{Int,String}()
    finalization_entries = Dict{Int,String}()
    for path in entries
        training_match = match(
            CHECKPOINT_FILENAME_PATTERN,
            basename(path),
        )
        finalization_match = match(
            FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
            basename(path),
        )
        if training_match !== nothing
            update = parse(Int, only(training_match.captures))
            haskey(training_entries, update) &&
                error(
                    "training lineage checkpoint directory duplicates " *
                    "training update $update",
                )
            training_entries[update] = path
        elseif finalization_match !== nothing
            update = parse(Int, only(finalization_match.captures))
            haskey(finalization_entries, update) &&
                error(
                    "training lineage checkpoint directory duplicates " *
                    "finalization update $update",
                )
            finalization_entries[update] = path
        else
            error(
                "training lineage checkpoint directory contains an " *
                "unmanifested extra, partial, latest, or aliased artifact: " *
                basename(path),
            )
        end
    end
    Set(keys(training_entries)) == Set(keys(manifest)) ||
        error(
            "training lineage checkpoint directory and manifest training " *
            "updates differ",
        )
    expected_training_paths = Set(
        normalized_declared_path(record.path)
        for record in values(manifest)
    )
    live_training_paths = Set(
        normalized_declared_path(path)
        for path in values(training_entries)
    )
    length(expected_training_paths) == length(manifest) ||
        error(
            "training lineage manifest contains a declared-path alias",
        )
    length(live_training_paths) == length(training_entries) ||
        error(
            "training lineage checkpoint directory contains a declared-path " *
            "alias",
        )
    live_training_paths == expected_training_paths ||
        error(
            "training lineage checkpoint directory and manifest paths differ",
        )
    expected_training_real_paths = Set(
        normalized_existing_path(
            record.path,
            "training lineage manifest checkpoint",
        )
        for record in values(manifest)
    )
    all(
        record -> dirname(normalized_existing_path(
            record.path,
            "training lineage manifest checkpoint",
        )) == owner_checkpoint_real_path,
        values(manifest),
    ) || error(
        "training lineage manifest checkpoint resolves outside its owner " *
        "directory",
    )
    live_training_real_paths = Set(
        normalized_existing_path(
            path,
            "training lineage live checkpoint",
        )
        for path in values(training_entries)
    )
    length(expected_training_real_paths) == length(manifest) ||
        error(
            "training lineage manifest contains a resolved-path alias",
        )
    length(live_training_real_paths) == length(training_entries) ||
        error(
            "training lineage checkpoint directory contains a resolved-path " *
            "alias",
        )
    live_training_real_paths == expected_training_real_paths ||
        error(
            "training lineage live/manifest checkpoint artifact sets differ",
        )

    directory_artifacts = NamedTuple[]
    for update in sort!(collect(keys(manifest)))
        manifest_record = manifest[update]
        verify_file_record(
            manifest_record,
            manifest_record.path,
            "training lineage manifest checkpoint $update",
        )
        filename_update = only(
            key
            for (key, path) in training_entries
            if normalized_declared_path(path) ==
                normalized_declared_path(manifest_record.path)
        )
        filename_update == update ||
            error(
                "training lineage manifest update differs from checkpoint " *
                "filename",
            )
        payload =
            normalized_existing_path(
                manifest_record.path,
                "training lineage manifest payload",
            ) == normalized_existing_path(
                branch_checkpoint.path,
                "training lineage branch payload",
            ) ?
            branch_checkpoint :
            load_analysis_checkpoint(
                manifest_record.path,
                manifest_record.sha256,
            )
        payload.production_schema ||
            error(
                "training lineage manifest contains a non-production payload",
            )
        String(payload.checkpoint_kind) == "training" ||
            error(
                "training lineage manifest contains a non-training payload",
            )
        payload.update == update ||
            error(
                "training lineage manifest record/payload updates differ",
            )
        require_canonical_equal(
            payload.config,
            owner_config,
            "training lineage manifest payload/config.json config",
        )
        payload_parent = payload.payload_parent_checkpoint
        if owner_scratch
            payload_parent === nothing ||
                error(
                    "scratch lineage manifest payload unexpectedly has a " *
                    "parent checkpoint",
                )
        else
            payload_parent === nothing &&
                error(
                    "resume lineage manifest payload has no parent checkpoint",
                )
            require_canonical_equal(
                checkpoint_record(
                    payload_parent,
                    "training lineage manifest payload parent checkpoint",
                ),
                owner_parent_record,
                "training lineage manifest payload/config parent checkpoint",
            )
        end
        payload_segment = payload.segment_state
        payload_segment === nothing &&
            error(
                "training lineage manifest payload has no segment state",
            )
        Int(required_property(
            payload_segment,
            :start_update,
            "training lineage manifest payload segment state",
        )) == segment_start ||
            error(
                "training lineage manifest payload segment start differs",
            )
        Int(required_property(
            payload_segment,
            :updates,
            "training lineage manifest payload segment state",
        )) == update - segment_start ||
            error(
                "training lineage manifest payload segment update count " *
                "differs",
            )
        payload_overall_seconds = Float64(required_property(
            payload_segment,
            :overall_seconds,
            "training lineage manifest payload segment state",
        ))
        isfinite(payload_overall_seconds) &&
            payload_overall_seconds >= 0.0 ||
            error(
                "training lineage manifest payload segment elapsed time is " *
                "invalid",
            )
        push!(directory_artifacts, manifest_record)
    end

    length(finalization_entries) <= 1 ||
        error(
            "training lineage checkpoint directory contains more than one " *
            "finalization checkpoint",
        )
    for (update, path) in finalization_entries
        haskey(manifest, update) ||
            error(
                "training lineage finalization checkpoint has no training " *
                "manifest record at update $update",
            )
        update == maximum(keys(manifest)) ||
            error(
                "training lineage finalization checkpoint is not for the " *
                "latest manifested update",
            )
        update == Int(required_property(
            owner_config,
            :maximum_updates,
            "training lineage config",
        )) || error(
            "training lineage finalization checkpoint is not for the " *
            "configured terminal update",
        )
        artifact = checkpoint_record(
            (;
                kind="finalization",
                path,
                bytes=filesize(path),
                sha256=file_sha256(path),
                update,
            ),
            "training lineage finalization checkpoint";
            allow_finalization=true,
        )
        payload = load_analysis_checkpoint(path, artifact.sha256)
        payload.production_schema ||
            error(
                "training lineage finalization checkpoint is not production " *
                "schema",
            )
        String(payload.checkpoint_kind) == "finalization" ||
            error(
                "training lineage finalization payload kind differs",
            )
        payload.update == update ||
            error(
                "training lineage finalization filename/payload updates differ",
            )
        require_canonical_equal(
            payload.config,
            owner_config,
            "training lineage finalization/config.json config",
        )
        payload_parent = checkpoint_record(
            payload.payload_parent_checkpoint,
            "training lineage finalization parent checkpoint",
        )
        require_canonical_equal(
            payload_parent,
            manifest[update],
            "training lineage finalization/manifest training checkpoint",
        )
        finalization_record = payload.payload_finalization
        finalization_record === nothing &&
            error(
                "training lineage finalization payload has no finalization " *
                "record",
            )
        String(required_property(
            finalization_record,
            :status,
            "training lineage finalization record",
        )) == "finalization_checkpoint_complete" ||
            error(
                "training lineage finalization record is incomplete",
            )
        Int(required_property(
            finalization_record,
            :optimizer_steps_after_target,
            "training lineage finalization record",
        )) == 0 ||
            error(
                "training lineage finalization record reports post-target " *
                "optimizer steps",
            )
        require_canonical_equal(
            checkpoint_record(
                required_property(
                    finalization_record,
                    :training_checkpoint,
                    "training lineage finalization record",
                ),
                "training lineage finalization record training checkpoint",
            ),
            manifest[update],
            "training lineage finalization record/manifest training checkpoint",
        )
        training_payload =
            normalized_existing_path(
                manifest[update].path,
                "training lineage finalization parent payload",
            ) == normalized_existing_path(
                branch_checkpoint.path,
                "training lineage branch payload",
            ) ?
            branch_checkpoint :
            load_analysis_checkpoint(
                manifest[update].path,
                manifest[update].sha256,
            )
        require_checkpoint_state_equal(
            payload,
            training_payload,
            "training lineage finalization/training checkpoint state",
        )
        push!(directory_artifacts, artifact)
    end
    declared_paths = Set(normalized_declared_path.(entries))
    resolved_paths = Set(
        normalized_existing_path(
            path,
            "training lineage checkpoint-directory entry",
        )
        for path in entries
    )
    length(declared_paths) == length(entries) ||
        error(
            "training lineage checkpoint directory contains a declared-path " *
            "alias",
        )
    length(resolved_paths) == length(entries) ||
        error(
            "training lineage checkpoint directory contains a resolved-path " *
            "alias",
        )
    return (;
        path=owner_checkpoint_dir,
        declared_paths,
        resolved_paths,
        artifacts=directory_artifacts,
    )
end

function verify_lineage_checkpoint_directory_snapshot!(
    snapshot,
    location,
)
    entries = readdir(snapshot.path; join=true)
    all(isfile, entries) ||
        error("$location contains a non-file entry")
    all(path -> !islink(path), entries) ||
        error("$location contains a symbolic-link or reparse entry")
    owner_checkpoint_real_path =
        normalized_existing_path(snapshot.path, "$location owner directory")
    all(
        path -> dirname(normalized_existing_path(
            path,
            "$location entry",
        )) == owner_checkpoint_real_path,
        entries,
    ) || error("$location entry resolves outside its owner directory")
    declared_paths = Set(normalized_declared_path.(entries))
    resolved_paths = Set(
        normalized_existing_path(path, "$location entry")
        for path in entries
    )
    length(declared_paths) == length(entries) ||
        error("$location contains a declared-path alias")
    length(resolved_paths) == length(entries) ||
        error("$location contains a resolved-path alias")
    declared_paths == snapshot.declared_paths ||
        error("$location declared artifact set changed")
    resolved_paths == snapshot.resolved_paths ||
        error("$location resolved artifact set changed")
    for artifact in snapshot.artifacts
        verify_file_record(
            artifact,
            artifact.path,
            "$location bound artifact",
        )
    end
    return true
end

function bind_training_lineage(target_checkpoint, target_record)
    entries = NamedTuple[]
    seen_paths = Set{String}()
    checkpoint = target_checkpoint
    record = target_record
    while true
        checkpoint.production_schema ||
            error("training lineage contains a non-production checkpoint")
        String(checkpoint.checkpoint_kind) == "training" ||
            error("training lineage contains a non-training checkpoint")
        checkpoint.update == record.update ||
            error("training lineage record/payload updates differ")
        checkpoint.bytes == record.bytes ||
            error("training lineage record/payload byte sizes differ")
        checkpoint.sha256 == record.sha256 ||
            error("training lineage record/payload SHA-256 values differ")
        require_same_existing_path(
            checkpoint.path,
            record.path,
            "training lineage record/payload path",
        )
        verify_file_record(
            record,
            record.path,
            "training lineage checkpoint",
        )
        normalized_checkpoint_path = normalized_existing_path(
            checkpoint.path,
            "training lineage checkpoint",
        )
        normalized_checkpoint_path in seen_paths &&
            error("training checkpoint lineage contains a cycle")
        push!(seen_paths, normalized_checkpoint_path)
        length(seen_paths) <= 1024 ||
            error("training checkpoint lineage exceeds 1024 checkpoints")

        owner_checkpoint_dir = dirname(checkpoint.path)
        owner_run_dir = dirname(owner_checkpoint_dir)
        require_same_existing_path(
            owner_checkpoint_dir,
            joinpath(owner_run_dir, "checkpoints"),
            "training lineage checkpoint directory",
        )
        config_snapshot = read_json_object_snapshot(
            joinpath(owner_run_dir, "config.json"),
            "training lineage config.json",
        )
        config_document = config_snapshot.document
        owner_config = required_property(
            config_document,
            :config,
            "training lineage config.json",
        )
        require_canonical_equal(
            owner_config,
            checkpoint.config,
            "training lineage config.json/checkpoint config",
        )
        owner_run_id = String(required_property(
            owner_config,
            :run_id,
            "training lineage config",
        ))
        (
            Sys.iswindows() ?
            lowercase(owner_run_id) == lowercase(basename(owner_run_dir)) :
            owner_run_id == basename(owner_run_dir)
        ) || error(
            "training lineage config run ID differs from its run directory",
        )
        owner_start_mode = replace(
            String(required_property(
                owner_config,
                :start_mode,
                "training lineage config",
            )),
            "_" => "-",
        )
        owner_start_mode in ("scratch", "resume") ||
            error("training lineage start mode is unsupported")
        owner_scratch = Bool(required_property(
            owner_config,
            :scratch,
            "training lineage config",
        ))
        owner_scratch == (owner_start_mode == "scratch") ||
            error("training lineage scratch/start-mode flags differ")
        segment_state = checkpoint.segment_state
        segment_state === nothing &&
            error("training lineage checkpoint has no segment state")
        segment_start = Int(required_property(
            segment_state,
            :start_update,
            "training lineage segment state",
        ))
        segment_updates = Int(required_property(
            segment_state,
            :updates,
            "training lineage segment state",
        ))
        segment_updates >= 0 ||
            error("training lineage segment update count is negative")
        segment_start + segment_updates == checkpoint.update ||
            error("training lineage segment does not end at checkpoint update")
        parent_config_raw = property_or(
            config_document,
            :parent_checkpoint,
            nothing,
        )
        owner_parent_record = if owner_scratch
            parent_config_raw === nothing ||
                error("scratch lineage config unexpectedly has a parent")
            nothing
        else
            parent_config_raw === nothing &&
                error("resume lineage config has no parent checkpoint")
            checkpoint_record(
                parent_config_raw,
                "training lineage config parent checkpoint",
            )
        end

        manifest_snapshot = checkpoint_manifest_snapshot(
            joinpath(owner_run_dir, "checkpoint_manifest.jsonl"),
        )
        manifest = manifest_snapshot.records
        haskey(manifest, checkpoint.update) ||
            error(
                "training lineage manifest does not contain checkpoint update " *
                string(checkpoint.update),
            )
        require_canonical_equal(
            manifest[checkpoint.update],
            record,
            "training lineage manifest/checkpoint record",
        )
        manifest_contract = expected_lineage_manifest_updates(
            owner_config,
            owner_scratch,
            segment_start,
            checkpoint.update,
            manifest,
        )
        checkpoint_directory_snapshot =
            bind_lineage_checkpoint_directory(
                owner_checkpoint_dir,
                owner_config,
                owner_scratch,
                owner_parent_record,
                segment_start,
                manifest,
                checkpoint,
            )
        parent_payload_raw = checkpoint.payload_parent_checkpoint
        parent_record = nothing
        if owner_scratch
            parent_payload_raw === nothing ||
                error("scratch lineage payload unexpectedly has a parent")
            segment_start == 0 ||
                error("scratch lineage segment does not start at zero")
        else
            parent_payload_raw === nothing &&
                error("resume lineage payload has no parent checkpoint")
            config_parent_record = owner_parent_record
            payload_parent_record = checkpoint_record(
                parent_payload_raw,
                "training lineage payload parent checkpoint",
            )
            require_canonical_equal(
                payload_parent_record,
                config_parent_record,
                "training lineage config/payload parent checkpoint",
            )
            verify_file_record(
                config_parent_record,
                config_parent_record.path,
                "training lineage parent checkpoint",
            )
            parent_run_dir = dirname(dirname(config_parent_record.path))
            require_same_existing_path(
                dirname(config_parent_record.path),
                joinpath(parent_run_dir, "checkpoints"),
                "training lineage parent checkpoint directory",
            )
            !path_is_within(parent_run_dir, owner_run_dir) &&
                !path_is_within(owner_run_dir, parent_run_dir) ||
                error(
                    "training lineage parent and child run directories must " *
                    "be disjoint",
                )
            config_parent_record.update < checkpoint.update ||
                error(
                    "training lineage parent update is not earlier than child",
                )
            segment_start == config_parent_record.update ||
                error(
                    "resume lineage segment start differs from parent update",
                )
            parent_record = config_parent_record
        end
        push!(entries, (;
            checkpoint,
            record,
            run_dir=owner_run_dir,
            run_id=owner_run_id,
            start_mode=owner_start_mode,
            scratch=owner_scratch,
            segment_start_update=segment_start,
            config_snapshot,
            manifest_snapshot,
            manifest_contract,
            checkpoint_directory_snapshot,
            parent_checkpoint=parent_record,
        ))
        parent_record === nothing && break
        checkpoint = load_analysis_checkpoint(
            parent_record.path,
            parent_record.sha256,
        )
        record = parent_record
    end
    return entries
end

function infer_model(parameters, config)
    preset = property_or(
        config,
        :preset,
        property_or(config, :model_preset, nothing),
    )
    if preset !== nothing
        model = build_model(Symbol(preset))
        validate_parameter_shapes(model, parameters)
        return model, "checkpoint_preset"
    end
    topology = property_or(config, :model, nothing)
    if topology !== nothing
        blocks = Int(property_or(topology, :blocks, 0))
        nodes = Int(property_or(topology, :nodes, 0))
        fanout = Int(property_or(topology, :fanout, 0))
        cycles = Int(property_or(topology, :cycles, 0))
        workspace_k = Int(property_or(
            topology,
            :workspace_capacity,
            property_or(topology, :workspace_k, 0),
        ))
        blocks > 0 && mod(nodes, blocks) == 0 || error(
            "checkpoint topology nodes are not divisible by blocks",
        )
        node_dim = blocks > 0 ? div(nodes, blocks) : 0
        if minimum((blocks, node_dim, fanout, cycles, workspace_k)) > 0
            hidden = size(parameters.head_weight, 1)
            for known_preset in (:tiny, :small, :scaled_v2, :scaled, :large)
                known_model = try
                    build_model(known_preset)
                catch
                    continue
                end
                topology_matches(
                    known_model,
                    blocks,
                    node_dim,
                    fanout,
                    cycles,
                    workspace_k,
                    hidden,
                ) || continue
                validate_parameter_shapes(known_model, parameters)
                return known_model, "known_preset_shape_$known_preset"
            end
            routing = property_or(config, :routing, NamedTuple())
            route_temperature = property_or(
                routing,
                :route_temperature,
                property_or(config, :route_temperature, nothing),
            )
            spike_temperature = property_or(
                topology,
                :spike_temperature,
                property_or(config, :spike_temperature, nothing),
            )
            route_temperature === nothing && error(
                "custom topology has no recorded route temperature",
            )
            spike_temperature === nothing && error(
                "custom topology has no recorded spike temperature",
            )
            model = SerialWorkspaceModel(
                blocks=blocks,
                node_dim=node_dim,
                fanout=fanout,
                cycles=cycles,
                workspace_k=workspace_k,
                hidden,
                route_temperature=Float32(route_temperature),
                spike_temperature=Float32(spike_temperature),
            )
            validate_parameter_shapes(model, parameters)
            return model, "checkpoint_custom_topology"
        end
    end
    model = build_model(:scaled)
    expected_nodes = model.blocks * model.node_dim
    size(parameters.synapse_weight) ==
        (expected_nodes, model.fanout) ||
        error(
            "checkpoint has no reconstructable model topology and does not " *
            "match :scaled",
        )
    validate_parameter_shapes(model, parameters)
    return model, "scaled_shape_fallback"
end

function split_rows(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        allowed = Set((:train, :validation))
        all(split -> split in allowed, dataset.predefined_split) ||
            error("predefined split contains an unknown value")
        training = Int.(findall(==(:train), dataset.predefined_split))
        validation =
            Int.(findall(==(:validation), dataset.predefined_split))
        isempty(training) && error("predefined training split is empty")
        isempty(validation) && error("predefined validation split is empty")
        training_groups = Set(dataset.split_group_ids[training])
        validation_groups = Set(dataset.split_group_ids[validation])
        isempty(intersect(training_groups, validation_groups)) ||
            error("seed leakage across predefined splits")
        return (; training, validation, kind="manifest_predefined_seed_group")
    end
    groups = sort(unique(dataset.split_group_ids))
    length(groups) >= 2 ||
        error("at least two split groups are required")
    shuffled = shuffle(Xoshiro(SPLIT_SEED), groups)
    validation_count =
        clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    validation_groups = Set(shuffled[1:validation_count])
    validation = Int.(findall(
        group -> group in validation_groups,
        dataset.split_group_ids,
    ))
    training = Int.(findall(
        group -> !(group in validation_groups),
        dataset.split_group_ids,
    ))
    return (; training, validation, kind="deterministic_seed_group")
end

function fixed_subset(rows, count::Int, seed::UInt64)
    selected = copy(Int.(rows))
    shuffle!(Xoshiro(seed), selected)
    resize!(selected, min(count, length(selected)))
    isempty(selected) && error("requested panel is empty")
    return selected
end

function read_custom_rows(path)
    rows_path = abspath(path)
    isfile(rows_path) || error("panel rows file does not exist: $rows_path")
    extension = lowercase(splitext(rows_path)[2])
    rows = if extension == ".jld2"
        saved = JLD2.load(rows_path)
        key = if haskey(saved, "rows")
            "rows"
        elseif haskey(saved, "panel_rows")
            "panel_rows"
        elseif haskey(saved, "validation_rows")
            "validation_rows"
        else
            error("JLD2 panel file has no rows key")
        end
        Int.(saved[key])
    elseif extension == ".json"
        value = JSON3.read(read(rows_path, String))
        source = value isa AbstractVector ? value :
            property_or(value, :rows, nothing)
        source === nothing && error("JSON panel file has no rows")
        Int.(source)
    else
        tokens = split(read(rows_path, String), r"[\s,]+")
        Int.(parse.(Int, filter(!isempty, tokens)))
    end
    isempty(rows) && error("custom panel is empty")
    length(unique(rows)) == length(rows) ||
        error("custom panel contains duplicate rows")
    return rows
end

function build_panels(panel_spec, split, states, validation_states)
    normalized = lowercase(strip(panel_spec))
    result = Pair{String,Vector{Int}}[]
    if normalized in ("fixed", "training", "train")
        push!(
            result,
            "fixed_training" =>
                fixed_subset(split.training, states, TRAIN_EVAL_SEED),
        )
    elseif normalized in ("validation", "val")
        push!(
            result,
            "fixed_validation" => fixed_subset(
                split.validation,
                validation_states,
                VALIDATION_EVAL_SEED,
            ),
        )
    elseif normalized == "both"
        push!(
            result,
            "fixed_training" =>
                fixed_subset(split.training, states, TRAIN_EVAL_SEED),
        )
        push!(
            result,
            "fixed_validation" => fixed_subset(
                split.validation,
                validation_states,
                VALIDATION_EVAL_SEED,
            ),
        )
    else
        push!(result, "custom" => read_custom_rows(panel_spec))
    end
    return result
end

@inline function rms_normalize_local(values, scale::Float32=1.0f0)
    if isdefined(SerialWorkspaceSNN, :rms_normalize)
        return SerialWorkspaceSNN.rms_normalize(values, scale)
    end
    epsilon = isdefined(SerialWorkspaceSNN, :RMS_NORM_EPS) ?
        Float32(SerialWorkspaceSNN.RMS_NORM_EPS) : 1.0f-4
    inverse_rms = inv.(sqrt.(
        sum(abs2, values; dims=1) ./ Float32(size(values, 1)) .+
        epsilon,
    ))
    return scale .* values .* inverse_rms
end

@inline function query_scale()
    if isdefined(SerialWorkspaceSNN, :QUERY_NORM_SCALE)
        return Float32(SerialWorkspaceSNN.QUERY_NORM_SCALE)
    elseif isdefined(SerialWorkspaceSNN, :QUERY_RMS_SCALE)
        return Float32(SerialWorkspaceSNN.QUERY_RMS_SCALE)
    end
    return 0.50f0
end

@inline function hidden_scale()
    if isdefined(SerialWorkspaceSNN, :HIDDEN_NORM_SCALE)
        return Float32(SerialWorkspaceSNN.HIDDEN_NORM_SCALE)
    elseif isdefined(SerialWorkspaceSNN, :HIDDEN_RMS_SCALE)
        return Float32(SerialWorkspaceSNN.HIDDEN_RMS_SCALE)
    end
    return 0.75f0
end

@inline function checkpoint_workspace_decay(parameters, v2::Bool)
    if v2 && isdefined(SerialWorkspaceSNN, :bounded_workspace_decay)
        return Float32(SerialWorkspaceSNN.bounded_workspace_decay(
            parameters.workspace_decay_logit[1],
        ))
    end
    return Float32(sigmoid(parameters.workspace_decay_logit[1]))
end

function hard_topk_mask(scores::AbstractMatrix, k::Int)
    blocks, candidates = size(scores)
    mask = zeros(Float32, blocks, candidates)
    for candidate in 1:candidates
        selected =
            partialsortperm(@view(scores[:, candidate]), 1:k; rev=true)
        mask[selected, candidate] .= 1.0f0
    end
    return mask
end

function route_base_probabilities(scores, model)
    surrogate = if isdefined(
        SerialWorkspaceSNN,
        :standardized_route_probabilities,
    )
        SerialWorkspaceSNN.standardized_route_probabilities(scores, model)
    else
        block_count = Float32(size(scores, 1))
        centered = scores .- sum(scores; dims=1) ./ block_count
        inverse_std = inv.(sqrt.(
            sum(abs2, centered; dims=1) ./ block_count .+ 1.0f-4,
        ))
        logits = centered .* inverse_std ./ model.route_temperature
        shifted = logits .- maximum(logits; dims=1)
        exp.(shifted)
    end
    mass = sum(surrogate; dims=1)
    base = surrogate ./ max.(mass, eps(Float32))
    return (; base, surrogate_mass=vec(mass))
end

function routing_exploration(config)
    routing = property_or(config, :routing, nothing)
    if routing !== nothing
        nested = property_or(
            routing,
            :exploration_probability,
            property_or(routing, :routing_exploration, nothing),
        )
        nested === nothing || return (
            value=Float32(nested),
            source="checkpoint_config.routing.exploration_probability",
        )
    end
    for name in (
        :routing_exploration,
        :route_exploration,
        :routing_exploration_probability,
    )
        value = property_or(config, name, nothing)
        value === nothing || return (
            value=Float32(value),
            source="checkpoint_config.$name",
        )
    end
    for name in (
        :DEFAULT_ROUTING_EXPLORATION,
        :ROUTING_EXPLORATION,
        :ROUTING_EXPLORATION_PROBABILITY,
    )
        if isdefined(SerialWorkspaceSNN, name)
            return (
                value=Float32(getfield(SerialWorkspaceSNN, name)),
                source="SerialWorkspaceSNN.$name",
            )
        end
    end
    return (
        value=DEFAULT_ROUTING_EXPLORATION,
        source="analysis_default_0.05",
    )
end

function saturation_eprop_contract(config; production_schema::Bool)
    eprop = property_or(config, :eprop, nothing)
    if eprop === nothing
        production_schema &&
            error("production config has no e-prop contract")
        return (;
            applicable=false,
            eligibility_mode=nothing,
            signal_schedule=nothing,
            trace_decay_scale=1.0f0,
            source="legacy_default",
        )
    end
    eligibility_mode = Symbol(property_or(
        eprop,
        :eligibility_mode,
        :unknown,
    ))
    signal_schedule = Symbol(property_or(
        eprop,
        :signal_schedule,
        :unknown,
    ))
    trace_decay_scale = Float32(property_or(
        eprop,
        :trace_decay_scale,
        NaN32,
    ))
    isfinite(trace_decay_scale) &&
        0.0f0 <= trace_decay_scale <= 1.0f0 ||
        error("e-prop trace_decay_scale is invalid")
    applicable =
        eligibility_mode === :membrane &&
        signal_schedule === :terminal
    if production_schema
        eligibility_mode === :membrane ||
            error(
                "production saturation eligibility requires " *
                "eligibility_mode=:membrane",
            )
        signal_schedule === :terminal ||
            error(
                "production saturation eligibility requires " *
                "signal_schedule=:terminal",
            )
    end
    return (;
        applicable,
        eligibility_mode,
        signal_schedule,
        trace_decay_scale,
        source="checkpoint_config.eprop",
    )
end

function routing_entropy_floor(config; production_schema::Bool)
    routing = property_or(config, :routing, nothing)
    value = routing === nothing ?
        nothing : property_or(routing, :entropy_floor, nothing)
    if value === nothing
        production_schema &&
            error("production routing config has no entropy floor")
        return (;
            value=ROUTING_ENTROPY_FLOOR_FALLBACK,
            source="analysis_fallback_0.70",
        )
    end
    floor_value = Float64(value)
    0.0 <= floor_value <= 1.0 ||
        error("routing entropy floor is outside [0,1]")
    return (;
        value=floor_value,
        source="checkpoint_config.routing.entropy_floor",
    )
end

function slice_input(input, count::Int)
    return (;
        board=input.board[:, :, :, 1:count],
        candidate=input.candidate[:, :, :, 1:count],
        difference=input.difference[:, :, :, 1:count],
        aux=input.aux[:, 1:count],
        next_hold=input.next_hold[:, :, 1:count],
        local_mask=input.local_mask[:, :, :, 1:count],
    )
end

function diagnostic_dynamics(
    model,
    input,
    parameters;
    v2::Bool,
    workspace_off::Bool=false,
    synapse_off::Bool=false,
    memory_off::Bool=false,
    exploration::Float32=DEFAULT_ROUTING_EXPLORATION,
    record_saturation::Bool=false,
    trace_decay_scale::Float32=1.0f0,
)
    record_saturation &&
        (workspace_off || synapse_off || memory_off) &&
        error("saturation telemetry is only valid for the full forward")
    rails = binary_rails(input)
    nodes = model.blocks * model.node_dim
    candidates = size(rails, 2)
    seed =
        parameters.input_gain .* rails[model.feature_for_node, :] .+
        parameters.input_bias
    query_pre = parameters.query_weight * rails
    query = v2 ?
        tanh.(rms_normalize_local(query_pre, query_scale())) :
        tanh.(query_pre)
    membrane = seed
    previous_active_spikes = zeros(Float32, nodes, candidates)
    workspace = zeros(Float32, model.node_dim, candidates)
    threshold =
        0.25f0 .+ 0.75f0 .* sigmoid.(parameters.threshold_logits)
    leak = 0.45f0 .+ 0.50f0 .* sigmoid.(parameters.leak_logits)
    threshold_matrix = reshape(threshold, nodes, 1)
    leak_matrix = reshape(leak, nodes, 1)
    feedback_gain =
        reshape(parameters.feedback_gain, model.node_dim, model.blocks, 1)
    workspace_key =
        reshape(parameters.workspace_key, model.node_dim, model.blocks, 1)
    zero_message = zeros(Float32, nodes, candidates)
    zero_spikes = zeros(Float32, nodes, candidates)
    masks = Array{Float32,3}(
        undef,
        model.blocks,
        model.cycles,
        candidates,
    )
    base_probabilities = similar(masks)
    policy_probabilities = similar(masks)
    surrogate_mass =
        Matrix{Float32}(undef, model.cycles, candidates)
    scores_by_cycle = record_saturation ?
        Array{Float32,3}(
            undef,
            model.blocks,
            model.cycles,
            candidates,
        ) : nothing
    workspace_pre_tanh = record_saturation ?
        Array{Float32,3}(
            undef,
            model.node_dim,
            model.cycles,
            candidates,
        ) : nothing
    workspace_by_cycle = record_saturation ?
        similar(workspace_pre_tanh) : nothing
    eligibility_sample_indices = if record_saturation
        sample_count = min(
            SATURATION_ELIGIBILITY_EDGE_SAMPLES,
            length(parameters.synapse_weight),
        )
        unique(round.(
            Int,
            range(
                1,
                length(parameters.synapse_weight);
                length=sample_count,
            ),
        ))
    else
        Int[]
    end
    eligibility_trace = record_saturation ?
        zeros(Float32, length(eligibility_sample_indices), candidates) :
        nothing
    gate_hard = record_saturation ?
        Float32.(parameters.gate_logits .>= 0.0f0) : nothing
    delay_probability = record_saturation ?
        sigmoid.(parameters.delay_logits) : nothing
    membrane_count = 0
    membrane_margin_sum_square = 0.0
    membrane_margin_abs_lt_1 = 0
    membrane_margin_abs_lt_2 = 0
    raw_spike_count = 0
    raw_spike_counts_per_node = record_saturation ?
        zeros(Int, nodes) : nothing
    surrogate_sum = 0.0
    surrogate_sum_square = 0.0
    surrogate_lt_1e_3 = 0
    surrogate_lt_1e_4 = 0

    for cycle in 1:model.cycles
        if memory_off && cycle > 1
            # Eliminate the previous cycle's membrane from routing, spikes,
            # writes, and the next state, rather than merely setting leak=0 in
            # an update whose result would still seed the following cycle.
            membrane = seed
        end
        block_state =
            reshape(membrane, model.node_dim, model.blocks, candidates)
        query3 = reshape(query, model.node_dim, 1, candidates)
        scores = dropdims(
            sum(block_state .* workspace_key .* query3; dims=1);
            dims=1,
        )
        scores .+= 0.05f0 .* dropdims(
            sum(abs.(block_state); dims=1);
            dims=1,
        )
        record_saturation &&
            (scores_by_cycle[:, cycle, :] .= scores)
        route = route_base_probabilities(scores, model)
        block_mask = hard_topk_mask(scores, model.workspace_k)
        policy = (1.0f0 - exploration) .* route.base .+
            exploration / Float32(model.blocks)
        masks[:, cycle, :] .= block_mask
        base_probabilities[:, cycle, :] .= route.base
        policy_probabilities[:, cycle, :] .= policy
        surrogate_mass[cycle, :] .= route.surrogate_mass

        surrogate_current = nothing
        if record_saturation
            margin =
                (membrane .- threshold_matrix) ./
                Float32(model.spike_temperature)
            soft = sigmoid.(margin)
            surrogate_current =
                soft .* (1.0f0 .- soft) ./
                Float32(model.spike_temperature)
            @inbounds for candidate in axes(margin, 2)
                for node in axes(margin, 1)
                    margin_value = Float64(margin[node, candidate])
                    sensitivity =
                        Float64(surrogate_current[node, candidate])
                    membrane_count += 1
                    membrane_margin_sum_square = muladd(
                        margin_value,
                        margin_value,
                        membrane_margin_sum_square,
                    )
                    membrane_margin_abs_lt_1 +=
                        abs(margin_value) < 1.0
                    membrane_margin_abs_lt_2 +=
                        abs(margin_value) < 2.0
                    fired = margin_value >= 0.0
                    raw_spike_count += fired
                    raw_spike_counts_per_node[node] += fired
                    surrogate_sum += sensitivity
                    surrogate_sum_square = muladd(
                        sensitivity,
                        sensitivity,
                        surrogate_sum_square,
                    )
                    surrogate_lt_1e_3 += sensitivity < 1.0e-3
                    surrogate_lt_1e_4 += sensitivity < 1.0e-4
                end
            end
        end
        spikes = Float32.(membrane .>= threshold_matrix)
        node_mask = reshape(
            repeat(
                reshape(block_mask, 1, model.blocks, candidates),
                model.node_dim,
                1,
                1,
            ),
            nodes,
            candidates,
        )
        active_spikes = spikes .* node_mask
        if record_saturation
            @inbounds for (
                sample,
                linear_index,
            ) in enumerate(eligibility_sample_indices)
                source = mod1(linear_index, nodes)
                relation = div(linear_index - 1, nodes) + 1
                destination =
                    model.destination_for_source[source, relation]
                delay = delay_probability[source, relation]
                gate = gate_hard[source, relation]
                for candidate in 1:candidates
                    current = active_spikes[source, candidate]
                    previous =
                        previous_active_spikes[source, candidate]
                    delayed_pre = muladd(
                        1.0f0 - delay,
                        current,
                        delay * previous,
                    )
                    recurrence = trace_decay_scale * (
                        leak[destination] -
                        threshold[destination] *
                        surrogate_current[
                            destination,
                            candidate,
                        ]
                    )
                    eligibility_trace[sample, candidate] = muladd(
                        recurrence,
                        eligibility_trace[sample, candidate],
                        gate * delayed_pre,
                    )
                end
            end
        end
        message = synapse_off ?
            zero_message :
            vectorized_synapse_scan(
                model,
                active_spikes,
                # With both arguments equal the delay interpolation is exactly
                # current_spike, preserving instantaneous synaptic gain while
                # removing delayed access to the previous cycle.
                memory_off ? active_spikes : previous_active_spikes,
                parameters,
            )
        selected =
            reshape(block_mask, 1, model.blocks, candidates)
        write = dropdims(
            sum(block_state .* selected; dims=2);
            dims=2,
        ) ./ Float32(model.workspace_k)
        decay = memory_off ?
            0.0f0 : checkpoint_workspace_decay(parameters, v2)
        if workspace_off
            workspace = zero(workspace)
            feedback = zero_message
        else
            workspace_pre = decay .* workspace .+ write
            workspace = tanh.(workspace_pre)
            if record_saturation
                workspace_pre_tanh[:, cycle, :] .= workspace_pre
                workspace_by_cycle[:, cycle, :] .= workspace
            end
            feedback = reshape(
                feedback_gain .*
                reshape(workspace, model.node_dim, 1, candidates),
                nodes,
                candidates,
            )
        end
        retained = memory_off ?
            zero_message : leak_matrix .* membrane
        membrane = retained .+ message .+ 0.18f0 .* seed .+
            feedback .- spikes .* threshold_matrix
        previous_active_spikes =
            memory_off ? zero_spikes : active_spikes
    end

    final_blocks =
        reshape(membrane, model.node_dim, model.blocks, candidates)
    final_mask = @view masks[:, model.cycles, :]
    pooled = if v2
        dropdims(sum(
            final_blocks .*
            reshape(final_mask, 1, model.blocks, candidates);
            dims=2,
        ); dims=2) ./ Float32(model.workspace_k)
    else
        dropdims(mean(final_blocks; dims=2); dims=2)
    end
    return (;
        workspace,
        query_pre,
        query,
        pooled,
        membrane,
        masks,
        base_probabilities,
        policy_probabilities,
        surrogate_mass,
        scores_by_cycle,
        workspace_pre_tanh,
        workspace_by_cycle,
        eligibility_trace,
        eligibility_sample_indices,
        eligibility_sample_active=record_saturation ?
            Bool[
                gate_hard[linear_index]
                for linear_index in eligibility_sample_indices
            ] : nothing,
        saturation_moments=record_saturation ? (;
            membrane_count,
            membrane_margin_sum_square,
            membrane_margin_abs_lt_1,
            membrane_margin_abs_lt_2,
            raw_spike_count,
            raw_spike_counts_per_node,
            observations_per_node=model.cycles * candidates,
            surrogate_sum,
            surrogate_sum_square,
            surrogate_lt_1e_3,
            surrogate_lt_1e_4,
        ) : nothing,
        trace_decay_scale,
        route_temperature=Float32(model.route_temperature),
    )
end

function diagnostic_head(
    dynamics,
    parameters;
    v2::Bool,
    selected_pool_off::Bool=false,
)
    pooled = selected_pool_off ?
        zero(dynamics.pooled) : dynamics.pooled
    features = if v2
        vcat(
            rms_normalize_local(dynamics.workspace),
            rms_normalize_local(pooled),
        )
    else
        vcat(dynamics.workspace, dynamics.query, pooled)
    end
    hidden_pre =
        parameters.head_weight * features .+ parameters.head_bias
    hidden = v2 ?
        tanh.(rms_normalize_local(hidden_pre, hidden_scale())) :
        tanh.(hidden_pre)
    raw =
        parameters.output_weight * hidden .+ parameters.output_bias
    return (;
        q=vec(raw[1, :]),
        raw,
        features,
        hidden_pre,
        hidden,
    )
end

function state_listnet_metrics(prediction, teacher_z)
    count = length(prediction)
    centered =
        prediction .- sum(prediction) / Float32(count)
    student_z = centered ./ sqrt(
        sum(abs2, centered) / Float32(count) + 1.0f-4,
    )
    temperature = Float64(
        getfield(BeatFirstTrainingCore, :LISTNET_TEMPERATURE),
    )
    teacher_logits = Float64.(teacher_z) ./ temperature
    student_logits = Float64.(student_z) ./ temperature
    teacher_shifted = teacher_logits .- maximum(teacher_logits)
    student_shifted = student_logits .- maximum(student_logits)
    teacher_exp = exp.(teacher_shifted)
    student_exp = exp.(student_shifted)
    teacher_probability = teacher_exp ./ sum(teacher_exp)
    student_probability = student_exp ./ sum(student_exp)
    entropy = -sum(
        probability > 0.0 ?
            probability * log(probability) : 0.0
        for probability in teacher_probability
    )
    cross_entropy = -sum(
        teacher_probability .* log.(max.(student_probability, eps(Float64))),
    )
    return (;
        cross_entropy,
        teacher_entropy=entropy,
        kl=cross_entropy - entropy,
    )
end

function stable_softmax(values)
    isempty(values) && return Float64[]
    shifted = Float64.(values) .- maximum(Float64.(values))
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function exact_listnet_state_gradient(
    prediction,
    teacher_z;
    state_batch::Int=1,
)
    count = length(prediction)
    count == length(teacher_z) ||
        throw(DimensionMismatch("prediction/teacher_z length differs"))
    count >= 1 || throw(ArgumentError("ListNet state is empty"))
    state_batch >= 1 ||
        throw(ArgumentError("state_batch must be positive"))
    q = Float64.(prediction)
    q_mean = mean(q)
    centered = q .- q_mean
    scale = sqrt(mean(abs2, centered) + 1.0e-4)
    inverse_scale = inv(scale)
    student_z = centered .* inverse_scale
    temperature = Float64(
        getfield(BeatFirstTrainingCore, :LISTNET_TEMPERATURE),
    )
    teacher_probability =
        stable_softmax(Float64.(teacher_z) ./ temperature)
    student_probability =
        stable_softmax(student_z ./ temperature)
    delta_z = (
        student_probability .- teacher_probability
    ) ./ (temperature * state_batch)
    mean_delta_z = mean(delta_z)
    mean_delta_z_times_z = mean(delta_z .* student_z)
    projected_delta_z =
        delta_z .- mean_delta_z .-
        student_z .* mean_delta_z_times_z
    delta_q = projected_delta_z .* inverse_scale
    teacher_entropy = entropy(teacher_probability)
    student_entropy = entropy(student_probability)
    order = sortperm(
        teacher_probability;
        rev=true,
        alg=Base.Sort.MergeSort,
    )
    teacher_probability_gap = count >= 2 ?
        teacher_probability[order[1]] -
        teacher_probability[order[2]] : nothing
    delta_z_norm = norm(delta_z)
    projected_norm = norm(projected_delta_z)
    return (;
        q_mean,
        q_state_rms=sqrt(mean(abs2, q)),
        centered_q_rms=sqrt(mean(abs2, centered)),
        standardization_scale=scale,
        standardization_inverse_scale=inverse_scale,
        student_z,
        teacher_probability,
        student_probability,
        teacher_entropy,
        normalized_teacher_entropy=count <= 1 ?
            0.0 : teacher_entropy / log(Float64(count)),
        teacher_effective_choices=exp(teacher_entropy),
        student_entropy,
        normalized_student_entropy=count <= 1 ?
            0.0 : student_entropy / log(Float64(count)),
        candidate_count=count,
        total_variation=0.5 * sum(abs.(
            student_probability .- teacher_probability
        )),
        teacher_probability_top1=teacher_probability[order[1]],
        teacher_probability_gap,
        delta_z,
        projected_delta_z,
        delta_q,
        delta_z_norm,
        projected_delta_z_norm=projected_norm,
        delta_q_norm=norm(delta_q),
        standardization_projection_retention=
            delta_z_norm == 0.0 ? nothing :
            projected_norm / delta_z_norm,
        raw_q_gradient_gain=
            delta_z_norm == 0.0 ? nothing :
            norm(delta_q) / delta_z_norm,
    )
end

@inline function finite_cosine(left, right)
    denominator = norm(left) * norm(right)
    return denominator == 0.0 ? nothing :
        dot(left, right) / denominator
end

function stable_top_two(values)
    count = length(values)
    count >= 1 || throw(ArgumentError("top-two input is empty"))
    top1 = 1
    @inbounds for candidate in 2:count
        values[candidate] > values[top1] &&
            (top1 = candidate)
    end
    top2 = count == 1 ? top1 : (top1 == 1 ? 2 : 1)
    @inbounds for candidate in 1:count
        candidate == top1 && continue
        values[candidate] > values[top2] &&
            (top2 = candidate)
    end
    return top1, top2
end

function gradient_component_summary(components::Dict{String,Vector{Float64}})
    names = sort!(collect(keys(components)))
    norms = Dict(
        name => norm(components[name])
        for name in names
    )
    total = zeros(Float64, isempty(names) ? 0 : length(components[first(names)]))
    for name in names
        length(components[name]) == length(total) ||
            error("gradient component lengths differ")
        total .+= components[name]
    end
    pairwise = Dict{String,Any}()
    if length(names) >= 2
        for left_index in 1:(length(names) - 1)
            for right_index in (left_index + 1):length(names)
                left = names[left_index]
                right = names[right_index]
                pairwise["$left/$right"] = (;
                    cosine=finite_cosine(
                        components[left],
                        components[right],
                    ),
                    dot=dot(components[left], components[right]),
                )
            end
        end
    end
    sum_component_norms = sum(values(norms); init=0.0)
    return (;
        component_norm=norms,
        component_alignment_with_total=Dict(
            name => finite_cosine(components[name], total)
            for name in names
        ),
        pairwise,
        total_norm=norm(total),
        sum_component_norms,
        cancellation_ratio=sum_component_norms == 0.0 ?
            0.0 : norm(total) / sum_component_norms,
        conflict_fraction=sum_component_norms == 0.0 ?
            0.0 : 1.0 - norm(total) / sum_component_norms,
    )
end

function conditional_metric_summary(records, selector)
    selected = filter(selector, records)
    return (;
        states=length(selected),
        mean_listnet_kl=isempty(selected) ? nothing :
            mean(getproperty.(selected, :kl)),
        mean_normalized_teacher_entropy=isempty(selected) ? nothing :
            mean(getproperty.(selected, :normalized_teacher_entropy)),
        mean_teacher_probability_gap=isempty(selected) ? nothing :
            begin
                gaps = Float64[
                    record.teacher_probability_gap
                    for record in selected
                    if record.teacher_probability_gap !== nothing
                ]
                isempty(gaps) ? nothing : mean(gaps)
            end,
    )
end

function panel_listnet_gradient_diagnostics(
    state_records;
    state_batch::Int,
)
    state_batch >= 1 ||
        throw(ArgumentError("state_batch must be positive"))
    isempty(state_records) && return nothing
    state_diagnostics = NamedTuple[]
    listnet_gradient = Float64[]
    old_q_gradient = Float64[]
    margin_gradient = Float64[]
    batch_records = Any[]
    for first_state in 1:state_batch:length(state_records)
        last_state = min(
            first_state + state_batch - 1,
            length(state_records),
        )
        group = @view state_records[first_state:last_state]
        batch_size = length(group)
        valid_total = sum(length(record.q) for record in group)
        batch_listnet = Float64[]
        batch_old_q = Float64[]
        batch_margin = Float64[]
        for record in group
            length(record.q) == length(record.teacher) ==
                length(record.teacher_z) ||
                throw(
                    DimensionMismatch(
                        "q/teacher/teacher_z length differs",
                    ),
                )
            isempty(record.q) &&
                throw(ArgumentError("panel state has no candidates"))
            diagnostic = exact_listnet_state_gradient(
                record.q,
                record.teacher_z;
                state_batch=batch_size,
            )
            listnet = state_listnet_metrics(
                record.q,
                record.teacher_z,
            )
            push!(state_diagnostics, merge(diagnostic, (;
                raw_kl=listnet.kl,
                kl=max(listnet.kl, 0.0),
            )))
            append!(batch_listnet, diagnostic.delta_q)
            append!(
                batch_old_q,
                0.25 .* clamp.(
                    Float64.(record.q) .-
                    Float64.(record.teacher),
                    -1.0,
                    1.0,
                ) ./ valid_total,
            )
            count = length(record.q)
            state_margin = zeros(Float64, count)
            if count >= 2
                top1, top2 = stable_top_two(record.teacher)
                target_margin = Float64(
                    record.teacher[top1] -
                    record.teacher[top2],
                )
                error_value =
                    Float64(record.q[top1] - record.q[top2]) -
                    target_margin
                value =
                    0.15 * clamp(error_value, -1.0, 1.0) /
                    batch_size
                state_margin[top1] += value
                state_margin[top2] -= value
            end
            append!(batch_margin, state_margin)
        end
        append!(listnet_gradient, batch_listnet)
        append!(old_q_gradient, batch_old_q)
        append!(margin_gradient, batch_margin)
        push!(batch_records, (;
            first_state,
            last_state,
            states=batch_size,
            candidates=valid_total,
            gradients=gradient_component_summary(Dict(
                "listnet_standardized" => batch_listnet,
                "old_q_huber" => batch_old_q,
                "teacher_top2_margin" => batch_margin,
            )),
        ))
    end
    entropy_buckets = Dict(
        "low_lt_0.50" => conditional_metric_summary(
            state_diagnostics,
            record -> record.normalized_teacher_entropy < 0.50,
        ),
        "mid_0.50_to_0.80" => conditional_metric_summary(
            state_diagnostics,
            record ->
                0.50 <= record.normalized_teacher_entropy < 0.80,
        ),
        "high_ge_0.80" => conditional_metric_summary(
            state_diagnostics,
            record -> record.normalized_teacher_entropy >= 0.80,
        ),
    )
    probability_gap_buckets = Dict(
        "ambiguous_lt_0.05" => conditional_metric_summary(
            state_diagnostics,
            record ->
                record.teacher_probability_gap !== nothing &&
                record.teacher_probability_gap < 0.05,
        ),
        "middle_0.05_to_0.15" => conditional_metric_summary(
            state_diagnostics,
            record ->
                record.teacher_probability_gap !== nothing &&
                0.05 <= record.teacher_probability_gap < 0.15,
        ),
        "clear_ge_0.15" => conditional_metric_summary(
            state_diagnostics,
            record ->
                record.teacher_probability_gap !== nothing &&
                record.teacher_probability_gap >= 0.15,
        ),
        "singleton" => conditional_metric_summary(
            state_diagnostics,
            record -> record.teacher_probability_gap === nothing,
        ),
    )
    candidate_count_buckets = Dict(
        "1_to_16" => conditional_metric_summary(
            state_diagnostics,
            record -> record.candidate_count <= 16,
        ),
        "17_to_40" => conditional_metric_summary(
            state_diagnostics,
            record -> 17 <= record.candidate_count <= 40,
        ),
        "ge_41" => conditional_metric_summary(
            state_diagnostics,
            record -> record.candidate_count >= 41,
        ),
    )
    return (;
        formula=(
            "p=softmax(teacher_z/tau); s=softmax(zq/tau); " *
            "delta_z=(s-p)/(tau*batch); " *
            "delta_q=(delta_z-mean(delta_z)-zq*mean(delta_z*zq))/scale"
        ),
        temperature=Float64(
            getfield(BeatFirstTrainingCore, :LISTNET_TEMPERATURE),
        ),
        states=length(state_diagnostics),
        batches=length(batch_records),
        fixed_panel_pseudo_batches=true,
        observed_training_batches=false,
        state_batch,
        partial_final_batch_states=
            mod(length(state_diagnostics), state_batch),
        q_state_rms=distribution_statistics(
            getproperty.(state_diagnostics, :q_state_rms),
        ),
        centered_q_rms=distribution_statistics(
            getproperty.(state_diagnostics, :centered_q_rms),
        ),
        standardization_scale=distribution_statistics(
            getproperty.(state_diagnostics, :standardization_scale),
        ),
        standardization_inverse_scale=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :standardization_inverse_scale,
            ),
        ),
        delta_z_norm=distribution_statistics(
            getproperty.(state_diagnostics, :delta_z_norm),
        ),
        projected_delta_z_norm=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :projected_delta_z_norm,
            ),
        ),
        delta_q_norm=distribution_statistics(
            getproperty.(state_diagnostics, :delta_q_norm),
        ),
        standardization_projection_retention=
            distribution_statistics(Float64[
                record.standardization_projection_retention
                for record in state_diagnostics
                if record.standardization_projection_retention !== nothing
            ]),
        raw_q_gradient_gain=distribution_statistics(
            Float64[
                record.raw_q_gradient_gain
                for record in state_diagnostics
                if record.raw_q_gradient_gain !== nothing
            ],
        ),
        zero_upstream_delta_states=count(
            record -> record.delta_z_norm == 0.0,
            state_diagnostics,
        ),
        teacher_normalized_entropy=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :normalized_teacher_entropy,
            ),
        ),
        teacher_effective_choices=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :teacher_effective_choices,
            ),
        ),
        teacher_probability_top1=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :teacher_probability_top1,
            ),
        ),
        teacher_probability_gap=distribution_statistics(
            Float64[
                record.teacher_probability_gap
                for record in state_diagnostics
                if record.teacher_probability_gap !== nothing
            ],
        ),
        total_variation=distribution_statistics(
            getproperty.(state_diagnostics, :total_variation),
        ),
        student_normalized_entropy=distribution_statistics(
            getproperty.(
                state_diagnostics,
                :normalized_student_entropy,
            ),
        ),
        conditional_kl=(;
            by_normalized_teacher_entropy=entropy_buckets,
            by_teacher_probability_gap=probability_gap_buckets,
            by_candidate_count=candidate_count_buckets,
        ),
        q_head_gradient_conflict=gradient_component_summary(Dict(
            "listnet_standardized" => listnet_gradient,
            "old_q_huber" => old_q_gradient,
            "teacher_top2_margin" => margin_gradient,
        )),
        per_batch=batch_records,
        teacher_choice_margin_note=(
            "teacher_probability_gap is the teacher ListNet probability " *
            "top1-top2 gap; metrics.teacher_choice_margin is instead the " *
            "student raw-Q margin at the teacher choice"
        ),
    )
end

function state_metric(prediction, teacher, teacher_z)
    listnet = state_listnet_metrics(prediction, teacher_z)
    predicted_order = partialsortperm(
        prediction,
        1:min(2, length(prediction));
        rev=true,
    )
    teacher_order = partialsortperm(
        teacher,
        1:min(2, length(teacher));
        rev=true,
    )
    predicted_gap = length(prediction) >= 2 ?
        Float64(
            prediction[predicted_order[1]] -
            prediction[predicted_order[2]],
        ) : 0.0
    teacher_gap = length(teacher) >= 2 ?
        Float64(
            teacher[teacher_order[1]] -
            teacher[teacher_order[2]],
        ) : 0.0
    teacher_choice_margin = length(prediction) >= 2 ?
        Float64(
            prediction[teacher_order[1]] -
            maximum(prediction[
                filter(!=(teacher_order[1]), eachindex(prediction))
            ]),
        ) : 0.0
    return merge(listnet, (;
        top1=argmax(prediction) == argmax(teacher),
        ndcg=BeatFirstTrainingCore._ndcg(prediction, teacher),
        pairwise=BeatFirstTrainingCore._pairwise_accuracy(
            prediction,
            teacher,
        ),
        teacher_gap,
        predicted_gap,
        teacher_choice_margin,
    ))
end

@inline safe_mean(values) =
    isempty(values) ? nothing : mean(Float64.(values))

function summarize_metrics(records)
    return (;
        states=length(records),
        listnet_cross_entropy=safe_mean(
            getproperty.(records, :cross_entropy),
        ),
        teacher_entropy=safe_mean(
            getproperty.(records, :teacher_entropy),
        ),
        listnet_kl=safe_mean(getproperty.(records, :kl)),
        top1_agreement=safe_mean(getproperty.(records, :top1)),
        ndcg=safe_mean(getproperty.(records, :ndcg)),
        pairwise_accuracy=safe_mean(
            getproperty.(records, :pairwise),
        ),
        teacher_top_gap=safe_mean(
            getproperty.(records, :teacher_gap),
        ),
        predicted_top_gap=safe_mean(
            getproperty.(records, :predicted_gap),
        ),
        teacher_choice_margin=safe_mean(
            getproperty.(records, :teacher_choice_margin),
        ),
    )
end

function metric_delta_from_full(metrics, full)
    return (;
        listnet_cross_entropy=
            metrics.listnet_cross_entropy -
            full.listnet_cross_entropy,
        listnet_kl=metrics.listnet_kl - full.listnet_kl,
        top1_agreement=
            metrics.top1_agreement - full.top1_agreement,
        ndcg=metrics.ndcg - full.ndcg,
        pairwise_accuracy=
            metrics.pairwise_accuracy - full.pairwise_accuracy,
        teacher_choice_margin=
            metrics.teacher_choice_margin -
            full.teacher_choice_margin,
    )
end

function gap_bucket_summary(records)
    result = Any[]
    for bucket in GAP_BUCKETS
        selected = filter(
            record ->
                bucket.lower <= record.teacher_gap < bucket.upper,
            records,
        )
        push!(result, merge(
            (;
                name=bucket.name,
                lower=bucket.lower,
                upper=isfinite(bucket.upper) ? bucket.upper : nothing,
            ),
            summarize_metrics(selected),
        ))
    end
    return result
end

mutable struct ActivationAccumulator
    count::Int
    absolute_over_095::Int
    absolute_over_099::Int
    sum_value::Float64
    sum_absolute::Float64
    sum_square::Float64
    sum_tanh_derivative::Float64
end

ActivationAccumulator() =
    ActivationAccumulator(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

function accumulate_activation!(accumulator, values)
    for value32 in values
        value = Float64(value32)
        absolute = abs(value)
        accumulator.count += 1
        accumulator.absolute_over_095 += absolute > 0.95
        accumulator.absolute_over_099 += absolute > 0.99
        accumulator.sum_value += value
        accumulator.sum_absolute += absolute
        accumulator.sum_square =
            muladd(value, value, accumulator.sum_square)
        accumulator.sum_tanh_derivative += 1.0 - value * value
    end
    return accumulator
end

function activation_summary(accumulator)
    inverse = inv(max(accumulator.count, 1))
    return (;
        count=accumulator.count,
        mean=accumulator.sum_value * inverse,
        mean_absolute=accumulator.sum_absolute * inverse,
        rms=sqrt(accumulator.sum_square * inverse),
        fraction_abs_gt_0_95=
            accumulator.absolute_over_095 * inverse,
        fraction_abs_gt_0_99=
            accumulator.absolute_over_099 * inverse,
        mean_tanh_derivative=
            accumulator.sum_tanh_derivative * inverse,
    )
end

mutable struct RmsAccumulator
    rms::Vector{Float64}
    inverse_rms::Vector{Float64}
end

RmsAccumulator() = RmsAccumulator(Float64[], Float64[])

function accumulate_rms!(accumulator::RmsAccumulator, values)
    epsilon = Float64(SerialWorkspaceSNN.RMS_NORM_EPS)
    for candidate in axes(values, 2)
        square_mean = mean(abs2, @view(values[:, candidate]))
        push!(accumulator.rms, sqrt(Float64(square_mean)))
        push!(
            accumulator.inverse_rms,
            inv(sqrt(Float64(square_mean) + epsilon)),
        )
    end
    return accumulator
end

function rms_summary(accumulator::RmsAccumulator)
    return (;
        pre_epsilon_rms=distribution_statistics(accumulator.rms),
        inverse_rms=distribution_statistics(accumulator.inverse_rms),
        epsilon=Float64(SerialWorkspaceSNN.RMS_NORM_EPS),
    )
end

mutable struct PreNormActivationAccumulator
    rms::RmsAccumulator
    unit_derivative_sum::Vector{Float64}
    observations_per_unit::Int
    derivative_count::Int
    derivative_sum::Float64
    derivative_lt_0_1::Int
    derivative_lt_0_01::Int
    derivative_lt_0_001::Int
end

function PreNormActivationAccumulator(units::Int)
    units >= 1 || throw(ArgumentError("units must be positive"))
    return PreNormActivationAccumulator(
        RmsAccumulator(),
        zeros(Float64, units),
        0,
        0,
        0.0,
        0,
        0,
        0,
    )
end

function accumulate_prenorm_activation!(
    accumulator::PreNormActivationAccumulator,
    pre,
    output,
)
    size(pre) == size(output) ||
        throw(DimensionMismatch("pre/output activation shape differs"))
    size(pre, 1) == length(accumulator.unit_derivative_sum) ||
        throw(DimensionMismatch("activation unit count differs"))
    accumulate_rms!(accumulator.rms, pre)
    candidates = size(pre, 2)
    accumulator.observations_per_unit += candidates
    accumulator.derivative_count += length(output)
    @inbounds for candidate in axes(output, 2)
        for unit in axes(output, 1)
            derivative =
                1.0 - Float64(output[unit, candidate])^2
            accumulator.unit_derivative_sum[unit] += derivative
            accumulator.derivative_sum += derivative
            accumulator.derivative_lt_0_1 += derivative < 0.1
            accumulator.derivative_lt_0_01 += derivative < 0.01
            accumulator.derivative_lt_0_001 += derivative < 0.001
        end
    end
    return accumulator
end

function prenorm_activation_summary(
    accumulator::PreNormActivationAccumulator,
)
    count_value = max(accumulator.derivative_count, 1)
    unit_denominator = max(accumulator.observations_per_unit, 1)
    unit_mean_derivative =
        accumulator.unit_derivative_sum ./ unit_denominator
    return (;
        rms_normalization=rms_summary(accumulator.rms),
        tanh_derivative=(;
            values=distribution_statistics(unit_mean_derivative),
            global_mean=
                accumulator.derivative_sum / count_value,
            fraction_lt_0_1=
                accumulator.derivative_lt_0_1 / count_value,
            fraction_lt_0_01=
                accumulator.derivative_lt_0_01 / count_value,
            fraction_lt_0_001=
                accumulator.derivative_lt_0_001 / count_value,
            units=length(unit_mean_derivative),
            units_mean_lt_0_1=count(<(0.1), unit_mean_derivative),
            units_mean_lt_0_01=count(<(0.01), unit_mean_derivative),
            units_mean_lt_0_001=count(<(0.001), unit_mean_derivative),
        ),
    )
end

mutable struct DynamicsSaturationAccumulator
    membrane_count::Int
    membrane_margin_sum_square::Float64
    membrane_margin_abs_lt_1::Int
    membrane_margin_abs_lt_2::Int
    spike_count::Int
    surrogate_sum::Float64
    surrogate_sum_square::Float64
    surrogate_lt_1e_3::Int
    surrogate_lt_1e_4::Int
    eligibility_count::Int
    eligibility_sum_square::Float64
    eligibility_absolute_sum::Float64
    eligibility_nonzero::Int
    active_eligibility_count::Int
    active_eligibility_sum_square::Float64
    active_eligibility_nonzero::Int
    inactive_eligibility_count::Int
    inactive_eligibility_sum_square::Float64
    inactive_eligibility_nonzero::Int
    raw_spike_counts_per_node::Vector{Int}
    observations_per_node::Int
    workspace_pre_rms::RmsAccumulator
    workspace_activation::ActivationAccumulator
end

DynamicsSaturationAccumulator(model) = DynamicsSaturationAccumulator(
    0,
    0.0,
    0,
    0,
    0,
    0.0,
    0.0,
    0,
    0,
    0,
    0.0,
    0.0,
    0,
    0,
    0.0,
    0,
    0,
    0.0,
    0,
    zeros(Int, model.blocks * model.node_dim),
    0,
    RmsAccumulator(),
    ActivationAccumulator(),
)

function accumulate_dynamics_saturation!(
    accumulator::DynamicsSaturationAccumulator,
    dynamics,
)
    moments = dynamics.saturation_moments
    moments === nothing &&
        error("saturation dynamics were not recorded")
    accumulator.membrane_count += moments.membrane_count
    accumulator.membrane_margin_sum_square +=
        moments.membrane_margin_sum_square
    accumulator.membrane_margin_abs_lt_1 +=
        moments.membrane_margin_abs_lt_1
    accumulator.membrane_margin_abs_lt_2 +=
        moments.membrane_margin_abs_lt_2
    accumulator.spike_count += moments.raw_spike_count
    accumulator.surrogate_sum += moments.surrogate_sum
    accumulator.surrogate_sum_square +=
        moments.surrogate_sum_square
    accumulator.surrogate_lt_1e_3 +=
        moments.surrogate_lt_1e_3
    accumulator.surrogate_lt_1e_4 +=
        moments.surrogate_lt_1e_4
    length(accumulator.raw_spike_counts_per_node) ==
        length(moments.raw_spike_counts_per_node) ||
        error("node spike telemetry length differs")
    accumulator.raw_spike_counts_per_node .+=
        moments.raw_spike_counts_per_node
    accumulator.observations_per_node +=
        moments.observations_per_node
    eligibility = dynamics.eligibility_trace
    eligibility === nothing &&
        error("eligibility trace was not recorded")
    active_sample = dynamics.eligibility_sample_active
    active_sample === nothing &&
        error("eligibility activity mask was not recorded")
    @inbounds for candidate in axes(eligibility, 2)
        for sample in axes(eligibility, 1)
            value32 = eligibility[sample, candidate]
        value = Float64(value32)
        accumulator.eligibility_count += 1
        accumulator.eligibility_sum_square = muladd(
            value,
            value,
            accumulator.eligibility_sum_square,
        )
        accumulator.eligibility_absolute_sum += abs(value)
        accumulator.eligibility_nonzero += !iszero(value)
            if active_sample[sample]
                accumulator.active_eligibility_count += 1
                accumulator.active_eligibility_sum_square = muladd(
                    value,
                    value,
                    accumulator.active_eligibility_sum_square,
                )
                accumulator.active_eligibility_nonzero += !iszero(value)
            else
                accumulator.inactive_eligibility_count += 1
                accumulator.inactive_eligibility_sum_square = muladd(
                    value,
                    value,
                    accumulator.inactive_eligibility_sum_square,
                )
                accumulator.inactive_eligibility_nonzero +=
                    !iszero(value)
            end
        end
    end
    for cycle in axes(dynamics.workspace_pre_tanh, 2)
        accumulate_rms!(
            accumulator.workspace_pre_rms,
            @view(dynamics.workspace_pre_tanh[:, cycle, :]),
        )
        accumulate_activation!(
            accumulator.workspace_activation,
            @view(dynamics.workspace_by_cycle[:, cycle, :]),
        )
    end
    return accumulator
end

function dynamics_saturation_summary(
    accumulator::DynamicsSaturationAccumulator,
    model,
    eprop_contract,
)
    membrane_count = max(accumulator.membrane_count, 1)
    eligibility_count = max(accumulator.eligibility_count, 1)
    active_eligibility_count =
        max(accumulator.active_eligibility_count, 1)
    inactive_eligibility_count =
        max(accumulator.inactive_eligibility_count, 1)
    node_observations = max(accumulator.observations_per_node, 1)
    per_node_firing_fraction =
        accumulator.raw_spike_counts_per_node ./ node_observations
    sample_count = min(
        SATURATION_ELIGIBILITY_EDGE_SAMPLES,
        model.blocks * model.node_dim * model.fanout,
    )
    sample_indices = unique(round.(
        Int64,
        range(
            1,
            model.blocks * model.node_dim * model.fanout;
            length=sample_count,
        ),
    ))
    return (;
        membrane_threshold=(;
            normalization="(membrane-threshold)/spike_temperature",
            spike_temperature=Float64(model.spike_temperature),
            normalized_margin_rms=sqrt(
                accumulator.membrane_margin_sum_square /
                membrane_count,
            ),
            fraction_abs_lt_1=
                accumulator.membrane_margin_abs_lt_1 /
                membrane_count,
            fraction_abs_lt_2=
                accumulator.membrane_margin_abs_lt_2 /
                membrane_count,
            firing_fraction=
                accumulator.spike_count / membrane_count,
            per_node_firing_fraction=
                distribution_statistics(per_node_firing_fraction),
            silent_node_fraction=count(
                iszero,
                accumulator.raw_spike_counts_per_node,
            ) / max(length(accumulator.raw_spike_counts_per_node), 1),
            always_spiking_node_fraction=count(
                ==(accumulator.observations_per_node),
                accumulator.raw_spike_counts_per_node,
            ) / max(length(accumulator.raw_spike_counts_per_node), 1),
        ),
        surrogate=(;
            formula="sigmoid(margin)*(1-sigmoid(margin))/temperature",
            mean=accumulator.surrogate_sum / membrane_count,
            rms=sqrt(
                accumulator.surrogate_sum_square /
                membrane_count,
            ),
            fraction_lt_1e_3=
                accumulator.surrogate_lt_1e_3 /
                membrane_count,
            fraction_lt_1e_4=
                accumulator.surrogate_lt_1e_4 /
                membrane_count,
        ),
        weight_eligibility=(;
            mode="sampled_terminal_membrane_eprop_without_third_factor",
            eligibility_mode=eprop_contract.eligibility_mode,
            signal_schedule=eprop_contract.signal_schedule,
            trace_decay_scale=Float64(
                eprop_contract.trace_decay_scale,
            ),
            production_formula_applicable=eprop_contract.applicable,
            routing="deterministic_inference_hard_top_k",
            sampled_edges_per_state=min(
                SATURATION_ELIGIBILITY_EDGE_SAMPLES,
                model.blocks * model.node_dim * model.fanout,
            ),
            total_edges=
                model.blocks * model.node_dim * model.fanout,
            sampling=(
                "approximately equidistant deterministic positions over " *
                "the column-major edge tape"
            ),
            sample_index_sha256=array_content_sha256(sample_indices),
            observations=accumulator.eligibility_count,
            rms=sqrt(
                accumulator.eligibility_sum_square /
                eligibility_count,
            ),
            mean_absolute=
                accumulator.eligibility_absolute_sum /
                eligibility_count,
            nonzero_fraction=
                accumulator.eligibility_nonzero /
                eligibility_count,
            active_edges=(;
                observations=
                    accumulator.active_eligibility_count,
                rms=sqrt(
                    accumulator.active_eligibility_sum_square /
                    active_eligibility_count,
                ),
                nonzero_fraction=
                    accumulator.active_eligibility_nonzero /
                    active_eligibility_count,
            ),
            inactive_edges=(;
                observations=
                    accumulator.inactive_eligibility_count,
                rms=sqrt(
                    accumulator.inactive_eligibility_sum_square /
                    inactive_eligibility_count,
                ),
                nonzero_fraction=
                    accumulator.inactive_eligibility_nonzero /
                    inactive_eligibility_count,
            ),
            limitation=(
                "eligibility excludes the downstream third factor; exact " *
                "local/VJP credit diagnostics are reported by the shadow " *
                "benchmark"
            ),
        ),
        workspace=(;
            pre_tanh_rms_normalization=
                rms_summary(accumulator.workspace_pre_rms),
            output_activation=
                activation_summary(
                    accumulator.workspace_activation,
                ),
        ),
    )
end

mutable struct RoutingAccumulator
    blocks::Int
    cycles::Int
    workspace_k::Int
    entropy_floor::Float64
    observations::Vector{Int}
    exploitation_entropy_sum::Vector{Float64}
    policy_entropy_sum::Vector{Float64}
    entropy_floor_violations::Vector{Int}
    maximum_base_probability_sum::Vector{Float64}
    selected_base_probability_mass_sum::Vector{Float64}
    score_rms_sum::Vector{Float64}
    centered_score_rms_sum::Vector{Float64}
    score_inverse_rms_sum::Vector{Float64}
    raw_kth_gap::Vector{Vector{Float64}}
    standardized_kth_gap::Vector{Vector{Float64}}
    route_logit_kth_gap::Vector{Vector{Float64}}
    hard_counts::Matrix{Int}
    mask_pair_count::Vector{Int}
    mask_jaccard_similarity_sum::Vector{Float64}
    mask_hamming_fraction_sum::Vector{Float64}
    distinct_mask_fraction_sum::Vector{Float64}
    state_cycle_count::Vector{Int}
    transition_count::Vector{Int}
    transition_retention_sum::Vector{Float64}
    transition_hamming_fraction_sum::Vector{Float64}
    maximum_base_mass_error::Float64
    maximum_policy_mass_error::Float64
    maximum_hard_mass_error::Float64
    maximum_surrogate_mass_error::Float64
end

function RoutingAccumulator(
    model;
    entropy_floor::Real=ROUTING_ENTROPY_FLOOR_FALLBACK,
)
    0.0 <= entropy_floor <= 1.0 ||
        throw(ArgumentError("routing entropy floor must be in [0, 1]"))
    return RoutingAccumulator(
        model.blocks,
        model.cycles,
        model.workspace_k,
        Float64(entropy_floor),
        zeros(Int, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Int, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        [Float64[] for _ in 1:model.cycles],
        [Float64[] for _ in 1:model.cycles],
        [Float64[] for _ in 1:model.cycles],
        zeros(Int, model.blocks, model.cycles),
        zeros(Int, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Float64, model.cycles),
        zeros(Int, model.cycles),
        zeros(Int, max(model.cycles - 1, 0)),
        zeros(Float64, max(model.cycles - 1, 0)),
        zeros(Float64, max(model.cycles - 1, 0)),
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

@inline function entropy(probabilities)
    result = 0.0
    for probability32 in probabilities
        probability = Float64(probability32)
        probability > 0.0 &&
            (result -= probability * log(probability))
    end
    return result
end

function accumulate_routing!(accumulator, dynamics)
    candidates = size(dynamics.masks, 3)
    expected_surrogate_mass = Float64(accumulator.workspace_k)
    log_blocks = log(Float64(accumulator.blocks))
    for cycle in 1:accumulator.cycles, candidate in 1:candidates
        base = @view dynamics.base_probabilities[:, cycle, candidate]
        policy =
            @view dynamics.policy_probabilities[:, cycle, candidate]
        mask = @view dynamics.masks[:, cycle, candidate]
        scores = @view dynamics.scores_by_cycle[:, cycle, candidate]
        accumulator.observations[cycle] += 1
        base_entropy = entropy(base)
        accumulator.exploitation_entropy_sum[cycle] += base_entropy
        accumulator.policy_entropy_sum[cycle] += entropy(policy)
        normalized_entropy = base_entropy / log_blocks
        accumulator.entropy_floor_violations[cycle] +=
            normalized_entropy < accumulator.entropy_floor
        accumulator.maximum_base_probability_sum[cycle] +=
            maximum(base)
        accumulator.selected_base_probability_mass_sum[cycle] +=
            sum(base .* mask)
        score_mean = mean(Float64.(scores))
        score_rms = sqrt(mean(abs2, Float64.(scores)))
        centered_rms = sqrt(
            mean(
                score -> (Float64(score) - score_mean)^2,
                scores,
            ) +
            Float64(SerialWorkspaceSNN.RMS_NORM_EPS),
        )
        inverse_rms = inv(centered_rms)
        accumulator.score_rms_sum[cycle] += score_rms
        accumulator.centered_score_rms_sum[cycle] += centered_rms
        accumulator.score_inverse_rms_sum[cycle] += inverse_rms
        if accumulator.workspace_k < accumulator.blocks
            order = partialsortperm(
                scores,
                1:(accumulator.workspace_k + 1);
                rev=true,
            )
            raw_gap = Float64(
                scores[order[accumulator.workspace_k]] -
                scores[order[accumulator.workspace_k + 1]],
            )
            push!(accumulator.raw_kth_gap[cycle], raw_gap)
            push!(
                accumulator.standardized_kth_gap[cycle],
                raw_gap * inverse_rms,
            )
            push!(
                accumulator.route_logit_kth_gap[cycle],
                raw_gap * inverse_rms /
                Float64(dynamics.route_temperature),
            )
        end
        accumulator.maximum_base_mass_error = max(
            accumulator.maximum_base_mass_error,
            abs(Float64(sum(base)) - 1.0),
        )
        accumulator.maximum_policy_mass_error = max(
            accumulator.maximum_policy_mass_error,
            abs(Float64(sum(policy)) - 1.0),
        )
        accumulator.maximum_hard_mass_error = max(
            accumulator.maximum_hard_mass_error,
            abs(Float64(sum(mask)) - accumulator.workspace_k),
        )
        accumulator.maximum_surrogate_mass_error = max(
            accumulator.maximum_surrogate_mass_error,
            abs(
                Float64(dynamics.surrogate_mass[cycle, candidate]) -
                expected_surrogate_mass
            ),
        )
        for block in 1:accumulator.blocks
            accumulator.hard_counts[block, cycle] +=
                mask[block] > 0.5f0
        end
    end
    for cycle in 1:accumulator.cycles
        signatures = Set{Tuple}()
        for candidate in 1:candidates
            selected = Tuple(findall(
                >(0.5f0),
                @view(dynamics.masks[:, cycle, candidate]),
            ))
            push!(signatures, selected)
        end
        accumulator.distinct_mask_fraction_sum[cycle] +=
            length(signatures) / max(candidates, 1)
        accumulator.state_cycle_count[cycle] += 1
        if candidates >= 2
            for left in 1:(candidates - 1)
                left_mask =
                    @view dynamics.masks[:, cycle, left]
                for right in (left + 1):candidates
                    right_mask =
                        @view dynamics.masks[:, cycle, right]
                    intersection = 0
                    @inbounds for block in 1:accumulator.blocks
                        intersection +=
                            left_mask[block] > 0.5f0 &&
                            right_mask[block] > 0.5f0
                    end
                    union_count =
                        2 * accumulator.workspace_k - intersection
                    accumulator.mask_pair_count[cycle] += 1
                    accumulator.mask_jaccard_similarity_sum[cycle] +=
                        intersection / max(union_count, 1)
                    accumulator.mask_hamming_fraction_sum[cycle] +=
                        2 * (
                            accumulator.workspace_k - intersection
                        ) / accumulator.blocks
                end
            end
        end
    end
    for cycle in 2:accumulator.cycles, candidate in 1:candidates
        previous = @view dynamics.masks[:, cycle - 1, candidate]
        current = @view dynamics.masks[:, cycle, candidate]
        intersection = 0
        @inbounds for block in 1:accumulator.blocks
            intersection +=
                previous[block] > 0.5f0 &&
                current[block] > 0.5f0
        end
        transition = cycle - 1
        accumulator.transition_count[transition] += 1
        accumulator.transition_retention_sum[transition] +=
            intersection / accumulator.workspace_k
        accumulator.transition_hamming_fraction_sum[transition] +=
            2 * (
                accumulator.workspace_k - intersection
            ) / accumulator.blocks
    end
    return accumulator
end

function hard_load_summary(counts, observations, workspace_k)
    total = observations * workspace_k
    loads = total == 0 ?
        zeros(Float64, length(counts)) :
        Float64.(counts) ./ Float64(total)
    load_entropy = entropy(loads)
    sorted_loads = sort(loads; rev=true)
    return (;
        normalized_entropy=length(loads) <= 1 ?
            0.0 : load_entropy / log(Float64(length(loads))),
        effective_blocks=exp(load_entropy),
        top8_share=sum(@view sorted_loads[1:min(8, end)]),
        maximum_block_share=isempty(sorted_loads) ?
            0.0 : first(sorted_loads),
        coverage_blocks=count(>(0), counts),
        coverage_fraction=count(>(0), counts) / max(length(counts), 1),
        selections=total,
    )
end

function routing_summary(accumulator)
    log_blocks = log(Float64(accumulator.blocks))
    per_cycle = Any[]
    for cycle in 1:accumulator.cycles
        observations = accumulator.observations[cycle]
        pairs = accumulator.mask_pair_count[cycle]
        state_cycles = accumulator.state_cycle_count[cycle]
        push!(per_cycle, (;
            cycle,
            candidates=observations,
            exploitation_entropy=
                accumulator.exploitation_entropy_sum[cycle] /
                max(observations, 1) / log_blocks,
            first_draw_policy_entropy=
                accumulator.policy_entropy_sum[cycle] /
                max(observations, 1) / log_blocks,
            entropy_floor=accumulator.entropy_floor,
            entropy_floor_violation_fraction=
                accumulator.entropy_floor_violations[cycle] /
                max(observations, 1),
            mean_maximum_base_probability=
                accumulator.maximum_base_probability_sum[cycle] /
                max(observations, 1),
            mean_selected_base_probability_mass=
                accumulator.selected_base_probability_mass_sum[cycle] /
                max(observations, 1),
            score=(;
                rms=accumulator.score_rms_sum[cycle] /
                    max(observations, 1),
                centered_rms=
                    accumulator.centered_score_rms_sum[cycle] /
                    max(observations, 1),
                inverse_centered_rms=
                    accumulator.score_inverse_rms_sum[cycle] /
                    max(observations, 1),
                raw_kth_boundary_gap=distribution_statistics(
                    accumulator.raw_kth_gap[cycle],
                ),
                standardized_kth_boundary_gap=
                    distribution_statistics(
                        accumulator.standardized_kth_gap[cycle],
                    ),
                route_logit_kth_boundary_gap=
                    distribution_statistics(
                        accumulator.route_logit_kth_gap[cycle],
                    ),
            ),
            within_state_mask_diversity=(;
                candidate_pairs=pairs,
                mean_jaccard_similarity=
                    accumulator.mask_jaccard_similarity_sum[cycle] /
                    max(pairs, 1),
                mean_jaccard_distance=
                    1.0 -
                    accumulator.mask_jaccard_similarity_sum[cycle] /
                    max(pairs, 1),
                mean_hamming_fraction=
                    accumulator.mask_hamming_fraction_sum[cycle] /
                    max(pairs, 1),
                mean_distinct_mask_fraction=
                    accumulator.distinct_mask_fraction_sum[cycle] /
                    max(state_cycles, 1),
            ),
            hard_load=hard_load_summary(
                @view(accumulator.hard_counts[:, cycle]),
                observations,
                accumulator.workspace_k,
            ),
        ))
    end
    total_observations = sum(accumulator.observations)
    total_counts = vec(sum(accumulator.hard_counts; dims=2))
    transition = Any[]
    for index in eachindex(accumulator.transition_count)
        count_value = accumulator.transition_count[index]
        push!(transition, (;
            from_cycle=index,
            to_cycle=index + 1,
            observations=count_value,
            mean_selected_block_retention=
                accumulator.transition_retention_sum[index] /
                max(count_value, 1),
            mean_selected_block_churn=
                1.0 -
                accumulator.transition_retention_sum[index] /
                max(count_value, 1),
            mean_hamming_fraction=
                accumulator.transition_hamming_fraction_sum[index] /
                max(count_value, 1),
        ))
    end
    return (;
        entropy_normalization="natural_entropy_divided_by_log_blocks",
        exploitation_entropy=sum(
            accumulator.exploitation_entropy_sum,
        ) / max(total_observations, 1) / log_blocks,
        first_draw_policy_entropy=sum(
            accumulator.policy_entropy_sum,
        ) / max(total_observations, 1) / log_blocks,
        entropy_floor=accumulator.entropy_floor,
        entropy_floor_violation_fraction=sum(
            accumulator.entropy_floor_violations,
        ) / max(total_observations, 1),
        policy_entropy_scope=(
            "entropy of the first Plackett-Luce categorical draw; " *
            "not entropy of the ordered top-k distribution"
        ),
        hard_load=hard_load_summary(
            total_counts,
            total_observations,
            accumulator.workspace_k,
        ),
        per_cycle,
        cycle_transition=transition,
        mass_checks=(;
            maximum_base_probability_mass_error=
                accumulator.maximum_base_mass_error,
            maximum_policy_probability_mass_error=
                accumulator.maximum_policy_mass_error,
            maximum_hard_mask_mass_error=
                accumulator.maximum_hard_mass_error,
            surrogate_mass_target=accumulator.workspace_k,
            maximum_surrogate_mass_error=
                accumulator.maximum_surrogate_mass_error,
        ),
    )
end

function array_statistics(array)
    count_values = length(array)
    finite_count = count(isfinite, array)
    sum_value = 0.0
    sum_square = 0.0
    minimum_value = Inf
    maximum_value = -Inf
    for value32 in array
        value = Float64(value32)
        isfinite(value) || continue
        sum_value += value
        sum_square = muladd(value, value, sum_square)
        minimum_value = min(minimum_value, value)
        maximum_value = max(maximum_value, value)
    end
    inverse = inv(max(finite_count, 1))
    return (;
        count=count_values,
        finite_count,
        all_finite=finite_count == count_values,
        mean=sum_value * inverse,
        rms=sqrt(sum_square * inverse),
        minimum=finite_count == 0 ? nothing : minimum_value,
        maximum=finite_count == 0 ? nothing : maximum_value,
    )
end

function array_content_sha256(array)
    dense = collect(vec(array))
    return bytes2hex(sha256(reinterpret(UInt8, dense)))
end

function parameter_statistics(parameters)
    result = Dict{String,Any}()
    for name in keys(parameters)
        value = getproperty(parameters, name)
        value isa AbstractArray ||
            continue
        result[string(name)] = merge(
            array_statistics(value),
            (;
                shape=collect(size(value)),
                element_type=string(eltype(value)),
                sha256=array_content_sha256(value),
            ),
        )
    end
    return result
end

function distribution_statistics(values)
    finite = sort!(Float64[
        Float64(value) for value in values if isfinite(value)
    ])
    base = array_statistics(values)
    isempty(finite) && return merge(base, (;
        median=nothing,
        p90=nothing,
        p95=nothing,
        p99=nothing,
    ))
    return merge(base, (;
        median=quantile(finite, 0.50),
        p90=quantile(finite, 0.90),
        p95=quantile(finite, 0.95),
        p99=quantile(finite, 0.99),
    ))
end

function bounded_sigmoid_statistics(
    logits;
    minimum::Real=0.0,
    span::Real=1.0,
)
    probability = sigmoid.(Float64.(logits))
    values = Float64(minimum) .+ Float64(span) .* probability
    derivative =
        Float64(span) .* probability .* (1.0 .- probability)
    count_value = max(length(probability), 1)
    return (;
        logits=distribution_statistics(logits),
        probability=distribution_statistics(probability),
        transformed_value=distribution_statistics(values),
        logit_derivative=distribution_statistics(derivative),
        probability_extremes=(;
            fraction_lt_0_01=count(<(0.01), probability) /
                count_value,
            fraction_gt_0_99=count(>(0.99), probability) /
                count_value,
        ),
        derivative_collapse=(;
            fraction_lt_1e_3=count(<(1.0e-3), derivative) /
                count_value,
            fraction_lt_1e_4=count(<(1.0e-4), derivative) /
                count_value,
        ),
        transform=(; minimum=Float64(minimum), span=Float64(span)),
    )
end

function parameter_transform_saturation(parameters)
    gate_mask = parameters.gate_logits .>= 0.0f0
    gate_all = bounded_sigmoid_statistics(parameters.gate_logits)
    return (;
        delay=bounded_sigmoid_statistics(parameters.delay_logits),
        leak=bounded_sigmoid_statistics(
            parameters.leak_logits;
            minimum=0.45,
            span=0.50,
        ),
        threshold=bounded_sigmoid_statistics(
            parameters.threshold_logits;
            minimum=0.25,
            span=0.75,
        ),
        gate=merge(gate_all, (;
            active=bounded_sigmoid_statistics(
                parameters.gate_logits[gate_mask],
            ),
            inactive=bounded_sigmoid_statistics(
                parameters.gate_logits[.!gate_mask],
            ),
        )),
        workspace_decay=bounded_sigmoid_statistics(
            parameters.workspace_decay_logit;
            minimum=Float64(
                SerialWorkspaceSNN.WORKSPACE_DECAY_MIN,
            ),
            span=Float64(
                SerialWorkspaceSNN.WORKSPACE_DECAY_RANGE,
            ),
        ),
    )
end

function adam_moment_implied_statistics(
    parameter,
    first_moment,
    second_moment;
    learning_rate::Real,
    weight_decay::Real,
    epsilon::Real,
    first_bias_power::Real,
    second_bias_power::Real,
)
    size(parameter) == size(first_moment) == size(second_moment) ||
        throw(DimensionMismatch("Adam parameter/moment shape differs"))
    0.0 <= first_bias_power < 1.0 ||
        throw(ArgumentError("invalid Adam first bias power"))
    0.0 <= second_bias_power < 1.0 ||
        throw(ArgumentError("invalid Adam second bias power"))
    count_value = length(parameter)
    adaptive_direction = Vector{Float64}(undef, count_value)
    adaptive_step = similar(adaptive_direction)
    decay_direction_values = similar(adaptive_direction)
    total_update = similar(adaptive_direction)
    update_to_weight = Float64[]
    sizehint!(update_to_weight, count_value)
    decay_dominant = 0
    opposing_direction = 0
    near_zero_weight = 0
    inverse_first_bias = inv(1.0 - Float64(first_bias_power))
    inverse_second_bias = inv(1.0 - Float64(second_bias_power))
    rate = Float64(learning_rate)
    decay = Float64(weight_decay)
    epsilon_value = Float64(epsilon)
    @inbounds for (flat, index) in enumerate(eachindex(parameter))
        first_hat =
            Float64(first_moment[index]) * inverse_first_bias
        second_hat =
            Float64(second_moment[index]) * inverse_second_bias
        direction = first_hat / (
            sqrt(max(second_hat, 0.0)) + epsilon_value
        )
        decay_direction = decay * Float64(parameter[index])
        step_value = rate * direction
        update_value = -rate * (direction + decay_direction)
        adaptive_direction[flat] = direction
        adaptive_step[flat] = step_value
        decay_direction_values[flat] = decay_direction
        total_update[flat] = update_value
        denominator = abs(Float64(parameter[index]))
        if denominator <= 1.0e-12
            near_zero_weight += 1
        else
            push!(
                update_to_weight,
                abs(update_value) / denominator,
            )
        end
        decay_dominant +=
            abs(decay_direction) > abs(direction)
        opposing_direction +=
            direction * decay_direction < 0.0
    end
    parameter_rms = array_statistics(parameter).rms
    update_rms = array_statistics(total_update).rms
    return (;
        moment_implied_direction=
            distribution_statistics(adaptive_direction),
        adaptive_step=distribution_statistics(adaptive_step),
        decay_direction=
            distribution_statistics(decay_direction_values),
        total_update=distribution_statistics(total_update),
        update_to_weight_elementwise=
            distribution_statistics(update_to_weight),
        rms_update_to_rms_weight=parameter_rms == 0.0 ?
            nothing : update_rms / parameter_rms,
        near_zero_weight_excluded=near_zero_weight,
        decay_dominant_fraction=
            decay_dominant / max(count_value, 1),
        adam_decay_opposing_fraction=
            opposing_direction / max(count_value, 1),
    )
end

function binary_auc(scores, positive_mask)
    length(scores) == length(positive_mask) ||
        throw(DimensionMismatch("AUC score/mask length differs"))
    dense_scores = Float64.(vec(scores))
    dense_positive = Bool.(vec(positive_mask))
    positive_count = count(identity, dense_positive)
    negative_count = length(dense_positive) - positive_count
    if positive_count == 0 || negative_count == 0
        return nothing
    end
    order = sortperm(
        dense_scores;
        alg=Base.Sort.MergeSort,
    )
    positive_rank_sum = 0.0
    first_tie = 1
    while first_tie <= length(order)
        last_tie = first_tie
        score = dense_scores[order[first_tie]]
        while last_tie < length(order) &&
              dense_scores[order[last_tie + 1]] == score
            last_tie += 1
        end
        average_rank = (first_tie + last_tie) / 2
        for rank_index in first_tie:last_tie
            dense_positive[order[rank_index]] &&
                (positive_rank_sum += average_rank)
        end
        first_tie = last_tie + 1
    end
    return (
        positive_rank_sum -
        positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
end

function optimizer_statistics(checkpoint)
    optimizer = checkpoint.optimizer
    optimizer === nothing && return nothing
    first_moment = required_property(
        optimizer,
        :first_moment,
        "checkpoint optimizer",
    )
    second_moment = required_property(
        optimizer,
        :second_moment,
        "checkpoint optimizer",
    )
    Set(keys(first_moment)) == Set(keys(checkpoint.parameters)) ||
        error("optimizer first-moment parameter registry differs")
    Set(keys(second_moment)) == Set(keys(checkpoint.parameters)) ||
        error("optimizer second-moment parameter registry differs")
    first = Dict{String,Any}()
    second = Dict{String,Any}()
    all_first_finite = true
    all_second_finite = true
    all_second_nonnegative = true
    for name in keys(checkpoint.parameters)
        parameter = getproperty(checkpoint.parameters, name)
        first_value = getproperty(first_moment, name)
        second_value = getproperty(second_moment, name)
        size(first_value) == size(parameter) ||
            error("optimizer first-moment shape differs for $name")
        size(second_value) == size(parameter) ||
            error("optimizer second-moment shape differs for $name")
        first_stats = array_statistics(first_value)
        second_stats = array_statistics(second_value)
        second_nonnegative = all(value -> value >= 0, second_value)
        first[string(name)] = merge(
            first_stats,
            (; sha256=array_content_sha256(first_value)),
        )
        second[string(name)] = merge(
            second_stats,
            (;
                all_nonnegative=second_nonnegative,
                sha256=array_content_sha256(second_value),
            ),
        )
        all_first_finite &= first_stats.all_finite
        all_second_finite &= second_stats.all_finite
        all_second_nonnegative &= second_nonnegative
    end
    all_first_finite ||
        error("optimizer first moments contain a non-finite value")
    all_second_finite ||
        error("optimizer second moments contain a non-finite value")
    all_second_nonnegative ||
        error("optimizer second moments contain a negative value")
    optimizer_step = Int(required_property(
        optimizer,
        :step,
        "checkpoint optimizer",
    ))
    optimizer_step >= 0 ||
        error("checkpoint optimizer step is negative")
    learning_rate = Float64(required_property(
        optimizer,
        :learning_rate,
        "checkpoint optimizer",
    ))
    beta1 = Float64(required_property(
        optimizer,
        :beta1,
        "checkpoint optimizer",
    ))
    beta2 = Float64(required_property(
        optimizer,
        :beta2,
        "checkpoint optimizer",
    ))
    beta1_power = Float64(required_property(
        optimizer,
        :beta1_power,
        "checkpoint optimizer",
    ))
    beta2_power = Float64(required_property(
        optimizer,
        :beta2_power,
        "checkpoint optimizer",
    ))
    epsilon = Float64(required_property(
        optimizer,
        :epsilon,
        "checkpoint optimizer",
    ))
    weight_decay = Float64(required_property(
        optimizer,
        :weight_decay,
        "checkpoint optimizer",
    ))
    0.0 <= beta1 < 1.0 ||
        error("checkpoint optimizer beta1 is invalid")
    0.0 <= beta2 < 1.0 ||
        error("checkpoint optimizer beta2 is invalid")
    current_first_bias_power = optimizer_step == 0 ? nothing :
        (beta1 == 0.0 ? 0.0 : beta1_power / beta1)
    current_second_bias_power = optimizer_step == 0 ? nothing :
        (beta2 == 0.0 ? 0.0 : beta2_power / beta2)
    moment_implied_update = optimizer_step == 0 ?
        nothing : Dict{String,Any}()
    if optimizer_step > 0
        for name in keys(checkpoint.parameters)
            moment_implied_update[string(name)] =
                adam_moment_implied_statistics(
                    getproperty(checkpoint.parameters, name),
                    getproperty(first_moment, name),
                    getproperty(second_moment, name);
                    learning_rate,
                    weight_decay,
                    epsilon,
                    first_bias_power=current_first_bias_power,
                    second_bias_power=current_second_bias_power,
                )
        end
    end
    return (;
        step=optimizer_step,
        all_finite=all_first_finite && all_second_finite,
        all_second_moments_nonnegative=all_second_nonnegative,
        hyperparameters=(;
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            stored_beta1_power=beta1_power,
            stored_beta2_power=beta2_power,
            current_moment_beta1_power=current_first_bias_power,
            current_moment_beta2_power=current_second_bias_power,
            bias_power_note=(
                "the checkpoint stores beta^(step+1) because powers advance " *
                "after each optimizer step; current stored moments use " *
                "stored_power/beta"
            ),
        ),
        first_moment=first,
        second_moment=second,
        moment_implied_update,
        moment_implied_update_scope=(
            "direction implied by final stored moments, not a fresh " *
            "gradient; gate sign projection and structural consolidation " *
            "are excluded"
        ),
    )
end

function structural_learning_statistics(checkpoint, model)
    gates = checkpoint.parameters.gate_logits
    mask = gates .>= 0
    node_counts = vec(sum(mask; dims=2))
    fanout_histogram = Dict{String,Int}()
    for count_value in node_counts
        key = string(Int(count_value))
        fanout_histogram[key] =
            get(fanout_histogram, key, 0) + 1
    end
    expected_budget = nothing
    if checkpoint.config !== nothing
        executor = property_or(checkpoint.config, :executor, nothing)
        structural = executor === nothing ?
            nothing : property_or(executor, :structural_learning, nothing)
        keep_fraction = structural === nothing ?
            nothing :
            property_or(structural, :utility_keep_fraction, nothing)
        keep_fraction === nothing ||
            (expected_budget = round(
                Int,
                model.fanout * Float64(keep_fraction),
            ))
    end
    budget_violations = expected_budget === nothing ?
        nothing : count(!=(expected_budget), node_counts)

    utility = checkpoint.synapse_utility
    utility_summary = nothing
    utility_swap = nothing
    if utility !== nothing
        size(utility) == size(gates) ||
            error("synapse utility shape differs from gate logits")
        all(isfinite, utility) ||
            error("synapse utility contains a non-finite value")
        all(value -> value >= 0, utility) ||
            error("synapse utility contains a negative value")
        utility_summary = (;
            all=distribution_statistics(utility),
            active=distribution_statistics(utility[mask]),
            inactive=distribution_statistics(utility[.!mask]),
            active_count=count(mask),
            inactive_count=count(!, mask),
            sha256=array_content_sha256(utility),
        )
        structural_config = checkpoint.config === nothing ?
            nothing : property_or(
                property_or(
                    checkpoint.config,
                    :executor,
                    NamedTuple(),
                ),
                :structural_learning,
                nothing,
            )
        connection_cost = structural_config === nothing ?
            0.0 : Float64(property_or(
                structural_config,
                :utility_connection_cost,
                0.0,
            ))
        turnover_period = structural_config === nothing ?
            nothing : Int(property_or(
                structural_config,
                :utility_turnover_period,
                0,
            ))
        structural_interval = checkpoint.config === nothing ?
            nothing : Int(property_or(
                checkpoint.config,
                :structural_interval,
                0,
            ))
        swap_gaps = Float64[]
        next_scheduled_swap_gaps = Float64[]
        events = structural_interval === nothing ||
            structural_interval < 1 ? 0 :
            div(checkpoint.update, structural_interval)
        next_event = events + 1
        @inbounds for node in axes(utility, 1)
            active_relations = findall(@view mask[node, :])
            inactive_relations = findall(!, @view mask[node, :])
            isempty(active_relations) && continue
            isempty(inactive_relations) && continue
            worst_active = minimum(
                Float64(utility[node, relation])
                for relation in active_relations
            )
            best_inactive = maximum(
                Float64(utility[node, relation])
                for relation in inactive_relations
            ) - connection_cost
            gap = best_inactive - worst_active
            push!(swap_gaps, gap)
            if turnover_period !== nothing &&
               turnover_period >= 1 &&
               mod(node - 1, turnover_period) ==
                   mod(next_event - 1, turnover_period)
                push!(next_scheduled_swap_gaps, gap)
            end
        end
        scheduled_node_visits = if turnover_period === nothing ||
            turnover_period < 1
            nothing
        else
            sum(
                begin
                    residue = mod(node - 1, turnover_period)
                    events > residue ?
                        1 + div(events - 1 - residue, turnover_period) :
                        0
                end
                for node in axes(utility, 1)
            )
        end
        cumulative_flips = checkpoint.total_structural_flips === nothing ?
            nothing : Int(checkpoint.total_structural_flips)
        initial_mask = checkpoint.initial_parameters !== nothing &&
            hasproperty(
                checkpoint.initial_parameters,
                :gate_logits,
            ) ?
            checkpoint.initial_parameters.gate_logits .>= 0.0f0 :
            nothing
        net_flips = initial_mask === nothing ?
            nothing : count(initial_mask .!= mask)
        if checkpoint.production_schema &&
           cumulative_flips !== nothing &&
           net_flips !== nothing
            net_flips <= cumulative_flips ||
                error(
                    "net gate-mask flips exceed cumulative structural " *
                    "flip telemetry",
                )
            iseven(cumulative_flips - net_flips) ||
                error(
                    "net/cumulative structural flip parity differs",
                )
        end
        exact_budget = expected_budget !== nothing &&
            all(==(expected_budget), node_counts)
        swap_interpretation_valid =
            exact_budget &&
            cumulative_flips !== nothing &&
            iseven(cumulative_flips)
        utility_swap = (;
            connection_cost,
            swap_gap=distribution_statistics(swap_gaps),
            positive_swap_gap_fraction=isempty(swap_gaps) ?
                nothing : count(>(0.0), swap_gaps) / length(swap_gaps),
            next_scheduled_event=(;
                event_ordinal=next_event,
                nodes=length(next_scheduled_swap_gaps),
                swap_gap=distribution_statistics(
                    next_scheduled_swap_gaps,
                ),
                positive_swap_gap_fraction=
                    isempty(next_scheduled_swap_gaps) ?
                    nothing :
                    count(>(0.0), next_scheduled_swap_gaps) /
                    length(next_scheduled_swap_gaps),
            ),
            active_inactive_separation=(;
                active_mean=utility_summary.active.mean,
                inactive_mean=utility_summary.inactive.mean,
                active_minus_inactive_mean=
                    utility_summary.active.mean -
                    utility_summary.inactive.mean,
                auc_active_greater_than_inactive=
                    binary_auc(utility, mask),
            ),
            turnover=(;
                structural_interval,
                turnover_period,
                completed_consolidation_events=events,
                scheduled_node_visits,
                cumulative_gate_bit_flips=cumulative_flips,
                exact_swap_interpretation_valid=
                    swap_interpretation_valid,
                cumulative_swap_decisions=
                    swap_interpretation_valid ?
                    div(cumulative_flips, 2) : nothing,
                decision_rate_per_scheduled_visit=
                    swap_interpretation_valid &&
                    scheduled_node_visits !== nothing ?
                    div(cumulative_flips, 2) /
                    max(scheduled_node_visits, 1) : nothing,
                net_gate_bit_flips_from_initial=net_flips,
                net_to_cumulative_flip_ratio=
                    cumulative_flips === nothing ||
                    cumulative_flips == 0 ||
                    net_flips === nothing ?
                    nothing : net_flips / cumulative_flips,
                reversion_lower_bound=
                    cumulative_flips === nothing ||
                    net_flips === nothing ?
                    nothing :
                    max((cumulative_flips - net_flips) / 2, 0),
                interpretation=(
                    "one steady-budget utility swap produces two gate-bit " *
                    "flips; reversion_lower_bound is a lower bound on " *
                    "previously reversed bit transitions"
                ),
            ),
        )
    elseif checkpoint.production_schema
        error("production checkpoint is missing synapse utility")
    end
    if checkpoint.production_schema
        checkpoint.utility_updates === nothing &&
            error("production checkpoint is missing utility update counter")
        checkpoint.total_structural_flips === nothing &&
            error("production checkpoint is missing structural flip counter")
        Int(checkpoint.utility_updates) >= 0 ||
            error("utility update counter is negative")
        Int(checkpoint.total_structural_flips) >= 0 ||
            error("structural flip counter is negative")
        learning_mode = Symbol(String(required_property(
            checkpoint.config,
            :learning_mode,
            "checkpoint config",
        )))
        if learning_mode === :local_hybrid
            Int(checkpoint.utility_updates) == checkpoint.update ||
                error(
                    "local-hybrid utility update counter differs from " *
                    "checkpoint update",
                )
        end
    end

    absolute_margin = abs.(gates)
    initial_mask_flips = nothing
    initial_mask_sha256 = nothing
    if checkpoint.initial_parameters !== nothing &&
       hasproperty(checkpoint.initial_parameters, :gate_logits)
        initial_gates = checkpoint.initial_parameters.gate_logits
        size(initial_gates) == size(gates) ||
            error("initial gate-logit shape differs")
        initial_mask = initial_gates .>= 0
        initial_mask_flips = count(initial_mask .!= mask)
        initial_mask_sha256 =
            array_content_sha256(UInt8.(initial_mask))
    end
    return (;
        synapse_utility=utility_summary,
        utility_swap,
        per_node_fanout_budget=(;
            nodes=length(node_counts),
            fanout=model.fanout,
            expected_active=expected_budget,
            active_minimum=minimum(node_counts),
            active_maximum=maximum(node_counts),
            active_mean=mean(node_counts),
            exact_budget_nodes=expected_budget === nothing ?
                nothing : count(==(expected_budget), node_counts),
            budget_violations,
            histogram=fanout_histogram,
        ),
        mask=(;
            active=count(mask),
            inactive=count(!, mask),
            density=count(mask) / length(mask),
            sha256=array_content_sha256(UInt8.(mask)),
            initial_sha256=initial_mask_sha256,
            flips_from_initial=initial_mask_flips,
            total_structural_flips=checkpoint.total_structural_flips,
            utility_updates=checkpoint.utility_updates,
        ),
        gate_margin=(;
            absolute=distribution_statistics(absolute_margin),
            fraction_lt_1e_4=count(
                value -> value < 1.0e-4,
                absolute_margin,
            ) /
                length(absolute_margin),
            fraction_lt_1e_3=count(
                value -> value < 1.0e-3,
                absolute_margin,
            ) /
                length(absolute_margin),
            fraction_lt_1e_2=count(
                value -> value < 1.0e-2,
                absolute_margin,
            ) /
                length(absolute_margin),
            fraction_lt_1e_1=count(
                value -> value < 1.0e-1,
                absolute_margin,
            ) /
                length(absolute_margin),
        ),
    )
end

function parameter_growth(current, initial)
    initial === nothing && return nothing
    result = Dict{String,Any}()
    for name in intersect(collect(keys(current)), collect(keys(initial)))
        current_value = getproperty(current, name)
        initial_value = getproperty(initial, name)
        current_value isa AbstractArray || continue
        initial_value isa AbstractArray || continue
        size(current_value) == size(initial_value) || continue
        current_rms = array_statistics(current_value).rms
        initial_rms = array_statistics(initial_value).rms
        difference_rms =
            array_statistics(current_value .- initial_value).rms
        result[string(name)] = (;
            initial_rms,
            current_rms,
            rms_ratio=initial_rms == 0.0 ?
                nothing : current_rms / initial_rms,
            difference_rms,
        )
    end
    return result
end

function panel_hash(rows)
    return bytes2hex(sha256(reinterpret(UInt8, rows)))
end

function canonical_field!(io, name::AbstractString, value)
    name_bytes = codeunits(String(name))
    value_bytes = codeunits(string(value))
    write(io, string(length(name_bytes)), ':')
    write(io, name_bytes)
    write(io, '=')
    write(io, string(length(value_bytes)), ':')
    write(io, value_bytes)
    write(io, '\n')
    return io
end

function current_source_fingerprint_files()
    return (
        joinpath(@__DIR__, "SerialWorkspaceSNN.jl"),
        joinpath(@__DIR__, "WorkspaceRoutingPolicy.jl"),
        joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"),
        joinpath(@__DIR__, "train_arena_100k.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "bounded_mpmc_queue.jl",
        ),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "windows_cpu_sets.jl",
        ),
    )
end

function current_source_file_inventory()
    files = current_source_fingerprint_files()
    all(isfile, files) || return nothing
    return [
        begin
            canonical_path = realpath(path)
            relative_path = replace(
                relpath(canonical_path, @__DIR__),
                '\\' => '/',
            )
            (;
                relative_path,
                bytes=filesize(canonical_path),
                sha256=file_sha256(canonical_path),
            )
        end
        for path in files
    ]
end

function current_training_source_fingerprint()
    inventory = current_source_file_inventory()
    inventory === nothing && return nothing
    io = IOBuffer()
    canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-source-fingerprint-v2",
    )
    for entry in inventory
        canonical_field!(io, "filename", entry.relative_path)
        canonical_field!(io, "length", entry.bytes)
        canonical_field!(io, "sha256", entry.sha256)
        write(io, "end-file\n")
    end
    return bytes2hex(sha256(take!(io)))
end

function parity_panel_indices(state_count::Int; maximum::Int=8)
    state_count >= 1 || return Int[]
    count = min(state_count, maximum)
    return unique(round.(
        Int,
        range(1, state_count; length=count),
    ))
end

function evaluate_panel(
    panel_name,
    rows,
    dataset,
    model,
    parameters,
    states,
    config,
    exploration;
    enforce_parity::Bool=true,
    eprop_contract=saturation_eprop_contract(
        config;
        production_schema=false,
    ),
    routing_floor_record=routing_entropy_floor(
        config;
        production_schema=false,
    ),
)
    width = 16 * cld(maximum(dataset.action_counts[rows]), 16)
    batch = allocate_host_batch(1; max_candidates=width)
    v2 = size(parameters.head_weight, 2) == 2 * model.node_dim
    legacy = size(parameters.head_weight, 2) == 3 * model.node_dim
    v2 || legacy || error(
        "unsupported head width $(size(parameters.head_weight, 2)); " *
        "expected 2*node_dim or 3*node_dim",
    )
    records = Dict(
        name => Any[] for name in ABLATION_NAMES
    )
    query_activation = ActivationAccumulator()
    hidden_activation = ActivationAccumulator()
    query_prenorm =
        PreNormActivationAccumulator(model.node_dim)
    hidden_prenorm =
        PreNormActivationAccumulator(model.hidden)
    workspace_rms = RmsAccumulator()
    selected_pool_rms = RmsAccumulator()
    dynamics_saturation =
        DynamicsSaturationAccumulator(model)
    listnet_inputs = NamedTuple[]
    routing = RoutingAccumulator(
        model;
        entropy_floor=routing_floor_record.value,
    )
    parity_indices = Set(parity_panel_indices(length(rows)))
    parity_panel_positions = Int[]
    parity_dataset_rows = Int[]
    parity_states_compared = 0
    parity_comparisons = 0
    parity_maximum_absolute_q_difference = 0.0
    parity_maximum_absolute_raw_difference = 0.0

    for (row_index, row) in enumerate(rows)
        pack_batch!(batch, dataset, [row])
        count_actions = dataset.action_counts[row]
        input = slice_input(batch.inputs, count_actions)
        full_dynamics = diagnostic_dynamics(
            model,
            input,
            parameters;
            v2,
            exploration=exploration,
            record_saturation=true,
            trace_decay_scale=
                Float32(eprop_contract.trace_decay_scale),
        )
        full_head = diagnostic_head(
            full_dynamics,
            parameters;
            v2,
        )
        workspace_off_dynamics = diagnostic_dynamics(
            model,
            input,
            parameters;
            v2,
            workspace_off=true,
            exploration=exploration,
        )
        workspace_off = diagnostic_head(
            workspace_off_dynamics,
            parameters;
            v2,
        )
        selected_pool_off = diagnostic_head(
            full_dynamics,
            parameters;
            v2,
            selected_pool_off=true,
        )
        both_off = diagnostic_head(
            workspace_off_dynamics,
            parameters;
            v2,
            selected_pool_off=true,
        )
        synapse_dynamics = diagnostic_dynamics(
            model,
            input,
            parameters;
            v2,
            synapse_off=true,
            exploration=exploration,
        )
        synapse_off = diagnostic_head(
            synapse_dynamics,
            parameters;
            v2,
        )
        memory_dynamics = diagnostic_dynamics(
            model,
            input,
            parameters;
            v2,
            memory_off=true,
            exploration=exploration,
        )
        memory_off = diagnostic_head(
            memory_dynamics,
            parameters;
            v2,
        )
        outputs = (;
            full=full_head,
            workspace_off,
            selected_pool_off,
            both_off,
            synapse_off,
            memory_off,
        )
        teacher =
            @view dataset.teacher_q[1:count_actions, row]
        teacher_z =
            @view batch.targets.teacher_z[1:count_actions, 1]
        push!(listnet_inputs, (;
            q=copy(full_head.q),
            teacher=Float32.(teacher),
            teacher_z=Float32.(teacher_z),
        ))
        for name in ABLATION_NAMES
            prediction = getproperty(outputs, name).q
            push!(
                records[name],
                state_metric(prediction, teacher, teacher_z),
            )
        end

        accumulate_activation!(
            query_activation,
            full_dynamics.query,
        )
        accumulate_activation!(
            hidden_activation,
            full_head.hidden,
        )
        accumulate_prenorm_activation!(
            query_prenorm,
            full_dynamics.query_pre,
            full_dynamics.query,
        )
        accumulate_prenorm_activation!(
            hidden_prenorm,
            full_head.hidden_pre,
            full_head.hidden,
        )
        accumulate_rms!(
            workspace_rms,
            full_dynamics.workspace,
        )
        accumulate_rms!(
            selected_pool_rms,
            full_dynamics.pooled,
        )
        accumulate_dynamics_saturation!(
            dynamics_saturation,
            full_dynamics,
        )
        accumulate_routing!(routing, full_dynamics)

        if v2 && row_index in parity_indices
            reference_output = first(model(input, parameters, states))
            reference_q = vec(reference_output.q)
            length(reference_q) == count_actions ||
                error("reference model Q length differs")
            q_difference = maximum(abs.(
                Float64.(reference_q) .-
                Float64.(full_head.q)
            ))
            reference_raw = vcat(
                reshape(reference_output.q, 1, :),
                reshape(reference_output.death_logit, 1, :),
                reference_output.quantiles,
                reference_output.geometry,
            )
            size(reference_raw) == size(full_head.raw) ||
                error("reference model raw output shape differs")
            raw_difference = maximum(abs.(
                Float64.(reference_raw) .-
                Float64.(full_head.raw)
            ))
            parity_comparisons += length(reference_raw)
            parity_states_compared += 1
            push!(parity_panel_positions, row_index)
            push!(parity_dataset_rows, row)
            parity_maximum_absolute_q_difference = max(
                parity_maximum_absolute_q_difference,
                q_difference,
            )
            parity_maximum_absolute_raw_difference = max(
                parity_maximum_absolute_raw_difference,
                raw_difference,
            )
            if enforce_parity && raw_difference > 1.0e-4
                error(
                    "diagnostic/reference forward parity failed on $panel_name: " *
                    "maximum raw difference $raw_difference",
                )
            end
        end
    end

    summarized = Dict{String,Any}()
    for name in ABLATION_NAMES
        summarized[string(name)] = summarize_metrics(records[name])
    end
    full_summary = summarized["full"]
    ablation_deltas = Dict{String,Any}()
    for name in ABLATION_NAMES
        name === :full && continue
        ablation_deltas[string(name)] = metric_delta_from_full(
            summarized[string(name)],
            full_summary,
        )
    end
    return (;
        panel=(;
            name=panel_name,
            states=length(rows),
            candidates=sum(dataset.action_counts[rows]),
            rows_sha256=panel_hash(rows),
        ),
        metrics=summarized,
        ablation_delta_from_full=ablation_deltas,
        full_gap_buckets=gap_bucket_summary(records[:full]),
        activation=(;
            query=activation_summary(query_activation),
            hidden=activation_summary(hidden_activation),
        ),
        saturation_diagnostics=(;
            listnet=panel_listnet_gradient_diagnostics(
                listnet_inputs;
                state_batch=Int(property_or(
                    config,
                    :state_batch,
                    1,
                )),
            ),
            activation=(;
                query_pre=query_prenorm |>
                    prenorm_activation_summary,
                workspace=rms_summary(workspace_rms),
                selected_pool=rms_summary(selected_pool_rms),
                head_pre=hidden_prenorm |>
                    prenorm_activation_summary,
                head_tanh_scale=Float64(hidden_scale()),
                query_tanh_scale=Float64(query_scale()),
                interpretation=(
                    "small tanh derivative indicates output saturation; " *
                    "small inverse RMS with healthy tanh derivative " *
                    "indicates norm-growth gradient starvation"
                ),
            ),
            recurrent=dynamics_saturation_summary(
                dynamics_saturation,
                model,
                eprop_contract,
            ),
            routing_entropy_floor_source=
                routing_floor_record.source,
        ),
        workspace_decay=checkpoint_workspace_decay(parameters, v2),
        routing=routing_summary(routing),
        reference_parity=(;
            enforced=enforce_parity,
            applicable=v2,
            sampling=(
                "up to eight approximately equidistant states spanning " *
                "the complete panel"
            ),
            panel_positions=parity_panel_positions,
            dataset_rows=parity_dataset_rows,
            states_compared=parity_states_compared,
            comparisons=parity_comparisons,
            maximum_absolute_q_difference=
                parity_comparisons == 0 ?
                nothing : parity_maximum_absolute_q_difference,
            maximum_absolute_raw_difference=
                parity_comparisons == 0 ?
                nothing : parity_maximum_absolute_raw_difference,
            tolerance=1.0e-4,
        ),
    )
end

function evaluate_full_metrics_only(
    rows,
    dataset,
    model,
    parameters,
    config,
    exploration,
)
    batch = allocate_host_batch(1; max_candidates=MAX_CANDIDATES)
    records = NamedTuple[]
    v2 = size(parameters.head_weight, 2) == 2 * model.node_dim
    for row in rows
        pack_batch!(batch, dataset, [row])
        count_actions = dataset.action_counts[row]
        input = slice_input(batch.inputs, count_actions)
        dynamics = diagnostic_dynamics(
            model,
            input,
            parameters;
            v2,
            exploration,
        )
        head = diagnostic_head(dynamics, parameters; v2)
        teacher = @view dataset.teacher_q[1:count_actions, row]
        teacher_z = @view batch.targets.teacher_z[1:count_actions, 1]
        push!(
            records,
            state_metric(head.q, teacher, teacher_z),
        )
    end
    return summarize_metrics(records)
end

function causal_lineage_checkpoint_records(lineage_chain)
    isempty(lineage_chain) && return NamedTuple[]
    records = Dict{Int,NamedTuple}()
    owners = Dict{Int,String}()
    for entry in reverse(lineage_chain)
        for update in sort!(collect(keys(
            entry.manifest_snapshot.records,
        )))
            causal =
                entry.segment_start_update < update <=
                    entry.record.update ||
                (entry.scratch && update == 0)
            causal || continue
            haskey(records, update) &&
                error(
                    "training lineage checkpoint curve duplicates update " *
                    string(update),
                )
            records[update] =
                entry.manifest_snapshot.records[update]
            owners[update] = entry.run_id
        end
    end
    return [
        merge(records[update], (;
            owner_run_id=owners[update],
        ))
        for update in sort!(collect(keys(records)))
    ]
end

function intermediate_checkpoint_curve(
    lineage_chain,
    rows,
    dataset,
    model,
    states,
    config,
    exploration,
)
    records = causal_lineage_checkpoint_records(lineage_chain)
    points = Any[]
    for record in records
        checkpoint = load_analysis_checkpoint(
            record.path,
            record.sha256,
        )
        checkpoint.production_schema ||
            error("checkpoint curve contains a non-production checkpoint")
        String(checkpoint.checkpoint_kind) == "training" ||
            error("checkpoint curve contains a non-training checkpoint")
        checkpoint.update == record.update ||
            error("checkpoint curve payload update differs")
        metrics = evaluate_full_metrics_only(
            rows,
            dataset,
            model,
            checkpoint.parameters,
            config,
            exploration,
        )
        push!(points, (;
            update=record.update,
            checkpoint_path=record.path,
            checkpoint_bytes=record.bytes,
            checkpoint_sha256=record.sha256,
            owner_run_id=record.owner_run_id,
            metrics,
        ))
    end
    return (;
        panel=(;
            states=length(rows),
            candidates=sum(dataset.action_counts[rows]),
            rows_sha256=panel_hash(rows),
        ),
        points,
        updates=Int[point.update for point in points],
        causal_lineage_only=true,
        same_fixed_panel_for_every_checkpoint=true,
        stochastic_scope=false,
    )
end

function compare_results_metrics(summary, recorded; tolerance=1.0e-4)
    Int(required_property(
        recorded,
        :states,
        "recorded evaluation metrics",
    )) == Int(summary.states) ||
        error("recorded evaluation state count differs")
    pairs_to_compare = (
        :listnet_loss => :listnet_cross_entropy,
        :teacher_entropy => :teacher_entropy,
        :listnet_kl => :listnet_kl,
        :top1_agreement => :top1_agreement,
        :ndcg => :ndcg,
        :pairwise_accuracy => :pairwise_accuracy,
    )
    differences = Dict{String,Float64}()
    for (result_name, analysis_name) in pairs_to_compare
        observed = Float64(required_property(
            recorded,
            result_name,
            "recorded evaluation metrics",
        ))
        recomputed = Float64(getproperty(summary, analysis_name))
        differences[String(result_name)] = recomputed - observed
    end
    maximum_absolute_difference = maximum(abs, values(differences))
    return (;
        compared=true,
        differences,
        maximum_absolute_difference,
        tolerance,
        match=maximum_absolute_difference <= tolerance,
    )
end

function assert_json_finite(value, path="report")
    if value isa AbstractFloat
        isfinite(value) ||
            error("non-finite JSON value at $path: $value")
    elseif value isa NamedTuple
        for (name, child) in pairs(value)
            assert_json_finite(child, "$path.$name")
        end
    elseif value isa AbstractDict
        for (name, child) in pairs(value)
            assert_json_finite(child, "$path[$(repr(name))]")
        end
    elseif value isa Tuple || value isa AbstractArray
        for (index, child) in pairs(value)
            assert_json_finite(child, "$path[$index]")
        end
    end
    return value
end

function validate_output_path(
    path;
    protected_paths=(),
    forbidden_directories=(),
)
    output_path = abspath(path)
    normalized_output = lowercase(normpath(output_path))
    for protected in protected_paths
        normalized_output == lowercase(normpath(abspath(protected))) &&
            error("refusing to overwrite protected input: $output_path")
    end
    for directory in forbidden_directories
        (
            declared_path_is_within(output_path, directory) ||
            resolved_declared_path_is_within(output_path, directory)
        ) &&
            error(
                "refusing to write analysis output inside a bound immutable " *
                "directory: $(abspath(directory))",
            )
    end
    ispath(output_path) &&
        error("refusing to overwrite existing output: $output_path")
    lowercase(splitext(output_path)[2]) == ".json" ||
        error("analysis output must have a .json extension")
    return output_path
end

function atomic_write_json(
    path,
    value;
    protected_paths=(),
    forbidden_directories=(),
    precommit_check=() -> nothing,
)
    output_path = validate_output_path(
        path;
        protected_paths,
        forbidden_directories,
    )
    assert_json_finite(value)
    output_parent = dirname(output_path)
    mkpath(output_parent)
    committed_parent_path = normalized_existing_path(
        output_parent,
        "analysis output parent directory",
    )
    temporary_path, temporary_io =
        mktemp(output_parent; cleanup=false)
    try
        JSON3.pretty(temporary_io, value)
        write(temporary_io, '\n')
        flush(temporary_io)
        close(temporary_io)
        for directory in forbidden_directories
            path_is_within(temporary_path, directory) &&
                error(
                    "temporary analysis output resolved inside a bound " *
                    "immutable directory",
                )
        end
        validate_output_path(
            output_path;
            protected_paths,
            forbidden_directories,
        )
        normalized_existing_path(
            dirname(temporary_path),
            "analysis temporary-output parent directory",
        ) == committed_parent_path ||
            error("analysis temporary-output directory changed")
        normalized_existing_path(
            output_parent,
            "analysis output parent directory before commit",
        ) == committed_parent_path ||
            error("analysis output directory changed before commit")
        precommit_check()
        validate_output_path(
            output_path;
            protected_paths,
            forbidden_directories,
        )
        for directory in forbidden_directories
            path_is_within(temporary_path, directory) &&
                error(
                    "temporary analysis output resolved inside a bound " *
                    "immutable directory immediately before commit",
                )
        end
        normalized_existing_path(
            dirname(temporary_path),
            "analysis temporary-output parent before commit",
        ) == committed_parent_path ||
            error(
                "analysis temporary-output directory changed immediately " *
                "before commit",
            )
        normalized_existing_path(
            output_parent,
            "analysis output parent immediately before commit",
        ) == committed_parent_path ||
            error(
                "analysis output directory changed immediately before commit",
            )
        # A same-directory hard link is an atomic, no-clobber commit: the
        # complete temporary inode becomes visible at output_path, while a
        # concurrently-created destination makes hardlink fail instead of
        # replacing it. Unsupported filesystems fail closed.
        hardlink(temporary_path, output_path)
    finally
        isopen(temporary_io) && close(temporary_io)
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return output_path
end

function validate_runtime_provenance(config, source_fingerprint_value)
    runtime = required_property(
        config,
        :runtime_provenance,
        "checkpoint config",
    )
    isempty(String(required_property(
        runtime,
        :schema,
        "runtime provenance",
    ))) && error("runtime provenance schema is empty")
    String(required_property(
        runtime,
        :julia_version,
        "runtime provenance",
    )) == string(VERSION) ||
        error("runtime Julia version differs from the analysis runtime")
    String(required_property(
        runtime,
        :julia_architecture,
        "runtime provenance",
    )) == string(Sys.ARCH) ||
        error("runtime Julia architecture differs")
    String(required_property(
        runtime,
        :julia_machine,
        "runtime provenance",
    )) == string(Sys.MACHINE) ||
        error("runtime Julia machine differs")
    String(required_property(
        runtime,
        :julia_kernel,
        "runtime provenance",
    )) == string(Sys.KERNEL) ||
        error("runtime Julia kernel differs")
    require_sha256(
        required_property(
            runtime,
            :source_fingerprint,
            "runtime provenance",
        ),
        "runtime source fingerprint",
    ) == source_fingerprint_value ||
        error("runtime and checkpoint source fingerprints differ")

    executable_path = abspath(String(required_property(
        runtime,
        :julia_executable_path,
        "runtime provenance",
    )))
    current_executable =
        abspath(joinpath(Sys.BINDIR, Base.julia_exename()))
    require_same_existing_path(
        executable_path,
        current_executable,
        "runtime Julia executable",
    )
    executable_sha256 = require_sha256(
        required_property(
            runtime,
            :julia_executable_sha256,
            "runtime provenance",
        ),
        "runtime Julia executable SHA-256",
    )
    file_sha256(executable_path) == executable_sha256 ||
        error("runtime Julia executable SHA-256 differs")

    active_project = Base.active_project()
    active_project === nothing &&
        error("analysis requires the recorded Julia project")
    current_project = abspath(active_project)
    current_manifest = joinpath(dirname(current_project), "Manifest.toml")
    for (stem, label, expected_path) in (
        (:project_toml, "Project.toml", current_project),
        (:manifest_toml, "Manifest.toml", current_manifest),
    )
        path_name = Symbol(String(stem) * "_path")
        sha_name = Symbol(String(stem) * "_sha256")
        path = abspath(String(required_property(
            runtime,
            path_name,
            "runtime provenance",
        )))
        isfile(path) ||
            error("runtime $label does not exist: $path")
        require_same_existing_path(
            path,
            expected_path,
            "runtime $label",
        )
        digest = require_sha256(
            required_property(runtime, sha_name, "runtime provenance"),
            "runtime $label SHA-256",
        )
        file_sha256(path) == digest ||
            error("runtime $label SHA-256 differs")
    end

    source_files = required_property(
        runtime,
        :source_files,
        "runtime provenance",
    )
    source_files isa AbstractVector ||
        source_files isa JSON3.Array ||
        error("runtime source_files must be an array")
    isempty(source_files) &&
        error("runtime source_files must not be empty")
    current_inventory = current_source_file_inventory()
    current_inventory === nothing &&
        error("current source inventory is unavailable")
    require_canonical_equal(
        source_files,
        current_inventory,
        "runtime source-file inventory",
    )
    return runtime
end

function validate_production_config(
    checkpoint,
    verified_run,
    analysis_source_fingerprint,
)
    checkpoint.production_schema ||
        error("validate_production_config requires production schema")
    config = checkpoint.config
    config === nothing && error("production checkpoint has no config")
    schema = required_property(
        config,
        :checkpoint_schema,
        "checkpoint config",
    )
    String(required_property(
        schema,
        :format,
        "checkpoint schema",
    )) == ARENA_CHECKPOINT_FORMAT ||
        error("configured checkpoint format differs")
    Int(required_property(
        schema,
        :version,
        "checkpoint schema",
    )) == PRODUCTION_ARENA_CHECKPOINT_VERSION ||
        error("configured checkpoint version differs")
    production_contract = required_property(
        config,
        :production_contract,
        "checkpoint config",
    )
    Int(required_property(
        production_contract,
        :version,
        "production contract",
    )) == 1 || error("production contract version differs")
    recorded_contract_sha256 = require_sha256(
        required_property(
            config,
            :production_contract_sha256,
            "checkpoint config",
        ),
        "production contract SHA-256",
    )
    actual_contract_sha256 = bytes2hex(sha256(
        codeunits(String(JSON3.write(production_contract))),
    ))
    actual_contract_sha256 == recorded_contract_sha256 ||
        error("production contract SHA-256 is corrupt")
    for name in (
        :experiment_id,
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
        :optimizer,
        :learning_mode,
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
        require_canonical_equal(
            required_property(
                config,
                name,
                "checkpoint config",
            ),
            required_property(
                production_contract,
                name,
                "production contract",
            ),
            "production contract/top-level $(String(name))",
        )
    end
    String(required_property(
        config,
        :experiment_id,
        "checkpoint config",
    )) == "serial_workspace_snn_arena_v3" ||
        error("production experiment ID differs")
    production_target = required_property(
        config,
        :production_target,
        "checkpoint config",
    )
    require_canonical_equal(
        production_target,
        (;
            model_preset="scaled_v2",
            start_mode="scratch",
            learning_mode="local_hybrid",
            maximum_updates=100_000,
        ),
        "checkpoint production target",
    )
    production_target_match =
        String(required_property(
            config,
            :model_preset,
            "checkpoint config",
        )) == "scaled_v2" &&
        String(required_property(
            config,
            :start_mode,
            "checkpoint config",
        )) == "scratch" &&
        String(required_property(
            config,
            :learning_mode,
            "checkpoint config",
        )) == "local_hybrid" &&
        Int(required_property(
            config,
            :maximum_updates,
            "checkpoint config",
        )) == 100_000
    Bool(required_property(
        config,
        :production_target_match,
        "checkpoint config",
    )) == production_target_match ||
        error("checkpoint production target match flag is inconsistent")
    run_id = String(required_property(
        config,
        :run_id,
        "checkpoint config",
    ))
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) ||
        error("checkpoint run ID is unsafe")
    if verified_run !== nothing
        run_id == verified_run.run_id ||
            error("checkpoint run ID differs from verification")
        Int(required_property(
            config,
            :maximum_updates,
            "checkpoint config",
        )) == verified_run.expected_updates ||
            error("configured maximum updates differs from verification")
    end
    source = require_sha256(
        required_property(
            config,
            :source_fingerprint,
            "checkpoint config",
        ),
        "checkpoint source fingerprint",
    )
    source == analysis_source_fingerprint ||
        error("checkpoint source fingerprint differs")
    dataset_content_sha256 = require_sha256(
        required_property(
            config,
            :dataset_content_sha256,
            "checkpoint config",
        ),
        "checkpoint dataset content SHA-256",
    )
    dataset_integrity = required_property(
        config,
        :dataset_integrity,
        "checkpoint config",
    )
    runtime = validate_runtime_provenance(config, source)
    checkpoint.payload_dataset_content_sha256 ==
        dataset_content_sha256 ||
        error("payload/config dataset content SHA-256 differs")
    require_canonical_equal(
        checkpoint.payload_dataset_integrity,
        dataset_integrity,
        "payload/config dataset integrity",
    )
    require_canonical_equal(
        checkpoint.payload_runtime_provenance,
        runtime,
        "payload/config runtime provenance",
    )
    return (;
        run_id,
        source_fingerprint=source,
        dataset_content_sha256,
        dataset_integrity,
        runtime,
        production_contract,
        production_contract_sha256=recorded_contract_sha256,
        production_target_match,
    )
end

function recorded_spike_temperature(config)
    model_config = property_or(config, :model, nothing)
    representation = property_or(config, :representation, nothing)
    for (container, name) in (
        (model_config, :spike_temperature),
        (representation, :spike_temperature),
        (property_or(config, :spiking, nothing), :spike_temperature),
        (config, :spike_temperature),
    )
        container === nothing && continue
        value = property_or(container, name, nothing)
        value === nothing || return Float32(value)
    end
    error("production config has no recorded spike temperature")
end

function validate_production_model_contract(model, parameters, config)
    preset = Symbol(String(required_property(
        config,
        :model_preset,
        "checkpoint config",
    )))
    expected_model = build_model(preset)
    topology_matches(
        model,
        expected_model.blocks,
        expected_model.node_dim,
        expected_model.fanout,
        expected_model.cycles,
        expected_model.workspace_k,
        expected_model.hidden,
    ) || error("reconstructed model differs from its recorded preset")
    recorded = required_property(
        config,
        :model,
        "checkpoint config",
    )
    expected_topology = graph_topology(model, parameters)
    for name in (
        :blocks,
        :nodes,
        :candidate_synapses,
        :enabled_synapses,
        :fanout,
        :cycles,
        :workspace_capacity,
        :input_rails,
    )
        Int(required_property(
            recorded,
            name,
            "recorded model topology",
        )) == Int(getproperty(expected_topology, name)) ||
            error("recorded model topology $(String(name)) differs")
    end
    actual_parameter_count = parameter_count(parameters)
    Int(required_property(
        config,
        :parameter_count,
        "checkpoint config",
    )) == actual_parameter_count ||
        error("recorded parameter count differs")
    routing = required_property(
        config,
        :routing,
        "checkpoint config",
    )
    Float32(required_property(
        routing,
        :route_temperature,
        "recorded routing config",
    )) == model.route_temperature ||
        error("recorded route temperature differs from the preset")
    recorded_spike_temperature(config) == model.spike_temperature ||
        error("recorded spike temperature differs from the preset")
    return (;
        preset=String(preset),
        route_temperature=model.route_temperature,
        spike_temperature=model.spike_temperature,
    )
end

function analysis_dataset_binding_preflight(path::AbstractString)
    source_path = abspath(path)
    binding_file_path = isdir(source_path) ?
        joinpath(source_path, "manifest.json") : source_path
    isfile(binding_file_path) ||
        error("dataset binding file does not exist: $binding_file_path")
    canonical_binding_file = realpath(binding_file_path)
    return (;
        kind=isdir(source_path) ? :sharded_manifest : :single_file,
        source_path=isdir(source_path) ?
            realpath(source_path) : canonical_binding_file,
        binding_file_path=canonical_binding_file,
        binding_file_sha256=file_sha256(canonical_binding_file),
    )
end

function ordered_manifest_counts(counts)
    names = sort!(String[String(name) for name in keys(counts)])
    return [
        (; name, count=Int(counts[name]))
        for name in names
    ]
end

function required_manifest_part_field(part, name::Symbol)
    return required_property(part, name, "dataset manifest part")
end

function analysis_canonical_manifest_parts_sha256(manifest)
    parts = collect(required_property(
        manifest,
        :parts,
        "dataset manifest",
    ))
    io = IOBuffer()
    canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-canonical-manifest-parts-v1",
    )
    for (index, part) in enumerate(parts)
        canonical_field!(io, "part_index", index)
        relative_path = replace(
            normpath(String(required_manifest_part_field(
                part,
                :relative_path,
            ))),
            '\\' => '/',
        )
        canonical_field!(io, "relative_path", relative_path)
        canonical_field!(
            io,
            "row_count",
            Int(required_manifest_part_field(part, :row_count)),
        )
        canonical_field!(
            io,
            "bytes",
            Int(required_manifest_part_field(part, :bytes)),
        )
        part_sha256 = require_sha256(
            required_manifest_part_field(part, :sha256),
            "dataset manifest part SHA-256",
        )
        canonical_field!(io, "sha256", part_sha256)
        canonical_field!(
            io,
            "split",
            String(required_manifest_part_field(part, :split)),
        )
        for optional_name in (
            :episode_key,
            :role,
            :seed,
            :candidate_count,
        )
            if property_or(part, optional_name, nothing) !== nothing
                canonical_field!(
                    io,
                    String(optional_name),
                    property_or(part, optional_name),
                )
            end
        end
        write(io, "end-part\n")
    end
    return bytes2hex(sha256(take!(io))), length(parts)
end

function bind_analysis_dataset(
    dataset_path,
    dataset,
    preflight,
)
    current = analysis_dataset_binding_preflight(dataset_path)
    current == preflight ||
        error("dataset binding file changed while it was loaded")
    if current.kind === :single_file
        integrity = (;
            schema="serial-workspace-snn-dataset-integrity-v1",
            kind="single_file",
            source_path=current.source_path,
            binding_file_path=current.binding_file_path,
            binding_file_sha256=current.binding_file_sha256,
            manifest_sha256=nothing,
            manifest_format_version=nothing,
            manifest_part_count=0,
            verified_part_count=0,
            part_integrity_verified=true,
            manifest_counts=NamedTuple[],
            canonical_parts_sha256=nothing,
            binding_file_stable=true,
        )
        return current.binding_file_sha256, integrity
    end

    normalized_existing_path(
        dataset.manifest_path,
        "loader manifest",
    ) == normalized_existing_path(
        current.binding_file_path,
        "bound manifest",
    ) || error("loader manifest path differs from the bound manifest")
    dataset.part_integrity_verified === true ||
        error("dataset loader did not verify every manifest part")
    Bool(property_or(dataset, :partial_dataset_allowed, true)) &&
        error("production analysis refuses a partial dataset")
    manifest = read_json_object(
        current.binding_file_path,
        "dataset manifest",
    )
    canonical_parts_sha256, manifest_part_count =
        analysis_canonical_manifest_parts_sha256(manifest)
    Int(dataset.verified_part_count) == manifest_part_count ||
        error("dataset loader verified part count differs from manifest")
    Int(dataset.manifest_format_version) ==
        Int(required_property(
            manifest,
            :format_version,
            "dataset manifest",
        )) || error(
            "dataset loader manifest format differs from bound manifest",
        )
    counts = ordered_manifest_counts(dataset.manifest_counts)
    content_io = IOBuffer()
    canonical_field!(
        content_io,
        "schema",
        "serial-workspace-snn-dataset-content-v1",
    )
    canonical_field!(
        content_io,
        "manifest_sha256",
        current.binding_file_sha256,
    )
    canonical_field!(
        content_io,
        "manifest_format_version",
        Int(dataset.manifest_format_version),
    )
    canonical_field!(
        content_io,
        "manifest_part_count",
        manifest_part_count,
    )
    canonical_field!(
        content_io,
        "canonical_parts_sha256",
        canonical_parts_sha256,
    )
    dataset_content_sha256 = bytes2hex(sha256(take!(content_io)))
    integrity = (;
        schema="serial-workspace-snn-dataset-integrity-v1",
        kind="sharded_manifest",
        source_path=current.source_path,
        binding_file_path=current.binding_file_path,
        binding_file_sha256=current.binding_file_sha256,
        manifest_sha256=current.binding_file_sha256,
        manifest_format_version=Int(dataset.manifest_format_version),
        manifest_part_count,
        verified_part_count=Int(dataset.verified_part_count),
        part_integrity_verified=true,
        manifest_counts=counts,
        canonical_parts_sha256,
        binding_file_stable=true,
    )
    return dataset_content_sha256, integrity
end

function dataset_part_metadata_witness(dataset_path, preflight)
    preflight.kind === :single_file && return [(
        path=normalized_declared_path(preflight.binding_file_path),
        bytes=filesize(preflight.binding_file_path),
        sha256=file_sha256(preflight.binding_file_path),
        mtime=stat(preflight.binding_file_path).mtime,
    )]
    manifest = read_json_object(
        preflight.binding_file_path,
        "dataset manifest",
    )
    root = realpath(dataset_path)
    root_key = normalized_declared_path(root)
    witness = NamedTuple[]
    for (index, part) in enumerate(required_property(
        manifest,
        :parts,
        "dataset manifest",
    ))
        path = normpath(joinpath(
            root,
            String(required_property(
                part,
                :relative_path,
                "dataset manifest part $index",
            )),
        ))
        isfile(path) ||
            error("dataset part $index disappeared: $path")
        resolved = realpath(path)
        relative = relpath(resolved, root)
        (!isabspath(relative) && first(splitpath(relative)) != "..") ||
            error("dataset part $index resolves outside the dataset root")
        expected_bytes = Int(required_property(
            part,
            :bytes,
            "dataset manifest part $index",
        ))
        filesize(resolved) == expected_bytes ||
            error("dataset part $index byte size changed")
        expected_sha256 = require_sha256(
            required_property(
                part,
                :sha256,
                "dataset manifest part $index",
            ),
            "dataset manifest part $index SHA-256",
        )
        actual_sha256 = file_sha256(resolved)
        actual_sha256 == expected_sha256 ||
            error("dataset part $index SHA-256 changed")
        push!(witness, (;
            path=normalized_declared_path(resolved),
            bytes=expected_bytes,
            sha256=actual_sha256,
            mtime=stat(resolved).mtime,
        ))
    end
    length(unique(getproperty.(witness, :path))) == length(witness) ||
        error("dataset part metadata witness contains duplicate paths")
    isempty(root_key) && error("dataset root normalization failed")
    return witness
end

function validate_recorded_dataset_integrity(
    recorded,
    actual;
    override_used::Bool,
)
    String(required_property(
        recorded,
        :schema,
        "recorded dataset integrity",
    )) == "serial-workspace-snn-dataset-integrity-v1" ||
        error("recorded dataset integrity schema differs")
    String(required_property(
        recorded,
        :kind,
        "recorded dataset integrity",
    )) == String(actual.kind) ||
        error("recorded dataset kind differs")
    Bool(required_property(
        recorded,
        :part_integrity_verified,
        "recorded dataset integrity",
    )) || error("recorded dataset integrity is not verified")
    Bool(required_property(
        recorded,
        :binding_file_stable,
        "recorded dataset integrity",
    )) || error("recorded dataset binding was not stable")
    if !override_used
        require_canonical_equal(
            recorded,
            actual,
            "recorded/live dataset integrity",
        )
        return true
    end
    # Paths may intentionally differ for an override. Every content and loader
    # witness remains exact.
    for name in (
        :schema,
        :kind,
        :binding_file_sha256,
        :manifest_sha256,
        :manifest_format_version,
        :manifest_part_count,
        :verified_part_count,
        :part_integrity_verified,
        :manifest_counts,
        :canonical_parts_sha256,
        :binding_file_stable,
    )
        require_canonical_equal(
            required_property(recorded, name, "recorded dataset integrity"),
            getproperty(actual, name),
            "dataset override integrity $(String(name))",
        )
    end
    return true
end

function verify_snapshot_file!(
    path,
    bytes::Integer,
    digest::AbstractString,
    location::AbstractString,
)
    isfile(path) || error("$location disappeared: $(abspath(path))")
    filesize(path) == bytes ||
        error("$location byte size changed")
    file_sha256(path) == digest ||
        error("$location SHA-256 changed")
    return true
end

function verify_exact_bound_checkpoint_sets!(verified_run)
    checkpoint_dir = joinpath(verified_run.run_dir, "checkpoints")
    live_entries = readdir(checkpoint_dir; join=true)
    all(isfile, live_entries) ||
        error("bound checkpoint directory contains a non-file entry")
    local_training_records = [
        record for record in values(verified_run.checkpoints)
        if dirname(normalized_existing_path(
            record.path,
            "bound training checkpoint",
        )) == normalized_existing_path(
            checkpoint_dir,
            "bound checkpoint directory",
        )
    ]
    expected_records = [
        local_training_records...,
        verified_run.checkpoint,
    ]
    expected_declared_paths = Set(
        normalized_declared_path(record.path)
        for record in expected_records
    )
    live_declared_paths =
        Set(normalized_declared_path.(live_entries))
    length(live_entries) == length(expected_declared_paths) ||
        error("bound checkpoint directory gained an alias or extra entry")
    live_declared_paths == expected_declared_paths ||
        error("bound checkpoint directory declared paths changed")
    expected_real_paths = Set(
        normalized_existing_path(
            record.path,
            "bound checkpoint record",
        )
        for record in expected_records
    )
    live_real_paths = Set(
        normalized_existing_path(path, "bound live checkpoint")
        for path in live_entries
    )
    length(expected_real_paths) == length(expected_records) ||
        error("bound checkpoint records contain a resolved-path alias")
    length(live_real_paths) == length(live_entries) ||
        error("bound checkpoint directory contains a resolved-path alias")
    live_real_paths == expected_real_paths ||
        error("bound checkpoint directory exact artifact set changed")
    for (update, record) in verified_run.checkpoints
        verify_file_record(
            record,
            record.path,
            "bound checkpoint $update precommit",
        )
    end
    verify_file_record(
        verified_run.checkpoint,
        verified_run.checkpoint.path,
        "bound finalization checkpoint precommit",
    )
    verified_run.parent_checkpoint === nothing ||
        verify_file_record(
            verified_run.parent_checkpoint,
            verified_run.parent_checkpoint.path,
            "bound parent checkpoint precommit",
        )
    for name in ("checkpoint_final.jld2", "checkpoint_latest.jld2")
        !ispath(joinpath(verified_run.run_dir, name)) ||
            error("an unverified checkpoint alias appeared: $name")
    end

    if verified_run.is_finalize_only
        parent_checkpoint_dir =
            dirname(verified_run.parent_checkpoint.path)
        parent_entries = readdir(parent_checkpoint_dir; join=true)
        all(isfile, parent_entries) ||
            error(
                "finalize-only parent checkpoint directory contains a " *
                "non-file entry",
            )
        expected_parent_records = collect(values(
            verified_run.manifest_records,
        ))
        if verified_run.parent_residual_finalization_record !== nothing
            push!(
                expected_parent_records,
                verified_run.parent_residual_finalization_record,
            )
        end
        expected_parent_declared_paths = Set(
            normalized_declared_path(record.path)
            for record in expected_parent_records
        )
        live_parent_declared_paths =
            Set(normalized_declared_path.(parent_entries))
        length(parent_entries) ==
            length(expected_parent_declared_paths) ||
            error(
                "finalize-only parent checkpoint directory gained an alias " *
                "or extra entry",
            )
        live_parent_declared_paths == expected_parent_declared_paths ||
            error(
                "finalize-only parent checkpoint directory paths changed",
            )
        expected_parent_real_paths = Set(
            normalized_existing_path(
                record.path,
                "finalize-only expected parent checkpoint",
            )
            for record in expected_parent_records
        )
        live_parent_real_paths = Set(
            normalized_existing_path(
                path,
                "finalize-only live parent checkpoint",
            )
            for path in parent_entries
        )
        length(expected_parent_real_paths) ==
            length(expected_parent_records) ||
            error(
                "finalize-only parent records contain a resolved-path alias",
            )
        length(live_parent_real_paths) == length(parent_entries) ||
            error(
                "finalize-only parent directory contains a resolved-path alias",
            )
        live_parent_real_paths == expected_parent_real_paths ||
            error(
                "finalize-only parent checkpoint exact artifact set changed",
            )
        for (update, record) in verified_run.manifest_records
            verify_file_record(
                record,
                record.path,
                "finalize-only parent manifest checkpoint $update precommit",
            )
        end
        if verified_run.parent_residual_finalization_record !== nothing
            verify_file_record(
                verified_run.parent_residual_finalization_record,
                verified_run.parent_residual_finalization_record.path,
                "finalize-only parent residual checkpoint precommit",
            )
        end
        verified_run.parent_residual_expected_results_path === nothing ||
            !ispath(
                verified_run.parent_residual_expected_results_path,
            ) || error(
                "finalize-only parent residual results artifact appeared",
            )
        verified_run.parent_residual_expected_manifest_path === nothing ||
            !ispath(
                verified_run.parent_residual_expected_manifest_path,
            ) || error(
                "finalize-only parent residual manifest appeared",
            )
    end
    return true
end

function verify_bound_analysis_inputs!(
    checkpoint,
    dataset_path,
    dataset_preflight,
    dataset_metadata_witness,
    analysis_source_fingerprint,
    verified_run,
    lineage_chain,
    lineage_origin_config_snapshot,
    lineage_origin_parent_checkpoint,
    analysis_script_artifact,
)
    verify_snapshot_file!(
        checkpoint.path,
        checkpoint.bytes,
        checkpoint.sha256,
        "analysis checkpoint",
    )
    analysis_dataset_binding_preflight(dataset_path) ==
        dataset_preflight ||
        error("dataset binding changed")
    dataset_part_metadata_witness(
        dataset_path,
        dataset_preflight,
    ) == dataset_metadata_witness ||
        error("dataset part content or metadata changed")
    current_training_source_fingerprint() ==
        analysis_source_fingerprint ||
        error("training sources changed")
    if checkpoint.production_schema
        validate_runtime_provenance(
            checkpoint.config,
            analysis_source_fingerprint,
        )
    end
    verify_snapshot_file!(
        analysis_script_artifact.path,
        analysis_script_artifact.bytes,
        analysis_script_artifact.sha256,
        "analysis script",
    )
    verified_run === nothing && return true

    verify_exact_bound_checkpoint_sets!(verified_run)
    for (path, bytes, digest, location) in (
        (
            verified_run.verification_path,
            verified_run.verification_bytes,
            verified_run.verification_sha256,
            "verification.json",
        ),
        (
            verified_run.results_path,
            verified_run.results_bytes,
            verified_run.results_sha256,
            "results.json",
        ),
        (
            verified_run.manifest_path,
            verified_run.manifest_bytes,
            verified_run.manifest_sha256,
            "checkpoint manifest",
        ),
        (
            verified_run.finalization_manifest_artifact.path,
            verified_run.finalization_manifest_artifact.bytes,
            verified_run.finalization_manifest_artifact.sha256,
            "finalization manifest",
        ),
        (
            verified_run.config_path,
            verified_run.config_bytes,
            verified_run.config_sha256,
            "config.json",
        ),
        (
            verified_run.launch_manifest_artifact.path,
            verified_run.launch_manifest_artifact.bytes,
            verified_run.launch_manifest_artifact.sha256,
            "launch manifest",
        ),
        (
            verified_run.trace_artifact.path,
            verified_run.trace_artifact.bytes,
            verified_run.trace_artifact.sha256,
            "training trace",
        ),
        (
            verified_run.team_teardown_artifact.path,
            verified_run.team_teardown_artifact.bytes,
            verified_run.team_teardown_artifact.sha256,
            "team teardown",
        ),
    )
        verify_snapshot_file!(path, bytes, digest, location)
    end
    for artifact in verified_run.launch_code_artifacts
        verify_snapshot_file!(
            artifact.path,
            artifact.bytes,
            artifact.sha256,
            "launch code artifact $(artifact.name)",
        )
    end
    if lineage_origin_config_snapshot !== nothing
        verify_snapshot_file!(
            lineage_origin_config_snapshot.path,
            lineage_origin_config_snapshot.bytes,
            lineage_origin_config_snapshot.sha256,
            "finalize-only origin config",
        )
    end
    lineage_origin_parent_checkpoint === nothing ||
        verify_file_record(
            lineage_origin_parent_checkpoint,
            lineage_origin_parent_checkpoint.path,
            "finalize-only origin parent checkpoint precommit",
        )
    for (index, entry) in enumerate(lineage_chain)
        verify_file_record(
            entry.record,
            entry.record.path,
            "training lineage checkpoint $index precommit",
        )
        verify_snapshot_file!(
            entry.config_snapshot.path,
            entry.config_snapshot.bytes,
            entry.config_snapshot.sha256,
            "training lineage config $index",
        )
        verify_snapshot_file!(
            entry.manifest_snapshot.path,
            entry.manifest_snapshot.bytes,
            entry.manifest_snapshot.sha256,
            "training lineage manifest $index",
        )
        verify_lineage_checkpoint_directory_snapshot!(
            entry.checkpoint_directory_snapshot,
            "training lineage checkpoint directory $index precommit",
        )
    end
    return true
end

function main(args=ARGS)
    options = parse_arguments(args)
    analysis_script_path = abspath(@__FILE__)
    analysis_script_bytes_snapshot = read(analysis_script_path)
    analysis_script_artifact = (;
        path=analysis_script_path,
        bytes=length(analysis_script_bytes_snapshot),
        sha256=bytes2hex(sha256(analysis_script_bytes_snapshot)),
    )
    verified_run = options.run_dir === nothing ?
        nothing : strict_verified_run_binding(
            options.run_dir,
            options.expected_verification_sha256,
        )
    if verified_run !== nothing
        isempty(options.expected_sha256) ||
            options.expected_sha256 ==
                verified_run.checkpoint.sha256 ||
            error(
                "caller checkpoint SHA-256 pin differs from the verified " *
                "final checkpoint",
            )
        options.required_update === nothing ||
            options.required_update == verified_run.expected_updates ||
            error(
                "caller update pin differs from verification expected update",
            )
        options.enforce_parity || error(
            "--no-parity-check is not allowed for verified production analysis",
        )
    end
    checkpoint_path = verified_run === nothing ?
        options.checkpoint : verified_run.checkpoint.path
    run_dir = verified_run === nothing ?
        (
            basename(dirname(checkpoint_path)) == "checkpoints" ?
            dirname(dirname(checkpoint_path)) :
            dirname(checkpoint_path)
        ) : verified_run.run_dir
    expected_checkpoint_sha256 = verified_run === nothing ?
        options.expected_sha256 : verified_run.checkpoint.sha256
    required_checkpoint_update = verified_run === nothing ?
        options.required_update : verified_run.expected_updates
    checkpoint = load_analysis_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256;
        allow_legacy_provenance=options.allow_legacy_provenance,
    )
    lineage_training_checkpoint = nothing
    lineage_parent_checkpoint = nothing
    lineage_origin_config_snapshot = nothing
    lineage_origin_parent_checkpoint = nothing
    lineage_chain = NamedTuple[]
    checkpoint.production_schema &&
        options.allow_legacy_provenance &&
        error(
            "--allow-legacy-provenance was supplied for a production " *
            "checkpoint",
        )
    checkpoint.update == required_checkpoint_update ||
        error(
            "checkpoint update $(checkpoint.update) differs from required " *
            "$(required_checkpoint_update)",
        )
    if verified_run !== nothing
        checkpoint.production_schema || error(
            "a verified production run must use checkpoint schema version " *
            string(PRODUCTION_ARENA_CHECKPOINT_VERSION),
        )
        checkpoint.sha256 == verified_run.checkpoint.sha256 ||
            error("loaded checkpoint SHA-256 differs from verification")
        checkpoint.bytes == verified_run.checkpoint.bytes ||
            error("loaded checkpoint size differs from verification")
        String(checkpoint.checkpoint_kind) == "finalization" ||
            error("verified final checkpoint is not a finalization artifact")
        require_canonical_equal(
            checkpoint.config,
            verified_run.config,
            "checkpoint/config.json configuration binding",
        )
        checkpoint.payload_finalization === nothing &&
            error("finalization checkpoint has no finalization record")
        String(required_property(
            checkpoint.payload_finalization,
            :status,
            "checkpoint finalization record",
        )) == "finalization_checkpoint_complete" ||
            error("checkpoint finalization record is incomplete")
        Int(required_property(
            checkpoint.payload_finalization,
            :optimizer_steps_after_target,
            "checkpoint finalization record",
        )) == 0 ||
            error("checkpoint finalization record has post-target steps")
        require_canonical_equal(
            checkpoint_record(
                required_property(
                    checkpoint.payload_finalization,
                    :training_checkpoint,
                    "checkpoint finalization record",
                ),
                "checkpoint finalization training checkpoint",
            ),
            verified_run.training_checkpoint,
            "checkpoint finalization/verified training checkpoint",
        )
        require_canonical_equal(
            checkpoint_record(
                checkpoint.payload_parent_checkpoint,
                "checkpoint parent training checkpoint",
            ),
            verified_run.training_checkpoint,
            "checkpoint parent/verified training checkpoint",
        )
        require_same_existing_path(
            required_property(
                checkpoint.payload_finalization,
                :expected_results_path,
                "checkpoint finalization record",
            ),
            verified_run.results_path,
            "checkpoint finalization expected results path",
        )
        require_same_existing_path(
            required_property(
                checkpoint.payload_finalization,
                :expected_manifest_path,
                "checkpoint finalization record",
            ),
            verified_run.finalization_manifest_artifact.path,
            "checkpoint finalization expected manifest path",
        )
        require_canonical_equal(
            required_property(
                checkpoint.payload_finalization,
                :final_metrics,
                "checkpoint finalization record",
            ),
            required_property(
                verified_run.results,
                :final,
                "verified results",
            ),
            "checkpoint finalization/results final metrics",
        )
        checkpoint_teardown_raw = required_property(
            checkpoint.payload_finalization,
            :team_teardown,
            "checkpoint finalization record",
        )
        String(required_property(
            checkpoint_teardown_raw,
            :kind,
            "checkpoint finalization team teardown",
        )) == "team_teardown" ||
            error("checkpoint finalization team teardown kind differs")
        Int(required_property(
            checkpoint_teardown_raw,
            :update,
            "checkpoint finalization team teardown",
        )) == checkpoint.update ||
            error("checkpoint finalization team teardown update differs")
        require_canonical_equal(
            file_artifact_record(
                checkpoint_teardown_raw,
                "checkpoint finalization team teardown",
            ),
            verified_run.team_teardown_artifact,
            "checkpoint finalization/verified team teardown",
        )

        lineage_training_checkpoint = load_analysis_checkpoint(
            verified_run.training_checkpoint.path,
            verified_run.training_checkpoint.sha256,
        )
        lineage_training_checkpoint.production_schema ||
            error("verified training checkpoint is not production schema")
        String(lineage_training_checkpoint.checkpoint_kind) == "training" ||
            error("verified training payload is not a training checkpoint")
        lineage_training_checkpoint.update == checkpoint.update ||
            error("training/finalization checkpoint updates differ")
        require_checkpoint_state_equal(
            checkpoint,
            lineage_training_checkpoint,
            "training/finalization checkpoint state",
        )
        lineage_chain = bind_training_lineage(
            lineage_training_checkpoint,
            verified_run.training_checkpoint,
        )
        training_segment_start = Int(required_property(
            lineage_training_checkpoint.segment_state,
            :start_update,
            "verified training checkpoint segment state",
        ))
        training_segment_start == verified_run.segment_start_update ||
            error(
                "verified checkpoint policy segment start differs from the " *
                "training payload",
            )

        if verified_run.is_finalize_only
            origin_run_dir =
                dirname(dirname(lineage_training_checkpoint.path))
            lineage_origin_config_snapshot = read_json_object_snapshot(
                joinpath(origin_run_dir, "config.json"),
                "finalize-only origin config.json",
            )
            origin_config_document =
                lineage_origin_config_snapshot.document
            origin_config = required_property(
                origin_config_document,
                :config,
                "finalize-only origin config.json",
            )
            require_canonical_equal(
                origin_config,
                lineage_training_checkpoint.config,
                "origin config.json/training checkpoint configuration",
            )
            origin_run_id = String(required_property(
                origin_config,
                :run_id,
                "finalize-only origin config",
            ))
            (
                Sys.iswindows() ?
                lowercase(origin_run_id) == lowercase(basename(origin_run_dir)) :
                origin_run_id == basename(origin_run_dir)
            ) || error(
                "finalize-only origin config run ID differs from its run " *
                "directory",
            )
            origin_start_mode = replace(
                String(required_property(
                    origin_config,
                    :start_mode,
                    "finalize-only origin config",
                )),
                "_" => "-",
            )
            origin_start_mode in ("scratch", "resume") ||
                error("finalize-only origin start mode is unsupported")
            origin_scratch = Bool(required_property(
                origin_config,
                :scratch,
                "finalize-only origin config",
            ))
            origin_scratch == (origin_start_mode == "scratch") ||
                error(
                    "finalize-only origin config scratch/start-mode flags " *
                    "differ",
                )
            origin_parent_raw = property_or(
                origin_config_document,
                :parent_checkpoint,
                nothing,
            )
            payload_origin_parent =
                lineage_training_checkpoint.payload_parent_checkpoint
            if origin_scratch
                origin_parent_raw === nothing ||
                    error(
                        "scratch origin config unexpectedly has a parent " *
                        "checkpoint",
                    )
                payload_origin_parent === nothing ||
                    error(
                        "scratch origin training payload unexpectedly has a " *
                        "parent checkpoint",
                    )
                training_segment_start == 0 ||
                    error(
                        "scratch origin training segment does not start at zero",
                    )
            else
                origin_parent_raw === nothing &&
                    error("resume origin config has no parent checkpoint")
                payload_origin_parent === nothing &&
                    error("resume origin training payload has no parent checkpoint")
                lineage_origin_parent_checkpoint = checkpoint_record(
                    origin_parent_raw,
                    "finalize-only origin parent checkpoint",
                )
                payload_origin_parent_checkpoint = checkpoint_record(
                    payload_origin_parent,
                    "finalize-only training payload parent checkpoint",
                )
                require_canonical_equal(
                    payload_origin_parent_checkpoint,
                    lineage_origin_parent_checkpoint,
                    "origin config/training payload parent checkpoint",
                )
                verify_file_record(
                    lineage_origin_parent_checkpoint,
                    lineage_origin_parent_checkpoint.path,
                    "finalize-only origin parent checkpoint",
                )
                origin_parent_run_dir =
                    dirname(dirname(lineage_origin_parent_checkpoint.path))
                require_same_existing_path(
                    dirname(lineage_origin_parent_checkpoint.path),
                    joinpath(origin_parent_run_dir, "checkpoints"),
                    "finalize-only origin parent checkpoint directory",
                )
                lineage_origin_parent_checkpoint.update <
                    lineage_training_checkpoint.update ||
                    error(
                        "finalize-only origin parent update is not earlier " *
                        "than the target",
                    )
                training_segment_start ==
                    lineage_origin_parent_checkpoint.update ||
                    error(
                        "resume origin segment start differs from its parent " *
                        "checkpoint update",
                    )
            end
        elseif verified_run.launch_start_mode == "scratch"
            lineage_training_checkpoint.payload_parent_checkpoint === nothing ||
                error(
                    "scratch training payload unexpectedly has a parent " *
                    "checkpoint",
                )
        else
            lineage_training_checkpoint.payload_parent_checkpoint === nothing &&
                error("resume training payload has no parent checkpoint")
            require_canonical_equal(
                checkpoint_record(
                    lineage_training_checkpoint.payload_parent_checkpoint,
                    "resume training payload parent checkpoint",
                ),
                verified_run.parent_checkpoint,
                "resume training payload/verified parent checkpoint",
            )
        end

        if verified_run.launch_start_mode != "scratch"
            lineage_parent_checkpoint = verified_run.is_finalize_only ?
                lineage_training_checkpoint :
                load_analysis_checkpoint(
                    verified_run.parent_checkpoint.path,
                    verified_run.parent_checkpoint.sha256,
                )
            lineage_parent_checkpoint.production_schema ||
                error(
                    "$(verified_run.launch_start_mode) parent checkpoint is " *
                    "not production schema",
                )
            String(lineage_parent_checkpoint.checkpoint_kind) == "training" ||
                error(
                    "$(verified_run.launch_start_mode) parent payload is not " *
                    "a training checkpoint",
                )
            lineage_parent_checkpoint.update ==
                verified_run.parent_checkpoint.update ||
                error("parent payload/verification updates differ")
            if verified_run.launch_start_mode == "resume"
                lineage_parent_checkpoint.update < checkpoint.update ||
                    error(
                        "resume parent update is not earlier than the target",
                    )
            else
                lineage_parent_checkpoint.update == checkpoint.update ||
                    error(
                        "finalize-only parent/child checkpoint updates differ",
                    )
            end
        end
        residual_checkpoint =
            verified_run.parent_residual_finalization_checkpoint
        if residual_checkpoint !== nothing
            require_checkpoint_state_equal(
                residual_checkpoint,
                lineage_training_checkpoint,
                "parent residual finalization/training checkpoint state",
            )
        end
    end
    numbered_name = match(
        CHECKPOINT_FILENAME_PATTERN,
        basename(checkpoint.path),
    )
    numbered_name === nothing ||
        parse(Int, numbered_name.captures[1]) == checkpoint.update ||
        error(
            "checkpoint filename update differs from payload update " *
            "$(checkpoint.update)",
        )
    default_output_path = if verified_run === nothing
        joinpath(
            run_dir,
            "checkpoint_analysis_u" *
            lpad(string(max(checkpoint.update, 0)), 9, '0') *
            ".json",
        )
    else
        joinpath(
            dirname(verified_run.run_dir),
            "_analysis",
            (
                basename(verified_run.run_dir) *
                ".checkpoint_analysis_u" *
                lpad(string(max(checkpoint.update, 0)), 9, '0') *
                "." * checkpoint.sha256[1:12] * ".json"
            ),
        )
    end
    output_path = isempty(strip(options.output)) ?
        default_output_path : abspath(options.output)
    protected_output_paths =
        String[checkpoint.path, analysis_script_artifact.path]
    forbidden_output_directories = String[]
    if verified_run !== nothing
        # A verified run is an immutable, exact-set artifact.  Analysis output
        # must be a sibling artifact and must never make the verified run gain
        # an unmanifested file.
        push!(
            forbidden_output_directories,
            verified_run.run_dir,
        )
        append!(
            protected_output_paths,
            (
                verified_run.verification_path,
                verified_run.results_path,
                verified_run.manifest_path,
                verified_run.finalization_manifest_artifact.path,
                verified_run.config_path,
                verified_run.launch_manifest_artifact.path,
                verified_run.trace_artifact.path,
                verified_run.team_teardown_artifact.path,
            ),
        )
        for record in values(verified_run.checkpoints)
            push!(protected_output_paths, record.path)
        end
        for entry in lineage_chain
            push!(
                forbidden_output_directories,
                entry.run_dir,
            )
            append!(
                protected_output_paths,
                (
                    entry.record.path,
                    entry.config_snapshot.path,
                    entry.manifest_snapshot.path,
                ),
            )
            for manifest_record in values(
                entry.manifest_snapshot.records,
            )
                push!(protected_output_paths, manifest_record.path)
            end
        end
        if verified_run.is_finalize_only
            push!(
                forbidden_output_directories,
                (
                    lowercase(basename(dirname(
                        verified_run.parent_checkpoint.path,
                    ))) == "checkpoints" ?
                    dirname(dirname(verified_run.parent_checkpoint.path)) :
                    dirname(verified_run.parent_checkpoint.path)
                ),
            )
            lineage_origin_config_snapshot === nothing ||
                push!(
                    protected_output_paths,
                    lineage_origin_config_snapshot.path,
                )
            lineage_origin_parent_checkpoint === nothing ||
                push!(
                    protected_output_paths,
                    lineage_origin_parent_checkpoint.path,
                )
            verified_run.parent_residual_expected_results_path === nothing ||
                push!(
                    protected_output_paths,
                    verified_run.parent_residual_expected_results_path,
                )
            verified_run.parent_residual_expected_manifest_path === nothing ||
                push!(
                    protected_output_paths,
                    verified_run.parent_residual_expected_manifest_path,
                )
        end
        unique!(forbidden_output_directories)
    end
    validate_output_path(
        output_path;
        protected_paths=protected_output_paths,
        forbidden_directories=forbidden_output_directories,
    )
    model, model_source =
        infer_model(checkpoint.parameters, checkpoint.config)
    current_parameters =
        parameter_statistics(checkpoint.parameters)
    all_parameters_finite = all(
        value -> value.all_finite,
        values(current_parameters),
    )
    if !all_parameters_finite
        invalid_names = sort([
            name for (name, statistics) in current_parameters
            if !statistics.all_finite
        ])
        error(
            "checkpoint contains non-finite parameters: " *
            join(invalid_names, ", "),
        )
    end
    learned_source_fingerprint = property_or(
        checkpoint.config,
        :source_fingerprint,
        nothing,
    )
    analysis_source_fingerprint =
        current_training_source_fingerprint()
    if checkpoint.production_schema
        learned_source_fingerprint === nothing &&
            error("production checkpoint has no source fingerprint")
        analysis_source_fingerprint === nothing &&
            error("current training source fingerprint is unavailable")
    end
    source_fingerprint_match =
        learned_source_fingerprint === nothing ? nothing :
        (
            analysis_source_fingerprint === nothing ?
            nothing :
            require_sha256(
                learned_source_fingerprint,
                "checkpoint source fingerprint",
            ) == analysis_source_fingerprint
        )
    source_fingerprint_match === false &&
        checkpoint.production_schema &&
        error(
            "checkpoint source fingerprint differs from current training " *
            "sources",
        )
    production_contract = checkpoint.production_schema ?
        validate_production_config(
            checkpoint,
            verified_run,
            analysis_source_fingerprint,
        ) : nothing
    lineage_training_contract = nothing
    lineage_parent_contract = nothing
    lineage_chain_contracts = NamedTuple[]
    if lineage_training_checkpoint !== nothing
        lineage_training_contract = validate_production_config(
            lineage_training_checkpoint,
            nothing,
            analysis_source_fingerprint,
        )
        require_lineage_config_equal(
            lineage_training_checkpoint.config,
            checkpoint.config,
            "training/finalization config",
        )
        for (index, entry) in enumerate(lineage_chain)
            entry_contract = index == 1 ?
                lineage_training_contract :
                validate_production_config(
                    entry.checkpoint,
                    nothing,
                    analysis_source_fingerprint,
                )
            require_lineage_config_equal(
                entry.checkpoint.config,
                lineage_training_checkpoint.config,
                "training lineage config $index",
            )
            push!(lineage_chain_contracts, (;
                index,
                update=entry.checkpoint.update,
                run_id=entry_contract.run_id,
                production_contract_sha256=
                    entry_contract.production_contract_sha256,
                production_target_match=
                    entry_contract.production_target_match,
            ))
        end
    end
    if lineage_parent_checkpoint !== nothing
        lineage_parent_contract = verified_run.is_finalize_only ?
            lineage_training_contract :
            validate_production_config(
                lineage_parent_checkpoint,
                nothing,
                analysis_source_fingerprint,
            )
        require_lineage_config_equal(
            lineage_parent_checkpoint.config,
            checkpoint.config,
            "$(verified_run.launch_start_mode) parent/child config",
        )
        parent_run_id = String(required_property(
            lineage_parent_checkpoint.config,
            :run_id,
            "$(verified_run.launch_start_mode) parent config",
        ))
        (
            Sys.iswindows() ?
            lowercase(parent_run_id) ==
                lowercase(basename(dirname(dirname(
                    lineage_parent_checkpoint.path,
                )))) :
            parent_run_id ==
                basename(dirname(dirname(lineage_parent_checkpoint.path)))
        ) || error(
            "$(verified_run.launch_start_mode) parent config run ID differs " *
            "from its run directory",
        )
        parent_start_mode = String(required_property(
            lineage_parent_checkpoint.config,
            :start_mode,
            "$(verified_run.launch_start_mode) parent config",
        ))
        parent_start_mode in ("scratch", "resume") ||
            error("lineage parent start mode is unsupported")
        parent_scratch = Bool(required_property(
            lineage_parent_checkpoint.config,
            :scratch,
            "$(verified_run.launch_start_mode) parent config",
        ))
        parent_scratch == (parent_start_mode == "scratch") ||
            error("lineage parent scratch/start-mode flags differ")
        if verified_run.is_finalize_only
            parent_segment_start = Int(required_property(
                lineage_parent_checkpoint.segment_state,
                :start_update,
                "finalize-only parent segment state",
            ))
            0 <= parent_segment_start < checkpoint.update ||
                error(
                    "finalize-only parent segment start update is out of range",
                )
            parent_segment_start == verified_run.segment_start_update ||
                error(
                    "finalize-only verification segment start differs from " *
                    "the parent payload",
                )
            parent_scratch && parent_segment_start != 0 &&
                error(
                    "scratch finalize-only parent segment must start at zero",
                )
            parent_checkpoint_interval = Int(required_property(
                lineage_parent_checkpoint.config,
                :checkpoint_interval,
                "finalize-only parent config",
            ))
            parent_checkpoint_interval >= 1 ||
                error(
                    "finalize-only parent checkpoint interval is invalid",
                )
            parent_checkpoint_interval == verified_run.checkpoint_interval ||
                error(
                    "finalize-only verification checkpoint interval differs " *
                    "from the parent config",
                )
            expected_parent_manifest_updates = Int[]
            parent_scratch && push!(expected_parent_manifest_updates, 0)
            for update in (parent_segment_start + 1):checkpoint.update
                update % parent_checkpoint_interval == 0 &&
                    push!(expected_parent_manifest_updates, update)
            end
            checkpoint.update in expected_parent_manifest_updates ||
                push!(expected_parent_manifest_updates, checkpoint.update)
            sort!(unique!(expected_parent_manifest_updates))
            observed_parent_manifest_updates =
                sort!(collect(keys(verified_run.manifest_records)))
            observed_parent_manifest_updates ==
                expected_parent_manifest_updates ||
                error(
                    "finalize-only parent manifest cadence differs: observed=" *
                    string(observed_parent_manifest_updates) *
                    " expected=" * string(expected_parent_manifest_updates),
                )
        end
    end
    residual_checkpoint =
        verified_run === nothing ?
        nothing : verified_run.parent_residual_finalization_checkpoint
    if residual_checkpoint !== nothing
        residual_contract = validate_production_config(
            residual_checkpoint,
            nothing,
            analysis_source_fingerprint,
        )
        require_lineage_config_equal(
            residual_checkpoint.config,
            lineage_training_checkpoint.config,
            "parent residual finalization/training config",
        )
        residual_contract.production_contract_sha256 ==
            lineage_training_contract.production_contract_sha256 ||
            error(
                "parent residual finalization production contract differs",
            )
    end
    raw_production_target_match =
        production_contract === nothing ?
        false : production_contract.production_target_match
    recovered_production_target =
        verified_run !== nothing &&
        verified_run.is_finalize_only &&
        lineage_parent_contract !== nothing &&
        lineage_parent_contract.production_target_match
    effective_production_target_match =
        recovered_production_target || raw_production_target_match
    production_model_contract = checkpoint.production_schema ?
        validate_production_model_contract(
            model,
            checkpoint.parameters,
            checkpoint.config,
        ) : nothing
    content_provenance_complete = checkpoint.production_schema &&
        learned_source_fingerprint !== nothing &&
        analysis_source_fingerprint !== nothing &&
        source_fingerprint_match === true
    provenance_complete =
        verified_run !== nothing && content_provenance_complete
    dataset_from_config =
        property_or(checkpoint.config, :dataset_path, DEFAULT_DATASET)
    dataset_path = abspath(
        isempty(strip(options.dataset)) ?
        String(dataset_from_config) : options.dataset,
    )
    dataset_override_used = !isempty(strip(options.dataset))
    if isdir(dataset_path)
        push!(forbidden_output_directories, dataset_path)
        unique!(forbidden_output_directories)
    elseif isfile(dataset_path)
        push!(protected_output_paths, dataset_path)
    end
    validate_output_path(
        output_path;
        protected_paths=protected_output_paths,
        forbidden_directories=forbidden_output_directories,
    )
    dataset_preflight =
        analysis_dataset_binding_preflight(dataset_path)
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(
            options.states,
            options.validation_states,
        ),
    )
    dataset_content_sha256, dataset_integrity =
        bind_analysis_dataset(
            dataset_path,
            dataset,
            dataset_preflight,
        )
    dataset_metadata_witness =
        dataset_part_metadata_witness(
            dataset_path,
            dataset_preflight,
        )
    dataset_content_sha256_match = nothing
    dataset_integrity_match = nothing
    if checkpoint.production_schema
        dataset_content_sha256_match =
            dataset_content_sha256 ==
            production_contract.dataset_content_sha256
        dataset_content_sha256_match || error(
            "live dataset content SHA-256 differs from the checkpoint",
        )
        dataset_integrity_match =
            validate_recorded_dataset_integrity(
                required_property(
                    checkpoint.config,
                    :dataset_integrity,
                    "checkpoint config",
                ),
                dataset_integrity;
                override_used=dataset_override_used,
            )
        content_provenance_complete &=
            dataset_content_sha256_match && dataset_integrity_match
        provenance_complete &=
            dataset_content_sha256_match && dataset_integrity_match
    elseif dataset_override_used
        error(
            "a legacy checkpoint cannot bind a dataset override; omit " *
            "--dataset or use a production checkpoint",
        )
    end
    split = split_rows(dataset)
    training_rows_sha256 = panel_hash(split.training)
    expected_training_rows_sha256 = property_or(
        checkpoint.config,
        :training_rows_sha256,
        nothing,
    )
    checkpoint.production_schema &&
        expected_training_rows_sha256 === nothing &&
        error("production checkpoint has no training-row SHA-256")
    training_rows_hash_match =
        expected_training_rows_sha256 === nothing ?
        nothing :
        lowercase(String(expected_training_rows_sha256)) ==
        training_rows_sha256
    training_rows_hash_match === false &&
        error("training split SHA-256 differs from the checkpoint")
    content_provenance_complete &=
        !checkpoint.production_schema ||
        training_rows_hash_match === true
    provenance_complete &=
        !checkpoint.production_schema ||
        training_rows_hash_match === true
    panels = build_panels(
        options.panel,
        split,
        options.states,
        options.validation_states,
    )
    for (_, rows) in panels
        all(row -> 1 <= row <= length(dataset.action_counts), rows) ||
            error("panel contains a row outside the dataset")
    end
    exploration_record = routing_exploration(checkpoint.config)
    0.0f0 <= exploration_record.value < 1.0f0 ||
        error("routing exploration must be in [0,1)")
    eprop_saturation_contract = saturation_eprop_contract(
        checkpoint.config;
        production_schema=checkpoint.production_schema,
    )
    routing_floor_record = routing_entropy_floor(
        checkpoint.config;
        production_schema=checkpoint.production_schema,
    )
    architecture = if size(checkpoint.parameters.head_weight, 2) ==
        2 * model.node_dim
        "v2_normalized_workspace_selected_pool"
    elseif size(checkpoint.parameters.head_weight, 2) ==
        3 * model.node_dim
        "legacy_v1_workspace_query_all_block_pool"
    else
        error("unsupported checkpoint head layout")
    end
    actual_parameter_count = parameter_count(checkpoint.parameters)
    expected_parameter_count = property_or(
        checkpoint.config,
        :parameter_count,
        nothing,
    )
    parameter_count_match = expected_parameter_count === nothing ?
        nothing :
        actual_parameter_count == Int(expected_parameter_count)
    parameter_count_match === false &&
        error("checkpoint parameter count differs from its config")
    optimizer_summary = optimizer_statistics(checkpoint)
    structural_summary =
        structural_learning_statistics(checkpoint, model)
    transform_saturation =
        parameter_transform_saturation(checkpoint.parameters)
    training_trace_analysis = nothing
    if verified_run !== nothing
        isempty(lineage_chain) &&
            error("verified run has no bound training lineage")
        trace_owner = first(lineage_chain)
        trace_config = trace_owner.checkpoint.config
        trace_log_interval = Int(required_property(
            trace_config,
            :log_interval,
            "training trace owner config",
        ))
        trace_state_batch = Int(required_property(
            trace_config,
            :state_batch,
            "training trace owner config",
        ))
        parsed_trace = parse_bound_training_trace(
            verified_run.trace_artifact;
            segment_start=trace_owner.segment_start_update,
            terminal_update=trace_owner.record.update,
            log_interval=trace_log_interval,
            state_batch=trace_state_batch,
            require_v3=checkpoint.production_schema,
        )
        training_trace_analysis =
            training_trace_summary(parsed_trace)
    end

    panel_results = Dict{String,Any}()
    for (name, rows) in panels
        println(
            stderr,
            "analyzing $name: $(length(rows)) states, " *
            "$(sum(dataset.action_counts[rows])) candidates",
        )
        evaluated = evaluate_panel(
            name,
            rows,
            dataset,
            model,
            checkpoint.parameters,
            checkpoint.states,
            checkpoint.config,
            exploration_record.value;
            enforce_parity=options.enforce_parity,
            eprop_contract=eprop_saturation_contract,
            routing_floor_record,
        )
        expected_panel_hash = if name == "fixed_training" &&
            length(rows) == Int(property_or(
                checkpoint.config,
                :training_eval_states,
                -1,
            ))
            property_or(
                checkpoint.config,
                :training_panel_rows_sha256,
                nothing,
            )
        elseif name == "fixed_validation" &&
               length(rows) == Int(property_or(
                   checkpoint.config,
                   :validation_eval_states,
                   -1,
               ))
            property_or(
                checkpoint.config,
                :validation_panel_rows_sha256,
                nothing,
            )
        else
            nothing
        end
        panel_hash_match = expected_panel_hash === nothing ?
            nothing :
            lowercase(String(expected_panel_hash)) ==
            evaluated.panel.rows_sha256
        panel_hash_match === false &&
            error("$name rows SHA-256 differs from the checkpoint")
        panel_results[name] = merge(evaluated, (;
            checkpoint_binding=(;
                expected_rows_sha256=expected_panel_hash,
                rows_sha256_match=panel_hash_match,
            ),
        ))
    end
    results_metric_binding = nothing
    checkpoint_curve = nothing
    if verified_run !== nothing
        initial_metrics = required_property(
            verified_run.results,
            :initial,
            "verified results",
        )
        final_metrics = required_property(
            verified_run.results,
            :final,
            "verified results",
        )
        recorded_panel_states = Int(required_property(
            checkpoint.config,
            :training_eval_states,
            "checkpoint config",
        ))
        recorded_panel_rows = fixed_subset(
            split.training,
            recorded_panel_states,
            TRAIN_EVAL_SEED,
        )
        recorded_panel_rows_sha256 = panel_hash(recorded_panel_rows)
        recorded_panel_rows_sha256 ==
            lowercase(String(required_property(
                checkpoint.config,
                :training_panel_rows_sha256,
                "checkpoint config",
            ))) || error(
                "reconstructed results panel differs from checkpoint config",
            )
        checkpoint_curve = intermediate_checkpoint_curve(
            lineage_chain,
            recorded_panel_rows,
            dataset,
            model,
            checkpoint.states,
            checkpoint.config,
            exploration_record.value,
        )
        existing_fixed_panel = get(
            panel_results,
            "fixed_training",
            nothing,
        )
        final_summary = if existing_fixed_panel !== nothing &&
            existing_fixed_panel.panel.rows_sha256 ==
                recorded_panel_rows_sha256
            existing_fixed_panel.metrics["full"]
        else
            evaluate_full_metrics_only(
                recorded_panel_rows,
                dataset,
                model,
                checkpoint.parameters,
                checkpoint.config,
                exploration_record.value,
            )
        end
        recomputed_final =
            compare_results_metrics(final_summary, final_metrics)
        recomputed_final.match || error(
            "recomputed final fixed-panel metrics differ from " *
            "verified results.json",
        )
        checkpoint.initial_parameters === nothing && error(
            "verified production checkpoint has no initial parameters",
        )
        initial_summary = evaluate_full_metrics_only(
            recorded_panel_rows,
            dataset,
            model,
            checkpoint.initial_parameters,
            checkpoint.config,
            exploration_record.value,
        )
        recomputed_initial =
            compare_results_metrics(initial_summary, initial_metrics)
        recomputed_initial.match || error(
            "recomputed initial fixed-panel metrics differ from " *
            "verified results.json",
        )
        results_metric_binding = (;
            results_path=verified_run.results_path,
            results_sha256=verified_run.results_sha256,
            results_bytes=verified_run.results_bytes,
            artifact_bound_by_verification=true,
            verifier_recomputed_metrics=
                property_or(
                    verified_run.verification,
                    :metrics_verified,
                    false,
                ) === true,
            initial=initial_metrics,
            final=final_metrics,
            recorded_panel=(;
                states=length(recorded_panel_rows),
                rows_sha256=recorded_panel_rows_sha256,
            ),
            recomputed_initial,
            recomputed_final,
            metric_schema=(;
                listnet_cross_entropy="results.listnet_loss",
                teacher_entropy="results.teacher_entropy",
                listnet_kl="results.listnet_kl",
                top1_agreement="results.top1_agreement",
                ndcg="results.ndcg",
                pairwise_accuracy="results.pairwise_accuracy",
            ),
        )
    end
    file_sha256(checkpoint.path) == checkpoint.sha256 ||
        error("checkpoint changed during analysis")
    filesize(checkpoint.path) == checkpoint.bytes ||
        error("checkpoint size changed during analysis")
    analysis_dataset_binding_preflight(dataset_path) ==
        dataset_preflight ||
        error("dataset binding changed during analysis")
    dataset_part_metadata_witness(
        dataset_path,
        dataset_preflight,
    ) == dataset_metadata_witness ||
        error("dataset part metadata changed during analysis")
    current_training_source_fingerprint() ==
        analysis_source_fingerprint ||
        error("training sources changed during analysis")
    if verified_run !== nothing
        checkpoint_dir = joinpath(verified_run.run_dir, "checkpoints")
        live_entries = readdir(checkpoint_dir; join=true)
        all(isfile, live_entries) ||
            error(
                "checkpoints directory gained a non-file entry during analysis",
            )
        live_paths = Set(
            normalized_existing_path(path, "live checkpoint entry")
            for path in live_entries
        )
        verified_paths = Set(
            normalized_existing_path(
                record.path,
                "verified checkpoint",
            )
            for record in values(verified_run.checkpoints)
            if dirname(normalized_existing_path(
                record.path,
                "verified checkpoint",
            )) == normalized_existing_path(
                checkpoint_dir,
                "checkpoint directory",
            )
        )
        push!(
            verified_paths,
            normalized_existing_path(
                verified_run.checkpoint.path,
                "verified finalization checkpoint",
            ),
        )
        expected_entry_paths = Set(
            normalized_declared_path(record.path)
            for record in values(verified_run.checkpoints)
            if dirname(normalized_existing_path(
                record.path,
                "verified checkpoint",
            )) == normalized_existing_path(
                checkpoint_dir,
                "checkpoint directory",
            )
        )
        push!(
            expected_entry_paths,
            normalized_declared_path(verified_run.checkpoint.path),
        )
        length(live_entries) == length(expected_entry_paths) ||
            error("checkpoint alias or extra entry appeared during analysis")
        length(verified_paths) == length(expected_entry_paths) ||
            error(
                "verified checkpoint resolved-path alias appeared during " *
                "analysis",
            )
        length(live_paths) == length(live_entries) ||
            error(
                "live checkpoint resolved-path alias appeared during analysis",
            )
        Set(normalized_declared_path.(live_entries)) ==
            expected_entry_paths ||
            error("checkpoint entry paths changed during analysis")
        live_paths == verified_paths ||
            error("checkpoint artifact set changed during analysis")
        for (update, record) in verified_run.checkpoints
            verify_file_record(
                record,
                record.path,
                "verified checkpoint $update end-of-analysis",
            )
        end
        verify_file_record(
            verified_run.checkpoint,
            verified_run.checkpoint.path,
            "verified finalization checkpoint end-of-analysis",
        )
        verified_run.parent_checkpoint === nothing ||
            verify_file_record(
                verified_run.parent_checkpoint,
                verified_run.parent_checkpoint.path,
                "verified parent checkpoint end-of-analysis",
            )
        if verified_run.is_finalize_only
            parent_checkpoint_dir =
                dirname(verified_run.parent_checkpoint.path)
            for (update, record) in verified_run.manifest_records
                verify_file_record(
                    record,
                    record.path,
                    "parent manifest checkpoint $update end-of-analysis",
                )
            end
            if verified_run.parent_residual_finalization_record !== nothing
                verify_file_record(
                    verified_run.parent_residual_finalization_record,
                    verified_run.parent_residual_finalization_record.path,
                    "parent residual finalization end-of-analysis",
                )
            end
            parent_entries = readdir(parent_checkpoint_dir; join=true)
            all(isfile, parent_entries) ||
                error(
                    "parent checkpoint directory gained a non-file entry " *
                    "during analysis",
                )
            parent_training_entries = String[]
            parent_finalization_entries = String[]
            for path in parent_entries
                training_match =
                    match(CHECKPOINT_FILENAME_PATTERN, basename(path))
                finalization_match = match(
                    FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
                    basename(path),
                )
                if training_match !== nothing
                    push!(parent_training_entries, path)
                elseif finalization_match !== nothing &&
                       parse(Int, only(finalization_match.captures)) ==
                           verified_run.expected_updates
                    push!(parent_finalization_entries, path)
                else
                    error(
                        "parent checkpoint directory artifact set changed " *
                        "during analysis",
                    )
                end
            end
            length(parent_finalization_entries) <= 1 ||
                error(
                    "parent checkpoint directory gained duplicate " *
                    "finalization artifacts during analysis",
                )
            expected_parent_finalization_paths =
                verified_run.parent_residual_finalization_record === nothing ?
                Set{String}() :
                Set([
                    normalized_declared_path(
                        verified_run.
                            parent_residual_finalization_record.path,
                    ),
                ])
            observed_parent_finalization_paths =
                Set(normalized_declared_path.(parent_finalization_entries))
            observed_parent_finalization_paths ==
                expected_parent_finalization_paths ||
                error(
                    "parent residual finalization membership changed during " *
                    "analysis",
                )
            length(parent_training_entries) ==
                length(verified_run.manifest_records) ||
                error(
                    "parent checkpoint directory gained a training alias, " *
                    "extra, or omission during analysis",
                )
            Set(normalized_declared_path.(parent_training_entries)) ==
                Set(
                    normalized_declared_path(record.path)
                    for record in values(verified_run.manifest_records)
                ) ||
                error(
                    "parent checkpoint directory training paths changed " *
                    "during analysis",
                )
            live_parent_real_paths = Set(
                normalized_existing_path(
                    path,
                    "parent live checkpoint end-of-analysis",
                )
                for path in parent_training_entries
            )
            manifest_parent_real_paths = Set(
                normalized_existing_path(
                    record.path,
                    "parent manifest checkpoint end-of-analysis",
                )
                for record in values(verified_run.manifest_records)
            )
            length(live_parent_real_paths) ==
                length(parent_training_entries) ||
                error(
                    "parent live training checkpoint resolved-path alias " *
                    "appeared during analysis",
                )
            length(manifest_parent_real_paths) ==
                length(verified_run.manifest_records) ||
                error(
                    "parent manifest checkpoint resolved-path alias appeared " *
                    "during analysis",
                )
            live_parent_real_paths == manifest_parent_real_paths || error(
                "parent checkpoint directory training artifact set changed " *
                "during analysis",
            )
            lineage_origin_config_snapshot === nothing &&
                error("finalize-only origin config snapshot is unavailable")
            filesize(lineage_origin_config_snapshot.path) ==
                lineage_origin_config_snapshot.bytes ||
                error(
                    "finalize-only origin config byte size changed during " *
                    "analysis",
                )
            file_sha256(lineage_origin_config_snapshot.path) ==
                lineage_origin_config_snapshot.sha256 ||
                error(
                    "finalize-only origin config changed during analysis",
                )
            lineage_origin_parent_checkpoint === nothing ||
                verify_file_record(
                    lineage_origin_parent_checkpoint,
                    lineage_origin_parent_checkpoint.path,
                    "finalize-only origin parent checkpoint end-of-analysis",
                )
            verified_run.parent_residual_expected_results_path === nothing ||
                !ispath(
                    verified_run.parent_residual_expected_results_path,
                ) || error(
                    "finalize-only parent residual results artifact appeared " *
                    "during analysis",
                )
            verified_run.parent_residual_expected_manifest_path === nothing ||
                !ispath(
                    verified_run.parent_residual_expected_manifest_path,
                ) || error(
                    "finalize-only parent residual manifest appeared during " *
                    "analysis",
                )
        end
        file_sha256(verified_run.trace_artifact.path) ==
            verified_run.trace_artifact.sha256 ||
            error("training trace changed during analysis")
        file_sha256(verified_run.team_teardown_artifact.path) ==
            verified_run.team_teardown_artifact.sha256 ||
            error("team teardown changed during analysis")
        file_sha256(verified_run.verification_path) ==
            verified_run.verification_sha256 ||
            error("verification.json changed during analysis")
        file_sha256(verified_run.results_path) ==
            verified_run.results_sha256 ||
            error("results.json changed during analysis")
        file_sha256(verified_run.manifest_path) ==
            verified_run.manifest_sha256 ||
            error("checkpoint manifest changed during analysis")
        file_sha256(
            verified_run.finalization_manifest_artifact.path,
        ) == verified_run.finalization_manifest_artifact.sha256 ||
            error("finalization manifest changed during analysis")
        file_sha256(verified_run.config_path) ==
            verified_run.config_sha256 ||
            error("config.json changed during analysis")
        file_sha256(verified_run.launch_manifest_artifact.path) ==
            verified_run.launch_manifest_artifact.sha256 ||
            error("launch manifest changed during analysis")
        for artifact in verified_run.launch_code_artifacts
            filesize(artifact.path) == artifact.bytes ||
                error(
                    "launch code artifact $(artifact.name) size changed " *
                    "during analysis",
                )
            file_sha256(artifact.path) == artifact.sha256 ||
                error(
                    "launch code artifact $(artifact.name) changed during " *
                    "analysis",
                )
        end
        for (index, entry) in enumerate(lineage_chain)
            verify_file_record(
                entry.record,
                entry.record.path,
                "training lineage checkpoint $index end-of-analysis",
            )
            filesize(entry.config_snapshot.path) ==
                entry.config_snapshot.bytes ||
                error(
                    "training lineage config $index byte size changed during " *
                    "analysis",
                )
            file_sha256(entry.config_snapshot.path) ==
                entry.config_snapshot.sha256 ||
                error(
                    "training lineage config $index changed during analysis",
                )
            filesize(entry.manifest_snapshot.path) ==
                entry.manifest_snapshot.bytes ||
                error(
                    "training lineage manifest $index byte size changed " *
                    "during analysis",
                )
            file_sha256(entry.manifest_snapshot.path) ==
                entry.manifest_snapshot.sha256 ||
                error(
                    "training lineage manifest $index changed during analysis",
                )
            verify_lineage_checkpoint_directory_snapshot!(
                entry.checkpoint_directory_snapshot,
                "training lineage checkpoint directory $index " *
                "end-of-analysis",
            )
        end
    end
    config_canonical_sha256 = checkpoint.config === nothing ?
        nothing :
        bytes2hex(sha256(codeunits(String(JSON3.write(
            checkpoint.config,
        )))))
    runtime_provenance_sha256 = production_contract === nothing ?
        nothing :
        bytes2hex(sha256(codeunits(String(JSON3.write(
            production_contract.runtime,
        )))))
    filesize(analysis_script_artifact.path) ==
        analysis_script_artifact.bytes ||
        error("analysis script byte size changed during analysis")
    file_sha256(analysis_script_artifact.path) ==
        analysis_script_artifact.sha256 ||
        error("analysis script changed during analysis")
    report = (;
        format=ANALYSIS_FORMAT,
        version=ANALYSIS_VERSION,
        analysis_tool=(;
            format=ANALYSIS_FORMAT,
            version=ANALYSIS_VERSION,
            script=analysis_script_artifact,
        ),
        checkpoint=(;
            path=checkpoint.path,
            sha256=checkpoint.sha256,
            bytes=checkpoint.bytes,
            schema=checkpoint.schema,
            format=checkpoint.checkpoint_format,
            version=checkpoint.checkpoint_version,
            update=checkpoint.update,
            checkpoint_kind=checkpoint.checkpoint_kind,
            required_update=required_checkpoint_update,
            required_update_satisfied=
                checkpoint.update == required_checkpoint_update,
            learned_source_fingerprint,
            analysis_source_fingerprint,
            source_fingerprint_match,
            production_schema=checkpoint.production_schema,
            provenance_complete,
            content_provenance_complete,
            verified_run_binding=verified_run === nothing ? nothing : (;
                verification_path=verified_run.verification_path,
                verification_sha256=
                    verified_run.verification_sha256,
                verification_bytes=
                    verified_run.verification_bytes,
                caller_pinned_verification_sha256=
                    isempty(options.expected_verification_sha256) ?
                    nothing :
                    options.expected_verification_sha256,
                trust_model=
                    isempty(options.expected_verification_sha256) ?
                    (
                        "local unsigned verification v2 plus reciprocal live " *
                        "artifact hashes"
                    ) :
                    "caller-pinned verification SHA-256",
                verification_format=RUN_VERIFICATION_FORMAT,
                verification_version=RUN_VERIFICATION_VERSION,
                verified_status="verified_complete",
                metrics_verified=true,
                run_id=verified_run.run_id,
                start_mode=verified_run.launch_start_mode,
                finalize_only_lineage=verified_run.is_finalize_only,
                final_checkpoint_path=
                    verified_run.checkpoint.path,
                final_checkpoint_update=
                    verified_run.checkpoint.update,
                final_checkpoint_sha256=
                    verified_run.checkpoint.sha256,
                final_checkpoint_bytes=
                    verified_run.checkpoint.bytes,
                training_checkpoint=
                    verified_run.training_checkpoint,
                checkpoint_manifest_path=
                    verified_run.manifest_path,
                checkpoint_manifest_sha256=
                    verified_run.manifest_sha256,
                checkpoint_manifest_bytes=
                    verified_run.manifest_bytes,
                finalization_manifest_path=
                    verified_run.finalization_manifest_artifact.path,
                finalization_manifest_sha256=
                    verified_run.finalization_manifest_artifact.sha256,
                finalization_manifest_bytes=
                    verified_run.finalization_manifest_artifact.bytes,
                config_path=verified_run.config_path,
                config_sha256=verified_run.config_sha256,
                config_bytes=verified_run.config_bytes,
                launch_manifest_path=
                    verified_run.launch_manifest_artifact.path,
                launch_manifest_sha256=
                    verified_run.launch_manifest_artifact.sha256,
                launch_manifest_bytes=
                    verified_run.launch_manifest_artifact.bytes,
                launch_output_root=verified_run.launch_output_root,
                launch_expected_contract_sha256=bytes2hex(sha256(
                    codeunits(String(JSON3.write(required_property(
                        verified_run.launch_manifest,
                        :expected_contract,
                        "verified launch manifest",
                    )))),
                )),
                launch_environment_sha256=bytes2hex(sha256(
                    codeunits(String(JSON3.write(required_property(
                        verified_run.launch_manifest,
                        :environment,
                        "verified launch manifest",
                    )))),
                )),
                launch_code_artifacts=
                    verified_run.launch_code_artifacts,
                results_path=verified_run.results_path,
                results_sha256=verified_run.results_sha256,
                trace=verified_run.trace_artifact,
                team_teardown=verified_run.team_teardown_artifact,
                parent_checkpoint=verified_run.parent_checkpoint,
                lineage_origin_config=
                    lineage_origin_config_snapshot === nothing ?
                    nothing :
                    (;
                        path=lineage_origin_config_snapshot.path,
                        bytes=lineage_origin_config_snapshot.bytes,
                        sha256=lineage_origin_config_snapshot.sha256,
                    ),
                lineage_origin_parent_checkpoint,
                training_lineage=[
                    (;
                        index,
                        update=entry.record.update,
                        checkpoint_path=entry.record.path,
                        checkpoint_bytes=entry.record.bytes,
                        checkpoint_sha256=entry.record.sha256,
                        run_dir=entry.run_dir,
                        run_id=entry.run_id,
                        start_mode=entry.start_mode,
                        scratch=entry.scratch,
                        segment_start_update=
                            entry.segment_start_update,
                        config_path=entry.config_snapshot.path,
                        config_bytes=entry.config_snapshot.bytes,
                        config_sha256=entry.config_snapshot.sha256,
                        manifest_path=entry.manifest_snapshot.path,
                        manifest_bytes=entry.manifest_snapshot.bytes,
                        manifest_sha256=entry.manifest_snapshot.sha256,
                        manifest_record_count=length(
                            entry.manifest_snapshot.records,
                        ),
                        manifest_checkpoint_interval=
                            entry.manifest_contract.checkpoint_interval,
                        manifest_maximum_updates=
                            entry.manifest_contract.maximum_updates,
                        checkpoint_directory_artifact_count=length(
                            entry.checkpoint_directory_snapshot.artifacts,
                        ),
                        checkpoint_directory_exact_set_verified=true,
                        parent_checkpoint=entry.parent_checkpoint,
                        production_contract_sha256=
                            lineage_chain_contracts[index].production_contract_sha256,
                        production_target_match=
                            lineage_chain_contracts[index].production_target_match,
                    )
                    for (index, entry) in enumerate(lineage_chain)
                ],
                parent_residual_finalization_checkpoint=
                    verified_run.parent_residual_finalization_record,
                parent_residual_expected_results_path=
                    verified_run.parent_residual_expected_results_path,
                parent_residual_expected_manifest_path=
                    verified_run.parent_residual_expected_manifest_path,
                parent_residual_outputs_absent=
                    verified_run.is_finalize_only ? true : nothing,
                parent_manifest_records_verified=
                    verified_run.is_finalize_only ?
                    length(verified_run.manifest_records) :
                    length(verified_run.checkpoints),
                exact_live_checkpoint_set_verified=true,
                verifier_fixed_panel_recomputation=
                    verified_run.fixed_panel_recomputation,
                verifier_runtime=verified_run.verifier_runtime,
            ),
            unverified_checkpoint_explicitly_allowed=
                options.allow_unverified_checkpoint,
            legacy_provenance_explicitly_allowed=
                options.allow_legacy_provenance,
            production_analysis=
                verified_run !== nothing &&
                provenance_complete &&
                effective_production_target_match,
            production_target_match=
                effective_production_target_match,
            raw_checkpoint_production_target_match=
                raw_production_target_match,
            recovered_production_target,
            production_claim_note=
                effective_production_target_match ?
                nothing :
                (
                    "the artifact may have complete content/run provenance, " *
                    "but it is not the configured 100k scratch production target"
                ),
            provenance_note=
                verified_run !== nothing && provenance_complete ?
                nothing :
                (
                    "explicit legacy or direct-unverified analysis; this " *
                    "report is not a production-quality claim"
                ),
        ),
        dataset=(;
            path=dataset_path,
            override_used=dataset_override_used,
            content_sha256=dataset_content_sha256,
            expected_content_sha256=
                production_contract === nothing ?
                nothing :
                production_contract.dataset_content_sha256,
            content_sha256_match=dataset_content_sha256_match,
            integrity=dataset_integrity,
            integrity_match=dataset_integrity_match,
            metadata_witness_entries=
                length(dataset_metadata_witness),
            metadata_stable_during_analysis=true,
            split_kind=split.kind,
            training_states=length(split.training),
            validation_states=length(split.validation),
            training_rows_sha256,
            expected_training_rows_sha256,
            training_rows_sha256_match=training_rows_hash_match,
        ),
        provenance=(;
            complete=provenance_complete,
            content_complete=content_provenance_complete,
            production_verified=
                verified_run !== nothing &&
                provenance_complete &&
                effective_production_target_match,
            verified_provenance_complete=
                verified_run !== nothing && provenance_complete,
            checkpoint_sha256=checkpoint.sha256,
            config_canonical_sha256,
            dataset_content_sha256,
            source_fingerprint=learned_source_fingerprint,
            runtime_provenance_sha256,
            runtime_provenance=
                production_contract === nothing ?
                nothing : production_contract.runtime,
            production_contract_sha256=
                production_contract === nothing ?
                nothing :
                production_contract.production_contract_sha256,
            metrics_recomputed_from_bound_checkpoint=true,
            metrics_artifact_binding=results_metric_binding === nothing ?
                "unverified_direct_checkpoint" :
                "verification_v2_results_sha256_and_final_recomputation",
        ),
        model=(;
            reconstruction_source=model_source,
            architecture,
            topology=graph_topology(model, checkpoint.parameters),
            head_input_width=size(
                checkpoint.parameters.head_weight,
                2,
            ),
            parameter_count=actual_parameter_count,
            expected_parameter_count,
            parameter_count_match,
            routing_exploration=exploration_record,
            production_contract=production_model_contract,
        ),
        metrics_binding=results_metric_binding,
        training_trace=training_trace_analysis,
        intermediate_checkpoint_curve=checkpoint_curve,
        panels=panel_results,
        parameters=(;
            all_finite=all_parameters_finite,
            current=current_parameters,
            growth_from_initial=parameter_growth(
                checkpoint.parameters,
                checkpoint.initial_parameters,
            ),
            transform_saturation,
            optimizer=optimizer_summary,
            structural_learning=structural_summary,
        ),
        implementation_contract=(;
            dependencies=(
                "SerialWorkspaceSNN.binary_rails",
                "SerialWorkspaceSNN.vectorized_synapse_scan",
                "SerialWorkspaceSNN.standardized_route_probabilities",
                "SerialWorkspaceSNN.bounded_workspace_decay",
                "BeatFirstTrainingCore.load_teacher_dataset",
                "BeatFirstTrainingCore.pack_batch!",
            ),
            v2_readout=(
                "query=tanh(query_scale*RMSNorm(query_weight*rails)); " *
                "query is routing-only",
                "selected_pool is the final-cycle hard-selected block mean",
                "head=[RMSNorm(workspace);RMSNorm(selected_pool)]",
                "hidden=tanh(hidden_scale*RMSNorm(head_pre))",
            ),
            routing=(
                "base=softmax(candidate-standardized score/temperature)",
                "policy=(1-exploration)*base+exploration/blocks",
                "hard inference=deterministic raw-score top-k",
            ),
            saturation_diagnostics=(;
                eligibility_sampling=(
                    "terminal membrane e-prop trace on a deterministic " *
                    "approximately equidistant edge-tape sample; no third factor"
                ),
                eligibility_contract=eprop_saturation_contract,
                routing_entropy_floor=routing_floor_record,
                training_trace=(
                    "hash-bound exact-v3 TSV with strict cadence, type, " *
                    "monotonicity, and finiteness validation"
                ),
                checkpoint_curve=(
                    "every causally bound manifested training checkpoint " *
                    "evaluated on the same fixed training panel"
                ),
                optimizer=(
                    "checkpoint-moment-implied unprojected direction; not " *
                    "a measured fresh-gradient optimizer step"
                ),
            ),
            ablations=(;
                full="unaltered deterministic inference",
                workspace_off=(
                    "disable workspace state, writes, feedback, and its " *
                    "readout branch while retaining routing and the " *
                    "selected-pool branch"
                ),
                selected_pool_off=(
                    "zero only the final selected-pool readout branch"
                ),
                both_off=(
                    "apply workspace_off and also zero the selected-pool " *
                    "readout; legacy checkpoints retain their direct query " *
                    "branch to expose that shortcut"
                ),
                synapse_off=(
                    "zero recurrent synaptic messages while preserving " *
                    "leak, workspace, feedback, and readout"
                ),
                memory_off=(
                    "reset membrane before every later cycle, replace delay " *
                    "interpolation by the current spike at full gain, and " *
                    "set workspace decay to zero; current-cycle synapses, " *
                    "writes, feedback, and readout remain"
                ),
            ),
        ),
    )
    atomic_write_json(
        output_path,
        report;
        protected_paths=protected_output_paths,
        forbidden_directories=forbidden_output_directories,
        precommit_check=() -> verify_bound_analysis_inputs!(
            checkpoint,
            dataset_path,
            dataset_preflight,
            dataset_metadata_witness,
            analysis_source_fingerprint,
            verified_run,
            lineage_chain,
            lineage_origin_config_snapshot,
            lineage_origin_parent_checkpoint,
            analysis_script_artifact,
        ),
    )
    output_sha256 = file_sha256(output_path)
    output_bytes = filesize(output_path)
    println(JSON3.write((;
        output_path,
        output_bytes,
        output_sha256,
        checkpoint_update=checkpoint.update,
        checkpoint_sha256=checkpoint.sha256,
        verified_run=verified_run !== nothing,
        provenance_complete,
        panels=collect(keys(panel_results)),
        all_parameters_finite,
    )))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
