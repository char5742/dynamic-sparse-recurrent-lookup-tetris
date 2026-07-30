using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const MODEL_SEED = UInt64(2026072703)
const SPLIT_SEED = UInt64(2026071817)
const SAMPLER_SEED = UInt64(2026071801) + UInt64(0x9e3779b97f4a7c15)
const TRAIN_EVAL_SEED = UInt64(2026071801) + UInt64(0x101)
const ROUTING_SEED = UInt64(0x524f555445534545)
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const CHECKPOINT_FORMAT = "serial-workspace-snn-arena-checkpoint"
const CHECKPOINT_VERSION = 3
const PRODUCTION_CONTRACT_VERSION = 1
const TRAINING_CHECKPOINT_FILENAME_PATTERN =
    r"^checkpoint_([0-9]{9})\.jld2$"
const FINALIZATION_CHECKPOINT_FILENAME_PATTERN =
    r"^finalization_checkpoint_([0-9]{9})\.jld2$"
const CHECKPOINT_MANIFEST_PROPERTIES =
    Set(("kind", "path", "bytes", "sha256", "update"))
const CHECKPOINT_MANIFEST_RECORD_TYPE = NamedTuple{
    (:kind, :path, :bytes, :sha256, :update),
    Tuple{String,String,Int,String,Int},
}
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = UInt32(0x00000400)
const WINDOWS_INVALID_FILE_ATTRIBUTES = typemax(UInt32)
const WINDOWS_RESERVED_BASENAMES = Set((
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
))
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

mutable struct ComponentLossTelemetry
    window_old_q_loss_sum::Float64
    window_q_huber_loss_sum::Float64
    window_margin_loss_sum::Float64
    window_raw_top_gap_loss_sum::Float64
    window_death_loss_sum::Float64
    window_quantile_teacher_loss_sum::Float64
    window_geometry_loss_sum::Float64
    window_line_clear_loss_sum::Float64
    window_max_height_loss_sum::Float64
    window_holes_loss_sum::Float64
    window_cavities_loss_sum::Float64
    window_structure_loss_sum::Float64
    active_window_old_q_loss_sum::Float64
    active_window_q_huber_loss_sum::Float64
    active_window_margin_loss_sum::Float64
    active_window_raw_top_gap_loss_sum::Float64
    active_window_death_loss_sum::Float64
    active_window_quantile_teacher_loss_sum::Float64
    active_window_geometry_loss_sum::Float64
    active_window_line_clear_loss_sum::Float64
    active_window_max_height_loss_sum::Float64
    active_window_holes_loss_sum::Float64
    active_window_cavities_loss_sum::Float64
    active_window_structure_loss_sum::Float64
    old_q_loss::Float64
    q_huber_loss::Float64
    margin_loss::Float64
    raw_top_gap_loss::Float64
    death_loss::Float64
    quantile_teacher_loss::Float64
    geometry_loss::Float64
    line_clear_loss::Float64
    max_height_loss::Float64
    holes_loss::Float64
    cavities_loss::Float64
    structure_loss::Float64
end

ComponentLossTelemetry() = ComponentLossTelemetry(
    ntuple(_ -> 0.0, 36)...,
)

function component_loss_snapshot(telemetry::ComponentLossTelemetry)
    return (;
        window_old_q_loss_sum=telemetry.window_old_q_loss_sum,
        window_q_huber_loss_sum=telemetry.window_q_huber_loss_sum,
        window_margin_loss_sum=telemetry.window_margin_loss_sum,
        window_raw_top_gap_loss_sum=
            telemetry.window_raw_top_gap_loss_sum,
        window_death_loss_sum=telemetry.window_death_loss_sum,
        window_quantile_teacher_loss_sum=
            telemetry.window_quantile_teacher_loss_sum,
        window_geometry_loss_sum=telemetry.window_geometry_loss_sum,
        window_line_clear_loss_sum=
            telemetry.window_line_clear_loss_sum,
        window_max_height_loss_sum=
            telemetry.window_max_height_loss_sum,
        window_holes_loss_sum=telemetry.window_holes_loss_sum,
        window_cavities_loss_sum=
            telemetry.window_cavities_loss_sum,
        window_structure_loss_sum=
            telemetry.window_structure_loss_sum,
        active_window_old_q_loss_sum=
            telemetry.active_window_old_q_loss_sum,
        active_window_q_huber_loss_sum=
            telemetry.active_window_q_huber_loss_sum,
        active_window_margin_loss_sum=
            telemetry.active_window_margin_loss_sum,
        active_window_raw_top_gap_loss_sum=
            telemetry.active_window_raw_top_gap_loss_sum,
        active_window_death_loss_sum=
            telemetry.active_window_death_loss_sum,
        active_window_quantile_teacher_loss_sum=
            telemetry.active_window_quantile_teacher_loss_sum,
        active_window_geometry_loss_sum=
            telemetry.active_window_geometry_loss_sum,
        active_window_line_clear_loss_sum=
            telemetry.active_window_line_clear_loss_sum,
        active_window_max_height_loss_sum=
            telemetry.active_window_max_height_loss_sum,
        active_window_holes_loss_sum=
            telemetry.active_window_holes_loss_sum,
        active_window_cavities_loss_sum=
            telemetry.active_window_cavities_loss_sum,
        active_window_structure_loss_sum=
            telemetry.active_window_structure_loss_sum,
        old_q_loss=telemetry.old_q_loss,
        q_huber_loss=telemetry.q_huber_loss,
        margin_loss=telemetry.margin_loss,
        raw_top_gap_loss=telemetry.raw_top_gap_loss,
        death_loss=telemetry.death_loss,
        quantile_teacher_loss=telemetry.quantile_teacher_loss,
        geometry_loss=telemetry.geometry_loss,
        line_clear_loss=telemetry.line_clear_loss,
        max_height_loss=telemetry.max_height_loss,
        holes_loss=telemetry.holes_loss,
        cavities_loss=telemetry.cavities_loss,
        structure_loss=telemetry.structure_loss,
    )
end

function restore_component_loss_telemetry(snapshot)
    validate_component_loss_snapshot(snapshot)
    expected = fieldnames(ComponentLossTelemetry)
    values = ntuple(
        index -> begin
            value = Float64(getproperty(snapshot, expected[index]))
            value
        end,
        length(expected),
    )
    return ComponentLossTelemetry(values...)
end

mutable struct ProgressTotals
    updates::Int
    teacher_states::Int
    candidates::Int
    hot_wall_seconds::Float64
    hot_cpu_seconds::Float64
    hot_allocation_bytes::Int128
    hot_gc_seconds::Float64
    pack_seconds::Float64
    forward_seconds::Float64
    loss_seconds::Float64
    shadow_seconds::Float64
    backward_seconds::Float64
    optimizer_seconds::Float64
    consolidation_seconds::Float64
    window_updates::Int
    window_composite_loss::Float64
    window_listnet_ce::Float64
    window_teacher_entropy::Float64
    window_listnet_kl::Float64
    window_composite_excess::Float64
    completed_component_loss_window_updates::Int
    component_losses::ComponentLossTelemetry
end

ProgressTotals() = ProgressTotals(
    0, 0, 0, 0.0, 0.0, Int128(0), 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0,
    ComponentLossTelemetry(),
)

function progress_snapshot(progress::ProgressTotals)
    return (;
        updates=progress.updates,
        teacher_states=progress.teacher_states,
        candidates=progress.candidates,
        hot_wall_seconds=progress.hot_wall_seconds,
        hot_cpu_seconds=progress.hot_cpu_seconds,
        hot_allocation_bytes=progress.hot_allocation_bytes,
        hot_gc_seconds=progress.hot_gc_seconds,
        pack_seconds=progress.pack_seconds,
        forward_seconds=progress.forward_seconds,
        loss_seconds=progress.loss_seconds,
        shadow_seconds=progress.shadow_seconds,
        backward_seconds=progress.backward_seconds,
        optimizer_seconds=progress.optimizer_seconds,
        consolidation_seconds=progress.consolidation_seconds,
        window_updates=progress.window_updates,
        window_composite_loss=progress.window_composite_loss,
        window_listnet_ce=progress.window_listnet_ce,
        window_teacher_entropy=progress.window_teacher_entropy,
        window_listnet_kl=progress.window_listnet_kl,
        window_composite_excess=progress.window_composite_excess,
        completed_component_loss_window_updates=
            progress.completed_component_loss_window_updates,
        telemetry_schema_version=3,
        component_loss_alias_contract=COMPONENT_LOSS_ALIAS_CONTRACT,
        component_losses=component_loss_snapshot(
            progress.component_losses,
        ),
    )
end

function restore_progress(snapshot)
    hasproperty(snapshot, :telemetry_schema_version) || error(
        "checkpoint progress is missing telemetry_schema_version",
    )
    snapshot.telemetry_schema_version isa Int || error(
        "checkpoint progress telemetry schema version must be Int",
    )
    snapshot.telemetry_schema_version == 3 || error(
        "checkpoint progress telemetry schema version differs",
    )
    hasproperty(snapshot, :component_loss_alias_contract) || error(
        "checkpoint progress is missing component-loss alias contract",
    )
    snapshot.component_loss_alias_contract ==
        COMPONENT_LOSS_ALIAS_CONTRACT || error(
        "checkpoint progress component-loss alias contract differs",
    )
    snapshot.completed_component_loss_window_updates isa Int || error(
        "checkpoint completed component-loss window updates must be Int",
    )
    snapshot.completed_component_loss_window_updates >= 0 || error(
        "checkpoint completed component-loss window updates is negative",
    )
    return ProgressTotals(
        Int(snapshot.updates),
        Int(snapshot.teacher_states),
        Int(snapshot.candidates),
        Float64(snapshot.hot_wall_seconds),
        Float64(snapshot.hot_cpu_seconds),
        Int128(snapshot.hot_allocation_bytes),
        Float64(snapshot.hot_gc_seconds),
        Float64(snapshot.pack_seconds),
        Float64(snapshot.forward_seconds),
        Float64(snapshot.loss_seconds),
        Float64(snapshot.shadow_seconds),
        Float64(snapshot.backward_seconds),
        Float64(snapshot.optimizer_seconds),
        Float64(snapshot.consolidation_seconds),
        Int(snapshot.window_updates),
        Float64(snapshot.window_composite_loss),
        Float64(snapshot.window_listnet_ce),
        Float64(snapshot.window_teacher_entropy),
        Float64(snapshot.window_listnet_kl),
        Float64(snapshot.window_composite_excess),
        snapshot.completed_component_loss_window_updates,
        restore_component_loss_telemetry(snapshot.component_losses),
    )
end

function accumulate!(
    progress::ProgressTotals,
    trainer::ArenaTrainer,
)
    metrics = trainer.metrics
    progress.updates += 1
    progress.teacher_states += trainer.arena.state_batch
    progress.candidates += trainer.arena.valid_count
    progress.hot_wall_seconds += metrics.wall_seconds
    progress.hot_cpu_seconds += metrics.cpu_seconds
    progress.hot_allocation_bytes += metrics.allocation_bytes
    progress.hot_gc_seconds += metrics.gc_seconds
    progress.pack_seconds += metrics.pack_seconds
    progress.forward_seconds += metrics.forward_seconds
    progress.loss_seconds += metrics.loss_seconds
    progress.shadow_seconds += metrics.shadow_seconds
    progress.backward_seconds += metrics.backward_seconds
    progress.optimizer_seconds += metrics.optimizer_seconds
    progress.consolidation_seconds += metrics.consolidation_seconds
    loss = trainer.last_loss
    validate_loss_record(loss)
    progress.window_updates += 1
    progress.window_composite_loss += loss.composite_loss
    progress.window_listnet_ce += loss.listnet_loss
    progress.window_teacher_entropy += loss.teacher_entropy
    progress.window_listnet_kl += loss.listnet_kl
    progress.window_composite_excess +=
        loss.composite_loss - loss.teacher_entropy
    telemetry = progress.component_losses
    telemetry.old_q_loss = Float64(loss.old_q_loss)
    telemetry.q_huber_loss = Float64(loss.q_huber_loss)
    telemetry.margin_loss = Float64(loss.margin_loss)
    telemetry.raw_top_gap_loss = Float64(loss.raw_top_gap_loss)
    telemetry.death_loss = Float64(loss.death_loss)
    telemetry.quantile_teacher_loss =
        Float64(loss.quantile_teacher_loss)
    telemetry.geometry_loss = Float64(loss.geometry_loss)
    telemetry.line_clear_loss = Float64(loss.line_clear_loss)
    telemetry.max_height_loss = Float64(loss.max_height_loss)
    telemetry.holes_loss = Float64(loss.holes_loss)
    telemetry.cavities_loss = Float64(loss.cavities_loss)
    telemetry.structure_loss = Float64(loss.structure_loss)
    telemetry.active_window_old_q_loss_sum += telemetry.old_q_loss
    telemetry.active_window_q_huber_loss_sum += telemetry.q_huber_loss
    telemetry.active_window_margin_loss_sum += telemetry.margin_loss
    telemetry.active_window_raw_top_gap_loss_sum +=
        telemetry.raw_top_gap_loss
    telemetry.active_window_death_loss_sum += telemetry.death_loss
    telemetry.active_window_quantile_teacher_loss_sum +=
        telemetry.quantile_teacher_loss
    telemetry.active_window_geometry_loss_sum += telemetry.geometry_loss
    telemetry.active_window_line_clear_loss_sum +=
        telemetry.line_clear_loss
    telemetry.active_window_max_height_loss_sum +=
        telemetry.max_height_loss
    telemetry.active_window_holes_loss_sum += telemetry.holes_loss
    telemetry.active_window_cavities_loss_sum += telemetry.cavities_loss
    telemetry.active_window_structure_loss_sum +=
        telemetry.structure_loss
    return progress
end

function reset_loss_window!(progress::ProgressTotals)
    progress.completed_component_loss_window_updates =
        progress.window_updates
    progress.window_updates = 0
    progress.window_composite_loss = 0.0
    progress.window_listnet_ce = 0.0
    progress.window_teacher_entropy = 0.0
    progress.window_listnet_kl = 0.0
    progress.window_composite_excess = 0.0
    telemetry = progress.component_losses
    telemetry.window_old_q_loss_sum =
        telemetry.active_window_old_q_loss_sum
    telemetry.window_q_huber_loss_sum =
        telemetry.active_window_q_huber_loss_sum
    telemetry.window_margin_loss_sum =
        telemetry.active_window_margin_loss_sum
    telemetry.window_raw_top_gap_loss_sum =
        telemetry.active_window_raw_top_gap_loss_sum
    telemetry.window_death_loss_sum =
        telemetry.active_window_death_loss_sum
    telemetry.window_quantile_teacher_loss_sum =
        telemetry.active_window_quantile_teacher_loss_sum
    telemetry.window_geometry_loss_sum =
        telemetry.active_window_geometry_loss_sum
    telemetry.window_line_clear_loss_sum =
        telemetry.active_window_line_clear_loss_sum
    telemetry.window_max_height_loss_sum =
        telemetry.active_window_max_height_loss_sum
    telemetry.window_holes_loss_sum =
        telemetry.active_window_holes_loss_sum
    telemetry.window_cavities_loss_sum =
        telemetry.active_window_cavities_loss_sum
    telemetry.window_structure_loss_sum =
        telemetry.active_window_structure_loss_sum
    telemetry.active_window_old_q_loss_sum = 0.0
    telemetry.active_window_q_huber_loss_sum = 0.0
    telemetry.active_window_margin_loss_sum = 0.0
    telemetry.active_window_raw_top_gap_loss_sum = 0.0
    telemetry.active_window_death_loss_sum = 0.0
    telemetry.active_window_quantile_teacher_loss_sum = 0.0
    telemetry.active_window_geometry_loss_sum = 0.0
    telemetry.active_window_line_clear_loss_sum = 0.0
    telemetry.active_window_max_height_loss_sum = 0.0
    telemetry.active_window_holes_loss_sum = 0.0
    telemetry.active_window_cavities_loss_sum = 0.0
    telemetry.active_window_structure_loss_sum = 0.0
    return progress
end

function env_int(name, default; minimum=0)
    value = parse(Int, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function env_float(name, default; minimum=0.0)
    value = parse(Float32, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function env_bool(name, default::Bool)
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$name must be true/false, 1/0, yes/no, or on/off")
end

function required_env(name::AbstractString)
    haskey(ENV, name) || error(
        "$name must be explicitly set for the production driver",
    )
    value = strip(ENV[name])
    isempty(value) && error("$name must not be empty")
    return value
end

sha256_file(path::AbstractString) =
    bytes2hex(open(sha256, path))

function _canonical_field!(
    io::IO,
    name::AbstractString,
    value,
)
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

function source_fingerprint_files()
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

function source_file_inventory()
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
                sha256=sha256_file(canonical_path),
            )
        end
        for path in source_fingerprint_files()
    ]
end

function source_fingerprint()
    io = IOBuffer()
    _canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-source-fingerprint-v2",
    )
    for entry in source_file_inventory()
        _canonical_field!(io, "filename", entry.relative_path)
        _canonical_field!(io, "length", entry.bytes)
        _canonical_field!(io, "sha256", entry.sha256)
        write(io, "end-file\n")
    end
    return bytes2hex(sha256(take!(io)))
end

function dataset_binding_preflight(path::AbstractString)
    source_path = abspath(path)
    binding_file_path = isdir(source_path) ?
        joinpath(source_path, "manifest.json") : source_path
    isfile(binding_file_path) || error(
        "dataset binding file does not exist: $binding_file_path",
    )
    canonical_binding_file = realpath(binding_file_path)
    return (;
        kind=isdir(source_path) ? :sharded_manifest : :single_file,
        source_path=isdir(source_path) ?
            realpath(source_path) : canonical_binding_file,
        binding_file_path=canonical_binding_file,
        binding_file_sha256=sha256_file(canonical_binding_file),
    )
end

function _ordered_manifest_counts(counts)
    names = sort!(String[String(name) for name in keys(counts)])
    return [
        (; name, count=Int(counts[name]))
        for name in names
    ]
end

function _required_manifest_part_field(part, name::Symbol)
    hasproperty(part, name) || error(
        "dataset manifest part is missing $(String(name))",
    )
    return getproperty(part, name)
end

function canonical_manifest_parts_sha256(manifest)
    hasproperty(manifest, :parts) ||
        error("dataset manifest has no parts")
    io = IOBuffer()
    _canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-canonical-manifest-parts-v1",
    )
    parts = collect(manifest.parts)
    for (index, part) in enumerate(parts)
        _canonical_field!(io, "part_index", index)
        relative_path = replace(
            normpath(String(_required_manifest_part_field(
                part,
                :relative_path,
            ))),
            '\\' => '/',
        )
        _canonical_field!(io, "relative_path", relative_path)
        _canonical_field!(
            io,
            "row_count",
            Int(_required_manifest_part_field(part, :row_count)),
        )
        _canonical_field!(
            io,
            "bytes",
            Int(_required_manifest_part_field(part, :bytes)),
        )
        part_sha256 = lowercase(String(
            _required_manifest_part_field(part, :sha256),
        ))
        occursin(r"^[0-9a-f]{64}$", part_sha256) || error(
            "dataset manifest part SHA-256 is not canonical",
        )
        _canonical_field!(io, "sha256", part_sha256)
        _canonical_field!(
            io,
            "split",
            String(_required_manifest_part_field(part, :split)),
        )
        # These fields are not required by the tensor loader, but when present
        # they make the ordered semantic identity human-auditable.  The raw
        # manifest digest below binds every other field as well.
        for optional_name in (
            :episode_key,
            :role,
            :seed,
            :candidate_count,
        )
            if hasproperty(part, optional_name)
                _canonical_field!(
                    io,
                    String(optional_name),
                    getproperty(part, optional_name),
                )
            end
        end
        write(io, "end-part\n")
    end
    return bytes2hex(sha256(take!(io))), length(parts)
end

function bind_loaded_dataset(
    dataset_path::AbstractString,
    dataset,
    preflight,
)
    current = dataset_binding_preflight(dataset_path)
    current.kind == preflight.kind ||
        error("dataset kind changed while loading")
    current.source_path == preflight.source_path ||
        error("dataset source path changed while loading")
    current.binding_file_path == preflight.binding_file_path ||
        error("dataset binding file path changed while loading")
    current.binding_file_sha256 == preflight.binding_file_sha256 ||
        error("dataset binding file changed while loading")

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

    dataset.manifest_path == current.binding_file_path || error(
        "loader manifest path differs from the bound manifest",
    )
    dataset.part_integrity_verified === true || error(
        "loader did not verify every manifest part SHA-256",
    )
    manifest = JSON3.read(read(current.binding_file_path, String))
    canonical_parts_sha256, manifest_part_count =
        canonical_manifest_parts_sha256(manifest)
    Int(dataset.verified_part_count) == manifest_part_count || error(
        "loader verified $(dataset.verified_part_count) parts, " *
        "but the manifest contains $manifest_part_count",
    )
    Int(dataset.manifest_format_version) ==
        Int(manifest.format_version) || error(
        "loader manifest format differs from the bound manifest",
    )
    counts = _ordered_manifest_counts(dataset.manifest_counts)
    content_io = IOBuffer()
    _canonical_field!(
        content_io,
        "schema",
        "serial-workspace-snn-dataset-content-v1",
    )
    _canonical_field!(
        content_io,
        "manifest_sha256",
        current.binding_file_sha256,
    )
    _canonical_field!(
        content_io,
        "manifest_format_version",
        Int(dataset.manifest_format_version),
    )
    _canonical_field!(
        content_io,
        "manifest_part_count",
        manifest_part_count,
    )
    _canonical_field!(
        content_io,
        "canonical_parts_sha256",
        canonical_parts_sha256,
    )
    dataset_content_sha256 =
        bytes2hex(sha256(take!(content_io)))
    integrity = (;
        schema="serial-workspace-snn-dataset-integrity-v1",
        kind="sharded_manifest",
        source_path=current.source_path,
        binding_file_path=current.binding_file_path,
        binding_file_sha256=current.binding_file_sha256,
        manifest_sha256=current.binding_file_sha256,
        manifest_format_version=
            Int(dataset.manifest_format_version),
        manifest_part_count,
        verified_part_count=Int(dataset.verified_part_count),
        part_integrity_verified=true,
        manifest_counts=counts,
        canonical_parts_sha256,
        binding_file_stable=true,
    )
    return dataset_content_sha256, integrity
end

const REQUIRED_STARTUP_FILE_OPTION = 2
const REQUIRED_HISTORY_FILE_OPTION = 0
const REQUIRED_JULIA_RUNTIME_ARGUMENTS = (
    "--startup-file=no",
    "--history-file=no",
)
const REQUIRED_PRODUCTION_PROJECT_PATH =
    realpath(joinpath(@__DIR__, ".."))
const REQUIRED_PRODUCTION_PROJECT_FILE = realpath(joinpath(
    REQUIRED_PRODUCTION_PROJECT_PATH,
    "Project.toml",
))

function validate_production_project_paths(
    project_option_path::AbstractString,
    active_project_file::AbstractString,
)
    realpath(project_option_path) ==
        REQUIRED_PRODUCTION_PROJECT_PATH || error(
        "production training requires --project=" *
        REQUIRED_PRODUCTION_PROJECT_PATH,
    )
    realpath(active_project_file) ==
        REQUIRED_PRODUCTION_PROJECT_FILE || error(
        "production training active Project.toml is noncanonical",
    )
    return true
end

function validate_hermetic_runtime_options(
    startup_file_option::Integer,
    history_file_option::Integer,
)
    startup_file_option == REQUIRED_STARTUP_FILE_OPTION || error(
        "production training requires --startup-file=no: " *
        "observed Julia option=$startup_file_option",
    )
    history_file_option == REQUIRED_HISTORY_FILE_OPTION || error(
        "production training requires --history-file=no: " *
        "observed Julia option=$history_file_option",
    )
    return true
end

function hermetic_runtime_options()
    startup_file_option = Int(Base.JLOptions().startupfile)
    history_file_option = Int(Base.JLOptions().historyfile)
    validate_hermetic_runtime_options(
        startup_file_option,
        history_file_option,
    )
    project_option_pointer = Base.JLOptions().project
    project_option_pointer == C_NULL && error(
        "production training requires an explicit --project option",
    )
    julia_project_option = unsafe_string(project_option_pointer)
    julia_project_option_path = realpath(abspath(julia_project_option))
    active_project_file = Base.active_project()
    active_project_file === nothing &&
        error("production training has no active Project.toml")
    validate_production_project_paths(
        julia_project_option_path,
        active_project_file,
    )
    return (;
        startup_file_option,
        history_file_option,
        startup_file_disabled=true,
        history_file_disabled=true,
        julia_runtime_arguments=collect(
            REQUIRED_JULIA_RUNTIME_ARGUMENTS,
        ),
        julia_project_option,
        julia_project_option_path,
        active_project_file=realpath(active_project_file),
        canonical_project_path=REQUIRED_PRODUCTION_PROJECT_PATH,
    )
end

function runtime_provenance(fingerprint::AbstractString)
    runtime_options = hermetic_runtime_options()
    blas_threads = BLAS.get_num_threads()
    blas_threads == 1 || error(
        "production runtime provenance requires one BLAS thread: " *
        "observed=$blas_threads",
    )
    project = Base.active_project()
    project === nothing && error(
        "the production driver requires an active Project.toml",
    )
    project_path = realpath(project)
    required_project_path = realpath(joinpath(
        REQUIRED_PRODUCTION_PROJECT_PATH,
        "Project.toml",
    ))
    project_path == required_project_path || error(
        "production training requires the canonical Project.toml: " *
        "observed=$project_path expected=$required_project_path",
    )
    manifest_path = joinpath(dirname(project_path), "Manifest.toml")
    isfile(manifest_path) || error(
        "the production driver requires Manifest.toml beside Project.toml",
    )
    executable_path = realpath(
        joinpath(Sys.BINDIR, Base.julia_exename()),
    )
    source_files = source_file_inventory()
    source_fingerprint() == fingerprint || error(
        "source changed while runtime provenance was collected",
    )
    return (;
        schema="serial-workspace-snn-runtime-provenance-v2",
        julia_version=string(VERSION),
        julia_executable_path=executable_path,
        julia_executable_sha256=sha256_file(executable_path),
        julia_architecture=String(Sys.ARCH),
        julia_machine=String(Sys.MACHINE),
        julia_kernel=String(Sys.KERNEL),
        project_toml_path=project_path,
        project_toml_sha256=sha256_file(project_path),
        manifest_toml_path=realpath(manifest_path),
        manifest_toml_sha256=sha256_file(manifest_path),
        startup_file_option=runtime_options.startup_file_option,
        history_file_option=runtime_options.history_file_option,
        startup_file_disabled=runtime_options.startup_file_disabled,
        history_file_disabled=runtime_options.history_file_disabled,
        julia_runtime_arguments=
            runtime_options.julia_runtime_arguments,
        julia_project_option=runtime_options.julia_project_option,
        julia_project_option_path=
            runtime_options.julia_project_option_path,
        canonical_project_path=
            runtime_options.canonical_project_path,
        blas_threads,
        source_fingerprint=String(fingerprint),
        source_files,
    )
end

function eprop_config_snapshot(config::EPropShadowConfig)
    return (;
        trace_decay_scale=config.trace_decay_scale,
        feedback_seed=config.feedback_seed,
        feedback_scale=config.feedback_scale,
        feedback_mode=config.feedback_mode,
        eligibility_mode=config.eligibility_mode,
        error_signal_mode=config.error_signal_mode,
        edge_parameter_mode=config.edge_parameter_mode,
        node_parameter_mode=config.node_parameter_mode,
        routing_parameter_mode=config.routing_parameter_mode,
        signal_schedule=config.signal_schedule,
        third_factor_mode=config.third_factor_mode,
        time_order=config.time_order,
        routing_entropy_weight=config.routing_entropy_weight,
        routing_entropy_floor=config.routing_entropy_floor,
        routing_load_weight=config.routing_load_weight,
    )
end

function production_contract_sha256(contract)
    return bytes2hex(sha256(codeunits(String(JSON3.write(contract)))))
end

function validate_resume_contract(saved_config, current_config)
    hasproperty(saved_config, :production_contract) || error(
        "resume checkpoint predates production contract v1; " *
        "v1/v2 checkpoints are not accepted",
    )
    hasproperty(saved_config, :production_contract_sha256) || error(
        "resume production contract digest is missing",
    )
    saved_contract = saved_config.production_contract
    Int(saved_contract.version) == PRODUCTION_CONTRACT_VERSION || error(
        "resume production contract version differs",
    )
    saved_digest = production_contract_sha256(saved_contract)
    saved_digest == String(saved_config.production_contract_sha256) ||
        error("resume production contract digest is corrupt")
    current_digest =
        production_contract_sha256(current_config.production_contract)
    current_digest ==
        String(current_config.production_contract_sha256) || error(
        "current production contract digest is corrupt",
    )
    saved_digest == current_digest || error(
        "resume production contract differs: saved=$saved_digest " *
        "current=$current_digest",
    )
    isequal(saved_contract, current_config.production_contract) || error(
        "resume production contract payload differs despite its digest",
    )
    return nothing
end

function validate_start_mode_update(
    start_mode::Symbol,
    checkpoint_update::Integer,
    maximum_updates::Integer,
)
    if start_mode === :resume
        Int(checkpoint_update) < Int(maximum_updates) || error(
            "resume checkpoint already reached the target; use " *
            "SWSNN_START_MODE=finalize_only",
        )
    elseif start_mode === :finalize_only
        Int(checkpoint_update) == Int(maximum_updates) || error(
            "finalize_only requires a checkpoint exactly at " *
            "SWSNN_MAX_UPDATES",
        )
    else
        start_mode === :scratch || error("unsupported start mode")
        Int(checkpoint_update) == 0 || error(
            "scratch mode cannot start from a checkpoint update",
        )
    end
    return nothing
end

function validate_optimizer_contract(optimizer, optimizer_config)
    optimizer.learning_rate == optimizer_config.learning_rate ||
        error("optimizer learning rate differs from production config")
    optimizer.weight_decay == optimizer_config.weight_decay ||
        error("optimizer weight decay differs from production config")
    optimizer.beta1 == optimizer_config.beta1 ||
        error("optimizer beta1 differs from production config")
    optimizer.beta2 == optimizer_config.beta2 ||
        error("optimizer beta2 differs from production config")
    optimizer.epsilon == optimizer_config.epsilon ||
        error("optimizer epsilon differs from production config")
    return nothing
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) && error("manifest training split is empty")
        return Int.(rows)
    end
    groups = sort(unique(dataset.split_group_ids))
    shuffled = shuffle(Xoshiro(SPLIT_SEED), groups)
    validation_count =
        clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    forbidden = Set(shuffled[1:validation_count])
    return findall(
        group -> !(group in forbidden),
        dataset.split_group_ids,
    )
end

function fixed_training_panel(rows, count::Int)
    selected = copy(Int.(rows))
    shuffle!(Xoshiro(TRAIN_EVAL_SEED), selected)
    resize!(selected, min(count, length(selected)))
    return selected
end

function write_json(path, value)
    temporary = path * ".tmp"
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
        end
        mv(temporary, path; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return path
end

function write_stage_status(
    run_dir,
    phase::Symbol;
    update::Integer,
    details=(;),
)
    payload = merge((;
        format="serial-workspace-snn-arena-stage-status",
        version=1,
        phase=String(phase),
        recorded_at=string(now()),
        process_id=getpid(),
        update=Int(update),
    ), details)
    write_json(joinpath(run_dir, "stage_status.json"), payload)
    println(
        stderr,
        "SWSNN_STAGE phase=$(String(phase)) update=$(Int(update))",
    )
    flush(stderr)
    return payload
end

function normalized_path_identity(path::AbstractString)
    normalized = normpath(abspath(path))
    return Sys.iswindows() ? lowercase(normalized) : normalized
end

function windows_file_attributes(path::AbstractString)
    Sys.iswindows() || return UInt32(0)
    attributes = ccall(
        (:GetFileAttributesW, "kernel32"),
        stdcall,
        UInt32,
        (Cwstring,),
        path,
    )
    attributes == WINDOWS_INVALID_FILE_ATTRIBUTES && error(
        "GetFileAttributesW failed for $path",
    )
    return attributes
end

function path_prefixes(path::AbstractString)
    components = splitpath(normpath(abspath(path)))
    isempty(components) && error("cannot enumerate an empty path")
    prefixes = String[]
    current = first(components)
    push!(prefixes, current)
    for component in Iterators.drop(components, 1)
        current = joinpath(current, component)
        push!(prefixes, current)
    end
    return prefixes
end

"""
Reject every existing symbolic-link, junction, mount point, or other Windows
reparse component. Missing suffix components are permitted only for a path
which has not been created yet.
"""
function reject_existing_reparse_chain(
    path::AbstractString,
    label::AbstractString,
)
    missing_suffix = false
    for prefix in path_prefixes(path)
        if !ispath(prefix)
            missing_suffix = true
            continue
        end
        missing_suffix && error(
            "$label has an existing descendant below a missing component: " *
            prefix,
        )
        if Sys.iswindows()
            iszero(
                windows_file_attributes(prefix) &
                WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT,
            ) || error("$label traverses a Windows reparse point: $prefix")
        else
            !islink(prefix) ||
                error("$label traverses a symbolic link: $prefix")
        end
    end
    return path
end

function canonical_future_path(path::AbstractString)
    candidate = normpath(abspath(path))
    reject_existing_reparse_chain(candidate, "future path")
    unresolved = String[]
    cursor = candidate
    while !ispath(cursor)
        parent = dirname(cursor)
        parent == cursor && break
        pushfirst!(unresolved, basename(cursor))
        cursor = parent
    end
    canonical = ispath(cursor) ? realpath(cursor) : cursor
    for component in unresolved
        canonical = joinpath(canonical, component)
    end
    return normpath(canonical)
end

function canonical_existing_directory(
    path::AbstractString,
    label::AbstractString,
)
    candidate = normpath(abspath(path))
    isdir(candidate) || error("$label is not an existing directory: $candidate")
    reject_existing_reparse_chain(candidate, label)
    canonical = realpath(candidate)
    normalized_path_identity(canonical) ==
        normalized_path_identity(candidate) || error(
        "$label is not canonical: supplied=$candidate resolved=$canonical",
    )
    return canonical
end

function canonical_existing_file(
    path::AbstractString,
    label::AbstractString,
)
    candidate = normpath(abspath(path))
    isfile(candidate) || error("$label is not an existing file: $candidate")
    reject_existing_reparse_chain(candidate, label)
    canonical = realpath(candidate)
    normalized_path_identity(canonical) ==
        normalized_path_identity(candidate) || error(
        "$label is not canonical: supplied=$candidate resolved=$canonical",
    )
    return canonical
end

function validate_run_id(raw_run_id::AbstractString)
    run_id = String(raw_run_id)
    isempty(run_id) && error("run ID cannot be empty")
    ncodeunits(run_id) <= 128 || error("run ID is too long")
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    startswith(run_id, ".") && error("run ID cannot begin with a dot")
    endswith(run_id, ".") && error("run ID cannot end with a dot")
    all(==('.'), run_id) && error("run ID cannot be dot-only")
    basename(run_id) == run_id || error("run ID is not a basename")
    normpath(run_id) == run_id || error("run ID is not canonical")
    device_stem = uppercase(first(split(run_id, '.'; limit=2)))
    device_stem in WINDOWS_RESERVED_BASENAMES && error(
        "run ID uses a reserved Windows device basename",
    )
    return run_id
end

function validate_run_destination(
    raw_output_root::AbstractString,
    raw_run_id::AbstractString,
)
    isabspath(raw_output_root) ||
        error("SWSNN_OUTPUT must be an explicit absolute path")
    output_root = normpath(abspath(raw_output_root))
    ispath(output_root) && !isdir(output_root) && error(
        "SWSNN_OUTPUT is not a directory: $output_root",
    )
    run_id = validate_run_id(raw_run_id)
    canonical_output = canonical_future_path(output_root)
    normalized_path_identity(canonical_output) ==
        normalized_path_identity(output_root) || error(
        "output root traverses a path alias or reparse point",
    )
    run_dir = normpath(joinpath(output_root, run_id))
    canonical_run = canonical_future_path(run_dir)
    normalized_path_identity(dirname(canonical_run)) ==
        normalized_path_identity(canonical_output) || error(
        "run directory is not an immediate child of the output root",
    )
    basename(canonical_run) == run_id || error(
        "run directory basename differs from the canonical run ID",
    )
    normalized_path_identity(canonical_run) ==
        normalized_path_identity(run_dir) || error(
        "run directory traverses a path alias or reparse point",
    )
    return output_root, run_dir
end

function required_json_string(value, name::AbstractString)
    value isa AbstractString || error("$name must be a string")
    result = String(value)
    isempty(result) && error("$name must not be empty")
    return result
end

function required_json_integer(value, name::AbstractString)
    value isa Integer && !(value isa Bool) ||
        error("$name must be an integer")
    return try
        Int(value)
    catch
        error("$name does not fit Int")
    end
end

function normalized_start_mode(value)
    return replace(
        lowercase(required_json_string(value, "launch start mode")),
        '-' => '_',
    )
end

function validate_launch_manifest_binding(
    raw_manifest_path::AbstractString,
    raw_manifest_sha256::AbstractString,
    output_root::AbstractString,
    run_dir::AbstractString,
    run_id::AbstractString,
    start_mode::Symbol,
    expected_updates::Integer,
)
    manifest_sha256 = String(raw_manifest_sha256)
    occursin(r"^[0-9a-f]{64}$", manifest_sha256) || error(
        "launch manifest SHA-256 is not canonical",
    )
    canonical_output =
        canonical_existing_directory(output_root, "launch output root")
    expected_manifest_path = joinpath(
        canonical_output,
        "_controllers",
        run_id,
        "launch_manifest.json",
    )
    manifest_path = canonical_existing_file(
        raw_manifest_path,
        "launch manifest",
    )
    manifest_path == String(raw_manifest_path) || error(
        "launch manifest path is not canonical",
    )
    manifest_path == expected_manifest_path || error(
        "launch manifest path is not the controller-owned path for this run",
    )
    manifest_bytes = read(manifest_path)
    bytes2hex(sha256(manifest_bytes)) == manifest_sha256 || error(
        "launch manifest SHA-256 differs from SWSNN_LAUNCH_MANIFEST_SHA256",
    )
    launch = JSON3.read(String(manifest_bytes))
    launch isa JSON3.Object ||
        error("launch manifest must be a JSON object")
    required_json_string(launch.format, "launch format") ==
        "serial-workspace-snn-arena-run-launch" || error(
        "launch manifest format differs",
    )
    required_json_integer(launch.version, "launch version") == 2 ||
        error("launch manifest version differs")
    required_json_string(launch.run_id, "launch run_id") == run_id ||
        error("launch manifest run ID differs")
    launch_run_directory = required_json_string(
        launch.run_directory,
        "launch run_directory",
    )
    isabspath(launch_run_directory) || error(
        "launch run directory is not absolute",
    )
    normpath(abspath(launch_run_directory)) == launch_run_directory || error(
        "launch run directory is not canonical",
    )
    normalized_path_identity(launch_run_directory) ==
        normalized_path_identity(run_dir) || error(
        "launch run directory differs",
    )
    normalized_start_mode(launch.start_mode) == String(start_mode) || error(
        "launch start mode differs",
    )
    required_json_integer(
        launch.expected_updates,
        "launch expected_updates",
    ) == Int(expected_updates) || error("launch expected updates differ")
    launch_output_root = required_json_string(
        launch.output_root,
        "launch output_root",
    )
    launch_output_root == canonical_output || error(
        "launch output root differs",
    )
    hasproperty(launch, :code_artifacts) ||
        error("launch manifest has no code artifacts")
    hasproperty(launch.code_artifacts, :training) ||
        error("launch manifest has no training code artifact")
    training_artifact = launch.code_artifacts.training
    training_path = required_json_string(
        training_artifact.path,
        "launch training artifact path",
    )
    canonical_training_path = canonical_existing_file(
        @__FILE__,
        "production training source",
    )
    training_path == canonical_training_path || error(
        "launch training artifact path differs",
    )
    required_json_integer(
        training_artifact.bytes,
        "launch training artifact bytes",
    ) == filesize(canonical_training_path) || error(
        "launch training artifact byte size differs",
    )
    training_sha256 = required_json_string(
        training_artifact.sha256,
        "launch training artifact SHA-256",
    )
    occursin(r"^[0-9a-f]{64}$", training_sha256) || error(
        "launch training artifact SHA-256 is not canonical",
    )
    training_sha256 == sha256_file(canonical_training_path) || error(
        "launch training artifact SHA-256 differs",
    )
    return (;
        path=manifest_path,
        sha256=manifest_sha256,
    )
end

function launch_binding_fields(value, label::AbstractString)
    hasproperty(value, :path) || error("$label is missing path")
    hasproperty(value, :sha256) || error("$label is missing sha256")
    observed = Set(String(name) for name in propertynames(value))
    observed == Set(("path", "sha256")) || error(
        "$label properties differ",
    )
    path = required_json_string(getproperty(value, :path), "$label path")
    sha256_value = required_json_string(
        getproperty(value, :sha256),
        "$label SHA-256",
    )
    occursin(r"^[0-9a-f]{64}$", sha256_value) || error(
        "$label SHA-256 is not canonical",
    )
    return (; path, sha256=sha256_value)
end

function validate_launch_parent_contract(
    launch_binding,
    start_mode::Symbol,
    resume_payload,
    checkpoint_reference,
)
    binding = launch_binding_fields(
        launch_binding,
        "current launch_binding",
    )
    manifest_path = canonical_existing_file(
        binding.path,
        "current launch manifest",
    )
    manifest_path == binding.path || error(
        "current launch manifest path is noncanonical",
    )
    manifest_bytes = read(manifest_path)
    bytes2hex(sha256(manifest_bytes)) == binding.sha256 || error(
        "current launch manifest changed after binding validation",
    )
    launch = JSON3.read(String(manifest_bytes))
    hasproperty(launch, :parent_checkpoint) || error(
        "current launch manifest is missing parent_checkpoint",
    )
    hasproperty(launch, :parent_lineage) || error(
        "current launch manifest is missing parent_lineage",
    )
    launch.parent_lineage isa AbstractVector || error(
        "current launch parent_lineage must be an array",
    )
    if start_mode === :scratch
        resume_payload === nothing && checkpoint_reference === nothing ||
            error("scratch launch unexpectedly loaded a parent checkpoint")
        launch.parent_checkpoint === nothing || error(
            "scratch launch manifest unexpectedly binds a parent checkpoint",
        )
        isempty(launch.parent_lineage) || error(
            "scratch launch manifest unexpectedly binds parent lineage",
        )
        return true
    end

    resume_payload === nothing && error(
        "non-scratch launch has no loaded parent checkpoint",
    )
    checkpoint_reference === nothing && error(
        "non-scratch launch has no normalized parent checkpoint reference",
    )
    launch.parent_checkpoint isa JSON3.Object || error(
        "non-scratch launch parent_checkpoint must be an object",
    )
    Set(String(key) for key in keys(launch.parent_checkpoint)) ==
        Set(("path", "sha256", "update")) || error(
        "current launch parent_checkpoint properties differ",
    )
    launch_parent_path = required_json_string(
        launch.parent_checkpoint.path,
        "current launch parent checkpoint path",
    )
    launch_parent_sha256 = required_json_string(
        launch.parent_checkpoint.sha256,
        "current launch parent checkpoint SHA-256",
    )
    occursin(r"^[0-9a-f]{64}$", launch_parent_sha256) || error(
        "current launch parent checkpoint SHA-256 is noncanonical",
    )
    launch_parent_update = required_json_integer(
        launch.parent_checkpoint.update,
        "current launch parent checkpoint update",
    )
    launch_parent_path == String(checkpoint_reference.path) || error(
        "current launch parent checkpoint path differs from SWSNN_RESUME",
    )
    launch_parent_sha256 == String(checkpoint_reference.sha256) || error(
        "current launch parent checkpoint SHA-256 differs from SWSNN_RESUME",
    )
    launch_parent_update == Int(checkpoint_reference.update) || error(
        "current launch parent checkpoint update differs from the payload",
    )
    isempty(launch.parent_lineage) && error(
        "non-scratch launch manifest has empty parent_lineage",
    )
    lineage_head = first(launch.parent_lineage)
    lineage_head isa JSON3.Object || error(
        "current launch parent_lineage head must be an object",
    )
    for name in (
        :run_id,
        :run_directory,
        :selected_update,
        :selected_checkpoint,
        :launch_manifest,
    )
        hasproperty(lineage_head, name) || error(
            "current launch parent_lineage head is missing $(String(name))",
        )
    end
    parent_run_id = required_json_string(
        resume_payload.config.run_id,
        "loaded parent run ID",
    )
    required_json_string(
        lineage_head.run_id,
        "current launch lineage run ID",
    ) == parent_run_id || error(
        "current launch lineage run ID differs from the parent payload",
    )
    parent_run_dir = dirname(dirname(String(checkpoint_reference.path)))
    required_json_string(
        lineage_head.run_directory,
        "current launch lineage run directory",
    ) == parent_run_dir || error(
        "current launch lineage run directory differs",
    )
    required_json_integer(
        lineage_head.selected_update,
        "current launch lineage selected update",
    ) == Int(checkpoint_reference.update) || error(
        "current launch lineage selected update differs",
    )
    selected_checkpoint = lineage_head.selected_checkpoint
    selected_checkpoint isa JSON3.Object || error(
        "current launch lineage selected_checkpoint must be an object",
    )
    Set(String(key) for key in keys(selected_checkpoint)) ==
        Set(("kind", "path", "bytes", "sha256", "update")) || error(
        "current launch lineage selected_checkpoint properties differ",
    )
    required_json_string(
        selected_checkpoint.kind,
        "current launch lineage checkpoint kind",
    ) == String(checkpoint_reference.kind) || error(
        "current launch lineage checkpoint kind differs",
    )
    required_json_string(
        selected_checkpoint.path,
        "current launch lineage checkpoint path",
    ) == String(checkpoint_reference.path) || error(
        "current launch lineage checkpoint path differs",
    )
    required_json_integer(
        selected_checkpoint.bytes,
        "current launch lineage checkpoint bytes",
    ) == Int(checkpoint_reference.bytes) || error(
        "current launch lineage checkpoint bytes differ",
    )
    required_json_string(
        selected_checkpoint.sha256,
        "current launch lineage checkpoint SHA-256",
    ) == String(checkpoint_reference.sha256) || error(
        "current launch lineage checkpoint SHA-256 differs",
    )
    required_json_integer(
        selected_checkpoint.update,
        "current launch lineage checkpoint update",
    ) == Int(checkpoint_reference.update) || error(
        "current launch lineage checkpoint update differs",
    )
    parent_binding = launch_binding_fields(
        resume_payload.config.launch_binding,
        "loaded parent launch_binding",
    )
    lineage_launch = lineage_head.launch_manifest
    lineage_launch isa JSON3.Object || error(
        "current launch lineage launch_manifest must be an object",
    )
    for name in (:kind, :path, :bytes, :sha256)
        hasproperty(lineage_launch, name) || error(
            "current launch lineage launch_manifest is missing " *
            String(name),
        )
    end
    required_json_string(
        lineage_launch.kind,
        "current launch lineage launch manifest kind",
    ) == "launch_manifest" || error(
        "current launch lineage launch manifest kind differs",
    )
    required_json_string(
        lineage_launch.path,
        "current launch lineage launch manifest path",
    ) == parent_binding.path || error(
        "current launch lineage launch manifest path differs",
    )
    required_json_integer(
        lineage_launch.bytes,
        "current launch lineage launch manifest bytes",
    ) == filesize(parent_binding.path) || error(
        "current launch lineage launch manifest bytes differ",
    )
    required_json_string(
        lineage_launch.sha256,
        "current launch lineage launch manifest SHA-256",
    ) == parent_binding.sha256 || error(
        "current launch lineage launch manifest SHA-256 differs",
    )
    return true
end

function validate_parent_launch_binding(
    resume_payload,
    checkpoint_reference,
)
    hasproperty(resume_payload.config, :launch_binding) || error(
        "parent checkpoint config is missing launch_binding",
    )
    payload_binding = launch_binding_fields(
        resume_payload.config.launch_binding,
        "parent checkpoint launch_binding",
    )
    parent_run_id = validate_run_id(required_json_string(
        resume_payload.config.run_id,
        "parent checkpoint run ID",
    ))
    parent_checkpoint_path = canonical_existing_file(
        String(checkpoint_reference.path),
        "parent checkpoint",
    )
    parent_run_dir = dirname(dirname(parent_checkpoint_path))
    canonical_parent_run = canonical_existing_directory(
        parent_run_dir,
        "parent run directory",
    )
    basename(canonical_parent_run) == parent_run_id || error(
        "parent checkpoint run ID differs from its run directory",
    )
    parent_output_root = canonical_existing_directory(
        dirname(canonical_parent_run),
        "parent output root",
    )
    parent_config_path = canonical_existing_file(
        joinpath(canonical_parent_run, "config.json"),
        "parent config.json",
    )
    parent_config = JSON3.read(read(parent_config_path, String))
    hasproperty(parent_config, :config) ||
        error("parent config.json has no config payload")
    hasproperty(parent_config.config, :launch_binding) || error(
        "parent config.json is missing launch_binding",
    )
    config_binding = launch_binding_fields(
        parent_config.config.launch_binding,
        "parent config.json launch_binding",
    )
    config_binding == payload_binding || error(
        "parent checkpoint and config.json launch bindings differ",
    )
    required_json_string(
        parent_config.config.run_id,
        "parent config.json run ID",
    ) == parent_run_id || error(
        "parent config.json run ID differs",
    )
    normalized_start_mode(parent_config.config.start_mode) ==
        String(resume_payload.config.start_mode) || error(
        "parent config.json start mode differs",
    )
    required_json_integer(
        parent_config.config.maximum_updates,
        "parent config.json maximum_updates",
    ) == Int(resume_payload.config.maximum_updates) || error(
        "parent config.json maximum updates differ",
    )
    validated_binding = validate_launch_manifest_binding(
        payload_binding.path,
        payload_binding.sha256,
        parent_output_root,
        canonical_parent_run,
        parent_run_id,
        Symbol(String(resume_payload.config.start_mode)),
        Int(resume_payload.config.maximum_updates),
    )
    validated_binding == payload_binding || error(
        "parent launch binding differs after live validation",
    )
    return validated_binding
end

function reserve_run_directory!(
    output_root::AbstractString,
    run_dir::AbstractString,
    run_id::AbstractString,
)
    validated_output, validated_run =
        validate_run_destination(output_root, run_id)
    normalized_path_identity(validated_run) ==
        normalized_path_identity(run_dir) || error(
        "run directory changed after destination validation",
    )
    ispath(validated_run) && error("run already exists: $validated_run")
    mkpath(validated_output)
    canonical_output = canonical_existing_directory(
        validated_output,
        "output root",
    )
    planned_run = joinpath(canonical_output, run_id)
    normalized_path_identity(planned_run) ==
        normalized_path_identity(validated_run) || error(
        "output root ownership changed before run reservation",
    )
    # `mkdir` is the atomic ownership boundary for this run.
    mkdir(validated_run)
    canonical_run = canonical_existing_directory(
        validated_run,
        "owned run directory",
    )
    normalized_path_identity(dirname(canonical_run)) ==
        normalized_path_identity(canonical_output) || error(
        "owned run directory escaped the output root",
    )
    basename(canonical_run) == run_id || error(
        "owned run directory basename changed",
    )
    checkpoint_dir = joinpath(canonical_run, "checkpoints")
    mkdir(checkpoint_dir)
    canonical_checkpoints = canonical_existing_directory(
        checkpoint_dir,
        "owned checkpoint directory",
    )
    normalized_path_identity(dirname(canonical_checkpoints)) ==
        normalized_path_identity(canonical_run) || error(
        "owned checkpoint directory escaped the run directory",
    )
    basename(canonical_checkpoints) == "checkpoints" || error(
        "owned checkpoint directory basename changed",
    )
    return canonical_run
end

function append_checkpoint_manifest(run_dir, artifact)
    String(artifact.kind) == "training" || error(
        "training checkpoint manifest accepts only training artifacts",
    )
    checkpoint_root = canonical_existing_directory(
        joinpath(run_dir, "checkpoints"),
        "owned checkpoint directory",
    )
    artifact_path = canonical_existing_file(
        String(artifact.path),
        "training checkpoint artifact",
    )
    relative_artifact = relpath(artifact_path, checkpoint_root)
    relative_components = splitpath(relative_artifact)
    (!isabspath(relative_artifact) &&
     !isempty(relative_components) &&
     first(relative_components) != "..") || error(
        "checkpoint artifact is outside the run checkpoint directory",
    )
    actual_sha256 = sha256_file(artifact_path)
    actual_sha256 == String(artifact.sha256) || error(
        "checkpoint artifact SHA-256 changed before manifest commit",
    )
    filesize(artifact_path) == Int(artifact.bytes) || error(
        "checkpoint artifact byte size changed before manifest commit",
    )
    artifact_match = match(
        TRAINING_CHECKPOINT_FILENAME_PATTERN,
        basename(artifact_path),
    )
    artifact_match === nothing && error(
        "training checkpoint artifact filename is noncanonical",
    )
    parse(Int, only(artifact_match.captures)) ==
        Int(artifact.update) || error(
        "training checkpoint artifact filename update differs",
    )
    path = joinpath(run_dir, "checkpoint_manifest.jsonl")
    existing_bytes = UInt8[]
    existing_sha256 = nothing
    matched = false
    manifest_records = Dict{Int,String}()
    if isfile(path)
        canonical_manifest = canonical_existing_file(
            path,
            "owned checkpoint manifest",
        )
        canonical_manifest == path || error(
            "owned checkpoint manifest path is noncanonical",
        )
        existing_bytes = read(canonical_manifest)
        isempty(existing_bytes) &&
            error("checkpoint manifest is empty")
        existing_sha256 = bytes2hex(sha256(existing_bytes))
        for (line_number, line) in
            enumerate(eachline(IOBuffer(existing_bytes)))
            isempty(strip(line)) && error(
                "checkpoint manifest contains an empty line at " *
                "$line_number",
            )
            existing = JSON3.read(line)
            existing isa JSON3.Object || error(
                "checkpoint manifest line $line_number must be a JSON object",
            )
            Set(String(key) for key in keys(existing)) ==
                CHECKPOINT_MANIFEST_PROPERTIES || error(
                "checkpoint manifest line $line_number properties differ",
            )
            strictly_typed_manifest_record(line, line_number)
            update = strict_manifest_integer(
                existing,
                :update,
                line_number,
            )
            update >= 0 || error(
                "checkpoint manifest line $line_number update is negative",
            )
            haskey(manifest_records, update) && error(
                "checkpoint manifest duplicates update $update",
            )
            kind = strict_manifest_string(
                existing,
                :kind,
                line_number,
            )
            kind == "training" || error(
                "checkpoint manifest line $line_number kind is not training",
            )
            existing_path = strict_manifest_string(
                existing,
                :path,
                line_number,
            )
            canonical_existing_path = canonical_existing_file(
                existing_path,
                "checkpoint manifest line $line_number artifact",
            )
            existing_path == canonical_existing_path || error(
                "checkpoint manifest line $line_number path is noncanonical",
            )
            normalized_path_identity(dirname(canonical_existing_path)) ==
                normalized_path_identity(checkpoint_root) || error(
                "checkpoint manifest line $line_number escaped the " *
                "checkpoint directory",
            )
            filename_match = match(
                TRAINING_CHECKPOINT_FILENAME_PATTERN,
                basename(canonical_existing_path),
            )
            filename_match === nothing && error(
                "checkpoint manifest line $line_number filename differs",
            )
            parse(Int, only(filename_match.captures)) == update || error(
                "checkpoint manifest line $line_number filename update " *
                "differs",
            )
            bytes = strict_manifest_integer(
                existing,
                :bytes,
                line_number,
            )
            bytes == filesize(canonical_existing_path) || error(
                "checkpoint manifest line $line_number byte size differs",
            )
            sha256_value = strict_manifest_string(
                existing,
                :sha256,
                line_number,
            )
            occursin(r"^[0-9a-f]{64}$", sha256_value) || error(
                "checkpoint manifest line $line_number SHA-256 is " *
                "noncanonical",
            )
            sha256_file(canonical_existing_path) == sha256_value || error(
                "checkpoint manifest line $line_number SHA-256 differs",
            )
            manifest_records[update] = canonical_existing_path
            if update == Int(artifact.update)
                kind == String(artifact.kind) &&
                    existing_path == String(artifact.path) &&
                    sha256_value == String(artifact.sha256) &&
                    bytes == Int(artifact.bytes) || error(
                    "checkpoint manifest has a conflicting " *
                    "$(artifact.kind) artifact at update " *
                    "$(artifact.update)",
                )
                matched = true
            end
        end
    end

    live_checkpoints = Dict{Int,String}()
    for entry in readdir(checkpoint_root; join=true)
        canonical_entry = canonical_existing_file(
            entry,
            "owned checkpoint directory entry",
        )
        entry == canonical_entry || error(
            "owned checkpoint directory entry is noncanonical",
        )
        filename_match = match(
            TRAINING_CHECKPOINT_FILENAME_PATTERN,
            basename(canonical_entry),
        )
        filename_match === nothing && error(
            "owned checkpoint directory contains an unexpected entry",
        )
        update = parse(Int, only(filename_match.captures))
        haskey(live_checkpoints, update) && error(
            "owned checkpoint directory duplicates update $update",
        )
        live_checkpoints[update] = canonical_entry
    end
    expected_live_updates = Set(keys(manifest_records))
    push!(expected_live_updates, Int(artifact.update))
    Set(keys(live_checkpoints)) == expected_live_updates || error(
        "owned checkpoint manifest differs from live checkpoint files",
    )
    for (update, existing_path) in manifest_records
        normalized_path_identity(live_checkpoints[update]) ==
            normalized_path_identity(existing_path) || error(
            "owned checkpoint manifest path differs at update $update",
        )
    end
    normalized_path_identity(
        live_checkpoints[Int(artifact.update)],
    ) == normalized_path_identity(artifact_path) || error(
        "training checkpoint artifact differs from the live update",
    )
    matched && return path

    temporary, temporary_io = mktemp(run_dir; cleanup=false)
    try
        if !isempty(existing_bytes)
            write(temporary_io, existing_bytes)
            if last(existing_bytes) != UInt8('\n')
                write(temporary_io, '\n')
            end
        end
        JSON3.write(temporary_io, artifact)
        write(temporary_io, '\n')
        flush(temporary_io)
        close(temporary_io)
        canonical_temporary = canonical_existing_file(
            temporary,
            "owned checkpoint manifest temporary",
        )
        dirname(canonical_temporary) ==
            canonical_existing_directory(run_dir, "owned run directory") ||
            error("checkpoint manifest temporary escaped the run directory")
        if existing_sha256 === nothing
            !ispath(path) || error(
                "checkpoint manifest appeared during its initial commit",
            )
        else
            isfile(path) && sha256_file(path) == existing_sha256 || error(
                "checkpoint manifest changed during append",
            )
        end
        mv(temporary, path; force=true)
    catch
        isopen(temporary_io) && close(temporary_io)
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return path
end

function normalize_checkpoint_reference(
    path::AbstractString,
    sha256_value::AbstractString,
    update::Integer;
    kind::AbstractString="training",
)
    canonical_path = realpath(path)
    canonical_sha256 = lowercase(String(sha256_value))
    occursin(r"^[0-9a-f]{64}$", canonical_sha256) || error(
        "checkpoint SHA-256 is not canonical",
    )
    actual_sha256 = sha256_file(canonical_path)
    actual_sha256 == canonical_sha256 || error(
        "checkpoint reference SHA-256 differs from the file",
    )
    Int(update) >= 0 || error("checkpoint update must be nonnegative")
    return (;
        kind=String(kind),
        path=canonical_path,
        bytes=filesize(canonical_path),
        sha256=canonical_sha256,
        update=Int(update),
    )
end

function strict_manifest_integer(record, name::Symbol, line_number::Int)
    hasproperty(record, name) || error(
        "checkpoint manifest line $line_number is missing $(String(name))",
    )
    value = getproperty(record, name)
    value isa Integer && !(value isa Bool) || error(
        "checkpoint manifest line $line_number $(String(name)) " *
        "must be an integer",
    )
    return try
        Int(value)
    catch
        error(
            "checkpoint manifest line $line_number $(String(name)) " *
            "does not fit Int",
        )
    end
end

function strictly_typed_manifest_record(
    line::AbstractString,
    line_number::Int,
)
    return try
        JSON3.read(line, CHECKPOINT_MANIFEST_RECORD_TYPE)
    catch exception
        error(
            "checkpoint manifest line $line_number has invalid field " *
            "types: $(sprint(showerror, exception))",
        )
    end
end

function strict_manifest_string(record, name::Symbol, line_number::Int)
    hasproperty(record, name) || error(
        "checkpoint manifest line $line_number is missing $(String(name))",
    )
    value = getproperty(record, name)
    value isa AbstractString || error(
        "checkpoint manifest line $line_number $(String(name)) " *
        "must be a string",
    )
    return String(value)
end

"""
Validate a resume/finalize parent checkpoint against immutable, pre-existing
evidence. This function is deliberately read-only: it never creates, repairs,
rewrites, or appends the parent checkpoint manifest.
"""
function validate_parent_checkpoint_manifest(checkpoint_reference)
    String(checkpoint_reference.kind) == "training" || error(
        "parent checkpoint manifest validation requires training kind",
    )
    checkpoint_path = canonical_existing_file(
        String(checkpoint_reference.path),
        "parent checkpoint",
    )
    checkpoint_dir = dirname(checkpoint_path)
    basename(checkpoint_dir) == "checkpoints" || error(
        "parent checkpoint must be stored in a run checkpoints directory",
    )
    run_dir = dirname(checkpoint_dir)
    canonical_run =
        canonical_existing_directory(run_dir, "parent run directory")
    canonical_checkpoints = canonical_existing_directory(
        checkpoint_dir,
        "parent checkpoint directory",
    )
    normalized_path_identity(dirname(canonical_checkpoints)) ==
        normalized_path_identity(canonical_run) || error(
        "parent checkpoint directory escaped the parent run",
    )
    normalized_path_identity(dirname(checkpoint_path)) ==
        normalized_path_identity(canonical_checkpoints) || error(
        "parent checkpoint escaped the parent checkpoint directory",
    )

    manifest_path = canonical_existing_file(
        joinpath(canonical_run, "checkpoint_manifest.jsonl"),
        "parent checkpoint manifest",
    )
    manifest_bytes = read(manifest_path)
    isempty(manifest_bytes) && error("parent checkpoint manifest is empty")
    records = Dict{Int,NamedTuple}()
    for (line_number, line) in
        enumerate(eachline(IOBuffer(manifest_bytes)))
        isempty(strip(line)) && error(
            "checkpoint manifest contains a blank line at line $line_number",
        )
        record = JSON3.read(line)
        record isa JSON3.Object || error(
            "checkpoint manifest line $line_number must be a JSON object",
        )
        observed_properties =
            Set(String(key) for key in keys(record))
        observed_properties == CHECKPOINT_MANIFEST_PROPERTIES || error(
            "checkpoint manifest line $line_number properties differ: " *
            "observed=$(sort!(collect(observed_properties)))",
        )
        strictly_typed_manifest_record(line, line_number)
        update = strict_manifest_integer(record, :update, line_number)
        update >= 0 || error(
            "checkpoint manifest line $line_number update is negative",
        )
        haskey(records, update) && error(
            "checkpoint manifest duplicates update $update",
        )
        kind = strict_manifest_string(record, :kind, line_number)
        kind == "training" || error(
            "checkpoint manifest line $line_number kind is not training",
        )
        record_path = strict_manifest_string(
            record,
            :path,
            line_number,
        )
        isabspath(record_path) || error(
            "checkpoint manifest line $line_number path is not absolute",
        )
        canonical_record_path = canonical_existing_file(
            record_path,
            "checkpoint manifest line $line_number artifact",
        )
        record_path == canonical_record_path || error(
            "checkpoint manifest line $line_number path is not canonical",
        )
        normalized_path_identity(dirname(canonical_record_path)) ==
            normalized_path_identity(canonical_checkpoints) || error(
            "checkpoint manifest line $line_number path escaped " *
            "the checkpoint directory",
        )
        filename_match = match(
            TRAINING_CHECKPOINT_FILENAME_PATTERN,
            basename(canonical_record_path),
        )
        filename_match === nothing && error(
            "checkpoint manifest line $line_number has a non-training " *
            "checkpoint filename",
        )
        parse(Int, only(filename_match.captures)) == update || error(
            "checkpoint manifest line $line_number filename update differs",
        )
        bytes = strict_manifest_integer(record, :bytes, line_number)
        bytes > 0 || error(
            "checkpoint manifest line $line_number byte size is not positive",
        )
        bytes == filesize(canonical_record_path) || error(
            "checkpoint manifest line $line_number byte size differs",
        )
        sha256_value = strict_manifest_string(
            record,
            :sha256,
            line_number,
        )
        occursin(r"^[0-9a-f]{64}$", sha256_value) || error(
            "checkpoint manifest line $line_number SHA-256 is not canonical",
        )
        sha256_file(canonical_record_path) == sha256_value || error(
            "checkpoint manifest line $line_number SHA-256 differs",
        )
        records[update] = (;
            kind,
            path=canonical_record_path,
            bytes,
            sha256=sha256_value,
            update,
        )
    end
    isempty(records) && error("parent checkpoint manifest has no records")

    live_checkpoints = Dict{Int,String}()
    finalization_count = 0
    for entry in readdir(canonical_checkpoints; join=true)
        canonical_entry = canonical_existing_file(
            entry,
            "parent checkpoint directory entry",
        )
        entry == canonical_entry || error(
            "parent checkpoint directory entry is not canonical",
        )
        name = basename(canonical_entry)
        training_match =
            match(TRAINING_CHECKPOINT_FILENAME_PATTERN, name)
        if training_match !== nothing
            update = parse(Int, only(training_match.captures))
            haskey(live_checkpoints, update) && error(
                "parent checkpoint directory duplicates update $update",
            )
            live_checkpoints[update] = canonical_entry
            continue
        end
        finalization_match =
            match(FINALIZATION_CHECKPOINT_FILENAME_PATTERN, name)
        finalization_match === nothing && error(
            "parent checkpoint directory contains an unexpected entry: " *
            canonical_entry,
        )
        finalization_count += 1
        finalization_count <= 1 || error(
            "parent checkpoint directory contains multiple finalization " *
            "checkpoints",
        )
        parse(Int, only(finalization_match.captures)) ==
            Int(checkpoint_reference.update) || error(
            "parent finalization checkpoint update differs from the " *
            "selected training checkpoint",
        )
    end
    Set(keys(records)) == Set(keys(live_checkpoints)) || error(
        "parent checkpoint manifest update set differs from live files",
    )
    for (update, live_path) in live_checkpoints
        normalized_path_identity(records[update].path) ==
            normalized_path_identity(live_path) || error(
            "parent checkpoint manifest path differs at update $update",
        )
    end

    selected_update = Int(checkpoint_reference.update)
    haskey(records, selected_update) || error(
        "selected parent checkpoint is missing from its manifest",
    )
    selected_update == maximum(keys(records)) || error(
        "selected parent checkpoint is not the latest manifest update",
    )
    selected = records[selected_update]
    selected.kind == String(checkpoint_reference.kind) || error(
        "selected parent checkpoint manifest kind differs",
    )
    selected.path == String(checkpoint_reference.path) || error(
        "selected parent checkpoint manifest path differs",
    )
    selected.bytes == Int(checkpoint_reference.bytes) || error(
        "selected parent checkpoint manifest byte size differs",
    )
    selected.sha256 == String(checkpoint_reference.sha256) || error(
        "selected parent checkpoint manifest SHA-256 differs",
    )
    return (;
        path=manifest_path,
        bytes=length(manifest_bytes),
        sha256=bytes2hex(sha256(manifest_bytes)),
        records=length(records),
    )
end

function append_trace(path, record)
    first_write = !isfile(path)
    open(path, "a") do io
        if first_write
            println(io, join(String.(keys(record)), '\t'))
        end
        println(io, join(values(record), '\t'))
    end
    return nothing
end

function training_dynamics(trainer)
    arena = trainer.arena
    model = trainer.model
    blocks = model.blocks
    spikes = 0.0
    spike_count = 0
    entropy = 0.0
    exploitation_entropy = 0.0
    route_count = 0
    probability_mass_error = 0.0
    probability_max_mass_error = 0.0
    hard_load = zeros(Float64, blocks)
    hard_total = 0.0
    workspace_square = 0.0
    workspace_count = 0
    head_pre_square = 0.0
    head_pre_count = 0
    hidden_inv_rms_sum = 0.0
    hidden_inv_rms_min = Inf
    hidden_inv_rms_max = -Inf
    hidden_tanh_derivative_sum = 0.0
    hidden_tanh_derivative_count = 0
    inverse_log_blocks = inv(log(Float64(blocks)))
    has_base_probability =
        hasproperty(arena, :route_base_probability)
    base_probability = has_base_probability ?
        getproperty(arena, :route_base_probability) : nothing
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        hidden_inverse = Float64(arena.hidden_inv_rms[flat])
        hidden_inv_rms_sum += hidden_inverse
        hidden_inv_rms_min = min(hidden_inv_rms_min, hidden_inverse)
        hidden_inv_rms_max = max(hidden_inv_rms_max, hidden_inverse)
        for hidden in 1:model.hidden
            head_pre = Float64(arena.hidden_pre[hidden, flat])
            head_pre_square =
                muladd(head_pre, head_pre, head_pre_square)
            head_pre_count += 1
            hidden_value = Float64(arena.hidden[hidden, flat])
            hidden_tanh_derivative_sum +=
                1.0 - hidden_value * hidden_value
            hidden_tanh_derivative_count += 1
        end
        for cycle in 1:model.cycles
            for node in 1:(blocks * model.node_dim)
                spikes += arena.active_spikes[node, cycle, flat]
                spike_count += 1
            end
            local_entropy = 0.0
            local_exploitation_entropy = 0.0
            probability_mass = 0.0
            for block in 1:blocks
                probability =
                    Float64(arena.route_probability[block, cycle, flat])
                probability_mass += probability
                probability > 0.0 &&
                    (local_entropy -= probability * log(probability))
                if has_base_probability
                    base = Float64(
                        base_probability[block, cycle, flat],
                    )
                    base > 0.0 &&
                        (local_exploitation_entropy -= base * log(base))
                end
                selected = Float64(
                    arena.block_mask[block, cycle, flat],
                )
                hard_load[block] += selected
                hard_total += selected
            end
            entropy += local_entropy * inverse_log_blocks
            exploitation_entropy +=
                local_exploitation_entropy * inverse_log_blocks
            mass_error = abs(probability_mass - 1.0)
            probability_mass_error += mass_error
            probability_max_mass_error =
                max(probability_max_mass_error, mass_error)
            route_count += 1
            for coordinate in 1:model.node_dim
                value = Float64(
                    arena.workspace[coordinate, cycle + 1, flat],
                )
                workspace_square =
                    muladd(value, value, workspace_square)
                workspace_count += 1
            end
        end
    end
    hard_entropy = 0.0
    if hard_total > 0.0
        @inbounds for count in hard_load
            probability = count / hard_total
            probability > 0.0 &&
                (hard_entropy -= probability * log(probability))
        end
    end
    sorted_hard_load = sort(hard_load; rev=true)
    top_count = min(8, length(sorted_hard_load))
    top8_share = hard_total > 0.0 ?
        sum(@view sorted_hard_load[1:top_count]) / hard_total : 0.0
    utility_sum = 0.0
    utility_nonzero = 0
    @inbounds for value in trainer.synapse_utility
        utility_sum += value
        utility_nonzero += value > 0.0f0
    end
    cache_mean(values) = begin
        total = 0.0
        @inbounds for value in values
            total += Float64(value)
        end
        total / max(length(values), 1)
    end
    metrics = trainer.metrics
    return (;
        schema_version=4,
        firing_rate=spikes / max(spike_count, 1),
        workspace_route_entropy=entropy / max(route_count, 1),
        workspace_exploitation_entropy=
            has_base_probability ?
            exploitation_entropy / max(route_count, 1) : NaN,
        hard_route_load_entropy=
            hard_entropy * inverse_log_blocks,
        hard_route_effective_blocks=exp(hard_entropy),
        hard_route_top8_share=top8_share,
        route_probability_mass_error=
            probability_mass_error / max(route_count, 1),
        route_probability_max_mass_error=
            probability_max_mass_error,
        workspace_rms=sqrt(
            workspace_square / max(workspace_count, 1),
        ),
        gate_density=Float64(
            ArenaWorkspaceTraining._gate_density(trainer.cache),
        ),
        utility_mean=
            utility_sum / length(trainer.synapse_utility),
        utility_nonzero_fraction=
            utility_nonzero / length(trainer.synapse_utility),
        head_pre_rms=sqrt(
            head_pre_square / max(head_pre_count, 1),
        ),
        hidden_inv_rms_mean=
            hidden_inv_rms_sum / max(arena.valid_count, 1),
        hidden_inv_rms_min=
            isfinite(hidden_inv_rms_min) ? hidden_inv_rms_min : 0.0,
        hidden_inv_rms_max=
            isfinite(hidden_inv_rms_max) ? hidden_inv_rms_max : 0.0,
        hidden_tanh_derivative_mean=
            hidden_tanh_derivative_sum /
            max(hidden_tanh_derivative_count, 1),
        route_selection_gap=metrics.route_selection_gap,
        route_score_rms=metrics.route_score_rms,
        hard_mask_unique_fraction=metrics.hard_mask_unique_fraction,
        hard_mask_cycle_churn=metrics.hard_mask_cycle_churn,
        entropy_floor_violation_fraction=
            metrics.entropy_floor_violation_fraction,
        utility_swap_gap=metrics.utility_swap_gap,
        consolidation_scheduled=metrics.consolidation_scheduled,
        consolidation_actual=metrics.consolidation_actual,
        net_mask_flips=metrics.net_mask_flips,
        gate_probability_mean=
            cache_mean(trainer.cache.gate_probability),
        gate_derivative_mean=
            cache_mean(trainer.cache.gate_derivative),
        delay_mean=cache_mean(trainer.cache.delay),
        delay_derivative_mean=
            cache_mean(trainer.cache.delay_derivative),
        leak_mean=cache_mean(trainer.cache.leak),
        leak_derivative_mean=
            cache_mean(trainer.cache.leak_derivative),
        threshold_mean=cache_mean(trainer.cache.threshold),
        threshold_derivative_mean=
            cache_mean(trainer.cache.threshold_derivative),
        workspace_decay=Float64(trainer.cache.workspace_decay),
        workspace_decay_derivative=
            Float64(trainer.cache.workspace_decay_derivative),
        membrane_threshold_margin_mean=
            metrics.membrane_threshold_margin_mean,
        membrane_threshold_margin_rms=
            metrics.membrane_threshold_margin_rms,
        surrogate_sensitivity_mean=
            metrics.surrogate_sensitivity_mean,
        surrogate_sensitivity_rms=
            metrics.surrogate_sensitivity_rms,
        eligibility_rms=metrics.eligibility_rms,
        local_q_loss=metrics.local_q_loss,
        local_death_loss=metrics.local_death_loss,
        local_quantile_loss=metrics.local_quantile_loss,
        local_geometry_loss=metrics.local_geometry_loss,
    )
end

const REQUIRED_TRAINING_DYNAMICS_PROPERTIES = (
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

const REQUIRED_LOSS_RECORD_PROPERTIES = (
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

const REQUIRED_COMPONENT_LOSS_PROPERTIES = (
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

const COMPONENT_LOSS_WINDOW_PROPERTIES = (
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
    REQUIRED_COMPONENT_LOSS_PROPERTIES...,
    COMPONENT_LOSS_WINDOW_PROPERTIES...,
    :gradient_norm,
    :enabled_synapses,
    :structural_flips_total,
    :training_dynamics_schema_version,
    Tuple(
        name for name in REQUIRED_TRAINING_DYNAMICS_PROPERTIES
        if name != :schema_version
    )...,
    :states_per_second,
    :cpu_percent,
    :hot_allocation_bytes,
    :hot_gc_seconds,
    :shadow_seconds,
)

const TRACE_INTEGER_PROPERTIES = (
    :trace_schema_version,
    :update,
    :teacher_states,
    :window_updates,
    :component_loss_alias_schema_version,
    :enabled_synapses,
    :structural_flips_total,
    :training_dynamics_schema_version,
    :net_mask_flips,
    :hot_allocation_bytes,
)

const TRACE_BOOL_PROPERTIES = (
    :consolidation_scheduled,
    :consolidation_actual,
)

const TRACE_STRING_PROPERTIES = (
    :q_huber_loss_alias_of,
    :raw_top_gap_loss_alias_of,
    :component_loss_alias_identity,
)

function validate_loss_record(loss)
    propertynames(loss) == REQUIRED_LOSS_RECORD_PROPERTIES || error(
        "loss record properties differ",
    )
    for name in REQUIRED_LOSS_RECORD_PROPERTIES
        name == :valid_candidates && continue
        raw_value = getproperty(loss, name)
        raw_value isa Float32 || error(
            "loss record field must be Float32: $(String(name))",
        )
        value = Float64(raw_value)
        isfinite(value) || error(
            "loss record is non-finite: $(String(name))",
        )
    end
    for name in REQUIRED_COMPONENT_LOSS_PROPERTIES
        Float64(getproperty(loss, name)) >= 0.0 || error(
            "component loss is negative: $(String(name))",
        )
    end
    loss.valid_candidates isa Int || error(
        "loss valid_candidates must be Int",
    )
    Int(loss.valid_candidates) >= 0 || error(
        "loss valid_candidates is negative",
    )
    reinterpret(UInt32, loss.q_huber_loss) ==
        reinterpret(UInt32, loss.old_q_loss) || error(
        "q_huber_loss is not the bit-exact old_q_loss compatibility alias",
    )
    reinterpret(UInt32, loss.raw_top_gap_loss) ==
        reinterpret(UInt32, loss.margin_loss) || error(
        "raw_top_gap_loss is not the bit-exact margin_loss " *
        "compatibility alias",
    )
    return true
end

function validate_component_loss_snapshot(snapshot)
    propertynames(snapshot) == fieldnames(ComponentLossTelemetry) || error(
        "component-loss telemetry properties differ",
    )
    for name in propertynames(snapshot)
        raw_value = getproperty(snapshot, name)
        raw_value isa Float64 || error(
            "component-loss telemetry field must be Float64: " *
            String(name),
        )
        value = raw_value
        isfinite(value) || error(
            "component-loss telemetry is non-finite: $(String(name))",
        )
        value >= 0.0 || error(
            "component-loss telemetry is negative: $(String(name))",
        )
    end
    for (alias_name, canonical_name) in (
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
        reinterpret(UInt64, getproperty(snapshot, alias_name)) ==
            reinterpret(UInt64, getproperty(snapshot, canonical_name)) ||
            error(
                "component-loss compatibility alias differs: " *
                String(alias_name),
            )
    end
    return true
end

function validate_training_dynamics_snapshot(snapshot)
    propertynames(snapshot) ==
        REQUIRED_TRAINING_DYNAMICS_PROPERTIES || error(
        "training dynamics telemetry properties differ",
    )
    snapshot.schema_version isa Int || error(
        "training dynamics telemetry schema version must be Int",
    )
    snapshot.schema_version == 4 || error(
        "training dynamics telemetry schema version differs",
    )
    snapshot.consolidation_scheduled isa Bool || error(
        "consolidation_scheduled telemetry must be Bool",
    )
    snapshot.consolidation_actual isa Bool || error(
        "consolidation_actual telemetry must be Bool",
    )
    Bool(snapshot.consolidation_actual) &&
        !Bool(snapshot.consolidation_scheduled) && error(
        "actual consolidation was not scheduled",
    )
    snapshot.net_mask_flips isa Int || error(
        "net_mask_flips telemetry must be Int",
    )
    Int(snapshot.net_mask_flips) >= 0 || error(
        "net_mask_flips telemetry is negative",
    )
    for name in REQUIRED_TRAINING_DYNAMICS_PROPERTIES
        name in (
            :schema_version,
            :consolidation_scheduled,
            :consolidation_actual,
            :net_mask_flips,
        ) && continue
        raw_value = getproperty(snapshot, name)
        raw_value isa Float64 || error(
            "training dynamics telemetry field must be Float64: " *
            String(name),
        )
        isfinite(raw_value) || error(
            "training dynamics telemetry is non-finite: $(String(name))",
        )
    end
    for name in (
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
        value = Float64(getproperty(snapshot, name))
        -1.0e-6 <= value <= 1.0 + 1.0e-6 || error(
            "training dynamics unit-interval telemetry differs: " *
            String(name),
        )
    end
    for name in (
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
        :leak_mean,
        :leak_derivative_mean,
        :threshold_mean,
        :threshold_derivative_mean,
        :workspace_decay_derivative,
        :membrane_threshold_margin_rms,
        :surrogate_sensitivity_mean,
        :surrogate_sensitivity_rms,
        :eligibility_rms,
        :local_q_loss,
        :local_death_loss,
        :local_quantile_loss,
        :local_geometry_loss,
    )
        Float64(getproperty(snapshot, name)) >= 0.0 || error(
            "training dynamics nonnegative telemetry differs: " *
            String(name),
        )
    end
    return true
end

function teacher_listnet_entropy(dataset, rows)
    isempty(rows) && error("teacher entropy rows are empty")
    total = 0.0
    @inbounds for row in rows
        count = Int(dataset.action_counts[row])
        count >= 1 || error("teacher state has no candidates")
        teacher = @view dataset.teacher_q[1:count, row]
        teacher_mean = Float32(mean(teacher))
        teacher_scale = max(
            Float32(std(teacher; corrected=false)),
            1.0f-4,
        )
        maximum_logit = -Inf32
        for candidate in 1:count
            logit = (
                Float32(teacher[candidate]) - teacher_mean
            ) / (teacher_scale * 0.50f0)
            maximum_logit = max(maximum_logit, logit)
        end
        normalizer = 0.0f0
        for candidate in 1:count
            logit = (
                Float32(teacher[candidate]) - teacher_mean
            ) / (teacher_scale * 0.50f0)
            normalizer += exp(logit - maximum_logit)
        end
        inverse_normalizer = inv(max(normalizer, 1.0f-12))
        state_entropy = 0.0
        for candidate in 1:count
            logit = (
                Float32(teacher[candidate]) - teacher_mean
            ) / (teacher_scale * 0.50f0)
            probability =
                Float64(exp(logit - maximum_logit) * inverse_normalizer)
            probability > 0.0 &&
                (state_entropy -= probability * log(probability))
        end
        total += state_entropy
    end
    return total / length(rows)
end

function evaluate(model, parameters, states, dataset, rows, batch)
    metrics = evaluation_metrics(
        dataset,
        rows,
        batch,
        packed -> first(model(packed.inputs, parameters, states)),
    )
    if hasproperty(metrics, :teacher_entropy) &&
       hasproperty(metrics, :listnet_kl)
        return metrics
    end
    teacher_entropy = teacher_listnet_entropy(dataset, rows)
    listnet_kl = max(
        Float64(metrics.listnet_loss) - teacher_entropy,
        0.0,
    )
    return merge(metrics, (; teacher_entropy, listnet_kl))
end

function reducible_objective_metrics(metrics)
    listnet_ce = hasproperty(metrics, :listnet_loss) ?
        Float64(metrics.listnet_loss) :
        Float64(metrics.diagnostic_listnet_loss)
    teacher_entropy = hasproperty(metrics, :teacher_entropy) ?
        Float64(metrics.teacher_entropy) : NaN
    listnet_kl = hasproperty(metrics, :listnet_kl) ?
        Float64(metrics.listnet_kl) :
        listnet_ce - teacher_entropy
    composite_excess = hasproperty(metrics, :composite_loss) ?
        Float64(metrics.composite_loss) - teacher_entropy : NaN
    return (;
        listnet_ce,
        teacher_entropy,
        listnet_kl,
        composite_excess,
    )
end

function checkpoint_payload(
    trainer,
    sampler,
    initial_parameters,
    config,
    initial_metrics,
    progress,
    parent_checkpoint,
    persistent_team_warmup,
    segment_state,
    last_training_dynamics;
    checkpoint_kind::Symbol=:training,
    finalization=nothing,
)
    validate_loss_record(trainer.last_loss)
    validate_component_loss_snapshot(
        component_loss_snapshot(progress.component_losses),
    )
    validate_training_dynamics_snapshot(last_training_dynamics)
    validate_optimizer_contract(trainer.optimizer, config.optimizer)
    trainer.structure_weight == config.structure_weight || error(
        "trainer structure weight differs from checkpoint config",
    )
    progress.updates == trainer.optimizer.step || error(
        "progress update total differs from optimizer step",
    )
    progress.teacher_states ==
        trainer.optimizer.step * trainer.arena.state_batch || error(
        "progress teacher-state total differs from optimizer step",
    )
    Int(segment_state.start_update) +
        Int(segment_state.updates) == trainer.optimizer.step || error(
        "checkpoint segment timing does not end at optimizer step",
    )
    Float64(segment_state.overall_seconds) >= 0.0 || error(
        "checkpoint segment timing is negative",
    )
    if trainer.optimizer.step > 0
        persistent_team_warmup === nothing && error(
            "trained checkpoint is missing persistent warmup artifact",
        )
        isfinite(trainer.last_loss.composite_loss) || error(
            "trained checkpoint has no finite last loss",
        )
        isfinite(trainer.last_gradient_norm) || error(
            "trained checkpoint has no finite last gradient norm",
        )
    end
    config.dataset_content_sha256 ==
        config.production_contract.dataset_content_sha256 || error(
        "config dataset content binding differs from production contract",
    )
    config.runtime_provenance ==
        config.production_contract.runtime_provenance || error(
        "config runtime provenance differs from production contract",
    )
    return (;
        format=CHECKPOINT_FORMAT,
        version=CHECKPOINT_VERSION,
        component_loss_alias_contract=COMPONENT_LOSS_ALIAS_CONTRACT,
        checkpoint_kind,
        update=trainer.optimizer.step,
        parent_checkpoint,
        dataset_content_sha256=config.dataset_content_sha256,
        dataset_integrity=config.dataset_integrity,
        runtime_provenance=config.runtime_provenance,
        parameters=trainer.parameters,
        optimizer=(;
            first_moment=trainer.optimizer.first_moment,
            second_moment=trainer.optimizer.second_moment,
            learning_rate=trainer.optimizer.learning_rate,
            beta1=trainer.optimizer.beta1,
            beta2=trainer.optimizer.beta2,
            beta1_power=trainer.optimizer.beta1_power,
            beta2_power=trainer.optimizer.beta2_power,
            epsilon=trainer.optimizer.epsilon,
            weight_decay=trainer.optimizer.weight_decay,
            step=trainer.optimizer.step,
        ),
        trainer_state=(;
            last_loss=trainer.last_loss,
            last_gradient_norm=trainer.last_gradient_norm,
            structure_weight=trainer.structure_weight,
        ),
        total_structural_flips=trainer.total_structural_flips,
        synapse_utility=trainer.synapse_utility,
        utility_updates=trainer.utility_updates,
        sampler_state=sampler_snapshot(sampler),
        initial_parameters,
        config,
        initial_metrics,
        progress=progress_snapshot(progress),
        persistent_team_warmup,
        segment_state,
        last_training_dynamics,
        finalization,
    )
end

function save_checkpoint!(
    path,
    trainer,
    sampler,
    initial_parameters,
    config,
    initial_metrics,
    progress,
    parent_checkpoint,
    persistent_team_warmup,
    segment_state,
    last_training_dynamics;
    checkpoint_kind::Symbol=:training,
    finalization=nothing,
)
    payload = checkpoint_payload(
        trainer,
        sampler,
        initial_parameters,
        config,
        initial_metrics,
        progress,
        parent_checkpoint,
        persistent_team_warmup,
        segment_state,
        last_training_dynamics;
        checkpoint_kind,
        finalization,
    )
    atomic_jldsave(path; payload)
    return normalize_checkpoint_reference(
        path,
        sha256_file(path),
        trainer.optimizer.step;
        kind=String(checkpoint_kind),
    )
end

function validate_loaded_checkpoint_invariants(payload)
    for name in (
        :update,
        :parameters,
        :optimizer,
        :initial_parameters,
        :config,
        :initial_metrics,
        :total_structural_flips,
        :synapse_utility,
        :utility_updates,
        :sampler_state,
    )
        hasproperty(payload, name) || error(
            "v3 resume checkpoint is missing $(String(name))",
        )
    end
    payload.update isa Int || error(
        "resume checkpoint update must be Int",
    )
    update = payload.update
    update >= 0 || error("resume checkpoint update is negative")
    hasproperty(payload.config, :state_batch) || error(
        "resume checkpoint config is missing state_batch",
    )
    payload.config.state_batch isa Int || error(
        "resume checkpoint config state_batch must be Int",
    )
    state_batch = payload.config.state_batch
    state_batch >= 1 || error(
        "resume checkpoint config state_batch is not positive",
    )
    hasproperty(payload.config, :optimizer) || error(
        "resume checkpoint config is missing optimizer contract",
    )
    validate_optimizer_contract(
        payload.optimizer,
        payload.config.optimizer,
    )
    hasproperty(payload.optimizer, :step) || error(
        "resume checkpoint optimizer is missing step",
    )
    payload.optimizer.step isa Int || error(
        "resume checkpoint optimizer step must be Int",
    )
    payload.optimizer.step == update || error(
        "resume checkpoint optimizer clock differs from update",
    )

    progress = restore_progress(payload.progress)
    progress.updates == update || error(
        "resume checkpoint progress update total differs",
    )
    progress.teacher_states == update * state_batch || error(
        "resume checkpoint progress teacher-state total differs",
    )
    progress.candidates >= 0 || error(
        "resume checkpoint progress candidate total is negative",
    )
    progress.hot_allocation_bytes >= 0 || error(
        "resume checkpoint hot allocation total is negative",
    )
    progress.window_updates >= 0 || error(
        "resume checkpoint active window update count is negative",
    )
    for name in (
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
        value = getproperty(progress, name)
        isfinite(value) && value >= 0.0 || error(
            "resume checkpoint progress field differs: $(String(name))",
        )
    end

    segment = payload.segment_state
    propertynames(segment) == (
        :start_update,
        :updates,
        :overall_seconds,
    ) || error("resume checkpoint segment-state properties differ")
    segment.start_update isa Int || error(
        "resume checkpoint segment start update must be Int",
    )
    segment.updates isa Int || error(
        "resume checkpoint segment updates must be Int",
    )
    segment.start_update >= 0 || error(
        "resume checkpoint segment start update is negative",
    )
    segment.updates >= 0 || error(
        "resume checkpoint segment updates is negative",
    )
    segment.start_update + segment.updates == update || error(
        "resume checkpoint segment timing does not end at update",
    )
    segment.overall_seconds isa Float64 || error(
        "resume checkpoint segment overall_seconds must be Float64",
    )
    isfinite(segment.overall_seconds) &&
        segment.overall_seconds >= 0.0 || error(
        "resume checkpoint segment overall_seconds differs",
    )

    payload.total_structural_flips isa Int || error(
        "resume checkpoint total_structural_flips must be Int",
    )
    payload.total_structural_flips >= 0 || error(
        "resume checkpoint total_structural_flips is negative",
    )
    payload.utility_updates isa Int || error(
        "resume checkpoint utility_updates must be Int",
    )
    payload.utility_updates >= 0 || error(
        "resume checkpoint utility_updates is negative",
    )
    if Symbol(payload.config.learning_mode) === :local_hybrid
        payload.utility_updates == update || error(
            "resume checkpoint utility update clock differs",
        )
    end
    if update > 0
        payload.persistent_team_warmup === nothing && error(
            "trained resume checkpoint is missing persistent warmup evidence",
        )
        isfinite(Float64(payload.trainer_state.last_gradient_norm)) ||
            error("trained resume checkpoint gradient norm is non-finite")
    end

    keys(payload.parameters) == keys(payload.initial_parameters) || error(
        "resume checkpoint initial parameter registry differs",
    )
    keys(payload.optimizer.first_moment) == keys(payload.parameters) ||
        error("resume checkpoint first-moment registry differs")
    keys(payload.optimizer.second_moment) == keys(payload.parameters) ||
        error("resume checkpoint second-moment registry differs")
    for name in keys(payload.parameters)
        parameter = getproperty(payload.parameters, name)
        initial = getproperty(payload.initial_parameters, name)
        first_moment = getproperty(payload.optimizer.first_moment, name)
        second_moment = getproperty(payload.optimizer.second_moment, name)
        size(initial) == size(parameter) || error(
            "resume checkpoint initial parameter shape differs for $name",
        )
        size(first_moment) == size(parameter) || error(
            "resume checkpoint first-moment shape differs for $name",
        )
        size(second_moment) == size(parameter) || error(
            "resume checkpoint second-moment shape differs for $name",
        )
        all(isfinite, parameter) || error(
            "resume checkpoint parameter is non-finite: $name",
        )
        all(isfinite, initial) || error(
            "resume checkpoint initial parameter is non-finite: $name",
        )
        all(isfinite, first_moment) || error(
            "resume checkpoint first moment is non-finite: $name",
        )
        all(isfinite, second_moment) || error(
            "resume checkpoint second moment is non-finite: $name",
        )
    end
    all(value -> isfinite(value) && value >= 0.0f0,
        payload.synapse_utility) || error(
        "resume checkpoint synapse utility differs",
    )
    return progress
end

function load_checkpoint(path, expected_sha256)
    checkpoint_path = abspath(path)
    isfile(checkpoint_path) || error(
        "resume checkpoint does not exist: $checkpoint_path",
    )
    isempty(strip(expected_sha256)) && error(
        "resume requires an explicit SWSNN_RESUME_SHA256",
    )
    actual_sha256 = sha256_file(checkpoint_path)
    lowercase(expected_sha256) == actual_sha256 ||
        error("resume checkpoint SHA-256 differs")
    file = JLD2.load(checkpoint_path)
    haskey(file, "payload") || error("resume checkpoint has no payload")
    payload = file["payload"]
    payload.format isa String || error(
        "resume checkpoint format must be String",
    )
    payload.format == CHECKPOINT_FORMAT || error(
        "resume checkpoint format differs",
    )
    payload.version isa Int || error(
        "resume checkpoint version must be Int",
    )
    payload.version == CHECKPOINT_VERSION || error(
        "unsupported resume checkpoint version $(payload.version); " *
        "production resume requires v$CHECKPOINT_VERSION and rejects v1/v2",
    )
    hasproperty(payload, :checkpoint_kind) || error(
        "resume checkpoint kind is missing",
    )
    payload.checkpoint_kind isa Symbol || error(
        "resume checkpoint kind must be Symbol",
    )
    payload.update isa Int || error(
        "resume checkpoint update must be Int",
    )
    checkpoint_reference = normalize_checkpoint_reference(
        checkpoint_path,
        actual_sha256,
        Int(payload.update);
        kind=String(payload.checkpoint_kind),
    )
    for name in (
        :parent_checkpoint,
        :component_loss_alias_contract,
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
        :trainer_state,
        :progress,
        :persistent_team_warmup,
        :segment_state,
        :last_training_dynamics,
    )
        hasproperty(payload, name) || error(
            "v3 resume checkpoint is missing $(String(name))",
        )
    end
    payload.component_loss_alias_contract ==
        COMPONENT_LOSS_ALIAS_CONTRACT || error(
        "checkpoint component-loss alias contract differs",
    )
    payload.dataset_content_sha256 ==
        payload.config.dataset_content_sha256 || error(
        "checkpoint dataset content binding differs from config",
    )
    payload.dataset_integrity == payload.config.dataset_integrity ||
        error("checkpoint dataset integrity differs from config")
    payload.runtime_provenance == payload.config.runtime_provenance ||
        error("checkpoint runtime provenance differs from config")
    validate_loss_record(payload.trainer_state.last_loss)
    validate_training_dynamics_snapshot(payload.last_training_dynamics)
    validate_loaded_checkpoint_invariants(payload)
    return payload, checkpoint_reference
end

function restore_trainer!(
    trainer,
    payload,
    optimizer_config,
)
    validate_optimizer_contract(payload.optimizer, optimizer_config)
    keys(payload.parameters) == keys(trainer.parameters) ||
        error("resume parameter registry differs")
    for name in keys(trainer.parameters)
        destination = getproperty(trainer.parameters, name)
        source = getproperty(payload.parameters, name)
        size(destination) == size(source) ||
            error("resume parameter shape differs for $name")
        destination .= source
        getproperty(trainer.optimizer.first_moment, name) .=
            getproperty(payload.optimizer.first_moment, name)
        getproperty(trainer.optimizer.second_moment, name) .=
            getproperty(payload.optimizer.second_moment, name)
    end
    trainer.optimizer.learning_rate =
        Float32(payload.optimizer.learning_rate)
    trainer.optimizer.beta1 = Float32(payload.optimizer.beta1)
    trainer.optimizer.beta2 = Float32(payload.optimizer.beta2)
    trainer.optimizer.beta1_power =
        Float32(payload.optimizer.beta1_power)
    trainer.optimizer.beta2_power =
        Float32(payload.optimizer.beta2_power)
    trainer.optimizer.epsilon = Float32(payload.optimizer.epsilon)
    trainer.optimizer.weight_decay =
        Float32(payload.optimizer.weight_decay)
    trainer.optimizer.step = Int(payload.optimizer.step)
    trainer.optimizer.step == Int(payload.update) ||
        error("resume optimizer clock differs")
    trainer.structure_weight =
        Float32(payload.trainer_state.structure_weight)
    trainer.structure_weight == optimizer_config.structure_weight ||
        error("resume structure weight differs")
    trainer.total_structural_flips =
        Int(payload.total_structural_flips)
    size(payload.synapse_utility) ==
        size(trainer.synapse_utility) ||
        error("resume synapse utility shape differs")
    trainer.synapse_utility .= payload.synapse_utility
    trainer.utility_updates = Int(payload.utility_updates)
    trainer.last_loss = payload.trainer_state.last_loss
    trainer.last_gradient_norm =
        Float64(payload.trainer_state.last_gradient_norm)
    ArenaWorkspaceTraining.refresh_parameter_cache!(
        trainer.cache,
        trainer.parameters,
    )
    validate_optimizer_contract(
        trainer.optimizer,
        optimizer_config,
    )
    # Defense in depth after the in-memory restore.  The same payload was
    # validated before run-directory reservation; this second check proves
    # that restoration did not coerce, omit, or alias any persisted state.
    validate_loaded_checkpoint_invariants(payload)
    trainer.optimizer.step == payload.update || error(
        "restored trainer optimizer clock differs",
    )
    trainer.total_structural_flips ==
        payload.total_structural_flips || error(
        "restored trainer structural flip count differs",
    )
    trainer.utility_updates == payload.utility_updates || error(
        "restored trainer utility update count differs",
    )
    trainer.last_loss == payload.trainer_state.last_loss || error(
        "restored trainer last loss differs",
    )
    trainer.last_gradient_norm ==
        Float64(payload.trainer_state.last_gradient_norm) || error(
        "restored trainer gradient norm differs",
    )
    trainer.synapse_utility == payload.synapse_utility || error(
        "restored trainer synapse utility differs",
    )
    for name in keys(trainer.parameters)
        getproperty(trainer.parameters, name) ==
            getproperty(payload.parameters, name) || error(
            "restored trainer parameter differs: $name",
        )
        getproperty(trainer.optimizer.first_moment, name) ==
            getproperty(payload.optimizer.first_moment, name) || error(
            "restored trainer first moment differs: $name",
        )
        getproperty(trainer.optimizer.second_moment, name) ==
            getproperty(payload.optimizer.second_moment, name) || error(
            "restored trainer second moment differs: $name",
        )
    end
    return trainer
end

function phase_totals(progress::ProgressTotals)
    accounted_seconds =
        progress.pack_seconds +
        progress.forward_seconds +
        progress.loss_seconds +
        progress.shadow_seconds +
        progress.backward_seconds +
        progress.optimizer_seconds +
        progress.consolidation_seconds
    return (;
        pack_seconds=progress.pack_seconds,
        forward_seconds=progress.forward_seconds,
        loss_seconds=progress.loss_seconds,
        shadow_seconds=progress.shadow_seconds,
        backward_seconds=progress.backward_seconds,
        optimizer_seconds=progress.optimizer_seconds,
        consolidation_seconds=progress.consolidation_seconds,
        accounted_seconds,
        hot_wall_seconds=progress.hot_wall_seconds,
        unattributed_hot_wall_seconds=
            progress.hot_wall_seconds - accounted_seconds,
    )
end

function file_artifact(
    path::AbstractString,
    kind::AbstractString,
    update::Integer,
)
    canonical_path = realpath(path)
    return (;
        kind=String(kind),
        path=canonical_path,
        bytes=filesize(canonical_path),
        sha256=sha256_file(canonical_path),
        update=Int(update),
    )
end

function read_bound_file_artifact(
    artifact,
    label::AbstractString,
)
    for name in (:kind, :path, :bytes, :sha256, :update)
        hasproperty(artifact, name) || error(
            "$label artifact is missing $(String(name))",
        )
    end
    artifact.path isa String || error(
        "$label artifact path must be String",
    )
    artifact.bytes isa Int || error(
        "$label artifact bytes must be Int",
    )
    artifact.sha256 isa String || error(
        "$label artifact SHA-256 must be String",
    )
    artifact.update isa Int || error(
        "$label artifact update must be Int",
    )
    artifact.bytes >= 0 || error(
        "$label artifact byte size is negative",
    )
    occursin(r"^[0-9a-f]{64}$", artifact.sha256) || error(
        "$label artifact SHA-256 is noncanonical",
    )
    canonical_path = canonical_existing_file(
        artifact.path,
        label,
    )
    canonical_path == artifact.path || error(
        "$label artifact path changed",
    )
    bytes = read(canonical_path)
    length(bytes) == artifact.bytes || error(
        "$label artifact byte size changed",
    )
    bytes2hex(sha256(bytes)) == artifact.sha256 || error(
        "$label artifact SHA-256 changed",
    )
    return bytes
end

function verify_bound_file_artifact(
    artifact,
    label::AbstractString,
)
    read_bound_file_artifact(artifact, label)
    return artifact
end

function record_team_teardown!(
    run_dir,
    executor,
    team,
    config,
    update::Integer,
)
    all(team.bindings_released) || error(
        "cannot record team teardown before every binding is released",
    )
    length(team.bindings) == executor.julia_workers || error(
        "team teardown binding count differs from Julia worker count",
    )
    all(binding -> binding !== nothing && binding.verified, team.bindings) ||
        error("team teardown contains an unverified startup binding")
    ArenaWorkspaceTraining.Queue.isclosed(executor.queue) || error(
        "cannot record team teardown while the arena queue is open",
    )
    ArenaWorkspaceTraining.Queue.approx_length(executor.queue) == 0 ||
        error("cannot record team teardown with queued work")
    executor.remaining[] == 0 || error(
        "cannot record team teardown with remaining work",
    )
    executor.shutdown_requested[] == UInt32(1) || error(
        "cannot record team teardown without shutdown acknowledgement",
    )
    bindings = [
        (;
            worker_slot=Int(binding.worker_slot),
            julia_thread_id=Int(binding.julia_thread_id),
            cpu_set_id=binding.cpu_set_id === nothing ?
                nothing : Int(binding.cpu_set_id),
            verified=Bool(binding.verified),
            released=Bool(team.bindings_released[index]),
        )
        for (index, binding) in enumerate(team.bindings)
    ]
    payload = (;
        format="serial-workspace-snn-team-teardown",
        version=1,
        recorded_at=string(now()),
        process_id=getpid(),
        update=Int(update),
        dataset_content_sha256=config.dataset_content_sha256,
        runtime_provenance=config.runtime_provenance,
        julia_workers=executor.julia_workers,
        active_workers=executor.active_workers,
        cpuset_mode=executor.cpuset_mode,
        booted_workers=executor.booted_workers[],
        ready_workers=executor.ready_workers[],
        shutdown_requested=executor.shutdown_requested[],
        queue_closed=true,
        queue_length=0,
        remaining=executor.remaining[],
        bindings,
        all_bindings_verified=true,
        all_bindings_released=true,
    )
    path = joinpath(run_dir, "team_teardown.json")
    write_json(path, payload)
    return file_artifact(path, "team_teardown", update)
end

function load_parent_training_trace(
    checkpoint_reference,
    config,
    checkpoint_payload,
)
    String(checkpoint_reference.kind) == "training" || error(
        "finalize-only requires a training checkpoint",
    )
    checkpoint_path = canonical_existing_file(
        String(checkpoint_reference.path),
        "finalize-only parent checkpoint",
    )
    parent_run_dir = canonical_existing_directory(
        dirname(dirname(checkpoint_path)),
        "finalize-only parent run directory",
    )
    trace_path = canonical_existing_file(
        joinpath(parent_run_dir, "training_trace.tsv"),
        "finalize-only parent training trace",
    )
    normalized_path_identity(dirname(trace_path)) ==
        normalized_path_identity(parent_run_dir) || error(
        "parent training trace is not an immediate run artifact",
    )
    trace_bytes = read(trace_path)
    artifact = file_artifact(
        trace_path,
        "training_trace",
        checkpoint_reference.update,
    )
    artifact.bytes == length(trace_bytes) || error(
        "parent training trace changed while reading",
    )
    artifact.sha256 == bytes2hex(sha256(trace_bytes)) || error(
        "parent training trace SHA-256 changed while reading",
    )
    text = String(trace_bytes)
    isempty(text) && error("parent training trace is empty")
    lines = split(chomp(text), '\n'; keepempty=true)
    length(lines) >= 2 || error(
        "parent training trace has no data rows",
    )
    header = split(chomp(lines[1]), '\t'; keepempty=true)
    (
        length(header) == length(REQUIRED_TRACE_COLUMNS) &&
        all(
            header[index] == String(REQUIRED_TRACE_COLUMNS[index])
            for index in eachindex(REQUIRED_TRACE_COLUMNS)
        )
    ) || error(
        "parent training trace header differs",
    )
    indices = Dict(
        name => index
        for (index, name) in enumerate(REQUIRED_TRACE_COLUMNS)
    )
    hasproperty(checkpoint_payload, :segment_state) || error(
        "finalize-only parent checkpoint is missing segment state",
    )
    segment_start_update =
        checkpoint_payload.segment_state.start_update
    segment_start_update isa Int || error(
        "finalize-only parent segment start update must be Int",
    )
    0 <= segment_start_update < Int(checkpoint_reference.update) || error(
        "finalize-only parent segment start update differs",
    )
    previous_update = Int128(segment_start_update)
    previous_structural_flips = Int128(0)
    for row_index in 2:length(lines)
        line = chomp(lines[row_index])
        isempty(line) && error(
            "parent training trace has a blank row at $row_index",
        )
        values = split(line, '\t'; keepempty=true)
        length(values) == length(REQUIRED_TRACE_COLUMNS) || error(
            "parent training trace column count differs at row $row_index",
        )
        integer_values = Dict{Symbol,Int128}()
        for name in TRACE_INTEGER_PROPERTIES
            parsed = tryparse(Int128, values[indices[name]])
            parsed === nothing && error(
                "parent training trace integer differs at row $row_index: " *
                String(name),
            )
            integer_values[name] = parsed
        end
        integer_values[:trace_schema_version] == 3 || error(
            "parent training trace schema version differs",
        )
        integer_values[:training_dynamics_schema_version] == 4 || error(
            "parent training dynamics trace schema version differs",
        )
        integer_values[:component_loss_alias_schema_version] ==
            COMPONENT_LOSS_ALIAS_CONTRACT.schema_version || error(
            "parent training trace alias schema version differs",
        )
        update = integer_values[:update]
        update > previous_update || error(
            "parent training trace updates are not strictly increasing",
        )
        (
            update == 1 ||
            update % Int(config.log_interval) == 0 ||
            update == Int(checkpoint_reference.update)
        ) || error(
            "parent training trace update violates logging cadence",
        )
        integer_values[:teacher_states] ==
            update * Int(config.state_batch) || error(
            "parent training trace teacher-state count differs",
        )
        integer_values[:window_updates] ==
            update - previous_update || error(
            "parent training trace window update count differs",
        )
        integer_values[:enabled_synapses] >= 0 || error(
            "parent training trace enabled-synapse count is negative",
        )
        structural_flips = integer_values[:structural_flips_total]
        structural_flips >= previous_structural_flips || error(
            "parent training trace structural flips decrease",
        )
        integer_values[:net_mask_flips] >= 0 || error(
            "parent training trace net mask flip count is negative",
        )
        integer_values[:hot_allocation_bytes] >= 0 || error(
            "parent training trace hot allocation is negative",
        )
        for name in TRACE_BOOL_PROPERTIES
            values[indices[name]] in ("true", "false") || error(
                "parent training trace Bool differs at row $row_index: " *
                String(name),
            )
        end
        values[indices[:q_huber_loss_alias_of]] ==
            COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.alias_of || error(
            "parent training trace q-Huber alias differs",
        )
        values[indices[:raw_top_gap_loss_alias_of]] ==
            COMPONENT_LOSS_ALIAS_CONTRACT.raw_top_gap_loss.alias_of || error(
            "parent training trace top-gap alias differs",
        )
        values[indices[:component_loss_alias_identity]] ==
            COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.identity || error(
            "parent training trace alias identity differs",
        )
        for name in REQUIRED_TRACE_COLUMNS
            name in TRACE_INTEGER_PROPERTIES && continue
            name in TRACE_BOOL_PROPERTIES && continue
            name in TRACE_STRING_PROPERTIES && continue
            parsed = tryparse(Float64, values[indices[name]])
            parsed === nothing && error(
                "parent training trace numeric field differs at row " *
                "$row_index: $(String(name))",
            )
            isfinite(parsed) || error(
                "parent training trace numeric field is non-finite at row " *
                "$row_index: $(String(name))",
            )
            if name in (
                REQUIRED_COMPONENT_LOSS_PROPERTIES...,
                COMPONENT_LOSS_WINDOW_PROPERTIES...,
            )
                parsed >= 0.0 || error(
                    "parent training trace component loss is negative at " *
                    "row $row_index: $(String(name))",
                )
            end
        end
        for (alias_name, canonical_name) in (
            (:q_huber_loss, :old_q_loss),
            (:raw_top_gap_loss, :margin_loss),
            (:window_q_huber_loss_sum, :window_old_q_loss_sum),
            (:window_raw_top_gap_loss_sum, :window_margin_loss_sum),
        )
            values[indices[alias_name]] ==
                values[indices[canonical_name]] || error(
                "parent training trace component alias differs at row " *
                "$row_index: $(String(alias_name))",
            )
        end
        previous_update = update
        previous_structural_flips = structural_flips
    end
    previous_update == Int(checkpoint_reference.update) || error(
        "parent training trace final update differs from checkpoint",
    )
    return artifact
end

function load_team_teardown(
    checkpoint_reference,
    config,
)
    String(checkpoint_reference.kind) == "training" || error(
        "finalize-only requires a training checkpoint, not " *
        "$(checkpoint_reference.kind)",
    )
    checkpoint_path = canonical_existing_file(
        String(checkpoint_reference.path),
        "finalize-only parent checkpoint",
    )
    run_dir = canonical_existing_directory(
        dirname(dirname(checkpoint_path)),
        "finalize-only parent run directory",
    )
    path = canonical_existing_file(
        joinpath(run_dir, "team_teardown.json"),
        "finalize-only parent team teardown",
    )
    normalized_path_identity(dirname(path)) ==
        normalized_path_identity(run_dir) || error(
        "parent team teardown is not an immediate run artifact",
    )
    artifact = file_artifact(
        path,
        "team_teardown",
        checkpoint_reference.update,
    )
    payload = JSON3.read(String(read_bound_file_artifact(
        artifact,
        "finalize-only parent team teardown",
    )))
    payload isa JSON3.Object || error(
        "parent team teardown must be a JSON object",
    )
    String(payload.format) ==
        "serial-workspace-snn-team-teardown" ||
        error("parent team teardown format differs")
    Int(payload.version) == 1 ||
        error("parent team teardown version differs")
    Int(payload.update) == Int(checkpoint_reference.update) ||
        error("parent team teardown update differs")
    String(payload.dataset_content_sha256) ==
        String(config.dataset_content_sha256) ||
        error("parent team teardown dataset binding differs")
    production_contract_sha256(payload.runtime_provenance) ==
        production_contract_sha256(config.runtime_provenance) || error(
        "parent team teardown runtime provenance differs",
    )
    Bool(payload.queue_closed) &&
        Int(payload.queue_length) == 0 &&
        Int(payload.remaining) == 0 &&
        Bool(payload.all_bindings_verified) &&
        Bool(payload.all_bindings_released) || error(
        "parent team teardown invariants are incomplete",
    )
    length(payload.bindings) == Int(payload.julia_workers) || error(
        "parent team teardown binding count differs",
    )
    all(
        binding ->
            Bool(binding.verified) && Bool(binding.released),
        payload.bindings,
    ) || error("parent team teardown binding evidence is incomplete")
    return artifact
end

function binding_summary_from_teardown(teardown_artifact)
    payload = JSON3.read(String(read_bound_file_artifact(
        teardown_artifact,
        "bound parent team teardown",
    )))
    active_workers = Int(payload.active_workers)
    return (;
        verified=
            Bool(payload.all_bindings_verified) &&
            Bool(payload.all_bindings_released),
        released=Bool(payload.all_bindings_released),
        cpu_set_ids=[
            payload.bindings[index].cpu_set_id === nothing ?
                nothing : Int(payload.bindings[index].cpu_set_id)
            for index in 1:active_workers
        ],
    )
end

function with_gc_boundary(body::F) where {F}
    previous_gc_state = GC.enable(true)
    try
        return body()
    finally
        GC.gc()
        GC.enable(previous_gc_state)
    end
end

function full_local_eprop_config()
    return EPropShadowConfig(;
        feedback_mode=:block_local,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
        signal_schedule=:all_cycles,
        third_factor_mode=:aligned,
        time_order=:forward,
        routing_entropy_weight=0.002f0,
        routing_entropy_floor=0.70f0,
        routing_load_weight=0.002f0,
    )
end

function array_registry_sha256(registry)
    registry_digest_input = UInt8[]
    for name in keys(registry)
        append!(registry_digest_input, codeunits(String(name)))
        push!(registry_digest_input, 0x00)
        array = getproperty(registry, name)
        append!(
            registry_digest_input,
            sha256(reinterpret(UInt8, vec(array))),
        )
    end
    return bytes2hex(sha256(registry_digest_input))
end

function trainer_isolation_witness(trainer)
    optimizer = trainer.optimizer
    mask = structural_mask(trainer.parameters)
    return (;
        parameters_sha256=
            array_registry_sha256(trainer.parameters),
        first_moment_sha256=
            array_registry_sha256(optimizer.first_moment),
        second_moment_sha256=
            array_registry_sha256(optimizer.second_moment),
        utility_sha256=bytes2hex(sha256(reinterpret(
            UInt8,
            vec(trainer.synapse_utility),
        ))),
        structural_mask_sha256=bytes2hex(sha256(reinterpret(
            UInt8,
            vec(mask),
        ))),
        cached_gate_mask_sha256=bytes2hex(sha256(reinterpret(
            UInt8,
            vec(trainer.cache.gate_hard),
        ))),
        optimizer_step=optimizer.step,
        learning_rate=optimizer.learning_rate,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        beta1_power=optimizer.beta1_power,
        beta2_power=optimizer.beta2_power,
        epsilon=optimizer.epsilon,
        weight_decay=optimizer.weight_decay,
        total_structural_flips=trainer.total_structural_flips,
        utility_updates=trainer.utility_updates,
    )
end

function assert_independent_warmup_storage!(
    production_trainer,
    warmup_trainer,
)
    production_trainer === warmup_trainer &&
        error("warmup trainer aliases production trainer")
    production_trainer.arena === warmup_trainer.arena &&
        error("warmup arena aliases production arena")
    production_trainer.synapse_utility ===
        warmup_trainer.synapse_utility &&
        error("warmup utility aliases production utility")
    for name in keys(production_trainer.parameters)
        getproperty(production_trainer.parameters, name) ===
            getproperty(warmup_trainer.parameters, name) &&
            error("warmup parameter storage aliases $name")
        getproperty(
            production_trainer.optimizer.first_moment,
            name,
        ) === getproperty(
            warmup_trainer.optimizer.first_moment,
            name,
        ) && error("warmup first moment aliases $name")
        getproperty(
            production_trainer.optimizer.second_moment,
            name,
        ) === getproperty(
            warmup_trainer.optimizer.second_moment,
            name,
        ) && error("warmup second moment aliases $name")
    end
    return nothing
end

function run_persistent_team_warmup!(
    executor,
    production_trainer,
    training_rows,
)
    executor.trainer === production_trainer ||
        error("persistent team does not own the production trainer")
    ArenaWorkspaceTraining.Queue.approx_length(executor.queue) == 0 ||
        error("arena queue is not empty before persistent warmup")
    executor.remaining[] == 0 ||
        error("arena has remaining work before persistent warmup")
    executor.failure_worker[] == 0 ||
        error("arena has a recorded failure before persistent warmup")

    before = trainer_isolation_witness(production_trainer)
    warmup_trainer = ArenaTrainer(
        production_trainer.model,
        copy_parameters(production_trainer.parameters);
        state_batch=production_trainer.arena.state_batch,
        width=production_trainer.arena.width,
        learning_rate=production_trainer.optimizer.learning_rate,
        weight_decay=production_trainer.optimizer.weight_decay,
        structure_weight=production_trainer.structure_weight,
    )
    typeof(warmup_trainer) === typeof(production_trainer) ||
        error("warmup trainer type differs from production trainer")
    assert_independent_warmup_storage!(
        production_trainer,
        warmup_trainer,
    )
    warmup_trainer.arena.rows .= @view training_rows[
        1:warmup_trainer.arena.state_batch
    ]

    try
        executor.trainer = warmup_trainer
        arena_update!(executor; structural_interval=1)
    finally
        executor.trainer = production_trainer
    end

    executor.trainer === production_trainer ||
        error("production trainer was not restored after warmup")
    ArenaWorkspaceTraining.Queue.approx_length(executor.queue) == 0 ||
        error("arena queue is not empty after persistent warmup")
    executor.remaining[] == 0 ||
        error("arena has remaining work after persistent warmup")
    executor.failure_worker[] == 0 ||
        error("arena recorded a failure during persistent warmup")
    !ArenaWorkspaceTraining.Queue.isclosed(executor.queue) ||
        error("arena queue closed during persistent warmup")
    executor.shutdown_requested[] == 0 ||
        error("arena shutdown was requested during persistent warmup")

    after = trainer_isolation_witness(production_trainer)
    before == after ||
        error("persistent warmup contaminated production trainer state")
    return (;
        isolation_verified=true,
        warmup_optimizer_step=warmup_trainer.optimizer.step,
        warmup_loss=warmup_trainer.last_loss.composite_loss,
        queue_length=
            ArenaWorkspaceTraining.Queue.approx_length(executor.queue),
        remaining=executor.remaining[],
        failure_worker=executor.failure_worker[],
    )
end

function main()
    hermetic_runtime_options()
    Threads.nthreads(:interactive) == 0 || error(
        "launch with --threads=N,0",
    )
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 || error(
        "production training could not enforce one BLAS thread",
    )
    maximum_updates = env_int("SWSNN_MAX_UPDATES", 100_000; minimum=1)
    state_batch = env_int("SWSNN_STATE_BATCH", 8; minimum=1)
    active_workers = env_int(
        "SWSNN_ACTIVE_WORKERS",
        Threads.nthreads(:default);
        minimum=2,
    )
    cpuset_mode = Symbol(lowercase(get(
        ENV,
        "SWSNN_CPUSET_MODE",
        active_workers == Threads.nthreads(:default) ? "all" : "none",
    )))
    learning_rate =
        env_float("SWSNN_LEARNING_RATE", 5.0f-4; minimum=0.0)
    weight_decay =
        env_float("SWSNN_WEIGHT_DECAY", 1.0f-5; minimum=0.0)
    structure_weight =
        env_float("SWSNN_STRUCTURE_WEIGHT", 1.0f-2; minimum=0.0)
    utility_decay =
        env_float("SWSNN_UTILITY_DECAY", 0.99f0; minimum=0.0)
    utility_decay < 1.0f0 || error(
        "SWSNN_UTILITY_DECAY must be in [0, 1)",
    )
    utility_connection_cost = env_float(
        "SWSNN_UTILITY_CONNECTION_COST",
        1.0f-6;
        minimum=0.0,
    )
    isfinite(utility_connection_cost) || error(
        "SWSNN_UTILITY_CONNECTION_COST must be finite and nonnegative",
    )
    utility_keep_fraction = env_float(
        "SWSNN_UTILITY_KEEP_FRACTION",
        0.50f0;
        minimum=0.0,
    )
    0.0f0 < utility_keep_fraction < 1.0f0 || error(
        "SWSNN_UTILITY_KEEP_FRACTION must be in (0, 1)",
    )
    utility_turnover_period = env_int(
        "SWSNN_UTILITY_TURNOVER_PERIOD",
        128;
        minimum=1,
    )
    structural_interval =
        env_int("SWSNN_STRUCTURAL_INTERVAL", 25; minimum=1)
    checkpoint_interval =
        env_int("SWSNN_CHECKPOINT_INTERVAL", 10_000; minimum=1)
    log_interval = env_int("SWSNN_LOG_INTERVAL", 1_000; minimum=1)
    evaluation_states =
        env_int("SWSNN_EVAL_STATES", 128; minimum=1)
    maximum_hot_allocation =
        env_int("SWSNN_MAX_HOT_ALLOCATION_BYTES", 4096; minimum=0)
    learning_mode = Symbol(lowercase(required_env(
        "SWSNN_LEARNING_MODE",
    )))
    learning_mode in (:vjp, :local_hybrid) || error(
        "SWSNN_LEARNING_MODE must be vjp or local_hybrid",
    )
    eprop_reducers = min(
        env_int(
            "SWSNN_EPROP_REDUCERS",
            active_workers;
            minimum=1,
        ),
        active_workers,
    )
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))
    raw_output_root = get(
        ENV,
        "SWSNN_OUTPUT",
        joinpath(@__DIR__, "trained"),
    )
    resume_path = strip(get(ENV, "SWSNN_RESUME_CHECKPOINT", ""))
    resume_sha256 = strip(get(ENV, "SWSNN_RESUME_SHA256", ""))
    isempty(resume_path) && !isempty(resume_sha256) && error(
        "SWSNN_RESUME_SHA256 requires SWSNN_RESUME_CHECKPOINT",
    )
    start_mode = Symbol(lowercase(required_env("SWSNN_START_MODE")))
    start_mode in (:scratch, :resume, :finalize_only) || error(
        "SWSNN_START_MODE must be scratch, resume, or finalize_only",
    )
    scratch = start_mode === :scratch
    scratch && !isempty(resume_path) && error(
        "scratch start mode cannot use SWSNN_RESUME_CHECKPOINT",
    )
    !scratch && isempty(resume_path) && error(
        "$start_mode start mode requires SWSNN_RESUME_CHECKPOINT",
    )
    !scratch && isempty(resume_sha256) && error(
        "$start_mode start mode requires SWSNN_RESUME_SHA256",
    )
    if haskey(ENV, "SWSNN_SCRATCH")
        env_bool("SWSNN_SCRATCH", scratch) == scratch || error(
            "SWSNN_SCRATCH contradicts SWSNN_START_MODE",
        )
    end
    model_preset = Symbol(lowercase(required_env(
        "SWSNN_MODEL_PRESET",
    )))
    run_id = validate_run_id(get(
        ENV,
        "SWSNN_RUN_ID",
        "arena_" * string(model_preset) * "_u" *
        string(maximum_updates) * "_" *
        Dates.format(now(), "yyyymmddTHHMMSS"),
    ))
    output_root, run_dir =
        validate_run_destination(raw_output_root, run_id)
    ispath(run_dir) && error("run already exists: $run_dir")
    launch_binding = validate_launch_manifest_binding(
        required_env("SWSNN_LAUNCH_MANIFEST_PATH"),
        required_env("SWSNN_LAUNCH_MANIFEST_SHA256"),
        output_root,
        run_dir,
        run_id,
        start_mode,
        maximum_updates,
    )

    resume_payload, parent_checkpoint = isempty(resume_path) ?
        (nothing, nothing) : load_checkpoint(resume_path, resume_sha256)
    parent_launch_evidence = nothing
    parent_manifest_evidence = nothing
    if resume_payload === nothing
        validate_start_mode_update(:scratch, 0, maximum_updates)
        validate_launch_parent_contract(
            launch_binding,
            start_mode,
            resume_payload,
            parent_checkpoint,
        )
    else
        Symbol(resume_payload.checkpoint_kind) === :training || error(
            "production resume/finalize-only requires a training checkpoint",
        )
        # Start-mode/update validation is intentionally performed immediately
        # after the read-only checkpoint load, before any output is created.
        validate_start_mode_update(
            start_mode,
            Int(resume_payload.update),
            maximum_updates,
        )
        validate_launch_parent_contract(
            launch_binding,
            start_mode,
            resume_payload,
            parent_checkpoint,
        )
        parent_launch_evidence = validate_parent_launch_binding(
            resume_payload,
            parent_checkpoint,
        )
        parent_manifest_evidence =
            validate_parent_checkpoint_manifest(parent_checkpoint)
    end

    dataset_preflight = dataset_binding_preflight(dataset_path)
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(state_batch, evaluation_states),
    )
    dataset_content_sha256, dataset_integrity =
        bind_loaded_dataset(dataset_path, dataset, dataset_preflight)
    width = 16 * cld(maximum(dataset.action_counts), 16)
    width == 80 || error("teacher_v3 candidate width drift: $width")
    training_rows = training_rows_only(dataset)
    panel_rows = fixed_training_panel(training_rows, evaluation_states)
    panel_sha256 = bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    training_rows_sha256 =
        bytes2hex(sha256(reinterpret(UInt8, training_rows)))
    model = build_model(model_preset)
    fresh_parameters, states = Lux.setup(Xoshiro(MODEL_SEED), model)
    fingerprint = source_fingerprint()
    provenance = runtime_provenance(fingerprint)

    production_eprop_config = full_local_eprop_config()

    model_topology = graph_topology(model, fresh_parameters)
    optimizer_config = (;
        learning_rate,
        weight_decay,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
        structure_weight,
    )
    representation_config = (;
        query_role="routing_only",
        head_features="workspace_plus_hard_selected_pool",
        query_normalization="candidate_local_rmsnorm_tanh",
        hidden_normalization="candidate_local_rmsnorm_tanh",
        rms_epsilon=Float32(SerialWorkspaceSNN.RMS_NORM_EPS),
        query_norm_scale=
            Float32(SerialWorkspaceSNN.QUERY_NORM_SCALE),
        hidden_norm_scale=
            Float32(SerialWorkspaceSNN.HIDDEN_NORM_SCALE),
    )
    workspace_retention_config = (;
        parameterization="bounded_sigmoid",
        minimum=Float32(SerialWorkspaceSNN.WORKSPACE_DECAY_MIN),
        maximum=Float32(SerialWorkspaceSNN.WORKSPACE_DECAY_MAX),
    )
    spiking_config = (;
        spike_temperature=model.spike_temperature,
    )
    eprop_config = merge(
        (; enabled=learning_mode === :local_hybrid),
        eprop_config_snapshot(production_eprop_config),
    )
    routing_config = (;
        routing_seed=ROUTING_SEED,
        inference_selection="deterministic_hard_top_k",
        training_selection=
            learning_mode === :local_hybrid ?
            "stochastic_hard_top_k_without_replacement" :
            "deterministic_hard_top_k",
        parameter_update=
            learning_mode === :local_hybrid ?
            "ordered_plackett_luce_score_eligibility_three_factor" :
            "analytic_vjp",
        learning_signal_semantics=
            learning_mode === :local_hybrid ?
            "supervised_reward_surrogate" :
            "analytic_supervised_gradient",
        reward_source=
            learning_mode === :local_hybrid ?
            "candidate_centered_listnet_and_auxiliary_loss_advantage" :
            "not_applicable",
        route_probability_mass=1.0f0,
        route_temperature=model.route_temperature,
        exploration_probability=Float32(
            ArenaWorkspaceTraining.Routing.DEFAULT_EXPLORATION,
        ),
        score_normalization_epsilon=Float32(
            ArenaWorkspaceTraining.Routing.DEFAULT_NORM_EPSILON,
        ),
        entropy_weight=
            learning_mode === :local_hybrid ?
            production_eprop_config.routing_entropy_weight : 0.0f0,
        entropy_floor=production_eprop_config.routing_entropy_floor,
        load_balance_weight=
            learning_mode === :local_hybrid ?
            production_eprop_config.routing_load_weight : 0.0f0,
    )
    structural_learning_config = (;
        mode=
            learning_mode === :local_hybrid ? "utility" : "legacy",
        utility_decay,
        utility_connection_cost,
        utility_keep_fraction,
        utility_turnover_period,
        responsibility=
            learning_mode === :local_hybrid ?
            "normalized_abs_block_signal_times_eligibility" :
            "legacy_weight_magnitude",
    )
    executor_config = (;
        fixed_candidate_arenas=true,
        analytic_vjp=learning_mode === :vjp,
        supervised_head_vjp=true,
        recurrent_credit_assignment=
            learning_mode === :local_hybrid ?
            "eprop_decolle_block_local_three_factor" :
            "analytic_vjp",
        eprop_reducers=
            learning_mode === :local_hybrid ? eprop_reducers : 0,
        worker_local_gradients=true,
        mpmc_isbits_jobs=true,
        queue_capacity=2048,
        parallel_in_place_adamw=true,
        gc_disabled_inside_hot_training=true,
        structural_learning=structural_learning_config,
    )
    production_contract = (;
        version=PRODUCTION_CONTRACT_VERSION,
        experiment_id=:serial_workspace_snn_arena_v3,
        model_preset,
        model=model_topology,
        parameter_count=parameter_count(fresh_parameters),
        maximum_updates,
        state_batch,
        candidate_width=width,
        active_workers,
        eprop_reducers,
        cpuset_mode,
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
        optimizer=optimizer_config,
        learning_mode,
        structural_interval,
        checkpoint_interval,
        log_interval,
        evaluation_states,
        maximum_hot_allocation_bytes=maximum_hot_allocation,
        dataset_path=dataset_integrity.source_path,
        dataset_content_sha256,
        dataset_integrity,
        training_rows=length(training_rows),
        training_rows_sha256,
        training_panel_rows_sha256=panel_sha256,
        model_seed=MODEL_SEED,
        sampler_seed=SAMPLER_SEED,
        representation=representation_config,
        workspace_retention=workspace_retention_config,
        spiking=spiking_config,
        eprop=eprop_config,
        routing=routing_config,
        executor=executor_config,
        runtime_provenance=provenance,
    )
    production_contract_digest =
        production_contract_sha256(production_contract)
    config = (;
        experiment_id=:serial_workspace_snn_arena_v3,
        checkpoint_schema=(;
            format=CHECKPOINT_FORMAT,
            version=CHECKPOINT_VERSION,
        ),
        production_contract,
        production_contract_sha256=production_contract_digest,
        role="third_model_after_preact_and_dsrln",
        run_id,
        launch_binding,
        start_mode,
        scratch,
        explicit_environment_contract=true,
        production_target=(;
            model_preset=:scaled_v2,
            start_mode=:scratch,
            learning_mode=:local_hybrid,
            maximum_updates=100_000,
        ),
        production_target_match=
            model_preset === :scaled_v2 &&
            start_mode === :scratch &&
            learning_mode === :local_hybrid &&
            maximum_updates == 100_000,
        vjp_explicitly_requested=learning_mode === :vjp,
        model_preset,
        model=model_topology,
        parameter_count=parameter_count(fresh_parameters),
        maximum_updates,
        state_batch,
        target_teacher_states=maximum_updates * state_batch,
        active_workers,
        cpuset_mode,
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
        learning_rate,
        weight_decay,
        structure_weight,
        optimizer=optimizer_config,
        learning_mode,
        eprop_reducers,
        structural_interval,
        checkpoint_interval,
        log_interval,
        evaluation_states,
        maximum_hot_allocation_bytes=maximum_hot_allocation,
        dataset_path=dataset_integrity.source_path,
        dataset_content_sha256,
        dataset_integrity,
        candidate_width=width,
        training_rows=length(training_rows),
        training_rows_sha256,
        training_eval_states=length(panel_rows),
        training_panel_rows_sha256=panel_sha256,
        validation_rows_used=false,
        game_validation_used=false,
        sealed_seeds_used=false,
        model_seed=MODEL_SEED,
        sampler_seed=SAMPLER_SEED,
        routing_seed=ROUTING_SEED,
        source_fingerprint=fingerprint,
        runtime_provenance=provenance,
        persistent_single_team_warmup=true,
        representation=representation_config,
        workspace_retention=workspace_retention_config,
        spiking=spiking_config,
        eprop=eprop_config,
        routing=routing_config,
        executor=executor_config,
    )
    parent_team_teardown_evidence = nothing
    parent_training_trace_evidence = nothing
    if start_mode === :finalize_only
        parent_team_teardown_evidence =
            load_team_teardown(parent_checkpoint, config)
        parent_training_trace_evidence =
            load_parent_training_trace(
                parent_checkpoint,
                config,
                resume_payload,
            )
    end
    if resume_payload !== nothing
        validate_resume_contract(resume_payload.config, config)
        String(resume_payload.dataset_content_sha256) ==
            dataset_content_sha256 ||
            error("resume dataset content binding differs")
        resume_payload.dataset_integrity == dataset_integrity ||
            error("resume dataset integrity differs")
        resume_payload.runtime_provenance == provenance ||
            error("resume runtime provenance differs")
        validate_parent_launch_binding(
            resume_payload,
            parent_checkpoint,
        ) == parent_launch_evidence || error(
            "parent launch evidence changed during resume validation",
        )
        # This is the final read-only gate before the first output mutation.
        # Missing or corrupt parent evidence is never repaired by a child run.
        validate_parent_checkpoint_manifest(parent_checkpoint) ==
            parent_manifest_evidence || error(
            "parent checkpoint manifest evidence changed during resume " *
            "validation",
        )
    end
    validate_launch_parent_contract(
        launch_binding,
        start_mode,
        resume_payload,
        parent_checkpoint,
    )
    if start_mode === :finalize_only
        verify_bound_file_artifact(
            parent_team_teardown_evidence,
            "finalize-only parent team teardown before reservation",
        )
        verify_bound_file_artifact(
            parent_training_trace_evidence,
            "finalize-only parent training trace before reservation",
        )
    end
    ispath(run_dir) && error("run already exists: $run_dir")
    run_dir = reserve_run_directory!(
        output_root,
        run_dir,
        run_id,
    )
    write_json(
        joinpath(run_dir, "config.json"),
        (; config, parent_checkpoint),
    )
    write_stage_status(
        run_dir,
        :configuration_written;
        update=resume_payload === nothing ? 0 : Int(resume_payload.update),
        details=(; start_mode=String(start_mode)),
    )

    initial_parameters = resume_payload === nothing ?
        copy_parameters(fresh_parameters) :
        copy_parameters(resume_payload.initial_parameters)
    trainer = ArenaTrainer(
        model,
        fresh_parameters;
        state_batch,
        width,
        learning_rate,
        weight_decay,
        structure_weight,
    )
    validate_optimizer_contract(trainer.optimizer, optimizer_config)
    sampler = resume_payload === nothing ?
        EpochSampler(training_rows, Xoshiro(SAMPLER_SEED)) :
        restore_sampler(training_rows, resume_payload.sampler_state)
    progress = resume_payload === nothing ?
        ProgressTotals() : restore_progress(resume_payload.progress)
    resume_payload === nothing ||
        restore_trainer!(trainer, resume_payload, optimizer_config)

    eval_batch = allocate_host_batch(1; max_candidates=width)
    initial_metrics = resume_payload === nothing ?
        evaluate(
            model,
            trainer.parameters,
            states,
            dataset,
            panel_rows,
            eval_batch,
        ) : resume_payload.initial_metrics
    write_stage_status(
        run_dir,
        :initial_evaluation_complete;
        update=trainer.optimizer.step,
        details=(;
            restored_from_checkpoint=resume_payload !== nothing,
            composite_loss=initial_metrics.composite_loss,
        ),
    )
    trace_path = joinpath(run_dir, "training_trace.tsv")
    last_checkpoint = parent_checkpoint
    last_checkpoint_segment = Ref{Any}(
        resume_payload === nothing ?
        (;
            start_update=trainer.optimizer.step,
            updates=0,
            overall_seconds=0.0,
        ) :
        resume_payload.segment_state,
    )
    last_training_dynamics = Ref{Any}(
        resume_payload === nothing ?
        training_dynamics(trainer) :
        resume_payload.last_training_dynamics,
    )
    warmup_summary = Ref{Any}(
        resume_payload === nothing ?
        nothing : resume_payload.persistent_team_warmup,
    )
    if start_mode === :scratch
        last_checkpoint = with_gc_boundary() do
            checkpoint_path = joinpath(
                run_dir,
                "checkpoints",
                "checkpoint_000000000.jld2",
            )
            artifact = save_checkpoint!(
                checkpoint_path,
                trainer,
                sampler,
                initial_parameters,
                config,
                initial_metrics,
                progress,
                parent_checkpoint,
                warmup_summary[],
                last_checkpoint_segment[],
                last_training_dynamics[],
            )
            append_checkpoint_manifest(run_dir, artifact)
            artifact
        end
    end
    write_stage_status(
        run_dir,
        :checkpoint_zero_complete;
        update=trainer.optimizer.step,
        details=(;
            checkpoint_created=start_mode === :scratch,
            checkpoint=last_checkpoint,
        ),
    )
    executor = nothing
    team = nothing
    team_teardown_artifact = nothing
    if start_mode === :finalize_only
        team_teardown_artifact = verify_bound_file_artifact(
            parent_team_teardown_evidence,
            "finalize-only parent team teardown after reservation",
        )
        verify_bound_file_artifact(
            parent_training_trace_evidence,
            "finalize-only parent training trace after reservation",
        )
        write_stage_status(
            run_dir,
            :finalize_only_restore_complete;
            update=trainer.optimizer.step,
            details=(;
                optimizer_steps_executed=0,
                parent_checkpoint,
                team_teardown=team_teardown_artifact,
            ),
        )
    else
        executor = ArenaExecutor(
            trainer,
            dataset;
            active_workers,
            cpuset_mode,
            queue_capacity=executor_config.queue_capacity,
            eprop_shadow_config=
                learning_mode === :local_hybrid ?
                production_eprop_config : nothing,
            eprop_reducer_count=
                learning_mode === :local_hybrid ?
                eprop_reducers : active_workers,
            synapse_learning_mode=
                learning_mode === :local_hybrid ?
                :local_eligibility : :vjp,
            stochastic_routing=learning_mode === :local_hybrid,
            routing_seed=ROUTING_SEED,
            structural_learning_mode=
                learning_mode === :local_hybrid ? :utility : :legacy,
            utility_decay,
            utility_connection_cost,
            utility_keep_fraction,
            utility_turnover_period,
        )
        overall_started = Ref{Float64}(0.0)
        segment_start_update = Ref{Int}(trainer.optimizer.step)
        team = run_with_arena_team!(executor) do running
            team_initial_gc_state = GC.enable(true)
            try
                write_stage_status(
                    run_dir,
                    :persistent_team_warmup_start;
                    update=trainer.optimizer.step,
                )
                warmup = run_persistent_team_warmup!(
                    running,
                    trainer,
                    training_rows,
                )
                warmup_summary[] = warmup
                GC.gc()
                write_stage_status(
                    run_dir,
                    :persistent_team_warmup_complete;
                    update=trainer.optimizer.step,
                    details=warmup,
                )
                write_stage_status(
                    run_dir,
                    :training;
                    update=trainer.optimizer.step,
                )
                # Stage serialization and warmup compilation are deliberately
                # outside the measured hot segment. This is the last
                # collection before the allocation-free update loop.
                GC.gc()
                overall_started[] = time()
                segment_start_update[] = trainer.optimizer.step
                GC.enable(false)
                while trainer.optimizer.step < maximum_updates
                    fill_next_rows!(trainer.arena.rows, sampler)
                    arena_update!(
                        running;
                        structural_interval,
                    )
                    accumulate!(progress, trainer)
                    trainer.metrics.gc_seconds == 0.0 || error(
                        "GC entered the arena hot update at " *
                        "$(trainer.optimizer.step)",
                    )
                    trainer.metrics.allocation_bytes <=
                        maximum_hot_allocation || error(
                        "hot allocation " *
                        "$(trainer.metrics.allocation_bytes) exceeds " *
                        "$maximum_hot_allocation bytes at update " *
                        "$(trainer.optimizer.step)",
                    )

                    update = trainer.optimizer.step
                    if update == 1 ||
                       update % log_interval == 0 ||
                       update == maximum_updates
                        with_gc_boundary() do
                            dynamics = training_dynamics(trainer)
                            validate_training_dynamics_snapshot(dynamics)
                            last_training_dynamics[] = dynamics
                            window_denominator =
                                max(progress.window_updates, 1)
                            component_losses =
                                progress.component_losses
                            validate_component_loss_snapshot(
                                component_loss_snapshot(component_losses),
                            )
                            record = (;
                                trace_schema_version=3,
                                update,
                                teacher_states=progress.teacher_states,
                                loss=trainer.last_loss.composite_loss,
                                listnet_ce=
                                    trainer.last_loss.listnet_loss,
                                teacher_entropy=
                                    trainer.last_loss.teacher_entropy,
                                listnet_kl=
                                    trainer.last_loss.listnet_kl,
                                composite_excess=
                                    trainer.last_loss.composite_loss -
                                    trainer.last_loss.teacher_entropy,
                                window_updates=progress.window_updates,
                                window_loss=
                                    progress.window_composite_loss /
                                    window_denominator,
                                window_listnet_ce=
                                    progress.window_listnet_ce /
                                    window_denominator,
                                window_teacher_entropy=
                                    progress.window_teacher_entropy /
                                    window_denominator,
                                window_listnet_kl=
                                    progress.window_listnet_kl /
                                    window_denominator,
                                window_composite_excess=
                                    progress.window_composite_excess /
                                    window_denominator,
                                component_loss_alias_schema_version=
                                    COMPONENT_LOSS_ALIAS_CONTRACT.schema_version,
                                q_huber_loss_alias_of=
                                    COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.alias_of,
                                raw_top_gap_loss_alias_of=
                                    COMPONENT_LOSS_ALIAS_CONTRACT.raw_top_gap_loss.alias_of,
                                component_loss_alias_identity=
                                    COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.identity,
                                old_q_loss=component_losses.old_q_loss,
                                q_huber_loss=
                                    component_losses.q_huber_loss,
                                margin_loss=component_losses.margin_loss,
                                raw_top_gap_loss=
                                    component_losses.raw_top_gap_loss,
                                death_loss=component_losses.death_loss,
                                quantile_teacher_loss=
                                    component_losses.quantile_teacher_loss,
                                geometry_loss=
                                    component_losses.geometry_loss,
                                line_clear_loss=
                                    component_losses.line_clear_loss,
                                max_height_loss=
                                    component_losses.max_height_loss,
                                holes_loss=component_losses.holes_loss,
                                cavities_loss=
                                    component_losses.cavities_loss,
                                structure_loss=
                                    component_losses.structure_loss,
                                window_old_q_loss_sum=
                                    component_losses.active_window_old_q_loss_sum,
                                window_q_huber_loss_sum=
                                    component_losses.active_window_q_huber_loss_sum,
                                window_margin_loss_sum=
                                    component_losses.active_window_margin_loss_sum,
                                window_raw_top_gap_loss_sum=
                                    component_losses.active_window_raw_top_gap_loss_sum,
                                window_death_loss_sum=
                                    component_losses.active_window_death_loss_sum,
                                window_quantile_teacher_loss_sum=
                                    component_losses.active_window_quantile_teacher_loss_sum,
                                window_geometry_loss_sum=
                                    component_losses.active_window_geometry_loss_sum,
                                window_line_clear_loss_sum=
                                    component_losses.active_window_line_clear_loss_sum,
                                window_max_height_loss_sum=
                                    component_losses.active_window_max_height_loss_sum,
                                window_holes_loss_sum=
                                    component_losses.active_window_holes_loss_sum,
                                window_cavities_loss_sum=
                                    component_losses.active_window_cavities_loss_sum,
                                window_structure_loss_sum=
                                    component_losses.active_window_structure_loss_sum,
                                gradient_norm=trainer.last_gradient_norm,
                                enabled_synapses=enabled_synapse_count(
                                    trainer.parameters,
                                ),
                                structural_flips_total=
                                    trainer.total_structural_flips,
                                training_dynamics_schema_version=
                                    dynamics.schema_version,
                                firing_rate=dynamics.firing_rate,
                                workspace_route_entropy=
                                    dynamics.workspace_route_entropy,
                                workspace_exploitation_entropy=
                                    dynamics.workspace_exploitation_entropy,
                                hard_route_load_entropy=
                                    dynamics.hard_route_load_entropy,
                                hard_route_effective_blocks=
                                    dynamics.hard_route_effective_blocks,
                                hard_route_top8_share=
                                    dynamics.hard_route_top8_share,
                                route_probability_mass_error=
                                    dynamics.route_probability_mass_error,
                                route_probability_max_mass_error=
                                    dynamics.route_probability_max_mass_error,
                                workspace_rms=dynamics.workspace_rms,
                                gate_density=dynamics.gate_density,
                                utility_mean=dynamics.utility_mean,
                                utility_nonzero_fraction=
                                    dynamics.utility_nonzero_fraction,
                                head_pre_rms=dynamics.head_pre_rms,
                                hidden_inv_rms_mean=
                                    dynamics.hidden_inv_rms_mean,
                                hidden_inv_rms_min=
                                    dynamics.hidden_inv_rms_min,
                                hidden_inv_rms_max=
                                    dynamics.hidden_inv_rms_max,
                                hidden_tanh_derivative_mean=
                                    dynamics.hidden_tanh_derivative_mean,
                                route_selection_gap=
                                    dynamics.route_selection_gap,
                                route_score_rms=
                                    dynamics.route_score_rms,
                                hard_mask_unique_fraction=
                                    dynamics.hard_mask_unique_fraction,
                                hard_mask_cycle_churn=
                                    dynamics.hard_mask_cycle_churn,
                                entropy_floor_violation_fraction=
                                    dynamics.entropy_floor_violation_fraction,
                                utility_swap_gap=
                                    dynamics.utility_swap_gap,
                                consolidation_scheduled=
                                    dynamics.consolidation_scheduled,
                                consolidation_actual=
                                    dynamics.consolidation_actual,
                                net_mask_flips=
                                    dynamics.net_mask_flips,
                                gate_probability_mean=
                                    dynamics.gate_probability_mean,
                                gate_derivative_mean=
                                    dynamics.gate_derivative_mean,
                                delay_mean=dynamics.delay_mean,
                                delay_derivative_mean=
                                    dynamics.delay_derivative_mean,
                                leak_mean=dynamics.leak_mean,
                                leak_derivative_mean=
                                    dynamics.leak_derivative_mean,
                                threshold_mean=
                                    dynamics.threshold_mean,
                                threshold_derivative_mean=
                                    dynamics.threshold_derivative_mean,
                                workspace_decay=
                                    dynamics.workspace_decay,
                                workspace_decay_derivative=
                                    dynamics.workspace_decay_derivative,
                                membrane_threshold_margin_mean=
                                    dynamics.membrane_threshold_margin_mean,
                                membrane_threshold_margin_rms=
                                    dynamics.membrane_threshold_margin_rms,
                                surrogate_sensitivity_mean=
                                    dynamics.surrogate_sensitivity_mean,
                                surrogate_sensitivity_rms=
                                    dynamics.surrogate_sensitivity_rms,
                                eligibility_rms=
                                    dynamics.eligibility_rms,
                                local_q_loss=
                                    dynamics.local_q_loss,
                                local_death_loss=
                                    dynamics.local_death_loss,
                                local_quantile_loss=
                                    dynamics.local_quantile_loss,
                                local_geometry_loss=
                                    dynamics.local_geometry_loss,
                                states_per_second=
                                    progress.teacher_states /
                                    progress.hot_wall_seconds,
                                cpu_percent=
                                    100.0 * progress.hot_cpu_seconds /
                                    (
                                        progress.hot_wall_seconds *
                                        Threads.nthreads(:default)
                                    ),
                                hot_allocation_bytes=
                                    progress.hot_allocation_bytes,
                                hot_gc_seconds=progress.hot_gc_seconds,
                                shadow_seconds=progress.shadow_seconds,
                            )
                            append_trace(trace_path, record)
                            @info "Arena workspace SNN training" record...
                            reset_loss_window!(progress)
                        end
                    end
                    if update % checkpoint_interval == 0 ||
                       update == maximum_updates
                        last_checkpoint = with_gc_boundary() do
                            dynamics = training_dynamics(trainer)
                            last_training_dynamics[] = dynamics
                            segment_state = (;
                                start_update=segment_start_update[],
                                updates=
                                    trainer.optimizer.step -
                                    segment_start_update[],
                                overall_seconds=
                                    time() - overall_started[],
                            )
                            last_checkpoint_segment[] = segment_state
                            checkpoint_path = joinpath(
                                run_dir,
                                "checkpoints",
                                "checkpoint_" *
                                lpad(string(update), 9, '0') *
                                ".jld2",
                            )
                            artifact = save_checkpoint!(
                                checkpoint_path,
                                trainer,
                                sampler,
                                initial_parameters,
                                config,
                                initial_metrics,
                                progress,
                                parent_checkpoint,
                                warmup_summary[],
                                segment_state,
                                dynamics,
                            )
                            append_checkpoint_manifest(run_dir, artifact)
                            @info "Arena workspace SNN checkpoint" artifact
                            artifact
                        end
                    end
                end
            finally
                GC.enable(team_initial_gc_state)
            end
            return nothing
        end
        team.result === nothing ||
            error("unexpected arena team result")
        team_teardown_artifact = record_team_teardown!(
            run_dir,
            executor,
            team,
            config,
            trainer.optimizer.step,
        )
    end
    last_checkpoint === nothing && error(
        "finalization requires a target training checkpoint",
    )
    trainer.optimizer.step == maximum_updates || error(
        "training stopped before the requested target",
    )
    overall_seconds =
        Float64(last_checkpoint_segment[].overall_seconds)
    segment_updates = Int(last_checkpoint_segment[].updates)

    write_stage_status(
        run_dir,
        :final_evaluation;
        update=trainer.optimizer.step,
        details=(;
            optimizer_steps_this_process=
                start_mode === :finalize_only ? 0 : segment_updates,
        ),
    )
    final_metrics = evaluate(
        model,
        trainer.parameters,
        states,
        dataset,
        panel_rows,
        eval_batch,
    )
    weight_delta = sqrt(sum(
        abs2,
        Float64.(
            trainer.parameters.synapse_weight .-
            initial_parameters.synapse_weight
        ),
    ))
    delay_delta = sqrt(sum(
        abs2,
        Float64.(
            sigmoid.(trainer.parameters.delay_logits) .-
            sigmoid.(initial_parameters.delay_logits)
        ),
    ))
    gate_flips = count(
        structural_mask(trainer.parameters) .!=
        structural_mask(initial_parameters),
    )
    initial_objective =
        reducible_objective_metrics(initial_metrics)
    final_objective =
        reducible_objective_metrics(final_metrics)
    final_training_dynamics = last_training_dynamics[]
    trace_artifact = if start_mode === :finalize_only
        verify_bound_file_artifact(
            parent_training_trace_evidence,
            "finalize-only parent training trace before finalization",
        )
    else
        isfile(trace_path) || error(
            "training completed without training_trace.tsv",
        )
        file_artifact(
            trace_path,
            "training_trace",
            trainer.optimizer.step,
        )
    end
    bindings = binding_summary_from_teardown(
        team_teardown_artifact,
    )
    finalization_checkpoint_path = joinpath(
        run_dir,
        "checkpoints",
        "finalization_checkpoint_" *
        lpad(string(trainer.optimizer.step), 9, '0') *
        ".jld2",
    )
    finalization_manifest_path =
        joinpath(run_dir, "finalization_manifest.json")
    results_path = joinpath(run_dir, "results.json")
    finalization_record = (;
        status="finalization_checkpoint_complete",
        finalized_at=string(now()),
        optimizer_steps_after_target=0,
        expected_results_path=abspath(results_path),
        expected_manifest_path=abspath(finalization_manifest_path),
        team_teardown=team_teardown_artifact,
        training_checkpoint=last_checkpoint,
        final_metrics,
        component_loss_alias_contract=
            COMPONENT_LOSS_ALIAS_CONTRACT,
        completed_component_loss_window_updates=
            progress.completed_component_loss_window_updates,
        component_loss_telemetry=
            component_loss_snapshot(progress.component_losses),
    )
    finalization_checkpoint = save_checkpoint!(
        finalization_checkpoint_path,
        trainer,
        sampler,
        initial_parameters,
        config,
        initial_metrics,
        progress,
        last_checkpoint,
        warmup_summary[],
        last_checkpoint_segment[],
        final_training_dynamics;
        checkpoint_kind=:finalization,
        finalization=finalization_record,
    )
    results = (;
        config,
        component_loss_alias_contract=COMPONENT_LOSS_ALIAS_CONTRACT,
        parent_checkpoint,
        dataset_content_sha256,
        dataset_integrity,
        runtime_provenance=provenance,
        persistent_team_warmup=warmup_summary[],
        team_teardown=team_teardown_artifact,
        training_trace=trace_artifact,
        initial=initial_metrics,
        final=final_metrics,
        final_training_dynamics,
        completed_component_loss_window_updates=
            progress.completed_component_loss_window_updates,
        component_loss_telemetry=
            component_loss_snapshot(progress.component_losses),
        reducible_objective=(;
            initial=initial_objective,
            final=final_objective,
        ),
        deltas=(;
            composite_loss=
                final_metrics.composite_loss -
                initial_metrics.composite_loss,
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
                final_metrics.top1_agreement -
                initial_metrics.top1_agreement,
            ndcg=final_metrics.ndcg - initial_metrics.ndcg,
            pairwise_accuracy=
                final_metrics.pairwise_accuracy -
                initial_metrics.pairwise_accuracy,
        ),
        learning_witness=(;
            final_update=trainer.optimizer.step,
            consumed_teacher_states=progress.teacher_states,
            last_batch_loss=trainer.last_loss.composite_loss,
            last_gradient_norm=trainer.last_gradient_norm,
            synapse_weight_l2_delta=weight_delta,
            continuous_delay_l2_delta=delay_delta,
            structural_consolidation_flips=
                trainer.total_structural_flips,
            final_mask_flips_from_initial=gate_flips,
        ),
        throughput=merge(progress_snapshot(progress), (;
            states_per_second=
                progress.teacher_states /
                max(progress.hot_wall_seconds, eps(Float64)),
            updates_per_second=
                progress.updates /
                max(progress.hot_wall_seconds, eps(Float64)),
            average_cpu_percent=
                100.0 * progress.hot_cpu_seconds /
                max(
                    progress.hot_wall_seconds *
                    Threads.nthreads(:default),
                    eps(Float64),
                ),
            overall_seconds,
            segment_updates,
            segment_updates_per_second=
                segment_updates / max(overall_seconds, eps(Float64)),
            optimizer_steps_this_process=
                start_mode === :finalize_only ? 0 : segment_updates,
            phases=phase_totals(progress),
        )),
        checkpoint=finalization_checkpoint,
        training_checkpoint=last_checkpoint,
        bindings,
        finalization=(;
            mode="finalization_checkpoint_then_results_then_manifest",
            optimizer_steps_after_target=0,
            checkpoint=finalization_checkpoint,
            training_checkpoint=last_checkpoint,
            manifest_path=abspath(finalization_manifest_path),
        ),
    )
    write_json(results_path, results)
    results_artifact = file_artifact(
        results_path,
        "results",
        trainer.optimizer.step,
    )
    write_json(
        finalization_manifest_path,
        (;
            format="serial-workspace-snn-finalization-manifest",
            version=1,
            update=trainer.optimizer.step,
            results=results_artifact,
            team_teardown=team_teardown_artifact,
            training_checkpoint=last_checkpoint,
            finalization_checkpoint,
            optimizer_steps_after_target=0,
            dataset_content_sha256,
            runtime_provenance=provenance,
        ),
    )
    finalization_manifest = file_artifact(
        finalization_manifest_path,
        "finalization_manifest",
        trainer.optimizer.step,
    )
    write_stage_status(
        run_dir,
        :complete;
        update=trainer.optimizer.step,
        details=(;
            results_written=true,
            checkpoint=finalization_checkpoint,
            training_checkpoint=last_checkpoint,
            finalization_checkpoint,
            finalization_manifest,
            optimizer_steps_after_target=0,
        ),
    )
    println(JSON3.write(results))
    return results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
