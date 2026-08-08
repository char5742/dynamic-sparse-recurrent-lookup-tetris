module CanonicalCheckpoint

using Serialization
using SHA
using ..CanonicalDendriticGraph

const Graph = CanonicalDendriticGraph

const CHECKPOINT_MAGIC = UInt64(0x43414e4f4e484447) # "CANONHDG"
const CHECKPOINT_SCHEMA = UInt32(1)
const CHECKPOINT_FORMAT =
    "route-free-ordered-multiscale-dendritic-event-graph-v1"
const FINGERPRINT_ALGORITHM = "sha256-canonical-binary-contract-v1"
const CANONICAL_PARAMETER_GROUPS = (
    :core_cell_raw,
    :semantic_projection_raw,
    :event_raw,
    :output_cell_raw,
    :output_projection_raw,
)
const CANONICAL_MECHANISM_COUNTERS = (
    :decolle_signal_nonzero,
    :subthreshold_updates,
    :nonspiking_updates,
    :homeostasis_events,
    :synaptic_scaling_events,
    :utility_updates,
    :rewires,
)
const _TOPOLOGY_FIELDS = (
    :child_offsets,
    :child_edges,
    :parent_offsets,
    :parent_edges,
    :edge_sources,
    :edge_destinations,
    :edge_kinds,
    :edge_slots,
    :edge_roles,
    :node_classes,
    :node_planes,
    :node_slots,
    :interval_firsts,
    :interval_lasts,
)
const _CANONICAL_NODE_COUNT = 1_458
const _CANONICAL_EDGE_COUNT = 2_216
const _CANONICAL_CORE_COUNT = 1_436
const _CANONICAL_OUTPUT_DIM = 22
const _CANONICAL_CONTINUOUS_OBSERVATION_DIM = 47
const _CANONICAL_PACKET_DIM = 12
const _CANONICAL_CELL_STATE_DIM = 48
const _CANONICAL_CELL_INPUT_DIM = 27
const _CANONICAL_CELL_PARAMETER_DIM = 46
const _CANONICAL_EVENT_DIM = 5
const _CANONICAL_STATIC_EVENT_COUNT = 2_040
const _CANONICAL_DYNAMIC_EVENT_COUNT = 80
const _CANONICAL_EVENT_PARAMETER_COUNT = 2_125
const _GRAPH_CONFIG_FIELDS = (
    :max_candidates,
    :max_event_waves,
    :tape_capacity,
    :event_overflow,
)

const _REGISTRY_KEYS = (:path, :element_type, :dimensions, :length)
const _SNAPSHOT_KEYS = (
    :magic,
    :schema,
    :format,
    :architecture_fingerprint,
    :input_fingerprint,
    :topology_fingerprint,
    :learning_fingerprint,
    :optimizer_fingerprint,
    :parameter_registry,
    :parameter_values,
    :first_moments,
    :second_moments,
    :optimizer_step,
    :counters,
    :state_fingerprint,
)

export CHECKPOINT_FORMAT,
       CHECKPOINT_MAGIC,
       CHECKPOINT_SCHEMA,
       CANONICAL_PARAMETER_GROUPS,
       CANONICAL_MECHANISM_COUNTERS,
       ResumeState,
       architecture_fingerprint,
       canonical_architecture_contract,
       canonical_input_contract,
       canonical_learning_contract,
       input_fingerprint,
       learning_fingerprint,
       load_checkpoint,
       optimizer_fingerprint,
       parameter_registry,
       restore_checkpoint!,
       save_checkpoint,
       topology_fingerprint

"""Mutable training clocks returned after arrays have been restored."""
struct ResumeState{T}
    optimizer_step::Int
    counters::T
end

@inline function _write_u64(io::IO, value::UInt64)
    @inbounds for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

function _write_string(io::IO, value::AbstractString)
    bytes = codeunits(value)
    _write_u64(io, UInt64(length(bytes)))
    write(io, bytes)
    return io
end

@inline function _type_label(type::Type)
    return string(type)
end

function _write_float(io::IO, value::T) where {T<:AbstractFloat}
    isfinite(value) || throw(DomainError(value, "contract float is not finite"))
    if T === Float16
        _write_u64(io, UInt64(reinterpret(UInt16, value)))
    elseif T === Float32
        _write_u64(io, UInt64(reinterpret(UInt32, value)))
    elseif T === Float64
        _write_u64(io, reinterpret(UInt64, value))
    else
        throw(ArgumentError("unsupported floating contract type $T"))
    end
    return io
end

"""Stable structural encoding used by every configuration fingerprint."""
function _write_contract(io::IO, value)
    _write_string(io, _type_label(typeof(value)))
    if value === nothing
        return io
    elseif value isa Bool
        write(io, UInt8(value))
    elseif value isa Enum
        _write_string(io, string(Integer(value)))
    elseif value isa Integer
        _write_string(io, string(value))
    elseif value isa AbstractFloat
        _write_float(io, value)
    elseif value isa AbstractString
        _write_string(io, value)
    elseif value isa Symbol
        _write_string(io, String(value))
    elseif value isa NamedTuple
        _write_u64(io, UInt64(length(value)))
        @inbounds for name in keys(value)
            _write_string(io, String(name))
            _write_contract(io, getproperty(value, name))
        end
    elseif value isa Tuple
        _write_u64(io, UInt64(length(value)))
        @inbounds for item in value
            _write_contract(io, item)
        end
    elseif value isa AbstractArray
        isbitstype(eltype(value)) || throw(ArgumentError(
            "contract array element type $(eltype(value)) is not bits-compatible",
        ))
        _write_u64(io, UInt64(ndims(value)))
        @inbounds for dimension in size(value)
            _write_u64(io, UInt64(dimension))
        end
        @inbounds for item in value
            _write_contract(io, item)
        end
    elseif isstructtype(typeof(value))
        names = fieldnames(typeof(value))
        _write_u64(io, UInt64(length(names)))
        @inbounds for name in names
            _write_string(io, String(name))
            _write_contract(io, getfield(value, name))
        end
    else
        throw(ArgumentError(
            "unsupported unordered contract value $(typeof(value))",
        ))
    end
    return io
end

function _contract_fingerprint(label::AbstractString, value)
    io = IOBuffer()
    _write_string(io, FINGERPRINT_ALGORITHM)
    _write_string(io, label)
    _write_contract(io, value)
    return bytes2hex(SHA.sha256(take!(io)))
end

function canonical_architecture_contract(config)
    fieldnames(typeof(config)) == _GRAPH_CONFIG_FIELDS || throw(ArgumentError(
        "graph config fields are missing, extra, or reordered",
    ))
    config.max_candidates isa Int && config.max_candidates > 0 || throw(
        ArgumentError("graph max_candidates must be a positive Int"),
    )
    maximum_waves = Graph.Events.CANONICAL_MAX_WAVES
    config.max_event_waves isa Int &&
        0 <= config.max_event_waves <= maximum_waves || throw(
            ArgumentError(
                "graph max_event_waves must be in 0:$maximum_waves",
            ),
        )
    required_tape_capacity = _CANONICAL_CORE_COUNT *
        (1 + config.max_event_waves)
    config.tape_capacity isa Int &&
        config.tape_capacity >= required_tape_capacity || throw(ArgumentError(
            "graph tape capacity cannot hold all canonical transitions",
        ))
    config.event_overflow in (:error, :fallback) || throw(ArgumentError(
        "graph event overflow policy is unsupported",
    ))
    return (;
        config,
        node_count=_CANONICAL_NODE_COUNT,
        core_node_count=_CANONICAL_CORE_COUNT,
        output_count=_CANONICAL_OUTPUT_DIM,
        cell_state_dim=_CANONICAL_CELL_STATE_DIM,
        cell_input_dim=_CANONICAL_CELL_INPUT_DIM,
        cell_parameter_dim=_CANONICAL_CELL_PARAMETER_DIM,
        packet_dim=_CANONICAL_PACKET_DIM,
        event_dim=_CANONICAL_EVENT_DIM,
        information_spine=:ordered_binary_full_packet,
        semantic_receiver=:typed_receptor_diagonal,
    )
end

architecture_fingerprint(config) = _contract_fingerprint(
    "architecture",
    canonical_architecture_contract(config),
)

@inline function _module_binding(owner::Module, name::Symbol)
    isdefined(owner, name) || throw(ArgumentError(
        "canonical input module has no $(String(name)) binding",
    ))
    return getfield(owner, name)
end

"""Plain, exhaustive semantic ABI of `CanonicalTetrisInput`."""
function canonical_input_contract(input::Module)
    code(name) = UInt8(_module_binding(input, name))
    rows = _module_binding(input, :BOARD_ROWS)
    columns = _module_binding(input, :BOARD_COLUMNS)
    placement_capacity = _module_binding(input, :PLACEMENT_CAPACITY)
    next_count = _module_binding(input, :NEXT_COUNT)
    rows == 24 && columns == 10 || throw(ArgumentError(
        "canonical input board must be 24 x 10",
    ))
    placement_capacity == 4 && next_count == 5 || throw(ArgumentError(
        "canonical placement/queue dimensions differ",
    ))
    return (;
        board_rows=rows,
        board_columns=columns,
        placement_capacity,
        queue_roles=(:hold, :next1, :next2, :next3, :next4, :next5),
        ren_storage=:Int32_exact_nonnegative,
        board=(empty=code(:EMPTY), occupied=code(:OCCUPIED)),
        placement=(absent=code(:ABSENT), present=code(:PRESENT)),
        pieces=(
            none=code(:NONE), i=code(:PIECE_I), o=code(:PIECE_O),
            t=code(:PIECE_T), s=code(:PIECE_S), z=code(:PIECE_Z),
            j=code(:PIECE_J), l=code(:PIECE_L),
        ),
        truth=(false_value=code(:FALSE_VALUE), true_value=code(:TRUE_VALUE)),
        event=(no_event=code(:NO_EVENT), present=code(:EVENT_PRESENT)),
        site=(
            empty=code(:SITE_EMPTY), occupied=code(:SITE_OCCUPIED),
            placed=code(:SITE_PLACED), outside=code(:OUTSIDE),
        ),
        candidate_path=(
            uninitialized=code(:UNINITIALIZED),
            no_clear_cow=code(:NO_CLEAR_COW),
            clear_slow_path=code(:CLEAR_SLOW_PATH),
        ),
        teacher_fields_absent=true,
    )
end

input_fingerprint(config) = _contract_fingerprint("input", config)

function _validate_topology_contract(config)
    fieldnames(typeof(config)) == _TOPOLOGY_FIELDS || throw(ArgumentError(
        "topology contract fields are missing, extra, or reordered",
    ))
    length(config.child_offsets) == _CANONICAL_NODE_COUNT + 1 || throw(
        DimensionMismatch("topology child offsets have the wrong length"),
    )
    length(config.parent_offsets) == _CANONICAL_NODE_COUNT + 1 || throw(
        DimensionMismatch("topology parent offsets have the wrong length"),
    )
    length(config.child_edges) == _CANONICAL_EDGE_COUNT || throw(
        DimensionMismatch("topology child edge order has the wrong length"),
    )
    length(config.parent_edges) == _CANONICAL_EDGE_COUNT || throw(
        DimensionMismatch("topology parent edge order has the wrong length"),
    )
    for name in (
        :edge_sources, :edge_destinations, :edge_kinds, :edge_slots, :edge_roles,
    )
        length(getproperty(config, name)) == _CANONICAL_EDGE_COUNT || throw(
            DimensionMismatch("topology $name has the wrong length"),
        )
    end
    for name in (
        :node_classes, :node_planes, :node_slots, :interval_firsts,
        :interval_lasts,
    )
        length(getproperty(config, name)) == _CANONICAL_NODE_COUNT || throw(
            DimensionMismatch("topology $name has the wrong length"),
        )
    end
    config.child_offsets[1] == 1 &&
        config.child_offsets[end] == _CANONICAL_EDGE_COUNT + 1 || throw(
            ArgumentError("topology child offsets are not canonical CSR"),
        )
    config.parent_offsets[1] == 1 &&
        config.parent_offsets[end] == _CANONICAL_EDGE_COUNT + 1 || throw(
            ArgumentError("topology parent offsets are not canonical CSR"),
        )
    @inbounds for edge in 1:_CANONICAL_EDGE_COUNT
        source = Int(config.edge_sources[edge])
        destination = Int(config.edge_destinations[edge])
        1 <= source < destination <= _CANONICAL_NODE_COUNT || throw(
            ArgumentError("topology edge $edge violates ordered DAG anatomy"),
        )
    end
    return config
end

function _canonical_event_parameter_order(model::Graph.CanonicalModel)
    _validate_topology_contract(model.topology)
    parameter_count = Graph.event_parameter_count(model)
    parameter_count == _CANONICAL_EVENT_PARAMETER_COUNT || throw(
        DimensionMismatch("canonical event parameter count differs"),
    )
    length(model.parameters.event_raw) == parameter_count || throw(
        DimensionMismatch(
            "model event_raw does not match static, dynamic, and kind order",
        ),
    )
    kind = Vector{UInt8}(undef, parameter_count)
    source = Vector{UInt16}(undef, parameter_count)
    destination = Vector{UInt16}(undef, parameter_count)
    channel = Vector{UInt8}(undef, parameter_count)
    family = Vector{UInt8}(undef, parameter_count)
    slot = Vector{UInt8}(undef, parameter_count)
    branch = Vector{UInt8}(undef, parameter_count)
    lane = Vector{UInt8}(undef, parameter_count)
    @inbounds for index in 1:parameter_count
        descriptor = Graph.event_parameter_descriptor(model, index)
        kind[index] = UInt8(descriptor.kind)
        source[index] = descriptor.source
        destination[index] = descriptor.destination
        channel[index] = descriptor.channel
        family[index] = descriptor.family
        slot[index] = descriptor.slot
        branch[index] = descriptor.branch
        lane[index] = descriptor.lane
    end

    static_kind = UInt8(Graph.STATIC_EVENT_CONTACT)
    dynamic_kind = UInt8(Graph.DYNAMIC_EVENT_CONTACT)
    shared_kind = UInt8(Graph.SHARED_EVENT_KIND_GAIN)
    all(==(static_kind), @view(kind[1:_CANONICAL_STATIC_EVENT_COUNT])) ||
        throw(ArgumentError("static event parameter prefix changed"))
    dynamic_first = _CANONICAL_STATIC_EVENT_COUNT + 1
    dynamic_last = _CANONICAL_STATIC_EVENT_COUNT +
        _CANONICAL_DYNAMIC_EVENT_COUNT
    all(==(dynamic_kind), @view(kind[dynamic_first:dynamic_last])) ||
        throw(ArgumentError("dynamic event parameter segment changed"))
    all(==(shared_kind), @view(kind[(dynamic_last + 1):end])) ||
        throw(ArgumentError("shared event-kind parameter suffix changed"))
    lane[(dynamic_last + 1):end] == UInt8.(1:_CANONICAL_EVENT_DIM) ||
        throw(ArgumentError("shared event-kind lane order changed"))

    # Only this descriptor stream defines optimizer-coordinate semantics.
    return (;
        parameter_count,
        kind,
        source,
        destination,
        channel,
        family,
        slot,
        branch,
        lane,
    )
end

function _canonical_event_parameter_order(value)
    throw(ArgumentError(
        "canonical topology contract requires the live CanonicalModel; " *
        "topology alone cannot identify event parameter semantics",
    ))
end

function _canonical_topology_contract(model::Graph.CanonicalModel)
    return (;
        topology=_validate_topology_contract(model.topology),
        event_parameter_order=_canonical_event_parameter_order(model),
    )
end

function _canonical_topology_contract(value)
    throw(ArgumentError(
        "canonical topology fingerprint requires CanonicalDendriticGraph.CanonicalModel",
    ))
end

topology_fingerprint(config) = _contract_fingerprint(
    "topology",
    _canonical_topology_contract(config),
)

@inline function _local_learning_config(config)
    hasproperty(config, :schedule) && hasproperty(config, :plasticity) &&
        return config
    hasproperty(config, :local) && return getproperty(config, :local)
    hasproperty(config, :local_learning) &&
        return getproperty(config, :local_learning)
    throw(ArgumentError(
        "learning config must be LocalLearningConfig or own one explicitly",
    ))
end

"""Include local/plasticity controls and fixed-map dimensional identity."""
function canonical_learning_contract(config)
    local_config = _local_learning_config(config)
    schedule = local_config.schedule
    plasticity = local_config.plasticity
    for name in (
        :analog_interval, :hard_event_interval, :homeostasis_interval,
        :structure_interval,
    )
        hasproperty(schedule, name) || throw(ArgumentError(
            "learning schedule is missing $(String(name))",
        ))
    end
    for name in (
        :firing_ema_decay, :target_rate_min, :target_rate_max,
        :threshold_homeostasis_step, :adaptation_homeostasis_step,
        :synaptic_scaling_rate, :conductance_floor, :conductance_ceiling,
        :structure_enabled, :utility_decay, :connection_cost,
        :max_swaps_per_node,
    )
        hasproperty(plasticity, name) || throw(ArgumentError(
            "plasticity config is missing $(String(name))",
        ))
    end
    return (;
        config,
        fixed_local_signal_map=(
            output_dim=_CANONICAL_OUTPUT_DIM,
            continuous_observation_dim=_CANONICAL_CONTINUOUS_OBSERVATION_DIM,
            packet_observation_dim=_CANONICAL_PACKET_DIM,
            cell_count=_CANONICAL_CORE_COUNT,
            seed_source=:feedback_seed,
            family_source=:ordered_node_class,
            cell_source=:ordered_node_id,
        ),
    )
end

learning_fingerprint(config) = _contract_fingerprint(
    "learning",
    canonical_learning_contract(config),
)
optimizer_fingerprint(config) = _contract_fingerprint("optimizer", config)

@inline function _is_parameter_container(value)
    return value isa NamedTuple || value isa Tuple || (
        isstructtype(typeof(value)) &&
        !(value isa Number) &&
        !(value isa AbstractString) &&
        !(value isa Symbol) &&
        !(value isa Enum)
    )
end

function _collect_parameter_arrays!(
    paths::Vector{String},
    arrays::Vector{DenseArray{Float32}},
    value,
    path::String,
)
    if value isa DenseArray{Float32}
        isempty(path) && throw(ArgumentError("parameter path cannot be empty"))
        @inbounds for previous in arrays
            Base.mightalias(previous, value) && throw(ArgumentError(
                "parameter array $path aliases another registry entry",
            ))
        end
        push!(paths, path)
        push!(arrays, value)
        return nothing
    elseif value isa AbstractArray
        throw(ArgumentError(
            "parameter $path must be a dense Float32 array, got $(typeof(value))",
        ))
    elseif value isa NamedTuple
        isempty(value) && throw(ArgumentError("parameter container $path is empty"))
        @inbounds for name in keys(value)
            child = isempty(path) ? String(name) : "$path/$(String(name))"
            _collect_parameter_arrays!(
                paths,
                arrays,
                getproperty(value, name),
                child,
            )
        end
        return nothing
    elseif value isa Tuple
        isempty(value) && throw(ArgumentError("parameter container $path is empty"))
        @inbounds for (index, child_value) in enumerate(value)
            child = isempty(path) ? "[$index]" : "$path/[$index]"
            _collect_parameter_arrays!(paths, arrays, child_value, child)
        end
        return nothing
    elseif _is_parameter_container(value)
        names = fieldnames(typeof(value))
        isempty(names) && throw(ArgumentError("parameter container $path is empty"))
        @inbounds for name in names
            child = isempty(path) ? String(name) : "$path/$(String(name))"
            _collect_parameter_arrays!(
                paths,
                arrays,
                getfield(value, name),
                child,
            )
        end
        return nothing
    end
    throw(ArgumentError(
        "parameter registry leaf $path must be a dense Float32 array, " *
        "got $(typeof(value))",
    ))
end

function _parameter_arrays(components)
    paths = String[]
    arrays = DenseArray{Float32}[]
    _collect_parameter_arrays!(paths, arrays, components, "")
    isempty(arrays) && throw(ArgumentError("parameter registry cannot be empty"))
    length(unique(paths)) == length(paths) || throw(ArgumentError(
        "parameter registry contains duplicate paths",
    ))
    return paths, arrays
end

function _registry_record(paths, arrays)
    records = Vector{NamedTuple{_REGISTRY_KEYS,Tuple{
        String,String,Tuple{Vararg{Int}},Int,
    }}}(undef, length(arrays))
    @inbounds for index in eachindex(arrays)
        array = arrays[index]
        records[index] = (;
            path=paths[index],
            element_type=_type_label(eltype(array)),
            dimensions=Tuple(size(array)),
            length=length(array),
        )
    end
    return records
end

"""Return the exact ordered path/type/shape registry for trainable arrays."""
function parameter_registry(components)
    paths, arrays = _parameter_arrays(components)
    return _registry_record(paths, arrays)
end

function _validate_numeric_arrays(arrays, label::AbstractString;
                                  nonnegative::Bool=false)
    @inbounds for (array_index, array) in enumerate(arrays)
        for value in array
            isfinite(value) || throw(DomainError(
                value,
                "$label array $array_index contains a non-finite value",
            ))
            nonnegative && value < 0.0f0 && throw(DomainError(
                value,
                "$label array $array_index contains a negative value",
            ))
        end
    end
    return arrays
end

function _validate_counter_value(value, path::String="counters")
    if value isa Bool
        throw(ArgumentError("$path cannot be Bool"))
    elseif value isa Integer
        value >= 0 || throw(ArgumentError("$path cannot be negative"))
    elseif value isa NamedTuple
        isempty(value) && throw(ArgumentError("$path cannot be empty"))
        @inbounds for name in keys(value)
            _validate_counter_value(getproperty(value, name), "$path/$(String(name))")
        end
    elseif value isa Tuple
        isempty(value) && throw(ArgumentError("$path cannot be empty"))
        @inbounds for (index, item) in enumerate(value)
            _validate_counter_value(item, "$path/[$index]")
        end
    elseif isstructtype(typeof(value)) && !(value isa Number)
        names = fieldnames(typeof(value))
        isempty(names) && throw(ArgumentError("$path cannot be empty"))
        @inbounds for name in names
            _validate_counter_value(getfield(value, name), "$path/$(String(name))")
        end
    else
        throw(ArgumentError(
            "$path must contain only non-negative integer counters",
        ))
    end
    return value
end

function _fingerprints(
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    return (;
        architecture=architecture_fingerprint(architecture_config),
        input=input_fingerprint(input_config),
        topology=topology_fingerprint(topology_config),
        learning=learning_fingerprint(learning_config),
        optimizer=optimizer_fingerprint(optimizer_config),
    )
end

function _write_array_state(io::IO, label::String, records, arrays)
    _write_string(io, label)
    _write_u64(io, UInt64(length(arrays)))
    @inbounds for index in eachindex(arrays)
        record = records[index]
        _write_contract(io, record)
        bytes = reinterpret(UInt8, vec(arrays[index]))
        _write_string(io, bytes2hex(SHA.sha256(bytes)))
    end
    return io
end

function _state_fingerprint(
    fingerprints,
    registry,
    values,
    first,
    second,
    optimizer_step::Int,
    counters,
)
    io = IOBuffer()
    _write_string(io, FINGERPRINT_ALGORITHM)
    _write_string(io, CHECKPOINT_FORMAT)
    _write_contract(io, fingerprints)
    _write_array_state(io, "parameters", registry, values)
    _write_array_state(io, "first_moments", registry, first)
    _write_array_state(io, "second_moments", registry, second)
    _write_contract(io, optimizer_step)
    _write_contract(io, counters)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _validate_registry(registry)
    registry isa Vector || throw(ArgumentError(
        "checkpoint parameter registry is not a Vector",
    ))
    isempty(registry) && throw(ArgumentError(
        "checkpoint parameter registry cannot be empty",
    ))
    paths = String[]
    @inbounds for record in registry
        record isa NamedTuple || throw(ArgumentError(
            "checkpoint registry entry is not a NamedTuple",
        ))
        keys(record) == _REGISTRY_KEYS || throw(ArgumentError(
            "checkpoint registry entry fields are missing or extra",
        ))
        record.path isa String && !isempty(record.path) || throw(ArgumentError(
            "checkpoint registry path is invalid",
        ))
        record.element_type == "Float32" || throw(ArgumentError(
            "checkpoint registry element type is not Float32",
        ))
        record.dimensions isa Tuple || throw(ArgumentError(
            "checkpoint registry dimensions are invalid",
        ))
        all(dimension -> dimension >= 0, record.dimensions) || throw(
            ArgumentError("checkpoint registry has a negative dimension"),
        )
        record.length isa Int && record.length >= 0 || throw(ArgumentError(
            "checkpoint registry length is invalid",
        ))
        prod(record.dimensions; init=1) == record.length || throw(ArgumentError(
            "checkpoint registry length disagrees with dimensions",
        ))
        push!(paths, record.path)
    end
    length(unique(paths)) == length(paths) || throw(ArgumentError(
        "checkpoint registry contains duplicate paths",
    ))
    return registry
end

function _validate_saved_arrays(arrays, registry, label; nonnegative=false)
    arrays isa Vector || throw(ArgumentError("checkpoint $label is not a Vector"))
    length(arrays) == length(registry) || throw(DimensionMismatch(
        "checkpoint $label count differs from the parameter registry",
    ))
    @inbounds for index in eachindex(arrays)
        array = arrays[index]
        array isa Array{Float32} || throw(ArgumentError(
            "checkpoint $label entry $index is not a dense Float32 Array",
        ))
        Tuple(size(array)) == registry[index].dimensions || throw(
            DimensionMismatch("checkpoint $label shape differs at $(registry[index].path)"),
        )
    end
    _validate_numeric_arrays(arrays, label; nonnegative=nonnegative)
    return arrays
end

function _validate_header(snapshot)
    snapshot isa NamedTuple || throw(ArgumentError(
        "checkpoint payload is not the canonical NamedTuple format",
    ))
    keys(snapshot) == _SNAPSHOT_KEYS || throw(ArgumentError(
        "checkpoint fields are missing or extra; legacy checkpoints are unsupported",
    ))
    snapshot.magic === CHECKPOINT_MAGIC || throw(ArgumentError(
        "checkpoint magic is not canonical",
    ))
    snapshot.schema === CHECKPOINT_SCHEMA || throw(ArgumentError(
        "checkpoint schema is unsupported",
    ))
    snapshot.format === CHECKPOINT_FORMAT || throw(ArgumentError(
        "checkpoint format is unsupported; relation/motif checkpoints cannot resume",
    ))
    return snapshot
end

function _validate_snapshot(snapshot)
    _validate_header(snapshot)
    fingerprints = (;
        architecture=snapshot.architecture_fingerprint,
        input=snapshot.input_fingerprint,
        topology=snapshot.topology_fingerprint,
        learning=snapshot.learning_fingerprint,
        optimizer=snapshot.optimizer_fingerprint,
    )
    @inbounds for (name, fingerprint) in pairs(fingerprints)
        fingerprint isa String && occursin(r"^[0-9a-f]{64}$", fingerprint) ||
            throw(ArgumentError("checkpoint $name fingerprint is malformed"))
    end
    registry = _validate_registry(snapshot.parameter_registry)
    _validate_saved_arrays(snapshot.parameter_values, registry, "parameters")
    _validate_saved_arrays(snapshot.first_moments, registry, "first moments")
    _validate_saved_arrays(
        snapshot.second_moments,
        registry,
        "second moments";
        nonnegative=true,
    )
    snapshot.optimizer_step isa Int || throw(ArgumentError(
        "checkpoint optimizer step must be Int",
    ))
    snapshot.optimizer_step >= 0 || throw(ArgumentError(
        "checkpoint optimizer step cannot be negative",
    ))
    _validate_counter_value(snapshot.counters)
    snapshot.state_fingerprint isa String || throw(ArgumentError(
        "checkpoint state fingerprint is not a String",
    ))
    expected = _state_fingerprint(
        fingerprints,
        registry,
        snapshot.parameter_values,
        snapshot.first_moments,
        snapshot.second_moments,
        snapshot.optimizer_step,
        snapshot.counters,
    )
    snapshot.state_fingerprint == expected || throw(ArgumentError(
        "checkpoint state fingerprint is false",
    ))
    return snapshot
end

"""Load and fully validate the new canonical format. Legacy formats fail closed."""
function load_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("checkpoint does not exist: $source"))
    snapshot = try
        open(deserialize, source)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError(
            "checkpoint cannot be decoded as the canonical format: " *
            sprint(showerror, error),
        ))
    end
    return _validate_snapshot(snapshot)
end

"""Atomically save the complete canonical train/AdamW state."""
function save_checkpoint(
    path::AbstractString;
    parameters,
    first_moments,
    second_moments,
    optimizer_step::Integer,
    counters,
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    optimizer_step isa Bool && throw(ArgumentError("optimizer step cannot be Bool"))
    step = try
        Int(optimizer_step)
    catch
        throw(ArgumentError("optimizer step does not fit Int"))
    end
    step >= 0 || throw(ArgumentError("optimizer step cannot be negative"))
    _validate_counter_value(counters)

    parameter_paths, parameter_arrays = _parameter_arrays(parameters)
    first_paths, first_arrays = _parameter_arrays(first_moments)
    second_paths, second_arrays = _parameter_arrays(second_moments)
    registry = _registry_record(parameter_paths, parameter_arrays)
    first_registry = _registry_record(first_paths, first_arrays)
    second_registry = _registry_record(second_paths, second_arrays)
    first_registry == registry || throw(ArgumentError(
        "first-moment registry does not exactly match parameters",
    ))
    second_registry == registry || throw(ArgumentError(
        "second-moment registry does not exactly match parameters",
    ))
    _validate_numeric_arrays(parameter_arrays, "parameters")
    _validate_numeric_arrays(first_arrays, "first moments")
    _validate_numeric_arrays(second_arrays, "second moments"; nonnegative=true)

    fingerprints = _fingerprints(
        architecture_config,
        input_config,
        topology_config,
        learning_config,
        optimizer_config,
    )
    parameter_values = [copy(array) for array in parameter_arrays]
    first_values = [copy(array) for array in first_arrays]
    second_values = [copy(array) for array in second_arrays]
    counter_values = deepcopy(counters)
    state_fingerprint = _state_fingerprint(
        fingerprints,
        registry,
        parameter_values,
        first_values,
        second_values,
        step,
        counter_values,
    )
    snapshot = (;
        magic=CHECKPOINT_MAGIC,
        schema=CHECKPOINT_SCHEMA,
        format=CHECKPOINT_FORMAT,
        architecture_fingerprint=fingerprints.architecture,
        input_fingerprint=fingerprints.input,
        topology_fingerprint=fingerprints.topology,
        learning_fingerprint=fingerprints.learning,
        optimizer_fingerprint=fingerprints.optimizer,
        parameter_registry=registry,
        parameter_values=parameter_values,
        first_moments=first_values,
        second_moments=second_values,
        optimizer_step=step,
        counters=counter_values,
        state_fingerprint=state_fingerprint,
    )
    _validate_snapshot(snapshot)

    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination); cleanup=false)
    try
        open(temporary, "w") do io
            serialize(io, snapshot)
            flush(io)
        end
        mv(temporary, destination; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

function _assert_contract_fingerprints(
    snapshot,
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    current = _fingerprints(
        architecture_config,
        input_config,
        topology_config,
        learning_config,
        optimizer_config,
    )
    snapshot.architecture_fingerprint == current.architecture || throw(
        ArgumentError("checkpoint architecture configuration differs"),
    )
    snapshot.input_fingerprint == current.input || throw(
        ArgumentError("checkpoint input configuration differs"),
    )
    snapshot.topology_fingerprint == current.topology || throw(
        ArgumentError("checkpoint topology configuration differs"),
    )
    snapshot.learning_fingerprint == current.learning || throw(
        ArgumentError("checkpoint learning configuration differs"),
    )
    snapshot.optimizer_fingerprint == current.optimizer || throw(
        ArgumentError("checkpoint optimizer configuration differs"),
    )
    return current
end

"""
Validate every contract and registry before mutating live arrays, then restore
parameters and both AdamW moment sets. Step/counters are returned explicitly so
their owning optimizer/learner can install them without hidden reflection.
"""
function restore_checkpoint!(
    parameters,
    first_moments,
    second_moments,
    snapshot;
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    _validate_snapshot(snapshot)
    _assert_contract_fingerprints(
        snapshot,
        architecture_config,
        input_config,
        topology_config,
        learning_config,
        optimizer_config,
    )

    parameter_paths, parameter_arrays = _parameter_arrays(parameters)
    first_paths, first_arrays = _parameter_arrays(first_moments)
    second_paths, second_arrays = _parameter_arrays(second_moments)
    current_registry = _registry_record(parameter_paths, parameter_arrays)
    current_registry == snapshot.parameter_registry || throw(ArgumentError(
        "live parameter registry does not exactly match checkpoint",
    ))
    _registry_record(first_paths, first_arrays) == current_registry || throw(
        ArgumentError("live first-moment registry does not match parameters"),
    )
    _registry_record(second_paths, second_arrays) == current_registry || throw(
        ArgumentError("live second-moment registry does not match parameters"),
    )

    # Mutation starts only after the complete snapshot and live contract pass.
    @inbounds for index in eachindex(parameter_arrays)
        copyto!(parameter_arrays[index], snapshot.parameter_values[index])
        copyto!(first_arrays[index], snapshot.first_moments[index])
        copyto!(second_arrays[index], snapshot.second_moments[index])
    end
    return ResumeState(snapshot.optimizer_step, deepcopy(snapshot.counters))
end

@inline function _require_property(value, name::Symbol, owner::AbstractString)
    hasproperty(value, name) || throw(ArgumentError(
        "$owner has no $(String(name)) field",
    ))
    return getproperty(value, name)
end

function _optimizer_registry_contract(registry)
    groups = _require_property(registry, :groups, "optimizer registry")
    groups isa Tuple && !isempty(groups) || throw(ArgumentError(
        "optimizer registry groups must be a non-empty Tuple",
    ))
    contracts = map(groups) do group
        name = _require_property(group, :name, "parameter group")
        parameter = _require_property(group, :parameter, "parameter group $name")
        gradient = _require_property(group, :gradient, "parameter group $name")
        transform_kind = _require_property(
            group,
            :transform_kind,
            "parameter group $name",
        )
        multiplier = Float32(_require_property(
            group,
            :multiplier,
            "parameter group $name",
        ))
        lower_bound = Float32(_require_property(
            group,
            :lower_bound,
            "parameter group $name",
        ))
        upper_bound = Float32(_require_property(
            group,
            :upper_bound,
            "parameter group $name",
        ))
        name isa Symbol && name !== Symbol("") || throw(ArgumentError(
            "parameter-group name must be a non-empty Symbol",
        ))
        parameter isa DenseArray{Float32} || throw(ArgumentError(
            "parameter group $name storage must be a dense Float32 array",
        ))
        gradient isa AbstractArray{Float32} && size(gradient) == size(parameter) ||
            throw(DimensionMismatch(
                "parameter group $name gradient shape differs",
            ))
        isfinite(multiplier) && multiplier >= 0.0f0 || throw(ArgumentError(
            "parameter group $name multiplier is invalid",
        ))
        !isnan(lower_bound) && !isnan(upper_bound) && lower_bound < upper_bound ||
            throw(ArgumentError("parameter group $name bounds are invalid"))
        return (;
            name,
            transform_kind=string(transform_kind),
            multiplier_bits=reinterpret(UInt32, multiplier),
            lower_bound_bits=reinterpret(UInt32, lower_bound),
            upper_bound_bits=reinterpret(UInt32, upper_bound),
            dimensions=Tuple(size(parameter)),
        )
    end
    names = map(contract -> contract.name, contracts)
    length(unique(names)) == length(names) || throw(ArgumentError(
        "optimizer registry contains duplicate group names",
    ))
    names == CANONICAL_PARAMETER_GROUPS || throw(ArgumentError(
        "optimizer registry must contain the five canonical graph groups " *
        "in canonical order",
    ))
    return contracts
end

function _optimizer_components(registry, state)
    contracts = _optimizer_registry_contract(registry)
    groups = getproperty(registry, :groups)
    moments = _require_property(state, :moments, "optimizer state")
    moments isa Tuple && length(moments) == length(groups) || throw(ArgumentError(
        "optimizer moment tuple differs from parameter groups",
    ))
    names = Tuple(contract.name for contract in contracts)
    parameters = NamedTuple{names}(Tuple(group.parameter for group in groups))
    first = NamedTuple{names}(ntuple(length(groups)) do index
        group = groups[index]
        moment = moments[index]
        moment_name = _require_property(moment, :name, "optimizer moment")
        moment_name == group.name || throw(ArgumentError(
            "optimizer moment name differs for group $(group.name)",
        ))
        for field in (:transform_kind, :multiplier, :lower_bound, :upper_bound)
            isequal(getproperty(moment, field), getproperty(group, field)) || throw(
                ArgumentError(
                    "optimizer moment metadata $field differs for group $(group.name)",
                ),
            )
        end
        array = _require_property(moment, :first, "optimizer moment $(group.name)")
        size(array) == size(group.parameter) || throw(DimensionMismatch(
            "first-moment shape differs for group $(group.name)",
        ))
        array
    end)
    second = NamedTuple{names}(ntuple(length(groups)) do index
        group = groups[index]
        moment = moments[index]
        array = _require_property(moment, :second, "optimizer moment $(group.name)")
        size(array) == size(group.parameter) || throw(DimensionMismatch(
            "second-moment shape differs for group $(group.name)",
        ))
        array
    end)
    group_steps = _require_property(state, :group_steps, "optimizer state")
    total_step = _require_property(state, :total_step, "optimizer state")
    group_steps isa Vector{UInt64} && length(group_steps) == length(groups) ||
        throw(ArgumentError("optimizer group-step vector differs from registry"))
    total_step isa UInt64 || throw(ArgumentError(
        "optimizer total_step must be UInt64",
    ))
    @inbounds for (index, group_step) in enumerate(group_steps)
        group_step <= total_step || throw(ArgumentError(
            "optimizer group step $index exceeds total step",
        ))
        contracts[index].multiplier_bits == reinterpret(UInt32, 0.0f0) &&
            group_step != 0 && throw(ArgumentError(
                "frozen optimizer group $(contracts[index].name) has a nonzero step",
            ))
    end
    total_step <= UInt64(typemax(Int)) || throw(ArgumentError(
        "optimizer total step does not fit checkpoint Int",
    ))
    return parameters, first, second, contracts, group_steps, Int(total_step)
end

@inline function _counter_fields(value)
    value isa NamedTuple && return keys(value)
    return fieldnames(typeof(value))
end

function _validate_training_counters(counters, learning_config, total_step::Int)
    _counter_fields(counters) ==
        (:learning_clock, :mechanisms, :training_updates) || throw(
            ArgumentError(
                "training counters must contain learning_clock, mechanisms, " *
                "training_updates in canonical order",
            ),
        )
    clock = getproperty(counters, :learning_clock)
    mechanisms = getproperty(counters, :mechanisms)
    updates = getproperty(counters, :training_updates)
    _counter_fields(clock) == (
        :update,
        :analog_ticks,
        :hard_event_ticks,
        :homeostasis_ticks,
        :structure_ticks,
    ) || throw(ArgumentError(
        "learning clock fields are missing, extra, or reordered",
    ))
    _counter_fields(mechanisms) == CANONICAL_MECHANISM_COUNTERS || throw(
        ArgumentError(
            "mechanism counter fields are missing, extra, or reordered",
        ),
    )
    updates isa UInt64 && updates == UInt64(total_step) || throw(ArgumentError(
        "training update counter differs from optimizer total step",
    ))
    clock.update isa Int && clock.update == total_step || throw(ArgumentError(
        "learning clock update differs from optimizer total step",
    ))
    local_config = _local_learning_config(learning_config)
    schedule = local_config.schedule
    expected = (
        fld(total_step, schedule.analog_interval),
        fld(total_step, schedule.hard_event_interval),
        fld(total_step, schedule.homeostasis_interval),
        fld(total_step, schedule.structure_interval),
    )
    actual = (
        clock.analog_ticks,
        clock.hard_event_ticks,
        clock.homeostasis_ticks,
        clock.structure_ticks,
    )
    actual == expected || throw(ArgumentError(
        "learning clock ticks disagree with the checkpointed schedule",
    ))
    _validate_counter_value(counters)
    return counters
end

function _validate_canonical_group_shapes(parameters, topology_config)
    keys(parameters) == CANONICAL_PARAMETER_GROUPS || throw(ArgumentError(
        "canonical parameter components are missing, extra, or reordered",
    ))
    size(parameters.core_cell_raw) ==
        (_CANONICAL_CELL_PARAMETER_DIM, _CANONICAL_CORE_COUNT) || throw(
            DimensionMismatch("core_cell_raw has the wrong canonical shape"),
        )
    size(parameters.semantic_projection_raw) == (4, 3, 8, 2) || throw(
        DimensionMismatch(
            "semantic_projection_raw has the wrong canonical shape",
        ),
    )
    event_order = _canonical_event_parameter_order(topology_config)
    length(parameters.event_raw) == event_order.parameter_count || throw(
        DimensionMismatch(
            "event_raw length differs from canonical static/dynamic/kind order",
        ),
    )
    size(parameters.output_cell_raw) ==
        (_CANONICAL_CELL_PARAMETER_DIM, _CANONICAL_OUTPUT_DIM) || throw(
            DimensionMismatch("output_cell_raw has the wrong canonical shape"),
        )
    size(parameters.output_projection_raw) == (4, 3, 5) || throw(
        DimensionMismatch(
            "output_projection_raw has the wrong canonical shape",
        ),
    )
    return parameters
end

"""Save directly from `CanonicalOptimizer.ParameterRegistry`/`AdamWState`."""
function save_checkpoint(
    path::AbstractString,
    registry,
    optimizer_state;
    counters,
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    parameters, first, second, group_contract, group_steps, total_step =
        _optimizer_components(registry, optimizer_state)
    _validate_canonical_group_shapes(parameters, topology_config)
    _validate_training_counters(counters, learning_config, total_step)
    optimizer_contract = (;
        adam=optimizer_config,
        parameter_groups=group_contract,
    )
    stored_counters = (;
        optimizer_group_steps=Tuple(group_steps),
        learner=deepcopy(counters),
    )
    return save_checkpoint(
        path;
        parameters,
        first_moments=first,
        second_moments=second,
        optimizer_step=total_step,
        counters=stored_counters,
        architecture_config,
        input_config,
        topology_config,
        learning_config,
        optimizer_config=optimizer_contract,
    )
end

"""Restore directly into the canonical optimizer after complete validation."""
function restore_checkpoint!(
    registry,
    optimizer_state,
    snapshot;
    architecture_config,
    input_config,
    topology_config,
    learning_config,
    optimizer_config,
)
    _validate_snapshot(snapshot)
    parameters, first, second, group_contract, _, _ =
        _optimizer_components(registry, optimizer_state)
    _validate_canonical_group_shapes(parameters, topology_config)
    optimizer_contract = (;
        adam=optimizer_config,
        parameter_groups=group_contract,
    )
    counters = snapshot isa NamedTuple && hasproperty(snapshot, :counters) ?
        snapshot.counters : nothing
    counters isa NamedTuple && keys(counters) == (:optimizer_group_steps, :learner) ||
        throw(ArgumentError("checkpoint optimizer counters are missing or extra"))
    group_steps = counters.optimizer_group_steps
    group_steps isa Tuple && length(group_steps) == length(group_contract) ||
        throw(ArgumentError("checkpoint optimizer group-step count differs"))
    all(step -> step isa UInt64 && step <= UInt64(snapshot.optimizer_step), group_steps) ||
        throw(ArgumentError("checkpoint optimizer group steps are invalid"))
    _validate_training_counters(
        counters.learner,
        learning_config,
        snapshot.optimizer_step,
    )

    resume = restore_checkpoint!(
        parameters,
        first,
        second,
        snapshot;
        architecture_config,
        input_config,
        topology_config,
        learning_config,
        optimizer_config=optimizer_contract,
    )
    optimizer_state.group_steps .= group_steps
    optimizer_state.total_step = UInt64(resume.optimizer_step)
    # `event_raw`, cell dynamics, and output projections all have derived
    # caches.  A resumed optimizer must never observe the constructor cache
    # paired with restored raw coordinates.
    Graph.refresh_cache!(topology_config)
    return ResumeState(resume.optimizer_step, deepcopy(counters.learner))
end

end # module CanonicalCheckpoint
