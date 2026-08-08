module CanonicalCheckpoint

using Serialization
using SHA
using ..CanonicalDendriticGraph
using ..CanonicalLocalLearning
using ..CanonicalOptimizer
using ..CanonicalTetrisInput
using ..ReducedHayCPUSampler

const Graph = CanonicalDendriticGraph
const Local = CanonicalLocalLearning
const Optimizer = CanonicalOptimizer
const Input = CanonicalTetrisInput
const Sampler = ReducedHayCPUSampler

const CHECKPOINT_MAGIC = UInt64(0x43414e4f4e484447) # "CANONHDG"
const CHECKPOINT_SCHEMA = UInt32(2)
const CHECKPOINT_FORMAT =
    "route-free-ordered-multiscale-dendritic-event-graph-v2"
const RUN_CONTRACT_SCHEMA = UInt32(1)
const FINGERPRINT_ALGORITHM = "sha256-canonical-binary-contract-v2"

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
    :hard_event_control_updates,
    :homeostasis_events,
    :synaptic_scaling_events,
    :utility_updates,
    :rewires,
)

const _CANONICAL_NODE_COUNT = 1_458
const _CANONICAL_EDGE_COUNT = 2_216
const _CANONICAL_CORE_COUNT = 1_436
const _CANONICAL_OUTPUT_DIM = 22
const _CANONICAL_CELL_PARAMETER_DIM = 46
const _CANONICAL_EVENT_PARAMETER_COUNT = 2_125
const _CANONICAL_EVENT_CONTACT_COUNT = 2_120
const _CANONICAL_STATE_BATCH = 8
const _CANONICAL_CANDIDATE_WIDTH = 80
const _CANONICAL_WORKERS = 20
const _CANONICAL_QUEUE_CAPACITY = 64
const _CANONICAL_CHUNK_SIZE = 4
const _CANONICAL_MAX_UPDATES = 100_000

export CHECKPOINT_FORMAT,
       CHECKPOINT_MAGIC,
       CHECKPOINT_SCHEMA,
       RUN_CONTRACT_SCHEMA,
       CANONICAL_PARAMETER_GROUPS,
       CANONICAL_MECHANISM_COUNTERS,
       CanonicalRunContract,
       ParameterGroupContract,
       CanonicalParameterState,
       CanonicalOptimizerStateSnapshot,
       LearningClockSnapshot,
       MechanismCounterSnapshot,
       PlasticityStateSnapshot,
       SamplerStateSnapshot,
       CanonicalTrainingStateSnapshot,
       PreparedTrainingCheckpoint,
       architecture_fingerprint,
       build_training_snapshot,
       canonical_architecture_contract,
       canonical_input_contract,
       canonical_learning_contract,
       input_fingerprint,
       learning_fingerprint,
       load_checkpoint,
       optimizer_fingerprint,
       prepare_training_checkpoint,
       run_contract_fingerprint,
       save_checkpoint,
       topology_fingerprint,
       validate_prepared_checkpoint

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

function _write_float(io::IO, value::T) where {T<:AbstractFloat}
    isfinite(value) || throw(DomainError(value, "contract float is not finite"))
    if T === Float16
        _write_u64(io, UInt64(reinterpret(UInt16, value)))
    elseif T === Float32
        _write_u64(io, UInt64(reinterpret(UInt32, value)))
    elseif T === Float64
        _write_u64(io, reinterpret(UInt64, value))
    else
        throw(ArgumentError("unsupported contract float type $T"))
    end
    return io
end

"""Stable encoding for immutable contracts and copied checkpoint state."""
function _write_contract(io::IO, value)
    _write_string(io, string(typeof(value)))
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
            "checkpoint array element type $(eltype(value)) is unsupported",
        ))
        _write_u64(io, UInt64(ndims(value)))
        @inbounds for dimension in size(value)
            _write_u64(io, UInt64(dimension))
        end
        _write_string(io, bytes2hex(SHA.sha256(reinterpret(UInt8, vec(value)))))
    elseif isstructtype(typeof(value))
        names = fieldnames(typeof(value))
        _write_u64(io, UInt64(length(names)))
        @inbounds for name in names
            _write_string(io, String(name))
            _write_contract(io, getfield(value, name))
        end
    else
        throw(ArgumentError("unsupported checkpoint value $(typeof(value))"))
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

@inline function _require_sha256(value::String, label::AbstractString)
    occursin(r"^[0-9a-f]{64}$", value) || throw(ArgumentError(
        "$label must be lowercase 64-hex SHA-256",
    ))
    return value
end

@inline function _require_commit(value::String)
    occursin(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", value) || throw(
        ArgumentError("source_commit must be a full lowercase 40/64-hex ID"),
    )
    return value
end

@inline function _require_positive(value::Int, label::AbstractString)
    value > 0 || throw(ArgumentError("$label must be positive"))
    return value
end

"""Immutable operational and dataset identity contract for the 100k run."""
struct CanonicalRunContract
    schema::UInt32
    dataset_format_version::Int
    dataset_manifest_sha256::String
    ordered_training_rows_sha256::String
    portable_dataset_sha256::String
    dataset_state_count::Int
    training_state_count::Int
    validation_state_count::Int
    dataset_candidate_count::Int
    dataset_part_count::Int
    dataset_schema_fingerprint::String
    teacher_target_transform_fingerprint::String
    split_identity::String
    source_commit::String
    source_tree_clean::Bool
    model_seed::UInt64
    sampler_seed::UInt64
    state_batch::Int
    candidate_width::Int
    workers::Int
    queue_capacity::Int
    chunk_size::Int
    binding::Symbol
    training_config_fingerprint::String
    architecture_fingerprint::String
    topology_fingerprint::String
    promotion_plan_fingerprint::String
    planned_max_updates::Int
    log_interval::Int
    evaluation_interval::Int
    checkpoint_interval::Int
    update_semantics::Symbol
end

function _validate_run_contract(contract::CanonicalRunContract)
    contract.schema == RUN_CONTRACT_SCHEMA || throw(ArgumentError(
        "run-contract schema is unsupported",
    ))
    _require_positive(contract.dataset_format_version, "dataset format version")
    for (label, value) in (
        ("dataset manifest SHA-256", contract.dataset_manifest_sha256),
        ("ordered training rows SHA-256", contract.ordered_training_rows_sha256),
        ("portable dataset SHA-256", contract.portable_dataset_sha256),
        ("dataset schema fingerprint", contract.dataset_schema_fingerprint),
        ("teacher target transform fingerprint",
            contract.teacher_target_transform_fingerprint),
        ("training configuration fingerprint",
            contract.training_config_fingerprint),
        ("architecture fingerprint", contract.architecture_fingerprint),
        ("topology fingerprint", contract.topology_fingerprint),
        ("promotion plan fingerprint", contract.promotion_plan_fingerprint),
    )
        _require_sha256(value, label)
    end
    _require_commit(contract.source_commit)
    contract.source_tree_clean || throw(ArgumentError(
        "run contract requires a clean source-tree attestation",
    ))
    isempty(contract.split_identity) && throw(ArgumentError(
        "split_identity cannot be empty",
    ))
    contract.update_semantics === Symbol("") && throw(ArgumentError(
        "update_semantics cannot be empty",
    ))
    _require_positive(contract.dataset_state_count, "dataset state count")
    _require_positive(contract.training_state_count, "training state count")
    _require_positive(contract.validation_state_count, "validation state count")
    _require_positive(contract.dataset_candidate_count, "dataset candidate count")
    _require_positive(contract.dataset_part_count, "dataset part count")
    contract.training_state_count + contract.validation_state_count <=
        contract.dataset_state_count || throw(ArgumentError(
            "training/validation states exceed dataset state count",
        ))
    contract.state_batch == _CANONICAL_STATE_BATCH || throw(ArgumentError(
        "canonical state_batch must be $_CANONICAL_STATE_BATCH",
    ))
    contract.candidate_width == _CANONICAL_CANDIDATE_WIDTH || throw(
        ArgumentError("canonical candidate_width must be $_CANONICAL_CANDIDATE_WIDTH"),
    )
    contract.dataset_candidate_count >= contract.dataset_state_count || throw(
        ArgumentError("dataset has fewer candidates than states"),
    )
    UInt128(contract.dataset_candidate_count) <=
        UInt128(contract.dataset_state_count) * UInt128(contract.candidate_width) ||
        throw(ArgumentError("dataset candidates exceed the width-80 bound"))
    contract.workers == _CANONICAL_WORKERS || throw(ArgumentError(
        "canonical workers must be $_CANONICAL_WORKERS",
    ))
    contract.queue_capacity == _CANONICAL_QUEUE_CAPACITY || throw(ArgumentError(
        "canonical queue capacity must be $_CANONICAL_QUEUE_CAPACITY",
    ))
    contract.chunk_size == _CANONICAL_CHUNK_SIZE || throw(ArgumentError(
        "canonical chunk size must be $_CANONICAL_CHUNK_SIZE",
    ))
    contract.binding === :none || throw(ArgumentError(
        "canonical binding must be :none",
    ))
    contract.planned_max_updates == _CANONICAL_MAX_UPDATES || throw(
        ArgumentError("canonical planned_max_updates must be $_CANONICAL_MAX_UPDATES"),
    )
    for (label, cadence) in (
        ("log interval", contract.log_interval),
        ("evaluation interval", contract.evaluation_interval),
        ("checkpoint interval", contract.checkpoint_interval),
    )
        _require_positive(cadence, label)
        cadence <= contract.planned_max_updates || throw(ArgumentError(
            "$label exceeds planned_max_updates",
        ))
    end
    return contract
end

run_contract_fingerprint(contract::CanonicalRunContract) =
    _contract_fingerprint("run-contract", _validate_run_contract(contract))

struct ParameterGroupContract
    name::Symbol
    transform_kind::Optimizer.ParameterTransformKind
    multiplier_bits::UInt32
    lower_bound_bits::UInt32
    upper_bound_bits::UInt32
    projected_lower_raw_bits::UInt32
    projected_upper_raw_bits::UInt32
    dimensions::Tuple
end

function ParameterGroupContract(
    name::Symbol,
    transform_kind::Optimizer.ParameterTransformKind,
    multiplier::Float32,
    lower_bound::Float32,
    upper_bound::Float32,
    projected_lower_raw::Float32,
    projected_upper_raw::Float32,
    dimensions::Tuple,
)
    name === Symbol("") && throw(ArgumentError("parameter-group name is empty"))
    isfinite(multiplier) && multiplier >= 0.0f0 || throw(ArgumentError(
        "parameter-group multiplier is invalid",
    ))
    !isnan(lower_bound) && !isnan(upper_bound) && lower_bound < upper_bound ||
        throw(ArgumentError("parameter-group physical bounds are invalid"))
    !isnan(projected_lower_raw) && !isnan(projected_upper_raw) &&
        projected_lower_raw < projected_upper_raw || throw(ArgumentError(
            "parameter-group raw bounds are invalid",
        ))
    !isempty(dimensions) && all(dimension -> dimension isa Int && dimension > 0,
        dimensions) || throw(ArgumentError("parameter-group dimensions are invalid"))
    return ParameterGroupContract(
        name,
        transform_kind,
        reinterpret(UInt32, multiplier),
        reinterpret(UInt32, lower_bound),
        reinterpret(UInt32, upper_bound),
        reinterpret(UInt32, projected_lower_raw),
        reinterpret(UInt32, projected_upper_raw),
        dimensions,
    )
end

@inline _multiplier(group::ParameterGroupContract) =
    reinterpret(Float32, group.multiplier_bits)
@inline _raw_lower(group::ParameterGroupContract) =
    reinterpret(Float32, group.projected_lower_raw_bits)
@inline _raw_upper(group::ParameterGroupContract) =
    reinterpret(Float32, group.projected_upper_raw_bits)

struct CanonicalParameterState
    core_cell_raw::Matrix{Float32}
    semantic_projection_raw::Array{Float32,4}
    event_raw::Vector{Float32}
    output_cell_raw::Matrix{Float32}
    output_projection_raw::Array{Float32,3}
end

function CanonicalParameterState(
    core_cell_raw::AbstractMatrix{Float32},
    semantic_projection_raw::AbstractArray{Float32,4},
    event_raw::AbstractVector{Float32},
    output_cell_raw::AbstractMatrix{Float32},
    output_projection_raw::AbstractArray{Float32,3},
)
    return CanonicalParameterState(
        Array(core_cell_raw),
        Array(semantic_projection_raw),
        Vector(event_raw),
        Array(output_cell_raw),
        Array(output_projection_raw),
    )
end

@inline _parameter_arrays(state::CanonicalParameterState) = (
    state.core_cell_raw,
    state.semantic_projection_raw,
    state.event_raw,
    state.output_cell_raw,
    state.output_projection_raw,
)

struct CanonicalOptimizerStateSnapshot
    registry::NTuple{5,ParameterGroupContract}
    parameters::CanonicalParameterState
    first_moments::CanonicalParameterState
    second_moments::CanonicalParameterState
    group_steps::NTuple{5,UInt64}
    total_step::UInt64
end

struct LearningClockSnapshot
    update::Int
    analog_ticks::Int
    hard_event_ticks::Int
    homeostasis_ticks::Int
    structure_ticks::Int
end

struct MechanismCounterSnapshot
    decolle_signal_nonzero::UInt64
    subthreshold_updates::UInt64
    nonspiking_updates::UInt64
    hard_event_control_updates::UInt64
    homeostasis_events::UInt64
    synaptic_scaling_events::UInt64
    utility_updates::UInt64
    rewires::UInt64
end

struct PlasticityStateSnapshot
    firing_rate::Vector{Float32}
    activity_ema::Vector{Float32}
    incoming_conductance_ema::Vector{Float32}
    utility::Vector{Float32}
    reduced_batches::UInt64
    homeostasis_events::UInt64
    synaptic_scaling_events::UInt64
    utility_updates::UInt64
    rewires::UInt64
end

function PlasticityStateSnapshot(
    firing_rate::AbstractVector{Float32},
    activity_ema::AbstractVector{Float32},
    incoming_conductance_ema::AbstractVector{Float32},
    utility::AbstractVector{Float32},
    reduced_batches::UInt64,
    homeostasis_events::UInt64,
    synaptic_scaling_events::UInt64,
    utility_updates::UInt64,
    rewires::UInt64,
)
    return PlasticityStateSnapshot(
        Vector(firing_rate),
        Vector(activity_ema),
        Vector(incoming_conductance_ema),
        Vector(utility),
        reduced_batches,
        homeostasis_events,
        synaptic_scaling_events,
        utility_updates,
        rewires,
    )
end

struct SamplerSourceIdentitySnapshot
    encoding::String
    count::Int
    sha256::String
end

struct SamplerStateSnapshot
    schema::Int
    algorithm::String
    seed::UInt64
    epoch::UInt64
    cursor::Int
    source_identity::SamplerSourceIdentitySnapshot
    source_rows::Vector{Int}
    permutation::Vector{Int}
end

function SamplerStateSnapshot(snapshot::NamedTuple)
    keys(snapshot) == (
        :schema, :algorithm, :seed, :epoch, :cursor, :source_identity,
        :source_rows, :permutation,
    ) || throw(ArgumentError("sampler snapshot fields differ"))
    identity = snapshot.source_identity
    identity isa NamedTuple && keys(identity) == (:encoding, :count, :sha256) ||
        throw(ArgumentError("sampler source identity fields differ"))
    typeof(snapshot.schema) === Int || throw(ArgumentError(
        "sampler snapshot schema type differs",
    ))
    typeof(snapshot.algorithm) === String || throw(ArgumentError(
        "sampler snapshot algorithm type differs",
    ))
    typeof(snapshot.seed) === UInt64 || throw(ArgumentError(
        "sampler snapshot seed type differs",
    ))
    typeof(snapshot.epoch) === UInt64 || throw(ArgumentError(
        "sampler snapshot epoch type differs",
    ))
    typeof(snapshot.cursor) === Int || throw(ArgumentError(
        "sampler snapshot cursor type differs",
    ))
    typeof(identity.encoding) === String || throw(ArgumentError(
        "sampler source encoding type differs",
    ))
    typeof(identity.count) === Int || throw(ArgumentError(
        "sampler source count type differs",
    ))
    typeof(identity.sha256) === String || throw(ArgumentError(
        "sampler source digest type differs",
    ))
    typeof(snapshot.source_rows) === Vector{Int} || throw(ArgumentError(
        "sampler source rows type differs",
    ))
    typeof(snapshot.permutation) === Vector{Int} || throw(ArgumentError(
        "sampler permutation type differs",
    ))
    return SamplerStateSnapshot(
        snapshot.schema,
        snapshot.algorithm,
        snapshot.seed,
        snapshot.epoch,
        snapshot.cursor,
        SamplerSourceIdentitySnapshot(
            identity.encoding, identity.count, identity.sha256,
        ),
        copy(snapshot.source_rows),
        copy(snapshot.permutation),
    )
end

function _sampler_named_tuple(snapshot::SamplerStateSnapshot)
    identity = snapshot.source_identity
    return (;
        schema=snapshot.schema,
        algorithm=snapshot.algorithm,
        seed=snapshot.seed,
        epoch=snapshot.epoch,
        cursor=snapshot.cursor,
        source_identity=(;
            encoding=identity.encoding,
            count=identity.count,
            sha256=identity.sha256,
        ),
        source_rows=copy(snapshot.source_rows),
        permutation=copy(snapshot.permutation),
    )
end

struct CanonicalTrainingStateSnapshot
    magic::UInt64
    schema::UInt32
    format::String
    architecture_fingerprint::String
    input_fingerprint::String
    topology_fingerprint::String
    learning_fingerprint::String
    optimizer_fingerprint::String
    run_contract_fingerprint::String
    run_contract::CanonicalRunContract
    learning_config::Local.LocalLearningConfig
    optimizer_config::Optimizer.AdamWConfig
    optimizer::CanonicalOptimizerStateSnapshot
    learning_clock::LearningClockSnapshot
    cumulative_mechanisms::MechanismCounterSnapshot
    training_updates::UInt64
    plasticity::PlasticityStateSnapshot
    sampler::SamplerStateSnapshot
    state_fingerprint::String
end

struct PreparedTrainingCheckpoint
    snapshot::CanonicalTrainingStateSnapshot
    sampler::Sampler.DeterministicEpochSampler
end

function canonical_architecture_contract(config::Graph.GraphConfig)
    maximum_waves = Graph.Events.CANONICAL_MAX_WAVES
    0 <= config.max_event_waves <= maximum_waves || throw(ArgumentError(
        "graph max_event_waves must be in 0:$maximum_waves",
    ))
    required_tape = _CANONICAL_CORE_COUNT * (1 + config.max_event_waves)
    config.tape_capacity >= required_tape || throw(ArgumentError(
        "graph tape capacity cannot hold all canonical transitions",
    ))
    config.event_overflow in (:error, :fallback) || throw(ArgumentError(
        "graph event overflow policy is unsupported",
    ))
    return (;
        config,
        node_count=_CANONICAL_NODE_COUNT,
        edge_count=_CANONICAL_EDGE_COUNT,
        core_count=_CANONICAL_CORE_COUNT,
        output_dim=_CANONICAL_OUTPUT_DIM,
        event_parameters=_CANONICAL_EVENT_PARAMETER_COUNT,
    )
end

architecture_fingerprint(config::Graph.GraphConfig) =
    _contract_fingerprint("architecture", canonical_architecture_contract(config))

function canonical_input_contract()
    return (;
        board_rows=Input.BOARD_ROWS,
        board_columns=Input.BOARD_COLUMNS,
        placement_capacity=Input.PLACEMENT_CAPACITY,
        next_count=Input.NEXT_COUNT,
        board=(empty=UInt8(Input.EMPTY), occupied=UInt8(Input.OCCUPIED)),
        placement=(absent=UInt8(Input.ABSENT), present=UInt8(Input.PRESENT)),
        pieces=(
            none=UInt8(Input.NONE), i=UInt8(Input.PIECE_I),
            o=UInt8(Input.PIECE_O), t=UInt8(Input.PIECE_T),
            s=UInt8(Input.PIECE_S), z=UInt8(Input.PIECE_Z),
            j=UInt8(Input.PIECE_J), l=UInt8(Input.PIECE_L),
        ),
        truth=(false_value=UInt8(Input.FALSE_VALUE),
            true_value=UInt8(Input.TRUE_VALUE)),
        event=(no_event=UInt8(Input.NO_EVENT),
            present=UInt8(Input.EVENT_PRESENT)),
        site=(
            empty=UInt8(Input.SITE_EMPTY), occupied=UInt8(Input.SITE_OCCUPIED),
            placed=UInt8(Input.SITE_PLACED), outside=UInt8(Input.OUTSIDE),
        ),
        candidate_path=(
            uninitialized=UInt8(Input.UNINITIALIZED),
            no_clear_cow=UInt8(Input.NO_CLEAR_COW),
            clear_slow_path=UInt8(Input.CLEAR_SLOW_PATH),
        ),
        teacher_fields_absent=true,
    )
end

input_fingerprint() = _contract_fingerprint("input", canonical_input_contract())

function _topology_contract(model::Graph.CanonicalModel)
    topology = model.topology
    length(topology.edge_sources) == _CANONICAL_EDGE_COUNT || throw(
        DimensionMismatch("canonical topology edge count differs"),
    )
    parameter_count = Graph.event_parameter_count(model)
    parameter_count == _CANONICAL_EVENT_PARAMETER_COUNT || throw(
        DimensionMismatch("canonical event parameter count differs"),
    )
    kinds = Vector{UInt8}(undef, parameter_count)
    sources = Vector{UInt16}(undef, parameter_count)
    destinations = Vector{UInt16}(undef, parameter_count)
    channels = Vector{UInt8}(undef, parameter_count)
    families = Vector{UInt8}(undef, parameter_count)
    slots = Vector{UInt8}(undef, parameter_count)
    branches = Vector{UInt8}(undef, parameter_count)
    lanes = Vector{UInt8}(undef, parameter_count)
    @inbounds for index in 1:parameter_count
        descriptor = Graph.event_parameter_descriptor(model, index)
        kinds[index] = UInt8(descriptor.kind)
        sources[index] = descriptor.source
        destinations[index] = descriptor.destination
        channels[index] = descriptor.channel
        families[index] = descriptor.family
        slots[index] = descriptor.slot
        branches[index] = descriptor.branch
        lanes[index] = descriptor.lane
    end
    return (;
        topology,
        event_parameter_order=(;
            kinds,
            sources,
            destinations,
            channels,
            families,
            slots,
            branches,
            lanes,
        ),
    )
end

topology_fingerprint(model::Graph.CanonicalModel) =
    _contract_fingerprint("topology", _topology_contract(model))

function canonical_learning_contract(config::Local.LocalLearningConfig)
    config.plasticity.structure_enabled && throw(ArgumentError(
        "schema2 fixed-spine resume requires structure_enabled=false",
    ))
    return (;
        config,
        fixed_local_signal_map=(
            output_dim=_CANONICAL_OUTPUT_DIM,
            observation_dim=47,
            packet_dim=12,
            cell_count=_CANONICAL_CORE_COUNT,
        ),
    )
end

learning_fingerprint(config::Local.LocalLearningConfig) =
    _contract_fingerprint("learning", canonical_learning_contract(config))

optimizer_fingerprint(
    config::Optimizer.AdamWConfig,
    registry::NTuple{5,ParameterGroupContract},
) = _contract_fingerprint("optimizer", (; config, registry))

function _validate_group_contracts(
    registry::NTuple{5,ParameterGroupContract},
)
    map(group -> group.name, registry) == CANONICAL_PARAMETER_GROUPS || throw(
        ArgumentError("parameter groups are missing, extra, or reordered"),
    )
    expected_dimensions = (
        (_CANONICAL_CELL_PARAMETER_DIM, _CANONICAL_CORE_COUNT),
        (4, 3, 8, 2),
        (_CANONICAL_EVENT_PARAMETER_COUNT,),
        (_CANONICAL_CELL_PARAMETER_DIM, _CANONICAL_OUTPUT_DIM),
        (4, 3, 5),
    )
    @inbounds for index in 1:5
        registry[index].dimensions == expected_dimensions[index] || throw(
            DimensionMismatch(
                "parameter-group $(registry[index].name) shape differs",
            ),
        )
    end
    return registry
end

function _validate_parameter_state(
    state::CanonicalParameterState,
    registry::NTuple{5,ParameterGroupContract};
    label::String,
    second_moment::Bool=false,
    enforce_bounds::Bool=false,
)
    arrays = _parameter_arrays(state)
    @inbounds for index in 1:5
        array = arrays[index]
        Tuple(size(array)) == registry[index].dimensions || throw(
            DimensionMismatch("$label shape differs for $(registry[index].name)"),
        )
        lower = _raw_lower(registry[index])
        upper = _raw_upper(registry[index])
        for value in array
            isfinite(value) || throw(DomainError(value, "$label is non-finite"))
            second_moment && value < 0.0f0 && throw(DomainError(
                value, "$label contains a negative value",
            ))
            enforce_bounds && !(lower <= value <= upper) && throw(DomainError(
                value, "$label is outside the registered raw bounds",
            ))
        end
        if _multiplier(registry[index]) == 0.0f0 && label != "parameters"
            all(iszero, array) || throw(ArgumentError(
                "frozen group $(registry[index].name) has nonzero $label",
            ))
        end
    end
    return state
end

@inline function _due_union_count(updates::UInt64, left::Int, right::Int)
    l = UInt128(left)
    r = UInt128(right)
    greatest = UInt128(gcd(left, right))
    least = (l ÷ greatest) * r
    total = UInt128(updates)
    value = total ÷ l + total ÷ r - total ÷ least
    value <= UInt128(typemax(UInt64)) || throw(OverflowError(
        "optimizer due-union count overflow",
    ))
    return UInt64(value)
end

function _expected_group_steps(
    updates::UInt64,
    config::Local.LocalLearningConfig,
    registry::NTuple{5,ParameterGroupContract},
)
    schedule = config.schedule
    analog = config.analog_multiplier > 0.0f0 ?
        updates ÷ UInt64(schedule.analog_interval) : UInt64(0)
    hard = config.hard_event_multiplier > 0.0f0 ?
        updates ÷ UInt64(schedule.hard_event_interval) : UInt64(0)
    recurrent = if iszero(analog)
        hard
    elseif iszero(hard)
        analog
    else
        _due_union_count(
            updates, schedule.analog_interval, schedule.hard_event_interval,
        )
    end
    # Every recurrent group can receive the hard-event lane, so their group
    # clocks advance on the union of analog and hard-event due updates.
    raw = (recurrent, recurrent, recurrent, updates, updates)
    return ntuple(5) do index
        _multiplier(registry[index]) > 0.0f0 ? raw[index] : UInt64(0)
    end
end

function _validate_clock(
    clock::LearningClockSnapshot,
    config::Local.LocalLearningConfig,
    updates::UInt64,
)
    updates <= UInt64(typemax(Int)) || throw(ArgumentError(
        "training update count does not fit LearningClock Int",
    ))
    expected_update = Int(updates)
    clock.update == expected_update || throw(ArgumentError(
        "learning clock update differs from training updates",
    ))
    schedule = config.schedule
    actual = (
        clock.analog_ticks,
        clock.hard_event_ticks,
        clock.homeostasis_ticks,
        clock.structure_ticks,
    )
    expected = (
        fld(expected_update, schedule.analog_interval),
        fld(expected_update, schedule.hard_event_interval),
        fld(expected_update, schedule.homeostasis_interval),
        fld(expected_update, schedule.structure_interval),
    )
    actual == expected || throw(ArgumentError(
        "learning clock ticks differ from the configured schedule",
    ))
    return clock
end

function _validate_plasticity(
    state::PlasticityStateSnapshot,
    clock::LearningClockSnapshot,
    config::Local.LocalLearningConfig,
    updates::UInt64,
)
    length(state.firing_rate) == _CANONICAL_NODE_COUNT || throw(
        DimensionMismatch("plasticity firing-rate shape differs"),
    )
    length(state.activity_ema) == _CANONICAL_NODE_COUNT || throw(
        DimensionMismatch("plasticity activity EMA shape differs"),
    )
    length(state.incoming_conductance_ema) == _CANONICAL_NODE_COUNT || throw(
        DimensionMismatch("plasticity conductance EMA shape differs"),
    )
    length(state.utility) == _CANONICAL_EVENT_CONTACT_COUNT || throw(
        DimensionMismatch("plasticity utility shape differs"),
    )
    @inbounds for value in state.firing_rate
        isfinite(value) && 0.0f0 <= value <= 1.0f0 || throw(DomainError(
            value, "plasticity firing rate is outside [0,1]",
        ))
    end
    for (label, values) in (
        ("activity EMA", state.activity_ema),
        ("incoming-conductance EMA", state.incoming_conductance_ema),
        ("utility", state.utility),
    )
        @inbounds for value in values
            isfinite(value) && value >= 0.0f0 || throw(DomainError(
                value, "plasticity $label is invalid",
            ))
        end
    end
    state.reduced_batches == updates || throw(ArgumentError(
        "plasticity reduced_batches differs from training updates",
    ))
    analog_active = config.analog_multiplier > 0.0f0 &&
        config.utility_mode === :combined
    utility_bound = analog_active ?
        UInt128(clock.analog_ticks) * UInt128(_CANONICAL_EVENT_CONTACT_COUNT) :
        UInt128(0)
    UInt128(state.utility_updates) <= utility_bound || throw(ArgumentError(
        "plasticity utility counter exceeds its scheduled bound",
    ))
    UInt128(state.homeostasis_events) <=
        UInt128(clock.homeostasis_ticks) * UInt128(_CANONICAL_NODE_COUNT) ||
        throw(ArgumentError("homeostasis counter exceeds its scheduled bound"))
    UInt128(state.synaptic_scaling_events) <=
        UInt128(clock.homeostasis_ticks) *
            UInt128(_CANONICAL_EVENT_CONTACT_COUNT) || throw(ArgumentError(
                "synaptic-scaling counter exceeds its scheduled bound",
            ))
    !config.plasticity.structure_enabled || throw(ArgumentError(
        "schema2 does not serialize mutable rewired topology",
    ))
    state.rewires == 0 || throw(ArgumentError(
        "fixed-spine schema2 requires rewires == 0",
    ))
    return state
end

function _validate_sampler(
    snapshot::SamplerStateSnapshot,
    contract::CanonicalRunContract,
    updates::UInt64,
)
    snapshot.seed == contract.sampler_seed || throw(ArgumentError(
        "sampler seed differs from the run contract",
    ))
    snapshot.source_identity.count == contract.training_state_count || throw(
        ArgumentError("sampler source count differs from DatasetIdentity"),
    )
    restored = Sampler.restore_sampler(
        snapshot.source_rows, _sampler_named_tuple(snapshot),
    )
    consumed = Sampler.sampler_consumed_rows(restored)
    expected = UInt128(updates) * UInt128(contract.state_batch)
    consumed == expected || throw(ArgumentError(
        "sampler position differs from completed training updates",
    ))
    return restored
end

function _state_fingerprint(snapshot::CanonicalTrainingStateSnapshot)
    return _contract_fingerprint("training-state", (;
        magic=snapshot.magic,
        schema=snapshot.schema,
        format=snapshot.format,
        architecture_fingerprint=snapshot.architecture_fingerprint,
        input_fingerprint=snapshot.input_fingerprint,
        topology_fingerprint=snapshot.topology_fingerprint,
        learning_fingerprint=snapshot.learning_fingerprint,
        optimizer_fingerprint=snapshot.optimizer_fingerprint,
        run_contract_fingerprint=snapshot.run_contract_fingerprint,
        run_contract=snapshot.run_contract,
        learning_config=snapshot.learning_config,
        optimizer_config=snapshot.optimizer_config,
        optimizer=snapshot.optimizer,
        learning_clock=snapshot.learning_clock,
        cumulative_mechanisms=snapshot.cumulative_mechanisms,
        training_updates=snapshot.training_updates,
        plasticity=snapshot.plasticity,
        sampler=snapshot.sampler,
    ))
end

function _validate_snapshot(snapshot::CanonicalTrainingStateSnapshot)
    snapshot.magic == CHECKPOINT_MAGIC || throw(ArgumentError(
        "checkpoint magic is not canonical",
    ))
    snapshot.schema == CHECKPOINT_SCHEMA || throw(ArgumentError(
        "checkpoint schema is unsupported; schema1 has no compatibility path",
    ))
    snapshot.format == CHECKPOINT_FORMAT || throw(ArgumentError(
        "checkpoint format is unsupported",
    ))
    for (label, value) in (
        ("architecture", snapshot.architecture_fingerprint),
        ("input", snapshot.input_fingerprint),
        ("topology", snapshot.topology_fingerprint),
        ("learning", snapshot.learning_fingerprint),
        ("optimizer", snapshot.optimizer_fingerprint),
        ("run contract", snapshot.run_contract_fingerprint),
        ("state", snapshot.state_fingerprint),
    )
        _require_sha256(value, "$label fingerprint")
    end
    _validate_run_contract(snapshot.run_contract)
    run_contract_fingerprint(snapshot.run_contract) ==
        snapshot.run_contract_fingerprint || throw(ArgumentError(
            "run-contract fingerprint is false",
        ))
    learning_fingerprint(snapshot.learning_config) ==
        snapshot.learning_fingerprint || throw(ArgumentError(
            "learning fingerprint is false",
        ))
    registry = _validate_group_contracts(snapshot.optimizer.registry)
    optimizer_fingerprint(snapshot.optimizer_config, registry) ==
        snapshot.optimizer_fingerprint || throw(ArgumentError(
            "optimizer fingerprint is false",
        ))
    _validate_parameter_state(
        snapshot.optimizer.parameters, registry;
        label="parameters", enforce_bounds=true,
    )
    _validate_parameter_state(
        snapshot.optimizer.first_moments, registry;
        label="first moments",
    )
    _validate_parameter_state(
        snapshot.optimizer.second_moments, registry;
        label="second moments", second_moment=true,
    )
    snapshot.optimizer.total_step == snapshot.training_updates || throw(
        ArgumentError("optimizer total step differs from training updates"),
    )
    expected_steps = _expected_group_steps(
        snapshot.training_updates, snapshot.learning_config, registry,
    )
    snapshot.optimizer.group_steps == expected_steps || throw(ArgumentError(
        "optimizer group steps differ from the exact due schedule",
    ))
    clock = _validate_clock(
        snapshot.learning_clock,
        snapshot.learning_config,
        snapshot.training_updates,
    )
    _validate_plasticity(
        snapshot.plasticity,
        clock,
        snapshot.learning_config,
        snapshot.training_updates,
    )
    snapshot.cumulative_mechanisms.rewires == 0 || throw(ArgumentError(
        "fixed-spine cumulative mechanism rewires must be zero",
    ))
    snapshot.cumulative_mechanisms.homeostasis_events ==
        snapshot.plasticity.homeostasis_events || throw(ArgumentError(
            "cumulative homeostasis telemetry differs from plasticity state",
        ))
    snapshot.cumulative_mechanisms.synaptic_scaling_events ==
        snapshot.plasticity.synaptic_scaling_events || throw(ArgumentError(
            "cumulative synaptic-scaling telemetry differs from plasticity state",
        ))
    snapshot.cumulative_mechanisms.utility_updates ==
        snapshot.plasticity.utility_updates || throw(ArgumentError(
            "cumulative utility telemetry differs from plasticity state",
        ))
    snapshot.cumulative_mechanisms.rewires == snapshot.plasticity.rewires || throw(
        ArgumentError("cumulative rewire telemetry differs from plasticity state"),
    )
    _validate_sampler(
        snapshot.sampler, snapshot.run_contract, snapshot.training_updates,
    )
    _state_fingerprint(snapshot) == snapshot.state_fingerprint || throw(
        ArgumentError("checkpoint state fingerprint is false"),
    )
    return snapshot
end

function build_training_snapshot(
    model::Graph.CanonicalModel,
    learning_config::Local.LocalLearningConfig,
    optimizer_config::Optimizer.AdamWConfig,
    optimizer::CanonicalOptimizerStateSnapshot,
    learning_clock::LearningClockSnapshot,
    cumulative_mechanisms::MechanismCounterSnapshot,
    training_updates::UInt64,
    plasticity::PlasticityStateSnapshot,
    sampler::SamplerStateSnapshot,
    run_contract::CanonicalRunContract,
)
    architecture = architecture_fingerprint(model.config)
    input = input_fingerprint()
    topology = topology_fingerprint(model)
    learning = learning_fingerprint(learning_config)
    optimizer_fp = optimizer_fingerprint(optimizer_config, optimizer.registry)
    architecture == run_contract.architecture_fingerprint || throw(
        ArgumentError("run-contract architecture fingerprint differs"),
    )
    topology == run_contract.topology_fingerprint || throw(ArgumentError(
        "run-contract topology fingerprint differs",
    ))
    model_parameters = Graph.parameter_components(model.parameters)
    live = (
        model_parameters.core_cell_raw,
        model_parameters.semantic_projection_raw,
        model_parameters.event_raw,
        model_parameters.output_cell_raw,
        model_parameters.output_projection_raw,
    )
    saved = _parameter_arrays(optimizer.parameters)
    @inbounds for index in 1:5
        live[index] == saved[index] || throw(ArgumentError(
            "optimizer parameter snapshot differs from the live model",
        ))
    end
    provisional = CanonicalTrainingStateSnapshot(
        CHECKPOINT_MAGIC,
        CHECKPOINT_SCHEMA,
        CHECKPOINT_FORMAT,
        architecture,
        input,
        topology,
        learning,
        optimizer_fp,
        run_contract_fingerprint(run_contract),
        run_contract,
        learning_config,
        optimizer_config,
        optimizer,
        learning_clock,
        cumulative_mechanisms,
        training_updates,
        plasticity,
        sampler,
        repeat("0", 64),
    )
    snapshot = CanonicalTrainingStateSnapshot(
        provisional.magic,
        provisional.schema,
        provisional.format,
        provisional.architecture_fingerprint,
        provisional.input_fingerprint,
        provisional.topology_fingerprint,
        provisional.learning_fingerprint,
        provisional.optimizer_fingerprint,
        provisional.run_contract_fingerprint,
        provisional.run_contract,
        provisional.learning_config,
        provisional.optimizer_config,
        provisional.optimizer,
        provisional.learning_clock,
        provisional.cumulative_mechanisms,
        provisional.training_updates,
        provisional.plasticity,
        provisional.sampler,
        _state_fingerprint(provisional),
    )
    return _validate_snapshot(snapshot)
end

"""Atomically serialize schema2. There is no schema1/generic overload."""
function save_checkpoint(
    path::AbstractString,
    snapshot::CanonicalTrainingStateSnapshot,
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

"""Load schema2 only. NamedTuple/schema1/relation checkpoints fail closed."""
function load_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("checkpoint does not exist: $source"))
    value = try
        open(deserialize, source)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError(
            "checkpoint cannot be decoded as canonical schema2: " *
            sprint(showerror, error),
        ))
    end
    value isa CanonicalTrainingStateSnapshot || throw(ArgumentError(
        "legacy/generic checkpoint payload is unsupported",
    ))
    return _validate_snapshot(value)
end

function prepare_training_checkpoint(
    snapshot::CanonicalTrainingStateSnapshot,
    model::Graph.CanonicalModel,
    expected_registry::NTuple{5,ParameterGroupContract},
    learning_config::Local.LocalLearningConfig,
    optimizer_config::Optimizer.AdamWConfig,
    run_contract::CanonicalRunContract,
    source_rows::AbstractVector{<:Integer},
)
    copied = deepcopy(snapshot)
    _validate_snapshot(copied)
    copied.architecture_fingerprint == architecture_fingerprint(model.config) ||
        throw(ArgumentError("checkpoint architecture differs"))
    copied.input_fingerprint == input_fingerprint() || throw(ArgumentError(
        "checkpoint input ABI differs",
    ))
    copied.topology_fingerprint == topology_fingerprint(model) || throw(
        ArgumentError("checkpoint topology/order differs"),
    )
    copied.learning_fingerprint == learning_fingerprint(learning_config) ||
        throw(ArgumentError("checkpoint learning configuration differs"))
    _validate_group_contracts(expected_registry)
    copied.optimizer.registry == expected_registry || throw(ArgumentError(
        "checkpoint parameter registry differs",
    ))
    copied.optimizer_fingerprint ==
        optimizer_fingerprint(optimizer_config, expected_registry) || throw(
            ArgumentError("checkpoint optimizer configuration differs"),
        )
    isequal(copied.run_contract, _validate_run_contract(run_contract)) || throw(
        ArgumentError("checkpoint run contract differs"),
    )
    restored_sampler = Sampler.restore_sampler(
        source_rows, _sampler_named_tuple(copied.sampler),
    )
    Sampler.sampler_consumed_rows(restored_sampler) ==
        UInt128(copied.training_updates) * UInt128(run_contract.state_batch) ||
        throw(ArgumentError("restored sampler progress differs"))
    return PreparedTrainingCheckpoint(copied, restored_sampler)
end

function prepare_training_checkpoint(
    path::AbstractString,
    model::Graph.CanonicalModel,
    expected_registry::NTuple{5,ParameterGroupContract},
    learning_config::Local.LocalLearningConfig,
    optimizer_config::Optimizer.AdamWConfig,
    run_contract::CanonicalRunContract,
    source_rows::AbstractVector{<:Integer},
)
    return prepare_training_checkpoint(
        load_checkpoint(path),
        model,
        expected_registry,
        learning_config,
        optimizer_config,
        run_contract,
        source_rows,
    )
end

validate_prepared_checkpoint(prepared::PreparedTrainingCheckpoint) =
    _validate_snapshot(prepared.snapshot)

end # module CanonicalCheckpoint
