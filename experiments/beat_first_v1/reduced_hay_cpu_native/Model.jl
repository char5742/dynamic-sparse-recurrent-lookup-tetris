module ReducedHayCPUNativeModel

using Random
using SHA

using ..Architecture
using ..ActiveApicalCell
using ..StateCodec
using ..Float32NumericCore
using ..Payload
using ..ReducedHayCPUNativeEventGraph
using ..Topology
using ..SensoryEncoder
using ..OutputCellBank

export BLOCKS,
    CELLS_PER_BLOCK,
    STATE_DIM,
    RECURRENT_STEPS,
    FANOUT,
    INPUT_RAILS,
    OUTPUT_DIM,
    CANONICAL_PARAMETER_COUNT,
    Parameters,
    CPUHayModel,
    ModelCache,
    PreparedModelState,
    PreparedSnapshot,
    ForwardBuffers,
    ForwardScratch,
    AbstractForwardDiagnostics,
    NoForwardDiagnostics,
    FullForwardDiagnostics,
    ReferenceForward,
    build_model,
    initialize_parameters,
    copy_parameters!,
    parameter_count,
    prepared_generation,
    prepared_snapshot,
    assert_generation,
    publish!,
    recurrent_off_output,
    validate_parameters,
    forward_candidate!,
    forward_reference,
    forward_reference

const Cell = ActiveApicalCell
const Codec = StateCodec
const Numeric = Float32NumericCore
const Graph = ReducedHayCPUNativeEventGraph
const NetworkTopology = Topology
const Encoder = SensoryEncoder
const OutputBank = OutputCellBank
const EventPayload = Payload

const BLOCKS = Architecture.BLOCK_COUNT
const CELLS_PER_BLOCK = Architecture.CELLS_PER_BLOCK
const TOTAL_CELLS = Architecture.TOTAL_CELLS
const STATE_DIM = Cell.STATE_DIM
const RECURRENT_STEPS = Architecture.CYCLES
const FANOUT = Architecture.FANOUT
const INPUT_RAILS = Architecture.RAIL_COUNT
const OUTPUT_DIM = Architecture.OUTPUT_COUNT

const CANONICAL_PARAMETER_COUNT =
    Cell.PARAM_DIM * CELLS_PER_BLOCK * BLOCKS +
    2 * INPUT_RAILS +
    FANOUT * CELLS_PER_BLOCK * BLOCKS +
    EventPayload.ANALOG_GAIN_COUNT +
    Cell.PARAM_DIM * OutputBank.OUTPUT_CELLS +
    OutputBank.OUTPUT_FANOUT * OutputBank.SOURCE_CELLS +
    OutputBank.Q_OUTPUT_CELLS +
    OutputBank.RECURRENT_STEPS * OutputBank.AUX_OUTPUT_CELLS +
    (OutputBank.OUTPUT_CHANNELS - 1)

const DEFAULT_MODEL_SEED = UInt64(0x6a09e667f3bcc909)
const FROZEN_NUMERIC_CORE_PATH = joinpath(
    @__DIR__,
    "results",
    "float32_numeric_core.bin",
)
const FROZEN_NUMERIC_CORE_SHA256 =
    "d189640c010b2b298fff0d1cf1d363378d307f96b7f09ce3b5692c19ed41a1df"

@inline _global_cell(block::Int, cell::Int) =
    (block - 1) * CELLS_PER_BLOCK + cell

"""The nine trainable parameter groups in their sole canonical layout."""
struct Parameters
    cell_raw::Array{Float32,3}
    sensory_gain_raw::Matrix{Float32}
    edge_strength_raw::Array{Float32,3}
    payload_gain_raw::Vector{Float32}
    output_cell_raw::Matrix{Float32}
    output_edge_raw::Matrix{Float32}
    output_q_basal_bias_raw::Vector{Float32}
    output_gain::Matrix{Float32}
    output_bias::Vector{Float32}
end

"""The fixed-slot/fanout recurrent graph and sparse output-cell bank."""
struct CPUHayModel{T<:AbstractFloat}
    graph::Graph.EventGraph
    output_topology::OutputBank.OutputTopology
    numeric_core::Numeric.BitSerialMachine
end

"""
Transformed quantities refreshed once after a parameter update.

Cell dynamics never transform their 46 raw coordinates inside a candidate
loop.  Edge polarity remains in `model.graph`; only the smooth positive edge
strength is cached here.
"""
struct ModelCache{T<:AbstractFloat}
    cell::Matrix{Cell.CellParameterCache{T}}
    cell_derivative::Matrix{Cell.CellParameterDerivativeCache{T}}
    sensory_gain::Matrix{T}
    sensory_gain_derivative::Matrix{T}
    edge_strength::Vector{T}
    edge_strength_derivative::Vector{T}
    payload_gain::Vector{T}
    payload_gain_derivative::Vector{T}
    output::OutputBank.OutputCache
end

mutable struct PreparedSlot{T<:AbstractFloat}
    parameters::Parameters
    cache::ModelCache{T}
    generation::UInt64
end

"""
Double-buffered published model state.

Only the inactive slot is written during `publish!`; the active raw snapshot
and every transformed cache remain immutable for the whole forward/replay/
reverse generation.
"""
mutable struct PreparedModelState{T<:AbstractFloat}
    slots::NTuple{2,PreparedSlot{T}}
    active_index::UInt8
    generation::UInt64
end

"""One generation-pinned raw/cache pair for an exact reverse pass."""
struct PreparedSnapshot{T<:AbstractFloat}
    parameters::Parameters
    cache::ModelCache{T}
    generation::UInt64
end

"""Caller-owned outputs retained for one forward candidate."""
struct ForwardBuffers{T<:AbstractFloat}
    physical_anchor::Array{T,3}
    physical_recurrent::Array{T,4}
    raw_output::Vector{T}
    output_trajectory::OutputBank.OutputTrajectory
end

function ForwardBuffers(::Type{T}=Float32) where {T<:AbstractFloat}
    return ForwardBuffers{T}(
        Array{T}(undef, STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        Array{T}(
            undef,
            STATE_DIM,
            CELLS_PER_BLOCK,
            BLOCKS,
            RECURRENT_STEPS,
        ),
        Vector{T}(undef, OUTPUT_DIM),
        OutputBank.OutputTrajectory(),
    )
end

"""Dispatch boundary between a storage-free training path and full audits."""
abstract type AbstractForwardDiagnostics end

"""Zero-field diagnostics sink used by the hot candidate path."""
struct NoForwardDiagnostics <: AbstractForwardDiagnostics end

"""Owning diagnostics sink used by the allocating mathematical references."""
mutable struct FullForwardDiagnostics{T<:AbstractFloat} <:
               AbstractForwardDiagnostics
    payload_amplitude::Array{T,3}
    payload_analog::Array{T,3}
    payload_source::Array{Int16,3}
    payload_analog_source::Array{Int16,3}
    pending_before_delivery::Vector{T}
    pending_after_delivery::Vector{T}
    pending_conductance_sum::T
end

function FullForwardDiagnostics(::Type{T}=Float32) where {T<:AbstractFloat}
    return FullForwardDiagnostics{T}(
        zeros(T, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        zeros(T, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        zeros(Int16, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        zeros(Int16, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        zeros(T, RECURRENT_STEPS),
        zeros(T, RECURRENT_STEPS),
        zero(T),
    )
end

"""
Reusable worker-local state for [`forward_candidate!`](@ref).

`recurrent_inputs` is the exact `[Cell.INPUT_DIM,cell,block,cycle]` input tape
consumed by every cell update. It is overwritten on every call and retained
for the exact-oracle adjoint.
The high-dimensional output-cell bank owns its own worker-local scratch; no
global dense observer or candidate-sized hidden cache exists.
"""
struct ForwardScratch{T<:AbstractFloat}
    sensory_input::Array{T,3}
    encoded_anchor::Array{T,3}
    current_physical::Array{T,3}
    current_encoded::Array{T,3}
    soma_threshold::Matrix{T}
    encoded_recurrent::Array{T,4}
    recurrent_inputs::Array{T,4}
    pending::Graph.ConductanceInbox{T}
    ring::Graph.DelayedPayloadRing{T}
    payload_channels::Vector{T}
    initial_state::Vector{T}
    current_sources::Vector{Int}
    previous_sources::Vector{Int}
    active_sources::Vector{Int}
    output::OutputBank.OutputScratch
end

function ForwardScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return ForwardScratch{T}(
        zeros(T, Cell.INPUT_DIM, CELLS_PER_BLOCK, BLOCKS),
        zeros(T, STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        zeros(T, STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        zeros(T, STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        zeros(T, CELLS_PER_BLOCK, BLOCKS),
        zeros(
            T,
            STATE_DIM,
            CELLS_PER_BLOCK,
            BLOCKS,
            RECURRENT_STEPS,
        ),
        zeros(
            T,
            Cell.INPUT_DIM,
            CELLS_PER_BLOCK,
            BLOCKS,
            RECURRENT_STEPS,
        ),
        Graph.ConductanceInbox(TOTAL_CELLS, T),
        Graph.DelayedPayloadRing(TOTAL_CELLS, T),
        zeros(T, EventPayload.PAYLOAD_DIM),
        zeros(T, STATE_DIM),
        Vector{Int}(undef, BLOCKS * CELLS_PER_BLOCK),
        Vector{Int}(undef, BLOCKS * CELLS_PER_BLOCK),
        Vector{Int}(undef, 2 * BLOCKS * CELLS_PER_BLOCK),
        OutputBank.OutputScratch(),
    )
end

"""
Owning result of the allocating mathematical reference.

`anchor` is the immutable all-block sensory snapshot and `recurrent` stores
every block's post-update state at each cycle.
"""
struct ReferenceForward{T<:AbstractFloat}
    physical_anchor::Array{T,3}
    encoded_anchor::Array{T,3}
    physical_recurrent::Array{T,4}
    encoded_recurrent::Array{T,4}
    raw_output::Vector{T}
    payload_amplitude::Array{T,3}
    payload_analog::Array{T,3}
    payload_source::Array{Int16,3}
    payload_analog_source::Array{Int16,3}
    pending_before_delivery::Vector{T}
    pending_after_delivery::Vector{T}
    pending_conductance_sum::T
end

@inline function _require_shape(value, shape, label::AbstractString)
    size(value) == shape || throw(
        DimensionMismatch("$label must have shape $shape"),
    )
    return nothing
end

@inline function _validate_parameter_array(value, label::AbstractString)
    @inbounds for index in eachindex(value)
        isfinite(value[index]) || throw(ArgumentError(
            "$label contains a non-finite value",
        ))
    end
    return nothing
end

@inline function _parameter_shapes_valid(parameters::Parameters)
    return size(parameters.cell_raw) ==
               (Cell.PARAM_DIM, CELLS_PER_BLOCK, BLOCKS) &&
           size(parameters.sensory_gain_raw) == (2, INPUT_RAILS) &&
           size(parameters.edge_strength_raw) ==
               (FANOUT, CELLS_PER_BLOCK, BLOCKS) &&
           size(parameters.payload_gain_raw) ==
               (EventPayload.ANALOG_GAIN_COUNT,) &&
           size(parameters.output_cell_raw) ==
               (Cell.PARAM_DIM, OutputBank.OUTPUT_CELLS) &&
           size(parameters.output_edge_raw) ==
               (OutputBank.OUTPUT_FANOUT, OutputBank.SOURCE_CELLS) &&
            size(parameters.output_q_basal_bias_raw) ==
               (OutputBank.Q_OUTPUT_CELLS,) &&
           size(parameters.output_gain) ==
               (OutputBank.RECURRENT_STEPS, OutputBank.AUX_OUTPUT_CELLS) &&
           size(parameters.output_bias) == (OUTPUT_DIM - 1,)
end

function validate_parameters(parameters::Parameters)
    _parameter_shapes_valid(parameters) || throw(DimensionMismatch(
        "one or more parameter groups have a noncanonical shape",
    ))

    _validate_parameter_array(parameters.cell_raw, "cell_raw")
    _validate_parameter_array(parameters.sensory_gain_raw, "sensory_gain_raw")
    _validate_parameter_array(parameters.edge_strength_raw, "edge_strength_raw")
    _validate_parameter_array(parameters.payload_gain_raw, "payload_gain_raw")
    _validate_parameter_array(parameters.output_cell_raw, "output_cell_raw")
    _validate_parameter_array(parameters.output_edge_raw, "output_edge_raw")
    _validate_parameter_array(
        parameters.output_q_basal_bias_raw,
        "output_q_basal_bias_raw",
    )
    _validate_parameter_array(parameters.output_gain, "output_gain")
    _validate_parameter_array(parameters.output_bias, "output_bias")
    @inbounds for output in 2:OutputBank.Q_OUTPUT_CELLS
        for cell_parameter in axes(parameters.output_cell_raw, 1)
            parameters.output_cell_raw[cell_parameter, output] ==
                parameters.output_cell_raw[cell_parameter, 1] || throw(
                    ArgumentError(
                        "numeric Q cells must share one exact internal parameter vector",
                    ),
                )
        end
    end
    return parameters
end

function parameter_count(parameters::Parameters)
    count = 0
    @inbounds for index in 1:fieldcount(Parameters)
        count += length(getfield(parameters, index))
    end
    return count
end

function Base.copy(parameters::Parameters)
    return Parameters(
        copy(parameters.cell_raw),
        copy(parameters.sensory_gain_raw),
        copy(parameters.edge_strength_raw),
        copy(parameters.payload_gain_raw),
        copy(parameters.output_cell_raw),
        copy(parameters.output_edge_raw),
        copy(parameters.output_q_basal_bias_raw),
        copy(parameters.output_gain),
        copy(parameters.output_bias),
    )
end

function copy_parameters!(destination::Parameters, source::Parameters)
    _parameter_shapes_valid(destination) || throw(DimensionMismatch(
        "destination parameter storage is malformed",
    ))
    _parameter_shapes_valid(source) || throw(DimensionMismatch(
        "source parameter storage is malformed",
    ))
    copyto!(destination.cell_raw, source.cell_raw)
    copyto!(destination.sensory_gain_raw, source.sensory_gain_raw)
    copyto!(destination.edge_strength_raw, source.edge_strength_raw)
    copyto!(destination.payload_gain_raw, source.payload_gain_raw)
    copyto!(destination.output_cell_raw, source.output_cell_raw)
    copyto!(destination.output_edge_raw, source.output_edge_raw)
    copyto!(
        destination.output_q_basal_bias_raw,
        source.output_q_basal_bias_raw,
    )
    copyto!(destination.output_gain, source.output_gain)
    copyto!(destination.output_bias, source.output_bias)
    return destination
end

function initialize_parameters(
    seed::Integer=DEFAULT_MODEL_SEED,
    ;
    numeric_cell_raw::AbstractVector{Float32}=
        Cell.default_raw_parameters(Float32),
)
    seed >= 0 || throw(ArgumentError("model seed must be nonnegative"))
    rng = MersenneTwister(seed)

    base_cell = Cell.default_raw_parameters(Float32)
    cell_raw = Array{Float32}(undef, Cell.PARAM_DIM, CELLS_PER_BLOCK, BLOCKS)
    @inbounds for block in 1:BLOCKS
        for cell in 1:CELLS_PER_BLOCK
            for parameter in 1:Cell.PARAM_DIM
                cell_raw[parameter, cell, block] =
                    base_cell[parameter] + 0.01f0 * randn(rng, Float32)
            end
        end
    end

    sensory_gain_raw = Encoder.default_raw_gains(Float32)
    edge_strength_raw = NetworkTopology.initialize_edge_strength_raw(Float32)
    # The hard-sigmoid payload transform maps 1.2 to a trainable 0.8 gain.
    # This keeps continuous event identity above numerical noise without
    # initializing at the transform's zero-derivative upper corner.
    payload_gain_raw = fill(1.2f0, EventPayload.ANALOG_GAIN_COUNT)

    output = OutputBank.initialize_parameters(
        rng;
        numeric_cell_raw,
    )
    parameters = Parameters(
        cell_raw,
        sensory_gain_raw,
        edge_strength_raw,
        payload_gain_raw,
        output.cell_raw,
        output.edge_raw,
        output.q_basal_bias_raw,
        output.gain,
        output.bias,
    )
    validate_parameters(parameters)
    parameter_count(parameters) == CANONICAL_PARAMETER_COUNT || error(
        "canonical parameter-count derivation is inconsistent",
    )
    return parameters
end

function _allocate_cache(parameters::Parameters)
    first_raw = @view parameters.cell_raw[:, 1, 1]
    first_cell = Cell.transform_parameters(first_raw)
    first_derivative = Cell.transform_parameter_derivatives(first_raw)
    cell = Matrix{typeof(first_cell)}(undef, CELLS_PER_BLOCK, BLOCKS)
    cell_derivative = Matrix{typeof(first_derivative)}(
        undef,
        CELLS_PER_BLOCK,
        BLOCKS,
    )
    return ModelCache(
        cell,
        cell_derivative,
        zeros(Float32, 2, INPUT_RAILS),
        zeros(Float32, 2, INPUT_RAILS),
        NetworkTopology.initialize_edge_strength_cache(Float32),
        NetworkTopology.initialize_edge_strength_cache(Float32),
        zeros(Float32, EventPayload.ANALOG_GAIN_COUNT),
        zeros(Float32, EventPayload.ANALOG_GAIN_COUNT),
        OutputBank.OutputCache(_output_parameters(parameters)),
    )
end

function _refresh_cache_unchecked!(
    cache::ModelCache{Float32},
    parameters::Parameters,
)
    @inbounds for block in 1:BLOCKS
        for cell in 1:CELLS_PER_BLOCK
            raw = @view parameters.cell_raw[:, cell, block]
            cache.cell[cell, block] = Cell.transform_parameters(raw)
            cache.cell_derivative[cell, block] =
                Cell.transform_parameter_derivatives(raw)
        end
    end
    Encoder.transform_sensory_gains!(
        cache.sensory_gain,
        cache.sensory_gain_derivative,
        parameters.sensory_gain_raw,
    )
    NetworkTopology.transform_edge_strengths!(
        cache.edge_strength,
        cache.edge_strength_derivative,
        parameters.edge_strength_raw,
    )
    EventPayload.transform_payload_gains!(
        cache.payload_gain,
        cache.payload_gain_derivative,
        parameters.payload_gain_raw,
    )
    OutputBank.refresh_cache!(cache.output, _output_parameters(parameters))
    return cache
end

function _cache_from_validated(parameters::Parameters)
    cache = _allocate_cache(parameters)
    return _refresh_cache_unchecked!(cache, parameters)
end

@inline function _cache_shapes_valid(cache::ModelCache)
    return size(cache.cell) == (CELLS_PER_BLOCK, BLOCKS) &&
           size(cache.cell_derivative) == (CELLS_PER_BLOCK, BLOCKS) &&
           size(cache.sensory_gain) == (2, INPUT_RAILS) &&
           size(cache.sensory_gain_derivative) == (2, INPUT_RAILS) &&
           size(cache.edge_strength) == (NetworkTopology.EDGE_COUNT,) &&
           size(cache.edge_strength_derivative) ==
               (NetworkTopology.EDGE_COUNT,) &&
           size(cache.payload_gain) == (EventPayload.ANALOG_GAIN_COUNT,) &&
           size(cache.payload_gain_derivative) ==
               (EventPayload.ANALOG_GAIN_COUNT,) &&
           length(cache.output.cell) == OutputBank.OUTPUT_CELLS
end

function PreparedModelState(
    staging::Parameters;
    generation::UInt64=UInt64(1),
)
    generation > UInt64(0) || throw(ArgumentError(
        "initial prepared generation must be positive",
    ))
    validate_parameters(staging)
    first_parameters = copy(staging)
    second_parameters = copy(staging)
    first_slot = PreparedSlot(
        first_parameters,
        _cache_from_validated(first_parameters),
        generation,
    )
    second_slot = PreparedSlot(
        second_parameters,
        _cache_from_validated(second_parameters),
        UInt64(0),
    )
    return PreparedModelState(
        (first_slot, second_slot),
        UInt8(1),
        generation,
    )
end

@inline prepared_generation(prepared::PreparedModelState) = prepared.generation

@inline function assert_generation(
    prepared::PreparedModelState{Float32},
    expected_generation::UInt64,
)
    active_index = Int(prepared.active_index)
    1 <= active_index <= 2 || throw(ArgumentError(
        "prepared model active slot is invalid",
    ))
    prepared.generation == expected_generation || throw(ArgumentError(
        "prepared model generation mismatch",
    ))
    slot = @inbounds prepared.slots[active_index]
    slot.generation == expected_generation || throw(ArgumentError(
        "prepared slot generation mismatch",
    ))
    _parameter_shapes_valid(slot.parameters) || throw(DimensionMismatch(
        "prepared parameter snapshot is malformed",
    ))
    _cache_shapes_valid(slot.cache) || throw(DimensionMismatch(
        "prepared transform cache is malformed",
    ))
    return slot
end

@inline function prepared_snapshot(
    prepared::PreparedModelState{Float32},
    expected_generation::UInt64,
)
    slot = assert_generation(prepared, expected_generation)
    return PreparedSnapshot(slot.parameters, slot.cache, slot.generation)
end

function publish!(prepared::PreparedModelState{Float32}, staging::Parameters)
    validate_parameters(staging)
    prepared.generation < typemax(UInt64) || throw(OverflowError(
        "prepared model generation exhausted",
    ))
    inactive_index = prepared.active_index == UInt8(1) ? 2 : 1
    inactive = @inbounds prepared.slots[inactive_index]

    # The active slot is untouched until every copy and transform succeeds.
    copy_parameters!(inactive.parameters, staging)
    _refresh_cache_unchecked!(inactive.cache, inactive.parameters)
    next_generation = prepared.generation + UInt64(1)
    inactive.generation = next_generation
    prepared.active_index = UInt8(inactive_index)
    prepared.generation = next_generation
    return next_generation
end

function _load_frozen_numeric_core()
    isfile(FROZEN_NUMERIC_CORE_PATH) || error(
        "frozen Float32 numeric core is absent: $FROZEN_NUMERIC_CORE_PATH",
    )
    digest = bytes2hex(SHA.sha256(read(FROZEN_NUMERIC_CORE_PATH)))
    digest == FROZEN_NUMERIC_CORE_SHA256 || error(
        "frozen Float32 numeric core checksum differs: $digest",
    )
    machine = Numeric.load_machine(FROZEN_NUMERIC_CORE_PATH)
    for kernel in (
        machine.adder,
        machine.subtractor,
        machine.sticky_or,
        machine.round_to_nearest_even,
    )
        kernel.hard_table == kernel.spec.target || error(
            "frozen Float32 numeric core contains an inexact transition kernel",
        )
    end
    machine.register_cell.hard_table == reshape(
        BitVector((false, false, false, true)),
        1,
        4,
    ) || error("frozen Float32 numeric register is inexact")
    reports = Numeric.validate_width_curriculum(machine; samples_per_width=64)
    all(report -> minimum(report.exact_rate) == 1.0, reports) || error(
        "frozen Float32 numeric core failed width validation",
    )
    return machine
end


function build_model(seed::Integer=DEFAULT_MODEL_SEED)
    seed >= 0 || throw(ArgumentError("model seed must be nonnegative"))
    seed_value = UInt64(seed)
    graph = NetworkTopology.build_topology(seed, Float32)
    numeric_core = _load_frozen_numeric_core()
    model = CPUHayModel{Float32}(
        graph,
        OutputBank.build_topology(xor(seed_value, UInt64(0x4f5554505554))),
        numeric_core,
    )
    staging = initialize_parameters(
        seed;
        numeric_cell_raw=numeric_core.register_cell.raw_parameters,
    )
    prepared = PreparedModelState(staging)
    return model, staging, prepared
end

@inline function _output_parameters(parameters::Parameters)
    return OutputBank.OutputParameters(
        parameters.output_cell_raw,
        parameters.output_edge_raw,
        parameters.output_q_basal_bias_raw,
        parameters.output_gain,
        parameters.output_bias,
    )
end

"""Exact anchor-only ablation; it is not a production execution mode."""
function recurrent_off_output(
    prepared::PreparedModelState{Float32},
    encoded_anchor::AbstractArray{Float32,3},
    expected_generation::UInt64,
)
    throw(ArgumentError(
        "anchor-only dense-head ablation was removed with the global head",
    ))
end

function _initial_anchor!(
    buffers::ForwardBuffers{Float32},
    scratch::ForwardScratch{Float32},
    cache::ModelCache{Float32},
    rails::AbstractVector{Float32},
)
    length(rails) == INPUT_RAILS || throw(DimensionMismatch(
        "rails must have length $INPUT_RAILS",
    ))
    Encoder.encode_sensory_cached!(
        scratch.sensory_input,
        rails,
        cache.sensory_gain,
    )
    @inbounds for block in 1:BLOCKS
        for cell in 1:CELLS_PER_BLOCK
            cell_cache = cache.cell[cell, block]
            scratch.soma_threshold[cell, block] = cell_cache.soma_threshold
            Cell.initial_state!(scratch.initial_state, cell_cache)
            Cell.cell_step!(
                @view(buffers.physical_anchor[:, cell, block]),
                scratch.initial_state,
                @view(scratch.sensory_input[:, cell, block]),
                cell_cache,
            )
            Codec.encode_state!(
                @view(scratch.encoded_anchor[:, cell, block]),
                @view(buffers.physical_anchor[:, cell, block]),
            )
        end
    end
    copyto!(scratch.current_physical, buffers.physical_anchor)
    copyto!(scratch.current_encoded, scratch.encoded_anchor)
    return nothing
end

@inline function _read_pending_input!(
    destination::AbstractVector{Float32},
    pending::Graph.ConductanceInbox{Float32},
    source::Int,
)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        destination[Cell.input_index(compartment, Cell.INPUT_AMPA)] =
            pending.ampa[compartment, source]
        destination[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
            pending.nmda[compartment, source]
        destination[Cell.input_index(compartment, Cell.INPUT_GABA)] =
            pending.gaba[compartment, source]
        pending.ampa[compartment, source] = 0.0f0
        pending.nmda[compartment, source] = 0.0f0
        pending.gaba[compartment, source] = 0.0f0
    end
    return destination
end

@inline function _merge_sorted_sources!(
    destination::Vector{Int},
    current_sources::Vector{Int},
    current_count::Int,
    previous_sources::Vector{Int},
    previous_count::Int,
)
    current_index = 1
    previous_index = 1
    destination_index = 0
    @inbounds while current_index <= current_count &&
                    previous_index <= previous_count
        current = current_sources[current_index]
        previous = previous_sources[previous_index]
        if current < previous
            destination_index += 1
            destination[destination_index] = current
            current_index += 1
        elseif previous < current
            destination_index += 1
            destination[destination_index] = previous
            previous_index += 1
        else
            destination_index += 1
            destination[destination_index] = current
            current_index += 1
            previous_index += 1
        end
    end
    @inbounds while current_index <= current_count
        destination_index += 1
        destination[destination_index] = current_sources[current_index]
        current_index += 1
    end
    @inbounds while previous_index <= previous_count
        destination_index += 1
        destination[destination_index] = previous_sources[previous_index]
        previous_index += 1
    end
    return destination_index
end

@inline function _pending_sum(pending::Graph.ConductanceInbox{Float32})
    return sum(pending.ampa) + sum(pending.nmda) + sum(pending.gaba)
end

function _precharge_anchor_events!(
    pending::Graph.ConductanceInbox{Float32},
    ring::Graph.DelayedPayloadRing{Float32},
    model::CPUHayModel{Float32},
    cache::ModelCache{Float32},
    encoded_anchor::AbstractArray{Float32,3},
    payload_channels::AbstractVector{Float32},
)
    @inbounds for block in 1:BLOCKS
        for cell in 1:CELLS_PER_BLOCK
            source = _global_cell(block, cell)
            EventPayload.payload_channels_cached_unchecked!(
                payload_channels,
                @view(encoded_anchor[:, cell, block]),
                cache.payload_gain,
            )
            Graph.set_current_payload!(ring, source, payload_channels)
        end
    end

    # Anchor activity has no temporal predecessor.  Mirroring it into both
    # taps lets every fixed current/previous edge contribute exactly once to
    # the first recurrent update, without changing the learned delay bit.
    copyto!(ring.previous, ring.current)
    Graph.deliver_payloads!(
        pending,
        model.graph,
        cache.edge_strength,
        ring,
        1:TOTAL_CELLS,
    )
    Graph.clear_payload_ring!(ring)
    return nothing
end

@inline _prepare_diagnostics!(::NoForwardDiagnostics) = nothing

@inline function _prepare_diagnostics!(
    diagnostics::FullForwardDiagnostics{Float32},
)
    fill!(diagnostics.payload_amplitude, 0.0f0)
    fill!(diagnostics.payload_analog, 0.0f0)
    fill!(diagnostics.payload_analog_source, Int16(0))
    diagnostics.pending_conductance_sum = 0.0f0
    return nothing
end

@inline _record_source!(
    ::NoForwardDiagnostics,
    ::Int,
    ::Int,
    ::Int,
    ::Int,
) = nothing

@inline function _record_source!(
    diagnostics::FullForwardDiagnostics{Float32},
    cell::Int,
    block::Int,
    step::Int,
    source::Int,
)
    @inbounds diagnostics.payload_source[cell, block, step] = Int16(source)
    return nothing
end

@inline _record_payload!(
    ::NoForwardDiagnostics,
    ::Int,
    ::Int,
    ::Int,
    ::Float32,
    ::Float32,
    ::Int,
) = nothing

@inline function _record_payload!(
    diagnostics::FullForwardDiagnostics{Float32},
    cell::Int,
    block::Int,
    step::Int,
    amplitude::Float32,
    analog::Float32,
    analog_source::Int,
)
    @inbounds begin
        diagnostics.payload_amplitude[cell, block, step] = amplitude
        diagnostics.payload_analog[cell, block, step] = analog
        diagnostics.payload_analog_source[cell, block, step] =
            Int16(analog_source)
    end
    return nothing
end


@inline _record_pending_before!(
    ::NoForwardDiagnostics,
    ::Graph.ConductanceInbox{Float32},
    ::Int,
) = nothing

@inline function _record_pending_before!(
    diagnostics::FullForwardDiagnostics{Float32},
    pending::Graph.ConductanceInbox{Float32},
    step::Int,
)
    @inbounds diagnostics.pending_before_delivery[step] = _pending_sum(pending)
    return nothing
end

@inline _record_pending_after!(
    ::NoForwardDiagnostics,
    ::Graph.ConductanceInbox{Float32},
    ::Int,
) = nothing

@inline function _record_pending_after!(
    diagnostics::FullForwardDiagnostics{Float32},
    pending::Graph.ConductanceInbox{Float32},
    step::Int,
)
    @inbounds diagnostics.pending_after_delivery[step] = _pending_sum(pending)
    return nothing
end

@inline _finish_diagnostics!(
    ::NoForwardDiagnostics,
    ::Graph.ConductanceInbox{Float32},
) = nothing

@inline function _finish_diagnostics!(
    diagnostics::FullForwardDiagnostics{Float32},
    pending::Graph.ConductanceInbox{Float32},
)
    diagnostics.pending_conductance_sum = _pending_sum(pending)
    return nothing
end

@inline function _validate_forward_storage(
    buffers::ForwardBuffers{Float32},
    scratch::ForwardScratch{Float32},
)
    _require_shape(
        buffers.physical_anchor,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        "physical_anchor",
    )
    _require_shape(
        buffers.physical_recurrent,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        "physical_recurrent",
    )
    _require_shape(buffers.raw_output, (OUTPUT_DIM,), "raw_output")
    _require_shape(
        scratch.sensory_input,
        (Cell.INPUT_DIM, CELLS_PER_BLOCK, BLOCKS),
        "sensory_input scratch",
    )
    _require_shape(
        scratch.encoded_anchor,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        "encoded_anchor scratch",
    )
    _require_shape(
        scratch.current_physical,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        "current_physical scratch",
    )
    _require_shape(
        scratch.current_encoded,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS),
        "current_encoded scratch",
    )
    _require_shape(
        scratch.soma_threshold,
        (CELLS_PER_BLOCK, BLOCKS),
        "soma_threshold scratch",
    )
    _require_shape(
        scratch.encoded_recurrent,
        (STATE_DIM, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        "encoded_recurrent scratch",
    )
    _require_shape(
        scratch.recurrent_inputs,
        (Cell.INPUT_DIM, CELLS_PER_BLOCK, BLOCKS, RECURRENT_STEPS),
        "recurrent_inputs scratch",
    )
    return nothing
end

"""
    forward_candidate!(buffers, scratch, model, prepared, rails,
                       diagnostics; ...)

The single route-free candidate-forward implementation. All numerical state,
event delivery, recurrent input tape, and output are overwritten
in caller-owned storage.  With `NoForwardDiagnostics()` this method performs no
diagnostic writes and allocates no memory after compilation.
"""
function forward_candidate!(
    buffers::ForwardBuffers{Float32},
    scratch::ForwardScratch{Float32},
    model::CPUHayModel{Float32},
    prepared::PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    diagnostics::D=NoForwardDiagnostics();
    expected_generation::UInt64,
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
) where {D<:AbstractForwardDiagnostics}
    slot = assert_generation(prepared, expected_generation)
    parameters = slot.parameters
    cache = slot.cache
    length(rails) == INPUT_RAILS || throw(DimensionMismatch(
        "rails must have length $INPUT_RAILS",
    ))
    _validate_forward_storage(buffers, scratch)
    0.0f0 <= event_floor <= 1.0f0 || throw(ArgumentError(
        "event floor must be in [0, 1]",
    ))
    0.0f0 <= spike_smoothing <= 1.0f0 || throw(ArgumentError(
        "spike smoothing must be in [0, 1]",
    ))
    _prepare_diagnostics!(diagnostics)
    Graph.clear_inbox!(scratch.pending)
    Graph.clear_payload_ring!(scratch.ring)
    _initial_anchor!(buffers, scratch, cache, rails)
    _precharge_anchor_events!(
        scratch.pending,
        scratch.ring,
        model,
        cache,
        scratch.encoded_anchor,
        scratch.payload_channels,
    )

    previous_count = 0
    @inbounds for step in 1:RECURRENT_STEPS
        # Every block advances. Only hard soma/plateau events emit onto the
        # sparse synaptic graph.
        for block in 1:BLOCKS
            for cell in 1:CELLS_PER_BLOCK
                source = _global_cell(block, cell)
                recurrent_input = @view scratch.recurrent_inputs[:, cell, block, step]
                fill!(recurrent_input, 0.0f0)
                _read_pending_input!(recurrent_input, scratch.pending, source)
                Cell.cell_step!(
                    @view(buffers.physical_recurrent[:, cell, block, step]),
                    @view(scratch.current_physical[:, cell, block]),
                    recurrent_input,
                    cache.cell[cell, block],
                    spike_smoothing,
                )
                Codec.encode_state!(
                    @view(scratch.encoded_recurrent[:, cell, block, step]),
                    @view(buffers.physical_recurrent[:, cell, block, step]),
                )
                copyto!(
                    @view(scratch.current_physical[:, cell, block]),
                    @view(buffers.physical_recurrent[:, cell, block, step]),
                )
                copyto!(
                    @view(scratch.current_encoded[:, cell, block]),
                    @view(scratch.encoded_recurrent[:, cell, block, step]),
                )
                _record_source!(
                    diagnostics,
                    cell,
                    block,
                    step,
                    source,
                )
            end
        end

        _record_pending_before!(diagnostics, scratch.pending, step)
        if step < RECURRENT_STEPS
            current_count = 0
            for block in 1:BLOCKS
                for cell in 1:CELLS_PER_BLOCK
                    source = _global_cell(block, cell)
                    own_state =
                        @view scratch.encoded_recurrent[:, cell, block, step]
                    EventPayload.has_payload_event(own_state, event_floor) ||
                        continue
                    current_count += 1
                    scratch.current_sources[current_count] = source
                    donor_source = source
                    donor_state = own_state
                    EventPayload.payload_channels_cached_unchecked!(
                        scratch.payload_channels,
                        donor_state,
                        cache.payload_gain,
                    )
                    spike_delta = own_state[Cell.SPIKE_INDEX] -
                                  donor_state[Cell.SPIKE_INDEX]
                    amplitude = 0.0f0
                    for compartment in 1:Cell.N_COMPARTMENTS
                        event_gain = EventPayload.compartment_event_gain(
                            own_state,
                            compartment,
                            event_floor,
                        )
                        for receptor in 1:Cell.INPUT_CHANNELS
                            channel = Cell.input_index(compartment, receptor)
                            scratch.payload_channels[channel] = event_gain * (
                                scratch.payload_channels[channel] + spike_delta
                            )
                            amplitude += scratch.payload_channels[channel]
                        end
                    end
                    amplitude /= length(scratch.payload_channels)
                    donor_analog = amplitude - own_state[Cell.SPIKE_INDEX]
                    Graph.set_current_payload!(
                        scratch.ring,
                        source,
                        scratch.payload_channels,
                    )
                    _record_payload!(
                        diagnostics,
                        cell,
                        block,
                        step,
                        amplitude,
                        donor_analog,
                        donor_source,
                    )
                end
            end

            active_count = _merge_sorted_sources!(
                scratch.active_sources,
                scratch.current_sources,
                current_count,
                scratch.previous_sources,
                previous_count,
            )
            Graph.deliver_payloads!(
                scratch.pending,
                model.graph,
                cache.edge_strength,
                scratch.ring,
                @view(scratch.active_sources[1:active_count]),
            )
            Graph.advance_payload_ring!(scratch.ring)
            for source_index in 1:current_count
                scratch.previous_sources[source_index] =
                    scratch.current_sources[source_index]
            end
            previous_count = current_count
        end
        _record_pending_after!(diagnostics, scratch.pending, step)
    end

    OutputBank.output_forward!(
        buffers.raw_output,
        buffers.output_trajectory,
        scratch.output,
        model.output_topology,
        _output_parameters(parameters),
        cache.output,
        model.numeric_core,
        cache.payload_gain,
        scratch.encoded_anchor,
        scratch.encoded_recurrent;
        event_floor,
        spike_smoothing,
    )
    _finish_diagnostics!(diagnostics, scratch.pending)
    return buffers
end

function _reference_forward(
    model::CPUHayModel{Float32},
    prepared::PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    expected_generation::UInt64=prepared_generation(prepared),
)
    buffers = ForwardBuffers(Float32)
    scratch = ForwardScratch(Float32)
    diagnostics = FullForwardDiagnostics(Float32)
    forward_candidate!(
        buffers,
        scratch,
        model,
        prepared,
        rails,
        diagnostics;
        expected_generation,
    )
    return ReferenceForward(
        buffers.physical_anchor,
        scratch.encoded_anchor,
        buffers.physical_recurrent,
        scratch.encoded_recurrent,
        buffers.raw_output,
        diagnostics.payload_amplitude,
        diagnostics.payload_analog,
        diagnostics.payload_source,
        diagnostics.payload_analog_source,
        diagnostics.pending_before_delivery,
        diagnostics.pending_after_delivery,
        diagnostics.pending_conductance_sum,
    )
end

"""Allocating mathematical reference for the canonical route-free forward."""
function forward_reference(
    model::CPUHayModel{Float32},
    prepared::PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    expected_generation::UInt64=prepared_generation(prepared),
)
    return _reference_forward(
        model,
        prepared,
        rails,
        expected_generation,
    )
end

end # module ReducedHayCPUNativeModel
