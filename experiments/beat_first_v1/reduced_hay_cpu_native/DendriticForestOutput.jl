module DendriticForestOutput

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..CompactDendriticNode

const Cell = ActiveApicalCell
const Delta = CandidateDeltaInput
const Node = CompactDendriticNode

export OUTPUT_CHANNELS,
       ANCHOR_COUNT,
       PLANE_COUNT,
       CONTEXT_DIM,
       BEFORE_PLANE,
       AFTER_PLANE,
       ForestOutputParameters,
       ForestOutputCache,
       ForestOutputTape,
       ForestOutputScratch,
       ForestOutputGradient,
       initialize_parameters,
       refresh_cache!,
       clear_gradient!,
       accumulate_gradient!,
       stored_parameter_count,
       fill_context!,
       hard_event_count,
       hard_event_denominator,
       forest_output_forward!,
       forest_output_pullback!

"""One private Reduced Hay output cell for every supervised Q channel."""
const OUTPUT_CHANNELS = 22

"""Twenty-four row roots followed by ten column roots in each plane."""
const ANCHOR_COUNT = 34
const PLANE_COUNT = 2
const BEFORE_PLANE = 1
const AFTER_PLANE = 2

const QUEUE_CONTEXT_DIM = Delta.QUEUE_PIECES * Delta.QUEUE_TOKENS
const CONTEXT_DIM =
    QUEUE_CONTEXT_DIM + 2 + Delta.AUX_FEATURES
const QUEUE_CONTEXT_FIRST = 1
const REN_CONTEXT_INDEX = QUEUE_CONTEXT_FIRST + QUEUE_CONTEXT_DIM
const BACK_TO_BACK_CONTEXT_INDEX = REN_CONTEXT_INDEX + 1
const AUX_CONTEXT_FIRST = BACK_TO_BACK_CONTEXT_INDEX + 1
const PLACEMENT_CAPACITY = 4
const AUXILIARY_OUTPUTS = OUTPUT_CHANNELS - 1

@assert Node.PAYLOAD_DIM == 3
@assert Cell.N_BASAL == 8
@assert CONTEXT_DIM == 81

"""
Canonical trainable output parameters.

Every output owns its complete cell dynamics.  Anchor contacts are ordinary
signed weights: a positive aggregate drive recruits AMPA/NMDA and a negative
aggregate drive recruits GABA inside `CompactDendriticNode`.  Before and after
planes never share a contact.  Context contacts are likewise signed and feed
only the active apical compartment.
"""
struct ForestOutputParameters{T<:AbstractFloat}
    cell_raw::Matrix{T}
    anchor_weight::Array{T,4}
    context_weight::Matrix{T}
    placement_weight::Matrix{T}
    cascade_weight::Matrix{T}
    gain::Vector{T}
    bias::Vector{T}

    function ForestOutputParameters(
        cell_raw::Matrix{T},
        anchor_weight::Array{T,4},
        context_weight::Matrix{T},
        placement_weight::Matrix{T},
        cascade_weight::Matrix{T},
        gain::Vector{T},
        bias::Vector{T},
    ) where {T<:AbstractFloat}
        size(cell_raw) == (Cell.PARAM_DIM, OUTPUT_CHANNELS) || throw(
            DimensionMismatch("cell raw parameters must have shape " *
                              "($(Cell.PARAM_DIM), $OUTPUT_CHANNELS)"),
        )
        size(anchor_weight) ==
            (Node.PAYLOAD_DIM, ANCHOR_COUNT, PLANE_COUNT, OUTPUT_CHANNELS) ||
            throw(DimensionMismatch(
                "anchor contacts must have shape (3, 34, 2, 22)",
            ))
        size(context_weight) == (CONTEXT_DIM, OUTPUT_CHANNELS) || throw(
            DimensionMismatch("context contacts must have shape (81, 22)"),
        )
        size(placement_weight) == (Delta.BOARD_CELLS, OUTPUT_CHANNELS) ||
            throw(DimensionMismatch(
                "placement contacts must have shape (240, 22)",
            ))
        size(cascade_weight) == (Node.PAYLOAD_DIM, AUXILIARY_OUTPUTS) ||
            throw(DimensionMismatch(
                "auxiliary-to-Q cascade contacts must have shape (3, 21)",
            ))
        length(gain) == OUTPUT_CHANNELS || throw(DimensionMismatch(
            "output gain must have 22 values",
        ))
        length(bias) == OUTPUT_CHANNELS || throw(DimensionMismatch(
            "output bias must have 22 values",
        ))
        return new{T}(
            cell_raw,
            anchor_weight,
            context_weight,
            placement_weight,
            cascade_weight,
            gain,
            bias,
        )
    end
end

@inline function _mix64(value::UInt64)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function _initial_signed(
    ::Type{T},
    family::UInt64,
    first::Int,
    second::Int,
    third::Int=0,
    fourth::Int=0,
) where {T<:AbstractFloat}
    word = _mix64(
        family ⊻
        UInt64(first) * UInt64(0x9e3779b97f4a7c15) ⊻
        UInt64(second) * UInt64(0xbf58476d1ce4e5b9) ⊻
        UInt64(third) * UInt64(0x94d049bb133111eb) ⊻
        UInt64(fourth) * UInt64(0xd6e8feb86659fd93),
    )
    fraction = T((word >> 40) & UInt64(0x00ff_ffff)) / T(0x0100_0000)
    magnitude = T(0.4) + T(0.2) * fraction
    return isodd(count_ones(word)) ? magnitude : -magnitude
end

function initialize_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    cell_raw = Matrix{T}(undef, Cell.PARAM_DIM, OUTPUT_CHANNELS)
    default_raw = Cell.default_raw_parameters(T)
    @inbounds for output in 1:OUTPUT_CHANNELS, parameter in 1:Cell.PARAM_DIM
        cell_raw[parameter, output] = default_raw[parameter]
    end

    anchor_weight = Array{T,4}(
        undef,
        Node.PAYLOAD_DIM,
        ANCHOR_COUNT,
        PLANE_COUNT,
        OUTPUT_CHANNELS,
    )
    @inbounds for output in 1:OUTPUT_CHANNELS,
                  plane in 1:PLANE_COUNT,
                  anchor in 1:ANCHOR_COUNT,
                  coordinate in 1:Node.PAYLOAD_DIM
        anchor_weight[coordinate, anchor, plane, output] = _initial_signed(
            T,
            UInt64(0x4d595df4d0f33173),
            output,
            plane,
            anchor,
            coordinate,
        )
    end

    context_weight = Matrix{T}(undef, CONTEXT_DIM, OUTPUT_CHANNELS)
    @inbounds for output in 1:OUTPUT_CHANNELS, source in 1:CONTEXT_DIM
        context_weight[source, output] = _initial_signed(
            T,
            UInt64(0x8d58ac26afe12e47),
            output,
            source,
        )
    end
    placement_weight = Matrix{T}(
        undef,
        Delta.BOARD_CELLS,
        OUTPUT_CHANNELS,
    )
    @inbounds for output in 1:OUTPUT_CHANNELS, position in 1:Delta.BOARD_CELLS
        placement_weight[position, output] = _initial_signed(
            T,
            UInt64(0x2d0f28c7e7e786b5),
            output,
            position,
        )
    end
    cascade_weight = Matrix{T}(
        undef,
        Node.PAYLOAD_DIM,
        AUXILIARY_OUTPUTS,
    )
    @inbounds for auxiliary in 1:AUXILIARY_OUTPUTS,
                  coordinate in 1:Node.PAYLOAD_DIM
        cascade_weight[coordinate, auxiliary] = _initial_signed(
            T,
            UInt64(0xa0761d6478bd642f),
            auxiliary,
            coordinate,
        )
    end
    return ForestOutputParameters(
        cell_raw,
        anchor_weight,
        context_weight,
        placement_weight,
        cascade_weight,
        ones(T, OUTPUT_CHANNELS),
        zeros(T, OUTPUT_CHANNELS),
    )
end

"""Transformed cell parameters cached outside the candidate hot path."""
mutable struct ForestOutputCache{T<:AbstractFloat}
    cell::Vector{Cell.CellParameterCache{T}}
    derivative::Vector{Cell.CellParameterDerivativeCache{T}}
end

function ForestOutputCache(parameters::ForestOutputParameters{T}) where {T}
    cell = Vector{Cell.CellParameterCache{T}}(undef, OUTPUT_CHANNELS)
    derivative = Vector{Cell.CellParameterDerivativeCache{T}}(
        undef,
        OUTPUT_CHANNELS,
    )
    cache = ForestOutputCache(cell, derivative)
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::ForestOutputCache{T},
    parameters::ForestOutputParameters{T},
) where {T}
    length(cache.cell) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "cell cache must have 22 entries",
    ))
    length(cache.derivative) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "cell derivative cache must have 22 entries",
    ))
    @inbounds for output in 1:OUTPUT_CHANNELS
        transformed, derivative = Cell.parameter_caches(
            @view(parameters.cell_raw[:, output]),
        )
        cache.cell[output] = transformed
        cache.derivative[output] = derivative
    end
    return cache
end

"""Caller-owned fixed forward trajectories and compact node payloads."""
struct ForestOutputTape{T<:AbstractFloat}
    traces::Vector{Node.NodeTrace{T}}
    drive::Matrix{T}
    payload::Matrix{T}
    placement_positions::Vector{UInt16}
    placement_count::Vector{UInt8}

    function ForestOutputTape(
        traces::Vector{Node.NodeTrace{T}},
        drive::Matrix{T},
        payload::Matrix{T},
        placement_positions::Vector{UInt16},
        placement_count::Vector{UInt8},
    ) where {T<:AbstractFloat}
        length(traces) == OUTPUT_CHANNELS || throw(DimensionMismatch(
            "output tape must own 22 node traces",
        ))
        size(drive) == (Node.DRIVE_DIM, OUTPUT_CHANNELS) || throw(
            DimensionMismatch("output drive tape must have shape (9, 22)"),
        )
        size(payload) == (Node.PAYLOAD_DIM, OUTPUT_CHANNELS) || throw(
            DimensionMismatch("output payload tape must have shape (3, 22)"),
        )
        length(placement_positions) == PLACEMENT_CAPACITY || throw(
            DimensionMismatch("placement tape must have capacity four"),
        )
        length(placement_count) == 1 || throw(DimensionMismatch(
            "placement count must be caller-owned scalar storage",
        ))
        return new{T}(
            traces,
            drive,
            payload,
            placement_positions,
            placement_count,
        )
    end
end

function ForestOutputTape(::Type{T}=Float32) where {T<:AbstractFloat}
    return ForestOutputTape(
        [Node.NodeTrace(T) for _ in 1:OUTPUT_CHANNELS],
        zeros(T, Node.DRIVE_DIM, OUTPUT_CHANNELS),
        zeros(T, Node.PAYLOAD_DIM, OUTPUT_CHANNELS),
        zeros(UInt16, PLACEMENT_CAPACITY),
        zeros(UInt8, 1),
    )
end

"""Number of hard output-cell events across all three physical phases."""
function hard_event_count(tape::ForestOutputTape{T}) where {T}
    count = 0
    @inbounds for output in 1:OUTPUT_CHANNELS, phase in 1:Node.PHASE_COUNT
        count += !iszero(tape.traces[output].events[phase])
    end
    return count
end

@inline hard_event_denominator() = OUTPUT_CHANNELS * Node.PHASE_COUNT

"""One reusable output-cell reverse workspace."""
struct ForestOutputScratch{T<:AbstractFloat}
    node::Node.NodeScratch{T}
    drive_bar::Vector{T}
    cell_raw_bar::Vector{T}
    analog_bar::Vector{T}
    cascade_analog_bar::Matrix{T}
    cascade_event_bar::Vector{T}
end

function ForestOutputScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return ForestOutputScratch(
        Node.NodeScratch(T),
        zeros(T, Node.DRIVE_DIM),
        zeros(T, Cell.PARAM_DIM),
        zeros(T, Node.ANALOG_DIM),
        zeros(T, Node.ANALOG_DIM, AUXILIARY_OUTPUTS),
        zeros(T, AUXILIARY_OUTPUTS),
    )
end

"""Gradient storage mirrors every trainable parameter exactly once."""
struct ForestOutputGradient{T<:AbstractFloat}
    cell_raw::Matrix{T}
    anchor_weight::Array{T,4}
    context_weight::Matrix{T}
    placement_weight::Matrix{T}
    cascade_weight::Matrix{T}
    gain::Vector{T}
    bias::Vector{T}
end

function ForestOutputGradient(::Type{T}=Float32) where {T<:AbstractFloat}
    return ForestOutputGradient(
        zeros(T, Cell.PARAM_DIM, OUTPUT_CHANNELS),
        zeros(T, Node.PAYLOAD_DIM, ANCHOR_COUNT, PLANE_COUNT, OUTPUT_CHANNELS),
        zeros(T, CONTEXT_DIM, OUTPUT_CHANNELS),
        zeros(T, Delta.BOARD_CELLS, OUTPUT_CHANNELS),
        zeros(T, Node.PAYLOAD_DIM, AUXILIARY_OUTPUTS),
        zeros(T, OUTPUT_CHANNELS),
        zeros(T, OUTPUT_CHANNELS),
    )
end

function clear_gradient!(gradient::ForestOutputGradient{T}) where {T}
    fill!(gradient.cell_raw, zero(T))
    fill!(gradient.anchor_weight, zero(T))
    fill!(gradient.context_weight, zero(T))
    fill!(gradient.placement_weight, zero(T))
    fill!(gradient.cascade_weight, zero(T))
    fill!(gradient.gain, zero(T))
    fill!(gradient.bias, zero(T))
    return gradient
end

function accumulate_gradient!(
    destination::ForestOutputGradient{T},
    source::ForestOutputGradient{T},
) where {T}
    @inbounds @simd for index in eachindex(destination.cell_raw)
        destination.cell_raw[index] += source.cell_raw[index]
    end
    @inbounds @simd for index in eachindex(destination.anchor_weight)
        destination.anchor_weight[index] += source.anchor_weight[index]
    end
    @inbounds @simd for index in eachindex(destination.context_weight)
        destination.context_weight[index] += source.context_weight[index]
    end
    @inbounds @simd for index in eachindex(destination.placement_weight)
        destination.placement_weight[index] += source.placement_weight[index]
    end
    @inbounds @simd for index in eachindex(destination.cascade_weight)
        destination.cascade_weight[index] += source.cascade_weight[index]
    end
    @inbounds @simd for index in eachindex(destination.gain)
        destination.gain[index] += source.gain[index]
        destination.bias[index] += source.bias[index]
    end
    return destination
end

@inline function stored_parameter_count(parameters::ForestOutputParameters)
    return length(parameters.cell_raw) + length(parameters.anchor_weight) +
           length(parameters.context_weight) +
           length(parameters.placement_weight) +
           length(parameters.cascade_weight) + length(parameters.gain) +
           length(parameters.bias)
end

@inline function _signed_bit(value::UInt8, name::AbstractString)
    (value == 0x00 || value == 0x01) || throw(ArgumentError(
        "$name must contain only zero/one values",
    ))
    return iszero(value) ? -1.0f0 : 1.0f0
end

"""
    fill_context!(context, common, materialization)

Build the canonical 81-value signed context without the retired
`ContextAfferents` path.  Queue identity is role-major and bipolar.  REN is
mapped from `[0,30]` to `[-1,1]`; B2B is bipolar; the 37 normalized candidate
auxiliaries are centered from `[0,1]` to `[-1,1]`.
"""
function fill_context!(
    context::AbstractVector{T},
    common::Delta.StateCommon,
    materialization::Delta.CandidateMaterialization,
) where {T<:AbstractFloat}
    length(context) == CONTEXT_DIM || throw(DimensionMismatch(
        "output context must have 81 values",
    ))
    index = 0
    @inbounds for role in 1:Delta.QUEUE_TOKENS, piece in 1:Delta.QUEUE_PIECES
        index += 1
        context[index] = T(_signed_bit(common.queue[piece, role], "queue"))
    end
    @inbounds begin
        ren = clamp(common.ren[1] / 30.0f0, 0.0f0, 1.0f0)
        context[REN_CONTEXT_INDEX] = T(muladd(2.0f0, ren, -1.0f0))
        back_to_back = common.back_to_back[1]
        context[BACK_TO_BACK_CONTEXT_INDEX] =
            T(muladd(2.0f0, back_to_back, -1.0f0))
        for auxiliary in 1:Delta.AUX_FEATURES
            context[AUX_CONTEXT_FIRST + auxiliary - 1] = T(muladd(
                2.0f0,
                materialization.aux[auxiliary],
                -1.0f0,
            ))
        end
    end
    return context
end

@inline function _check_anchors(anchors)
    size(anchors) == (Node.PAYLOAD_DIM, ANCHOR_COUNT, PLANE_COUNT) || throw(
        DimensionMismatch("forest anchors must have shape (3, 34, 2)"),
    )
    return nothing
end

@inline function _check_context(context)
    length(context) == CONTEXT_DIM || throw(DimensionMismatch(
        "output context must have 81 values",
    ))
    return nothing
end

@inline function _check_placement(placement)
    size(placement) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) || throw(
        DimensionMismatch("candidate placement must have shape (24, 10)"),
    )
    return nothing
end

function _capture_placement!(tape::ForestOutputTape, placement)
    _check_placement(placement)
    count = 0
    @inbounds for column in 1:Delta.BOARD_COLUMNS, row in 1:Delta.BOARD_ROWS
        value = placement[row, column]
        (value == 0x00 || value == 0x01) || throw(ArgumentError(
            "candidate placement must contain only zero/one values",
        ))
        iszero(value) && continue
        count += 1
        count <= PLACEMENT_CAPACITY || throw(ArgumentError(
            "candidate placement exceeds four occupied positions",
        ))
        tape.placement_positions[count] = UInt16(
            row + (column - 1) * Delta.BOARD_ROWS,
        )
    end
    tape.placement_count[1] = UInt8(count)
    @inbounds for index in (count + 1):PLACEMENT_CAPACITY
        tape.placement_positions[index] = UInt16(0)
    end
    return tape
end

@inline _anchor_branch(anchor::Int) = (anchor - 1) % Cell.N_BASAL + 1

@inline function _branch_anchor_count(branch::Int)
    return fld(ANCHOR_COUNT - branch, Cell.N_BASAL) + 1
end

@inline function _branch_normalization(::Type{T}, branch::Int) where {T}
    contact_count =
        Node.PAYLOAD_DIM * PLANE_COUNT * _branch_anchor_count(branch)
    return inv(sqrt(T(contact_count)))
end

@inline _context_normalization(::Type{T}) where {T} =
    inv(sqrt(T(CONTEXT_DIM)))

@inline _placement_branch(position::Int) =
    (position - 1) % Cell.N_BASAL + 1
@inline _placement_normalization(::Type{T}) where {T} =
    inv(sqrt(T(PLACEMENT_CAPACITY)))

@inline _cascade_branch(auxiliary_output::Int) =
    (auxiliary_output - 2) % Cell.N_BASAL + 1

@inline function _cascade_branch_count(branch::Int)
    return fld(AUXILIARY_OUTPUTS - branch, Cell.N_BASAL) + 1
end

@inline function _cascade_normalization(::Type{T}, branch::Int) where {T}
    contact_count = Node.PAYLOAD_DIM * _cascade_branch_count(branch)
    return inv(sqrt(T(contact_count)))
end

@inline function _assemble_direct_drive!(
    drive::AbstractVector{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    tape::ForestOutputTape{T},
    parameters::ForestOutputParameters{T},
    output::Int,
) where {T<:AbstractFloat}
    fill!(drive, zero(T))
    @inbounds for plane in 1:PLANE_COUNT, anchor in 1:ANCHOR_COUNT
        branch = _anchor_branch(anchor)
        normalization = _branch_normalization(T, branch)
        for coordinate in 1:Node.PAYLOAD_DIM
            drive[branch] = muladd(
                parameters.anchor_weight[coordinate, anchor, plane, output] *
                normalization,
                anchors[coordinate, anchor, plane],
                drive[branch],
            )
        end
    end
    placement_normalization = _placement_normalization(T)
    @inbounds for placement_index in 1:Int(tape.placement_count[1])
        position = Int(tape.placement_positions[placement_index])
        branch = _placement_branch(position)
        drive[branch] += parameters.placement_weight[position, output] *
                         placement_normalization
    end
    apical = zero(T)
    normalization = _context_normalization(T)
    @inbounds for source in 1:CONTEXT_DIM
        apical = muladd(
            parameters.context_weight[source, output] * normalization,
            context[source],
            apical,
        )
    end
    drive[Node.DRIVE_DIM] = apical
    return drive
end

@inline function _add_cascade_drive!(
    drive::AbstractVector{T},
    tape::ForestOutputTape{T},
    parameters::ForestOutputParameters{T},
) where {T<:AbstractFloat}
    @inbounds for auxiliary_output in 2:OUTPUT_CHANNELS
        auxiliary = auxiliary_output - 1
        branch = _cascade_branch(auxiliary_output)
        normalization = _cascade_normalization(T, branch)
        for coordinate in 1:Node.PAYLOAD_DIM
            drive[branch] = muladd(
                parameters.cascade_weight[coordinate, auxiliary] *
                normalization,
                tape.payload[coordinate, auxiliary_output],
                drive[branch],
            )
        end
    end
    return drive
end

"""
Run exactly 22 private Reduced Hay output cells.  Cells 2--22 first produce
the auxiliary channels.  Their complete compact payloads then enter cell 1
through trainable signed basal contacts, making Q a state-dependent dendritic
cascade rather than a single apical ridge interaction.

`anchors[3, :, :]` contains hard root events.  Those bits causally enter the
basal drive, but the supervised Q value itself is *only* the resting-centered
continuous final soma margin.  `output_events` is terminal diagnostic/control
state and is never added to `raw_output`.
"""
function forest_output_forward!(
    raw_output::AbstractVector{T},
    output_events::AbstractVector{T},
    tape::ForestOutputTape{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    placement::AbstractMatrix{UInt8},
    parameters::ForestOutputParameters{T},
    cache::ForestOutputCache{T},
) where {T<:AbstractFloat}
    length(raw_output) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "forest output must have 22 values",
    ))
    length(output_events) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "output event vector must have 22 values",
    ))
    _check_anchors(anchors)
    _check_context(context)
    _capture_placement!(tape, placement)

    # Auxiliary cells must precede Q because Q consumes their full payload.
    @inbounds for output in 2:OUTPUT_CHANNELS
        drive = @view tape.drive[:, output]
        payload = @view tape.payload[:, output]
        _assemble_direct_drive!(
            drive, anchors, context, tape, parameters, output,
        )
        event = Node.node_forward!(
            payload,
            tape.traces[output],
            drive,
            cache.cell[output],
        )
        raw_output[output] = muladd(
            parameters.gain[output],
            payload[Node.CENTERED_MARGIN_INDEX],
            parameters.bias[output],
        )
        output_events[output] = event
    end

    output = 1
    drive = @view tape.drive[:, output]
    payload = @view tape.payload[:, output]
    _assemble_direct_drive!(drive, anchors, context, tape, parameters, output)
    _add_cascade_drive!(drive, tape, parameters)
    event = Node.node_forward!(
        payload,
        tape.traces[output],
        drive,
        cache.cell[output],
    )
    raw_output[output] = muladd(
        parameters.gain[output],
        payload[Node.CENTERED_MARGIN_INDEX],
        parameters.bias[output],
    )
    output_events[output] = event
    return raw_output
end

@inline function _distribute_direct_drive_pullback!(
    anchor_bar::AbstractArray{T,3},
    context_bar::AbstractVector{T},
    gradient::ForestOutputGradient{T},
    scratch::ForestOutputScratch{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    tape::ForestOutputTape{T},
    parameters::ForestOutputParameters{T},
    output::Int,
) where {T<:AbstractFloat}
    @inbounds for plane in 1:PLANE_COUNT, anchor in 1:ANCHOR_COUNT
        branch = _anchor_branch(anchor)
        normalization = _branch_normalization(T, branch)
        drive_seed = scratch.drive_bar[branch] * normalization
        for coordinate in 1:Node.PAYLOAD_DIM
            weight = parameters.anchor_weight[
                coordinate,
                anchor,
                plane,
                output,
            ]
            anchor_bar[coordinate, anchor, plane] = muladd(
                drive_seed,
                weight,
                anchor_bar[coordinate, anchor, plane],
            )
            gradient.anchor_weight[coordinate, anchor, plane, output] =
                drive_seed * anchors[coordinate, anchor, plane]
        end
    end

    placement_normalization = _placement_normalization(T)
    @inbounds for placement_index in 1:Int(tape.placement_count[1])
        position = Int(tape.placement_positions[placement_index])
        branch = _placement_branch(position)
        gradient.placement_weight[position, output] =
            scratch.drive_bar[branch] * placement_normalization
    end

    normalization = _context_normalization(T)
    drive_seed = scratch.drive_bar[Node.DRIVE_DIM] * normalization
    @inbounds for source in 1:CONTEXT_DIM
        context_bar[source] = muladd(
            drive_seed,
            parameters.context_weight[source, output],
            context_bar[source],
        )
        gradient.context_weight[source, output] =
            drive_seed * context[source]
    end
    return nothing
end

@inline function _distribute_cascade_pullback!(
    gradient::ForestOutputGradient{T},
    scratch::ForestOutputScratch{T},
    tape::ForestOutputTape{T},
    parameters::ForestOutputParameters{T},
) where {T<:AbstractFloat}
    fill!(scratch.cascade_analog_bar, zero(T))
    fill!(scratch.cascade_event_bar, zero(T))
    @inbounds for auxiliary_output in 2:OUTPUT_CHANNELS
        auxiliary = auxiliary_output - 1
        branch = _cascade_branch(auxiliary_output)
        drive_seed = scratch.drive_bar[branch] *
                     _cascade_normalization(T, branch)
        for coordinate in 1:Node.PAYLOAD_DIM
            gradient.cascade_weight[coordinate, auxiliary] =
                drive_seed * tape.payload[coordinate, auxiliary_output]
        end
        for coordinate in 1:Node.ANALOG_DIM
            scratch.cascade_analog_bar[coordinate, auxiliary] =
                drive_seed * parameters.cascade_weight[coordinate, auxiliary]
        end
        scratch.cascade_event_bar[auxiliary] =
            drive_seed * parameters.cascade_weight[
                Node.HARD_EVENT_INDEX,
                auxiliary,
            ]
    end
    return nothing
end

function _forest_output_pullback!(
    anchor_bar::AbstractArray{T,3},
    context_bar::AbstractVector{T},
    gradient::ForestOutputGradient{T},
    scratch::ForestOutputScratch{T},
    tape::ForestOutputTape{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    placement::AbstractMatrix{UInt8},
    parameters::ForestOutputParameters{T},
    cache::ForestOutputCache{T},
    raw_bar::AbstractVector{T},
    output_event_bar,
) where {T<:AbstractFloat}
    _check_anchors(anchors)
    _check_anchors(anchor_bar)
    _check_context(context)
    _check_context(context_bar)
    _check_placement(placement)
    length(raw_bar) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "forest output cotangent must have 22 values",
    ))
    if output_event_bar !== nothing
        length(output_event_bar) == OUTPUT_CHANNELS || throw(
            DimensionMismatch("output event cotangent must have 22 values"),
        )
    end
    fill!(anchor_bar, zero(T))
    fill!(context_bar, zero(T))
    clear_gradient!(gradient)

    # Reverse Q first so its cascade cotangents are available to auxiliaries.
    output = 1
    output_seed = raw_bar[output]
    centered_margin = tape.payload[Node.CENTERED_MARGIN_INDEX, output]
    gradient.gain[output] = output_seed * centered_margin
    gradient.bias[output] = output_seed
    scratch.analog_bar[Node.CENTERED_MARGIN_INDEX] =
        output_seed * parameters.gain[output]
    scratch.analog_bar[Node.MEAN_PLATEAU_INDEX] = zero(T)
    event_seed = output_event_bar === nothing ?
        zero(T) : output_event_bar[output]
    Node.node_pullback!(
        scratch.drive_bar,
        scratch.cell_raw_bar,
        scratch.node,
        tape.traces[output],
        @view(tape.drive[:, output]),
        cache.cell[output],
        cache.derivative[output],
        scratch.analog_bar,
        event_seed,
    )
    @inbounds @simd for parameter in 1:Cell.PARAM_DIM
        gradient.cell_raw[parameter, output] =
            scratch.cell_raw_bar[parameter]
    end
    _distribute_direct_drive_pullback!(
        anchor_bar,
        context_bar,
        gradient,
        scratch,
        anchors,
        context,
        tape,
        parameters,
        output,
    )
    _distribute_cascade_pullback!(gradient, scratch, tape, parameters)

    @inbounds for output in 2:OUTPUT_CHANNELS
        auxiliary = output - 1
        output_seed = raw_bar[output]
        centered_margin = tape.payload[
            Node.CENTERED_MARGIN_INDEX,
            output,
        ]
        gradient.gain[output] = output_seed * centered_margin
        gradient.bias[output] = output_seed
        scratch.analog_bar[Node.CENTERED_MARGIN_INDEX] =
            output_seed * parameters.gain[output] +
            scratch.cascade_analog_bar[
                Node.CENTERED_MARGIN_INDEX,
                auxiliary,
            ]
        scratch.analog_bar[Node.MEAN_PLATEAU_INDEX] =
            scratch.cascade_analog_bar[
                Node.MEAN_PLATEAU_INDEX,
                auxiliary,
            ]
        event_seed = scratch.cascade_event_bar[auxiliary] +
            (output_event_bar === nothing ? zero(T) : output_event_bar[output])

        Node.node_pullback!(
            scratch.drive_bar,
            scratch.cell_raw_bar,
            scratch.node,
            tape.traces[output],
            @view(tape.drive[:, output]),
            cache.cell[output],
            cache.derivative[output],
            scratch.analog_bar,
            event_seed,
        )
        @simd for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output] =
                scratch.cell_raw_bar[parameter]
        end
        _distribute_direct_drive_pullback!(
            anchor_bar,
            context_bar,
            gradient,
            scratch,
            anchors,
            context,
            tape,
            parameters,
            output,
        )
    end
    return gradient
end

"""
Exact conditional reverse of the continuous 22-D Q path.

The third anchor coordinate is returned in `anchor_bar[3, :, :]`, deliberately
separate from the first two analog coordinates.  Its caller must pass that
value to the root node as the explicit hard-event surrogate cotangent; it must
not be mixed into that node's analog payload cotangent.
"""
function forest_output_pullback!(
    anchor_bar::AbstractArray{T,3},
    context_bar::AbstractVector{T},
    gradient::ForestOutputGradient{T},
    scratch::ForestOutputScratch{T},
    tape::ForestOutputTape{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    placement::AbstractMatrix{UInt8},
    parameters::ForestOutputParameters{T},
    cache::ForestOutputCache{T},
    raw_bar::AbstractVector{T},
) where {T<:AbstractFloat}
    return _forest_output_pullback!(
        anchor_bar,
        context_bar,
        gradient,
        scratch,
        tape,
        anchors,
        context,
        placement,
        parameters,
        cache,
        raw_bar,
        nothing,
    )
end

"""As above, with an explicit terminal output-event surrogate cotangent."""
function forest_output_pullback!(
    anchor_bar::AbstractArray{T,3},
    context_bar::AbstractVector{T},
    gradient::ForestOutputGradient{T},
    scratch::ForestOutputScratch{T},
    tape::ForestOutputTape{T},
    anchors::AbstractArray{T,3},
    context::AbstractVector{T},
    placement::AbstractMatrix{UInt8},
    parameters::ForestOutputParameters{T},
    cache::ForestOutputCache{T},
    raw_bar::AbstractVector{T},
    output_event_bar::AbstractVector{T},
) where {T<:AbstractFloat}
    return _forest_output_pullback!(
        anchor_bar,
        context_bar,
        gradient,
        scratch,
        tape,
        anchors,
        context,
        placement,
        parameters,
        cache,
        raw_bar,
        output_event_bar,
    )
end

end # module DendriticForestOutput
