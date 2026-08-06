module DendriticDeltaForest

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..CompactDendriticNode
using ..DendriticDeltaForestTopology
using ..DendriticProgramBank

const Cell = ActiveApicalCell
const Delta = CandidateDeltaInput
const Node = CompactDendriticNode
const Topology = DendriticDeltaForestTopology
const Bank = DendriticProgramBank

export AFTER_PLANE,
       BEFORE_PLANE,
       INTERNAL_CLASS_COUNT,
       PAYLOAD_DIM,
       PlaneParameters,
       PlaneCache,
       PlaneGradient,
       PlaneState,
       COWPlaneState,
       PlaneWorkspace,
       PlaneForwardStats,
       initialize_parameters,
       refresh_cache!,
       clear_gradient!,
       base_forward!,
       candidate_forward!,
       candidate_reverse!,
       base_reverse!,
       copy_anchor_payloads!,
       payload_value,
       leaf_program_rows,
       leaf_drive!,
       stored_parameter_count

const BEFORE_PLANE = UInt8(1)
const AFTER_PLANE = UInt8(2)
const PAYLOAD_DIM = Node.PAYLOAD_DIM
const INTERNAL_CLASS_COUNT = 4
const CONTACT_LANES = PAYLOAD_DIM
const _DEFAULT_SEED = UInt64(0x64656c7461666f72)
const _PROGRAM_SCALE = 0.5f0
const _RESTING_GAIN = 0.55f0

# Internal classes are stored in the same semantic order in every plane.
const _ROW_HALF_PARAMETER = 1
const _COLUMN_GROUP_PARAMETER = 2
const _ROW_ROOT_PARAMETER = 3
const _COLUMN_ROOT_PARAMETER = 4

@inline function _parameter_class(node_class::UInt8)
    node_class == Topology.ROW_HALF_CLASS && return _ROW_HALF_PARAMETER
    node_class == Topology.COLUMN_GROUP_CLASS && return _COLUMN_GROUP_PARAMETER
    node_class == Topology.ROW_ROOT_CLASS && return _ROW_ROOT_PARAMETER
    node_class == Topology.COLUMN_ROOT_CLASS && return _COLUMN_ROOT_PARAMETER
    throw(ArgumentError("leaves do not own internal-node parameters"))
end

@inline function _mix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline _softplus(value::T) where {T<:AbstractFloat} =
    max(value, zero(T)) + log1p(exp(-abs(value)))
@inline _softplus_derivative(value::T) where {T<:AbstractFloat} =
    inv(one(T) + exp(-value))
@inline _inverse_softplus(value::T) where {T<:AbstractFloat} =
    value + log(-expm1(-value))

"""Trainable state of one anatomical before/after plane."""
struct PlaneParameters{T<:AbstractFloat}
    internal_raw::Matrix{T}             # Cell.PARAM_DIM x four node classes
    child_contact::Matrix{T}            # three typed payload lanes x 568 edges
end

function initialize_parameters(
    seed::Integer=_DEFAULT_SEED,
    ::Type{T}=Float32,
) where {T<:AbstractFloat}
    internal_raw = Matrix{T}(undef, Cell.PARAM_DIM, INTERNAL_CLASS_COUNT)
    default = Cell.default_raw_parameters(T)
    @inbounds for class in 1:INTERNAL_CLASS_COUNT, parameter in 1:Cell.PARAM_DIM
        internal_raw[parameter, class] = default[parameter]
    end

    # Each child slot is one dendritic branch.  The three independently
    # trainable signed contacts preserve message type: soma-margin evidence,
    # plateau evidence, and the exact hard event.  Small deterministic jitter
    # breaks reducer symmetries without changing the topology.
    child_contact = Matrix{T}(undef, CONTACT_LANES, Topology.CHILD_EDGE_COUNT)
    word = UInt64(seed)
    base = (T(0.32), T(0.22), T(0.38))
    @inbounds for edge in 1:Topology.CHILD_EDGE_COUNT
        for lane in 1:CONTACT_LANES
            word = _mix64(word + UInt64(edge * 5 + lane))
            jitter = T(Int((word >> 56) & UInt64(0xff)) - 127) / T(8192)
            child_contact[lane, edge] = base[lane] + jitter
        end
    end
    return PlaneParameters(internal_raw, child_contact)
end

"""Cached physical parameters and raw-transform derivatives for four classes."""
mutable struct PlaneCache{T<:AbstractFloat}
    physical::Vector{Cell.CellParameterCache{T}}
    derivative::Vector{Cell.CellParameterDerivativeCache{T}}
end

function PlaneCache(parameters::PlaneParameters{T}) where {T<:AbstractFloat}
    physical = Vector{Cell.CellParameterCache{T}}(undef, INTERNAL_CLASS_COUNT)
    derivative = Vector{Cell.CellParameterDerivativeCache{T}}(
        undef,
        INTERNAL_CLASS_COUNT,
    )
    cache = PlaneCache(physical, derivative)
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::PlaneCache{T},
    parameters::PlaneParameters{T},
) where {T<:AbstractFloat}
    @inbounds for class in 1:INTERNAL_CLASS_COUNT
        raw = @view parameters.internal_raw[:, class]
        physical, derivative = Cell.parameter_caches(raw)
        cache.physical[class] = physical
        cache.derivative[class] = derivative
    end
    return cache
end

"""Dense gradient of the small always-active internal plane parameters."""
struct PlaneGradient{T<:AbstractFloat}
    internal_raw::Matrix{T}
    child_contact::Matrix{T}
end

function PlaneGradient(parameters::PlaneParameters{T}) where {T<:AbstractFloat}
    return PlaneGradient(
        zeros(T, size(parameters.internal_raw)),
        zeros(T, size(parameters.child_contact)),
    )
end

function clear_gradient!(gradient::PlaneGradient)
    fill!(gradient.internal_raw, zero(eltype(gradient.internal_raw)))
    fill!(gradient.child_contact, zero(eltype(gradient.child_contact)))
    return gradient
end

"""Allocation-free execution counters returned by full and COW forwards."""
struct PlaneForwardStats
    leaf_hard_events::Int32
    internal_hard_events::Int32
    root_hard_events::Int32
    dirty_leaves::Int32
    dirty_ancestors::Int32
    compact_messages::Int32
end

PlaneForwardStats() = PlaneForwardStats(0, 0, 0, 0, 0, 0)

"""Compact persistent payload of a fully evaluated 362-node plane."""
mutable struct PlaneState{T<:AbstractFloat}
    payload::Matrix{T}
    stats::PlaneForwardStats
end

PlaneState(::Type{T}=Float32) where {T<:AbstractFloat} =
    PlaneState(zeros(T, PAYLOAD_DIM, Topology.NODE_COUNT), PlaneForwardStats())

"""
Generation-tagged candidate overlay.  Only the exact dirty closure is written;
all clean payloads are read directly from the common `PlaneState`.
"""
mutable struct COWPlaneState{T<:AbstractFloat}
    payload::Matrix{T}
    generation::Vector{UInt32}
    epoch::UInt32
    stats::PlaneForwardStats
end

COWPlaneState(::Type{T}=Float32) where {T<:AbstractFloat} = COWPlaneState(
    zeros(T, PAYLOAD_DIM, Topology.NODE_COUNT),
    zeros(UInt32, Topology.NODE_COUNT),
    UInt32(0),
    PlaneForwardStats(),
)

@inline function _next_epoch!(state::COWPlaneState)
    if state.epoch == typemax(UInt32)
        fill!(state.generation, UInt32(0))
        state.epoch = UInt32(1)
    else
        state.epoch += UInt32(1)
    end
    return state.epoch
end

@inline function _is_current(state::COWPlaneState, node::Int)
    return @inbounds state.generation[node] == state.epoch
end

@inline payload_value(state::PlaneState, lane::Integer, node::Integer) =
    @inbounds state.payload[Int(lane), Int(node)]

@inline function payload_value(
    candidate::COWPlaneState,
    base::PlaneState,
    lane::Integer,
    node::Integer,
)
    physical_node = Int(node)
    return @inbounds candidate.generation[physical_node] == candidate.epoch ?
        candidate.payload[Int(lane), physical_node] :
        base.payload[Int(lane), physical_node]
end

"""All fixed scratch used by full/COW forward and replay reverse."""
mutable struct PlaneWorkspace{T<:AbstractFloat}
    leaf_trace::Node.NodeTrace{T}
    internal_trace::Node.NodeTrace{T}
    node_reverse::Node.NodeScratch{T}
    leaf_drive::Vector{T}
    effective_drive::Vector{T}
    internal_drive::Vector{T}
    program_payload::Vector{Float32}
    raw_program::Vector{T}
    default_program::Vector{T}
    local_payload::Vector{T}
    danalog::Vector{T}
    ddrive::Vector{T}
    draw_node::Vector{T}
    dprogram::Vector{T}
    candidate_bar::Matrix{T}
    candidate_bar_generation::Vector{UInt32}
    candidate_bar_epoch::UInt32
end

function PlaneWorkspace(::Type{T}=Float32) where {T<:AbstractFloat}
    default_program = fill(_inverse_softplus(T(_RESTING_GAIN)), 2 * Cell.N_BASAL)
    return PlaneWorkspace(
        Node.NodeTrace(T),
        Node.NodeTrace(T),
        Node.NodeScratch(T),
        zeros(T, Node.DRIVE_DIM),
        zeros(T, Node.DRIVE_DIM),
        zeros(T, Node.DRIVE_DIM),
        zeros(Float32, Bank.PAYLOAD_WIDTH),
        zeros(T, Bank.PAYLOAD_WIDTH),
        default_program,
        zeros(T, PAYLOAD_DIM),
        zeros(T, Node.ANALOG_DIM),
        zeros(T, Node.DRIVE_DIM),
        zeros(T, Cell.PARAM_DIM),
        zeros(T, Bank.PAYLOAD_WIDTH),
        zeros(T, PAYLOAD_DIM, Topology.NODE_COUNT),
        zeros(UInt32, Topology.NODE_COUNT),
        UInt32(0),
    )
end

@inline function _next_bar_epoch!(workspace::PlaneWorkspace)
    if workspace.candidate_bar_epoch == typemax(UInt32)
        fill!(workspace.candidate_bar_generation, UInt32(0))
        workspace.candidate_bar_epoch = UInt32(1)
    else
        workspace.candidate_bar_epoch += UInt32(1)
    end
    return workspace.candidate_bar_epoch
end

@inline function _touch_bar!(workspace::PlaneWorkspace{T}, node::Int) where {T}
    @inbounds if workspace.candidate_bar_generation[node] !=
                 workspace.candidate_bar_epoch
        workspace.candidate_bar_generation[node] = workspace.candidate_bar_epoch
        for lane in 1:PAYLOAD_DIM
            workspace.candidate_bar[lane, node] = zero(T)
        end
    end
    return nothing
end

# -- Exact compact semantic leaf address ------------------------------------

@inline function _coordinates(position::Int)
    column = div(position - 1, Delta.BOARD_ROWS) + 1
    board_row = position - (column - 1) * Delta.BOARD_ROWS
    return board_row, column
end

@inline function _occupancy_mask(board, board_row::Int, column::Int)
    mask = UInt16(0)
    lane = 0
    @inbounds for column_offset in -1:1, row_offset in -1:1
        row = board_row + row_offset
        col = column + column_offset
        if row < 1 || row > Delta.BOARD_ROWS ||
           col < 1 || col > Delta.BOARD_COLUMNS
            continue
        end
        !iszero(board[row, col]) && (mask |= UInt16(1) << lane)
        lane += 1
    end
    return Int32(mask)
end

@inline function _table1_base(row::Int, column::Int)
    top = row == 1
    bottom = row == Delta.BOARD_ROWS
    left = column == 1
    right = column == Delta.BOARD_COLUMNS
    top && left && return Int32(768)
    top && right && return Int32(784)
    bottom && left && return Int32(800)
    bottom && right && return Int32(816)
    top && return Int32(512)
    bottom && return Int32(576)
    left && return Int32(640)
    right && return Int32(704)
    return Int32(0)
end

@inline function _table2_base(row::Int, column::Int)
    boundary = row == 1 || row == Delta.BOARD_ROWS
    row_prefix = row == 1 ? 0 : row == Delta.BOARD_ROWS ? 14_176 :
                 96 + (row - 2) * 640
    position_prefix = boundary ?
        (column == 1 ? 0 : column == Delta.BOARD_COLUMNS ? 80 : 16) :
        (column == 1 ? 0 : column == Delta.BOARD_COLUMNS ? 576 : 64)
    return Int32(row_prefix + position_prefix)
end

@inline function _table3_base(row::Int, column::Int)
    boundary = column == 1 || column == Delta.BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == Delta.BOARD_COLUMNS ? 5_216 : 96 + (column - 2) * 640
    position_prefix = boundary ?
        (row == 1 ? 0 : row == Delta.BOARD_ROWS ? 80 : 16) :
        (row == 1 ? 0 : row == Delta.BOARD_ROWS ? 576 : 64)
    return Int32(column_prefix + position_prefix)
end

@inline function _table4_base(row::Int, column::Int)
    boundary = column == 1 || column == Delta.BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == Delta.BOARD_COLUMNS ? 92_576 :
        1_440 + (column - 2) * 11_392
    position_prefix = boundary ?
        (row == 1 ? 0 : row == Delta.BOARD_ROWS ? 1_424 :
         16 + (row - 2) * 64) :
        (row == 1 ? 0 : row == Delta.BOARD_ROWS ? 11_328 :
         64 + (row - 2) * 512)
    return Int32(column_prefix + position_prefix)
end

const _TABLE1_BASE = ntuple(Topology.LEAF_COUNT) do position
    row, column = _coordinates(position)
    _table1_base(row, column)
end
const _TABLE2_BASE = ntuple(Topology.LEAF_COUNT) do position
    row, column = _coordinates(position)
    _table2_base(row, column)
end
const _TABLE3_BASE = ntuple(Topology.LEAF_COUNT) do position
    row, column = _coordinates(position)
    _table3_base(row, column)
end
const _TABLE4_BASE = ntuple(Topology.LEAF_COUNT) do position
    row, column = _coordinates(position)
    _table4_base(row, column)
end

@inline function _check_board(board)
    size(board) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) ||
        throw(DimensionMismatch("board must be 24 x 10"))
    return nothing
end

@inline function _check_plane(plane::UInt8)
    plane == BEFORE_PLANE || plane == AFTER_PLANE ||
        throw(ArgumentError("plane must be BEFORE_PLANE or AFTER_PLANE"))
    return nothing
end

"""Exact collision-free four-row semantic address owned by the forest leaf."""
@inline function leaf_program_rows(board, position::Integer, plane::UInt8)
    _check_board(board)
    1 <= position <= Topology.LEAF_COUNT ||
        throw(BoundsError(1:Topology.LEAF_COUNT, position))
    _check_plane(plane)
    physical_position = Int(position)
    row, column = _coordinates(physical_position)
    mask = _occupancy_mask(board, row, column)
    plane_offset = plane == AFTER_PLANE ? Int32(94_016) : Int32(0)
    return Bank.ProgramRows(
        Int32(Bank.TABLE_ROW_OFFSETS[1]) + _TABLE1_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[2]) + _TABLE2_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[3]) + _TABLE3_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[4]) + plane_offset +
            _TABLE4_BASE[physical_position] + mask + 1,
    )
end

@inline function _site(board, row::Int, column::Int)
    if row < 1 || row > Delta.BOARD_ROWS ||
       column < 1 || column > Delta.BOARD_COLUMNS
        return UInt8(2)
    end
    return iszero(@inbounds board[row, column]) ? UInt8(0) : UInt8(1)
end

@inline function _basal_symbol(site::UInt8, ::Type{T}) where {T}
    site == UInt8(0) && return T(-0.25)
    site == UInt8(1) && return T(1.0)
    return T(-0.60)
end

"""Write the active signed 3x3 morphology into eight basal + one apical lane."""
function leaf_drive!(drive::AbstractVector{T}, board, position::Integer) where {T<:AbstractFloat}
    length(drive) == Node.DRIVE_DIM ||
        throw(DimensionMismatch("leaf drive must have $(Node.DRIVE_DIM) lanes"))
    _check_board(board)
    1 <= position <= Topology.LEAF_COUNT ||
        throw(BoundsError(1:Topology.LEAF_COUNT, position))
    row, column = _coordinates(Int(position))
    basal = 0
    @inbounds for column_offset in -1:1, row_offset in -1:1
        iszero(row_offset) && iszero(column_offset) && continue
        basal += 1
        drive[basal] = _basal_symbol(
            _site(board, row + row_offset, column + column_offset),
            T,
        )
    end
    centre = _site(board, row, column)
    drive[end] = centre == UInt8(1) ? T(0.35) : T(-0.35)
    return drive
end

@inline function _prepare_leaf!(
    workspace::PlaneWorkspace{T},
    bank::Bank.ProgramBank,
    board,
    position::Int,
    plane::UInt8,
) where {T}
    rows = leaf_program_rows(board, position, plane)
    Bank.accumulate_active_payload!(workspace.program_payload, bank, rows)
    leaf_drive!(workspace.leaf_drive, board, position)
    @inbounds for lane in 1:Bank.PAYLOAD_WIDTH
        workspace.raw_program[lane] = workspace.default_program[lane] +
                                      T(workspace.program_payload[lane]) *
                                      T(_PROGRAM_SCALE)
    end
    @inbounds for branch in 1:Cell.N_BASAL
        activity = workspace.leaf_drive[branch]
        raw_index = activity >= zero(T) ? branch : Cell.N_BASAL + branch
        workspace.effective_drive[branch] =
            activity * _softplus(workspace.raw_program[raw_index])
    end
    workspace.effective_drive[end] = workspace.leaf_drive[end]
    return rows
end

@inline function _leaf_forward!(
    destination,
    workspace::PlaneWorkspace{T},
    bank,
    board,
    position::Int,
    plane::UInt8,
    shared_cache,
) where {T}
    _prepare_leaf!(workspace, bank, board, position, plane)
    Node.node_forward!(
        destination,
        workspace.leaf_trace,
        workspace.effective_drive,
        shared_cache,
    )
    return @inbounds destination[Node.HARD_EVENT_INDEX]
end

@inline function _child_payload(
    state::PlaneState,
    child::Int,
    lane::Int,
)
    return @inbounds state.payload[lane, child]
end

@inline function _child_payload(
    candidate::COWPlaneState,
    base::PlaneState,
    child::Int,
    lane::Int,
)
    return payload_value(candidate, base, lane, child)
end

@inline function _prepare_internal_drive!(
    drive::AbstractVector{T},
    topology,
    node::Int,
    parameters::PlaneParameters{T},
    child_source,
    base_source=nothing,
) where {T}
    fill!(drive, zero(T))
    first_edge = Int(topology.child_offsets[node])
    child_limit = Int(topology.child_offsets[node + 1]) - 1
    child_slot = 0
    @inbounds for edge in first_edge:child_limit
        child_slot += 1
        child = Int(topology.children[edge])
        total = zero(T)
        for lane in 1:PAYLOAD_DIM
            value = base_source === nothing ?
                _child_payload(child_source, child, lane) :
                _child_payload(child_source, base_source, child, lane)
            total = muladd(parameters.child_contact[lane, edge], value, total)
        end
        drive[child_slot] = total
    end
    return child_limit - first_edge + 1
end

@inline function _add_residual!(payload, topology, node::Int, child_source, base_source=nothing)
    count = Topology.child_count(topology, node)
    inv_count = inv(eltype(payload)(count))
    @inbounds for lane in 1:Node.ANALOG_DIM
        total = zero(eltype(payload))
        for child_index in 1:count
            child = Int(Topology.child_node(topology, node, child_index))
            value = base_source === nothing ?
                _child_payload(child_source, child, lane) :
                _child_payload(child_source, base_source, child, lane)
            total += value
        end
        payload[lane] += total * inv_count
    end
    return payload
end

@inline function _internal_forward!(
    destination,
    workspace::PlaneWorkspace{T},
    topology,
    node::Int,
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    child_source,
    base_source=nothing,
) where {T}
    _prepare_internal_drive!(
        workspace.internal_drive,
        topology,
        node,
        parameters,
        child_source,
        base_source,
    )
    class = _parameter_class(Topology.node_class(topology, node))
    Node.node_forward!(
        destination,
        workspace.internal_trace,
        workspace.internal_drive,
        cache.physical[class],
    )
    _add_residual!(destination, topology, node, child_source, base_source)
    return @inbounds destination[Node.HARD_EVENT_INDEX]
end

"""Evaluate every leaf and internal node of a plane exactly once."""
function base_forward!(
    state::PlaneState{T},
    workspace::PlaneWorkspace{T},
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    bank::Bank.ProgramBank,
    board,
    plane::UInt8,
    shared_cache::Cell.CellParameterCache{T},
    topology::Topology.DeltaForestTopology=Topology.canonical_topology(),
) where {T<:AbstractFloat}
    _check_board(board)
    _check_plane(plane)
    leaf_events = 0
    internal_events = 0
    root_events = 0
    messages = 0
    @inbounds for node in 1:Topology.NODE_COUNT
        destination = @view state.payload[:, node]
        class = Topology.node_class(topology, node)
        event = if class == Topology.LEAF_CLASS
            _leaf_forward!(
                destination,
                workspace,
                bank,
                board,
                node,
                plane,
                shared_cache,
            )
        else
            messages += Topology.child_count(topology, node)
            _internal_forward!(
                destination,
                workspace,
                topology,
                node,
                parameters,
                cache,
                state,
            )
        end
        if class == Topology.LEAF_CLASS
            leaf_events += !iszero(event)
        elseif Topology.is_anchor(topology, node)
            root_events += !iszero(event)
        else
            internal_events += !iszero(event)
        end
    end
    state.stats = PlaneForwardStats(
        leaf_events,
        internal_events,
        root_events,
        Topology.LEAF_COUNT,
        Topology.NODE_COUNT - Topology.LEAF_COUNT,
        messages,
    )
    return state.stats
end

"""Evaluate exactly the affected leaves and their structural ancestor closure."""
function candidate_forward!(
    candidate::COWPlaneState{T},
    base::PlaneState{T},
    workspace::PlaneWorkspace{T},
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    bank::Bank.ProgramBank,
    board,
    plane::UInt8,
    shared_cache::Cell.CellParameterCache{T},
    affected,
    closure::Topology.DirtyClosure,
    topology::Topology.DeltaForestTopology=Topology.canonical_topology(),
) where {T<:AbstractFloat}
    _check_board(board)
    _check_plane(plane)
    Topology.fill_dirty_closure!(closure, topology, affected)
    epoch = _next_epoch!(candidate)
    leaf_events = 0
    internal_events = 0
    root_events = 0
    leaves = 0
    ancestors = 0
    messages = 0
    @inbounds for index in 1:Topology.dirty_count(closure)
        node = Int(Topology.dirty_forward_node(closure, index))
        candidate.generation[node] = epoch
        destination = @view candidate.payload[:, node]
        class = Topology.node_class(topology, node)
        event = if class == Topology.LEAF_CLASS
            leaves += 1
            _leaf_forward!(
                destination,
                workspace,
                bank,
                board,
                node,
                plane,
                shared_cache,
            )
        else
            ancestors += 1
            messages += Topology.child_count(topology, node)
            _internal_forward!(
                destination,
                workspace,
                topology,
                node,
                parameters,
                cache,
                candidate,
                base,
            )
        end
        if class == Topology.LEAF_CLASS
            leaf_events += !iszero(event)
        elseif Topology.is_anchor(topology, node)
            root_events += !iszero(event)
        else
            internal_events += !iszero(event)
        end
    end
    candidate.stats = PlaneForwardStats(
        leaf_events,
        internal_events,
        root_events,
        leaves,
        ancestors,
        messages,
    )
    return candidate.stats
end

function copy_anchor_payloads!(destination, state::PlaneState, topology=Topology.canonical_topology())
    size(destination) == (PAYLOAD_DIM, Topology.ANCHOR_COUNT) ||
        throw(DimensionMismatch("anchor payload must be 3 x 34"))
    @inbounds for anchor in 1:Topology.ANCHOR_COUNT
        node = Int(Topology.anchor_node(anchor))
        for lane in 1:PAYLOAD_DIM
            destination[lane, anchor] = state.payload[lane, node]
        end
    end
    return destination
end

function copy_anchor_payloads!(destination, candidate::COWPlaneState, base::PlaneState, topology=Topology.canonical_topology())
    size(destination) == (PAYLOAD_DIM, Topology.ANCHOR_COUNT) ||
        throw(DimensionMismatch("anchor payload must be 3 x 34"))
    @inbounds for anchor in 1:Topology.ANCHOR_COUNT
        node = Int(Topology.anchor_node(anchor))
        for lane in 1:PAYLOAD_DIM
            destination[lane, anchor] = payload_value(candidate, base, lane, node)
        end
    end
    return destination
end

@inline function _accumulate_leaf_reverse!(
    shared_bar,
    bank_bar,
    workspace::PlaneWorkspace{T},
    bank,
    board,
    position::Int,
    plane::UInt8,
    shared_cache,
    shared_derivative,
    payload_bar,
) where {T}
    rows = _prepare_leaf!(workspace, bank, board, position, plane)
    Node.node_forward!(
        workspace.local_payload,
        workspace.leaf_trace,
        workspace.effective_drive,
        shared_cache,
    )
    workspace.danalog[1] = payload_bar[1]
    workspace.danalog[2] = payload_bar[2]
    Node.node_pullback!(
        workspace.ddrive,
        workspace.draw_node,
        workspace.node_reverse,
        workspace.leaf_trace,
        workspace.effective_drive,
        shared_cache,
        shared_derivative,
        workspace.danalog,
        payload_bar[3],
    )
    fill!(workspace.dprogram, zero(T))
    @inbounds for branch in 1:Cell.N_BASAL
        activity = workspace.leaf_drive[branch]
        raw_index = activity >= zero(T) ? branch : Cell.N_BASAL + branch
        workspace.dprogram[raw_index] +=
            workspace.ddrive[branch] * activity *
            _softplus_derivative(workspace.raw_program[raw_index])
    end
    @inbounds for parameter in 1:Cell.PARAM_DIM
        shared_bar[parameter] += workspace.draw_node[parameter]
    end
    @inbounds for active_index in 1:Bank.active_count(rows)
        row = Int(Bank.active_row(rows, active_index))
        Bank.accumulate_program_gradient!(
            bank_bar,
            row,
            workspace.dprogram,
            T(_PROGRAM_SCALE),
        )
    end
    return nothing
end

@inline function _internal_reverse!(
    gradient::PlaneGradient{T},
    workspace::PlaneWorkspace{T},
    topology,
    node::Int,
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    child_source,
    base_source,
    payload_bar,
    accumulate_child!,
) where {T}
    _prepare_internal_drive!(
        workspace.internal_drive,
        topology,
        node,
        parameters,
        child_source,
        base_source,
    )
    class = _parameter_class(Topology.node_class(topology, node))
    Node.node_forward!(
        workspace.local_payload,
        workspace.internal_trace,
        workspace.internal_drive,
        cache.physical[class],
    )
    workspace.danalog[1] = payload_bar[1]
    workspace.danalog[2] = payload_bar[2]
    Node.node_pullback!(
        workspace.ddrive,
        workspace.draw_node,
        workspace.node_reverse,
        workspace.internal_trace,
        workspace.internal_drive,
        cache.physical[class],
        cache.derivative[class],
        workspace.danalog,
        payload_bar[3],
    )
    @inbounds for parameter in 1:Cell.PARAM_DIM
        gradient.internal_raw[parameter, class] += workspace.draw_node[parameter]
    end

    count = Topology.child_count(topology, node)
    inv_count = inv(T(count))
    first_edge = Int(topology.child_offsets[node])
    @inbounds for child_index in 1:count
        edge = first_edge + child_index - 1
        child = Int(topology.children[edge])
        drive_bar = workspace.ddrive[child_index]
        for lane in 1:PAYLOAD_DIM
            value = base_source === nothing ?
                _child_payload(child_source, child, lane) :
                _child_payload(child_source, base_source, child, lane)
            gradient.child_contact[lane, edge] += drive_bar * value
            child_bar = drive_bar * parameters.child_contact[lane, edge]
            lane <= Node.ANALOG_DIM && (child_bar += payload_bar[lane] * inv_count)
            accumulate_child!(child, lane, child_bar)
        end
    end
    return nothing
end

"""
Reverse one candidate overlay only.  Any cotangent entering a clean child (or
a clean anchor seed) is added to caller-owned `base_accumulator`, enabling one
grouped common-board reverse after all candidates.
"""
function candidate_reverse!(
    gradient::PlaneGradient{T},
    shared_bar::AbstractVector{T},
    bank_bar,
    base_accumulator::AbstractMatrix{T},
    workspace::PlaneWorkspace{T},
    candidate::COWPlaneState{T},
    base::PlaneState{T},
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    bank::Bank.ProgramBank,
    board,
    plane::UInt8,
    shared_cache::Cell.CellParameterCache{T},
    shared_derivative::Cell.CellParameterDerivativeCache{T},
    anchor_bar::AbstractMatrix{T},
    closure::Topology.DirtyClosure,
    topology::Topology.DeltaForestTopology=Topology.canonical_topology(),
) where {T<:AbstractFloat}
    size(base_accumulator) == (PAYLOAD_DIM, Topology.NODE_COUNT) ||
        throw(DimensionMismatch("base accumulator must be 3 x 362"))
    size(anchor_bar) == (PAYLOAD_DIM, Topology.ANCHOR_COUNT) ||
        throw(DimensionMismatch("anchor cotangent must be 3 x 34"))
    _next_bar_epoch!(workspace)
    @inbounds for anchor in 1:Topology.ANCHOR_COUNT
        node = Int(Topology.anchor_node(anchor))
        if _is_current(candidate, node)
            _touch_bar!(workspace, node)
            for lane in 1:PAYLOAD_DIM
                workspace.candidate_bar[lane, node] += anchor_bar[lane, anchor]
            end
        else
            for lane in 1:PAYLOAD_DIM
                base_accumulator[lane, node] += anchor_bar[lane, anchor]
            end
        end
    end

    @inbounds for index in 1:Topology.dirty_count(closure)
        node = Int(Topology.dirty_reverse_node(closure, index))
        _touch_bar!(workspace, node)
        payload_bar = @view workspace.candidate_bar[:, node]
        class = Topology.node_class(topology, node)
        if class == Topology.LEAF_CLASS
            _accumulate_leaf_reverse!(
                shared_bar,
                bank_bar,
                workspace,
                bank,
                board,
                node,
                plane,
                shared_cache,
                shared_derivative,
                payload_bar,
            )
        else
            accumulate_child! = function (child::Int, lane::Int, value::T)
                if _is_current(candidate, child)
                    _touch_bar!(workspace, child)
                    @inbounds workspace.candidate_bar[lane, child] += value
                else
                    @inbounds base_accumulator[lane, child] += value
                end
                return nothing
            end
            _internal_reverse!(
                gradient,
                workspace,
                topology,
                node,
                parameters,
                cache,
                candidate,
                base,
                payload_bar,
                accumulate_child!,
            )
        end
    end
    return gradient, shared_bar, bank_bar, base_accumulator
end

"""Reverse a full plane from a caller-owned arbitrary per-node cotangent."""
function base_reverse!(
    gradient::PlaneGradient{T},
    shared_bar::AbstractVector{T},
    bank_bar,
    node_bar::AbstractMatrix{T},
    workspace::PlaneWorkspace{T},
    state::PlaneState{T},
    parameters::PlaneParameters{T},
    cache::PlaneCache{T},
    bank::Bank.ProgramBank,
    board,
    plane::UInt8,
    shared_cache::Cell.CellParameterCache{T},
    shared_derivative::Cell.CellParameterDerivativeCache{T},
    topology::Topology.DeltaForestTopology=Topology.canonical_topology(),
) where {T<:AbstractFloat}
    size(node_bar) == (PAYLOAD_DIM, Topology.NODE_COUNT) ||
        throw(DimensionMismatch("node cotangent must be 3 x 362"))
    @inbounds for node in Topology.NODE_COUNT:-1:1
        payload_bar = @view node_bar[:, node]
        class = Topology.node_class(topology, node)
        if class == Topology.LEAF_CLASS
            _accumulate_leaf_reverse!(
                shared_bar,
                bank_bar,
                workspace,
                bank,
                board,
                node,
                plane,
                shared_cache,
                shared_derivative,
                payload_bar,
            )
        else
            accumulate_child! = function (child::Int, lane::Int, value::T)
                @inbounds node_bar[lane, child] += value
                return nothing
            end
            _internal_reverse!(
                gradient,
                workspace,
                topology,
                node,
                parameters,
                cache,
                state,
                nothing,
                payload_bar,
                accumulate_child!,
            )
        end
    end
    return gradient, shared_bar, bank_bar, node_bar
end

@inline stored_parameter_count(parameters::PlaneParameters) =
    length(parameters.internal_raw) + length(parameters.child_contact)

end # module DendriticDeltaForest
