module TypedRelationCellBank

using ..ActiveApicalCell
using ..HighDimensionalCellPacket

const Cell = ActiveApicalCell
const Packet = HighDimensionalCellPacket

export RELATION_CELLS,
       PHASE_COUNT,
       RelationParameters,
       RelationCache,
       RelationTape,
       RelationScratch,
       RelationGradient,
       initialize_parameters,
       refresh_cache!,
       clear_gradient!,
       accumulate_gradient!,
       stored_parameter_count,
       hard_event_count,
       hard_event_denominator,
       relation_initial_state!,
       relation_initial_state_pullback!,
       relation_forward!,
       relation_forward_selected!,
       relation_replay_selected!,
       relation_pullback!,
       relation_pullback_selected!

"""Private Reduced Hay cells in the canonical typed relation bank."""
const RELATION_CELLS = 48

"""One mandatory typed-inbox transition per base or candidate overlay."""
const PHASE_COUNT = 1

"""
Independent trainable dynamics for every relation cell.

There is deliberately no shared cell parameter vector: `cell_raw[:, cell]`
owns one complete 48-state `ActiveApicalCell`.  The bank itself adds no dense
projection and no trainable readout to the 47-lane anatomical packet.
"""
struct RelationParameters{T<:AbstractFloat}
    cell_raw::Matrix{T}

    function RelationParameters(cell_raw::Matrix{T}) where {T<:AbstractFloat}
        size(cell_raw) == (Cell.PARAM_DIM, RELATION_CELLS) || throw(
            DimensionMismatch(
                "relation cell parameters must have shape " *
                "($(Cell.PARAM_DIM), $RELATION_CELLS)",
            ),
        )
        return new{T}(cell_raw)
    end
end

function initialize_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    raw = Cell.default_raw_parameters(T)
    cell_raw = Matrix{T}(undef, Cell.PARAM_DIM, RELATION_CELLS)
    @inbounds for cell in 1:RELATION_CELLS, parameter in 1:Cell.PARAM_DIM
        cell_raw[parameter, cell] = raw[parameter]
    end
    return RelationParameters(cell_raw)
end

"""Transformed parameters cached outside every base/candidate hot path."""
mutable struct RelationCache{T<:AbstractFloat}
    cell::Vector{Cell.CellParameterCache{T}}
    derivative::Vector{Cell.CellParameterDerivativeCache{T}}
end

function RelationCache(parameters::RelationParameters{T}) where {T}
    cache = RelationCache(
        Vector{Cell.CellParameterCache{T}}(undef, RELATION_CELLS),
        Vector{Cell.CellParameterDerivativeCache{T}}(
            undef,
            RELATION_CELLS,
        ),
    )
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::RelationCache{T},
    parameters::RelationParameters{T},
) where {T<:AbstractFloat}
    length(cache.cell) == RELATION_CELLS || throw(
        DimensionMismatch("relation cell cache has the wrong length"),
    )
    length(cache.derivative) == RELATION_CELLS || throw(
        DimensionMismatch("relation derivative cache has the wrong length"),
    )
    @inbounds for cell in 1:RELATION_CELLS
        transformed, derivative = Cell.parameter_caches(
            @view(parameters.cell_raw[:, cell]),
        )
        cache.cell[cell] = transformed
        cache.derivative[cell] = derivative
    end
    return cache
end

"""
Fixed structure-of-arrays trajectory for all private cells.

`driven_input` is a receptor-typed `Cell.INPUT_DIM x RELATION_CELLS` inbox.
`states[:, :, 1]` records the explicit caller-owned initial state.  A base pass
supplies parameterized rest; a candidate pass supplies the common base final
state and a delta-only inbox.  The bank copies only selected columns during a
COW forward.  Those recorded columns can subsequently be replayed without
retaining an external sample or allocating a per-cell object.
"""
struct RelationTape{T<:AbstractFloat}
    states::Array{T,3}
    driven_input::Matrix{T}
    events::Matrix{T}

    function RelationTape(
        states::Array{T,3},
        driven_input::Matrix{T},
        events::Matrix{T},
    ) where {T<:AbstractFloat}
        size(states) == (
            Cell.STATE_DIM,
            RELATION_CELLS,
            PHASE_COUNT + 1,
        ) || throw(DimensionMismatch(
            "relation state tape has the wrong shape",
        ))
        size(driven_input) == (Cell.INPUT_DIM, RELATION_CELLS) || throw(
            DimensionMismatch("relation driven-input tape has the wrong shape"),
        )
        size(events) == (PHASE_COUNT, RELATION_CELLS) || throw(
            DimensionMismatch("relation event tape has the wrong shape"),
        )
        return new{T}(states, driven_input, events)
    end
end

function RelationTape(::Type{T}=Float32) where {T<:AbstractFloat}
    return RelationTape(
        Array{T,3}(
            undef,
            Cell.STATE_DIM,
            RELATION_CELLS,
            PHASE_COUNT + 1,
        ),
        Matrix{T}(undef, Cell.INPUT_DIM, RELATION_CELLS),
        Matrix{T}(undef, PHASE_COUNT, RELATION_CELLS),
    )
end

"""One reusable reverse workspace for every source-major cell traversal."""
struct RelationScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw_step::Vector{T}
    dnext::Vector{T}
end

function RelationScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return RelationScratch(
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, Cell.INPUT_DIM),
        Vector{T}(undef, Cell.PARAM_DIM),
        Vector{T}(undef, Cell.STATE_DIM),
    )
end

"""Gradient storage mirrors the cell-private raw parameters exactly."""
struct RelationGradient{T<:AbstractFloat}
    cell_raw::Matrix{T}
end

function RelationGradient(::Type{T}=Float32) where {T<:AbstractFloat}
    return RelationGradient(zeros(T, Cell.PARAM_DIM, RELATION_CELLS))
end

function clear_gradient!(gradient::RelationGradient{T}) where {T}
    fill!(gradient.cell_raw, zero(T))
    return gradient
end

function accumulate_gradient!(
    destination::RelationGradient{T},
    source::RelationGradient{T},
) where {T}
    @inbounds @simd for index in eachindex(destination.cell_raw)
        destination.cell_raw[index] += source.cell_raw[index]
    end
    return destination
end

@inline stored_parameter_count(::RelationParameters) =
    Cell.PARAM_DIM * RELATION_CELLS

@inline function _check_initial_state(initial_state)
    size(initial_state) == (Cell.STATE_DIM, RELATION_CELLS) || throw(
        DimensionMismatch(
            "relation initial state must have shape " *
            "($(Cell.STATE_DIM), $RELATION_CELLS)",
        ),
    )
    return nothing
end

"""
    relation_initial_state!(initial_state, cache)

Fill all relation columns with their private parameterized resting state.
Only a common base pass should call this function.  Candidate COW overlays
must instead receive the common base final state explicitly.
"""
function relation_initial_state!(
    initial_state::AbstractMatrix{T},
    cache::RelationCache{T},
) where {T<:AbstractFloat}
    _check_initial_state(initial_state)
    @inbounds for cell in 1:RELATION_CELLS
        Cell.initial_state!(@view(initial_state[:, cell]), cache.cell[cell])
    end
    return initial_state
end

"""
    relation_initial_state_pullback!(gradient, scratch, dinitial_state, cache)

Accumulate the parameterized-rest VJP for a base pass.  Transition pullbacks
return `dinitial_state` without applying this map, so candidate credit can flow
into the common base trajectory rather than being incorrectly attributed to a
new resting state.
"""
function relation_initial_state_pullback!(
    gradient::RelationGradient{T},
    scratch::RelationScratch{T},
    dinitial_state::AbstractMatrix{T},
    cache::RelationCache{T},
) where {T<:AbstractFloat}
    _check_initial_state(dinitial_state)
    @inbounds for cell in 1:RELATION_CELLS
        fill!(scratch.draw_step, zero(T))
        Cell.initial_state_pullback!(
            scratch.draw_step,
            @view(dinitial_state[:, cell]),
            cache.derivative[cell],
        )
        for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, cell] +=
                scratch.draw_step[parameter]
        end
    end
    return gradient
end

@inline function _cell_step!(
    destination::AbstractVector{Float32},
    state::AbstractVector{Float32},
    input::AbstractVector{Float32},
    cache::Cell.CellParameterCache{Float32},
)
    return Cell.cell_step!(destination, state, input, cache)
end

# Float64 remains a focused finite-difference oracle.  The production
# Float32 path above dispatches to the allocation-free in-place kernel.
function _cell_step!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    copyto!(destination, Cell.cell_step_cached_functional(state, input, cache))
    return destination
end

@inline function _check_output_shapes(packet, hard_event, tape::RelationTape)
    size(packet) == (Packet.PACKET_DIM, RELATION_CELLS) || throw(
        DimensionMismatch(
            "relation packet must have shape " *
            "($(Packet.PACKET_DIM), $RELATION_CELLS)",
        ),
    )
    length(hard_event) == RELATION_CELLS || throw(
        DimensionMismatch(
            "relation hard-event output must have $RELATION_CELLS entries",
        ),
    )
    size(tape.states) == (
        Cell.STATE_DIM,
        RELATION_CELLS,
        PHASE_COUNT + 1,
    ) || throw(DimensionMismatch("relation state tape has the wrong shape"))
    return nothing
end

@inline function _check_inbox(inbox)
    size(inbox) == (Cell.INPUT_DIM, RELATION_CELLS) || throw(
        DimensionMismatch(
            "typed relation inbox must have shape " *
            "($(Cell.INPUT_DIM), $RELATION_CELLS)",
        ),
    )
    return nothing
end

@inline function _check_cells(cells)
    @inbounds for index in eachindex(cells)
        cell = Int(cells[index])
        1 <= cell <= RELATION_CELLS || throw(
            BoundsError(1:RELATION_CELLS, cell),
        )
        for previous_index in eachindex(cells)
            previous_index == index && break
            Int(cells[previous_index]) == cell && throw(
                ArgumentError("relation cell list must not contain duplicates"),
            )
        end
    end
    return nothing
end

@inline function _copy_inbox_column!(
    tape::RelationTape{T},
    inbox::AbstractMatrix{T},
    cell::Int,
) where {T<:AbstractFloat}
    @inbounds @simd for input in 1:Cell.INPUT_DIM
        tape.driven_input[input, cell] = inbox[input, cell]
    end
    return nothing
end

@inline function _copy_initial_column!(
    tape::RelationTape{T},
    initial_state::AbstractMatrix{T},
    cell::Int,
) where {T<:AbstractFloat}
    @inbounds @simd for state in 1:Cell.STATE_DIM
        tape.states[state, cell, 1] = initial_state[state, cell]
    end
    return nothing
end

@inline function _run_cell!(
    packet::AbstractMatrix{T},
    hard_event::AbstractVector{T},
    tape::RelationTape{T},
    cache::RelationCache{T},
    cell::Int,
) where {T<:AbstractFloat}
    previous_state = @view tape.states[:, cell, 1]
    next_state = @view tape.states[:, cell, 2]
    _cell_step!(
        next_state,
        previous_state,
        @view(tape.driven_input[:, cell]),
        cache.cell[cell],
    )
    event = @inbounds next_state[Cell.SPIKE_INDEX]
    @inbounds tape.events[1, cell] = event
    Packet.cell_packet_column!(
        packet,
        tape.states,
        cell,
        1,
        PHASE_COUNT + 1,
        cache.cell[cell],
    )
    hard_event[cell] = event
    return nothing
end

"""
    relation_forward!(packet, hard_event, tape, initial_state, inbox,
                      parameters, cache)

Run all 48 private relation cells from the explicit caller-owned state and
typed inbox.  Every cell performs one mandatory transition. `packet` is a 47-by-48
branch-preserving continuous packet; `hard_event` records the exact spike of
that transition for diagnostics/control only and never gates the analog path.
"""
function relation_forward!(
    packet::AbstractMatrix{T},
    hard_event::AbstractVector{T},
    tape::RelationTape{T},
    initial_state::AbstractMatrix{T},
    inbox::AbstractMatrix{T},
    parameters::RelationParameters{T},
    cache::RelationCache{T},
) where {T<:AbstractFloat}
    _check_output_shapes(packet, hard_event, tape)
    _check_initial_state(initial_state)
    _check_inbox(inbox)
    @inbounds for cell in 1:RELATION_CELLS
        _copy_initial_column!(tape, initial_state, cell)
    end
    copyto!(tape.driven_input, inbox)
    @inbounds for cell in 1:RELATION_CELLS
        _run_cell!(packet, hard_event, tape, cache, cell)
    end
    return packet, hard_event
end

"""
    relation_forward_selected!(..., cells)

COW forward for a unique source-major cell list.  Only selected inbox columns,
trajectories, packet columns and hard-event entries are overwritten.  Every
unselected caller-owned value remains bit-for-bit unchanged.
"""
function relation_forward_selected!(
    packet::AbstractMatrix{T},
    hard_event::AbstractVector{T},
    tape::RelationTape{T},
    initial_state::AbstractMatrix{T},
    inbox::AbstractMatrix{T},
    parameters::RelationParameters{T},
    cache::RelationCache{T},
    cells::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_output_shapes(packet, hard_event, tape)
    _check_initial_state(initial_state)
    _check_inbox(inbox)
    _check_cells(cells)
    @inbounds for index in eachindex(cells)
        cell = Int(cells[index])
        _copy_initial_column!(tape, initial_state, cell)
        _copy_inbox_column!(tape, inbox, cell)
        _run_cell!(packet, hard_event, tape, cache, cell)
    end
    return packet, hard_event
end

"""
    relation_replay_selected!(packet, hard_event, tape, parameters, cache,
                              cells)

Regenerate selected trajectories from their already recorded typed inbox.
This is the fixed-memory replay primitive: it neither reads a dataset row nor
copies an external input, and leaves every unselected cell untouched.
"""
function relation_replay_selected!(
    packet::AbstractMatrix{T},
    hard_event::AbstractVector{T},
    tape::RelationTape{T},
    parameters::RelationParameters{T},
    cache::RelationCache{T},
    cells::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_output_shapes(packet, hard_event, tape)
    _check_cells(cells)
    @inbounds for index in eachindex(cells)
        _run_cell!(
            packet,
            hard_event,
            tape,
            cache,
            Int(cells[index]),
        )
    end
    return packet, hard_event
end

@inline function hard_event_count(tape::RelationTape)
    count = 0
    @inbounds for cell in 1:RELATION_CELLS, phase in 1:PHASE_COUNT
        count += !iszero(tape.events[phase, cell])
    end
    return count
end

@inline hard_event_denominator() = RELATION_CELLS * PHASE_COUNT

@inline function _pullback_cell!(
    dinitial_state::AbstractMatrix{T},
    dinbox::AbstractMatrix{T},
    gradient::RelationGradient{T},
    scratch::RelationScratch{T},
    tape::RelationTape{T},
    cache::RelationCache{T},
    packet_bar::AbstractMatrix{T},
    cell::Int,
) where {T<:AbstractFloat}
    margin_bar = Packet.cell_packet_column_pullback!(
        scratch.dnext,
        packet_bar,
        tape.states,
        cell,
        1,
        PHASE_COUNT + 1,
        cache.cell[cell],
    )
    previous_state = @view tape.states[:, cell, 1]
    next_state = @view tape.states[:, cell, 2]
    Cell.cell_step_conditional_pullback!(
        scratch.dstate,
        scratch.dinput,
        scratch.draw_step,
        previous_state,
        @view(tape.driven_input[:, cell]),
        cache.cell[cell],
        cache.derivative[cell],
        next_state,
        scratch.dnext,
        zero(T),
        zero(T),
        margin_bar,
    )
    @inbounds for parameter in 1:Cell.PARAM_DIM
        gradient.cell_raw[parameter, cell] += scratch.draw_step[parameter]
    end
    @inbounds for input in 1:Cell.INPUT_DIM
        dinbox[input, cell] = scratch.dinput[input]
    end
    @inbounds @simd for state in 1:Cell.STATE_DIM
        dinitial_state[state, cell] = scratch.dstate[state]
    end
    return nothing
end

"""
    relation_pullback!(dinitial_state, dinbox, gradient, scratch, tape,
                       parameters, cache, packet_bar)

Exact VJP of every continuous 47-lane packet, conditional on the recorded hard
event sequence.  The hard OR event has no cotangent and cannot inject a spike
surrogate into this task gradient. `dinitial_state` and `dinbox` are
overwritten and `gradient` is accumulated. The initial-state cotangent is not
mapped to parameterized rest here; candidate callers must return it to their
common base trajectory.
"""
function relation_pullback!(
    dinitial_state::AbstractMatrix{T},
    dinbox::AbstractMatrix{T},
    gradient::RelationGradient{T},
    scratch::RelationScratch{T},
    tape::RelationTape{T},
    parameters::RelationParameters{T},
    cache::RelationCache{T},
    packet_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_initial_state(dinitial_state)
    _check_inbox(dinbox)
    size(packet_bar) == (Packet.PACKET_DIM, RELATION_CELLS) || throw(
        DimensionMismatch("relation packet cotangent has the wrong shape"),
    )
    fill!(dinitial_state, zero(T))
    fill!(dinbox, zero(T))
    @inbounds for cell in 1:RELATION_CELLS
        _pullback_cell!(
            dinitial_state,
            dinbox,
            gradient,
            scratch,
            tape,
            cache,
            packet_bar,
            cell,
        )
    end
    return dinitial_state, dinbox, gradient
end

"""
    relation_pullback_selected!(..., cells)

Selected-cell exact VJP paired with `relation_forward_selected!` or replay.
Only selected initial-state-bar, inbox-bar and private cell-gradient columns
are written; unselected storage is never cleared or otherwise modified.
"""
function relation_pullback_selected!(
    dinitial_state::AbstractMatrix{T},
    dinbox::AbstractMatrix{T},
    gradient::RelationGradient{T},
    scratch::RelationScratch{T},
    tape::RelationTape{T},
    parameters::RelationParameters{T},
    cache::RelationCache{T},
    packet_bar::AbstractMatrix{T},
    cells::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_initial_state(dinitial_state)
    _check_inbox(dinbox)
    size(packet_bar) == (Packet.PACKET_DIM, RELATION_CELLS) || throw(
        DimensionMismatch("relation packet cotangent has the wrong shape"),
    )
    _check_cells(cells)
    @inbounds for index in eachindex(cells)
        _pullback_cell!(
            dinitial_state,
            dinbox,
            gradient,
            scratch,
            tape,
            cache,
            packet_bar,
            Int(cells[index]),
        )
    end
    return dinitial_state, dinbox, gradient
end

end # module TypedRelationCellBank
