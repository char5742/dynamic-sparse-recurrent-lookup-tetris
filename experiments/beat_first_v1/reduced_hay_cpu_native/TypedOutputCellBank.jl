module TypedOutputCellBank

using ..ActiveApicalCell
using ..HighDimensionalCellPacket

const Cell = ActiveApicalCell
const Packet = HighDimensionalCellPacket

export OUTPUT_CELLS,
       READOUT_DIM,
       TypedOutputParameters,
       TypedOutputCache,
       TypedOutputTape,
       TypedOutputScratch,
       TypedOutputGradient,
       initialize_parameters,
       refresh_cache!,
       output_initial_state!,
       output_initial_state_pullback!,
       clear_gradient!,
       accumulate_gradient!,
       stored_parameter_count,
       hard_event_count,
       hard_event_denominator,
       typed_output_forward!,
       typed_output_pullback!

"""One private high-dimensional Reduced Hay cell per supervised output."""
const OUTPUT_CELLS = 22

"""The complete branch-preserving packet read by each local output."""
const READOUT_DIM = Packet.PACKET_DIM

"""
Trainable parameters of the typed output-cell bank.

`cell_raw[:, output]` owns the complete 48-state Reduced Hay dynamics for one
output. `readout_weight[:, output]` reads that same cell's complete 47-lane
anatomical packet. There is no cross-output projection, branch averaging, soma
margin shortcut or hard-event task parameter.
"""
struct TypedOutputParameters{T<:AbstractFloat}
    cell_raw::Matrix{T}
    readout_weight::Matrix{T}
    bias::Vector{T}

    function TypedOutputParameters(
        cell_raw::Matrix{T},
        readout_weight::Matrix{T},
        bias::Vector{T},
    ) where {T<:AbstractFloat}
        size(cell_raw) == (Cell.PARAM_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "cell raw parameters must have shape ($(Cell.PARAM_DIM), 22)",
            ),
        )
        size(readout_weight) == (READOUT_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "readout weights must have shape ($READOUT_DIM, 22)",
            ),
        )
        length(bias) == OUTPUT_CELLS || throw(
            DimensionMismatch("bias must have 22 values"),
        )
        return new{T}(cell_raw, readout_weight, bias)
    end
end

"""
Create a deterministic, physically scaled initial bank. Every anatomical lane
starts with equal nonzero gain, so the first task VJP reaches all eight branch
voltages and all eight signed slow-state lanes. The weights remain independent
and trainable thereafter.
"""
function initialize_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    raw = Cell.default_raw_parameters(T)
    cell_raw = Matrix{T}(undef, Cell.PARAM_DIM, OUTPUT_CELLS)
    @inbounds for output in 1:OUTPUT_CELLS, parameter in 1:Cell.PARAM_DIM
        cell_raw[parameter, output] = raw[parameter]
    end
    initial_readout_gain = inv(T(READOUT_DIM))
    return TypedOutputParameters(
        cell_raw,
        fill(initial_readout_gain, READOUT_DIM, OUTPUT_CELLS),
        zeros(T, OUTPUT_CELLS),
    )
end

"""Transformed cell parameters cached outside the candidate hot path."""
mutable struct TypedOutputCache{T<:AbstractFloat}
    cell::Vector{Cell.CellParameterCache{T}}
    derivative::Vector{Cell.CellParameterDerivativeCache{T}}
end

function TypedOutputCache(parameters::TypedOutputParameters{T}) where {T}
    cache = TypedOutputCache(
        Vector{Cell.CellParameterCache{T}}(undef, OUTPUT_CELLS),
        Vector{Cell.CellParameterDerivativeCache{T}}(undef, OUTPUT_CELLS),
    )
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::TypedOutputCache{T},
    parameters::TypedOutputParameters{T},
) where {T<:AbstractFloat}
    length(cache.cell) == OUTPUT_CELLS || throw(
        DimensionMismatch("cell cache must have 22 entries"),
    )
    length(cache.derivative) == OUTPUT_CELLS || throw(
        DimensionMismatch("cell derivative cache must have 22 entries"),
    )
    @inbounds for output in 1:OUTPUT_CELLS
        transformed, derivative = Cell.parameter_caches(
            @view(parameters.cell_raw[:, output]),
        )
        cache.cell[output] = transformed
        cache.derivative[output] = derivative
    end
    return cache
end

"""
    output_initial_state!(state, cache)

Materialize the parameterized resting state once for a common/base pass.  A
candidate pass must receive the resulting state explicitly; it never resets
itself to rest.
"""
function output_initial_state!(
    state::AbstractMatrix{T},
    cache::TypedOutputCache{T},
) where {T<:AbstractFloat}
    size(state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "initial output state must have shape ($(Cell.STATE_DIM), 22)",
        ),
    )
    @inbounds for output in 1:OUTPUT_CELLS
        Cell.initial_state!(@view(state[:, output]), cache.cell[output])
    end
    return state
end

"""
Caller-owned structure-of-arrays trajectory.

The inbox is already receptor typed and has exactly `Cell.INPUT_DIM` rows:
AMPA, NMDA and GABA for each of eight basal plus one apical compartment.  Every
candidate performs exactly one mandatory transition from a caller-supplied
base state.  No hidden initialization, relaxation phase or event-dependent
skip exists here.
"""
struct TypedOutputTape{T<:AbstractFloat}
    base_state::Matrix{T}
    next_state::Matrix{T}
    inbox::Matrix{T}
    packet::Matrix{T}
    event::Vector{T}

    function TypedOutputTape(
        base_state::Matrix{T},
        next_state::Matrix{T},
        inbox::Matrix{T},
        packet::Matrix{T},
        event::Vector{T},
    ) where {T<:AbstractFloat}
        size(base_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "base-state tape must have shape ($(Cell.STATE_DIM), 22)",
            ),
        )
        size(next_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "next-state tape must have shape ($(Cell.STATE_DIM), 22)",
            ),
        )
        size(inbox) == (Cell.INPUT_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "typed inbox tape must have shape ($(Cell.INPUT_DIM), 22)",
            ),
        )
        size(packet) == (READOUT_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "output packet tape must have shape ($READOUT_DIM, 22)",
            ),
        )
        length(event) == OUTPUT_CELLS || throw(
            DimensionMismatch("event tape must have 22 values"),
        )
        return new{T}(
            base_state,
            next_state,
            inbox,
            packet,
            event,
        )
    end
end

function TypedOutputTape(::Type{T}=Float32) where {T<:AbstractFloat}
    return TypedOutputTape(
        Matrix{T}(undef, Cell.STATE_DIM, OUTPUT_CELLS),
        Matrix{T}(undef, Cell.STATE_DIM, OUTPUT_CELLS),
        Matrix{T}(undef, Cell.INPUT_DIM, OUTPUT_CELLS),
        Matrix{T}(undef, READOUT_DIM, OUTPUT_CELLS),
        Vector{T}(undef, OUTPUT_CELLS),
    )
end

"""One reusable reverse workspace; no cell-sized object is created per call."""
struct TypedOutputScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw_step::Vector{T}
    dnext::Vector{T}
    packet_bar::Vector{T}
end

function TypedOutputScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return TypedOutputScratch(
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, Cell.INPUT_DIM),
        Vector{T}(undef, Cell.PARAM_DIM),
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, READOUT_DIM),
    )
end

"""Gradient storage mirrors the three trainable parameter groups exactly."""
struct TypedOutputGradient{T<:AbstractFloat}
    cell_raw::Matrix{T}
    readout_weight::Matrix{T}
    bias::Vector{T}
end

function TypedOutputGradient(::Type{T}=Float32) where {T<:AbstractFloat}
    return TypedOutputGradient(
        zeros(T, Cell.PARAM_DIM, OUTPUT_CELLS),
        zeros(T, READOUT_DIM, OUTPUT_CELLS),
        zeros(T, OUTPUT_CELLS),
    )
end

function clear_gradient!(gradient::TypedOutputGradient{T}) where {T}
    fill!(gradient.cell_raw, zero(T))
    fill!(gradient.readout_weight, zero(T))
    fill!(gradient.bias, zero(T))
    return gradient
end

"""
    output_initial_state_pullback!(gradient, scratch, dstate, cache)

Accumulate parameter credit from a common/base resting-state boundary.  The
canonical integrator calls this once after summing every candidate's
`dbase_state`; candidate kernels themselves never apply this pullback.
"""
function output_initial_state_pullback!(
    gradient::TypedOutputGradient{T},
    scratch::TypedOutputScratch{T},
    dstate::AbstractMatrix{T},
    cache::TypedOutputCache{T},
) where {T<:AbstractFloat}
    size(dstate) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "initial-state cotangent must have shape ($(Cell.STATE_DIM), 22)",
        ),
    )
    @inbounds for output in 1:OUTPUT_CELLS
        fill!(scratch.draw_step, zero(T))
        Cell.initial_state_pullback!(
            scratch.draw_step,
            @view(dstate[:, output]),
            cache.derivative[output],
        )
        for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output] += scratch.draw_step[parameter]
        end
    end
    return gradient
end

function accumulate_gradient!(
    destination::TypedOutputGradient{T},
    source::TypedOutputGradient{T},
) where {T}
    @inbounds @simd for index in eachindex(destination.cell_raw)
        destination.cell_raw[index] += source.cell_raw[index]
    end
    @inbounds @simd for index in eachindex(destination.readout_weight)
        destination.readout_weight[index] += source.readout_weight[index]
    end
    @inbounds @simd for output in 1:OUTPUT_CELLS
        destination.bias[output] += source.bias[output]
    end
    return destination
end

@inline stored_parameter_count(::TypedOutputParameters) =
    Cell.PARAM_DIM * OUTPUT_CELLS + READOUT_DIM * OUTPUT_CELLS + OUTPUT_CELLS

@inline function _cell_step!(
    destination::AbstractVector{Float32},
    state::AbstractVector{Float32},
    input::AbstractVector{Float32},
    cache::Cell.CellParameterCache{Float32},
)
    return Cell.cell_step!(destination, state, input, cache)
end

# Float64 is an exact-derivative oracle.  The production Float32 dispatch
# above is the allocation-free in-place kernel.
function _cell_step!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    copyto!(destination, Cell.cell_step_cached_functional(state, input, cache))
    return destination
end

@inline function _check_forward_shapes(
    continuous::AbstractVector,
    hard_event::AbstractVector,
    tape::TypedOutputTape,
    base_state::AbstractMatrix,
    inbox::AbstractMatrix,
)
    length(continuous) == OUTPUT_CELLS || throw(
        DimensionMismatch("continuous output must have 22 values"),
    )
    length(hard_event) == OUTPUT_CELLS || throw(
        DimensionMismatch("hard-event output must have 22 values"),
    )
    size(base_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "base state must have shape ($(Cell.STATE_DIM), 22)",
        ),
    )
    size(inbox) == (Cell.INPUT_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "typed inbox must have shape ($(Cell.INPUT_DIM), 22)",
        ),
    )
    size(tape.base_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch("output-cell base-state tape has the wrong shape"),
    )
    size(tape.next_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch("output-cell next-state tape has the wrong shape"),
    )
    size(tape.packet) == (READOUT_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch("output-cell packet tape has the wrong shape"),
    )
    return nothing
end

"""
    typed_output_forward!(continuous, hard_event, tape, base_state, inbox,
                          parameters, cache)

Run 22 independent Reduced Hay cells.  The caller supplies the already-typed
`Cell.INPUT_DIM x 22` receptor inbox and a `Cell.STATE_DIM x 22` base state.
Each cell performs one mandatory transition. Its complete branch-preserving
packet is then read by a cell-local linear functional:

```
dot(readout_weight[:, output], packet[:, output]) + bias[output]
```

No parameter connects different outputs and no branch coordinate is averaged
before learning. The second result is the exact hard event of that transition.
It is a control signal only and is deliberately absent from the task VJP.
"""
function typed_output_forward!(
    continuous::AbstractVector{T},
    hard_event::AbstractVector{T},
    tape::TypedOutputTape{T},
    base_state::AbstractMatrix{T},
    inbox::AbstractMatrix{T},
    parameters::TypedOutputParameters{T},
    cache::TypedOutputCache{T},
) where {T<:AbstractFloat}
    _check_forward_shapes(continuous, hard_event, tape, base_state, inbox)
    copyto!(tape.base_state, base_state)
    copyto!(tape.inbox, inbox)

    @inbounds for output in 1:OUTPUT_CELLS
        _cell_step!(
            @view(tape.next_state[:, output]),
            @view(tape.base_state[:, output]),
            @view(tape.inbox[:, output]),
            cache.cell[output],
        )
        Packet.cell_packet!(
            @view(tape.packet[:, output]),
            @view(tape.base_state[:, output]),
            @view(tape.next_state[:, output]),
            cache.cell[output],
        )
        tape.event[output] = tape.next_state[Cell.SPIKE_INDEX, output]
        value = parameters.bias[output]
        for lane in 1:READOUT_DIM
            value = muladd(
                parameters.readout_weight[lane, output],
                tape.packet[lane, output],
                value,
            )
        end
        continuous[output] = value
        hard_event[output] = tape.event[output]
    end
    return continuous, hard_event
end

@inline function hard_event_count(tape::TypedOutputTape{T}) where {T}
    count = 0
    @inbounds for output in 1:OUTPUT_CELLS
        count += !iszero(tape.event[output])
    end
    return count
end

@inline hard_event_denominator() = OUTPUT_CELLS

"""
    typed_output_pullback!(dbase_state, dinbox, gradient, scratch, tape,
                           parameters, cache, continuous_bar)

Exact reverse of the continuous output conditional on the recorded hard-event
sequence.  Cotangents are returned for the caller-supplied base state, typed
receptor inbox, all private cell raw parameters, the complete local readout
vector and the bias. There is no hard event cotangent: hard events remain
control decisions and cannot silently inject a surrogate into the task VJP.

`dbase_state` and `dinbox` are overwritten.  `gradient` is accumulated,
allowing a caller to reduce several candidate gradients without temporary
allocations.  Parameter dependence of the base-state producer is deliberately
owned by that producer; this one-step kernel differentiates the supplied base
state as an ordinary input.
"""
function typed_output_pullback!(
    dbase_state::AbstractMatrix{T},
    dinbox::AbstractMatrix{T},
    gradient::TypedOutputGradient{T},
    scratch::TypedOutputScratch{T},
    tape::TypedOutputTape{T},
    parameters::TypedOutputParameters{T},
    cache::TypedOutputCache{T},
    continuous_bar::AbstractVector{T},
) where {T<:AbstractFloat}
    size(dbase_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "base-state cotangent must have shape ($(Cell.STATE_DIM), 22)",
        ),
    )
    size(dinbox) == (Cell.INPUT_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "typed inbox cotangent must have shape ($(Cell.INPUT_DIM), 22)",
        ),
    )
    length(continuous_bar) == OUTPUT_CELLS || throw(
        DimensionMismatch("continuous cotangent must have 22 values"),
    )
    fill!(dbase_state, zero(T))
    fill!(dinbox, zero(T))

    @inbounds for output in 1:OUTPUT_CELLS
        output_bar = continuous_bar[output]
        gradient.bias[output] += output_bar

        for lane in 1:READOUT_DIM
            packet_value = tape.packet[lane, output]
            gradient.readout_weight[lane, output] +=
                output_bar * packet_value
            scratch.packet_bar[lane] =
                output_bar * parameters.readout_weight[lane, output]
        end
        margin_bar = Packet.cell_packet_pullback!(
            scratch.dnext,
            scratch.packet_bar,
            @view(tape.base_state[:, output]),
            @view(tape.next_state[:, output]),
            cache.cell[output],
        )

        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            @view(tape.base_state[:, output]),
            @view(tape.inbox[:, output]),
            cache.cell[output],
            cache.derivative[output],
            @view(tape.next_state[:, output]),
            scratch.dnext,
            zero(T),
            zero(T),
            margin_bar,
        )
        for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output] +=
                scratch.draw_step[parameter]
        end
        for channel in 1:Cell.INPUT_DIM
            dinbox[channel, output] = scratch.dinput[channel]
        end
        # `ActiveApicalCell` exposes an identity eligibility for a previous
        # spike so local event learners can train the bAP path.  This bank is
        # the analog task-VJP boundary: the supplied base spike is a hard
        # control bit, whose exact almost-everywhere derivative is zero.
        scratch.dstate[Cell.SPIKE_INDEX] = zero(T)
        for state in 1:Cell.STATE_DIM
            dbase_state[state, output] = scratch.dstate[state]
        end
    end
    return dbase_state, dinbox, gradient
end

end # module TypedOutputCellBank
