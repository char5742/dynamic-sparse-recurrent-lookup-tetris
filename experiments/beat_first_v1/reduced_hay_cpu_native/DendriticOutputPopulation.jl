module DendriticOutputPopulation

using ..ActiveApicalCell
using ..DendriticAxonPacket

const Cell = ActiveApicalCell
const Axon = DendriticAxonPacket

export OUTPUT_DIM,
       OUTPUT_CELLS,
       EVIDENCE_DIM,
       MAX_EVIDENCE,
       PROJECTION_PARAMETER_COUNT,
       ROLE_COUNT,
       ROLE_VALUE,
       ROLE_ADVANTAGE,
       ROLE_DEATH,
       ROLE_GEOMETRY,
       ROLE_UNCERTAINTY,
       VALUE_CELLS,
       ADVANTAGE_CELLS,
       DEATH_CELLS,
       GEOMETRY_CELLS,
       UNCERTAINTY_CELLS,
       Q_INDEX,
       DEATH_INDEX,
       QUANTILE_RANGE,
       GEOMETRY_RANGE,
       QUANTILE_COEFFICIENTS,
       OutputComponents,
       OutputComponentGradient,
       OutputPopulationParameters,
       OutputPopulationCache,
       OutputPopulationTape,
       OutputPopulationScratch,
       OutputPopulationGradient,
       initialize_parameters,
       refresh_cache!,
       output_initial_state!,
       output_initial_state_pullback!,
       clear_gradient!,
       clear_component_gradient!,
       accumulate_gradient!,
       stored_parameter_count,
       evidence_lane,
       cell_role,
       uncertainty_scale,
       assemble_output!,
       q_cotangent,
       assemble_output_pullback!,
       hard_event_count,
       hard_event_denominator,
       output_population_forward!,
       output_population_pullback!

"""External ranking ABI: Q, death, sixteen quantiles and four geometry values."""
const OUTPUT_DIM = 22

"""Twenty-two private Reduced-Hay cells; no cell is shared by two roles."""
const OUTPUT_CELLS = 22

"""Width of the bounded, nonnegative axon packet arriving on one semantic edge."""
const EVIDENCE_DIM = Axon.PACKET_DIM

"""One evidence source is assigned to each of the eight basal compartments."""
const MAX_EVIDENCE = Cell.N_BASAL

const ROLE_VALUE = 1
const ROLE_ADVANTAGE = 2
const ROLE_DEATH = 3
const ROLE_GEOMETRY = 4
const ROLE_UNCERTAINTY = 5
const ROLE_COUNT = 5

"""Four groups times three receptors times five roles; no cross-type map."""
const PROJECTION_PARAMETER_COUNT =
    Axon.GROUP_COUNT * Cell.INPUT_CHANNELS * ROLE_COUNT

const VALUE_CELLS = 1:2
const ADVANTAGE_CELLS = 3:10
const DEATH_CELLS = 11:12
const GEOMETRY_CELLS = 13:20
const UNCERTAINTY_CELLS = 21:22

const Q_INDEX = 1
const DEATH_INDEX = 2
const QUANTILE_RANGE = 3:18
const GEOMETRY_RANGE = 19:22
const QUANTILE_COUNT = length(QUANTILE_RANGE)
const GEOMETRY_COUNT = length(GEOMETRY_RANGE)

"""
Fixed ordered offsets for the compatibility quantile ABI.

The current teacher is deterministic, so these coefficients are deliberately
symmetric around zero.  Every nonzero uncertainty scale separates at least one
quantile from Q and the unique limiting optimum is therefore `sigma -> 0`.
"""
const QUANTILE_COEFFICIENTS = ntuple(
    quantile -> Float32(2 * quantile - QUANTILE_COUNT - 1) /
                Float32(QUANTILE_COUNT),
    Val(QUANTILE_COUNT),
)

@assert OUTPUT_DIM == 1 + 1 + QUANTILE_COUNT + GEOMETRY_COUNT
@assert OUTPUT_CELLS == 2 + 8 + 2 + 8 + 2
@assert MAX_EVIDENCE == 8

"""
The seven semantic scalars emitted by the physical output population.

`advantage` is uncentered.  Candidate-set centering is owned by
[`assemble_output!`](@ref), after every candidate of one Tetris state has been
evaluated.  Hiding the mean inside one candidate forward would make the model
mathematically incorrect.
"""
mutable struct OutputComponents{T<:AbstractFloat}
    value::T
    advantage::T
    death::T
    geometry::Vector{T}
    uncertainty_raw::T

    function OutputComponents(
        value::T,
        advantage::T,
        death::T,
        geometry::Vector{T},
        uncertainty_raw::T,
    ) where {T<:AbstractFloat}
        length(geometry) == GEOMETRY_COUNT || throw(DimensionMismatch(
            "geometry component must have $GEOMETRY_COUNT values",
        ))
        return new{T}(value, advantage, death, geometry, uncertainty_raw)
    end
end

OutputComponents(::Type{T}=Float32) where {T<:AbstractFloat} =
    OutputComponents(zero(T), zero(T), zero(T), zeros(T, GEOMETRY_COUNT), zero(T))

"""Caller-owned cotangents for [`OutputComponents`](@ref)."""
mutable struct OutputComponentGradient{T<:AbstractFloat}
    value::T
    advantage::T
    death::T
    geometry::Vector{T}
    uncertainty_raw::T

    function OutputComponentGradient(
        value::T,
        advantage::T,
        death::T,
        geometry::Vector{T},
        uncertainty_raw::T,
    ) where {T<:AbstractFloat}
        length(geometry) == GEOMETRY_COUNT || throw(DimensionMismatch(
            "geometry cotangent must have $GEOMETRY_COUNT values",
        ))
        return new{T}(value, advantage, death, geometry, uncertainty_raw)
    end
end

OutputComponentGradient(::Type{T}=Float32) where {T<:AbstractFloat} =
    OutputComponentGradient(
        zero(T), zero(T), zero(T), zeros(T, GEOMETRY_COUNT), zero(T),
    )

function clear_component_gradient!(gradient::OutputComponentGradient{T}) where {T}
    gradient.value = zero(T)
    gradient.advantage = zero(T)
    gradient.death = zero(T)
    fill!(gradient.geometry, zero(T))
    gradient.uncertainty_raw = zero(T)
    return gradient
end

"""
The only trainable output-boundary parameters.

Every cell owns complete Reduced-Hay dynamics.  The five semantic roles own a
small positive receptor-diagonal `12 -> 3` projection shared by cells of that
role.  Its 60 parameters are indexed `(group, receptor, role)`: AMPA evidence
can affect only AMPA, NMDA only NMDA and GABA only GABA.  There is intentionally
no cross-receptor, raw-input, dense all-source, residual, packet decoder,
output gain or output bias parameter.
"""
struct OutputPopulationParameters{T<:AbstractFloat}
    cell_raw::Matrix{T}
    projection_raw::Array{T,3}

    function OutputPopulationParameters(
        cell_raw::Matrix{T},
        projection_raw::Array{T,3},
    ) where {T<:AbstractFloat}
        size(cell_raw) == (Cell.PARAM_DIM, OUTPUT_CELLS) || throw(
            DimensionMismatch(
                "cell raw parameters must have shape " *
                "($(Cell.PARAM_DIM), $OUTPUT_CELLS)",
            ),
        )
        size(projection_raw) ==
            (Axon.GROUP_COUNT, Cell.INPUT_CHANNELS, ROLE_COUNT) || throw(
            DimensionMismatch(
                "typed projection must have shape " *
                "($(Axon.GROUP_COUNT), $(Cell.INPUT_CHANNELS), $ROLE_COUNT)",
            ),
        )
        return new{T}(cell_raw, projection_raw)
    end
end

@inline function _softplus(value::T) where {T<:AbstractFloat}
    return max(value, zero(T)) + log1p(exp(-abs(value)))
end

@inline function _softplus_derivative(value::T) where {T<:AbstractFloat}
    if value >= zero(T)
        return inv(one(T) + exp(-value))
    end
    exponential = exp(value)
    return exponential / (one(T) + exponential)
end

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("softplus target must be positive"))
    return log(expm1(value))
end

@inline function _projection_initial_value(
    ::Type{T},
    group::Int,
    receptor::Int,
    role::Int,
) where {T<:AbstractFloat}
    # Group, receptor and role factors break exact symmetry while retaining a
    # physically diagonal receptor map.  No cross-type fallback exists.
    base = T(0.055)
    group_factor = one(T) +
                   T(0.015) * T(2 * group - Axon.GROUP_COUNT - 1)
    receptor_factor = one(T) + T(0.01) * T(receptor - 2)
    role_factor = one(T) + T(0.025) * T(role - 3)
    return base * group_factor * receptor_factor * role_factor
end

function initialize_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    default_raw = Cell.default_raw_parameters(T)
    cell_raw = Matrix{T}(undef, Cell.PARAM_DIM, OUTPUT_CELLS)
    @inbounds for output_cell in 1:OUTPUT_CELLS,
                  parameter in 1:Cell.PARAM_DIM
        cell_raw[parameter, output_cell] = default_raw[parameter]
    end
    projection_raw = Array{T,3}(
        undef,
        Axon.GROUP_COUNT,
        Cell.INPUT_CHANNELS,
        ROLE_COUNT,
    )
    @inbounds for role in 1:ROLE_COUNT,
                  receptor in 1:Cell.INPUT_CHANNELS,
                  group in 1:Axon.GROUP_COUNT
        projection_raw[group, receptor, role] = _inverse_softplus(
            _projection_initial_value(T, group, receptor, role),
        )
    end
    return OutputPopulationParameters(cell_raw, projection_raw)
end

"""Transformed cell and positive receptor-projection parameters."""
mutable struct OutputPopulationCache{T<:AbstractFloat}
    cell::Vector{Cell.CellParameterCache{T}}
    derivative::Vector{Cell.CellParameterDerivativeCache{T}}
    projection::Array{T,3}
    projection_derivative::Array{T,3}
end

function OutputPopulationCache(
    parameters::OutputPopulationParameters{T},
) where {T<:AbstractFloat}
    cache = OutputPopulationCache(
        Vector{Cell.CellParameterCache{T}}(undef, OUTPUT_CELLS),
        Vector{Cell.CellParameterDerivativeCache{T}}(undef, OUTPUT_CELLS),
        Array{T,3}(
            undef,
            Axon.GROUP_COUNT,
            Cell.INPUT_CHANNELS,
            ROLE_COUNT,
        ),
        Array{T,3}(
            undef,
            Axon.GROUP_COUNT,
            Cell.INPUT_CHANNELS,
            ROLE_COUNT,
        ),
    )
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::OutputPopulationCache{T},
    parameters::OutputPopulationParameters{T},
) where {T<:AbstractFloat}
    @inbounds for output_cell in 1:OUTPUT_CELLS
        transformed, derivative = Cell.parameter_caches(
            @view(parameters.cell_raw[:, output_cell]),
        )
        cache.cell[output_cell] = transformed
        cache.derivative[output_cell] = derivative
    end
    @inbounds @simd for index in eachindex(parameters.projection_raw)
        raw = parameters.projection_raw[index]
        cache.projection[index] = _softplus(raw)
        cache.projection_derivative[index] = _softplus_derivative(raw)
    end
    return cache
end

@inline function cell_role(output_cell::Integer)
    cell = Int(output_cell)
    1 <= cell <= OUTPUT_CELLS || throw(BoundsError(1:OUTPUT_CELLS, cell))
    return cell <= last(VALUE_CELLS) ? ROLE_VALUE :
           cell <= last(ADVANTAGE_CELLS) ? ROLE_ADVANTAGE :
           cell <= last(DEATH_CELLS) ? ROLE_DEATH :
           cell <= last(GEOMETRY_CELLS) ? ROLE_GEOMETRY :
           ROLE_UNCERTAINTY
end

function output_initial_state!(
    state::AbstractMatrix{T},
    cache::OutputPopulationCache{T},
) where {T<:AbstractFloat}
    size(state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "initial output state must have shape " *
            "($(Cell.STATE_DIM), $OUTPUT_CELLS)",
        ),
    )
    @inbounds for output_cell in 1:OUTPUT_CELLS
        Cell.initial_state!(@view(state[:, output_cell]), cache.cell[output_cell])
    end
    return state
end

"""Fixed-size forward trajectory; no candidate-local object is created."""
struct OutputPopulationTape{T<:AbstractFloat}
    base_state::Matrix{T}
    next_state::Matrix{T}
    inbox::Matrix{T}
    evidence::Array{T,3}
    evidence_count::Vector{UInt8}
    margin::Vector{T}
    event::Vector{T}
end

function OutputPopulationTape(::Type{T}=Float32) where {T<:AbstractFloat}
    return OutputPopulationTape(
        Matrix{T}(undef, Cell.STATE_DIM, OUTPUT_CELLS),
        Matrix{T}(undef, Cell.STATE_DIM, OUTPUT_CELLS),
        zeros(T, Cell.INPUT_DIM, OUTPUT_CELLS),
        zeros(T, EVIDENCE_DIM, MAX_EVIDENCE, OUTPUT_CELLS),
        zeros(UInt8, OUTPUT_CELLS),
        Vector{T}(undef, OUTPUT_CELLS),
        Vector{T}(undef, OUTPUT_CELLS),
    )
end

"""One caller-owned reverse workspace for all twenty-two cells."""
struct OutputPopulationScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw_step::Vector{T}
    dnext::Vector{T}
    margin_bar::Vector{T}
end

function OutputPopulationScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return OutputPopulationScratch(
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, Cell.INPUT_DIM),
        Vector{T}(undef, Cell.PARAM_DIM),
        zeros(T, Cell.STATE_DIM),
        zeros(T, OUTPUT_CELLS),
    )
end

struct OutputPopulationGradient{T<:AbstractFloat}
    cell_raw::Matrix{T}
    projection_raw::Array{T,3}
end

function OutputPopulationGradient(::Type{T}=Float32) where {T<:AbstractFloat}
    return OutputPopulationGradient(
        zeros(T, Cell.PARAM_DIM, OUTPUT_CELLS),
        zeros(T, Axon.GROUP_COUNT, Cell.INPUT_CHANNELS, ROLE_COUNT),
    )
end

function clear_gradient!(gradient::OutputPopulationGradient{T}) where {T}
    fill!(gradient.cell_raw, zero(T))
    fill!(gradient.projection_raw, zero(T))
    return gradient
end

function accumulate_gradient!(
    destination::OutputPopulationGradient{T},
    source::OutputPopulationGradient{T},
) where {T<:AbstractFloat}
    @inbounds @simd for index in eachindex(destination.cell_raw)
        destination.cell_raw[index] += source.cell_raw[index]
    end
    @inbounds @simd for index in eachindex(destination.projection_raw)
        destination.projection_raw[index] += source.projection_raw[index]
    end
    return destination
end

@inline stored_parameter_count(::OutputPopulationParameters) =
    Cell.PARAM_DIM * OUTPUT_CELLS +
    PROJECTION_PARAMETER_COUNT

"""Canonical lane `3*(group-1)+receptor` in the 12D axon packet."""
@inline function evidence_lane(group::Integer, receptor::Integer)
    1 <= group <= Axon.GROUP_COUNT ||
        throw(BoundsError(1:Axon.GROUP_COUNT, group))
    1 <= receptor <= Cell.INPUT_CHANNELS ||
        throw(BoundsError(1:Cell.INPUT_CHANNELS, receptor))
    return Cell.INPUT_CHANNELS * (Int(group) - 1) + Int(receptor)
end

function output_initial_state_pullback!(
    gradient::OutputPopulationGradient{T},
    scratch::OutputPopulationScratch{T},
    dstate::AbstractMatrix{T},
    cache::OutputPopulationCache{T},
) where {T<:AbstractFloat}
    size(dstate) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "initial-state cotangent must have shape " *
            "($(Cell.STATE_DIM), $OUTPUT_CELLS)",
        ),
    )
    @inbounds for output_cell in 1:OUTPUT_CELLS
        fill!(scratch.draw_step, zero(T))
        Cell.initial_state_pullback!(
            scratch.draw_step,
            @view(dstate[:, output_cell]),
            cache.derivative[output_cell],
        )
        for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output_cell] +=
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

# Float64 exists only for finite-difference and exact-oracle tests.
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
    hard_event::AbstractVector,
    tape::OutputPopulationTape,
    base_state::AbstractMatrix,
    evidence::AbstractArray,
    evidence_count::AbstractVector,
)
    length(hard_event) == OUTPUT_CELLS || throw(DimensionMismatch(
        "hard-event output must have $OUTPUT_CELLS values",
    ))
    size(base_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "base state must have shape ($(Cell.STATE_DIM), $OUTPUT_CELLS)",
        ),
    )
    size(evidence) == (EVIDENCE_DIM, MAX_EVIDENCE, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "evidence must have shape " *
            "($EVIDENCE_DIM, $MAX_EVIDENCE, $OUTPUT_CELLS)",
        ),
    )
    length(evidence_count) == OUTPUT_CELLS || throw(DimensionMismatch(
        "evidence count must have $OUTPUT_CELLS entries",
    ))
    return nothing
end

@inline function _populate_components!(
    components::OutputComponents{T},
    margin::AbstractVector{T},
) where {T<:AbstractFloat}
    inverse_sqrt_two = inv(sqrt(T(2)))
    inverse_sqrt_eight = inv(sqrt(T(8)))
    components.value = (margin[1] - margin[2]) * inverse_sqrt_two

    advantage = zero(T)
    @inbounds for offset in 1:length(ADVANTAGE_CELLS)
        coefficient = isodd(offset) ? one(T) : -one(T)
        advantage = muladd(
            coefficient,
            margin[first(ADVANTAGE_CELLS) + offset - 1],
            advantage,
        )
    end
    components.advantage = advantage * inverse_sqrt_eight
    components.death = (margin[11] - margin[12]) * inverse_sqrt_two
    @inbounds for geometry in 1:GEOMETRY_COUNT
        first_cell = first(GEOMETRY_CELLS) + 2 * (geometry - 1)
        components.geometry[geometry] =
            (margin[first_cell] - margin[first_cell + 1]) * inverse_sqrt_two
    end
    components.uncertainty_raw = (margin[21] - margin[22]) * inverse_sqrt_two
    return components
end

"""
    output_population_forward!(components, hard_event, tape, base_state,
                               evidence, evidence_count, parameters, cache)

Run the mandatory analog transition of twenty-two private output cells.  Each
used evidence source supplies one nonnegative 12D packet to one distinct basal
branch.  Its cell-role projection produces three positive receptor-typed
conductances.  No hard event gates this transition.

The sole numeric readout of every cell is the exact continuous pre-reset soma
margin `V_pre - theta`.  Hard spikes are copied to `hard_event` only as control
events and receive no task cotangent here.
"""
function output_population_forward!(
    components::OutputComponents{T},
    hard_event::AbstractVector{T},
    tape::OutputPopulationTape{T},
    base_state::AbstractMatrix{T},
    evidence::AbstractArray{T,3},
    evidence_count::AbstractVector{UInt8},
    parameters::OutputPopulationParameters{T},
    cache::OutputPopulationCache{T},
) where {T<:AbstractFloat}
    _check_forward_shapes(hard_event, tape, base_state, evidence, evidence_count)
    copyto!(tape.base_state, base_state)
    copyto!(tape.evidence, evidence)
    copyto!(tape.evidence_count, evidence_count)
    fill!(tape.inbox, zero(T))

    @inbounds for output_cell in 1:OUTPUT_CELLS
        source_count = Int(evidence_count[output_cell])
        source_count <= MAX_EVIDENCE || throw(ArgumentError(
            "output cell $output_cell has $source_count evidence sources; " *
            "maximum is $MAX_EVIDENCE",
        ))
        role = cell_role(output_cell)
        for source in 1:source_count
            for receptor in 1:Cell.INPUT_CHANNELS
                typed_input = zero(T)
                for group in 1:Axon.GROUP_COUNT
                    lane = evidence_lane(group, receptor)
                    packet_value = evidence[lane, source, output_cell]
                    isfinite(packet_value) || throw(ArgumentError(
                        "output evidence must be finite",
                    ))
                    packet_value >= zero(T) || throw(ArgumentError(
                        "12D axon evidence must be nonnegative",
                    ))
                    typed_input = muladd(
                        cache.projection[group, receptor, role],
                        packet_value,
                        typed_input,
                    )
                end
                tape.inbox[
                    Cell.input_index(source, receptor),
                    output_cell,
                ] = typed_input
            end
        end

        _cell_step!(
            @view(tape.next_state[:, output_cell]),
            @view(tape.base_state[:, output_cell]),
            @view(tape.inbox[:, output_cell]),
            cache.cell[output_cell],
        )
        margin = Cell.spike_margin_from_transition(
            @view(tape.base_state[:, output_cell]),
            @view(tape.next_state[:, output_cell]),
            cache.cell[output_cell],
        )
        event = tape.next_state[Cell.SPIKE_INDEX, output_cell]
        tape.margin[output_cell] = margin
        tape.event[output_cell] = event
        hard_event[output_cell] = event
    end
    return _populate_components!(components, tape.margin), hard_event
end

@inline function uncertainty_scale(components::OutputComponents{T}) where {T}
    return _softplus(components.uncertainty_raw)
end

"""
    assemble_output!(output, components, advantage_mean)

Assemble the external 22D ABI after all candidate advantages of one state are
known.  `advantage_mean` is the arithmetic mean across those candidates, never
a mean across the eight physical advantage cells.
"""
function assemble_output!(
    output::AbstractVector{T},
    components::OutputComponents{T},
    advantage_mean::T,
) where {T<:AbstractFloat}
    length(output) == OUTPUT_DIM || throw(DimensionMismatch(
        "assembled output must have $OUTPUT_DIM values",
    ))
    q = components.value + components.advantage - advantage_mean
    sigma = uncertainty_scale(components)
    @inbounds begin
        output[Q_INDEX] = q
        output[DEATH_INDEX] = components.death
        for quantile in 1:QUANTILE_COUNT
            output[first(QUANTILE_RANGE) + quantile - 1] = muladd(
                T(QUANTILE_COEFFICIENTS[quantile]),
                sigma,
                q,
            )
        end
        for geometry in 1:GEOMETRY_COUNT
            output[first(GEOMETRY_RANGE) + geometry - 1] =
                components.geometry[geometry]
        end
    end
    return output
end

"""Total Q cotangent carried by raw Q and all sixteen compatibility quantiles."""
@inline function q_cotangent(output_bar::AbstractVector{T}) where {T<:AbstractFloat}
    length(output_bar) == OUTPUT_DIM || throw(DimensionMismatch(
        "output cotangent must have $OUTPUT_DIM values",
    ))
    total = output_bar[Q_INDEX]
    @inbounds @simd for output in QUANTILE_RANGE
        total += output_bar[output]
    end
    return total
end

"""
    assemble_output_pullback!(component_bar, output_bar, components,
                              centered_advantage_bar) -> q_bar

Reverse the 22D assembly.  The caller must provide the candidate-set-centered
advantage cotangent.  For candidates `a` of one state it is

```
centered_advantage_bar[a] = q_bar[a] - mean(q_bar)
```

Requiring it explicitly prevents a single-candidate kernel from silently
dropping the derivative of the dueling mean.  The returned `q_bar` is what a
shared/cached value population must accumulate over candidates.
"""
function assemble_output_pullback!(
    component_bar::OutputComponentGradient{T},
    output_bar::AbstractVector{T},
    components::OutputComponents{T},
    centered_advantage_bar::T,
) where {T<:AbstractFloat}
    length(output_bar) == OUTPUT_DIM || throw(DimensionMismatch(
        "output cotangent must have $OUTPUT_DIM values",
    ))
    clear_component_gradient!(component_bar)
    q_bar = q_cotangent(output_bar)
    component_bar.value = q_bar
    component_bar.advantage = centered_advantage_bar
    component_bar.death = output_bar[DEATH_INDEX]
    @inbounds for geometry in 1:GEOMETRY_COUNT
        component_bar.geometry[geometry] =
            output_bar[first(GEOMETRY_RANGE) + geometry - 1]
    end
    sigma_bar = zero(T)
    @inbounds for quantile in 1:QUANTILE_COUNT
        sigma_bar = muladd(
            T(QUANTILE_COEFFICIENTS[quantile]),
            output_bar[first(QUANTILE_RANGE) + quantile - 1],
            sigma_bar,
        )
    end
    component_bar.uncertainty_raw =
        sigma_bar * _softplus_derivative(components.uncertainty_raw)
    return q_bar
end

@inline function _components_to_margin_bar!(
    margin_bar::AbstractVector{T},
    components_bar::OutputComponentGradient{T},
) where {T<:AbstractFloat}
    fill!(margin_bar, zero(T))
    inverse_sqrt_two = inv(sqrt(T(2)))
    inverse_sqrt_eight = inv(sqrt(T(8)))
    @inbounds begin
        margin_bar[1] = components_bar.value * inverse_sqrt_two
        margin_bar[2] = -components_bar.value * inverse_sqrt_two
        for offset in 1:length(ADVANTAGE_CELLS)
            coefficient = isodd(offset) ? one(T) : -one(T)
            margin_bar[first(ADVANTAGE_CELLS) + offset - 1] =
                components_bar.advantage * coefficient * inverse_sqrt_eight
        end
        margin_bar[11] = components_bar.death * inverse_sqrt_two
        margin_bar[12] = -components_bar.death * inverse_sqrt_two
        for geometry in 1:GEOMETRY_COUNT
            first_cell = first(GEOMETRY_CELLS) + 2 * (geometry - 1)
            value = components_bar.geometry[geometry] * inverse_sqrt_two
            margin_bar[first_cell] = value
            margin_bar[first_cell + 1] = -value
        end
        margin_bar[21] = components_bar.uncertainty_raw * inverse_sqrt_two
        margin_bar[22] = -components_bar.uncertainty_raw * inverse_sqrt_two
    end
    return margin_bar
end

"""
Exact conditional reverse of [`output_population_forward!`](@ref).

`dbase_state` and `devidence` are overwritten; `gradient` is accumulated.  The
hard-event cotangent is exactly zero.  Projection credit reaches only the 60
role-specific receptor-diagonal gains, never a cross-receptor or
source-by-output dense path.
"""
function output_population_pullback!(
    dbase_state::AbstractMatrix{T},
    devidence::AbstractArray{T,3},
    gradient::OutputPopulationGradient{T},
    scratch::OutputPopulationScratch{T},
    tape::OutputPopulationTape{T},
    parameters::OutputPopulationParameters{T},
    cache::OutputPopulationCache{T},
    components_bar::OutputComponentGradient{T},
) where {T<:AbstractFloat}
    size(dbase_state) == (Cell.STATE_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "base-state cotangent must have shape " *
            "($(Cell.STATE_DIM), $OUTPUT_CELLS)",
        ),
    )
    size(devidence) == (EVIDENCE_DIM, MAX_EVIDENCE, OUTPUT_CELLS) || throw(
        DimensionMismatch(
            "evidence cotangent must have shape " *
            "($EVIDENCE_DIM, $MAX_EVIDENCE, $OUTPUT_CELLS)",
        ),
    )
    size(parameters.projection_raw) ==
        (Axon.GROUP_COUNT, Cell.INPUT_CHANNELS, ROLE_COUNT) || throw(
        DimensionMismatch("projection parameter shape changed"),
    )
    fill!(dbase_state, zero(T))
    fill!(devidence, zero(T))
    _components_to_margin_bar!(scratch.margin_bar, components_bar)

    @inbounds for output_cell in 1:OUTPUT_CELLS
        fill!(scratch.dnext, zero(T))
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            @view(tape.base_state[:, output_cell]),
            @view(tape.inbox[:, output_cell]),
            cache.cell[output_cell],
            cache.derivative[output_cell],
            @view(tape.next_state[:, output_cell]),
            scratch.dnext,
            zero(T),
            zero(T),
            scratch.margin_bar[output_cell],
        )
        for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output_cell] +=
                scratch.draw_step[parameter]
        end
        # The previous hard spike is a control coordinate.  Exact continuous
        # task credit must not create a surrogate hard-event path.
        scratch.dstate[Cell.SPIKE_INDEX] = zero(T)
        for state in 1:Cell.STATE_DIM
            dbase_state[state, output_cell] = scratch.dstate[state]
        end

        role = cell_role(output_cell)
        source_count = Int(tape.evidence_count[output_cell])
        for source in 1:source_count
            for receptor in 1:Cell.INPUT_CHANNELS
                input_bar = scratch.dinput[
                    Cell.input_index(source, receptor),
                ]
                for group in 1:Axon.GROUP_COUNT
                    lane = evidence_lane(group, receptor)
                    packet_value = tape.evidence[lane, source, output_cell]
                    devidence[lane, source, output_cell] +=
                        input_bar * cache.projection[group, receptor, role]
                    gradient.projection_raw[group, receptor, role] +=
                        input_bar * packet_value *
                        cache.projection_derivative[group, receptor, role]
                end
            end
        end
    end
    return dbase_state, devidence, gradient
end

@inline function hard_event_count(tape::OutputPopulationTape{T}) where {T}
    count = 0
    @inbounds for output_cell in 1:OUTPUT_CELLS
        count += !iszero(tape.event[output_cell])
    end
    return count
end

@inline hard_event_denominator() = OUTPUT_CELLS

end # module DendriticOutputPopulation
