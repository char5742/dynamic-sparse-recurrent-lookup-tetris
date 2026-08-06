module SensoryEncoder

using ..Architecture
using ..ActiveApicalCell: INPUT_AMPA,
    INPUT_DIM,
    INPUT_GABA,
    INPUT_NMDA,
    N_COMPARTMENTS,
    input_index

export BLOCKS,
    BOARD_COLUMNS,
    BOARD_PLANES,
    BOARD_RAILS,
    BOARD_ROWS,
    CELLS_PER_BLOCK,
    EXCITATORY_COORDINATES,
    EXCITATORY_GAIN,
    EXCITATORY_GAIN_MAX,
    INHIBITORY_GAIN,
    INHIBITORY_GAIN_MAX,
    INPUT_RAILS,
    NONBOARD_RAILS,
    RailDestination,
    board_rail_index,
    bounded_gain,
    bounded_gain_derivative,
    default_raw_gains,
    destination_linear_index,
    encode_sensory,
    encode_sensory!,
    encode_sensory_cached!,
    excitatory_destination,
    associative_destination,
    gaba_destination,
    associative_gaba_destination,
    rail_code,
    scatter_add_sensory!,
    scatter_add_sensory_cached!,
    sensory_cached_raw_vjp!,
    sensory_vjp!,
    transform_sensory_gains!

const INPUT_RAILS = 1_298
const CELLS_PER_BLOCK = Architecture.CELLS_PER_BLOCK
const SPATIAL_POSITIONS_PER_BLOCK = Architecture.SPATIAL_POSITIONS_PER_BLOCK
const LANES_PER_POSITION = Architecture.LANES_PER_POSITION
const BLOCKS = Architecture.BLOCK_COUNT
const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_PLANES = 4
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const BOARD_RAILS = BOARD_PLANES * BOARD_CELLS
const NONBOARD_RAILS = INPUT_RAILS - BOARD_RAILS

const EXCITATORY_GAIN = 1
const INHIBITORY_GAIN = 2
const GAIN_ROWS = 2

const EXCITATORY_GAIN_MAX = 2.0f0
const INHIBITORY_GAIN_MAX = 1.0f0
const DEFAULT_EXCITATORY_GAIN = 0.50f0
const DEFAULT_INHIBITORY_GAIN = 0.05f0

# Every cell has one AMPA and one NMDA coordinate for each compartment.
const EXCITATORY_CHANNELS_PER_COMPARTMENT = 2
const EXCITATORY_COORDINATES =
    BLOCKS * CELLS_PER_BLOCK * N_COMPARTMENTS *
    EXCITATORY_CHANNELS_PER_COMPARTMENT

# The four spatial planes consume AMPA/NMDA in basal compartments 1 and 2 at
# the exact matching board position.  Queue and auxiliary rails use only the
# four AMPA/NMDA coordinates in basal compartments 3/4. Additional basal
# compartments and the final active apical compartment are recurrent-only and
# receive no direct sensory current.
const NONBOARD_COMPARTMENT_FIRST = 3
const NONBOARD_COMPARTMENTS = 2
const NONBOARD_EXCITATORY_COORDINATES =
    BLOCKS * SPATIAL_POSITIONS_PER_BLOCK * NONBOARD_COMPARTMENTS *
    EXCITATORY_CHANNELS_PER_COMPARTMENT

# The associative lane receives the same rail identity in independent cells.
# Its separation is therefore on the cell axis, not the branch axis: both
# lanes use the strongly coupled sensory compartments 1--4 and retain
# compartments 5--8 for recurrent/context traffic.  Mapping the second lane
# into 5--8 would silently attenuate all of its sensory evidence because those
# branches intentionally start with a weak recurrent-only role.
const ASSOCIATIVE_CELL_OFFSET = SPATIAL_POSITIONS_PER_BLOCK

# 887 is coprime to 960.  Its permutation spreads the 338 non-spatial rails
# through otherwise unused basal coordinates without colliding with one
# another or with the exact board map.  Only these non-spatial rails use a
# permutation; board identity is never scrambled and apical input stays zero.
const NONBOARD_STRIDE = 887

struct RailDestination
    input::Int
    cell::Int
    block::Int
end

@inline function _check_rail(rail::Integer)
    1 <= rail <= INPUT_RAILS || throw(BoundsError(1:INPUT_RAILS, rail))
    return Int(rail)
end

"""
    board_rail_index(plane, row, column)

Return the TetrisRankingBatch rail index for one board coordinate.  Planes are
`1=board`, `2=after`, `3=added`, `4=removed`; positions follow the packer's
column-major `(column, row)` traversal.
"""
@inline function board_rail_index(plane::Integer, row::Integer, column::Integer)
    1 <= plane <= BOARD_PLANES || throw(BoundsError(1:BOARD_PLANES, plane))
    1 <= row <= BOARD_ROWS || throw(BoundsError(1:BOARD_ROWS, row))
    1 <= column <= BOARD_COLUMNS || throw(BoundsError(1:BOARD_COLUMNS, column))
    return (Int(plane) - 1) * BOARD_CELLS +
           (Int(column) - 1) * BOARD_ROWS + Int(row)
end

"""
    excitatory_destination(rail) -> RailDestination

Map one sensory rail to its unique AMPA or NMDA coordinate in the canonical
`input[Cell.INPUT_DIM, 8, 30]` cell-input tensor.  Each group of eight vertical board rows
is one model block, giving three blocks per Tetris column.  The four board
planes remain colocated at that exact block/cell but use distinct receptors.
Queue and auxiliary rails occupy collision-free coordinates in basal
compartments 3--4. Remaining compartments are reserved for recurrent/context input.
The mapping is deterministic and has no parameters or gradient.
"""
@inline function excitatory_destination(rail::Integer)
    rail_index = _check_rail(rail) - 1
    if rail_index < BOARD_RAILS
        plane_index, board_position = divrem(rail_index, BOARD_CELLS)
        column_index, row_index = divrem(board_position, BOARD_ROWS)
        vertical_block, cell_index = divrem(
            row_index,
            SPATIAL_POSITIONS_PER_BLOCK,
        )
        compartment_index, receptor_index =
            divrem(plane_index, EXCITATORY_CHANNELS_PER_COMPARTMENT)
        receptor = receptor_index == 0 ? INPUT_AMPA : INPUT_NMDA
        return RailDestination(
            input_index(compartment_index + 1, receptor),
            cell_index + 1,
            column_index * 3 + vertical_block + 1,
        )
    end

    nonboard_index = rail_index - BOARD_RAILS
    coordinate = mod(
        nonboard_index * NONBOARD_STRIDE,
        NONBOARD_EXCITATORY_COORDINATES,
    )
    cell_position, local_excitatory = divrem(
        coordinate,
        NONBOARD_COMPARTMENTS * EXCITATORY_CHANNELS_PER_COMPARTMENT,
    )
    block_index, cell_index = divrem(
        cell_position,
        SPATIAL_POSITIONS_PER_BLOCK,
    )
    compartment_index, receptor_index =
        divrem(local_excitatory, EXCITATORY_CHANNELS_PER_COMPARTMENT)
    receptor = receptor_index == 0 ? INPUT_AMPA : INPUT_NMDA
    return RailDestination(
        input_index(
            NONBOARD_COMPARTMENT_FIRST + compartment_index,
            receptor,
        ),
        cell_index + 1,
        block_index + 1,
    )
end

"""Exact rail projection into the independent associative cell lane."""
@inline function associative_destination(rail::Integer)
    primary = excitatory_destination(rail)
    return RailDestination(
        primary.input,
        primary.cell + ASSOCIATIVE_CELL_OFFSET,
        primary.block,
    )
end

"""
    gaba_destination(rail) -> RailDestination

Return the inhibitory companion of a rail.  It is the GABA coordinate in the
same cell and compartment as the rail's unique excitatory destination.  AMPA
and NMDA coordinates in the same compartment intentionally share this local
inhibitory companion, so encoding uses scatter-add.
"""
@inline function gaba_destination(rail::Integer)
    excitatory = excitatory_destination(rail)
    compartment = div(excitatory.input - 1, 3) + 1
    return RailDestination(
        input_index(compartment, INPUT_GABA),
        excitatory.cell,
        excitatory.block,
    )
end


@inline function associative_gaba_destination(rail::Integer)
    excitatory = associative_destination(rail)
    compartment = div(excitatory.input - 1, 3) + 1
    return RailDestination(
        input_index(compartment, INPUT_GABA),
        excitatory.cell,
        excitatory.block,
    )
end

"""Linear Julia index of a destination in an `input[Cell.INPUT_DIM, 8, 30]` tensor."""
@inline function destination_linear_index(destination::RailDestination)
    return destination.input +
           INPUT_DIM * (destination.cell - 1) +
           INPUT_DIM * CELLS_PER_BLOCK * (destination.block - 1)
end

"""
Stable fixed rail identity.  It is the exact tensor index of the rail's unique
excitatory destination; consequently all 1,298 codes are distinct.
"""
@inline rail_code(rail::Integer) = destination_linear_index(excitatory_destination(rail))

@inline function _sigmoid(raw::T) where {T<:AbstractFloat}
    if raw >= zero(T)
        inverse = exp(-raw)
        return inv(one(T) + inverse)
    end
    exponential = exp(raw)
    return exponential / (one(T) + exponential)
end

@inline function _gain_max(::Type{T}, gain::Integer) where {T<:AbstractFloat}
    gain == EXCITATORY_GAIN && return T(EXCITATORY_GAIN_MAX)
    gain == INHIBITORY_GAIN && return T(INHIBITORY_GAIN_MAX)
    throw(BoundsError(1:GAIN_ROWS, gain))
end

"""Transform an unconstrained raw gain to its bounded nonnegative amplitude."""
@inline function bounded_gain(raw::T, gain::Integer) where {T<:AbstractFloat}
    return _gain_max(T, gain) * _sigmoid(raw)
end

"""Exact derivative of [`bounded_gain`](@ref) with respect to its raw value."""
@inline function bounded_gain_derivative(raw::T, gain::Integer) where {T<:AbstractFloat}
    probability = _sigmoid(raw)
    return _gain_max(T, gain) * probability * (one(T) - probability)
end

@inline function _raw_for_gain(value::T, maximum::T) where {T<:AbstractFloat}
    probability = clamp(value / maximum, eps(T), one(T) - eps(T))
    return log(probability / (one(T) - probability))
end

"""
    default_raw_gains([T=Float32])

Create the caller-owned `2 x 1298` raw gain matrix.  Row 1 is excitatory and
row 2 inhibitory.  Initial transformed gains are 0.50 and 0.05 respectively:
excitation is dominant, while inhibition is present from the first update.
"""
function default_raw_gains(::Type{T}=Float32) where {T<:AbstractFloat}
    raw = Matrix{T}(undef, GAIN_ROWS, INPUT_RAILS)
    excitatory_raw = _raw_for_gain(T(DEFAULT_EXCITATORY_GAIN), T(EXCITATORY_GAIN_MAX))
    inhibitory_raw = _raw_for_gain(T(DEFAULT_INHIBITORY_GAIN), T(INHIBITORY_GAIN_MAX))
    @inbounds for rail in 1:INPUT_RAILS
        raw[EXCITATORY_GAIN, rail] = excitatory_raw
        raw[INHIBITORY_GAIN, rail] = inhibitory_raw
    end
    return raw
end

@inline function _check_destination(destination::AbstractArray)
    ndims(destination) == 3 ||
        throw(DimensionMismatch("sensory destination must have three axes"))
    size(destination, 1) == INPUT_DIM ||
        throw(DimensionMismatch("sensory input axis must have length $INPUT_DIM"))
    size(destination, 2) == CELLS_PER_BLOCK ||
        throw(DimensionMismatch("sensory cell axis must have length $CELLS_PER_BLOCK"))
    size(destination, 3) == BLOCKS ||
        throw(DimensionMismatch("sensory block axis must have length $BLOCKS"))
    return nothing
end

@inline function _check_rails(rails::AbstractVector)
    length(rails) == INPUT_RAILS ||
        throw(DimensionMismatch("sensory rail vector must have length $INPUT_RAILS"))
    return nothing
end

@inline function _check_raw_gains(raw_gains::AbstractMatrix)
    size(raw_gains, 1) == GAIN_ROWS ||
        throw(DimensionMismatch("raw sensory gains must have two rows"))
    size(raw_gains, 2) == INPUT_RAILS ||
        throw(DimensionMismatch("raw sensory gains must have $INPUT_RAILS columns"))
    return nothing
end

"""
    transform_sensory_gains!(gains, raw_derivatives, raw_gains)

Prepare the sensory parameter cache at the model publish boundary.  Both
caller-owned outputs have the same `2 x 1298` shape as `raw_gains`.  The first
stores bounded gains and the second stores their exact derivatives with
respect to the raw parameters.  Production forward and reverse kernels consume
only these caches and therefore never evaluate a sigmoid per candidate.
"""
function transform_sensory_gains!(
    gains::AbstractMatrix{T},
    raw_derivatives::AbstractMatrix{T},
    raw_gains::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_raw_gains(gains)
    _check_raw_gains(raw_derivatives)
    _check_raw_gains(raw_gains)
    @inbounds for rail in 1:INPUT_RAILS
        for gain in 1:GAIN_ROWS
            raw = raw_gains[gain, rail]
            isfinite(raw) ||
                throw(ArgumentError("raw sensory gains must be finite"))
            gains[gain, rail] = bounded_gain(raw, gain)
            raw_derivatives[gain, rail] = bounded_gain_derivative(raw, gain)
        end
    end
    return gains, raw_derivatives
end

"""
    scatter_add_sensory_cached!(destination, rails, gains)

Production scatter using gains prepared by [`transform_sensory_gains!`](@ref).
Only fixed-size shape checks are performed here; finiteness and transform
validity are publish-boundary invariants.
"""
function scatter_add_sensory_cached!(
    destination::AbstractArray{T,3},
    rails::AbstractVector{R},
    gains::AbstractMatrix{G},
) where {T<:AbstractFloat,R<:AbstractFloat,G<:AbstractFloat}
    _check_destination(destination)
    _check_rails(rails)
    _check_raw_gains(gains)

    @inbounds for rail in 1:INPUT_RAILS
        excitatory = excitatory_destination(rail)
        inhibitory = gaba_destination(rail)
        associative_excitatory = associative_destination(rail)
        associative_inhibitory = associative_gaba_destination(rail)
        rail_value = rails[rail]
        destination[excitatory.input, excitatory.cell, excitatory.block] +=
            T(rail_value * gains[EXCITATORY_GAIN, rail])
        destination[inhibitory.input, inhibitory.cell, inhibitory.block] +=
            T(rail_value * gains[INHIBITORY_GAIN, rail])
        destination[
            associative_excitatory.input,
            associative_excitatory.cell,
            associative_excitatory.block,
        ] += T(rail_value * gains[EXCITATORY_GAIN, rail])
        destination[
            associative_inhibitory.input,
            associative_inhibitory.cell,
            associative_inhibitory.block,
        ] += T(rail_value * gains[INHIBITORY_GAIN, rail])
    end
    return destination
end

"""Clear and encode with a publish-boundary sensory gain cache."""
function encode_sensory_cached!(
    destination::AbstractArray{T,3},
    rails::AbstractVector{R},
    gains::AbstractMatrix{G},
) where {T<:AbstractFloat,R<:AbstractFloat,G<:AbstractFloat}
    _check_destination(destination)
    fill!(destination, zero(T))
    return scatter_add_sensory_cached!(destination, rails, gains)
end

"""
    scatter_add_sensory!(destination, rails, raw_gains)

Add all rail currents to a pre-existing canonical input tensor.  The loop is
fixed-cost `O(1298)`, performs no allocation, and does not clear `destination`.
Each rail writes one unique AMPA/NMDA coordinate and its local GABA companion.
"""
function scatter_add_sensory!(
    destination::AbstractArray{T,3},
    rails::AbstractVector{R},
    raw_gains::AbstractMatrix{G},
) where {T<:AbstractFloat,R<:AbstractFloat,G<:AbstractFloat}
    _check_destination(destination)
    _check_rails(rails)
    _check_raw_gains(raw_gains)

    @inbounds for rail in 1:INPUT_RAILS
        excitatory = excitatory_destination(rail)
        inhibitory = gaba_destination(rail)
        associative_excitatory = associative_destination(rail)
        associative_inhibitory = associative_gaba_destination(rail)
        rail_value = rails[rail]
        excitatory_amplitude = bounded_gain(raw_gains[EXCITATORY_GAIN, rail], EXCITATORY_GAIN)
        inhibitory_amplitude = bounded_gain(raw_gains[INHIBITORY_GAIN, rail], INHIBITORY_GAIN)
        destination[excitatory.input, excitatory.cell, excitatory.block] +=
            T(rail_value * excitatory_amplitude)
        destination[inhibitory.input, inhibitory.cell, inhibitory.block] +=
            T(rail_value * inhibitory_amplitude)
        destination[
            associative_excitatory.input,
            associative_excitatory.cell,
            associative_excitatory.block,
        ] += T(rail_value * excitatory_amplitude)
        destination[
            associative_inhibitory.input,
            associative_inhibitory.cell,
            associative_inhibitory.block,
        ] += T(rail_value * inhibitory_amplitude)
    end
    return destination
end

"""Clear and encode into caller-owned canonical storage."""
function encode_sensory!(
    destination::AbstractArray{T,3},
    rails::AbstractVector{R},
    raw_gains::AbstractMatrix{G},
) where {T<:AbstractFloat,R<:AbstractFloat,G<:AbstractFloat}
    _check_destination(destination)
    fill!(destination, zero(T))
    return scatter_add_sensory!(destination, rails, raw_gains)
end

"""Allocating reference encoder used outside the fixed-arena hot path."""
function encode_sensory(
    rails::AbstractVector{R},
    raw_gains::AbstractMatrix{G},
) where {R<:AbstractFloat,G<:AbstractFloat}
    T = promote_type(R, G)
    destination = zeros(T, INPUT_DIM, CELLS_PER_BLOCK, BLOCKS)
    return scatter_add_sensory!(destination, rails, raw_gains)
end

"""
    sensory_vjp!(rail_bar, raw_gain_bar, input_bar, rails, raw_gains)

Exact analytic reverse pass for the sensory scatter.  The fixed destination
mapping has no cotangent.  Gradient arrays are overwritten, not accumulated.
"""
function sensory_vjp!(
    rail_bar::AbstractVector{TR},
    raw_gain_bar::AbstractMatrix{TG},
    input_bar::AbstractArray{TI,3},
    rails::AbstractVector{R},
    raw_gains::AbstractMatrix{G},
) where {
    TR<:AbstractFloat,
    TG<:AbstractFloat,
    TI<:AbstractFloat,
    R<:AbstractFloat,
    G<:AbstractFloat,
}
    _check_rails(rail_bar)
    _check_raw_gains(raw_gain_bar)
    _check_destination(input_bar)
    _check_rails(rails)
    _check_raw_gains(raw_gains)

    @inbounds for rail in 1:INPUT_RAILS
        excitatory = excitatory_destination(rail)
        inhibitory = gaba_destination(rail)
        associative_excitatory = associative_destination(rail)
        associative_inhibitory = associative_gaba_destination(rail)
        excitatory_bar =
            input_bar[excitatory.input, excitatory.cell, excitatory.block] +
            input_bar[
                associative_excitatory.input,
                associative_excitatory.cell,
                associative_excitatory.block,
            ]
        inhibitory_bar =
            input_bar[inhibitory.input, inhibitory.cell, inhibitory.block] +
            input_bar[
                associative_inhibitory.input,
                associative_inhibitory.cell,
                associative_inhibitory.block,
            ]
        raw_excitatory = raw_gains[EXCITATORY_GAIN, rail]
        raw_inhibitory = raw_gains[INHIBITORY_GAIN, rail]
        excitatory_amplitude = bounded_gain(raw_excitatory, EXCITATORY_GAIN)
        inhibitory_amplitude = bounded_gain(raw_inhibitory, INHIBITORY_GAIN)
        rail_value = rails[rail]

        rail_bar[rail] = TR(
            excitatory_bar * excitatory_amplitude +
            inhibitory_bar * inhibitory_amplitude,
        )
        raw_gain_bar[EXCITATORY_GAIN, rail] = TG(
            excitatory_bar * rail_value *
            bounded_gain_derivative(raw_excitatory, EXCITATORY_GAIN),
        )
        raw_gain_bar[INHIBITORY_GAIN, rail] = TG(
            inhibitory_bar * rail_value *
            bounded_gain_derivative(raw_inhibitory, INHIBITORY_GAIN),
        )
    end
    return rail_bar, raw_gain_bar
end

"""
    sensory_cached_raw_vjp!(rail_bar, raw_gain_bar, input_bar, rails,
                            gains, raw_derivatives)

Exact production VJP.  `gains` and `raw_derivatives` are the two arrays
prepared together by [`transform_sensory_gains!`](@ref).  Cotangents are
written directly in raw-parameter coordinates without consulting raw values or
re-evaluating their transform.  Outputs are overwritten, matching
[`sensory_vjp!`](@ref).
"""
function sensory_cached_raw_vjp!(
    rail_bar::AbstractVector{TR},
    raw_gain_bar::AbstractMatrix{TG},
    input_bar::AbstractArray{TI,3},
    rails::AbstractVector{R},
    gains::AbstractMatrix{G},
    raw_derivatives::AbstractMatrix{D},
) where {
    TR<:AbstractFloat,
    TG<:AbstractFloat,
    TI<:AbstractFloat,
    R<:AbstractFloat,
    G<:AbstractFloat,
    D<:AbstractFloat,
}
    _check_rails(rail_bar)
    _check_raw_gains(raw_gain_bar)
    _check_destination(input_bar)
    _check_rails(rails)
    _check_raw_gains(gains)
    _check_raw_gains(raw_derivatives)

    @inbounds for rail in 1:INPUT_RAILS
        excitatory = excitatory_destination(rail)
        inhibitory = gaba_destination(rail)
        associative_excitatory = associative_destination(rail)
        associative_inhibitory = associative_gaba_destination(rail)
        excitatory_bar =
            input_bar[excitatory.input, excitatory.cell, excitatory.block] +
            input_bar[
                associative_excitatory.input,
                associative_excitatory.cell,
                associative_excitatory.block,
            ]
        inhibitory_bar =
            input_bar[inhibitory.input, inhibitory.cell, inhibitory.block] +
            input_bar[
                associative_inhibitory.input,
                associative_inhibitory.cell,
                associative_inhibitory.block,
            ]
        rail_value = rails[rail]
        rail_bar[rail] = TR(
            excitatory_bar * gains[EXCITATORY_GAIN, rail] +
            inhibitory_bar * gains[INHIBITORY_GAIN, rail],
        )
        raw_gain_bar[EXCITATORY_GAIN, rail] = TG(
            excitatory_bar * rail_value *
            raw_derivatives[EXCITATORY_GAIN, rail],
        )
        raw_gain_bar[INHIBITORY_GAIN, rail] = TG(
            inhibitory_bar * rail_value *
            raw_derivatives[INHIBITORY_GAIN, rail],
        )
    end
    return rail_bar, raw_gain_bar
end

end # module SensoryEncoder
