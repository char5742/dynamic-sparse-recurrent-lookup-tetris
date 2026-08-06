module StructuredMotifReadout

using ..ActiveApicalCell
using ..HighDimensionalCellPacket

const Cell = ActiveApicalCell
const Packet = HighDimensionalCellPacket

export SOURCE_COUNT,
       FAMILY_COUNT,
       OUTPUT_COUNT,
       RELATION_RESIDUAL_SCALE,
       StructuredReadoutParameters,
       StructuredReadoutCache,
       StructuredReadoutGradient,
       initialize_parameters,
       refresh_cache!,
       clear_gradient!,
       accumulate_gradient!,
       stored_parameter_count,
       family_index,
       positive_branch,
       negative_branch,
       output_field_count,
       output_field,
       fixed_weight,
       deposit_readout!,
       deposit_readout_selected!,
       deposit_readout_pullback!,
       deposit_readout_selected_pullback!

"""The semantic packet bank consists of 48 branch-preserving cell packets."""
const SOURCE_COUNT = 48

"""Rows, columns, tiles and stripes are four fixed anatomical families."""
const FAMILY_COUNT = 4

"""The supervised ranking interface is fixed at 22 typed output cells."""
const OUTPUT_COUNT = 22

"""Canonical scale for the shallow relation-to-output residual."""
const RELATION_RESIDUAL_SCALE = 0.125f0

# Source identities are fixed and contiguous.  The counts are also the exact
# normalization populations used by `fixed_weight`.
const FAMILY_FIRST = (1, 25, 35, 43)
const FAMILY_LAST = (24, 34, 42, 48)
const FAMILY_SOURCE_COUNT = (24, 10, 8, 6)

# Semantic sign changes the receiving branch, never the receptor identity.
const POSITIVE_BRANCH = (1, 3, 5, 7)
const NEGATIVE_BRANCH = (2, 4, 6, 8)

# The readout exposes the complete 47-lane cell transition without restoring a
# dense all-to-all head.  Q sees every lane; death sees a stratified 24-lane
# sample; the 16 quantiles use rotated eight-lane samples whose union covers
# all 47 coordinates; geometry outputs use 16 stratified lanes each.
const OUTPUT_FIELD_COUNT = (
    47, 24,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    16, 16, 16, 16,
)

const DEATH_FIELDS = (
    1, 5, 9,
    10, 14, 18,
    19, 23, 27,
    28, 32, 36,
    37, 41, 45,
    3, 7, 12, 16, 21, 25, 30,
    46, 47,
)

@inline function _scheduled_field(output::Int, slot::Int)
    output == 1 && return slot
    output == 2 && return DEATH_FIELDS[slot]
    if output <= 18
        return 1 + mod(3 * (output - 3) + 7 * (slot - 1), Packet.PACKET_DIM)
    end
    slot == 1 && return Packet.MARGIN_LANE
    slot == 2 && return Packet.ADAPTATION_LANE
    return 1 + mod(11 * (output - 19) + 7 * (slot - 3), 45)
end

# A fixed tuple-of-tuples avoids heap activity in the hot readout while keeping
# all schedules inspectable through `output_field`.
const OUTPUT_FIELDS = ntuple(
    output -> ntuple(
        slot -> UInt8(
            slot <= OUTPUT_FIELD_COUNT[output] ?
            _scheduled_field(output, slot) : 0
        ),
        Packet.PACKET_DIM,
    ),
    OUTPUT_COUNT,
)

const GAIN_MIN = 0.25f0
const GAIN_SPAN = 1.75f0

@inline _sigmoid(value) = inv(one(value) + exp(-value))

@inline function family_index(source::Integer)
    1 <= source <= SOURCE_COUNT || throw(BoundsError(1:SOURCE_COUNT, source))
    source <= FAMILY_LAST[1] && return 1
    source <= FAMILY_LAST[2] && return 2
    source <= FAMILY_LAST[3] && return 3
    return 4
end

@inline function positive_branch(family::Integer)
    1 <= family <= FAMILY_COUNT || throw(BoundsError(1:FAMILY_COUNT, family))
    return @inbounds POSITIVE_BRANCH[family]
end

@inline function negative_branch(family::Integer)
    1 <= family <= FAMILY_COUNT || throw(BoundsError(1:FAMILY_COUNT, family))
    return @inbounds NEGATIVE_BRANCH[family]
end

@inline function output_field_count(output::Integer)
    1 <= output <= OUTPUT_COUNT || throw(BoundsError(1:OUTPUT_COUNT, output))
    return @inbounds OUTPUT_FIELD_COUNT[output]
end

@inline function output_field(output::Integer, field_slot::Integer)
    count = output_field_count(output)
    1 <= field_slot <= count || throw(BoundsError(1:count, field_slot))
    return @inbounds OUTPUT_FIELDS[output][field_slot]
end

"""
    fixed_weight(T, family, output)

Fixed fan-in normalization.  A family with `n` sources and an output reading
`f` packet fields assigns every source-field contact weight
`1 / sqrt(n*f)`.  Hence changing semantic population size does not silently
change its initial drive scale.
"""
@inline function fixed_weight(
    ::Type{T},
    family::Integer,
    output::Integer,
) where {T<:AbstractFloat}
    1 <= family <= FAMILY_COUNT || throw(BoundsError(1:FAMILY_COUNT, family))
    count = output_field_count(output)
    return inv(sqrt(T(@inbounds FAMILY_SOURCE_COUNT[family] * count)))
end

"""
The only trainable readout quantity: one bounded gain per source/output.

The family fixes branch anatomy and fan-in normalization, but never averages
source identity.  Keeping 48 distinct gains is the minimal positional binding
that makes two row, column, tile or stripe motifs distinguishable downstream.
"""
struct StructuredReadoutParameters{T<:AbstractFloat}
    source_gain_raw::Matrix{T}

    function StructuredReadoutParameters(
        source_gain_raw::Matrix{T},
    ) where {T<:AbstractFloat}
        size(source_gain_raw) == (SOURCE_COUNT, OUTPUT_COUNT) || throw(
            DimensionMismatch("source gain raw must have shape (48, 22)"),
        )
        return new{T}(source_gain_raw)
    end
end

"""Initialize every physical family gain to exactly one."""
function initialize_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    # 1 = 0.25 + 1.75 * sigmoid(raw), hence sigmoid(raw) = 3/7.
    raw = log(T(3) / T(4))
    return StructuredReadoutParameters(
        fill(raw, SOURCE_COUNT, OUTPUT_COUNT),
    )
end

"""Optimizer-step cache of the bounded gains and their raw derivatives."""
struct StructuredReadoutCache{T<:AbstractFloat}
    source_gain::Matrix{T}
    source_gain_derivative::Matrix{T}
end

function StructuredReadoutCache(
    parameters::StructuredReadoutParameters{T},
) where {T<:AbstractFloat}
    cache = StructuredReadoutCache(
        Matrix{T}(undef, SOURCE_COUNT, OUTPUT_COUNT),
        Matrix{T}(undef, SOURCE_COUNT, OUTPUT_COUNT),
    )
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(
    cache::StructuredReadoutCache{T},
    parameters::StructuredReadoutParameters{T},
) where {T<:AbstractFloat}
    size(cache.source_gain) == (SOURCE_COUNT, OUTPUT_COUNT) || throw(
        DimensionMismatch("source gain cache must have shape (48, 22)"),
    )
    size(cache.source_gain_derivative) == (SOURCE_COUNT, OUTPUT_COUNT) ||
        throw(DimensionMismatch(
            "source gain derivative cache must have shape (48, 22)",
        ))
    lower = T(GAIN_MIN)
    span = T(GAIN_SPAN)
    @inbounds @simd for index in eachindex(parameters.source_gain_raw)
        probability = _sigmoid(parameters.source_gain_raw[index])
        cache.source_gain[index] = muladd(span, probability, lower)
        cache.source_gain_derivative[index] =
            span * probability * (one(T) - probability)
    end
    return cache
end

"""Gradient storage mirrors the single trainable parameter group exactly."""
struct StructuredReadoutGradient{T<:AbstractFloat}
    source_gain_raw::Matrix{T}
end

function StructuredReadoutGradient(::Type{T}=Float32) where {T<:AbstractFloat}
    return StructuredReadoutGradient(
        zeros(T, SOURCE_COUNT, OUTPUT_COUNT),
    )
end

function clear_gradient!(gradient::StructuredReadoutGradient{T}) where {T}
    fill!(gradient.source_gain_raw, zero(T))
    return gradient
end

function accumulate_gradient!(
    destination::StructuredReadoutGradient{T},
    source::StructuredReadoutGradient{T},
) where {T<:AbstractFloat}
    @inbounds @simd for index in eachindex(destination.source_gain_raw)
        destination.source_gain_raw[index] += source.source_gain_raw[index]
    end
    return destination
end

@inline stored_parameter_count(::StructuredReadoutParameters) =
    SOURCE_COUNT * OUTPUT_COUNT

@inline function _check_packet(packet)
    size(packet) == (Packet.PACKET_DIM, SOURCE_COUNT) || throw(
        DimensionMismatch(
            "packet must have shape ($(Packet.PACKET_DIM), 48)",
        ),
    )
    return nothing
end

@inline function _check_destination(destination)
    size(destination) == (Cell.INPUT_DIM, OUTPUT_COUNT) || throw(
        DimensionMismatch(
            "typed output inbox must have shape ($(Cell.INPUT_DIM), 22)",
        ),
    )
    return nothing
end

@inline function _check_sources(sources)
    @inbounds for index in eachindex(sources)
        source = Int(sources[index])
        1 <= source <= SOURCE_COUNT || throw(BoundsError(1:SOURCE_COUNT, source))
        for previous in firstindex(sources):(index - 1)
            Int(sources[previous]) == source && throw(ArgumentError(
                "source list must not contain duplicates",
            ))
        end
    end
    return nothing
end

@inline function _check_scale(scale)
    isfinite(scale) || throw(ArgumentError("readout scale must be finite"))
    return nothing
end

@inline _packet_compartment(field::Int) = 1 + mod(field - 1, Cell.N_COMPARTMENTS)

@inline function _packet_contact(
    ::Type{T},
    family::Int,
    field::Int,
    value::T,
) where {T<:AbstractFloat}
    signed = field <= Cell.N_COMPARTMENTS || field == Packet.MARGIN_LANE
    compartment = field <= 45 ? _packet_compartment(field) : 0
    apical = compartment == Cell.N_COMPARTMENTS
    phase = 2 * (family - 1)
    target = apical ? Cell.N_COMPARTMENTS :
             1 + mod(compartment - 1 + phase, Cell.N_BASAL)

    if signed
        iszero(value) && return 0, zero(T), zero(T)
        positive = value > zero(T)
        receptor = positive ? Cell.INPUT_AMPA : Cell.INPUT_GABA
        return Cell.input_index(target, receptor), abs(value),
               positive ? one(T) : -one(T)
    end

    value > zero(T) || return 0, zero(T), zero(T)
    if field == Packet.ADAPTATION_LANE
        return Cell.input_index(Cell.N_COMPARTMENTS, Cell.INPUT_GABA),
               value, one(T)
    end

    receptor = field < Packet.NMDA_LANE_FIRST ? Cell.INPUT_AMPA :
               field < Packet.GABA_LANE_FIRST ? Cell.INPUT_NMDA :
               field < Packet.PLATEAU_LANE_FIRST ? Cell.INPUT_GABA :
               Cell.INPUT_NMDA
    return Cell.input_index(target, receptor), value, one(T)
end

@inline function _deposit_source!(
    destination::AbstractMatrix{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    source::Int,
    scale::T,
) where {T<:AbstractFloat}
    family = family_index(source)
    @inbounds for output in 1:OUTPUT_COUNT
        count = OUTPUT_FIELD_COUNT[output]
        coefficient = scale * cache.source_gain[source, output] *
                      inv(sqrt(T(FAMILY_SOURCE_COUNT[family] * count)))
        for field_slot in 1:count
            field = Int(OUTPUT_FIELDS[output][field_slot])
            value = packet[field, source]
            input, activity, _ = _packet_contact(
                T,
                family,
                field,
                value,
            )
            if !iszero(input)
                destination[input, output] = muladd(
                    coefficient,
                    activity,
                    destination[input, output],
                )
            end
        end
    end
    return nothing
end

"""
    deposit_readout!(destination, packet, cache, scale)

Add the full fixed semantic packet readout to a caller-owned typed output
inbox.  Positive and negative packet values occupy opponent basal branches;
the packet field fixes the receptor for both signs.  `destination` is never
cleared.  Canonical motif drive uses `scale=1`, while the relation residual
uses [`RELATION_RESIDUAL_SCALE`](@ref).
"""
function deposit_readout!(
    destination::AbstractMatrix{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    scale::T,
) where {T<:AbstractFloat}
    _check_destination(destination)
    _check_packet(packet)
    _check_scale(scale)
    @inbounds for source in 1:SOURCE_COUNT
        _deposit_source!(destination, packet, cache, source, scale)
    end
    return destination
end

"""Allocation-free source-selected form of [`deposit_readout!`](@ref)."""
function deposit_readout_selected!(
    destination::AbstractMatrix{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    sources::AbstractVector{<:Integer},
    scale::T,
) where {T<:AbstractFloat}
    _check_destination(destination)
    _check_packet(packet)
    _check_sources(sources)
    _check_scale(scale)
    @inbounds for index in eachindex(sources)
        _deposit_source!(
            destination,
            packet,
            cache,
            Int(sources[index]),
            scale,
        )
    end
    return destination
end

@inline function _pullback_source!(
    packet_bar::AbstractMatrix{T},
    gradient::StructuredReadoutGradient{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    destination_bar::AbstractMatrix{T},
    source::Int,
    scale::T,
) where {T<:AbstractFloat}
    family = family_index(source)
    @inbounds for output in 1:OUTPUT_COUNT
        count = OUTPUT_FIELD_COUNT[output]
        normalization = inv(sqrt(T(FAMILY_SOURCE_COUNT[family] * count)))
        fixed_scale = scale * normalization
        gain = cache.source_gain[source, output]
        gain_bar = zero(T)
        for field_slot in 1:count
            field = Int(OUTPUT_FIELDS[output][field_slot])
            value = packet[field, source]
            input, activity, derivative = _packet_contact(
                T,
                family,
                field,
                value,
            )
            if !iszero(input)
                cotangent = destination_bar[input, output]
                packet_bar[field, source] = muladd(
                    fixed_scale * gain * derivative,
                    cotangent,
                    packet_bar[field, source],
                )
                gain_bar = muladd(fixed_scale * activity, cotangent, gain_bar)
            end
        end
        gradient.source_gain_raw[source, output] = muladd(
            gain_bar,
            cache.source_gain_derivative[source, output],
            gradient.source_gain_raw[source, output],
        )
    end
    return nothing
end

"""
    deposit_readout_pullback!(packet_bar, gradient, packet, cache,
                              destination_bar, scale)

Exact additive pullback of [`deposit_readout!`](@ref), away from the signed
opponent split at packet value zero.  Both `packet_bar` and `gradient` are
accumulated and never cleared.
"""
function deposit_readout_pullback!(
    packet_bar::AbstractMatrix{T},
    gradient::StructuredReadoutGradient{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    destination_bar::AbstractMatrix{T},
    scale::T,
) where {T<:AbstractFloat}
    _check_packet(packet_bar)
    _check_packet(packet)
    _check_destination(destination_bar)
    _check_scale(scale)
    @inbounds for source in 1:SOURCE_COUNT
        _pullback_source!(
            packet_bar,
            gradient,
            packet,
            cache,
            destination_bar,
            source,
            scale,
        )
    end
    return packet_bar, gradient
end

"""Exact additive pullback restricted to the same unique source list."""
function deposit_readout_selected_pullback!(
    packet_bar::AbstractMatrix{T},
    gradient::StructuredReadoutGradient{T},
    packet::AbstractMatrix{T},
    cache::StructuredReadoutCache{T},
    sources::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix{T},
    scale::T,
) where {T<:AbstractFloat}
    _check_packet(packet_bar)
    _check_packet(packet)
    _check_destination(destination_bar)
    _check_sources(sources)
    _check_scale(scale)
    @inbounds for index in eachindex(sources)
        _pullback_source!(
            packet_bar,
            gradient,
            packet,
            cache,
            destination_bar,
            Int(sources[index]),
            scale,
        )
    end
    return packet_bar, gradient
end

@assert Packet.PACKET_DIM == 47
@assert output_field_count(1) == Packet.PACKET_DIM
@assert all(field -> any(
    output -> any(
        slot -> output_field(output, slot) == field,
        1:output_field_count(output),
    ),
    1:OUTPUT_COUNT,
), 1:Packet.PACKET_DIM)
@assert FAMILY_FIRST == (1, 25, 35, 43)
@assert FAMILY_LAST == (24, 34, 42, 48)
@assert sum(FAMILY_SOURCE_COUNT) == SOURCE_COUNT

end # module StructuredMotifReadout
