module TypedSparseAfferents

export AMPA_RECEPTOR,
       DECISION_CELL_COUNT,
       FANOUT,
       FEATURE_COUNT,
       GABA_RECEPTOR,
       INPUT_COUNT,
       NMDA_RECEPTOR,
       POSITION_COUNT,
       RECEPTOR_COUNT,
       TypedSparseAfferentGraph,
       build_typed_sparse_afferents,
       deposit_affected_delta!,
       deposit_affected_delta_pullback!,
       deposit_full!,
       deposit_full_pullback!,
       destination_input,
       edge_count,
       edge_magnitude,
       edge_slot,
       source_index,
       validate_typed_sparse_afferents

# The canonical factor interface is intentionally independent of the current
# recurrent model dimensions.  A spatial factor owns 27 continuous features;
# a decision cell accepts the same 9-compartment x 3-receptor typed input.
const POSITION_COUNT = 240
const FEATURE_COUNT = 27
const DECISION_CELL_COUNT = 50
const RECEPTOR_COUNT = 3
const INPUT_COUNT = 27
const FANOUT = 8

# A spatial contact is present in both the before and after anatomical planes,
# and the resulting conductance is held for every physical decision phase.
# The original single-plane/single-phase fan-in calibration must therefore be
# divided by this exposure count.  This is an input-energy normalization, not
# a spike-threshold tuning constant.
const _SPATIAL_PLANE_COUNT = 2
const _DECISION_PHASE_COUNT = 3
const _INITIAL_EXPOSURE_COUNT = _SPATIAL_PLANE_COUNT * _DECISION_PHASE_COUNT

const AMPA_RECEPTOR = UInt8(1)
const NMDA_RECEPTOR = UInt8(2)
const GABA_RECEPTOR = UInt8(3)
const _DEFAULT_SEED = UInt64(0x7479706564616666)

"""
Source-major, fixed-fanout typed afferent graph.

Every source is the ordered pair `(spatial_position, continuous_feature)`.
`source_position` and `source_feature` are stored explicitly so position and
feature identity cannot disappear in a commutative pool.  Each feature has a
fixed receptor identity for positive evidence.  Negative evidence is carried
by the complementary E/I receptor, so a signed continuous feature is never
written as a negative conductance.  Each edge selects a decision cell and a
destination compartment.  `raw_magnitude` is the only trainable edge field.
"""
struct TypedSparseAfferentGraph{T<:AbstractFloat}
    source_position::Memory{UInt16}
    source_feature::Memory{UInt8}
    destination_cell::Memory{UInt8}
    destination_compartment::Memory{UInt8}
    receptor::Memory{UInt8}
    raw_magnitude::Vector{T}
end

@inline edge_count(::TypedSparseAfferentGraph) = POSITION_COUNT * FEATURE_COUNT * FANOUT

@inline function source_index(position::Integer, feature::Integer)
    1 <= position <= POSITION_COUNT || throw(BoundsError(1:POSITION_COUNT, position))
    1 <= feature <= FEATURE_COUNT || throw(BoundsError(1:FEATURE_COUNT, feature))
    return (Int(position) - 1) * FEATURE_COUNT + Int(feature)
end

@inline function edge_slot(
    ::TypedSparseAfferentGraph,
    position::Integer,
    feature::Integer,
    relation::Integer,
)
    1 <= relation <= FANOUT || throw(BoundsError(1:FANOUT, relation))
    return (source_index(position, feature) - 1) * FANOUT + Int(relation)
end

@inline function destination_input(graph::TypedSparseAfferentGraph, slot::Integer)
    1 <= slot <= edge_count(graph) || throw(BoundsError(1:edge_count(graph), slot))
    @inbounds begin
        compartment = Int(graph.destination_compartment[Int(slot)])
        receptor = Int(graph.receptor[Int(slot)])
    end
    return (compartment - 1) * RECEPTOR_COUNT + receptor
end

@inline function _mix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function _feature_receptor(feature::Int)
    # SharedDendriticFactor feature contract:
    #   voltage 1:8       signed fast evidence
    #   NMDA 9:16        non-negative slow evidence
    #   plateau 17:24    non-negative regenerative evidence
    #   apical V / margin signed evidence
    #   adaptation       non-negative inhibitory evidence
    feature <= 8 && return AMPA_RECEPTOR
    feature <= 16 && return NMDA_RECEPTOR
    feature <= 24 && return NMDA_RECEPTOR
    feature <= 26 && return AMPA_RECEPTOR
    return GABA_RECEPTOR
end

@inline _opposite_receptor(receptor::UInt8) =
    receptor == GABA_RECEPTOR ? AMPA_RECEPTOR : GABA_RECEPTOR

@inline function _signed_receptor_and_amplitude(value, positive_receptor::UInt8)
    if value >= zero(value)
        return positive_receptor, value, one(value)
    end
    return _opposite_receptor(positive_receptor), -value, -one(value)
end

@inline function _unit_interval(word::UInt64, ::Type{T}) where {T<:AbstractFloat}
    numerator = (word >> 40) & UInt64(0x00ff_ffff)
    return T(numerator) / T(0x0100_0000)
end

@inline function edge_magnitude(raw::T) where {T<:AbstractFloat}
    if raw > T(16)
        return raw
    elseif raw < T(-16)
        return exp(raw)
    end
    return log1p(exp(raw))
end

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("edge magnitude must be positive"))
    return value + log(-expm1(-value))
end

@inline function _exposure_normalized_raw(raw::T) where {T<:AbstractFloat}
    return _inverse_softplus(
        edge_magnitude(raw) / T(_INITIAL_EXPOSURE_COUNT),
    )
end

@inline function _edge_magnitude_derivative(raw::T) where {T<:AbstractFloat}
    if raw >= zero(T)
        negative = exp(-raw)
        return inv(one(T) + negative)
    end
    positive = exp(raw)
    return positive / (one(T) + positive)
end

@inline function _duplicate_destination(
    destination_cell::Vector{UInt8},
    destination_compartment::Vector{UInt8},
    source::Int,
    relation::Int,
    cell::UInt8,
    compartment::UInt8,
)
    first_slot = (source - 1) * FANOUT + 1
    @inbounds for previous in 1:(relation - 1)
        slot = first_slot + previous - 1
        if destination_cell[slot] == cell &&
           destination_compartment[slot] == compartment
            return true
        end
    end
    return false
end

"""Construct the deterministic topology and its trainable positive magnitudes."""
function build_typed_sparse_afferents(
    seed::Integer=_DEFAULT_SEED,
    ::Type{T}=Float32,
) where {T<:AbstractFloat}
    slots = POSITION_COUNT * FEATURE_COUNT * FANOUT
    source_position = Vector{UInt16}(undef, slots)
    source_feature = Vector{UInt8}(undef, slots)
    destination_cell = Vector{UInt8}(undef, slots)
    destination_compartment = Vector{UInt8}(undef, slots)
    receptor = Vector{UInt8}(undef, slots)
    raw_magnitude = Vector{T}(undef, slots)
    seed_word = UInt64(seed)

    @inbounds for position in 1:POSITION_COUNT
        for feature in 1:FEATURE_COUNT
            source = source_index(position, feature)
            fixed_receptor = _feature_receptor(feature)
            source_word = _mix64(
                xor(seed_word, UInt64(source) * UInt64(0xd6e8feb86659fd93)),
            )
            for relation in 1:FANOUT
                slot = (source - 1) * FANOUT + relation
                word = _mix64(
                    xor(source_word, UInt64(relation) * UInt64(0xa0761d6478bd642f)),
                )
                cell = UInt8(mod(word, UInt64(DECISION_CELL_COUNT)) + 1)
                compartment = UInt8(mod(word >> 17, UInt64(INPUT_COUNT ÷ RECEPTOR_COUNT)) + 1)
                attempt = 0
                while _duplicate_destination(
                    destination_cell,
                    destination_compartment,
                    source,
                    relation,
                    cell,
                    compartment,
                )
                    attempt += 1
                    word = _mix64(word + UInt64(attempt))
                    cell = UInt8(mod(word, UInt64(DECISION_CELL_COUNT)) + 1)
                    compartment = UInt8(
                        mod(word >> 17, UInt64(INPUT_COUNT ÷ RECEPTOR_COUNT)) + 1,
                    )
                end

                source_position[slot] = UInt16(position)
                source_feature[slot] = UInt8(feature)
                destination_cell[slot] = cell
                destination_compartment[slot] = compartment
                receptor[slot] = fixed_receptor
                # `-6 .. -5` is the calibrated single-plane/single-phase
                # contact scale.  A canonical decision receives two planes
                # and integrates the same drive for three phases, so preserve
                # that energy by dividing the *physical softplus magnitude*
                # by 2*3.  Transforming the raw value after this division is
                # exact and avoids relying on a tail-only `raw - log(6)`
                # approximation.
                base_raw = T(-6) + _unit_interval(_mix64(word), T)
                raw_magnitude[slot] = _exposure_normalized_raw(base_raw)
            end
        end
    end

    graph = TypedSparseAfferentGraph(
        Memory{UInt16}(source_position),
        Memory{UInt8}(source_feature),
        Memory{UInt8}(destination_cell),
        Memory{UInt8}(destination_compartment),
        Memory{UInt8}(receptor),
        raw_magnitude,
    )
    return validate_typed_sparse_afferents(graph)
end

function validate_typed_sparse_afferents(graph::TypedSparseAfferentGraph)
    expected = POSITION_COUNT * FEATURE_COUNT * FANOUT
    for array in (
        graph.source_position,
        graph.source_feature,
        graph.destination_cell,
        graph.destination_compartment,
        graph.receptor,
        graph.raw_magnitude,
    )
        length(array) == expected || throw(ArgumentError(
            "typed afferent storage violates fixed fanout",
        ))
    end

    @inbounds for position in 1:POSITION_COUNT
        for feature in 1:FEATURE_COUNT
            source = source_index(position, feature)
            first_slot = (source - 1) * FANOUT + 1
            expected_receptor = _feature_receptor(feature)
            for relation in 1:FANOUT
                slot = first_slot + relation - 1
                graph.source_position[slot] == UInt16(position) || throw(ArgumentError(
                    "edge $slot lost spatial position identity",
                ))
                graph.source_feature[slot] == UInt8(feature) || throw(ArgumentError(
                    "edge $slot lost source feature identity",
                ))
                1 <= graph.destination_cell[slot] <= DECISION_CELL_COUNT ||
                    throw(ArgumentError("edge $slot has an invalid destination cell"))
                1 <= graph.destination_compartment[slot] <= INPUT_COUNT ÷ RECEPTOR_COUNT ||
                    throw(ArgumentError("edge $slot has an invalid compartment"))
                graph.receptor[slot] == expected_receptor || throw(ArgumentError(
                    "edge $slot changed the fixed source receptor identity",
                ))
                isfinite(graph.raw_magnitude[slot]) || throw(ArgumentError(
                    "edge $slot has a non-finite raw magnitude",
                ))
                for previous in 1:(relation - 1)
                    prior_slot = first_slot + previous - 1
                    same_cell = graph.destination_cell[prior_slot] ==
                                graph.destination_cell[slot]
                    same_compartment = graph.destination_compartment[prior_slot] ==
                                       graph.destination_compartment[slot]
                    same_cell && same_compartment && throw(ArgumentError(
                        "source $source contains duplicate typed destinations",
                    ))
                end
            end
        end
    end
    return graph
end

@inline function _check_feature_shape(features)
    size(features, 1) == FEATURE_COUNT && size(features, 2) == POSITION_COUNT ||
        throw(DimensionMismatch(
            "source features must have shape ($FEATURE_COUNT, $POSITION_COUNT)",
        ))
    return nothing
end

@inline function _check_destination_shape(destination)
    size(destination, 1) == INPUT_COUNT &&
        size(destination, 2) == DECISION_CELL_COUNT ||
        throw(DimensionMismatch(
            "decision inputs must have shape ($INPUT_COUNT, $DECISION_CELL_COUNT)",
        ))
    return nothing
end

@inline function _check_raw_bar(graph, raw_bar)
    length(raw_bar) == edge_count(graph) || throw(DimensionMismatch(
        "raw magnitude cotangent must have $(edge_count(graph)) entries",
    ))
    return nothing
end

@inline function _validate_affected_positions(affected_positions)
    @inbounds for index in eachindex(affected_positions)
        position = Int(affected_positions[index])
        1 <= position <= POSITION_COUNT || throw(BoundsError(
            1:POSITION_COUNT,
            position,
        ))
        for previous in firstindex(affected_positions):(index - 1)
            Int(affected_positions[previous]) == position && throw(ArgumentError(
                "affected_positions must not contain duplicates",
            ))
        end
    end
    return nothing
end

"""Add all 240 x 27 typed source features to caller-owned decision inputs."""
function deposit_full!(
    destination::AbstractMatrix,
    graph::TypedSparseAfferentGraph,
    features::AbstractMatrix,
)
    _check_destination_shape(destination)
    _check_feature_shape(features)
    @inbounds for position in 1:POSITION_COUNT
        for feature in 1:FEATURE_COUNT
            value = features[feature, position]
            source = (position - 1) * FEATURE_COUNT + feature
            first_slot = (source - 1) * FANOUT + 1
            for relation in 1:FANOUT
                slot = first_slot + relation - 1
                receptor, amplitude, _ = _signed_receptor_and_amplitude(
                    value,
                    graph.receptor[slot],
                )
                input = (Int(graph.destination_compartment[slot]) - 1) *
                        RECEPTOR_COUNT + Int(receptor)
                cell = Int(graph.destination_cell[slot])
                destination[input, cell] = muladd(
                    amplitude,
                    edge_magnitude(graph.raw_magnitude[slot]),
                    destination[input, cell],
                )
            end
        end
    end
    return destination
end

"""
Replace the base deposit by the candidate deposit only at caller-provided
affected positions.

Thus `deposit_full!(y, graph, base); deposit_affected_delta!(y, ...)` is the
incremental counterpart of depositing the complete candidate feature tensor.

The subtraction is performed *after* the signed receptor transform.  In
general `typed(candidate) - typed(base) != typed(candidate - base)` when the
feature crosses zero, so pushing the scalar difference through that transform
would silently change both receptor identity and the represented value.
"""
function deposit_affected_delta!(
    destination::AbstractMatrix,
    graph::TypedSparseAfferentGraph,
    candidate::AbstractMatrix,
    base::AbstractMatrix,
    affected_positions::AbstractVector{<:Integer},
)
    _check_destination_shape(destination)
    _check_feature_shape(candidate)
    _check_feature_shape(base)
    _validate_affected_positions(affected_positions)
    @inbounds for affected_index in eachindex(affected_positions)
        position = Int(affected_positions[affected_index])
        for feature in 1:FEATURE_COUNT
            candidate_value = candidate[feature, position]
            base_value = base[feature, position]
            source = (position - 1) * FEATURE_COUNT + feature
            first_slot = (source - 1) * FANOUT + 1
            for relation in 1:FANOUT
                slot = first_slot + relation - 1
                candidate_receptor, candidate_amplitude, _ =
                    _signed_receptor_and_amplitude(
                        candidate_value,
                        graph.receptor[slot],
                    )
                base_receptor, base_amplitude, _ = _signed_receptor_and_amplitude(
                    base_value,
                    graph.receptor[slot],
                )
                input_offset = (Int(graph.destination_compartment[slot]) - 1) *
                               RECEPTOR_COUNT
                cell = Int(graph.destination_cell[slot])
                magnitude = edge_magnitude(graph.raw_magnitude[slot])
                candidate_input = input_offset + Int(candidate_receptor)
                base_input = input_offset + Int(base_receptor)
                destination[candidate_input, cell] = muladd(
                    candidate_amplitude,
                    magnitude,
                    destination[candidate_input, cell],
                )
                destination[base_input, cell] = muladd(
                    -base_amplitude,
                    magnitude,
                    destination[base_input, cell],
                )
            end
        end
    end
    return destination
end

"""Exact additive pullback for `deposit_full!`."""
function deposit_full_pullback!(
    feature_bar::AbstractMatrix,
    raw_bar::AbstractVector,
    graph::TypedSparseAfferentGraph,
    features::AbstractMatrix,
    destination_bar::AbstractMatrix,
)
    _check_feature_shape(feature_bar)
    _check_feature_shape(features)
    _check_destination_shape(destination_bar)
    _check_raw_bar(graph, raw_bar)
    @inbounds for position in 1:POSITION_COUNT
        for feature in 1:FEATURE_COUNT
            value = features[feature, position]
            accumulated = zero(eltype(feature_bar))
            source = (position - 1) * FEATURE_COUNT + feature
            first_slot = (source - 1) * FANOUT + 1
            for relation in 1:FANOUT
                slot = first_slot + relation - 1
                receptor, amplitude, value_derivative =
                    _signed_receptor_and_amplitude(
                        value,
                        graph.receptor[slot],
                    )
                input = (Int(graph.destination_compartment[slot]) - 1) *
                        RECEPTOR_COUNT + Int(receptor)
                cell = Int(graph.destination_cell[slot])
                cotangent = destination_bar[input, cell]
                raw = graph.raw_magnitude[slot]
                accumulated = muladd(
                    cotangent * value_derivative,
                    edge_magnitude(raw),
                    accumulated,
                )
                raw_bar[slot] = muladd(
                    cotangent * amplitude,
                    _edge_magnitude_derivative(raw),
                    raw_bar[slot],
                )
            end
            feature_bar[feature, position] += accumulated
        end
    end
    return feature_bar, raw_bar
end

"""Exact additive pullback for `deposit_affected_delta!`."""
function deposit_affected_delta_pullback!(
    candidate_bar::AbstractMatrix,
    base_bar::AbstractMatrix,
    raw_bar::AbstractVector,
    graph::TypedSparseAfferentGraph,
    candidate::AbstractMatrix,
    base::AbstractMatrix,
    affected_positions::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix,
)
    _check_feature_shape(candidate_bar)
    _check_feature_shape(base_bar)
    _check_feature_shape(candidate)
    _check_feature_shape(base)
    _check_destination_shape(destination_bar)
    _check_raw_bar(graph, raw_bar)
    _validate_affected_positions(affected_positions)
    @inbounds for affected_index in eachindex(affected_positions)
        position = Int(affected_positions[affected_index])
        for feature in 1:FEATURE_COUNT
            candidate_value = candidate[feature, position]
            base_value = base[feature, position]
            candidate_accumulated = zero(eltype(candidate_bar))
            base_accumulated = zero(eltype(base_bar))
            source = (position - 1) * FEATURE_COUNT + feature
            first_slot = (source - 1) * FANOUT + 1
            for relation in 1:FANOUT
                slot = first_slot + relation - 1
                candidate_receptor, candidate_amplitude, candidate_derivative =
                    _signed_receptor_and_amplitude(
                        candidate_value,
                        graph.receptor[slot],
                    )
                base_receptor, base_amplitude, base_derivative =
                    _signed_receptor_and_amplitude(
                        base_value,
                        graph.receptor[slot],
                    )
                input_offset = (Int(graph.destination_compartment[slot]) - 1) *
                               RECEPTOR_COUNT
                candidate_input = input_offset + Int(candidate_receptor)
                base_input = input_offset + Int(base_receptor)
                cell = Int(graph.destination_cell[slot])
                candidate_cotangent = destination_bar[candidate_input, cell]
                base_cotangent = destination_bar[base_input, cell]
                raw = graph.raw_magnitude[slot]
                magnitude = edge_magnitude(raw)
                candidate_accumulated = muladd(
                    candidate_cotangent * candidate_derivative,
                    magnitude,
                    candidate_accumulated,
                )
                base_accumulated = muladd(
                    -base_cotangent * base_derivative,
                    magnitude,
                    base_accumulated,
                )
                raw_bar[slot] = muladd(
                    candidate_cotangent * candidate_amplitude -
                    base_cotangent * base_amplitude,
                    _edge_magnitude_derivative(raw),
                    raw_bar[slot],
                )
            end
            candidate_bar[feature, position] += candidate_accumulated
            base_bar[feature, position] += base_accumulated
        end
    end
    return candidate_bar, base_bar, raw_bar
end

end
