module ContextAfferents

using ..ActiveApicalCell
using ..CandidateDeltaInput

export AUXILIARY_FEATURES,
    BACK_TO_BACK_SOURCE_INDEX,
    CONTEXT_FANOUT,
    CONTEXT_SOURCE_COUNT,
    DECISION_CELLS,
    INPUT_DIM,
    QUEUE_PIECES,
    QUEUE_ROLES,
    REN_SOURCE_INDEX,
    SourceKind,
    ReceptorKind,
    ContextSource,
    ContextEdge,
    ContextTopology,
    AUXILIARY_SOURCE,
    QUEUE_SOURCE,
    REN_SOURCE,
    BACK_TO_BACK_SOURCE,
    AMPA_RECEPTOR,
    NMDA_RECEPTOR,
    GABA_RECEPTOR,
    auxiliary_source_index,
    queue_source_index,
    build_topology,
    default_raw_magnitudes,
    magnitude,
    magnitude_derivative,
    deposit_state_common!,
    deposit_candidate_aux!,
    state_common_pullback!,
    candidate_aux_pullback!

const AUXILIARY_FEATURES = CandidateDeltaInput.AUX_FEATURES
const QUEUE_PIECES = CandidateDeltaInput.QUEUE_PIECES
const QUEUE_ROLES = CandidateDeltaInput.QUEUE_TOKENS
const DECISION_CELLS = 50
const INPUT_DIM = ActiveApicalCell.INPUT_DIM
const CONTEXT_FANOUT = 8


# Context is deposited once per candidate but held through all three physical
# decision phases.  Normalize initial contact energy by the number of repeated
# integrations; queue bits remain explicit bipolar events.
const _DECISION_PHASE_COUNT = 3

const AUXILIARY_SOURCE_FIRST = 1
const QUEUE_SOURCE_FIRST = AUXILIARY_SOURCE_FIRST + AUXILIARY_FEATURES
const REN_SOURCE_INDEX =
    QUEUE_SOURCE_FIRST + QUEUE_PIECES * QUEUE_ROLES
const BACK_TO_BACK_SOURCE_INDEX = REN_SOURCE_INDEX + 1
const CONTEXT_SOURCE_COUNT = BACK_TO_BACK_SOURCE_INDEX

@assert AUXILIARY_FEATURES == 37
@assert QUEUE_PIECES == 7
@assert QUEUE_ROLES == 6
@assert INPUT_DIM == 27
@assert CONTEXT_SOURCE_COUNT == 81

@enum SourceKind::UInt8 begin
    AUXILIARY_SOURCE = 0x01
    QUEUE_SOURCE = 0x02
    REN_SOURCE = 0x03
    BACK_TO_BACK_SOURCE = 0x04
end

@enum ReceptorKind::UInt8 begin
    AMPA_RECEPTOR = 0x01
    NMDA_RECEPTOR = 0x02
    GABA_RECEPTOR = 0x03
end

"""One stable context identity. Queue `role` is never pooled away."""
struct ContextSource
    kind::SourceKind
    index::UInt8
    role::UInt8
end

"""One fixed destination in the relation-fastest, source-major graph."""
struct ContextEdge
    decision_cell::UInt8
    input::UInt8
    receptor::ReceptorKind
end

"""
Typed, fixed-fanout context graph.

`edge[relation, source]` is contiguous for each source under Julia's
column-major layout. Only `raw_magnitude[relation, source]` is trainable;
source identity, destination cell, compartment and receptor are fixed.
"""
struct ContextTopology
    source::Vector{ContextSource}
    edge::Matrix{ContextEdge}

    function ContextTopology(
        source::Vector{ContextSource},
        edge::Matrix{ContextEdge},
    )
        length(source) == CONTEXT_SOURCE_COUNT || throw(DimensionMismatch(
            "context topology must have $CONTEXT_SOURCE_COUNT sources",
        ))
        size(edge) == (CONTEXT_FANOUT, CONTEXT_SOURCE_COUNT) || throw(
            DimensionMismatch(
                "context edge table must have shape " *
                "($CONTEXT_FANOUT, $CONTEXT_SOURCE_COUNT)",
            ),
        )
        return new(source, edge)
    end
end

@inline function auxiliary_source_index(feature::Integer)
    1 <= feature <= AUXILIARY_FEATURES || throw(
        BoundsError(1:AUXILIARY_FEATURES, feature),
    )
    return AUXILIARY_SOURCE_FIRST + Int(feature) - 1
end

"""
Return the source identity for one `(piece, role)` queue bit.

Role is the queue position, not a pooled categorical feature. Consequently the
same tetromino in two queue positions always maps to two different sources.
"""
@inline function queue_source_index(piece::Integer, role::Integer)
    1 <= piece <= QUEUE_PIECES || throw(BoundsError(1:QUEUE_PIECES, piece))
    1 <= role <= QUEUE_ROLES || throw(BoundsError(1:QUEUE_ROLES, role))
    return QUEUE_SOURCE_FIRST +
           (Int(role) - 1) * QUEUE_PIECES + Int(piece) - 1
end

@inline function _mix64(value::UInt64)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function _seeded_word(seed::UInt64, source::Int, salt::UInt64)
    return _mix64(xor(
        seed,
        UInt64(source) * UInt64(0x9e3779b97f4a7c15),
        salt,
    ))
end

const _DECISION_STRIDES = (
    1, 3, 7, 9, 11, 13, 17, 19, 21, 23,
    27, 29, 31, 33, 37, 39, 41, 43, 47, 49,
)
const _RELATION_STRIDES = (1, 3, 5, 7)

@inline function _receptor_from_rank(rank::Int)
    rank <= 3 && return AMPA_RECEPTOR
    rank <= 6 && return NMDA_RECEPTOR
    return GABA_RECEPTOR
end

function _build_sources()
    source = Vector{ContextSource}(undef, CONTEXT_SOURCE_COUNT)
    @inbounds for feature in 1:AUXILIARY_FEATURES
        source[auxiliary_source_index(feature)] = ContextSource(
            AUXILIARY_SOURCE,
            UInt8(feature),
            0x00,
        )
    end
    @inbounds for role in 1:QUEUE_ROLES, piece in 1:QUEUE_PIECES
        source[queue_source_index(piece, role)] = ContextSource(
            QUEUE_SOURCE,
            UInt8(piece),
            UInt8(role),
        )
    end
    source[REN_SOURCE_INDEX] = ContextSource(REN_SOURCE, 0x01, 0x00)
    source[BACK_TO_BACK_SOURCE_INDEX] = ContextSource(
        BACK_TO_BACK_SOURCE,
        0x01,
        0x00,
    )
    return source
end

"""
Build the seed-fixed typed context graph.

Each of the 81 sources owns eight unique decision cells. A seed-fixed
permutation assigns exactly three AMPA, three NMDA and two GABA destinations
per source. The destination input is the receptor coordinate of one of the
nine Reduced Hay compartments and is never learned.
"""
function build_topology(seed::Integer=0x434f4e5445585401)
    seed >= 0 || throw(ArgumentError("context topology seed must be nonnegative"))
    seed_value = UInt64(seed)
    source = _build_sources()
    edge = Matrix{ContextEdge}(undef, CONTEXT_FANOUT, CONTEXT_SOURCE_COUNT)

    @inbounds for source_index in 1:CONTEXT_SOURCE_COUNT
        cell_word = _seeded_word(
            seed_value,
            source_index,
            UInt64(0x243f6a8885a308d3),
        )
        start = Int(cell_word % UInt64(DECISION_CELLS)) + 1
        stride = _DECISION_STRIDES[
            Int((cell_word >> 12) % UInt64(length(_DECISION_STRIDES))) + 1
        ]

        receptor_word = _seeded_word(
            seed_value,
            source_index,
            UInt64(0x13198a2e03707344),
        )
        receptor_offset = Int(receptor_word % UInt64(CONTEXT_FANOUT))
        receptor_stride = _RELATION_STRIDES[
            Int((receptor_word >> 12) % UInt64(length(_RELATION_STRIDES))) + 1
        ]

        compartment_word = _seeded_word(
            seed_value,
            source_index,
            UInt64(0xa4093822299f31d0),
        )
        compartment_offset = Int(
            compartment_word % UInt64(ActiveApicalCell.N_COMPARTMENTS),
        )
        compartment_stride = Int(
            (compartment_word >> 12) %
            UInt64(ActiveApicalCell.N_COMPARTMENTS - 1),
        ) + 1

        for relation in 1:CONTEXT_FANOUT
            decision_cell = mod(
                start - 1 + (relation - 1) * stride,
                DECISION_CELLS,
            ) + 1
            receptor_rank = mod(
                receptor_offset + (relation - 1) * receptor_stride,
                CONTEXT_FANOUT,
            ) + 1
            receptor = _receptor_from_rank(receptor_rank)
            compartment = mod(
                compartment_offset + (relation - 1) * compartment_stride,
                ActiveApicalCell.N_COMPARTMENTS,
            ) + 1
            input = ActiveApicalCell.input_index(compartment, Int(receptor))
            edge[relation, source_index] = ContextEdge(
                UInt8(decision_cell),
                UInt8(input),
                receptor,
            )
        end
    end
    return ContextTopology(source, edge)
end

@inline function _sigmoid(raw::T) where {T<:AbstractFloat}
    if raw >= zero(T)
        inverse = exp(-raw)
        return inv(one(T) + inverse)
    end
    exponential = exp(raw)
    return exponential / (one(T) + exponential)
end

"""Positive edge magnitude. Raw sign cannot change the fixed receptor type."""
@inline function magnitude(raw::T) where {T<:AbstractFloat}
    raw > T(20) && return raw
    raw < T(-20) && return exp(raw)
    return log1p(exp(raw))
end

"""Exact raw derivative of [`magnitude`](@ref), including its stable tails."""
@inline function magnitude_derivative(raw::T) where {T<:AbstractFloat}
    raw > T(20) && return one(T)
    raw < T(-20) && return exp(raw)
    return _sigmoid(raw)
end

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("context magnitude must be positive"))
    return value + log(-expm1(-value))
end

@inline function _phase_normalized_raw(raw::T) where {T<:AbstractFloat}
    return _inverse_softplus(magnitude(raw) / T(_DECISION_PHASE_COUNT))
end

@inline function _signed_input(input::UInt8, receptor::ReceptorKind, value)
    value >= zero(value) && return Int(input), value, one(value)
    compartment = div(Int(input) - 1, ActiveApicalCell.INPUT_CHANNELS) + 1
    complement = receptor == GABA_RECEPTOR ? AMPA_RECEPTOR : GABA_RECEPTOR
    return ActiveApicalCell.input_index(compartment, Int(complement)),
        -value,
        -one(value)
end

"""
Create deterministic relation-fastest raw magnitudes.

The historical `[-3,-2)` contact scale describes one integration exposure.
Canonical decision cells hold context for three physical phases, so the
physical softplus magnitude is divided by three before converting back to raw
space. Every relation/source remains an independent parameter.
"""
function default_raw_magnitudes(
    ::Type{T}=Float32;
    seed::Integer=0x4d41474e49545544,
) where {T<:AbstractFloat}
    seed >= 0 || throw(ArgumentError("magnitude seed must be nonnegative"))
    seed_value = UInt64(seed)
    raw = Matrix{T}(undef, CONTEXT_FANOUT, CONTEXT_SOURCE_COUNT)
    @inbounds for source in 1:CONTEXT_SOURCE_COUNT
        for relation in 1:CONTEXT_FANOUT
            word = _mix64(xor(
                seed_value,
                UInt64(source) * UInt64(0x9e3779b97f4a7c15),
                UInt64(relation) * UInt64(0xbf58476d1ce4e5b9),
            ))
            fraction = T(word >> 40) / T(UInt64(1) << 24)
            base_raw = T(-3) + fraction
            raw[relation, source] = _phase_normalized_raw(base_raw)
        end
    end
    return raw
end

@inline function _check_destination(destination::AbstractMatrix)
    size(destination) == (INPUT_DIM, DECISION_CELLS) || throw(
        DimensionMismatch(
            "decision input must have shape ($INPUT_DIM, $DECISION_CELLS)",
        ),
    )
    return nothing
end

@inline function _check_raw(raw::AbstractMatrix)
    size(raw) == (CONTEXT_FANOUT, CONTEXT_SOURCE_COUNT) || throw(
        DimensionMismatch(
            "raw context magnitude must have shape " *
            "($CONTEXT_FANOUT, $CONTEXT_SOURCE_COUNT)",
        ),
    )
    return nothing
end

@inline function _check_topology(topology::ContextTopology)
    length(topology.source) == CONTEXT_SOURCE_COUNT || error(
        "context source count changed after construction",
    )
    size(topology.edge) == (CONTEXT_FANOUT, CONTEXT_SOURCE_COUNT) || error(
        "context edge shape changed after construction",
    )
    return nothing
end

@inline function _deposit_source!(
    destination::AbstractMatrix{T},
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
    source::Int,
    value,
) where {T<:AbstractFloat,R<:AbstractFloat}
    iszero(value) && return nothing
    typed_value = T(value)
    @inbounds for relation in 1:CONTEXT_FANOUT
        edge = topology.edge[relation, source]
        input, amplitude, _ = _signed_input(
            edge.input,
            edge.receptor,
            typed_value,
        )
        cell = Int(edge.decision_cell)
        destination[input, cell] = muladd(
            amplitude,
            T(magnitude(raw_magnitude[relation, source])),
            destination[input, cell],
        )
    end
    return nothing
end

"""
Add state-common queue, REN and back-to-back context to an existing `27 x 50`
decision input.

Queue values are either one-hot or all-zero independently for all six roles;
the all-zero hold role is a valid Tetris state.  Every `(piece, role)` bit is
an explicit bipolar event (`0 -> -1`, `1 -> +1`) rather than silence.  Thus an
empty role remains distinguishable without adding a special compatibility
token, while the signed-input transform maps the two bit values to
complementary E/I receptors. REN and back-to-back remain their exact Float32
values; this function performs no thresholding or quantization.
"""
function deposit_state_common!(
    destination::AbstractMatrix{T},
    common::CandidateDeltaInput.StateCommon,
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
) where {T<:AbstractFloat,R<:AbstractFloat}
    _check_destination(destination)
    _check_raw(raw_magnitude)
    _check_topology(topology)

    @inbounds for role in 1:QUEUE_ROLES
        active = 0
        for piece in 1:QUEUE_PIECES
            value = common.queue[piece, role]
            (value == 0x00 || value == 0x01) || throw(ArgumentError(
                "queue context must contain only zero/one values",
            ))
            active += Int(value)
            _deposit_source!(
                destination,
                topology,
                raw_magnitude,
                queue_source_index(piece, role),
                iszero(value) ? -one(T) : one(T),
            )
        end
        active <= 1 || throw(ArgumentError(
            "each queue role must contain at most one active piece",
        ))
    end

    ren = common.ren[1]
    back_to_back = common.back_to_back[1]
    isfinite(ren) || throw(ArgumentError("REN context must be finite"))
    isfinite(back_to_back) || throw(ArgumentError(
        "back-to-back context must be finite",
    ))
    _deposit_source!(
        destination,
        topology,
        raw_magnitude,
        REN_SOURCE_INDEX,
        ren,
    )
    _deposit_source!(
        destination,
        topology,
        raw_magnitude,
        BACK_TO_BACK_SOURCE_INDEX,
        back_to_back,
    )
    return destination
end

@inline function _aux_matrix(materialization::CandidateDeltaInput.CandidateMaterialization)
    return materialization.aux
end
@inline _aux_matrix(aux::AbstractMatrix) = aux

"""
Add exact candidate-local auxiliary Float32 values to an existing decision
input. Values are not converted to threshold rails and are not quantized.

Features 35/36 are REN/B2B mirrors in the legacy 37-value geometry vector.
They are intentionally skipped here because the exact state-common REN/B2B
sources have already deposited them once.  Feature 37 (T-spin) remains
candidate-local.
"""
function deposit_candidate_aux!(
    destination::AbstractMatrix{T},
    candidate_aux::Union{
        CandidateDeltaInput.CandidateMaterialization,
        AbstractMatrix,
    },
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
) where {T<:AbstractFloat,R<:AbstractFloat}
    _check_destination(destination)
    _check_raw(raw_magnitude)
    _check_topology(topology)
    aux = _aux_matrix(candidate_aux)
    size(aux) == (AUXILIARY_FEATURES, 1) || throw(DimensionMismatch(
        "candidate auxiliary context must have shape ($AUXILIARY_FEATURES, 1)",
    ))
    @inbounds for feature in 1:AUXILIARY_FEATURES
        feature in (35, 36) && continue
        value = aux[feature]
        isfinite(value) || throw(ArgumentError(
            "candidate auxiliary context must be finite",
        ))
        _deposit_source!(
            destination,
            topology,
            raw_magnitude,
            auxiliary_source_index(feature),
            value,
        )
    end
    return destination
end

@inline function _source_pullback!(
    raw_bar::AbstractMatrix{B},
    input_bar::AbstractMatrix{I},
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
    source::Int,
    value,
) where {B<:AbstractFloat,I<:AbstractFloat,R<:AbstractFloat}
    source_bar = zero(I)
    typed_value = I(value)
    @inbounds for relation in 1:CONTEXT_FANOUT
        edge = topology.edge[relation, source]
        input, amplitude, value_derivative = _signed_input(
            edge.input,
            edge.receptor,
            typed_value,
        )
        cotangent = input_bar[input, Int(edge.decision_cell)]
        raw = raw_magnitude[relation, source]
        source_bar = muladd(
            cotangent * value_derivative,
            I(magnitude(raw)),
            source_bar,
        )
        raw_bar[relation, source] = muladd(
            B(cotangent * amplitude),
            B(magnitude_derivative(raw)),
            raw_bar[relation, source],
        )
    end
    return source_bar
end

"""
Accumulate the exact VJP for [`deposit_state_common!`](@ref).

All bar arrays are additive. `queue_bar` is Float storage even though the
forward queue is discrete; it records the conditional derivative on the
currently selected bipolar receptor branch.  It is useful for diagnostics but
is not a surrogate for a discrete zero/one flip, which changes receptor type.
"""
function state_common_pullback!(
    queue_bar::AbstractMatrix{Q},
    ren_bar::AbstractMatrix{N},
    back_to_back_bar::AbstractMatrix{K},
    raw_bar::AbstractMatrix{B},
    input_bar::AbstractMatrix{I},
    common::CandidateDeltaInput.StateCommon,
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
) where {
    Q<:AbstractFloat,
    N<:AbstractFloat,
    K<:AbstractFloat,
    B<:AbstractFloat,
    I<:AbstractFloat,
    R<:AbstractFloat,
}
    size(queue_bar) == (QUEUE_PIECES, QUEUE_ROLES) || throw(
        DimensionMismatch("queue cotangent must have shape (7, 6)"),
    )
    size(ren_bar) == (1, 1) || throw(DimensionMismatch(
        "REN cotangent must have shape (1, 1)",
    ))
    size(back_to_back_bar) == (1, 1) || throw(DimensionMismatch(
        "back-to-back cotangent must have shape (1, 1)",
    ))
    _check_raw(raw_bar)
    _check_destination(input_bar)
    _check_raw(raw_magnitude)
    _check_topology(topology)

    @inbounds for role in 1:QUEUE_ROLES, piece in 1:QUEUE_PIECES
        value = common.queue[piece, role]
        bipolar_value = iszero(value) ? -one(I) : one(I)
        source_bar = _source_pullback!(
            raw_bar,
            input_bar,
            topology,
            raw_magnitude,
            queue_source_index(piece, role),
            bipolar_value,
        )
        # d(2x-1)/dx = 2 on the currently selected receptor branch.  A hard
        # bit flip also changes receptor identity and is deliberately not
        # represented as a continuous queue gradient.
        queue_bar[piece, role] += Q(2source_bar)
    end
    ren_bar[1] += N(_source_pullback!(
        raw_bar,
        input_bar,
        topology,
        raw_magnitude,
        REN_SOURCE_INDEX,
        common.ren[1],
    ))
    back_to_back_bar[1] += K(_source_pullback!(
        raw_bar,
        input_bar,
        topology,
        raw_magnitude,
        BACK_TO_BACK_SOURCE_INDEX,
        common.back_to_back[1],
    ))
    return queue_bar, ren_bar, back_to_back_bar, raw_bar
end

"""Accumulate the exact VJP for [`deposit_candidate_aux!`](@ref)."""
function candidate_aux_pullback!(
    aux_bar::AbstractMatrix{A},
    raw_bar::AbstractMatrix{B},
    input_bar::AbstractMatrix{I},
    candidate_aux::Union{
        CandidateDeltaInput.CandidateMaterialization,
        AbstractMatrix,
    },
    topology::ContextTopology,
    raw_magnitude::AbstractMatrix{R},
) where {
    A<:AbstractFloat,
    B<:AbstractFloat,
    I<:AbstractFloat,
    R<:AbstractFloat,
}
    size(aux_bar) == (AUXILIARY_FEATURES, 1) || throw(DimensionMismatch(
        "auxiliary cotangent must have shape ($AUXILIARY_FEATURES, 1)",
    ))
    _check_raw(raw_bar)
    _check_destination(input_bar)
    _check_raw(raw_magnitude)
    _check_topology(topology)
    aux = _aux_matrix(candidate_aux)
    size(aux) == (AUXILIARY_FEATURES, 1) || throw(DimensionMismatch(
        "candidate auxiliary context must have shape ($AUXILIARY_FEATURES, 1)",
    ))

    @inbounds for feature in 1:AUXILIARY_FEATURES
        feature in (35, 36) && continue
        source_bar = _source_pullback!(
            raw_bar,
            input_bar,
            topology,
            raw_magnitude,
            auxiliary_source_index(feature),
            aux[feature],
        )
        aux_bar[feature] += A(source_bar)
    end
    return aux_bar, raw_bar
end

end # module ContextAfferents
