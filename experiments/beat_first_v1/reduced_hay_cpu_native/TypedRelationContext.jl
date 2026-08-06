module TypedRelationContext

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..DendriticRelationTopology
using ..TypedDendriticAfferents

const Cell = ActiveApicalCell
const Input = CandidateDeltaInput
const RelationTopology = DendriticRelationTopology
const Afferents = TypedDendriticAfferents

export AUX_RELATION_FANOUT,
       AUX_SOURCE_COUNT,
       BACK_TO_BACK_SOURCE,
       COMMON_OUTPUT_FANOUT,
       COMMON_RELATION_FANOUT,
       COMMON_SOURCE_COUNT,
       QUEUE_SOURCE_COUNT,
       REN_SOURCE,
       RelationContextGradient,
       RelationContextGraphs,
       RelationContextScratch,
       accumulate_aux_pullback!,
       accumulate_common_pullback!,
       aux_feature_index,
       aux_source_index,
       build_relation_context,
       clear_context_gradient!,
       clear_packet_bars!,
       deposit_candidate_aux_context!,
       deposit_common_context!,
       pack_candidate_aux!,
       pack_state_common!,
       pullback_candidate_aux_context!,
       pullback_common_context!,
       queue_source_index

const QUEUE_SOURCE_COUNT = Input.QUEUE_PIECES * Input.QUEUE_TOKENS
const REN_SOURCE = QUEUE_SOURCE_COUNT + 1
const BACK_TO_BACK_SOURCE = REN_SOURCE + 1
const COMMON_SOURCE_COUNT = BACK_TO_BACK_SOURCE
# Auxiliary coordinates 35/36 repeat state-common REN/B2B.  They are not
# candidate-local sources here: common context owns those identities exactly
# once.  Candidate context retains 1:34 plus feature 37 (T-spin).
const AUX_SOURCE_COUNT = Input.AUX_FEATURES - 2

const COMMON_RELATION_FANOUT = 8
const COMMON_OUTPUT_FANOUT = 8
const AUX_RELATION_FANOUT = 8

const _COMMON_RELATION_SEED = UInt64(0x4354_5854_5f52_454c)
const _COMMON_OUTPUT_SEED = UInt64(0x4354_5854_5f4f_5554)
const _AUX_RELATION_SEED = UInt64(0x4155_585f_5f52_454c)
const _COMMON_INITIAL_CONDUCTANCE = 0.1
const _AUX_INITIAL_CONDUCTANCE = 0.05

@assert QUEUE_SOURCE_COUNT == 42
@assert COMMON_SOURCE_COUNT == 44
@assert AUX_SOURCE_COUNT == 35
@assert RelationTopology.RELATION_COUNT == 48
@assert RelationTopology.CROSS_ACTION_RELATION_COUNT == 14
@assert Cell.INPUT_DIM == 27

"""
The three explicit typed context graphs used by the candidate-delta relation
model.

Context never passes through a dense projection or a pooled workspace.  Queue
role, queue piece, REN/B2B identity, and every candidate auxiliary coordinate
remain distinct source-major identities all the way to fixed dendritic
contacts.  Only contact conductance is trainable.
"""
struct RelationContextGraphs{T<:AbstractFloat}
    common_relation::Afferents.TypedAfferentGraph{T}
    common_output::Afferents.TypedAfferentGraph{T}
    aux_relation::Afferents.TypedAfferentGraph{T}
end

"""Caller-owned packets and packet cotangents; no hot-path allocation."""
struct RelationContextScratch{T<:AbstractFloat}
    common_packet::Matrix{T}
    common_packet_bar::Matrix{T}
    aux_packet::Matrix{T}
    aux_packet_bar::Matrix{T}
end

RelationContextScratch(::Type{T}=Float32) where {T<:AbstractFloat} =
    RelationContextScratch(
        zeros(T, 2, COMMON_SOURCE_COUNT),
        zeros(T, 2, COMMON_SOURCE_COUNT),
        zeros(T, 1, AUX_SOURCE_COUNT),
        zeros(T, 1, AUX_SOURCE_COUNT),
    )

"""Trainable-conductance cotangents for the three anatomical graphs."""
struct RelationContextGradient{T<:AbstractFloat}
    common_relation_raw::Vector{T}
    common_output_raw::Vector{T}
    aux_relation_raw::Vector{T}
end

function RelationContextGradient(graphs::RelationContextGraphs{T}) where {T}
    return RelationContextGradient(
        zeros(T, Afferents.contact_count(graphs.common_relation)),
        zeros(T, Afferents.contact_count(graphs.common_output)),
        zeros(T, Afferents.contact_count(graphs.aux_relation)),
    )
end

@inline function queue_source_index(piece::Integer, role::Integer)
    1 <= piece <= Input.QUEUE_PIECES ||
        throw(BoundsError(1:Input.QUEUE_PIECES, piece))
    1 <= role <= Input.QUEUE_TOKENS ||
        throw(BoundsError(1:Input.QUEUE_TOKENS, role))
    return (Int(role) - 1) * Input.QUEUE_PIECES + Int(piece)
end

@inline function aux_source_index(feature::Integer)
    1 <= feature <= Input.AUX_FEATURES ||
        throw(BoundsError(1:Input.AUX_FEATURES, feature))
    feature <= 34 && return Int(feature)
    feature == 37 && return AUX_SOURCE_COUNT
    throw(ArgumentError(
        "aux features 35/36 are state-common REN/B2B, not candidate sources",
    ))
end

@inline function aux_feature_index(source::Integer)
    1 <= source <= AUX_SOURCE_COUNT ||
        throw(BoundsError(1:AUX_SOURCE_COUNT, source))
    return source <= 34 ? Int(source) : 37
end

@inline function _mix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function _word(seed::UInt64, source::Int, rank::Int, salt::UInt64)
    return _mix64(xor(
        seed,
        UInt64(source) * UInt64(0xd6e8feb86659fd93),
        UInt64(rank) * UInt64(0xa0761d6478bd642f),
        salt,
    ))
end

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("initial conductance must be positive"))
    return value + log(-expm1(-value))
end

@inline function _initial_raw(
    seed::UInt64,
    source::Int,
    rank::Int,
    initial_conductance::T,
    ::Type{T},
) where {T<:AbstractFloat}
    # This is a physical contact scale.  It is intentionally not divided by a
    # presumed number of recurrent phases: the canonical relation/output banks
    # own their own temporal depth and may change it independently.
    word = _word(seed, source, rank, UInt64(0x6a09e667f3bcc909))
    jitter = T((word >> 40) & UInt64(0x00ff_ffff)) / T(0x0100_0000)
    magnitude = initial_conductance * (T(0.9) + T(0.2) * jitter)
    return _inverse_softplus(magnitude)
end

@inline function _cyclic_destination(
    word::UInt64,
    rank::Int,
    first::Int,
    count::Int,
)
    # The rank term makes the few contacts of a source distinct even when two
    # hash words happen to share their low bits.
    start = Int(mod(word, UInt64(count)))
    stride = count == 10 ? 3 : 5
    return first + mod(start + stride * (rank - 1), count)
end

@inline function _common_relation_destination(
    seed::UInt64,
    source::Int,
    unit::Int,
)
    word = _word(seed, source, 0, UInt64(0xbb67ae8584caa73b))
    if unit <= 4
        first = RelationTopology.ROW_RELATION_COUNT +
                RelationTopology.COLUMN_RELATION_COUNT + 1
        return _cyclic_destination(
            word,
            unit,
            first,
            RelationTopology.CROSS_ACTION_RELATION_COUNT,
        )
    elseif unit <= 6
        return _cyclic_destination(
            word,
            unit - 4,
            1,
            RelationTopology.ROW_RELATION_COUNT,
        )
    end
    return _cyclic_destination(
        word,
        unit - 6,
        RelationTopology.ROW_RELATION_COUNT + 1,
        RelationTopology.COLUMN_RELATION_COUNT,
    )
end

@inline function _aux_relation_destination(
    seed::UInt64,
    source::Int,
    unit::Int,
)
    word = _word(seed, source, 0, UInt64(0x3c6ef372fe94f82b))
    # The first eight cross/action cells are tile-local board binders.  Global
    # candidate geometry must not overwrite that spatial identity, so auxiliary
    # context reaches only the final six small-world/action stripe cells.
    first = RelationTopology.ROW_RELATION_COUNT +
            RelationTopology.COLUMN_RELATION_COUNT + 9
    return _cyclic_destination(
        word,
        unit,
        first,
        6,
    )
end

@inline function _output_destination(
    seed::UInt64,
    source::Int,
    unit::Int,
)
    word = _word(seed, source, 0, UInt64(0xa54ff53a5f1d36f1))
    return _cyclic_destination(word, unit, 1, 22)
end

@inline function _receptor(seed::UInt64, source::Int, unit::Int)
    word = _word(seed, source, unit, UInt64(0x510e527fade682d1))
    return UInt8(mod(word, UInt64(Cell.INPUT_CHANNELS)) + 1)
end

@inline function _positive_compartment(seed::UInt64, source::Int, unit::Int)
    word = _word(seed, source, unit, UInt64(0x9b05688c2b3e6c1f))
    return Int(mod(word, UInt64(Cell.N_COMPARTMENTS))) + 1
end

@inline function _negative_compartment(
    positive::Int,
    seed::UInt64,
    source::Int,
    unit::Int,
)
    word = _word(seed, source, unit, UInt64(0x1f83d9abfb41bd6b))
    offset = Int(mod(word, UInt64(Cell.N_COMPARTMENTS - 1))) + 1
    return mod(positive - 1 + offset, Cell.N_COMPARTMENTS) + 1
end

function _explicit_graph(
    source_count::Int,
    field_kind::Vector{UInt8},
    source_field_kind::Vector{UInt16},
    signed_source::BitVector,
    destination_count::Int,
    fanout::Int,
    seed::UInt64,
    destination_for::F,
    initial_conductance::Real,
    ::Type{T},
) where {F,T<:AbstractFloat}
    iseven(fanout) || throw(ArgumentError("opponent fanout must be even"))
    length(source_field_kind) == source_count ||
        throw(DimensionMismatch("source field selector has the wrong length"))
    length(signed_source) == source_count ||
        throw(DimensionMismatch("source sign selector has the wrong length"))
    initial_conductance > 0 ||
        throw(ArgumentError("initial conductance must be positive"))
    physical_initial_conductance = T(initial_conductance)

    slots = source_count * fanout
    fields = Vector{UInt16}(undef, slots)
    polarities = Vector{Int8}(undef, slots)
    cells = Vector{UInt16}(undef, slots)
    compartments = Vector{UInt8}(undef, slots)
    receptors = Vector{UInt8}(undef, slots)
    raw = Vector{T}(undef, slots)

    @inbounds for source in 1:source_count
        first_slot = (source - 1) * fanout + 1
        if signed_source[source]
            # A signed scalar is represented by anatomical opponent contacts.
            # Both members of a pair retain the same receptor identity; the
            # semantic sign therefore never turns excitation into inhibition.
            for unit in 1:div(fanout, 2)
                destination = destination_for(seed, source, unit)
                receptor = _receptor(seed, source, unit)
                positive_compartment =
                    _positive_compartment(seed, source, unit)
                negative_compartment = _negative_compartment(
                    positive_compartment,
                    seed,
                    source,
                    unit,
                )
                for member in 1:2
                    rank = 2 * unit - 2 + member
                    slot = first_slot + rank - 1
                    fields[slot] = source_field_kind[source]
                    polarities[slot] = member == 1 ? Int8(1) : Int8(-1)
                    cells[slot] = UInt16(destination)
                    compartments[slot] = UInt8(
                        member == 1 ? positive_compartment : negative_compartment,
                    )
                    receptors[slot] = receptor
                    raw[slot] = _initial_raw(
                        seed,
                        source,
                        rank,
                        physical_initial_conductance,
                        T,
                    )
                end
            end
        else
            # Queue sources are exact hard 0/1 events.  Zero is silence, not a
            # learned negative value, so these contacts never invent an OFF rail.
            for rank in 1:fanout
                slot = first_slot + rank - 1
                fields[slot] = source_field_kind[source]
                polarities[slot] = Int8(1)
                cells[slot] = UInt16(destination_for(seed, source, rank))
                compartments[slot] = UInt8(
                    _positive_compartment(seed, source, rank),
                )
                receptors[slot] = _receptor(seed, source, rank)
                raw[slot] = _initial_raw(
                    seed,
                    source,
                    rank,
                    physical_initial_conductance,
                    T,
                )
            end
        end
    end

    return Afferents.TypedAfferentGraph(
        source_count,
        field_kind,
        destination_count,
        fanout,
        fields,
        polarities,
        cells,
        compartments,
        receptors,
        raw,
    )
end

"""
Build the canonical seed-fixed typed relation context.

All fanouts and seeds are part of this module's architecture contract.  There
is deliberately no phase-count argument and no hidden three-phase scaling.
"""
function build_relation_context(::Type{T}=Float32) where {T<:AbstractFloat}
    common_field_kind = UInt8[Afferents.ANALOG_FIELD, Afferents.HARD_BIT_FIELD]
    common_source_field = fill(UInt16(2), COMMON_SOURCE_COUNT)
    common_source_field[REN_SOURCE] = UInt16(1)
    common_source_field[BACK_TO_BACK_SOURCE] = UInt16(1)
    common_signed = falses(COMMON_SOURCE_COUNT)
    common_signed[REN_SOURCE] = true
    common_signed[BACK_TO_BACK_SOURCE] = true

    aux_field_kind = UInt8[Afferents.ANALOG_FIELD]
    aux_source_field = fill(UInt16(1), AUX_SOURCE_COUNT)
    aux_signed = trues(AUX_SOURCE_COUNT)

    common_relation = _explicit_graph(
        COMMON_SOURCE_COUNT,
        common_field_kind,
        common_source_field,
        common_signed,
        RelationTopology.RELATION_COUNT,
        COMMON_RELATION_FANOUT,
        _COMMON_RELATION_SEED,
        _common_relation_destination,
        _COMMON_INITIAL_CONDUCTANCE,
        T,
    )
    common_output = _explicit_graph(
        COMMON_SOURCE_COUNT,
        common_field_kind,
        common_source_field,
        common_signed,
        22,
        COMMON_OUTPUT_FANOUT,
        _COMMON_OUTPUT_SEED,
        _output_destination,
        _COMMON_INITIAL_CONDUCTANCE,
        T,
    )
    aux_relation = _explicit_graph(
        AUX_SOURCE_COUNT,
        aux_field_kind,
        aux_source_field,
        aux_signed,
        RelationTopology.RELATION_COUNT,
        AUX_RELATION_FANOUT,
        _AUX_RELATION_SEED,
        _aux_relation_destination,
        _AUX_INITIAL_CONDUCTANCE,
        T,
    )
    return RelationContextGraphs(
        common_relation,
        common_output,
        aux_relation,
    )
end

@inline function _check_common_packet(packet)
    size(packet) == (2, COMMON_SOURCE_COUNT) || throw(DimensionMismatch(
        "common packet must have shape (2, $COMMON_SOURCE_COUNT)",
    ))
    return nothing
end


@inline function _check_aux_packet(packet)
    size(packet) == (1, AUX_SOURCE_COUNT) || throw(DimensionMismatch(
        "aux packet must have shape (1, $AUX_SOURCE_COUNT)",
    ))
    return nothing
end


"""Pack 42 role-preserving hard queue bits plus signed REN and B2B scalars."""
function pack_state_common!(
    packet::AbstractMatrix{T},
    common::Input.StateCommon,
) where {T<:AbstractFloat}
    _check_common_packet(packet)
    fill!(packet, zero(T))
    @inbounds for role in 1:Input.QUEUE_TOKENS, piece in 1:Input.QUEUE_PIECES
        source = queue_source_index(piece, role)
        packet[2, source] = T(common.queue[piece, role])
    end
    packet[1, REN_SOURCE] = T(common.ren[1])
    packet[1, BACK_TO_BACK_SOURCE] = T(common.back_to_back[1])
    return packet
end

"""
Pack candidate-only auxiliary coordinates 1:34 and 37 without pooling/scaling.

Features 35/36 duplicate REN/B2B and are deliberately excluded: those values
already enter through the two signed state-common sources.
"""
function pack_candidate_aux!(
    packet::AbstractMatrix{T},
    materialization::Input.CandidateMaterialization,
) where {T<:AbstractFloat}
    _check_aux_packet(packet)
    @inbounds for source in 1:AUX_SOURCE_COUNT
        packet[1, source] =
            T(materialization.aux[aux_feature_index(source)])
    end
    return packet
end

"""Deposit state-common context into `27 x 48` and `27 x 22` inboxes."""
function deposit_common_context!(
    relation_inbox::AbstractMatrix{T},
    output_inbox::AbstractMatrix{T},
    graphs::RelationContextGraphs{T},
    common::Input.StateCommon,
    scratch::RelationContextScratch{T},
) where {T<:AbstractFloat}
    pack_state_common!(scratch.common_packet, common)
    Afferents.deposit_typed!(
        relation_inbox,
        graphs.common_relation,
        scratch.common_packet,
    )
    Afferents.deposit_typed!(
        output_inbox,
        graphs.common_output,
        scratch.common_packet,
    )
    return relation_inbox, output_inbox
end

"""
Deposit candidate-local auxiliaries into six small-world/action stripe cells.
Tile-local relation cells 35:42 remain exclusively owned by board/placement
binding.  Candidate auxiliaries reach outputs only through relation cells.
"""
function deposit_candidate_aux_context!(
    relation_inbox::AbstractMatrix{T},
    graphs::RelationContextGraphs{T},
    materialization::Input.CandidateMaterialization,
    scratch::RelationContextScratch{T},
) where {T<:AbstractFloat}
    pack_candidate_aux!(scratch.aux_packet, materialization)
    Afferents.deposit_typed!(
        relation_inbox,
        graphs.aux_relation,
        scratch.aux_packet,
    )
    return relation_inbox
end

"""Exact additive pullback of both state-common typed deposits."""
function pullback_common_context!(
    scratch::RelationContextScratch{T},
    gradient::RelationContextGradient{T},
    graphs::RelationContextGraphs{T},
    relation_inbox_bar::AbstractMatrix{T},
    output_inbox_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    Afferents.deposit_typed_pullback!(
        scratch.common_packet_bar,
        gradient.common_relation_raw,
        graphs.common_relation,
        scratch.common_packet,
        relation_inbox_bar,
    )
    Afferents.deposit_typed_pullback!(
        scratch.common_packet_bar,
        gradient.common_output_raw,
        graphs.common_output,
        scratch.common_packet,
        output_inbox_bar,
    )
    return scratch.common_packet_bar, gradient
end

"""Exact additive pullback of the relation-only candidate-auxiliary deposit."""
function pullback_candidate_aux_context!(
    scratch::RelationContextScratch{T},
    gradient::RelationContextGradient{T},
    graphs::RelationContextGraphs{T},
    relation_inbox_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    Afferents.deposit_typed_pullback!(
        scratch.aux_packet_bar,
        gradient.aux_relation_raw,
        graphs.aux_relation,
        scratch.aux_packet,
        relation_inbox_bar,
    )
    return scratch.aux_packet_bar, gradient
end

"""Accumulate differentiable REN/B2B source cotangents; queue bits stay hard."""
function accumulate_common_pullback!(
    ren_bar::AbstractMatrix{T},
    back_to_back_bar::AbstractMatrix{T},
    packet_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    size(ren_bar) == (1, 1) ||
        throw(DimensionMismatch("REN cotangent must have shape (1, 1)"))
    size(back_to_back_bar) == (1, 1) || throw(DimensionMismatch(
        "back-to-back cotangent must have shape (1, 1)",
    ))
    _check_common_packet(packet_bar)
    ren_bar[1] += packet_bar[1, REN_SOURCE]
    back_to_back_bar[1] += packet_bar[1, BACK_TO_BACK_SOURCE]
    return ren_bar, back_to_back_bar
end

"""Accumulate candidate-only cotangents; REN/B2B duplicate slots stay zero."""
function accumulate_aux_pullback!(
    aux_bar::AbstractMatrix{T},
    packet_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    size(aux_bar) == (Input.AUX_FEATURES, 1) || throw(DimensionMismatch(
        "aux cotangent must have shape ($(Input.AUX_FEATURES), 1)",
    ))
    _check_aux_packet(packet_bar)
    @inbounds for source in 1:AUX_SOURCE_COUNT
        aux_bar[aux_feature_index(source)] += packet_bar[1, source]
    end
    return aux_bar
end

function clear_packet_bars!(scratch::RelationContextScratch)
    fill!(scratch.common_packet_bar, zero(eltype(scratch.common_packet_bar)))
    fill!(scratch.aux_packet_bar, zero(eltype(scratch.aux_packet_bar)))
    return scratch
end

function clear_context_gradient!(gradient::RelationContextGradient)
    fill!(gradient.common_relation_raw, zero(eltype(gradient.common_relation_raw)))
    fill!(gradient.common_output_raw, zero(eltype(gradient.common_output_raw)))
    fill!(gradient.aux_relation_raw, zero(eltype(gradient.aux_relation_raw)))
    return gradient
end

end # module TypedRelationContext
