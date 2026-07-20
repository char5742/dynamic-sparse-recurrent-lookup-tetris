module BeatFirstSparseFeatures

export CANDIDATE_BOARD_HEIGHT,
       CANDIDATE_BOARD_WIDTH,
       CANDIDATE_BOARD_CELLS,
       NEXT_HOLD_PIECES,
       NEXT_HOLD_TOKENS,
       NEXT_HOLD_FEATURES,
       AUX_FEATURES,
       ROUTE_FEATURES,
       VALUE_FEATURES,
       BOARD_ROUTE_SKETCH_SCALE,
       BOARD_ROUTE_SKETCH_MULADDS,
       ROUTE_AUX_INDICES,
       VALUE_AUX_INDICES,
       board_route_sketch_slot,
       board_route_sketch_sign,
       validate_candidate_feature_input,
       split_candidate_features!,
       split_candidate_features

const CANDIDATE_BOARD_HEIGHT = 24
const CANDIDATE_BOARD_WIDTH = 10
const CANDIDATE_BOARD_CELLS = CANDIDATE_BOARD_HEIGHT * CANDIDATE_BOARD_WIDTH

const NEXT_HOLD_PIECES = 7
const NEXT_HOLD_TOKENS = 6
const NEXT_HOLD_FEATURES = NEXT_HOLD_PIECES * NEXT_HOLD_TOKENS
const AUX_FEATURES = 37

const ROUTE_FEATURES = 64
const VALUE_FEATURES = 496

# A fixed balanced CountSketch makes the hard route depend on raw board cells
# without adding a learned dense stem or changing the 64-dimensional contract.
# Across candidate+difference streams, every route slot receives seven or eight
# signed cells. The sketch is computed from this independent candidate only.
const BOARD_ROUTE_SKETCH_SCALE =
    sqrt(Float32(ROUTE_FEATURES) / Float32(2 * CANDIDATE_BOARD_CELLS))
const BOARD_ROUTE_SKETCH_MULADDS = 2 * CANDIDATE_BOARD_CELLS

@inline function board_route_sketch_slot(stream::Integer, linear_cell::Integer)
    stream == 1 || stream == 2 ||
        throw(ArgumentError("board sketch stream must be 1 (candidate) or 2 (difference)"))
    1 <= linear_cell <= CANDIDATE_BOARD_CELLS ||
        throw(BoundsError(Base.OneTo(CANDIDATE_BOARD_CELLS), linear_cell))
    multiplier = stream == 1 ? 13 : 17
    offset = stream == 1 ? 7 : 24
    return mod(multiplier * (Int(linear_cell) - 1) + offset, ROUTE_FEATURES) + 1
end

@inline function board_route_sketch_sign(stream::Integer, linear_cell::Integer)
    stream == 1 || stream == 2 ||
        throw(ArgumentError("board sketch stream must be 1 (candidate) or 2 (difference)"))
    1 <= linear_cell <= CANDIDATE_BOARD_CELLS ||
        throw(BoundsError(Base.OneTo(CANDIDATE_BOARD_CELLS), linear_cell))
    salt = stream == 1 ? UInt32(0x9e3779b9) : UInt32(0x85ebca6b)
    word = UInt32(linear_cell) * salt
    return isodd(count_ones(word)) ? -1.0f0 : 1.0f0
end

# Route input: column heights, holes per column, unreachable cavities, and
# maximum height. These indices follow BeatFirstModels.INPUT_CONTRACT.
const ROUTE_AUX_INDICES = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    31, 34,
)

# Value input: well depths, aggregate height, skyline bumpiness, REN, B2B,
# and T-spin. The trailing value feature is a constant one, not an aux row.
const VALUE_AUX_INDICES = (
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    32, 33, 35, 36, 37,
)

NEXT_HOLD_FEATURES + length(ROUTE_AUX_INDICES) == ROUTE_FEATURES ||
    error("invalid route feature layout")
2 * CANDIDATE_BOARD_CELLS + length(VALUE_AUX_INDICES) + 1 == VALUE_FEATURES ||
    error("invalid value feature layout")

"""Validate the self-contained tensors used by the dynamic sparse model.

The split intentionally consumes one independent candidate only. Required
fields and shapes are:

  * `candidate`: `24 x 10 x 1 x N`
  * `difference`: `24 x 10 x 1 x N`
  * `next_hold`: `7 x 6 x N`
  * `aux`: `37 x N`

No parent-board cache, NNUE accumulator, or cached/external sketch is consulted.
The fixed CountSketch is recomputed directly from this independent candidate.
Validation is shape-only and constant-time; canonical input construction is
responsible for finite-value checks. The returned integer is the number of
candidates `N`.
"""
function validate_candidate_feature_input(input)
    hasproperty(input, :candidate) ||
        throw(ArgumentError("input is missing required field :candidate"))
    hasproperty(input, :difference) ||
        throw(ArgumentError("input is missing required field :difference"))
    hasproperty(input, :next_hold) ||
        throw(ArgumentError("input is missing required field :next_hold"))
    hasproperty(input, :aux) ||
        throw(ArgumentError("input is missing required field :aux"))

    candidate = input.candidate
    difference = input.difference
    next_hold = input.next_hold
    aux = input.aux

    n = size(candidate, 4)
    size(candidate) == (
        CANDIDATE_BOARD_HEIGHT,
        CANDIDATE_BOARD_WIDTH,
        1,
        n,
    ) || throw(DimensionMismatch("candidate must be 24x10x1xN"))
    size(difference) == size(candidate) ||
        throw(DimensionMismatch("difference shape must equal candidate shape"))
    size(next_hold) == (NEXT_HOLD_PIECES, NEXT_HOLD_TOKENS, n) ||
        throw(DimensionMismatch("next_hold must be 7x6xN"))
    size(aux) == (AUX_FEATURES, n) ||
        throw(DimensionMismatch("aux must be 37xN"))

    return n
end

@inline function _validate_destination(destination, expected::Int, name::String)
    ndims(destination) == 1 ||
        throw(DimensionMismatch("$name must be a one-dimensional vector"))
    axes(destination, 1) == Base.OneTo(expected) ||
        throw(DimensionMismatch("$name must use one-based axes of length $expected"))
    return nothing
end

@inline function _validate_candidate_index(candidate_index::Integer, n::Int)
    1 <= candidate_index <= n || throw(BoundsError(Base.OneTo(n), candidate_index))
    return nothing
end

"""Split one independent candidate into the exact route/value feature vectors.

`q64` first receives, in order, flattened `next_hold` (42 values, Julia
column-major order) followed by aux rows `1:20, 31, 34`. A fixed signed
CountSketch of all 240 candidate cells and all 240 difference cells is then
added into those same 64 slots. Thus the feature positions remain exact while
the route also changes for raw boards that share the same summaries. `x496`
receives flattened
`candidate` (240), flattened `difference` (240), aux rows
`21:30, 32, 33, 35:37`, and a final constant one.

Both destinations must be ordinary one-based vectors of lengths 64 and 496.
On a valid input this method performs no heap allocation: it uses direct,
bounded indexing and does not create views or temporary flattened arrays.
It returns `(q64, x496)` for convenience.
"""
function split_candidate_features!(
    q64::AbstractVector,
    x496::AbstractVector,
    input,
    candidate_index::Integer,
)
    n = validate_candidate_feature_input(input)
    _validate_destination(q64, ROUTE_FEATURES, "q64")
    _validate_destination(x496, VALUE_FEATURES, "x496")
    _validate_candidate_index(candidate_index, n)

    route_position = 1
    @inbounds for token in 1:NEXT_HOLD_TOKENS
        for piece in 1:NEXT_HOLD_PIECES
            q64[route_position] = input.next_hold[piece, token, candidate_index]
            route_position += 1
        end
    end
    @inbounds for aux_index in ROUTE_AUX_INDICES
        q64[route_position] = input.aux[aux_index, candidate_index]
        route_position += 1
    end

    linear_cell = 1
    @inbounds for column in 1:CANDIDATE_BOARD_WIDTH
        for row in 1:CANDIDATE_BOARD_HEIGHT
            candidate_slot = board_route_sketch_slot(1, linear_cell)
            q64[candidate_slot] += BOARD_ROUTE_SKETCH_SCALE *
                board_route_sketch_sign(1, linear_cell) *
                input.candidate[row, column, 1, candidate_index]
            difference_slot = board_route_sketch_slot(2, linear_cell)
            q64[difference_slot] += BOARD_ROUTE_SKETCH_SCALE *
                board_route_sketch_sign(2, linear_cell) *
                input.difference[row, column, 1, candidate_index]
            linear_cell += 1
        end
    end

    value_position = 1
    @inbounds for column in 1:CANDIDATE_BOARD_WIDTH
        for row in 1:CANDIDATE_BOARD_HEIGHT
            x496[value_position] = input.candidate[row, column, 1, candidate_index]
            value_position += 1
        end
    end
    @inbounds for column in 1:CANDIDATE_BOARD_WIDTH
        for row in 1:CANDIDATE_BOARD_HEIGHT
            x496[value_position] = input.difference[row, column, 1, candidate_index]
            value_position += 1
        end
    end
    @inbounds for aux_index in VALUE_AUX_INDICES
        x496[value_position] = input.aux[aux_index, candidate_index]
        value_position += 1
    end
    @inbounds x496[value_position] = one(eltype(x496))

    return q64, x496
end

"""Allocate and return `(q64, x496)` for one independent candidate.

Use `split_candidate_features!` in latency-sensitive inference and training
loops. This convenience method chooses promoted element types from the source
tensors, allocates exactly the two output vectors, then delegates to the
validated allocation-free splitter.
"""
function split_candidate_features(input, candidate_index::Integer)
    route_type = promote_type(
        Float32,
        eltype(input.next_hold),
        eltype(input.aux),
        eltype(input.candidate),
        eltype(input.difference),
    )
    value_type = promote_type(
        eltype(input.candidate),
        eltype(input.difference),
        eltype(input.aux),
    )
    q64 = Vector{route_type}(undef, ROUTE_FEATURES)
    x496 = Vector{value_type}(undef, VALUE_FEATURES)
    return split_candidate_features!(q64, x496, input, candidate_index)
end

end # module BeatFirstSparseFeatures
