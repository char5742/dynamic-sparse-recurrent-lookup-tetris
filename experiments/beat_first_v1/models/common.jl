using Lux
using Random
using Statistics

const BOARD_HEIGHT = 24
const BOARD_WIDTH = 10
const PIECE_TYPES = 7
const QUEUE_TOKENS = 6
const AUX_FEATURES = 37

"""Canonical fixed-shape candidate input used by every beat-first model.

`N == max_candidates * state_batch`; the candidate-validity mask deliberately
lives in the loss, not in the model. Spatial inputs use occupancy semantics
(one means filled):

  * `board`: pre-action board, `24 x 10 x 1 x N`
  * `candidate`: post-action/post-line-clear board, same shape
  * `difference`: exactly `candidate - board`, same shape
  * `local_mask`: changed cells plus their one-cell halo, same shape
  * `next_hold`: one-hot `(hold, next1, ..., next5)`, `7 x 6 x N`
  * `aux`: engineered candidate features, `37 x N`

Auxiliary rows are: column heights 1:10, holes per column 11:20, well
depths per column 21:30, unreachable-cavity count 31, aggregate height 32,
skyline bumpiness 33, maximum height 34, REN 35, B2B 36, T-spin 37. The
packer normalizes these quantities; see `INPUT_CONTRACT` below.
"""
const INPUT_CONTRACT = (;
    board=(BOARD_HEIGHT, BOARD_WIDTH, 1, :N),
    candidate=(BOARD_HEIGHT, BOARD_WIDTH, 1, :N),
    difference=(BOARD_HEIGHT, BOARD_WIDTH, 1, :N),
    aux=(AUX_FEATURES, :N),
    next_hold=(PIECE_TYPES, QUEUE_TOKENS, :N),
    local_mask=(BOARD_HEIGHT, BOARD_WIDTH, 1, :N),
    aux_order=(;
        column_height=1:10,
        holes_per_column=11:20,
        well_depth_per_column=21:30,
        unreachable_cavities=31,
        aggregate_height=32,
        skyline_bumpiness=33,
        maximum_height=34,
        ren=35,
        back_to_back=36,
        tspin=37,
    ),
    aux_scale=(;
        column_height=24.0f0,
        holes_per_column=24.0f0,
        well_depth_per_column=24.0f0,
        unreachable_cavities=240.0f0,
        aggregate_height=240.0f0,
        skyline_bumpiness=216.0f0,
        maximum_height=24.0f0,
        ren=30.0f0,
    ),
)

"""Validate a host-side packed input before it enters a compiled train step."""
function validate_candidate_input(input)
    n = size(input.board, 4)
    size(input.board) == (BOARD_HEIGHT, BOARD_WIDTH, 1, n) ||
        error("board must be 24x10x1xN")
    size(input.candidate) == size(input.board) ||
        error("candidate shape differs from board")
    size(input.difference) == size(input.board) ||
        error("difference shape differs from board")
    size(input.local_mask) == size(input.board) ||
        error("local_mask shape differs from board")
    size(input.aux) == (AUX_FEATURES, n) || error("aux must be 37xN")
    size(input.next_hold) == (PIECE_TYPES, QUEUE_TOKENS, n) ||
        error("next_hold must be 7x6xN")
    return n
end

"""Explicit adapter for the historical six-tensor input.

Engineered `aux` and `local_mask` are supplied by the packer so the model never
contains variable-control-flow board analysis. The historical tensors use zero
for filled cells, hence the occupancy inversion here.
"""
function pack_legacy_candidate_input(
    legacy_input; aux, local_mask
)
    old_board, old_candidate, _, _, _, next_hold = legacy_input
    board = 1.0f0 .- old_board
    candidate = 1.0f0 .- old_candidate
    input = (;
        board,
        candidate,
        difference=candidate .- board,
        aux,
        next_hold,
        local_mask,
    )
    validate_candidate_input(input)
    return input
end

"""Position-aware encoder for the six short HOLD/NEXT tokens."""
struct NextHoldEncoder <: Lux.AbstractLuxContainerLayer{(:token, :summary)}
    token
    summary
end

function NextHoldEncoder(; token_features::Int=32, output_features::Int=64)
    return NextHoldEncoder(
        Dense(PIECE_TYPES => token_features, gelu),
        Dense(QUEUE_TOKENS * token_features => output_features, gelu),
    )
end

function (layer::NextHoldEncoder)(next_hold, ps, st)
    token, token_st = layer.token(next_hold, ps.token, st.token)
    token = reshape(token, :, size(token, 3))
    encoded, summary_st = layer.summary(token, ps.summary, st.summary)
    return encoded, (; token=token_st, summary=summary_st)
end

"""Shared scalar/death/distributional output projection."""
struct CandidateHeads <: Lux.AbstractLuxContainerLayer{
    (:shared, :q, :death, :quantiles)
}
    shared
    q
    death
    quantiles
    n_quantiles::Int
end

"""Current multi-task head; legacy `CandidateHeads` remains checkpoint-compatible."""
struct GeometryCandidateHeads <: Lux.AbstractLuxContainerLayer{
    (:shared, :q, :death, :quantiles, :geometry)
}
    shared
    q
    death
    quantiles
    geometry
    n_quantiles::Int
end

function CandidateHeads(
    input_features::Int; hidden_features::Int=256, n_quantiles::Int=16
)
    n_quantiles >= 1 || error("n_quantiles must be positive")
    return CandidateHeads(
        Chain(
            Dense(input_features => hidden_features, gelu),
            Dense(hidden_features => hidden_features, gelu),
        ),
        Dense(hidden_features => 1),
        Dense(hidden_features => 1),
        Dense(hidden_features => n_quantiles),
        n_quantiles,
    )
end

function GeometryCandidateHeads(
    input_features::Int; hidden_features::Int=256, n_quantiles::Int=16
)
    n_quantiles >= 1 || error("n_quantiles must be positive")
    return GeometryCandidateHeads(
        Chain(
            Dense(input_features => hidden_features, gelu),
            Dense(hidden_features => hidden_features, gelu),
        ),
        Dense(hidden_features => 1),
        Dense(hidden_features => 1),
        Dense(hidden_features => n_quantiles),
        Dense(hidden_features => 4),
        n_quantiles,
    )
end

make_candidate_heads(
    input_features::Int;
    hidden_features::Int,
    n_quantiles::Int,
    geometry_heads::Bool,
) = geometry_heads ?
    GeometryCandidateHeads(input_features; hidden_features, n_quantiles) :
    CandidateHeads(input_features; hidden_features, n_quantiles)

function (layer::CandidateHeads)(features, ps, st)
    shared, shared_st = layer.shared(features, ps.shared, st.shared)
    q, q_st = layer.q(shared, ps.q, st.q)
    death_logit, death_st = layer.death(shared, ps.death, st.death)
    quantiles, quantiles_st = layer.quantiles(
        shared, ps.quantiles, st.quantiles
    )
    output = (; q, death_logit, quantiles)
    next_state = (;
        shared=shared_st,
        q=q_st,
        death=death_st,
        quantiles=quantiles_st,
    )
    return output, next_state
end

function (layer::GeometryCandidateHeads)(features, ps, st)
    shared, shared_st = layer.shared(features, ps.shared, st.shared)
    q, q_st = layer.q(shared, ps.q, st.q)
    death_logit, death_st = layer.death(shared, ps.death, st.death)
    quantiles, quantiles_st = layer.quantiles(
        shared, ps.quantiles, st.quantiles
    )
    geometry, geometry_st = layer.geometry(shared, ps.geometry, st.geometry)
    output = (; q, death_logit, quantiles, geometry)
    next_state = (;
        shared=shared_st,
        q=q_st,
        death=death_st,
        quantiles=quantiles_st,
        geometry=geometry_st,
    )
    return output, next_state
end

global_mean_2d(x) = reshape(mean(x; dims=(1, 2)), size(x, 3), size(x, 4))

function local_mean_2d(x, mask)
    numerator = sum(x .* mask; dims=(1, 2))
    denominator = max.(sum(mask; dims=(1, 2)), 1.0f0)
    return reshape(numerator ./ denominator, size(x, 3), size(x, 4))
end

"""Fixed 4x5 mean grid for the canonical 24x10 candidate board.

The grid retains coarse height and column location while moving the expensive
capacity out of the full-resolution spatial trunk. Each output cell summarizes
an exact 6x2 input tile, so this operation has no padding or boundary ambiguity.
"""
function fixed_grid_mean_4x5(x)
    size(x, 1) == BOARD_HEIGHT || error("grid pool expects board height 24")
    size(x, 2) == BOARD_WIDTH || error("grid pool expects board width 10")
    channels = size(x, 3)
    batch = size(x, 4)
    tiled = reshape(x, 6, 4, 2, 5, channels, batch)
    pooled = mean(tiled; dims=(1, 3))
    return reshape(pooled, 20 * channels, batch)
end

"""Count nested Lux parameters without importing an optimizer package."""
parameter_count(x::AbstractArray) = length(x)
parameter_count(x::NamedTuple) = sum(parameter_count, values(x); init=0)
parameter_count(x::Tuple) = sum(parameter_count, x; init=0)
parameter_count(x) = 0

"""Count every trainable array leaf, including every auxiliary output head."""
all_parameter_count(ps) = parameter_count(ps)

function smoke_input(; candidates::Int=74, state_batch::Int=1)
    n = candidates * state_batch
    board = zeros(Float32, BOARD_HEIGHT, BOARD_WIDTH, 1, n)
    candidate = zeros(Float32, BOARD_HEIGHT, BOARD_WIDTH, 1, n)
    difference = zeros(Float32, BOARD_HEIGHT, BOARD_WIDTH, 1, n)
    aux = zeros(Float32, AUX_FEATURES, n)
    next_hold = zeros(Float32, PIECE_TYPES, QUEUE_TOKENS, n)
    local_mask = ones(Float32, BOARD_HEIGHT, BOARD_WIDTH, 1, n)
    return (; board, candidate, difference, aux, next_hold, local_mask)
end

"""Setup/forward hook; intentionally not executed at include time."""
function forward_smoke(model; seed::UInt64=0x426561745631, candidates::Int=74)
    ps, st = Lux.setup(Xoshiro(seed), model)
    input = smoke_input(; candidates)
    output, next_state = model(input, ps, st)
    n = candidates
    size(output.q) == (1, n) || error("q smoke shape mismatch")
    size(output.death_logit) == (1, n) || error("death smoke shape mismatch")
    size(output.quantiles, 2) == n || error("quantile smoke shape mismatch")
    if hasproperty(output, :geometry)
        size(output.geometry) == (4, n) || error("geometry smoke shape mismatch")
        all(isfinite, output.geometry) || error("geometry smoke contains non-finite values")
    end
    all(isfinite, output.q) || error("q smoke contains non-finite values")
    return (; output, next_state, parameters=parameter_count(ps))
end
