module EpisodicViTRecurrentLookup

using LinearAlgebra
using Random

if !isdefined(Main, :DynamicSparseRecurrentLookup)
    Base.include(
        Main,
        joinpath(
            @__DIR__, "..", "dynamic_sparse_recurrent_lookup",
            "DynamicSparseRecurrentLookup.jl",
        ),
    )
end
const SparseLookup = Main.DynamicSparseRecurrentLookup
const SparseEngine = Main.ResidualLookupSlide

# This module deliberately defines its input boundary independently of the
# historical sparse-feature adapters.  These are exactly the five tensors read
# by PreActECAQ.  `local_mask`, targets, candidate masks and sibling candidates
# cannot enter the model.
const BOARD_HEIGHT = 24
const BOARD_WIDTH = 10
const CELL_COUNT = BOARD_HEIGHT * BOARD_WIDTH
const PIECE_TYPES = 7
const NEXT_HOLD_TOKENS = 6
const AUX_FEATURES = 37
const TOKEN_COUNT = CELL_COUNT + NEXT_HOLD_TOKENS + AUX_FEATURES
const OUTPUT_DIM = 22

const MODEL_DIM = SparseLookup.CARRIER_DIM
const ATTENTION_DIM = let raw = strip(get(ENV, "EVRL_ATTENTION_DIM", "32"))
    value = parse(Int, raw)
    32 <= value <= MODEL_DIM || throw(ArgumentError(
        "EVRL_ATTENTION_DIM must be in 32:MODEL_DIM",
    ))
    value
end
const ATTENTION_HEADS = let raw = strip(get(ENV, "EVRL_ATTENTION_HEADS", "4"))
    value = parse(Int, raw)
    1 <= value <= ATTENTION_DIM || throw(ArgumentError(
        "EVRL_ATTENTION_HEADS must be in 1:EVRL_ATTENTION_DIM",
    ))
    ATTENTION_DIM % value == 0 || throw(ArgumentError(
        "EVRL_ATTENTION_DIM must be divisible by EVRL_ATTENTION_HEADS",
    ))
    value
end
const SPATIAL_PROJECTION_BLOCK_WIDTH = cld(MODEL_DIM, ATTENTION_DIM)
const REGISTER_COUNT = let raw = strip(get(ENV, "EVRL_REGISTERS", "8"))
    value = parse(Int, raw)
    2 <= value <= 32 || throw(ArgumentError("EVRL_REGISTERS must be in 2:32"))
    value
end
const EPISODIC_ROUTER_TABLES = let raw = strip(get(ENV, "EVRL_ROUTER_TABLES", "2"))
    value = parse(Int, raw)
    1 <= value <= 4 || throw(ArgumentError("EVRL_ROUTER_TABLES must be in 1:4"))
    value
end
const EPISODIC_ROUTER_BITS = let raw = strip(get(ENV, "EVRL_ROUTER_BITS", "4"))
    value = parse(Int, raw)
    2 <= value <= 6 || throw(ArgumentError("EVRL_ROUTER_BITS must be in 2:6"))
    value
end
const EPISODIC_ROUTER_DIM = EPISODIC_ROUTER_TABLES * EPISODIC_ROUTER_BITS
const EPISODIC_ROUTER_BLOCK_WIDTH = cld(MODEL_DIM, EPISODIC_ROUTER_DIM)
const EPISODIC_BUCKETS = 1 << EPISODIC_ROUTER_BITS
const EPISODIC_BUCKET_CAP = let raw = strip(get(ENV, "EVRL_ROUTER_BUCKET_CAP", "16"))
    value = parse(Int, raw)
    1 <= value <= 64 || throw(ArgumentError("EVRL_ROUTER_BUCKET_CAP must be in 1:64"))
    value
end
const EPISODIC_SHORTLIST = let raw = strip(get(ENV, "EVRL_EPISODIC_SHORTLIST", "16"))
    value = parse(Int, raw)
    1 <= value <= 32 || throw(ArgumentError("EVRL_EPISODIC_SHORTLIST must be in 1:32"))
    value
end
const EPISODIC_CANDIDATE_CAP = let raw = strip(get(
    ENV, "EVRL_EPISODIC_CANDIDATE_CAP", "16",
))
    value = parse(Int, raw)
    EPISODIC_SHORTLIST <= value <= 64 || throw(ArgumentError(
        "EVRL_EPISODIC_CANDIDATE_CAP must be in EVRL_EPISODIC_SHORTLIST:64",
    ))
    value
end
const SPATIAL_ANCHORS = let raw = strip(get(ENV, "EVRL_SPATIAL_ANCHORS", "2"))
    value = parse(Int, raw)
    1 <= value <= EPISODIC_SHORTLIST || throw(ArgumentError(
        "EVRL_SPATIAL_ANCHORS must be in 1:EVRL_EPISODIC_SHORTLIST",
    ))
    value
end
const SPATIAL_SHORTLIST = let raw = strip(get(
    ENV, "EVRL_SPATIAL_SHORTLIST", "2",
))
    value = parse(Int, raw)
    1 <= value <= EPISODIC_SHORTLIST - 1 || throw(ArgumentError(
        "EVRL_SPATIAL_SHORTLIST must be smaller than EVRL_EPISODIC_SHORTLIST",
    ))
    value
end
const SPATIAL_CANDIDATE_CAP = let raw = strip(get(
    ENV, "EVRL_SPATIAL_CANDIDATE_CAP", string(min(3, EPISODIC_SHORTLIST - 1)),
))
    value = parse(Int, raw)
    SPATIAL_SHORTLIST <= value <= EPISODIC_SHORTLIST - 1 || throw(ArgumentError(
        "EVRL_SPATIAL_CANDIDATE_CAP must fit inside the routed shortlist",
    ))
    value
end
const EPISODIC_ROOT_CAP = min(TOKEN_COUNT, max(64, EPISODIC_CANDIDATE_CAP))
const ROUTER_STE_SCALE = 0.25f0
const FFN_MULTIPLIER = let raw = strip(get(ENV, "EVRL_FFN_MULTIPLIER", "2"))
    value = parse(Int, raw)
    1 <= value <= 4 || throw(ArgumentError("EVRL_FFN_MULTIPLIER must be in 1:4"))
    value
end
const FFN_DIM = let raw = strip(get(ENV, "EVRL_FFN_DIM", ""))
    value = isempty(raw) ? MODEL_DIM * FFN_MULTIPLIER : parse(Int, raw)
    32 <= value <= 4 * MODEL_DIM || throw(ArgumentError(
        "EVRL_FFN_DIM must be in 32:$(4 * MODEL_DIM)",
    ))
    value
end
const INITIAL_HALT_PROBABILITY = let raw = strip(get(
    ENV, "EVRL_INITIAL_HALT_PROBABILITY", "0.8",
))
    value = parse(Float32, raw)
    0.0f0 < value < 1.0f0 || throw(ArgumentError(
        "EVRL_INITIAL_HALT_PROBABILITY must be strictly between zero and one",
    ))
    value
end
const MIN_RECURRENT_STEPS = SparseLookup.MIN_RECURRENT_STEPS
const MAX_RECURRENT_STEPS = SparseLookup.MAX_RECURRENT_STEPS
const WARMUP_MAX_STEPS = SparseLookup.WARMUP_MAX_STEPS
const BLOCKS = SparseLookup.BLOCKS
const LOCAL_SPATIAL_NEIGHBORS = 8
const VISUAL_CHANNELS = 3
const VISUAL_STAGES = 5
const VISUAL_DILATIONS = (1, 2, 4, 8, 16)
const VISUAL_RECEPTIVE_FIELD = 1 + 2 * sum(VISUAL_DILATIONS)
const CANDIDATE_SUPPORT_CAP = let raw = strip(get(
    ENV, "EVRL_CANDIDATE_SUPPORT_CAP", "4",
))
    value = parse(Int, raw)
    1 <= value <= EPISODIC_SHORTLIST || throw(ArgumentError(
        "EVRL_CANDIDATE_SUPPORT_CAP must fit inside the episodic shortlist",
    ))
    value
end
const MIN_RESIDUAL_SCALE = 0.10f0
const RESIDUAL_SCALE_SPAN = 0.90f0
const INITIAL_RESIDUAL_SCALE = 0.25f0
const INITIAL_RESIDUAL_LOGIT = log(
    ((INITIAL_RESIDUAL_SCALE - MIN_RESIDUAL_SCALE) / RESIDUAL_SCALE_SPAN) /
    (1.0f0 - (INITIAL_RESIDUAL_SCALE - MIN_RESIDUAL_SCALE) / RESIDUAL_SCALE_SPAN),
)

@assert MODEL_DIM in (128, 256, 512)
@assert OUTPUT_DIM == SparseLookup.OUTPUT_DIM

@inline _sigmoid(x::Float32) = inv(1.0f0 + exp(-x))
@inline _swish(x::Float32) = x * _sigmoid(x)
@inline _swish_derivative(x::Float32) = begin
    probability = _sigmoid(x)
    probability * (1.0f0 + x * (1.0f0 - probability))
end
@inline _residual_scale(x::Float32) =
    MIN_RESIDUAL_SCALE + RESIDUAL_SCALE_SPAN * _sigmoid(x)
@inline _residual_scale_derivative(x::Float32) = begin
    probability = _sigmoid(x)
    RESIDUAL_SCALE_SPAN * probability * (1.0f0 - probability)
end

"""One independently evaluated candidate at the exact PreAct input boundary."""
struct EpisodicCandidateInput{
    B<:AbstractMatrix{Float32}, C<:AbstractMatrix{Float32},
    D<:AbstractMatrix{Float32}, N<:AbstractMatrix{Float32},
    A<:AbstractVector{Float32},
}
    board::B
    candidate::C
    difference::D
    next_hold::N
    aux::A
end

function EpisodicCandidateInput(board, candidate, difference, next_hold, aux)
    size(board) == (BOARD_HEIGHT, BOARD_WIDTH) ||
        throw(DimensionMismatch("board must be 24x10"))
    size(candidate) == (BOARD_HEIGHT, BOARD_WIDTH) ||
        throw(DimensionMismatch("candidate must be 24x10"))
    size(difference) == (BOARD_HEIGHT, BOARD_WIDTH) ||
        throw(DimensionMismatch("difference must be 24x10"))
    size(next_hold) == (PIECE_TYPES, NEXT_HOLD_TOKENS) ||
        throw(DimensionMismatch("next_hold must be 7x6"))
    length(aux) == AUX_FEATURES ||
        throw(DimensionMismatch("aux must have length 37"))
    all(isfinite, board) || throw(ArgumentError("board is non-finite"))
    all(isfinite, candidate) || throw(ArgumentError("candidate is non-finite"))
    all(isfinite, difference) || throw(ArgumentError("difference is non-finite"))
    all(isfinite, next_hold) || throw(ArgumentError("next_hold is non-finite"))
    all(isfinite, aux) || throw(ArgumentError("aux is non-finite"))

    # Canonical teacher batches are Float32.  Their candidate slices can be
    # held as read-only views until the immediately following backward pass,
    # eliminating five per-candidate copies.  Non-Float32 public inputs retain
    # the prior conversion semantics.
    board32 = eltype(board) === Float32 ? board : Matrix{Float32}(board)
    candidate32 = eltype(candidate) === Float32 ? candidate : Matrix{Float32}(candidate)
    difference32 = eltype(difference) === Float32 ? difference : Matrix{Float32}(difference)
    next_hold32 = eltype(next_hold) === Float32 ? next_hold : Matrix{Float32}(next_hold)
    aux32 = eltype(aux) === Float32 ? aux : Vector{Float32}(aux)
    return EpisodicCandidateInput{
        typeof(board32), typeof(candidate32), typeof(difference32),
        typeof(next_hold32), typeof(aux32),
    }(board32, candidate32, difference32, next_hold32, aux32)
end

"""Extract one candidate from the canonical fixed-shape PreAct batch."""
function candidate_input(input, candidate_index::Int)
    n = size(input.board, 4)
    1 <= candidate_index <= n || throw(BoundsError(input.board, candidate_index))
    size(input.board) == (BOARD_HEIGHT, BOARD_WIDTH, 1, n) ||
        throw(DimensionMismatch("board must be 24x10x1xN"))
    size(input.candidate) == size(input.board) ||
        throw(DimensionMismatch("candidate shape differs from board"))
    size(input.difference) == size(input.board) ||
        throw(DimensionMismatch("difference shape differs from board"))
    size(input.next_hold) == (PIECE_TYPES, NEXT_HOLD_TOKENS, n) ||
        throw(DimensionMismatch("next_hold must be 7x6xN"))
    size(input.aux) == (AUX_FEATURES, n) ||
        throw(DimensionMismatch("aux must be 37xN"))
    return EpisodicCandidateInput(
        @view(input.board[:, :, 1, candidate_index]),
        @view(input.candidate[:, :, 1, candidate_index]),
        @view(input.difference[:, :, 1, candidate_index]),
        @view(input.next_hold[:, :, candidate_index]),
        @view(input.aux[:, candidate_index]),
    )
end

mutable struct EpisodicViTLookupModel
    lookup::SparseLookup.DynamicLookupModel

    # Per-cell tokens retain the direct [board, candidate, difference]
    # projection and add a deliberately small visual stem: one learned 3x3
    # depthwise kernel per raw channel followed by a 3 -> MODEL_DIM pointwise
    # linear map.  The stem is residual, shared over the board, and does not
    # touch NEXT/HOLD or aux tokens.
    cell_projection::Matrix{Float32}
    cell_bias::Vector{Float32}
    cell_position::Matrix{Float32}
    visual_depthwise::Array{Float32,4}
    visual_channel_mix::Array{Float32,3}
    visual_pointwise::Matrix{Float32}
    visual_scale_logit::Vector{Float32}

    # Six independent HOLD/NEXT tokens and 37 independent scalar aux tokens.
    next_projection::Matrix{Float32}
    next_bias::Vector{Float32}
    next_position::Matrix{Float32}
    aux_value::Matrix{Float32}
    aux_position::Matrix{Float32}

    register_seed::Matrix{Float32}

    # Shared sparse spatial block.  Every recurrent step updates the 24x10
    # cell memory over the physically materialized 8-neighbour graph.  No
    # dense TOKEN_COUNT^2 score matrix or mask is ever formed.
    spatial_q::Matrix{Float32}
    spatial_k::Matrix{Float32}
    spatial_v::Matrix{Float32}
    spatial_o::Matrix{Float32}
    spatial_relative_bias::Array{Float32,3}
    spatial_scale_logit::Vector{Float32}

    cross_q::Matrix{Float32}
    cross_k::Matrix{Float32}
    cross_v::Matrix{Float32}
    cross_o::Matrix{Float32}
    cross_scale_logit::Vector{Float32}
    relation_scale_logit::Vector{Float32}

    # Reverse cross-attention write path.  The register-specific read support
    # is reused to write the updated register state back into the sample-local
    # episodic memory, making it a true recurrent working memory rather than a
    # read-only encoder cache.
    memory_write_v::Matrix{Float32}
    memory_write_o::Matrix{Float32}
    memory_write_scale_logit::Vector{Float32}

    self_q::Matrix{Float32}
    self_k::Matrix{Float32}
    self_v::Matrix{Float32}
    self_o::Matrix{Float32}
    self_scale_logit::Vector{Float32}

    ffn_gate::Matrix{Float32}
    ffn_up::Matrix{Float32}
    ffn_down::Matrix{Float32}
    ffn_scale_logit::Vector{Float32}

    # Every register retrieves its own long-memory trajectory from the shared
    # LookupFFN banks.  The banks remain parameter-shared, but neither the
    # address path nor the retrieved residual is pooled across registers.
    lookup_register_gate::Vector{Float32}
end

function _xavier(rng, rows, columns)
    result = randn(rng, Float32, rows, columns)
    result .*= sqrt(2.0f0 / Float32(rows + columns))
    return result
end

function initialize_model(rng::AbstractRNG=Xoshiro(0))
    lookup = SparseLookup.initialize_model(rng)
    lookup.halt_bias[1] = log(
        INITIAL_HALT_PROBABILITY / (1.0f0 - INITIAL_HALT_PROBABILITY),
    )
    cell_projection = _xavier(rng, MODEL_DIM, 3)
    visual_depthwise = zeros(Float32, 3, 3, VISUAL_CHANNELS, VISUAL_STAGES)
    visual_channel_mix = zeros(Float32, VISUAL_CHANNELS, VISUAL_CHANNELS, VISUAL_STAGES)
    visual_blur = Float32[1 2 1; 2 4 2; 1 2 1] ./ 16.0f0
    @inbounds for stage in 1:VISUAL_STAGES, channel in 1:VISUAL_CHANNELS
        visual_depthwise[:, :, channel, stage] .= visual_blur
        visual_channel_mix[channel, channel, stage] = 1.0f0
    end
    visual_channel_mix .+= 0.01f0 .* randn(
        rng, Float32, VISUAL_CHANNELS, VISUAL_CHANNELS, VISUAL_STAGES,
    )
    visual_pointwise = _xavier(rng, MODEL_DIM, VISUAL_CHANNELS)
    next_projection = _xavier(rng, MODEL_DIM, PIECE_TYPES)
    aux_value = randn(rng, Float32, MODEL_DIM, AUX_FEATURES) .* inv(sqrt(Float32(MODEL_DIM)))
    position_scale = 0.02f0
    register_seed = randn(rng, Float32, MODEL_DIM, REGISTER_COUNT) .* position_scale
    return EpisodicViTLookupModel(
        lookup,
        cell_projection,
        zeros(Float32, MODEL_DIM),
        randn(rng, Float32, MODEL_DIM, CELL_COUNT) .* position_scale,
        visual_depthwise,
        visual_channel_mix,
        visual_pointwise,
        Float32[INITIAL_RESIDUAL_LOGIT],
        next_projection,
        zeros(Float32, MODEL_DIM),
        randn(rng, Float32, MODEL_DIM, NEXT_HOLD_TOKENS) .* position_scale,
        aux_value,
        randn(rng, Float32, MODEL_DIM, AUX_FEATURES) .* position_scale,
        register_seed,
        _xavier(rng, SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM),
        _xavier(rng, SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM),
        _xavier(rng, SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM),
        _xavier(rng, SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM),
        zeros(Float32, 3, 3, ATTENTION_HEADS),
        Float32[INITIAL_RESIDUAL_LOGIT],
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, MODEL_DIM, ATTENTION_DIM),
        Float32[INITIAL_RESIDUAL_LOGIT],
        Float32[INITIAL_RESIDUAL_LOGIT],
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, MODEL_DIM, ATTENTION_DIM),
        Float32[INITIAL_RESIDUAL_LOGIT],
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, ATTENTION_DIM, MODEL_DIM),
        _xavier(rng, MODEL_DIM, ATTENTION_DIM),
        Float32[INITIAL_RESIDUAL_LOGIT],
        _xavier(rng, FFN_DIM, MODEL_DIM),
        _xavier(rng, FFN_DIM, MODEL_DIM),
        _xavier(rng, MODEL_DIM, FFN_DIM),
        Float32[INITIAL_RESIDUAL_LOGIT],
        zeros(Float32, REGISTER_COUNT),
    )
end

function _dense_parameters(model::EpisodicViTLookupModel)
    return (;
        cell_projection=model.cell_projection,
        cell_bias=model.cell_bias,
        cell_position=model.cell_position,
        visual_depthwise=model.visual_depthwise,
        visual_channel_mix=model.visual_channel_mix,
        visual_pointwise=model.visual_pointwise,
        visual_scale_logit=model.visual_scale_logit,
        next_projection=model.next_projection,
        next_bias=model.next_bias,
        next_position=model.next_position,
        aux_value=model.aux_value,
        aux_position=model.aux_position,
        register_seed=model.register_seed,
        spatial_q=model.spatial_q,
        spatial_k=model.spatial_k,
        spatial_v=model.spatial_v,
        spatial_o=model.spatial_o,
        spatial_relative_bias=model.spatial_relative_bias,
        spatial_scale_logit=model.spatial_scale_logit,
        cross_q=model.cross_q,
        cross_k=model.cross_k,
        cross_v=model.cross_v,
        cross_o=model.cross_o,
        cross_scale_logit=model.cross_scale_logit,
        relation_scale_logit=model.relation_scale_logit,
        memory_write_v=model.memory_write_v,
        memory_write_o=model.memory_write_o,
        memory_write_scale_logit=model.memory_write_scale_logit,
        self_q=model.self_q,
        self_k=model.self_k,
        self_v=model.self_v,
        self_o=model.self_o,
        self_scale_logit=model.self_scale_logit,
        ffn_gate=model.ffn_gate,
        ffn_up=model.ffn_up,
        ffn_down=model.ffn_down,
        ffn_scale_logit=model.ffn_scale_logit,
        lookup_register_gate=model.lookup_register_gate,
    )
end

parameter_count(model::EpisodicViTLookupModel) =
    SparseLookup.parameter_count(model.lookup) +
    sum(length, values(_dense_parameters(model)); init=0)

function topology(model::EpisodicViTLookupModel)
    return (;
        architecture="episodic-vit-recurrent-active-lookup",
        input_contract="preact-board-candidate-difference-next-hold-aux37",
        spatial_encoder="light-global-receptive-field-dilated-depthwise-pointwise-visual-stack-plus-shared-recurrent-learned-qkvo-8-neighbour-sparse-cell-attention-no-countsketch",
        visual_stem="raw3-five-stage-dilated-depthwise3x3-silu-pointwise3x3-residual-then-pointwise3-to-model-dim",
        visual_dilations=VISUAL_DILATIONS,
        visual_receptive_field=(VISUAL_RECEPTIVE_FIELD, VISUAL_RECEPTIVE_FIELD),
        visual_stem_parameters=length(model.visual_depthwise) +
            length(model.visual_channel_mix) +
            length(model.visual_pointwise) + length(model.visual_scale_logit),
        visual_stem_scalar_macs_per_candidate=
            CELL_COUNT * (
                VISUAL_STAGES * (
                    VISUAL_CHANNELS * 9 + VISUAL_CHANNELS * VISUAL_CHANNELS
                ) + VISUAL_CHANNELS * MODEL_DIM
            ),
        episodic_tokens=TOKEN_COUNT,
        cell_tokens=CELL_COUNT,
        next_hold_tokens=NEXT_HOLD_TOKENS,
        aux_tokens=AUX_FEATURES,
        registers=REGISTER_COUNT,
        model_dim=MODEL_DIM,
        attention_dim=ATTENTION_DIM,
        attention_heads=ATTENTION_HEADS,
        ffn_dim=FFN_DIM,
        recurrent_block="shared-full-token-cross-register-self-swiglu-register-local-active-lookup-reverse-cross-working-memory-write",
        episodic_router="none-full-token-cross-attention",
        episodic_attention="shared-learned-qkvo-full-283-token-softmax",
        episodic_projected_token_occurrences_per_step=TOKEN_COUNT,
        episodic_qk_pairs_per_step=REGISTER_COUNT * TOKEN_COUNT,
        episodic_exact_rerank_pairs_per_step=0,
        episodic_gather_rows_per_step=REGISTER_COUNT * TOKEN_COUNT,
        episodic_exact_dot_scalar_macs_per_step=
            REGISTER_COUNT * TOKEN_COUNT * ATTENTION_DIM,
        episodic_weighted_gather_scalar_macs_per_step=
            REGISTER_COUNT * TOKEN_COUNT * ATTENTION_DIM,
        working_memory="input-specific-recurrent-read-write-token-memory",
        working_memory_write="register-specific-reverse-cross-attention-using-normalized-read-support",
        episodic_router_scalar_macs_per_trajectory=0,
        episodic_retrieval_entry_reads_upper_bound_per_step=0,
        spatial_relation="physical-8-neighbour-relative-position-learned-qkvo",
        spatial_local_edges_upper_bound=CELL_COUNT * LOCAL_SPATIAL_NEIGHBORS,
        candidate_support_cap=0,
        spatial_dense_all_token_scores=0,
        long_memory_carriers_per_step=REGISTER_COUNT,
        long_memory_micro_calls_per_step=BLOCKS * REGISTER_COUNT,
        long_memory_register_injection="register-local-sigmoid-gated-residual",
        long_memory_compute_reduction_vs_per_register=1,
        long_memory_balance="state-local-hard-frequency-times-soft-probability-auxiliary-credit",
        register_relation="full-model-dim-direct-normalized-dot-softmax-gather",
        initial_halt_probability=INITIAL_HALT_PROBABILITY,
        recurrent_steps=(MIN_RECURRENT_STEPS, MAX_RECURRENT_STEPS),
        total_parameters=parameter_count(model),
        sparse_lookup=SparseLookup.topology(model.lookup),
        sparse_update="selected-bank-rows-only-lazy-adamw",
        halting="sampled-hard-hazard-policy-gradient-plus-sparse-one-step-probe-credit",
        candidate_evaluation="independent",
    )
end

@inline function _visual_input_channel(
    input::EpisodicCandidateInput,
    row::Int,
    column::Int,
    channel::Int,
)
    channel == 1 && return input.board[row, column]
    channel == 2 && return input.candidate[row, column]
    return input.difference[row, column]
end

function _visual_stack_forward_tokens!(tokens, model, input)
    @inbounds for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
        token = (column - 1) * BOARD_HEIGHT + row
        for channel in 1:VISUAL_CHANNELS
            tokens[channel, token] = _visual_input_channel(
                input, row, column, channel,
            )
        end
    end
    source_offset = 0
    target_offset = VISUAL_CHANNELS
    @inbounds for stage in 1:VISUAL_STAGES
        dilation = VISUAL_DILATIONS[stage]
        for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
            token = (column - 1) * BOARD_HEIGHT + row
            pre_1 = 0.0f0
            pre_2 = 0.0f0
            pre_3 = 0.0f0
            for delta_column in -1:1, delta_row in -1:1
                source_row = row + dilation * delta_row
                source_column = column + dilation * delta_column
                1 <= source_row <= BOARD_HEIGHT || continue
                1 <= source_column <= BOARD_WIDTH || continue
                source_token = (source_column - 1) * BOARD_HEIGHT + source_row
                pre_1 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 1, stage,
                    ],
                    tokens[source_offset + 1, source_token], pre_1,
                )
                pre_2 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 2, stage,
                    ],
                    tokens[source_offset + 2, source_token], pre_2,
                )
                pre_3 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 3, stage,
                    ],
                    tokens[source_offset + 3, source_token], pre_3,
                )
            end
            visual_1 = _swish(pre_1)
            visual_2 = _swish(pre_2)
            visual_3 = _swish(pre_3)
            for output_channel in 1:VISUAL_CHANNELS
                mixed = model.visual_channel_mix[output_channel, 1, stage] *
                    visual_1 +
                    model.visual_channel_mix[output_channel, 2, stage] *
                    visual_2 +
                    model.visual_channel_mix[output_channel, 3, stage] *
                    visual_3
                tokens[target_offset + output_channel, token] = mixed +
                    tokens[source_offset + output_channel, token]
            end
        end
        source_offset, target_offset = target_offset, source_offset
    end
    return source_offset
end

function _tokenize!(tokens::Matrix{Float32}, model, input::EpisodicCandidateInput)
    size(tokens) == (MODEL_DIM, TOKEN_COUNT) ||
        throw(DimensionMismatch("episodic token buffer shape changed"))
    token = 0
    visual_offset = _visual_stack_forward_tokens!(tokens, model, input)
    visual_alpha = _residual_scale(model.visual_scale_logit[1])
    @inbounds for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
        token += 1
        board = input.board[row, column]
        candidate = input.candidate[row, column]
        difference = input.difference[row, column]
        visual_1 = tokens[visual_offset + 1, token]
        visual_2 = tokens[visual_offset + 2, token]
        visual_3 = tokens[visual_offset + 3, token]
        @simd for coordinate in 1:MODEL_DIM
            visual = model.visual_pointwise[coordinate, 1] * visual_1 +
                model.visual_pointwise[coordinate, 2] * visual_2 +
                model.visual_pointwise[coordinate, 3] * visual_3
            tokens[coordinate, token] = visual_alpha * visual +
                model.cell_bias[coordinate] +
                model.cell_position[coordinate, token] +
                model.cell_projection[coordinate, 1] * board +
                model.cell_projection[coordinate, 2] * candidate +
                model.cell_projection[coordinate, 3] * difference
        end
    end
    @inbounds for queue_token in 1:NEXT_HOLD_TOKENS
        token += 1
        for coordinate in 1:MODEL_DIM
            value = model.next_bias[coordinate] +
                model.next_position[coordinate, queue_token]
            @simd for piece in 1:PIECE_TYPES
                value = muladd(
                    model.next_projection[coordinate, piece],
                    input.next_hold[piece, queue_token], value,
                )
            end
            tokens[coordinate, token] = value
        end
    end
    @inbounds for feature in 1:AUX_FEATURES
        token += 1
        value = input.aux[feature]
        @simd for coordinate in 1:MODEL_DIM
            tokens[coordinate, token] = model.aux_position[coordinate, feature] +
                model.aux_value[coordinate, feature] * value
        end
    end
    token == TOKEN_COUNT || error("episodic token count changed")
    return tokens
end

function _tokenize(model, input::EpisodicCandidateInput)
    return _tokenize!(Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT), model, input)
end

"""Bounded candidate-specific support derived only from the allowed tensors.

The support is not a sibling-candidate feature: every candidate is processed
independently and only its own `board`, `candidate`, and `difference` cells are
inspected.  These IDs guarantee that the placement evidence cannot disappear
before the learned episodic router has received a useful gradient.
"""
function _candidate_support(input::EpisodicCandidateInput)
    ids = fill(Int16(0), CANDIDATE_SUPPORT_CAP)
    scores = fill(-Inf32, CANDIDATE_SUPPORT_CAP)
    @inbounds for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
        difference = abs(input.difference[row, column])
        changed = abs(input.candidate[row, column] - input.board[row, column])
        salience = difference + changed
        salience <= 1.0f-7 && continue
        token = Int16((column - 1) * BOARD_HEIGHT + row)
        insertion = CANDIDATE_SUPPORT_CAP + 1
        for index in 1:CANDIDATE_SUPPORT_CAP
            if salience > scores[index] ||
               (salience == scores[index] && (iszero(ids[index]) || token < ids[index]))
                insertion = index
                break
            end
        end
        insertion > CANDIDATE_SUPPORT_CAP && continue
        for index in CANDIDATE_SUPPORT_CAP:-1:(insertion + 1)
            scores[index] = scores[index - 1]
            ids[index] = ids[index - 1]
        end
        scores[insertion] = salience
        ids[insertion] = token
    end
    count = findlast(x -> !iszero(x), ids)
    return ids, (count === nothing ? 0 : Int(count))
end

function _rmsnorm_columns!(input::Matrix{Float32}, inverse_rms::Vector{Float32})
    length(inverse_rms) == size(input, 2) ||
        throw(DimensionMismatch("episodic inverse-RMS buffer shape changed"))
    @inbounds for column in axes(input, 2)
        square_sum = 0.0f0
        @simd for coordinate in axes(input, 1)
            value = input[coordinate, column]
            square_sum = muladd(value, value, square_sum)
        end
        inverse = inv(sqrt(square_sum / Float32(size(input, 1)) + 1.0f-6))
        inverse_rms[column] = inverse
        @simd for coordinate in axes(input, 1)
            input[coordinate, column] *= inverse
        end
    end
    return input, inverse_rms
end

function _rmsnorm_columns(input::Matrix{Float32})
    output = copy(input)
    inverse_rms = Vector{Float32}(undef, size(input, 2))
    return _rmsnorm_columns!(output, inverse_rms)
end

function _rmsnorm_columns_vjp(normalized, inverse_rms, cotangent)
    result = similar(cotangent)
    return _rmsnorm_columns_vjp!(result, normalized, inverse_rms, cotangent)
end

function _rmsnorm_columns_vjp!(result, normalized, inverse_rms, cotangent)
    size(result) == size(cotangent) || throw(DimensionMismatch(
        "episodic RMSNorm VJP scratch shape changed",
    ))
    width = Float32(size(normalized, 1))
    @inbounds for column in axes(normalized, 2)
        projection = dot(
            @view(cotangent[:, column]), @view(normalized[:, column]),
        ) / width
        inverse = inverse_rms[column]
        @simd for coordinate in axes(normalized, 1)
            result[coordinate, column] = inverse * (
                cotangent[coordinate, column] -
                normalized[coordinate, column] * projection
            )
        end
    end
    return result
end

function _add_selected_vector!(dictionary, token::Int, value)
    target = get!(dictionary, token) do
        zeros(Float32, length(value))
    end
    @inbounds @simd for coordinate in eachindex(target)
        target[coordinate] += value[coordinate]
    end
    return nothing
end

function _rmsnorm_selected_vjp(normalized, inverse_rms, cotangents)
    result = Dict{Int,Vector{Float32}}()
    width = Float32(size(normalized, 1))
    for (token, cotangent) in cotangents
        normalized_token = @view normalized[:, token]
        projection = dot(cotangent, normalized_token) / width
        inverse = inverse_rms[token]
        gradient = Vector{Float32}(undef, size(normalized, 1))
        @inbounds @simd for coordinate in eachindex(gradient)
            gradient[coordinate] = inverse * (
                cotangent[coordinate] -
                normalized_token[coordinate] * projection
            )
        end
        result[token] = gradient
    end
    return result
end

function _structured_router_forward!(output, weights, input)
    size(output) == (EPISODIC_ROUTER_DIM, size(input, 2)) ||
        throw(DimensionMismatch("structured-router scratch shape changed"))
    @inbounds for route in 1:EPISODIC_ROUTER_DIM
        first_coordinate = (route - 1) * EPISODIC_ROUTER_BLOCK_WIDTH + 1
        last_coordinate = min(route * EPISODIC_ROUTER_BLOCK_WIDTH, MODEL_DIM)
        for column in axes(input, 2)
            value = 0.0f0
            @simd for coordinate in first_coordinate:last_coordinate
                local_coordinate = coordinate - first_coordinate + 1
                value = muladd(
                    weights[local_coordinate, route], input[coordinate, column], value,
                )
            end
            output[route, column] = value
        end
    end
    return output
end

function _structured_router_forward(weights, input)
    output = Matrix{Float32}(undef, EPISODIC_ROUTER_DIM, size(input, 2))
    return _structured_router_forward!(output, weights, input)
end

function _structured_router_selected_vjp!(
    weight_gradient, input_gradient, weights, input, column, route_gradient,
)
    @inbounds for route in 1:EPISODIC_ROUTER_DIM
        first_coordinate = (route - 1) * EPISODIC_ROUTER_BLOCK_WIDTH + 1
        last_coordinate = min(route * EPISODIC_ROUTER_BLOCK_WIDTH, MODEL_DIM)
        gradient = route_gradient[route]
        @simd for coordinate in first_coordinate:last_coordinate
            local_coordinate = coordinate - first_coordinate + 1
            weight_gradient[local_coordinate, route] = muladd(
                gradient, input[coordinate, column],
                weight_gradient[local_coordinate, route],
            )
            input_gradient[coordinate] = muladd(
                weights[local_coordinate, route], gradient,
                input_gradient[coordinate],
            )
        end
    end
    return nothing
end

"""Learned block-structured MODEL_DIM -> ATTENTION_DIM projection.

Every output coordinate owns one contiguous input block.  It remains learned
and shared across cells, but its fixed work is O(MODEL_DIM) per cell rather
than a dense O(MODEL_DIM*ATTENTION_DIM) projection.
"""
function _structured_attention_forward!(output, weights, input)
    size(output) == (ATTENTION_DIM, size(input, 2)) ||
        throw(DimensionMismatch("structured attention output shape changed"))
    size(weights) == (SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM) ||
        throw(DimensionMismatch("structured attention weight shape changed"))
    @inbounds for route in 1:ATTENTION_DIM
        first_coordinate = (route - 1) * SPATIAL_PROJECTION_BLOCK_WIDTH + 1
        last_coordinate = min(route * SPATIAL_PROJECTION_BLOCK_WIDTH, MODEL_DIM)
        for column in axes(input, 2)
            value = 0.0f0
            @simd for coordinate in first_coordinate:last_coordinate
                local_coordinate = coordinate - first_coordinate + 1
                value = muladd(
                    weights[local_coordinate, route],
                    input[coordinate, column],
                    value,
                )
            end
            output[route, column] = value
        end
    end
    return output
end

"""Learned block-structured ATTENTION_DIM -> MODEL_DIM expansion."""
function _structured_attention_expand!(output, weights, input)
    size(input, 1) == ATTENTION_DIM ||
        throw(DimensionMismatch("structured attention context width changed"))
    size(output) == (MODEL_DIM, size(input, 2)) ||
        throw(DimensionMismatch("structured attention expansion shape changed"))
    size(weights) == (SPATIAL_PROJECTION_BLOCK_WIDTH, ATTENTION_DIM) ||
        throw(DimensionMismatch("structured attention expansion weights changed"))
    @inbounds for route in 1:ATTENTION_DIM
        first_coordinate = (route - 1) * SPATIAL_PROJECTION_BLOCK_WIDTH + 1
        last_coordinate = min(route * SPATIAL_PROJECTION_BLOCK_WIDTH, MODEL_DIM)
        for column in axes(input, 2)
            route_value = input[route, column]
            @simd for coordinate in first_coordinate:last_coordinate
                local_coordinate = coordinate - first_coordinate + 1
                output[coordinate, column] =
                    weights[local_coordinate, route] * route_value
            end
        end
    end
    return output
end

function _structured_attention_vjp!(
    weight_gradient, input_gradient, weights, input, output_cotangent,
)
    fill!(input_gradient, 0.0f0)
    @inbounds for route in 1:ATTENTION_DIM
        first_coordinate = (route - 1) * SPATIAL_PROJECTION_BLOCK_WIDTH + 1
        last_coordinate = min(route * SPATIAL_PROJECTION_BLOCK_WIDTH, MODEL_DIM)
        for column in axes(input, 2)
            gradient = output_cotangent[route, column]
            @simd for coordinate in first_coordinate:last_coordinate
                local_coordinate = coordinate - first_coordinate + 1
                weight_gradient[local_coordinate, route] = muladd(
                    gradient, input[coordinate, column],
                    weight_gradient[local_coordinate, route],
                )
                input_gradient[coordinate, column] = muladd(
                    weights[local_coordinate, route], gradient,
                    input_gradient[coordinate, column],
                )
            end
        end
    end
    return input_gradient
end

function _structured_attention_expand_vjp!(
    weight_gradient, input_gradient, weights, input, output_cotangent,
)
    fill!(input_gradient, 0.0f0)
    @inbounds for route in 1:ATTENTION_DIM
        first_coordinate = (route - 1) * SPATIAL_PROJECTION_BLOCK_WIDTH + 1
        last_coordinate = min(route * SPATIAL_PROJECTION_BLOCK_WIDTH, MODEL_DIM)
        for column in axes(input, 2)
            route_value = input[route, column]
            route_gradient = 0.0f0
            @simd for coordinate in first_coordinate:last_coordinate
                local_coordinate = coordinate - first_coordinate + 1
                gradient = output_cotangent[coordinate, column]
                weight_gradient[local_coordinate, route] = muladd(
                    gradient, route_value,
                    weight_gradient[local_coordinate, route],
                )
                route_gradient = muladd(
                    weights[local_coordinate, route], gradient, route_gradient,
                )
            end
            input_gradient[route, column] = route_gradient
        end
    end
    return input_gradient
end

function _softmax_columns!(weights, scores, head)
    @inbounds for query in axes(scores, 2)
        maximum_score = maximum(@view scores[:, query])
        denominator = 0.0f0
        for key in axes(scores, 1)
            value = exp(scores[key, query] - maximum_score)
            weights[key, query, head] = value
            denominator += value
        end
        inverse = inv(denominator)
        @simd for key in axes(scores, 1)
            weights[key, query, head] *= inverse
        end
    end
    return nothing
end

struct AnchorRelationTape
    anchor_id::Int16
    related_ids::Vector{Int16}
    weights::Vector{Float32}
    context::Vector{Float32}
end

"""One physically sparse recurrent update of the 24x10 cell memory."""
struct SpatialAttentionTape
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    neighbor_ids::Matrix{Int16}
    neighbor_counts::Vector{Int8}
    weights::Array{Float32,3}
    context::Matrix{Float32}
    projected::Matrix{Float32}
    output::Matrix{Float32}
    output_normalized::Matrix{Float32}
    output_inverse_rms::Vector{Float32}
    alpha::Float32
end

mutable struct SpatialAttentionForwardScratch
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    weights::Array{Float32,3}
    context::Matrix{Float32}
    projected::Matrix{Float32}
    output::Matrix{Float32}
    output_normalized::Matrix{Float32}
    output_inverse_rms::Vector{Float32}
end

function SpatialAttentionForwardScratch()
    return SpatialAttentionForwardScratch(
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        Vector{Float32}(undef, TOKEN_COUNT),
        Matrix{Float32}(undef, ATTENTION_DIM, CELL_COUNT),
        Matrix{Float32}(undef, ATTENTION_DIM, CELL_COUNT),
        Matrix{Float32}(undef, ATTENTION_DIM, CELL_COUNT),
        Array{Float32,3}(undef,
            LOCAL_SPATIAL_NEIGHBORS, CELL_COUNT, ATTENTION_HEADS),
        Matrix{Float32}(undef, ATTENTION_DIM, CELL_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, CELL_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        Vector{Float32}(undef, TOKEN_COUNT),
    )
end

@inline function _cell_row_column(token::Int)
    return mod(token - 1, BOARD_HEIGHT) + 1,
           div(token - 1, BOARD_HEIGHT) + 1
end

function _local_neighbor_ids()
    ids = fill(Int16(0), LOCAL_SPATIAL_NEIGHBORS, CELL_COUNT)
    counts = zeros(Int8, CELL_COUNT)
    @inbounds for token in 1:CELL_COUNT
        row, column = _cell_row_column(token)
        count = 0
        for delta_column in -1:1, delta_row in -1:1
            iszero(delta_row) && iszero(delta_column) && continue
            neighbor_row = row + delta_row
            neighbor_column = column + delta_column
            1 <= neighbor_row <= BOARD_HEIGHT || continue
            1 <= neighbor_column <= BOARD_WIDTH || continue
            count += 1
            ids[count, token] = Int16(
                (neighbor_column - 1) * BOARD_HEIGHT + neighbor_row,
            )
        end
        counts[token] = Int8(count)
    end
    return ids, counts
end

const LOCAL_NEIGHBOR_IDS, LOCAL_NEIGHBOR_COUNTS = _local_neighbor_ids()

function _spatial_attention_forward!(scratch::SpatialAttentionForwardScratch,
                                     model, memory)
    size(memory) == (MODEL_DIM, TOKEN_COUNT) || throw(DimensionMismatch(
        "recurrent episodic memory shape changed",
    ))
    normalized = scratch.normalized
    copyto!(normalized, memory)
    inverse_rms = scratch.inverse_rms
    _rmsnorm_columns!(normalized, inverse_rms)
    cell_normalized = @view normalized[:, 1:CELL_COUNT]
    query = scratch.query
    key = scratch.key
    value = scratch.value
    _structured_attention_forward!(query, model.spatial_q, cell_normalized)
    _structured_attention_forward!(key, model.spatial_k, cell_normalized)
    _structured_attention_forward!(value, model.spatial_v, cell_normalized)
    weights = scratch.weights
    fill!(weights, 0.0f0)
    context = scratch.context
    fill!(context, 0.0f0)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    scale = inv(sqrt(Float32(head_dim)))
    @inbounds for token in 1:CELL_COUNT
        row, column = _cell_row_column(token)
        count = Int(LOCAL_NEIGHBOR_COUNTS[token])
        for head in 1:ATTENTION_HEADS
            first_coordinate = (head - 1) * head_dim + 1
            last_coordinate = head * head_dim
            maximum_score = -Inf32
            for edge in 1:count
                neighbor = Int(LOCAL_NEIGHBOR_IDS[edge, token])
                neighbor_row, neighbor_column = _cell_row_column(neighbor)
                score = 0.0f0
                @simd for coordinate in first_coordinate:last_coordinate
                    score = muladd(
                        query[coordinate, token], key[coordinate, neighbor], score,
                    )
                end
                score = score * scale + model.spatial_relative_bias[
                    neighbor_row - row + 2,
                    neighbor_column - column + 2,
                    head,
                ]
                weights[edge, token, head] = score
                maximum_score = max(maximum_score, score)
            end
            denominator = 0.0f0
            for edge in 1:count
                probability = exp(weights[edge, token, head] - maximum_score)
                weights[edge, token, head] = probability
                denominator += probability
            end
            inverse_denominator = inv(denominator)
            for edge in 1:count
                probability = weights[edge, token, head] * inverse_denominator
                weights[edge, token, head] = probability
                neighbor = Int(LOCAL_NEIGHBOR_IDS[edge, token])
                @simd for coordinate in first_coordinate:last_coordinate
                    context[coordinate, token] = muladd(
                        probability, value[coordinate, neighbor],
                        context[coordinate, token],
                    )
                end
            end
        end
    end
    projected = scratch.projected
    _structured_attention_expand!(projected, model.spatial_o, context)
    output = scratch.output
    copyto!(output, memory)
    alpha = _residual_scale(model.spatial_scale_logit[1])
    @inbounds for token in 1:CELL_COUNT
        @simd for coordinate in 1:MODEL_DIM
            output[coordinate, token] = muladd(
                alpha, projected[coordinate, token], output[coordinate, token],
            )
        end
    end
    output_normalized = scratch.output_normalized
    copyto!(output_normalized, output)
    output_inverse_rms = scratch.output_inverse_rms
    _rmsnorm_columns!(output_normalized, output_inverse_rms)
    return output, SpatialAttentionTape(
        normalized, inverse_rms, query, key, value,
        LOCAL_NEIGHBOR_IDS, LOCAL_NEIGHBOR_COUNTS, weights, context,
        projected, output, output_normalized, output_inverse_rms, alpha,
    )
end


function _spatial_attention_forward(model, memory)
    return _spatial_attention_forward!(
        SpatialAttentionForwardScratch(), model, memory,
    )
end

struct CrossAttentionTape
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    attention_weights::Array{Float32,3}
    attention_context::Matrix{Float32}
    output::Matrix{Float32}
    alpha::Float32
end

"""Candidate-owned storage for one full register-to-token cross step.

The 283 token K/V projections are materialized exactly once per recurrent
step and shared by every register.  No candidate retrieval, hard top-k, dense
mask, or token-routing parameter participates in this path.
"""
struct CrossAttentionForwardScratch
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    attention_weights::Array{Float32,3}
    attention_context::Matrix{Float32}
    output::Matrix{Float32}
    next::Matrix{Float32}
end


CrossAttentionForwardScratch() = CrossAttentionForwardScratch(
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Vector{Float32}(undef, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, TOKEN_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, TOKEN_COUNT),
    Array{Float32}(undef, TOKEN_COUNT, REGISTER_COUNT, ATTENTION_HEADS),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
)

struct WorkingMemoryWriteTape
    registers::Matrix{Float32}
    value::Matrix{Float32}
    weights::Array{Float32,3}
    inverse_weight_sums::Matrix{Float32}
    context::Matrix{Float32}
    projected::Matrix{Float32}
    output::Matrix{Float32}
    alpha::Float32
end

struct WorkingMemoryWriteForwardScratch
    registers::Matrix{Float32}
    value::Matrix{Float32}
    weights::Array{Float32,3}
    inverse_weight_sums::Matrix{Float32}
    context::Matrix{Float32}
    projected::Matrix{Float32}
    output::Matrix{Float32}
end

WorkingMemoryWriteForwardScratch() = WorkingMemoryWriteForwardScratch(
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Array{Float32}(undef, TOKEN_COUNT, REGISTER_COUNT, ATTENTION_HEADS),
    Matrix{Float32}(undef, TOKEN_COUNT, ATTENTION_HEADS),
    Matrix{Float32}(undef, ATTENTION_DIM, TOKEN_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
)

struct EpisodicBuckets
    entries::Matrix{Vector{Int16}}
    root::Vector{Int16}
end

function EpisodicBuckets()
    entries = Matrix{Vector{Int16}}(
        undef, EPISODIC_ROUTER_TABLES, EPISODIC_BUCKETS,
    )
    for index in eachindex(entries)
        entries[index] = Int16[]
        sizehint!(entries[index], EPISODIC_BUCKET_CAP)
    end
    return EpisodicBuckets(entries, Vector{Int16}(undef, EPISODIC_ROOT_CAP))
end

@inline function _route_code(route, table::Int, column::Int)
    offset = (table - 1) * EPISODIC_ROUTER_BITS
    code = 0
    @inbounds for bit in 1:EPISODIC_ROUTER_BITS
        route[offset + bit, column] >= 0.0f0 && (code |= 1 << (bit - 1))
    end
    return code + 1
end

function _build_token_buckets!(buckets::EpisodicBuckets, token_route)
    entries = buckets.entries
    @inbounds for index in eachindex(entries)
        empty!(entries[index])
    end
    root_seed = 0
    @inbounds for token in 1:TOKEN_COUNT, table in 1:EPISODIC_ROUTER_TABLES
        code = _route_code(token_route, table, token)
        bucket = entries[table, code]
        length(bucket) < EPISODIC_BUCKET_CAP && push!(bucket, Int16(token))
        # Bounded input-dependent fingerprint of the complete tiny router
        # output.  It controls fallback content but never scores memory tokens.
        root_seed = mod(
            root_seed * 33 + code + token * table,
            TOKEN_COUNT,
        )
    end
    # One fixed-cap direct fallback bucket.  The 131-stride permutation is
    # deterministic and coprime with the 283-token memory; retrieval never
    # scans TOKEN_COUNT.
    root = buckets.root
    @inbounds for index in 0:(EPISODIC_ROOT_CAP - 1)
        root[index + 1] = Int16(
            mod(root_seed + index * 131, TOKEN_COUNT) + 1,
        )
    end
    return buckets
end

function _build_token_buckets(token_route)
    return _build_token_buckets!(EpisodicBuckets(), token_route)
end

@inline function _append_bucket_rotated!(
    destination, count, seen, bucket, rotation, limit=length(destination),
)
    count == limit && return count
    length(bucket) == 0 && return count
    start = mod(rotation, length(bucket))
    @inbounds for offset in 0:(length(bucket) - 1)
        token16 = bucket[mod(start + offset, length(bucket)) + 1]
        token = Int(token16)
        seen[token] && continue
        count += 1
        destination[count] = token16
        seen[token] = true
        count == limit && break
    end
    return count
end

@inline function _append_bucket!(
    destination, count, seen, bucket, limit=length(destination),
)
    @inbounds for token16 in bucket
        token = Int(token16)
        seen[token] && continue
        count += 1
        destination[count] = token16
        seen[token] = true
        count == limit && break
    end
    return count
end

function _retrieve_candidates!(selected, seen_matrix, buckets::EpisodicBuckets, register_route)
    size(selected) == (EPISODIC_CANDIDATE_CAP, REGISTER_COUNT) ||
        throw(DimensionMismatch("episodic candidate scratch shape changed"))
    size(seen_matrix) == (TOKEN_COUNT, REGISTER_COUNT) ||
        throw(DimensionMismatch("episodic seen scratch shape changed"))
    table_quota = cld(EPISODIC_CANDIDATE_CAP, EPISODIC_ROUTER_TABLES)
    @inbounds for register in 1:REGISTER_COUNT
        seen = @view seen_matrix[:, register]
        fill!(seen, false)
        count = 0
        query_hash = 0
        for table in 1:EPISODIC_ROUTER_TABLES
            primary = _route_code(register_route, table, register)
            query_hash = query_hash * EPISODIC_BUCKETS + primary - 1
            table_limit = min(EPISODIC_CANDIDATE_CAP, count + table_quota)
            # Exact bucket followed by a fixed Hamming-1 probe schedule.  Work
            # is bounded by tables * (bits+1) * bucket_cap.
            for probe in 0:EPISODIC_ROUTER_BITS
                bucket_index = probe == 0 ? primary :
                    xor(primary - 1, 1 << (probe - 1)) + 1
                count = _append_bucket!(
                    @view(selected[:, register]), count, seen,
                    buckets.entries[table, bucket_index], table_limit,
                )
                count == table_limit && break
            end
            count == EPISODIC_CANDIDATE_CAP && break
        end
        count = _append_bucket_rotated!(
            @view(selected[:, register]), count, seen, buckets.root,
            query_hash, EPISODIC_CANDIDATE_CAP,
        )
        count == EPISODIC_CANDIDATE_CAP || error(
            "episodic candidate retrieval fill failed",
        )
    end
    return selected
end

function _retrieve_candidates(buckets::EpisodicBuckets, register_route)
    selected = Matrix{Int16}(undef, EPISODIC_CANDIDATE_CAP, REGISTER_COUNT)
    seen = Matrix{Bool}(undef, TOKEN_COUNT, REGISTER_COUNT)
    return _retrieve_candidates!(selected, seen, buckets, register_route)
end

function _anchor_relation_forward(
    anchor::Int,
    routed_ids,
    memory_normalized,
)
    related_ids = fill(typemax(Int16), SPATIAL_SHORTLIST)
    weights = fill(-Inf32, SPATIAL_SHORTLIST)
    scale = inv(sqrt(Float32(MODEL_DIM)))
    candidates_seen = 0
    @inbounds for candidate16 in routed_ids
        candidate = Int(candidate16)
        candidate == anchor && continue
        candidates_seen += 1
        candidates_seen > SPATIAL_CANDIDATE_CAP && break
        score = 0.0f0
        @simd for coordinate in 1:MODEL_DIM
            score = muladd(
                memory_normalized[coordinate, anchor],
                memory_normalized[coordinate, candidate],
                score,
            )
        end
        score *= scale
        insertion = SPATIAL_SHORTLIST + 1
        for shortlist_index in 1:SPATIAL_SHORTLIST
            old_score = weights[shortlist_index]
            old_token = related_ids[shortlist_index]
            if score > old_score || (score == old_score && candidate16 < old_token)
                insertion = shortlist_index
                break
            end
        end
        if insertion <= SPATIAL_SHORTLIST
            for shortlist_index in SPATIAL_SHORTLIST:-1:(insertion + 1)
                weights[shortlist_index] = weights[shortlist_index - 1]
                related_ids[shortlist_index] = related_ids[shortlist_index - 1]
            end
            weights[insertion] = score
            related_ids[insertion] = candidate16
        end
    end
    related_ids[end] != typemax(Int16) || error(
        "episodic spatial exact rerank failed to fill shortlist",
    )
    maximum_score = weights[1]
    denominator = 0.0f0
    @inbounds for shortlist_index in 1:SPATIAL_SHORTLIST
        value = exp(weights[shortlist_index] - maximum_score)
        weights[shortlist_index] = value
        denominator += value
    end
    inverse_denominator = inv(denominator)
    context = zeros(Float32, MODEL_DIM)
    @inbounds for shortlist_index in 1:SPATIAL_SHORTLIST
        weight = weights[shortlist_index] * inverse_denominator
        weights[shortlist_index] = weight
        token = Int(related_ids[shortlist_index])
        @simd for coordinate in 1:MODEL_DIM
            context[coordinate] = muladd(
                weight, memory_normalized[coordinate, token], context[coordinate],
            )
        end
    end
    return AnchorRelationTape(Int16(anchor), related_ids, weights, context)
end

struct SelfAttentionTape
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    weights::Array{Float32,3}
    context::Matrix{Float32}
    output::Matrix{Float32}
    alpha::Float32
end

struct SelfAttentionForwardScratch
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    query::Matrix{Float32}
    key::Matrix{Float32}
    value::Matrix{Float32}
    weights::Array{Float32,3}
    context::Matrix{Float32}
    output::Matrix{Float32}
    next::Matrix{Float32}
end

SelfAttentionForwardScratch() = SelfAttentionForwardScratch(
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Vector{Float32}(undef, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Array{Float32}(undef, REGISTER_COUNT, REGISTER_COUNT, ATTENTION_HEADS),
    Matrix{Float32}(undef, ATTENTION_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
)

function _attention_context(query, key, value)
    queries = size(query, 2)
    keys = size(key, 2)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    scale = inv(sqrt(Float32(head_dim)))
    weights = Array{Float32}(undef, keys, queries, ATTENTION_HEADS)
    context = zeros(Float32, ATTENTION_DIM, queries)
    @inbounds for head in 1:ATTENTION_HEADS
        coordinates = ((head - 1) * head_dim + 1):(head * head_dim)
        scores = Matrix{Float32}(
            transpose(@view(key[coordinates, :])) * @view(query[coordinates, :])
        )
        scores .*= scale
        _softmax_columns!(weights, scores, head)
        @views context[coordinates, :] .=
            value[coordinates, :] * weights[:, :, head]
    end
    return context, weights
end

function _attention_context!(weights, context, query, key, value)
    queries = size(query, 2)
    keys = size(key, 2)
    size(weights) == (keys, queries, ATTENTION_HEADS) ||
        throw(DimensionMismatch("attention weight scratch shape changed"))
    size(context) == (ATTENTION_DIM, queries) ||
        throw(DimensionMismatch("attention context scratch shape changed"))
    fill!(context, 0.0f0)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    scale = inv(sqrt(Float32(head_dim)))
    @inbounds for head in 1:ATTENTION_HEADS
        first_coordinate = (head - 1) * head_dim + 1
        last_coordinate = head * head_dim
        for query_index in 1:queries
            maximum_score = -Inf32
            for key_index in 1:keys
                score = 0.0f0
                @simd for coordinate in first_coordinate:last_coordinate
                    score = muladd(query[coordinate, query_index],
                        key[coordinate, key_index], score)
                end
                score *= scale
                weights[key_index, query_index, head] = score
                maximum_score = max(maximum_score, score)
            end
            denominator = 0.0f0
            for key_index in 1:keys
                probability = exp(weights[key_index, query_index, head] - maximum_score)
                weights[key_index, query_index, head] = probability
                denominator += probability
            end
            inverse_denominator = inv(denominator)
            for key_index in 1:keys
                probability = weights[key_index, query_index, head] * inverse_denominator
                weights[key_index, query_index, head] = probability
                @simd for coordinate in first_coordinate:last_coordinate
                    context[coordinate, query_index] = muladd(
                        probability, value[coordinate, key_index],
                        context[coordinate, query_index])
                end
            end
        end
    end
    return context
end

function _cross_attention_forward(model, registers, memory_normalized)
    return _cross_attention_forward!(
        CrossAttentionForwardScratch(), model, registers, memory_normalized,
    )
end


function _cross_attention_forward!(
    scratch::CrossAttentionForwardScratch,
    model,
    registers,
    memory_normalized,
)
    copyto!(scratch.input, registers)
    copyto!(scratch.normalized, registers)
    _rmsnorm_columns!(scratch.normalized, scratch.inverse_rms)

    # Project every episodic token once.  Every register then attends the same
    # complete, input-specific memory with exact multi-head softmax.
    mul!(scratch.query, model.cross_q, scratch.normalized)
    mul!(scratch.key, model.cross_k, memory_normalized)
    mul!(scratch.value, model.cross_v, memory_normalized)
    _attention_context!(
        scratch.attention_weights,
        scratch.attention_context,
        scratch.query,
        scratch.key,
        scratch.value,
    )
    mul!(scratch.output, model.cross_o, scratch.attention_context)
    alpha = _residual_scale(model.cross_scale_logit[1])
    @inbounds @simd for index in eachindex(scratch.next)
        scratch.next[index] = muladd(
            alpha, scratch.output[index], registers[index],
        )
    end
    return scratch.next, CrossAttentionTape(
        scratch.input,
        scratch.normalized,
        scratch.inverse_rms,
        scratch.query,
        scratch.key,
        scratch.value,
        scratch.attention_weights,
        scratch.attention_context,
        scratch.output,
        alpha,
    )
end

function _working_memory_write_forward!(
    scratch::WorkingMemoryWriteForwardScratch,
    model,
    memory,
    registers,
    read_weights,
)
    copyto!(scratch.registers, registers)
    mul!(scratch.value, model.memory_write_v, registers)
    fill!(scratch.context, 0.0f0)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    @inbounds for head in 1:ATTENTION_HEADS
        first_coordinate = (head - 1) * head_dim + 1
        last_coordinate = head * head_dim
        for token in 1:TOKEN_COUNT
            denominator = 1.0f-8
            for register in 1:REGISTER_COUNT
                denominator += read_weights[token, register, head]
            end
            inverse = inv(denominator)
            scratch.inverse_weight_sums[token, head] = inverse
            for register in 1:REGISTER_COUNT
                weight = read_weights[token, register, head] * inverse
                scratch.weights[token, register, head] = weight
                @simd for coordinate in first_coordinate:last_coordinate
                    scratch.context[coordinate, token] = muladd(
                        weight, scratch.value[coordinate, register],
                        scratch.context[coordinate, token],
                    )
                end
            end
        end
    end
    mul!(scratch.projected, model.memory_write_o, scratch.context)
    alpha = _residual_scale(model.memory_write_scale_logit[1])
    @inbounds @simd for index in eachindex(scratch.output)
        scratch.output[index] = muladd(
            alpha, scratch.projected[index], memory[index],
        )
    end
    return scratch.output, WorkingMemoryWriteTape(
        scratch.registers,
        scratch.value,
        scratch.weights,
        scratch.inverse_weight_sums,
        scratch.context,
        scratch.projected,
        scratch.output,
        alpha,
    )
end

function _working_memory_write_forward(model, memory, registers, read_weights)
    return _working_memory_write_forward!(
        WorkingMemoryWriteForwardScratch(), model, memory, registers,
        read_weights,
    )
end

function _self_attention_forward(model, registers)
    normalized, inverse_rms = _rmsnorm_columns(registers)
    query = model.self_q * normalized
    key = model.self_k * normalized
    value = model.self_v * normalized
    context, weights = _attention_context(query, key, value)
    output = model.self_o * context
    alpha = _residual_scale(model.self_scale_logit[1])
    next = similar(registers)
    @inbounds @simd for index in eachindex(next)
        next[index] = muladd(alpha, output[index], registers[index])
    end
    return next, SelfAttentionTape(
        copy(registers), normalized, inverse_rms, query, key, value,
        weights, context, output, alpha,
    )
end

function _self_attention_forward!(
    scratch::SelfAttentionForwardScratch,
    model,
    registers,
)
    copyto!(scratch.input, registers)
    copyto!(scratch.normalized, registers)
    _rmsnorm_columns!(scratch.normalized, scratch.inverse_rms)
    mul!(scratch.query, model.self_q, scratch.normalized)
    mul!(scratch.key, model.self_k, scratch.normalized)
    mul!(scratch.value, model.self_v, scratch.normalized)
    _attention_context!(scratch.weights, scratch.context,
        scratch.query, scratch.key, scratch.value)
    mul!(scratch.output, model.self_o, scratch.context)
    alpha = _residual_scale(model.self_scale_logit[1])
    next = scratch.next
    @inbounds @simd for index in eachindex(next)
        next[index] = muladd(alpha, scratch.output[index], registers[index])
    end
    return next, SelfAttentionTape(
        scratch.input,
        scratch.normalized,
        scratch.inverse_rms,
        scratch.query,
        scratch.key,
        scratch.value,
        scratch.weights,
        scratch.context,
        scratch.output,
        alpha,
    )
end

struct SwiGLUTape
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    gate::Matrix{Float32}
    up::Matrix{Float32}
    hidden::Matrix{Float32}
    output::Matrix{Float32}
    alpha::Float32
end

struct SwiGLUForwardScratch
    input::Matrix{Float32}
    normalized::Matrix{Float32}
    inverse_rms::Vector{Float32}
    gate::Matrix{Float32}
    up::Matrix{Float32}
    hidden::Matrix{Float32}
    output::Matrix{Float32}
    next::Matrix{Float32}
end

SwiGLUForwardScratch() = SwiGLUForwardScratch(
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Vector{Float32}(undef, REGISTER_COUNT),
    Matrix{Float32}(undef, FFN_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, FFN_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, FFN_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
)

function _swiglu_forward(model, registers)
    normalized, inverse_rms = _rmsnorm_columns(registers)
    gate = model.ffn_gate * normalized
    up = model.ffn_up * normalized
    hidden = similar(gate)
    @inbounds @simd for index in eachindex(hidden)
        hidden[index] = _swish(gate[index]) * up[index]
    end
    output = model.ffn_down * hidden
    alpha = _residual_scale(model.ffn_scale_logit[1])
    next = registers .+ alpha .* output
    return next, SwiGLUTape(
        copy(registers), normalized, inverse_rms, gate, up, hidden, output, alpha,
    )
end

function _swiglu_forward!(scratch::SwiGLUForwardScratch, model, registers)
    copyto!(scratch.input, registers)
    copyto!(scratch.normalized, registers)
    _rmsnorm_columns!(scratch.normalized, scratch.inverse_rms)
    mul!(scratch.gate, model.ffn_gate, scratch.normalized)
    mul!(scratch.up, model.ffn_up, scratch.normalized)
    @inbounds @simd for index in eachindex(scratch.hidden)
        scratch.hidden[index] = _swish(scratch.gate[index]) * scratch.up[index]
    end
    mul!(scratch.output, model.ffn_down, scratch.hidden)
    alpha = _residual_scale(model.ffn_scale_logit[1])
    @. scratch.next = registers + alpha * scratch.output
    return scratch.next, SwiGLUTape(
        scratch.input,
        scratch.normalized,
        scratch.inverse_rms,
        scratch.gate,
        scratch.up,
        scratch.hidden,
        scratch.output,
        alpha,
    )
end

function _pool_registers(registers)
    pooled = zeros(Float32, MODEL_DIM)
    return _pool_registers!(pooled, registers)
end

function _pool_registers!(pooled, registers)
    length(pooled) == MODEL_DIM || throw(DimensionMismatch(
        "episodic pooled-register scratch shape changed",
    ))
    fill!(pooled, 0.0f0)
    inverse_count = inv(Float32(REGISTER_COUNT))
    @inbounds for register in 1:REGISTER_COUNT
        @simd for coordinate in 1:MODEL_DIM
            pooled[coordinate] += registers[coordinate, register] * inverse_count
        end
    end
    return pooled
end

struct CarrierLookupTape
    blocks::Matrix{SparseLookup.LookupMicroTape}
    residual::Matrix{Float32}
    gates::Vector{Float32}
end


struct CarrierLookupForwardScratch
    blocks::Matrix{SparseLookup.LookupMicroTape}
    pooled::Vector{Float32}
    residual::Matrix{Float32}
    gates::Vector{Float32}
    next::Matrix{Float32}
end


function CarrierLookupForwardScratch(
    arenas::Vector{SparseLookup.LookupTrajectoryArena},
    step_index::Int,
)
    length(arenas) == REGISTER_COUNT || throw(DimensionMismatch(
        "one Lookup trajectory arena is required per register",
    ))
    return CarrierLookupForwardScratch(
        [SparseLookup.lookup_micro_tape(arenas[register], step_index, block)
         for block in 1:BLOCKS, register in 1:REGISTER_COUNT],
        Vector{Float32}(undef, MODEL_DIM),
        Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
        Vector{Float32}(undef, REGISTER_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
    )
end

function _lookup_carrier_forward(model, registers, temperature, materialize)
    block_tapes = Matrix{SparseLookup.LookupMicroTape}(
        undef, BLOCKS, REGISTER_COUNT,
    )
    residual = Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT)
    gates = Vector{Float32}(undef, REGISTER_COUNT)
    output = copy(registers)
    @inbounds for register in 1:REGISTER_COUNT
        state = copy(@view(registers[:, register]))
        for block in 1:BLOCKS
            state, block_tapes[block, register] = SparseLookup._lookup_micro_forward(
                model.lookup, state, block, temperature, materialize,
            )
        end
        gate = _sigmoid(model.lookup_register_gate[register])
        gates[register] = gate
        @simd for coordinate in 1:MODEL_DIM
            residual[coordinate, register] =
                state[coordinate] - registers[coordinate, register]
            output[coordinate, register] = muladd(
                gate, residual[coordinate, register], output[coordinate, register],
            )
        end
    end
    return output, CarrierLookupTape(block_tapes, residual, gates)
end

function _lookup_carrier_forward(
    model,
    registers,
    temperature,
    materialize,
    arenas::Vector{SparseLookup.LookupTrajectoryArena},
    step_index::Int,
)
    block_tapes = Matrix{SparseLookup.LookupMicroTape}(
        undef, BLOCKS, REGISTER_COUNT,
    )
    residual = Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT)
    gates = Vector{Float32}(undef, REGISTER_COUNT)
    output = copy(registers)
    @inbounds for register in 1:REGISTER_COUNT
        state = copy(@view(registers[:, register]))
        for block in 1:BLOCKS
            state, block_tapes[block, register] = SparseLookup.lookup_micro_forward!(
                arenas[register], model.lookup, state, step_index, block,
                temperature, materialize,
            )
        end
        gate = _sigmoid(model.lookup_register_gate[register])
        gates[register] = gate
        @simd for coordinate in 1:MODEL_DIM
            residual[coordinate, register] =
                state[coordinate] - registers[coordinate, register]
            output[coordinate, register] = muladd(
                gate, residual[coordinate, register], output[coordinate, register],
            )
        end
    end
    return output, CarrierLookupTape(block_tapes, residual, gates)
end


function _lookup_carrier_forward!(
    scratch::CarrierLookupForwardScratch,
    model,
    registers,
    temperature,
    materialize,
    arenas::Vector{SparseLookup.LookupTrajectoryArena},
    step_index::Int,
)
    residual = scratch.residual
    output = scratch.next
    copyto!(output, registers)
    @inbounds for register in 1:REGISTER_COUNT
        state = @view registers[:, register]
        for block in 1:BLOCKS
            state, tape = SparseLookup.lookup_micro_forward!(
                arenas[register], model.lookup, state, step_index, block,
                temperature, materialize,
            )
            tape === scratch.blocks[block, register] ||
                error("candidate register lookup tape changed")
        end
        gate = _sigmoid(model.lookup_register_gate[register])
        scratch.gates[register] = gate
        @simd for coordinate in 1:MODEL_DIM
            residual[coordinate, register] =
                state[coordinate] - registers[coordinate, register]
            output[coordinate, register] = muladd(
                gate, residual[coordinate, register], output[coordinate, register],
            )
        end
    end
    return output, CarrierLookupTape(scratch.blocks, residual, scratch.gates)
end

struct RecurrentStepTape
    spatial::SpatialAttentionTape
    cross::CrossAttentionTape
    self::SelfAttentionTape
    swiglu::SwiGLUTape
    lookup::CarrierLookupTape
    memory_write::WorkingMemoryWriteTape
    final_registers::Matrix{Float32}
    halt_probability::Float32
    stochastic_decision::Bool
    stopped::Bool
    forced_stop::Bool
end

"""Candidate-owned forward storage reused only after its backward completes."""
struct ForwardCandidateArena
    registers::Matrix{Float32}
    steps::Vector{RecurrentStepTape}
    lookup::Vector{SparseLookup.LookupTrajectoryArena}
    spatial_scratch::Vector{Union{Nothing,SpatialAttentionForwardScratch}}
    cross_scratch::Vector{Union{Nothing,CrossAttentionForwardScratch}}
    self_scratch::Vector{SelfAttentionForwardScratch}
    swiglu_scratch::Vector{SwiGLUForwardScratch}
    carrier_scratch::Vector{CarrierLookupForwardScratch}
    memory_write_scratch::Vector{WorkingMemoryWriteForwardScratch}
end

function ForwardCandidateArena()
    steps = RecurrentStepTape[]
    sizehint!(steps, MAX_RECURRENT_STEPS)
    lookup = [SparseLookup.LookupTrajectoryArena(MAX_RECURRENT_STEPS)
              for _ in 1:REGISTER_COUNT]
    return ForwardCandidateArena(
        Matrix{Float32}(undef, MODEL_DIM, REGISTER_COUNT),
        steps,
        lookup,
        Union{Nothing,SpatialAttentionForwardScratch}[
            step <= MIN_RECURRENT_STEPS ? SpatialAttentionForwardScratch() : nothing
            for step in 1:MAX_RECURRENT_STEPS
        ],
        Union{Nothing,CrossAttentionForwardScratch}[
            step <= MIN_RECURRENT_STEPS ? CrossAttentionForwardScratch() : nothing
            for step in 1:MAX_RECURRENT_STEPS
        ],
        [SelfAttentionForwardScratch() for _ in 1:MAX_RECURRENT_STEPS],
        [SwiGLUForwardScratch() for _ in 1:MAX_RECURRENT_STEPS],
        [CarrierLookupForwardScratch(lookup, step)
         for step in 1:MAX_RECURRENT_STEPS],
        [WorkingMemoryWriteForwardScratch() for _ in 1:MAX_RECURRENT_STEPS],
    )
end

struct TrajectoryTape
    input::EpisodicCandidateInput
    initial_memory::Matrix{Float32}
    steps::Vector{RecurrentStepTape}
    output::Vector{Float32}
    warmup_depth::Bool
end

mutable struct RouteUsage
    sparse::SparseLookup.RouteUsage
    trajectories::UInt64
    recurrent_steps::UInt64
end
RouteUsage() = RouteUsage(SparseLookup.RouteUsage(), 0, 0)

mutable struct LookupBalanceStats
    hard_frequencies::Array{Float32,4}
    observations::Vector{Int32}
end

LookupBalanceStats() = LookupBalanceStats(
    zeros(
        Float32,
        SparseLookup.WTA_CHOICES,
        SparseLookup.WTA_DIGITS,
        SparseLookup.TABLES_PER_BLOCK,
        BLOCKS,
    ),
    zeros(Int32, BLOCKS),
)

function lookup_balance_stats!(
    stats::LookupBalanceStats,
    tapes,
    candidate_count::Int,
)
    fill!(stats.hard_frequencies, 0.0f0)
    fill!(stats.observations, 0)
    @inbounds for candidate in 1:candidate_count
        trajectory = tapes[candidate]
        trajectory === nothing && error("lookup balance saw a missing trajectory")
        for step in trajectory.steps, block in 1:BLOCKS, register in 1:REGISTER_COUNT
            micro = step.lookup.blocks[block, register]
            stats.observations[block] += 1
            for table in 1:SparseLookup.TABLES_PER_BLOCK
                primary_slot = SparseLookup._lookup_slot(1, table)
                for digit in 1:SparseLookup.WTA_DIGITS
                    choice = Int(micro.winner_choices[digit, primary_slot])
                    stats.hard_frequencies[choice, digit, table, block] += 1.0f0
                end
            end
        end
    end
    @inbounds for block in 1:BLOCKS
        count = Int(stats.observations[block])
        count > 0 || continue
        inverse = inv(Float32(count))
        for table in 1:SparseLookup.TABLES_PER_BLOCK,
                digit in 1:SparseLookup.WTA_DIGITS,
                choice in 1:SparseLookup.WTA_CHOICES
            stats.hard_frequencies[choice, digit, table, block] *= inverse
        end
    end
    return stats
end

function _record_usage!(usage::RouteUsage, steps)
    usage.trajectories += 1
    usage.recurrent_steps += UInt64(length(steps))
    sparse = usage.sparse
    sparse.trajectories += 1
    sparse.recurrent_steps += UInt64(length(steps))
    @inbounds for step in steps, block in 1:BLOCKS
        sparse.block_visits[block] += REGISTER_COUNT
        for register in 1:REGISTER_COUNT
            tape = step.lookup.blocks[block, register]
            for table in 1:SparseLookup.TABLES_PER_BLOCK
                for lookup in 1:SparseLookup.ROWS_PER_TABLE_LOOKUP
                    slot = SparseLookup._lookup_slot(lookup, table)
                    sparse.counts[Int(tape.addresses[slot]), table, block] += 1
                end
            end
        end
    end
    return nothing
end

"""Record one completed trajectory after any parallel forward region."""
function record_usage!(usage::RouteUsage, tape::TrajectoryTape)
    _record_usage!(usage, tape.steps)
    return usage
end

function usage_summary(usage::RouteUsage)
    sparse = SparseLookup.usage_summary(usage.sparse)
    return (;
        trajectories=Int(usage.trajectories),
        recurrent_steps=Int(usage.recurrent_steps),
        mean_steps=usage.trajectories == 0 ? 0.0 :
            Float64(usage.recurrent_steps) / Float64(usage.trajectories),
        sparse,
    )
end

mutable struct ForwardTrajectoryState
    input::EpisodicCandidateInput
    memory::Matrix{Float32}
    registers::Matrix{Float32}
    steps::Vector{RecurrentStepTape}
    arena::Union{Nothing,ForwardCandidateArena}
    rng::Union{Nothing,Xoshiro}
    forced_depth::Int16
    temperature::Float32
    training::Bool
    stopped::Bool
end

"""Prepare one trajectory without executing a recurrent step.

This split is used by the CPU scheduler to mix candidates from every state in
one queue and compact the active set after every hard-halting decision.  The
public `forward_trajectory` wrapper below retains the original semantics.
"""
function prepare_trajectory(
    model::EpisodicViTLookupModel,
    input::EpisodicCandidateInput;
    rng=nothing,
    training::Bool=false,
    forced_depth=nothing,
    temperature=0.50f0,
    memory_buffer=nothing,
    inverse_rms_buffer=nothing,
    arena::Union{Nothing,ForwardCandidateArena}=nothing,
)
    training == (rng !== nothing) ||
        throw(ArgumentError("training mode requires exactly one owned RNG"))
    forced_depth === nothing || MIN_RECURRENT_STEPS <= forced_depth <= MAX_RECURRENT_STEPS ||
        throw(ArgumentError("forced depth is outside recurrent bounds"))

    (memory_buffer === nothing) == (inverse_rms_buffer === nothing) ||
        throw(ArgumentError("episodic memory buffers must be supplied together"))
    memory = memory_buffer === nothing ? _tokenize(model, input) :
        _tokenize!(memory_buffer, model, input)
    if arena === nothing
        registers = copy(model.register_seed)
        steps = RecurrentStepTape[]
        sizehint!(steps, MAX_RECURRENT_STEPS)
    else
        registers = arena.registers
        copyto!(registers, model.register_seed)
        steps = arena.steps
        empty!(steps)
    end
    return ForwardTrajectoryState(
        input,
        memory,
        registers,
        steps,
        arena,
        rng,
        forced_depth === nothing ? Int16(0) : Int16(forced_depth),
        Float32(temperature),
        training,
        false,
    )
end

"""Execute exactly one recurrent step and return whether it hard-stopped."""
function advance_trajectory!(
    model::EpisodicViTLookupModel,
    state::ForwardTrajectoryState;
    materialize=nothing,
)
    state.stopped && error("cannot advance a stopped trajectory")
    step_index = length(state.steps) + 1
    step_index <= MAX_RECURRENT_STEPS || error("trajectory exceeded recurrent bound")
    memory, spatial = if state.arena === nothing
        _spatial_attention_forward(model, state.memory)
    else
        spatial_scratch = state.arena.spatial_scratch[step_index]
        if spatial_scratch === nothing
            spatial_scratch = SpatialAttentionForwardScratch()
            state.arena.spatial_scratch[step_index] = spatial_scratch
        end
        _spatial_attention_forward!(spatial_scratch, model, state.memory)
    end
    cross_scratch = if state.arena === nothing
        nothing
    else
        value = state.arena.cross_scratch[step_index]
        if value === nothing
            value = CrossAttentionForwardScratch()
            state.arena.cross_scratch[step_index] = value
        end
        value
    end
    registers, cross = if state.arena === nothing
        _cross_attention_forward(
            model,
            state.registers,
            spatial.output_normalized,
        )
    else
        _cross_attention_forward!(
            cross_scratch,
            model,
            state.registers,
            spatial.output_normalized,
        )
    end
    if state.arena === nothing
        registers, self = _self_attention_forward(model, registers)
        registers, swiglu = _swiglu_forward(model, registers)
        registers, lookup = _lookup_carrier_forward(
            model, registers, state.temperature, materialize,
        )
        pooled = _pool_registers(registers)
        final_registers = copy(registers)
    else
        registers, self = _self_attention_forward!(
            state.arena.self_scratch[step_index], model, registers,
        )
        registers, swiglu = _swiglu_forward!(
            state.arena.swiglu_scratch[step_index], model, registers,
        )
        carrier_scratch = state.arena.carrier_scratch[step_index]
        registers, lookup = _lookup_carrier_forward!(
            carrier_scratch,
            model,
            registers,
            state.temperature,
            materialize,
            state.arena.lookup,
            step_index,
        )
        pooled = _pool_registers!(carrier_scratch.pooled, registers)
        final_registers = registers
    end
    memory, memory_write = if state.arena === nothing
        _working_memory_write_forward(
            model, memory, registers, cross.attention_weights,
        )
    else
        _working_memory_write_forward!(
            state.arena.memory_write_scratch[step_index],
            model, memory, registers, cross.attention_weights,
        )
    end
    halt_probability = _sigmoid(
        model.lookup.halt_bias[1] + dot(model.lookup.halt_weight, pooled),
    )
    forced_depth = iszero(state.forced_depth) ? nothing : Int(state.forced_depth)
    forced_stop = step_index == MAX_RECURRENT_STEPS ||
        (forced_depth !== nothing && step_index == forced_depth)
    stochastic = state.training && forced_depth === nothing &&
        step_index >= MIN_RECURRENT_STEPS && step_index < MAX_RECURRENT_STEPS
    stopped = forced_stop ? true :
        (step_index < MIN_RECURRENT_STEPS || forced_depth !== nothing ? false :
         (state.training ? begin
             state.rng === nothing && error("training trajectory lost its RNG")
             rand(state.rng) < halt_probability
         end : halt_probability >= 0.50f0))
    push!(state.steps, RecurrentStepTape(
        spatial, cross, self, swiglu, lookup, memory_write, final_registers,
        halt_probability,
        stochastic, stopped, forced_stop,
    ))
    state.memory = memory
    state.registers = registers
    state.stopped = stopped
    return stopped
end

"""Finalize a stopped trajectory into the original output/tape contract."""
function finalize_trajectory(
    model::EpisodicViTLookupModel,
    state::ForwardTrajectoryState;
    usage=nothing,
)
    state.stopped || error("trajectory did not stop")
    pooled = if state.arena === nothing
        _pool_registers(state.registers)
    else
        scratch = state.arena.carrier_scratch[length(state.steps)].pooled
        _pool_registers!(scratch, state.registers)
    end
    output = copy(model.lookup.bias)
    mul!(output, model.lookup.head, pooled, 1.0f0, 1.0f0)
    tape = TrajectoryTape(
        state.input,
        first(state.steps).spatial.normalized,
        state.steps,
        output,
        !iszero(state.forced_depth),
    )
    usage === nothing || _record_usage!(usage, state.steps)
    return (; output, tape, depth=length(state.steps))
end

"""Evaluate exactly one counterfactual recurrent step after a sampled stop.

The supplied arena and output vector are caller-owned scratch.  The original
trajectory, its RNG history, and route-usage accounting are left untouched.
This is the physical one-step primitive used by sparse halting probes.
"""
function probe_one_step!(
    output::AbstractVector{Float32},
    model::EpisodicViTLookupModel,
    tape::TrajectoryTape,
    arena::ForwardCandidateArena;
    temperature=0.50f0,
)
    length(output) == OUTPUT_DIM || throw(DimensionMismatch(
        "halting probe output shape changed",
    ))
    depth = length(tape.steps)
    depth < MAX_RECURRENT_STEPS || throw(ArgumentError(
        "cannot probe beyond the maximum recurrent depth",
    ))
    stopped_step = last(tape.steps)
    stopped_step.stopped || throw(ArgumentError(
        "halting probe requires a stopped trajectory",
    ))

    copyto!(arena.registers, stopped_step.final_registers)
    empty!(arena.steps)
    append!(arena.steps, tape.steps)
    state = ForwardTrajectoryState(
        tape.input,
        stopped_step.memory_write.output,
        arena.registers,
        arena.steps,
        arena,
        nothing,
        Int16(depth + 1),
        Float32(temperature),
        false,
        false,
    )
    advance_trajectory!(model, state)
    state.stopped || error("forced one-step halting probe did not stop")
    pooled = arena.carrier_scratch[depth + 1].pooled
    _pool_registers!(pooled, state.registers)
    copyto!(output, model.lookup.bias)
    mul!(output, model.lookup.head, pooled, 1.0f0, 1.0f0)
    return output
end


function forward_trajectory(
    model::EpisodicViTLookupModel,
    input::EpisodicCandidateInput;
    rng=nothing,
    training::Bool=false,
    forced_depth=nothing,
    temperature=0.50f0,
    usage=nothing,
    materialize=nothing,
    memory_buffer=nothing,
    inverse_rms_buffer=nothing,
)
    state = prepare_trajectory(
        model,
        input;
        rng,
        training,
        forced_depth,
        temperature,
        memory_buffer,
        inverse_rms_buffer,
    )
    while !state.stopped
        advance_trajectory!(model, state; materialize)
    end
    return finalize_trajectory(model, state; usage)
end

"""Persistent worker-local storage for one complete candidate backward job."""
mutable struct BackwardScratch
    dy::Vector{Float32}
    pooled::Vector{Float32}
    state_a::Matrix{Float32}
    state_b::Matrix{Float32}
    halt_gradients::Vector{Float32}
    context_cotangent::Matrix{Float32}
    attention_dcontext::Matrix{Float32}
    attention_dquery::Matrix{Float32}
    attention_dkey::Matrix{Float32}
    attention_dvalue::Matrix{Float32}
    attention_dscore::Matrix{Float32}
    attention_dweights::Matrix{Float32}
    attention_external_dweights::Array{Float32,3}
    memory_write_dcontext::Matrix{Float32}
    memory_write_dvalue::Matrix{Float32}
    memory_write_dregisters::Matrix{Float32}
    dnormalized::Matrix{Float32}
    dregister_route::Matrix{Float32}
    dweights::Vector{Float32}
    register_input_gradient::Vector{Float32}
    token_input_gradient::Vector{Float32}
    ffn_output_cotangent::Matrix{Float32}
    ffn_hidden_cotangent::Matrix{Float32}
    ffn_gate_cotangent::Matrix{Float32}
    ffn_up_cotangent::Matrix{Float32}
    ffn_normalized_a::Matrix{Float32}
    ffn_normalized_b::Matrix{Float32}
    ffn_down_product::Matrix{Float32}
    ffn_parameter_product::Matrix{Float32}
    residual_cotangent::Vector{Float32}
    memory_gradient::Matrix{Float32}
    memory_postnorm::Matrix{Float32}
    memory_previous::Matrix{Float32}
    visual_features::Array{Float32,3}
    visual_cotangents::Array{Float32,3}
    memory_stamp::Vector{UInt32}
    memory_ids::Vector{Int16}
    memory_count::Int
    memory_generation::UInt32
    route_gradient::Matrix{Float32}
    route_stamp::Vector{UInt32}
    route_ids::Vector{Int16}
    route_count::Int
    route_generation::UInt32
    spatial_dprojected::Matrix{Float32}
    spatial_dcontext::Matrix{Float32}
    spatial_dquery::Matrix{Float32}
    spatial_dkey::Matrix{Float32}
    spatial_dvalue::Matrix{Float32}
    spatial_dnormalized::Matrix{Float32}
    spatial_projection_input::Matrix{Float32}
end

function BackwardScratch()
    return BackwardScratch(
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, MODEL_DIM),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, MAX_RECURRENT_STEPS),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, ATTENTION_DIM, REGISTER_COUNT),
        zeros(Float32, ATTENTION_DIM, REGISTER_COUNT),
        zeros(Float32, ATTENTION_DIM, TOKEN_COUNT),
        zeros(Float32, ATTENTION_DIM, TOKEN_COUNT),
        zeros(Float32, TOKEN_COUNT, REGISTER_COUNT),
        zeros(Float32, TOKEN_COUNT, REGISTER_COUNT),
        zeros(Float32, TOKEN_COUNT, REGISTER_COUNT, ATTENTION_HEADS),
        zeros(Float32, ATTENTION_DIM, TOKEN_COUNT),
        zeros(Float32, ATTENTION_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, EPISODIC_ROUTER_DIM, REGISTER_COUNT),
        # Shared softmax VJP scratch must fit every attention support.  The
        # physical 8-neighbour spatial path is wider than the current
        # episodic/register shortlists; undersizing this vector silently wrote
        # past the Julia array under @inbounds and eventually corrupted GC.
        zeros(Float32, max(
            EPISODIC_SHORTLIST,
            REGISTER_COUNT,
            SPATIAL_SHORTLIST,
            LOCAL_SPATIAL_NEIGHBORS,
        )),
        zeros(Float32, MODEL_DIM),
        zeros(Float32, MODEL_DIM),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, FFN_DIM, REGISTER_COUNT),
        zeros(Float32, FFN_DIM, REGISTER_COUNT),
        zeros(Float32, FFN_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        zeros(Float32, MODEL_DIM, FFN_DIM),
        zeros(Float32, FFN_DIM, MODEL_DIM),
        zeros(Float32, MODEL_DIM),
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        Matrix{Float32}(undef, MODEL_DIM, TOKEN_COUNT),
        zeros(Float32, VISUAL_CHANNELS, CELL_COUNT, VISUAL_STAGES + 1),
        zeros(Float32, VISUAL_CHANNELS, CELL_COUNT, VISUAL_STAGES + 1),
        zeros(UInt32, TOKEN_COUNT),
        zeros(Int16, TOKEN_COUNT),
        0,
        UInt32(0),
        Matrix{Float32}(undef, EPISODIC_ROUTER_DIM, TOKEN_COUNT),
        zeros(UInt32, TOKEN_COUNT),
        zeros(Int16, TOKEN_COUNT),
        0,
        UInt32(0),
        zeros(Float32, MODEL_DIM, CELL_COUNT),
        zeros(Float32, ATTENTION_DIM, CELL_COUNT),
        zeros(Float32, ATTENTION_DIM, CELL_COUNT),
        zeros(Float32, ATTENTION_DIM, CELL_COUNT),
        zeros(Float32, ATTENTION_DIM, CELL_COUNT),
        zeros(Float32, MODEL_DIM, CELL_COUNT),
        zeros(Float32, MODEL_DIM, CELL_COUNT),
    )
end

@inline function _next_generation!(stamps::Vector{UInt32}, generation::UInt32)
    if generation == typemax(UInt32)
        fill!(stamps, UInt32(0))
        return UInt32(1)
    end
    return generation + UInt32(1)
end

@inline function _begin_memory_gradients!(scratch::BackwardScratch)
    scratch.memory_generation = _next_generation!(
        scratch.memory_stamp, scratch.memory_generation,
    )
    scratch.memory_count = 0
    return nothing
end

@inline function _touch_memory_gradient!(scratch::BackwardScratch, token::Int)
    if scratch.memory_stamp[token] != scratch.memory_generation
        scratch.memory_stamp[token] = scratch.memory_generation
        scratch.memory_count += 1
        scratch.memory_ids[scratch.memory_count] = Int16(token)
        @inbounds @simd for coordinate in 1:MODEL_DIM
            scratch.memory_gradient[coordinate, token] = 0.0f0
        end
    end
    return nothing
end

@inline function _begin_route_gradients!(scratch::BackwardScratch)
    scratch.route_generation = _next_generation!(
        scratch.route_stamp, scratch.route_generation,
    )
    scratch.route_count = 0
    return nothing
end

@inline function _touch_route_gradient!(scratch::BackwardScratch, token::Int)
    if scratch.route_stamp[token] != scratch.route_generation
        scratch.route_stamp[token] = scratch.route_generation
        scratch.route_count += 1
        scratch.route_ids[scratch.route_count] = Int16(token)
        @inbounds @simd for coordinate in 1:EPISODIC_ROUTER_DIM
            scratch.route_gradient[coordinate, token] = 0.0f0
        end
    end
    return nothing
end

mutable struct GradientAccumulator
    lookup::SparseLookup.GradientAccumulator
    dense::Dict{Symbol,Array{Float32}}
    active_tokens::BitVector
    backward_scratch::BackwardScratch
end

function _zero_dense_gradients()
    return Dict{Symbol,Array{Float32}}(
        :cell_projection => zeros(Float32, MODEL_DIM, 3),
        :cell_bias => zeros(Float32, MODEL_DIM),
        :cell_position => zeros(Float32, MODEL_DIM, CELL_COUNT),
        :visual_depthwise => zeros(
            Float32, 3, 3, VISUAL_CHANNELS, VISUAL_STAGES,
        ),
        :visual_channel_mix => zeros(
            Float32, VISUAL_CHANNELS, VISUAL_CHANNELS, VISUAL_STAGES,
        ),
        :visual_pointwise => zeros(Float32, MODEL_DIM, 3),
        :visual_scale_logit => zeros(Float32, 1),
        :next_projection => zeros(Float32, MODEL_DIM, PIECE_TYPES),
        :next_bias => zeros(Float32, MODEL_DIM),
        :next_position => zeros(Float32, MODEL_DIM, NEXT_HOLD_TOKENS),
        :aux_value => zeros(Float32, MODEL_DIM, AUX_FEATURES),
        :aux_position => zeros(Float32, MODEL_DIM, AUX_FEATURES),
        :register_seed => zeros(Float32, MODEL_DIM, REGISTER_COUNT),
        :spatial_q => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :spatial_k => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :spatial_v => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :spatial_o => zeros(Float32, MODEL_DIM, ATTENTION_DIM),
        :spatial_relative_bias => zeros(Float32, 3, 3, ATTENTION_HEADS),
        :spatial_scale_logit => zeros(Float32, 1),
        :cross_q => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :cross_k => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :cross_v => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :cross_o => zeros(Float32, MODEL_DIM, ATTENTION_DIM),
        :cross_scale_logit => zeros(Float32, 1),
        :relation_scale_logit => zeros(Float32, 1),
        :memory_write_v => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :memory_write_o => zeros(Float32, MODEL_DIM, ATTENTION_DIM),
        :memory_write_scale_logit => zeros(Float32, 1),
        :self_q => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :self_k => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :self_v => zeros(Float32, ATTENTION_DIM, MODEL_DIM),
        :self_o => zeros(Float32, MODEL_DIM, ATTENTION_DIM),
        :self_scale_logit => zeros(Float32, 1),
        :ffn_gate => zeros(Float32, FFN_DIM, MODEL_DIM),
        :ffn_up => zeros(Float32, FFN_DIM, MODEL_DIM),
        :ffn_down => zeros(Float32, MODEL_DIM, FFN_DIM),
        :ffn_scale_logit => zeros(Float32, 1),
        :lookup_register_gate => zeros(Float32, REGISTER_COUNT),
    )
end

function GradientAccumulator()
    return GradientAccumulator(
        SparseLookup.GradientAccumulator(), _zero_dense_gradients(),
        falses(TOKEN_COUNT), BackwardScratch(),
    )
end

function GradientAccumulator(model::EpisodicViTLookupModel)
    dense = Dict{Symbol,Array{Float32}}()
    for (name, parameter) in pairs(_dense_parameters(model))
        dense[name] = zeros(Float32, size(parameter))
    end
    return GradientAccumulator(
        SparseLookup.GradientAccumulator(), dense, falses(TOKEN_COUNT),
        BackwardScratch(),
    )
end

@inline function _dgm(accumulator::GradientAccumulator, name::Symbol)
    return accumulator.dense[name]::Matrix{Float32}
end

@inline function _dgv(accumulator::GradientAccumulator, name::Symbol)
    return accumulator.dense[name]::Vector{Float32}
end

"""Clear a reusable thread-local accumulator without reallocating dense buffers."""
function reset_gradients!(accumulator::GradientAccumulator)
    lookup = accumulator.lookup
    SparseLookup.recycle_bank_gradients!(lookup)
    for block in 1:BLOCKS
        fill!(lookup.dbh4[block], 0.0f0)
    end
    fill!(lookup.dalpha_logits, 0.0f0)
    fill!(lookup.dhead, 0.0f0)
    fill!(lookup.dbias, 0.0f0)
    fill!(lookup.dhalt_weight, 0.0f0)
    fill!(lookup.dhalt_bias, 0.0f0)
    fill!(lookup.dreinject_logit, 0.0f0)
    fill!(lookup.selected_row_events, 0)
    for gradient in values(accumulator.dense)
        fill!(gradient, 0.0f0)
    end
    fill!(accumulator.active_tokens, false)
    return accumulator
end

"""Deterministically merge a thread-local gradient into the destination."""
function merge_gradients!(
    destination::GradientAccumulator,
    source::GradientAccumulator,
)
    destination_lookup = destination.lookup
    source_lookup = source.lookup
    for block in 1:BLOCKS
        source_columns = sort!(collect(keys(source_lookup.bank_gradients[block])))
        for column in source_columns
            source_gradient = source_lookup.bank_gradients[block][column]
            target_gradient = SparseLookup._bank_gradient_target!(
                destination_lookup, block, column,
            )
            @inbounds @simd for coordinate in eachindex(source_gradient)
                target_gradient[coordinate] += source_gradient[coordinate]
            end
        end
        destination_lookup.dbh4[block] .+= source_lookup.dbh4[block]
    end
    destination_lookup.dalpha_logits .+= source_lookup.dalpha_logits
    destination_lookup.dhead .+= source_lookup.dhead
    destination_lookup.dbias .+= source_lookup.dbias
    destination_lookup.dhalt_weight .+= source_lookup.dhalt_weight
    destination_lookup.dhalt_bias .+= source_lookup.dhalt_bias
    destination_lookup.dreinject_logit .+= source_lookup.dreinject_logit
    destination_lookup.selected_row_events .+= source_lookup.selected_row_events
    for name in sort!(collect(keys(source.dense)); by=String)
        destination.dense[name] .+= source.dense[name]
    end
    destination.active_tokens .|= source.active_tokens
    return destination
end

function _spatial_attention_vjp!(
    accumulator, model, tape::SpatialAttentionTape,
    output_cotangent, dinput, scratch::BackwardScratch,
)
    copyto!(dinput, output_cotangent)
    scale_gradient = dot(
        @view(output_cotangent[:, 1:CELL_COUNT]), tape.projected,
    ) * _residual_scale_derivative(model.spatial_scale_logit[1])
    _dgv(accumulator, :spatial_scale_logit)[1] += scale_gradient

    dprojected = scratch.spatial_dprojected
    @inbounds for token in 1:CELL_COUNT
        @simd for coordinate in 1:MODEL_DIM
            dprojected[coordinate, token] =
                tape.alpha * output_cotangent[coordinate, token]
        end
    end
    dcontext = scratch.spatial_dcontext
    _structured_attention_expand_vjp!(
        _dgm(accumulator, :spatial_o),
        dcontext,
        model.spatial_o,
        tape.context,
        dprojected,
    )
    dquery = scratch.spatial_dquery
    dkey = scratch.spatial_dkey
    dvalue = scratch.spatial_dvalue
    fill!(dquery, 0.0f0)
    fill!(dkey, 0.0f0)
    fill!(dvalue, 0.0f0)
    relative_gradient = accumulator.dense[:spatial_relative_bias]::Array{Float32,3}
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    score_scale = inv(sqrt(Float32(head_dim)))
    dweights = scratch.dweights
    @inbounds for token in 1:CELL_COUNT
        row, column = _cell_row_column(token)
        count = Int(tape.neighbor_counts[token])
        for head in 1:ATTENTION_HEADS
            first_coordinate = (head - 1) * head_dim + 1
            last_coordinate = head * head_dim
            projection = 0.0f0
            for edge in 1:count
                neighbor = Int(tape.neighbor_ids[edge, token])
                value_gradient = 0.0f0
                @simd for coordinate in first_coordinate:last_coordinate
                    value_gradient = muladd(
                        tape.value[coordinate, neighbor],
                        dcontext[coordinate, token], value_gradient,
                    )
                end
                dweights[edge] = value_gradient
                projection = muladd(
                    tape.weights[edge, token, head], value_gradient, projection,
                )
            end
            for edge in 1:count
                neighbor = Int(tape.neighbor_ids[edge, token])
                neighbor_row, neighbor_column = _cell_row_column(neighbor)
                probability = tape.weights[edge, token, head]
                dscore = probability * (dweights[edge] - projection)
                relative_gradient[
                    neighbor_row - row + 2,
                    neighbor_column - column + 2,
                    head,
                ] += dscore
                @simd for coordinate in first_coordinate:last_coordinate
                    dquery[coordinate, token] = muladd(
                        dscore * score_scale, tape.key[coordinate, neighbor],
                        dquery[coordinate, token],
                    )
                    dkey[coordinate, neighbor] = muladd(
                        dscore * score_scale, tape.query[coordinate, token],
                        dkey[coordinate, neighbor],
                    )
                    dvalue[coordinate, neighbor] = muladd(
                        probability, dcontext[coordinate, token],
                        dvalue[coordinate, neighbor],
                    )
                end
            end
        end
    end
    cell_normalized = @view tape.normalized[:, 1:CELL_COUNT]
    dnormalized = scratch.spatial_dnormalized
    projection_input_gradient = scratch.spatial_projection_input
    _structured_attention_vjp!(
        _dgm(accumulator, :spatial_q),
        dnormalized,
        model.spatial_q,
        cell_normalized,
        dquery,
    )
    _structured_attention_vjp!(
        _dgm(accumulator, :spatial_k),
        projection_input_gradient,
        model.spatial_k,
        cell_normalized,
        dkey,
    )
    dnormalized .+= projection_input_gradient
    _structured_attention_vjp!(
        _dgm(accumulator, :spatial_v),
        projection_input_gradient,
        model.spatial_v,
        cell_normalized,
        dvalue,
    )
    dnormalized .+= projection_input_gradient
    _rmsnorm_columns_vjp!(
        projection_input_gradient,
        cell_normalized,
        @view(tape.inverse_rms[1:CELL_COUNT]),
        dnormalized,
    )
    @inbounds for token in 1:CELL_COUNT
        @simd for coordinate in 1:MODEL_DIM
            dinput[coordinate, token] +=
                projection_input_gradient[coordinate, token]
        end
    end
    return dinput
end

function _attention_core_vjp!(
    dquery, dkey, dvalue, dscore_total, dweights,
    query, key, value, weights, context_cotangent,
    external_dweights=nothing,
)
    queries = size(query, 2)
    keys = size(key, 2)
    size(dquery) == size(query) ||
        throw(DimensionMismatch("attention dquery scratch shape changed"))
    size(dkey) == size(key) ||
        throw(DimensionMismatch("attention dkey scratch shape changed"))
    size(dvalue) == size(value) ||
        throw(DimensionMismatch("attention dvalue scratch shape changed"))
    size(dscore_total) == (keys, queries) ||
        throw(DimensionMismatch("attention dscore scratch shape changed"))
    size(dweights) == (keys, queries) ||
        throw(DimensionMismatch("attention dweight scratch shape changed"))
    fill!(dquery, 0.0f0)
    fill!(dkey, 0.0f0)
    fill!(dvalue, 0.0f0)
    fill!(dscore_total, 0.0f0)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    scale = inv(sqrt(Float32(head_dim)))
    @inbounds for head in 1:ATTENTION_HEADS
        first_coordinate = (head - 1) * head_dim + 1
        last_coordinate = head * head_dim
        for query_index in 1:queries
            projection = 0.0f0
            for key_index in 1:keys
                value_gradient = 0.0f0
                @simd for coordinate in first_coordinate:last_coordinate
                    value_gradient = muladd(
                        value[coordinate, key_index],
                        context_cotangent[coordinate, query_index],
                        value_gradient,
                    )
                    dvalue[coordinate, key_index] = muladd(
                        weights[key_index, query_index, head],
                        context_cotangent[coordinate, query_index],
                        dvalue[coordinate, key_index],
                    )
                end
                if external_dweights !== nothing
                    value_gradient += external_dweights[
                        key_index, query_index, head,
                    ]
                end
                dweights[key_index, query_index] = value_gradient
                projection = muladd(
                    weights[key_index, query_index, head],
                    value_gradient,
                    projection,
                )
            end
            for key_index in 1:keys
                dscore = weights[key_index, query_index, head] *
                    (dweights[key_index, query_index] - projection) * scale
                dscore_total[key_index, query_index] += dscore
                @simd for coordinate in first_coordinate:last_coordinate
                    dquery[coordinate, query_index] = muladd(
                        key[coordinate, key_index], dscore,
                        dquery[coordinate, query_index],
                    )
                    dkey[coordinate, key_index] = muladd(
                        query[coordinate, query_index], dscore,
                        dkey[coordinate, key_index],
                    )
                end
            end
        end
    end
    return dquery, dkey, dvalue, dscore_total
end

function _attention_core_vjp(query, key, value, weights, context_cotangent)
    dquery = zeros(Float32, size(query))
    dkey = zeros(Float32, size(key))
    dvalue = zeros(Float32, size(value))
    dscore_total = zeros(Float32, size(key, 2), size(query, 2))
    dweights = similar(dscore_total)
    return _attention_core_vjp!(
        dquery, dkey, dvalue, dscore_total, dweights,
        query, key, value, weights, context_cotangent,
    )
end

function _cross_attention_vjp!(
    accumulator, model, tape, memory_normalized, state_cotangent,
    scratch::BackwardScratch, dinput, dmemory_normalized,
    external_dweights=nothing,
)
    fill!(dmemory_normalized, 0.0f0)
    _dgv(accumulator, :cross_scale_logit)[1] +=
        dot(state_cotangent, tape.output) *
        _residual_scale_derivative(model.cross_scale_logit[1])
    output_cotangent = scratch.context_cotangent
    @inbounds @simd for index in eachindex(output_cotangent)
        output_cotangent[index] = tape.alpha * state_cotangent[index]
    end
    cross_o_gradient = _dgm(accumulator, :cross_o)
    @inbounds for attention_coordinate in 1:ATTENTION_DIM
        for coordinate in 1:MODEL_DIM
            gradient = 0.0f0
            @simd for register in 1:REGISTER_COUNT
                gradient = muladd(
                    output_cotangent[coordinate, register],
                    tape.attention_context[attention_coordinate, register],
                    gradient,
                )
            end
            cross_o_gradient[coordinate, attention_coordinate] += gradient
        end
    end
    attention_context_cotangent = scratch.attention_dcontext
    mul!(
        attention_context_cotangent,
        transpose(model.cross_o),
        output_cotangent,
    )
    dquery = scratch.attention_dquery
    dkey = scratch.attention_dkey
    dvalue = scratch.attention_dvalue
    _attention_core_vjp!(
        dquery,
        dkey,
        dvalue,
        scratch.attention_dscore,
        scratch.attention_dweights,
        tape.query,
        tape.key,
        tape.value,
        tape.attention_weights,
        attention_context_cotangent,
        external_dweights,
    )

    # All 283 memory tokens participate, so task gradients reach every K/V
    # projection and every episodic token without a routing STE.
    mul!(
        _dgm(accumulator, :cross_q), dquery, transpose(tape.normalized),
        1.0f0, 1.0f0,
    )
    mul!(
        _dgm(accumulator, :cross_k), dkey, transpose(memory_normalized),
        1.0f0, 1.0f0,
    )
    mul!(
        _dgm(accumulator, :cross_v), dvalue, transpose(memory_normalized),
        1.0f0, 1.0f0,
    )
    dnormalized = scratch.dnormalized
    mul!(dnormalized, transpose(model.cross_q), dquery)
    mul!(dmemory_normalized, transpose(model.cross_k), dkey)
    mul!(
        dmemory_normalized, transpose(model.cross_v), dvalue,
        1.0f0, 1.0f0,
    )
    _rmsnorm_columns_vjp!(
        dinput, tape.normalized, tape.inverse_rms, dnormalized,
    )
    @inbounds @simd for index in eachindex(dinput)
        dinput[index] += state_cotangent[index]
    end
    return dinput
end

function _working_memory_write_vjp!(
    accumulator,
    model,
    tape::WorkingMemoryWriteTape,
    memory_cotangent,
    state_cotangent,
    scratch::BackwardScratch,
)
    _dgv(accumulator, :memory_write_scale_logit)[1] +=
        dot(memory_cotangent, tape.projected) *
        _residual_scale_derivative(model.memory_write_scale_logit[1])

    dprojected = scratch.memory_previous
    @inbounds @simd for index in eachindex(dprojected)
        dprojected[index] = tape.alpha * memory_cotangent[index]
    end
    mul!(
        _dgm(accumulator, :memory_write_o),
        dprojected,
        transpose(tape.context),
        1.0f0,
        1.0f0,
    )
    dcontext = scratch.memory_write_dcontext
    mul!(dcontext, transpose(model.memory_write_o), dprojected)
    dvalue = scratch.memory_write_dvalue
    fill!(dvalue, 0.0f0)
    external = scratch.attention_external_dweights
    fill!(external, 0.0f0)
    head_dim = ATTENTION_DIM ÷ ATTENTION_HEADS
    @inbounds for head in 1:ATTENTION_HEADS
        first_coordinate = (head - 1) * head_dim + 1
        last_coordinate = head * head_dim
        for token in 1:TOKEN_COUNT
            projection = 0.0f0
            for register in 1:REGISTER_COUNT
                dweight = 0.0f0
                @simd for coordinate in first_coordinate:last_coordinate
                    dweight = muladd(
                        dcontext[coordinate, token],
                        tape.value[coordinate, register],
                        dweight,
                    )
                    dvalue[coordinate, register] = muladd(
                        tape.weights[token, register, head],
                        dcontext[coordinate, token],
                        dvalue[coordinate, register],
                    )
                end
                external[token, register, head] = dweight
                projection = muladd(
                    tape.weights[token, register, head], dweight, projection,
                )
            end
            inverse = tape.inverse_weight_sums[token, head]
            for register in 1:REGISTER_COUNT
                external[token, register, head] = inverse * (
                    external[token, register, head] - projection
                )
            end
        end
    end
    mul!(
        _dgm(accumulator, :memory_write_v),
        dvalue,
        transpose(tape.registers),
        1.0f0,
        1.0f0,
    )
    dregisters = scratch.memory_write_dregisters
    mul!(dregisters, transpose(model.memory_write_v), dvalue)
    @inbounds @simd for index in eachindex(state_cotangent)
        state_cotangent[index] += dregisters[index]
    end
    return external
end

function _self_attention_vjp!(
    accumulator, model, tape, state_cotangent,
    scratch::BackwardScratch, dinput,
)
    _dgv(accumulator, :self_scale_logit)[1] +=
        dot(state_cotangent, tape.output) *
        _residual_scale_derivative(model.self_scale_logit[1])
    doutput = scratch.context_cotangent
    @inbounds @simd for index in eachindex(doutput)
        doutput[index] = tape.alpha * state_cotangent[index]
    end
    self_o_gradient = _dgm(accumulator, :self_o)
    @inbounds for attention_coordinate in 1:ATTENTION_DIM
        for coordinate in 1:MODEL_DIM
            gradient = 0.0f0
            @simd for register in 1:REGISTER_COUNT
                gradient = muladd(
                    doutput[coordinate, register],
                    tape.context[attention_coordinate, register],
                    gradient,
                )
            end
            self_o_gradient[coordinate, attention_coordinate] += gradient
        end
    end
    dcontext = scratch.attention_dcontext
    mul!(dcontext, transpose(model.self_o), doutput)
    dquery = scratch.attention_dquery
    dkey = @view scratch.attention_dkey[:, 1:REGISTER_COUNT]
    dvalue = @view scratch.attention_dvalue[:, 1:REGISTER_COUNT]
    dscore = @view scratch.attention_dscore[1:REGISTER_COUNT, 1:REGISTER_COUNT]
    dweights = @view scratch.attention_dweights[1:REGISTER_COUNT, 1:REGISTER_COUNT]
    _attention_core_vjp!(
        dquery, dkey, dvalue, dscore, dweights,
        tape.query, tape.key, tape.value, tape.weights, dcontext,
    )
    self_q_gradient = _dgm(accumulator, :self_q)
    self_k_gradient = _dgm(accumulator, :self_k)
    self_v_gradient = _dgm(accumulator, :self_v)
    @inbounds for attention_coordinate in 1:ATTENTION_DIM
        for coordinate in 1:MODEL_DIM
            q_gradient = 0.0f0
            k_gradient = 0.0f0
            v_gradient = 0.0f0
            @simd for register in 1:REGISTER_COUNT
                normalized_value = tape.normalized[coordinate, register]
                q_gradient = muladd(
                    dquery[attention_coordinate, register],
                    normalized_value,
                    q_gradient,
                )
                k_gradient = muladd(
                    dkey[attention_coordinate, register],
                    normalized_value,
                    k_gradient,
                )
                v_gradient = muladd(
                    dvalue[attention_coordinate, register],
                    normalized_value,
                    v_gradient,
                )
            end
            self_q_gradient[attention_coordinate, coordinate] += q_gradient
            self_k_gradient[attention_coordinate, coordinate] += k_gradient
            self_v_gradient[attention_coordinate, coordinate] += v_gradient
        end
    end
    dnormalized = scratch.dnormalized
    fill!(dnormalized, 0.0f0)
    @inbounds for register in 1:REGISTER_COUNT
        for attention_coordinate in 1:ATTENTION_DIM
            q_gradient = dquery[attention_coordinate, register]
            k_gradient = dkey[attention_coordinate, register]
            v_gradient = dvalue[attention_coordinate, register]
            @simd for coordinate in 1:MODEL_DIM
                dnormalized[coordinate, register] = muladd(
                    model.self_q[attention_coordinate, coordinate], q_gradient,
                    muladd(
                        model.self_k[attention_coordinate, coordinate], k_gradient,
                        muladd(
                            model.self_v[attention_coordinate, coordinate], v_gradient,
                            dnormalized[coordinate, register],
                        ),
                    ),
                )
            end
        end
    end
    _rmsnorm_columns_vjp!(
        dinput, tape.normalized, tape.inverse_rms, dnormalized,
    )
    @inbounds @simd for index in eachindex(dinput)
        dinput[index] += state_cotangent[index]
    end
    return dinput
end

function _swiglu_vjp!(
    accumulator, model, tape, state_cotangent,
    scratch::BackwardScratch, dinput,
    ffn_scale_contributions,
    step_index::Int,
)
    ffn_scale_contribution =
        dot(state_cotangent, tape.output) *
        _residual_scale_derivative(model.ffn_scale_logit[1])
    if ffn_scale_contributions === nothing
        _dgv(accumulator, :ffn_scale_logit)[1] += ffn_scale_contribution
    else
        @inbounds ffn_scale_contributions[step_index] = ffn_scale_contribution
    end
    output_cotangent = scratch.ffn_output_cotangent
    @inbounds @simd for index in eachindex(output_cotangent)
        output_cotangent[index] = tape.alpha * state_cotangent[index]
    end
    down_product = scratch.ffn_down_product
    mul!(down_product, output_cotangent, transpose(tape.hidden))
    down_gradient = _dgm(accumulator, :ffn_down)
    @inbounds @simd for index in eachindex(down_gradient)
        down_gradient[index] += down_product[index]
    end
    dhidden = scratch.ffn_hidden_cotangent
    mul!(dhidden, transpose(model.ffn_down), output_cotangent)
    dgate = scratch.ffn_gate_cotangent
    dup = scratch.ffn_up_cotangent
    @inbounds @simd for index in eachindex(dhidden)
        dgate[index] = dhidden[index] * tape.up[index] *
            _swish_derivative(tape.gate[index])
        dup[index] = dhidden[index] * _swish(tape.gate[index])
    end
    parameter_product = scratch.ffn_parameter_product
    gate_gradient = _dgm(accumulator, :ffn_gate)
    mul!(parameter_product, dgate, transpose(tape.normalized))
    @inbounds @simd for index in eachindex(gate_gradient)
        gate_gradient[index] += parameter_product[index]
    end
    up_gradient = _dgm(accumulator, :ffn_up)
    mul!(parameter_product, dup, transpose(tape.normalized))
    @inbounds @simd for index in eachindex(up_gradient)
        up_gradient[index] += parameter_product[index]
    end
    dnormalized_a = scratch.ffn_normalized_a
    dnormalized_b = scratch.ffn_normalized_b
    mul!(dnormalized_a, transpose(model.ffn_gate), dgate)
    mul!(dnormalized_b, transpose(model.ffn_up), dup)
    @inbounds @simd for index in eachindex(dnormalized_a)
        dnormalized_a[index] += dnormalized_b[index]
    end
    _rmsnorm_columns_vjp!(
        dinput, tape.normalized, tape.inverse_rms, dnormalized_a,
    )
    @inbounds @simd for index in eachindex(dinput)
        dinput[index] += state_cotangent[index]
    end
    return dinput
end

function _lookup_carrier_vjp!(
    accumulator, model, tape, state_cotangent, temperature,
    scratch::BackwardScratch, dinput,
    balance_stats=nothing,
    balance_weight::Float32=0.0f0,
)
    # The direct residual and every LookupFFN trajectory are register-local.
    # Parameters are shared, so gradients accumulate into the same sparse bank
    # rows without pooling the credit signal between registers.
    copyto!(dinput, state_cotangent)
    gate_gradient = _dgv(accumulator, :lookup_register_gate)
    @inbounds for register in 1:REGISTER_COUNT
        gate = tape.gates[register]
        register_cotangent = @view state_cotangent[:, register]
        gate_gradient[register] += dot(
            register_cotangent, @view(tape.residual[:, register]),
        ) *
            gate * (1.0f0 - gate)
        carrier_cotangent = scratch.residual_cotangent
        @simd for coordinate in 1:MODEL_DIM
            carrier_cotangent[coordinate] = gate * register_cotangent[coordinate]
        end
        lookup_input_cotangent = carrier_cotangent
        for block in BLOCKS:-1:1
            lookup_input_cotangent = SparseLookup._lookup_micro_vjp!(
                accumulator.lookup, model.lookup, tape.blocks[block, register],
                block, lookup_input_cotangent, Float32(temperature),
                balance_stats === nothing ? nothing :
                    balance_stats.hard_frequencies,
                balance_stats === nothing ? 0 :
                    Int(balance_stats.observations[block]),
                balance_weight,
            )
        end
        @simd for coordinate in 1:MODEL_DIM
            # output = input + gate * (lookup(input) - input)
            dinput[coordinate, register] +=
                lookup_input_cotangent[coordinate] - carrier_cotangent[coordinate]
        end
    end
    return dinput
end

function _rmsnorm_selected_vjp!(
    scratch::BackwardScratch, normalized, inverse_rms,
)
    width = Float32(size(normalized, 1))
    @inbounds for active_index in 1:scratch.memory_count
        token = Int(scratch.memory_ids[active_index])
        projection = dot(
            @view(scratch.memory_gradient[:, token]),
            @view(normalized[:, token]),
        ) / width
        inverse = inverse_rms[token]
        @simd for coordinate in 1:MODEL_DIM
            scratch.memory_postnorm[coordinate, token] = inverse * (
                scratch.memory_gradient[coordinate, token] -
                normalized[coordinate, token] * projection
            )
        end
    end
    return nothing
end

function _cell_token_vjp!(
    accumulator,
    _model,
    input,
    row::Int,
    column::Int,
    token::Int,
    token_cotangent,
)
    cell_projection = _dgm(accumulator, :cell_projection)
    cell_bias = _dgv(accumulator, :cell_bias)
    cell_position = _dgm(accumulator, :cell_position)
    board = input.board[row, column]
    candidate = input.candidate[row, column]
    difference = input.difference[row, column]
    @inbounds for coordinate in 1:MODEL_DIM
        gradient = token_cotangent[coordinate]
        cell_bias[coordinate] += gradient
        cell_position[coordinate, token] += gradient
        cell_projection[coordinate, 1] = muladd(
            gradient, board, cell_projection[coordinate, 1],
        )
        cell_projection[coordinate, 2] = muladd(
            gradient, candidate, cell_projection[coordinate, 2],
        )
        cell_projection[coordinate, 3] = muladd(
            gradient, difference, cell_projection[coordinate, 3],
        )
    end
    return nothing
end

function _visual_stack_forward_features!(features, model, input)
    @inbounds for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
        token = (column - 1) * BOARD_HEIGHT + row
        for channel in 1:VISUAL_CHANNELS
            features[channel, token, 1] = _visual_input_channel(
                input, row, column, channel,
            )
        end
    end
    @inbounds for stage in 1:VISUAL_STAGES
        dilation = VISUAL_DILATIONS[stage]
        for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
            token = (column - 1) * BOARD_HEIGHT + row
            pre_1 = 0.0f0
            pre_2 = 0.0f0
            pre_3 = 0.0f0
            for delta_column in -1:1, delta_row in -1:1
                source_row = row + dilation * delta_row
                source_column = column + dilation * delta_column
                1 <= source_row <= BOARD_HEIGHT || continue
                1 <= source_column <= BOARD_WIDTH || continue
                source_token = (source_column - 1) * BOARD_HEIGHT + source_row
                pre_1 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 1, stage,
                    ], features[1, source_token, stage], pre_1,
                )
                pre_2 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 2, stage,
                    ], features[2, source_token, stage], pre_2,
                )
                pre_3 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 3, stage,
                    ], features[3, source_token, stage], pre_3,
                )
            end
            visual_1 = _swish(pre_1)
            visual_2 = _swish(pre_2)
            visual_3 = _swish(pre_3)
            for output_channel in 1:VISUAL_CHANNELS
                mixed = model.visual_channel_mix[output_channel, 1, stage] *
                    visual_1 +
                    model.visual_channel_mix[output_channel, 2, stage] *
                    visual_2 +
                    model.visual_channel_mix[output_channel, 3, stage] *
                    visual_3
                features[output_channel, token, stage + 1] = mixed +
                    features[output_channel, token, stage]
            end
        end
    end
    return features
end

function _visual_stem_vjp!(
    accumulator,
    model,
    input,
    token_cotangents,
    scratch::BackwardScratch,
)
    features = _visual_stack_forward_features!(
        scratch.visual_features, model, input,
    )
    feature_cotangents = scratch.visual_cotangents
    fill!(feature_cotangents, 0.0f0)
    ddepthwise = accumulator.dense[:visual_depthwise]::Array{Float32,4}
    dchannel_mix = accumulator.dense[:visual_channel_mix]::Array{Float32,3}
    dpointwise = _dgm(accumulator, :visual_pointwise)
    dscale = _dgv(accumulator, :visual_scale_logit)
    final_stage = VISUAL_STAGES + 1
    output_alpha = _residual_scale(model.visual_scale_logit[1])
    output_scale_derivative = _residual_scale_derivative(
        model.visual_scale_logit[1],
    )
    @inbounds for token in 1:CELL_COUNT
        visual_1 = features[1, token, final_stage]
        visual_2 = features[2, token, final_stage]
        visual_3 = features[3, token, final_stage]
        for coordinate in 1:MODEL_DIM
            gradient = token_cotangents[coordinate, token]
            visual_output = model.visual_pointwise[coordinate, 1] * visual_1 +
                model.visual_pointwise[coordinate, 2] * visual_2 +
                model.visual_pointwise[coordinate, 3] * visual_3
            dscale[1] = muladd(
                gradient, visual_output * output_scale_derivative, dscale[1],
            )
            scaled_gradient = output_alpha * gradient
            dpointwise[coordinate, 1] = muladd(
                scaled_gradient, visual_1, dpointwise[coordinate, 1],
            )
            dpointwise[coordinate, 2] = muladd(
                scaled_gradient, visual_2, dpointwise[coordinate, 2],
            )
            dpointwise[coordinate, 3] = muladd(
                scaled_gradient, visual_3, dpointwise[coordinate, 3],
            )
            feature_cotangents[1, token, final_stage] = muladd(
                scaled_gradient, model.visual_pointwise[coordinate, 1],
                feature_cotangents[1, token, final_stage],
            )
            feature_cotangents[2, token, final_stage] = muladd(
                scaled_gradient, model.visual_pointwise[coordinate, 2],
                feature_cotangents[2, token, final_stage],
            )
            feature_cotangents[3, token, final_stage] = muladd(
                scaled_gradient, model.visual_pointwise[coordinate, 3],
                feature_cotangents[3, token, final_stage],
            )
        end
    end
    @inbounds for stage in VISUAL_STAGES:-1:1
        dilation = VISUAL_DILATIONS[stage]
        for column in 1:BOARD_WIDTH, row in 1:BOARD_HEIGHT
            token = (column - 1) * BOARD_HEIGHT + row
            pre_1 = 0.0f0
            pre_2 = 0.0f0
            pre_3 = 0.0f0
            for delta_column in -1:1, delta_row in -1:1
                source_row = row + dilation * delta_row
                source_column = column + dilation * delta_column
                1 <= source_row <= BOARD_HEIGHT || continue
                1 <= source_column <= BOARD_WIDTH || continue
                source_token = (source_column - 1) * BOARD_HEIGHT + source_row
                pre_1 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 1, stage,
                    ], features[1, source_token, stage], pre_1,
                )
                pre_2 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 2, stage,
                    ], features[2, source_token, stage], pre_2,
                )
                pre_3 = muladd(
                    model.visual_depthwise[
                        delta_row + 2, delta_column + 2, 3, stage,
                    ], features[3, source_token, stage], pre_3,
                )
            end
            visual_1 = _swish(pre_1)
            visual_2 = _swish(pre_2)
            visual_3 = _swish(pre_3)
            dout_1 = feature_cotangents[1, token, stage + 1]
            dout_2 = feature_cotangents[2, token, stage + 1]
            dout_3 = feature_cotangents[3, token, stage + 1]
            feature_cotangents[1, token, stage] += dout_1
            feature_cotangents[2, token, stage] += dout_2
            feature_cotangents[3, token, stage] += dout_3
            dvisual_1 = 0.0f0
            dvisual_2 = 0.0f0
            dvisual_3 = 0.0f0
            for output_channel in 1:VISUAL_CHANNELS
                dout = output_channel == 1 ? dout_1 :
                    output_channel == 2 ? dout_2 : dout_3
                scaled_dout = dout
                dchannel_mix[output_channel, 1, stage] = muladd(
                    scaled_dout, visual_1,
                    dchannel_mix[output_channel, 1, stage],
                )
                dchannel_mix[output_channel, 2, stage] = muladd(
                    scaled_dout, visual_2,
                    dchannel_mix[output_channel, 2, stage],
                )
                dchannel_mix[output_channel, 3, stage] = muladd(
                    scaled_dout, visual_3,
                    dchannel_mix[output_channel, 3, stage],
                )
                dvisual_1 = muladd(
                    scaled_dout,
                    model.visual_channel_mix[output_channel, 1, stage],
                    dvisual_1,
                )
                dvisual_2 = muladd(
                    scaled_dout,
                    model.visual_channel_mix[output_channel, 2, stage],
                    dvisual_2,
                )
                dvisual_3 = muladd(
                    scaled_dout,
                    model.visual_channel_mix[output_channel, 3, stage],
                    dvisual_3,
                )
            end
            dpre_1 = dvisual_1 * _swish_derivative(pre_1)
            dpre_2 = dvisual_2 * _swish_derivative(pre_2)
            dpre_3 = dvisual_3 * _swish_derivative(pre_3)
            for delta_column in -1:1, delta_row in -1:1
                source_row = row + dilation * delta_row
                source_column = column + dilation * delta_column
                1 <= source_row <= BOARD_HEIGHT || continue
                1 <= source_column <= BOARD_WIDTH || continue
                source_token = (source_column - 1) * BOARD_HEIGHT + source_row
                for channel in 1:VISUAL_CHANNELS
                    dpre = channel == 1 ? dpre_1 :
                        channel == 2 ? dpre_2 : dpre_3
                    kernel_row = delta_row + 2
                    kernel_column = delta_column + 2
                    ddepthwise[kernel_row, kernel_column, channel, stage] =
                        muladd(
                            dpre, features[channel, source_token, stage],
                            ddepthwise[
                                kernel_row, kernel_column, channel, stage,
                            ],
                        )
                    feature_cotangents[channel, source_token, stage] = muladd(
                        dpre,
                        model.visual_depthwise[
                            kernel_row, kernel_column, channel, stage,
                        ],
                        feature_cotangents[channel, source_token, stage],
                    )
                end
            end
        end
    end
    return nothing
end

function _tokenize_vjp!(accumulator, model, input, token_cotangents)
    next_projection = _dgm(accumulator, :next_projection)
    next_bias = _dgv(accumulator, :next_bias)
    next_position = _dgm(accumulator, :next_position)
    aux_value = _dgm(accumulator, :aux_value)
    aux_position = _dgm(accumulator, :aux_position)
    for (token, token_cotangent) in token_cotangents
        if token <= CELL_COUNT
            row = mod(token - 1, BOARD_HEIGHT) + 1
            column = div(token - 1, BOARD_HEIGHT) + 1
            _cell_token_vjp!(
                accumulator, model, input, row, column, token,
                token_cotangent,
            )
        elseif token <= CELL_COUNT + NEXT_HOLD_TOKENS
            queue_token = token - CELL_COUNT
            @inbounds for coordinate in 1:MODEL_DIM
                gradient = token_cotangent[coordinate]
                next_bias[coordinate] += gradient
                next_position[coordinate, queue_token] += gradient
                @simd for piece in 1:PIECE_TYPES
                    next_projection[coordinate, piece] = muladd(
                        gradient, input.next_hold[piece, queue_token],
                        next_projection[coordinate, piece],
                    )
                end
            end
        else
            feature = token - CELL_COUNT - NEXT_HOLD_TOKENS
            value = input.aux[feature]
            @inbounds @simd for coordinate in 1:MODEL_DIM
                gradient = token_cotangent[coordinate]
                aux_position[coordinate, feature] += gradient
                aux_value[coordinate, feature] = muladd(
                    gradient, value, aux_value[coordinate, feature],
                )
            end
        end
    end
    return nothing
end


function _tokenize_vjp!(
    accumulator, model, input, scratch::BackwardScratch,
)
    next_projection = _dgm(accumulator, :next_projection)
    next_bias = _dgv(accumulator, :next_bias)
    next_position = _dgm(accumulator, :next_position)
    aux_value = _dgm(accumulator, :aux_value)
    aux_position = _dgm(accumulator, :aux_position)
    @inbounds for active_index in 1:scratch.memory_count
        token = Int(scratch.memory_ids[active_index])
        token_cotangent = @view scratch.memory_postnorm[:, token]
        if token <= CELL_COUNT
            row = mod(token - 1, BOARD_HEIGHT) + 1
            column = div(token - 1, BOARD_HEIGHT) + 1
            _cell_token_vjp!(
                accumulator, model, input, row, column, token,
                token_cotangent,
            )
        elseif token <= CELL_COUNT + NEXT_HOLD_TOKENS
            queue_token = token - CELL_COUNT
            for coordinate in 1:MODEL_DIM
                gradient = token_cotangent[coordinate]
                next_bias[coordinate] += gradient
                next_position[coordinate, queue_token] += gradient
                @simd for piece in 1:PIECE_TYPES
                    next_projection[coordinate, piece] = muladd(
                        gradient, input.next_hold[piece, queue_token],
                        next_projection[coordinate, piece],
                    )
                end
            end
        else
            feature = token - CELL_COUNT - NEXT_HOLD_TOKENS
            value = input.aux[feature]
            @simd for coordinate in 1:MODEL_DIM
                gradient = token_cotangent[coordinate]
                aux_position[coordinate, feature] += gradient
                aux_value[coordinate, feature] = muladd(
                    gradient, value, aux_value[coordinate, feature],
                )
            end
        end
    end
    return nothing
end

"""Backpropagate a dense recurrent-memory cotangent into the exact input tokens.

The spatial block gives previously unselected neighbour cells a real gradient,
so the old shortlist-only stamp cannot describe the active token set anymore.
This routine still preserves active-only optimiser semantics: a token column is
marked active only when its accumulated cotangent is actually non-zero.
"""
function _tokenize_vjp_dense!(
    accumulator, model, input, token_cotangents, scratch::BackwardScratch,
)
    size(token_cotangents) == (MODEL_DIM, TOKEN_COUNT) ||
        throw(DimensionMismatch("recurrent memory cotangent shape changed"))
    cell_projection = _dgm(accumulator, :cell_projection)
    cell_bias = _dgv(accumulator, :cell_bias)
    cell_position = _dgm(accumulator, :cell_position)
    next_projection = _dgm(accumulator, :next_projection)
    next_bias = _dgv(accumulator, :next_bias)
    next_position = _dgm(accumulator, :next_position)
    aux_value = _dgm(accumulator, :aux_value)
    aux_position = _dgm(accumulator, :aux_position)
    @inbounds for token in 1:TOKEN_COUNT
        active = false
        @simd for coordinate in 1:MODEL_DIM
            active |= !iszero(token_cotangents[coordinate, token])
        end
        active || continue
        accumulator.active_tokens[token] = true
        if token <= CELL_COUNT
            row, column = _cell_row_column(token)
            _cell_token_vjp!(
                accumulator, model, input, row, column, token,
                @view(token_cotangents[:, token]),
            )
        elseif token <= CELL_COUNT + NEXT_HOLD_TOKENS
            queue_token = token - CELL_COUNT
            for coordinate in 1:MODEL_DIM
                gradient = token_cotangents[coordinate, token]
                next_bias[coordinate] += gradient
                next_position[coordinate, queue_token] += gradient
                @simd for piece in 1:PIECE_TYPES
                    next_projection[coordinate, piece] = muladd(
                        gradient, input.next_hold[piece, queue_token],
                        next_projection[coordinate, piece],
                    )
                end
            end
        else
            feature = token - CELL_COUNT - NEXT_HOLD_TOKENS
            value = input.aux[feature]
            @simd for coordinate in 1:MODEL_DIM
                gradient = token_cotangents[coordinate, token]
                aux_position[coordinate, feature] += gradient
                aux_value[coordinate, feature] = muladd(
                    gradient, value, aux_value[coordinate, feature],
                )
            end
        end
    end
    _visual_stem_vjp!(
        accumulator, model, input, token_cotangents, scratch,
    )
    return nothing
end

function backward_trajectory!(
    accumulator::GradientAccumulator,
    model::EpisodicViTLookupModel,
    tape::TrajectoryTape,
    output_cotangent;
    realized_loss,
    baseline,
    compute_price=0.02f0,
    policy_weight=0.05f0,
    entropy_weight=0.001f0,
    halt_probe_mode::Bool=false,
    halt_probe_target=Float32(NaN),
    halt_probe_weight=1.0f0,
    temperature=0.50f0,
    ffn_scale_contributions=nothing,
    lookup_balance_stats=nothing,
    lookup_balance_weight=0.0f0,
)
    scratch = accumulator.backward_scratch
    length(output_cotangent) == OUTPUT_DIM || throw(DimensionMismatch(
        "episodic output cotangent shape changed",
    ))
    dy = scratch.dy
    @inbounds @simd for output_index in 1:OUTPUT_DIM
        dy[output_index] = Float32(output_cotangent[output_index])
    end
    final_registers = last(tape.steps).final_registers
    pooled = _pool_registers!(scratch.pooled, final_registers)
    state_cotangent = scratch.state_a
    fill!(state_cotangent, 0.0f0)
    @inbounds for output_index in 1:OUTPUT_DIM
        coefficient = dy[output_index]
        accumulator.lookup.dbias[output_index] += coefficient
        @simd for coordinate in 1:MODEL_DIM
            accumulator.lookup.dhead[output_index, coordinate] = muladd(
                coefficient, pooled[coordinate],
                accumulator.lookup.dhead[output_index, coordinate],
            )
            gradient = model.lookup.head[output_index, coordinate] * coefficient /
                Float32(REGISTER_COUNT)
            for register in 1:REGISTER_COUNT
                state_cotangent[coordinate, register] += gradient
            end
        end
    end

    depth = length(tape.steps)
    if ffn_scale_contributions !== nothing
        length(ffn_scale_contributions) >= depth || throw(DimensionMismatch(
            "FFN scale contribution scratch is shorter than trajectory depth",
        ))
        fill!(ffn_scale_contributions, 0.0f0)
    end
    depth <= length(scratch.halt_gradients) || throw(DimensionMismatch(
        "episodic recurrent depth exceeded backward scratch",
    ))
    halt_logit_gradients = scratch.halt_gradients
    fill!(halt_logit_gradients, 0.0f0)
    # Sparse one-step probes supplement the trajectory policy gradient; they
    # must not replace it.  Replacing it previously left every non-probed stop
    # and every earlier continue decision without any halting credit.
    if !tape.warmup_depth
        advantage = clamp(
            Float32(realized_loss) + Float32(compute_price) * Float32(depth) -
            Float32(baseline), -8.0f0, 8.0f0,
        )
        @inbounds for step_index in 1:depth
            step = tape.steps[step_index]
            step.stochastic_decision || continue
            probability = clamp(step.halt_probability, 1.0f-5, 1.0f0 - 1.0f-5)
            action_gradient = step.stopped ? 1.0f0 - probability : -probability
            entropy_gradient = -Float32(entropy_weight) * probability *
                (1.0f0 - probability) * log((1.0f0 - probability) / probability)
            halt_logit_gradients[step_index] = Float32(policy_weight) *
                advantage * action_gradient + entropy_gradient
        end
    end
    if halt_probe_mode
        target = Float32(halt_probe_target)
        if isfinite(target)
            0.0f0 <= target <= 1.0f0 || throw(ArgumentError(
                "halting probe target must lie in [0,1]",
            ))
            step = last(tape.steps)
            step.stochastic_decision && step.stopped && !step.forced_stop ||
                error("halting probe target was attached to an ineligible stop")
            probability = clamp(
                step.halt_probability, 1.0f-5, 1.0f0 - 1.0f-5,
            )
            halt_logit_gradients[depth] += Float32(halt_probe_weight) *
                (probability - target)
        end
    end

    # `memory_gradient` is the raw-memory cotangent arriving from the future
    # recurrent step.  Cross attention first produces a cotangent for the
    # normalized memory of this step; it is unnormalised, combined with the
    # future path, then propagated through the sparse spatial cell update.
    fill!(scratch.memory_gradient, 0.0f0)
    for step_index in depth:-1:1
        step = tape.steps[step_index]
        external_cross_dweights = _working_memory_write_vjp!(
            accumulator,
            model,
            step.memory_write,
            scratch.memory_gradient,
            state_cotangent,
            scratch,
        )
        halt_gradient = halt_logit_gradients[step_index]
        if !iszero(halt_gradient)
            step_pooled = _pool_registers!(scratch.pooled, step.final_registers)
            accumulator.lookup.dhalt_bias[1] += halt_gradient
            @inbounds @simd for coordinate in 1:MODEL_DIM
                accumulator.lookup.dhalt_weight[coordinate] = muladd(
                    halt_gradient, step_pooled[coordinate],
                    accumulator.lookup.dhalt_weight[coordinate],
                )
                gradient = model.lookup.halt_weight[coordinate] * halt_gradient /
                    Float32(REGISTER_COUNT)
                for register in 1:REGISTER_COUNT
                    state_cotangent[coordinate, register] += gradient
                end
            end
        end
        next_cotangent = state_cotangent === scratch.state_a ?
            scratch.state_b : scratch.state_a
        state_cotangent = _lookup_carrier_vjp!(
            accumulator, model, step.lookup, state_cotangent,
            Float32(temperature), scratch, next_cotangent,
            lookup_balance_stats, Float32(lookup_balance_weight),
        )
        next_cotangent = state_cotangent === scratch.state_a ?
            scratch.state_b : scratch.state_a
        state_cotangent = _swiglu_vjp!(
            accumulator, model, step.swiglu, state_cotangent,
            scratch, next_cotangent, ffn_scale_contributions, step_index,
        )
        next_cotangent = state_cotangent === scratch.state_a ?
            scratch.state_b : scratch.state_a
        state_cotangent = _self_attention_vjp!(
            accumulator, model, step.self, state_cotangent,
            scratch, next_cotangent,
        )
        next_cotangent = state_cotangent === scratch.state_a ?
            scratch.state_b : scratch.state_a
        state_cotangent = _cross_attention_vjp!(
            accumulator, model, step.cross, step.spatial.output_normalized,
            state_cotangent, scratch, next_cotangent,
            scratch.memory_postnorm,
            external_cross_dweights,
        )

        # Cross reads the RMS-normalized post-spatial memory.  Convert that
        # gradient back to the raw post-spatial state before joining the BPTT
        # edge from the following recurrent step.
        _rmsnorm_columns_vjp!(
            scratch.memory_previous,
            step.spatial.output_normalized,
            step.spatial.output_inverse_rms,
            scratch.memory_postnorm,
        )
        @inbounds @simd for index in eachindex(scratch.memory_previous)
            scratch.memory_previous[index] += scratch.memory_gradient[index]
        end
        _spatial_attention_vjp!(
            accumulator,
            model,
            step.spatial,
            scratch.memory_previous,
            scratch.memory_gradient,
            scratch,
        )
    end
    register_seed_gradient = _dgm(accumulator, :register_seed)
    @inbounds @simd for index in eachindex(register_seed_gradient)
        register_seed_gradient[index] += state_cotangent[index]
    end
    _tokenize_vjp_dense!(
        accumulator, model, tape.input, scratch.memory_gradient, scratch,
    )
    return nothing
end

mutable struct Optimizer
    lookup::SparseLookup.DynamicLookupOptimizer
    dense_states::Dict{Symbol,Tuple{Array{Float32},Array{Float32}}}
    token_event_count::Vector{UInt64}
    dense_step::UInt64
end

function initialize_optimizer(
    model::EpisodicViTLookupModel;
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    bank_learning_rate::Real=1.0f-4,
    bank_weight_decay::Real=0.0f0,
)
    lookup = SparseLookup.initialize_optimizer(
        model.lookup;
        beta1,
        beta2,
        epsilon,
        bank_learning_rate,
        bank_weight_decay,
    )
    states = Dict{Symbol,Tuple{Array{Float32},Array{Float32}}}()
    for (name, parameter) in pairs(_dense_parameters(model))
        states[name] = (
            zeros(Float32, size(parameter)), zeros(Float32, size(parameter)),
        )
    end
    return Optimizer(lookup, states, zeros(UInt64, TOKEN_COUNT), 0)
end

function configure_bank_optimizer!(
    optimizer::Optimizer;
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    bank_learning_rate::Real=1.0f-4,
    bank_weight_decay::Real=0.0f0,
    allow_moment_hyperparameter_change::Bool=false,
)
    SparseLookup.configure_bank_optimizer!(
        optimizer.lookup;
        beta1,
        beta2,
        epsilon,
        bank_learning_rate,
        bank_weight_decay,
        allow_moment_hyperparameter_change,
    )
    return optimizer
end

function materialize_selected_columns!(
    model::EpisodicViTLookupModel,
    optimizer::Optimizer,
    block::Int,
    columns,
)
    1 <= block <= BLOCKS || throw(BoundsError(model.lookup.banks, block))
    return SparseEngine.materialize_selected_columns!(
        model.lookup.banks[block], optimizer.lookup.bank_states[block], columns,
    )
end

@inline function _dense_array_abs2_sum(gradient::Array{Float32,N}) where {N}
    return sum(abs2, gradient)
end

@inline function _selected_token_gradient_abs2_sum(
    total::Float64,
    gradient::Matrix{Float32},
    active_tokens::BitVector,
    token_offset::Int,
)
    @inbounds for token in eachindex(active_tokens)
        active_tokens[token] || continue
        column = token - token_offset
        1 <= column <= size(gradient, 2) || continue
        @simd for row in axes(gradient, 1)
            total += abs2(Float64(gradient[row, column]))
        end
    end
    return total
end

@inline function _scale_dense_array!(
    gradient::Array{Float32,N},
    scale::Float32,
) where {N}
    @inbounds @simd for index in eachindex(gradient)
        gradient[index] *= scale
    end
    return nothing
end

@inline function _scale_selected_token_columns!(
    gradient::Matrix{Float32},
    active_tokens::BitVector,
    token_offset::Int,
    scale::Float32,
)
    @inbounds for token in eachindex(active_tokens)
        active_tokens[token] || continue
        column = token - token_offset
        1 <= column <= size(gradient, 2) || continue
        @simd for row in axes(gradient, 1)
            gradient[row, column] *= scale
        end
    end
    return nothing
end

function gradient_norm(accumulator::GradientAccumulator)
    total = Float64(SparseLookup.gradient_norm(accumulator.lookup))^2
    for (name, gradient) in accumulator.dense
        token_offset = _token_column_offset(name)
        if token_offset < 0
            total += _dense_array_abs2_sum(gradient)
            continue
        end
        total = _selected_token_gradient_abs2_sum(
            total,
            gradient::Matrix{Float32},
            accumulator.active_tokens,
            token_offset,
        )
    end
    return sqrt(total)
end

function scale_gradients!(accumulator::GradientAccumulator, scale)
    SparseLookup.scale_gradients!(accumulator.lookup, scale)
    scale32 = Float32(scale)
    for (name, gradient) in accumulator.dense
        token_offset = _token_column_offset(name)
        if token_offset < 0
            _scale_dense_array!(gradient, scale32)
            continue
        end
        _scale_selected_token_columns!(
            gradient::Matrix{Float32},
            accumulator.active_tokens,
            token_offset,
            scale32,
        )
    end
    return accumulator
end

@inline function _dense_group(name::Symbol)
    if name in (
        :cell_projection, :cell_bias, :cell_position,
        :visual_depthwise, :visual_channel_mix, :visual_pointwise,
        :visual_scale_logit,
        :next_projection, :next_bias, :next_position,
        :aux_value, :aux_position,
    )
        return :token
    elseif name in (:register_seed, :lookup_register_gate)
        return :register
    elseif name in (:ffn_gate, :ffn_up, :ffn_down, :ffn_scale_logit)
        return :ffn
    else
        return :attention
    end
end

@inline _token_column_offset(name::Symbol) =
    name === :cell_position ? 0 :
    name === :next_position ? CELL_COUNT :
    name in (:aux_value, :aux_position) ? CELL_COUNT + NEXT_HOLD_TOKENS : -1

function _adam_update_selected_token_columns!(
    parameter,
    m,
    v,
    gradient,
    active_tokens,
    token_event_count,
    token_offset,
    learning_rate;
    beta1,
    beta2,
    epsilon,
    weight_decay,
)
    decay = 1.0f0 - Float32(learning_rate) * Float32(weight_decay)
    @inbounds for column in axes(parameter, 2)
        token = token_offset + column
        active_tokens[token] || continue
        event = token_event_count[token]
        correction1 = 1.0f0 - Float32(beta1)^event
        correction2 = 1.0f0 - Float32(beta2)^event
        @simd for row in axes(parameter, 1)
            value = gradient[row, column]
            updated_m = muladd(
                Float32(beta1), m[row, column],
                (1.0f0 - Float32(beta1)) * value,
            )
            updated_v = muladd(
                Float32(beta2), v[row, column],
                (1.0f0 - Float32(beta2)) * value * value,
            )
            parameter[row, column] = parameter[row, column] * decay -
                Float32(learning_rate) * (updated_m / correction1) /
                (sqrt(updated_v / correction2) + Float32(epsilon))
            m[row, column] = updated_m
            v[row, column] = updated_v
        end
    end
    return nothing
end

function optimizer_step!(
    model::EpisodicViTLookupModel,
    optimizer::Optimizer,
    accumulator::GradientAccumulator;
    clip_norm=5.0f0,
    beta1=0.9f0,
    beta2=0.999f0,
    epsilon=1.0f-8,
    router_learning_rate=2.0f-4,
    lookup_alpha_learning_rate=1.0f-4,
    head_learning_rate=1.0f-4,
    halt_learning_rate=5.0f-5,
    token_learning_rate=2.0f-4,
    attention_learning_rate=2.0f-4,
    ffn_learning_rate=2.0f-4,
    register_learning_rate=2.0f-4,
    dense_weight_decay=0.0f0,
)
    optimizer.lookup.step == optimizer.dense_step ||
        error("optimizer clocks diverged")
    norm = gradient_norm(accumulator)
    isfinite(norm) || error("gradient norm is non-finite")
    scale = norm > clip_norm ? Float32(clip_norm / norm) : 1.0f0
    scale_gradients!(accumulator, scale)
    sparse_telemetry = SparseLookup.optimizer_step!(
        model.lookup,
        optimizer.lookup,
        accumulator.lookup;
        clip_norm=Inf32,
        beta1,
        beta2,
        epsilon,
        bh4_learning_rate=router_learning_rate,
        alpha_learning_rate=lookup_alpha_learning_rate,
        head_learning_rate,
        halt_learning_rate,
        reinject_learning_rate=register_learning_rate,
        dense_weight_decay,
    )
    next_step = optimizer.lookup.step
    @inbounds for token in eachindex(accumulator.active_tokens)
        accumulator.active_tokens[token] || continue
        optimizer.token_event_count[token] += 1
    end
    parameters = _dense_parameters(model)
    for name in propertynames(parameters)
        parameter = getproperty(parameters, name)
        gradient = accumulator.dense[name]
        m, v = optimizer.dense_states[name]
        group = _dense_group(name)
        learning_rate = group === :token ? token_learning_rate :
            group === :register ? register_learning_rate :
            group === :ffn ? ffn_learning_rate : attention_learning_rate
        token_offset = _token_column_offset(name)
        if token_offset >= 0
            _adam_update_selected_token_columns!(
                parameter,
                m,
                v,
                gradient,
                accumulator.active_tokens,
                optimizer.token_event_count,
                token_offset,
                learning_rate;
                beta1,
                beta2,
                epsilon,
                weight_decay=dense_weight_decay,
            )
        else
            SparseLookup._adam_update!(
                parameter,
                m,
                v,
                gradient,
                next_step,
                learning_rate;
                beta1,
                beta2,
                epsilon,
                weight_decay=dense_weight_decay,
            )
        end
    end
    optimizer.dense_step = next_step
    return (;
        step=Int(next_step),
        gradient_norm=norm,
        gradient_scale=Float64(scale),
        active_columns=sparse_telemetry.active_columns,
        active_elements=sparse_telemetry.active_elements,
    )
end

export BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT, PIECE_TYPES, NEXT_HOLD_TOKENS,
       AUX_FEATURES, TOKEN_COUNT, OUTPUT_DIM, MODEL_DIM, ATTENTION_DIM,
       ATTENTION_HEADS, REGISTER_COUNT, FFN_DIM, INITIAL_HALT_PROBABILITY,
       EPISODIC_ROUTER_TABLES,
       EPISODIC_ROUTER_BITS, EPISODIC_ROUTER_DIM, EPISODIC_BUCKET_CAP,
       EPISODIC_ROOT_CAP, EPISODIC_CANDIDATE_CAP, EPISODIC_SHORTLIST,
       SPATIAL_ANCHORS, SPATIAL_CANDIDATE_CAP, SPATIAL_SHORTLIST,
       MIN_RECURRENT_STEPS,
       MAX_RECURRENT_STEPS, WARMUP_MAX_STEPS, EpisodicCandidateInput,
       EpisodicViTLookupModel, Optimizer, TrajectoryTape, GradientAccumulator,
       RouteUsage, LookupBalanceStats, lookup_balance_stats!, candidate_input,
       initialize_model, initialize_optimizer,
       configure_bank_optimizer!, materialize_selected_columns!, parameter_count,
       topology, usage_summary, record_usage!, ForwardCandidateArena,
       ForwardTrajectoryState,
       prepare_trajectory, advance_trajectory!, finalize_trajectory,
       probe_one_step!, forward_trajectory,
       backward_trajectory!, reset_gradients!, merge_gradients!, gradient_norm,
       scale_gradients!, optimizer_step!

end
