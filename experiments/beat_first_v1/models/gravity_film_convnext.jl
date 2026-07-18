struct GravityFiLMBlock <: Lux.AbstractLuxContainerLayer{
    (:depthwise, :norm, :condition, :expand, :project)
}
    depthwise
    norm
    condition
    expand
    project
    channels::Int
end

function GravityFiLMBlock(
    channels::Int, condition_features::Int; expansion::Int=4
)
    hidden = channels * expansion
    return GravityFiLMBlock(
        Conv(
            (7, 3), channels => channels;
            pad=SamePad(), groups=channels, use_bias=false,
        ),
        GroupNorm(channels, 1),
        Dense(condition_features => 2 * channels),
        Conv((1, 1), channels => hidden, gelu),
        Conv((1, 1), hidden => channels),
        channels,
    )
end

function (layer::GravityFiLMBlock)((x, condition), ps, st)
    residual, depthwise_st = layer.depthwise(
        x, ps.depthwise, st.depthwise
    )
    residual, norm_st = layer.norm(residual, ps.norm, st.norm)
    modulation, condition_st = layer.condition(
        condition, ps.condition, st.condition
    )
    gamma = reshape(
        @view(modulation[1:layer.channels, :]),
        1,
        1,
        layer.channels,
        :,
    )
    beta = reshape(
        @view(modulation[(layer.channels + 1):(2 * layer.channels), :]),
        1,
        1,
        layer.channels,
        :,
    )
    # Bounded modulation keeps early imitation training well conditioned while
    # still injecting HOLD/NEXT and board geometry at every block.
    residual = residual .* (1.0f0 .+ 0.25f0 .* tanh.(gamma)) .+
               0.25f0 .* tanh.(beta)
    residual, expand_st = layer.expand(residual, ps.expand, st.expand)
    residual, project_st = layer.project(residual, ps.project, st.project)
    return (x .+ residual, condition), (;
        depthwise=depthwise_st,
        norm=norm_st,
        condition=condition_st,
        expand=expand_st,
        project=project_st,
    )
end

struct GravityFiLMConvNeXtQ <: Lux.AbstractLuxContainerLayer{
    (
        :board_stem,
        :candidate_stem,
        :difference_stem,
        :fusion,
        :queue,
        :aux,
        :condition,
        :blocks,
        :final_norm,
        :global_projection,
        :local_projection,
        :heads,
    )
}
    board_stem
    candidate_stem
    difference_stem
    fusion
    queue
    aux
    condition
    blocks
    final_norm
    global_projection
    local_projection
    heads
    channels::Int
end

function GravityFiLMConvNeXtQ(;
    channels::Int=112,
    depth::Int=10,
    expansion::Int=4,
    branch_channels::Int=32,
    queue_features::Int=64,
    aux_features::Int=64,
    condition_features::Int=128,
    pooled_features::Int=96,
    hidden_features::Int=320,
    n_quantiles::Int=16,
    geometry_heads::Bool=false,
)
    queue_features + aux_features == condition_features || error(
        "condition_features must equal queue_features + aux_features"
    )
    blocks = Chain(
        (
            GravityFiLMBlock(channels, condition_features; expansion)
            for _ in 1:depth
        )...,
    )
    head_features = 2 * pooled_features + branch_channels + condition_features
    return GravityFiLMConvNeXtQ(
        Conv((5, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv((5, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv((7, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv(
            (1, 1), 3 * branch_channels => channels, gelu;
            use_bias=false,
        ),
        NextHoldEncoder(; output_features=queue_features),
        Chain(
            Dense(AUX_FEATURES => aux_features, gelu),
            Dense(aux_features => aux_features, gelu),
        ),
        Dense(condition_features => condition_features, gelu),
        blocks,
        GroupNorm(channels, 1, gelu),
        Conv((1, 1), channels => pooled_features, gelu),
        Conv((1, 1), channels => pooled_features, gelu),
        make_candidate_heads(
            head_features; hidden_features, n_quantiles, geometry_heads,
        ),
        channels,
    )
end

function (model::GravityFiLMConvNeXtQ)(input, ps, st)
    board, board_st = model.board_stem(
        input.board, ps.board_stem, st.board_stem
    )
    candidate, candidate_st = model.candidate_stem(
        input.candidate, ps.candidate_stem, st.candidate_stem
    )
    difference, difference_st = model.difference_stem(
        input.difference, ps.difference_stem, st.difference_stem
    )
    value, fusion_st = model.fusion(
        cat(board, candidate, difference; dims=3), ps.fusion, st.fusion
    )

    queue, queue_st = model.queue(input.next_hold, ps.queue, st.queue)
    aux, aux_st = model.aux(input.aux, ps.aux, st.aux)
    condition, condition_st = model.condition(
        vcat(queue, aux), ps.condition, st.condition
    )
    (value, _), blocks_st = model.blocks(
        (value, condition), ps.blocks, st.blocks
    )
    value, final_norm_st = model.final_norm(
        value, ps.final_norm, st.final_norm
    )

    global_value, global_projection_st = model.global_projection(
        value, ps.global_projection, st.global_projection
    )
    local_value, local_projection_st = model.local_projection(
        value, ps.local_projection, st.local_projection
    )
    global_features = global_mean_2d(global_value)
    local_features = local_mean_2d(local_value, input.local_mask)
    # The pre-action board branch is retained separately so global board
    # stability is not forced through the candidate/difference fusion alone.
    board_features = global_mean_2d(board)
    output, heads_st = model.heads(
        vcat(global_features, local_features, board_features, condition),
        ps.heads,
        st.heads,
    )
    return output, (;
        board_stem=board_st,
        candidate_stem=candidate_st,
        difference_stem=difference_st,
        fusion=fusion_st,
        queue=queue_st,
        aux=aux_st,
        condition=condition_st,
        blocks=blocks_st,
        final_norm=final_norm_st,
        global_projection=global_projection_st,
        local_projection=local_projection_st,
        heads=heads_st,
    )
end

"""Compute-efficient Gravity-FiLM block with a low-rank pointwise MLP.

The legacy block above is intentionally preserved byte-for-byte in structure
for old checkpoint loading. This v2 block keeps vertical depthwise mixing and
per-block NEXT/HOLD FiLM, but removes the expensive 4x spatial expansion.
"""
struct EfficientGravityFiLMBlock <: Lux.AbstractLuxContainerLayer{
    (:depthwise, :norm, :condition, :reduce, :project)
}
    depthwise
    norm
    condition
    reduce
    project
    channels::Int
end

function EfficientGravityFiLMBlock(
    channels::Int,
    condition_features::Int;
    pointwise_features::Int,
)
    return EfficientGravityFiLMBlock(
        Conv(
            (7, 3), channels => channels;
            pad=SamePad(), groups=channels, use_bias=false,
        ),
        GroupNorm(channels, 1),
        Dense(condition_features => 2 * channels),
        Conv((1, 1), channels => pointwise_features, gelu),
        Conv((1, 1), pointwise_features => channels),
        channels,
    )
end

function (layer::EfficientGravityFiLMBlock)((x, condition), ps, st)
    residual, depthwise_st = layer.depthwise(
        x, ps.depthwise, st.depthwise
    )
    residual, norm_st = layer.norm(residual, ps.norm, st.norm)
    modulation, condition_st = layer.condition(
        condition, ps.condition, st.condition
    )
    gamma = reshape(
        @view(modulation[1:layer.channels, :]),
        1,
        1,
        layer.channels,
        :,
    )
    beta = reshape(
        @view(modulation[(layer.channels + 1):(2 * layer.channels), :]),
        1,
        1,
        layer.channels,
        :,
    )
    residual = residual .* (1.0f0 .+ 0.25f0 .* tanh.(gamma)) .+
               0.25f0 .* tanh.(beta)
    residual, reduce_st = layer.reduce(residual, ps.reduce, st.reduce)
    residual, project_st = layer.project(residual, ps.project, st.project)
    return (x .+ residual, condition), (;
        depthwise=depthwise_st,
        norm=norm_st,
        condition=condition_st,
        reduce=reduce_st,
        project=project_st,
    )
end

struct EfficientGravityFiLMConvNeXtQ <: Lux.AbstractLuxContainerLayer{
    (
        :board_stem,
        :candidate_stem,
        :difference_stem,
        :fusion,
        :queue,
        :aux,
        :condition,
        :blocks,
        :final_norm,
        :grid_projection,
        :local_projection,
        :heads,
    )
}
    board_stem
    candidate_stem
    difference_stem
    fusion
    queue
    aux
    condition
    blocks
    final_norm
    grid_projection
    local_projection
    heads
    channels::Int
end

function EfficientGravityFiLMConvNeXtQ(;
    channels::Int=112,
    depth::Int=10,
    pointwise_features::Int=48,
    branch_channels::Int=32,
    queue_features::Int=64,
    aux_features::Int=64,
    condition_features::Int=128,
    grid_features::Int=256,
    local_features::Int=96,
    hidden_features::Int=704,
    n_quantiles::Int=16,
    geometry_heads::Bool=false,
)
    queue_features + aux_features == condition_features || error(
        "condition_features must equal queue_features + aux_features"
    )
    blocks = Chain(
        (
            EfficientGravityFiLMBlock(
                channels,
                condition_features;
                pointwise_features,
            ) for _ in 1:depth
        )...,
    )
    head_features =
        grid_features + local_features + branch_channels + condition_features
    return EfficientGravityFiLMConvNeXtQ(
        Conv((5, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv((5, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv((7, 3), 1 => branch_channels; pad=SamePad(), use_bias=false),
        Conv(
            (1, 1), 3 * branch_channels => channels, gelu;
            use_bias=false,
        ),
        NextHoldEncoder(; output_features=queue_features),
        Chain(
            Dense(AUX_FEATURES => aux_features, gelu),
            Dense(aux_features => aux_features, gelu),
        ),
        Dense(condition_features => condition_features, gelu),
        blocks,
        GroupNorm(channels, 1, gelu),
        Dense(20 * channels => grid_features, gelu),
        Dense(channels => local_features, gelu),
        make_candidate_heads(
            head_features; hidden_features, n_quantiles, geometry_heads,
        ),
        channels,
    )
end

function (model::EfficientGravityFiLMConvNeXtQ)(input, ps, st)
    board, board_st = model.board_stem(
        input.board, ps.board_stem, st.board_stem
    )
    candidate, candidate_st = model.candidate_stem(
        input.candidate, ps.candidate_stem, st.candidate_stem
    )
    difference, difference_st = model.difference_stem(
        input.difference, ps.difference_stem, st.difference_stem
    )
    value, fusion_st = model.fusion(
        cat(board, candidate, difference; dims=3), ps.fusion, st.fusion
    )

    queue, queue_st = model.queue(input.next_hold, ps.queue, st.queue)
    aux, aux_st = model.aux(input.aux, ps.aux, st.aux)
    condition, condition_st = model.condition(
        vcat(queue, aux), ps.condition, st.condition
    )
    (value, _), blocks_st = model.blocks(
        (value, condition), ps.blocks, st.blocks
    )
    value, final_norm_st = model.final_norm(
        value, ps.final_norm, st.final_norm
    )

    grid, grid_projection_st = model.grid_projection(
        fixed_grid_mean_4x5(value),
        ps.grid_projection,
        st.grid_projection,
    )
    local_features, local_projection_st = model.local_projection(
        local_mean_2d(value, input.local_mask),
        ps.local_projection,
        st.local_projection,
    )
    board_features = global_mean_2d(board)
    output, heads_st = model.heads(
        vcat(grid, local_features, board_features, condition),
        ps.heads,
        st.heads,
    )
    return output, (;
        board_stem=board_st,
        candidate_stem=candidate_st,
        difference_stem=difference_st,
        fusion=fusion_st,
        queue=queue_st,
        aux=aux_st,
        condition=condition_st,
        blocks=blocks_st,
        final_norm=final_norm_st,
        grid_projection=grid_projection_st,
        local_projection=local_projection_st,
        heads=heads_st,
    )
end
