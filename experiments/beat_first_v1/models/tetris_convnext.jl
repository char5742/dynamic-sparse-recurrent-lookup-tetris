struct TetrisConvNeXtBlock <: Lux.AbstractLuxContainerLayer{
    (:depthwise, :norm, :expand, :project)
}
    depthwise
    norm
    expand
    project
end

function TetrisConvNeXtBlock(channels::Int; expansion::Int=4)
    hidden = channels * expansion
    return TetrisConvNeXtBlock(
        Conv(
            (5, 3), channels => channels;
            pad=SamePad(), groups=channels, use_bias=false,
        ),
        GroupNorm(channels, 1),
        Conv((1, 1), channels => hidden, gelu),
        Conv((1, 1), hidden => channels),
    )
end

function (layer::TetrisConvNeXtBlock)(x, ps, st)
    residual, depthwise_st = layer.depthwise(
        x, ps.depthwise, st.depthwise
    )
    residual, norm_st = layer.norm(residual, ps.norm, st.norm)
    residual, expand_st = layer.expand(residual, ps.expand, st.expand)
    residual, project_st = layer.project(residual, ps.project, st.project)
    return x .+ residual, (;
        depthwise=depthwise_st,
        norm=norm_st,
        expand=expand_st,
        project=project_st,
    )
end

struct TetrisConvNeXtQ <: Lux.AbstractLuxContainerLayer{
    (:stem, :blocks, :final_norm, :queue, :aux, :heads)
}
    stem
    blocks
    final_norm
    queue
    aux
    heads
    channels::Int
end

function TetrisConvNeXtQ(;
    channels::Int=112,
    depth::Int=10,
    expansion::Int=4,
    queue_features::Int=64,
    aux_features::Int=64,
    hidden_features::Int=256,
    n_quantiles::Int=16,
)
    blocks = Chain(
        (TetrisConvNeXtBlock(channels; expansion) for _ in 1:depth)...
    )
    feature_count = 2 * channels + queue_features + aux_features
    return TetrisConvNeXtQ(
        Conv((3, 3), 3 => channels; pad=SamePad()),
        blocks,
        GroupNorm(channels, 1, gelu),
        NextHoldEncoder(; output_features=queue_features),
        Chain(
            Dense(AUX_FEATURES => aux_features, gelu),
            Dense(aux_features => aux_features, gelu),
        ),
        CandidateHeads(feature_count; hidden_features, n_quantiles),
        channels,
    )
end

function (model::TetrisConvNeXtQ)(input, ps, st)
    spatial = cat(input.board, input.candidate, input.difference; dims=3)
    value, stem_st = model.stem(spatial, ps.stem, st.stem)
    value, blocks_st = model.blocks(value, ps.blocks, st.blocks)
    value, final_norm_st = model.final_norm(
        value, ps.final_norm, st.final_norm
    )
    global_features = global_mean_2d(value)
    local_features = local_mean_2d(value, input.local_mask)
    queue, queue_st = model.queue(input.next_hold, ps.queue, st.queue)
    aux, aux_st = model.aux(input.aux, ps.aux, st.aux)
    output, heads_st = model.heads(
        vcat(global_features, local_features, queue, aux),
        ps.heads,
        st.heads,
    )
    return output, (;
        stem=stem_st,
        blocks=blocks_st,
        final_norm=final_norm_st,
        queue=queue_st,
        aux=aux_st,
        heads=heads_st,
    )
end
