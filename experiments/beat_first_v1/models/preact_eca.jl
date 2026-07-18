"""Efficient channel attention with a three-channel-neighbour kernel."""
struct ECAGate <: Lux.AbstractLuxContainerLayer{(:pool, :channel_conv)}
    pool
    channel_conv
end

ECAGate() = ECAGate(
    GlobalMeanPool(),
    Conv((3,), 1 => 1; pad=SamePad(), use_bias=false),
)

function (layer::ECAGate)(x, ps, st)
    pooled, pool_st = layer.pool(x, ps.pool, st.pool)
    channel_signal = reshape(pooled, size(x, 3), 1, size(x, 4))
    gate, conv_st = layer.channel_conv(
        channel_signal, ps.channel_conv, st.channel_conv
    )
    gate = reshape(sigmoid.(gate), 1, 1, size(x, 3), size(x, 4))
    return x .* gate, (; pool=pool_st, channel_conv=conv_st)
end

struct PreActECABlock <: Lux.AbstractLuxContainerLayer{
    (:norm1, :conv1, :norm2, :conv2, :attention)
}
    norm1
    conv1
    norm2
    conv2
    attention
end

function PreActECABlock(channels::Int; groups::Int=8)
    channels % groups == 0 || error("channels must be divisible by groups")
    return PreActECABlock(
        GroupNorm(channels, groups, swish),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        GroupNorm(channels, groups, swish),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        ECAGate(),
    )
end

function (layer::PreActECABlock)(x, ps, st)
    residual, norm1_st = layer.norm1(x, ps.norm1, st.norm1)
    residual, conv1_st = layer.conv1(residual, ps.conv1, st.conv1)
    residual, norm2_st = layer.norm2(residual, ps.norm2, st.norm2)
    residual, conv2_st = layer.conv2(residual, ps.conv2, st.conv2)
    residual, attention_st = layer.attention(
        residual, ps.attention, st.attention
    )
    return x .+ residual, (;
        norm1=norm1_st,
        conv1=conv1_st,
        norm2=norm2_st,
        conv2=conv2_st,
        attention=attention_st,
    )
end

struct PreActECAQ <: Lux.AbstractLuxContainerLayer{
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

function PreActECAQ(;
    channels::Int=96,
    depth::Int=8,
    groups::Int=8,
    queue_features::Int=64,
    aux_features::Int=64,
    hidden_features::Int=256,
    n_quantiles::Int=16,
)
    blocks = Chain((PreActECABlock(channels; groups) for _ in 1:depth)...)
    feature_count = channels + queue_features + aux_features
    return PreActECAQ(
        Conv((3, 3), 3 => channels; pad=SamePad()),
        blocks,
        GroupNorm(channels, groups, swish),
        NextHoldEncoder(; output_features=queue_features),
        Chain(
            Dense(AUX_FEATURES => aux_features, swish),
            Dense(aux_features => aux_features, swish),
        ),
        CandidateHeads(feature_count; hidden_features, n_quantiles),
        channels,
    )
end

function (model::PreActECAQ)(input, ps, st)
    spatial = cat(input.board, input.candidate, input.difference; dims=3)
    value, stem_st = model.stem(spatial, ps.stem, st.stem)
    value, blocks_st = model.blocks(value, ps.blocks, st.blocks)
    value, final_norm_st = model.final_norm(
        value, ps.final_norm, st.final_norm
    )
    pooled = global_mean_2d(value)
    queue, queue_st = model.queue(input.next_hold, ps.queue, st.queue)
    aux, aux_st = model.aux(input.aux, ps.aux, st.aux)
    output, heads_st = model.heads(
        vcat(pooled, queue, aux), ps.heads, st.heads
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

"""Pre-activation ECA bottleneck used by the compute-efficient Medium model.

The 192-channel residual stream retains representation width, while the only
full 3x3 mixing happens in the 48-channel bottleneck. Twelve blocks still give
a 25x25 theoretical receptive field, covering the canonical 24x10 board.
"""
struct PreActECABottleneckBlock <: Lux.AbstractLuxContainerLayer{
    (:norm1, :reduce, :norm2, :spatial, :norm3, :expand, :attention)
}
    norm1
    reduce
    norm2
    spatial
    norm3
    expand
    attention
end

function PreActECABottleneckBlock(
    channels::Int, bottleneck_channels::Int; groups::Int=8
)
    channels % groups == 0 || error("channels must be divisible by groups")
    bottleneck_channels % groups == 0 ||
        error("bottleneck_channels must be divisible by groups")
    return PreActECABottleneckBlock(
        GroupNorm(channels, groups, swish),
        Conv((1, 1), channels => bottleneck_channels; use_bias=false),
        GroupNorm(bottleneck_channels, groups, swish),
        Conv(
            (3, 3), bottleneck_channels => bottleneck_channels;
            pad=SamePad(), use_bias=false,
        ),
        GroupNorm(bottleneck_channels, groups, swish),
        Conv((1, 1), bottleneck_channels => channels; use_bias=false),
        ECAGate(),
    )
end

function (layer::PreActECABottleneckBlock)(x, ps, st)
    residual, norm1_st = layer.norm1(x, ps.norm1, st.norm1)
    residual, reduce_st = layer.reduce(residual, ps.reduce, st.reduce)
    residual, norm2_st = layer.norm2(residual, ps.norm2, st.norm2)
    residual, spatial_st = layer.spatial(residual, ps.spatial, st.spatial)
    residual, norm3_st = layer.norm3(residual, ps.norm3, st.norm3)
    residual, expand_st = layer.expand(residual, ps.expand, st.expand)
    residual, attention_st = layer.attention(
        residual, ps.attention, st.attention
    )
    return x .+ residual, (;
        norm1=norm1_st,
        reduce=reduce_st,
        norm2=norm2_st,
        spatial=spatial_st,
        norm3=norm3_st,
        expand=expand_st,
        attention=attention_st,
    )
end

struct EfficientPreActECAQ <: Lux.AbstractLuxContainerLayer{
    (:stem, :blocks, :final_norm, :grid_projection, :queue, :aux, :heads)
}
    stem
    blocks
    final_norm
    grid_projection
    queue
    aux
    heads
    channels::Int
end

function EfficientPreActECAQ(;
    channels::Int=192,
    depth::Int=12,
    bottleneck_channels::Int=48,
    groups::Int=8,
    grid_features::Int=512,
    queue_features::Int=96,
    aux_features::Int=96,
    hidden_features::Int=1536,
    n_quantiles::Int=16,
)
    blocks = Chain(
        (
            PreActECABottleneckBlock(
                channels, bottleneck_channels; groups
            ) for _ in 1:depth
        )...,
    )
    head_features = grid_features + queue_features + aux_features
    return EfficientPreActECAQ(
        Conv((3, 3), 3 => channels; pad=SamePad()),
        blocks,
        GroupNorm(channels, groups, swish),
        Dense(20 * channels => grid_features, swish),
        NextHoldEncoder(; output_features=queue_features),
        Chain(
            Dense(AUX_FEATURES => aux_features, swish),
            Dense(aux_features => aux_features, swish),
        ),
        CandidateHeads(head_features; hidden_features, n_quantiles),
        channels,
    )
end

function (model::EfficientPreActECAQ)(input, ps, st)
    spatial = cat(input.board, input.candidate, input.difference; dims=3)
    value, stem_st = model.stem(spatial, ps.stem, st.stem)
    value, blocks_st = model.blocks(value, ps.blocks, st.blocks)
    value, final_norm_st = model.final_norm(
        value, ps.final_norm, st.final_norm
    )
    grid, grid_projection_st = model.grid_projection(
        fixed_grid_mean_4x5(value),
        ps.grid_projection,
        st.grid_projection,
    )
    queue, queue_st = model.queue(input.next_hold, ps.queue, st.queue)
    aux, aux_st = model.aux(input.aux, ps.aux, st.aux)
    output, heads_st = model.heads(vcat(grid, queue, aux), ps.heads, st.heads)
    return output, (;
        stem=stem_st,
        blocks=blocks_st,
        final_norm=final_norm_st,
        grid_projection=grid_projection_st,
        queue=queue_st,
        aux=aux_st,
        heads=heads_st,
    )
end
