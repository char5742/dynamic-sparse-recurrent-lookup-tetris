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
