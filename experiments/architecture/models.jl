using Lux
using Random

const ARCH_SWISH = swish

struct SEGate <: Lux.AbstractLuxContainerLayer{(:pool, :reduce, :expand)}
    pool
    reduce
    expand
end

function SEGate(channels::Int; ratio::Int=8)
    hidden = max(8, channels ÷ ratio)
    return SEGate(
        GlobalMeanPool(),
        Conv((1, 1), channels => hidden, ARCH_SWISH),
        Conv((1, 1), hidden => channels, sigmoid),
    )
end

function (layer::SEGate)(x, ps, st)
    gate, pool_st = layer.pool(x, ps.pool, st.pool)
    gate, reduce_st = layer.reduce(gate, ps.reduce, st.reduce)
    gate, expand_st = layer.expand(gate, ps.expand, st.expand)
    return x .* gate, (; pool=pool_st, reduce=reduce_st, expand=expand_st)
end

struct PreActSEBlock <: Lux.AbstractLuxContainerLayer{
    (:norm1, :conv1, :norm2, :conv2, :gate)
}
    norm1
    conv1
    norm2
    conv2
    gate
end

function PreActSEBlock(channels::Int)
    return PreActSEBlock(
        GroupNorm(channels, 8, ARCH_SWISH),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        GroupNorm(channels, 8, ARCH_SWISH),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        SEGate(channels),
    )
end

function (layer::PreActSEBlock)(x, ps, st)
    y, s1 = layer.norm1(x, ps.norm1, st.norm1)
    y, s2 = layer.conv1(y, ps.conv1, st.conv1)
    y, s3 = layer.norm2(y, ps.norm2, st.norm2)
    y, s4 = layer.conv2(y, ps.conv2, st.conv2)
    y, s5 = layer.gate(y, ps.gate, st.gate)
    return x .+ y, (; norm1=s1, conv1=s2, norm2=s3, conv2=s4, gate=s5)
end

struct ConvNeXtBlock <: Lux.AbstractLuxContainerLayer{
    (:depthwise, :norm, :expand, :project)
}
    depthwise
    norm
    expand
    project
end


function ConvNeXtBlock(channels::Int; expansion::Int=4)
    return ConvNeXtBlock(
        Conv(
            (5, 5), channels => channels;
            pad=SamePad(), groups=channels, use_bias=false,
        ),
        GroupNorm(channels, 1),
        Conv((1, 1), channels => expansion * channels, gelu),
        Conv((1, 1), expansion * channels => channels),
    )
end

function (layer::ConvNeXtBlock)(x, ps, st)
    y, s1 = layer.depthwise(x, ps.depthwise, st.depthwise)
    y, s2 = layer.norm(y, ps.norm, st.norm)
    y, s3 = layer.expand(y, ps.expand, st.expand)
    y, s4 = layer.project(y, ps.project, st.project)
    return x .+ y, (; depthwise=s1, norm=s2, expand=s3, project=s4)
end

struct FiLMResidualBlock <: Lux.AbstractLuxContainerLayer{
    (:norm1, :conv1, :norm2, :conv2, :condition)
}
    norm1
    conv1
    norm2
    conv2
    condition
    channels::Int
end

function FiLMResidualBlock(channels::Int, condition_features::Int)
    return FiLMResidualBlock(
        GroupNorm(channels, 8, ARCH_SWISH),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        GroupNorm(channels, 8, ARCH_SWISH),
        Conv((3, 3), channels => channels; pad=SamePad(), use_bias=false),
        Dense(condition_features => 2 * channels),
        channels,
    )
end

function (layer::FiLMResidualBlock)((x, condition), ps, st)
    y, s1 = layer.norm1(x, ps.norm1, st.norm1)
    y, s2 = layer.conv1(y, ps.conv1, st.conv1)
    gamma_beta, s5 = layer.condition(condition, ps.condition, st.condition)
    gamma = reshape(gamma_beta[1:layer.channels, :], 1, 1, layer.channels, :)
    beta = reshape(
        gamma_beta[(layer.channels + 1):(2 * layer.channels), :],
        1, 1, layer.channels, :,
    )
    y = y .* (1.0f0 .+ 0.1f0 .* tanh.(gamma)) .+ beta
    y, s3 = layer.norm2(y, ps.norm2, st.norm2)
    y, s4 = layer.conv2(y, ps.conv2, st.conv2)
    return (x .+ y, condition),
        (; norm1=s1, conv1=s2, norm2=s3, conv2=s4, condition=s5)
end

struct QueueEncoder <: Lux.AbstractLuxContainerLayer{(:token, :summary)}
    token
    summary
end

function QueueEncoder(features::Int=64)
    return QueueEncoder(
        Dense(7 => features, gelu),
        Dense(6 * features => features, gelu),
    )
end

function (layer::QueueEncoder)(queue, ps, st)
    y, s1 = layer.token(queue, ps.token, st.token)
    y = reshape(y, :, size(y, 3))
    y, s2 = layer.summary(y, ps.summary, st.summary)
    return y, (; token=s1, summary=s2)
end

struct PlainValueNet <: Lux.AbstractLuxContainerLayer{
    (:stem, :blocks, :compress, :queue, :head)
}
    stem
    blocks
    compress
    queue
    head
    channels::Int
end

function PlainValueNet(block_factory; channels::Int=64, depth::Int=6)
    blocks = Chain([block_factory(channels) for _ in 1:depth]...)
    queue_features = 64
    compressed_channels = 4
    head_features = 24 * 10 * compressed_channels + queue_features + 3
    return PlainValueNet(
        Conv((3, 3), 2 => channels, ARCH_SWISH; pad=SamePad()),
        blocks,
        Conv((1, 1), channels => compressed_channels, ARCH_SWISH),
        QueueEncoder(queue_features),
        Chain(
            Dense(head_features => 256, ARCH_SWISH),
            Dense(256 => 64, ARCH_SWISH),
            Dense(64 => 1),
        ),
        channels,
    )
end

function (layer::PlainValueNet)((board, queue, scalars), ps, st)
    x, s1 = layer.stem(board, ps.stem, st.stem)
    x, s2 = layer.blocks(x, ps.blocks, st.blocks)
    x, s3 = layer.compress(x, ps.compress, st.compress)
    x = reshape(x, :, size(x, 4))
    q, s4 = layer.queue(queue, ps.queue, st.queue)
    y, s5 = layer.head(vcat(x, q, scalars), ps.head, st.head)
    return y, (; stem=s1, blocks=s2, compress=s3, queue=s4, head=s5)
end

PreActSEValueNet(; channels::Int=64, depth::Int=6) =
    PlainValueNet(PreActSEBlock; channels, depth)

ConvNeXtValueNet(; channels::Int=64, depth::Int=8) =
    PlainValueNet(ConvNeXtBlock; channels, depth)

struct FiLMValueNet <: Lux.AbstractLuxContainerLayer{
    (:stem, :queue, :blocks, :compress, :head)
}
    stem
    queue
    blocks
    compress
    head
    channels::Int
end

function FiLMValueNet(; channels::Int=64, depth::Int=6)
    queue_features = 64
    compressed_channels = 4
    blocks = Chain(
        [FiLMResidualBlock(channels, queue_features) for _ in 1:depth]...
    )
    head_features = 24 * 10 * compressed_channels + queue_features + 3
    return FiLMValueNet(
        Conv((3, 3), 2 => channels, ARCH_SWISH; pad=SamePad()),
        QueueEncoder(queue_features),
        blocks,
        Conv((1, 1), channels => compressed_channels, ARCH_SWISH),
        Chain(
            Dense(head_features => 256, ARCH_SWISH),
            Dense(256 => 64, ARCH_SWISH),
            Dense(64 => 1),
        ),
        channels,
    )
end

function (layer::FiLMValueNet)((board, queue, scalars), ps, st)
    x, s1 = layer.stem(board, ps.stem, st.stem)
    q, s2 = layer.queue(queue, ps.queue, st.queue)
    (x, _), s3 = layer.blocks((x, q), ps.blocks, st.blocks)
    x, s4 = layer.compress(x, ps.compress, st.compress)
    x = reshape(x, :, size(x, 4))
    y, s5 = layer.head(vcat(x, q, scalars), ps.head, st.head)
    return y,
        (; stem=s1, queue=s2, blocks=s3, compress=s4, head=s5)
end

function analytical_macs(name::Symbol, batch::Int)
    h, w, c = 24, 10, 64
    stem = h * w * 3 * 3 * 2 * c
    queue = 6 * 7 * 64 + (6 * 64) * 64
    compress = h * w * c * 4
    head = (h * w * 4 + 64 + 3) * 256 + 256 * 64 + 64
    if name === :preact_se
        per_block = 2 * h * w * 3 * 3 * c * c + 2 * c * (c ÷ 8)
        depth = 6
    elseif name === :convnext
        per_block = h * w * (5 * 5 * c + 2 * c * (4 * c))
        depth = 8
    elseif name === :film
        per_block = 2 * h * w * 3 * 3 * c * c + 64 * 2 * c
        depth = 6
    else
        error("unknown architecture")
    end
    return batch * (stem + depth * per_block + queue + compress + head)
end

function analytical_activation_bytes(name::Symbol, batch::Int)
    # Conservative forward major-activation estimate.  Training frameworks may
    # retain additional temporaries, so measured allocations are also reported.
    h, w, c = 24, 10, 64
    depth = name === :convnext ? 8 : 6
    expansion = name === :convnext ? 4 : 1
    spatial_floats = h * w * c * (1 + depth * (2 + expansion))
    queue_head_floats = 6 * 64 + 64 + 24 * 10 * 4 + 256 + 64
    return 4 * batch * (spatial_floats + queue_head_floats)
end
