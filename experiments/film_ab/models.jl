using Lux
using Random

const FILM_SWISH = swish

zero_init(rng::AbstractRNG, dims...) = zeros(Float32, dims...)

struct PlainResidualBlock <: Lux.AbstractLuxContainerLayer{(:conv1, :conv2)}
    conv1
    conv2
end

PlainResidualBlock(channels::Int) = PlainResidualBlock(
    Conv((3, 3), channels => channels; pad=SamePad()),
    Conv((3, 3), channels => channels; pad=SamePad()),
)

function (layer::PlainResidualBlock)(x, ps, st)
    y, s1 = layer.conv1(x, ps.conv1, st.conv1)
    y = FILM_SWISH.(y)
    y, s2 = layer.conv2(y, ps.conv2, st.conv2)
    return FILM_SWISH.(x .+ y), (; conv1=s1, conv2=s2)
end

"""The plain block plus a queue-conditioned affine transform.

The transform is applied after the first convolution/nonlinearity.  Its Dense
layer is zero initialized, so the FiLM network is exactly identical to the
plain network at update zero when the shared parameters are copied.
"""
struct FiLMResidualBlock <: Lux.AbstractLuxContainerLayer{
    (:conv1, :conv2, :condition)
}
    conv1
    conv2
    condition
    channels::Int
end


function FiLMResidualBlock(channels::Int, condition_features::Int)
    return FiLMResidualBlock(
        Conv((3, 3), channels => channels; pad=SamePad()),
        Conv((3, 3), channels => channels; pad=SamePad()),
        Dense(
            condition_features => 2 * channels;
            init_weight=zero_init,
            init_bias=zero_init,
        ),
        channels,
    )
end

function (layer::FiLMResidualBlock)((x, condition), ps, st)
    y, s1 = layer.conv1(x, ps.conv1, st.conv1)
    y = FILM_SWISH.(y)
    gamma_beta, s3 = layer.condition(condition, ps.condition, st.condition)
    gamma = reshape(
        @view(gamma_beta[1:layer.channels, :]), 1, 1, layer.channels, :
    )
    beta = reshape(
        @view(gamma_beta[(layer.channels + 1):(2 * layer.channels), :]),
        1,
        1,
        layer.channels,
        :,
    )
    # Bounded residual modulation keeps the initially identical policy stable.
    y = y .* (1.0f0 .+ 0.1f0 .* tanh.(gamma)) .+
        0.1f0 .* tanh.(beta)
    y, s2 = layer.conv2(y, ps.conv2, st.conv2)
    return FILM_SWISH.(x .+ y),
        (; conv1=s1, conv2=s2, condition=s3)
end

function queue_encoder()
    return Chain(Dense(42 => 64, FILM_SWISH), Dense(64 => 64, FILM_SWISH))
end

function candidate_head(; spatial_channels::Int=2)
    features = 24 * 10 * spatial_channels + 64 + 3
    return Chain(
        Dense(features => 256, FILM_SWISH),
        Dense(256 => 64, FILM_SWISH),
        Dense(64 => 1),
    )
end

struct PlainCompactCandidateQ <: Lux.AbstractLuxContainerLayer{
    (:stem, :block, :projection, :queue_encoder, :head)
}
    stem
    block
    projection
    queue_encoder
    head
end

function PlainCompactCandidateQ(; channels::Int=8, spatial_channels::Int=2)
    return PlainCompactCandidateQ(
        Chain(
            Conv((3, 3), 2 => channels; pad=SamePad()),
            WrappedFunction(FILM_SWISH),
        ),
        PlainResidualBlock(channels),
        Chain(
            Conv((1, 1), channels => spatial_channels),
            WrappedFunction(FILM_SWISH),
        ),
        queue_encoder(),
        candidate_head(; spatial_channels),
    )
end

struct FiLMCompactCandidateQ <: Lux.AbstractLuxContainerLayer{
    (:stem, :block, :projection, :queue_encoder, :head)
}
    stem
    block
    projection
    queue_encoder
    head
end


function FiLMCompactCandidateQ(; channels::Int=8, spatial_channels::Int=2)
    return FiLMCompactCandidateQ(
        Chain(
            Conv((3, 3), 2 => channels; pad=SamePad()),
            WrappedFunction(FILM_SWISH),
        ),
        FiLMResidualBlock(channels, 64),
        Chain(
            Conv((1, 1), channels => spatial_channels),
            WrappedFunction(FILM_SWISH),
        ),
        queue_encoder(),
        candidate_head(; spatial_channels),
    )
end

function candidate_inputs(input)
    board, placement, ren, back_to_back, tspin, queue = input
    spatial = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    queue_flat = reshape(queue, :, size(queue, 3))
    scalars = (ren ./ 30.0f0, back_to_back, tspin)
    return spatial, queue_flat, scalars
end

function (model::PlainCompactCandidateQ)(input, ps, st)
    spatial, queue_flat, scalars = candidate_inputs(input)
    x, s1 = model.stem(spatial, ps.stem, st.stem)
    x, s2 = model.block(x, ps.block, st.block)
    x, s3 = model.projection(x, ps.projection, st.projection)
    x = reshape(x, :, size(x, 4))
    q, s4 = model.queue_encoder(queue_flat, ps.queue_encoder, st.queue_encoder)
    y, s5 = model.head(vcat(x, q, scalars...), ps.head, st.head)
    return y, (;
        stem=s1,
        block=s2,
        projection=s3,
        queue_encoder=s4,
        head=s5,
    )
end

function (model::FiLMCompactCandidateQ)(input, ps, st)
    spatial, queue_flat, scalars = candidate_inputs(input)
    x, s1 = model.stem(spatial, ps.stem, st.stem)
    q, s4 = model.queue_encoder(queue_flat, ps.queue_encoder, st.queue_encoder)
    x, s2 = model.block((x, q), ps.block, st.block)
    x, s3 = model.projection(x, ps.projection, st.projection)
    x = reshape(x, :, size(x, 4))
    y, s5 = model.head(vcat(x, q, scalars...), ps.head, st.head)
    return y, (;
        stem=s1,
        block=s2,
        projection=s3,
        queue_encoder=s4,
        head=s5,
    )
end

"""Set every shared parameter/state in FiLM to the plain initialization.

Only `block.condition` remains FiLM-specific (and is zero initialized).  This
guarantees equal update-zero outputs and removes initialization luck from A/B.
"""
function matched_initialization(seed::UInt64)
    plain = PlainCompactCandidateQ()
    film = FiLMCompactCandidateQ()
    plain_ps, plain_st = Lux.setup(Xoshiro(seed), plain)
    film_ps0, film_st0 = Lux.setup(Xoshiro(seed), film)
    film_ps = (;
        stem=deepcopy(plain_ps.stem),
        block=(;
            conv1=deepcopy(plain_ps.block.conv1),
            conv2=deepcopy(plain_ps.block.conv2),
            condition=film_ps0.block.condition,
        ),
        projection=deepcopy(plain_ps.projection),
        queue_encoder=deepcopy(plain_ps.queue_encoder),
        head=deepcopy(plain_ps.head),
    )
    film_st = (;
        stem=deepcopy(plain_st.stem),
        block=(;
            conv1=deepcopy(plain_st.block.conv1),
            conv2=deepcopy(plain_st.block.conv2),
            condition=film_st0.block.condition,
        ),
        projection=deepcopy(plain_st.projection),
        queue_encoder=deepcopy(plain_st.queue_encoder),
        head=deepcopy(plain_st.head),
    )
    return (; plain, plain_ps, plain_st, film, film_ps, film_st)
end

function analytical_macs(kind::Symbol; channels::Int=8, spatial_channels::Int=2)
    spatial = 24 * 10
    stem = spatial * 3 * 3 * 2 * channels
    residual = 2 * spatial * 3 * 3 * channels * channels
    projection = spatial * channels * spatial_channels
    queue = 42 * 64 + 64 * 64
    head_features = spatial * spatial_channels + 64 + 3
    head = head_features * 256 + 256 * 64 + 64
    conditioning = kind === :film ? 64 * (2 * channels) : 0
    return stem + residual + projection + queue + head + conditioning
end
