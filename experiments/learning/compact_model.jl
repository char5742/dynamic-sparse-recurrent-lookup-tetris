using Lux

"""Small end-to-end candidate value model; no historical features are frozen."""
struct CompactCandidateQ <: Lux.AbstractLuxContainerLayer{
    (:stem, :trunk, :projection, :queue_encoder, :head)
}
    stem
    trunk
    projection
    queue_encoder
    head
end

function compact_residual_block(channels::Int)
    residual = Chain(
        Conv((3, 3), channels => channels; pad=SamePad()),
        WrappedFunction(swish),
        Conv((3, 3), channels => channels; pad=SamePad()),
    )
    return Chain(SkipConnection(residual, +), WrappedFunction(swish))
end

function CompactCandidateQ(; channels::Int=32, blocks::Int=4, spatial_channels::Int=4)
    return CompactCandidateQ(
        Chain(
            Conv((3, 3), 2 => channels; pad=SamePad()),
            WrappedFunction(swish),
        ),
        Chain((compact_residual_block(channels) for _ in 1:blocks)...),
        Chain(
            Conv((1, 1), channels => spatial_channels),
            WrappedFunction(swish),
        ),
        Chain(Dense(42 => 64, swish), Dense(64 => 64, swish)),
        Chain(
            Dense(24 * 10 * spatial_channels + 64 + 3 => 256, swish),
            Dense(256 => 64, swish),
            Dense(64 => 1),
        ),
    )
end

function (model::CompactCandidateQ)(input, parameters, state)
    board, placement, ren, back_to_back, tspin, queue = input
    value = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    value, stem_state = model.stem(value, parameters.stem, state.stem)
    value, trunk_state = model.trunk(value, parameters.trunk, state.trunk)
    value, projection_state = model.projection(
        value, parameters.projection, state.projection
    )
    value = reshape(value, :, size(value, 4))
    queue_value = reshape(queue, :, size(queue, 3))
    queue_value, queue_state = model.queue_encoder(
        queue_value, parameters.queue_encoder, state.queue_encoder
    )
    combined = vcat(value, queue_value, ren ./ 30.0f0, back_to_back, tspin)
    score, head_state = model.head(combined, parameters.head, state.head)
    return score, (;
        stem=stem_state,
        trunk=trunk_state,
        projection=projection_state,
        queue_encoder=queue_state,
        head=head_state,
    )
end

