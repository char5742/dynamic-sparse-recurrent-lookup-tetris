struct LegacyPositionalEncoding <: Lux.AbstractLuxLayer
    features::Int
    sequence_length::Int
end

Lux.initialparameters(::AbstractRNG, ::LegacyPositionalEncoding) = NamedTuple()

function Lux.initialstates(::AbstractRNG, layer::LegacyPositionalEncoding)
    positional_encoding = Float32[
        isodd(feature) ?
        sin(position / 10_000.0f0^(2 * feature / layer.features)) :
        cos(position / 10_000.0f0^(2 * feature / layer.features))
        for feature in 1:layer.features, position in 1:layer.sequence_length, _ in 1:1
    ]
    return (; pos_enc=positional_encoding)
end

Lux.parameterlength(::LegacyPositionalEncoding) = 0
Lux.statelength(layer::LegacyPositionalEncoding) = layer.features * layer.sequence_length

function (layer::LegacyPositionalEncoding)(input, _parameters, state)
    return input .+ state.pos_enc, state
end

struct LegacyQueueMixer <: Lux.AbstractLuxContainerLayer{
    (:embedding, :positional_encoding, :blocks, :output)
}
    embedding
    positional_encoding
    blocks
    output
end

function token_mixing_block(features::Int, sequence_length::Int)
    token_mlp = Chain(
        LayerNorm((features, sequence_length)),
        WrappedFunction(input -> permutedims(input, (2, 1, 3))),
        Dense(sequence_length => 4 * sequence_length),
        WrappedFunction(gelu),
        Dense(4 * sequence_length => sequence_length),
        WrappedFunction(input -> permutedims(input, (2, 1, 3))),
    )
    return SkipConnection(token_mlp, +)
end

function channel_mixing_block(features::Int, sequence_length::Int)
    channel_mlp = Chain(
        LayerNorm((features, sequence_length)),
        Dense(features => 4 * features),
        WrappedFunction(gelu),
        Dense(4 * features => features),
    )
    return SkipConnection(channel_mlp, +)
end

function LegacyQueueMixer(
    features::Int=128, sequence_length::Int=6, mixer_depth::Int=6
)
    mixer_layers = Any[]
    for _ in 1:mixer_depth
        push!(mixer_layers, token_mixing_block(features, sequence_length))
        push!(mixer_layers, channel_mixing_block(features, sequence_length))
    end
    return LegacyQueueMixer(
        Dense(features => features),
        LegacyPositionalEncoding(features, sequence_length),
        Chain(mixer_layers...),
        Chain(FlattenLayer(), Dense(features * sequence_length => sequence_length)),
    )
end

function (mixer::LegacyQueueMixer)(input, parameters, state)
    value, _ = mixer.embedding(input, parameters.embedding, state.embedding)
    value, _ = mixer.positional_encoding(
        value, parameters.positional_encoding, state.positional_encoding
    )
    value, blocks_state = mixer.blocks(value, parameters.blocks, state.blocks)
    value, output_state = mixer.output(value, parameters.output, state.output)
    return value, merge(state, (; blocks=blocks_state, output=output_state))
end

struct LegacyBoardNetwork <: Lux.AbstractLuxContainerLayer{
    (:conv1, :norm1, :resblocks, :conv2, :norm2, :gmp)
}
    conv1
    norm1
    resblocks
    conv2
    norm2
    gmp
end

function squeeze_excitation_block(channels::Int, ratio::Int=4)
    gate = Chain(
        GlobalMeanPool(),
        Conv((1, 1), channels => channels ÷ ratio, swish),
        Conv((1, 1), channels ÷ ratio => channels, sigmoid),
    )
    return SkipConnection(gate, .*)
end

function legacy_residual_block(channels::Int)
    residual = Chain(
        Conv((3, 3), channels => channels; pad=SamePad()),
        BatchNorm(
            channels,
            swish;
            epsilon=Float16(1.0f-5),
            momentum=Float16(0.1f0),
        ),
        Conv((3, 3), channels => channels; pad=SamePad()),
        BatchNorm(
            channels;
            epsilon=Float16(1.0f-5),
            momentum=Float16(0.1f0),
        ),
        squeeze_excitation_block(channels),
    )
    return (SkipConnection(residual, +), WrappedFunction(swish))
end

function LegacyBoardNetwork(channels::Int=256, residual_blocks::Int=16)
    residual_layers = Any[]
    for _ in 1:residual_blocks
        append!(residual_layers, legacy_residual_block(channels))
    end
    return LegacyBoardNetwork(
        Conv((3, 3), 2 => channels; pad=SamePad()),
        BatchNorm(
            channels;
            epsilon=Float16(1.0f-6),
            momentum=Float16(0.1f0),
        ),
        Chain(residual_layers...),
        Conv((3, 3), channels => 1; pad=SamePad()),
        BatchNorm(1; epsilon=Float16(1.0f-6), momentum=Float16(0.1f0)),
        GlobalMeanPool(),
    )
end

function (network::LegacyBoardNetwork)((board, placement), parameters, state)
    value = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    value, _ = network.conv1(value, parameters.conv1, state.conv1)
    value, norm1_state = network.norm1(value, parameters.norm1, state.norm1)
    value = swish(value)
    value, residual_state = network.resblocks(
        value, parameters.resblocks, state.resblocks
    )
    value, _ = network.conv2(value, parameters.conv2, state.conv2)
    value, norm2_state = network.norm2(value, parameters.norm2, state.norm2)
    value = swish(value)
    # The historical 16-block checkpoint intentionally retained the global
    # pooling layer in its structure while flattening all 24x10 cells.
    value = reshape(value, :, size(value, 4))
    return value, merge(
        state,
        (; norm1=norm1_state, resblocks=residual_state, norm2=norm2_state),
    )
end

struct LegacyQNetwork <: Lux.AbstractLuxContainerLayer{
    (
        :board_net,
        :board_encoder,
        :ren_encoder,
        :btb_encoder,
        :tspin_encoder,
        :mino_list_encoder,
        :attention,
        :score_net,
    )
}
    board_net
    board_encoder
    ren_encoder
    btb_encoder
    tspin_encoder
    mino_list_encoder
    attention
    score_net
end

function LegacyQNetwork()
    return LegacyQNetwork(
        LegacyBoardNetwork(256, 16),
        NoOpLayer(),
        NoOpLayer(),
        NoOpLayer(),
        NoOpLayer(),
        Dense(7 => 128),
        LegacyQueueMixer(128, 6, 6),
        Chain(
            Dense(249 => 1024, swish),
            Dense(1024 => 256, swish),
            Dense(256 => 1),
        ),
    )
end

function (network::LegacyQNetwork)(
    (board, placement, ren, back_to_back, tspin, mino_list), parameters, state
)
    board_features, board_state = network.board_net(
        (board, placement), parameters.board_net, state.board_net
    )
    mino_features, _ = network.mino_list_encoder(
        mino_list, parameters.mino_list_encoder, state.mino_list_encoder
    )
    queue_features, attention_state = network.attention(
        mino_features, parameters.attention, state.attention
    )
    combined = vcat(
        board_features,
        ren ./ 30.0f0,
        back_to_back,
        tspin,
        queue_features,
    )
    score, score_state = network.score_net(
        combined, parameters.score_net, state.score_net
    )
    return score, merge(
        state,
        (; board_net=board_state, attention=attention_state, score_net=score_state),
    )
end

function modernize_legacy_parameters(parameters::NamedTuple)
    names = keys(parameters)
    converted = map(values(parameters)) do value
        value isa NamedTuple ? modernize_legacy_parameters(value) : value
    end
    result = NamedTuple{names}(converted)
    if haskey(result, :weight) &&
       haskey(result, :bias) &&
       result.weight isa AbstractArray &&
       result.bias isa AbstractArray &&
       ndims(result.bias) > 1
        return merge(result, (; bias=vec(result.bias)))
    end
    return result
end
