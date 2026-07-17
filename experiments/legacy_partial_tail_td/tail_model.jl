function parameter_layer(parameters, index::Integer)
    return getproperty(parameters, Symbol("layer_", index))
end

function model_layer(chain, index::Integer)
    return getproperty(chain.layers, Symbol("layer_", index))
end

function trainable_parameters(parameters)
    return (;
        board_net=(;
            resblocks=(;
                layer_29=parameters.board_net.resblocks.layer_29,
                layer_31=parameters.board_net.resblocks.layer_31,
            ),
            conv2=parameters.board_net.conv2,
            norm2=parameters.board_net.norm2,
        ),
        score_net=parameters.score_net,
    )
end

function merge_trainable(parameters, trainable)
    residuals = merge(
        parameters.board_net.resblocks,
        (;
            layer_29=trainable.board_net.resblocks.layer_29,
            layer_31=trainable.board_net.resblocks.layer_31,
        ),
    )
    board_net = merge(
        parameters.board_net,
        (;
            resblocks=residuals,
            conv2=trainable.board_net.conv2,
            norm2=trainable.board_net.norm2,
        ),
    )
    return merge(parameters, (; board_net, score_net=trainable.score_net))
end

function slice_input(input, range::UnitRange{Int})
    return (
        @view(input[1][:, :, :, range]),
        @view(input[2][:, :, :, range]),
        @view(input[3][:, range]),
        @view(input[4][:, range]),
        @view(input[5][:, range]),
        @view(input[6][:, :, range]),
    )
end

function row_input(data, slot::Int, count::Int)
    board_one = permutedims(@view(data["boards"][slot:slot, :, :, :]), (2, 3, 4, 1))
    board = repeat(Float32.(board_one); outer=(1, 1, 1, count))
    placement = permutedims(
        Float32.(@view(data["placements"][slot, 1:count, :, :, :])),
        (2, 3, 4, 1),
    )
    ren = fill(Float32(data["ren"][slot]), 1, count)
    back_to_back = fill(Float32(data["back_to_back"][slot]), 1, count)
    tspin = reshape(Float32.(@view(data["tspin"][slot, 1:count])), 1, count)
    queue_one = Float32.(@view(data["queues"][slot, :, :]))
    queue = repeat(reshape(queue_one, 7, 6, 1); outer=(1, 1, count))
    return (board, placement, ren, back_to_back, tspin, queue)
end

function frozen_prefix(model, parameters, fixed_state, input)
    board, placement, ren, back_to_back, tspin, mino_list = input
    value = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    value, _ = model.board_net.conv1(
        value, parameters.board_net.conv1, fixed_state.board_net.conv1
    )
    value, _ = model.board_net.norm1(
        value, parameters.board_net.norm1, fixed_state.board_net.norm1
    )
    value = Lux.swish(value)
    for index in 1:28
        value, _ = model_layer(model.board_net.resblocks, index)(
            value,
            parameter_layer(parameters.board_net.resblocks, index),
            parameter_layer(fixed_state.board_net.resblocks, index),
        )
    end
    mino_features, _ = model.mino_list_encoder(
        mino_list, parameters.mino_list_encoder, fixed_state.mino_list_encoder
    )
    queue_features, _ = model.attention(
        mino_features, parameters.attention, fixed_state.attention
    )
    return (; board_prefix=value, ren, back_to_back, tspin, queue_features)
end

function tail_scores(model, trainable, fixed_state, prefix)
    value, _ = model_layer(model.board_net.resblocks, 29)(
        prefix.board_prefix,
        trainable.board_net.resblocks.layer_29,
        fixed_state.board_net.resblocks.layer_29,
    )
    value, _ = model_layer(model.board_net.resblocks, 30)(
        value,
        NamedTuple(),
        fixed_state.board_net.resblocks.layer_30,
    )
    value, _ = model_layer(model.board_net.resblocks, 31)(
        value,
        trainable.board_net.resblocks.layer_31,
        fixed_state.board_net.resblocks.layer_31,
    )
    value, _ = model_layer(model.board_net.resblocks, 32)(
        value,
        NamedTuple(),
        fixed_state.board_net.resblocks.layer_32,
    )
    value, _ = model.board_net.conv2(
        value, trainable.board_net.conv2, fixed_state.board_net.conv2
    )
    value, _ = model.board_net.norm2(
        value, trainable.board_net.norm2, fixed_state.board_net.norm2
    )
    value = Lux.swish(value)
    board_features = reshape(value, :, size(value, 4))
    combined = vcat(
        board_features,
        prefix.ren ./ 30.0f0,
        prefix.back_to_back,
        prefix.tspin,
        prefix.queue_features,
    )
    scores, _ = model.score_net(
        combined, trainable.score_net, fixed_state.score_net
    )
    return vec(scores)
end

function split_historical_scores(model, trainable, parameters, fixed_state, input, count)
    output = Vector{Float32}(undef, count)
    for range in chunk_ranges(count)
        chunk = slice_input(input, range)
        prefix = frozen_prefix(model, parameters, fixed_state, chunk)
        output[range] .= Float32.(Array(tail_scores(model, trainable, fixed_state, prefix)))
    end
    return output
end

function full_historical_scores(model, parameters, fixed_state, input, count)
    output = Vector{Float32}(undef, count)
    for range in chunk_ranges(count)
        values, _ = model(slice_input(input, range), parameters, fixed_state)
        output[range] .= vec(Float32.(Array(values)))
    end
    return output
end

function selected_training_problem(
    model, parameters, fixed_state, input, selected::Int, count::Int, old_scores, target
)
    range = selected_chunk_range(selected, count)
    prefix = frozen_prefix(model, parameters, fixed_state, slice_input(input, range))
    return (;
        prefix,
        selected_local=selected - first(range) + 1,
        old_scores=Float32.(old_scores[range]),
        target=Float32(target),
        range,
    )
end

function training_loss(model, trainable, fixed_state, problem)
    scores = tail_scores(model, trainable, fixed_state, problem.prefix)
    return anchored_loss(scores, problem.selected_local, problem.target, problem.old_scores)
end
