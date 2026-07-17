using JLD2
using JSON3
using Lux
using NPZ
using Random
using TetrisPaperPlus

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CHECKPOINT = joinpath(ROOT, "1313", "mainmodel copy 3.jld2")
const OUTPUT_DIRECTORY = joinpath(ROOT, "artifacts", "legacy_openvino")

function collect_arrays!(output, value, path="")
    if value isa AbstractArray
        output[path] = Array(value)
    elseif value isa NamedTuple
        for name in keys(value)
            child_path = isempty(path) ? string(name) : string(path, ".", name)
            collect_arrays!(output, getproperty(value, name), child_path)
        end
    end
    return output
end

function main()
    parameters, state = jldopen(CHECKPOINT, "r") do file
        modernize_legacy_parameters(file["ps"]), file["st"]
    end
    arrays = Dict{String,Array}()
    collect_arrays!(arrays, parameters, "ps")
    collect_arrays!(arrays, state, "st")

    mkpath(OUTPUT_DIRECTORY)
    npz_path = joinpath(OUTPUT_DIRECTORY, "legacy_1313_weights.npz")
    npzwrite(npz_path, arrays)

    rng = Xoshiro(0x1313)
    batch_size = 8
    reference_input = (
        Float32.(rand(rng, 0:1, 24, 10, 1, batch_size)),
        Float32.(rand(rng, 0:1, 24, 10, 1, batch_size)),
        reshape(Float32.(rand(rng, 0:30, batch_size)), 1, :),
        reshape(Float32.(rand(rng, Bool, batch_size)), 1, :),
        reshape(Float32.(rand(rng, Bool, batch_size)), 1, :),
        Float32.(rand(rng, 0:1, 7, 6, batch_size)),
    )
    model = LegacyQNetwork()
    inference_state = Lux.testmode(state)
    reference_output, _ = model(reference_input, parameters, inference_state)
    board_features, _ = model.board_net(
        reference_input[1:2], parameters.board_net, inference_state.board_net
    )
    board_input = cat(1.0f0 .- reference_input[1], 1.0f0 .- reference_input[2]; dims=3)
    board_conv1, _ = model.board_net.conv1(
        board_input, parameters.board_net.conv1, inference_state.board_net.conv1
    )
    board_norm1, _ = model.board_net.norm1(
        board_conv1, parameters.board_net.norm1, inference_state.board_net.norm1
    )
    board_norm1 = swish(board_norm1)
    first_residual, _ = model.board_net.resblocks.layers.layer_1(
        board_norm1,
        parameters.board_net.resblocks.layer_1,
        inference_state.board_net.resblocks.layer_1,
    )
    first_residual, _ = model.board_net.resblocks.layers.layer_2(
        first_residual,
        parameters.board_net.resblocks.layer_2,
        inference_state.board_net.resblocks.layer_2,
    )
    mino_features, _ = model.mino_list_encoder(
        reference_input[6],
        parameters.mino_list_encoder,
        inference_state.mino_list_encoder,
    )
    queue_embedding, _ = model.attention.embedding(
        mino_features,
        parameters.attention.embedding,
        inference_state.attention.embedding,
    )
    queue_position, _ = model.attention.positional_encoding(
        queue_embedding,
        parameters.attention.positional_encoding,
        inference_state.attention.positional_encoding,
    )
    queue_block1, _ = model.attention.blocks.layers.layer_1(
        queue_position,
        parameters.attention.blocks.layer_1,
        inference_state.attention.blocks.layer_1,
    )
    token_chain = model.attention.blocks.layers.layer_1.layers.layers
    token_parameters = parameters.attention.blocks.layer_1
    token_state = inference_state.attention.blocks.layer_1
    queue_norm, _ = token_chain.layer_1(
        queue_position, token_parameters.layer_1, token_state.layer_1
    )
    queue_token_order, _ = token_chain.layer_2(
        queue_norm, token_parameters.layer_2, token_state.layer_2
    )
    queue_token_dense1, _ = token_chain.layer_3(
        queue_token_order, token_parameters.layer_3, token_state.layer_3
    )
    queue_token_gelu, _ = token_chain.layer_4(
        queue_token_dense1, token_parameters.layer_4, token_state.layer_4
    )
    queue_token_dense2, _ = token_chain.layer_5(
        queue_token_gelu, token_parameters.layer_5, token_state.layer_5
    )
    queue_token_result, _ = token_chain.layer_6(
        queue_token_dense2, token_parameters.layer_6, token_state.layer_6
    )
    queue_features, _ = model.attention(
        mino_features, parameters.attention, inference_state.attention
    )
    combined = vcat(
        board_features,
        reference_input[3] ./ 30.0f0,
        reference_input[4],
        reference_input[5],
        queue_features,
    )
    reference_path = joinpath(OUTPUT_DIRECTORY, "legacy_1313_reference.npz")
    npzwrite(
        reference_path,
        Dict(
            "board" => reference_input[1],
            "placement" => reference_input[2],
            "ren" => reference_input[3],
            "back_to_back" => reference_input[4],
            "tspin" => reference_input[5],
            "queue" => reference_input[6],
            "board_conv1" => Array(board_conv1),
            "board_norm1" => Array(board_norm1),
            "board_residual1" => Array(first_residual),
            "board_features" => Array(board_features),
            "mino_features" => Array(mino_features),
            "queue_embedding" => Array(queue_embedding),
            "queue_position" => Array(queue_position),
            "queue_norm" => Array(queue_norm),
            "queue_token_order" => Array(queue_token_order),
            "queue_token_dense1" => Array(queue_token_dense1),
            "queue_token_gelu" => Array(queue_token_gelu),
            "queue_token_dense2" => Array(queue_token_dense2),
            "queue_token_result" => Array(queue_token_result),
            "queue_block1" => Array(queue_block1),
            "queue_features" => Array(queue_features),
            "combined" => Array(combined),
            "output" => Array(reference_output),
        ),
    )
    metadata_path = joinpath(OUTPUT_DIRECTORY, "legacy_1313_weights.json")
    open(metadata_path, "w") do io
        JSON3.pretty(io, (; checkpoint=CHECKPOINT, arrays=sort(collect(keys(arrays)))))
    end
    println("npz=", npz_path)
    println("metadata=", metadata_path)
    println("reference=", reference_path)
    println("arrays=", length(arrays))
end

main()
