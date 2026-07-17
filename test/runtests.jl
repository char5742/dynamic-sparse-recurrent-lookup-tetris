using Test
using Random
using Tetris

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "upstream", "TetrisAI", "src", "core", "components", "node.jl"))
include(joinpath(ROOT, "upstream", "TetrisAI", "src", "core", "analyzer.jl"))

@testset "candidate simulation preserves root RNG and replay" begin
    state = GameState(Xoshiro(5742))
    state.mino_list = state.mino_list[(end - 7):end]
    expected_rng = copy(state.rng)

    nodes = get_node_list(state)
    @test !isempty(nodes)
    @test rand(state.rng, UInt64) == rand(expected_rng, UInt64)

    # Restore equivalent roots after the RNG assertion consumed one value.
    root = GameState(Xoshiro(5742))
    root.mino_list = root.mino_list[(end - 7):end]
    node = first(get_node_list(root))
    replay = GameState(root)
    for action in node.action_list
        action!(replay, action)
    end
    put_mino!(replay)

    @test replay.current_game_board.binary == node.game_state.current_game_board.binary
    @test replay.score == node.game_state.score
    @test typeof(replay.current_mino) == typeof(node.game_state.current_mino)
    @test typeof.(replay.mino_list) == typeof.(node.game_state.mino_list)
end

include(joinpath(@__DIR__, "conformance_tests.jl"))
include(joinpath(@__DIR__, "evaluation_artifact_tests.jl"))
