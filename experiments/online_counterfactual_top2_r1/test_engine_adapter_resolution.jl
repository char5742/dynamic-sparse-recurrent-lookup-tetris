using Test
using Random
using SHA

include(joinpath(@__DIR__, "collector.jl"))
include(joinpath(@__DIR__, "engine_adapter.jl"))

using .R1ProductionEngineAdapter

@testset "production adapter resolves without loading production" begin
    @test isdefined(R1ProductionEngineAdapter, :make_production_adapter)
    @test isdefined(R1ProductionEngineAdapter, :build_production_adapter)
    # Including the adapter is deliberately inert.  The historical evaluator
    # and Python/OpenVINO are loaded only if make_production_adapter is called.
    @test !isdefined(Main, :evaluate_openvino_episode)
end

abstract type FakeMino end
for name in (:FI, :FO, :FS, :FZ, :FJ, :FL, :FT)
    @eval begin
        struct $name <: FakeMino
            direction::Int
            block::Matrix{Int}
        end
        $name() = $name(0, ones(Int, 1, 1))
    end
end
const FAKE_MINO_TYPES = (FI, FO, FS, FZ, FJ, FL, FT)

mutable struct FakeBoard
    binary::Matrix{Int}
    color::Matrix{Int}
end
struct FakePosition
    x::Int
    y::Int
end
mutable struct FakeState
    current_game_board::FakeBoard
    current_mino::FakeMino
    current_position::FakePosition
    hold_mino::Union{Nothing,FakeMino}
    mino_list::Vector{FakeMino}
    score::Int
    ren::Int
    ren_flag::Bool
    back_to_back_flag::Bool
    game_over_flag::Bool
    hold_flag::Bool
    hard_drop_flag::Bool
    t_spin_flag::Bool
    srs_index::Int
    rng::Xoshiro
end
struct FakeAction
    id::Int
end
struct FakeNode
    action_list::Vector{FakeAction}
    mino::FakeMino
    position::FakePosition
    tspin::Int
    game_state::FakeState
end

function fake_state(rng::Xoshiro)
    queue = FakeMino[FI(), FO(), FS(), FZ(), FJ(), FL(), FT(), FI()]
    return FakeState(
        FakeBoard(zeros(Int, 6, 2), zeros(Int, 6, 2)), FI(), FakePosition(1, 1),
        nothing, queue, 0, 0, false, false, false, true, false, false, -1, rng,
    )
end
function fake_state(state::FakeState)
    return FakeState(
        FakeBoard(copy(state.current_game_board.binary), copy(state.current_game_board.color)),
        state.current_mino,
        state.current_position,
        state.hold_mino,
        copy(state.mino_list),
        state.score,
        state.ren,
        state.ren_flag,
        state.back_to_back_flag,
        state.game_over_flag,
        state.hold_flag,
        state.hard_drop_flag,
        state.t_spin_flag,
        state.srs_index,
        copy(state.rng),
    )
end
function fake_nodes(state::FakeState)
    nodes = FakeNode[]
    for id in 1:2
        after = fake_state(state)
        after.score += id == 1 ? 60 : 120
        after.current_game_board.binary[6, id] = 1
        push!(nodes, FakeNode([FakeAction(id)], id == 1 ? FI() : FO(), FakePosition(id, 6), 0, after))
    end
    return nodes
end
function copy_fake_node_state!(state::FakeState, node::FakeNode)
    after = fake_state(node.game_state)
    for field in fieldnames(FakeState)
        setfield!(state, field, getfield(after, field))
    end
    return state
end
fake_board_features(board) = (;
    holes=0, aggregate_height=sum(board), max_height=maximum(sum(board; dims=1)),
    bumpiness=0, wells=0, row_transitions=0, column_transitions=0,
)
fake_bag(rng) = shuffle!(rng, FakeMino[FI(), FO(), FS(), FZ(), FJ(), FL(), FT()])

@testset "synthetic production adapter callbacks execute" begin
    accounting = EngineAccounting()
    adapter = build_production_adapter(
        nothing;
        game_state_fn=fake_state,
        stable_node_list_fn=fake_nodes,
        stable_node_key_fn=node -> node.action_list[1].id,
        legacy_candidate_batch_fn=(state, nodes; next_count) -> length(nodes),
        openvino_scores_fn=(inference, candidate_count) -> Float32[2, 1][1:candidate_count],
        apply_node_fn=copy_fake_node_state!,
        board_features_fn=fake_board_features,
        generate_mino_list_fn=fake_bag,
        mino_types=FAKE_MINO_TYPES,
        accounting,
    )
    root = adapter.initial_state(17)
    before = adapter.state_digest(root)
    future = adapter.future_stream_digest(root, 6)
    @test !isempty(future)
    @test adapter.state_digest(root) == before
    actions = adapter.candidate_actions(root)
    @test length(actions) == 2
    @test adapter.q_values(root, actions) == Float32[2, 1]
    @test accounting.logical_network_calls == 1
    @test accounting.physical_network_calls == 1
    branch = adapter.clone_state(root)
    adapter.apply_action(branch, actions[1])
    @test adapter.state_digest(branch) == adapter.state_digest(actions[1].game_state)
    @test adapter.state_digest(root) == before
    queue = adapter.queue_onehot(root)
    @test length(queue) == 42
    @test sum(queue[1:7]) == 0 # empty HOLD
    @test queue[7 + 1] == 1 # immediate NEXT is queue[end] == I
    row = R1CounterfactualCollector.collect_counterfactual(
        adapter, root; seed=17, episode_id=17, piece_index=10,
    )
    @test isapprox(row.advantage_unclipped, 0.1; atol=1e-12)
    @test length(row.top1_branch.decision_evidence) == 7 # root + 5 policy + s6
    @test row.top1_branch.decision_evidence[end].kind == :bootstrap
    @test adapter.state_digest(root) == before
end

@testset "covered_cells is unique-cell coverage on the full board" begin
    board = zeros(Int, 6, 2)
    board[2:3, 1] .= 1
    board[5, 1] = 1
    # Holes at rows 4 and 6 are both covered by the same three occupied cells;
    # unique coverage counts each occupied cell once, rather than hole_depth=5.
    @test covered_cells(board) == 3
    board[1, 2] = 1
    @test covered_cells(board) == 4
end

@testset "engine binding is explicit and frozen" begin
    source = read(joinpath(@__DIR__, "engine_adapter.jl"), String)
    @test occursin("evaluate_openvino_checkpoint.jl", source)
    @test occursin("LegacyOpenVINOInference(\"NPU\", 16)", source)
    @test occursin("legacy_1313_weights.npz", source)
    @test occursin("2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba", source)
    @test occursin("stable_node_key_fn=getfield(engine, :stable_node_key)", source)
    @test !occursin("stable_node_key_fn=Main.stable_node_key", source)
    @test occursin("_load_frozen_evaluator", source)
    @test occursin("unexpected recursive include in frozen legacy evaluator", source)
    @test occursin("GameState(root)", source)
    @test occursin("node_replay_mismatch", source)
    @test occursin("future_oracle_mutated_state", source)
end

@testset "vendored candidate generator is byte-identical and privately loaded" begin
    repository = normpath(joinpath(@__DIR__, "..", ".."))
    original_node = joinpath(
        repository, "upstream", "TetrisAI", "src", "core", "components", "node.jl",
    )
    original_analyzer = joinpath(
        repository, "upstream", "TetrisAI", "src", "core", "analyzer.jl",
    )
    vendored_node = joinpath(
        @__DIR__, "vendor", "TetrisAI", "src", "core", "components", "node.jl",
    )
    vendored_analyzer = joinpath(
        @__DIR__, "vendor", "TetrisAI", "src", "core", "analyzer.jl",
    )
    @test read(vendored_node) == read(original_node)
    @test read(vendored_analyzer) == read(original_analyzer)
    @test bytes2hex(open(SHA.sha256, vendored_node)) ==
          "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82"
    @test bytes2hex(open(SHA.sha256, vendored_analyzer)) ==
          "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc"

    evaluator = joinpath(repository, "scripts", "evaluate_openvino_checkpoint.jl")
    private_module = R1ProductionEngineAdapter._load_frozen_evaluator(repository, evaluator)
    @test private_module !== Main
    @test isdefined(private_module, :stable_node_list)
    @test isdefined(private_module, :stable_node_key)
    @test isdefined(private_module, :legacy_candidate_batch)
    @test isdefined(private_module, :openvino_scores)
    @test !isdefined(Main, :stable_node_list)
    @test !isdefined(Main, :evaluate_openvino_episode)

    graph = R1ProductionEngineAdapter._engine_dependency_graph(repository)
    @test graph.schema_version == "r1-engine-dependency-graph-v1"
    @test graph.upstream_tetrisai.head ==
          "6fdfb1d30197246fd862b716438e998f0315c830"
    @test graph.upstream_tetrisai.clean
    @test length(graph.records) == 7
    @test issorted(getproperty.(graph.records, :path))
    @test graph.node_source_sha256 ==
          "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82"
    @test graph.analyzer_source_sha256 ==
          "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc"
end
