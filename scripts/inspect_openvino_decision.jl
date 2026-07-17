include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

sys = pyimport("sys")
sys.path.insert(0, joinpath(ROOT, "tools"))
legacy_openvino = pyimport("legacy_openvino")
inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)
seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
state = GameState(Xoshiro(seed))
decision_step = parse(Int, get(ENV, "DECISION_STEP", "0"))
for _ in 1:decision_step
    node, _, _, _ = select_openvino_node(state, inference; next_count=5)
    isnothing(node) && break
    apply_node!(state, node)
end
nodes = stable_node_list(state)
scores = openvino_scores(inference, legacy_candidate_batch(state, nodes; next_count=5))
rewards = Float32[(node.game_state.score - state.score) / 600 for node in nodes]
indices = partialsortperm(scores, 1:min(12, length(scores)); rev=true)
println("step=", decision_step, " state_score=", state.score, " score_range=", extrema(scores), " reward_range=", extrema(rewards))
for index in indices
    node = nodes[index]
    println(
        (; index, q=scores[index], reward=rewards[index], tspin=node.tspin, score=node.game_state.score)
    )
end
