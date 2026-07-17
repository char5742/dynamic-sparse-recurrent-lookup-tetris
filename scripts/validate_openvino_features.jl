include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

using JLD2
using Statistics

sys = pyimport("sys")
sys.path.insert(0, joinpath(ROOT, "tools"))
legacy_openvino = pyimport("legacy_openvino")
score_inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)
feature_inference = legacy_openvino.LegacyOpenVINOFeatures("NPU", 16)

parameters, model_state = jldopen(joinpath(ROOT, "1313", "mainmodel copy 3.jld2"), "r") do file
    modernize_legacy_parameters(file["ps"]), Lux.testmode(file["st"])
end
model = LegacyQNetwork()
state = GameState(Xoshiro(5742))
nodes = stable_node_list(state)
input = legacy_candidate_batch(state, nodes; next_count=5)
reference = openvino_scores(score_inference, input)
python_features = lock(PYTHON_INFERENCE_LOCK) do
    feature_inference.predict(input[1], input[2], input[3], input[4], input[5], input[6])
end
features = permutedims(pyconvert(Matrix{Float32}, python_features))
scores, _ = model.score_net(features, parameters.score_net, model_state.score_net)
errors = abs.(vec(scores) .- reference)
println(
    (; candidates=length(nodes), feature_shape=size(features), maximum_absolute_error=maximum(errors), mean_absolute_error=mean(errors))
)
