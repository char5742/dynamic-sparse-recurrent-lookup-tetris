ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

using Dates
using JSON3
using NPZ
using PythonCall

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyPartialTailTDContract
include(joinpath(@__DIR__, "tail_model.jl"))

function main(args=ARGS)
    length(args) == 3 || error(
        "usage: offline_gate.jl OFFLINE_NPZ CANDIDATE_WEIGHTS OUTPUT_JSON"
    )
    offline_path, candidate_weights, output_path = abspath.(args)
    ispath(output_path) && error("refusing to overwrite offline gate")
    data = npzread(offline_path)
    rows = Int.(vec(data["source_rows"]))
    rows == collect(OFFLINE_ROWS) || error("offline rows must be exactly 1501--2000")
    episodes = Int.(vec(data["episode_ids"]))
    sort(unique(episodes)) == collect(OFFLINE_EPISODES) || error(
        "offline episodes must be exactly 7--8"
    )
    counts = Int.(vec(data["action_counts"]))
    length(counts) == 500 || error("offline gate requires exactly 500 fixed rows")
    training = JSON3.read(read(joinpath(dirname(output_path), "training_phase.json"), String))
    training.status == "training_phase_complete" || error("training phase incomplete")
    training.weights_sha256 == hex_sha256(candidate_weights) || error(
        "offline weights differ from fresh training export"
    )
    openvino = JSON3.read(read(joinpath(dirname(output_path), "openvino_gate.json"), String))
    openvino.status == "openvino_gate_pass" || error("fresh OpenVINO gate incomplete")
    started = time()
    sys = pyimport("sys")
    sys.path.insert(0, @__DIR__)
    weighted = pyimport("weighted_inference")
    repository = normpath(joinpath(@__DIR__, "..", ".."))
    compile_seconds = @elapsed inference = weighted.WeightedLegacyOpenVINOInference(
        repository, candidate_weights, "NPU", LEGACY_BATCH
    )
    agreements = 0
    all_q_finite = true
    candidate_evaluations = 0
    inference_started = time()
    for slot in eachindex(rows)
        count = counts[slot]
        input = row_input(data, slot, count)
        scores = pyconvert(
            Vector{Float32},
            inference.predict(input[1], input[2], input[3], input[4], input[5], input[6]),
        )
        old_scores = Float32.(@view(data["stored_q"][slot, 1:count]))
        all_q_finite &= all(isfinite, scores)
        agreements += argmax(scores) == argmax(old_scores)
        candidate_evaluations += count
    end
    agreement = agreements / length(rows)
    passed = all_q_finite && agreement >= OFFLINE_TOP1_AGREEMENT
    result = (;
        status=passed ? "offline_gate_pass" : "offline_gate_reject",
        generated_at=string(now()),
        candidate_weights,
        candidate_weights_sha256=hex_sha256(candidate_weights),
        offline_npz_sha256=hex_sha256(offline_path),
        offline_npz_path=offline_path,
        training_phase_sha256=hex_sha256(joinpath(dirname(output_path), "training_phase.json")),
        openvino_gate_sha256=hex_sha256(joinpath(dirname(output_path), "openvino_gate.json")),
        explicit_fresh_weight_constructor=true,
        compile_seconds,
        rows=[first(OFFLINE_ROWS), last(OFFLINE_ROWS)],
        row_count=length(rows),
        episodes=collect(OFFLINE_EPISODES),
        seeds=collect(OFFLINE_SEEDS),
        top1_agreements=agreements,
        top1_agreement=agreement,
        required_top1_agreement=OFFLINE_TOP1_AGREEMENT,
        all_q_finite,
        candidate_evaluations,
        logical_network_calls=length(rows),
        physical_network_calls=sum(cld(count, LEGACY_BATCH) for count in counts),
        inference_loop_seconds=time() - inference_started,
        wall_seconds=time() - started,
        checkpoint_selection_performed=false,
        earlier_checkpoint_rollback=false,
        validation_or_test_seed_loaded=false,
        game_evaluation_run=false,
    )
    atomic_write_json(output_path, result)
    passed || error("offline top-1/finite gate rejected fixed update-300 candidate")
    println(JSON3.write((; status=result.status, output=output_path)))
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
