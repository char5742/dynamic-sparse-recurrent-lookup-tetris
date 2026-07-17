ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

using Dates
using JSON3
using PythonCall
using Statistics

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyPartialTailTDContract
include(joinpath(@__DIR__, "..", "..", "scripts", "evaluate_openvino_checkpoint.jl"))

function write_progress(path, started, status, records, pairs, compilation)
    atomic_write_json(
        path,
        (;
            status,
            generated_at=string(now()),
            wall_seconds=time() - started,
            seeds=collect(DEVELOPMENT_SEEDS),
            next_count=5,
            hold_enabled=true,
            candidate_order="stable_node_key",
            max_pieces=DEVELOPMENT_PIECES,
            lookahead_expansions=0,
            logical_full_candidate_score_calls_per_decision=1,
            compilation,
            training_phase_sha256=hex_sha256(joinpath(dirname(path), "training_phase.json")),
            openvino_gate_sha256=hex_sha256(joinpath(dirname(path), "openvino_gate.json")),
            offline_gate_sha256=hex_sha256(joinpath(dirname(path), "offline_gate.json")),
            evaluations=records,
            pairs,
            validation_or_test_seed_loaded=false,
        ),
    )
end

function main(args=ARGS)
    length(args) == 4 || error(
        "usage: evaluate_development.jl REPOSITORY CANDIDATE_WEIGHTS BASELINE_WEIGHTS OUTPUT_JSON"
    )
    repository, candidate_weights, baseline_weights, output_path = abspath.(args)
    ispath(output_path) && error("refusing to overwrite development result")
    require_hash(
        baseline_weights,
        CANONICAL_BASELINE_WEIGHTS_SHA256,
        "canonical old OpenVINO baseline weights",
    )
    training_path = joinpath(dirname(output_path), "training_phase.json")
    openvino_path = joinpath(dirname(output_path), "openvino_gate.json")
    offline_path = joinpath(dirname(output_path), "offline_gate.json")
    JSON3.read(read(training_path, String)).status == "training_phase_complete" || error(
        "training gate incomplete"
    )
    JSON3.read(read(openvino_path, String)).status == "openvino_gate_pass" || error(
        "OpenVINO gate incomplete"
    )
    JSON3.read(read(offline_path, String)).status == "offline_gate_pass" || error(
        "offline gate incomplete"
    )
    sys = pyimport("sys")
    sys.path.insert(0, @__DIR__)
    weighted = pyimport("weighted_inference")
    started = time()
    candidate_compile_seconds = @elapsed candidate_inference =
        weighted.WeightedLegacyOpenVINOInference(
            repository, candidate_weights, "NPU", LEGACY_BATCH
        )
    baseline_compile_seconds = @elapsed baseline_inference =
        weighted.WeightedLegacyOpenVINOInference(
            repository, baseline_weights, "NPU", LEGACY_BATCH
        )
    compilation = (;
        candidate_seconds=candidate_compile_seconds,
        baseline_seconds=baseline_compile_seconds,
        candidate_weights,
        candidate_weights_sha256=hex_sha256(candidate_weights),
        baseline_weights,
        baseline_weights_sha256=hex_sha256(baseline_weights),
        explicit_weight_constructor=true,
    )
    records = Any[]
    pairs = Any[]
    write_progress(output_path, started, "development_running", records, pairs, compilation)
    for seed in DEVELOPMENT_SEEDS
        candidate = evaluate_openvino_episode(
            seed,
            candidate_inference;
            next_count=5,
            max_steps=DEVELOPMENT_PIECES,
            inference_batch_size=LEGACY_BATCH,
        )
        push!(records, merge((; role="candidate"), candidate))
        write_progress(output_path, started, "development_running", records, pairs, compilation)
        if candidate.game_over || candidate.steps != DEVELOPMENT_PIECES
            write_progress(
                output_path,
                started,
                "development_reject_game_over",
                records,
                pairs,
                compilation,
            )
            error("candidate did not complete 100 pieces on seed $seed")
        end
        baseline = evaluate_openvino_episode(
            seed,
            baseline_inference;
            next_count=5,
            max_steps=DEVELOPMENT_PIECES,
            inference_batch_size=LEGACY_BATCH,
        )
        push!(records, merge((; role="canonical_old_baseline"), baseline))
        if baseline.game_over || baseline.steps != DEVELOPMENT_PIECES
            write_progress(
                output_path,
                started,
                "development_reject_game_over",
                records,
                pairs,
                compilation,
            )
            error("baseline did not complete 100 pieces on seed $seed")
        end
        difference = candidate.score - baseline.score
        pair = (;
            seed,
            candidate_score=candidate.score,
            baseline_score=baseline.score,
            difference,
        )
        push!(pairs, pair)
        if seed == first(DEVELOPMENT_SEEDS) && difference <= 0
            write_progress(
                output_path,
                started,
                "development_reject_first_pair_nonpositive",
                records,
                pairs,
                compilation,
            )
            error("seed 5756 difference is nonpositive; seed 5757 remains unused")
        end
        write_progress(output_path, started, "development_running", records, pairs, compilation)
    end
    differences = Float64[pair.difference for pair in pairs]
    pair_signs_pass = length(differences) == 2 && all(>(0), differences)
    mean_difference = mean(differences)
    median_difference = median(differences)
    passed = pair_signs_pass && mean_difference >= DEVELOPMENT_MEAN_MINIMUM &&
             median_difference > 0
    accounting = (;
        candidate_evaluations=sum(record.candidate_evaluations for record in records),
        logical_network_calls=sum(record.logical_network_calls for record in records),
        physical_network_calls=sum(record.physical_network_calls for record in records),
        generation_seconds=sum(record.generation_seconds for record in records),
        inference_seconds=sum(record.inference_seconds for record in records),
        episode_wall_seconds=sum(record.wall_seconds for record in records),
    )
    final = (;
        status=passed ? "P1-development-pass" : "development_reject_summary_gate",
        generated_at=string(now()),
        wall_seconds=time() - started,
        seeds=collect(DEVELOPMENT_SEEDS),
        next_count=5,
        hold_enabled=true,
        candidate_order="stable_node_key",
        max_pieces=DEVELOPMENT_PIECES,
        lookahead_expansions=0,
        logical_full_candidate_score_calls_per_decision=1,
        compilation,
        training_phase_sha256=hex_sha256(training_path),
        openvino_gate_sha256=hex_sha256(openvino_path),
        offline_gate_sha256=hex_sha256(offline_path),
        evaluations=records,
        pairs,
        paired_differences=differences,
        both_strictly_positive=pair_signs_pass,
        paired_mean_difference=mean_difference,
        paired_median_difference=median_difference,
        required_mean_difference=DEVELOPMENT_MEAN_MINIMUM,
        accounting,
        statistical_model_beat_claim=false,
        sealed_test_authorized=false,
        validation_or_test_seed_loaded=false,
    )
    atomic_write_json(output_path, final)
    passed || error("two-seed development summary gate rejected fixed candidate")
    println(JSON3.write((; status=final.status, output=output_path)))
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
