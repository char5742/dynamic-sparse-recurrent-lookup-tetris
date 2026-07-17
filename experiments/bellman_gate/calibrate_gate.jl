include(joinpath(@__DIR__, "train_gate.jl"))

using JLD2
using JSON3
using Statistics

function main()
    dataset_path = abspath(ENV["GATE_DATASET_PATH"])
    checkpoint_path = abspath(ENV["GATE_CHECKPOINT_PATH"])
    output_path = abspath(get(
        ENV,
        "GATE_CALIBRATION_PATH",
        replace(checkpoint_path, r"\.jld2$" => "_calibration.json"),
    ))
    data = load(dataset_path)
    checkpoint = load(checkpoint_path)
    hidden = Int(checkpoint["hidden"])
    model = build_residual_gate(size(checkpoint["feature_mean"], 1); hidden)
    ps = checkpoint["ps"]
    st = Lux.testmode(checkpoint["st"])
    validation_ids = Int.(checkpoint["validation_episode_ids"])
    episode_ids = Int.(data["episode_ids"])
    indices = findall(id -> id in validation_ids, episode_ids)
    first_feature = Float32.(data["feature_top1"][:, indices])
    second_feature = Float32.(data["feature_top2"][:, indices])
    q_first = reshape(Float32.(data["q_top1"][indices]), 1, :)
    q_second = reshape(Float32.(data["q_top2"][indices]), 1, :)
    logits, _ = residual_logits(
        model,
        ps,
        st,
        (first_feature, second_feature, q_first, q_second),
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        Float32(checkpoint["residual_scale"]),
    )
    logits = vec(logits)
    metadata = data["metadata"]
    blend = Float32(metadata.blend)
    teacher_first = (1.0f0 - blend) .* vec(q_first) .+
                    blend .* Float32.(data["bellman_top1"][indices])
    teacher_second = (1.0f0 - blend) .* vec(q_second) .+
                     blend .* Float32.(data["bellman_top2"][indices])
    teacher_labels = teacher_second .> teacher_first

    sorted_logits = sort(unique(logits))
    thresholds = Float32[Inf32, 0.0f0]
    append!(thresholds, sorted_logits)
    append!(
        thresholds,
        Float32[(sorted_logits[i] + sorted_logits[i + 1]) / 2
                for i in 1:(length(sorted_logits) - 1)],
    )
    candidates = NamedTuple[]
    for threshold in thresholds
        predictions = logits .> threshold
        gain = mean(ifelse.(predictions, teacher_second .- teacher_first, 0.0f0))
        tp = count(predictions .& teacher_labels)
        fp = count(predictions .& .!teacher_labels)
        push!(candidates, (;
            threshold,
            mean_teacher_gain=Float64(gain),
            accuracy=mean(predictions .== teacher_labels),
            flip_rate=mean(predictions),
            precision=tp / max(tp + fp, 1),
            true_positives=tp,
            false_positives=fp,
        ))
    end
    candidate_keys = [
        (item.mean_teacher_gain, item.accuracy, -item.flip_rate) for item in candidates
    ]
    best_index = argmax(candidate_keys)
    best = candidates[best_index]
    result = (;
        experiment="BG01_holdout_threshold_calibration",
        dataset_path,
        checkpoint_path,
        validation_episode_ids=validation_ids,
        state_count=length(indices),
        selection_rule="maximize holdout mean Bellman-teacher gain; then accuracy; then fewer flips",
        uncalibrated=candidates[findfirst(item -> item.threshold == 0.0f0, candidates)],
        calibrated=best,
        held_out_test_seeds_used=false,
    )
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Calibrated Bellman gate threshold" output_path result
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
