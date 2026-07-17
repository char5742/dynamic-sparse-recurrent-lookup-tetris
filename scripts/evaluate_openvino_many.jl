include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

using Statistics

function main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    seed_start = parse(Int, get(ENV, "EVAL_SEED_START", "5742"))
    seed_count = parse(Int, get(ENV, "EVAL_SEED_COUNT", "4"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    seeds = collect(seed_start:(seed_start + seed_count - 1))
    @info "Compiling shared OpenVINO policy" device batch_size seeds
    inference = legacy_openvino.LegacyOpenVINOInference(device, batch_size)
    episodes = Vector{NamedTuple}(undef, length(seeds))
    started = time()
    # PythonCall's embedded CPython is not safe to enter from Julia worker
    # threads on Windows. Keep this evaluator sequential; process-level
    # parallelism is used by the orchestration script when needed.
    for index in eachindex(seeds)
        episodes[index] = evaluate_openvino_episode(
            seeds[index], inference; next_count=5, max_steps
        )
        @info "Completed parallel episode" episodes[index]
    end
    scores = getproperty.(episodes, :score)
    summary = (;
        timestamp=string(now()),
        device,
        batch_size,
        seeds,
        max_steps,
        scores,
        mean_score=mean(scores),
        median_score=median(scores),
        minimum_score=minimum(scores),
        maximum_score=maximum(scores),
        over_target=count(>=(SCORE_TARGET), scores),
        wall_seconds=time() - started,
        episodes,
    )
    @info "Parallel evaluation summary" summary
    output_path = joinpath(
        ROOT, "runs", "openvino_many_$(first(seeds))_$(last(seeds)).json"
    )
    open(output_path, "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Saved parallel evaluation" output_path
end

main()
