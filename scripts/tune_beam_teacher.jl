include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

function main()
    weights = best_teacher_weights()
    seeds = parse.(Int, split(get(ENV, "BEAM_SEEDS", "1001"), ','))
    configs = [
        (name="beam4_d10", width=4, discount=1.0),
        (name="beam8_d10", width=8, discount=1.0),
        (name="beam8_d15", width=8, discount=1.5),
        (name="beam16_d10", width=16, discount=1.0),
        (name="beam16_d15", width=16, discount=1.5),
    ]
    config_filter = get(ENV, "BEAM_CONFIG_FILTER", "")
    !isempty(config_filter) && (configs = filter(config -> contains(config.name, config_filter), configs))
    jobs = [(config, seed) for config in configs for seed in seeds]
    results = Vector{NamedTuple}(undef, length(jobs))

    Threads.@threads for index in eachindex(jobs)
        config, seed = jobs[index]
        elapsed = @elapsed episode = play_episode(
            seed,
            weights;
            beam_width=config.width,
            beam_discount=config.discount,
        )
        results[index] = (; name=config.name, seed, episode..., seconds=elapsed)
    end

    for config in configs
        subset = filter(result -> result.name == config.name, results)
        scores = getproperty.(subset, :score)
        @printf("config=%-12s mean=%7.1f median=%7.1f min=%5d max=%5d scores=%s seconds=%s\n",
            config.name, mean(scores), median(scores), minimum(scores), maximum(scores),
            join(scores, ","), join(round.(getproperty.(subset, :seconds); digits=1), ","))
    end
end

main()
