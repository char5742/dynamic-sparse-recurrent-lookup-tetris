include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

function main()
    episode_count = parse(Int, get(ENV, "EVAL_EPISODES", "5"))
    seed_offset = parse(Int, get(ENV, "EVAL_SEED_OFFSET", "1000"))
    weights = best_teacher_weights()
    results = Vector{NamedTuple}(undef, episode_count)

    Threads.@threads for index in eachindex(results)
        seed = seed_offset + index
        elapsed = @elapsed episode = play_episode(seed, weights)
        results[index] = (; seed, episode..., seconds=elapsed)
    end

    sort!(results; by=result -> result.seed)
    for result in results
        @printf("seed=%d score=%d tetrises=%d lines=%d holes=%d maxh=%d game_over=%s seconds=%.3f\n",
            result.seed, result.score, result.tetrises, result.lines_cleared,
            result.final_holes, result.final_max_height, result.game_over, result.seconds)
    end
    scores = getproperty.(results, :score)
    @printf("summary episodes=%d mean=%.1f median=%.1f min=%d max=%d over_12k=%d\n",
        length(scores), mean(scores), median(scores), minimum(scores), maximum(scores),
        count(>=(12_000), scores))
end

main()

