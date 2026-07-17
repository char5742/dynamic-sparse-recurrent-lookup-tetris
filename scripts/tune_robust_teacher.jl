include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

function robust_weights(; holes, hole_depth, max_height=-2.0, aggregate_height=-0.2)
    return HeuristicWeights(
        reward=0.5,
        aggregate_height=aggregate_height,
        holes=holes,
        bumpiness=0.0,
        max_height=max_height,
        wells=0.0,
        row_transitions=-0.5,
        column_transitions=-1.0,
        back_to_back=30.0,
        internal_bumpiness=-4.0,
        right_well_depth=12.0,
        right_column_height=-15.0,
        tetris_clear=300.0,
        non_tetris_line=-100.0,
        left_stack_range=-4.0,
        hole_depth=hole_depth,
    )
end

const ROBUST_CANDIDATES = [
    "h70_d0" => robust_weights(holes=-70.0, hole_depth=0.0),
    "h200_d20" => robust_weights(holes=-200.0, hole_depth=-20.0),
    "h500_d50" => robust_weights(holes=-500.0, hole_depth=-50.0),
    "h1000_d100" => robust_weights(holes=-1000.0, hole_depth=-100.0),
    "h300_d50_height" => robust_weights(
        holes=-300.0,
        hole_depth=-50.0,
        max_height=-10.0,
        aggregate_height=-0.5,
    ),
]

function main()
    seeds = 1001:1003
    jobs = [(name, weights, seed) for (name, weights) in ROBUST_CANDIDATES for seed in seeds]
    results = Vector{NamedTuple}(undef, length(jobs))

    Threads.@threads for index in eachindex(jobs)
        name, weights, seed = jobs[index]
        elapsed = @elapsed episode = play_episode(seed, weights)
        results[index] = (; name, seed, episode..., seconds=elapsed)
    end

    for (name, _) in ROBUST_CANDIDATES
        subset = filter(result -> result.name == name, results)
        sort!(subset; by=result -> result.seed)
        scores = getproperty.(subset, :score)
        @printf("candidate=%-20s mean=%7.1f median=%7.1f min=%5d max=%5d scores=%s\n",
            name, mean(scores), median(scores), minimum(scores), maximum(scores), join(scores, ","))
    end
end


main()

