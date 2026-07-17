include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

function pc_weights(overhang)
    base = best_teacher_weights()
    return HeuristicWeights(;
        reward=base.reward,
        aggregate_height=base.aggregate_height,
        holes=base.holes,
        bumpiness=base.bumpiness,
        max_height=base.max_height,
        wells=base.wells,
        row_transitions=base.row_transitions,
        column_transitions=base.column_transitions,
        back_to_back=base.back_to_back,
        internal_bumpiness=base.internal_bumpiness,
        right_well_depth=base.right_well_depth,
        right_column_height=base.right_column_height,
        hold_i_reserve=base.hold_i_reserve,
        tetris_clear=base.tetris_clear,
        non_tetris_line=base.non_tetris_line,
        wasted_i=base.wasted_i,
        left_stack_range=base.left_stack_range,
        hole_depth=base.hole_depth,
        tetris_overhang=overhang,
    )
end

const PC_CANDIDATES = [
    "overhang_0" => pc_weights(0.0),
    "overhang_5" => pc_weights(-5.0),
    "overhang_20" => pc_weights(-20.0),
    "overhang_50" => pc_weights(-50.0),
]

function main()
    seeds = 1001:1005
    jobs = [(name, weights, seed) for (name, weights) in PC_CANDIDATES for seed in seeds]
    results = Vector{NamedTuple}(undef, length(jobs))

    Threads.@threads for index in eachindex(jobs)
        name, weights, seed = jobs[index]
        elapsed = @elapsed episode = play_episode(seed, weights)
        results[index] = (; name, seed, episode..., seconds=elapsed)
    end

    for (name, _) in PC_CANDIDATES
        subset = filter(result -> result.name == name, results)
        sort!(subset; by=result -> result.seed)
        scores = getproperty.(subset, :score)
        pcs = getproperty.(subset, :perfect_clears)
        @printf("candidate=%-12s mean=%7.1f median=%7.1f min=%5d max=%5d over12k=%d pcs=%s scores=%s\n",
            name, mean(scores), median(scores), minimum(scores), maximum(scores),
            count(>=(12_000), scores), join(pcs, ","), join(scores, ","))
    end
end


main()

