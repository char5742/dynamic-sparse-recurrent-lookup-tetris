using JSON3
using LinearAlgebra
using Lux
using Profile
using Random

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const DATASET_PATH = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(2026072703)

function training_rows_only(dataset)
    rows = hasproperty(dataset, :predefined_split) ?
        findall(==(:train), dataset.predefined_split) : Int[]
    return isempty(rows) ? collect(eachindex(dataset.action_counts)) : Int.(rows)
end

function actionable_frame(stacktrace)
    fallback = nothing
    for frame in stacktrace
        frame.from_c && continue
        Int(frame.line) > 0 || continue
        file = replace(String(frame.file), '\\' => '/')
        function_name = String(frame.func)
        fallback === nothing && (fallback = (function_name, file, Int(frame.line)))
        normalized = lowercase(file)
        startswith(normalized, "./") && continue
        occursin("/share/julia/", normalized) && continue
        occursin("/workdir/", normalized) && continue
        return (function_name, file, Int(frame.line))
    end
    return fallback === nothing ? ("<unknown>", "<unknown>", -1) : fallback
end

function main()
    Threads.nthreads(:default) == 20 || error("launch with --threads=20,0")
    Threads.nthreads(:interactive) == 0 || error("interactive pool must be zero")
    BLAS.set_num_threads(1)
    dataset = load_teacher_dataset(
        DATASET_PATH;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=8,
    )
    rows = training_rows_only(dataset)
    width = 16 * cld(maximum(dataset.action_counts), 16)
    model = build_model(:scaled)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    trainer = ArenaTrainer(
        model,
        parameters;
        state_batch=8,
        width,
        parameter_shard_size=4096,
    )
    trainer.arena.rows .= @view rows[1:8]
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers=20,
        cpuset_mode=:all,
    )
    team = run_with_arena_team!(executor) do running
        arena_update!(running)
        arena_update!(running)
        GC.gc()
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 arena_update!(running)
        profile = Profile.Allocs.fetch()
        return (profile=profile, metrics=deepcopy(trainer.metrics))
    end
    profile = team.result.profile
    sites = Dict{Tuple{String,String,Int},Tuple{UInt128,Int}}()
    for allocation in profile.allocs
        site = actionable_frame(allocation.stacktrace)
        prior = get(sites, site, (UInt128(0), 0))
        sites[site] = (prior[1] + UInt128(allocation.size), prior[2] + 1)
    end
    ranked = collect(sites)
    sort!(ranked; by=entry -> (-Int128(last(entry)[1]), -last(entry)[2]))
    top = [
        (;
            function_name=first(entry)[1],
            file=first(entry)[2],
            line=first(entry)[3],
            bytes=last(entry)[1],
            allocations=last(entry)[2],
        )
        for entry in ranked[1:min(40, length(ranked))]
    ]
    report = (;
        sampled_allocations=length(profile.allocs),
        sampled_bytes=sum(
            (UInt128(allocation.size) for allocation in profile.allocs);
            init=UInt128(0),
        ),
        metrics=team.result.metrics,
        top,
    )
    println(JSON3.write(report))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
