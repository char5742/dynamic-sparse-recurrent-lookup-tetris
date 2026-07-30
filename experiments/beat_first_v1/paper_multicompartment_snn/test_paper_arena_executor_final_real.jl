using Lux
using Random
using Test

include(joinpath(@__DIR__, "LoadPaperArenaCanonical.jl"))
for filename in (
    "PaperArenaExecutorFinal.jl",
    "PaperArenaExecutorFinalHotfix.jl",
    "PaperArenaExecutorFinalBindings.jl",
)
    Base.include(
        Main.PaperArenaTrainingFinal,
        joinpath(@__DIR__, filename),
    )
end

const PEFR = Main.PaperArenaTrainingFinal
const PMFR = Main.PaperModelCanonical
const CORER = Main.BeatFirstTrainingCore
const DEFAULT_REAL_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const REAL_DATASET = abspath(get(
    ENV,
    "HDSWSNN_TWINPROP_DATASET",
    DEFAULT_REAL_DATASET,
))

function real_training_rows(dataset)
    if hasproperty(dataset, :predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) || return Int.(rows)
    end
    return collect(eachindex(dataset.action_counts))
end

function maximum_parameter_difference(left, right)
    result = 0.0f0
    for name in keys(left)
        a = getproperty(left, name)
        b = getproperty(right, name)
        result = max(result, maximum(abs, a .- b))
    end
    return result
end

@testset "real teacher serial vs MPMC equivalence and CPU smoke" begin
    if Base.Threads.nthreads(:interactive) != 0 ||
       Base.Threads.nthreads(:default) < 2 ||
       !isdir(REAL_DATASET)
        @test_skip false
    else
        dataset = CORER.load_teacher_dataset(
            REAL_DATASET;
            max_candidates=CORER.MAX_CANDIDATES,
            allow_partial_dataset=false,
            geometry_cache_max_states=1,
        )
        row = first(real_training_rows(dataset))
        model = PMFR.build_paper_model(:tiny)
        initial, _ = Lux.setup(
            Xoshiro(0x5041504552524541),
            model,
        )
        serial_trainer = PEFR.PaperTrainer(
            model,
            PEFR.Optim.parameter_copy(initial);
            state_batch=1,
            width=80,
            cell_mode=:detailed,
            location_interval=64,
        )
        parallel_trainer = PEFR.PaperTrainer(
            model,
            PEFR.Optim.parameter_copy(initial);
            state_batch=1,
            width=80,
            cell_mode=:detailed,
            location_interval=64,
        )
        workers = min(
            4,
            Base.Threads.nthreads(:default),
        )
        serial_executor = PEFR.PaperExecutorFinal(
            serial_trainer,
            dataset;
            active_workers=workers,
            stochastic_routing=false,
            cpuset_mode=:none,
        )
        parallel_executor = PEFR.PaperExecutorFinal(
            parallel_trainer,
            dataset;
            active_workers=workers,
            stochastic_routing=false,
            cpuset_mode=:none,
        )

        # First update compiles the complete detailed forward/replay path.
        # The second update is the hot allocation/GC measurement.
        for _ in 1:2
            serial_trainer.tape.base.rows[1] = row
            PEFR.paper_arena_update_serial_final!(
                serial_executor,
            )
        end
        team_result = PEFR.run_with_paper_team!(
            parallel_executor,
        ) do running
            PEFR.paper_arena_update!(running)
            parallel_trainer.tape.base.rows[1] = row
            PEFR.paper_arena_update!(running)
            PEFR.paper_final_phase_snapshot(running)
        end
        snapshot = team_result.result

        @test serial_trainer.optimizer.step == 2
        @test parallel_trainer.optimizer.step == 2
        @test serial_trainer.last_loss.composite_loss ≈
            parallel_trainer.last_loss.composite_loss atol=1.0f-7 rtol=1.0f-7
        @test maximum(
            abs,
            serial_trainer.tape.base.raw .-
            parallel_trainer.tape.base.raw,
        ) <= 1.0f-7
        @test maximum_parameter_difference(
            serial_trainer.parameters,
            parallel_trainer.parameters,
        ) <= 2.0f-7
        @test parallel_trainer.metrics.allocation_bytes <= 16_384
        @test parallel_trainer.metrics.gc_seconds == 0.0
        @test snapshot.process_cpu_utilization > 0.0
        @test count(worker -> worker.jobs > 0, snapshot.per_worker) >= 2
        @test snapshot.phases.pack.jobs ==
            parallel_trainer.tape.base.valid_count
        @test snapshot.phases.forward.jobs ==
            parallel_trainer.tape.base.valid_count
        @test snapshot.phases.replay.jobs ==
            parallel_trainer.tape.base.valid_count
        @test length(team_result.bindings) ==
            Base.Threads.nthreads(:default)
        @test length(PEFR._WORKER_ELIGIBILITY) == 0
    end
end

