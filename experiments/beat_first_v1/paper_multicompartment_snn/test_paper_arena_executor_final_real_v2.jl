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

const PEFR2 = Main.PaperArenaTrainingFinal
const PMFR2 = Main.PaperModelCanonical
const CORER2 = Main.BeatFirstTrainingCore
const REAL_DATASET2 = abspath(get(
    ENV,
    "HDSWSNN_TWINPROP_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
))

function real_training_rows_v2(dataset)
    if hasproperty(dataset, :predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) || return Int.(rows)
    end
    return collect(eachindex(dataset.action_counts))
end

function maximum_parameter_difference_v2(left, right)
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
       !isdir(REAL_DATASET2)
        @test_skip false
    else
        dataset = CORER2.load_teacher_dataset(
            REAL_DATASET2;
            max_candidates=CORER2.MAX_CANDIDATES,
            allow_partial_dataset=false,
            geometry_cache_max_states=1,
        )
        row = first(real_training_rows_v2(dataset))
        model = PMFR2.build_paper_model(:tiny)
        initial, _ = Lux.setup(
            Xoshiro(0x5041504552524541),
            model,
        )
        serial_trainer = PEFR2.PaperTrainer(
            model,
            PEFR2.Optim.parameter_copy(initial);
            state_batch=1,
            width=80,
            cell_mode=:detailed,
            location_interval=64,
        )
        parallel_trainer = PEFR2.PaperTrainer(
            model,
            PEFR2.Optim.parameter_copy(initial);
            state_batch=1,
            width=80,
            cell_mode=:detailed,
            location_interval=64,
        )
        workers = min(
            4,
            Base.Threads.nthreads(:default),
        )
        serial_executor = PEFR2.PaperExecutorFinal(
            serial_trainer,
            dataset;
            active_workers=workers,
            stochastic_routing=false,
            cpuset_mode=:none,
        )
        parallel_executor = PEFR2.PaperExecutorFinal(
            parallel_trainer,
            dataset;
            active_workers=workers,
            stochastic_routing=false,
            cpuset_mode=:none,
        )
        serial_trainer.tape.base.rows[1] = row
        parallel_trainer.tape.base.rows[1] = row

        for _ in 1:2
            serial_trainer.tape.base.rows[1] = row
            PEFR2.paper_arena_update_serial_final!(
                serial_executor,
            )
        end
        team_result = PEFR2.run_with_paper_team!(
            parallel_executor,
        ) do running
            PEFR2.paper_arena_update!(running)
            parallel_trainer.tape.base.rows[1] = row
            PEFR2.paper_arena_update!(running)
            PEFR2.paper_final_phase_snapshot(running)
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
        @test maximum_parameter_difference_v2(
            serial_trainer.parameters,
            parallel_trainer.parameters,
        ) <= 2.0f-7
        @test parallel_trainer.metrics.allocation_bytes <= 16_384
        @test parallel_trainer.metrics.gc_seconds == 0.0
        @test snapshot.process_cpu_utilization > 0.0
        @test count(
            worker -> worker.jobs > 0,
            snapshot.per_worker,
        ) >= 2
        @test snapshot.phases.pack.jobs ==
            parallel_trainer.tape.base.valid_count
        @test snapshot.phases.forward.jobs ==
            parallel_trainer.tape.base.valid_count
        @test snapshot.phases.replay.jobs ==
            parallel_trainer.tape.base.valid_count
        @test length(team_result.bindings) ==
            Base.Threads.nthreads(:default)
        @test length(PEFR2._WORKER_ELIGIBILITY) == 0
    end
end

