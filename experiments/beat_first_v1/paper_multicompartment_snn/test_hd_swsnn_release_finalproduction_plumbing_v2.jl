# Corrected executable plumbing test.  Reuse only the fixture/helper prefix
# from v1; its test body is intentionally not evaluated.
using Test

const _PLUMBING_V1 = joinpath(
    @__DIR__,
    "test_hd_swsnn_release_finalproduction_plumbing.jl",
)
const _PLUMBING_SOURCE = read(_PLUMBING_V1, String)
const _PLUMBING_BOUNDARY =
    findfirst("@testset \"FinalProduction release plumbing\"",
        _PLUMBING_SOURCE)
_PLUMBING_BOUNDARY === nothing &&
    error("plumbing helper boundary is absent")
include_string(
    Main,
    _PLUMBING_SOURCE[
        firstindex(_PLUMBING_SOURCE):
        prevind(_PLUMBING_SOURCE, first(_PLUMBING_BOUNDARY))
    ],
    _PLUMBING_V1 * ":helpers",
)

@testset "FinalProduction release plumbing v2" begin
    @test PlumbingArena ===
        Main.HDSWSNNTwinPropProduction.Training
    @test PlumbingArena.ReleaseCell.Final === PlumbingCell
    @test which(
        PlumbingArena.paper_arena_update!,
        Tuple{PlumbingArena.PaperExecutorFinal},
    ).file == Symbol(joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalBindings.jl",
    ))

    mktempdir() do directory
        artifact = _write_plumbing_artifact(
            joinpath(directory, "synthetic_release_v2.jld2"),
        )
        trainer = _plumbing_trainer(artifact)
        aux = PlumbingArena.register_paper_trainer_aux!(trainer)
        @test aux isa PlumbingArena.PaperReleaseAux
        @test eltype(aux.input_location) === UInt16
        @test aux.location_catalog == UInt16.(1:642)
        @test PlumbingArena.paper_preflight_integrity!(trainer) ==
            aux.trusted.expected_parameter_sha256

        dataset = load_teacher_dataset(
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3";
            max_candidates=208,
            allow_partial_dataset=false,
            geometry_cache_max_states=1,
        )
        rows = findall(==(:train), dataset.predefined_split)
        _, local_index =
            findmin(dataset.action_counts[rows])
        trainer.tape.base.rows[1] = rows[local_index]
        executor = PlumbingArena.PaperExecutorFinal(
            trainer,
            dataset;
            active_workers=min(4, Threads.nthreads(:default)),
            cpuset_mode=:none,
            stochastic_routing=true,
            routing_seed=UInt64(0x48445357534e4e52),
        )
        @test all(
            worker -> worker.runtime isa
                PlumbingArena.ReleaseCellRuntime,
            executor.workers,
        )
        before =
            PlumbingArena.paper_internal_parameter_sha256(
                trainer,
            )
        PlumbingArena.run_with_paper_team!(executor) do running
            PlumbingArena.paper_arena_update!(running)
        end
        @test trainer.optimizer.step == 1
        @test isfinite(trainer.last_loss.composite_loss)
        @test PlumbingArena.paper_checkpoint_integrity!(
            trainer,
        ) == before
        @test PlumbingArena.paper_internal_max_delta(trainer) ==
            0.0f0
        @test PlumbingArena.paper_end_run_integrity!(trainer) ==
            before
        output = PlumbingArena.paper_arena_output(trainer)
        @test all(
            all(isfinite, value)
            for value in values(output)
        )
    end
end
