using Test
using LinearAlgebra
using Lux
using Random

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV3.jl",
))
include(joinpath(@__DIR__, "..", "training", "core.jl"))

using .BeatFirstTrainingCore

const ReleaseProduction = Main.HDSWSNNTwinPropProduction
const ReleaseArena = ReleaseProduction.Training
const ReleaseModel = Main.PaperModelCanonical

const INTEGRATION_TEACHER_MANIFEST = get(
    ENV,
    "PAPER_OFFICIAL_TEACHER_MANIFEST",
    "",
)
const INTEGRATION_FROZEN_TWIN = get(
    ENV,
    "PAPER_RELEASE_FROZEN_TWIN",
    "",
)
const INTEGRATION_DISTILLED_CELL = get(
    ENV,
    "PAPER_RELEASE_TEST_ARTIFACT",
    "",
)
const INTEGRATION_TETRIS_DATASET = get(
    ENV,
    "PAPER_RELEASE_TEST_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
)

function require_finalproduction_inputs()
    for (label, path, predicate) in (
        (
            "official teacher manifest",
            INTEGRATION_TEACHER_MANIFEST,
            isfile,
        ),
        ("frozen official ELM twin", INTEGRATION_FROZEN_TWIN, isfile),
        (
            "genuine release-v2 distilled cell",
            INTEGRATION_DISTILLED_CELL,
            isfile,
        ),
        ("Tetris teacher_v3 dataset", INTEGRATION_TETRIS_DATASET, isdir),
    )
        !isempty(path) && predicate(path) || error(
            "FAIL-CLOSED: $label is absent: $path",
        )
    end
end

function build_finalproduction_test_trainer(bundle)
    model = ReleaseModel.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x48445357534e4e4d)),
        model,
    )
    trainer = ReleaseProduction.build_production_trainer(
        bundle,
        model,
        parameters;
        state_batch=1,
        width=208,
        learning_rate=5.0f-4,
        weight_decay=1.0f-5,
        location_interval=128,
    )
    return trainer
end

function run_finalproduction_integration()
    require_finalproduction_inputs()
    BLAS.set_num_threads(1)
    @testset "single production type universe" begin
        @test ReleaseArena ===
            Main.PaperArenaTrainingFinalProduction
        @test ReleaseArena.Distilled ===
            Main.DistilledElevenStateCellFinal
        @test ReleaseArena.ReleaseCell.Final ===
            Main.DistilledElevenStateCellFinal
        @test isdefined(ReleaseArena, :PaperExecutorFinal)
        @test isbitstype(ReleaseArena.PaperFinalWorkItem)
        @test which(
            ReleaseArena.paper_arena_update!,
            Tuple{ReleaseArena.PaperExecutorFinal},
        ).file == Symbol(joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalBindings.jl",
        ))
    end

    bundle = ReleaseProduction.load_production_bundle(
        INTEGRATION_TEACHER_MANIFEST,
        INTEGRATION_FROZEN_TWIN,
        INTEGRATION_DISTILLED_CELL;
        verify_teacher_shards=true,
    )
    ReleaseProduction.assert_production_bundle_unchanged!(bundle)
    serial_trainer =
        build_finalproduction_test_trainer(bundle)
    parallel_trainer =
        build_finalproduction_test_trainer(bundle)
    serial_aux =
        ReleaseArena.register_paper_trainer_aux!(serial_trainer)
    parallel_aux =
        ReleaseArena.register_paper_trainer_aux!(parallel_trainer)

    @testset "release trainer contract" begin
        @test serial_aux isa ReleaseArena.PaperReleaseAux
        @test parallel_aux isa ReleaseArena.PaperReleaseAux
        @test eltype(serial_aux.input_location) === UInt16
        @test serial_aux.location_catalog == UInt16.(1:642)
        @test size(
            serial_aux.internal_parameters.compartment_projection,
        ) == (4, 642)
        @test Set(keys(serial_trainer.parameters)) == Set((
            :input_conductance,
            :recurrent_conductance,
            :workspace_conductance,
            :query_weight,
            :workspace_key,
            :workspace_decay_logit,
            :head_weight,
            :head_bias,
            :output_weight,
            :output_bias,
        ))
        @test !(:internal_parameters in
            keys(serial_trainer.optimizer.first_moment))
        @test ReleaseArena.paper_preflight_integrity!(
            serial_trainer,
        ) == bundle.distilled_parameter_sha256
    end

    dataset = load_teacher_dataset(
        INTEGRATION_TETRIS_DATASET;
        max_candidates=208,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    training_rows =
        findall(==(:train), dataset.predefined_split)
    _, local_index =
        findmin(dataset.action_counts[training_rows])
    chosen = training_rows[local_index]
    ReleaseArena.paper_training_arena(
        serial_trainer,
    ).rows[1] = chosen
    ReleaseArena.paper_training_arena(
        parallel_trainer,
    ).rows[1] = chosen

    serial_executor = ReleaseArena.PaperExecutorFinal(
        serial_trainer,
        dataset;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=UInt64(0x48445357534e4e52),
    )
    parallel_executor = ReleaseArena.PaperExecutorFinal(
        parallel_trainer,
        dataset;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=UInt64(0x48445357534e4e52),
    )
    @test all(
        worker -> worker.runtime isa
            ReleaseArena.ReleaseCellRuntime,
        parallel_executor.workers,
    )
    serial_hash =
        ReleaseArena.paper_internal_parameter_sha256(
            serial_trainer,
        )
    parallel_hash =
        ReleaseArena.paper_internal_parameter_sha256(
            parallel_trainer,
        )

    ReleaseArena.paper_arena_update_serial_final!(
        serial_executor,
    )
    ReleaseArena.run_with_paper_team!(
        parallel_executor,
    ) do running
        ReleaseArena.paper_arena_update!(running)
    end

    @testset "one-update serial/MPMC equivalence" begin
        @test serial_trainer.optimizer.step == 1
        @test parallel_trainer.optimizer.step == 1
        @test isfinite(
            serial_trainer.last_loss.composite_loss,
        )
        @test isapprox(
            serial_trainer.last_loss.composite_loss,
            parallel_trainer.last_loss.composite_loss;
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        @test isapprox(
            ReleaseArena.paper_arena_output(serial_trainer),
            ReleaseArena.paper_arena_output(parallel_trainer);
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        for name in keys(serial_trainer.parameters)
            @test isapprox(
                getproperty(serial_trainer.parameters, name),
                getproperty(parallel_trainer.parameters, name);
                rtol=2.0e-4,
                atol=2.0e-6,
            )
        end
        @test ReleaseArena.paper_checkpoint_integrity!(
            serial_trainer,
        ) == serial_hash
        @test ReleaseArena.paper_checkpoint_integrity!(
            parallel_trainer,
        ) == parallel_hash
        @test ReleaseArena.paper_internal_max_delta(
            serial_trainer,
        ) == 0.0f0
        @test ReleaseArena.paper_internal_max_delta(
            parallel_trainer,
        ) == 0.0f0
    end
    @test ReleaseArena.paper_end_run_integrity!(
        serial_trainer,
    ) == serial_hash
    @test ReleaseArena.paper_end_run_integrity!(
        parallel_trainer,
    ) == parallel_hash
end

run_finalproduction_integration()
