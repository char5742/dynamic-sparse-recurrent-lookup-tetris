using Test
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "PaperModelCanonical.jl"))
include(joinpath(@__DIR__, "LoadPaperArenaReleaseV2.jl"))

using .BeatFirstTrainingCore
using .PaperModelCanonical
using .PaperArenaTrainingFinal

const ReleaseArena = PaperArenaTrainingFinal
const RELEASE_INTEGRATION_DATASET = get(
    ENV,
    "PAPER_RELEASE_TEST_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
)
const RELEASE_INTEGRATION_ARTIFACT = get(
    ENV,
    "PAPER_RELEASE_TEST_ARTIFACT",
    joinpath(
        @__DIR__,
        "artifacts",
        "paper_cell_distilled_release_v2.jld2",
    ),
)

function require_release_inputs_final()
    isfile(RELEASE_INTEGRATION_ARTIFACT) || error(
        "FAIL-CLOSED: genuine release-v2 artifact is absent: " *
        RELEASE_INTEGRATION_ARTIFACT,
    )
    isdir(RELEASE_INTEGRATION_DATASET) || error(
        "real teacher_v3 dataset is absent: " *
        RELEASE_INTEGRATION_DATASET,
    )
end

function release_trainer_final()
    model = build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x48445357534e4e4d)),
        model,
    )
    trainer = PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=208,
        learning_rate=5.0f-4,
        weight_decay=1.0f-5,
        location_interval=128,
        cell_mode=:distilled_frozen,
        cell_artifact=RELEASE_INTEGRATION_ARTIFACT,
    )
    aux = enable_release_runtime!(
        trainer,
        RELEASE_INTEGRATION_ARTIFACT,
    )
    return trainer, aux
end

function run_release_integration_final()
    require_release_inputs_final()
    BLAS.set_num_threads(1)
    trainer, aux = release_trainer_final()

    @testset "release-v2 contract" begin
        @test aux isa ReleaseArena.PaperReleaseAux
        @test eltype(aux.location_catalog) === UInt16
        @test aux.location_catalog == UInt16.(1:642)
        @test eltype(aux.input_location) === UInt16
        @test eltype(aux.recurrent_location) === UInt16
        @test eltype(aux.workspace_location) === UInt16
        @test all(location -> 1 <= location <= 642,
            aux.input_location)
        @test all(location -> 1 <= location <= 642,
            aux.recurrent_location)
        @test all(location -> 1 <= location <= 642,
            aux.workspace_location)
        @test size(
            aux.internal_parameters.compartment_projection,
        ) == (4, 642)
        @test ReleaseArena.ReleaseCell.SEMANTIC_STATE_SCALE ===
            :normalized_unit_interval
        @test paper_preflight_integrity!(trainer) ==
            aux.trusted.expected_parameter_sha256
        @test paper_internal_max_delta(trainer) == 0.0f0
    end

    worker = ReleaseArena.PaperWorker(trainer)
    @testset "trusted hard-spike hot path" begin
        @test worker.runtime isa ReleaseArena.ReleaseCellRuntime
        runtime = worker.runtime
        ReleaseArena.reset_runtime!(runtime)
        ReleaseArena.reset_cell_drive!(runtime, 1)
        ReleaseArena.add_cell_event!(
            runtime,
            1,
            642,
            PaperModelCanonical.EXCITATORY,
            0.5f0,
        )
        ReleaseArena.step_cell!(runtime, 1)
        ReleaseArena.reset_cell_drive!(runtime, 1)
        allocated =
            @allocated ReleaseArena.step_cell!(runtime, 1)
        @test allocated == 0
        @test runtime.diagnostics[1].soma_spike in
            (0.0f0, 1.0f0)
        @test length(runtime.states[1].value) == 11
    end

    dataset = load_teacher_dataset(
        RELEASE_INTEGRATION_DATASET;
        max_candidates=208,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    train_rows = findall(==(:train), dataset.predefined_split)
    _, local_index =
        findmin(dataset.action_counts[train_rows])
    chosen_row = train_rows[local_index]
    @test dataset.action_counts[chosen_row] >= 2
    paper_training_arena(trainer).rows[1] = chosen_row
    executor = PaperExecutor(
        trainer,
        dataset;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=UInt64(0x48445357534e4e52),
    )
    before_parameter_hash =
        paper_internal_parameter_sha256(trainer)
    before_artifact_hash = paper_internal_sha256(trainer)
    run_with_paper_team!(executor) do running
        paper_arena_update!(running)
    end

    @testset "real teacher full update" begin
        @test trainer.optimizer.step == 1
        @test isfinite(trainer.last_loss.composite_loss)
        @test paper_internal_parameter_sha256(trainer) ==
            before_parameter_hash
        @test paper_internal_sha256(trainer) ==
            before_artifact_hash
        @test paper_internal_max_delta(trainer) == 0.0f0
        @test paper_checkpoint_integrity!(trainer) ==
            before_parameter_hash
        snapshot = paper_aux_snapshot(trainer)
        @test snapshot.location_index_type == "UInt16"
        @test snapshot.official_segment_count == 642
        @test snapshot.location_mapping_sha256 ==
            aux.location_mapping_sha256
        @test eltype(snapshot.input_location) === UInt16
        @test all(isfinite, paper_arena_output(trainer))
    end
    @test paper_end_run_integrity!(trainer) ==
        before_parameter_hash
    return trainer
end

run_release_integration_final()
