# Plumbing-only integration test.
#
# The synthetic cell below is deliberately NOT an accepted research artifact.
# Its purpose is to exercise the exact FinalProduction type universe,
# UInt16/642 adapter, release replay, and final MPMC executor while the genuine
# official ELM/distillation artifacts are still being produced.  The canonical
# artifact-gated test remains `test_hd_swsnn_release_finalproduction_integration.jl`.

using Test
using JLD2
using LinearAlgebra
using Lux
using Random
using Serialization
using SHA

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV3.jl",
))
include(joinpath(@__DIR__, "..", "training", "core.jl"))

using .BeatFirstTrainingCore

const PlumbingArena =
    Main.PaperArenaTrainingFinalProduction
const PlumbingCell =
    Main.DistilledElevenStateCellFinal
const PlumbingModel = Main.PaperModelCanonical

function _plumbing_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _plumbing_parameters()
    projection = zeros(Float32, 4, 642)
    @inbounds for location in 1:642
        projection[mod1(location, 4), location] = 1.0f0
    end
    region_projection = Matrix{Float32}(I, 4, 4)
    recurrent = zeros(Float32, 11, 11)
    readout = zeros(Float32, 11, 11)
    @inbounds for coordinate in 1:11
        recurrent[coordinate, coordinate] = 0.10f0
        readout[coordinate, coordinate] = 1.0f0
    end
    input = zeros(Float32, 11, 16)
    @inbounds for branch in 1:4
        input[branch, branch] = 0.20f0
        input[4 + branch, 4 + branch] = 0.20f0
    end
    return PlumbingCell.DistilledParameters(
        dt_ms=1.0f0,
        transition_decay=fill(0.8f0, 11),
        recurrent_weight=recurrent,
        input_weight=input,
        transition_bias=zeros(Float32, 11),
        readout_weight=readout,
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection=projection,
        region_projection=region_projection,
        spike_threshold=0.5f0,
        teacher_schema=
            "hd_swsnn_twinprop.elm_frozen.synthetic_plumbing",
        detailed_kernel_hash=repeat("1", 64),
        morphology_hash=repeat("2", 64),
        frozen_twin_parameter_hash=repeat("3", 64),
        frozen_twin_artifact_hash=repeat("4", 64),
        distillation_dataset_hash=repeat("5", 64),
        distillation_config_hash=repeat("6", 64),
    )
end

function _write_plumbing_artifact(path)
    parameters = _plumbing_parameters()
    parameter_hash = PlumbingCell.parameter_sha256(parameters)
    source_catalog = repeat("7", 64)
    regions = Tuple(
        isodd(location) ? "basal" : "apical"
        for location in 1:642
    )
    mapping_hash = _plumbing_sha256((
        source_catalog,
        regions,
        parameters.compartment_projection,
    ))
    semantic_names =
        PlumbingArena.ReleaseCell.SEMANTIC_COORDINATE_NAMES
    payload = (;
        schema=PlumbingCell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        ablation_mode=:full,
        official_segment_count=642,
        official_segment_region=regions,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256=mapping_hash,
        source_segment_catalog_sha256=source_catalog,
        semantic_coordinate_gate=(;
            passed=true,
            coordinate_names=semantic_names,
            per_coordinate_passed=fill(true, 11),
        ),
        structured_transition_contract=(;
            structured_readout=true,
            coordinate_wise_semantic_supervision=true,
            dense_rotational_hidden_basis=false,
        ),
        metrics=(;
            test=(; spike_auroc=0.999),
        ),
        gate=(;
            passed=true,
            minimum_spike_auroc=0.985,
            multi_target_passed=true,
            voltage_passed=true,
            nmda_passed=true,
            calcium_passed=true,
            dendritic_voltage_passed=true,
            semantic_state_passed=true,
        ),
        teacher_hash=repeat("8", 64),
        detailed_kernel_hash=parameters.detailed_kernel_hash,
        cell_mechanism_sha256=parameters.detailed_kernel_hash,
        morphology_hash=parameters.morphology_hash,
        digital_twin_hash=
            parameters.frozen_twin_artifact_hash,
        frozen_twin_parameter_hash=
            parameters.frozen_twin_parameter_hash,
        frozen_twin_artifact_hash=
            parameters.frozen_twin_artifact_hash,
        distillation_dataset_hash=
            parameters.distillation_dataset_hash,
        distillation_config_hash=
            parameters.distillation_config_hash,
        config=(; dt_ms=1.0),
    )
    JLD2.jldsave(path; payload)
    return path
end

function _plumbing_trainer(artifact)
    model = PlumbingModel.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x48445357534e4e4d)),
        model,
    )
    trainer = PlumbingArena.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=208,
        learning_rate=5.0f-4,
        weight_decay=1.0f-5,
        location_interval=128,
        cell_mode=:distilled_frozen,
        cell_artifact=artifact,
    )
    PlumbingArena.enable_release_runtime!(trainer, artifact)
    return trainer
end

@testset "FinalProduction release plumbing" begin
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
        @test all(isfinite, PlumbingArena.paper_arena_output(trainer))
    end
end
