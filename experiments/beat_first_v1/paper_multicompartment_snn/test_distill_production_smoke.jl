using JLD2
using Lux
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_production_v4.jl",
))

@testset "strict mixed-supervision production distiller smoke" begin
    rng = Xoshiro(0xd157111)
    config = Twin.TwinConfig(
        segments=4,
        nmda_regions=4,
        memory_units=16,
        core_dim=8,
        dt_ms=1.0f0,
    )
    model = Twin.build_paper_twin(config; input_density=0.5)
    parameters, _ = Lux.setup(rng, model)
    normalizer = Twin.TwinNormalizer(
        zeros(Float32, config.input_dim),
        ones(Float32, config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = Twin.freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            held_out_spike_auroc=0.99,
            detailed_teacher_hash=repeat("1", 64),
        ),
    )
    input = rand(rng, Float32, config.input_dim, 8, 6)
    prediction = Twin.twin_forward(frozen, input)
    target_ca = Float32.(
        rand(rng, Float32, 8, 6) .> 0.75f0,
    )
    target_dendritic =
        -65.0f0 .+ 5.0f0 .* randn(rng, Float32, 4, 8, 6)

    mktempdir() do directory
        twin_path = joinpath(directory, "frozen_twin.jld2")
        Twin.save_frozen_twin(twin_path, frozen)
        dataset_path = joinpath(directory, "prepared.jld2")
        dataset = (;
            schema=PREPARED_DATASET_SCHEMA,
            mixed_supervision=true,
            digital_twin_gate_passed=true,
            input,
            target_voltage=prediction.voltage,
            target_spike=prediction.spike_probability,
            target_nmda=prediction.nmda,
            target_calcium_event=target_ca,
            target_dendritic_voltage=target_dendritic,
            train_indices=Int32[1, 2],
            validation_indices=Int32[3, 4],
            test_indices=Int32[5, 6],
            segment_region=(
                "soma",
                "basal",
                "apical_trunk",
                "apical_tuft",
            ),
            frozen_twin_parameter_hash=frozen.parameter_sha256,
            frozen_twin_artifact_hash=frozen.artifact_sha256,
            official_neuron_schema=
                "hd_swsnn_twinprop.neuron_teacher.v1",
            detailed_teacher_hash=repeat("2", 64),
            detailed_kernel_hash=repeat("3", 64),
            morphology_hash=repeat("4", 64),
            official_modeldb_source_hash=repeat("5", 64),
            source_dataset_hash=repeat("6", 64),
        )
        JLD2.jldsave(dataset_path; dataset)

        loaded_frozen = Twin.load_frozen_twin(twin_path)
        loaded = _load_prepared_dataset(
            dataset_path,
            loaded_frozen,
        )
        live = _run_frozen_twin(
            loaded_frozen,
            loaded.input;
            chunk_size=2,
        )
        cache_check = _verify_cached_twin!(loaded, live)
        @test cache_check.voltage_error <= cache_check.tolerance
        @test cache_check.spike_error <= cache_check.tolerance
        @test cache_check.nmda_error <= cache_check.tolerance
        @test loaded.provenance.frozen_twin_parameter_hash ==
            frozen.parameter_sha256
        @test loaded.provenance.frozen_twin_artifact_hash ==
            frozen.artifact_sha256

        target = _target_tensor(loaded, live)
        target_mean, target_scale =
            _target_statistics(target, loaded.train_indices)
        training_parameters = _initial_parameters(rng, config.segments)
        input_window = loaded.input[:, 1:4, loaded.train_indices]
        target_window = target[:, 1:4, loaded.train_indices]
        teacher_loss = _sequence_loss(
            training_parameters,
            input_window,
            target_window,
            target_mean,
            target_scale,
            config.segments,
            0.0f0,
        )
        rollout_loss = _sequence_loss(
            training_parameters,
            input_window,
            target_window,
            target_mean,
            target_scale,
            config.segments,
            1.0f0,
        )
        @test isfinite(teacher_loss)
        @test isfinite(rollout_loss)
        gradient = only(Zygote.gradient(
            candidate -> _sequence_loss(
                candidate,
                input_window,
                target_window,
                target_mean,
                target_scale,
                config.segments,
                0.5f0,
            ),
            training_parameters,
        ))
        @test all(isfinite, gradient.recurrent_weight)
        @test sum(abs, gradient.recurrent_weight) > 0
        @test sum(abs, gradient.spatial_logits) > 0
        @test Twin.assert_frozen_unchanged(loaded_frozen).max_delta ==
            0.0f0
    end
end

println("production distillation smoke passed")
