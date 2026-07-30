using JLD2
using Lux
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_release.jl",
))

@testset "standalone release distiller strict smoke" begin
    rng = Xoshiro(0x64211)
    twin_config = ReleaseTwin.TwinConfig(
        segments=RELEASE_SEGMENTS,
        nmda_regions=4,
        memory_units=16,
        core_dim=8,
        dt_ms=1.0f0,
    )
    twin_model =
        ReleaseTwin.build_paper_twin(twin_config; input_density=0.01)
    twin_parameters, _ = Lux.setup(rng, twin_model)
    twin_normalizer = ReleaseTwin.TwinNormalizer(
        zeros(Float32, twin_config.input_dim),
        ones(Float32, twin_config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = ReleaseTwin.freeze_twin(
        twin_model,
        twin_parameters,
        twin_normalizer;
        metadata=(;
            held_out_metrics=(; spike_auroc=0.99),
            gate=(; passed=true),
            detailed_teacher_hash=repeat("1", 64),
            morphology_hash=repeat("2", 64),
        ),
    )
    input = 0.05f0 .*
        rand(rng, Float32, twin_config.input_dim, 4, 6)
    prediction = ReleaseTwin.twin_forward(frozen, input)
    calcium = Float32.(
        reshape(collect(0:23), 4, 6) .% 3 .== 0,
    )
    dendritic =
        -65.0f0 .+ 4.0f0 .* randn(rng, Float32, 4, 4, 6)
    segment_region = Vector{String}(undef, RELEASE_SEGMENTS)
    segment_region[1] = "soma"
    segment_region[2:321] .= "basal"
    segment_region[322:500] .= "apical_trunk"
    segment_region[501:end] .= "apical_tuft"

    mktempdir() do directory
        twin_path = joinpath(directory, "frozen_twin.jld2")
        ReleaseTwin.save_frozen_twin(twin_path, frozen)
        twin_file_hash = _release_file_sha256(twin_path)
        dataset_path = joinpath(directory, "prepared.jld2")
        metadata = (;
            source_kind="official_neuron",
            official_neuron_source=true,
            official_neuron_schema=RELEASE_OFFICIAL_TEACHER_SCHEMA,
            source_completion_state="complete",
            mixed_supervision=true,
            twin_gate=(;
                passed=true,
                minimum_spike_auroc=0.985,
            ),
            twin_held_out_metrics=(; spike_auroc=0.99),
            segment_region,
        )
        dataset = (;
            schema=RELEASE_DATASET_SCHEMA,
            input,
            target_voltage=prediction.voltage,
            target_spike=prediction.spike_probability,
            target_nmda=prediction.nmda,
            target_calcium_event=calcium,
            target_dendritic_voltage=dendritic,
            train_indices=Int32[1, 2],
            validation_indices=Int32[3, 4],
            test_indices=Int32[5, 6],
            mixed_supervision=true,
            frozen_twin_parameter_hash=frozen.parameter_sha256,
            frozen_twin_artifact_hash=frozen.artifact_sha256,
            frozen_twin_file_sha256=twin_file_hash,
            detailed_teacher_hash=repeat("3", 64),
            detailed_kernel_hash=repeat("4", 64),
            morphology_hash=repeat("5", 64),
            official_modeldb_source_hash=repeat("6", 64),
            official_teacher_file_hash=repeat("7", 64),
            original_dataset_sha256=repeat("8", 64),
            source_manifest_sha256=repeat("9", 64),
            segment_catalog_sha256=repeat("a", 64),
            segment_region,
            metadata,
        )
        JLD2.jldsave(dataset_path; dataset)

        loaded_frozen = ReleaseTwin.load_frozen_twin(twin_path)
        loaded = _release_load_dataset(dataset_path, loaded_frozen)
        live = _release_twin_inference(
            loaded_frozen,
            loaded.input;
            batch_size=2,
        )
        cache = _release_cache_check(loaded, live)
        @test maximum((
            cache.voltage_error,
            cache.spike_error,
            cache.nmda_error,
        )) <= cache.tolerance
        target = _release_targets(loaded, live)
        target_mean, target_scale =
            _release_target_statistics(target, loaded.train_indices)
        parameters =
            _release_initial_parameters(rng, RELEASE_SEGMENTS)
        loss_teacher = _release_sequence_loss(
            parameters,
            loaded.input[:, :, loaded.train_indices],
            target[:, :, loaded.train_indices],
            target_mean,
            target_scale,
            RELEASE_SEGMENTS,
            0.0f0,
        )
        loss_free = _release_sequence_loss(
            parameters,
            loaded.input[:, :, loaded.train_indices],
            target[:, :, loaded.train_indices],
            target_mean,
            target_scale,
            RELEASE_SEGMENTS,
            1.0f0,
        )
        @test isfinite(loss_teacher)
        @test isfinite(loss_free)
        gradient = only(Zygote.gradient(
            candidate -> _release_sequence_loss(
                candidate,
                loaded.input[:, :, loaded.train_indices],
                target[:, :, loaded.train_indices],
                target_mean,
                target_scale,
                RELEASE_SEGMENTS,
                0.5f0,
            ),
            parameters,
        ))
        @test all(isfinite, gradient.recurrent_weight)
        @test sum(abs, gradient.recurrent_weight) > 0
        @test sum(abs, gradient.location_logits) > 0
        @test all(
            iszero,
            gradient.recurrent_weight[
                RELEASE_RECURRENT_MASK .== 0.0f0
            ],
        )
        @test all(
            iszero,
            gradient.input_weight[
                RELEASE_INPUT_MASK .== 0.0f0
            ],
        )
        @test ReleaseTwin.assert_frozen_unchanged(
            loaded_frozen,
        ).max_delta == 0.0f0
    end
end

println("standalone release distillation smoke passed")
