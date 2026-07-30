using Test
using JLD2
using LinearAlgebra
using Lux
using Random
using Serialization
using SHA

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropPinnedV5CanonicalFinal.jl",
))

const PinnedArena = Main.HD_RELEASE_V5_FINAL_ARENA
const PinnedCell = PinnedArena.ReleasePinnedCell
const FrozenCell = Main.DistilledElevenStateCellFinal
const PaperModelV6 = Main.PaperModelCanonical
const SEALED_ATTESTATION =
    repeat("9", 64)

function _fixture_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _fixture_parameters()
    compartment_projection = zeros(Float32, 4, 642)
    @inbounds for location in 1:642
        compartment_projection[mod1(location, 4), location] = 1.0f0
    end
    return FrozenCell.DistilledParameters(
        dt_ms=1.0f0,
        transition_decay=fill(0.8f0, 11),
        recurrent_weight=zeros(Float32, 11, 11),
        input_weight=zeros(Float32, 11, 16),
        transition_bias=zeros(Float32, 11),
        readout_weight=zeros(Float32, 11, 11),
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection,
        region_projection=Matrix{Float32}(I, 4, 4),
        spike_threshold=0.75f0,
        teacher_schema=PinnedCell.SEALED_RELEASE_SCHEMA,
        detailed_kernel_hash=repeat("1", 64),
        morphology_hash=repeat("2", 64),
        frozen_twin_parameter_hash=repeat("3", 64),
        frozen_twin_artifact_hash=repeat("4", 64),
        distillation_dataset_hash=repeat("5", 64),
        distillation_config_hash=repeat("6", 64),
    )
end

function _fixture_payload(; sealed_v2::Bool)
    parameters = _fixture_parameters()
    source_catalog_sha256 = repeat("7", 64)
    official_segment_region = Tuple(
        isodd(location) ? "basal" : "apical"
        for location in 1:642
    )
    location_mapping_sha256 = _fixture_sha256((
        source_catalog_sha256,
        official_segment_region,
        parameters.compartment_projection,
    ))
    test_indices_sha256 = repeat("d", 64)
    base = (;
        schema=FrozenCell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=FrozenCell.parameter_sha256(parameters),
        frozen_internal=true,
        ablation_mode=:full,
        training_input_mode="sharded_streaming_v2",
        dense_full_memory_path=false,
        provisional=false,
        prepared_dataset_schema=
            "hd-swsnn-twinprop-distillation-dataset-sharded-release-v2",
        paper_scale=false,
        official_segment_count=642,
        official_segment_region,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256,
        source_segment_catalog_sha256,
        semantic_coordinate_gate=(;
            passed=true,
            coordinate_names=PinnedCell.SEMANTIC_COORDINATE_NAMES,
            per_coordinate_passed=fill(true, 11),
        ),
        structured_transition_contract=(;
            structured_readout=true,
            coordinate_wise_semantic_supervision=true,
            dense_rotational_hidden_basis=false,
            semantic_state_scale="normalized_unit_interval",
            location_index_type="UInt16",
        ),
        metrics=(;
            test=(;
                spike_auroc=0.999,
                auroc_is_conservative_lower_bound=true,
                evaluation_split="test",
                exact_dataset_split=true,
                evaluated_indices_sha256=test_indices_sha256,
                evaluated_index_count=1,
                sparse_auxiliary_metrics_use_observed_times_only=true,
                interpolated_auxiliary_values_excluded_from_gate=true,
            ),
        ),
        gate=(;
            gate_schema="hd_swsnn.eleven_state.strict_gate.final.v2",
            passed=true,
            per_region_and_branch_gating=true,
            spike_passed=true,
            minimum_spike_auroc=0.985,
            multi_target_passed=true,
            voltage_passed=true,
            nmda_passed=true,
            nmda_region_passed=fill(true, 4),
            calcium_passed=true,
            dendritic_voltage_passed=true,
            dendritic_branch_passed=fill(true, 4),
            semantic_state_passed=true,
            semantic_coordinate_passed=fill(true, 11),
            observation_counts_passed=true,
        ),
        detailed_kernel_hash=parameters.detailed_kernel_hash,
        morphology_hash=parameters.morphology_hash,
        frozen_twin_parameter_hash=
            parameters.frozen_twin_parameter_hash,
        frozen_twin_artifact_hash=
            parameters.frozen_twin_artifact_hash,
        distillation_dataset_hash=
            parameters.distillation_dataset_hash,
        distillation_config_hash=
            parameters.distillation_config_hash,
    )
    sealed_v2 || return base
    return merge(
        base,
        (;
            official_training_input_mode=
                "signed_1278_sealed_v2_neuronio_windows_v1",
            official_elm_input_dim=1_278,
            sealed_execution_type=PinnedCell.SEALED_RELEASE_TYPE,
            sealed_release_schema=PinnedCell.SEALED_RELEASE_SCHEMA,
            sealed_release_artifact_kind=
                PinnedCell.SEALED_RELEASE_ARTIFACT_KIND,
            legacy_3852_twin_fallback=false,
            sealed_attestation_sha256=SEALED_ATTESTATION,
            source_dataset_sha256=repeat("a", 64),
            source_bound_sealed_elm=(;
                source_manifest_sha256=repeat("b", 64),
                source_teacher_contract_sha256=repeat("c", 64),
                parameter_sha256=repeat("d", 64),
                base_artifact_sha256=repeat("e", 64),
                sealed_attestation_sha256=SEALED_ATTESTATION,
                sealed_release_schema=PinnedCell.SEALED_RELEASE_SCHEMA,
                sealed_execution_type=PinnedCell.SEALED_RELEASE_TYPE,
                executable_mlp_activation="tanh",
                compatibility_profile="official-1278-v2",
            ),
            primary_frozen_twin_targets=(;
                cache_verified_all_samples=true,
                bit_exact=true,
                measurement_schema=
                    "hd_swsnn.distillation.primary_cache_live_replay.final.v2",
                measurement_sha256=repeat("f", 64),
                official_elm_input_dim=1_278,
                sealed_execution_type=PinnedCell.SEALED_RELEASE_TYPE,
                sealed_release_schema=PinnedCell.SEALED_RELEASE_SCHEMA,
                sealed_release_artifact_kind=
                    PinnedCell.SEALED_RELEASE_ARTIFACT_KIND,
                sealed_attestation_sha256=SEALED_ATTESTATION,
                soma_voltage_max_delta=0.0,
                spike_probability_max_delta=0.0,
                spike_logit_max_delta=0.0,
                nmda_max_delta=0.0,
            ),
            detailed_model_auxiliary_state_targets=(;
                primary_teacher=false,
                sparse_observation_grid=true,
            ),
            neuronio_window_contract=(;
                full_time_steps=1_500,
                training_ignore_steps=500,
                training_window_steps=500,
                training_start_range=(501, 1_000),
                training_sampling="uniform_with_replacement",
                heldout_burnin_steps=500,
                heldout_evaluation_range=(501, 1_500),
            ),
            split_identity=(;
                train=(;
                    count=1,
                    indices_sha256=repeat("1", 64),
                ),
                validation=(;
                    count=1,
                    indices_sha256=repeat("2", 64),
                ),
                test=(;
                    count=1,
                    indices_sha256=test_indices_sha256,
                ),
            ),
            evaluation_protocol=(;
                candidate_restarts=1,
                validation_evaluations=1,
                test_evaluations=1,
                selection_split="validation",
                final_gate_split="test",
            ),
        ),
    )
end

function _write_fixture(path; sealed_v2::Bool)
    JLD2.jldsave(path; payload=_fixture_payload(; sealed_v2))
    return path
end

function _fixture_trainer(path)
    model = PaperModelV6.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x52554e54494d4556)),
        model,
    )
    return PinnedArena.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=208,
        location_interval=128,
        cell_mode=:distilled_frozen,
        cell_artifact=path,
    )
end

@testset "pinned V5 artifact / RuntimeV6 minimal boundary" begin
    @test PinnedArena.paper_release_adapter_contract_version() == 6
    @test PinnedArena.RELEASE_REQUIRED_ARTIFACT_RUNTIME ===
        :pinned_v5_artifact_v6_verifier
    @test PinnedCell.SEALED_RELEASE_SCHEMA ==
        "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test PinnedCell.SEALED_RELEASE_ARTIFACT_KIND ==
        "SealedOfficialELMReleaseV2"
    @test which(
        PinnedArena.enable_release_runtime!,
        Tuple{PinnedArena.PaperTrainer,AbstractString},
    ).file == Symbol(joinpath(
        @__DIR__,
        "PaperArenaReleaseAdapterV6PinnedFinal.jl",
    ))

    mktempdir() do directory
        sealed_path = _write_fixture(
            joinpath(directory, "sealed_v2_v5.jld2");
            sealed_v2=true,
        )
        legacy_path = _write_fixture(
            joinpath(directory, "legacy_v3.jld2");
            sealed_v2=false,
        )
        artifact_sha256 = bytes2hex(SHA.sha256(read(sealed_path)))
        legacy_sha256 = bytes2hex(SHA.sha256(read(legacy_path)))

        runtime = PinnedCell.load_release_runtime(
            sealed_path;
            expected_artifact_sha256=artifact_sha256,
            expected_sealed_attestation_sha256=
                SEALED_ATTESTATION,
        )
        @test runtime isa PinnedCell.TrustedReleaseRuntime
        @test runtime.paper_scale === false
        @test PinnedCell.preflight_integrity!(runtime) ==
            runtime.expected_parameter_sha256
        @test_throws ErrorException PinnedCell.load_release_runtime(
            sealed_path;
            expected_artifact_sha256=repeat("0", 64),
            expected_sealed_attestation_sha256=
                SEALED_ATTESTATION,
        )
        @test_throws ErrorException PinnedCell.load_release_runtime(
            sealed_path;
            expected_artifact_sha256=artifact_sha256,
            expected_sealed_attestation_sha256=repeat("8", 64),
        )
        @test_throws ErrorException PinnedCell.load_release_runtime(
            legacy_path;
            expected_artifact_sha256=legacy_sha256,
            expected_sealed_attestation_sha256=
                SEALED_ATTESTATION,
        )

        state = PinnedCell.release_new_state(runtime)
        drive = PinnedCell.release_new_drive(runtime)
        diagnostics = PinnedCell.release_new_diagnostics(runtime)
        spike = PinnedCell.trusted_cell_step!(
            runtime,
            state,
            drive,
            diagnostics,
        )
        @test diagnostics.spike_probability == 0.5f0
        @test spike == 0.0f0
        @test spike == diagnostics.soma_spike
        @test spike != diagnostics.spike_probability

        trainer = _fixture_trainer(sealed_path)
        @test_throws UndefKeywordError PinnedArena.enable_development_release_runtime!(
            trainer,
            sealed_path;
            development_scale_chain=true,
        )
        @test_throws ErrorException PinnedArena.register_paper_trainer_aux!(
            trainer,
        )
    end
end
