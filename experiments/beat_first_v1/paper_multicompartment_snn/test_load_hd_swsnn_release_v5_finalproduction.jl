using Test
using JLD2
using LinearAlgebra
using Lux
using Random
using Serialization
using SHA

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5FinalProduction.jl",
))

const ArenaV5 = Main.HD_RELEASE_V5_ARENA
const CellV5 = Main.DistilledElevenStateCellFinal
const ModelV5 = Main.PaperModelCanonical

function _v5_fixture_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _v5_fixture_parameters()
    projection = zeros(Float32, 4, 642)
    @inbounds for location in 1:642
        projection[mod1(location, 4), location] = 1.0f0
    end
    return CellV5.DistilledParameters(
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
        compartment_projection=projection,
        region_projection=Matrix{Float32}(I, 4, 4),
        spike_threshold=0.75f0,
        teacher_schema=
            "hd_swsnn.paper_elm_v2.sealed_release.final.v1",
        detailed_kernel_hash=repeat("1", 64),
        morphology_hash=repeat("2", 64),
        frozen_twin_parameter_hash=repeat("3", 64),
        frozen_twin_artifact_hash=repeat("4", 64),
        distillation_dataset_hash=repeat("5", 64),
        distillation_config_hash=repeat("6", 64),
    )
end

function _v5_fixture_payload(; sealed::Bool)
    parameters = _v5_fixture_parameters()
    source_catalog = repeat("7", 64)
    regions = Tuple(
        isodd(location) ? "basal" : "apical"
        for location in 1:642
    )
    mapping_hash = _v5_fixture_sha256((
        source_catalog,
        regions,
        parameters.compartment_projection,
    ))
    teacher_hash = repeat("8", 64)
    attestation_hash = repeat("9", 64)
    base = (;
        schema=CellV5.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=CellV5.parameter_sha256(parameters),
        frozen_internal=true,
        ablation_mode=:full,
        training_input_mode="sharded_streaming_v2",
        dense_full_memory_path=false,
        provisional=false,
        prepared_dataset_schema=
            "hd-swsnn-twinprop-distillation-dataset-sharded-release-v2",
        paper_scale=false,
        official_segment_count=642,
        official_segment_region=regions,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256=mapping_hash,
        source_segment_catalog_sha256=source_catalog,
        semantic_coordinate_gate=(;
            passed=true,
            coordinate_names=
                ArenaV5.ReleaseSecurityCell.SEMANTIC_COORDINATE_NAMES,
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
            ),
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
        teacher_hash,
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
    sealed || return base
    return merge(
        base,
        (;
            official_training_input_mode=
                "signed_1278_sealed_neuronio_windows_v1",
            official_elm_input_dim=1_278,
            sealed_execution_type=
                "PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease",
            sealed_release_schema=
                "hd_swsnn.paper_elm_v2.sealed_release.final.v1",
            legacy_3852_twin_fallback=false,
            sealed_attestation_sha256=attestation_hash,
            primary_frozen_twin_targets=(;
                cache_verified_all_samples=true,
                bit_exact=true,
                official_elm_input_dim=1_278,
                sealed_execution_type=
                    "PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease",
                soma_voltage_max_delta=0.0,
                spike_probability_max_delta=0.0,
                spike_logit_max_delta=0.0,
                nmda_max_delta=0.0,
                primary_targets=(
                    "soma_voltage",
                    "spike_probability",
                    "spike_logit",
                    "nmda_soma",
                    "nmda_basal",
                    "nmda_apical_trunk",
                    "nmda_apical_tuft",
                ),
                sealed_attestation_sha256=attestation_hash,
            ),
            detailed_model_auxiliary_state_targets=(;
                primary_teacher=false,
                sparse_observation_grid=true,
                targets=(
                    "calcium_event_sparse",
                    "dendritic_voltage_sparse",
                ),
                teacher_hash,
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
        ),
    )
end

function _write_v5_fixture(path; sealed::Bool)
    JLD2.jldsave(path; payload=_v5_fixture_payload(; sealed))
    return path
end

function _v5_fixture_trainer(path)
    model = ModelV5.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x52554e54494d4555)),
        model,
    )
    return ArenaV5.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=208,
        learning_rate=5.0f-4,
        weight_decay=1.0f-5,
        location_interval=128,
        cell_mode=:distilled_frozen,
        cell_artifact=path,
    )
end

@testset "release-v5 exact sealed FinalProduction loader" begin
    @test ArenaV5 === Main.PaperArenaTrainingFinalProduction
    @test ArenaV5.ReleaseSecurityCell ===
        ArenaV5.DistilledElevenStateCellReleaseRuntimeV5
    @test ArenaV5.paper_release_adapter_contract_version() == 5
    @test ArenaV5.RELEASE_REQUIRED_ARTIFACT_RUNTIME ===
        :sealed_runtime_v5
    @test ArenaV5.ReleaseSecurityCell.OFFICIAL_ELM_INPUT_DIM == 1_278
    @test ArenaV5.ReleaseSecurityCell.TrustedReleaseRuntime ===
        ArenaV5.ReleaseCell.TrustedReleaseRuntime
    @test which(
        ArenaV5.enable_release_runtime!,
        Tuple{ArenaV5.PaperTrainer,AbstractString},
    ).file == Symbol(joinpath(
        @__DIR__,
        "PaperArenaReleaseAdapterV5.jl",
    ))
    @test isbitstype(ArenaV5.PaperFinalWorkItem)

    mktempdir() do directory
        legacy_path = _write_v5_fixture(
            joinpath(directory, "legacy_v3.jld2");
            sealed=false,
        )
        sealed_path = _write_v5_fixture(
            joinpath(directory, "sealed_v5.jld2");
            sealed=true,
        )

        # Prove this is a genuinely V3-compatible legacy artifact, then prove
        # the canonical V5 boundary rejects it.
        legacy_runtime =
            ArenaV5.ReleaseCell.load_release_runtime(legacy_path)
        @test legacy_runtime isa
            ArenaV5.ReleaseCell.TrustedReleaseRuntime
        @test_throws ErrorException ArenaV5.ReleaseSecurityCell.load_release_runtime(
            legacy_path,
        )
        legacy_trainer = _v5_fixture_trainer(legacy_path)
        @test_throws ErrorException ArenaV5.enable_development_release_runtime!(
            legacy_trainer,
            legacy_path;
            development_scale_chain=true,
        )

        runtime =
            ArenaV5.ReleaseSecurityCell.load_release_runtime(sealed_path)
        @test runtime isa
            ArenaV5.ReleaseSecurityCell.TrustedReleaseRuntime
        @test runtime.paper_scale === false
        @test ArenaV5.ReleaseSecurityCell.preflight_integrity!(runtime) ==
            runtime.expected_parameter_sha256

        trainer = _v5_fixture_trainer(sealed_path)
        aux = ArenaV5.enable_development_release_runtime!(
            trainer,
            sealed_path;
            development_scale_chain=true,
        )
        @test aux.trusted.artifact_sha256 == runtime.artifact_sha256
        @test ArenaV5.paper_release_scale_mode(trainer) === :development
        @test ArenaV5.paper_preflight_integrity!(trainer) ==
            runtime.artifact_sha256
        @test_throws ErrorException ArenaV5.enable_release_runtime!(
            _v5_fixture_trainer(sealed_path),
            sealed_path,
        )

        state = ArenaV5.ReleaseSecurityCell.release_new_state(runtime)
        drive = ArenaV5.ReleaseSecurityCell.release_new_drive(runtime)
        diagnostics =
            ArenaV5.ReleaseSecurityCell.release_new_diagnostics(runtime)
        spike = ArenaV5.ReleaseSecurityCell.trusted_cell_step!(
            runtime,
            state,
            drive,
            diagnostics,
        )
        @test diagnostics.spike_probability == 0.5f0
        @test spike == 0.0f0
        @test spike == diagnostics.soma_spike
        @test spike in (0.0f0, 1.0f0)
        @test spike != diagnostics.spike_probability
    end
end
