using JLD2
using JSON3
using Lux
using Random
using SHA
using Test

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_release_streaming_sealed_v2.jl",
))

mutable struct SyntheticTracker
    peak_loaded_shard_bytes::Int
    peak_dense_window_bytes::Int
    peak_combined_bytes::Int
    windows_materialized::Int
    samples_materialized::Int
end

struct SyntheticSealedDataset
    manifest
    time_steps::Int
    train_indices::Vector{Int}
    validation_indices::Vector{Int}
    test_indices::Vector{Int}
    segment_region::Vector{String}
    provenance
    dataset_sha256::String
    segment_catalog_sha256::String
    manifest_sha256::String
    tracker::SyntheticTracker
end

const SYNTHETIC_BUNDLE = Ref{Any}(nothing)
const SYNTHETIC_OPENED = Ref{Any}(nothing)

function Sealed.load_verified_sealed_official_elm_release(
    ::String,
    ::String,
    ::String;
    require_production::Bool=true,
    scratch_root=nothing,
)
    require_production &&
        error("synthetic fixture is deliberately development scale")
    return SYNTHETIC_BUNDLE[]
end

function StreamFinal.open_live_verified_sealed_stream_dataset(
    ::String,
    ::Sealed.SealedOfficialELMRelease,
    ::String,
    ::String;
    kwargs...,
)
    return SYNTHETIC_OPENED[]
end

Stream.stream_dataset_integrity!(dataset::SyntheticSealedDataset) =
    dataset.dataset_sha256

function Stream.sealed_stream_materialize_window(
    dataset::SyntheticSealedDataset,
    indices,
    first_time::Integer,
    window::Integer,
)
    samples = Int.(collect(indices))
    count = Int(window)
    raw = zeros(Float32, 1278, count, length(samples))
    target = zeros(Float32, 11, count, length(samples))
    observed = trues(11, count, length(samples))
    logit = zeros(Float32, count, length(samples))
    for (batch, sample) in enumerate(samples)
        for local_time in 1:count
            time = Int(first_time) + local_time - 1
            phase = 0.01f0 * Float32(time + 17 * sample)
            target[1, local_time, batch] = -67.0f0 + sin(phase)
            target[2, local_time, batch] =
                isodd(time + sample) ? 1.0f0 : 0.0f0
            logit[local_time, batch] =
                target[2, local_time, batch] > 0.5f0 ? 8.0f0 : -8.0f0
            for region in 1:4
                target[2 + region, local_time, batch] =
                    0.1f0 * region + 0.03f0 * sin(phase + region)
                target[7 + region, local_time, batch] =
                    -70.0f0 + 2.0f0 * region + cos(phase + region)
            end
            target[7, local_time, batch] =
                isodd(div(time, 3) + sample) ? 1.0f0 : 0.0f0
        end
    end
    dataset.tracker.windows_materialized += 1
    dataset.tracker.samples_materialized += length(samples)
    dataset.tracker.peak_dense_window_bytes = max(
        dataset.tracker.peak_dense_window_bytes,
        sizeof(raw) + sizeof(target),
    )
    dataset.tracker.peak_combined_bytes =
        dataset.tracker.peak_dense_window_bytes
    return (;
        raw_input=raw,
        target,
        target_spike_logit=logit,
        observed,
    )
end

function _perfect_metrics(dataset::SyntheticSealedDataset, split::Symbol)
    indices =
        split === :validation ?
        dataset.validation_indices :
        split === :test ? dataset.test_indices :
        dataset.train_indices
    samples = length(indices)
    dense = samples * 1000
    return (;
        samples,
        free_rollout_horizon=1000,
        soma_voltage_rmse_mv=0.1,
        soma_voltage_correlation=0.999,
        spike_auroc=0.999,
        spike_auroc_estimate=1.0,
        spike_auroc_ambiguity_bound=0.001,
        nmda_rmse_by_region=[0.01, 0.01, 0.01, 0.01],
        nmda_correlation_by_region=[0.99, 0.99, 0.99, 0.99],
        calcium_event_auroc=0.99,
        calcium_event_auroc_estimate=1.0,
        calcium_event_auroc_ambiguity_bound=0.01,
        dendritic_voltage_rmse_mv=[0.1, 0.1, 0.1, 0.1],
        semantic_coordinate_names=
            DistillCore.SEMANTIC_COORDINATE_NAMES,
        semantic_coordinate_rmse=fill(0.01, 11),
        semantic_coordinate_correlation=fill(0.99, 11),
        semantic_coordinate_passed=trues(11),
        sparse_auxiliary_metrics_use_observed_times_only=true,
        interpolated_auxiliary_values_excluded_from_gate=true,
        auroc_is_conservative_lower_bound=true,
        auroc_histogram_bins=256,
        evaluation_split=String(split),
        evaluated_indices_sha256=
            Training.split_indices_sha256(indices),
        evaluated_index_count=samples,
        exact_dataset_split=true,
        state_warmup_steps=500,
        evaluated_time_indices_one_based=(501, 1500),
        evaluated_steps_per_trial=1000,
        expected_dense_observations=dense,
        physical_observation_count_by_coordinate=(
            dense, dense, dense, dense, dense, dense,
            samples * 100, samples * 100, samples * 100,
            samples * 100, samples * 100,
        ),
        semantic_observation_count_by_coordinate=
            ntuple(_ -> samples * 100, 11),
        spike_positive_observations=max(1, div(dense, 2)),
        spike_negative_observations=max(1, div(dense, 2)),
        calcium_positive_observations=max(1, samples * 50),
        calcium_negative_observations=max(1, samples * 50),
        training_window_contract_is_distinct=true,
    )
end

function Training.evaluate_streaming_split_post_burnin(
    parameters,
    dataset::SyntheticSealedDataset,
    materialize_window,
    split::Symbol,
    target_mean,
    target_scale,
    config;
    time_chunk::Integer,
    auroc_bins::Integer,
)
    return _perfect_metrics(dataset, split)
end

function _synthetic_frozen()
    config = Stream.Twin.OfficialELMConfig(
        num_memory=2,
        hidden_size=4,
    )
    model = Stream.Twin.build_official_elm_twin(config)
    parameters = Lux.initialparameters(Xoshiro(7), model)
    normalizer = Stream.Twin.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    return Stream.Twin.freeze_official_elm_twin(
        model,
        parameters,
        normalizer,
    )
end

@testset "sealed-v2 -> 11-state -> strict gate -> runtime V6 smoke" begin
    mktempdir() do directory
        source_manifest = joinpath(directory, "manifest.json")
        write(source_manifest, "{\"synthetic\":true}")
        source_manifest_sha = bytes2hex(SHA.sha256(read(source_manifest)))
        contract_sha = "c"^64
        frozen = _synthetic_frozen()
        payload = (;
            teacher=(;
                manifest_sha256=source_manifest_sha,
                teacher_contract_sha256=contract_sha,
            ),
            model=(;
                input_dim=1278,
                parameter_sha256=frozen.parameter_sha256,
                base_artifact_sha256=frozen.artifact_sha256,
                executable_mlp_activation=:silu,
                compatibility_profile=:synthetic_final_v2,
            ),
            outcome=(;
                gate_passed=true,
                paper_scale=false,
                development_scale=true,
                promotable_production=false,
            ),
            split=(; duration_ms=1500.0),
        )
        attestation =
            Sealed.SealedOfficialELMReleaseAttestation(
                payload,
                Sealed.canonical_sha256(payload),
            )
        bundle =
            Sealed.SealedOfficialELMRelease(frozen, attestation)
        SYNTHETIC_BUNDLE[] = bundle

        regions = fill("apical_trunk", 642)
        regions[1] = "soma"
        regions[2:200] .= "basal"
        regions[501:640] .= "apical_tuft"
        regions[641:642] .= "axon"
        manifest = (;
            neuronio_training_window=(;
                full_time_steps=1500,
                sample_dt_ms=1.0,
                ignore_time_from_start_ms=500.0,
                input_window_steps=500,
                valid_window_start_indices_one_based=(501, 1000),
                sampling="uniform_with_replacement",
            ),
            heldout_evaluation_window=(;
                evaluated_time_indices_one_based=(501, 1500),
                evaluated_steps_per_trial=1000,
            ),
        )
        dataset = SyntheticSealedDataset(
            manifest,
            1500,
            [1, 2],
            [3],
            [4],
            regions,
            (;
                detailed_teacher_hash="d"^64,
                detailed_kernel_hash="e"^64,
                morphology_hash="f"^64,
                source_manifest_sha256=source_manifest_sha,
                source_teacher_contract_sha256=contract_sha,
            ),
            "1"^64,
            "2"^64,
            "3"^64,
            SyntheticTracker(0, 0, 0, 0, 0),
        )
        live = (;
            cache_verified_all_samples=true,
            bit_exact=true,
            samples_verified=4,
            time_points_verified=6000,
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
            detailed_model_auxiliary_targets=(
                "calcium_event_sparse",
                "dendritic_voltage_sparse",
            ),
            sealed_execution_type=Stream.SEALED_EXECUTION_TYPE,
            official_elm_input_dim=1278,
            sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
            sealed_release_artifact_kind=
                Sealed.SEALED_RELEASE_ARTIFACT_KIND,
            sealed_attestation_sha256=
                attestation.attestation_sha256,
        )
        SYNTHETIC_OPENED[] = (;
            dataset,
            live_replay=live,
            measurement=(;
                verified=true,
                measurement_sha256="4"^64,
                measurement_schema=StreamFinal.PRIMARY_REPLAY_SCHEMA,
                samples_verified=4,
                time_points_verified=6000,
                bit_exact=true,
            ),
        )
        bridge = joinpath(directory, "bridge")
        shards = joinpath(directory, "shards")
        mkpath(bridge)
        mkpath(shards)
        sealed_artifact = joinpath(directory, "sealed.jld2")
        write(sealed_artifact, "synthetic exact-type fixture")
        output = joinpath(directory, "distilled.jld2")
        report_path = joinpath(directory, "report.json")
        report = run_sealed_v2_eleven_state_distillation(
            SealedV2ElevenStateDistillationConfig(
                bridge_dataset=bridge,
                sealed_artifact=sealed_artifact,
                source_manifest=source_manifest,
                source_shards=shards,
                output=output,
                metrics=report_path,
                epochs=1,
                steps_per_epoch=1,
                batch=1,
                free_rollout_epochs=0,
                restarts=1,
                metric_time_chunk=500,
                metric_auroc_bins=256,
                cache_replay_time_chunk=500,
                statistics_time_chunk=500,
                expected_source_manifest_sha256=
                    source_manifest_sha,
                expected_teacher_contract_sha256=contract_sha,
            ),
        )
        @test report.accepted
        @test isfile(output)
        @test isfile(report_path)
        @test report.evaluation_protocol.test_evaluations == 1
        @test report.gate.per_region_and_branch_gating
        data = JLD2.load(output)
        parameters = data["payload"].parameters
        @test size(parameters.compartment_projection) == (4, 642)
        @test all(iszero, parameters.compartment_projection[:, 1])
        @test all(iszero, parameters.compartment_projection[:, 641:642])
        runtime = RuntimeV6.load_release_runtime(
            output;
            expected_artifact_sha256=report.artifact_sha256,
            expected_sealed_attestation_sha256=
                attestation.attestation_sha256,
        )
        @test RuntimeV6.preflight_integrity!(runtime) ==
            report.parameter_sha256
    end
end
