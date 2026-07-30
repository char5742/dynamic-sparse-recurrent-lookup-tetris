using Test
using JLD2
using JSON3
using SHA
using Statistics

include("StreamingReleaseDataset.jl")
using .StreamingReleaseDataset

const TEST_SHA = (
    twin_parameter=repeat("1", 64),
    twin_artifact=repeat("2", 64),
    twin_file=repeat("3", 64),
    teacher=repeat("4", 64),
    kernel=repeat("5", 64),
    morphology=repeat("6", 64),
    modeldb=repeat("7", 64),
    source_dataset=repeat("8", 64),
    source_manifest=repeat("9", 64),
    segment_catalog=repeat("a", 64),
)

test_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function write_stream_fixture(root; shard_sizes=(2, 1, 2))
    mkpath(root)
    time_steps = 5
    total_samples = sum(shard_sizes)
    diagnostic_time_indices = Int32.(0:(time_steps - 1))
    records = NamedTuple[]
    global_first = 1
    dense_input = zeros(Float32, 3852, time_steps, total_samples)
    dense_target =
        zeros(Float32, 11, time_steps, total_samples)
    split_code = UInt8[1, 1, 1, 2, 3]
    for (shard_index, samples) in enumerate(shard_sizes)
        contact_trial_offset = Int64[2(index - 1) for index in 1:(samples + 1)]
        event_trial_offset = Int64[2(index - 1) for index in 1:(samples + 1)]
        contact_axon = Int32[]
        contact_segment = Int32[]
        contact_kind = UInt8[]
        contact_strength = Float32[]
        event_axon = Int32[]
        event_time_bin = Int32[]
        event_count = UInt8[]
        for local in 1:samples
            global_index = global_first + local - 1
            append!(contact_axon, Int32[1, 2])
            append!(contact_segment, Int32[1, 2])
            append!(contact_kind, UInt8[1, 2])
            append!(contact_strength, Float32[0.5, 0.25])
            append!(event_axon, Int32[1, 2])
            append!(
                event_time_bin,
                Int32[mod(global_index - 1, time_steps), 4],
            )
            append!(event_count, UInt8[1, 2])
            for time in 1:time_steps
                dense_input[1927, time, global_index] = 0.5f0
                dense_input[2569, time, global_index] = 0.5f0
                dense_input[3212, time, global_index] = 0.25f0
            end
            event_one = mod(global_index - 1, time_steps) + 1
            dense_input[1, event_one, global_index] = 0.5f0
            dense_input[643, event_one, global_index] = 0.5f0
            dense_input[1286, 5, global_index] = 0.5f0
        end
        voltage = zeros(Float32, time_steps, samples)
        spike = zeros(Float32, time_steps, samples)
        nmda = zeros(Float32, 4, time_steps, samples)
        calcium = zeros(Float32, time_steps, samples)
        dendritic = zeros(Float32, 4, time_steps, samples)
        for local in 1:samples
            global_index = global_first + local - 1
            for time in 1:time_steps
                voltage[time, local] =
                    Float32(0.1global_index + 0.01time)
                spike[time, local] =
                    Float32(isodd(global_index + time))
                calcium[time, local] =
                    Float32(iszero(mod(global_index + time, 3)))
                for region in 1:4
                    nmda[region, time, local] =
                        Float32(0.01region + 0.001time + 0.1global_index)
                    dendritic[region, time, local] =
                        Float32(region + 0.1time + 0.01global_index)
                end
            end
            dense_target[1, :, global_index] .= voltage[:, local]
            dense_target[2, :, global_index] .= spike[:, local]
            dense_target[3:6, :, global_index] .= nmda[:, :, local]
            dense_target[7, :, global_index] .= calcium[:, local]
            dense_target[8:11, :, global_index] .=
                dendritic[:, :, local]
        end
        global_last = global_first + samples - 1
        payload = (;
            schema=RELEASE_SHARD_SCHEMA,
            input_representation=
                "compact_ragged_contact_event_location_v2",
            input=nothing,
            contact_trial_offset,
            contact_axon,
            contact_segment,
            contact_kind,
            contact_strength,
            event_trial_offset,
            event_axon,
            event_time_bin,
            event_count,
            target_voltage=voltage,
            target_spike=spike,
            target_spike_logit=zeros(Float32, time_steps, samples),
            target_nmda=nmda,
            target_calcium_event=calcium,
            target_dendritic_voltage=dendritic,
            split_code=split_code[global_first:global_last],
            global_output_indices=
                Int32.(global_first:global_last),
            diagnostic_time_indices,
            frozen_twin_parameter_hash=TEST_SHA.twin_parameter,
            frozen_twin_artifact_hash=TEST_SHA.twin_artifact,
            frozen_twin_file_sha256=TEST_SHA.twin_file,
            detailed_teacher_hash=TEST_SHA.teacher,
            detailed_kernel_hash=TEST_SHA.kernel,
            morphology_hash=TEST_SHA.morphology,
            segment_catalog_sha256=TEST_SHA.segment_catalog,
            time_steps,
        )
        name = "shard_" * lpad(shard_index, 3, '0') * ".jld2"
        path = joinpath(root, name)
        JLD2.jldsave(path; dataset=payload)
        push!(
            records,
            (;
                path=name,
                sha256=test_file_sha256(path),
                bytes=filesize(path),
                samples,
                global_first,
                global_last,
            ),
        )
        global_first = global_last + 1
    end
    segment_region = vcat(
        ["soma", "basal", "apical", "tuft"],
        fill("basal", 638),
    )
    manifest = (;
        schema=RELEASE_DATASET_SCHEMA,
        shard_schema=RELEASE_SHARD_SCHEMA,
        completion_state="complete",
        promotion_eligible=false,
        official_neuron_schema=
            "hd_swsnn_twinprop.neuron_teacher.final.v2",
        mixed_supervision=true,
        digital_twin_gate_passed=true,
        twin_self_report_trusted=false,
        recomputed_twin_gate=(; spike_auroc=0.999),
        integrity_before=(; max_delta=0),
        integrity_after=(; max_delta=0),
        input_representation=
            "compact_ragged_contact_event_location_v2",
        dense_memory_scales_with_total_samples=false,
        total_samples,
        time_steps,
        diagnostic_time_indices,
        segment_region,
        segment_catalog_sha256=TEST_SHA.segment_catalog,
        train_indices=Int32[1, 2, 3],
        validation_indices=Int32[4],
        test_indices=Int32[5],
        frozen_twin_parameter_hash=TEST_SHA.twin_parameter,
        frozen_twin_artifact_hash=TEST_SHA.twin_artifact,
        frozen_twin_file_sha256=TEST_SHA.twin_file,
        detailed_teacher_hash=TEST_SHA.teacher,
        detailed_kernel_hash=TEST_SHA.kernel,
        morphology_hash=TEST_SHA.morphology,
        official_modeldb_source_hash=TEST_SHA.modeldb,
        source_dataset_hash=TEST_SHA.source_dataset,
        source_manifest_sha256=TEST_SHA.source_manifest,
        shards=records,
    )
    manifest_path = joinpath(root, "manifest.json")
    open(manifest_path, "w") do stream
        JSON3.pretty(stream, manifest)
    end
    frozen = (;
        model=(; config=(; segments=642, input_dim=3852)),
        parameter_sha256=TEST_SHA.twin_parameter,
        artifact_sha256=TEST_SHA.twin_artifact,
    )
    return (; manifest_path, frozen, dense_input, dense_target, records)
end

@testset "sharded release consumer equals dense fixture" begin
    mktempdir() do directory
        fixture = write_stream_fixture(directory)
        dataset = open_stream_dataset(
            fixture.manifest_path,
            fixture.frozen;
            require_promotion_eligible=false,
        )
        window = stream_materialize_window(
            dataset,
            [5, 1, 3],
            1,
            5,
        )
        @test window.raw_input ==
            fixture.dense_input[:, :, [5, 1, 3]]
        @test window.target ==
            fixture.dense_target[:, :, [5, 1, 3]]
        @test all(window.observed)

        mean_value, scale_value =
            stream_target_statistics(dataset)
        for coordinate in (1, 3, 4, 5, 6, 8, 9, 10, 11)
            reference = vec(
                fixture.dense_target[coordinate, :, 1:3],
            )
            @test mean_value[coordinate] ≈
                Float32(mean(reference)) atol=2.0f-6
            @test scale_value[coordinate] ≈
                max(
                    Float32(std(reference; corrected=false)),
                    1.0f-4,
                ) atol=2.0f-6
        end
        @test stream_dataset_integrity!(dataset) ==
            dataset.dataset_sha256

        maximum_shard_bytes =
            maximum(record.bytes for record in dataset.records)
        @test dataset.tracker.peak_loaded_shard_bytes ==
            maximum_shard_bytes
        @test dataset.tracker.peak_combined_bytes <=
            maximum_shard_bytes +
            dataset.tracker.peak_dense_window_bytes
        @test dataset.tracker.peak_dense_window_bytes <
            sizeof(Float32) * length(fixture.dense_input)
    end
end

@testset "sparse diagnostics are interpolated but masked" begin
    mktempdir() do directory
        fixture = write_stream_fixture(directory)
        manifest = JSON3.read(
            read(fixture.manifest_path, String),
            Dict{String,Any},
        )
        manifest["diagnostic_time_indices"] = Any[0, 2, 4]
        for record in manifest["shards"]
            path = joinpath(directory, String(record["path"]))
            payload = JLD2.load(path)["dataset"]
            payload = merge(
                payload,
                (;
                    diagnostic_time_indices=Int32[0, 2, 4],
                    target_calcium_event=
                        payload.target_calcium_event[[1, 3, 5], :],
                    target_dendritic_voltage=
                        payload.target_dendritic_voltage[:, [1, 3, 5], :],
                ),
            )
            JLD2.jldsave(path; dataset=payload)
            record["sha256"] = test_file_sha256(path)
            record["bytes"] = filesize(path)
        end
        open(fixture.manifest_path, "w") do stream
            JSON3.pretty(stream, manifest)
        end
        dataset = open_stream_dataset(
            fixture.manifest_path,
            fixture.frozen;
            require_promotion_eligible=false,
        )
        window = stream_materialize_window(dataset, [1], 1, 5)
        @test all(window.observed[1:6, :, :])
        @test all(window.observed[7:11, [1, 3, 5], :])
        @test !any(window.observed[7:11, [2, 4], :])
        @test window.target[8:11, 2, 1] ≈
            0.5f0 .* (
                fixture.dense_target[8:11, 1, 1] .+
                fixture.dense_target[8:11, 3, 1]
            )
    end
end

println("streaming release dataset tests passed")
