using Test
using JLD2
using JSON3
using NPZ

if !isdefined(Main, :DistillationDatasetBridgeStreamingV5)
    include(joinpath(
        @__DIR__,
        "prepare_distillation_dataset_streaming_v5.jl",
    ))
end
using .DistillationDatasetBridgeStreamingV5

fixture_file = joinpath(
    @__DIR__,
    "test_prepare_distillation_dataset_final.jl",
)
fixture_source = first(split(read(fixture_file, String), "\n@testset"))
fixture_source = replace(
    fixture_source,
    r"if !isdefined\(Main, :DistillationDatasetBridgeFinal\)[\s\S]*?using \.DistillationDatasetBridgeFinal\n" =>
        "",
)
Base.include_string(Main, fixture_source, fixture_file)

function make_stream_fixture_faithful!(fixture)
    shard = DistillationDatasetBridgeStreamingV5._read_numeric_shard(
        fixture.shard_path,
    )
    target_voltage = similar(shard["target_voltage"])
    target_spike = similar(shard["target_spike"])
    target_nmda = similar(shard["target_nmda"])
    for trial in axes(target_voltage, 2)
        sparse = DistillationDatasetBridgeStreamingV5._sparse_sample(
            shard,
            trial,
            fixture.twin_config,
        )
        prediction =
            DistillationDatasetBridgeStreamingV5._infer_sample(
                fixture.frozen,
                sparse,
                size(target_voltage, 1),
                3,
            )
        target_voltage[:, trial] .= prediction.voltage
        target_nmda[:, :, trial] .= prediction.nmda
        order = sortperm(prediction.spike)
        target_spike[:, trial] .= 0
        positives = max(1, length(order) ÷ 2)
        target_spike[
            order[(end - positives + 1):end],
            trial,
        ] .= 1
    end
    shard["target_voltage"] = target_voltage
    shard["target_spike"] = target_spike
    shard["target_nmda"] = target_nmda
    NPZ.npzwrite(fixture.shard_path, shard)
    manifest = JSON3.read(
        read(fixture.manifest_path, String),
        Dict{String,Any},
    )
    manifest["shards"][1]["sha256"] =
        file_sha256(fixture.shard_path)
    open(fixture.manifest_path, "w") do stream
        JSON3.pretty(stream, manifest)
    end
    return fixture
end

@testset "streaming sparse bridge is memory bounded" begin
    mktempdir() do directory
        fixture = make_stream_fixture_faithful!(
            write_official_fixture(directory),
        )
        output = joinpath(directory, "prepared_shards")
        report = prepare_distillation_dataset_streaming(
            StreamingPrepareConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_directory=output,
                source_kind=:official_neuron,
                time_chunk=3,
                output_shard_samples=1,
                minimum_twin_spike_auroc=0.985,
                auroc_histogram_bins=1024,
            ),
        )
        @test report.schema == SHARDED_DATASET_SCHEMA
        @test report.total_samples == 3
        @test report.shard_count == 3
        @test report.digital_twin_gate_passed
        @test report.recomputed_twin_gate.spike_auroc == 1.0
        @test !report.dense_memory_scales_with_total_samples
        @test report.peak_dense_chunk_bytes ==
            fixture.twin_config.input_dim * 3 * sizeof(Float32)
        manifest = JSON3.read(read(report.manifest_path, String))
        @test manifest.completion_state == "complete"
        @test manifest.input_representation ==
            "compact_contact_event_location_v1"
        @test length(manifest.shards) == 3
        for record in manifest.shards
            shard_path = joinpath(output, String(record.path))
            @test file_sha256(shard_path) == String(record.sha256)
            dataset = JLD2.load(shard_path)["dataset"]
            @test dataset.schema == SHARD_SCHEMA
            @test dataset.input === nothing
            @test dataset.input_representation ==
                "compact_contact_event_location_v1"
            @test size(dataset.target_voltage, 2) == 1
            @test size(dataset.target_dendritic_voltage, 1) == 4
        end
    end
end

@testset "streaming bridge rejects self-reported random twin" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory; spike_auroc=0.99)
        output = joinpath(directory, "not_published")
        @test_throws ErrorException prepare_distillation_dataset_streaming(
            StreamingPrepareConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_directory=output,
                source_kind=:official_neuron,
                time_chunk=3,
                output_shard_samples=1,
                auroc_histogram_bins=1024,
            ),
        )
        @test !ispath(output)
    end
end

println("prepare_distillation_dataset_streaming_v5 tests passed")
