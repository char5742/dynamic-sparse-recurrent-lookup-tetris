using Test
using JLD2
using JSON3
using NPZ

if !isdefined(Main, :DistillationDatasetBridgeProductionV4)
    include(joinpath(
        @__DIR__,
        "prepare_distillation_dataset_production_v4.jl",
    ))
end
using .DistillationDatasetBridgeProductionV4

# Reuse the official NPZ fixture constructor without executing older tests.
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

function make_twin_faithful!(fixture)
    config = PrepareDistillationConfig(
        dataset_path=fixture.dataset_root,
        frozen_twin_path=fixture.twin_path,
        output_path=joinpath(dirname(fixture.twin_path), "unused.jld2"),
        source_kind=:official_neuron,
    )
    source = DistillationDatasetBridgeProductionV4._load_source(config)
    shard = NPZ.npzread(fixture.shard_path)
    dense = DistillationDatasetBridgeProductionV4._dense_input(
        config,
        source,
        shard,
        collect(1:size(shard["target_voltage"], 2)),
        fixture.twin_config,
    )
    prediction = Main.PaperDigitalTwin.twin_forward(
        fixture.frozen,
        dense,
    )
    @test all(isfinite, dense)
    @test all(isfinite, prediction.voltage)
    @test all(isfinite, prediction.spike_probability)
    @test all(isfinite, prediction.nmda)
    shard["target_voltage"] = Float32.(prediction.voltage)
    shard["target_nmda"] = Float32.(prediction.nmda)
    target_spike = zeros(Float32, size(prediction.spike_probability))
    for trial in axes(target_spike, 2)
        score = prediction.spike_probability[:, trial]
        order = sortperm(score)
        positives = max(1, length(order) ÷ 2)
        target_spike[order[(end - positives + 1):end], trial] .= 1.0f0
    end
    shard["target_spike"] = target_spike
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

@testset "standalone production bridge, official NPZ" begin
    mktempdir() do directory
        fixture = make_twin_faithful!(
            write_official_fixture(directory),
        )
        output = joinpath(directory, "prepared.jld2")
        report = prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=output,
                source_kind=:official_neuron,
                twin_batch_size=2,
                minimum_twin_spike_auroc=0.985,
            ),
        )
        @test report.digital_twin_gate_passed
        @test report.recomputed_twin_gate.spike_auroc == 1.0
        @test !report.twin_self_report_trusted
        @test report.frozen_max_delta_before == 0
        @test report.frozen_max_delta_after == 0
        payload = JLD2.load(output)["dataset"]
        @test payload.official_neuron_schema == OFFICIAL_NEURON_SCHEMA
        @test payload.source_completion_state == "complete"
        @test payload.teacher_hash == fixture.detailed_teacher_hash
        @test payload.frozen_twin_artifact_hash ==
            fixture.frozen.artifact_sha256
        @test payload.digital_twin_gate_passed
        @test payload.mixed_supervision
        @test payload.source_sample_indices == Int32[101, 201, 301]
        @test size(payload.target_dendritic_voltage) == (4, 7, 3)
        @test size(payload.target_calcium_event) == (7, 3)
        @test length(payload.segment_region) == 8
        prediction = Main.PaperDigitalTwin.twin_forward(
            fixture.frozen,
            payload.input,
        )
        @test payload.target_voltage == prediction.voltage
        @test payload.target_spike == prediction.spike_probability
        @test payload.target_spike_logit == prediction.spike_logit
        @test payload.target_nmda == prediction.nmda
    end
end

@testset "random twin self-report cannot pass recomputed gate" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory; spike_auroc=0.99)
        output = joinpath(directory, "must_not_exist.jld2")
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=output,
                source_kind=:official_neuron,
            ),
        )
        @test !isfile(output)
    end
end

@testset "hash and source-schema checks fail closed" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory)
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "bad_hash.jld2"),
                source_kind=:official_neuron,
                expected_morphology_sha256=repeat("f", 64),
            ),
        )
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "relabeled.jld2"),
                source_kind=:canonical_julia,
            ),
        )
    end
end

println("prepare_distillation_dataset_production_v4 tests passed")
