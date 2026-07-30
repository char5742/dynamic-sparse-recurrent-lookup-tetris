using Test
using JLD2
using JSON3
using NPZ
using Statistics

if !isdefined(Main, :DistillationDatasetBridgeProductionV2)
    include(joinpath(
        @__DIR__,
        "prepare_distillation_dataset_production_v2.jl",
    ))
end
using .DistillationDatasetBridgeProductionV2

# Reuse only the fixture definition, not the obsolete testsets.
fixture_source = read(
    joinpath(@__DIR__, "test_prepare_distillation_dataset_final.jl"),
    String,
)
fixture_prefix = first(split(fixture_source, "@testset"))
fixture_prefix = replace(
    fixture_prefix,
    r"if !isdefined\(Main, :DistillationDatasetBridgeFinal\)[\s\S]*?using \.DistillationDatasetBridgeFinal\n" =>
        "",
)
Base.include_string(Main, fixture_prefix, "production_fixture.jl")

function make_fixture_twin_faithful!(fixture)
    shard = NPZ.npzread(fixture.shard_path)
    contacts = size(shard["contact_axon"], 1)
    time_steps = size(shard["target_voltage"], 1)
    trials = size(shard["target_voltage"], 2)
    event_spike = falses(contacts, time_steps, trials)
    @inbounds for trial in 1:trials
        first_event = Int(shard["event_trial_offset"][trial]) + 1
        last_event = Int(shard["event_trial_offset"][trial + 1])
        active = Dict{Int,Vector{Int}}()
        for event in first_event:last_event
            push!(
                get!(
                    active,
                    Int(shard["event_axon"][event]),
                    Int[],
                ),
                Int(shard["event_time_bin"][event]) + 1,
            )
        end
        for contact in 1:contacts
            for time in get(
                active,
                Int(shard["contact_axon"][contact, trial]),
                Int[],
            )
                event_spike[contact, time, trial] = true
            end
        end
    end
    dense = Main.TwinDatasetGeneration.expand_compact_twin_input(
        shard["contact_segment"],
        shard["contact_kind"],
        shard["contact_strength"],
        event_spike,
        fixture.twin_config,
    )
    prediction = Main.PaperDigitalTwin.twin_forward(
        fixture.frozen,
        dense,
    )
    shard["event_spike"] = UInt8.(event_spike)
    shard["target_voltage"] = Float32.(prediction.voltage)
    shard["target_nmda"] = Float32.(prediction.nmda)
    target_spike = zeros(Float32, size(prediction.spike_probability))
    for trial in 1:trials
        score = prediction.spike_probability[:, trial]
        ordering = sortperm(score)
        positive_count = max(1, length(score) ÷ 2)
        target_spike[
            ordering[(end - positive_count + 1):end],
            trial,
        ] .= 1.0f0
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

@testset "production bridge recomputes official held-out fidelity" begin
    mktempdir() do directory
        fixture = make_fixture_twin_faithful!(
            write_official_fixture(directory),
        )
        output = joinpath(directory, "production_prepared.jld2")
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
        @test !report.twin_self_report_trusted
        @test report.recomputed_twin_gate.spike_auroc == 1.0
        @test isfinite(report.recomputed_twin_gate.voltage_rmse)
        @test isfinite(report.recomputed_twin_gate.nmda_rmse)
        payload = JLD2.load(output)["dataset"]
        @test payload.digital_twin_gate_passed
        @test payload.recomputed_twin_gate.spike_auroc == 1.0
        @test payload.teacher_hash == fixture.detailed_teacher_hash
        @test payload.official_neuron_schema == OFFICIAL_NEURON_SCHEMA
        @test length(payload.segment_region) == 8
        @test payload.source_completion_state == "complete"
        @test payload.metadata.twin_self_report_trusted == false
    end
end

@testset "self-reported 0.99 cannot admit a random twin" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory; spike_auroc=0.99)
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "must_not_publish.jld2"),
                source_kind=:official_neuron,
            ),
        )
        @test !isfile(joinpath(directory, "must_not_publish.jld2"))
    end
end

println("prepare_distillation_dataset_production_v2 tests passed")
