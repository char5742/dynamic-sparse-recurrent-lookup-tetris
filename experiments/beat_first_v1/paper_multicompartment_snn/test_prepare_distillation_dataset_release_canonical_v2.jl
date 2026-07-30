using Test
using JLD2
using JSON3
using NPZ

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_release_canonical.jl",
))

const CanonicalReleaseBridge =
    Main.DistillationDatasetBridgeReleaseCanonical

# Keep the sizeable, standalone final-v2 fixture constructor isolated in a
# module and evaluate definitions only.  This uses normal parsed-file include,
# never include_string, and exports no fixture globals into Main.
module CanonicalReleaseFixture
function definitions_only(expression)
    if expression isa Expr &&
       expression.head === :call &&
       first(expression.args) === :include
        return :(nothing)
    elseif expression isa Expr &&
           expression.head === :macrocall &&
           occursin("@testset", string(first(expression.args)))
        return :(nothing)
    elseif expression isa Expr &&
           expression.head === :call &&
           first(expression.args) === :println
        return :(nothing)
    end
    return expression
end

Base.include(
    definitions_only,
    @__MODULE__,
    joinpath(
        @__DIR__,
        "test_prepare_distillation_dataset_release_canonical.jl",
    ),
)
end

const ReleaseFixture = CanonicalReleaseFixture

function make_histogram_separable!(
    fixture;
    mode::Symbol,
    bins::Int=1024,
)
    arrays = NPZ.npzread(fixture.shard_path)
    target = arrays["target_spike"]
    target .= 0
    for trial in axes(target, 2)
        probability = fixture.predictions[trial].spike
        histogram_bin = clamp.(
            floor.(Int, probability .* bins) .+ 1,
            1,
            bins,
        )
        chosen = if mode === :highest
            findall(==(maximum(histogram_bin)), histogram_bin)
        elseif mode === :lowest
            findall(==(minimum(histogram_bin)), histogram_bin)
        else
            error("unknown separation mode")
        end
        length(chosen) < length(histogram_bin) ||
            error("fixture probabilities occupy only one AUROC bin")
        target[chosen, trial] .= 1
    end
    arrays["target_spike"] = target
    NPZ.npzwrite(fixture.shard_path, arrays)
    manifest = JSON3.read(
        read(fixture.manifest_path, String),
        Dict{String,Any},
    )
    manifest["shards"][1]["sha256"] =
        ReleaseFixture.release_file_sha256(fixture.shard_path)
    manifest["shards"][1]["bytes"] =
        filesize(fixture.shard_path)
    manifest["shards"][1]["spike_positive_bins"] =
        count(>=(0.5), target)
    ReleaseFixture.release_write_json(
        fixture.manifest_path,
        manifest,
    )
    return fixture
end

function event_order_is_valid(dataset)
    dataset.event_order == "time_then_axon" || return false
    for trial in 1:(length(dataset.event_trial_offset) - 1)
        first_event = Int(dataset.event_trial_offset[trial]) + 1
        last_event = Int(dataset.event_trial_offset[trial + 1])
        first_event > last_event && continue
        issorted(collect(zip(
            dataset.event_time_bin[first_event:last_event],
            dataset.event_axon[first_event:last_event],
        ))) || return false
    end
    return true
end

@testset "final-v2 release success and bounded sparse contract" begin
    mktempdir() do directory
        fixture = make_histogram_separable!(
            ReleaseFixture.release_write_fixture(directory);
            mode=:highest,
        )
        output = joinpath(directory, "release")
        config =
            CanonicalReleaseBridge.ReleaseStreamingPrepareConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_directory=output,
                validation_samples=1,
                time_chunk=3,
                output_shard_samples=2,
                minimum_twin_spike_auroc=0.985,
                auroc_histogram_bins=1024,
                require_full_public_counts=false,
            )
        report =
            CanonicalReleaseBridge.
            prepare_distillation_dataset_release(config)
        @test report.total_samples == 4
        @test report.split_counts ==
            (train=2, validation=1, test=1)
        @test report.recomputed_twin_gate.spike_auroc == 1.0
        @test report.digital_twin_gate_passed
        @test !report.dense_memory_scales_with_total_samples
        @test report.peak_dense_chunk_bytes ==
            fixture.twin_config.input_dim * 3 * sizeof(Float32)
        manifest = JSON3.read(read(report.manifest_path, String))
        @test manifest.official_neuron_schema ==
            CanonicalReleaseBridge.FINAL_NEURON_SCHEMA
        @test manifest.event_order == "time_then_axon"
        @test !manifest.promotion_eligible
        @test manifest.source_public_counts.train_pool == 50_000
        @test manifest.source_public_counts.held_out_test == 2_000
        @test length(manifest.train_indices) == 2
        @test length(manifest.validation_indices) == 1
        @test length(manifest.test_indices) == 1
        @test length(String(manifest.segment_catalog_sha256)) == 64
        @test String(manifest.frozen_twin_file_sha256) ==
            ReleaseFixture.release_file_sha256(fixture.twin_path)
        saw_multiplicity = false
        for record in manifest.shards
            path = joinpath(output, String(record.path))
            @test ReleaseFixture.release_file_sha256(path) ==
                String(record.sha256)
            dataset = JLD2.load(path)["dataset"]
            @test dataset.input === nothing
            @test event_order_is_valid(dataset)
            @test dataset.digital_twin_gate_passed
            @test size(dataset.target_voltage, 1) == 7
            @test size(dataset.target_nmda, 1) == 4
            @test size(dataset.target_calcium_event, 1) == 4
            @test size(dataset.target_dendritic_voltage)[1:2] ==
                (4, 4)
            saw_multiplicity |= any(>(UInt8(1)), dataset.event_count)
            for trial in axes(dataset.target_voltage, 2)
                source_index =
                    findfirst(==(
                        dataset.source_sample_indices[trial],
                    ), Int32[11, 22, 33, 44])
                prediction = fixture.predictions[source_index]
                @test dataset.target_voltage[:, trial] ==
                    prediction.voltage
                @test dataset.target_spike[:, trial] ==
                    prediction.spike
                @test dataset.target_nmda[:, :, trial] ==
                    prediction.nmda
                if dataset.split_code[trial] == UInt8(2)
                    @test dataset.source_split_code[trial] == UInt8(1)
                end
            end
        end
        @test saw_multiplicity
    end
end

@testset "gate, raw hash, old schema and partial promotion reject" begin
    mktempdir() do directory
        fixture = make_histogram_separable!(
            ReleaseFixture.release_write_fixture(directory);
            mode=:lowest,
        )
        output = joinpath(directory, "must_not_publish")
        config =
            CanonicalReleaseBridge.ReleaseStreamingPrepareConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_directory=output,
                validation_samples=1,
                time_chunk=3,
                output_shard_samples=2,
                minimum_twin_spike_auroc=0.985,
                auroc_histogram_bins=1024,
                require_full_public_counts=false,
            )
        @test_throws ErrorException(
            CanonicalReleaseBridge.
            prepare_distillation_dataset_release(config)
        )
        @test !ispath(output)
        @test isempty(filter(
            name -> startswith(
                name,
                basename(output) * ".staging.",
            ),
            readdir(dirname(output)),
        ))

        strict = CanonicalReleaseBridge.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=joinpath(directory, "strict"),
            validation_samples=1,
            require_full_public_counts=true,
        )
        @test_throws ErrorException(
            CanonicalReleaseBridge.
            prepare_distillation_dataset_release(strict)
        )

        manifest = JSON3.read(
            read(fixture.manifest_path, String),
            Dict{String,Any},
        )
        canonical =
            manifest["teacher_contract_canonical_json"]
        manifest["teacher_contract_canonical_json"] =
            canonical * " "
        ReleaseFixture.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        loader =
            CanonicalReleaseBridge.Production.OrderedBridge.
            FinalBridge._load_release_source
        @test_throws ErrorException loader(config)
        manifest["teacher_contract_canonical_json"] = canonical
        manifest["schema_name"] =
            "hd_swsnn_twinprop.neuron_teacher.v1"
        ReleaseFixture.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        @test_throws ErrorException loader(config)
    end
end

println("canonical final-v2 release bridge v2 tests passed")
