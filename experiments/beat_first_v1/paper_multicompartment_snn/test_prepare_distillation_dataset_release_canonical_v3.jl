using Test
using JLD2
using JSON3

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_release_canonical.jl",
))

module ReleaseV3Definitions
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
        "test_prepare_distillation_dataset_release_canonical_v2.jl",
    ),
)
end

const V3 = ReleaseV3Definitions
const BridgeV3 = V3.CanonicalReleaseBridge
const FixtureV3 = V3.ReleaseFixture

@testset "release-v3 success contract" begin
    mktempdir() do directory
        fixture = V3.make_histogram_separable!(
            FixtureV3.release_write_fixture(directory);
            mode=:highest,
        )
        output = joinpath(directory, "release")
        config = BridgeV3.ReleaseStreamingPrepareConfig(
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
        report = BridgeV3.prepare_distillation_dataset_release(config)
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
            BridgeV3.FINAL_NEURON_SCHEMA
        @test manifest.event_order == "time_then_axon"
        @test manifest.source_public_counts.train_pool == 50_000
        @test manifest.source_public_counts.held_out_test == 2_000
        @test length(manifest.validation_indices) == 1
        @test length(String(manifest.segment_catalog_sha256)) == 64
        @test String(manifest.frozen_twin_file_sha256) ==
            FixtureV3.release_file_sha256(fixture.twin_path)
        saw_multiplicity = false
        for record in manifest.shards
            path = joinpath(output, String(record.path))
            @test FixtureV3.release_file_sha256(path) ==
                String(record.sha256)
            dataset = JLD2.load(path)["dataset"]
            @test dataset.input === nothing
            @test V3.event_order_is_valid(dataset)
            @test dataset.digital_twin_gate_passed
            @test size(dataset.target_calcium_event, 1) == 4
            @test size(dataset.target_dendritic_voltage)[1:2] ==
                (4, 4)
            saw_multiplicity |= any(>(UInt8(1)), dataset.event_count)
            for trial in axes(dataset.target_voltage, 2)
                source_index = findfirst(
                    ==(dataset.source_sample_indices[trial]),
                    Int32[11, 22, 33, 44],
                )
                prediction = fixture.predictions[source_index]
                @test dataset.target_voltage[:, trial] ==
                    prediction.voltage
                @test dataset.target_spike[:, trial] ==
                    prediction.spike
                @test dataset.target_nmda[:, :, trial] ==
                    prediction.nmda
                dataset.split_code[trial] == UInt8(2) &&
                    @test(
                        dataset.source_split_code[trial] == UInt8(1)
                    )
            end
        end
        @test saw_multiplicity
    end
end

@testset "release-v3 fail closed" begin
    mktempdir() do directory
        fixture = V3.make_histogram_separable!(
            FixtureV3.release_write_fixture(directory);
            mode=:lowest,
        )
        output = joinpath(directory, "must_not_publish")
        config = BridgeV3.ReleaseStreamingPrepareConfig(
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
            BridgeV3.prepare_distillation_dataset_release(config)
        )
        @test !ispath(output)
        @test isempty(filter(
            name -> startswith(
                name,
                basename(output) * ".staging.",
            ),
            readdir(dirname(output)),
        ))

        strict = BridgeV3.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=joinpath(directory, "strict"),
            validation_samples=1,
            require_full_public_counts=true,
        )
        @test_throws ErrorException(
            BridgeV3.prepare_distillation_dataset_release(strict)
        )

        manifest = JSON3.read(
            read(fixture.manifest_path, String),
            Dict{String,Any},
        )
        canonical =
            manifest["teacher_contract_canonical_json"]
        manifest["teacher_contract_canonical_json"] =
            canonical * " "
        FixtureV3.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        loader = BridgeV3.Production.OrderedBridge.
            FinalBridge._load_release_source
        @test_throws ErrorException loader(config)
        manifest["teacher_contract_canonical_json"] = canonical
        manifest["schema_name"] =
            "hd_swsnn_twinprop.neuron_teacher.v1"
        FixtureV3.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        @test_throws ErrorException loader(config)
    end
end

println("canonical final-v2 release bridge v3 tests passed")
