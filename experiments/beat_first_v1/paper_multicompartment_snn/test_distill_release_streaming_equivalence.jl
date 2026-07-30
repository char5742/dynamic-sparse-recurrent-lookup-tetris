using Test
using JLD2
using JSON3
using Random
using Zygote

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_release_streaming.jl",
))

# Reuse the final-v2 [2, 1, 2]-shard oracle without executing that file's own
# testsets.  The old fixture draft contains `local` as an identifier, so apply
# the same lexical repair as its canonical test entry point.  Names are made
# private to this test because the dense release driver intentionally defines
# a different `RELEASE_DATASET_SCHEMA` in Main.
fixture_path = joinpath(
    @__DIR__,
    "test_streaming_release_dataset.jl",
)
fixture_source = first(split(read(fixture_path, String), "@testset"))
fixture_source = replace(
    fixture_source,
    "include(\"StreamingReleaseDataset.jl\")" => "",
    "using .StreamingReleaseDataset" => "",
    "RELEASE_DATASET_SCHEMA" => "EQ_RELEASE_DATASET_SCHEMA",
    "RELEASE_SHARD_SCHEMA" => "EQ_RELEASE_SHARD_SCHEMA",
    "TEST_SHA" => "EQ_TEST_SHA",
    "test_file_sha256" => "equivalence_file_sha256",
    "write_stream_fixture" => "write_stream_equivalence_fixture",
    r"\blocal\b" => "local_index",
)
fixture_source =
    """
    const EQ_RELEASE_DATASET_SCHEMA =
        StreamingReleaseDataset.RELEASE_DATASET_SCHEMA
    const EQ_RELEASE_SHARD_SCHEMA =
        StreamingReleaseDataset.RELEASE_SHARD_SCHEMA
    """ * fixture_source
Base.include_string(Main, fixture_source, fixture_path)

function separable_histogram_labels(scores, bins::Int)
    histogram_bin = map(scores) do raw
        score = Float64(raw)
        0.0 <= score <= 1.0 ||
            error("fixture probability is outside [0, 1]")
        score == 1.0 ? bins : floor(Int, score * bins) + 1
    end
    occupied = sort!(unique(vec(histogram_bin)))
    length(occupied) >= 2 ||
        error("fixture predictions occupy fewer than two bins")
    threshold = occupied[cld(length(occupied), 2)]
    labels = Float32.(histogram_bin .>= threshold)
    any(iszero, labels) && any(==(1.0f0), labels) ||
        error("fixture AUROC labels need both classes")
    return labels
end

function make_metric_bins_separable!(
    fixture,
    parameters,
    auroc_bins::Int,
)
    initial_mean, initial_scale = _release_target_statistics(
        fixture.dense_target,
        1:3,
    )
    rollout = _release_rollout(
        parameters,
        fixture.dense_input,
        fixture.dense_target,
        initial_mean,
        initial_scale,
        642,
    )
    physical = _release_physical_output(
        rollout.output,
        initial_mean,
        initial_scale,
    )
    fixture.dense_target[2, :, :] .=
        separable_histogram_labels(
            @view(physical[2, :, :]),
            auroc_bins,
        )
    fixture.dense_target[7, :, :] .=
        separable_histogram_labels(
            @view(physical[7, :, :]),
            auroc_bins,
        )

    manifest = JSON3.read(
        read(fixture.manifest_path, String),
        Dict{String,Any},
    )
    root = dirname(fixture.manifest_path)
    for record in manifest["shards"]
        first_global = Int(record["global_first"])
        last_global = Int(record["global_last"])
        path = joinpath(root, String(record["path"]))
        payload = JLD2.load(path)["dataset"]
        payload = merge(
            payload,
            (;
                target_spike=copy(@view(
                    fixture.dense_target[
                        2,
                        :,
                        first_global:last_global,
                    ]
                )),
                target_calcium_event=copy(@view(
                    fixture.dense_target[
                        7,
                        :,
                        first_global:last_global,
                    ]
                )),
            ),
        )
        JLD2.jldsave(path; dataset=payload)
        record["sha256"] = equivalence_file_sha256(path)
        record["bytes"] = filesize(path)
    end
    open(fixture.manifest_path, "w") do stream
        JSON3.pretty(stream, manifest)
    end
    return fixture
end

function test_parameter_gradients_equal(dense, streaming)
    @test keys(dense) == keys(streaming)
    for name in keys(dense)
        dense_value = getproperty(dense, name)
        stream_value = getproperty(streaming, name)
        @test size(dense_value) == size(stream_value)
        @test dense_value == stream_value
    end
end

function test_metric_values_equal(dense, streaming)
    @test streaming.samples == dense.samples
    @test streaming.free_rollout_horizon ==
        dense.free_rollout_horizon
    @test isapprox(
        streaming.soma_voltage_rmse_mv,
        dense.soma_voltage_rmse_mv;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test isapprox(
        streaming.soma_voltage_correlation,
        dense.soma_voltage_correlation;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test streaming.spike_auroc_ambiguity_bound == 0.0
    @test streaming.spike_auroc_estimate == dense.spike_auroc
    @test streaming.spike_auroc == dense.spike_auroc
    @test isapprox(
        streaming.nmda_rmse_by_region,
        dense.nmda_rmse_by_region;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test isapprox(
        streaming.nmda_correlation_by_region,
        dense.nmda_correlation_by_region;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test streaming.calcium_event_auroc_ambiguity_bound == 0.0
    @test streaming.calcium_event_auroc_estimate ==
        dense.calcium_event_auroc
    @test streaming.calcium_event_auroc ==
        dense.calcium_event_auroc
    @test isapprox(
        streaming.dendritic_voltage_rmse_mv,
        dense.dendritic_voltage_rmse_mv;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test streaming.semantic_coordinate_names ==
        STREAM_SEMANTIC_COORDINATE_NAMES
    @test isapprox(
        streaming.semantic_coordinate_rmse,
        dense.semantic_coordinate_rmse;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test isapprox(
        streaming.semantic_coordinate_correlation,
        dense.semantic_coordinate_correlation;
        atol=2.0e-9,
        rtol=2.0e-7,
        nans=true,
    )
    @test streaming.semantic_coordinate_passed ==
        dense.semantic_coordinate_passed
    @test streaming.auroc_is_conservative_lower_bound
    return nothing
end

@testset "dense and streaming release distillation are equivalent" begin
    mktempdir() do directory
        auroc_bins = 65_536
        parameters =
            _release_initial_parameters(Xoshiro(0x1234), 642)
        fixture = make_metric_bins_separable!(
            write_stream_equivalence_fixture(directory),
            parameters,
            auroc_bins,
        )
        dataset = StreamingReleaseDataset.open_stream_dataset(
            fixture.manifest_path,
            fixture.frozen;
            require_promotion_eligible=false,
        )
        target_mean, target_scale =
            StreamingReleaseDataset.stream_target_statistics(dataset)
        dense_mean, dense_scale = _release_target_statistics(
            fixture.dense_target,
            1:3,
        )
        @test target_mean ≈ dense_mean atol=2.0f-6
        @test target_scale ≈ dense_scale atol=2.0f-6

        # Release random-window training deliberately resets to initial_state
        # at the start of an interior window.  The streaming consumer must
        # preserve that existing behaviour rather than replaying the prefix.
        indices = [5, 1, 3]
        first_time = 2
        window_length = 3
        time_range =
            first_time:(first_time + window_length - 1)
        batch = StreamingReleaseDataset.stream_materialize_window(
            dataset,
            indices,
            first_time,
            window_length,
        )
        @test batch.raw_input ==
            fixture.dense_input[:, time_range, indices]
        @test batch.target ==
            fixture.dense_target[:, time_range, indices]
        @test all(batch.observed)

        for free_fraction in (0.0f0, 0.375f0, 1.0f0)
            dense_loss = _release_sequence_loss(
                parameters,
                fixture.dense_input[:, time_range, indices],
                fixture.dense_target[:, time_range, indices],
                target_mean,
                target_scale,
                642,
                free_fraction,
            )
            stream_loss = _stream_sequence_loss(
                parameters,
                batch.raw_input,
                batch.target,
                target_mean,
                target_scale,
                642,
                free_fraction,
            )
            @test stream_loss == dense_loss
        end

        free_fraction = 0.375f0
        dense_gradient = Zygote.gradient(parameters) do candidate
            _release_sequence_loss(
                candidate,
                fixture.dense_input[:, time_range, indices],
                fixture.dense_target[:, time_range, indices],
                target_mean,
                target_scale,
                642,
                free_fraction,
            )
        end |> only
        stream_gradient = Zygote.gradient(parameters) do candidate
            _stream_sequence_loss(
                candidate,
                batch.raw_input,
                batch.target,
                target_mean,
                target_scale,
                642,
                free_fraction,
            )
        end |> only
        test_parameter_gradients_equal(
            dense_gradient,
            stream_gradient,
        )

        dense_metrics = _release_metrics(
            parameters,
            (; input=fixture.dense_input),
            fixture.dense_target,
            1:5,
            target_mean,
            target_scale,
            642,
        )
        stream_metrics = _stream_metrics(
            parameters,
            dataset,
            1:5,
            target_mean,
            target_scale,
            642;
            time_chunk=2,
            auroc_bins,
        )
        test_metric_values_equal(dense_metrics, stream_metrics)

        maximum_shard_bytes =
            maximum(record.bytes for record in dataset.records)
        expected_loss_window_bytes =
            sizeof(Float32) *
            (3852 + 11) *
            window_length *
            length(indices) +
            sizeof(Bool) *
            11 *
            window_length *
            length(indices)
        expected_metric_chunk_bytes =
            sizeof(Float32) * (3852 + 11) * 2 +
            sizeof(Bool) * 11 * 2
        expected_peak_window_bytes = max(
            expected_loss_window_bytes,
            expected_metric_chunk_bytes,
        )
        dense_dataset_bytes =
            sizeof(Float32) * (
                length(fixture.dense_input) +
                length(fixture.dense_target)
            ) +
            sizeof(Bool) * length(fixture.dense_target)

        @test dataset.tracker.peak_loaded_shard_bytes ==
            maximum_shard_bytes
        @test dataset.tracker.peak_dense_window_bytes ==
            expected_peak_window_bytes
        @test dataset.tracker.peak_combined_bytes <=
            maximum_shard_bytes + expected_peak_window_bytes
        @test dataset.tracker.peak_dense_window_bytes <
            dense_dataset_bytes
        @test dataset.tracker.samples_materialized ==
            length(indices)
    end
end

println("streaming release dense-equivalence tests passed")
