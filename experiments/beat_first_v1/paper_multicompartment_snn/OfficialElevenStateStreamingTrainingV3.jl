module OfficialElevenStateStreamingTrainingV3

# Final bounded-memory NeuronIO training/evaluation contract for the
# source-bound 1,278-input sealed ELM.  V3 makes split identity and every
# observation count part of the metric record so a release gate cannot trust a
# caller-selected subset or silently accept a coordinate with no observations.

using Serialization
using SHA

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :OfficialElevenStateStreamingTrainingV2)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "OfficialElevenStateStreamingTrainingV2.jl",
        ),
    )
end

const V2 =
    getfield(_PARENT, :OfficialElevenStateStreamingTrainingV2)
const BaseTraining = V2.BaseTraining
const Core = V2.Core
const Metrics = V2.Metrics

export evaluate_streaming_split_post_burnin,
    neuronio_window_contract,
    split_indices_sha256,
    train_streaming_neuronio_windows

const neuronio_window_contract = V2.neuronio_window_contract
const train_streaming_neuronio_windows =
    V2.train_streaming_neuronio_windows

function split_indices_sha256(indices)
    stream = IOBuffer()
    Serialization.serialize(stream, Tuple(Int.(collect(indices))))
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _split_indices(dataset, split::Symbol)
    split === :train && return dataset.train_indices
    split === :validation && return dataset.validation_indices
    split === :test && return dataset.test_indices
    throw(ArgumentError(
        "evaluation split must be :train, :validation, or :test",
    ))
end

"""
Evaluate one exact dataset split after a stateful 500-step warm-up.

The public entry point accepts a split name, not arbitrary indices.  This is
intentional: model selection may inspect only `:validation`, while the final
release gate consumes exactly `dataset.test_indices`.
"""
function evaluate_streaming_split_post_burnin(
    parameters,
    dataset,
    materialize_window,
    split::Symbol,
    target_mean,
    target_scale,
    config;
    time_chunk::Integer,
    auroc_bins::Integer,
)
    contract = neuronio_window_contract(dataset, config)
    selected = Int.(collect(_split_indices(dataset, split)))
    isempty(selected) &&
        error("the $(String(split)) split is empty")
    chunk = Int(time_chunk)
    chunk >= 1 || throw(ArgumentError("time_chunk must be positive"))
    metrics = Metrics.StreamingMetrics(
        length(selected),
        contract.heldout_evaluated_steps;
        auroc_bins=Int(auroc_bins),
        maximum_coordinate_rmse=
            BaseTraining.MAXIMUM_COORDINATE_RMSE,
        minimum_coordinate_correlation=
            BaseTraining.MINIMUM_COORDINATE_CORRELATION,
    )
    first_metric = contract.heldout_evaluation_range[1]
    for global_index in selected
        state = repeat(parameters.initial_state, 1, 1)
        for first_time in 1:chunk:dataset.time_steps
            count = min(chunk, dataset.time_steps - first_time + 1)
            batch = materialize_window(
                dataset,
                [global_index],
                first_time,
                count,
            )
            raw_input = @view batch.raw_input[:, :, 1]
            target = @view batch.target[:, :, 1]
            observed = @view batch.observed[:, :, 1]
            for local_time in 1:count
                global_time = first_time + local_time - 1
                input = Core.project_official_input(
                    @view(raw_input[:, local_time:local_time]),
                    parameters.location_logits,
                )
                state = Core.transition(parameters, state, input)
                global_time < first_metric && continue
                raw_output =
                    Core.structured_readout(parameters, state)
                normalized = Core.normalize_target(
                    @view(target[:, local_time:local_time]),
                    target_mean,
                    target_scale,
                )
                semantic = Core.semantic_target(normalized)
                physical = Core.physical_output(
                    @view(raw_output[:, 1]),
                    target_mean,
                    target_scale,
                )
                physical_validity = @view observed[:, local_time]
                semantic_validity =
                    BaseTraining._semantic_validity(physical_validity)
                BaseTraining._update_metrics!(
                    metrics,
                    physical,
                    @view(target[:, local_time]),
                    @view(state[:, 1]),
                    @view(semantic[:, 1]),
                    physical_validity,
                    semantic_validity,
                )
            end
        end
    end
    result = BaseTraining._finalize_metrics(metrics)
    physical_counts =
        Tuple(statistic.count for statistic in metrics.output_statistics)
    semantic_counts =
        Tuple(statistic.count for statistic in metrics.semantic_statistics)
    expected_dense_count =
        length(selected) * contract.heldout_evaluated_steps
    for coordinate in 1:6
        physical_counts[coordinate] == expected_dense_count ||
            error(
                "primary coordinate $coordinate did not cover every " *
                "post-burn-in sample/time point",
            )
    end
    return merge(
        result,
        (;
            evaluation_split=String(split),
            evaluated_indices_sha256=
                split_indices_sha256(selected),
            evaluated_index_count=length(selected),
            exact_dataset_split=true,
            state_warmup_steps=contract.heldout_burnin_steps,
            evaluated_time_indices_one_based=
                contract.heldout_evaluation_range,
            evaluated_steps_per_trial=
                contract.heldout_evaluated_steps,
            expected_dense_observations=expected_dense_count,
            physical_observation_count_by_coordinate=
                physical_counts,
            semantic_observation_count_by_coordinate=
                semantic_counts,
            spike_positive_observations=
                metrics.spike_auroc.positive_count,
            spike_negative_observations=
                metrics.spike_auroc.negative_count,
            calcium_positive_observations=
                metrics.calcium_auroc.positive_count,
            calcium_negative_observations=
                metrics.calcium_auroc.negative_count,
            sparse_auxiliary_metrics_use_observed_times_only=true,
            interpolated_auxiliary_values_excluded_from_gate=true,
            training_window_contract_is_distinct=true,
        ),
    )
end

end # module OfficialElevenStateStreamingTrainingV3
