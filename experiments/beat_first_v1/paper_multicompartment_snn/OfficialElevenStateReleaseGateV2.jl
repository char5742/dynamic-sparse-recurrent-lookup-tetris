module OfficialElevenStateReleaseGateV2

# Immutable per-coordinate release gate.  No mean-over-regions or
# mean-over-branches criterion is permitted: one catastrophic coordinate must
# fail the candidate.

export MAXIMUM_DENDRITIC_RMSE_MV,
    MAXIMUM_NMDA_NORMALIZED_RMSE,
    MAXIMUM_VOLTAGE_RMSE_MV,
    MINIMUM_CALCIUM_AUROC,
    MINIMUM_NMDA_CORRELATION,
    MINIMUM_SPIKE_AUROC,
    MINIMUM_VOLTAGE_CORRELATION,
    strict_release_gate

const MAXIMUM_VOLTAGE_RMSE_MV = 5.0
const MINIMUM_VOLTAGE_CORRELATION = 0.90
const MAXIMUM_NMDA_NORMALIZED_RMSE = 1.0
const MINIMUM_NMDA_CORRELATION = 0.80
const MINIMUM_SPIKE_AUROC = 0.985
const MINIMUM_CALCIUM_AUROC = 0.80
const MAXIMUM_DENDRITIC_RMSE_MV = 8.0

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("strict gate metric lacks $(String(name))")
    return value
end

function _exact_length(values, expected, label)
    result = collect(values)
    length(result) == expected ||
        error("$label must contain $expected entries")
    return result
end

function _positive_counts(metrics, name::Symbol)
    values = Int.(_exact_length(_required(metrics, name), 11, String(name)))
    all(>(0), values) ||
        error("strict gate has a zero-observation coordinate in $(String(name))")
    return values
end

"""
Gate exactly one final test evaluation.

`target_scale[3:6]` converts each physical NMDA RMSE to its independently
normalized error.  Every NMDA region, every dendritic voltage branch, and
every semantic coordinate must pass its own threshold.
"""
function strict_release_gate(
    test_metrics,
    target_scale;
    minimum_spike_auroc::Real=MINIMUM_SPIKE_AUROC,
    expected_test_indices_sha256::AbstractString,
)
    minimum_spike_auroc >= MINIMUM_SPIKE_AUROC ||
        throw(ArgumentError("spike gate cannot be weaker than 0.985"))
    String(_required(test_metrics, :evaluation_split)) == "test" ||
        error("release gate accepts only the exact test split")
    _required(test_metrics, :exact_dataset_split) === true ||
        error("release metrics were computed on caller-selected indices")
    String(_required(test_metrics, :evaluated_indices_sha256)) ==
        String(expected_test_indices_sha256) ||
        error("release test index identity differs from the dataset")
    Int(_required(test_metrics, :evaluated_index_count)) ==
        Int(_required(test_metrics, :samples)) ||
        error("release test sample count differs")
    Tuple(Int.(collect(_required(
        test_metrics,
        :evaluated_time_indices_one_based,
    )))) == (
        501,
        500 + Int(_required(
            test_metrics,
            :evaluated_steps_per_trial,
        )),
    ) || error("release test time interval differs")
    Int(_required(test_metrics, :state_warmup_steps)) == 500 ||
        error("release test did not warm recurrent state for 500 steps")

    physical_counts = _positive_counts(
        test_metrics,
        :physical_observation_count_by_coordinate,
    )
    semantic_counts = _positive_counts(
        test_metrics,
        :semantic_observation_count_by_coordinate,
    )
    dense_expected =
        Int(_required(test_metrics, :expected_dense_observations))
    all(==(dense_expected), physical_counts[1:6]) ||
        error("primary test targets are not dense over the fixed interval")
    for name in (
        :spike_positive_observations,
        :spike_negative_observations,
        :calcium_positive_observations,
        :calcium_negative_observations,
    )
        Int(_required(test_metrics, name)) > 0 ||
            error("strict gate lacks both classes for $(String(name))")
    end
    _required(
        test_metrics,
        :sparse_auxiliary_metrics_use_observed_times_only,
    ) === true ||
        error("sparse auxiliary metrics included interpolated targets")
    _required(
        test_metrics,
        :interpolated_auxiliary_values_excluded_from_gate,
    ) === true ||
        error("interpolated auxiliary values entered the gate")
    _required(test_metrics, :auroc_is_conservative_lower_bound) === true ||
        error("AUROC gate is not a conservative lower bound")

    voltage_rmse =
        Float64(_required(test_metrics, :soma_voltage_rmse_mv))
    voltage_correlation =
        Float64(_required(test_metrics, :soma_voltage_correlation))
    voltage_passed =
        isfinite(voltage_rmse) &&
        isfinite(voltage_correlation) &&
        voltage_rmse <= MAXIMUM_VOLTAGE_RMSE_MV &&
        voltage_correlation >= MINIMUM_VOLTAGE_CORRELATION

    scales = Float64.(_exact_length(target_scale, 11, "target_scale"))
    all(value -> isfinite(value) && value > 0.0, scales) ||
        error("strict gate target scales are invalid")
    nmda_rmse = Float64.(_exact_length(
        _required(test_metrics, :nmda_rmse_by_region),
        4,
        "nmda_rmse_by_region",
    ))
    nmda_correlation = Float64.(_exact_length(
        _required(test_metrics, :nmda_correlation_by_region),
        4,
        "nmda_correlation_by_region",
    ))
    nmda_normalized_rmse = nmda_rmse ./ scales[3:6]
    nmda_region_passed = [
        isfinite(nmda_normalized_rmse[region]) &&
        isfinite(nmda_correlation[region]) &&
        nmda_normalized_rmse[region] <=
            MAXIMUM_NMDA_NORMALIZED_RMSE &&
        nmda_correlation[region] >= MINIMUM_NMDA_CORRELATION
        for region in 1:4
    ]
    nmda_passed = all(nmda_region_passed)

    spike_auroc = Float64(_required(test_metrics, :spike_auroc))
    spike_passed =
        isfinite(spike_auroc) &&
        spike_auroc >= Float64(minimum_spike_auroc)
    calcium_auroc =
        Float64(_required(test_metrics, :calcium_event_auroc))
    calcium_passed =
        isfinite(calcium_auroc) &&
        calcium_auroc >= MINIMUM_CALCIUM_AUROC

    dendritic_rmse = Float64.(_exact_length(
        _required(test_metrics, :dendritic_voltage_rmse_mv),
        4,
        "dendritic_voltage_rmse_mv",
    ))
    dendritic_branch_passed = [
        isfinite(value) && value <= MAXIMUM_DENDRITIC_RMSE_MV
        for value in dendritic_rmse
    ]
    dendritic_voltage_passed = all(dendritic_branch_passed)

    semantic_coordinate_passed = Bool.(_exact_length(
        _required(test_metrics, :semantic_coordinate_passed),
        11,
        "semantic_coordinate_passed",
    ))
    semantic_state_passed = all(semantic_coordinate_passed)
    observation_counts_passed =
        all(>(0), physical_counts) &&
        all(>(0), semantic_counts)
    multi_target_passed =
        voltage_passed &&
        nmda_passed &&
        calcium_passed &&
        dendritic_voltage_passed &&
        semantic_state_passed &&
        observation_counts_passed
    return (;
        passed=spike_passed && multi_target_passed,
        gate_schema="hd_swsnn.eleven_state.strict_gate.final.v2",
        minimum_spike_auroc=Float64(minimum_spike_auroc),
        held_out_spike_auroc=spike_auroc,
        spike_passed,
        multi_target_passed,
        voltage_passed,
        voltage_thresholds=(;
            maximum_rmse_mv=MAXIMUM_VOLTAGE_RMSE_MV,
            minimum_correlation=MINIMUM_VOLTAGE_CORRELATION,
        ),
        nmda_passed,
        nmda_region_passed,
        nmda_normalized_rmse,
        nmda_correlation_by_region=nmda_correlation,
        nmda_thresholds=(;
            maximum_normalized_rmse=
                MAXIMUM_NMDA_NORMALIZED_RMSE,
            minimum_correlation=MINIMUM_NMDA_CORRELATION,
        ),
        calcium_passed,
        minimum_calcium_auroc=MINIMUM_CALCIUM_AUROC,
        dendritic_voltage_passed,
        dendritic_branch_passed,
        maximum_dendritic_rmse_mv=MAXIMUM_DENDRITIC_RMSE_MV,
        semantic_state_passed,
        semantic_coordinate_passed,
        observation_counts_passed,
        physical_observation_count_by_coordinate=physical_counts,
        semantic_observation_count_by_coordinate=semantic_counts,
        exact_test_indices_sha256=
            String(expected_test_indices_sha256),
        per_region_and_branch_gating=true,
    )
end

end # module OfficialElevenStateReleaseGateV2
