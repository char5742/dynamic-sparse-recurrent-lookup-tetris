"""
Production entry point for the detailed->digital-twin->11-state bridge.

`prepare_distillation_dataset_final.jl` was created while the Windows
workspace ACL helper prevented in-place correction.  Load its source with
the one lexical correction required by Julia (`local` is reserved), then add
the fail-closed held-out recomputation gate here.  The prepared artifact is
written to a staging path and is published only after the frozen twin is
re-evaluated against official detailed-teacher test targets.
"""

if !isdefined(Main, :DistillationDatasetBridgeFinal)
    source_path = joinpath(@__DIR__, "prepare_distillation_dataset_final.jl")
    source = read(source_path, String)
    source = replace(
        source,
        "for (local, code) in enumerate(codes)" =>
            "for (local_index, code) in enumerate(codes)",
    )
    source = replace(
        source,
        "push!(chosen, local)" => "push!(chosen, local_index)",
    )
    Base.include_string(Main, source, source_path)
end

module DistillationDatasetBridgeProductionV2

using JLD2
using JSON3
using SHA
using Statistics

const Bridge = Main.DistillationDatasetBridgeFinal
const Twin = Main.PaperDigitalTwin

export PREPARED_DATASET_SCHEMA,
    OFFICIAL_NEURON_SCHEMA,
    PrepareDistillationConfig,
    prepare_distillation_dataset,
    main

const PREPARED_DATASET_SCHEMA = Bridge.PREPARED_DATASET_SCHEMA
const OFFICIAL_NEURON_SCHEMA = Bridge.OFFICIAL_NEURON_SCHEMA

Base.@kwdef struct PrepareDistillationConfig
    dataset_path::String
    frozen_twin_path::String
    output_path::String
    source_kind::Symbol = :official_neuron
    maximum_train_samples::Int = typemax(Int)
    maximum_validation_samples::Int = typemax(Int)
    maximum_test_samples::Int = typemax(Int)
    twin_batch_size::Int = 8
    minimum_twin_spike_auroc::Float64 = 0.985
    expected_source_dataset_sha256::String = ""
    expected_modeldb_source_sha256::String = ""
    expected_detailed_teacher_sha256::String = ""
    expected_detailed_kernel_sha256::String = ""
    expected_morphology_sha256::String = ""
    expected_twin_parameter_sha256::String = ""
    expected_twin_artifact_sha256::String = ""
    selected_dendritic_segments::Vector{Int} = Int[]
end

function _base_config(
    config::PrepareDistillationConfig,
    output_path::AbstractString,
)
    return Bridge.PrepareDistillationConfig(
        dataset_path=config.dataset_path,
        frozen_twin_path=config.frozen_twin_path,
        output_path=String(output_path),
        source_kind=config.source_kind,
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        twin_batch_size=config.twin_batch_size,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        expected_source_dataset_sha256=
            config.expected_source_dataset_sha256,
        expected_modeldb_source_sha256=
            config.expected_modeldb_source_sha256,
        expected_detailed_teacher_sha256=
            config.expected_detailed_teacher_sha256,
        expected_detailed_kernel_sha256=
            config.expected_detailed_kernel_sha256,
        expected_morphology_sha256=
            config.expected_morphology_sha256,
        expected_twin_parameter_sha256=
            config.expected_twin_parameter_sha256,
        expected_twin_artifact_sha256=
            config.expected_twin_artifact_sha256,
        selected_dendritic_segments=
            config.selected_dendritic_segments,
    )
end

mutable struct _Moments
    count::Int
    sum_x::Float64
    sum_y::Float64
    sum_x2::Float64
    sum_y2::Float64
    sum_xy::Float64
    sum_error2::Float64
end

_Moments() = _Moments(0, 0, 0, 0, 0, 0, 0)

function _update!(moments::_Moments, prediction, target)
    size(prediction) == size(target) ||
        throw(DimensionMismatch("gate prediction/target shape differs"))
    @inbounds for (x_raw, y_raw) in zip(prediction, target)
        x = Float64(x_raw)
        y = Float64(y_raw)
        isfinite(x) && isfinite(y) ||
            error("held-out gate encountered a non-finite target")
        moments.count += 1
        moments.sum_x += x
        moments.sum_y += y
        moments.sum_x2 += x * x
        moments.sum_y2 += y * y
        moments.sum_xy += x * y
        difference = x - y
        moments.sum_error2 += difference * difference
    end
    return moments
end

function _finish(moments::_Moments)
    moments.count > 0 || error("held-out metric stream is empty")
    count = Float64(moments.count)
    covariance =
        moments.sum_xy - moments.sum_x * moments.sum_y / count
    variance_x =
        moments.sum_x2 - moments.sum_x * moments.sum_x / count
    variance_y =
        moments.sum_y2 - moments.sum_y * moments.sum_y / count
    correlation = if variance_x > 0 && variance_y > 0
        covariance / sqrt(variance_x * variance_y)
    else
        NaN
    end
    return (;
        rmse=sqrt(moments.sum_error2 / count),
        correlation,
        count=moments.count,
    )
end

function _binary_auroc(score::Vector{Float64}, label::Vector{Float64})
    length(score) == length(label) ||
        throw(DimensionMismatch("spike AUROC arrays differ"))
    positive = findall(>=(0.5), label)
    negative = findall(<(0.5), label)
    isempty(positive) &&
        error("official held-out target contains no soma-spike positives")
    isempty(negative) &&
        error("official held-out target contains no soma-spike negatives")
    wins = 0.0
    @inbounds for positive_index in positive
        for negative_index in negative
            wins += score[positive_index] > score[negative_index] ? 1.0 :
                score[positive_index] == score[negative_index] ? 0.5 : 0.0
        end
    end
    return wins / (length(positive) * length(negative))
end

function _official_segment_regions(source)
    records = Bridge._value(source.manifest, :segments)
    records === nothing && return String[]
    total = Int(Bridge._value(source.manifest, :total_segments, 0))
    result = fill("", total)
    for record in records
        index = Int(Bridge._value(record, :index, 0))
        1 <= index <= total ||
            error("official segment catalog index is invalid")
        result[index] = String(
            Bridge._value(record, :region_name, ""),
        )
    end
    all(!isempty, result) ||
        error("official segment catalog has missing region labels")
    return result
end

function _recompute_gate(
    base_config,
    staged_dataset,
)
    source = Bridge._load_source(base_config)
    plan, _, _ = Bridge._sample_plan(source, base_config)
    voltage_moments = _Moments()
    nmda_moments = _Moments()
    spike_score = Float64[]
    spike_label = Float64[]
    spike_bce_sum = 0.0
    spike_count = 0
    cursor = 1
    held_out_samples = 0

    for item in plan
        shard = Bridge._unwrap_shard(Bridge._read_shard(item.path))
        local_count = length(item.indices)
        destination = cursor:(cursor + local_count - 1)
        held_out_local = findall(==(Bridge.TEST_SPLIT), item.split_code)
        if !isempty(held_out_local)
            held_out_samples += length(held_out_local)
            output_indices = first(destination) .+ held_out_local .- 1
            source_indices = item.indices[held_out_local]
            detailed_voltage = Float32.(
                @view Bridge._value(
                    shard,
                    :target_voltage,
                )[:, source_indices]
            )
            detailed_spike = Float32.(
                @view Bridge._value(
                    shard,
                    :target_spike,
                )[:, source_indices]
            )
            detailed_nmda = Float32.(
                @view Bridge._value(
                    shard,
                    :target_nmda,
                )[:, :, source_indices]
            )
            predicted_voltage = @view(
                staged_dataset.target_voltage[:, output_indices]
            )
            predicted_spike = @view(
                staged_dataset.target_spike[:, output_indices]
            )
            predicted_nmda = @view(
                staged_dataset.target_nmda[:, :, output_indices]
            )
            _update!(
                voltage_moments,
                predicted_voltage,
                detailed_voltage,
            )
            _update!(nmda_moments, predicted_nmda, detailed_nmda)
            append!(spike_score, vec(Float64.(predicted_spike)))
            append!(spike_label, vec(Float64.(detailed_spike)))
            probability = clamp.(
                Float64.(predicted_spike),
                1.0e-7,
                1.0 - 1.0e-7,
            )
            spike_bce_sum += sum(
                -Float64.(detailed_spike) .* log.(probability) .-
                (1.0 .- Float64.(detailed_spike)) .* log1p.(-probability),
            )
            spike_count += length(detailed_spike)
        end
        cursor += local_count
    end
    cursor == size(staged_dataset.input, 3) + 1 ||
        error("held-out gate/source sample order mismatch")
    held_out_samples > 0 ||
        error("official held-out gate has no test trajectories")
    voltage = _finish(voltage_moments)
    nmda = _finish(nmda_moments)
    spike_auroc = _binary_auroc(spike_score, spike_label)
    spike_accuracy = sum(
        (spike_score .>= 0.5) .== (spike_label .>= 0.5),
    ) / length(spike_label)
    return (;
        source="recomputed by production bridge against held-out " *
            "official detailed-teacher targets",
        self_report_trusted=false,
        held_out_samples,
        held_out_bins=length(spike_label),
        voltage_rmse=voltage.rmse,
        voltage_correlation=voltage.correlation,
        spike_auroc,
        spike_bce=spike_bce_sum / spike_count,
        spike_accuracy,
        nmda_rmse=nmda.rmse,
        nmda_correlation=nmda.correlation,
    ), source
end

function _atomic_publish(path::AbstractString, dataset)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".publishing." * string(getpid())
    JLD2.jldsave(temporary; dataset)
    mv(temporary, destination; force=true)
    return destination
end

function prepare_distillation_dataset(
    config::PrepareDistillationConfig,
)
    destination = abspath(config.output_path)
    staging = destination * ".ungated." * string(getpid()) * ".jld2"
    base_config = _base_config(config, staging)
    base_report = Bridge.prepare_distillation_dataset(base_config)
    staged_file = JLD2.load(staging)
    haskey(staged_file, "dataset") ||
        error("base bridge did not create a dataset payload")
    staged = staged_file["dataset"]
    recomputed_gate, source = _recompute_gate(base_config, staged)
    isfinite(recomputed_gate.spike_auroc) ||
        error("recomputed frozen-twin spike AUROC is non-finite")
    recomputed_gate.spike_auroc >=
        config.minimum_twin_spike_auroc || error(
        "frozen twin failed recomputed official held-out gate: " *
        "spike AUROC $(recomputed_gate.spike_auroc) < " *
        "$(config.minimum_twin_spike_auroc). Ungated staging artifact " *
        "was not published.",
    )
    frozen = Twin.load_frozen_twin(config.frozen_twin_path)
    after = Twin.assert_frozen_unchanged(
        frozen;
        expected_artifact_sha256=staged.frozen_twin_artifact_hash,
    )
    after.max_delta == 0.0f0 ||
        error("frozen twin changed during production gate")

    segment_region = _official_segment_regions(source)
    completion_state = String(
        Bridge._value(source.manifest, :completion_state, ""),
    )
    config.source_kind === :official_neuron &&
        completion_state != "complete" &&
        error("official teacher completion_state is not complete")
    updated_metadata = merge(
        staged.metadata,
        (;
            digital_twin_gate_passed=true,
            recomputed_twin_gate,
            twin_artifact_reported_metrics=
                staged.metadata.twin_held_out_metrics,
            twin_gate_source="production recomputation",
            twin_self_report_trusted=false,
            segment_region,
            source_completion_state=completion_state,
            integrity_after_production_gate=after,
        ),
    )
    updated = merge(
        staged,
        (;
            digital_twin_gate_passed=true,
            recomputed_twin_gate,
            segment_region,
            source_completion_state=completion_state,
            metadata=updated_metadata,
        ),
    )
    output_path = _atomic_publish(destination, updated)
    output_file_sha256 = Bridge._sha256_file(output_path)
    return merge(
        base_report,
        (;
            output_path,
            output_file_sha256,
            recomputed_twin_gate,
            digital_twin_gate_passed=true,
            twin_self_report_trusted=false,
            frozen_max_delta_after=after.max_delta,
            staging_path=staging,
        ),
    )
end

function _parse_arguments(arguments)
    values = Dict{String,String}(
        "dataset" => get(ENV, "HD_TWINPROP_TEACHER_PATH", ""),
        "frozen-twin" => get(ENV, "HD_TWINPROP_TWIN_PATH", ""),
        "output" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET",
            joinpath(@__DIR__, "artifacts", "distillation_dataset.jld2"),
        ),
        "source-kind" => get(
            ENV,
            "HD_TWINPROP_SOURCE_KIND",
            "official-neuron",
        ),
        "max-train" => get(ENV, "HD_TWINPROP_MAX_TRAIN", string(typemax(Int))),
        "max-validation" => get(
            ENV,
            "HD_TWINPROP_MAX_VALIDATION",
            string(typemax(Int)),
        ),
        "max-test" => get(ENV, "HD_TWINPROP_MAX_TEST", string(typemax(Int))),
        "twin-batch" => get(ENV, "HD_TWINPROP_TWIN_BATCH", "8"),
        "minimum-twin-spike-auroc" => get(
            ENV,
            "HD_TWINPROP_MIN_TWIN_AUROC",
            "0.985",
        ),
        "expected-source-dataset-sha256" => "",
        "expected-modeldb-source-sha256" => "",
        "expected-detailed-teacher-sha256" => "",
        "expected-detailed-kernel-sha256" => "",
        "expected-morphology-sha256" => "",
        "expected-twin-parameter-sha256" => "",
        "expected-twin-artifact-sha256" => "",
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        key = token[3:end]
        haskey(values, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value after --$key")
        values[key] = arguments[index + 1]
        index += 2
    end
    isempty(values["dataset"]) &&
        error("--dataset or HD_TWINPROP_TEACHER_PATH is required")
    isempty(values["frozen-twin"]) &&
        error("--frozen-twin or HD_TWINPROP_TWIN_PATH is required")
    return PrepareDistillationConfig(
        dataset_path=abspath(values["dataset"]),
        frozen_twin_path=abspath(values["frozen-twin"]),
        output_path=abspath(values["output"]),
        source_kind=Symbol(replace(
            lowercase(values["source-kind"]),
            "-" => "_",
        )),
        maximum_train_samples=parse(Int, values["max-train"]),
        maximum_validation_samples=
            parse(Int, values["max-validation"]),
        maximum_test_samples=parse(Int, values["max-test"]),
        twin_batch_size=parse(Int, values["twin-batch"]),
        minimum_twin_spike_auroc=parse(
            Float64,
            values["minimum-twin-spike-auroc"],
        ),
        expected_source_dataset_sha256=
            values["expected-source-dataset-sha256"],
        expected_modeldb_source_sha256=
            values["expected-modeldb-source-sha256"],
        expected_detailed_teacher_sha256=
            values["expected-detailed-teacher-sha256"],
        expected_detailed_kernel_sha256=
            values["expected-detailed-kernel-sha256"],
        expected_morphology_sha256=
            values["expected-morphology-sha256"],
        expected_twin_parameter_sha256=
            values["expected-twin-parameter-sha256"],
        expected_twin_artifact_sha256=
            values["expected-twin-artifact-sha256"],
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset(_parse_arguments(arguments))
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeProductionV2

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeProductionV2.main()
end
