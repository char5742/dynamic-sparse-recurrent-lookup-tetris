module StreamingTwinCacheVerification

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :StreamingReleaseDataset)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "StreamingReleaseDataset.jl"),
    )
end
if !isdefined(_PARENT, :PaperDigitalTwin)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "PaperDigitalTwin.jl"),
    )
end

const StreamData = getfield(_PARENT, :StreamingReleaseDataset)
const Twin = getfield(_PARENT, :PaperDigitalTwin)

export verify_primary_frozen_twin_cache

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function verify_primary_frozen_twin_cache(
    dataset::StreamData.StreamDataset,
    frozen;
    time_chunk::Integer=256,
    tolerance::Real=2.0e-5,
)
    chunk = Int(time_chunk)
    chunk >= 1 || throw(ArgumentError("time_chunk must be positive"))
    error_tolerance = Float64(tolerance)
    error_tolerance >= 0 && isfinite(error_tolerance) ||
        throw(ArgumentError("cache tolerance must be finite/nonnegative"))
    _get(dataset.manifest, :event_order, "") ==
        "time_then_axon" ||
        error("release manifest lacks time-ordered compact events")
    before = Twin.assert_frozen_unchanged(frozen)
    maximum_voltage_error = 0.0
    maximum_spike_error = 0.0
    maximum_nmda_error = 0.0
    verified_samples = 0
    verified_time_bins = 0
    peak_input_chunk_bytes = 0
    for (shard_index, record) in enumerate(dataset.records)
        shard = StreamData._load_shard(dataset, shard_index)
        _get(shard, :event_order, "") == "time_then_axon" ||
            error("release shard lacks time-ordered compact events")
        for local_index in 1:record.samples
            memory = nothing
            for first_time in chunk:chunk:0
                error("unreachable")
            end
            for first_time in 1:chunk:dataset.time_steps
                last_time = min(
                    first_time + chunk - 1,
                    dataset.time_steps,
                )
                count = last_time - first_time + 1
                raw = zeros(Float32, dataset.input_dim, count)
                StreamData._fill_raw_window!(
                    raw,
                    shard,
                    local_index,
                    first_time,
                    last_time,
                )
                peak_input_chunk_bytes = max(
                    peak_input_chunk_bytes,
                    sizeof(Float32) * length(raw),
                )
                prediction = Twin.twin_forward(
                    frozen,
                    reshape(raw, dataset.input_dim, count, 1);
                    initial_memory=memory,
                )
                voltage_error = maximum(
                    abs,
                    Float64.(vec(prediction.voltage)) .-
                    Float64.(
                        @view(
                            StreamData._required(
                                shard,
                                :target_voltage,
                            )[
                                first_time:last_time,
                                local_index,
                            ]
                        )
                    ),
                )
                spike_error = maximum(
                    abs,
                    Float64.(vec(prediction.spike_probability)) .-
                    Float64.(
                        @view(
                            StreamData._required(
                                shard,
                                :target_spike,
                            )[
                                first_time:last_time,
                                local_index,
                            ]
                        )
                    ),
                )
                nmda_error = maximum(
                    abs,
                    Float64.(reshape(prediction.nmda, 4, count)) .-
                    Float64.(
                        @view(
                            StreamData._required(
                                shard,
                                :target_nmda,
                            )[
                                :,
                                first_time:last_time,
                                local_index,
                            ]
                        )
                    ),
                )
                maximum_voltage_error =
                    max(maximum_voltage_error, voltage_error)
                maximum_spike_error =
                    max(maximum_spike_error, spike_error)
                maximum_nmda_error =
                    max(maximum_nmda_error, nmda_error)
                maximum((
                    voltage_error,
                    spike_error,
                    nmda_error,
                )) <= error_tolerance || error(
                    "cached primary frozen-twin target differs: " *
                    "shard=$(record.relative_path) " *
                    "sample=$local_index time=$first_time:$last_time " *
                    "voltage=$voltage_error spike=$spike_error " *
                    "nmda=$nmda_error tolerance=$error_tolerance",
                )
                memory = prediction.final_memory
                verified_time_bins += count
            end
            verified_samples += 1
        end
    end
    verified_samples == dataset.total_samples ||
        error("not every stream sample was cache verified")
    verified_time_bins ==
        dataset.total_samples * dataset.time_steps ||
        error("not every stream time bin was cache verified")
    after = Twin.assert_frozen_unchanged(frozen)
    before == after ||
        error("frozen twin changed during primary-cache verification")
    return (;
        passed=true,
        verified_all_samples=true,
        verified_all_time_bins=true,
        verified_samples,
        verified_time_bins,
        tolerance=error_tolerance,
        maximum_voltage_error,
        maximum_spike_probability_error=maximum_spike_error,
        maximum_nmda_error,
        peak_input_chunk_bytes,
        primary_targets=(
            "soma_voltage",
            "soma_spike_logit_probability",
            "soma_basal_apical_trunk_apical_tuft_nmda",
        ),
        frozen_parameter_sha256=frozen.parameter_sha256,
        frozen_artifact_sha256=frozen.artifact_sha256,
        integrity_before=before,
        integrity_after=after,
    )
end

end # module StreamingTwinCacheVerification
