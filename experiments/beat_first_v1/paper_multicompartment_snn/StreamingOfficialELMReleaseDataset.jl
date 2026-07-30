module StreamingOfficialELMReleaseDataset

"""
Bounded-memory reader for the sealed official Paper ELM release.

This module deliberately has no method accepting `PaperDigitalTwin.FrozenTwin`
or a bare `FrozenOfficialELMTwin`.  The only public entry points which bind a
teacher require the capability returned after independently recomputing the
release attestation and held-out gates.
"""

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :StreamingReleaseDataset)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "StreamingReleaseDataset.jl"),
    )
end
if !isdefined(_PARENT, :PaperELMTwinOfficialV2ReleaseExecution)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2ReleaseExecution.jl",
        ),
    )
end

const BaseStream = getfield(_PARENT, :StreamingReleaseDataset)
const Execution =
    getfield(_PARENT, :PaperELMTwinOfficialV2ReleaseExecution)
const ELM = Execution.ELM

export OFFICIAL_ELM_INPUT_DIM,
    OFFICIAL_ELM_EXECUTION_TYPE,
    open_official_stream_dataset,
    official_stream_materialize_window,
    verify_primary_cache_against_live_official_elm!,
    stream_dataset_integrity!,
    stream_target_statistics

const OFFICIAL_ELM_INPUT_DIM = 1_278
const OFFICIAL_DENDRITIC_LOCATIONS = 639
const OFFICIAL_ELM_EXECUTION_TYPE =
    "PaperELMTwinOfficialV2ReleaseExecution.VerifiedOfficialELMExecution"
const EVENT_ORDER = "time_then_axon"
const INPUT_SEMANTICS =
    "E:+strength*event_count, I:-strength*event_count"

const StreamDataset = BaseStream.StreamDataset
const stream_dataset_integrity! = BaseStream.stream_dataset_integrity!
const stream_target_statistics = BaseStream.stream_target_statistics

@inline _get(object, name::Symbol, default=nothing) =
    BaseStream._get(object, name, default)

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("official streaming field $(String(name)) is absent")
    return value
end

function _official_execution_type(execution)
    type = typeof(execution)
    return string(parentmodule(type), ".", nameof(type))
end

function _assert_official_execution(
    execution::Execution.VerifiedOfficialELMExecution,
)
    Execution.assert_verified_release_unchanged!(execution)
    frozen = Execution.verified_release_frozen(execution)
    frozen.model.config.num_input == OFFICIAL_ELM_INPUT_DIM ||
        error("sealed official ELM does not expose 1278 inputs")
    frozen.model.config.num_branch == 45 ||
        error("sealed official ELM branch count differs")
    frozen.model.config.num_synapse_per_branch == 100 ||
        error("sealed official ELM branch fan-in differs")
    return frozen
end

function _require_manifest_official_contract(manifest, execution)
    String(_required(manifest, :event_order)) == EVENT_ORDER ||
        error("official shard events are not ordered time_then_axon")
    input_layout = _required(manifest, :input_layout)
    Int(_required(input_layout, :input_dim)) == OFFICIAL_ELM_INPUT_DIM ||
        error("official release manifest input_dim is not 1278")
    Int(_required(input_layout, :dendritic_locations)) ==
        OFFICIAL_DENDRITIC_LOCATIONS ||
        error("official release manifest does not expose 639 locations")
    String(_required(input_layout, :input_semantics)) ==
        INPUT_SEMANTICS ||
        error("official release signed E/I semantics differ")
    _get(manifest, :input_static_plane, false) === false ||
        error("official ELM input must not contain a static plane")

    declared_type = String(_required(manifest, :digital_twin_type))
    declared_type in (
        OFFICIAL_ELM_EXECUTION_TYPE,
        "VerifiedOfficialELMExecution",
    ) || error("dataset was not built by the sealed execution capability")
    _official_execution_type(execution) == OFFICIAL_ELM_EXECUTION_TYPE ||
        error("runtime official ELM capability has the wrong Julia type")

    bundle = Execution.verified_release_bundle(execution)
    attestation = String(_required(
        manifest,
        :verified_attestation_sha256,
    ))
    attestation == execution.attestation_sha256 ||
        error("dataset/execution release attestation digest mismatch")
    attestation == bundle.attestation.attestation_sha256 ||
        error("sealed execution attestation changed")
    return true
end

"""
Open the final-v2 shards using only a sealed official ELM execution capability.

The legacy reader supplies the lineage/range/hash/split validation.  A private
shape-only adapter is used solely to reuse those manifest checks; no legacy
twin is loaded or called.  The returned dataset has the real 1278 input
dimension and is subsequently materialized only by this module.
"""
function open_official_stream_dataset(
    path::AbstractString,
    execution::Execution.VerifiedOfficialELMExecution;
    minimum_spike_auroc::Real=0.985,
    verify_shard_hashes::Bool=true,
    require_promotion_eligible::Bool=true,
)
    frozen = _assert_official_execution(execution)
    adapter = (;
        model=(; config=(; segments=642, input_dim=6 * 642)),
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
    )
    parsed = BaseStream.open_stream_dataset(
        path,
        adapter;
        minimum_spike_auroc,
        verify_shard_hashes,
        require_promotion_eligible,
    )
    _require_manifest_official_contract(parsed.manifest, execution)
    provenance = merge(
        parsed.provenance,
        (;
            official_elm_execution_type=
                _official_execution_type(execution),
            official_elm_attestation_sha256=
                execution.attestation_sha256,
            official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
            official_elm_input_semantics=INPUT_SEMANTICS,
        ),
    )
    return StreamDataset(
        parsed.manifest_path,
        parsed.root,
        parsed.manifest_sha256,
        parsed.dataset_sha256,
        parsed.manifest,
        parsed.records,
        parsed.total_samples,
        parsed.time_steps,
        OFFICIAL_ELM_INPUT_DIM,
        parsed.train_indices,
        parsed.validation_indices,
        parsed.test_indices,
        parsed.split_code,
        parsed.global_to_shard,
        parsed.diagnostic_time_indices,
        parsed.segment_region,
        parsed.segment_catalog_sha256,
        provenance,
        parsed.frozen_twin_file_sha256,
        parsed.verified_shard_hashes,
        parsed.tracker,
    )
end

function _offset_range(offsets, local_index, total, label)
    return BaseStream._offset_range(
        offsets,
        local_index,
        total,
        label,
    )
end

@inline function _contact_channel(segment::Int, kind::UInt8)
    2 <= segment <= 640 ||
        error("official ELM contacts must target Hay dendrites 2:640")
    location = segment - 1
    kind == UInt8(1) && return location
    kind == UInt8(2) &&
        return OFFICIAL_DENDRITIC_LOCATIONS + location
    error("official ELM contact violates Dale E/I coding")
end

function _assert_event_order(event_time, event_axon, event_range)
    previous_time = typemin(Int)
    previous_axon = typemin(Int)
    @inbounds for event in event_range
        time = event_time[event]
        axon = event_axon[event]
        (
            time > previous_time ||
            (time == previous_time && axon >= previous_axon)
        ) || error("compact events violate time_then_axon order")
        previous_time = time
        previous_axon = axon
    end
    return true
end

"""
Expand one compact trial into the exact official 1278 signed input.

Channels 1:639 contain positive excitatory strength-weighted event counts;
channels 640:1278 contain negative inhibitory values.  There is no duplicated
AMPA/NMDA plane and no static-strength plane.
"""
function _fill_official_raw_window!(
    destination::AbstractMatrix{Float32},
    shard,
    local_index::Int,
    first_time::Int,
    last_time::Int,
)
    size(destination) ==
        (OFFICIAL_ELM_INPUT_DIM, last_time - first_time + 1) ||
        throw(DimensionMismatch("official raw destination shape differs"))
    fill!(destination, 0.0f0)

    contact_axon = Int.(vec(_required(shard, :contact_axon)))
    contact_segment = Int.(vec(_required(shard, :contact_segment)))
    contact_kind = UInt8.(vec(_required(shard, :contact_kind)))
    contact_strength =
        Float32.(vec(_required(shard, :contact_strength)))
    contact_range = _offset_range(
        vec(_required(shard, :contact_trial_offset)),
        local_index,
        length(contact_axon),
        "contact",
    )
    contacts_by_axon = Dict{Int,Vector{Int}}()
    @inbounds for contact in contact_range
        _contact_channel(
            contact_segment[contact],
            contact_kind[contact],
        )
        strength = contact_strength[contact]
        isfinite(strength) && strength >= 0.0f0 ||
            error("official ELM contact strength is invalid")
        push!(
            get!(contacts_by_axon, contact_axon[contact], Int[]),
            contact,
        )
    end

    event_axon = Int.(vec(_required(shard, :event_axon)))
    event_time = Int.(vec(_required(shard, :event_time_bin)))
    event_count = UInt8.(vec(_required(shard, :event_count)))
    event_range = _offset_range(
        vec(_required(shard, :event_trial_offset)),
        local_index,
        length(event_axon),
        "event",
    )
    _assert_event_order(event_time, event_axon, event_range)
    isempty(event_range) && return destination
    local_times = @view event_time[event_range]
    first_event = searchsortedfirst(local_times, first_time - 1)
    last_event = searchsortedlast(local_times, last_time - 1)
    first_event > last_event && return destination
    offset = first(event_range) - 1
    @inbounds for relative in first_event:last_event
        event = offset + relative
        global_time = event_time[event] + 1
        local_time = global_time - first_time + 1
        multiplicity = Float32(event_count[event])
        multiplicity > 0.0f0 ||
            error("official ELM event multiplicity is not positive")
        for contact in get(
            contacts_by_axon,
            event_axon[event],
            Int[],
        )
            channel = _contact_channel(
                contact_segment[contact],
                contact_kind[contact],
            )
            signed_strength =
                contact_kind[contact] == UInt8(1) ?
                contact_strength[contact] :
                -contact_strength[contact]
            destination[channel, local_time] +=
                signed_strength * multiplicity
        end
    end
    all(isfinite, destination) ||
        error("official ELM stream-expanded input is non-finite")
    all(value -> value >= 0.0f0, @view(destination[1:639, :])) ||
        error("official excitatory input became negative")
    all(value -> value <= 0.0f0, @view(destination[640:1278, :])) ||
        error("official inhibitory input became positive")
    return destination
end

function _validate_logit_target(shard, time_steps, samples)
    logit = _required(shard, :target_spike_logit)
    size(logit) == (time_steps, samples) ||
        error("official cached spike-logit shape differs")
    all(isfinite, logit) ||
        error("official cached spike logits are non-finite")
    return logit
end

function official_stream_materialize_window(
    dataset::StreamDataset,
    global_indices,
    first_time::Integer,
    window::Integer,
)
    dataset.input_dim == OFFICIAL_ELM_INPUT_DIM ||
        error("dataset is not an official 1278-input stream")
    samples = Int.(collect(global_indices))
    isempty(samples) &&
        throw(ArgumentError("stream window needs at least one sample"))
    first = Int(first_time)
    count = Int(window)
    count >= 1 || throw(ArgumentError("stream window must be positive"))
    last = first + count - 1
    1 <= first <= last <= dataset.time_steps ||
        throw(BoundsError(1:dataset.time_steps, first:last))
    all(index -> 1 <= index <= dataset.total_samples, samples) ||
        throw(BoundsError(1:dataset.total_samples, samples))

    batch = length(samples)
    raw_input =
        zeros(Float32, OFFICIAL_ELM_INPUT_DIM, count, batch)
    target = zeros(Float32, 11, count, batch)
    target_spike_logit = zeros(Float32, count, batch)
    observed = falses(11, count, batch)
    shard_indices = Int.(dataset.global_to_shard[samples])
    order = sortperm(shard_indices)
    current_shard_index = 0
    current_shard = nothing
    current_shard_bytes = 0
    for ordered_position in order
        global_index = samples[ordered_position]
        shard_index = shard_indices[ordered_position]
        if shard_index != current_shard_index
            current_shard =
                BaseStream._load_shard(dataset, shard_index)
            current_shard_index = shard_index
            current_shard_bytes =
                dataset.records[shard_index].bytes
            _validate_logit_target(
                current_shard,
                dataset.time_steps,
                dataset.records[shard_index].samples,
            )
        end
        record = dataset.records[shard_index]
        local_index = global_index - record.global_first + 1
        _fill_official_raw_window!(
            @view(raw_input[:, :, ordered_position]),
            current_shard,
            local_index,
            first,
            last,
        )
        BaseStream._fill_target_window!(
            @view(target[:, :, ordered_position]),
            @view(observed[:, :, ordered_position]),
            dataset,
            current_shard,
            local_index,
            first,
            last,
        )
        target_spike_logit[:, ordered_position] .= @view(
            _required(current_shard, :target_spike_logit)[
                first:last,
                local_index,
            ]
        )
        window_bytes =
            sizeof(Float32) * (
                length(raw_input) +
                length(target) +
                length(target_spike_logit)
            ) +
            sizeof(Bool) * length(observed)
        dataset.tracker.peak_dense_window_bytes = max(
            dataset.tracker.peak_dense_window_bytes,
            window_bytes,
        )
        dataset.tracker.peak_combined_bytes = max(
            dataset.tracker.peak_combined_bytes,
            current_shard_bytes + window_bytes,
        )
    end
    dataset.tracker.windows_materialized += 1
    dataset.tracker.samples_materialized += batch
    return (; raw_input, target, target_spike_logit, observed)
end

@inline function _maximum_absolute_delta(left, right)
    size(left) == size(right) ||
        throw(DimensionMismatch("live/cache target shapes differ"))
    isempty(left) && return 0.0
    return maximum(abs.(Float64.(left) .- Float64.(right)))
end

function _require_bit_exact(name, live, cached)
    delta = _maximum_absolute_delta(live, cached)
    delta == 0.0 ||
        error("$name cache differs from live official ELM; max_delta=$delta")
    return delta
end

"""
Re-run the sealed official ELM for every sample and every time step.

All six primary trajectories (soma voltage, spike probability, spike logit,
and four regional NMDA currents) must be bit-identical to the shard cache.
Detailed Hay dendritic voltage and calcium remain separate sparse auxiliary
targets and are intentionally not compared to the ELM.
"""
function verify_primary_cache_against_live_official_elm!(
    dataset::StreamDataset,
    execution::Execution.VerifiedOfficialELMExecution;
    time_chunk::Integer=256,
)
    dataset.input_dim == OFFICIAL_ELM_INPUT_DIM ||
        error("primary-cache verification requires official 1278 input")
    chunk = Int(time_chunk)
    chunk >= 1 ||
        throw(ArgumentError("cache verification chunk must be positive"))
    _assert_official_execution(execution)

    maxima = zeros(Float64, 4)
    samples_verified = 0
    time_points_verified = 0
    for (shard_index, record) in enumerate(dataset.records)
        shard = BaseStream._load_shard(dataset, shard_index)
        logit = _validate_logit_target(
            shard,
            dataset.time_steps,
            record.samples,
        )
        for local_index in 1:record.samples
            state = nothing
            for first_time in 1:chunk:dataset.time_steps
                last_time = min(
                    first_time + chunk - 1,
                    dataset.time_steps,
                )
                count = last_time - first_time + 1
                raw = zeros(
                    Float32,
                    OFFICIAL_ELM_INPUT_DIM,
                    count,
                    1,
                )
                _fill_official_raw_window!(
                    @view(raw[:, :, 1]),
                    shard,
                    local_index,
                    first_time,
                    last_time,
                )
                live = Execution.twin_forward_after_verified(
                    execution,
                    raw;
                    normalized=false,
                    initial_state=state,
                )
                state = live.final_state
                maxima[1] = max(
                    maxima[1],
                    _require_bit_exact(
                        "soma voltage",
                        live.voltage,
                        @view(_required(shard, :target_voltage)[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[2] = max(
                    maxima[2],
                    _require_bit_exact(
                        "spike probability",
                        live.spike_probability,
                        @view(_required(shard, :target_spike)[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[3] = max(
                    maxima[3],
                    _require_bit_exact(
                        "spike logit",
                        live.spike_logit,
                        @view(logit[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[4] = max(
                    maxima[4],
                    _require_bit_exact(
                        "NMDA current",
                        live.nmda,
                        @view(_required(shard, :target_nmda)[
                            :,
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                window_bytes =
                    sizeof(Float32) * (
                        length(raw) +
                        length(live.voltage) +
                        length(live.spike_probability) +
                        length(live.spike_logit) +
                        length(live.nmda)
                    )
                dataset.tracker.peak_dense_window_bytes = max(
                    dataset.tracker.peak_dense_window_bytes,
                    window_bytes,
                )
                dataset.tracker.peak_combined_bytes = max(
                    dataset.tracker.peak_combined_bytes,
                    record.bytes + window_bytes,
                )
                time_points_verified += count
            end
            samples_verified += 1
        end
    end
    samples_verified == dataset.total_samples ||
        error("not every release sample was cache-verified")
    time_points_verified ==
        dataset.total_samples * dataset.time_steps ||
        error("not every release time point was cache-verified")
    all(iszero, maxima) ||
        error("primary cache verification was not bit exact")
    Execution.assert_verified_release_unchanged!(execution)
    return (;
        cache_verified_all_samples=true,
        bit_exact=true,
        samples_verified,
        time_points_verified,
        soma_voltage_max_delta=maxima[1],
        spike_probability_max_delta=maxima[2],
        spike_logit_max_delta=maxima[3],
        nmda_max_delta=maxima[4],
        primary_targets=(
            "soma_voltage",
            "spike_probability",
            "spike_logit",
            "nmda_soma",
            "nmda_basal",
            "nmda_apical_trunk",
            "nmda_apical_tuft",
        ),
        detailed_model_auxiliary_targets=(
            "calcium_event_sparse",
            "dendritic_voltage_sparse",
        ),
        official_elm_execution_type=
            _official_execution_type(execution),
        official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
        verified_attestation_sha256=
            execution.attestation_sha256,
    )
end

end # module StreamingOfficialELMReleaseDataset
