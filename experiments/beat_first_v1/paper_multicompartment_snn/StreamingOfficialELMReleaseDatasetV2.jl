module StreamingOfficialELMReleaseDatasetV2

using JLD2

const _PARENT = parentmodule(@__MODULE__)
for (name, file) in (
    (:StreamingReleaseDataset, "StreamingReleaseDataset.jl"),
    (
        :PaperELMTwinOfficialV2SealedRelease,
        "PaperELMTwinOfficialV2SealedRelease.jl",
    ),
    (
        :OfficialSignedInputReconstruction,
        "OfficialSignedInputReconstruction.jl",
    ),
)
    if !isdefined(_PARENT, name)
        Base.include(_PARENT, joinpath(@__DIR__, file))
    end
end

const BaseStream = getfield(_PARENT, :StreamingReleaseDataset)
const Sealed =
    getfield(_PARENT, :PaperELMTwinOfficialV2SealedRelease)
const Signed =
    getfield(_PARENT, :OfficialSignedInputReconstruction)
const Twin = Sealed.Twin

export OFFICIAL_ELM_INPUT_DIM,
    SEALED_EXECUTION_TYPE,
    open_sealed_stream_dataset,
    sealed_stream_materialize_window,
    stream_dataset_integrity!,
    stream_target_statistics,
    verify_primary_cache_against_live_sealed_elm!

const OFFICIAL_ELM_INPUT_DIM = 1_278
const SEALED_EXECUTION_TYPE =
    "PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease"
const StreamDataset = BaseStream.StreamDataset
const stream_dataset_integrity! = BaseStream.stream_dataset_integrity!
const stream_target_statistics = BaseStream.stream_target_statistics

@inline _get(object, name::Symbol, default=nothing) =
    BaseStream._get(object, name, default)

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("sealed stream field $(String(name)) is absent")
    return value
end

function _tuple2(value, label)
    values = Int.(collect(value))
    length(values) == 2 ||
        error("$label must have exactly two bounds")
    return (values[1], values[2])
end

function _assert_sha(value, label)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a complete SHA-256")
    return digest
end

function _assert_time_contract(manifest, time_steps)
    training = _required(manifest, :neuronio_training_window)
    full_time_steps = Int(_required(training, :full_time_steps))
    full_time_steps == time_steps ||
        error("sealed bridge did not preserve the full trajectory")
    Float64(_required(training, :sample_dt_ms)) == 1.0 ||
        error("NeuronIO sample interval differs")
    Float64(_required(training, :ignore_time_from_start_ms)) == 500.0 ||
        error("NeuronIO training ignore interval differs")
    Float64(_required(training, :input_window_size_ms)) == 500.0 ||
        error("NeuronIO training window duration differs")
    input_window_steps = Int(_required(training, :input_window_steps))
    input_window_steps == 500 ||
        error("NeuronIO training window must contain 500 steps")
    starts = _tuple2(
        _required(
            training,
            :valid_window_start_indices_one_based,
        ),
        "NeuronIO start range",
    )
    expected_starts = (501, full_time_steps - input_window_steps)
    starts == expected_starts ||
        error("NeuronIO random-window start bounds differ")
    starts[2] + input_window_steps - 1 ==
        full_time_steps - 1 ||
        error("Python choice-exclusive upper bound was not preserved")
    String(_required(training, :sampling)) ==
        "uniform_with_replacement" ||
        error("NeuronIO training windows are not sampled with replacement")
    _assert_sha(
        _required(training, :reference_commit),
        "NeuronIO reference commit",
    )
    _assert_sha(
        _required(training, :loader_sha256),
        "NeuronIO loader",
    )

    heldout = _required(manifest, :heldout_evaluation_window)
    Float64(_required(heldout, :ignore_time_at_start_ms)) == 500.0 ||
        error("held-out evaluator burn-in differs")
    evaluation = _tuple2(
        _required(
            heldout,
            :evaluated_time_indices_one_based,
        ),
        "held-out evaluation range",
    )
    evaluation == (501, full_time_steps) ||
        error("held-out metrics do not cover every post-burn-in bin")
    Int(_required(heldout, :evaluated_steps_per_trial)) ==
        full_time_steps - 500 ||
        error("held-out evaluated-bin count differs")
    _assert_sha(
        _required(heldout, :evaluator_sha256),
        "held-out evaluator",
    )
    return (;
        full_time_steps,
        training_window_start_range=starts,
        input_window_steps,
        heldout_evaluation_range=evaluation,
        heldout_evaluated_steps=full_time_steps - 500,
    )
end

function _assert_input_contract(manifest)
    String(_required(manifest, :event_order)) ==
        Signed.EVENT_ORDER ||
        error("sealed bridge events are not time_then_axon")
    layout = _required(manifest, :input_layout)
    Int(_required(layout, :total_hay_segments)) == 642 ||
        error("sealed input layout is not the Hay 642-segment catalog")
    _tuple2(_required(layout, :legal_contact_segments), "legal contacts") ==
        (2, 640) ||
        error("sealed input permits soma/axon contacts")
    Int(_required(layout, :excluded_soma_segment)) == 1 ||
        error("sealed input did not exclude soma")
    _tuple2(_required(layout, :excluded_axon_segments), "excluded axon") ==
        (641, 642) ||
        error("sealed input did not exclude the axon")
    Int(_required(layout, :dendritic_locations)) == 639 ||
        error("sealed input does not expose 639 dendritic locations")
    _tuple2(_required(layout, :excitatory_channels), "E channels") ==
        (1, 639) ||
        error("sealed excitatory channel interval differs")
    _tuple2(_required(layout, :inhibitory_channels), "I channels") ==
        (640, 1278) ||
        error("sealed inhibitory channel interval differs")
    Int(_required(layout, :input_dim)) == OFFICIAL_ELM_INPUT_DIM ||
        error("sealed input_dim is not 1278")
    String(_required(layout, :input_semantics)) ==
        Signed.INPUT_SEMANTICS ||
        error("sealed signed E/I semantics differ")
    _required(layout, :input_static_plane) === false ||
        error("sealed official input contains a static plane")
    Int(_required(layout, :num_branch)) == 45 ||
        error("sealed official branch count differs")
    Int(_required(layout, :num_synapse_per_branch)) == 100 ||
        error("sealed official branch fan-in differs")
    String(_required(layout, :input_to_synapse_routing)) ==
        "neuronio_routing" ||
        error("sealed official routing differs")
    return true
end

function _assert_primary_claim(manifest)
    claim = _required(manifest, :primary_cache_live_equality)
    _required(claim, :required) === true ||
        error("sealed bridge does not require live primary verification")
    _required(claim, :all_samples) === true ||
        error("sealed bridge did not live-check all cached samples")
    _required(claim, :bit_exact) === true ||
        error("sealed bridge primary cache is not bit exact")
    Tuple(String.(collect(_required(claim, :targets)))) == (
        "soma_voltage",
        "spike_probability",
        "spike_logit",
        "regional_nmda_current",
    ) || error("sealed bridge primary target identity/order differs")
    Tuple(String.(collect(
        _required(claim, :detailed_auxiliary_excluded),
    ))) == (
        "calcium_event_sparse",
        "dendritic_voltage_sparse",
    ) || error("detailed states were not excluded from primary equality")
    return true
end

function _verify_bundle_and_manifest!(
    bundle::Sealed.SealedOfficialELMRelease,
    manifest,
    source_teacher_manifest,
    source_teacher_shard_directory;
    require_production,
    scratch_root,
)
    Sealed.verify_sealed_official_elm_release(
        bundle,
        source_teacher_manifest,
        source_teacher_shard_directory;
        require_gate=true,
        require_production,
        scratch_root,
    )
    frozen = bundle.frozen
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("sealed bundle does not own the canonical Final frozen type")
    Twin.assert_frozen_official_elm_unchanged(frozen)
    payload = bundle.attestation.payload
    String(_required(manifest, :digital_twin_type)) ==
        SEALED_EXECUTION_TYPE ||
        error("bridge did not consume exact SealedOfficialELMRelease")
    String(_required(manifest, :digital_twin_schema)) ==
        Sealed.SEALED_RELEASE_SCHEMA ||
        error("bridge sealed-release schema differs")
    attestation = _assert_sha(
        _required(manifest, :sealed_attestation_sha256),
        "bridge sealed attestation",
    )
    attestation == bundle.attestation.attestation_sha256 ||
        error("bridge/sealed bundle attestation digest mismatch")
    hashes = _required(manifest, :hashes)
    _assert_sha(
        _required(hashes, :sealed_attestation_sha256),
        "nested bridge sealed attestation",
    ) == attestation ||
        error("top-level/nested sealed attestation digest differs")
    _assert_sha(
        _required(manifest, :source_manifest_sha256),
        "bridge source manifest",
    ) == payload.teacher.manifest_sha256 ||
        error("bridge source manifest differs from sealed teacher")
    _assert_sha(
        _required(manifest, :source_teacher_contract_sha256),
        "bridge source teacher contract",
    ) == payload.teacher.teacher_contract_sha256 ||
        error("bridge source contract differs from sealed teacher")
    Int(payload.model.input_dim) == OFFICIAL_ELM_INPUT_DIM ||
        error("sealed bundle model input_dim differs")
    payload.outcome.gate_passed === true ||
        error("sealed official ELM held-out gate failed")
    payload.outcome.paper_scale === require_production ||
        error("sealed bundle scale mode differs from requested mode")
    return true
end

"""
Open final-v2 compact shards against an independently source-bound sealed ELM.

No method exists for `PaperDigitalTwin.FrozenTwin`,
`FrozenOfficialELMTwin`, `VerifiedOfficialELMTwin`, or the superseded release
execution wrappers.
"""
function open_sealed_stream_dataset(
    path::AbstractString,
    bundle::Sealed.SealedOfficialELMRelease,
    source_teacher_manifest::AbstractString,
    source_teacher_shard_directory::AbstractString;
    minimum_spike_auroc::Real=0.985,
    verify_shard_hashes::Bool=true,
    require_promotion_eligible::Bool=true,
    require_production::Bool=require_promotion_eligible,
    scratch_root=nothing,
)
    require_production == require_promotion_eligible ||
        error("bridge and sealed bundle scale modes must agree")
    frozen = bundle.frozen
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
    _verify_bundle_and_manifest!(
        bundle,
        parsed.manifest,
        source_teacher_manifest,
        source_teacher_shard_directory;
        require_production,
        scratch_root,
    )
    _assert_input_contract(parsed.manifest)
    time_contract =
        _assert_time_contract(parsed.manifest, parsed.time_steps)
    _assert_primary_claim(parsed.manifest)
    provenance = merge(
        parsed.provenance,
        (;
            sealed_execution_type=SEALED_EXECUTION_TYPE,
            sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
            official_elm_input_semantics=Signed.INPUT_SEMANTICS,
            source_teacher_contract_sha256=
                bundle.attestation.payload.teacher.
                    teacher_contract_sha256,
            time_contract,
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

function _validate_logit_target(shard, time_steps, samples)
    logit = _required(shard, :target_spike_logit)
    size(logit) == (time_steps, samples) ||
        error("sealed cached spike-logit shape differs")
    all(isfinite, logit) ||
        error("sealed cached spike logits are non-finite")
    return logit
end

function sealed_stream_materialize_window(
    dataset::StreamDataset,
    global_indices,
    first_time::Integer,
    window::Integer,
)
    dataset.input_dim == OFFICIAL_ELM_INPUT_DIM ||
        error("dataset is not a sealed 1278-input stream")
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
    for original_batch_position in order
        global_index = samples[original_batch_position]
        shard_index = shard_indices[original_batch_position]
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
        Signed.fill_official_raw_window!(
            @view(raw_input[:, :, original_batch_position]),
            current_shard,
            local_index,
            first,
            last,
        )
        BaseStream._fill_target_window!(
            @view(target[:, :, original_batch_position]),
            @view(observed[:, :, original_batch_position]),
            dataset,
            current_shard,
            local_index,
            first,
            last,
        )
        target_spike_logit[:, original_batch_position] .= @view(
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

function _bit_exact(name, live, cached)
    size(live) == size(cached) ||
        throw(DimensionMismatch("$name live/cache shapes differ"))
    delta = isempty(live) ? 0.0 :
        maximum(abs.(Float64.(live) .- Float64.(cached)))
    delta == 0.0 ||
        error("$name cache differs from live sealed ELM; max_delta=$delta")
    return delta
end

"""
Re-run every cached primary trajectory through the exact sealed frozen ELM.

The sparse detailed Hay voltage/Ca targets are explicitly excluded: they are
auxiliary semantic-state supervision, not outputs of the digital twin.
"""
function verify_primary_cache_against_live_sealed_elm!(
    dataset::StreamDataset,
    bundle::Sealed.SealedOfficialELMRelease;
    time_chunk::Integer=256,
)
    dataset.input_dim == OFFICIAL_ELM_INPUT_DIM ||
        error("live-cache verification requires signed 1278 input")
    chunk = Int(time_chunk)
    chunk >= 1 ||
        throw(ArgumentError("cache verification chunk must be positive"))
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
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
                Signed.fill_official_raw_window!(
                    @view(raw[:, :, 1]),
                    shard,
                    local_index,
                    first_time,
                    last_time,
                )
                live = Twin.twin_forward(
                    bundle.frozen,
                    raw;
                    normalized=false,
                    initial_state=state,
                )
                state = live.final_state
                maxima[1] = max(
                    maxima[1],
                    _bit_exact(
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
                    _bit_exact(
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
                    _bit_exact(
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
                    _bit_exact(
                        "regional NMDA current",
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
        error("not every sealed release sample was cache-verified")
    time_points_verified ==
        dataset.total_samples * dataset.time_steps ||
        error("not every sealed release time point was cache-verified")
    all(iszero, maxima) ||
        error("sealed primary cache verification was not bit exact")
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
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
        sealed_execution_type=SEALED_EXECUTION_TYPE,
        official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
    )
end

end # module StreamingOfficialELMReleaseDatasetV2
