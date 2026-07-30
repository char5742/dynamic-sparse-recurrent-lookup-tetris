module DistillationDatasetBridgeOfficial1278

using JSON3

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(
    _PARENT_MODULE,
    :DistillationDatasetBridgeReleaseCanonical,
)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_release_canonical.jl",
        ),
    )
end
if !isdefined(_PARENT_MODULE, :PaperELMTwinOfficialV2Final)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "PaperELMTwinOfficialV2Final.jl"),
    )
end

using ..DistillationDatasetBridgeReleaseCanonical
using ..PaperDigitalTwin
using ..PaperELMTwinOfficialV2Final

const ReleaseBridge =
    DistillationDatasetBridgeReleaseCanonical
const Production = ReleaseBridge.Production
const OrderedBridge = Production.OrderedBridge
const FinalBridge = OrderedBridge.FinalBridge
const V6 = FinalBridge.V6
const Legacy = FinalBridge.Legacy
const BaseBridge = FinalBridge.BaseBridge
const LegacyTwinRuntime = PaperDigitalTwin
const OfficialTwin = PaperELMTwinOfficialV2Final
const ReleaseStreamingPrepareConfig =
    ReleaseBridge.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    OFFICIAL_INPUT_DIM,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = ReleaseBridge.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = ReleaseBridge.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = ReleaseBridge.RELEASE_SHARD_SCHEMA
const OFFICIAL_INPUT_DIM = OfficialTwin.OFFICIAL_ELM_INPUT_DIM

struct OfficialBridgeConfig
    input_dim::Int
    segments::Int
    nmda_regions::Int
    dt_ms::Float32
end

struct OfficialBridgeModel
    config::OfficialBridgeConfig
end

struct OfficialBridgeTwin{V,M,A}
    verified::V
    model::OfficialBridgeModel
    metadata::M
    attestation::A
    parameter_sha256::String
    artifact_sha256::String
    attestation_sha256::String
end

function _bridge_twin(verified::OfficialTwin.VerifiedOfficialELMTwin)
    config = verified.model.config
    config.num_input == OFFICIAL_INPUT_DIM ||
        error("verified official ELM does not have 1278 inputs")
    config.nmda_regions == 4 ||
        error("verified official ELM does not have four NMDA outputs")
    config.num_branch == OfficialTwin.OFFICIAL_ELM_BRANCHES ||
        error("verified official ELM branch count differs")
    config.num_synapse_per_branch ==
        OfficialTwin.OFFICIAL_ELM_SYNAPSES_PER_BRANCH ||
        error("verified official ELM synapses/branch differs")
    config.input_to_synapse_routing === :neuronio_routing ||
        error("verified official ELM routing is not neuronio_routing")
    config.num_memory == 1_000 ||
        error("verified TwinProp ELM must have 1,000 memory units")
    config.memory_tau_min_ms == 0.1f0 ||
        error("verified TwinProp ELM minimum tau differs")
    config.memory_tau_max_ms == 300.0f0 ||
        error("verified TwinProp ELM maximum tau differs")
    config.delta_t_ms == 1.0f0 ||
        error("verified TwinProp ELM must run at 1 ms")
    return OfficialBridgeTwin(
        verified,
        OfficialBridgeModel(OfficialBridgeConfig(
            OFFICIAL_INPUT_DIM,
            OfficialTwin.Core.HAY_TOTAL_SEGMENTS,
            4,
            config.delta_t_ms,
        )),
        verified.metadata,
        verified.attestation,
        verified.parameter_sha256,
        verified.artifact_sha256,
        verified.attestation_sha256,
    )
end

function BaseBridge._verify_twin(
    config::BaseBridge.PrepareDistillationConfig,
    source::BaseBridge._Source,
)
    path = abspath(config.frozen_twin_path)
    isfile(path) ||
        error("verified official ELM artifact is absent: $path")
    verified = OfficialTwin.load_verified_official_elm(path)
    OfficialTwin.assert_verified_official_elm(verified)
    attestation = verified.attestation
    attestation.teacher_manifest_sha256 ==
        source.source_manifest_hash ||
        error("official ELM attestation/teacher manifest mismatch")
    attestation.teacher_contract_sha256 ==
        source.detailed_teacher_hash ||
        error("official ELM attestation/teacher contract mismatch")
    attestation.metrics.spike_auroc >=
        config.minimum_twin_spike_auroc || error(
        "attested official ELM spike AUROC is below the bridge gate",
    )
    BaseBridge._expected_hash(
        "twin parameter",
        verified.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    BaseBridge._expected_hash(
        "twin artifact",
        verified.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )
    bridge = _bridge_twin(verified)
    integrity = (;
        frozen=true,
        verified=true,
        max_delta=0.0f0,
        parameter_sha256=verified.parameter_sha256,
        artifact_sha256=verified.artifact_sha256,
        attestation_sha256=verified.attestation_sha256,
    )
    return (
        bridge,
        integrity,
        attestation.metrics,
        BaseBridge._sha256_file(path),
    )
end

function LegacyTwinRuntime.assert_frozen_unchanged(
    frozen::OfficialBridgeTwin;
    expected_artifact_sha256::AbstractString="",
)
    OfficialTwin.assert_verified_official_elm(frozen.verified)
    isempty(expected_artifact_sha256) ||
        frozen.artifact_sha256 ==
            lowercase(String(expected_artifact_sha256)) ||
        error("verified official ELM artifact changed")
    return (;
        frozen=true,
        verified=true,
        max_delta=0.0f0,
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
        attestation_sha256=frozen.attestation_sha256,
    )
end

LegacyTwinRuntime.twin_input_layout(::OfficialBridgeTwin) =
    OfficialTwin.official_elm_input_layout()

function BaseBridge._input_anatomy(::OfficialBridgeConfig)
    locations = collect(
        OfficialTwin.Core.HAY_FIRST_DENDRITIC_SEGMENT:
        OfficialTwin.Core.HAY_LAST_DENDRITIC_SEGMENT,
    )
    return (
        vcat(locations, locations),
        vcat(
            fill(1, length(locations)),
            fill(2, length(locations)),
        ),
        ones(Int, OFFICIAL_INPUT_DIM),
    )
end

struct OfficialSparseSample
    contact_axon::Vector{Int32}
    contact_segment::Vector{Int32}
    contact_kind::Vector{UInt8}
    contact_strength::Vector{Float32}
    event_axon::Vector{Int32}
    event_time_bin::Vector{Int32}
    event_count::Vector{UInt8}
    contacts_by_axon::Dict{Int,Vector{Int}}
    events_by_time::Vector{Vector{Tuple{Int,UInt8}}}
end

function V6._sparse_sample(
    shard,
    trial::Int,
    ::OfficialBridgeConfig,
)
    layout = V6._validate_final_shard(shard)
    contact_range =
        V6._ragged_range(layout.contact_offsets, trial)
    event_range = V6._ragged_range(layout.event_offsets, trial)
    contact_axon = Int32.(
        vec(V6._required(shard, :contact_axon))[contact_range],
    )
    contact_segment = Int32.(
        vec(V6._required(shard, :contact_segment))[contact_range],
    )
    contact_kind = UInt8.(
        vec(V6._required(shard, :contact_kind))[contact_range],
    )
    contact_strength = Float32.(
        vec(V6._required(shard, :contact_strength))[contact_range],
    )
    event_axon = Int32.(
        vec(V6._required(shard, :event_axon))[event_range],
    )
    event_time = Int32.(
        vec(V6._required(shard, :event_time_bin))[event_range],
    )
    event_count = UInt8.(
        vec(V6._required(shard, :event_count))[event_range],
    )
    contacts_by_axon = Dict{Int,Vector{Int}}()
    for contact in eachindex(contact_axon)
        OfficialTwin.official_contact_channel(
            contact_segment[contact],
            contact_kind[contact],
        )
        0.0f0 <= contact_strength[contact] <= 1.0f0 ||
            error("official contact strength lies outside [0,1]")
        push!(
            get!(contacts_by_axon, Int(contact_axon[contact]), Int[]),
            contact,
        )
    end
    events_by_time =
        [Tuple{Int,UInt8}[] for _ in 1:layout.time_steps]
    for event in eachindex(event_axon)
        time = Int(event_time[event]) + 1
        1 <= time <= layout.time_steps ||
            error("official event lies outside trajectory")
        push!(
            events_by_time[time],
            (Int(event_axon[event]), event_count[event]),
        )
    end
    return OfficialSparseSample(
        contact_axon,
        contact_segment,
        contact_kind,
        contact_strength,
        event_axon,
        event_time,
        event_count,
        contacts_by_axon,
        events_by_time,
    )
end

function V6._dense_time_chunk(
    sample::OfficialSparseSample,
    config::OfficialBridgeConfig,
    first_time::Int,
    last_time::Int,
)
    input = zeros(
        Float32,
        OFFICIAL_INPUT_DIM,
        last_time - first_time + 1,
        1,
    )
    for output_time in axes(input, 2)
        source_time = first_time + output_time - 1
        for (axon, multiplicity) in sample.events_by_time[source_time]
            for contact in get(
                sample.contacts_by_axon,
                axon,
                Int[],
            )
                channel = OfficialTwin.official_contact_channel(
                    sample.contact_segment[contact],
                    sample.contact_kind[contact],
                )
                sign = sample.contact_kind[contact] == UInt8(1) ?
                    1.0f0 : -1.0f0
                input[channel, output_time, 1] +=
                    sign *
                    sample.contact_strength[contact] *
                    Float32(multiplicity)
            end
        end
    end
    all(isfinite, input) ||
        error("official 1278 input expansion is non-finite")
    return input
end

function V6._infer_sample(
    frozen::OfficialBridgeTwin,
    sample::OfficialSparseSample,
    time_steps,
    chunk,
)
    voltage = Vector{Float32}(undef, time_steps)
    spike = Vector{Float32}(undef, time_steps)
    spike_logit = Vector{Float32}(undef, time_steps)
    nmda = Matrix{Float32}(undef, 4, time_steps)
    state = nothing
    for first_time in 1:chunk:time_steps
        last_time = min(first_time + chunk - 1, time_steps)
        input = V6._dense_time_chunk(
            sample,
            frozen.model.config,
            first_time,
            last_time,
        )
        prediction = OfficialTwin.twin_forward(
            frozen.verified,
            input;
            initial_state=state,
        )
        for (label, values) in (
            ("voltage", prediction.voltage),
            ("spike probability", prediction.spike_probability),
            ("NMDA", prediction.nmda),
        )
            all(isfinite, values) ||
                error("verified official ELM produced non-finite $label")
        end
        voltage[first_time:last_time] .= vec(prediction.voltage)
        spike[first_time:last_time] .=
            vec(prediction.spike_probability)
        spike_logit[first_time:last_time] .=
            vec(prediction.spike_logit)
        nmda[:, first_time:last_time] .=
            reshape(prediction.nmda, 4, :)
        state = prediction.final_state
    end
    return (; voltage, spike, spike_logit, nmda)
end

function FinalBridge._release_manifest(
    config::ReleaseStreamingPrepareConfig,
    source::BaseBridge._Source,
    source_counts,
    plan_data,
    frozen::OfficialBridgeTwin,
    integrity_before,
    integrity_after,
    reported_metrics,
    recomputed_gate,
    twin_file_hash,
    config_sha256,
    segment_catalog_sha256,
    selected_segments,
    diagnostic_segments,
    diagnostic_times,
    train_indices,
    validation_indices,
    test_indices,
    shard_records,
    promotion_eligible,
)
    arguments = (
        config,
        source,
        source_counts,
        plan_data,
        frozen,
        integrity_before,
        integrity_after,
        reported_metrics,
        recomputed_gate,
        twin_file_hash,
        config_sha256,
        segment_catalog_sha256,
        selected_segments,
        diagnostic_segments,
        diagnostic_times,
        train_indices,
        validation_indices,
        test_indices,
        shard_records,
        promotion_eligible,
    )
    base = invoke(
        FinalBridge._release_manifest,
        Tuple{
            ReleaseStreamingPrepareConfig,
            BaseBridge._Source,
            Vararg{Any,18},
        },
        arguments...,
    )
    official_lineage = (;
        digital_twin_type="VerifiedOfficialELMTwin",
        official_elm_input_dim=OFFICIAL_INPUT_DIM,
        official_elm_dendritic_locations=
            OfficialTwin.OFFICIAL_DENDRITIC_LOCATIONS,
        official_elm_branches=
            OfficialTwin.OFFICIAL_ELM_BRANCHES,
        official_elm_synapses_per_branch=
            OfficialTwin.OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
        official_elm_memory_units=
            frozen.verified.model.config.num_memory,
        official_elm_input_semantics=
            "E:+strength*event_count, I:-strength*event_count",
        verified_attestation_sha256=
            frozen.attestation_sha256,
        attested_teacher_manifest_sha256=
            frozen.attestation.teacher_manifest_sha256,
        attested_teacher_contract_sha256=
            frozen.attestation.teacher_contract_sha256,
        attested_metrics=frozen.attestation.metrics,
        attested_thresholds=frozen.attestation.thresholds,
        attestation_evaluator_id=
            frozen.attestation.evaluator_id,
    )
    return merge(
        base,
        official_lineage,
        (;
            hashes=merge(
                base.hashes,
                (;
                    verified_attestation_sha256=
                        frozen.attestation_sha256,
                ),
            ),
        ),
    )
end

prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
) = ReleaseBridge.prepare_distillation_dataset_release(config)

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeOfficial1278

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeOfficial1278.main()
end
