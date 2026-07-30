module StreamingOfficialELMReleaseDatasetFinalV2

# Canonical consumer entry point: independently replays every persisted primary
# output and binds it to the writer's post-write measurement.

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :StreamingOfficialELMReleaseDatasetV4)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "StreamingOfficialELMReleaseDatasetV4.jl",
        ),
    )
end

const Stream =
    getfield(_PARENT, :StreamingOfficialELMReleaseDatasetV4)
const Sealed = Stream.Sealed

export PRIMARY_REPLAY_SCHEMA,
    open_live_verified_sealed_stream_dataset,
    verify_recorded_primary_measurement!

const PRIMARY_REPLAY_SCHEMA =
    "hd_swsnn.distillation.primary_cache_live_replay.final.v2"

function _zero_deltas(object)
    names = (
        :soma_voltage,
        :spike_probability,
        :spike_logit,
        :regional_nmda_current,
    )
    values = ntuple(
        index -> Float64(Stream._required(object, names[index])),
        length(names),
    )
    all(iszero, values) ||
        error("recorded primary replay has a nonzero delta")
    return values
end

function verify_recorded_primary_measurement!(dataset, live_report)
    claim = Stream._required(
        dataset.manifest,
        :primary_cache_live_equality,
    )
    String(Stream._required(claim, :measurement)) ==
        "postwrite_live_replay" ||
        error("primary cache was not measured by post-write replay")
    String(Stream._required(claim, :measurement_schema)) ==
        PRIMARY_REPLAY_SCHEMA ||
        error("primary replay measurement schema differs")
    recorded = Stream._required(claim, :measurement_report)
    recorded_sha256 = lowercase(String(
        Stream._required(claim, :measurement_sha256),
    ))
    Sealed.canonical_sha256(recorded) == recorded_sha256 ||
        error("recorded primary replay measurement digest differs")
    String(Stream._required(
        recorded,
        :execution_type,
    )) == Stream.SEALED_EXECUTION_TYPE ||
        error("recorded replay used another execution type")
    String(Stream._required(
        recorded,
        :sealed_release_schema,
    )) == Sealed.SEALED_RELEASE_SCHEMA ||
        error("recorded replay used another sealed schema")
    String(Stream._required(
        recorded,
        :sealed_attestation_sha256,
    )) == dataset.provenance.sealed_attestation_sha256 ||
        error("recorded replay belongs to another sealed-v2 ELM")
    String(Stream._required(
        recorded,
        :source_manifest_sha256,
    )) == dataset.provenance.source_manifest_sha256 ||
        error("recorded replay belongs to another source manifest")
    String(Stream._required(
        recorded,
        :source_teacher_contract_sha256,
    )) == dataset.provenance.source_teacher_contract_sha256 ||
        error("recorded replay belongs to another teacher contract")
    result = Stream._required(recorded, :result)
    Stream._required(result, :all_samples) === true ||
        error("recorded replay omitted cached samples")
    Stream._required(result, :bit_exact) === true ||
        error("recorded replay did not establish bit equality")
    Int(Stream._required(result, :samples_verified)) ==
        live_report.samples_verified ==
        dataset.total_samples ||
        error("recorded/live replay sample counts differ")
    Int(Stream._required(result, :time_points_verified)) ==
        live_report.time_points_verified ==
        dataset.total_samples * dataset.time_steps ||
        error("recorded/live replay time-point counts differ")
    recorded_delta =
        _zero_deltas(Stream._required(result, :max_absolute_delta))
    live_delta = (
        Float64(live_report.soma_voltage_max_delta),
        Float64(live_report.spike_probability_max_delta),
        Float64(live_report.spike_logit_max_delta),
        Float64(live_report.nmda_max_delta),
    )
    all(iszero, live_delta) ||
        error("independent live replay was not bit exact")
    recorded_delta == live_delta ||
        error("recorded/live replay deltas differ")
    return (;
        verified=true,
        measurement_sha256=recorded_sha256,
        measurement_schema=PRIMARY_REPLAY_SCHEMA,
        samples_verified=live_report.samples_verified,
        time_points_verified=live_report.time_points_verified,
        bit_exact=true,
    )
end

function open_live_verified_sealed_stream_dataset(
    path::AbstractString,
    bundle::Sealed.SealedOfficialELMRelease,
    source_teacher_manifest::AbstractString,
    source_teacher_shard_directory::AbstractString;
    time_chunk::Integer=256,
    minimum_spike_auroc::Real=0.985,
    verify_shard_hashes::Bool=true,
    require_promotion_eligible::Bool=true,
    require_production::Bool=require_promotion_eligible,
    scratch_root=nothing,
)
    dataset = Stream.open_sealed_stream_dataset(
        path,
        bundle,
        source_teacher_manifest,
        source_teacher_shard_directory;
        minimum_spike_auroc,
        verify_shard_hashes,
        require_promotion_eligible,
        require_production,
        scratch_root,
    )
    live_report =
        Stream.verify_primary_cache_against_live_sealed_elm!(
            dataset,
            bundle;
            time_chunk,
        )
    measurement =
        verify_recorded_primary_measurement!(dataset, live_report)
    return (; dataset, live_replay=live_report, measurement)
end

end # module StreamingOfficialELMReleaseDatasetFinalV2
