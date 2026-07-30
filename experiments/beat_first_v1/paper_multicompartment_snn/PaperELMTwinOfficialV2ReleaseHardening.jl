# Release-only hardening layered onto the immutable fixed-gate implementation.
# This file is included inside `PaperELMTwinOfficialV2Release`.

export preflight_verified_official_elm_release

const _FORBIDDEN_BASE_GATE_METADATA = (
    :fixed_gate,
    :release_gate,
    :release_thresholds,
    :threshold_override_allowed,
    :minimum_spike_auroc,
    :min_spike_auroc,
    :spike_auroc_min,
    :maximum_voltage_rmse_mv,
    :max_voltage_rmse_mv,
    :max_voltage_rmse,
    :voltage_rmse_max,
    :maximum_nmda_normalized_rmse,
    :max_nmda_normalized_rmse,
    :max_nmda_rmse,
    :nmda_rmse_max,
)

function _reject_base_gate_metadata(metadata)
    for name in _FORBIDDEN_BASE_GATE_METADATA
        hasproperty(metadata, name) && throw(ArgumentError(
            "release gate field `$name` cannot be supplied by base metadata",
        ))
    end
    return metadata
end

function _assert_canonical_final_contract(metadata)
    _reject_base_gate_metadata(metadata)
    _required_metadata(metadata, :identity_input_normalization) === true ||
        error("release candidate must use identity presynaptic input")
    Float32(_required_metadata(metadata, :soma_clip_mv)) ==
        Development.OFFICIAL_SOMA_CLIP_MV ||
        error("release soma clip differs from the canonical transform")
    Float32(_required_metadata(metadata, :soma_bias_mv)) ==
        Development.OFFICIAL_SOMA_BIAS_MV ||
        error("release soma bias differs from the canonical transform")
    Float32(_required_metadata(metadata, :soma_train_scale)) ==
        Development.OFFICIAL_SOMA_TRAIN_SCALE ||
        error("release soma scale differs from the canonical transform")
    return true
end

# Replace the base candidate constructors with stricter Final-contract
# versions. The fixed verification thresholds remain private constants in the
# base release implementation and cannot be supplied here.
function prepare_official_elm_release_candidate(
    frozen::Development.FrozenOfficialELMTwin,
)
    Development.assert_frozen_official_elm_unchanged(frozen)
    metadata = frozen.metadata
    _assert_canonical_final_contract(metadata)
    _required_metadata(metadata, :held_out_split) ==
        CANONICAL_HELD_OUT_SPLIT ||
        error("release candidate split is not held_out_test")
    Int(_required_metadata(metadata, :duration_ms)) ==
        CANONICAL_PAPER_DURATION_MS ||
        error("release candidate duration is not paper-scale 10,000 ms")
    Int(_required_metadata(metadata, :sample_dt_ms)) ==
        CANONICAL_PAPER_SAMPLE_DT_MS ||
        error("release candidate sample_dt_ms is not 1")
    _required_metadata(metadata, :paper_scale) === true ||
        error("release candidate is not paper_scale=true")
    manifest = _require_sha256(
        "official_teacher_manifest_sha256",
        _required_metadata(
            metadata,
            :official_teacher_manifest_sha256,
        ),
    )
    contract = _require_sha256(
        "teacher_contract_sha256",
        _required_metadata(metadata, :teacher_contract_sha256),
    )
    candidate_digest = Development._digest_hex(
        _candidate_payload(frozen, manifest, contract),
    )
    return OfficialELMReleaseCandidate(
        frozen,
        manifest,
        contract,
        candidate_digest,
    )
end

function assert_official_elm_release_candidate(
    candidate::OfficialELMReleaseCandidate,
)
    Development.assert_frozen_official_elm_unchanged(candidate.frozen)
    metadata = candidate.frozen.metadata
    _assert_canonical_final_contract(metadata)
    _required_metadata(metadata, :held_out_split) ==
        CANONICAL_HELD_OUT_SPLIT ||
        error("release split changed")
    Int(_required_metadata(metadata, :duration_ms)) ==
        CANONICAL_PAPER_DURATION_MS ||
        error("release duration changed")
    Int(_required_metadata(metadata, :sample_dt_ms)) ==
        CANONICAL_PAPER_SAMPLE_DT_MS ||
        error("release sample interval changed")
    _required_metadata(metadata, :paper_scale) === true ||
        error("release paper_scale changed")
    _require_sha256(
        "teacher manifest",
        _required_metadata(
            metadata,
            :official_teacher_manifest_sha256,
        ),
    ) == candidate.teacher_manifest_sha256 ||
        error("release teacher manifest changed")
    _require_sha256(
        "teacher contract",
        _required_metadata(metadata, :teacher_contract_sha256),
    ) == candidate.teacher_contract_sha256 ||
        error("release teacher contract changed")
    Development._digest_hex(_candidate_payload(
        candidate.frozen,
        candidate.teacher_manifest_sha256,
        candidate.teacher_contract_sha256,
    )) == candidate.candidate_sha256 ||
        error("release candidate digest changed")
    return true
end

preflight_verified_official_elm_release(
    verified::VerifiedOfficialELMRelease,
) = assert_verified_official_elm_release(verified)

