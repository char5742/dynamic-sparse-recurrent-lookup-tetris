# Load this file into `HDSWSNNTwinPropProduction`.
#
# It supersedes the first draft's impossible self-referential raw-file hash
# check. Raw twin bytes are bound by the separate distilled/release manifest;
# the twin itself carries a non-self-referential attestation over its logical
# artifact, official source, held-out metrics, and gate thresholds.

function _twin_attestation_payload_final(frozen, teacher)
    metadata = frozen.metadata
    return (;
        schema=PRODUCTION_ATTESTATION_SCHEMA,
        model_family=MODEL_FAMILY,
        official_teacher_manifest_sha256=
            teacher.manifest_sha256,
        teacher_contract_sha256=
            teacher.teacher_contract_sha256,
        mechanism_library_sha256=
            teacher.mechanism_library_sha256,
        mechanism_sources_sha256=
            teacher.mechanism_sources_sha256,
        morphology_sha256=teacher.morphology_sha256,
        modeldb_tree_sha256=teacher.modeldb_tree_sha256,
        generator_source_sha256=
            teacher.generator_source_sha256,
        twin_parameter_sha256=frozen.parameter_sha256,
        twin_artifact_sha256=frozen.artifact_sha256,
        twin_config=frozen.model.config,
        held_out_metrics=_held_out_metrics(frozen),
        gate=_required(metadata, :gate),
        loss_targets=_required(metadata, :loss_targets),
    )
end

production_twin_attestation_sha256(frozen, teacher) =
    _serialized_sha256(
        _twin_attestation_payload_final(frozen, teacher),
    )

function _validate_normalizer!(normalizer, config)
    length(normalizer.input_mean) == config.input_dim ||
        error("twin input normalizer dimension differs")
    length(normalizer.input_scale) == config.input_dim ||
        error("twin input scale dimension differs")
    length(normalizer.nmda_mean) == config.nmda_regions ||
        error("twin NMDA normalizer dimension differs")
    length(normalizer.nmda_scale) == config.nmda_regions ||
        error("twin NMDA scale dimension differs")
    all(isfinite, normalizer.input_mean) ||
        error("twin input mean is non-finite")
    all(value -> isfinite(value) && value > 0,
        normalizer.input_scale) ||
        error("twin input scale is non-finite/nonpositive")
    isfinite(normalizer.voltage_mean) ||
        error("twin voltage mean is non-finite")
    isfinite(normalizer.voltage_scale) &&
        normalizer.voltage_scale > 0 ||
        error("twin voltage scale is non-finite/nonpositive")
    all(isfinite, normalizer.nmda_mean) ||
        error("twin NMDA mean is non-finite")
    all(value -> isfinite(value) && value > 0,
        normalizer.nmda_scale) ||
        error("twin NMDA scale is non-finite/nonpositive")
    return nothing
end

function _metric_threshold(
    gate,
    metrics,
    metric::Symbol,
    threshold::Symbol,
    comparison,
)
    measured = Float64(_required(metrics, metric))
    limit = Float64(_required(gate, threshold))
    isfinite(measured) && isfinite(limit) ||
        error("non-finite official held-out gate value: $metric")
    comparison(measured, limit) || error(
        "official held-out gate failed: $metric=$measured, " *
        "$threshold=$limit",
    )
    return measured
end

function _validate_twin!(frozen, teacher, twin_file_sha256)
    integrity = Twin.assert_frozen_unchanged(frozen)
    config = frozen.model.config
    config.model_name == MODEL_FAMILY ||
        error("frozen twin model family differs")
    config.segments == OFFICIAL_HAY_SEGMENTS &&
        config.segments == teacher.total_segments ||
        error("twin is not the official 642-segment teacher")
    config.receptors == 3 ||
        error("production twin must have three receptor families")
    config.input_planes == 2 ||
        error("production twin must have two input planes")
    config.input_dim == 6config.segments ||
        error("production twin input dimension must be 6*segments")
    config.nmda_regions == 4 ||
        error("production twin must predict four NMDA regions")
    config.memory_units == 1_000 ||
        error("production twin must contain 1,000 memory units")
    config.core_dim == 128 ||
        error("production twin core_dim must be 128")
    config.bank_seed == UInt64(0x5457494e50524f50) ||
        error("production twin random bank seed differs")
    config.dt_ms == 1.0f0 ||
        error("production twin output step must be 1 ms")
    config.tau_min_ms == 0.1f0 ||
        error("production twin tau_min must be 0.1 ms")
    config.tau_max_ms == 300.0f0 ||
        error("production twin tau_max must be 300 ms")
    _validate_normalizer!(frozen.normalizer, config)

    metadata = frozen.metadata
    _value(metadata, :frozen, false) === true ||
        error("digital twin is not marked frozen")
    _required_string(metadata, :cell_mechanism_sha256) ==
        teacher.mechanism_library_sha256 ||
        error("twin mechanism hash differs from official teacher")
    _required_string(metadata, :morphology_sha256) ==
        teacher.morphology_sha256 ||
        error("twin morphology hash differs from official teacher")
    _required_string(
        metadata,
        :official_teacher_manifest_sha256,
    ) == teacher.manifest_sha256 ||
        error("twin is not bound to this official manifest")

    targets = Set(lowercase.(String.(
        collect(_required(metadata, :loss_targets)),
    )))
    targets == Set((
        "soma_voltage",
        "soma_spike",
        "region_nmda_current",
    )) || error(
        "digital twin must be trained jointly on voltage/spike/NMDA",
    )
    metrics = _held_out_metrics(frozen)
    gate = _required(metadata, :gate)
    _value(gate, :passed, false) === true ||
        error("frozen twin held-out gate did not pass")
    _value(gate, :recomputed_from_official_held_out, false) === true ||
        error("twin gate was not recomputed from official held-out data")
    _required_string(
        gate,
        :official_teacher_manifest_sha256,
    ) == teacher.manifest_sha256 ||
        error("twin held-out gate used another teacher manifest")
    _metric_threshold(
        gate,
        metrics,
        :spike_auroc,
        :minimum_spike_auroc,
        >=,
    ) >= MINIMUM_TWIN_SPIKE_AUROC ||
        error("twin spike AUROC gate is weaker than production")
    _metric_threshold(
        gate,
        metrics,
        :voltage_rmse,
        :maximum_voltage_rmse_mv,
        <=,
    )
    _metric_threshold(
        gate,
        metrics,
        :voltage_correlation,
        :minimum_voltage_correlation,
        >=,
    )
    _metric_threshold(
        gate,
        metrics,
        :nmda_normalized_mse,
        :maximum_nmda_normalized_mse,
        <=,
    )
    _metric_threshold(
        gate,
        metrics,
        :nmda_correlation,
        :minimum_nmda_correlation,
        >=,
    )

    attestation =
        production_twin_attestation_sha256(frozen, teacher)
    _required_string(
        metadata,
        :production_manifest_sha256,
    ) == attestation ||
        error("frozen twin production-attestation hash mismatch")
    integrity.parameter_sha256 == frozen.parameter_sha256 ||
        error("frozen twin parameter integrity differs")
    integrity.artifact_sha256 == frozen.artifact_sha256 ||
        error("frozen twin artifact integrity differs")
    # `twin_file_sha256` is deliberately not required inside `frozen`.
    # It is recorded by the separately hashed distilled/release artifact.
    occursin(r"^[0-9a-f]{64}$", twin_file_sha256) ||
        error("raw twin file digest is malformed")
    return attestation
end

