module HDSWSNNTwinPropProduction

using JLD2
using LinearAlgebra
using Serialization
using SHA

const _ROOT = Main
for (name, file) in (
    (
        :OfficialNeuronTeacherMetadataProduction,
        "OfficialNeuronTeacherMetadataProduction.jl",
    ),
    (:PaperDigitalTwin, "PaperDigitalTwin.jl"),
    (
        :DistilledElevenStateCellFinal,
        "DistilledElevenStateCellFinal.jl",
    ),
    (
        :PaperArenaTrainingFinalProduction,
        "PaperArenaTrainingFinalProduction.jl",
    ),
)
    if !isdefined(_ROOT, name)
        Base.include(_ROOT, joinpath(@__DIR__, file))
    end
end

const Teacher = Main.OfficialNeuronTeacherMetadataProduction
const Twin = Main.PaperDigitalTwin
const Cell = Main.DistilledElevenStateCellFinal
const Training = Main.PaperArenaTrainingFinalProduction

export MODEL_FAMILY,
    MINIMUM_TWIN_SPIKE_AUROC,
    STATE_SEMANTICS,
    ProductionBundle,
    assert_production_bundle_unchanged!,
    build_production_trainer,
    load_production_bundle,
    production_twin_attestation_sha256

const MODEL_FAMILY = "HD-SWSNN-TwinProp"
const MINIMUM_TWIN_SPIKE_AUROC = 0.985
const OFFICIAL_HAY_SEGMENTS = 642
const PRODUCTION_ATTESTATION_SCHEMA =
    "hd-swsnn-twinprop-twin-attestation-v1"
const STATE_SEMANTICS = (
    "dendritic_voltage_latent_1",
    "dendritic_voltage_latent_2",
    "dendritic_voltage_latent_3",
    "dendritic_voltage_latent_4",
    "nmda_current_latent_1",
    "nmda_current_latent_2",
    "nmda_current_latent_3",
    "nmda_current_latent_4",
    "apical_context_latent",
    "soma_voltage_latent",
    "adaptation_calcium_summary",
)

struct ProductionBundle{T,F,P}
    teacher::T
    frozen_twin::F
    distilled_parameters::P
    twin_path::String
    distilled_path::String
    twin_file_sha256::String
    distilled_file_sha256::String
    twin_parameter_sha256::String
    twin_artifact_sha256::String
    distilled_parameter_sha256::String
    twin_attestation_sha256::String
end

@inline function _value(object, name::Symbol, default=nothing)
    if object isa AbstractDict
        return get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        return getproperty(object, name)
    end
    return default
end

function _required(object, name::Symbol)
    value = _value(object, name, nothing)
    value === nothing && error("production lineage lacks `$name`")
    return value
end

function _required_string(object, name::Symbol)
    value = String(_required(object, name))
    isempty(value) && error("production lineage `$name` is empty")
    return value
end

function _sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("production artifact is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _held_out_metrics(frozen)
    metadata = frozen.metadata
    for name in (
        :held_out_test,
        :held_out_metrics,
        :test_metrics,
    )
        metrics = _value(metadata, name, nothing)
        metrics === nothing || return metrics
    end
    error("frozen twin has no held-out metrics")
end

function _twin_attestation_payload(frozen, teacher)
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
        loss_targets=_required(metadata, :loss_targets),
    )
end

"""
Digest that binds the immutable twin bank/normalizer/parameters to the
official teacher source and held-out metrics.  Production refuses a plain
metadata AUROC claim unless this digest is stored in the frozen artifact.
"""
production_twin_attestation_sha256(frozen, teacher) =
    _serialized_sha256(_twin_attestation_payload(frozen, teacher))

function _validate_twin!(frozen, teacher, twin_file_sha256)
    integrity = Twin.assert_frozen_unchanged(frozen)
    config = frozen.model.config
    config.model_name == MODEL_FAMILY ||
        error("frozen twin model family differs")
    config.memory_units == 1_000 ||
        error("production twin must contain 1,000 memory units")
    config.nmda_regions == 4 ||
        error("production twin must predict four NMDA regions")
    config.segments == teacher.total_segments ||
        error("twin segment axis differs from official teacher")
    config.segments == OFFICIAL_HAY_SEGMENTS ||
        error("production twin is not the official 642-segment Hay cell")
    config.dt_ms == 1.0f0 ||
        error("production twin output step must be 1 ms")
    config.tau_min_ms == 0.1f0 ||
        error("production twin tau_min must be 0.1 ms")
    config.tau_max_ms == 300.0f0 ||
        error("production twin tau_max must be 300 ms")

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
        error("twin is not bound to this official teacher manifest")
    _required_string(metadata, :raw_twin_file_sha256) ==
        twin_file_sha256 ||
        error("twin raw-file SHA-256 attestation differs")

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
    spike_auroc = Float64(_value(metrics, :spike_auroc, NaN))
    isfinite(spike_auroc) &&
        spike_auroc >= MINIMUM_TWIN_SPIKE_AUROC ||
        error(
            "frozen twin held-out spike AUROC $spike_auroc is below " *
            "$MINIMUM_TWIN_SPIKE_AUROC",
        )
    gate = _value(metadata, :gate, nothing)
    gate === nothing ||
        _value(gate, :passed, false) === true ||
        error("frozen twin metadata records a failed gate")

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
    return attestation
end

function _assert_finite_parameters(parameters)
    for name in (
        :transition_decay,
        :recurrent_weight,
        :input_weight,
        :transition_bias,
        :readout_weight,
        :readout_bias,
        :target_mean,
        :target_scale,
        :initial_state,
        :compartment_projection,
        :region_projection,
    )
        all(isfinite, getfield(parameters, name)) ||
            error("distilled parameter `$name` contains non-finite values")
    end
    return nothing
end

function _readout_off_coordinate_fraction(parameters)
    allowed = falses(11, 11)
    allowed[1, 10] = true             # soma voltage <- soma latent
    allowed[2, 10] = true             # soma spike <- soma latent
    for region in 1:4
        allowed[2 + region, 4 + region] = true
        allowed[7 + region, region] = true
    end
    allowed[7, 11] = true             # Ca/adaptation
    total = sum(abs2, parameters.readout_weight)
    total > 0 || error("distilled readout is identically zero")
    off = sum(
        abs2(parameters.readout_weight[index])
        for index in eachindex(allowed)
        if !allowed[index]
    )
    return sqrt(off / total)
end

function _validate_distilled!(
    path,
    parameters,
    frozen,
    teacher,
    twin_file_sha256,
)
    Cell.DISTILLED_STATE_DIM == 11 ||
        error("Final cell does not expose exactly 11 states")
    _assert_finite_parameters(parameters)
    size(parameters.recurrent_weight) == (11, 11) ||
        error("Final recurrent transition is not 11 x 11")
    size(parameters.input_weight) == (11, 16) ||
        error("Final input transition is not 11 x 16")
    size(parameters.readout_weight) == (11, 11) ||
        error("Final readout is not 11 x 11")
    size(parameters.compartment_projection) ==
        (4, teacher.total_segments) ||
        error("Final cell does not retain the official segment axis")
    parameters.detailed_kernel_hash ==
        teacher.mechanism_library_sha256 ||
        error("distilled mechanism lineage differs")
    parameters.morphology_hash == teacher.morphology_sha256 ||
        error("distilled morphology lineage differs")
    parameters.frozen_twin_parameter_hash ==
        frozen.parameter_sha256 ||
        error("distilled twin-parameter lineage differs")
    parameters.frozen_twin_artifact_hash ==
        frozen.artifact_sha256 ||
        error("distilled twin-artifact lineage differs")
    occursin("official-neuron", lowercase(parameters.teacher_schema)) ||
        error("distilled teacher schema is not official NEURON")
    occursin("paperdigitaltwin", lowercase(parameters.teacher_schema)) ||
        error("distilled teacher schema omits PaperDigitalTwin")
    isempty(keys(Cell.trainable_parameters(parameters))) ||
        error("distilled internals leaked into a trainable tree")

    data = JLD2.load(path)
    payload = _required(data, :payload)
    _value(payload, :frozen_internal, false) === true ||
        error("distilled artifact is not frozen_internal")
    Symbol(_value(payload, :ablation_mode, :full)) === :full ||
        error("production accepts only a full-mechanism artifact")
    _required_string(
        payload,
        :official_teacher_manifest_sha256,
    ) == teacher.manifest_sha256 ||
        error("distilled artifact is not bound to official manifest")
    _required_string(payload, :raw_twin_file_sha256) ==
        twin_file_sha256 ||
        error("distilled raw-twin file lineage differs")
    _required_string(payload, :frozen_twin_parameter_hash) ==
        frozen.parameter_sha256 ||
        error("distilled payload twin-parameter hash differs")
    _required_string(payload, :frozen_twin_artifact_hash) ==
        frozen.artifact_sha256 ||
        error("distilled payload twin-artifact hash differs")
    _required_string(payload, :detailed_kernel_hash) ==
        teacher.mechanism_library_sha256 ||
        error("distilled payload mechanism hash differs")
    _required_string(payload, :morphology_hash) ==
        teacher.morphology_sha256 ||
        error("distilled payload morphology hash differs")
    _required_string(payload, :official_modeldb_source_hash) ==
        teacher.modeldb_tree_sha256 ||
        error("distilled ModelDB source hash differs")
    for name in (
        :source_dataset_hash,
        :distillation_dataset_hash,
        :distillation_config_hash,
    )
        _required_string(payload, name)
    end

    config = _required(payload, :config)
    semantics = Tuple(String.(
        collect(_required(config, :state_semantics)),
    ))
    semantics == STATE_SEMANTICS ||
        error("distilled state coordinates are not canonical")
    alignment = _required(payload, :coordinate_alignment)
    _value(alignment, :passed, false) === true ||
        error("coordinate-wise distillation alignment did not pass")
    String(_required(alignment, :supervision)) ==
        "coordinate_wise" ||
        error("distilled hidden state was not coordinate-supervised")
    maximum_fraction = Float64(_required(
        alignment,
        :maximum_off_coordinate_readout_fraction,
    ))
    maximum_fraction <= 0.05 ||
        error("coordinate-alignment allowance exceeds 5 percent")
    observed_fraction =
        _readout_off_coordinate_fraction(parameters)
    observed_fraction <= maximum_fraction ||
        error(
            "unconstrained hidden rotation detected: off-coordinate " *
            "readout fraction $observed_fraction > $maximum_fraction",
        )
    return observed_fraction
end

function load_production_bundle(
    teacher_manifest::AbstractString,
    twin_path::AbstractString,
    distilled_path::AbstractString;
    verify_teacher_shards::Bool=true,
)
    teacher = Teacher.load_official_teacher_metadata(
        teacher_manifest;
        verify_shards=verify_teacher_shards,
    )
    twin_source = abspath(twin_path)
    distilled_source = abspath(distilled_path)
    twin_file_sha256 = _sha256_file(twin_source)
    distilled_file_sha256 = _sha256_file(distilled_source)
    frozen = Twin.load_frozen_twin(twin_source)
    attestation =
        _validate_twin!(frozen, teacher, twin_file_sha256)
    parameters = Cell.load_distilled_artifact(distilled_source)
    _validate_distilled!(
        distilled_source,
        parameters,
        frozen,
        teacher,
        twin_file_sha256,
    )
    parameter_hash = Cell.parameter_sha256(parameters)
    return ProductionBundle(
        teacher,
        frozen,
        parameters,
        twin_source,
        distilled_source,
        twin_file_sha256,
        distilled_file_sha256,
        frozen.parameter_sha256,
        frozen.artifact_sha256,
        parameter_hash,
        attestation,
    )
end

function assert_production_bundle_unchanged!(
    bundle::ProductionBundle,
)
    _sha256_file(bundle.twin_path) == bundle.twin_file_sha256 ||
        error("frozen twin file changed after production preflight")
    _sha256_file(bundle.distilled_path) ==
        bundle.distilled_file_sha256 ||
        error("frozen distilled file changed after production preflight")
    Twin.assert_frozen_unchanged(
        bundle.frozen_twin;
        expected_artifact_sha256=
            bundle.twin_artifact_sha256,
    )
    Twin.parameter_sha256(bundle.frozen_twin.parameters) ==
        bundle.twin_parameter_sha256 ||
        error("in-memory frozen twin parameters changed")
    Cell.assert_parameter_sha256(
        bundle.distilled_parameters,
        bundle.distilled_parameter_sha256,
    )
    production_twin_attestation_sha256(
        bundle.frozen_twin,
        bundle.teacher,
    ) == bundle.twin_attestation_sha256 ||
        error("in-memory frozen twin metadata changed")
    return (
        frozen=true,
        twin_max_delta=0.0f0,
        distilled_max_delta=0.0f0,
        twin_parameter_sha256=bundle.twin_parameter_sha256,
        distilled_parameter_sha256=
            bundle.distilled_parameter_sha256,
    )
end

"""
Production construction boundary.

This intentionally fails closed while the inherited arena stores anatomical
locations as UInt8.  Once the arena is widened to UInt16, this function is
the single place that should construct a Tetris trainer from the validated
bundle.
"""
function build_production_trainer(
    bundle::ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    assert_production_bundle_unchanged!(bundle)
    Training.assert_official_location_index_supported(
        bundle.distilled_parameters,
    )
    return Training.PaperTrainer(
        model,
        parameters;
        cell_mode=:distilled_frozen,
        cell_artifact=bundle.distilled_path,
        kwargs...,
    )
end

end # module HDSWSNNTwinPropProduction
