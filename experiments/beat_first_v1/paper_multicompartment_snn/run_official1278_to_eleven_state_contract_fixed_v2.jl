# Critical-path entry point:
#
# corrected sealed V2 loader -> official 1278 bridge -> streaming 11-state
# distillation -> frozen artifact provenance verification.
#
# This file deliberately loads the canonical contract-fixed loader before the
# unchanged bridge and distiller. Their fallback loader branches therefore
# resolve to the already-patched sealed V2 module.

if !isdefined(
    Main,
    :PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2,
)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
        ),
    )
end

if !isdefined(
    Main,
    :DistillationDatasetBridgeOfficial1278SealedFinalV2,
)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_official1278_sealed_final_v2.jl",
        ),
    )
end

if !isdefined(
    Main,
    :run_sealed_v2_eleven_state_distillation,
)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "distill_eleven_state_cell_release_streaming_sealed_v2.jl",
        ),
    )
end

module Official1278ToElevenStateContractFixedV2

using JLD2
using JSON3
using SHA

const Canonical =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2
const Bridge =
    Main.DistillationDatasetBridgeOfficial1278SealedFinalV2
const DistillCore = Main.OfficialElevenStateDistillationCore
const Cell = DistillCore.Cell
const BridgeConfig = Bridge.ReleaseStreamingPrepareConfig
const DistillationConfig =
    Main.SealedV2ElevenStateDistillationConfig

export CHAIN_SCHEMA,
    CORRECTED_EVALUATOR_SOURCE_SHA256,
    CONTRACT_FIXED_LOADER_SOURCE_SHA256,
    ContractFixedV2SourcePins,
    assert_contract_fixed_loader!,
    run_contract_fixed_v2_chain,
    verify_distilled_artifact_provenance,
    verify_distilled_payload_provenance,
    verify_source_chain_inputs,
    main

const CHAIN_SCHEMA =
    "hd_swsnn.official1278_to_eleven_state.contract_fixed_v2.v1"
const CORRECTED_EVALUATOR_SOURCE_SHA256 =
    "f451a10e579bec1bf58e579920c68d811b70d32fcf6feacf7edb03c73f3f5d7e"
const CONTRACT_FIXED_LOADER_SOURCE_SHA256 =
    "0cf68cbcd6b7dec31e21d6b62896861b498105cf19cd3f32a88064c7b50d7d4b"
const CORRECTED_EVALUATOR_ID =
    "official-final-v2-signed1278-paper-window-reset-exact-auroc-v4-contract-fix"

Base.@kwdef struct ContractFixedV2SourcePins
    source_artifact_sha256::String
    source_manifest_sha256::String
    teacher_contract_sha256::String
end

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
        error("contract-fixed chain lacks $(String(name))")
    return value
end

function _sha(value, label)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a complete SHA-256")
    return digest
end

function _file_sha(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("required file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _normalized_pins(pins::ContractFixedV2SourcePins)
    return (;
        source_artifact_sha256=_sha(
            pins.source_artifact_sha256,
            "source artifact",
        ),
        source_manifest_sha256=_sha(
            pins.source_manifest_sha256,
            "source manifest",
        ),
        teacher_contract_sha256=_sha(
            pins.teacher_contract_sha256,
            "teacher contract",
        ),
    )
end

"""
Verify that the bridge and distiller both resolve through the audited
contract-fixed canonical loader.
"""
function assert_contract_fixed_loader!()
    loader_path = joinpath(
        @__DIR__,
        "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
    )
    _file_sha(loader_path) == CONTRACT_FIXED_LOADER_SOURCE_SHA256 ||
        error("contract-fixed canonical loader source differs")
    isdefined(Canonical, :SEALED_V2_CONTRACT_FIX_V2_APPLIED) &&
        Canonical.SEALED_V2_CONTRACT_FIX_V2_APPLIED === true ||
        error("corrected sealed V2 evaluator overlay is absent")
    Canonical.corrected_evaluator_source_sha256_v2() ==
        CORRECTED_EVALUATOR_SOURCE_SHA256 ||
        error("corrected evaluator source SHA-256 differs")
    Bridge.Bridge.Sealed === Canonical ||
        error("official1278 bridge bypassed the contract-fixed module")
    Main.Sealed === Canonical ||
        error("11-state distiller bypassed the contract-fixed module")
    return (;
        loader_source_sha256=CONTRACT_FIXED_LOADER_SOURCE_SHA256,
        evaluator_source_sha256=
            CORRECTED_EVALUATOR_SOURCE_SHA256,
        evaluator_id=CORRECTED_EVALUATOR_ID,
    )
end

function _load_source_bundle(path::AbstractString)
    artifact = abspath(path)
    bundle = Canonical._load(artifact)
    bundle isa Canonical.SealedOfficialELMRelease ||
        error("source artifact has the wrong sealed V2 type")
    Canonical.Twin.assert_frozen_official_elm_unchanged(
        bundle.frozen,
    )
    Canonical.canonical_sha256(
        bundle.attestation.payload,
    ) == bundle.attestation.attestation_sha256 ||
        error("source sealed attestation digest differs")
    return bundle
end

"""
Fail-closed preflight shared by bridge generation and distillation.

The source artifact pin is the exact sealed JLD2 file SHA-256. The returned
identity additionally binds the internal frozen twin parameter/artifact hashes
and the sealed attestation.
"""
function verify_source_chain_inputs(
    bridge_config::BridgeConfig,
    distill_config::DistillationConfig,
    pins::ContractFixedV2SourcePins,
)
    loader = assert_contract_fixed_loader!()
    expected = _normalized_pins(pins)
    abspath(bridge_config.output_directory) ==
        abspath(distill_config.bridge_dataset) ||
        error("bridge output and distillation input directories differ")
    bridge_config.require_full_public_counts ==
        distill_config.paper_scale ||
        error("bridge and distillation scale modes differ")
    bridge_config.minimum_twin_spike_auroc ==
        distill_config.minimum_spike_auroc ||
        error("bridge and distillation spike gates differ")
    _sha(
        distill_config.expected_source_manifest_sha256,
        "distillation source manifest",
    ) == expected.source_manifest_sha256 ||
        error("distillation manifest pin differs from chain pin")
    _sha(
        distill_config.expected_teacher_contract_sha256,
        "distillation teacher contract",
    ) == expected.teacher_contract_sha256 ||
        error("distillation teacher-contract pin differs from chain pin")
    _file_sha(distill_config.source_manifest) ==
        expected.source_manifest_sha256 ||
        error("source manifest bytes differ from chain pin")
    _file_sha(distill_config.sealed_artifact) ==
        expected.source_artifact_sha256 ||
        error("source sealed artifact bytes differ from chain pin")

    bundle = _load_source_bundle(distill_config.sealed_artifact)
    payload = bundle.attestation.payload
    evaluator = _required(payload, :evaluator)
    String(_required(evaluator, :id)) ==
        CORRECTED_EVALUATOR_ID ||
        error("sealed artifact used another evaluator contract")
    _sha(
        _required(evaluator, :source_sha256),
        "sealed evaluator source",
    ) == CORRECTED_EVALUATOR_SOURCE_SHA256 ||
        error("sealed artifact evaluator source differs")
    _sha(
        _required(_required(payload, :teacher), :manifest_sha256),
        "attested source manifest",
    ) == expected.source_manifest_sha256 ||
        error("sealed artifact belongs to another source manifest")
    _sha(
        _required(
            _required(payload, :teacher),
            :teacher_contract_sha256,
        ),
        "attested teacher contract",
    ) == expected.teacher_contract_sha256 ||
        error("sealed artifact belongs to another teacher contract")
    _required(_required(payload, :outcome), :gate_passed) === true ||
        error("source sealed V2 gate is not passed")

    raw = Canonical.Twin.load_frozen_official_elm(
        abspath(bridge_config.frozen_twin_path),
    )
    Canonical.Twin.assert_frozen_official_elm_unchanged(raw)
    raw.parameter_sha256 == bundle.frozen.parameter_sha256 ||
        error("bridge raw twin and sealed artifact parameters differ")
    raw.artifact_sha256 == bundle.frozen.artifact_sha256 ||
        error("bridge raw twin and sealed artifact identity differ")
    isempty(bridge_config.expected_twin_parameter_sha256) &&
        error("bridge twin parameter SHA-256 must be externally pinned")
    isempty(bridge_config.expected_twin_artifact_sha256) &&
        error("bridge twin artifact SHA-256 must be externally pinned")
    _sha(
        bridge_config.expected_twin_parameter_sha256,
        "bridge twin parameter",
    ) == raw.parameter_sha256 ||
        error("bridge twin parameter pin differs")
    _sha(
        bridge_config.expected_twin_artifact_sha256,
        "bridge twin artifact",
    ) == raw.artifact_sha256 ||
        error("bridge twin artifact pin differs")

    return (;
        schema=CHAIN_SCHEMA,
        source_artifact_sha256=
            expected.source_artifact_sha256,
        source_manifest_sha256=
            expected.source_manifest_sha256,
        teacher_contract_sha256=
            expected.teacher_contract_sha256,
        frozen_twin_parameter_sha256=raw.parameter_sha256,
        frozen_twin_artifact_sha256=raw.artifact_sha256,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
        corrected_evaluator_source_sha256=
            loader.evaluator_source_sha256,
        corrected_evaluator_id=loader.evaluator_id,
        contract_fixed_loader_source_sha256=
            loader.loader_source_sha256,
    )
end

"""
Verify provenance and internal freezing directly on a distilled payload.
"""
function verify_distilled_payload_provenance(payload, identity)
    _required(payload, :frozen_internal) === true ||
        error("distilled cell internals are not frozen")
    _required(payload, :ablation_mode) === :full ||
        error("distilled artifact is not the full mechanism")
    _required(payload, :provisional) === false ||
        error("provisional distillation cannot enter the chain")
    before = _required(payload, :frozen_twin_integrity_before)
    after = _required(payload, :frozen_twin_integrity_after)
    before === true && after === true ||
        error("upstream twin freeze integrity was not maintained")

    parameters = _required(payload, :parameters)
    parameters isa Cell.DistilledParameters ||
        error("distilled parameters have the wrong canonical type")
    Cell.is_frozen(parameters) ||
        error("11-state parameters are not frozen")
    isempty(keys(Cell.trainable_parameters(parameters))) ||
        error("11-state parameters expose optimizer-visible fields")
    parameter_sha256 = Cell.parameter_sha256(parameters)
    _sha(
        _required(payload, :parameter_sha256),
        "distilled parameter",
    ) == parameter_sha256 ||
        error("distilled parameter SHA-256 differs")
    size(parameters.compartment_projection) == (4, 642) ||
        error("frozen compartment projection is not 4 x 642")
    all(iszero, @view(parameters.compartment_projection[:, 1])) ||
        error("frozen projection permits a soma contact")
    all(iszero, @view(
        parameters.compartment_projection[:, 641:642],
    )) || error("frozen projection permits an axon contact")

    source = _required(payload, :source_bound_sealed_elm)
    _sha(
        _required(source, :sealed_artifact_sha256),
        "payload source artifact",
    ) == identity.source_artifact_sha256 ||
        error("distilled artifact source-file pin differs")
    for (field, expected) in (
        (:source_manifest_sha256, identity.source_manifest_sha256),
        (
            :source_teacher_contract_sha256,
            identity.teacher_contract_sha256,
        ),
    )
        _sha(_required(payload, field), "payload $(String(field))") ==
            expected ||
            error("distilled payload $(String(field)) differs")
        _sha(_required(source, field), "source-bound $(String(field))") ==
            expected ||
            error("source-bound $(String(field)) differs")
    end
    _sha(
        _required(source, :sealed_attestation_sha256),
        "source-bound attestation",
    ) == identity.sealed_attestation_sha256 ||
        error("source-bound sealed attestation differs")
    _sha(
        _required(source, :base_artifact_sha256),
        "source-bound frozen artifact",
    ) == identity.frozen_twin_artifact_sha256 ||
        error("source-bound frozen artifact differs")
    _sha(
        _required(source, :parameter_sha256),
        "source-bound frozen parameter",
    ) == identity.frozen_twin_parameter_sha256 ||
        error("source-bound frozen parameters differ")
    _sha(
        _required(payload, :frozen_twin_artifact_hash),
        "payload frozen twin artifact",
    ) == identity.frozen_twin_artifact_sha256 ||
        error("payload frozen twin artifact differs")
    _sha(
        _required(payload, :frozen_twin_parameter_hash),
        "payload frozen twin parameter",
    ) == identity.frozen_twin_parameter_sha256 ||
        error("payload frozen twin parameters differ")
    parameters.frozen_twin_artifact_hash ==
        identity.frozen_twin_artifact_sha256 ||
        error("frozen cell embeds another twin artifact")
    parameters.frozen_twin_parameter_hash ==
        identity.frozen_twin_parameter_sha256 ||
        error("frozen cell embeds another twin parameter set")

    return (;
        schema=CHAIN_SCHEMA,
        source_artifact_sha256=identity.source_artifact_sha256,
        source_manifest_sha256=identity.source_manifest_sha256,
        teacher_contract_sha256=identity.teacher_contract_sha256,
        frozen_twin_artifact_sha256=
            identity.frozen_twin_artifact_sha256,
        frozen_twin_parameter_sha256=
            identity.frozen_twin_parameter_sha256,
        sealed_attestation_sha256=
            identity.sealed_attestation_sha256,
        corrected_evaluator_source_sha256=
            identity.corrected_evaluator_source_sha256,
        distilled_parameter_sha256=parameter_sha256,
        frozen_internal=true,
        optimizer_visible_parameter_count=0,
        soma_contacts=0,
        axon_contacts=0,
    )
end

function verify_distilled_artifact_provenance(
    path::AbstractString,
    identity;
    expected_artifact_sha256::AbstractString="",
)
    source = abspath(path)
    artifact_sha256 = _file_sha(source)
    if !isempty(expected_artifact_sha256)
        artifact_sha256 ==
            _sha(expected_artifact_sha256, "distilled artifact") ||
            error("distilled artifact bytes differ from external pin")
    end
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("distilled artifact has no payload")
    verified =
        verify_distilled_payload_provenance(data["payload"], identity)
    return merge(
        verified,
        (; distilled_artifact_sha256=artifact_sha256),
    )
end

function _atomic_json(path::AbstractString, value)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination)) * ".json"
    try
        open(temporary, "w") do io
            JSON3.write(io, value)
        end
        mv(temporary, destination; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

"""
Run only the critical bridge -> distillation chain.

Set `prepare_bridge=false` to reuse an already verified bridge directory. The
distiller still performs its mandatory shard hashes and live frozen-ELM replay.
"""
function run_contract_fixed_v2_chain(
    bridge_config::BridgeConfig,
    distill_config::DistillationConfig,
    pins::ContractFixedV2SourcePins;
    prepare_bridge::Bool=true,
    receipt_path::Union{Nothing,AbstractString}=nothing,
)
    identity =
        verify_source_chain_inputs(
            bridge_config,
            distill_config,
            pins,
        )
    bridge_report = if prepare_bridge
        Bridge.prepare_distillation_dataset_release(bridge_config)
    else
        isdir(distill_config.bridge_dataset) ||
            error("prebuilt bridge dataset is absent")
        nothing
    end
    report =
        Main.run_sealed_v2_eleven_state_distillation(
            distill_config,
        )
    report.accepted === true ||
        error("distillation did not produce an accepted artifact")
    verified = verify_distilled_artifact_provenance(
        report.artifact_path,
        identity;
        expected_artifact_sha256=report.artifact_sha256,
    )
    receipt = merge(
        verified,
        (;
            bridge_prepared=prepare_bridge,
            bridge_manifest_sha256=
                bridge_report === nothing ?
                nothing :
                bridge_report.manifest_sha256,
            distillation_accepted=true,
            distillation_metrics_path=
                abspath(distill_config.metrics),
        ),
    )
    receipt_path === nothing ||
        _atomic_json(String(receipt_path), receipt)
    return (;
        identity,
        bridge=bridge_report,
        distillation=report,
        receipt,
    )
end

function _split_arguments(arguments)
    bridge = String[]
    distill = String[]
    chain = Dict{String,String}(
        "source-artifact-sha256" => get(
            ENV,
            "HD_TWINPROP_SOURCE_ARTIFACT_SHA256",
            "",
        ),
        "source-manifest-sha256" => get(
            ENV,
            "HD_TWINPROP_SOURCE_MANIFEST_SHA256",
            "",
        ),
        "teacher-contract-sha256" => get(
            ENV,
            "HD_TWINPROP_TEACHER_CONTRACT_SHA256",
            "",
        ),
        "prepare-bridge" => get(
            ENV,
            "HD_TWINPROP_PREPARE_BRIDGE",
            "true",
        ),
        "receipt" => get(
            ENV,
            "HD_TWINPROP_CHAIN_RECEIPT",
            "",
        ),
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        index == length(arguments) &&
            error("missing value for $token")
        value = arguments[index + 1]
        key = token[3:end]
        if startswith(key, "bridge-")
            push!(bridge, "--" * key[8:end], value)
        elseif startswith(key, "distill-")
            push!(distill, "--" * key[9:end], value)
        elseif haskey(chain, key)
            chain[key] = value
        else
            error("unknown chain option $token")
        end
        index += 2
    end
    return bridge, distill, chain
end

function main(arguments=ARGS)
    bridge_arguments, distill_arguments, chain =
        _split_arguments(arguments)
    bridge_config = Bridge.V6._parse_arguments(bridge_arguments)
    distill_config = Main._parse_arguments(distill_arguments)
    pins = ContractFixedV2SourcePins(
        source_artifact_sha256=
            chain["source-artifact-sha256"],
        source_manifest_sha256=
            chain["source-manifest-sha256"],
        teacher_contract_sha256=
            chain["teacher-contract-sha256"],
    )
    receipt =
        isempty(chain["receipt"]) ?
        nothing :
        chain["receipt"]
    return run_contract_fixed_v2_chain(
        bridge_config,
        distill_config,
        pins;
        prepare_bridge=parse(Bool, chain["prepare-bridge"]),
        receipt_path=receipt,
    )
end

end # module Official1278ToElevenStateContractFixedV2

if abspath(PROGRAM_FILE) == @__FILE__
    Official1278ToElevenStateContractFixedV2.main()
end
