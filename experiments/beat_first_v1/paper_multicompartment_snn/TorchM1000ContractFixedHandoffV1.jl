module TorchM1000ContractFixedHandoffV1

using JSON3
using NPZ
using SHA

if !isdefined(Main, :PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V3)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "LoadPaperELMTwinOfficialV2ProfiledCanonicalV3.jl",
        ),
    )
end
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

const Twin = Main.PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V3
const CorrectedV2 =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2

export ImportedTorchM1000Handoff,
    TORCH_CHECKPOINT_SCHEMA,
    TORCH_HANDOFF_SCHEMA,
    TORCH_FIT4096_VALIDATION_SCHEMA,
    materialize_torch_m1000_handoff,
    import_exported_torch_m1000_handoff,
    import_torch_m1000_checkpoint,
    corrected_v2_validation_gate,
    assert_imported_torch_m1000_unchanged,
    freeze_torch_m1000_handoff

const TORCH_CHECKPOINT_SCHEMA = "paper_elm_torch_cpu_variable.v1"
const TORCH_HANDOFF_SCHEMA =
    "hd_swsnn.torch_m1000.contract_fixed_handoff.v1"
const TORCH_EXPORT_SCHEMA =
    "hd_swsnn.torch_m1000.checkpoint_export.v1"
const TORCH_FIT4096_VALIDATION_SCHEMA =
    "paper_elm_torch_fit4096_base_val8.v1"
const COMPOSITE_CONTRACT_SCHEMA = "paper_elm_composite_fit.v1"
const EXPECTED_MEMORY = 1_000
const EXPECTED_HIDDEN = 2_000
const EXPECTED_OUTPUT = 6
const EXPECTED_VALIDATION_TRIALS = 8
const EXPECTED_VALIDATION_OBSERVATIONS = 8_000
const NUMERIC_ORACLE_MAX_ABS = 5.0e-5

const PINNED_FIT4096_VALIDATOR_SHA256 =
    "ca9cd5d5794929ec510fe25de1fd377edcbdec0dd5805f59541a66b2ed55a1e4"
const PINNED_CORRECTED_V2_SOURCE_SHA256 =
    "f451a10e579bec1bf58e579920c68d811b70d32fcf6feacf7edb03c73f3f5d7e"

const _FIT4096_VALIDATOR_SOURCE = joinpath(
    @__DIR__,
    "evaluate_paper_elm_torch_fit4096_base_val8_m1000.py",
)
const _CORRECTED_V2_SOURCE = joinpath(
    @__DIR__,
    "PaperELMTwinOfficialV2SealedReleaseV2ContractFixV2.jl",
)

const _PARAMETER_NAMES = (
    "proto_w_s",
    "input_weight",
    "input_bias",
    "memory_weight",
    "memory_bias",
    "output_weight",
    "output_bias",
)
const _BUFFER_NAMES = (
    "route_indices",
    "valid_indices_mask",
    "kappa_b",
    "kappa_m",
    "kappa_lambda",
    "nmda_mean",
    "nmda_scale",
)

"""
An imported but not yet frozen Torch M1000 checkpoint.

The parameters remain mutable so the digest must be checked immediately before
freezing.  `freeze_torch_m1000_handoff` is the only provided freeze path and
refuses a checkpoint whose base-val8 metrics do not pass the corrected V2
thresholds.
"""
struct ImportedTorchM1000Handoff{M,P,N,R,G}
    model::M
    parameters::P
    normalizer::N
    provenance::R
    validation_gate::G
    parameter_sha256::String
    import_sha256::String
end

@inline function _required(object, name::Symbol)
    if object isa AbstractDict
        haskey(object, name) && return object[name]
        haskey(object, String(name)) && return object[String(name)]
    elseif hasproperty(object, name)
        return getproperty(object, name)
    end
    error("required field `$name` is absent")
end

@inline function _optional(object, name::Symbol, default=nothing)
    if object isa AbstractDict
        haskey(object, name) && return object[name]
        haskey(object, String(name)) && return object[String(name)]
    elseif hasproperty(object, name)
        return getproperty(object, name)
    end
    return default
end

function _sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("file is absent: $source"))
    return bytes2hex(SHA.sha256(read(source)))
end

_sha256_text(value::AbstractString) =
    bytes2hex(SHA.sha256(codeunits(value)))

function _required_sha256(object, name::Symbol)
    value = lowercase(String(_required(object, name)))
    occursin(r"^[0-9a-f]{64}$", value) ||
        error("`$name` is not a SHA-256 digest")
    return value
end

function _same_path(left::AbstractString, right::AbstractString)
    normalize(path) = begin
        value = normpath(abspath(path))
        Sys.iswindows() ? lowercase(value) : value
    end
    return normalize(left) == normalize(right)
end

function _assert_no_heldout_opened(value)
    if value isa AbstractDict || value isa JSON3.Object ||
       value isa NamedTuple
        names = value isa AbstractDict ?
            collect(keys(value)) :
            collect(propertynames(value))
        for raw_name in names
            name = String(raw_name)
            child = value isa AbstractDict ?
                value[raw_name] :
                getproperty(value, Symbol(raw_name))
            lowered = lowercase(name)
            if occursin("heldout", lowered) &&
               occursin("opened", lowered)
                child === false ||
                    error("provenance claims held-out access in `$name`")
            end
            _assert_no_heldout_opened(child)
        end
    elseif value isa AbstractArray || value isa Tuple
        for child in value
            _assert_no_heldout_opened(child)
        end
    end
    return true
end

function _assert_pinned_sources()
    _sha256_file(_FIT4096_VALIDATOR_SOURCE) ==
        PINNED_FIT4096_VALIDATOR_SHA256 ||
        error("fit4096 corrected validator source changed")
    _sha256_file(_CORRECTED_V2_SOURCE) ==
        PINNED_CORRECTED_V2_SOURCE_SHA256 ||
        error("corrected V2 gate source changed")
    CorrectedV2.corrected_evaluator_source_sha256_v2() ==
        PINNED_CORRECTED_V2_SOURCE_SHA256 ||
        error("loaded corrected V2 evaluator differs from pinned source")
    return true
end

const _PYTHON_EXPORTER = raw"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT_SCHEMA = "paper_elm_torch_cpu_variable.v1"
EXPORT_SCHEMA = "hd_swsnn.torch_m1000.checkpoint_export.v1"
PARAMETERS = (
    "proto_w_s",
    "input_weight",
    "input_bias",
    "memory_weight",
    "memory_bias",
    "output_weight",
    "output_bias",
)
BUFFERS = (
    "route_indices",
    "valid_indices_mask",
    "kappa_b",
    "kappa_m",
    "kappa_lambda",
    "nmda_mean",
    "nmda_scale",
)

checkpoint_path = Path(sys.argv[1]).resolve()
weights_path = Path(sys.argv[2]).resolve()
metadata_path = Path(sys.argv[3]).resolve()
payload = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
if payload.get("schema") != CHECKPOINT_SCHEMA:
    raise ValueError("Torch checkpoint schema differs")
if int(payload.get("memory", -1)) != 1000:
    raise ValueError("Torch checkpoint is not M1000")
if int(payload.get("hidden", -1)) != 2000:
    raise ValueError("Torch checkpoint hidden width is not 2000")
if bool(payload.get("heldout_opened", True)):
    raise ValueError("Torch checkpoint does not assert heldout exclusion")

state = payload["model"]
required = set(PARAMETERS + BUFFERS)
if set(state) != required:
    raise ValueError(
        "Torch state_dict keys differ: "
        + repr(sorted(set(state) ^ required))
    )

def cpu_tensor(name):
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state_dict value {name} is not a tensor")
    value = value.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"state_dict value {name} is non-finite")
    return value

tensors = {name: cpu_tensor(name) for name in required}
expected_shapes = {
    "proto_w_s": (4500,),
    "input_weight": (2000, 1045),
    "input_bias": (2000,),
    "memory_weight": (1000, 2000),
    "memory_bias": (1000,),
    "output_weight": (6, 1000),
    "output_bias": (6,),
    "route_indices": (4500,),
    "valid_indices_mask": (4500,),
    "kappa_b": (45,),
    "kappa_m": (1000,),
    "kappa_lambda": (1000,),
    "nmda_mean": (4,),
    "nmda_scale": (4,),
}
for name, shape in expected_shapes.items():
    if tuple(tensors[name].shape) != shape:
        raise ValueError(
            f"state_dict shape {name}={tuple(tensors[name].shape)} "
            f"!= {shape}"
        )

# A fixed sparse input checks routing, SiLU, recurrence and all six readouts
# after Julia import.  It never opens a dataset or a held-out target.
x = torch.zeros((4, 1, 1278), dtype=torch.float32)
for t in range(4):
    x[t, 0, (17 * t + 3) % 639] = 0.25 * (t + 1)
    x[t, 0, 639 + ((29 * t + 7) % 639)] = -0.125 * (t + 1)
    x[t, 0, (53 * t + 11) % 1278] += 0.0625

route = tensors["route_indices"].to(torch.int64)
routed = x.index_select(2, route)
weighted = (
    routed
    * tensors["valid_indices_mask"].view(1, 1, -1)
    * torch.clamp_min(tensors["proto_w_s"], 0.0).view(1, 1, -1)
)
branch_input = weighted.reshape(4, 1, 45, 100).sum(dim=3)
branch = torch.zeros((1, 45), dtype=torch.float32)
memory = torch.zeros((1, 1000), dtype=torch.float32)
raw_steps = []
for t in range(4):
    branch = branch * tensors["kappa_b"] + branch_input[t]
    decayed = memory * tensors["kappa_m"]
    hidden_pre = F.linear(
        torch.cat((branch, decayed), dim=1),
        tensors["input_weight"],
        tensors["input_bias"],
    )
    hidden = hidden_pre / (1.0 + torch.exp(-hidden_pre))
    delta = 1.7159 * torch.tanh(
        (2.0 / 3.0)
        * F.linear(
            hidden,
            tensors["memory_weight"],
            tensors["memory_bias"],
        )
    )
    memory = (
        decayed
        + (1.0 - tensors["kappa_lambda"]) * delta
    )
    raw_steps.append(
        F.linear(
            memory,
            tensors["output_weight"],
            tensors["output_bias"],
        )
    )
raw = torch.stack(raw_steps, dim=0)
if not bool(torch.isfinite(raw).all()):
    raise ValueError("numeric oracle is non-finite")

arrays = {
    f"state_{name}": value.numpy()
    for name, value in tensors.items()
}
arrays["oracle_input_signed1278"] = (
    x.permute(2, 0, 1).contiguous().numpy()
)
arrays["oracle_raw"] = raw.permute(2, 0, 1).contiguous().numpy()
np.savez(weights_path, **arrays)

def safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint metadata type {type(value)!r}")

events = payload.get("events", [])
checkpoint_metadata = {
    str(key): safe(value)
    for key, value in payload.items()
    if key not in ("model", "julia_adam")
}
optimizer = payload.get("julia_adam", {})
optimizer_summary = {
    "present": isinstance(optimizer, dict),
    "first_parameter_names": sorted(
        str(value) for value in optimizer.get("first", {}).keys()
    ) if isinstance(optimizer, dict) else [],
    "second_parameter_names": sorted(
        str(value) for value in optimizer.get("second", {}).keys()
    ) if isinstance(optimizer, dict) else [],
}
checkpoint_sha256 = hashlib.sha256(
    checkpoint_path.read_bytes()
).hexdigest()
weights_sha256 = hashlib.sha256(
    weights_path.read_bytes()
).hexdigest()
metadata = {
    "schema": EXPORT_SCHEMA,
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
    "weights_npz_sha256": weights_sha256,
    "checkpoint_schema": payload["schema"],
    "memory": int(payload["memory"]),
    "hidden": int(payload["hidden"]),
    "update_index": int(payload["update_index"]),
    "heldout_opened": bool(payload["heldout_opened"]),
    "event_count": len(events),
    "source_bridge_metadata": safe(
        payload.get("source_bridge_metadata", {})
    ),
    "checkpoint_metadata": checkpoint_metadata,
    "optimizer_summary": optimizer_summary,
}
metadata_path.write_text(
    json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ),
    encoding="utf-8",
)
"""

"""
Export a trusted local `paper_elm_torch_cpu_variable.v1` checkpoint to a
language-neutral NPZ plus truthful JSON metadata.

This reads only the checkpoint.  No dataset path is accepted, so the handoff
cannot open validation or held-out teacher shards.
"""
function materialize_torch_m1000_handoff(
    checkpoint_path::AbstractString,
    output_npz::AbstractString,
    output_metadata_json::AbstractString;
    python::AbstractString="python",
)
    checkpoint = abspath(checkpoint_path)
    weights = abspath(output_npz)
    metadata = abspath(output_metadata_json)
    isfile(checkpoint) ||
        throw(ArgumentError("Torch checkpoint is absent: $checkpoint"))
    ispath(weights) &&
        throw(ArgumentError("refusing to overwrite $weights"))
    ispath(metadata) &&
        throw(ArgumentError("refusing to overwrite $metadata"))
    mkpath(dirname(weights))
    mkpath(dirname(metadata))
    command = Cmd([
        String(python),
        "-c",
        _PYTHON_EXPORTER,
        checkpoint,
        weights,
        metadata,
    ])
    run(command)
    isfile(weights) || error("Python handoff did not create NPZ")
    isfile(metadata) || error("Python handoff did not create metadata JSON")
    return (;
        checkpoint,
        weights_npz=weights,
        metadata_json=metadata,
    )
end

function _array(arrays, name::AbstractString)
    haskey(arrays, name) || error("handoff array `$name` is absent")
    return arrays[name]
end

function _float_vector(arrays, name, length_expected)
    value = vec(Float32.(_array(arrays, name)))
    length(value) == length_expected ||
        error("handoff vector `$name` has the wrong length")
    all(isfinite, value) ||
        error("handoff vector `$name` is non-finite")
    return value
end

function _float_matrix(arrays, name, size_expected)
    value = Matrix{Float32}(_array(arrays, name))
    size(value) == size_expected ||
        error("handoff matrix `$name` has the wrong shape")
    all(isfinite, value) ||
        error("handoff matrix `$name` is non-finite")
    return value
end

function _model_and_parameters(arrays)
    config = Twin.OfficialELMConfig()
    model = Twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    Twin.assert_profiled_official_elm_contract(model)

    route = vec(Int64.(_array(arrays, "state_route_indices")))
    length(route) == 4_500 ||
        error("Torch routing vector has the wrong length")
    route .+ 1 == model.input_indices ||
        error("Torch routing differs from canonical Julia routing")
    valid = _float_vector(
        arrays,
        "state_valid_indices_mask",
        4_500,
    )
    valid == model.valid_indices_mask ||
        error("Torch valid routing mask differs from canonical Julia")
    kappa_b = _float_vector(arrays, "state_kappa_b", 45)
    maximum(abs.(kappa_b .- model.kappa_b)) <= 2.0f-7 ||
        error("Torch branch decay differs from canonical Julia")
    decay = Twin.Core.memory_decay_factors(
        model,
        (;
            proto_w_s=zeros(Float32, 4_500),
        ),
    )
    kappa_m = _float_vector(arrays, "state_kappa_m", 1_000)
    kappa_lambda =
        _float_vector(arrays, "state_kappa_lambda", 1_000)
    maximum(abs.(kappa_m .- decay.kappa_m)) <= 2.0f-7 ||
        error("Torch memory decay differs from canonical Julia")
    maximum(abs.(kappa_lambda .- decay.kappa_lambda)) <= 2.0f-7 ||
        error("Torch lambda decay differs from canonical Julia")

    parameters = (;
        proto_w_s=_float_vector(
            arrays,
            "state_proto_w_s",
            4_500,
        ),
        input_weight=_float_matrix(
            arrays,
            "state_input_weight",
            (2_000, 1_045),
        ),
        input_bias=_float_vector(
            arrays,
            "state_input_bias",
            2_000,
        ),
        memory_weight=_float_matrix(
            arrays,
            "state_memory_weight",
            (1_000, 2_000),
        ),
        memory_bias=_float_vector(
            arrays,
            "state_memory_bias",
            1_000,
        ),
        output_weight=_float_matrix(
            arrays,
            "state_output_weight",
            (6, 1_000),
        ),
        output_bias=_float_vector(
            arrays,
            "state_output_bias",
            6,
        ),
    )
    normalizer = Twin.OfficialELMNormalizer(
        _float_vector(arrays, "state_nmda_mean", 4),
        _float_vector(arrays, "state_nmda_scale", 4),
    )
    return model, parameters, normalizer
end

function _numeric_oracle_delta(model, parameters, arrays)
    input = Array{Float32,3}(
        _array(arrays, "oracle_input_signed1278"),
    )
    expected = Array{Float32,3}(_array(arrays, "oracle_raw"))
    size(input) == (1_278, 4, 1) ||
        error("Torch numeric oracle input shape differs")
    size(expected) == (EXPECTED_OUTPUT, 4, 1) ||
        error("Torch numeric oracle output shape differs")
    output = Twin.Core.twin_forward(model, parameters, input)
    actual = cat(
        reshape(output.spike_logit, 1, size(input, 2), size(input, 3)),
        reshape(output.voltage, 1, size(input, 2), size(input, 3)),
        output.nmda;
        dims=1,
    )
    size(actual) == size(expected) ||
        error("Julia numeric oracle output shape differs")
    all(isfinite, actual) ||
        error("Julia numeric oracle output is non-finite")
    delta = maximum(abs.(Float64.(actual) .- Float64.(expected)))
    delta <= NUMERIC_ORACLE_MAX_ABS ||
        error(
            "Torch/Julia numeric oracle max abs $delta exceeds " *
            "$NUMERIC_ORACLE_MAX_ABS",
        )
    return delta
end

function _validation_metrics(validation)
    String(_required(validation, :schema)) ==
        TORCH_FIT4096_VALIDATION_SCHEMA ||
        error("fit4096 validation schema differs")
    metrics = _required(validation, :metrics)
    Bool(_required(metrics, :heldout_opened)) === false ||
        error("validation does not assert held-out exclusion")
    String(_required(metrics, :validation_source)) == "base-only" ||
        error("fit4096 validation is not base-only")
    Int(_required(metrics, :validation_trials)) ==
        EXPECTED_VALIDATION_TRIALS ||
        error("fit4096 validation trial count differs")
    Int(_required(metrics, :observations)) ==
        EXPECTED_VALIDATION_OBSERVATIONS ||
        error("fit4096 validation observation count differs")
    spike_auroc = Float64(_required(metrics, :exact_spike_auroc))
    voltage_rmse_mv =
        Float64(_required(metrics, :clip_voltage_rmse_mv))
    nmda = Float64.(collect(_required(
        metrics,
        :nmda_normalized_rmse,
    )))
    length(nmda) == 4 ||
        error("fit4096 validation must report four NMDA regions")
    all(isfinite, (spike_auroc, voltage_rmse_mv)) &&
        all(isfinite, nmda) ||
        error("fit4096 validation metrics are non-finite")
    return (;
        spike_auroc,
        voltage_rmse_mv,
        nmda_normalized_rmse_by_region=nmda,
        validation_trials=EXPECTED_VALIDATION_TRIALS,
        observations=EXPECTED_VALIDATION_OBSERVATIONS,
        validation_source="base-only",
        heldout_opened=false,
    )
end

"""
Apply the immutable corrected V2 thresholds to a fit4096 base-val8 JSON object.

This is a validation gate only.  Passing it does not claim held-out
verification and does not create a sealed release attestation.
"""
function corrected_v2_validation_gate(validation)
    _assert_pinned_sources()
    metrics = _validation_metrics(validation)
    gate = CorrectedV2._gate((;
        spike_auroc=metrics.spike_auroc,
        voltage_rmse_mv=metrics.voltage_rmse_mv,
        nmda_normalized_rmse_by_region=
            metrics.nmda_normalized_rmse_by_region,
    ))
    return merge(
        gate,
        (;
            contract="corrected-v2-base-val8",
            metrics,
            fit4096_validator_sha256=
                PINNED_FIT4096_VALIDATOR_SHA256,
            corrected_v2_evaluator_sha256=
                PINNED_CORRECTED_V2_SOURCE_SHA256,
            heldout_evaluated=false,
        ),
    )
end

function _checkpoint_provenance(
    checkpoint_path,
    checkpoint_metadata_raw,
    validation_path,
    validation_raw,
    weights_npz_sha256,
    oracle_max_abs,
)
    checkpoint_metadata = JSON3.read(checkpoint_metadata_raw)
    validation = JSON3.read(validation_raw)
    _assert_no_heldout_opened(checkpoint_metadata)
    _assert_no_heldout_opened(validation)
    String(_required(checkpoint_metadata, :schema)) ==
        TORCH_EXPORT_SCHEMA ||
        error("Torch export metadata schema differs")
    String(_required(checkpoint_metadata, :checkpoint_schema)) ==
        TORCH_CHECKPOINT_SCHEMA ||
        error("Torch checkpoint schema differs")
    Int(_required(checkpoint_metadata, :memory)) == EXPECTED_MEMORY ||
        error("Torch checkpoint is not M1000")
    Int(_required(checkpoint_metadata, :hidden)) == EXPECTED_HIDDEN ||
        error("Torch checkpoint hidden width differs")
    Bool(_required(checkpoint_metadata, :heldout_opened)) === false ||
        error("Torch checkpoint does not assert held-out exclusion")
    _same_path(
        String(_required(checkpoint_metadata, :checkpoint_path)),
        checkpoint_path,
    ) || error("Torch export metadata names a different checkpoint")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    _required_sha256(checkpoint_metadata, :checkpoint_sha256) ==
        checkpoint_sha256 ||
        error("Torch checkpoint digest differs")
    _required_sha256(checkpoint_metadata, :weights_npz_sha256) ==
        weights_npz_sha256 ||
        error("Torch handoff NPZ digest differs")

    source = _required(checkpoint_metadata, :source_bridge_metadata)
    details = _required(source, :m1000_fit4096_mainline)
    String(_required(details, :schema)) == COMPOSITE_CONTRACT_SCHEMA ||
        error("fit4096 composite contract schema differs")
    base_manifest_sha256 =
        _required_sha256(details, :base_manifest_sha256)
    augmentation_manifest_sha256 =
        _required_sha256(details, :augmentation_manifest_sha256)
    composite_sha256 = _required_sha256(details, :composite_sha256)
    _required_sha256(source, :manifest_sha256) == composite_sha256 ||
        error("checkpoint source manifest is not the composite contract")
    _required_sha256(validation, :base_manifest_sha256) ==
        base_manifest_sha256 ||
        error("validation/base manifest digest differs")
    _required_sha256(validation, :augmentation_manifest_sha256) ==
        augmentation_manifest_sha256 ||
        error("validation/augmentation manifest digest differs")
    _same_path(
        String(_required(validation, :checkpoint)),
        checkpoint_path,
    ) || error("validation JSON names a different checkpoint")

    update_index = Int(_required(checkpoint_metadata, :update_index))
    update_count = Int(_required(details, :updates))
    initial_update_index =
        Int(_required(details, :initial_update_index))
    update_index == initial_update_index + update_count ||
        error("checkpoint update index differs from actual stage count")
    Int(_required(checkpoint_metadata, :event_count)) == update_count ||
        error("checkpoint event count differs from actual update count")
    Int(_required(validation, :update_index)) == update_index ||
        error("validation update index differs from checkpoint")
    Int(_required(validation, :memory)) == EXPECTED_MEMORY ||
        error("validation memory width differs")
    Int(_required(validation, :hidden)) == EXPECTED_HIDDEN ||
        error("validation hidden width differs")
    batch_size = Int(_required(details, :batch_size))
    batch_size in (8, 32) ||
        error("fit4096 checkpoint batch size must be explicitly 8 or 32")
    Bool(_required(details, :heldout_opened)) === false ||
        error("fit4096 training provenance claims held-out access")
    Int(_required(details, :augmentation_heldout_trials)) == 0 ||
        error("fit4096 augmentation is not train-only")
    Int(_required(details, :fit_trials)) == 4_096 ||
        error("fit4096 composite fit count differs")

    gate = corrected_v2_validation_gate(validation)
    return (;
        schema=TORCH_HANDOFF_SCHEMA,
        checkpoint_path=abspath(checkpoint_path),
        checkpoint_sha256,
        checkpoint_schema=TORCH_CHECKPOINT_SCHEMA,
        checkpoint_metadata_json=checkpoint_metadata_raw,
        checkpoint_metadata_sha256=
            _sha256_text(checkpoint_metadata_raw),
        weights_npz_sha256,
        validation_path=abspath(validation_path),
        validation_json=validation_raw,
        validation_json_sha256=_sha256_text(validation_raw),
        validation_schema=TORCH_FIT4096_VALIDATION_SCHEMA,
        base_manifest_sha256,
        augmentation_manifest_sha256,
        composite_contract_sha256=composite_sha256,
        batch_size,
        initial_update_index,
        update_count,
        update_index,
        fit_trials=4_096,
        validation_trials=EXPECTED_VALIDATION_TRIALS,
        numeric_oracle_max_abs=oracle_max_abs,
        numeric_oracle_tolerance=NUMERIC_ORACLE_MAX_ABS,
        actual_training_provenance=:torch_update_count,
        legacy_three_by_thirty_five_evidence_claimed=false,
        heldout_targets_opened=false,
    ), gate
end

function _import_digest(parameter_sha256, provenance, gate)
    return CorrectedV2.canonical_sha256((;
        schema=TORCH_HANDOFF_SCHEMA,
        parameter_sha256,
        checkpoint_sha256=provenance.checkpoint_sha256,
        checkpoint_metadata_sha256=
            provenance.checkpoint_metadata_sha256,
        weights_npz_sha256=provenance.weights_npz_sha256,
        validation_json_sha256=provenance.validation_json_sha256,
        base_manifest_sha256=provenance.base_manifest_sha256,
        augmentation_manifest_sha256=
            provenance.augmentation_manifest_sha256,
        composite_contract_sha256=
            provenance.composite_contract_sha256,
        batch_size=provenance.batch_size,
        update_count=provenance.update_count,
        update_index=provenance.update_index,
        numeric_oracle_max_abs=provenance.numeric_oracle_max_abs,
        validation_gate=gate,
    ))
end

function _assemble_import(
    checkpoint_path,
    arrays,
    checkpoint_metadata_raw,
    validation_path,
    validation_raw,
    weights_npz_sha256,
)
    model, parameters, normalizer =
        _model_and_parameters(arrays)
    oracle_max_abs =
        _numeric_oracle_delta(model, parameters, arrays)
    provenance, gate = _checkpoint_provenance(
        checkpoint_path,
        checkpoint_metadata_raw,
        validation_path,
        validation_raw,
        weights_npz_sha256,
        oracle_max_abs,
    )
    parameter_sha256 =
        Twin.official_parameter_sha256(parameters)
    import_sha256 =
        _import_digest(parameter_sha256, provenance, gate)
    return ImportedTorchM1000Handoff(
        model,
        parameters,
        normalizer,
        provenance,
        gate,
        parameter_sha256,
        import_sha256,
    )
end

"""
Import a previously materialized NPZ/JSON handoff and its exact validation
JSON.  The checkpoint itself is hashed again; neither this function nor the
Python materializer accepts a teacher dataset path.
"""
function import_exported_torch_m1000_handoff(
    checkpoint_path::AbstractString,
    weights_npz::AbstractString,
    checkpoint_metadata_json::AbstractString,
    validation_json::AbstractString,
)
    checkpoint = abspath(checkpoint_path)
    weights = abspath(weights_npz)
    metadata_path = abspath(checkpoint_metadata_json)
    validation_path = abspath(validation_json)
    for path in (checkpoint, weights, metadata_path, validation_path)
        isfile(path) || throw(ArgumentError("handoff input is absent: $path"))
    end
    weights_sha256 = _sha256_file(weights)
    arrays = NPZ.npzread(weights)
    checkpoint_metadata_raw = read(metadata_path, String)
    validation_raw = read(validation_path, String)
    return _assemble_import(
        checkpoint,
        arrays,
        checkpoint_metadata_raw,
        validation_path,
        validation_raw,
        weights_sha256,
    )
end

"""
One-call trusted-local import from `.pt` plus the already produced corrected
base-val8 validation JSON.

The temporary NPZ is deleted only after all arrays have been copied into the
Julia object.  No training or validation computation is performed here.
"""
function import_torch_m1000_checkpoint(
    checkpoint_path::AbstractString,
    validation_json::AbstractString;
    python::AbstractString="python",
)
    return mktempdir() do temporary
        weights = joinpath(temporary, "weights.npz")
        metadata = joinpath(temporary, "checkpoint_metadata.json")
        materialize_torch_m1000_handoff(
            checkpoint_path,
            weights,
            metadata;
            python,
        )
        import_exported_torch_m1000_handoff(
            checkpoint_path,
            weights,
            metadata,
            validation_json,
        )
    end
end

function assert_imported_torch_m1000_unchanged(
    imported::ImportedTorchM1000Handoff,
)
    Twin.assert_profiled_official_elm_contract(imported.model)
    Twin.official_parameter_sha256(imported.parameters) ==
        imported.parameter_sha256 ||
        error("imported Torch parameters changed")
    imported.import_sha256 ==
        _import_digest(
            imported.parameter_sha256,
            imported.provenance,
            imported.validation_gate,
        ) || error("imported Torch handoff provenance changed")
    imported.provenance.heldout_targets_opened === false ||
        error("imported handoff claims held-out access")
    imported.provenance.
        legacy_three_by_thirty_five_evidence_claimed === false ||
        error("imported handoff forged legacy 3x35 evidence")
    return true
end

"""
Freeze the imported weights only after the corrected V2 base-val8 gate passes.

The returned `FrozenOfficialELMTwin` intentionally remains
`verification_status=:unverified`: held-out targets have not been opened and
this handoff does not forge the legacy three-restart/35-epoch evidence expected
by the older sealed-release finalizer.
"""
function freeze_torch_m1000_handoff(
    imported::ImportedTorchM1000Handoff,
)
    assert_imported_torch_m1000_unchanged(imported)
    imported.validation_gate.passed === true ||
        error("corrected V2 validation gate did not pass; freeze refused")
    provenance = imported.provenance
    frozen = Twin.freeze_official_elm_twin(
        imported.model,
        imported.parameters,
        imported.normalizer;
        metadata=(;
            handoff_schema=TORCH_HANDOFF_SCHEMA,
            torch_checkpoint_schema=TORCH_CHECKPOINT_SCHEMA,
            torch_checkpoint_path=provenance.checkpoint_path,
            torch_checkpoint_sha256=provenance.checkpoint_sha256,
            torch_checkpoint_metadata_json=
                provenance.checkpoint_metadata_json,
            torch_checkpoint_metadata_sha256=
                provenance.checkpoint_metadata_sha256,
            torch_weights_npz_sha256=provenance.weights_npz_sha256,
            torch_batch_size=provenance.batch_size,
            torch_update_count=provenance.update_count,
            torch_update_index=provenance.update_index,
            base_manifest_sha256=provenance.base_manifest_sha256,
            augmentation_manifest_sha256=
                provenance.augmentation_manifest_sha256,
            composite_contract_sha256=
                provenance.composite_contract_sha256,
            validation_schema=provenance.validation_schema,
            validation_json=provenance.validation_json,
            validation_json_sha256=
                provenance.validation_json_sha256,
            corrected_v2_validation_gate=
                imported.validation_gate,
            numeric_oracle_max_abs=
                provenance.numeric_oracle_max_abs,
            numeric_oracle_tolerance=
                provenance.numeric_oracle_tolerance,
            actual_training_provenance=
                provenance.actual_training_provenance,
            legacy_three_by_thirty_five_evidence_claimed=false,
            heldout_targets_opened=false,
            freeze_basis=
                :corrected_v2_validation_pass_without_heldout,
            imported_parameter_sha256=imported.parameter_sha256,
            imported_handoff_sha256=imported.import_sha256,
        ),
    )
    Twin.assert_frozen_official_elm_unchanged(frozen)
    frozen.metadata.verification_status === :unverified ||
        error("Torch handoff must remain unverified before held-out")
    return frozen
end

end # module
