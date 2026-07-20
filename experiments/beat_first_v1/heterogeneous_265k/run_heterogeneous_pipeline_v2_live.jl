#!/usr/bin/env julia

# One-process production entry point for the v2 heterogeneous pipeline.  This
# file is intentionally source-bound by an explicit CLI SHA-256 and by the
# runner's concrete-adapter constructor.  It never launches a child process.

include(joinpath(@__DIR__, "post_idle_substrate.jl"))
using .PostIdleBenchmarkSubstrate
include(joinpath(@__DIR__, "post_idle_production_provider.jl"))
include(joinpath(@__DIR__, "heterogeneous_pipeline_runner_v2.jl"))

module BeatFirstHeterogeneousPipelineLiveCLI

using Dates
using JLD2
using JSON3
using SHA
using Serialization

using Main.PostIdleBenchmarkSubstrate
using Main.BeatFirstHeterogeneousPipelineV2

const H = Main.BeatFirstHeterogeneous265K
const S = Main.SparseDynamic3Layer
const LIVE_ADAPTER_SOURCE_PATH = abspath(@__FILE__)
const LIVE_SCHEMA = "heterogeneous-265k-pipeline-v2-live-run-v1"
const WORKLOAD_SCHEMA = "heterogeneous-265k-pipeline-v2-train-workload-v1"
const FORBIDDEN_VALIDATION_SEEDS = Set(8001:8008)
const FORBIDDEN_SEALED_SEEDS = Set(91001:91032)

struct LiveWorkItem
    order::UInt64
    workload_id::String
    expected_payload_sha256::String
    set_index::Int
    candidate_start::Int
    candidate_count::Int
end

struct LivePackedPayload
    item::LiveWorkItem
    set_id::String
    candidate_input::Any
    candidate_indices::Vector{Int}
    action_digests::Vector{String}
    raw_values::Vector{Vector{Float32}}
    packed_hashstem::Matrix{Float32}
    payload_sha256::String
end

struct LiveHashedPayload
    packed::LivePackedPayload
    hashstem_output::Matrix{Float32}
    hashstem_output_sha256::String
    actual_backend::Symbol
    physical_calls::Dict{String,Int}
    host_to_device_bytes::Int
    device_to_host_bytes::Int
    synchronization_count::Int
    physical_receipts::Vector{Dict{String,Any}}
end

struct LiveSparsePayload
    hashed::LiveHashedPayload
    outputs::Matrix{Float32}
    route_ids::Vector{NTuple{3,Vector{Int32}}}
    output_sha256::String
    route_sha256::String
    top1_index::Int
    top1_action_sha256::String
    accounting::Dict{String,Any}
    physical_receipts::Vector{Dict{String,Any}}
end

struct LiveStageOutcome
    stage::Symbol
    payload::Any
    physical_receipt::Dict{String,Any}
end

mutable struct LiveAdapter
    provider::Main.PostIdleProductionProvider
    workload_manifest::Dict{String,Any}
    workload_items::Vector{LiveWorkItem}
    provider_binding_sha256::String
    adapter_source_sha256::String
    source_bindings::Dict{String,String}
    input_bindings::Dict{String,String}
    sparse_variant::Symbol
    active_counts::NTuple{3,Int}
    hash_backend::Symbol
    ranking_output_index::Int
    sparse_index_sha256::String
end

_valid_sha256(value) = value isa AbstractString &&
    length(value) == 64 && all(c -> c in "0123456789abcdef", value)

function _require_sha256(value, name)
    text = String(value)
    _valid_sha256(text) || throw(ArgumentError("$name must be lowercase SHA-256"))
    return text
end

function _sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("file does not exist: $source"))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

_sha256_bytes(bytes) = bytes2hex(SHA.sha256(bytes))

function _write_u64!(io, value::Integer)
    write(io, htol(UInt64(value)))
end

function _write_string!(io, value::AbstractString)
    bytes = codeunits(String(value))
    _write_u64!(io, length(bytes))
    write(io, bytes)
end

function _write_float_array!(io, value::AbstractArray{Float32})
    _write_u64!(io, ndims(value))
    for dimension in size(value)
        _write_u64!(io, dimension)
    end
    for element in value
        write(io, htol(reinterpret(UInt32, element)))
    end
end

function _sha_float_matrix(value::Matrix{Float32})
    io = IOBuffer()
    _write_float_array!(io, value)
    return _sha256_bytes(take!(io))
end

function _sha_routes(routes::Vector{NTuple{3,Vector{Int32}}})
    io = IOBuffer()
    _write_u64!(io, length(routes))
    for route in routes, layer in route
        _write_u64!(io, length(layer))
        for neuron in layer
            write(io, htol(reinterpret(UInt32, neuron)))
        end
    end
    return _sha256_bytes(take!(io))
end

function _sha_sparse_indexes(runtime)
    io = IOBuffer()
    Serialization.serialize(io, runtime.indexes)
    return _sha256_bytes(take!(io))
end

function _workload_payload_sha256(
    set_id::String,
    indices::Vector{Int},
    action_digests::Vector{String},
    raw_values::Vector{Vector{Float32}},
    packed::Matrix{Float32},
)
    io = IOBuffer()
    _write_string!(io, set_id)
    _write_u64!(io, length(indices))
    for index in indices
        _write_u64!(io, index)
    end
    for digest in action_digests
        _write_string!(io, _require_sha256(digest, "action digest"))
    end
    for raw in raw_values
        _write_float_array!(io, raw)
    end
    _write_float_array!(io, packed)
    return _sha256_bytes(take!(io))
end

function _canonical_binding_sha256(
    source_bindings::Dict{String,String},
    input_bindings::Dict{String,String},
    config_bindings::Dict{String,String},
)
    io = IOBuffer()
    for (namespace, values) in (
        ("source", source_bindings),
        ("input", input_bindings),
        ("config", config_bindings),
    )
        for key in sort!(collect(keys(values)))
            _write_string!(io, namespace)
            _write_string!(io, key)
            _write_string!(io, values[key])
        end
    end
    return _sha256_bytes(take!(io))
end

function live_adapter_identity(adapter::LiveAdapter)
    return (
        adapter_source_path=LIVE_ADAPTER_SOURCE_PATH,
        adapter_source_sha256=adapter.adapter_source_sha256,
        provider_binding_sha256=adapter.provider_binding_sha256,
    )
end

function _dict(value, name)
    value isa AbstractDict || throw(ArgumentError("$name must be a JSON object"))
    return Dict{String,Any}(string(key) => item for (key, item) in pairs(value))
end

function _strict_bool(value, name)
    value isa Bool || throw(ArgumentError("$name must be a JSON boolean"))
    return value
end

function _strict_int(value, name)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$name must be a JSON integer",
    ))
    return Int(value)
end

function _read_json(path::AbstractString, name::AbstractString)
    parsed = JSON3.read(read(path, String), Dict{String,Any})
    return _dict(parsed, name)
end

function _finite_number(value, name)
    value isa Real && !(value isa Bool) || throw(ArgumentError("$name must be numeric"))
    observed = Float64(value)
    isfinite(observed) || throw(ArgumentError("$name must be finite"))
    return observed
end

function _validate_npu_gate!(path::String, expected_sha256::String)
    _sha256_file(path) == expected_sha256 || error("NPU adoption gate SHA-256 mismatch")
    decision = _read_json(path, "NPU adoption decision")
    decision["schema"] == "hashstem-sparse3-direct-k-scale-decision-v1" || error(
        "NPU adoption decision schema mismatch",
    )
    _strict_bool(get(decision, "passed", nothing), "NPU decision passed") || error(
        "NPU k-scale decision did not pass",
    )
    reasons = get(decision, "reasons", nothing)
    reasons isa AbstractVector && isempty(reasons) || error("NPU decision has reasons")
    gate = _dict(get(decision, "gate", nothing), "NPU decision gate")
    gate["schema"] == "hashstem-sparse3-cpu-k64-vs-npu-k128-gate-v1" || error(
        "NPU nested gate schema mismatch",
    )
    _strict_bool(get(gate, "passed", nothing), "NPU nested gate passed") || error(
        "NPU nested gate did not pass",
    )
    gate_reasons = get(gate, "reasons", nothing)
    gate_reasons isa AbstractVector && isempty(gate_reasons) || error(
        "NPU nested gate has reasons",
    )
    for field in (
        "cpu_k64_p50_ns", "cpu_k64_p95_ns", "npu_k128_p50_ns", "npu_k128_p95_ns",
        "component_p50_speedup", "baseline_quality", "candidate_quality",
    )
        _finite_number(gate[field], "NPU gate $field")
    end
    _finite_number(gate["component_p50_speedup"], "NPU component speedup") >= 1.15 ||
        error("NPU gate component speedup is below 1.15x")
    numeric = _dict(get(decision, "numeric", nothing), "NPU numeric evidence")
    for prefix in ("route", "action_top1", "state_action_top1", "state_action_top2")
        matches = _strict_int(numeric["npu_k128_$(prefix)_matches"],
            "NPU $prefix matches")
        total = _strict_int(numeric["npu_k128_$(prefix)_total"], "NPU $prefix total")
        total > 0 && matches == total || error("NPU $prefix identity gate failed")
    end
    _strict_int(numeric["npu_k128_top2_mismatches"], "NPU top2 mismatches") == 0 ||
        error("NPU top2 mismatch gate failed")
    _strict_int(numeric["npu_k128_top2_total"], "NPU top2 total") > 0 || error(
        "NPU top2 total is empty",
    )
    for field in ("npu_k128_hashstem_max_abs_error",
                  "npu_k128_sparse_output_max_abs_error_vs_cpu_k128")
        _finite_number(numeric[field], field) >= 0 || error("$field is negative")
    end
    openvino = _dict(get(decision, "openvino", nothing), "NPU OpenVINO identity")
    String.(get(openvino, "npu_execution_devices", Any[])) == ["NPU"] || error(
        "NPU gate did not execute exclusively on exact NPU",
    )
    artifacts = _dict(get(decision, "artifacts", nothing), "NPU gate artifacts")
    for name in (
        "teacher_checkpoint", "hashstem_weights", "snapshot_metadata", "fixed_ir",
        "fixed_bin", "dynamic_ir", "dynamic_bin", "workload",
    )
        artifact = _dict(get(artifacts, name, nothing), "NPU artifact $name")
        digest = _require_sha256(artifact["sha256"], "NPU artifact $name SHA-256")
        artifact_path = abspath(String(artifact["path"]))
        isfile(artifact_path) || error("NPU artifact $name is missing")
        filesize(artifact_path) == _strict_int(artifact["bytes"], "NPU artifact $name bytes") ||
            error("NPU artifact $name byte count changed")
        _sha256_file(artifact_path) == digest || error("NPU artifact $name bytes changed")
    end
    workload = _dict(get(decision, "workload", nothing), "NPU gate workload")
    _require_sha256(workload["packed_inputs_sha256"], "NPU packed-input digest")
    _require_sha256(workload["state_order_sha256"], "NPU state-order digest")
    raw = _dict(get(decision, "raw_records", nothing), "NPU raw records")
    raw_path = abspath(joinpath(dirname(path), String(raw["path"])))
    _inside_root(dirname(realpath(path)), raw_path) || error("NPU raw records escape root")
    _sha256_file(raw_path) == _require_sha256(raw["sha256"], "NPU raw-record SHA-256") ||
        error("NPU raw-record bytes differ")
    source = _dict(get(decision, "source", nothing), "NPU gate source closure")
    local_sources = Dict(
        "runner_sha256" => joinpath(@__DIR__, "run_hashstem_sparse3_k_scale_benchmark.jl"),
        "bridge_sha256" => joinpath(@__DIR__, "sparse_hashstem_bridge.jl"),
        "gate_sha256" => joinpath(@__DIR__, "hashstem_sparse3_k_scale_gate.jl"),
    )
    for (field, source_path) in local_sources
        _sha256_file(source_path) == _require_sha256(source[field], "NPU $field") || error(
            "NPU gate source closure changed: $field",
        )
    end
    H.sparse_source_closure_sha256() == _require_sha256(
        source["sparse_source_closure_sha256"], "NPU sparse source closure",
    ) || error("NPU sparse source closure changed")
    return decision
end

function _bind_npu_gate_to_live!(decision, input_bindings, loaded_master, snapshot_metadata)
    artifacts = _dict(decision["artifacts"], "NPU gate artifacts")
    artifact_sha(name) = _require_sha256(
        _dict(artifacts[name], "NPU artifact $name")["sha256"],
        "NPU artifact $name SHA-256",
    )
    artifact_sha("teacher_checkpoint") == input_bindings["sparse_checkpoint"] || error(
        "NPU gate teacher checkpoint differs from live sparse checkpoint",
    )
    artifact_sha("snapshot_metadata") == input_bindings["snapshot_metadata"] || error(
        "NPU gate snapshot metadata differs from live snapshot",
    )
    artifact_sha("hashstem_weights") == String(loaded_master.manifest.weights_sha256) ||
        error("NPU gate HashStem weights differ from live master")
    artifact_sha("fixed_ir") == String(snapshot_metadata["xml_sha256"]) || error(
        "NPU gate fixed IR differs from live snapshot",
    )
    artifact_sha("fixed_bin") == String(snapshot_metadata["bin_sha256"]) || error(
        "NPU gate fixed BIN differs from live snapshot",
    )
    teacher = _dict(decision["teacher_checkpoint"], "NPU teacher checkpoint")
    String(teacher["variant"]) == "k64" || error(
        "NPU k-scale decision is not derived from the ordinary k64 teacher checkpoint",
    )
    return true
end

function _inside_root(root::String, path::String)
    relative = relpath(path, root)
    components = splitpath(relative)
    return relative != ".." && !isempty(components) && first(components) != ".." &&
           !isabspath(relative)
end

function _validate_dataset_and_workload!(
    dataset_manifest_path::String,
    dataset_manifest_sha256::String,
    workload_manifest::Dict{String,Any},
    workload_sha256::String,
)
    _sha256_file(dataset_manifest_path) == dataset_manifest_sha256 || error(
        "teacher_v3 dataset manifest SHA-256 mismatch",
    )
    manifest = _read_json(dataset_manifest_path, "teacher_v3 manifest")
    _strict_int(get(manifest, "format_version", nothing), "dataset format_version") == 3 ||
        error("teacher_v3 dataset format must be 3")
    run_metadata = _dict(get(manifest, "run_metadata", nothing), "dataset run_metadata")
    _strict_bool(
        get(run_metadata, "held_out_development_validation_sealed_seeds_used", nothing),
        "dataset held-out-seed flag",
    ) === false || error("teacher_v3 manifest used held-out seeds")
    workload_manifest["schema"] == WORKLOAD_SCHEMA || error("workload schema mismatch")
    workload_manifest["split"] == "teacher_v3_train" || error(
        "live workload must be teacher_v3_train",
    )
    _strict_bool(get(workload_manifest, "reserved_seed_free", nothing),
        "workload reserved_seed_free") || error("workload did not exclude reserved seeds")
    for field in ("development_seeds_used", "validation_seeds_used", "sealed_seeds_used")
        _strict_bool(get(workload_manifest, field, nothing), "workload $field") === false ||
            error("workload $field must be false")
    end
    String(workload_manifest["workload_sha256"]) == workload_sha256 || error(
        "workload manifest/input SHA-256 binding mismatch",
    )
    String(workload_manifest["dataset_manifest_sha256"]) == dataset_manifest_sha256 ||
        error("workload/dataset manifest binding mismatch")

    parts = get(manifest, "parts", nothing)
    parts isa AbstractVector && !isempty(parts) || error("dataset parts are missing")
    indexed = Dict{String,Dict{String,Any}}()
    for raw_part in parts
        part = _dict(raw_part, "dataset part")
        key = String(get(part, "episode_key", ""))
        isempty(key) && error("dataset episode key is empty")
        haskey(indexed, key) && error("duplicate dataset episode key")
        indexed[key] = part
    end
    source_parts = get(workload_manifest, "source_parts", nothing)
    source_parts isa AbstractVector && !isempty(source_parts) || error(
        "workload source_parts are missing",
    )
    dataset_root = dirname(realpath(dataset_manifest_path))
    observed_seeds = Int[]
    observed_keys = Set{String}()
    for raw_binding in source_parts
        binding = _dict(raw_binding, "workload source-part binding")
        Set(keys(binding)) == Set(("episode_key", "part_sha256")) || error(
            "source-part binding key set changed",
        )
        key = String(binding["episode_key"])
        key in observed_keys && error("duplicate workload episode key")
        push!(observed_keys, key)
        part = get(indexed, key, nothing)
        part === nothing && error("workload episode is absent from dataset")
        String(get(part, "split", "")) == "train" && startswith(key, "v3|train|") ||
            error("workload source part is not teacher_v3 train")
        expected = _require_sha256(binding["part_sha256"], "source part digest")
        expected == _require_sha256(part["sha256"], "dataset part digest") || error(
            "workload source-part digest mismatch",
        )
        relative = String(get(part, "relative_path", ""))
        isempty(relative) && error("dataset part path is empty")
        isabspath(relative) && error("dataset part path must be relative")
        part_path = abspath(joinpath(dataset_root, relative))
        _inside_root(dataset_root, part_path) || error("dataset part path escapes root")
        isfile(part_path) || error("dataset part is missing")
        filesize(part_path) == _strict_int(part["bytes"], "dataset part bytes") || error(
            "dataset part byte count mismatch",
        )
        _sha256_file(part_path) == expected || error("dataset part SHA-256 mismatch")
        seed = _strict_int(part["seed"], "dataset part seed")
        push!(observed_seeds, seed)
    end
    length(unique(observed_seeds)) == length(observed_seeds) || error(
        "workload source seeds are duplicated",
    )
    environment_seeds = get(workload_manifest, "environment_seeds", nothing)
    environment_seeds isa AbstractVector || error("environment_seeds must be an array")
    expected_seeds = [_strict_int(seed, "environment seed") for seed in environment_seeds]
    sort(expected_seeds) == sort(observed_seeds) || error(
        "environment seeds differ from source-part seeds",
    )
    forbidden = union(
        intersect(Set(expected_seeds), FORBIDDEN_VALIDATION_SEEDS),
        intersect(Set(expected_seeds), FORBIDDEN_SEALED_SEEDS),
    )
    isempty(forbidden) || error("reserved validation/sealed seeds are forbidden")
    return nothing
end

function _load_work_items(manifest::Dict{String,Any}, candidate_sets)
    raw_items = get(manifest, "items", nothing)
    raw_items isa AbstractVector && length(raw_items) >= 2 || error(
        "overlap workload requires at least two explicitly ordered items",
    )
    items = LiveWorkItem[]
    seen_ids = Set{String}()
    for (expected_order, raw_item) in enumerate(raw_items)
        item = _dict(raw_item, "workload item")
        Set(keys(item)) == Set((
            "order", "workload_id", "workload_payload_sha256", "set_index",
            "candidate_start", "candidate_count",
        )) || error("workload item key set changed")
        order = _strict_int(item["order"], "workload order")
        order == expected_order || error("workload order is not exact and contiguous")
        workload_id = String(item["workload_id"])
        isempty(workload_id) && error("workload ID is empty")
        workload_id in seen_ids && error("workload ID is duplicated")
        push!(seen_ids, workload_id)
        set_index = _strict_int(item["set_index"], "set index")
        1 <= set_index <= length(candidate_sets) || error("set index is out of range")
        candidate_start = _strict_int(item["candidate_start"], "candidate start")
        candidate_count = _strict_int(item["candidate_count"], "candidate count")
        1 <= candidate_count <= 16 || error("candidate count must be in 1:16")
        _, available = Main._candidate_input(candidate_sets[set_index])
        1 <= candidate_start <= available &&
            candidate_start + candidate_count - 1 <= available || error(
                "candidate chunk is outside its candidate set",
            )
        push!(items, LiveWorkItem(
            UInt64(order), workload_id,
            _require_sha256(item["workload_payload_sha256"], "workload payload digest"),
            set_index, candidate_start, candidate_count,
        ))
    end
    return items
end

function _metadata_value(metadata, key::String)
    haskey(metadata, key) && return metadata[key]
    haskey(metadata, Symbol(key)) && return metadata[Symbol(key)]
    throw(KeyError(key))
end

function _runtime_for_variant(loaded_sparse, variant)
    source_variant = String(_metadata_value(loaded_sparse.metadata, "variant"))
    source_variant == "k64" || source_variant == String(variant.name) || error(
        "live sparse checkpoint must be ordinary k64 or the exact requested variant",
    )
    source = loaded_sparse.runtime
    S.parameter_count(source.model) == S.TOTAL_PARAMETERS || error(
        "sparse checkpoint is not the exact 19.924M bank",
    )
    source_counts = ntuple(layer -> source.model.layers[layer].active_count, 3)
    if source_counts == variant.active_counts
        return source, false
    end
    source_variant == "k64" || error("only ordinary k64 may derive another active width")
    cloned = deepcopy(source)
    layers = ntuple(3) do layer_id
        original = cloned.model.layers[layer_id]
        S.DynamicSparseLayer(
            original.theta,
            original.value_dim,
            variant.active_counts[layer_id],
            layer_id,
        )
    end
    model = S.ThreeLayerSparseModel(layers, cloned.model.head, cloned.model.bias)
    indexes = ntuple(3) do layer_id
        S.WTALSHIndex.WTAIndex(
            model.layers[layer_id].theta;
            config=S._layer_wta_config(layer_id, variant.active_counts[layer_id]),
            route_dims=S.ROUTE_DIM,
        )
    end
    runtime = S.ThreeLayerRuntime(
        model, indexes, cloned.bank_optimizers, cloned.head_optimizer,
    )
    S.parameter_count(runtime.model) == S.TOTAL_PARAMETERS || error(
        "derived live sparse parameter count drift",
    )
    S.active_parameter_count(runtime.model) == variant.active_parameters || error(
        "derived live sparse active-parameter count drift",
    )
    return runtime, true
end

function _load_live_adapter(
    config::RunConfig,
    workload_manifest_path::String,
    workload_manifest_sha256::String,
    dataset_manifest_path::String,
    dataset_manifest_sha256::String,
    sparse_variant::Symbol,
    hash_backend::Symbol,
    source_bindings::Dict{String,String},
    input_bindings::Dict{String,String},
    provider_binding_sha256::String,
)
    workload_manifest = _read_json(workload_manifest_path, "live workload manifest")
    _validate_dataset_and_workload!(
        dataset_manifest_path, dataset_manifest_sha256, workload_manifest,
        config.input.initial_sha256,
    )
    witness_metadata, candidate_sets = Main._load_witness(config)
    items = _load_work_items(workload_manifest, candidate_sets)
    if hash_backend == :NPU
        any(item -> item.candidate_count == 16, items) || error(
            "NPU live workload must contain at least one exact full batch16",
        )
        any(item -> 1 <= item.candidate_count < 16, items) || error(
            "NPU live workload must contain at least one real CPU_TAIL item",
        )
    end
    loaded_sparse = S.load_checkpoint(config.sparse_checkpoint.path)
    variant = H.sparse_bridge_variant_spec(sparse_variant)
    sparse_runtime, derived_active_width = _runtime_for_variant(loaded_sparse, variant)
    ntuple(layer -> sparse_runtime.model.layers[layer].active_count, 3) ==
        variant.active_counts || error("live sparse active width drift")
    sparse_workspace = S.ThreeLayerWorkspace(sparse_runtime)
    loaded_master, weights = Main._load_hash_weights(config)
    snapshot_metadata, xml = Main._snapshot_binding(config, loaded_master)
    npu_runtime = hash_backend == :NPU ? Main._compile_npu(xml) : nothing
    provider_metadata = Dict{String,Any}(
        "implementation_status" => "LIVE_PRODUCTION_ADAPTER_PENDING_VALIDATION",
        "hash_backend" => String(hash_backend),
        "sparse_variant" => String(variant.name),
        "active_counts" => collect(variant.active_counts),
        "source_checkpoint_variant" =>
            String(_metadata_value(loaded_sparse.metadata, "variant")),
        "derived_active_width_view" => derived_active_width,
        "sparse_index_sha256" => _sha_sparse_indexes(sparse_runtime),
        "ranking_output_index" => Int(witness_metadata["ranking_output_index"]),
        "NPU_execution_devices" => npu_runtime === nothing ? String[] :
            npu_runtime.execution_devices,
        "NPU_compile_ticks" => npu_runtime === nothing ? nothing : npu_runtime.compile_ticks,
        "master_weights_sha256" => String(loaded_master.manifest.weights_sha256),
        "snapshot_version" => Int(snapshot_metadata["snapshot_version"]),
    )
    post_idle = Main.PostIdleProductionProvider(
        config,
        qpc_frequency(),
        witness_metadata,
        candidate_sets,
        sparse_runtime,
        sparse_workspace,
        weights,
        npu_runtime,
        provider_metadata,
    )
    return LiveAdapter(
        post_idle,
        workload_manifest,
        items,
        provider_binding_sha256,
        source_bindings["live_cli"],
        source_bindings,
        input_bindings,
        variant.name,
        variant.active_counts,
        hash_backend,
        Int(witness_metadata["ranking_output_index"]),
        String(provider_metadata["sparse_index_sha256"]),
    ), loaded_master, snapshot_metadata
end

function _pack_stage!(adapter::LiveAdapter, context, item::LiveWorkItem)
    item.order == context.workload_order || error("pack workload order drift")
    item.workload_id == context.workload_id || error("pack workload ID drift")
    item.expected_payload_sha256 == context.workload_payload_sha256 || error(
        "pack workload digest drift",
    )
    item.candidate_count == context.candidate_count || error("pack candidate count drift")
    set = adapter.provider.candidate_sets[item.set_index]
    input, available = Main._candidate_input(set)
    last = item.candidate_start + item.candidate_count - 1
    last <= available || error("pack chunk exceeded candidate set")
    indices = collect(item.candidate_start:last)
    set_id = String(Main._property(set, :set_id))
    all_actions = Main._action_digests(set, available)
    actions = all_actions[indices]
    raw_values = Vector{Vector{Float32}}(undef, length(indices))
    q = Vector{Float32}(undef, 64)
    x = Vector{Float32}(undef, 496)
    for (position, candidate) in enumerate(indices)
        S.BeatFirstSparseFeatures.split_candidate_features!(q, x, input, candidate)
        raw_values[position] = copy(x)
    end
    packed = Matrix{Float32}(undef, length(indices), 559)
    H.pack_hashstem_input!(packed, input, indices)
    all(isfinite, packed) || error("packed HashStem input is non-finite")
    digest = _workload_payload_sha256(set_id, indices, actions, raw_values, packed)
    digest == item.expected_payload_sha256 || error(
        "computed workload payload SHA-256 differs from manifest",
    )
    payload = LivePackedPayload(
        item, set_id, input, indices, actions, raw_values, packed, digest,
    )
    receipt = Dict{String,Any}(
        "stage" => "pack",
        "operation" => "E_CORE_PACK",
        "candidate_count" => length(indices),
        "packed_bytes" => sizeof(packed) + sum(sizeof, raw_values),
        "workload_payload_sha256" => digest,
        "action_digest_sha256" => _sha256_bytes(codeunits(join(actions, ""))),
        "synchronized" => true,
        "readback_complete" => true,
    )
    return LiveStageOutcome(:pack, payload, receipt)
end

function _cpu_hash_stage!(adapter::LiveAdapter, context, previous::LiveStageOutcome)
    packed = previous.payload::LivePackedPayload
    count = size(packed.packed_hashstem, 1)
    output = Matrix{Float32}(undef, count, 256)
    scratch = H.HashStemReferenceScratch(count)
    H.hashstem_reference!(output, scratch, packed.packed_hashstem, adapter.provider.hash_weights)
    all(isfinite, output) || error("CPU HashStem output is non-finite")
    backend = context.backend in (:CPU_TAIL, :CPU_FALLBACK) ? context.backend : :CPU
    receipt = Dict{String,Any}(
        "stage" => "hashstem",
        "operation" => String(backend),
        "physical_calls" => Dict(String(backend) => 1),
        "synchronized" => true,
        "readback_complete" => true,
        "output_sha256" => _sha_float_matrix(output),
    )
    payload = LiveHashedPayload(
        packed, output, String(receipt["output_sha256"]), backend,
        Dict(String(backend) => 1), 0, 0, 0,
        Dict{String,Any}[copy(previous.physical_receipt), copy(receipt)],
    )
    return LiveStageOutcome(:hash_cpu, payload, receipt)
end

function _npu_hash_stage!(adapter::LiveAdapter, context, previous::LiveStageOutcome)
    context.backend == :NPU || error("NPU callback received non-NPU context")
    packed = previous.payload::LivePackedPayload
    size(packed.packed_hashstem) == (16, 559) || error(
        "NPU HashStem requires one exact 16x559 call",
    )
    trace = RawStageTrace(qpc_frequency())
    output = Main._npu_chunk!(adapter.provider, packed.packed_hashstem, trace)
    all(isfinite, output) || error("NPU HashStem output is non-finite")
    receipt = Dict{String,Any}(
        "stage" => "hashstem",
        "operation" => "NPU_FIXED_B16",
        "physical_calls" => Dict("NPU" => 1),
        "execution_devices" => adapter.provider.npu_runtime.execution_devices,
        "host_to_device_bytes" => 16 * 559 * sizeof(Float32),
        "device_to_host_bytes" => 16 * 256 * sizeof(Float32),
        "synchronization_count" => 1,
        "synchronized" => true,
        "readback_complete" => true,
        "output_sha256" => _sha_float_matrix(output),
    )
    payload = LiveHashedPayload(
        packed, output, String(receipt["output_sha256"]), :NPU, Dict("NPU" => 1),
        16 * 559 * sizeof(Float32), 16 * 256 * sizeof(Float32), 1,
        Dict{String,Any}[copy(previous.physical_receipt), copy(receipt)],
    )
    return LiveStageOutcome(:hash_npu, payload, receipt)
end

function _sparse_stage!(adapter::LiveAdapter, context, previous::LiveStageOutcome)
    hashed = previous.payload::LiveHashedPayload
    views = H.split_hashstem_output(hashed.hashstem_output)
    count = context.candidate_count
    outputs = Matrix{Float32}(undef, S.OUTPUT_DIM, count)
    routes = Vector{NTuple{3,Vector{Int32}}}(undef, count)
    scored_rows = zeros(Int, 3)
    rerank_macs = zeros(Int, 3)
    gross_gather_bytes = 0
    for candidate in 1:count
        sparse_input = S.ThreeLayerInput(
            (
                Vector{Float32}(@view views.query_1[candidate, :]),
                Vector{Float32}(@view views.query_2[candidate, :]),
                Vector{Float32}(@view views.query_3[candidate, :]),
            ),
            hashed.packed.raw_values[candidate],
            Vector{Float32}(@view views.context[candidate, :]),
            Vector{Float32}(@view views.next_hold_passthrough[candidate, :]),
        )
        result = S.route_forward!(
            adapter.provider.sparse_runtime,
            adapter.provider.sparse_workspace,
            sparse_input,
        )
        outputs[:, candidate] .= result.output
        routes[candidate] = ntuple(layer -> copy(result.tape.ids[layer]), 3)
        for layer in 1:3
            scored_rows[layer] += result.telemetry.scored_rows[layer]
            rerank_macs[layer] += result.telemetry.rerank_macs[layer]
        end
        gross_gather_bytes += result.telemetry.gross_weight_gather_bytes
    end
    all(isfinite, outputs) || error("sparse output is non-finite")
    all(route -> ntuple(layer -> length(route[layer]), 3) == adapter.active_counts, routes) ||
        error("sparse active width drift")
    ranking = @view outputs[adapter.ranking_output_index, :]
    top1 = first(eachindex(ranking))
    for index in Iterators.drop(eachindex(ranking), 1)
        ranking[index] > ranking[top1] && (top1 = index)
    end
    accounting = Dict{String,Any}(
        "candidate_count" => count,
        "active_counts" => collect(adapter.active_counts),
        "total_parameters" => S.parameter_count(adapter.provider.sparse_runtime.model),
        "active_parameters_per_candidate" =>
            S.active_parameter_count(adapter.provider.sparse_runtime.model),
        "scored_rows" => scored_rows,
        "rerank_macs" => rerank_macs,
        "gross_weight_gather_bytes" => gross_gather_bytes,
    )
    receipt = Dict{String,Any}(
        "stage" => "sparse",
        "operation" => "P_CORE_ALL_SPARSE_LAYERS_ONE_CALL",
        "candidate_count" => count,
        "output_sha256" => _sha_float_matrix(outputs),
        "route_sha256" => _sha_routes(routes),
        "top1_index" => top1,
        "top1_action_sha256" => hashed.packed.action_digests[top1],
        "accounting" => accounting,
        "synchronized" => true,
        "readback_complete" => true,
    )
    physical_receipts = copy(hashed.physical_receipts)
    push!(physical_receipts, copy(receipt))
    payload = LiveSparsePayload(
        hashed, outputs, routes, String(receipt["output_sha256"]),
        String(receipt["route_sha256"]), top1,
        hashed.packed.action_digests[top1], accounting, physical_receipts,
    )
    return LiveStageOutcome(:sparse, payload, receipt)
end

function run_live_adapter_stage!(adapter::LiveAdapter, stage::Symbol, context, payload)
    stage == :pack && return _pack_stage!(adapter, context, payload::LiveWorkItem)
    stage == :hash_cpu && return _cpu_hash_stage!(adapter, context, payload::LiveStageOutcome)
    stage == :hash_npu && return _npu_hash_stage!(adapter, context, payload::LiveStageOutcome)
    stage == :sparse && return _sparse_stage!(adapter, context, payload::LiveStageOutcome)
    stage == :stem_train_igpu && error(
        "iGPU live HashStem trainer has no production adapter; enablement is rejected",
    )
    throw(ArgumentError("unknown concrete live stage $stage"))
end

function validate_live_adapter_stage_outcome!(
    adapter::LiveAdapter,
    stage::Symbol,
    context,
    outcome::LiveStageOutcome,
)
    outcome.stage == stage || error("live stage outcome tag mismatch")
    receipt = outcome.physical_receipt
    haskey(receipt, "synchronized") && haskey(receipt, "readback_complete") || error(
        "physical receipt omitted synchronization/readback fields",
    )
    _strict_bool(receipt["synchronized"], "physical synchronized") || error(
        "physical stage did not synchronize",
    )
    _strict_bool(receipt["readback_complete"], "physical readback") || error(
        "physical stage did not complete readback",
    )
    if stage == :pack
        packed = outcome.payload::LivePackedPayload
        packed.payload_sha256 == context.workload_payload_sha256 || error(
            "packed payload digest/context mismatch",
        )
        length(packed.candidate_indices) == context.candidate_count || error(
            "packed candidate count/context mismatch",
        )
    elseif stage in (:hash_cpu, :hash_npu)
        hashed = outcome.payload::LiveHashedPayload
        size(hashed.hashstem_output) == (context.candidate_count, 256) || error(
            "HashStem output shape/context mismatch",
        )
        _sha_float_matrix(hashed.hashstem_output) == hashed.hashstem_output_sha256 || error(
            "HashStem output digest mismatch",
        )
        if stage == :hash_npu
            context.backend == :NPU && context.candidate_count == 16 || error(
                "NPU physical receipt is not one full batch16",
            )
            hashed.physical_calls == Dict("NPU" => 1) &&
                hashed.synchronization_count == 1 || error("NPU physical call receipt mismatch")
        else
            context.backend in (:CPU, :CPU_TAIL, :CPU_FALLBACK) || error(
                "CPU HashStem context backend mismatch",
            )
            context.backend == :CPU_TAIL && context.candidate_count == 16 && error(
                "CPU_TAIL cannot represent a full batch16",
            )
        end
    elseif stage == :sparse
        sparse = outcome.payload::LiveSparsePayload
        size(sparse.outputs) == (S.OUTPUT_DIM, context.candidate_count) || error(
            "sparse output shape/context mismatch",
        )
        _sha_float_matrix(sparse.outputs) == sparse.output_sha256 || error(
            "sparse output digest mismatch",
        )
        _sha_routes(sparse.route_ids) == sparse.route_sha256 || error(
            "sparse route digest mismatch",
        )
        _require_sha256(sparse.top1_action_sha256, "top1 action digest")
    else
        error("production iGPU stage is unavailable")
    end
    return outcome
end

function _parse_cli(arguments)
    length(arguments) % 2 == 0 || throw(ArgumentError(
        "every live CLI option must be an explicit --name value pair",
    ))
    options = Dict{String,String}()
    for index in 1:2:length(arguments)
        key = arguments[index]
        startswith(key, "--") || throw(ArgumentError("invalid CLI option $key"))
        haskey(options, key) && throw(ArgumentError("duplicate CLI option $key"))
        options[key] = arguments[index + 1]
    end
    required = Set((
        "--workload", "--workload-sha256", "--workload-manifest",
        "--workload-manifest-sha256", "--dataset-manifest",
        "--dataset-manifest-sha256", "--sparse-checkpoint",
        "--sparse-checkpoint-sha256", "--sparse-variant", "--master-manifest",
        "--master-manifest-sha256", "--snapshot-metadata",
        "--snapshot-metadata-sha256", "--system-contract",
        "--system-contract-sha256", "--post-idle-contract-sha256",
        "--post-idle-provider-sha256", "--source-manifest",
        "--source-manifest-sha256", "--runner-source-sha256",
        "--live-source-sha256", "--model-id", "--hash-backend", "--slot-count",
        "--overlap-threshold", "--npu-gate", "--npu-gate-sha256",
        "--enable-igpu-stem-training", "--output-directory",
    ))
    Set(keys(options)) == required || throw(ArgumentError(
        "live CLI option set differs; missing=$(collect(setdiff(required, Set(keys(options))))) " *
        "extra=$(collect(setdiff(Set(keys(options)), required)))",
    ))
    return options
end

function _binding(name, path, digest)
    return ArtifactBinding(name, abspath(path), _require_sha256(digest, "$name SHA-256"))
end

function _source_closure(options)
    paths = Dict(
        "live_cli" => LIVE_ADAPTER_SOURCE_PATH,
        "pipeline_runner_v2" => joinpath(@__DIR__, "heterogeneous_pipeline_runner_v2.jl"),
        "post_idle_provider" => joinpath(@__DIR__, "post_idle_production_provider.jl"),
        "post_idle_substrate" => joinpath(@__DIR__, "post_idle_substrate.jl"),
        "heterogeneous_module" => joinpath(@__DIR__, "Heterogeneous265K.jl"),
        "windows_cpu_sets" => joinpath(@__DIR__, "windows_cpu_sets.jl"),
        "hashstem" => joinpath(@__DIR__, "hashstem.jl"),
        "hashstem_master" => joinpath(@__DIR__, "hashstem_master.jl"),
        "sparse_bridge" => joinpath(@__DIR__, "sparse_hashstem_bridge.jl"),
        "sparse_module" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),
        "sparse_geometry" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "geometry.jl"),
        "sparse_model" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "model.jl"),
        "sparse_optimizer" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "optimizer.jl"),
        "sparse_runtime" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "runtime.jl"),
        "sparse_checkpoint" => joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "checkpoint.jl"),
        "project" => joinpath(@__DIR__, "..", "Project.toml"),
        "manifest" => joinpath(@__DIR__, "..", "Manifest.toml"),
    )
    bindings = Dict(key => _sha256_file(path) for (key, path) in paths)
    bindings["live_cli"] == _require_sha256(options["--live-source-sha256"],
        "live source SHA-256") || error("live CLI source SHA-256 mismatch")
    bindings["pipeline_runner_v2"] == _require_sha256(
        options["--runner-source-sha256"], "runner source SHA-256",
    ) || error("pipeline runner source SHA-256 mismatch")
    bindings["post_idle_provider"] == _require_sha256(
        options["--post-idle-provider-sha256"], "post-idle provider SHA-256",
    ) || error("post-idle provider source SHA-256 mismatch")
    return bindings, paths
end

function _lineage(adapter, loaded_master, snapshot_metadata, model_id)
    master_version = Int(loaded_master.metadata.master_version)
    master_superstep = Int(loaded_master.metadata.backend_updates)
    snapshot_version = Int(snapshot_metadata["snapshot_version"])
    sparse_step = adapter.provider.sparse_runtime.bank_optimizers[1].global_step
    sparse_step < typemax(UInt64) || error("sparse checkpoint step cannot form lineage")
    sparse_version = sparse_step + UInt64(1)
    return V2LineageToken(
        model_id=String(model_id),
        master_version=master_version,
        snapshot_version=snapshot_version,
        master_superstep=master_superstep,
        sparse_bank_version=sparse_version,
        sparse_index_version=sparse_version,
        snapshot_sha256=adapter.input_bindings["snapshot_metadata"],
        sparse_bank_sha256=adapter.input_bindings["sparse_checkpoint"],
        sparse_index_sha256=adapter.sparse_index_sha256,
    )
end

function _run_overlapped!(runner, items, slot_count)
    results = V2PipelineResult[]
    submitted = 0
    completed = 0
    while submitted < length(items)
        if submitted - completed >= slot_count
            result = take_v2_result!(runner)
            result.success || error("overlapped pipeline item failed: $(result.error)")
            push!(results, result)
            completed += 1
        end
        item = items[submitted + 1]
        submit_v2!(
            runner, item;
            candidate_count=item.candidate_count,
            workload_id=item.workload_id,
            workload_payload_sha256=item.expected_payload_sha256,
            workload_order=item.order,
        )
        submitted += 1
    end
    while completed < submitted
        result = take_v2_result!(runner)
        result.success || error("overlapped pipeline item failed: $(result.error)")
        push!(results, result)
        completed += 1
    end
    sort!(results; by=result -> result.workload_order)
    return results
end

function _run_phase!(config, providers, hooks, lineage, items)
    guard = V2PhaseSequenceGuard()
    results = V2PipelineResult[]
    for (sequence, item) in enumerate(items)
        result = run_phase_separated_v2!(
            config, providers, hooks, lineage, item;
            guard,
            candidate_count=item.candidate_count,
            workload_id=item.workload_id,
            workload_payload_sha256=item.expected_payload_sha256,
            workload_order=item.order,
            sequence=sequence,
            slot_id=1,
            slot_generation=sequence,
        )
        result.success || error("phase-separated item failed: $(result.error)")
        push!(results, result)
    end
    return results
end

function _sparse_payload(result::V2PipelineResult)
    outcome = result.output::LiveStageOutcome
    outcome.stage == :sparse || error("terminal live output is not sparse")
    return outcome.payload::LiveSparsePayload
end

function _validate_paired_results!(overlapped, phase)
    length(overlapped) == length(phase) || error("paired result counts differ")
    for (left, right) in zip(overlapped, phase)
        left.workload_order == right.workload_order &&
            left.workload_id == right.workload_id &&
            left.workload_payload_sha256 == right.workload_payload_sha256 &&
            left.candidate_count == right.candidate_count || error(
                "paired result workload identities differ",
            )
        a = _sparse_payload(left)
        b = _sparse_payload(right)
        a.hashed.hashstem_output_sha256 == b.hashed.hashstem_output_sha256 &&
            a.output_sha256 == b.output_sha256 && a.route_sha256 == b.route_sha256 &&
            a.top1_index == b.top1_index &&
            a.top1_action_sha256 == b.top1_action_sha256 || error(
                "overlap/phase physical results differ",
            )
    end
    return true
end

function _validate_backend_schedule!(results, hash_backend::Symbol)
    for result in results
        result.hash_fallback_used && error("live production result used HashStem fallback")
        successful = filter(
            receipt -> receipt.stage == :hashstem && receipt.success,
            result.receipts,
        )
        length(successful) == 1 || error("result lacks one successful HashStem receipt")
        observed = only(successful).backend
        expected = hash_backend == :NPU ?
            (result.candidate_count == 16 ? :NPU : :CPU_TAIL) : :CPU
        observed == expected || error(
            "HashStem backend schedule drift: expected $expected observed $observed",
        )
    end
    return true
end

function _lineage_dict(lineage)
    return Dict(
        "model_id" => lineage.model_id,
        "master_version" => lineage.master_version,
        "snapshot_version" => lineage.snapshot_version,
        "master_superstep" => lineage.master_superstep,
        "sparse_bank_version" => lineage.sparse_bank_version,
        "sparse_index_version" => lineage.sparse_index_version,
        "snapshot_sha256" => lineage.snapshot_sha256,
        "sparse_bank_sha256" => lineage.sparse_bank_sha256,
        "sparse_index_sha256" => lineage.sparse_index_sha256,
    )
end

function _receipt_dict(receipt)
    return Dict(
        "stage" => String(receipt.stage), "worker_role" => String(receipt.worker_role),
        "backend" => String(receipt.backend), "slot_id" => receipt.slot_id,
        "slot_generation" => receipt.slot_generation, "sequence" => receipt.sequence,
        "workload_order" => receipt.workload_order,
        "workload_id" => receipt.workload_id,
        "workload_payload_sha256" => receipt.workload_payload_sha256,
        "candidate_count" => receipt.candidate_count,
        "queue_enter_qpc" => receipt.queue_enter_qpc, "begin_qpc" => receipt.begin_qpc,
        "end_qpc" => receipt.end_qpc, "qpc_frequency" => receipt.qpc_frequency,
        "julia_thread_id" => receipt.julia_thread_id,
        "runtime_kind" => String(receipt.runtime_kind),
        "provider_completion_kind" => String(receipt.provider_completion_kind),
        "provider_binding_sha256" => receipt.provider_binding_sha256,
        "provider_synchronized" => receipt.provider_synchronized,
        "provider_readback_complete" => receipt.provider_readback_complete,
        "success" => receipt.success, "error" => receipt.error,
    )
end

function _binding_dict(binding)
    return Dict(
        "worker_role" => String(binding.worker_role),
        "requested_cpu_set_ids" => binding.requested_cpu_set_ids,
        "readback_cpu_set_ids" => binding.readback_cpu_set_ids,
        "topology_sha256" => binding.topology_sha256,
        "julia_thread_id" => binding.julia_thread_id,
        "process_id" => binding.process_id, "os_thread_id" => binding.os_thread_id,
        "apply_begin_qpc" => binding.apply_begin_qpc,
        "apply_end_qpc" => binding.apply_end_qpc,
        "qpc_frequency" => binding.qpc_frequency,
        "runtime_kind" => String(binding.runtime_kind), "status" => binding.status,
    )
end

function _result_dict(result, execution_kind)
    sparse = _sparse_payload(result)
    return Dict{String,Any}(
        "schema" => "heterogeneous-265k-pipeline-v2-live-item-v1",
        "execution_kind" => execution_kind,
        "success" => result.success,
        "slot_id" => result.slot_id, "slot_generation" => result.slot_generation,
        "sequence" => result.sequence, "workload_order" => result.workload_order,
        "workload_id" => result.workload_id,
        "workload_payload_sha256" => result.workload_payload_sha256,
        "candidate_count" => result.candidate_count,
        "lineage" => _lineage_dict(result.lineage),
        "hash_fallback_used" => result.hash_fallback_used,
        "admission_mode" => String(result.admission_mode),
        "submitted_qpc" => result.submitted_qpc, "admitted_qpc" => result.admitted_qpc,
        "completed_qpc" => result.completed_qpc, "qpc_frequency" => result.qpc_frequency,
        "runtime_kind" => String(result.runtime_kind),
        "stage_receipts" => [_receipt_dict(receipt) for receipt in result.receipts],
        "worker_bindings" => [_binding_dict(binding) for binding in result.worker_bindings],
        "physical_receipts" => sparse.physical_receipts,
        "hash_backend" => String(sparse.hashed.actual_backend),
        "hash_physical_calls" => sparse.hashed.physical_calls,
        "hashstem_output_sha256" => sparse.hashed.hashstem_output_sha256,
        "host_to_device_bytes" => sparse.hashed.host_to_device_bytes,
        "device_to_host_bytes" => sparse.hashed.device_to_host_bytes,
        "synchronization_count" => sparse.hashed.synchronization_count,
        "output_sha256" => sparse.output_sha256, "route_sha256" => sparse.route_sha256,
        "top1_index" => sparse.top1_index,
        "top1_action_sha256" => sparse.top1_action_sha256,
        "sparse_accounting" => sparse.accounting,
    )
end

function _summary_dict(summary)
    return Dict{String,Any}(
        "schema" => summary.schema,
        "status" => String(summary.status), "runtime_kind" => String(summary.runtime_kind),
        "lineage" => _lineage_dict(summary.lineage), "sample_count" => summary.sample_count,
        "sequences" => summary.sequences, "workload_sequence" => summary.workload_sequence,
        "admission_mode" => String(summary.admission_mode),
        "total_candidates" => summary.total_candidates,
        "qpc_frequency" => summary.qpc_frequency, "process_id" => summary.process_id,
        "provider_binding_sha256" => summary.provider_binding_sha256,
        "end_to_end" => summary.end_to_end,
        "admission_backpressure" => summary.admission_backpressure,
        "stage_service" => summary.stage_service, "stage_queue" => summary.stage_queue,
        "makespan" => summary.makespan,
        "maximum_concurrent_items" => summary.maximum_concurrent_items,
        "maximum_concurrent_stage_work" => summary.maximum_concurrent_stage_work,
        "stage_interval_receipts" => summary.stage_interval_receipts,
        "distinct_stage_overlap_witnesses" => summary.distinct_stage_overlap_witnesses,
        "actual_stage_overlap_observed" => summary.actual_stage_overlap_observed,
        "note" => summary.note,
    )
end

function _verify_bindings!(paths::Dict{String,String}, digests::Dict{String,String})
    for (key, path) in paths
        _sha256_file(path) == digests[key] || error("binding changed during run: $key")
    end
    return true
end

function _fresh_output_directory(path)
    target = abspath(path)
    ispath(target) && throw(ArgumentError("output directory must be fresh: $target"))
    mkpath(target)
    return target
end

function _atomic_json(path, payload)
    return atomic_write_json(path, payload)
end

function _atomic_jsonl(path, records)
    return atomic_write_jsonl(path, records)
end

function run_live(options::Dict{String,String})
    options["--enable-igpu-stem-training"] == "false" || error(
        "iGPU live trainer is unavailable; only explicit false is accepted",
    )
    hash_backend = Symbol(uppercase(options["--hash-backend"]))
    hash_backend in (:CPU, :NPU) || error("hash backend must be CPU or NPU")
    slot_count = parse(Int, options["--slot-count"])
    slot_count in (2, 3) || error("slot count must be 2 or 3")
    overlap_threshold = parse(Float64, options["--overlap-threshold"])
    isfinite(overlap_threshold) && overlap_threshold >= 1.0 || error(
        "overlap threshold must be finite and >=1",
    )
    sparse_variant = Symbol(options["--sparse-variant"])
    sparse_variant in (:k64, :k128, :k256) || error(
        "sparse variant must be k64, k128, or k256",
    )
    npu_gate_path = options["--npu-gate"]
    npu_gate_sha256 = options["--npu-gate-sha256"]
    npu_gate_decision = nothing
    if hash_backend == :NPU
        sparse_variant == :k128 || error(
            "the measured NPU adoption gate authorizes only sparse variant k128",
        )
        npu_gate_path = abspath(npu_gate_path)
        npu_gate_sha256 = _require_sha256(npu_gate_sha256, "NPU gate SHA-256")
        npu_gate_decision = _validate_npu_gate!(npu_gate_path, npu_gate_sha256)
    else
        npu_gate_path == "NONE" && npu_gate_sha256 == repeat("0", 64) || error(
            "CPU mode requires --npu-gate NONE and the all-zero disabled digest",
        )
    end

    output_directory = _fresh_output_directory(options["--output-directory"])
    source_bindings, source_paths = _source_closure(options)
    contract_path = joinpath(@__DIR__, "post_idle_benchmark_contract.json")
    provider_path = joinpath(@__DIR__, "post_idle_production_provider.jl")
    runner_path = joinpath(@__DIR__, "heterogeneous_pipeline_runner_v2.jl")
    input_bindings = Dict(
        "workload" => _require_sha256(options["--workload-sha256"], "workload SHA-256"),
        "workload_manifest" => _require_sha256(
            options["--workload-manifest-sha256"], "workload manifest SHA-256",
        ),
        "dataset_manifest" => _require_sha256(
            options["--dataset-manifest-sha256"], "dataset manifest SHA-256",
        ),
        "sparse_checkpoint" => _require_sha256(
            options["--sparse-checkpoint-sha256"], "sparse checkpoint SHA-256",
        ),
        "master_manifest" => _require_sha256(
            options["--master-manifest-sha256"], "master manifest SHA-256",
        ),
        "snapshot_metadata" => _require_sha256(
            options["--snapshot-metadata-sha256"], "snapshot metadata SHA-256",
        ),
        "system_contract" => _require_sha256(
            options["--system-contract-sha256"], "system contract SHA-256",
        ),
        "post_idle_contract" => _require_sha256(
            options["--post-idle-contract-sha256"], "post-idle contract SHA-256",
        ),
        "source_manifest" => _require_sha256(
            options["--source-manifest-sha256"], "source manifest SHA-256",
        ),
        "npu_gate" => npu_gate_sha256,
    )
    input_paths = Dict(
        "workload" => abspath(options["--workload"]),
        "workload_manifest" => abspath(options["--workload-manifest"]),
        "dataset_manifest" => abspath(options["--dataset-manifest"]),
        "sparse_checkpoint" => abspath(options["--sparse-checkpoint"]),
        "master_manifest" => abspath(options["--master-manifest"]),
        "snapshot_metadata" => abspath(options["--snapshot-metadata"]),
        "system_contract" => abspath(options["--system-contract"]),
        "post_idle_contract" => contract_path,
        "source_manifest" => abspath(options["--source-manifest"]),
    )
    hash_backend == :NPU && (input_paths["npu_gate"] = npu_gate_path)
    _verify_bindings!(input_paths, input_bindings)
    input_paths["post_idle_contract"] == abspath(contract_path) || error(
        "post-idle contract path drift",
    )
    source_paths["pipeline_runner_v2"] == abspath(runner_path) || error(
        "runner path drift",
    )
    source_paths["post_idle_provider"] == abspath(provider_path) || error(
        "provider path drift",
    )
    basename(options["--master-manifest"]) == "checkpoint_manifest.json" || error(
        "master input must be checkpoint_manifest.json",
    )
    basename(options["--snapshot-metadata"]) == "snapshot_metadata.json" || error(
        "snapshot input must be snapshot_metadata.json",
    )

    config_bindings = Dict(
        "model_id" => options["--model-id"],
        "hash_backend" => String(hash_backend),
        "sparse_variant" => String(sparse_variant),
        "slot_count" => string(slot_count),
        "overlap_threshold" => repr(overlap_threshold),
        "igpu_stem_training" => "false",
        "npu_gate" => hash_backend == :NPU ? "ADOPTED_MEASURED_BYTE_PINNED" : "DISABLED",
        "julia_version" => string(VERSION),
    )
    provider_binding = _canonical_binding_sha256(
        source_bindings, input_bindings, config_bindings,
    )
    cell_id = hash_backend == :NPU ? "H1_npu_hashstem_cpu_sparse" :
        "H0_cpu_hashstem_cpu_sparse"
    config = RunConfig(
        cell_id=cell_id,
        repetition=1,
        contract=_binding("post-idle contract", contract_path,
            input_bindings["post_idle_contract"]),
        input=_binding("workload", input_paths["workload"], input_bindings["workload"]),
        sparse_checkpoint=_binding("sparse checkpoint", input_paths["sparse_checkpoint"],
            input_bindings["sparse_checkpoint"]),
        master=_binding("HashStem master manifest", input_paths["master_manifest"],
            input_bindings["master_manifest"]),
        snapshot=_binding("HashStem snapshot metadata", input_paths["snapshot_metadata"],
            input_bindings["snapshot_metadata"]),
        system_contract=_binding("system contract", input_paths["system_contract"],
            input_bindings["system_contract"]),
        provider_source=_binding("post-idle provider", provider_path,
            source_bindings["post_idle_provider"]),
        source_manifest=_binding("source manifest", input_paths["source_manifest"],
            input_bindings["source_manifest"]),
        output_directory=output_directory,
        warmup_candidate_sets=0,
        timed_candidate_sets=0,
    )
    adapter, loaded_master, snapshot_metadata = _load_live_adapter(
        config,
        input_paths["workload_manifest"],
        input_bindings["workload_manifest"],
        input_paths["dataset_manifest"],
        input_bindings["dataset_manifest"],
        sparse_variant,
        hash_backend,
        source_bindings,
        input_bindings,
        provider_binding,
    )
    hash_backend == :NPU && _bind_npu_gate_to_live!(
        npu_gate_decision, input_bindings, loaded_master, snapshot_metadata,
    )
    lineage = _lineage(adapter, loaded_master, snapshot_metadata, options["--model-id"])
    providers = build_concrete_live_stage_providers(adapter, provider_binding)
    hooks = windows_v2_runtime_hooks()
    pipeline_config = V2PipelineConfig(
        slot_count=slot_count,
        hash_backend=hash_backend,
        allow_cpu_hash_fallback=false,
        enable_igpu_stem_training=false,
        npu_gate=(hash_backend == :NPU ? :ADOPTED_MEASURED : :DISABLED),
        igpu_gate=:DISABLED,
        execution_mode=:overlapped,
        overlap_slowdown_threshold=overlap_threshold,
    )
    runner = start_v2_pipeline!(pipeline_config, providers, lineage; hooks)
    overlapped = V2PipelineResult[]
    phase = V2PipelineResult[]
    comparator = nothing
    stopped = false
    try
        overlapped = _run_overlapped!(runner, adapter.workload_items, slot_count)
        phase = _run_phase!(pipeline_config, providers, hooks, lineage, adapter.workload_items)
        _validate_backend_schedule!(overlapped, hash_backend)
        _validate_backend_schedule!(phase, hash_backend)
        _validate_paired_results!(overlapped, phase)
        comparator = observe_overlap_comparator!(
            runner; overlapped_results=overlapped, phase_separated_results=phase,
        )
    finally
        stop_v2_pipeline!(runner)
        stopped = true
    end
    stopped || error("pipeline runner did not stop")
    overlap_summary = summarize_v2_receipts(overlapped; require_production=true)
    phase_summary = summarize_v2_receipts(phase; require_production=true)
    _verify_bindings!(source_paths, source_bindings)
    _verify_bindings!(input_paths, input_bindings)
    live_adapter_identity(adapter)

    records = Any[]
    append!(records, (_result_dict(result, "overlapped") for result in overlapped))
    append!(records, (_result_dict(result, "phase_separated") for result in phase))
    records_path = _atomic_jsonl(joinpath(output_directory, "receipts.jsonl"), records)
    records_sha256 = _sha256_file(records_path)
    status = comparator.decision == :KEEP_OVERLAP ?
        "MEASURED_KEEP_OVERLAP_PENDING_INDEPENDENT_VALIDATION" :
        "MEASURED_PHASE_SEPARATED_FAILSAFE_PENDING_INDEPENDENT_VALIDATION"
    summary = Dict{String,Any}(
        "schema" => LIVE_SCHEMA,
        "status" => status,
        "adoption_allowed" => false,
        "process_id" => getpid(),
        "single_process" => true,
        "child_processes_launched" => false,
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        "lineage" => _lineage_dict(lineage),
        "provider_binding_sha256" => provider_binding,
        "source_bindings" => source_bindings,
        "input_bindings" => input_bindings,
        "config_bindings" => config_bindings,
        "workload_sequence" => [Dict(
            "order" => item.order, "workload_id" => item.workload_id,
            "workload_payload_sha256" => item.expected_payload_sha256,
            "set_index" => item.set_index, "candidate_start" => item.candidate_start,
            "candidate_count" => item.candidate_count,
        ) for item in adapter.workload_items],
        "sparse_variant" => String(adapter.sparse_variant),
        "active_counts" => collect(adapter.active_counts),
        "hash_backend" => String(hash_backend),
        "igpu_stem_training" => Dict(
            "enabled" => false,
            "status" => "DISABLED_NO_PRODUCTION_LIVE_ADAPTER",
        ),
        "provider_metadata" => adapter.provider.metadata,
        "overlapped" => _summary_dict(overlap_summary),
        "phase_separated" => _summary_dict(phase_summary),
        "overlap_comparator" => Dict(string(key) => value for (key, value) in pairs(comparator)),
        "paired_physical_result_identity" => true,
        "receipts_file" => basename(records_path),
        "receipts_sha256" => records_sha256,
        "note" => "Application QPC/CPU-set receipts only; no strength, ETW residency, IMC, or adoption claim",
    )
    summary_path = _atomic_json(joinpath(output_directory, "summary.json"), summary)
    return summary_path
end

function main(arguments=ARGS)
    options = _parse_cli(arguments)
    output = abspath(options["--output-directory"])
    try
        summary = run_live(options)
        println("LIVE_PIPELINE_ARTIFACT=", summary)
        return 0
    catch error_value
        if isdir(output) && !isfile(joinpath(output, "summary.json"))
            failure = Dict(
                "schema" => LIVE_SCHEMA,
                "status" => "FAILED_CLOSED",
                "adoption_allowed" => false,
                "process_id" => getpid(),
                "timestamp_utc" => Dates.format(Dates.now(Dates.UTC),
                    dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
                "error" => sprint(showerror, error_value),
            )
            try
                _atomic_json(joinpath(output, "summary.json"), failure)
            catch
            end
        end
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
        return 2
    end
end

end # module BeatFirstHeterogeneousPipelineLiveCLI

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(BeatFirstHeterogeneousPipelineLiveCLI.main(ARGS))
end
