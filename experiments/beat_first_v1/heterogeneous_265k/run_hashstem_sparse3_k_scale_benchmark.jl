#!/usr/bin/env julia

"""Direct, fail-closed CPU-k64 / CPU-k128 / NPU-k128 production gate.

This is deliberately one executable and one process.  PythonCall embeds the
OpenVINO host API in this Julia process; it never launches a Python child.  An
ordinary 3-layer teacher checkpoint supplies one exact sparse bank.  Two
inference-only views rebuild only the active-width-specific WTA indexes so k64
and k128 use byte-identical source weights and optimizer clocks.
"""

using JLD2
using JSON3
using PythonCall
using SHA

module HashStemSparse3DirectRuntime

using JSON3
using SHA

if !isdefined(Main, :SparseDynamic3Layer)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),
    )
end
using Main.SparseDynamic3Layer

# The full heterogeneous module imports training-only dependencies (including
# NPZ).  This direct inference gate deliberately loads only its frozen
# HashStem/bridge/gate source closure.  The hooks type is part of the bridge's
# exported backward API even though this inference runner never calls it.
const UNEXECUTED_STATIC_ONLY = "UNEXECUTED_STATIC_ONLY"
const VALIDATED_RUNTIME = "VALIDATED_RUNTIME"

_sha256_file(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

include("hashstem.jl")

Base.@kwdef struct HashStemLossHooks
    q1::Matrix{Float32}
    q2::Matrix{Float32}
    q3::Matrix{Float32}
    context::Matrix{Float32}
    next_hold::Matrix{Float32}
    auxiliary_target::Matrix{Float32}
    auxiliary_mask::Matrix{Float32}
end

function _validate_loss_hooks(hooks::HashStemLossHooks, batch::Int)
    for (name, value, width) in (
        (:q1, hooks.q1, 64),
        (:q2, hooks.q2, 64),
        (:q3, hooks.q3, 64),
        (:context, hooks.context, 22),
        (:next_hold, hooks.next_hold, 42),
        (:auxiliary_target, hooks.auxiliary_target, 10),
        (:auxiliary_mask, hooks.auxiliary_mask, 10),
    )
        size(value) == (batch, width) || throw(DimensionMismatch(
            "$name must be $(batch)x$(width)",
        ))
        all(isfinite, value) || throw(ArgumentError("$name is non-finite"))
    end
    all(value -> value == 0.0f0 || value == 1.0f0, hooks.auxiliary_mask) ||
        throw(ArgumentError("auxiliary_mask must contain only 0f0/1f0"))
    return hooks
end

include("sparse_hashstem_bridge.jl")
include("hashstem_sparse3_k_scale_gate.jl")
include("sparse_hashstem_named_inference_view.jl")

end # module HashStemSparse3DirectRuntime

const H = HashStemSparse3DirectRuntime
const S = Main.SparseDynamic3Layer

const BENCHMARK_SCHEMA = "hashstem-sparse3-direct-k-scale-benchmark-v1"
const WORKLOAD_SCHEMA = "hashstem-sparse3-train-workload-v1"
const RAW_RECORD_SCHEMA = "hashstem-sparse3-direct-k-scale-raw-v1"
const DECISION_SCHEMA = "hashstem-sparse3-direct-k-scale-decision-v1"
const CELL_IDS = ("cpu_k64", "cpu_k128", "npu_k128")
const CELL_ORDERS = (
    ("cpu_k64", "cpu_k128", "npu_k128"),
    ("cpu_k64", "npu_k128", "cpu_k128"),
    ("cpu_k128", "npu_k128", "cpu_k64"),
    ("cpu_k128", "cpu_k64", "npu_k128"),
    ("npu_k128", "cpu_k64", "cpu_k128"),
    ("npu_k128", "cpu_k128", "cpu_k64"),
)
const FORBIDDEN_VALIDATION_SEEDS = Set(8001:8008)
const FORBIDDEN_SEALED_SEEDS = Set(91001:91032)
const MAXIMUM_SPARSE_Q_ABSOLUTE_ERROR = 1.0e-2
const ACTION_MARGIN_ABSOLUTE_ERROR_FLOOR = 1.0e-2
const ACTION_MARGIN_RELATIVE_ERROR = 5.0e-2
const STAGES = (
    "packing", "hashstem", "routing", "gather", "sparse", "head", "sync",
    "end_to_end",
)

struct ArtifactBinding
    label::String
    path::String
    expected_sha256::String
    initial_sha256::String
    bytes::Int64
end

struct OpenVINOExecutors
    ov::Any
    np::Any
    cpu_fixed::Any
    cpu_dynamic::Any
    npu_fixed::Any
    devices::Vector{String}
    cpu_execution_devices::Vector{String}
    npu_execution_devices::Vector{String}
end

mutable struct CellContext
    cell_id::String
    variant::String
    view::H.SparseNamedInferenceView
    workspace::S.ThreeLayerWorkspace
    records::Vector{Dict{String,Any}}
    reference_outputs::Union{Nothing,Matrix{Float32}}
    reference_routes::Union{Nothing,Vector{NTuple{3,Vector{Int32}}}}
    maximum_hashstem_absolute_error::Float64
end

function sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("artifact is not a file: $source"))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function valid_sha256(value::AbstractString)
    ncodeunits(value) == 64 || return false
    return all(character -> character in "0123456789abcdef", lowercase(value))
end

function ArtifactBinding(label, path, expected_sha256)
    canonical = abspath(String(path))
    expected = lowercase(String(expected_sha256))
    valid_sha256(expected) || throw(ArgumentError("$label expected SHA-256 is invalid"))
    observed = sha256_file(canonical)
    observed == expected || error(
        "$label SHA-256 mismatch: expected $expected, observed $observed",
    )
    return ArtifactBinding(String(label), canonical, expected, observed, filesize(canonical))
end

function verify(binding::ArtifactBinding)
    isfile(binding.path) || error("$(binding.label) disappeared during the benchmark")
    filesize(binding.path) == binding.bytes || error(
        "$(binding.label) byte length changed during the benchmark",
    )
    sha256_file(binding.path) == binding.initial_sha256 || error(
        "$(binding.label) bytes changed during the benchmark",
    )
    return nothing
end

function parse_cli(arguments::Vector{String})
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || throw(ArgumentError("unexpected positional argument $key"))
        index < length(arguments) || throw(ArgumentError("missing value after $key"))
        haskey(options, key) && throw(ArgumentError("duplicate option $key"))
        options[key] = arguments[index + 1]
        index += 2
    end
    return options
end

function required(options, key)
    haskey(options, key) || throw(ArgumentError("missing required option $key"))
    isempty(options[key]) && throw(ArgumentError("$key must not be empty"))
    return options[key]
end

binding(options, label, path_key, sha_key) = ArtifactBinding(
    label, required(options, path_key), required(options, sha_key),
)

function property_value(value, name::Symbol)
    if value isa AbstractDict
        haskey(value, String(name)) && return value[String(name)]
        haskey(value, name) && return value[name]
        throw(KeyError(name))
    end
    hasproperty(value, name) || throw(ArgumentError("workload item is missing $name"))
    return getproperty(value, name)
end

string_dict(value) = Dict{String,Any}(string(key) => item for (key, item) in pairs(value))

function array_sha256(values::AbstractArray{T}) where {T<:Union{Float32,Int32,UInt32}}
    Base.ENDIAN_BOM == 0x04030201 || error("little-endian host is required")
    return bytes2hex(SHA.sha256(collect(reinterpret(UInt8, collect(vec(values))))))
end

function hashstem_normalization_sha256(weights::H.HashStemWeights)
    Base.ENDIAN_BOM == 0x04030201 || error("little-endian host is required")
    bytes = vcat(
        collect(reinterpret(UInt8, weights.input_mean)),
        collect(reinterpret(UInt8, weights.input_inv_std)),
    )
    return bytes2hex(SHA.sha256(bytes))
end

function canonical_string_digest(values::AbstractVector{<:AbstractString})
    io = IOBuffer()
    write(io, htol(UInt32(length(values))))
    for value in values
        bytes = Vector{UInt8}(codeunits(String(value)))
        write(io, htol(UInt32(length(bytes))))
        write(io, bytes)
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function route_digest(routes::Vector{NTuple{3,Vector{Int32}}})
    io = IOBuffer()
    write(io, htol(UInt32(length(routes))))
    for route in routes, layer_id in 1:3
        write(io, htol(UInt32(length(route[layer_id]))))
        for neuron_id in route[layer_id]
            write(io, htol(neuron_id))
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function parse_snapshot(
    metadata_binding::ArtifactBinding,
    weights_binding::ArtifactBinding,
    fixed_ir::ArtifactBinding,
    fixed_bin::ArtifactBinding,
    dynamic_ir::ArtifactBinding,
    dynamic_bin::ArtifactBinding,
)
    metadata = JSON3.read(read(metadata_binding.path, String), Dict{String,Any})
    String(get(metadata, "schema", "")) == H.HASHSTEM_SCHEMA || error(
        "HashStem snapshot schema mismatch",
    )
    Int(get(metadata, "snapshot_version", 0)) > 0 || error(
        "HashStem snapshot version must be positive",
    )
    Int(get(metadata, "fixed_batch", 0)) == H.HASHSTEM_BATCH || error(
        "HashStem fixed batch differs from 16",
    )
    Int(get(metadata, "input_features", 0)) == H.HASHSTEM_INPUT_FEATURES || error(
        "HashStem input width mismatch",
    )
    Int(get(metadata, "output_features", 0)) == H.HASHSTEM_OUTPUT_FEATURES || error(
        "HashStem output width mismatch",
    )
    uppercase(String(get(metadata, "device", ""))) == "NPU" || error(
        "snapshot metadata device must be exact NPU",
    )
    expected = (
        ("weights_sha256", weights_binding.initial_sha256),
        ("fixed_xml_sha256", fixed_ir.initial_sha256),
        ("fixed_bin_sha256", fixed_bin.initial_sha256),
        ("dynamic_xml_sha256", dynamic_ir.initial_sha256),
        ("dynamic_bin_sha256", dynamic_bin.initial_sha256),
    )
    for (field, digest) in expected
        lowercase(String(get(metadata, field, ""))) == digest || error(
            "snapshot $field does not bind the supplied artifact",
        )
    end
    lowercase(String(get(metadata, "xml_sha256", ""))) == fixed_ir.initial_sha256 ||
        error("snapshot fixed XML compatibility alias mismatch")
    lowercase(String(get(metadata, "bin_sha256", ""))) == fixed_bin.initial_sha256 ||
        error("snapshot fixed BIN compatibility alias mismatch")
    splitext(fixed_ir.path)[1] * ".bin" == fixed_bin.path || error(
        "fixed XML and BIN must be same-stem siblings",
    )
    splitext(dynamic_ir.path)[1] * ".bin" == dynamic_bin.path || error(
        "dynamic XML and BIN must be same-stem siblings",
    )
    np = pyimport("numpy")
    archive = np.load(weights_binding.path, allow_pickle=false)
    weights = try
        expected_keys = Set((
            "conv3",
            "conv3_bias",
            "depthwise5x1",
            "depthwise_bias",
            "pointwise",
            "pointwise_bias",
            "dense",
            "dense_bias",
            "input_mean",
            "input_inv_std",
        ))
        observed_keys = Set(String.(pyconvert(Vector{String}, archive.files)))
        observed_keys == expected_keys || error("HashStem NPZ key set mismatch")
        host_array(name, ::Type{T}) where {T<:AbstractArray} = pyconvert(
            T, np.array(archive[name], dtype=np.float32, copy=true),
        )
        H.HashStemWeights(
            host_array("conv3", Array{Float32,4}),
            host_array("conv3_bias", Vector{Float32}),
            host_array("depthwise5x1", Matrix{Float32}),
            host_array("depthwise_bias", Vector{Float32}),
            host_array("pointwise", Matrix{Float32}),
            host_array("pointwise_bias", Vector{Float32}),
            host_array("dense", Matrix{Float32}),
            host_array("dense_bias", Vector{Float32}),
            host_array("input_mean", Vector{Float32}),
            host_array("input_inv_std", Vector{Float32}),
        )
    finally
        archive.close()
    end
    H.validate_hashstem_weights(weights)
    hashstem_normalization_sha256(weights) ==
        lowercase(String(metadata["normalization_sha256"])) ||
        error("HashStem normalization arrays differ from snapshot metadata")
    return metadata, weights
end

function load_workload(binding::ArtifactBinding, teacher_metadata)
    payload = JLD2.load(binding.path)
    get(payload, "schema", nothing) == WORKLOAD_SCHEMA || error(
        "train workload schema mismatch",
    )
    haskey(payload, "metadata") || error("train workload metadata is missing")
    haskey(payload, "candidate_sets") || error("train workload candidate_sets are missing")
    metadata = string_dict(payload["metadata"])
    String(get(metadata, "split", "")) == "teacher_v3_train" || error(
        "workload must be teacher_v3_train, not development/validation/sealed",
    )
    Bool(get(metadata, "training_only", false)) || error(
        "workload does not declare training_only=true",
    )
    Bool(get(metadata, "reserved_seed_free", false)) || error(
        "workload does not certify reserved-seed exclusion",
    )
    manifest_digest = lowercase(String(get(metadata, "dataset_manifest_sha256", "")))
    valid_sha256(manifest_digest) || error("workload dataset manifest digest is invalid")
    lowercase(String(get(teacher_metadata, "dataset_manifest_sha256", ""))) ==
        manifest_digest || error("teacher checkpoint/workload dataset manifest mismatch")

    sets = Vector{Any}(payload["candidate_sets"])
    length(sets) >= 2 || error("workload requires at least two candidate states")
    state_ids = String[]
    seeds = Int[]
    action_counts = Int[]
    teacher_q = Vector{Vector{Float32}}()
    action_digests = Vector{Vector{String}}()
    for (position, set) in pairs(sets)
        state_id = String(property_value(set, :state_id))
        isempty(state_id) && error("workload state $position has an empty state_id")
        push!(state_ids, state_id)
        seed = Int(property_value(set, :seed))
        seed >= 0 || error("workload seed must be non-negative")
        seed in FORBIDDEN_VALIDATION_SEEDS && error("validation seed $seed is forbidden")
        seed in FORBIDDEN_SEALED_SEEDS && error("sealed seed $seed is forbidden")
        push!(seeds, seed)
        input = property_value(set, :input)
        count = S.BeatFirstSparseFeatures.validate_candidate_feature_input(input)
        count > 1 || error("workload state $state_id has fewer than two actions")
        push!(action_counts, count)
        digests = String.(property_value(set, :action_digests))
        length(digests) == count || error("action digest count mismatch for $state_id")
        all(valid_sha256, digests) || error("invalid action digest for $state_id")
        push!(action_digests, digests)
        teacher = Float32.(property_value(set, :teacher_q))
        length(teacher) == count || error("teacher-Q count mismatch for $state_id")
        all(isfinite, teacher) || error("non-finite teacher Q for $state_id")
        push!(teacher_q, teacher)
    end
    length(unique(state_ids)) == length(state_ids) || error("workload state IDs repeat")
    issorted(state_ids) || error("workload state IDs must be in canonical sorted order")
    expected_order = lowercase(String(get(metadata, "state_order_sha256", "")))
    valid_sha256(expected_order) || error("workload state-order digest is invalid")
    canonical_string_digest(state_ids) == expected_order || error(
        "workload state order differs from its metadata digest",
    )
    total_candidates = sum(action_counts)
    total_candidates >= H.HASHSTEM_BATCH || error(
        "workload must contain at least one complete batch-16 HashStem call",
    )
    return (;
        metadata,
        sets,
        state_ids,
        seeds,
        action_counts,
        teacher_q,
        action_digests,
        total_candidates,
        state_order_sha256=expected_order,
    )
end

function pack_workload(workload)
    packed = Matrix{Float32}(undef, workload.total_candidates, H.HASHSTEM_INPUT_FEATURES)
    offsets = Vector{UnitRange{Int}}(undef, length(workload.sets))
    position = 1
    for (set_index, set) in pairs(workload.sets)
        input = property_value(set, :input)
        count = workload.action_counts[set_index]
        rows = position:(position + count - 1)
        local_packed = Matrix{Float32}(undef, count, H.HASHSTEM_INPUT_FEATURES)
        H.pack_hashstem_input!(local_packed, input, collect(1:count))
        packed[rows, :] .= local_packed
        offsets[set_index] = rows
        position += count
    end
    position == workload.total_candidates + 1 || error("workload packing lost rows")
    all(isfinite, packed) || error("packed workload is non-finite")
    return packed, offsets
end

function execution_devices(compiled, requested::String)
    values = String.(pyconvert(Vector{String}, compiled.get_property("EXECUTION_DEVICES")))
    uppercase.(values) == [requested] || error(
        "OpenVINO requested $requested but compiled for $(values)",
    )
    return values
end

function compile_openvino(fixed_ir::String, dynamic_ir::String)
    ov = pyimport("openvino")
    np = pyimport("numpy")
    core = ov.Core()
    devices = String.(pyconvert(Vector{String}, core.available_devices))
    "CPU" in uppercase.(devices) || error("OpenVINO did not enumerate exact CPU")
    "NPU" in uppercase.(devices) || error("OpenVINO did not enumerate exact NPU")
    cpu_fixed_model = core.compile_model(fixed_ir, "CPU")
    cpu_dynamic_model = core.compile_model(dynamic_ir, "CPU")
    npu_fixed_model = core.compile_model(fixed_ir, "NPU")
    cpu_fixed_devices = execution_devices(cpu_fixed_model, "CPU")
    execution_devices(cpu_dynamic_model, "CPU")
    npu_devices = execution_devices(npu_fixed_model, "NPU")
    return OpenVINOExecutors(
        ov,
        np,
        cpu_fixed_model.create_infer_request(),
        cpu_dynamic_model.create_infer_request(),
        npu_fixed_model.create_infer_request(),
        devices,
        cpu_fixed_devices,
        npu_devices,
    )
end

function elapsed_ns(operation::Function)
    started = time_ns()
    value = operation()
    elapsed = Int64(time_ns() - started)
    return value, max(elapsed, Int64(1))
end

function infer_request!(executors::OpenVINOExecutors, request, chunk::Matrix{Float32})
    py_input = executors.np.ascontiguousarray(
        PythonCall.Py(chunk), dtype=executors.np.float32,
    )
    tensor = executors.ov.Tensor(py_input)
    request.set_input_tensor(tensor)
    request.start_async()
    _, sync_ns = elapsed_ns(() -> request.wait())
    py_output = executors.np.array(
        request.get_output_tensor(0).data,
        dtype=executors.np.float32,
        copy=true,
    )
    output = pyconvert(Matrix{Float32}, py_output)
    size(output) == (size(chunk, 1), H.HASHSTEM_OUTPUT_FEATURES) || error(
        "OpenVINO output shape mismatch",
    )
    all(isfinite, output) || error("OpenVINO output is non-finite")
    return output, sync_ns
end

function hashstem_openvino!(
    executors::OpenVINOExecutors,
    packed::Matrix{Float32},
    device::String,
)
    rows = size(packed, 1)
    output = Matrix{Float32}(undef, rows, H.HASHSTEM_OUTPUT_FEATURES)
    position = 1
    sync_ns = Int64(0)
    full_calls = 0
    while position + H.HASHSTEM_BATCH - 1 <= rows
        last = position + H.HASHSTEM_BATCH - 1
        chunk = Matrix{Float32}(@view packed[position:last, :])
        request = device == "NPU" ? executors.npu_fixed : executors.cpu_fixed
        chunk_output, waited = infer_request!(executors, request, chunk)
        output[position:last, :] .= chunk_output
        sync_ns += waited
        full_calls += 1
        position = last + 1
    end
    tail_rows = rows - position + 1
    tail_calls = 0
    if tail_rows > 0
        chunk = Matrix{Float32}(@view packed[position:rows, :])
        chunk_output, waited = infer_request!(executors, executors.cpu_dynamic, chunk)
        output[position:rows, :] .= chunk_output
        sync_ns += waited
        tail_calls = 1
    end
    return output, sync_ns, full_calls, tail_calls, max(tail_rows, 0)
end

function stable_order(scores)
    all(isfinite, scores) || error("ranking scores are non-finite")
    return sortperm(scores; rev=true, alg=MergeSort)
end

function ndcg(prediction, teacher)
    count = length(teacher)
    teacher_order = stable_order(teacher)
    relevance = zeros(Float64, count)
    for (rank, action) in enumerate(teacher_order)
        relevance[action] = count - rank
    end
    prediction_order = stable_order(prediction)
    discount(rank) = 1.0 / log2(rank + 1.0)
    dcg = sum(relevance[action] * discount(rank) for (rank, action) in enumerate(prediction_order))
    idcg = sum((count - rank) * discount(rank) for rank in 1:count)
    return idcg == 0 ? 1.0 : dcg / idcg
end

function pairwise_accuracy(prediction, teacher)
    correct = 0
    compared = 0
    for left in 1:(length(teacher) - 1), right in (left + 1):length(teacher)
        difference = teacher[left] - teacher[right]
        difference == 0 && continue
        correct += (prediction[left] - prediction[right]) * difference > 0
        compared += 1
    end
    return compared == 0 ? 1.0 : correct / compared
end

function quality_metrics(outputs::Matrix{Float32}, workload, offsets)
    top1 = Bool[]
    ndcgs = Float64[]
    pairwise = Float64[]
    top2 = Vector{NTuple{2,Int}}()
    for state in eachindex(offsets)
        prediction = Vector{Float32}(@view outputs[1, offsets[state]])
        teacher = workload.teacher_q[state]
        order = stable_order(prediction)
        teacher_order = stable_order(teacher)
        push!(top1, order[1] == teacher_order[1])
        push!(top2, (order[1], order[2]))
        push!(ndcgs, ndcg(prediction, teacher))
        push!(pairwise, pairwise_accuracy(prediction, teacher))
    end
    return (;
        teacher_top1_agreement=sum(top1) / length(top1),
        teacher_ndcg=sum(ndcgs) / length(ndcgs),
        teacher_pairwise_accuracy=sum(pairwise) / length(pairwise),
        action_top1=[value[1] for value in top2],
        action_top2=top2,
    )
end

function replay_head_timing!(view::H.SparseNamedInferenceView, results)
    total = Int64(0)
    for result in results
        replay, elapsed = elapsed_ns() do
            values = Vector{Float32}(undef, S.OUTPUT_DIM)
            @inbounds for output_id in 1:S.OUTPUT_DIM
                accumulator = view.runtime.model.bias[output_id]
                @simd for hidden in 1:S.LATENT_DIM
                    accumulator = muladd(
                        view.runtime.model.head[output_id, hidden],
                        result.tape.latent[hidden],
                        accumulator,
                    )
                end
                values[output_id] = accumulator
            end
            values
        end
        all(isequal.(replay, result.output)) || error(
            "head replay differs from production route output",
        )
        total += elapsed
    end
    return total
end

function run_cell_once!(
    context::CellContext,
    workload,
    offsets,
    executors::OpenVINOExecutors,
    scalar_oracle::Matrix{Float32},
    expected_packed_sha256::String,
    repetition::Int;
    warmup::Bool,
)
    total_started = time_ns()
    packed, packing_ns = elapsed_ns(() -> pack_workload(workload))
    packed_matrix, observed_offsets = packed
    observed_offsets == offsets || error("workload offsets changed between cells")
    device = context.cell_id == "npu_k128" ? "NPU" : "CPU"
    stem_tuple, hashstem_ns = elapsed_ns() do
        hashstem_openvino!(executors, packed_matrix, device)
    end
    hashstem_output, sync_ns, full_calls, tail_calls, tail_rows = stem_tuple
    inputs, bridge_pack_ns = elapsed_ns() do
        H.three_layer_inputs_from_hashstem(packed_matrix, hashstem_output)
    end
    forward_batch, _ = elapsed_ns() do
        H.route_sparse_named_inputs!(context.view, context.workspace, inputs)
    end
    results = forward_batch.results
    length(results) == workload.total_candidates || error("production route lost candidates")
    outputs = Matrix{Float32}(undef, S.OUTPUT_DIM, workload.total_candidates)
    routes = Vector{NTuple{3,Vector{Int32}}}(undef, workload.total_candidates)
    for candidate in eachindex(results)
        outputs[:, candidate] .= results[candidate].output
        routes[candidate] = ntuple(
            layer_id -> copy(results[candidate].tape.ids[layer_id]), 3,
        )
    end
    end_to_end_ns = max(Int64(time_ns() - total_started), Int64(1))

    # Numerical checks, digests, teacher metrics, and the diagnostic head
    # replay are outside the adoption end-to-end boundary.
    observed_packed_sha256 = array_sha256(packed_matrix)
    observed_packed_sha256 == expected_packed_sha256 || error(
        "cell packed bytes differ from the canonical ordered workload",
    )
    hashstem_error = Float64(maximum(abs.(hashstem_output .- scalar_oracle)))
    context.maximum_hashstem_absolute_error = max(
        context.maximum_hashstem_absolute_error, hashstem_error,
    )
    metrics = quality_metrics(outputs, workload, offsets)
    head_replay_ns = replay_head_timing!(context.view, results)
    routing_ns = Int64(sum(
        sum(result.telemetry.routing_nanoseconds) for result in results;
        init=UInt64(0),
    ))
    gather_ns = Int64(sum(
        sum(result.telemetry.materialization_nanoseconds) for result in results;
        init=UInt64(0),
    ))
    sparse_including_head_ns = Int64(sum(
        result.telemetry.selected_compute_nanoseconds for result in results;
        init=UInt64(0),
    ))
    packing_total_ns = packing_ns + bridge_pack_ns
    stage_ns = Dict{String,Int64}(
        "packing" => packing_total_ns,
        "hashstem" => hashstem_ns,
        "routing" => routing_ns,
        "gather" => gather_ns,
        "sparse" => sparse_including_head_ns,
        "head" => head_replay_ns,
        "sync" => sync_ns,
        "end_to_end" => end_to_end_ns,
    )
    all(stage -> stage_ns[stage] > 0, STAGES) || error("a required timing stage is zero")
    accounted_ns = packing_total_ns + hashstem_ns + routing_ns + gather_ns +
        sparse_including_head_ns
    end_to_end_ns >= accounted_ns || error(
        "production non-overlapping stage durations exceed end-to-end",
    )
    output_digest = array_sha256(outputs)
    routes_digest = route_digest(routes)
    record = Dict{String,Any}(
        "schema" => RAW_RECORD_SCHEMA,
        "benchmark_schema" => BENCHMARK_SCHEMA,
        "cell_id" => context.cell_id,
        "sparse_variant" => context.variant,
        "full_batch_device" => device,
        "tail_device" => "CPU",
        "warmup" => warmup,
        "repetition" => repetition,
        "state_order_sha256" => workload.state_order_sha256,
        "packed_inputs_sha256" => observed_packed_sha256,
        "candidate_count" => workload.total_candidates,
        "action_counts" => workload.action_counts,
        "full_b16_calls" => full_calls,
        "cpu_tail_calls" => tail_calls,
        "cpu_tail_rows" => tail_rows,
        "stage_ns" => stage_ns,
        "sparse_stage_includes_production_head" => true,
        "head_stage_is_bitwise_checked_replay_outside_end_to_end" => true,
        "hashstem_component_including_packing_transfer_wait_ns" =>
            packing_total_ns + hashstem_ns,
        "hashstem_maximum_absolute_error_vs_scalar_cpu_fp32" => hashstem_error,
        "route_ids_sha256" => routes_digest,
        "output_bits_sha256" => output_digest,
        "ranking_q" => Float64.(vec(outputs[1, :])),
        "teacher_top1_agreement" => metrics.teacher_top1_agreement,
        "teacher_ndcg" => metrics.teacher_ndcg,
        "teacher_pairwise_accuracy" => metrics.teacher_pairwise_accuracy,
        "action_top1" => metrics.action_top1,
        "action_top2" => [[pair[1], pair[2]] for pair in metrics.action_top2],
        "packing_included" => true,
        "transfer_wait_included" => true,
        "sync_is_subset_of_hashstem" => true,
        "source_checkpoint_sha256" => context.view.source_checkpoint_sha256,
        "source_state_sha256" => context.view.source_state_sha256,
        "topology_sha256" => context.view.topology_sha256,
    )
    if !warmup
        push!(context.records, record)
        if context.reference_outputs === nothing
            context.reference_outputs = copy(outputs)
            context.reference_routes = deepcopy(routes)
        end
    end
    return record, outputs, routes, metrics
end

function nearest_rank(values::Vector{Int64}, probability::Float64)
    isempty(values) && throw(ArgumentError("empty latency distribution"))
    ordered = sort(values)
    return ordered[clamp(ceil(Int, probability * length(ordered)), 1, length(ordered))]
end

function stage_summary(records)
    return Dict(stage => Dict(
        "p50_ns" => nearest_rank(Int64[record["stage_ns"][stage] for record in records], 0.50),
        "p95_ns" => nearest_rank(Int64[record["stage_ns"][stage] for record in records], 0.95),
    ) for stage in STAGES)
end

function npu_sparse_numeric_parity(cpu_records, npu_records, offsets)
    length(cpu_records) == length(npu_records) || error(
        "CPU/NPU timed record counts differ",
    )
    maximum_q_error = 0.0
    maximum_margin_error = 0.0
    maximum_margin_budget_ratio = 0.0
    for repetition in eachindex(cpu_records)
        cpu_q = Float64.(cpu_records[repetition]["ranking_q"])
        npu_q = Float64.(npu_records[repetition]["ranking_q"])
        length(cpu_q) == length(npu_q) || error("CPU/NPU ranking-Q widths differ")
        maximum_q_error = max(maximum_q_error, maximum(abs.(cpu_q .- npu_q)))
        for rows in offsets
            cpu_state = @view cpu_q[rows]
            npu_state = @view npu_q[rows]
            cpu_order = stable_order(cpu_state)
            npu_order = stable_order(npu_state)
            cpu_order[1:2] == npu_order[1:2] || continue
            cpu_margin = cpu_state[cpu_order[1]] - cpu_state[cpu_order[2]]
            npu_margin = npu_state[npu_order[1]] - npu_state[npu_order[2]]
            margin_error = abs(npu_margin - cpu_margin)
            budget = max(
                ACTION_MARGIN_ABSOLUTE_ERROR_FLOOR,
                ACTION_MARGIN_RELATIVE_ERROR * abs(cpu_margin),
            )
            maximum_margin_error = max(maximum_margin_error, margin_error)
            maximum_margin_budget_ratio = max(
                maximum_margin_budget_ratio, margin_error / budget,
            )
        end
    end
    return (;
        maximum_q_error,
        maximum_margin_error,
        maximum_margin_budget_ratio,
    )
end

function gate_cell(
    context::CellContext,
    workload_binding::ArtifactBinding,
    packed_sha256::String,
    weights_sha256::String,
    sparse_weights_sha256::String,
    cpu_k128_context::CellContext,
)
    records = context.records
    isempty(records) && error("$(context.cell_id) has no timed records")
    reference = context.cell_id == "npu_k128" ? cpu_k128_context : context
    reference_routes = reference.reference_routes
    reference_outputs = reference.reference_outputs
    reference_routes === nothing && error("missing route reference")
    reference_outputs === nothing && error("missing output reference")
    candidate_count = Int(records[1]["candidate_count"])
    observed_routes = context.reference_routes
    observed_outputs = context.reference_outputs
    observed_routes === nothing && error("missing observed routes")
    observed_outputs === nothing && error("missing observed outputs")
    route_matches = 0
    for candidate in 1:candidate_count, layer_id in 1:3
        route_matches += observed_routes[candidate][layer_id] ==
            reference_routes[candidate][layer_id]
    end
    route_total = 3 * candidate_count
    reference_route_digest = route_digest(reference_routes)
    all(record -> record["route_ids_sha256"] == reference_route_digest, records) ||
        (route_matches = 0)
    top1_exact = true
    top2_exact = true
    for record in records
        top1_exact &= record["action_top1"] == reference.records[1]["action_top1"]
        top2_exact &= record["action_top2"] == reference.records[1]["action_top2"]
    end
    quality = Float64(records[1]["teacher_top1_agreement"])
    all(record -> isequal(Float64(record["teacher_top1_agreement"]), quality), records) ||
        error("$(context.cell_id) teacher quality changed between repetitions")
    full_calls = Int(records[1]["full_b16_calls"])
    tail_calls = Int(records[1]["cpu_tail_calls"])
    tail_rows = Int(records[1]["cpu_tail_rows"])
    all(record -> Int(record["full_b16_calls"]) == full_calls, records) || error(
        "full-b16 call count changed between repetitions",
    )
    summary = stage_summary(records)
    return H.HashStemSparse3GateCell(
        cell_id=context.cell_id,
        sparse_variant=context.variant,
        full_batch_device=context.cell_id == "npu_k128" ? "NPU" : "CPU",
        tail_device="CPU",
        candidate_count,
        sample_ids=UInt64.(1:length(records)),
        end_to_end_nanoseconds=Int64[record["stage_ns"]["end_to_end"] for record in records],
        hashstem_component_nanoseconds=Int64[
            record["hashstem_component_including_packing_transfer_wait_ns"] for record in records
        ],
        workload_sha256=workload_binding.initial_sha256,
        packed_inputs_sha256=packed_sha256,
        hashstem_weights_sha256=weights_sha256,
        sparse_bank_weights_sha256=sparse_weights_sha256,
        full_b16_calls_per_sample=full_calls,
        cpu_tail_calls_per_sample=tail_calls,
        cpu_tail_rows=tail_rows,
        packing_included=true,
        transfer_wait_included=true,
        hashstem_component_includes_packing_transfer_wait=true,
        maximum_hashstem_absolute_error=context.maximum_hashstem_absolute_error,
        route_id_matches=route_matches,
        route_id_total=route_total,
        action_top1_matches=top1_exact ? length(records) : 0,
        action_top1_total=length(records),
        top2_swap_count=top2_exact ? 0 : length(records),
        top2_total=length(records),
        quality_metric="teacher_top1_agreement",
        quality_direction="higher_is_better",
        quality_value=quality,
    ), summary
end

function struct_dict(value)
    return Dict{String,Any}(
        string(field) => getfield(value, field) for field in fieldnames(typeof(value))
    )
end

function atomic_write_json(path::String, value)
    ispath(path) && throw(ArgumentError("refusing to overwrite $path"))
    temporary = path * ".tmp." * string(getpid())
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
        end
        mv(temporary, path; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
end

function atomic_write_jsonl(path::String, records)
    ispath(path) && throw(ArgumentError("refusing to overwrite $path"))
    temporary = path * ".tmp." * string(getpid())
    try
        open(temporary, "w") do io
            for record in records
                JSON3.write(io, record)
                write(io, '\n')
            end
            flush(io)
        end
        mv(temporary, path; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
end

function run_benchmark(options, output_directory::String)
    timed_repetitions = parse(Int, get(options, "--timed-repetitions", "30"))
    warmup_repetitions = parse(Int, get(options, "--warmup-repetitions", "2"))
    timed_repetitions >= 30 || throw(ArgumentError("at least 30 timed repetitions are required"))
    timed_repetitions % length(CELL_ORDERS) == 0 || throw(ArgumentError(
        "timed repetitions must be a multiple of six for balanced cell order",
    ))
    warmup_repetitions >= 1 || throw(ArgumentError("at least one warmup repetition is required"))
    bindings = Dict(
        "teacher_checkpoint" => binding(
            options, "ordinary 3-layer teacher checkpoint", "--teacher-checkpoint",
            "--teacher-checkpoint-sha256",
        ),
        "hashstem_weights" => binding(
            options, "HashStem weight NPZ", "--hashstem-weights", "--hashstem-weights-sha256",
        ),
        "snapshot_metadata" => binding(
            options, "HashStem snapshot metadata", "--snapshot-metadata",
            "--snapshot-metadata-sha256",
        ),
        "fixed_ir" => binding(
            options, "HashStem fixed XML", "--fixed-ir", "--fixed-ir-sha256",
        ),
        "fixed_bin" => binding(
            options, "HashStem fixed BIN", "--fixed-bin", "--fixed-bin-sha256",
        ),
        "dynamic_ir" => binding(
            options, "HashStem dynamic CPU XML", "--dynamic-ir", "--dynamic-ir-sha256",
        ),
        "dynamic_bin" => binding(
            options, "HashStem dynamic CPU BIN", "--dynamic-bin", "--dynamic-bin-sha256",
        ),
        "workload" => binding(
            options, "train-only workload", "--workload", "--workload-sha256",
        ),
    )
    snapshot_metadata, weights = parse_snapshot(
        bindings["snapshot_metadata"], bindings["hashstem_weights"],
        bindings["fixed_ir"], bindings["fixed_bin"],
        bindings["dynamic_ir"], bindings["dynamic_bin"],
    )
    views = Dict(
        "cpu_k64" => H.load_sparse_named_inference_view(
            bindings["teacher_checkpoint"].path,
            bindings["teacher_checkpoint"].initial_sha256,
            "k64",
        ),
        "cpu_k128" => H.load_sparse_named_inference_view(
            bindings["teacher_checkpoint"].path,
            bindings["teacher_checkpoint"].initial_sha256,
            "k128",
        ),
        "npu_k128" => H.load_sparse_named_inference_view(
            bindings["teacher_checkpoint"].path,
            bindings["teacher_checkpoint"].initial_sha256,
            "k128",
        ),
    )
    source_state_sha = views["cpu_k64"].source_state_sha256
    all(view -> view.source_state_sha256 == source_state_sha, values(views)) || error(
        "named views do not share byte-identical bank/head/optimizer state",
    )
    workload = load_workload(bindings["workload"], views["cpu_k64"].metadata)
    lowercase(String(views["cpu_k64"].metadata["dataset_manifest_sha256"])) ==
        String(workload.metadata["dataset_manifest_sha256"]) || error(
            "teacher checkpoint/workload dataset binding changed",
        )
    packed_oracle, offsets = pack_workload(workload)
    packed_sha256 = array_sha256(packed_oracle)
    scalar_oracle = Matrix{Float32}(
        undef, workload.total_candidates, H.HASHSTEM_OUTPUT_FEATURES,
    )
    H.hashstem_reference!(
        scalar_oracle,
        H.HashStemReferenceScratch(workload.total_candidates),
        packed_oracle,
        weights,
    )
    executors = compile_openvino(bindings["fixed_ir"].path, bindings["dynamic_ir"].path)
    contexts = Dict{String,CellContext}()
    for cell_id in CELL_IDS
        view = views[cell_id]
        contexts[cell_id] = CellContext(
            cell_id,
            view.variant,
            view,
            S.ThreeLayerWorkspace(view.runtime),
            Dict{String,Any}[],
            nothing,
            nothing,
            0.0,
        )
    end

    for warmup in 1:warmup_repetitions
        for cell_id in CELL_IDS
            run_cell_once!(
                contexts[cell_id], workload, offsets, executors, scalar_oracle,
                packed_sha256, warmup; warmup=true,
            )
        end
    end
    for repetition in 1:timed_repetitions
        order = CELL_ORDERS[mod1(repetition, length(CELL_ORDERS))]
        for (order_position, cell_id) in pairs(order)
            measurement = @timed run_cell_once!(
                contexts[cell_id], workload, offsets, executors, scalar_oracle,
                packed_sha256, repetition; warmup=false,
            )
            record, _, _, _ = measurement.value
            record["execution_order_position"] = order_position
            record["execution_order"] = collect(order)
            record["julia_total_allocated_bytes"] = measurement.bytes
            record["julia_gc_seconds"] = measurement.gctime
            record["timed_gc_observed"] = measurement.gctime > 0
        end
    end

    for artifact in values(bindings)
        verify(artifact)
    end
    sparse_weight_sha = bindings["teacher_checkpoint"].initial_sha256
    cpu64_cell, cpu64_summary = gate_cell(
        contexts["cpu_k64"], bindings["workload"], packed_sha256,
        bindings["hashstem_weights"].initial_sha256, sparse_weight_sha,
        contexts["cpu_k128"],
    )
    cpu128_cell, cpu128_summary = gate_cell(
        contexts["cpu_k128"], bindings["workload"], packed_sha256,
        bindings["hashstem_weights"].initial_sha256, sparse_weight_sha,
        contexts["cpu_k128"],
    )
    npu128_cell, npu128_summary = gate_cell(
        contexts["npu_k128"], bindings["workload"], packed_sha256,
        bindings["hashstem_weights"].initial_sha256, sparse_weight_sha,
        contexts["cpu_k128"],
    )
    gate = H.evaluate_hashstem_sparse3_k_scale_gate(
        cpu64_cell, cpu128_cell, npu128_cell,
    )
    numeric_parity = npu_sparse_numeric_parity(
        contexts["cpu_k128"].records,
        contexts["npu_k128"].records,
        offsets,
    )
    cpu128_actions = contexts["cpu_k128"].records[1]["action_top1"]
    npu128_actions = contexts["npu_k128"].records[1]["action_top1"]
    state_action_top1_matches = sum(cpu128_actions .== npu128_actions)
    cpu128_top2 = contexts["cpu_k128"].records[1]["action_top2"]
    npu128_top2 = contexts["npu_k128"].records[1]["action_top2"]
    state_action_top2_matches = sum(cpu128_top2 .== npu128_top2)
    timed_gc_records = sum(
        Bool(record["timed_gc_observed"])
        for context in values(contexts), record in context.records;
        init=0,
    )
    reasons = copy(gate.reasons)
    numeric_parity.maximum_q_error <= MAXIMUM_SPARSE_Q_ABSOLUTE_ERROR || push!(
        reasons,
        "NPU-k128 sparse ranking-Q maximum absolute error exceeds 1e-2",
    )
    numeric_parity.maximum_margin_budget_ratio <= 1.0 || push!(
        reasons,
        "NPU-k128 action-margin error exceeds its 1e-2/5% budget",
    )
    timed_gc_records == 0 || push!(
        reasons,
        "timed Julia GC was observed; latency promotion is non-authoritative",
    )
    final_passed = gate.passed && isempty(reasons)
    all_records = Dict{String,Any}[]
    for repetition in 1:timed_repetitions, cell_id in CELL_IDS
        push!(all_records, contexts[cell_id].records[repetition])
    end
    raw_path = joinpath(output_directory, "raw_records.jsonl")
    atomic_write_jsonl(raw_path, all_records)
    raw_sha256 = sha256_file(raw_path)
    decision = Dict{String,Any}(
        "schema" => DECISION_SCHEMA,
        "benchmark_schema" => BENCHMARK_SCHEMA,
        "passed" => final_passed,
        "reasons" => reasons,
        "gate" => struct_dict(gate),
        "stage_summary" => Dict(
            "cpu_k64" => cpu64_summary,
            "cpu_k128" => cpu128_summary,
            "npu_k128" => npu128_summary,
        ),
        "numeric" => Dict(
            "cpu_k64_hashstem_max_abs_error" => cpu64_cell.maximum_hashstem_absolute_error,
            "cpu_k128_hashstem_max_abs_error" => cpu128_cell.maximum_hashstem_absolute_error,
            "npu_k128_hashstem_max_abs_error" => npu128_cell.maximum_hashstem_absolute_error,
            "npu_k128_route_matches" => npu128_cell.route_id_matches,
            "npu_k128_route_total" => npu128_cell.route_id_total,
            "npu_k128_action_top1_matches" => npu128_cell.action_top1_matches,
            "npu_k128_action_top1_total" => npu128_cell.action_top1_total,
            "npu_k128_top2_mismatches" => npu128_cell.top2_swap_count,
            "npu_k128_top2_total" => npu128_cell.top2_total,
            "npu_k128_sparse_output_max_abs_error_vs_cpu_k128" =>
                numeric_parity.maximum_q_error,
            "npu_k128_action_margin_max_abs_error_vs_cpu_k128" =>
                numeric_parity.maximum_margin_error,
            "npu_k128_action_margin_max_budget_ratio" =>
                numeric_parity.maximum_margin_budget_ratio,
            "sparse_q_max_abs_error_limit" => MAXIMUM_SPARSE_Q_ABSOLUTE_ERROR,
            "action_margin_abs_error_floor" => ACTION_MARGIN_ABSOLUTE_ERROR_FLOOR,
            "action_margin_relative_error_limit" => ACTION_MARGIN_RELATIVE_ERROR,
            "timed_gc_records" => timed_gc_records,
            "npu_k128_state_action_top1_matches" => state_action_top1_matches,
            "npu_k128_state_action_top1_total" => length(cpu128_actions),
            "npu_k128_state_action_top2_matches" => state_action_top2_matches,
            "npu_k128_state_action_top2_total" => length(cpu128_top2),
            "cpu_k64_teacher_top1" => cpu64_cell.quality_value,
            "cpu_k128_teacher_top1" => cpu128_cell.quality_value,
            "npu_k128_teacher_top1" => npu128_cell.quality_value,
        ),
        "workload" => Dict(
            "split" => workload.metadata["split"],
            "state_count" => length(workload.sets),
            "candidate_count" => workload.total_candidates,
            "state_order_sha256" => workload.state_order_sha256,
            "packed_inputs_sha256" => packed_sha256,
            "warmup_repetitions" => warmup_repetitions,
            "timed_repetitions" => timed_repetitions,
            "cell_order_cycle" => [[order...] for order in CELL_ORDERS],
        ),
        "artifacts" => Dict(name => Dict(
            "path" => artifact.path,
            "bytes" => artifact.bytes,
            "sha256" => artifact.initial_sha256,
        ) for (name, artifact) in bindings),
        "source" => Dict(
            "runner_sha256" => sha256_file(@__FILE__),
            "bridge_sha256" => sha256_file(joinpath(@__DIR__, "sparse_hashstem_bridge.jl")),
            "gate_sha256" => sha256_file(joinpath(@__DIR__, "hashstem_sparse3_k_scale_gate.jl")),
            "named_inference_view_sha256" => sha256_file(joinpath(
                @__DIR__, "sparse_hashstem_named_inference_view.jl",
            )),
            "sparse_source_closure_sha256" => H.sparse_source_closure_sha256(),
        ),
        "openvino" => Dict(
            "available_devices" => executors.devices,
            "cpu_execution_devices" => executors.cpu_execution_devices,
            "npu_execution_devices" => executors.npu_execution_devices,
            "snapshot_model_id" => String(snapshot_metadata["model_id"]),
            "snapshot_version" => Int(snapshot_metadata["snapshot_version"]),
        ),
        "teacher_checkpoint" => Dict(
            "variant" => String(get(views["cpu_k64"].metadata, "variant", "")),
            "ordinary_checkpoint_derived_views" => ["k64", "k128"],
            "optimizer_updates_performed" => 0,
        ),
        "raw_records" => Dict(
            "path" => raw_path,
            "sha256" => raw_sha256,
            "records" => length(all_records),
        ),
        "npu_host_api" => "embedded PythonCall; no Python child process",
    )
    atomic_write_json(joinpath(output_directory, "decision.json"), decision)
    return final_passed
end

function main(arguments=ARGS)
    options = parse_cli(arguments)
    output_directory = abspath(required(options, "--output-directory"))
    ispath(output_directory) && throw(ArgumentError(
        "output directory must be fresh: $output_directory",
    ))
    mkpath(output_directory)
    try
        passed = run_benchmark(options, output_directory)
        passed || exit(2)
    catch error_value
        decision_path = joinpath(output_directory, "decision.json")
        if !ispath(decision_path)
            atomic_write_json(decision_path, Dict(
                "schema" => DECISION_SCHEMA,
                "benchmark_schema" => BENCHMARK_SCHEMA,
                "passed" => false,
                "reasons" => [sprint(showerror, error_value)],
                "failure_type" => string(typeof(error_value)),
                "raw_records" => nothing,
            ))
        end
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
        exit(1)
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
