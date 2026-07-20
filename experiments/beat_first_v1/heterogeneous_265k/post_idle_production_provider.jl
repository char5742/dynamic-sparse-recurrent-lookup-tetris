const POST_IDLE_PROVIDER_IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"

using Main.PostIdleBenchmarkSubstrate
using JLD2
using JSON3
using NPZ
using PythonCall
using SHA

include(joinpath(@__DIR__, "Heterogeneous265K.jl"))
include(joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"))

const H = BeatFirstHeterogeneous265K
const S = SparseDynamic3Layer

const WITNESS_SCHEMA = "heterogeneous-265k-post-idle-witness-v1"
const SNAPSHOT_SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"

mutable struct PostIdleProductionProvider
    config::RunConfig
    frequency::Int64
    witness_metadata::Dict{String,Any}
    candidate_sets::Vector{Any}
    sparse_runtime::S.ThreeLayerRuntime
    sparse_workspace::S.ThreeLayerWorkspace
    hash_weights::H.HashStemWeights
    npu_runtime::Any
    metadata::Dict{String,Any}
end

function _property(value, name::Symbol)
    if value isa AbstractDict
        haskey(value, String(name)) && return value[String(name)]
        haskey(value, name) && return value[name]
        throw(KeyError(name))
    end
    hasproperty(value, name) || throw(ArgumentError("object is missing property $name"))
    return getproperty(value, name)
end

function _string_dict(value)
    return Dict{String,Any}(string(key) => item for (key, item) in pairs(value))
end

function _sha256_file(path::AbstractString)
    return PostIdleBenchmarkSubstrate.sha256_file(path)
end

function _load_witness(config::RunConfig)
    payload = JLD2.load(config.input.path)
    get(payload, "schema", nothing) == WITNESS_SCHEMA || error("witness schema mismatch")
    haskey(payload, "metadata") || error("witness metadata is missing")
    haskey(payload, "candidate_sets") || error("witness candidate sets are missing")
    metadata = _string_dict(payload["metadata"])
    expected = Dict(
        "input_sha256" => config.input.initial_sha256,
        "sparse_checkpoint_sha256" => config.sparse_checkpoint.initial_sha256,
        "master_sha256" => config.master.initial_sha256,
        "snapshot_sha256" => config.snapshot.initial_sha256,
        "system_contract_sha256" => config.system_contract.initial_sha256,
    )
    # A witness cannot contain its own whole-file digest without a cycle. The
    # input digest is bound by the runner; all other scientific artifacts must
    # be embedded by exact digest in the immutable witness metadata.
    for (name, expected_digest) in expected
        name == "input_sha256" && continue
        string(get(metadata, name, "")) == expected_digest || error(
            "witness metadata $name mismatch",
        )
    end
    String(get(metadata, "split", "")) == "teacher_v3_train" || error(
        "post-idle witness must use teacher_v3 train split",
    )
    Bool(get(metadata, "reserved_seed_free", false)) || error(
        "witness does not certify exclusion of development/validation/sealed seeds",
    )
    ranking = Int(get(metadata, "ranking_output_index", 0))
    1 <= ranking <= 22 || error("witness ranking output index must be in 1:22")
    candidate_sets = Vector{Any}(payload["candidate_sets"])
    length(candidate_sets) >= 4352 || error("witness requires at least 4352 candidate sets")
    return metadata, candidate_sets
end

function _load_sparse(config::RunConfig)
    checkpoint = S.load_checkpoint(config.sparse_checkpoint.path)
    S.assert_exact_geometry(checkpoint.runtime.model)
    return checkpoint.runtime, S.ThreeLayerWorkspace(checkpoint.runtime)
end

function _load_hash_weights(config::RunConfig)
    basename(config.master.path) == "checkpoint_manifest.json" || error(
        "--master must bind the HashStem checkpoint_manifest.json",
    )
    loaded = H.load_hashstem_master_checkpoint(dirname(config.master.path))
    loaded.manifest_sha256 == config.master.initial_sha256 || error(
        "HashStem master manifest binding mismatch",
    )
    archive = NPZ.npzread(loaded.weights_path)
    weights = H._weights_from_archive(archive)
    return loaded, weights
end

function _snapshot_binding(config::RunConfig, loaded_master)
    basename(config.snapshot.path) == "snapshot_metadata.json" || error(
        "--snapshot must bind snapshot_metadata.json",
    )
    metadata = JSON3.read(read(config.snapshot.path, String), Dict{String,Any})
    get(metadata, "schema", nothing) == SNAPSHOT_SCHEMA || error("snapshot schema mismatch")
    String(get(metadata, "weights_sha256", "")) ==
        String(loaded_master.manifest.weights_sha256) || error(
            "snapshot/master weight digest mismatch",
        )
    Int(get(metadata, "fixed_batch", 0)) == 16 || error("snapshot batch mismatch")
    Int(get(metadata, "input_features", 0)) == 559 || error("snapshot input mismatch")
    Int(get(metadata, "output_features", 0)) == 256 || error("snapshot output mismatch")
    xml = joinpath(dirname(config.snapshot.path), "hashstem_b16.xml")
    bin = joinpath(dirname(config.snapshot.path), "hashstem_b16.bin")
    _sha256_file(xml) == String(metadata["xml_sha256"]) || error("snapshot XML mismatch")
    _sha256_file(bin) == String(metadata["bin_sha256"]) || error("snapshot BIN mismatch")
    return metadata, xml
end

function _compile_npu(xml::String)
    ov = try
        pyimport("openvino")
    catch error_value
        throw(BackendUnavailable("NPU", String[], "OpenVINO import failed: $(sprint(showerror, error_value))"))
    end
    np = try
        pyimport("numpy")
    catch error_value
        throw(BackendUnavailable("NPU", String[], "NumPy import failed: $(sprint(showerror, error_value))"))
    end
    core = ov.Core()
    devices = String.(pyconvert(Vector{String}, core.available_devices))
    any(device -> uppercase(device) == "NPU", devices) || throw(
        BackendUnavailable("NPU", devices, "OpenVINO did not enumerate exact NPU")
    )
    started = qpc_now()
    compiled = try
        core.compile_model(xml, "NPU")
    catch error_value
        throw(BackendUnavailable(
            "NPU", devices, "NPU compile failed: $(sprint(showerror, error_value))",
        ))
    end
    compile_ticks = qpc_now() - started
    execution_devices = try
        String.(pyconvert(Vector{String}, compiled.get_property("EXECUTION_DEVICES")))
    catch error_value
        throw(BackendUnavailable(
            "NPU", devices,
            "cannot verify compiled execution device: $(sprint(showerror, error_value))",
        ))
    end
    normalized = uppercase.(execution_devices)
    normalized == ["NPU"] || throw(BackendUnavailable(
        "NPU", execution_devices, "compiled graph is not bound exclusively to NPU",
    ))
    request = compiled.create_infer_request()
    return (;
        ov,
        np,
        core,
        compiled,
        request,
        devices,
        execution_devices,
        compile_ticks,
    )
end

function build_post_idle_provider(config::RunConfig, frequency::Integer)
    config.cell_id in SUPPORTED_CELLS || error("production provider does not implement $(config.cell_id)")
    frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    witness_metadata, candidate_sets = _load_witness(config)
    sparse_runtime, sparse_workspace = _load_sparse(config)
    loaded_master, weights = _load_hash_weights(config)
    snapshot_metadata, xml = _snapshot_binding(config, loaded_master)
    npu_runtime = config.cell_id == "H1_npu_hashstem_cpu_sparse" ? _compile_npu(xml) : nothing
    enumerated = npu_runtime === nothing ? ["CPU"] : npu_runtime.devices
    metadata = Dict{String,Any}(
        "implementation_status" => POST_IDLE_PROVIDER_IMPLEMENTATION_STATUS,
        "cell_id" => config.cell_id,
        "input_sha256" => config.input.initial_sha256,
        "sparse_checkpoint_sha256" => config.sparse_checkpoint.initial_sha256,
        "master_sha256" => config.master.initial_sha256,
        "snapshot_sha256" => config.snapshot.initial_sha256,
        "system_contract_sha256" => config.system_contract.initial_sha256,
        "source_manifest_sha256" => config.source_manifest.initial_sha256,
        "qpc_frequency" => Int64(frequency),
        "ranking_output_index" => Int(witness_metadata["ranking_output_index"]),
        "enumerated_backends" => enumerated,
        "NPU_execution_devices" => npu_runtime === nothing ? String[] : npu_runtime.execution_devices,
        "NPU_compile_ticks" => npu_runtime === nothing ? nothing : npu_runtime.compile_ticks,
        "master_weights_sha256" => String(loaded_master.manifest.weights_sha256),
        "snapshot_version" => Int(snapshot_metadata["snapshot_version"]),
        "snapshot_weights_sha256" => String(snapshot_metadata["weights_sha256"]),
        "snapshot_xml_sha256" => String(snapshot_metadata["xml_sha256"]),
        "snapshot_bin_sha256" => String(snapshot_metadata["bin_sha256"]),
    )
    return PostIdleProductionProvider(
        config,
        Int64(frequency),
        witness_metadata,
        candidate_sets,
        sparse_runtime,
        sparse_workspace,
        weights,
        npu_runtime,
        metadata,
    )
end

post_idle_provider_metadata(provider::PostIdleProductionProvider) = provider.metadata
post_idle_candidate_set_count(provider::PostIdleProductionProvider) = length(provider.candidate_sets)

function _candidate_input(set)
    input = _property(set, :input)
    n = S.BeatFirstSparseFeatures.validate_candidate_feature_input(input)
    return input, n
end

function _action_digests(set, count::Int)
    digests = String.(_property(set, :action_digests))
    length(digests) == count || error("action digest count mismatch")
    return digests
end

function _reference_hash_output(set, count::Int)
    output = Float32.(Array(_property(set, :reference_hashstem_output)))
    size(output) == (count, 256) || throw(DimensionMismatch(
        "reference_hashstem_output must be candidate_count x 256",
    ))
    all(isfinite, output) || error("reference HashStem output is non-finite")
    return output
end

function _raw_values!(input, count::Int)
    values = Vector{Vector{Float32}}(undef, count)
    q = Vector{Float32}(undef, 64)
    x = Vector{Float32}(undef, 496)
    for candidate in 1:count
        S.BeatFirstSparseFeatures.split_candidate_features!(q, x, input, candidate)
        values[candidate] = copy(x)
    end
    return values
end

function _cpu_hashstem!(provider, packed::Matrix{Float32}, trace::RawStageTrace, stage::String)
    count = size(packed, 1)
    output = Matrix{Float32}(undef, count, 256)
    scratch = H.HashStemReferenceScratch(count)
    record_stage!(trace, stage) do
        H.hashstem_reference!(output, scratch, packed, provider.hash_weights)
    end
    return output
end

function _npu_chunk!(provider, chunk::Matrix{Float32}, trace::RawStageTrace)
    runtime = provider.npu_runtime
    runtime === nothing && throw(BackendUnavailable("NPU", String[], "NPU runtime missing"))
    py_input = record_stage!(trace, "hash_h2d") do
        runtime.np.ascontiguousarray(PythonCall.Py(chunk), dtype=runtime.np.float32)
    end
    tensor = record_stage!(trace, "hash_bind") do
        runtime.ov.Tensor(py_input)
    end
    record_stage!(trace, "hash_bind") do
        runtime.request.set_input_tensor(tensor)
    end
    record_stage!(trace, "hash_submit") do
        runtime.request.start_async()
    end
    record_stage!(trace, "hash_wait") do
        runtime.request.wait()
    end
    py_output = record_stage!(trace, "hash_d2h") do
        runtime.np.array(runtime.request.get_output_tensor(0).data, dtype=runtime.np.float32, copy=true)
    end
    output = pyconvert(Matrix{Float32}, py_output)
    size(output) == (16, 256) || throw(DimensionMismatch("NPU output must be 16x256"))
    all(isfinite, output) || error("NPU output is non-finite")
    return output
end

function _hash_outputs!(
    provider::PostIdleProductionProvider,
    set,
    input,
    count::Int,
    packed::Union{Nothing,Matrix{Float32}},
    trace::RawStageTrace,
)
    cell = provider.config.cell_id
    if cell == "A0_cpu_route_cpu_active"
        output = record_stage!(trace, "hash_reference_copy") do
            copy(_reference_hash_output(set, count))
        end
        return output, ["REFERENCE", "CPU_SPARSE"], Dict("REFERENCE" => 1), 0, 0, 0
    elseif cell == "H0_cpu_hashstem_cpu_sparse"
        output = _cpu_hashstem!(provider, packed, trace, "hash_cpu_compute")
        copied = record_stage!(trace, "hash_copy") do
            copy(output)
        end
        return copied, ["CPU_REFERENCE", "CPU_SPARSE"], Dict("CPU_REFERENCE" => 1), 0, 0, 0
    end

    output = Matrix{Float32}(undef, count, 256)
    full_calls = 0
    position = 1
    while position + 15 <= count
        chunk = Matrix{Float32}(@view packed[position:(position + 15), :])
        chunk_output = _npu_chunk!(provider, chunk, trace)
        output[position:(position + 15), :] .= chunk_output
        full_calls += 1
        position += 16
    end
    tail = count - position + 1
    tail_calls = 0
    if tail > 0
        tail_packed = Matrix{Float32}(@view packed[position:count, :])
        tail_output = _cpu_hashstem!(provider, tail_packed, trace, "hash_cpu_tail")
        output[position:count, :] .= tail_output
        tail_calls = 1
    end
    copied = record_stage!(trace, "hash_copy") do
        copy(output)
    end
    backends = if full_calls == 0
        ["CPU_TAIL", "CPU_SPARSE"]
    elseif tail_calls == 0
        ["NPU", "CPU_SPARSE"]
    else
        ["NPU", "CPU_TAIL", "CPU_SPARSE"]
    end
    calls = Dict("NPU" => full_calls, "CPU_TAIL" => tail_calls)
    h2d = full_calls * 16 * 559 * sizeof(Float32)
    d2h = full_calls * 16 * 256 * sizeof(Float32)
    return copied, backends, calls, h2d, d2h, full_calls
end

function _routed_forward_traced!(
    provider::PostIdleProductionProvider,
    input::S.ThreeLayerInput,
    trace::RawStageTrace,
)
    runtime = provider.sparse_runtime
    workspace = provider.sparse_workspace
    q1 = copy(input.base_queries[1])
    x1 = copy(input.raw_value)
    ids1 = record_stage!(trace, "l1_route") do
        S._query_eventtime!(
            workspace.selected_ids[1], runtime.indexes[1], workspace.query_scratch[1],
            runtime.model.layers[1].theta, runtime.bank_optimizers[1], q1;
            target=runtime.model.layers[1].active_count,
            max_scored_rows=S.LAYER_MAX_SCORED_ROWS[1],
            max_bucket_entries=S.LAYER_MAX_BUCKET_ENTRIES[1],
            training_probe_count=0,
            probe_token=UInt64(0x9e3779b97f4a7c15),
        )
    end
    record_stage!(trace, "l1_gather") do
        S.materialize_rows!(
            runtime.model.layers[1].theta, runtime.bank_optimizers[1], ids1,
        )
    end
    ids1_copy, _, a1 = record_stage!(trace, "l1_active") do
        S._forward_layer(runtime.model.layers[1], q1, x1, ids1)
    end

    q2 = Vector{Float32}(undef, S.ROUTE_DIM)
    x2 = Vector{Float32}(undef, S.DEEP_VALUE_DIM)
    record_stage!(trace, "l1_scatter") do
        S._compose_deep_query!(q2, input.base_queries[2], ids1_copy, a1, 1)
        S._compose_deep_value!(x2, ids1_copy, a1, input.context, input.next_hold, 1)
    end
    ids2 = record_stage!(trace, "l2_route") do
        S._query_eventtime!(
            workspace.selected_ids[2], runtime.indexes[2], workspace.query_scratch[2],
            runtime.model.layers[2].theta, runtime.bank_optimizers[2], q2;
            target=runtime.model.layers[2].active_count,
            max_scored_rows=S.LAYER_MAX_SCORED_ROWS[2],
            max_bucket_entries=S.LAYER_MAX_BUCKET_ENTRIES[2],
            training_probe_count=0,
            probe_token=UInt64(0xbf58476d1ce4e5b9),
        )
    end
    record_stage!(trace, "l2_gather") do
        S.materialize_rows!(
            runtime.model.layers[2].theta, runtime.bank_optimizers[2], ids2,
        )
    end
    ids2_copy, _, a2 = record_stage!(trace, "l2_active") do
        S._forward_layer(runtime.model.layers[2], q2, x2, ids2)
    end

    q3 = Vector{Float32}(undef, S.ROUTE_DIM)
    x3 = Vector{Float32}(undef, S.DEEP_VALUE_DIM)
    record_stage!(trace, "l2_scatter") do
        S._compose_deep_query!(q3, input.base_queries[3], ids2_copy, a2, 2)
        S._compose_deep_value!(x3, ids2_copy, a2, input.context, input.next_hold, 2)
    end
    ids3 = record_stage!(trace, "l3_route") do
        S._query_eventtime!(
            workspace.selected_ids[3], runtime.indexes[3], workspace.query_scratch[3],
            runtime.model.layers[3].theta, runtime.bank_optimizers[3], q3;
            target=runtime.model.layers[3].active_count,
            max_scored_rows=S.LAYER_MAX_SCORED_ROWS[3],
            max_bucket_entries=S.LAYER_MAX_BUCKET_ENTRIES[3],
            training_probe_count=0,
            probe_token=UInt64(0x94d049bb133111eb),
        )
    end
    record_stage!(trace, "l3_gather") do
        S.materialize_rows!(
            runtime.model.layers[3].theta, runtime.bank_optimizers[3], ids3,
        )
    end
    ids3_copy, _, a3 = record_stage!(trace, "l3_active") do
        S._forward_layer(runtime.model.layers[3], q3, x3, ids3)
    end

    latent = Vector{Float32}(undef, S.LATENT_DIM)
    record_stage!(trace, "l3_scatter") do
        S._compose_latent!(latent, ids3_copy, a3, input.context, input.next_hold, 3)
    end
    output = Vector{Float32}(undef, S.OUTPUT_DIM)
    record_stage!(trace, "head") do
        @inbounds for output_id in 1:S.OUTPUT_DIM
            accumulator = runtime.model.bias[output_id]
            @simd for hidden in 1:S.LATENT_DIM
                accumulator = muladd(
                    runtime.model.head[output_id, hidden], latent[hidden], accumulator,
                )
            end
            output[output_id] = accumulator
        end
    end
    return output, (copy(ids1_copy), copy(ids2_copy), copy(ids3_copy))
end

function run_post_idle_candidate_set!(
    provider::PostIdleProductionProvider,
    cell_id::AbstractString,
    sequence::Integer,
    trace::RawStageTrace;
    warmup::Bool,
)
    String(cell_id) == provider.config.cell_id || error("provider cell changed")
    1 <= sequence <= length(provider.candidate_sets) || throw(BoundsError(
        provider.candidate_sets, sequence,
    ))
    set = provider.candidate_sets[sequence]
    input, count = _candidate_input(set)
    action_digests = _action_digests(set, count)
    set_id = String(_property(set, :set_id))
    raw_values = record_stage!(trace, "pack") do
        _raw_values!(input, count)
    end
    packed = if cell_id == "A0_cpu_route_cpu_active"
        nothing
    else
        destination = Matrix{Float32}(undef, count, 559)
        record_stage!(trace, "hash_pack") do
            H.pack_hashstem_input!(destination, input, collect(1:count))
        end
        destination
    end
    hash_output, backends, physical_calls, h2d, d2h, synchronizations =
        _hash_outputs!(provider, set, input, count, packed, trace)
    views = H.split_hashstem_output(hash_output)
    outputs = Matrix{Float32}(undef, 22, count)
    routes = Vector{NTuple{3,Vector{Int32}}}(undef, count)
    for candidate in 1:count
        sparse_input = S.ThreeLayerInput(
            (
                Vector{Float32}(@view views.query_1[candidate, :]),
                Vector{Float32}(@view views.query_2[candidate, :]),
                Vector{Float32}(@view views.query_3[candidate, :]),
            ),
            raw_values[candidate],
            Vector{Float32}(@view views.context[candidate, :]),
            Vector{Float32}(@view views.next_hold_passthrough[candidate, :]),
        )
        output, selected = _routed_forward_traced!(provider, sparse_input, trace)
        outputs[:, candidate] .= output
        routes[candidate] = selected
    end
    packed_bytes = count * 496 * sizeof(Float32) +
        (packed === nothing ? 0 : length(packed) * sizeof(Float32))
    return CandidateSetResult(
        set_id,
        action_digests,
        routes,
        outputs,
        backends,
        physical_calls,
        packed_bytes,
        h2d,
        d2h,
        synchronizations,
        trace,
    )
end
