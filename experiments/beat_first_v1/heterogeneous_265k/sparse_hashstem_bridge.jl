using SHA

export SPARSE_HASHSTEM_BRIDGE_SCHEMA,
       SPARSE_ROUTE_WITNESS_SCHEMA,
       PRODUCTION_ROUTE_SEMANTICS,
       sparse_bridge_variant_spec,
       SparseBridgeLineage,
       SparseHashStemBridge,
       SparseHashStemForwardBatch,
       SparseHashStemBackwardBatch,
       sparse_source_closure_sha256,
       sparse_bridge_checkpoint_metadata,
       bind_teacher_checkpoint_for_hashstem_bridge,
       load_sparse_hashstem_bridge,
       raw_values_from_hashstem_packed,
       three_layer_inputs_from_hashstem,
       route_hashstem_batch!,
       sparse_vjp_to_hashstem_hooks,
       production_route_conformance_witness

const SPARSE_HASHSTEM_BRIDGE_SCHEMA = "hashstem-to-slide3-semantic-bridge-v1"
const SPARSE_ROUTE_WITNESS_SCHEMA = "hashstem-slide3-production-route-witness-v1"

# These names intentionally mirror, but do not import, the teacher trainer.
# The heterogeneous adapter remains usable with a frozen sparse checkpoint
# without loading Zygote or the teacher dataset.  A checkpoint binds one exact
# name and topology; custom active widths and the historical (26,22,22)
# baseline are not silently relabelled as a k-sweep member.
const _SPARSE_BRIDGE_VARIANTS = (
    k64=(;
        name=:k64,
        active_counts=(24, 20, 20),
        training_probes=(3, 2, 2),
        active_parameters=31_934,
        forward_macs=31_912,
        inclusive_training_macs=78_504,
    ),
    k128=(;
        name=:k128,
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        active_parameters=58_214,
        forward_macs=58_192,
        inclusive_training_macs=141_520,
    ),
    k256=(;
        name=:k256,
        active_counts=(96, 80, 80),
        training_probes=(12, 10, 10),
        active_parameters=110_774,
        forward_macs=110_752,
        inclusive_training_macs=267_552,
    ),
)

function sparse_bridge_variant_spec(name::Union{Symbol,AbstractString})
    key = name isa Symbol ? name : Symbol(strip(name))
    key in propertynames(_SPARSE_BRIDGE_VARIANTS) || throw(ArgumentError(
        "HashStem sparse variant must be exactly k64, k128, or k256; got $name",
    ))
    return getproperty(_SPARSE_BRIDGE_VARIANTS, key)
end

"""The semantics inherited by calling the frozen production route verbatim."""
const PRODUCTION_ROUTE_SEMANTICS = (
    retrieval="WTA collision count descending before exact score and neuron ID",
    rerank="Float64 exact route-key dot multiplied by pending logical decay",
    materialization="only final selected rows materialize pending lazy decay",
    deeper_routes="ID-keyed CountSketch residual recomputed from actual L1/L2 activations",
    hard_route_derivative="selected IDs are fixed in the sparse VJP",
    implementation="SparseDynamic3Layer.route_forward! and vjp_selected",
)

const _SPARSE_SOURCE_PATHS = (
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "geometry.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "model.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "optimizer.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "runtime.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "checkpoint.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic", "features.jl"),
    joinpath(@__DIR__, "..", "sparse_dynamic", "wta_index.jl"),
    joinpath(@__DIR__, "..", "Project.toml"),
    joinpath(@__DIR__, "..", "Manifest.toml"),
)

_sha256_text(value::AbstractString) = bytes2hex(
    SHA.sha256(Vector{UInt8}(codeunits(String(value)))),
)

function _primitive_array_sha256(value::AbstractArray{T}) where {T<:Union{Float32,Int32}}
    Base.ENDIAN_BOM == 0x04030201 || error("little-endian host required for ABI digests")
    contiguous = collect(vec(value))
    return bytes2hex(SHA.sha256(collect(reinterpret(UInt8, contiguous))))
end

"""Hash the complete frozen sparse routing/training source closure."""
function sparse_source_closure_sha256()
    root = normpath(joinpath(@__DIR__, ".."))
    records = String[]
    for source in _SPARSE_SOURCE_PATHS
        absolute = abspath(source)
        isfile(absolute) || error("sparse source closure member is missing: $absolute")
        relative = replace(relpath(absolute, root), '\\' => '/')
        push!(records, relative * "\0" * _sha256_file(absolute))
    end
    return _sha256_text(join(records, "\n"))
end

"""Immutable combined-model lineage written into every sparse checkpoint.

The checkpoint byte digest is intentionally not a field: it is only known
after the atomic sparse checkpoint has been committed and is required by the
loader separately.
"""
Base.@kwdef struct SparseBridgeLineage
    sparse_model_id::String
    sparse_variant::String
    sparse_bank_version::UInt64
    sparse_teacher_checkpoint_sha256::String
    sparse_source_closure_sha256::String
    hashstem_model_id::String
    hashstem_master_version::UInt64
    hashstem_snapshot_version::UInt64
    hashstem_weights_sha256::String
    hashstem_normalization_sha256::String
end

function _validate_lineage(lineage::SparseBridgeLineage)
    isempty(lineage.sparse_model_id) && throw(ArgumentError("sparse_model_id is empty"))
    isempty(lineage.hashstem_model_id) && throw(ArgumentError("hashstem_model_id is empty"))
    variant = sparse_bridge_variant_spec(lineage.sparse_variant)
    lineage.sparse_variant == String(variant.name) || throw(ArgumentError(
        "sparse_variant must use the canonical lowercase named-variant spelling",
    ))
    for (label, digest) in (
        ("sparse teacher checkpoint", lineage.sparse_teacher_checkpoint_sha256),
        ("sparse source closure", lineage.sparse_source_closure_sha256),
        ("HashStem weights", lineage.hashstem_weights_sha256),
        ("HashStem normalization", lineage.hashstem_normalization_sha256),
    )
        _valid_sha256(digest) || throw(ArgumentError("invalid $label SHA-256"))
    end
    lineage.hashstem_snapshot_version == 0 ||
        lineage.hashstem_master_version > 0 || throw(ArgumentError(
            "a published HashStem snapshot cannot bind master version zero",
        ))
    return lineage
end

function _same_lineage(left::SparseBridgeLineage, right::SparseBridgeLineage)
    return all(
        getfield(left, field) == getfield(right, field)
        for field in fieldnames(SparseBridgeLineage)
    )
end

"""Metadata to pass to `SparseDynamic3Layer.save_checkpoint`.

This does not save a checkpoint and cannot self-certify runtime validation.
"""
function sparse_bridge_checkpoint_metadata(lineage::SparseBridgeLineage)
    _validate_lineage(lineage)
    variant = sparse_bridge_variant_spec(lineage.sparse_variant)
    lineage.sparse_source_closure_sha256 == sparse_source_closure_sha256() ||
        error("lineage sparse source closure differs from the loaded source")
    return Dict{String,Any}(
        "bridge_schema" => SPARSE_HASHSTEM_BRIDGE_SCHEMA,
        "bridge_source_sha256" => _sha256_file(@__FILE__),
        "implementation_provenance" => UNEXECUTED_STATIC_ONLY,
        "runtime_validation_status" => UNEXECUTED_STATIC_ONLY,
        "sparse_model_id" => lineage.sparse_model_id,
        "sparse_variant" => lineage.sparse_variant,
        "sparse_active_counts" => collect(variant.active_counts),
        "sparse_training_probes" => collect(variant.training_probes),
        "sparse_bank_version" => lineage.sparse_bank_version,
        "sparse_teacher_checkpoint_sha256" =>
            lineage.sparse_teacher_checkpoint_sha256,
        "sparse_source_closure_sha256" => lineage.sparse_source_closure_sha256,
        "hashstem_model_id" => lineage.hashstem_model_id,
        "hashstem_master_version" => lineage.hashstem_master_version,
        "hashstem_snapshot_version" => lineage.hashstem_snapshot_version,
        "hashstem_weights_sha256" => lineage.hashstem_weights_sha256,
        "hashstem_normalization_sha256" => lineage.hashstem_normalization_sha256,
    )
end

struct SparseHashStemBridge
    lineage::SparseBridgeLineage
    checkpoint_path::String
    checkpoint_sha256::String
    checkpoint_format::String
    checkpoint_version::Int
    topology_sha256::String
    bridge_source_sha256::String
    runtime::SparseDynamic3Layer.ThreeLayerRuntime
    metadata::Dict{String,Any}
    implementation_provenance::String
end

function _topology_record(topology)
    return (
        routing_policy=String(topology.routing_policy),
        route_dim=Int(topology.route_dim),
        value_dims=Tuple(Int.(topology.value_dims)),
        row_dims=Tuple(Int.(topology.row_dims)),
        neuron_counts=Tuple(Int.(topology.neuron_counts)),
        active_counts=Tuple(Int.(topology.active_counts)),
        output_dim=Int(topology.output_dim),
        latent_dim=Int(topology.latent_dim),
        parameter_count=Int(topology.parameter_count),
    )
end

function _validate_production_topology(topology, runtime, variant)
    record = _topology_record(topology)
    record.routing_policy == SparseDynamic3Layer.ROUTING_POLICY ||
        error("sparse routing policy drift")
    record.route_dim == SparseDynamic3Layer.ROUTE_DIM || error("route width drift")
    record.value_dims == SparseDynamic3Layer.LAYER_VALUE_DIMS || error("value width drift")
    record.row_dims == SparseDynamic3Layer.LAYER_ROW_DIMS || error("row width drift")
    record.neuron_counts == SparseDynamic3Layer.LAYER_NEURON_COUNTS ||
        error("sparse bank width drift")
    record.active_counts == variant.active_counts ||
        error("active width drift for sparse variant $(variant.name)")
    record.output_dim == SparseDynamic3Layer.OUTPUT_DIM || error("output width drift")
    record.latent_dim == SparseDynamic3Layer.LATENT_DIM || error("latent width drift")
    record.parameter_count == SparseDynamic3Layer.TOTAL_PARAMETERS ||
        error("parameter count drift")
    SparseDynamic3Layer.PRODUCTION_DENSE_FALLBACK && error("dense fallback is forbidden")
    SparseDynamic3Layer.parameter_count(runtime.model) ==
        SparseDynamic3Layer.TOTAL_PARAMETERS || error("runtime parameter count drift")
    SparseDynamic3Layer.active_parameter_count(runtime.model) ==
        variant.active_parameters || error("runtime active-parameter count drift")
    for layer_id in 1:3
        layer = runtime.model.layers[layer_id]
        size(layer.theta) == (
            SparseDynamic3Layer.LAYER_ROW_DIMS[layer_id],
            SparseDynamic3Layer.LAYER_NEURON_COUNTS[layer_id],
        ) || error("layer $layer_id sparse-bank geometry drift")
        layer.active_count == variant.active_counts[layer_id] || error(
            "layer $layer_id active width differs from $(variant.name)",
        )
    end
    wta_configs = ntuple(3) do layer_id
        observed = runtime.indexes[layer_id].config
        expected = SparseDynamic3Layer._layer_wta_config(
            layer_id, runtime.model.layers[layer_id].active_count,
        )
        for field in fieldnames(typeof(expected))
            getfield(observed, field) == getfield(expected, field) ||
                error("layer $layer_id production WTA config field $field drifted")
        end
        (
            m=observed.m,
            K=observed.K,
            L=observed.L,
            target=observed.target,
            min=observed.min,
            max=observed.max,
            training_probes=observed.training_probes,
            seed=observed.seed,
        )
    end
    return merge(record, (; wta_configs))
end

function _metadata_value(metadata::AbstractDict, key::String)
    haskey(metadata, key) || error("sparse checkpoint metadata is missing $key")
    return metadata[key]
end

function _validate_checkpoint_lineage(metadata::AbstractDict, lineage::SparseBridgeLineage)
    expected = sparse_bridge_checkpoint_metadata(lineage)
    for key in keys(expected)
        observed = _metadata_value(metadata, key)
        if key in (
            "sparse_bank_version", "hashstem_master_version", "hashstem_snapshot_version",
        )
            UInt64(observed) == UInt64(expected[key]) || error("checkpoint $key mismatch")
        elseif key in ("sparse_active_counts", "sparse_training_probes")
            Tuple(Int.(observed)) == Tuple(Int.(expected[key])) ||
                error("checkpoint $key mismatch")
        elseif key == "runtime_validation_status"
            String(observed) in (UNEXECUTED_STATIC_ONLY, VALIDATED_RUNTIME) ||
                error("checkpoint runtime validation status is invalid")
        else
            String(observed) == String(expected[key]) || error("checkpoint $key mismatch")
        end
    end
    return metadata
end

function _runtime_optimizer_step(runtime::SparseDynamic3Layer.ThreeLayerRuntime)
    steps = ntuple(i -> runtime.bank_optimizers[i].global_step, 3)
    (steps[1] == steps[2] && steps[2] == steps[3]) ||
        error("sparse optimizer clocks diverged")
    runtime.head_optimizer.step == steps[1] || error("sparse/head clocks diverged")
    return steps[1]
end

function _validate_teacher_checkpoint_binding(loaded, lineage, variant)
    metadata = loaded.metadata
    expected_keys = Set((
        "source_sha256",
        "julia_version",
        "project_sha256",
        "manifest_sha256",
        "dataset_manifest_sha256",
        "pairing_contract_sha256",
        "variant",
        "routing_policy",
        "ranking_source",
        "ranking_output_index",
    ))
    Set(String.(keys(metadata))) == expected_keys || error(
        "teacher checkpoint metadata key set is not canonical",
    )
    for key in (
        "source_sha256", "project_sha256", "manifest_sha256",
        "dataset_manifest_sha256",
    )
        _valid_sha256(String(_metadata_value(metadata, key))) || error(
            "teacher checkpoint $key is not a SHA-256",
        )
    end
    String(_metadata_value(metadata, "julia_version")) == string(VERSION) || error(
        "teacher checkpoint Julia version differs from the bridge process",
    )
    String(_metadata_value(metadata, "variant")) == lineage.sparse_variant ||
        error("teacher checkpoint variant differs from bridge lineage")
    String(_metadata_value(metadata, "routing_policy")) ==
        SparseDynamic3Layer.ROUTING_POLICY || error(
        "teacher checkpoint routing policy differs from the bridge runtime",
    )
    String(_metadata_value(metadata, "ranking_source")) == "q" || error(
        "teacher checkpoint ranking source differs from q",
    )
    Int(_metadata_value(metadata, "ranking_output_index")) == 1 || error(
        "teacher checkpoint ranking output index differs from one",
    )
    return _validate_teacher_continuation(loaded, lineage, variant)
end

function _validate_teacher_continuation(loaded, lineage, variant)
    state = loaded.training_state
    state === nothing && error("teacher checkpoint has no continuation state")
    hasproperty(state, :update) || error("teacher checkpoint update is missing")
    hasproperty(state, :config) || error("teacher checkpoint config is missing")
    config = state.config
    hasproperty(config, :variant) || error("teacher config variant is missing")
    hasproperty(config, :training_probes) || error(
        "teacher config training_probes are missing",
    )
    hasproperty(config, :routing_policy) || error(
        "teacher config routing_policy is missing",
    )
    String(config.variant) == lineage.sparse_variant || error(
        "teacher config variant differs from bridge lineage",
    )
    Tuple(Int.(config.training_probes)) == variant.training_probes || error(
        "teacher config probes differ from the named bridge variant",
    )
    String(config.routing_policy) == SparseDynamic3Layer.ROUTING_POLICY ||
        error("teacher config routing policy differs from the bridge runtime")
    step = _runtime_optimizer_step(loaded.runtime)
    UInt64(state.update) == step || error(
        "teacher checkpoint update differs from sparse optimizer clocks",
    )
    return step
end

"""Create a fresh bridge-bound checkpoint without changing the CPU trainer.

The ordinary teacher checkpoint remains the immutable parent.  Its runtime,
optimizer/index state, sampler/RNG continuation payload, and existing metadata
are reserialized unchanged into a fresh destination carrying the additional
HashStem lineage.  The destination is immediately consumed by the strict
bridge loader; this is the only supported producer for that loader.
"""
function bind_teacher_checkpoint_for_hashstem_bridge(
    teacher_checkpoint_path::AbstractString,
    destination_path::AbstractString,
    lineage::SparseBridgeLineage;
    expected_teacher_checkpoint_sha256::AbstractString,
)
    _validate_lineage(lineage)
    _valid_sha256(expected_teacher_checkpoint_sha256) || throw(ArgumentError(
        "invalid expected teacher checkpoint SHA-256",
    ))
    source = abspath(teacher_checkpoint_path)
    destination = abspath(destination_path)
    lowercase(normpath(source)) == lowercase(normpath(destination)) &&
        throw(ArgumentError("bridge destination must differ from teacher checkpoint"))
    isfile(source) || throw(ArgumentError(
        "teacher checkpoint does not exist: $source",
    ))
    ispath(destination) && throw(ArgumentError(
        "bridge checkpoint destination must be fresh: $destination",
    ))
    source_digest = _sha256_file(source)
    source_digest == String(expected_teacher_checkpoint_sha256) || error(
        "teacher checkpoint SHA-256 differs from the external expectation",
    )
    source_digest == lineage.sparse_teacher_checkpoint_sha256 || error(
        "teacher checkpoint SHA-256 differs from bridge lineage",
    )
    loaded = SparseDynamic3Layer.load_checkpoint(source)
    variant = sparse_bridge_variant_spec(lineage.sparse_variant)
    topology = _validate_production_topology(loaded.topology, loaded.runtime, variant)
    step = _validate_teacher_checkpoint_binding(loaded, lineage, variant)
    metadata = Dict{String,Any}(loaded.metadata)
    merge!(metadata, sparse_bridge_checkpoint_metadata(lineage))
    metadata["bridge_parent_checkpoint_format"] =
        SparseDynamic3Layer.CHECKPOINT_FORMAT
    metadata["bridge_parent_checkpoint_version"] =
        SparseDynamic3Layer.CHECKPOINT_VERSION
    SparseDynamic3Layer.save_checkpoint(
        destination,
        loaded.runtime;
        training_state=loaded.training_state,
        metadata,
    )
    bridge_digest = _sha256_file(destination)
    bridge = load_sparse_hashstem_bridge(
        destination,
        lineage;
        expected_checkpoint_sha256=bridge_digest,
    )
    return (;
        path=destination,
        bytes=filesize(destination),
        sha256=bridge_digest,
        parent_checkpoint_sha256=source_digest,
        variant=variant.name,
        active_counts=topology.active_counts,
        optimizer_step=step,
        bridge,
    )
end

"""Load a byte-pinned production 3-layer runtime and reject all lineage drift."""
function load_sparse_hashstem_bridge(
    checkpoint_path::AbstractString,
    lineage::SparseBridgeLineage;
    expected_checkpoint_sha256::AbstractString,
)
    _validate_lineage(lineage)
    _valid_sha256(expected_checkpoint_sha256) ||
        throw(ArgumentError("invalid expected sparse checkpoint SHA-256"))
    source_digest = sparse_source_closure_sha256()
    source_digest == lineage.sparse_source_closure_sha256 ||
        error("frozen sparse source closure changed")
    source = abspath(checkpoint_path)
    isfile(source) || throw(ArgumentError("sparse checkpoint does not exist: $source"))
    checkpoint_digest = _sha256_file(source)
    checkpoint_digest == String(expected_checkpoint_sha256) ||
        error("sparse checkpoint SHA-256 mismatch")
    loaded = SparseDynamic3Layer.load_checkpoint(source)
    variant = sparse_bridge_variant_spec(lineage.sparse_variant)
    topology = _validate_production_topology(loaded.topology, loaded.runtime, variant)
    _validate_checkpoint_lineage(loaded.metadata, lineage)
    _validate_teacher_continuation(loaded, lineage, variant)
    topology_digest = _sha256_text(JSON3.write(topology))
    return SparseHashStemBridge(
        lineage,
        source,
        checkpoint_digest,
        SparseDynamic3Layer.CHECKPOINT_FORMAT,
        SparseDynamic3Layer.CHECKPOINT_VERSION,
        topology_digest,
        _sha256_file(@__FILE__),
        loaded.runtime,
        Dict{String,Any}(loaded.metadata),
        UNEXECUTED_STATIC_ONLY,
    )
end

"""Rebuild exact sparse `x496` from the same canonical HashStem `[B,559]` rows."""
function raw_values_from_hashstem_packed(packed::AbstractMatrix{Float32})
    batch = size(packed, 1)
    size(packed) == (batch, HASHSTEM_INPUT_FEATURES) ||
        throw(DimensionMismatch("packed HashStem input must be Bx559"))
    all(isfinite, packed) || throw(ArgumentError("packed HashStem input is non-finite"))
    raw = Matrix{Float32}(undef, batch, SparseDynamic3Layer.RAW_VALUE_DIM)
    raw[:, 1:(2 * BOARD_FEATURES)] .= packed[:, 1:(2 * BOARD_FEATURES)]
    auxiliary_offset = 2 * BOARD_FEATURES + NEXT_HOLD_FEATURES
    value_offset = 2 * BOARD_FEATURES
    for (position, auxiliary_index) in enumerate(
        SparseDynamic3Layer.BeatFirstSparseFeatures.VALUE_AUX_INDICES,
    )
        raw[:, value_offset + position] .= packed[:, auxiliary_offset + auxiliary_index]
    end
    raw[:, end] .= 1.0f0
    all(isfinite, raw) || error("constructed sparse raw values are non-finite")
    return raw
end

function _validate_hashstem_pair(
    packed::AbstractMatrix{Float32},
    hashstem_output::AbstractMatrix{Float32},
)
    batch = size(packed, 1)
    size(packed) == (batch, HASHSTEM_INPUT_FEATURES) ||
        throw(DimensionMismatch("packed HashStem input must be Bx559"))
    size(hashstem_output) == (batch, HASHSTEM_OUTPUT_FEATURES) ||
        throw(DimensionMismatch("HashStem output must be Bx256"))
    all(isfinite, packed) || throw(ArgumentError("packed HashStem input is non-finite"))
    all(isfinite, hashstem_output) || throw(ArgumentError("HashStem output is non-finite"))
    @inbounds for row in 1:batch, offset in 1:NEXT_HOLD_FEATURES
        isequal(
            hashstem_output[row, first(NEXT_HOLD_PASSTHROUGH_RANGE) + offset - 1],
            packed[row, 2 * BOARD_FEATURES + offset],
        ) || error("HashStem NEXT/HOLD passthrough does not match its packed row")
    end
    return batch
end

"""Construct exact independent-candidate `ThreeLayerInput`s from one stem call."""
function three_layer_inputs_from_hashstem(
    packed::AbstractMatrix{Float32},
    hashstem_output::AbstractMatrix{Float32},
)
    batch = _validate_hashstem_pair(packed, hashstem_output)
    raw_values = raw_values_from_hashstem_packed(packed)
    inputs = Vector{SparseDynamic3Layer.ThreeLayerInput}(undef, batch)
    @inbounds for row in 1:batch
        inputs[row] = SparseDynamic3Layer.ThreeLayerInput(
            (
                view(hashstem_output, row, QUERY_1_RANGE),
                view(hashstem_output, row, QUERY_2_RANGE),
                view(hashstem_output, row, QUERY_3_RANGE),
            ),
            view(raw_values, row, :),
            view(hashstem_output, row, CONTEXT_RANGE),
            view(hashstem_output, row, NEXT_HOLD_PASSTHROUGH_RANGE),
        )
    end
    return inputs
end

struct SparseHashStemForwardBatch
    inputs::Vector{SparseDynamic3Layer.ThreeLayerInput}
    results::Vector{SparseDynamic3Layer.RoutedForwardResult}
    lineage::SparseBridgeLineage
    checkpoint_sha256::String
    topology_sha256::String
    optimizer_step::UInt64
    training_probes::NTuple{3,Int}
end

"""Sequentially invoke the frozen production route; no worker or accelerator call."""
function route_hashstem_batch!(
    bridge::SparseHashStemBridge,
    workspace::SparseDynamic3Layer.ThreeLayerWorkspace,
    packed::AbstractMatrix{Float32},
    hashstem_output::AbstractMatrix{Float32};
    training_probes::NTuple{3,Int}=(0, 0, 0),
    probe_token_base::UInt64=UInt64(0),
)
    variant = sparse_bridge_variant_spec(bridge.lineage.sparse_variant)
    training_probes in ((0, 0, 0), variant.training_probes) || throw(ArgumentError(
        "training_probes must be zero for inference or exactly " *
        "$(variant.training_probes) for $(variant.name)",
    ))
    inputs = three_layer_inputs_from_hashstem(packed, hashstem_output)
    batch = length(inputs)
    batch == 0 && throw(ArgumentError("empty HashStem batch is not routable"))
    probe_token_base <= typemax(UInt64) - UInt64(batch - 1) ||
        throw(OverflowError("probe token range overflows UInt64"))
    step_before = _runtime_optimizer_step(bridge.runtime)
    results = Vector{SparseDynamic3Layer.RoutedForwardResult}(undef, batch)
    for row in 1:batch
        results[row] = SparseDynamic3Layer.route_forward!(
            bridge.runtime,
            workspace,
            inputs[row];
            training_probes,
            probe_token=probe_token_base + UInt64(row - 1),
        )
        all(isfinite, results[row].output) || error("sparse forward output is non-finite")
    end
    _runtime_optimizer_step(bridge.runtime) == step_before ||
        error("routing unexpectedly advanced the sparse optimizer")
    return SparseHashStemForwardBatch(
        inputs,
        results,
        bridge.lineage,
        bridge.checkpoint_sha256,
        bridge.topology_sha256,
        step_before,
        training_probes,
    )
end

struct SparseHashStemBackwardBatch
    hooks::HashStemLossHooks
    sparse_vjps::Vector{SparseDynamic3Layer.ThreeLayerVJP}
    lineage::SparseBridgeLineage
    checkpoint_sha256::String
    optimizer_step::UInt64
end

function _validate_forward_binding(
    bridge::SparseHashStemBridge,
    forward::SparseHashStemForwardBatch,
)
    _same_lineage(bridge.lineage, forward.lineage) || error("forward lineage changed")
    bridge.checkpoint_sha256 == forward.checkpoint_sha256 ||
        error("forward sparse checkpoint changed")
    bridge.topology_sha256 == forward.topology_sha256 || error("forward topology changed")
    _runtime_optimizer_step(bridge.runtime) == forward.optimizer_step ||
        error("sparse bank was updated between forward and VJP")
    return forward
end

function _validate_vjp_finite(vjp::SparseDynamic3Layer.ThreeLayerVJP)
    for value in (
        vjp.dtheta..., vjp.dhead, vjp.dbias, vjp.dbase_queries...,
        vjp.dcontext, vjp.dnext_hold,
    )
        all(isfinite, value) || error("sparse VJP contains non-finite values")
    end
    return vjp
end

"""Return full sparse VJPs plus exact q1/q2/q3/context HashStem hooks.

The hard selected IDs have zero derivative. `next_hold` cotangents are exposed
for ABI completeness, although the current HashStem output is a raw passthrough
and therefore has no trainable weights on that branch.
"""
function sparse_vjp_to_hashstem_hooks(
    bridge::SparseHashStemBridge,
    forward::SparseHashStemForwardBatch,
    sparse_output_cotangent::AbstractMatrix{Float32};
    auxiliary_target::AbstractMatrix{Float32}=zeros(
        Float32, length(forward.results), 10,
    ),
    auxiliary_mask::AbstractMatrix{Float32}=zeros(
        Float32, length(forward.results), 10,
    ),
)
    _validate_forward_binding(bridge, forward)
    batch = length(forward.results)
    size(sparse_output_cotangent) == (batch, SparseDynamic3Layer.OUTPUT_DIM) ||
        throw(DimensionMismatch("sparse output cotangent must be Bx22"))
    all(isfinite, sparse_output_cotangent) ||
        throw(ArgumentError("sparse output cotangent is non-finite"))
    size(auxiliary_target) == (batch, 10) ||
        throw(DimensionMismatch("auxiliary_target must be Bx10"))
    size(auxiliary_mask) == (batch, 10) ||
        throw(DimensionMismatch("auxiliary_mask must be Bx10"))
    all(isfinite, auxiliary_target) ||
        throw(ArgumentError("auxiliary_target is non-finite"))
    all(value -> value == 0.0f0 || value == 1.0f0, auxiliary_mask) ||
        throw(ArgumentError("auxiliary_mask must contain only 0f0/1f0"))

    q1 = Matrix{Float32}(undef, batch, 64)
    q2 = similar(q1)
    q3 = similar(q1)
    context = Matrix{Float32}(undef, batch, 22)
    next_hold = Matrix{Float32}(undef, batch, 42)
    vjps = Vector{SparseDynamic3Layer.ThreeLayerVJP}(undef, batch)
    @inbounds for row in 1:batch
        vjp = SparseDynamic3Layer.vjp_selected(
            bridge.runtime.model,
            forward.results[row].tape,
            view(sparse_output_cotangent, row, :),
        )
        _validate_vjp_finite(vjp)
        vjps[row] = vjp
        q1[row, :] .= vjp.dbase_queries[1]
        q2[row, :] .= vjp.dbase_queries[2]
        q3[row, :] .= vjp.dbase_queries[3]
        context[row, :] .= vjp.dcontext
        next_hold[row, :] .= vjp.dnext_hold
    end
    hooks = HashStemLossHooks(
        q1=q1,
        q2=q2,
        q3=q3,
        context=context,
        next_hold=next_hold,
        auxiliary_target=copy(auxiliary_target),
        auxiliary_mask=copy(auxiliary_mask),
    )
    _validate_loss_hooks(hooks, batch)
    return SparseHashStemBackwardBatch(
        hooks,
        vjps,
        bridge.lineage,
        bridge.checkpoint_sha256,
        forward.optimizer_step,
    )
end

function _capture_decay_clocks(runtime::SparseDynamic3Layer.ThreeLayerRuntime)
    return (
        global=ntuple(i -> runtime.bank_optimizers[i].global_log_decay, 3),
        last=ntuple(i -> copy(runtime.bank_optimizers[i].last_log_decay), 3),
    )
end

function _decay_witness(runtime, before, tape)
    layers = ntuple(3) do layer_id
        state = runtime.bank_optimizers[layer_id]
        ids = tape.ids[layer_id]
        selected = Set(Int.(ids))
        changed = findall(index -> !isequal(
            before.last[layer_id][index], state.last_log_decay[index],
        ), eachindex(state.last_log_decay))
        unexpected = filter(index -> !(index in selected), changed)
        pending_before = [
            exp(before.global[layer_id] - before.last[layer_id][Int(id)]) for id in ids
        ]
        scale_after = [SparseDynamic3Layer.logical_decay_scale(state, Int(id)) for id in ids]
        return (
            selected_ids=Int.(ids),
            pending_scale_before=pending_before,
            selected_scale_after=scale_after,
            changed_clock_ids=changed,
            unexpected_clock_changes=unexpected,
            selected_clocks_materialized=all(
                state.last_log_decay[Int(id)] == state.global_log_decay for id in ids
            ),
        )
    end
    return layers
end

function _route_record(result, decay)
    return (
        output_sha256=_primitive_array_sha256(result.output),
        selected_ids=ntuple(i -> Int.(result.tape.ids[i]), 3),
        routed_query_sha256=ntuple(
            i -> _primitive_array_sha256(result.tape.queries[i]), 3,
        ),
        activation_sha256=ntuple(
            i -> _primitive_array_sha256(result.tape.activation[i]), 3,
        ),
        scored_rows=result.telemetry.scored_rows,
        bucket_entries=result.telemetry.bucket_entries,
        rerank_macs=result.telemetry.rerank_macs,
        decay,
    )
end

"""Run two fresh production routes from one checkpoint for parity evidence.

The adapter performs no NPU call. The caller supplies a reference stem output
and an accelerator-perturbed output from the same byte-pinned HashStem weights.
Both sparse runtimes are independently restored so reference materialization
cannot perturb the accelerator branch. L2/L3 queries are taken from each
branch's production tape after its own preceding activations.
"""
function production_route_conformance_witness(
    checkpoint_path::AbstractString,
    lineage::SparseBridgeLineage,
    packed::AbstractMatrix{Float32},
    reference_hashstem_output::AbstractMatrix{Float32},
    accelerator_hashstem_output::AbstractMatrix{Float32};
    expected_checkpoint_sha256::AbstractString,
)
    size(packed, 1) == 1 || throw(DimensionMismatch("route witness requires B=1"))
    reference_bridge = load_sparse_hashstem_bridge(
        checkpoint_path, lineage; expected_checkpoint_sha256,
    )
    accelerator_bridge = load_sparse_hashstem_bridge(
        checkpoint_path, lineage; expected_checkpoint_sha256,
    )
    reference_input = only(three_layer_inputs_from_hashstem(
        packed, reference_hashstem_output,
    ))
    accelerator_input = only(three_layer_inputs_from_hashstem(
        packed, accelerator_hashstem_output,
    ))
    reference_before = _capture_decay_clocks(reference_bridge.runtime)
    accelerator_before = _capture_decay_clocks(accelerator_bridge.runtime)
    reference_result = SparseDynamic3Layer.route_forward!(
        reference_bridge.runtime,
        SparseDynamic3Layer.ThreeLayerWorkspace(reference_bridge.runtime),
        reference_input,
    )
    accelerator_result = SparseDynamic3Layer.route_forward!(
        accelerator_bridge.runtime,
        SparseDynamic3Layer.ThreeLayerWorkspace(accelerator_bridge.runtime),
        accelerator_input,
    )
    reference_decay = _decay_witness(
        reference_bridge.runtime, reference_before, reference_result.tape,
    )
    accelerator_decay = _decay_witness(
        accelerator_bridge.runtime, accelerator_before, accelerator_result.tape,
    )
    output_error = maximum(abs.(reference_result.output .- accelerator_result.output))
    route_ids_equal = ntuple(
        i -> reference_result.tape.ids[i] == accelerator_result.tape.ids[i], 3,
    )
    deeper_query_error = ntuple(2) do offset
        layer_id = offset + 1
        maximum(abs.(
            reference_result.tape.queries[layer_id] .-
            accelerator_result.tape.queries[layer_id],
        ))
    end
    payload = (
        schema=SPARSE_ROUTE_WITNESS_SCHEMA,
        implementation_provenance=UNEXECUTED_STATIC_ONLY,
        bridge_schema=SPARSE_HASHSTEM_BRIDGE_SCHEMA,
        production_route_semantics=PRODUCTION_ROUTE_SEMANTICS,
        sparse_model_id=lineage.sparse_model_id,
        sparse_variant=lineage.sparse_variant,
        sparse_active_counts=sparse_bridge_variant_spec(
            lineage.sparse_variant,
        ).active_counts,
        sparse_training_probes=sparse_bridge_variant_spec(
            lineage.sparse_variant,
        ).training_probes,
        sparse_bank_version=lineage.sparse_bank_version,
        sparse_checkpoint_sha256=reference_bridge.checkpoint_sha256,
        sparse_checkpoint_format=reference_bridge.checkpoint_format,
        sparse_checkpoint_version=reference_bridge.checkpoint_version,
        sparse_topology_sha256=reference_bridge.topology_sha256,
        sparse_source_closure_sha256=lineage.sparse_source_closure_sha256,
        bridge_source_sha256=reference_bridge.bridge_source_sha256,
        hashstem_model_id=lineage.hashstem_model_id,
        hashstem_master_version=lineage.hashstem_master_version,
        hashstem_snapshot_version=lineage.hashstem_snapshot_version,
        hashstem_weights_sha256=lineage.hashstem_weights_sha256,
        hashstem_normalization_sha256=lineage.hashstem_normalization_sha256,
        packed_input_sha256=_primitive_array_sha256(packed),
        reference_hashstem_output_sha256=_primitive_array_sha256(
            reference_hashstem_output,
        ),
        accelerator_hashstem_output_sha256=_primitive_array_sha256(
            accelerator_hashstem_output,
        ),
        reference=_route_record(reference_result, reference_decay),
        accelerator=_route_record(accelerator_result, accelerator_decay),
        exact_route_ids=route_ids_equal,
        all_route_ids_exact=all(route_ids_equal),
        maximum_sparse_output_absolute_error=output_error,
        deeper_routed_query_maximum_absolute_error=deeper_query_error,
        deeper_residuals_recomputed_by_production_route=true,
    )
    return merge(payload, (; witness_sha256=_sha256_text(JSON3.write(payload))))
end
