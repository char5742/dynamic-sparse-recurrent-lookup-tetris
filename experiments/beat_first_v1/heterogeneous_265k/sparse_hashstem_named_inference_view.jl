using Serialization

export SparseNamedInferenceView,
       SparseNamedForwardBatch,
       load_sparse_named_inference_view,
       route_sparse_named_inputs!

const SPARSE_NAMED_INFERENCE_VIEW_SCHEMA = "sparse3-named-inference-view-v1"

struct SparseNamedInferenceView
    schema::String
    variant::String
    source_checkpoint_path::String
    source_checkpoint_sha256::String
    source_state_sha256::String
    sparse_source_closure_sha256::String
    topology_sha256::String
    runtime::SparseDynamic3Layer.ThreeLayerRuntime
    metadata::Dict{String,Any}
end

struct SparseNamedForwardBatch
    view::SparseNamedInferenceView
    results::Vector{SparseDynamic3Layer.RoutedForwardResult}
    optimizer_step::UInt64
end

function _named_state_sha256(runtime::SparseDynamic3Layer.ThreeLayerRuntime)
    payload = (
        theta=ntuple(i -> runtime.model.layers[i].theta, 3),
        head=runtime.model.head,
        bias=runtime.model.bias,
        bank_optimizers=runtime.bank_optimizers,
        head_optimizer=runtime.head_optimizer,
    )
    io = IOBuffer()
    Serialization.serialize(io, payload)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _validate_ordinary_teacher_metadata(metadata::AbstractDict)
    canonical = Dict{String,Any}(string(key) => value for (key, value) in metadata)
    expected = Set((
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
    Set(keys(canonical)) == expected || error(
        "named inference view requires an ordinary trainer checkpoint",
    )
    String(canonical["julia_version"]) == string(VERSION) || error(
        "teacher checkpoint Julia version differs from the view process",
    )
    for field in (
        "source_sha256", "project_sha256", "manifest_sha256",
        "dataset_manifest_sha256",
    )
        _valid_sha256(lowercase(String(canonical[field]))) || error(
            "teacher checkpoint $field is not a valid SHA-256",
        )
    end
    String(canonical["ranking_source"]) == "q" || error(
        "teacher checkpoint ranking source must be q",
    )
    String(canonical["routing_policy"]) == SparseDynamic3Layer.ROUTING_POLICY ||
        error("teacher checkpoint routing policy differs from this runtime")
    Int(canonical["ranking_output_index"]) == 1 || error(
        "teacher checkpoint ranking output index must be one",
    )
    return canonical
end

"""Create one inference-only named active-width view from an ordinary checkpoint.

The source checkpoint is byte-pinned. Bank/head values and optimizer/event
clocks are deep-copied exactly; only `active_count` and the derived WTA indexes
change. No optimizer update or lazy materialization is performed here.
"""
function load_sparse_named_inference_view(
    checkpoint_path::AbstractString,
    expected_checkpoint_sha256::AbstractString,
    variant_name::Union{Symbol,AbstractString},
)
    expected = lowercase(String(expected_checkpoint_sha256))
    _valid_sha256(expected) || throw(ArgumentError("invalid checkpoint SHA-256"))
    source = abspath(checkpoint_path)
    isfile(source) || throw(ArgumentError("teacher checkpoint does not exist: $source"))
    _sha256_file(source) == expected || error("teacher checkpoint SHA-256 mismatch")
    loaded = SparseDynamic3Layer.load_checkpoint(source)
    SparseDynamic3Layer.parameter_count(loaded.runtime.model) ==
        SparseDynamic3Layer.TOTAL_PARAMETERS || error(
            "teacher checkpoint is not the exact 19,924,022-parameter bank",
        )
    metadata = _validate_ordinary_teacher_metadata(loaded.metadata)
    variant = sparse_bridge_variant_spec(variant_name)
    cloned = deepcopy(loaded.runtime)
    source_state = _named_state_sha256(cloned)
    layers = ntuple(3) do layer_id
        original = cloned.model.layers[layer_id]
        SparseDynamic3Layer.DynamicSparseLayer(
            original.theta,
            original.value_dim,
            variant.active_counts[layer_id],
            layer_id,
        )
    end
    model = SparseDynamic3Layer.ThreeLayerSparseModel(
        layers, cloned.model.head, cloned.model.bias,
    )
    indexes = ntuple(3) do layer_id
        SparseDynamic3Layer.WTALSHIndex.WTAIndex(
            model.layers[layer_id].theta;
            config=SparseDynamic3Layer._layer_wta_config(
                layer_id, variant.active_counts[layer_id],
            ),
            route_dims=SparseDynamic3Layer.ROUTE_DIM,
        )
    end
    runtime = SparseDynamic3Layer.ThreeLayerRuntime(
        model, indexes, cloned.bank_optimizers, cloned.head_optimizer,
    )
    _named_state_sha256(runtime) == source_state || error(
        "named view changed bank/head/optimizer state",
    )
    topology = SparseDynamic3Layer._checkpoint_topology(runtime)
    validated = _validate_production_topology(topology, runtime, variant)
    topology_sha = _sha256_text(JSON3.write(validated))
    return SparseNamedInferenceView(
        SPARSE_NAMED_INFERENCE_VIEW_SCHEMA,
        String(variant.name),
        source,
        expected,
        source_state,
        sparse_source_closure_sha256(),
        topology_sha,
        runtime,
        metadata,
    )
end

"""Evaluate independent inputs through the frozen production router."""
function route_sparse_named_inputs!(
    view::SparseNamedInferenceView,
    workspace::SparseDynamic3Layer.ThreeLayerWorkspace,
    inputs::AbstractVector{<:SparseDynamic3Layer.ThreeLayerInput},
)
    isempty(inputs) && throw(ArgumentError("empty sparse input batch"))
    step_before = _runtime_optimizer_step(view.runtime)
    results = Vector{SparseDynamic3Layer.RoutedForwardResult}(undef, length(inputs))
    for row in eachindex(inputs)
        results[row] = SparseDynamic3Layer.route_forward!(
            view.runtime,
            workspace,
            inputs[row];
            training_probes=(0, 0, 0),
            probe_token=UInt64(row - 1),
        )
        all(isfinite, results[row].output) || error(
            "production sparse output is non-finite",
        )
    end
    _runtime_optimizer_step(view.runtime) == step_before || error(
        "production inference unexpectedly advanced optimizer clocks",
    )
    return SparseNamedForwardBatch(view, results, step_before)
end
