const CHECKPOINT_FORMAT = "TETRIS_SLIDE_3L_EVENTTIME_SPARSE_V1"
const CHECKPOINT_VERSION = 2

struct ThreeLayerCheckpointEnvelope
    format::String
    version::Int
    topology::NamedTuple
    runtime::ThreeLayerRuntime
    training_state::Any
    metadata::Dict{String,Any}
end

function _checkpoint_topology(runtime::ThreeLayerRuntime)
    return (
        routing_policy=ROUTING_POLICY,
        route_dim=ROUTE_DIM,
        value_dims=ntuple(i -> runtime.model.layers[i].value_dim, 3),
        row_dims=ntuple(i -> size(runtime.model.layers[i].theta, 1), 3),
        neuron_counts=ntuple(i -> size(runtime.model.layers[i].theta, 2), 3),
        active_counts=ntuple(i -> runtime.model.layers[i].active_count, 3),
        output_dim=OUTPUT_DIM,
        latent_dim=LATENT_DIM,
        parameter_count=parameter_count(runtime.model),
    )
end

function _validate_index_contents(layer::DynamicSparseLayer, index::WTALSHIndex.WTAIndex)
    length(index.positions) ==
        index.config.m * index.config.K * index.config.L ||
        error("checkpoint WTA sampled-coordinate array is malformed")
    length(index.head) == index.config.L * index.bucket_count ||
        error("checkpoint WTA head array is malformed")
    seen = falses(index.neurons)
    for table in 1:index.config.L
        fill!(seen, false)
        for code in 0:(index.bucket_count - 1)
            bucket = WTALSHIndex._bucket_slot(index, code, table)
            neuron = @inbounds index.head[bucket]
            previous = Int32(0)
            traversed = 0
            while neuron != 0
                id = Int(neuron)
                1 <= id <= index.neurons || error("checkpoint WTA link has invalid ID")
                !seen[id] || error("checkpoint WTA chain contains a duplicate/cycle")
                seen[id] = true
                slot = WTALSHIndex._slot(index, neuron, table)
                @inbounds index.prev[slot] == previous ||
                    error("checkpoint WTA backward link is inconsistent")
                @inbounds Int(index.codes[slot]) == code ||
                    error("checkpoint WTA neuron is linked under the wrong code")
                previous = neuron
                neuron = @inbounds index.next[slot]
                traversed += 1
                traversed <= index.neurons || error("checkpoint WTA chain cycles")
            end
        end
        all(seen) || error("checkpoint WTA table does not contain every neuron exactly once")
        for neuron in 1:index.neurons
            slot = WTALSHIndex._slot(index, neuron, table)
            expected = WTALSHIndex._theta_code(index, layer.theta, neuron, table)
            @inbounds Int(index.codes[slot]) == expected || error(
                "checkpoint WTA code is stale relative to route-key weights",
            )
        end
    end
    return nothing
end

function _validate_runtime(runtime::ThreeLayerRuntime; full_validation::Bool=true)
    for layer_id in 1:3
        layer = runtime.model.layers[layer_id]
        index = runtime.indexes[layer_id]
        optimizer = runtime.bank_optimizers[layer_id]
        _assert_optimizer_layout(layer.theta, optimizer)
        index.neurons == size(layer.theta, 2) || error(
            "checkpoint layer $layer_id index width differs from theta",
        )
        index.route_dims == ROUTE_DIM || error(
            "checkpoint layer $layer_id index route width differs",
        )
        length(index.positions) ==
            index.config.m * index.config.K * index.config.L || error(
                "checkpoint layer $layer_id sampled-coordinate array is malformed",
            )
        length(index.head) == index.config.L * index.bucket_count || error(
            "checkpoint layer $layer_id WTA head array is malformed",
        )
        length(index.next) == index.config.L * index.neurons || error(
            "checkpoint layer $layer_id intrusive next array is malformed",
        )
        length(index.prev) == length(index.next) || error(
            "checkpoint layer $layer_id intrusive prev array is malformed",
        )
        length(index.codes) == length(index.next) || error(
            "checkpoint layer $layer_id code array is malformed",
        )
        if full_validation
            _validate_index_contents(layer, index)
            for neuron in axes(layer.theta, 2)
                optimizer.last_event_step[neuron] <= optimizer.global_step || error(
                    "checkpoint layer $layer_id has a future per-row event clock",
                )
                optimizer.last_log_decay[neuron] + 32 * eps(Float64) >=
                    optimizer.global_log_decay || error(
                    "checkpoint layer $layer_id row decay clock is ahead of global decay",
                )
            end
        end
    end
    clocks = ntuple(i -> runtime.bank_optimizers[i].global_step, 3)
    clocks[1] == clocks[2] == clocks[3] || error(
        "checkpoint sparse-layer global clocks diverged",
    )
    runtime.head_optimizer.step == clocks[1] || error(
        "checkpoint dense-head and sparse-layer steps diverged",
    )
    return runtime
end

"""Atomically serialize the distinct three-layer event-time state.

Lazy physical state is preserved exactly: saving never materializes an
inactive bank row.  The envelope includes all three intrusive WTA indexes,
theta/m/v/event clocks, the dense head optimizer, and caller-owned sampler/RNG
state in `training_state`.
"""
function save_checkpoint(
    path::AbstractString,
    runtime::ThreeLayerRuntime;
    training_state=nothing,
    metadata::AbstractDict=Dict{String,Any}(),
    full_validation::Bool=true,
)
    _validate_runtime(runtime; full_validation)
    target = abspath(path)
    mkpath(dirname(target))
    envelope = ThreeLayerCheckpointEnvelope(
        CHECKPOINT_FORMAT,
        CHECKPOINT_VERSION,
        _checkpoint_topology(runtime),
        runtime,
        training_state,
        Dict{String,Any}(string(key) => value for (key, value) in metadata),
    )
    temporary = target * ".tmp." * string(getpid()) * "." * string(time_ns())
    committed = false
    try
        open(temporary, "w") do io
            Serialization.serialize(io, envelope)
            flush(io)
        end
        mv(temporary, target; force=true)
        committed = true
    finally
        if !committed && isfile(temporary)
            rm(temporary; force=true)
        end
    end
    return target
end

function load_checkpoint(path::AbstractString; full_validation::Bool=true)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("checkpoint does not exist: $source"))
    envelope = open(source, "r") do io
        Serialization.deserialize(io)
    end
    envelope isa ThreeLayerCheckpointEnvelope || error(
        "checkpoint type is not the three-layer sparse envelope",
    )
    envelope.format == CHECKPOINT_FORMAT || error(
        "checkpoint format mismatch: $(envelope.format)",
    )
    envelope.version == CHECKPOINT_VERSION || error(
        "checkpoint version mismatch: $(envelope.version)",
    )
    runtime = _validate_runtime(envelope.runtime; full_validation)
    envelope.topology == _checkpoint_topology(runtime) || error(
        "checkpoint topology metadata differs from restored arrays",
    )
    return (
        runtime=runtime,
        training_state=envelope.training_state,
        metadata=envelope.metadata,
        topology=envelope.topology,
    )
end
