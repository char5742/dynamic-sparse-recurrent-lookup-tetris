import Serialization
import SHA

const RESIDUAL_LOOKUP_CHECKPOINT_FORMAT =
    "TETRIS_RESIDUAL_LOOKUP_SLIDE_R0_EVENTTIME_V1"
const RESIDUAL_LOOKUP_CHECKPOINT_VERSION = 1

struct ResidualLookupCheckpointEnvelope
    format::String
    version::Int
    topology::NamedTuple
    router_seeds::NTuple{3,UInt64}
    model::Any
    optimizer::ResidualLookupOptimizerState
    rng_state::Any
    training_state::Any
    metadata::Dict{String,Any}
end

function _normalize_router_seeds(router_seeds)
    router_seeds isa Tuple || throw(ArgumentError(
        "router_seeds must be a three-element tuple",
    ))
    length(router_seeds) == 3 || throw(DimensionMismatch(
        "router_seeds must contain one seed per residual block",
    ))
    return ntuple(3) do layer
        seed = router_seeds[layer]
        seed isa Integer || throw(ArgumentError(
            "router seed $layer is not an integer",
        ))
        UInt64(seed)
    end
end

function _assert_residual_lookup_checkpoint_state(
    model,
    optimizer::ResidualLookupOptimizerState;
    full_validation::Bool=true,
)
    1 <= model.tables_per_block <= TABLES_PER_BLOCK || error(
        "checkpoint model table count is outside live geometry",
    )
    length(model.banks) == BLOCKS || error(
        "checkpoint model block count differs from live geometry",
    )
    expected_bank_shape = (
        VALUE_DIM,
        ROWS_PER_TABLE * model.tables_per_block,
    )
    all(bank -> size(bank) == expected_bank_shape, model.banks) || error(
        "checkpoint bank shape differs from live table-major geometry",
    )
    size(model.head) == (OUTPUT_DIM, CARRIER_DIM) || error(
        "checkpoint head shape differs from live geometry",
    )
    length(model.bias) == OUTPUT_DIM || error(
        "checkpoint bias width differs from live geometry",
    )
    length(model.alpha_logits) == BLOCKS || error(
        "checkpoint alpha width differs from live geometry",
    )
    _assert_residual_lookup_optimizer_layout(model, optimizer)
    frozen_topology = topology(model)
    frozen_topology isa NamedTuple || error(
        "Residual Lookup-SLIDE topology must be a NamedTuple",
    )
    for layer in 1:3
        bank = model.banks[layer]
        state = optimizer.bank_states[layer]
        _validated_adamw_hyperparameters(
            state.beta1,
            state.beta2,
            state.epsilon,
            state.learning_rate,
            state.weight_decay,
        )
        isfinite(state.global_log_decay) || error(
            "lookup bank $layer has a non-finite global decay clock",
        )
        if full_validation
            all(isfinite, bank) || error("lookup bank $layer is non-finite")
            all(isfinite, state.m) || error(
                "lookup bank $layer first moment is non-finite",
            )
            all(isfinite, state.v) || error(
                "lookup bank $layer second moment is non-finite",
            )
            all(value -> value >= 0.0f0, state.v) || error(
                "lookup bank $layer second moment is negative",
            )
            for column in axes(bank, 2)
                state.last_event_step[column] <= state.global_step || error(
                    "lookup bank $layer column $column has a future event clock",
                )
                state.last_log_decay[column] + 32 * eps(Float64) >=
                    state.global_log_decay || error(
                        "lookup bank $layer column $column decay clock is ahead",
                    )
                state.event_count[column] <= state.last_event_step[column] || error(
                    "lookup bank $layer column $column has impossible event time",
                )
            end
        end
    end
    if full_validation
        all(isfinite, model.head) && all(isfinite, model.bias) || error(
            "dense head is non-finite",
        )
        all(isfinite, model.alpha_logits) || error("alpha logits are non-finite")
        head_state = optimizer.head_state
        alpha_state = optimizer.alpha_state
        _validated_adamw_hyperparameters(
            head_state.beta1,
            head_state.beta2,
            head_state.epsilon,
            head_state.learning_rate,
            head_state.weight_decay,
        )
        _validated_adamw_hyperparameters(
            alpha_state.beta1,
            alpha_state.beta2,
            alpha_state.epsilon,
            alpha_state.learning_rate,
            alpha_state.weight_decay,
        )
        all(isfinite, head_state.m_head) && all(isfinite, head_state.v_head) &&
            all(isfinite, head_state.m_bias) && all(isfinite, head_state.v_bias) ||
            error("dense-head optimizer state is non-finite")
        all(value -> value >= 0.0f0, head_state.v_head) &&
            all(value -> value >= 0.0f0, head_state.v_bias) ||
            error("dense-head second moment is negative")
        all(isfinite, alpha_state.m) && all(isfinite, alpha_state.v) || error(
            "dense-alpha optimizer state is non-finite",
        )
        all(value -> value >= 0.0f0, alpha_state.v) || error(
            "dense-alpha second moment is negative",
        )
    end
    return frozen_topology
end

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""Publish one fresh exact-continuation checkpoint by same-directory rename.

The envelope binds the model-derived topology, immutable router seeds, all
sparse and dense optimizer clocks/moments, a copied RNG object, and arbitrary
caller training state.  Saving does not materialize lazy lookup decay.
"""
function save_residual_lookup_checkpoint(
    path::AbstractString,
    model,
    optimizer::ResidualLookupOptimizerState;
    router_seeds=ROUTER_SEEDS,
    rng,
    training_state=nothing,
    metadata::AbstractDict=Dict{String,Any}(),
    full_validation::Bool=true,
)
    frozen_topology = _assert_residual_lookup_checkpoint_state(
        model,
        optimizer;
        full_validation,
    )
    seeds = _normalize_router_seeds(router_seeds)
    seeds == _normalize_router_seeds(ROUTER_SEEDS) || error(
        "checkpoint router seeds differ from the live fixed router",
    )
    rng isa Random.Xoshiro || throw(ArgumentError(
        "an owned Random.Xoshiro state is required for exact continuation",
    ))
    target = abspath(path)
    ispath(target) && throw(ArgumentError(
        "checkpoint target already exists; publish a fresh immutable path",
    ))
    mkpath(dirname(target))
    envelope = ResidualLookupCheckpointEnvelope(
        RESIDUAL_LOOKUP_CHECKPOINT_FORMAT,
        RESIDUAL_LOOKUP_CHECKPOINT_VERSION,
        frozen_topology,
        seeds,
        model,
        optimizer,
        deepcopy(rng),
        deepcopy(training_state),
        Dict{String,Any}(string(key) => value for (key, value) in metadata),
    )
    temporary = target * ".tmp." * string(getpid()) * "." * string(time_ns())
    ispath(temporary) && error("checkpoint temporary path collision")
    committed = false
    try
        open(temporary, "w") do io
            Serialization.serialize(io, envelope)
            flush(io)
        end
        # The temporary and target paths share a directory.  With a fresh
        # target, this rename is the only publication point and never deletes
        # a previous scientific checkpoint.
        mv(temporary, target; force=false)
        committed = true
    finally
        if !committed && isfile(temporary)
            rm(temporary; force=true)
        end
    end
    return (;
        path=target,
        bytes=filesize(target),
        sha256=_sha256_file(target),
        topology=frozen_topology,
        router_seeds=seeds,
        optimizer_step=Int(optimizer.step),
    )
end

function load_residual_lookup_checkpoint(
    path::AbstractString;
    expected_bytes::Integer,
    expected_sha256::AbstractString,
    expected_topology=nothing,
    expected_router_seeds=nothing,
    full_validation::Bool=true,
)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("checkpoint does not exist: $source"))
    expected_bytes >= 0 || throw(ArgumentError(
        "expected_bytes must be nonnegative",
    ))
    filesize(source) == Int(expected_bytes) || error(
        "checkpoint byte count differs from its receipt",
    )
    observed_sha256 = _sha256_file(source)
    expected = strip(String(expected_sha256))
    occursin(r"^[0-9a-f]{64}$", expected) || throw(ArgumentError(
        "expected_sha256 must be 64 lowercase hexadecimal characters",
    ))
    observed_sha256 == expected || error(
        "checkpoint SHA-256 differs from its receipt",
    )
    envelope = open(source, "r") do io
        value = Serialization.deserialize(io)
        eof(io) || error("checkpoint contains trailing bytes")
        value
    end
    envelope isa ResidualLookupCheckpointEnvelope || error(
        "checkpoint is not a Residual Lookup-SLIDE envelope",
    )
    envelope.format == RESIDUAL_LOOKUP_CHECKPOINT_FORMAT || error(
        "checkpoint format mismatch",
    )
    envelope.version == RESIDUAL_LOOKUP_CHECKPOINT_VERSION || error(
        "checkpoint version mismatch",
    )
    restored_topology = _assert_residual_lookup_checkpoint_state(
        envelope.model,
        envelope.optimizer;
        full_validation,
    )
    restored_topology == envelope.topology || error(
        "checkpoint topology metadata differs from restored model arrays",
    )
    normalized_seeds = _normalize_router_seeds(envelope.router_seeds)
    normalized_seeds == envelope.router_seeds || error(
        "checkpoint router seeds are not canonical UInt64 values",
    )
    normalized_seeds == _normalize_router_seeds(ROUTER_SEEDS) || error(
        "checkpoint router seeds differ from the live fixed router",
    )
    if expected_topology !== nothing
        expected_topology == envelope.topology || error(
            "checkpoint topology differs from the requested continuation",
        )
    end
    if expected_router_seeds !== nothing
        expected = _normalize_router_seeds(expected_router_seeds)
        expected == envelope.router_seeds || error(
            "checkpoint router seeds differ from the requested continuation",
        )
    end
    envelope.rng_state isa Random.Xoshiro || error(
        "checkpoint lacks the owned Random.Xoshiro required for exact continuation",
    )
    return (;
        model=envelope.model,
        optimizer=envelope.optimizer,
        rng=envelope.rng_state,
        training_state=envelope.training_state,
        metadata=envelope.metadata,
        topology=envelope.topology,
        router_seeds=envelope.router_seeds,
        path=source,
        bytes=filesize(source),
        sha256=observed_sha256,
    )
end
