using Dates
using JLD2
using JSON3
using Lux
using NPZ
using Optimisers
using Random
using SHA

using .BeatFirstFixedShapeBackend

export LearnedHashStem,
       HashStemLossHooks,
       HashStemLayoutEvidence,
       HashStemMasterTrainer,
       assemble_hashstem_cotangent!,
       hashstem_master_batch,
       hashstem_master_backend_status,
       init_hashstem_master,
       train_hashstem_master_step!,
       host_hashstem_weights,
       save_hashstem_master_checkpoint!,
       load_hashstem_master_checkpoint,
       resume_hashstem_master,
       export_hashstem_snapshot_source!

const HASHSTEM_MASTER_CHECKPOINT_SCHEMA = "learned-hashstem-master-checkpoint-v1"
const HASHSTEM_SNAPSHOT_SOURCE_SCHEMA = "learned-hashstem-snapshot-source-v1"
const UNEXECUTED_STATIC_ONLY = "UNEXECUTED_STATIC_ONLY"
const VALIDATED_RUNTIME = "VALIDATED_RUNTIME"

"""Frozen normalization carried as Lux state, never as an optimizer parameter."""
struct HashStemNormalization <: Lux.AbstractLuxLayer
    input_mean::Vector{Float32}
    input_inv_std::Vector{Float32}
end

Lux.initialparameters(::AbstractRNG, ::HashStemNormalization) = NamedTuple()
Lux.initialstates(::AbstractRNG, layer::HashStemNormalization) = (;
    input_mean=copy(layer.input_mean),
    input_inv_std=copy(layer.input_inv_std),
)

function (layer::HashStemNormalization)(packed, ps, st)
    normalized = (packed .- reshape(st.input_mean, 1, :)) .*
                 reshape(st.input_inv_std, 1, :)
    return normalized, st
end

"""Canonical trainable HashStem. Input/output are exactly `[B,559] -> [B,256]`.

The native Lux convolution tensors are converted explicitly by
`host_hashstem_weights` to the immutable OpenVINO schema. Snapshot export is
forbidden until an independent scalar-oracle parity witness is supplied.
"""
struct LearnedHashStem <: Lux.AbstractLuxContainerLayer{
    (:normalization, :conv3, :depthwise5x1, :pointwise, :pool, :dense)
}
    normalization
    conv3
    depthwise5x1
    pointwise
    pool
    dense
end

function LearnedHashStem(
    input_mean::AbstractVector{<:Real}=zeros(Float32, HASHSTEM_INPUT_FEATURES),
    input_inv_std::AbstractVector{<:Real}=ones(Float32, HASHSTEM_INPUT_FEATURES),
)
    mean_f32 = Float32.(input_mean)
    inv_std_f32 = Float32.(input_inv_std)
    length(mean_f32) == HASHSTEM_INPUT_FEATURES ||
        throw(DimensionMismatch("input_mean must have length 559"))
    length(inv_std_f32) == HASHSTEM_INPUT_FEATURES ||
        throw(DimensionMismatch("input_inv_std must have length 559"))
    all(isfinite, mean_f32) || throw(ArgumentError("input_mean is non-finite"))
    all(value -> isfinite(value) && value > 0.0f0, inv_std_f32) ||
        throw(ArgumentError("input_inv_std must be finite and positive"))
    return LearnedHashStem(
        HashStemNormalization(mean_f32, inv_std_f32),
        Conv(
            (3, 3), 2 => 16, relu;
            pad=SamePad(), cross_correlation=true,
        ),
        Conv(
            (5, 1), 16 => 16, relu;
            pad=SamePad(), groups=16, cross_correlation=true,
        ),
        Conv((1, 1), 16 => 32, relu; cross_correlation=true),
        MeanPool((4, 2); stride=(4, 2)),
        Dense(HASHSTEM_DENSE_INPUTS => HASHSTEM_LEARNED_OUTPUTS),
    )
end

function (model::LearnedHashStem)(packed, ps, st)
    size(packed, 2) == HASHSTEM_INPUT_FEATURES ||
        throw(DimensionMismatch("HashStem input must be Bx559"))
    batch = size(packed, 1)
    normalized, normalization_st = model.normalization(
        packed, ps.normalization, st.normalization,
    )
    # The packed ABI is row-fastest inside each column. Julia's first index is
    # contiguous, so 480xB reshapes directly to row,column,channel,batch.
    board = reshape(permutedims(normalized[:, 1:480]), 24, 10, 2, batch)
    value, conv3_st = model.conv3(board, ps.conv3, st.conv3)
    value, depthwise_st = model.depthwise5x1(
        value, ps.depthwise5x1, st.depthwise5x1,
    )
    value, pointwise_st = model.pointwise(value, ps.pointwise, st.pointwise)
    value, pool_st = model.pool(value, ps.pool, st.pool)
    size(value) == (6, 5, 32, batch) ||
        throw(DimensionMismatch("HashStem pool must produce 6x5x32xB"))
    # OpenVINO and the scalar oracle flatten pooled values column,row,channel.
    pooled = reshape(permutedims(value, (2, 1, 3, 4)), 960, batch)
    auxiliary = permutedims(normalized[:, 523:559])
    next_hold_normalized = permutedims(normalized[:, 481:522])
    learned, dense_st = model.dense(
        vcat(pooled, auxiliary, next_hold_normalized), ps.dense, st.dense,
    )
    raw_next_hold = permutedims(packed[:, 481:522])
    output = permutedims(vcat(learned, raw_next_hold))
    next_state = (;
        normalization=normalization_st,
        conv3=conv3_st,
        depthwise5x1=depthwise_st,
        pointwise=pointwise_st,
        pool=pool_st,
        dense=dense_st,
    )
    return output, next_state
end

"""External selected-only sparse VJP plus auxiliary supervision hooks."""
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
        size(value) == (batch, width) ||
            throw(DimensionMismatch("$name must be $(batch)x$(width)"))
        all(isfinite, value) || throw(ArgumentError("$name is non-finite"))
    end
    all(value -> value == 0.0f0 || value == 1.0f0, hooks.auxiliary_mask) ||
        throw(ArgumentError("auxiliary_mask must contain only 0f0/1f0"))
    return hooks
end

function assemble_hashstem_cotangent!(
    destination::Matrix{Float32}, hooks::HashStemLossHooks,
)
    batch = size(destination, 1)
    size(destination) == (batch, HASHSTEM_OUTPUT_FEATURES) ||
        throw(DimensionMismatch("cotangent destination must be Bx256"))
    _validate_loss_hooks(hooks, batch)
    destination[:, QUERY_1_RANGE] .= hooks.q1
    destination[:, QUERY_2_RANGE] .= hooks.q2
    destination[:, QUERY_3_RANGE] .= hooks.q3
    destination[:, CONTEXT_RANGE] .= hooks.context
    destination[:, NEXT_HOLD_PASSTHROUGH_RANGE] .= hooks.next_hold
    return destination
end

struct HashStemObjective
    auxiliary_weight::Float32
    huber_beta::Float32
end

function (objective::HashStemObjective)(model, ps, st, batch)
    output, next_state = model(batch.inputs.packed, ps, st)
    cotangent = batch.targets.output_cotangent
    # The caller supplies a true dL/doutput; do not silently rescale it.
    vjp_surrogate = sum(output .* cotangent)
    prediction = output[:, AUXILIARY_RANGE]
    difference = prediction .- batch.targets.auxiliary_target
    absolute = abs.(difference)
    beta = objective.huber_beta
    huber = ifelse.(
        absolute .<= beta,
        0.5f0 .* difference .* difference ./ beta,
        absolute .- 0.5f0 * beta,
    )
    weighted = huber .* batch.targets.auxiliary_mask
    denominator = max(sum(batch.targets.auxiliary_mask), 1.0f0)
    auxiliary_loss = sum(weighted) / denominator
    loss = vjp_surrogate + objective.auxiliary_weight * auxiliary_loss
    statistics = (;
        vjp_surrogate,
        auxiliary_loss,
        output_mean=sum(output) / Float32(length(output)),
    )
    return loss, next_state, statistics
end

function hashstem_master_batch(
    packed::Matrix{Float32}, hooks::HashStemLossHooks;
    state_mask::Matrix{Float32}=ones(Float32, 1, size(packed, 1)),
)
    batch = size(packed, 1)
    size(packed) == (batch, HASHSTEM_INPUT_FEATURES) ||
        throw(DimensionMismatch("packed input must be Bx559"))
    all(isfinite, packed) || throw(ArgumentError("packed input is non-finite"))
    size(state_mask) == (1, batch) ||
        throw(DimensionMismatch("state_mask must be 1xB"))
    all(value -> value == 0.0f0 || value == 1.0f0, state_mask) ||
        throw(ArgumentError("state_mask must contain only 0f0/1f0"))
    _validate_loss_hooks(hooks, batch)
    cotangent = zeros(Float32, batch, HASHSTEM_OUTPUT_FEATURES)
    assemble_hashstem_cotangent!(cotangent, hooks)
    cotangent .*= permutedims(state_mask)
    return (;
        inputs=(; packed=copy(packed)),
        targets=(;
            output_cotangent=cotangent,
            auxiliary_target=copy(hooks.auxiliary_target),
            auxiliary_mask=copy(hooks.auxiliary_mask) .* permutedims(state_mask),
        ),
        mask=copy(state_mask),
    )
end

"""Fail-closed backend inventory. iGPU is an interface, not a support claim."""
function hashstem_master_backend_status(backend::Symbol)
    backend === :cpu && return (;
        available=true,
        implementation="Lux+Reactant+EnzymeMLIR CPU",
        precision="FP32 master",
        status=UNEXECUTED_STATIC_ONLY,
    )
    backend === :igpu && return (;
        available=false,
        implementation="deferred optional adapter",
        precision="no precision authorized",
        status="FAIL_CLOSED_NO_VALIDATED_WINDOWS_IGPU_BACKEND",
    )
    throw(ArgumentError("backend must be :cpu or :igpu"))
end

mutable struct HashStemMasterTrainer
    model_id::String
    backend::Symbol
    model::LearnedHashStem
    learner::Any
    master_version::UInt64
    last_snapshot_version::UInt64
    last_snapshot_master_version::UInt64
    batch_size::Int
    learning_rate::Float32
    weight_decay::Float32
    auxiliary_weight::Float32
    huber_beta::Float32
    initialization_seed::UInt64
end

function _empty_hooks(batch::Int)
    return HashStemLossHooks(
        q1=zeros(Float32, batch, 64),
        q2=zeros(Float32, batch, 64),
        q3=zeros(Float32, batch, 64),
        context=zeros(Float32, batch, 22),
        next_hold=zeros(Float32, batch, 42),
        auxiliary_target=zeros(Float32, batch, 10),
        auxiliary_mask=zeros(Float32, batch, 10),
    )
end

function init_hashstem_master(;
    model_id::AbstractString,
    seed::UInt64,
    batch_size::Int=HASHSTEM_BATCH,
    input_mean::AbstractVector{<:Real}=zeros(Float32, HASHSTEM_INPUT_FEATURES),
    input_inv_std::AbstractVector{<:Real}=ones(Float32, HASHSTEM_INPUT_FEATURES),
    learning_rate::Float32=3.0f-4,
    weight_decay::Float32=1.0f-4,
    auxiliary_weight::Float32=1.0f0,
    huber_beta::Float32=1.0f0,
    backend::Symbol=:cpu,
    restore=nothing,
    master_version::Integer=0,
    last_snapshot_version::Integer=0,
    last_snapshot_master_version::Integer=0,
)
    isempty(model_id) && throw(ArgumentError("model_id must not be empty"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    status = hashstem_master_backend_status(backend)
    status.available || error(status.status)
    learning_rate > 0.0f0 || throw(ArgumentError("learning_rate must be positive"))
    weight_decay >= 0.0f0 || throw(ArgumentError("weight_decay must be nonnegative"))
    auxiliary_weight >= 0.0f0 || throw(ArgumentError("auxiliary_weight must be nonnegative"))
    huber_beta > 0.0f0 || throw(ArgumentError("huber_beta must be positive"))
    restore === nothing && master_version != 0 &&
        throw(ArgumentError("a fresh master must start at version zero"))
    restore === nothing && (last_snapshot_version != 0 || last_snapshot_master_version != 0) &&
        throw(ArgumentError("a fresh master cannot claim a published snapshot"))
    master_version >= 0 || throw(ArgumentError("master_version must be nonnegative"))
    last_snapshot_version >= 0 ||
        throw(ArgumentError("last_snapshot_version must be nonnegative"))
    last_snapshot_master_version >= 0 ||
        throw(ArgumentError("last_snapshot_master_version must be nonnegative"))
    (last_snapshot_version == 0) == (last_snapshot_master_version == 0) ||
        throw(ArgumentError("snapshot version/master-version lineage must be a zero pair"))
    last_snapshot_master_version <= master_version ||
        throw(ArgumentError("published snapshot cannot come from a future master"))
    model = LearnedHashStem(input_mean, input_inv_std)
    parameters, states = Lux.setup(Xoshiro(seed), model)
    objective = HashStemObjective(auxiliary_weight, huber_beta)
    template = hashstem_master_batch(
        zeros(Float32, batch_size, HASHSTEM_INPUT_FEATURES),
        _empty_hooks(batch_size),
    )
    optimiser = Optimisers.AdamW(
        learning_rate, (0.9f0, 0.999f0), weight_decay,
    )
    learner = BeatFirstFixedShapeBackend.init_backend(
        model,
        parameters,
        states,
        optimiser,
        objective,
        template;
        max_candidates=1,
        backend="cpu",
        restore,
    )
    return HashStemMasterTrainer(
        String(model_id),
        backend,
        model,
        learner,
        UInt64(master_version),
        UInt64(last_snapshot_version),
        UInt64(last_snapshot_master_version),
        batch_size,
        learning_rate,
        weight_decay,
        auxiliary_weight,
        huber_beta,
        seed,
    )
end

function train_hashstem_master_step!(
    trainer::HashStemMasterTrainer,
    packed::Matrix{Float32},
    hooks::HashStemLossHooks;
    state_mask::Matrix{Float32}=ones(Float32, 1, size(packed, 1)),
)
    trainer.backend === :cpu || error("only the fail-closed CPU master is authorized")
    trainer.last_snapshot_version == 0 || error(
        "online snapshot-bound master updates are not implemented; drain/coordinator proof required",
    )
    batch = hashstem_master_batch(packed, hooks; state_mask)
    return _apply_hashstem_master_batch!(trainer, batch)
end

"""Internal one-update primitive; online callers require a lifecycle barrier proof."""
function _apply_hashstem_master_batch!(trainer::HashStemMasterTrainer, batch)
    trainer.backend === :cpu || error("only the fail-closed CPU master is authorized")
    trainer.master_version == typemax(UInt64) && error("master version overflow")
    result = BeatFirstFixedShapeBackend.train_step!(trainer.learner, batch)
    trainer.master_version += UInt64(1)
    return merge(result, (; master_version=trainer.master_version))
end

function _require_shape(value, shape, label)
    size(value) == shape || throw(DimensionMismatch("$label must be $shape"))
    return Float32.(Array(value))
end

"""Convert Lux host parameters to the exact ten-array OpenVINO NPZ schema."""
function host_hashstem_weights(trainer::HashStemMasterTrainer)
    checkpoint = BeatFirstFixedShapeBackend.host_checkpoint(trainer.learner)
    ps, st = checkpoint.parameters, checkpoint.states
    conv3_native = _require_shape(ps.conv3.weight, (3, 3, 2, 16), "Lux conv3")
    depthwise_native = _require_shape(
        ps.depthwise5x1.weight, (5, 1, 1, 16), "Lux depthwise5x1",
    )
    pointwise_native = _require_shape(
        ps.pointwise.weight, (1, 1, 16, 32), "Lux pointwise",
    )
    dense_native = _require_shape(ps.dense.weight, (214, 1039), "Lux dense")
    weights = HashStemWeights(
        permutedims(conv3_native, (4, 3, 1, 2)),
        vec(Float32.(Array(ps.conv3.bias))),
        reshape(permutedims(depthwise_native, (4, 3, 1, 2)), 16, 5),
        vec(Float32.(Array(ps.depthwise5x1.bias))),
        reshape(permutedims(pointwise_native, (4, 3, 1, 2)), 32, 16),
        vec(Float32.(Array(ps.pointwise.bias))),
        dense_native,
        vec(Float32.(Array(ps.dense.bias))),
        vec(Float32.(Array(st.normalization.input_mean))),
        vec(Float32.(Array(st.normalization.input_inv_std))),
    )
    return validate_hashstem_weights(weights), checkpoint
end

function _weights_dictionary(weights::HashStemWeights)
    return Dict{String,Array}(
        "conv3" => weights.conv3,
        "conv3_bias" => weights.conv3_bias,
        "depthwise5x1" => weights.depthwise5x1,
        "depthwise_bias" => weights.depthwise_bias,
        "pointwise" => weights.pointwise,
        "pointwise_bias" => weights.pointwise_bias,
        "dense" => weights.dense,
        "dense_bias" => weights.dense_bias,
        "input_mean" => weights.input_mean,
        "input_inv_std" => weights.input_inv_std,
    )
end

_sha256_file(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function _normalization_sha256(weights::HashStemWeights)
    Base.ENDIAN_BOM == 0x04030201 || error("little-endian host required for NPZ binding")
    bytes = vcat(
        collect(reinterpret(UInt8, weights.input_mean)),
        collect(reinterpret(UInt8, weights.input_inv_std)),
    )
    return bytes2hex(SHA.sha256(bytes))
end

function _utc_now()
    return string(Dates.now(Dates.UTC)) * "Z"
end

function _write_json(path::AbstractString, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end

function save_hashstem_master_checkpoint!(
    trainer::HashStemMasterTrainer,
    destination::AbstractString;
    source_checkpoint_sha256::AbstractString=repeat("0", 64),
)
    trainer.backend === :cpu || error("unsupported master backend")
    _valid_sha256(source_checkpoint_sha256) ||
        throw(ArgumentError("invalid source checkpoint SHA-256"))
    final = abspath(destination)
    temporary = final * ".tmp"
    (ispath(final) || ispath(temporary)) &&
        throw(ArgumentError("checkpoint and .tmp destinations must both be fresh"))
    mkpath(temporary)
    try
        weights, host = host_hashstem_weights(trainer)
        weights_path = joinpath(temporary, "hashstem_master_weights.npz")
        NPZ.npzwrite(weights_path, _weights_dictionary(weights))
        restore_path = joinpath(temporary, "hashstem_optimizer.jld2")
        JLD2.jldsave(
            restore_path;
            restore=(;
                parameters=host.parameters,
                states=host.states,
                optimizer_state=host.optimizer_state,
                step=host.step,
                backend_updates=host.backend_updates,
            ),
        )
        weights_sha256 = _sha256_file(weights_path)
        optimizer_sha256 = _sha256_file(restore_path)
        metadata = (;
            schema=HASHSTEM_MASTER_CHECKPOINT_SCHEMA,
            hashstem_schema=HASHSTEM_SCHEMA,
            implementation_provenance=UNEXECUTED_STATIC_ONLY,
            # This writer cannot self-certify its unexecuted implementation.
            # A later audited promotion step must create VALIDATED_RUNTIME.
            runtime_validation_status=UNEXECUTED_STATIC_ONLY,
            model_id=trainer.model_id,
            backend="Reactant CPU",
            precision="FP32 master weights and AdamW state",
            master_version=trainer.master_version,
            optimizer_step=Int(host.step),
            backend_updates=host.backend_updates,
            last_snapshot_version=trainer.last_snapshot_version,
            last_snapshot_master_version=trainer.last_snapshot_master_version,
            batch_size=trainer.batch_size,
            learning_rate=trainer.learning_rate,
            weight_decay=trainer.weight_decay,
            auxiliary_weight=trainer.auxiliary_weight,
            huber_beta=trainer.huber_beta,
            initialization_seed=trainer.initialization_seed,
            weights_file="hashstem_master_weights.npz",
            weights_sha256,
            optimizer_file="hashstem_optimizer.jld2",
            optimizer_sha256,
            normalization_sha256=_normalization_sha256(weights),
            source_checkpoint_sha256=String(source_checkpoint_sha256),
            julia_version=string(VERSION),
            lux_version=string(Base.pkgversion(Lux)),
            reactant_version=string(Base.pkgversion(BeatFirstFixedShapeBackend.Reactant)),
            master_source_sha256=_sha256_file(@__FILE__),
            hashstem_source_sha256=_sha256_file(joinpath(@__DIR__, "hashstem.jl")),
            backend_source_sha256=_sha256_file(joinpath(
                @__DIR__, "..", "backend", "fixedshape_learner.jl",
            )),
            project_sha256=_sha256_file(joinpath(@__DIR__, "..", "Project.toml")),
            manifest_sha256=_sha256_file(joinpath(@__DIR__, "..", "Manifest.toml")),
            created_utc=_utc_now(),
            implicit_snapshot_publish=false,
            npu_backward=false,
            igpu_authorized=false,
        )
        _write_json(joinpath(temporary, "master_metadata.json"), metadata)
        manifest = (;
            schema=HASHSTEM_MASTER_CHECKPOINT_SCHEMA,
            metadata_file="master_metadata.json",
            metadata_sha256=_sha256_file(joinpath(temporary, "master_metadata.json")),
            weights_file=metadata.weights_file,
            weights_sha256,
            optimizer_file=metadata.optimizer_file,
            optimizer_sha256,
        )
        _write_json(joinpath(temporary, "checkpoint_manifest.json"), manifest)
        mv(temporary, final)
        return final
    catch
        # Leave a failed transaction visible for diagnosis; never publish it.
        rethrow()
    end
end

function load_hashstem_master_checkpoint(path::AbstractString)
    root = abspath(path)
    manifest_path = joinpath(root, "checkpoint_manifest.json")
    isfile(manifest_path) || error("checkpoint manifest is missing")
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.schema) == HASHSTEM_MASTER_CHECKPOINT_SCHEMA ||
        error("checkpoint schema mismatch")
    String(manifest.metadata_file) == "master_metadata.json" ||
        error("checkpoint metadata filename changed")
    String(manifest.weights_file) == "hashstem_master_weights.npz" ||
        error("checkpoint weights filename changed")
    String(manifest.optimizer_file) == "hashstem_optimizer.jld2" ||
        error("checkpoint optimizer filename changed")
    metadata_path = joinpath(root, "master_metadata.json")
    weights_path = joinpath(root, "hashstem_master_weights.npz")
    optimizer_path = joinpath(root, "hashstem_optimizer.jld2")
    for (file, expected) in (
        (metadata_path, String(manifest.metadata_sha256)),
        (weights_path, String(manifest.weights_sha256)),
        (optimizer_path, String(manifest.optimizer_sha256)),
    )
        isfile(file) || error("checkpoint member is missing: $file")
        _sha256_file(file) == expected || error("checkpoint member hash mismatch: $file")
    end
    metadata = JSON3.read(read(metadata_path, String))
    String(metadata.schema) == HASHSTEM_MASTER_CHECKPOINT_SCHEMA ||
        error("master metadata schema mismatch")
    String(metadata.hashstem_schema) == HASHSTEM_SCHEMA ||
        error("HashStem graph schema mismatch")
    String(metadata.weights_file) == String(manifest.weights_file) ||
        error("metadata/manifest weights filename mismatch")
    String(metadata.optimizer_file) == String(manifest.optimizer_file) ||
        error("metadata/manifest optimizer filename mismatch")
    String(metadata.weights_sha256) == String(manifest.weights_sha256) ||
        error("metadata/manifest weights digest mismatch")
    String(metadata.optimizer_sha256) == String(manifest.optimizer_sha256) ||
        error("metadata/manifest optimizer digest mismatch")
    String(metadata.implementation_provenance) == UNEXECUTED_STATIC_ONLY ||
        error("unexpected implementation provenance")
    String(metadata.runtime_validation_status) in
        (UNEXECUTED_STATIC_ONLY, VALIDATED_RUNTIME) ||
        error("unexpected runtime validation status")
    String(metadata.julia_version) == string(VERSION) ||
        error("Julia version changed; exact continuation rejected")
    String(metadata.lux_version) == string(Base.pkgversion(Lux)) ||
        error("Lux version changed; exact continuation rejected")
    String(metadata.reactant_version) ==
        string(Base.pkgversion(BeatFirstFixedShapeBackend.Reactant)) ||
        error("Reactant version changed; exact continuation rejected")
    for (label, observed, expected) in (
        ("master source", _sha256_file(@__FILE__), String(metadata.master_source_sha256)),
        ("HashStem source", _sha256_file(joinpath(@__DIR__, "hashstem.jl")),
            String(metadata.hashstem_source_sha256)),
        ("fixed-shape backend", _sha256_file(joinpath(
            @__DIR__, "..", "backend", "fixedshape_learner.jl",
        )), String(metadata.backend_source_sha256)),
        ("Project", _sha256_file(joinpath(@__DIR__, "..", "Project.toml")),
            String(metadata.project_sha256)),
        ("Manifest", _sha256_file(joinpath(@__DIR__, "..", "Manifest.toml")),
            String(metadata.manifest_sha256)),
    )
        observed == expected || error("$label changed; exact continuation rejected")
    end
    Int(metadata.master_version) == Int(metadata.backend_updates) ||
        error("master version and backend update count diverged")
    Int(metadata.optimizer_step) == Int(metadata.backend_updates) ||
        error("optimizer step and backend update count diverged")
    last_snapshot_version = Int(metadata.last_snapshot_version)
    last_snapshot_master_version = Int(metadata.last_snapshot_master_version)
    (last_snapshot_version == 0) == (last_snapshot_master_version == 0) ||
        error("checkpoint snapshot lineage is not a zero pair")
    0 <= last_snapshot_master_version <= Int(metadata.master_version) ||
        error("checkpoint snapshot lineage refers to a future master")
    optimizer = JLD2.load(optimizer_path)
    haskey(optimizer, "restore") || error("optimizer restore payload is missing")
    return (;
        root,
        manifest,
        metadata,
        weights_path,
        optimizer_path,
        restore=optimizer["restore"],
        manifest_sha256=_sha256_file(manifest_path),
    )
end

function _weights_from_archive(archive)
    expected = Set((
        "conv3", "conv3_bias", "depthwise5x1", "depthwise_bias",
        "pointwise", "pointwise_bias", "dense", "dense_bias",
        "input_mean", "input_inv_std",
    ))
    Set(String.(keys(archive))) == expected || error("master NPZ key set changed")
    array(name, shape) = begin
        value = Float32.(Array(archive[name]))
        size(value) == shape || error("master NPZ $name shape changed")
        value
    end
    weights = HashStemWeights(
        array("conv3", (16, 2, 3, 3)),
        vec(array("conv3_bias", (16,))),
        array("depthwise5x1", (16, 5)),
        vec(array("depthwise_bias", (16,))),
        array("pointwise", (32, 16)),
        vec(array("pointwise_bias", (32,))),
        array("dense", (214, 1039)),
        vec(array("dense_bias", (214,))),
        vec(array("input_mean", (559,))),
        vec(array("input_inv_std", (559,))),
    )
    return validate_hashstem_weights(weights)
end

function _weights_bitwise_equal(left::HashStemWeights, right::HashStemWeights)
    for name in fieldnames(HashStemWeights)
        a = getfield(left, name)
        b = getfield(right, name)
        size(a) == size(b) || return false
        reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b)) || return false
    end
    return true
end

function resume_hashstem_master(path::AbstractString)
    loaded = load_hashstem_master_checkpoint(path)
    metadata = loaded.metadata
    weights_archive = NPZ.npzread(loaded.weights_path)
    archived_weights = _weights_from_archive(weights_archive)
    _normalization_sha256(archived_weights) == String(metadata.normalization_sha256) ||
        error("master NPZ normalization digest mismatch")
    trainer = init_hashstem_master(
        model_id=String(metadata.model_id),
        seed=UInt64(metadata.initialization_seed),
        batch_size=Int(metadata.batch_size),
        input_mean=archived_weights.input_mean,
        input_inv_std=archived_weights.input_inv_std,
        learning_rate=Float32(metadata.learning_rate),
        weight_decay=Float32(metadata.weight_decay),
        auxiliary_weight=Float32(metadata.auxiliary_weight),
        huber_beta=Float32(metadata.huber_beta),
        restore=loaded.restore,
        master_version=Int(metadata.master_version),
        last_snapshot_version=Int(metadata.last_snapshot_version),
        last_snapshot_master_version=Int(metadata.last_snapshot_master_version),
    )
    restored_weights, host = host_hashstem_weights(trainer)
    _weights_bitwise_equal(restored_weights, archived_weights) ||
        error("restored TrainState and canonical NPZ differ bitwise")
    Int(host.step) == Int(metadata.optimizer_step) ||
        error("restored optimizer step mismatch")
    host.backend_updates == Int(metadata.backend_updates) ||
        error("restored backend update count mismatch")
    return trainer
end

Base.@kwdef struct HashStemLayoutEvidence
    weights_sha256::String
    witness_sha256::String
    output_shape_matches::Bool
    maximum_absolute_error::Float64
    exact_route_ids::Bool
end

function _layout_evidence_passes(evidence::HashStemLayoutEvidence, weights_sha256::String)
    return evidence.weights_sha256 == weights_sha256 &&
           _valid_sha256(evidence.weights_sha256) &&
           _valid_sha256(evidence.witness_sha256) &&
           evidence.output_shape_matches &&
           isfinite(evidence.maximum_absolute_error) &&
           0.0 <= evidence.maximum_absolute_error <= 1.0e-5 &&
           evidence.exact_route_ids
end

"""Create a fresh immutable NPU snapshot *source*; never compile or publish it."""
function export_hashstem_snapshot_source!(
    checkpoint_path::AbstractString,
    destination::AbstractString;
    snapshot_version::Integer,
    evidence::HashStemLayoutEvidence,
)
    snapshot_version > 0 || throw(ArgumentError("snapshot_version must be positive"))
    loaded = load_hashstem_master_checkpoint(checkpoint_path)
    metadata = loaded.metadata
    String(metadata.runtime_validation_status) == VALIDATED_RUNTIME ||
        error("snapshot export requires an explicitly runtime-validated master")
    weights_sha256 = String(metadata.weights_sha256)
    snapshot_archive = NPZ.npzread(loaded.weights_path)
    snapshot_weights = _weights_from_archive(snapshot_archive)
    observed_normalization_sha256 = _normalization_sha256(snapshot_weights)
    observed_normalization_sha256 == String(metadata.normalization_sha256) ||
        error("snapshot-source normalization digest mismatch")
    _layout_evidence_passes(evidence, weights_sha256) ||
        error("Lux/OpenVINO/scalar layout evidence did not pass")
    Int(snapshot_version) > Int(metadata.last_snapshot_version) ||
        error("snapshot version must exceed the last published version")
    final = abspath(destination)
    temporary = final * ".tmp"
    (ispath(final) || ispath(temporary)) &&
        throw(ArgumentError("snapshot source and .tmp destinations must be fresh"))
    mkpath(temporary)
    source_weights = loaded.weights_path
    destination_weights = joinpath(temporary, "hashstem_snapshot_weights.npz")
    cp(source_weights, destination_weights; force=false)
    binding = (;
        schema=HASHSTEM_SNAPSHOT_SOURCE_SCHEMA,
        hashstem_schema=HASHSTEM_SCHEMA,
        status=UNEXECUTED_STATIC_ONLY,
        model_id=String(metadata.model_id),
        master_version=Int(metadata.master_version),
        snapshot_version=Int(snapshot_version),
        weights_file="hashstem_snapshot_weights.npz",
        weights_sha256,
        normalization_sha256=observed_normalization_sha256,
        source_checkpoint_manifest_sha256=loaded.manifest_sha256,
        parity_witness_sha256=evidence.witness_sha256,
        parity_maximum_absolute_error=evidence.maximum_absolute_error,
        exact_route_ids=evidence.exact_route_ids,
        immutable_source=true,
        openvino_compiled=false,
        publish_authorized=false,
        npu_backward=false,
        exported_utc=_utc_now(),
    )
    _write_json(joinpath(temporary, "snapshot_source_metadata.json"), binding)
    mv(temporary, final)
    return final
end
