module RelationGraphCheckpoint

using Serialization
using SHA
using ..CandidateDeltaRelationGraph
using ..DendriticProgramBank
using ..RelationGraphOptimizer
using ..ReducedHayCPUSampler
using ..TypedDendriticAfferents

const Model = CandidateDeltaRelationGraph
const Bank = DendriticProgramBank
const Optimizer = RelationGraphOptimizer
const Sampler = ReducedHayCPUSampler
const Afferents = TypedDendriticAfferents

const CHECKPOINT_SCHEMA = UInt32(2)
const CHECKPOINT_FORMAT = "candidate-delta-relation-motif-graph-exact-v2"
const SOURCE_CLOSURE_SCHEMA = UInt32(2)
const SOURCE_HASH_ALGORITHM = "sha256-file-closure-v2"
const MODEL_FINGERPRINT_ALGORITHM =
    "sha256-relation-motif-model-contract-v2"
const OPTIMIZER_FINGERPRINT_ALGORITHM = "sha256-motif-optimizer-config-v2"
const RUN_CONTRACT_FINGERPRINT_ALGORITHM = "sha256-run-contract-v1"
const TRAINING_STATE_FINGERPRINT_ALGORITHM = "sha256-training-state-v2"

"""
Every repository-local source that can affect canonical relation-graph
forward, reverse, optimization, sampling, or barrierless scheduling.

Paths are relative to this file's directory and are intentionally fixed.  A
resume made from a different source closure is rejected; there is no legacy
DDF/local-learning compatibility path.
"""
const CANONICAL_SOURCE_FILES = (
    "ReducedHayCPU.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SpatialProgramPackets.jl",
    "DendriticRelationTopology.jl",
    "DendriticMotifTopology.jl",
    "TypedDendriticAfferents.jl",
    "HighDimensionalCellPacket.jl",
    "TypedRelationCellBank.jl",
    "TypedRelationContext.jl",
    "TypedOutputCellBank.jl",
    "StructuredMotifReadout.jl",
    "CandidateDeltaRelationGraph.jl",
    "TetrisRankingBatch.jl",
    "RelationGraphOptimizer.jl",
    "RelationGraphTraining.jl",
    "BarrierlessScheduler.jl",
    "RelationGraphBarrierless.jl",
    "Sampler.jl",
    "ExperimentData.jl",
    "train_scratch.jl",
    "Checkpoint.jl",
    "../training/core.jl",
    "../episodic_vit_recurrent_lookup/bounded_mpmc_queue.jl",
    "../episodic_vit_recurrent_lookup/windows_cpu_sets.jl",
    "../../../Project.toml",
    "../../../Manifest.toml",
)

const _SNAPSHOT_KEYS = (
    :schema,
    :format,
    :update,
    :model_fingerprint,
    :optimizer_config_fingerprint,
    :run_contract_fingerprint,
    :training_state_fingerprint,
    :source_closure,
    :run_contract,
    :parameters,
    :optimizer_config,
    :optimizer_state,
    :sampler,
)
const _SOURCE_CLOSURE_KEYS = (:schema, :algorithm, :files, :aggregate)
const _SOURCE_FILE_KEYS = (:path, :sha256)

export CANONICAL_SOURCE_FILES,
       CHECKPOINT_SCHEMA,
       ResumeState,
       canonical_source_closure,
       load_checkpoint,
       model_fingerprint,
       optimizer_config_fingerprint,
       run_contract_fingerprint,
       training_state_fingerprint,
       restore_checkpoint!,
       save_checkpoint

"""The two pieces of mutable run progress returned after a successful resume."""
struct ResumeState
    update::Int
    sampler::Sampler.DeterministicEpochSampler
end

@inline function _write_u64(io::IO, value::UInt64)
    @inbounds for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

@inline _write_int(io::IO, value::Integer) = _write_u64(io, UInt64(value))

function _write_string(io::IO, value::AbstractString)
    bytes = codeunits(value)
    _write_int(io, length(bytes))
    write(io, bytes)
    return io
end

function _write_shape(io::IO, array::AbstractArray)
    _write_string(io, string(eltype(array)))
    _write_int(io, ndims(array))
    @inbounds for dimension in size(array)
        _write_int(io, dimension)
    end
    return io
end

function _write_integer_array(io::IO, array::AbstractArray{T}) where {T<:Integer}
    _write_shape(io, array)
    @inbounds for value in array
        if T <: Signed
            _write_u64(io, reinterpret(UInt64, Int64(value)))
        else
            _write_u64(io, UInt64(value))
        end
    end
    return io
end

@inline function _sha256_file(path::AbstractString)
    isfile(path) || error("canonical source is missing: $path")
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""Return the exact, path-independent digest set of canonical source files."""
function canonical_source_closure(source_root::AbstractString=@__DIR__)
    root = abspath(source_root)
    files = Vector{NamedTuple{_SOURCE_FILE_KEYS,Tuple{String,String}}}(
        undef,
        length(CANONICAL_SOURCE_FILES),
    )
    aggregate_io = IOBuffer()
    _write_string(aggregate_io, SOURCE_HASH_ALGORITHM)
    @inbounds for (index, relative_path) in enumerate(CANONICAL_SOURCE_FILES)
        normalized = replace(normpath(relative_path), '\\' => '/')
        digest = _sha256_file(normpath(joinpath(root, relative_path)))
        files[index] = (; path=normalized, sha256=digest)
        _write_string(aggregate_io, normalized)
        _write_string(aggregate_io, digest)
    end
    aggregate = bytes2hex(SHA.sha256(take!(aggregate_io)))
    return (;
        schema=SOURCE_CLOSURE_SCHEMA,
        algorithm=SOURCE_HASH_ALGORITHM,
        files=files,
        aggregate=aggregate,
    )
end

function _validate_source_closure(snapshot)
    snapshot isa NamedTuple || throw(ArgumentError(
        "checkpoint source closure is not a named tuple",
    ))
    keys(snapshot) == _SOURCE_CLOSURE_KEYS || throw(ArgumentError(
        "checkpoint source closure fields are missing or extra",
    ))
    snapshot.schema === SOURCE_CLOSURE_SCHEMA || throw(ArgumentError(
        "checkpoint source closure schema is unsupported",
    ))
    snapshot.algorithm === SOURCE_HASH_ALGORITHM || throw(ArgumentError(
        "checkpoint source hash algorithm differs",
    ))
    snapshot.files isa Vector{NamedTuple{_SOURCE_FILE_KEYS,Tuple{String,String}}} ||
        throw(ArgumentError("checkpoint source file records have the wrong type"))
    length(snapshot.files) == length(CANONICAL_SOURCE_FILES) || throw(
        ArgumentError("checkpoint source closure length differs"),
    )
    occursin(r"^[0-9a-f]{64}$", snapshot.aggregate) || throw(ArgumentError(
        "checkpoint aggregate source digest is malformed",
    ))
    @inbounds for file in snapshot.files
        keys(file) == _SOURCE_FILE_KEYS || throw(ArgumentError(
            "checkpoint source record fields are missing or extra",
        ))
        occursin(r"^[0-9a-f]{64}$", file.sha256) || throw(ArgumentError(
            "checkpoint source digest is malformed",
        ))
    end
    current = canonical_source_closure()
    snapshot == current || throw(ArgumentError(
        "checkpoint canonical source closure differs from the current tree",
    ))
    return current
end

@inline function _write_graph_contract!(
    io::IO,
    label::AbstractString,
    graph::Afferents.TypedAfferentGraph,
)
    Afferents.validate_typed_afferents(graph)
    _write_string(io, label)
    _write_int(io, graph.source_count)
    _write_int(io, graph.field_count)
    _write_int(io, graph.destination_count)
    _write_int(io, graph.fanout)
    _write_integer_array(io, graph.field_kind)
    _write_integer_array(io, graph.source_field)
    _write_integer_array(io, graph.source_polarity)
    _write_integer_array(io, graph.destination_cell)
    _write_integer_array(io, graph.destination_compartment)
    _write_integer_array(io, graph.receptor)
    _write_integer_array(io, graph.destination_input)
    _write_shape(io, graph.raw_conductance)
    return io
end

"""
Hash only the model contract: trainable shapes plus every fixed anatomical
identity.  Parameter values are saved separately and deliberately excluded.
"""
function _model_contract_fingerprint(parameters::Model.ModelParameters)
    _validate_parameters(parameters)
    io = IOBuffer()
    _write_string(io, MODEL_FINGERPRINT_ALGORITHM)
    _write_shape(io, parameters.program_bank.payload)
    _write_graph_contract!(io, "leaf_relation", parameters.leaf_relation)
    _write_shape(io, parameters.relation.cell_raw)
    _write_graph_contract!(io, "relation_motif", parameters.relation_motif)
    _write_shape(io, parameters.motif.cell_raw)
    _write_graph_contract!(
        io,
        "common_relation",
        parameters.context.common_relation,
    )
    _write_graph_contract!(
        io,
        "common_output",
        parameters.context.common_output,
    )
    _write_graph_contract!(
        io,
        "aux_relation",
        parameters.context.aux_relation,
    )
    _write_graph_contract!(
        io,
        "placement_relation",
        parameters.placement_relation,
    )
    _write_shape(io, parameters.motif_readout.source_gain_raw)
    _write_shape(io, parameters.output.cell_raw)
    _write_shape(io, parameters.output.readout_weight)
    _write_shape(io, parameters.output.bias)
    _write_int(io, Model.stored_parameter_count(parameters))
    return bytes2hex(SHA.sha256(take!(io)))
end

const _CANONICAL_MODEL_FINGERPRINT = Ref{Union{Nothing,String}}(nothing)

function _canonical_model_fingerprint()
    fingerprint = _CANONICAL_MODEL_FINGERPRINT[]
    if isnothing(fingerprint)
        fingerprint = _model_contract_fingerprint(Model.initialize_model())
        _CANONICAL_MODEL_FINGERPRINT[] = fingerprint
    end
    return fingerprint
end

function model_fingerprint(parameters::Model.ModelParameters)
    fingerprint = _model_contract_fingerprint(parameters)
    fingerprint == _canonical_model_fingerprint() || throw(ArgumentError(
        "model anatomy differs from the canonical relation graph",
    ))
    return fingerprint
end

function optimizer_config_fingerprint(config::Optimizer.OptimizerConfig)
    validated = _validate_optimizer_config(config)
    io = IOBuffer()
    _write_string(io, OPTIMIZER_FINGERPRINT_ALGORITHM)
    @inbounds for name in fieldnames(Optimizer.OptimizerConfig)
        _write_string(io, String(name))
        value = getfield(validated, name)
        write(io, reinterpret(UInt32, value))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _write_contract_value(io::IO, value)
    _write_string(io, string(typeof(value)))
    if value isa Nothing
        return io
    elseif value isa Bool
        write(io, UInt8(value))
    elseif value isa Integer
        _write_string(io, string(value))
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError(
            "run contract floating values must be finite",
        ))
        if value isa Float16
            _write_u64(io, UInt64(reinterpret(UInt16, value)))
        elseif value isa Float32
            _write_u64(io, UInt64(reinterpret(UInt32, value)))
        elseif value isa Float64
            _write_u64(io, reinterpret(UInt64, value))
        else
            _write_string(io, string(value))
        end
    elseif value isa AbstractString
        _write_string(io, value)
    elseif value isa Symbol
        _write_string(io, String(value))
    elseif value isa NamedTuple
        _write_int(io, length(value))
        for name in keys(value)
            _write_string(io, String(name))
            _write_contract_value(io, getproperty(value, name))
        end
    elseif value isa Tuple
        _write_int(io, length(value))
        for element in value
            _write_contract_value(io, element)
        end
    elseif value isa AbstractVector
        _write_int(io, length(value))
        for element in value
            _write_contract_value(io, element)
        end
    else
        throw(ArgumentError(
            "run contract contains unsupported value type $(typeof(value))",
        ))
    end
    return io
end

"""Hash stable, plain-data run metadata supplied by the production CLI."""
function run_contract_fingerprint(run_contract::NamedTuple)
    isempty(run_contract) && throw(ArgumentError("run contract cannot be empty"))
    io = IOBuffer()
    _write_string(io, RUN_CONTRACT_FINGERPRINT_ALGORITHM)
    _write_contract_value(io, run_contract)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _write_array_digest!(
    io::IO,
    label::AbstractString,
    array::DenseArray{T},
) where {T<:Union{Float32,UInt32}}
    _write_string(io, label)
    _write_shape(io, array)
    bytes = reinterpret(UInt8, vec(array))
    _write_string(io, bytes2hex(SHA.sha256(bytes)))
    return io
end

"""Content checksum for all trainable values, AdamW state, and run clocks."""
function training_state_fingerprint(
    parameters::Model.ModelParameters,
    optimizer_state::Optimizer.AdamWState,
    optimizer_config::Optimizer.OptimizerConfig,
    update::Int,
    sampler_snapshot::NamedTuple,
    run_contract::NamedTuple,
)
    io = IOBuffer()
    _write_string(io, TRAINING_STATE_FINGERPRINT_ALGORITHM)
    for (label, array) in zip(_PARAMETER_LABELS, _parameter_arrays(parameters))
        _write_array_digest!(io, "parameter/$label", array)
    end
    for prefix in ("first", "second")
        moments = prefix == "first" ? optimizer_state.first : optimizer_state.second
        for name in fieldnames(Optimizer.DenseMoments)
            _write_array_digest!(
                io,
                "$prefix/$(String(name))",
                getfield(moments, name),
            )
        end
    end
    _write_array_digest!(io, "program_first", optimizer_state.program_first)
    _write_array_digest!(io, "program_second", optimizer_state.program_second)
    _write_array_digest!(
        io,
        "program_step_by_row",
        optimizer_state.program_step_by_row,
    )
    _write_int(io, update)
    for name in fieldnames(Optimizer.AdamWStepCounters)
        _write_string(io, String(name))
        _write_int(io, getfield(optimizer_state.steps, name))
    end
    _write_string(io, optimizer_config_fingerprint(optimizer_config))
    _write_contract_value(io, sampler_snapshot)
    _write_contract_value(io, run_contract)
    return bytes2hex(SHA.sha256(take!(io)))
end

@inline function _all_finite(array, label::AbstractString)
    @inbounds for value in array
        isfinite(value) || throw(DomainError(value, "$label is not finite"))
    end
    return array
end

function _parameter_arrays(parameters::Model.ModelParameters)
    return (
        parameters.program_bank.payload,
        parameters.leaf_relation.raw_conductance,
        parameters.relation.cell_raw,
        parameters.relation_motif.raw_conductance,
        parameters.motif.cell_raw,
        parameters.context.common_relation.raw_conductance,
        parameters.context.common_output.raw_conductance,
        parameters.context.aux_relation.raw_conductance,
        parameters.placement_relation.raw_conductance,
        parameters.motif_readout.source_gain_raw,
        parameters.output.cell_raw,
        parameters.output.readout_weight,
        parameters.output.bias,
    )
end

const _PARAMETER_LABELS = (
    "program_bank",
    "leaf_relation",
    "relation_cell",
    "relation_motif",
    "motif_cell",
    "common_relation",
    "common_output",
    "auxiliary_relation",
    "placement_relation",
    "motif_readout",
    "output_cell",
    "output_readout_weight",
    "output_bias",
)

function _validate_parameters(parameters::Model.ModelParameters)
    size(parameters.program_bank.payload) == (Bank.PAYLOAD_WIDTH, Bank.ROW_COUNT) ||
        throw(DimensionMismatch("program bank shape is noncanonical"))
    for graph in (
        parameters.leaf_relation,
        parameters.relation_motif,
        parameters.context.common_relation,
        parameters.context.common_output,
        parameters.context.aux_relation,
        parameters.placement_relation,
    )
        Afferents.validate_typed_afferents(graph)
    end
    size(parameters.relation.cell_raw) ==
        (Model.Cell.PARAM_DIM, Model.RELATION_CELLS) || throw(
            DimensionMismatch("relation-cell parameter shape is noncanonical"),
        )
    size(parameters.motif.cell_raw) ==
        (Model.Cell.PARAM_DIM, Model.MOTIF_CELLS) || throw(
            DimensionMismatch("motif-cell parameter shape is noncanonical"),
        )
    size(parameters.motif_readout.source_gain_raw) ==
        (Model.MOTIF_CELLS, Model.OUTPUT_CELLS) || throw(
            DimensionMismatch("motif-readout parameter shape is noncanonical"),
        )
    size(parameters.output.cell_raw) ==
        (Model.Cell.PARAM_DIM, Model.OUTPUT_CELLS) || throw(
            DimensionMismatch("output-cell parameter shape is noncanonical"),
        )
    size(parameters.output.readout_weight) ==
        (Model.Outputs.READOUT_DIM, Model.OUTPUT_CELLS) || throw(
            DimensionMismatch("output-readout parameter shape is noncanonical"),
        )
    length(parameters.output.bias) == Model.OUTPUT_CELLS || throw(
        DimensionMismatch("output bias shape is noncanonical"),
    )
    @inbounds for (label, array) in zip(
        _PARAMETER_LABELS,
        _parameter_arrays(parameters),
    )
        _all_finite(array, label)
    end
    Model.stored_parameter_count(parameters) ==
        sum(length, _parameter_arrays(parameters)) || throw(ArgumentError(
            "stored parameter count disagrees with canonical groups",
        ))
    return parameters
end

function _validate_run_progress(update::Int, sampler_snapshot, run_contract)
    hasproperty(run_contract, :state_batch) || throw(ArgumentError(
        "run contract must include state_batch",
    ))
    state_batch = run_contract.state_batch
    typeof(state_batch) === Int && state_batch >= 1 || throw(ArgumentError(
        "run contract state_batch must be a positive Int",
    ))
    sampler_snapshot isa NamedTuple || throw(ArgumentError(
        "checkpoint sampler payload has the wrong type",
    ))
    hasproperty(sampler_snapshot, :source_rows) || throw(ArgumentError(
        "checkpoint sampler payload has no source rows",
    ))
    source_rows = sampler_snapshot.source_rows
    source_rows isa Vector{Int} || throw(ArgumentError(
        "checkpoint sampler source rows have the wrong type",
    ))
    restored = Sampler.restore_sampler(source_rows, sampler_snapshot)
    consumed = Sampler.sampler_consumed_rows(restored)
    expected = UInt128(update) * UInt128(state_batch)
    consumed == expected || throw(ArgumentError(
        "sampler consumed-row count $consumed differs from " *
        "update*state_batch $expected",
    ))
    return restored
end

function _validate_optimizer_config(config::Optimizer.OptimizerConfig)
    names = fieldnames(Optimizer.OptimizerConfig)
    values = NamedTuple{names}(ntuple(
        index -> getfield(config, names[index]),
        length(names),
    ))
    reconstructed = Optimizer.OptimizerConfig(; values...)
    isequal(reconstructed, config) || throw(ArgumentError(
        "optimizer configuration failed canonical reconstruction",
    ))
    return config
end

function _dense_parameter_arrays(parameters::Model.ModelParameters)
    return (;
        leaf_relation=parameters.leaf_relation.raw_conductance,
        relation_cell=parameters.relation.cell_raw,
        relation_motif=parameters.relation_motif.raw_conductance,
        motif_cell=parameters.motif.cell_raw,
        common_relation=parameters.context.common_relation.raw_conductance,
        common_output=parameters.context.common_output.raw_conductance,
        auxiliary_relation=parameters.context.aux_relation.raw_conductance,
        placement_relation=parameters.placement_relation.raw_conductance,
        motif_readout=parameters.motif_readout.source_gain_raw,
        output_cell=parameters.output.cell_raw,
        output_readout_weight=parameters.output.readout_weight,
        output_bias=parameters.output.bias,
    )
end

function _validate_optimizer_state(
    state::Optimizer.AdamWState,
    parameters::Model.ModelParameters,
    config::Optimizer.OptimizerConfig,
    update::Int,
)
    update >= 0 || throw(ArgumentError("checkpoint update must be non-negative"))
    dense_parameters = _dense_parameter_arrays(parameters)
    dense_names = keys(dense_parameters)
    fieldnames(Optimizer.DenseMoments) == dense_names || throw(ArgumentError(
        "optimizer moment groups differ from canonical model groups",
    ))
    fieldnames(Optimizer.AdamWStepCounters) == (
        :total,
        :program_batches,
        :program_rows,
        dense_names...,
    ) || throw(ArgumentError(
        "optimizer counter groups differ from canonical model groups",
    ))
    for name in dense_names
        parameter = getproperty(dense_parameters, name)
        first = getfield(state.first, name)
        second = getfield(state.second, name)
        size(first) == size(parameter) || throw(DimensionMismatch(
            "first-moment shape differs for $name",
        ))
        size(second) == size(parameter) || throw(DimensionMismatch(
            "second-moment shape differs for $name",
        ))
        _all_finite(first, "first moment $name")
        _all_finite(second, "second moment $name")
        @inbounds for value in second
            value >= 0.0f0 || throw(DomainError(
                value,
                "second moment $name is negative",
            ))
        end
    end
    size(state.program_first) == size(parameters.program_bank.payload) || throw(
        DimensionMismatch("program first-moment shape differs"),
    )
    size(state.program_second) == size(parameters.program_bank.payload) || throw(
        DimensionMismatch("program second-moment shape differs"),
    )
    length(state.program_step_by_row) == Bank.ROW_COUNT || throw(
        DimensionMismatch("program row-clock shape differs"),
    )
    _all_finite(state.program_first, "program first moment")
    _all_finite(state.program_second, "program second moment")
    @inbounds for value in state.program_second
        value >= 0.0f0 || throw(DomainError(
            value,
            "program second moment is negative",
        ))
    end

    steps = state.steps
    @inbounds for name in fieldnames(Optimizer.AdamWStepCounters)
        value = getfield(steps, name)
        value >= 0 || throw(ArgumentError("optimizer counter $name is negative"))
    end
    steps.total == update || throw(ArgumentError(
        "checkpoint update differs from optimizer total-step counter",
    ))
    steps.program_batches <= steps.total || throw(ArgumentError(
        "program batch counter exceeds total optimizer steps",
    ))
    row_step_sum = UInt128(0)
    @inbounds for step in state.program_step_by_row
        Int(step) <= steps.program_batches || throw(ArgumentError(
            "program row clock exceeds program batch counter",
        ))
        row_step_sum += UInt128(step)
    end
    row_step_sum == UInt128(steps.program_rows) || throw(ArgumentError(
        "program row aggregate counter disagrees with row clocks",
    ))

    dense_multipliers = NamedTuple{dense_names}(ntuple(
        index -> getfield(
            config,
            Symbol(dense_names[index], "_multiplier"),
        ),
        length(dense_names),
    ))
    for name in keys(dense_multipliers)
        expected = getproperty(dense_multipliers, name) > 0.0f0 ? update : 0
        getfield(steps, name) == expected || throw(ArgumentError(
            "optimizer counter $name is inconsistent with its multiplier",
        ))
    end
    if config.program_multiplier == 0.0f0
        steps.program_batches == 0 && steps.program_rows == 0 || throw(
            ArgumentError("frozen program group has nonzero counters"),
        )
        all(iszero, state.program_step_by_row) || throw(ArgumentError(
            "frozen program group has nonzero row clocks",
        ))
    end
    return state
end

@inline function _trainer_contract(trainer)
    hasproperty(trainer, :parameters) || throw(ArgumentError(
        "trainer does not own canonical parameters",
    ))
    hasproperty(trainer, :cache) || throw(ArgumentError(
        "trainer does not own a canonical model cache",
    ))
    hasproperty(trainer, :optimizer_state) || throw(ArgumentError(
        "trainer does not own canonical AdamW state",
    ))
    hasproperty(trainer, :optimizer_config) || throw(ArgumentError(
        "trainer does not own canonical optimizer configuration",
    ))
    parameters = getproperty(trainer, :parameters)
    cache = getproperty(trainer, :cache)
    optimizer_state = getproperty(trainer, :optimizer_state)
    optimizer_config = getproperty(trainer, :optimizer_config)
    parameters isa Model.ModelParameters || throw(ArgumentError(
        "trainer parameters have the wrong canonical type",
    ))
    cache isa Model.ModelCache || throw(ArgumentError(
        "trainer cache has the wrong canonical type",
    ))
    optimizer_state isa Optimizer.AdamWState || throw(ArgumentError(
        "trainer optimizer state has the wrong canonical type",
    ))
    optimizer_config isa Optimizer.OptimizerConfig || throw(ArgumentError(
        "trainer optimizer config has the wrong canonical type",
    ))
    return parameters, cache, optimizer_state, optimizer_config
end

function _validate_snapshot(snapshot; validate_current_source::Bool=true)
    snapshot isa NamedTuple || throw(ArgumentError(
        "checkpoint payload is not a named tuple",
    ))
    keys(snapshot) == _SNAPSHOT_KEYS || throw(ArgumentError(
        "checkpoint payload fields are missing or extra",
    ))
    snapshot.schema === CHECKPOINT_SCHEMA || throw(ArgumentError(
        "unsupported checkpoint schema $(snapshot.schema)",
    ))
    snapshot.format === CHECKPOINT_FORMAT || throw(ArgumentError(
        "checkpoint format is not the canonical relation graph",
    ))
    snapshot.update isa Int || throw(ArgumentError(
        "checkpoint update has the wrong type",
    ))
    snapshot.parameters isa Model.ModelParameters || throw(ArgumentError(
        "checkpoint parameters have the wrong type",
    ))
    snapshot.optimizer_config isa Optimizer.OptimizerConfig || throw(
        ArgumentError("checkpoint optimizer configuration has the wrong type"),
    )
    snapshot.optimizer_state isa Optimizer.AdamWState || throw(ArgumentError(
        "checkpoint optimizer state has the wrong type",
    ))
    snapshot.run_contract isa NamedTuple || throw(ArgumentError(
        "checkpoint run contract is not a named tuple",
    ))
    snapshot.sampler isa NamedTuple || throw(ArgumentError(
        "checkpoint sampler payload has the wrong type",
    ))
    _validate_parameters(snapshot.parameters)
    _validate_optimizer_config(snapshot.optimizer_config)
    _validate_optimizer_state(
        snapshot.optimizer_state,
        snapshot.parameters,
        snapshot.optimizer_config,
        snapshot.update,
    )
    snapshot.model_fingerprint === model_fingerprint(snapshot.parameters) ||
        throw(ArgumentError("checkpoint model fingerprint is false"))
    snapshot.optimizer_config_fingerprint ===
        optimizer_config_fingerprint(snapshot.optimizer_config) || throw(
            ArgumentError("checkpoint optimizer fingerprint is false"),
        )
    snapshot.run_contract_fingerprint ===
        run_contract_fingerprint(snapshot.run_contract) || throw(
            ArgumentError("checkpoint run-contract fingerprint is false"),
        )
    snapshot.training_state_fingerprint === training_state_fingerprint(
        snapshot.parameters,
        snapshot.optimizer_state,
        snapshot.optimizer_config,
        snapshot.update,
        snapshot.sampler,
        snapshot.run_contract,
    ) || throw(ArgumentError("checkpoint training-state fingerprint is false"))
    validate_current_source && _validate_source_closure(snapshot.source_closure)
    _validate_run_progress(
        snapshot.update,
        snapshot.sampler,
        snapshot.run_contract,
    )
    return snapshot
end

"""
Atomically save all state required for an exact canonical resume.

`update` is mandatory and must equal the AdamW total-step counter.  This keeps
the entrypoint's logging/checkpoint cadence from drifting from optimizer time.
"""
function save_checkpoint(
    path,
    trainer,
    sampler::Sampler.DeterministicEpochSampler;
    update::Integer,
    run_contract::NamedTuple,
)
    update isa Bool && throw(ArgumentError("checkpoint update cannot be Bool"))
    update_value = Int(update)
    parameters, _, optimizer_state, optimizer_config = _trainer_contract(trainer)
    _validate_parameters(parameters)
    _validate_optimizer_config(optimizer_config)
    _validate_optimizer_state(
        optimizer_state,
        parameters,
        optimizer_config,
        update_value,
    )
    sampler_record = Sampler.sampler_snapshot(sampler)
    Sampler.restore_sampler(sampler.source_rows, sampler_record)
    snapshot = (;
        schema=CHECKPOINT_SCHEMA,
        format=CHECKPOINT_FORMAT,
        update=update_value,
        model_fingerprint=model_fingerprint(parameters),
        optimizer_config_fingerprint=
            optimizer_config_fingerprint(optimizer_config),
        run_contract_fingerprint=run_contract_fingerprint(run_contract),
        training_state_fingerprint=training_state_fingerprint(
            parameters,
            optimizer_state,
            optimizer_config,
            update_value,
            sampler_record,
            run_contract,
        ),
        source_closure=canonical_source_closure(),
        run_contract=deepcopy(run_contract),
        parameters=deepcopy(parameters),
        optimizer_config=optimizer_config,
        optimizer_state=deepcopy(optimizer_state),
        sampler=sampler_record,
    )
    _validate_snapshot(snapshot)

    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp"
    try
        open(temporary, "w") do io
            serialize(io, snapshot)
        end
        mv(temporary, destination; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

"""Load and fully validate a canonical relation-graph checkpoint."""
function load_checkpoint(path)
    source = abspath(path)
    isfile(source) || error("checkpoint does not exist: $source")
    snapshot = open(deserialize, source)
    return _validate_snapshot(snapshot)
end

function _copy_parameters!(
    destination::Model.ModelParameters,
    source::Model.ModelParameters,
)
    model_fingerprint(destination) == model_fingerprint(source) || throw(
        ArgumentError("checkpoint model contract differs from live trainer"),
    )
    @inbounds for (destination_array, source_array) in zip(
        _parameter_arrays(destination),
        _parameter_arrays(source),
    )
        size(destination_array) == size(source_array) || throw(
            DimensionMismatch("checkpoint parameter shape differs"),
        )
        copyto!(destination_array, source_array)
    end
    return destination
end

function _copy_dense_moments!(destination, source)
    @inbounds for name in fieldnames(Optimizer.DenseMoments)
        copyto!(getfield(destination, name), getfield(source, name))
    end
    return destination
end

function _copy_optimizer_state!(
    destination::Optimizer.AdamWState,
    source::Optimizer.AdamWState,
)
    _copy_dense_moments!(destination.first, source.first)
    _copy_dense_moments!(destination.second, source.second)
    copyto!(destination.program_first, source.program_first)
    copyto!(destination.program_second, source.program_second)
    copyto!(destination.program_step_by_row, source.program_step_by_row)
    @inbounds for name in fieldnames(Optimizer.AdamWStepCounters)
        setfield!(destination.steps, name, getfield(source.steps, name))
    end
    return destination
end

"""
Restore only after every model/config/source/sampler check has passed.

The live trainer is not mutated on any validation failure.  On success its
parameter arrays, AdamW moments/clocks, and transformed model cache are
updated in place.  The caller receives the exact update and sampler progress.
"""
function restore_checkpoint!(
    trainer,
    snapshot,
    source_rows::AbstractVector{<:Integer},
    ;
    expected_run_contract::NamedTuple,
)
    _validate_snapshot(snapshot)
    parameters, cache, optimizer_state, optimizer_config =
        _trainer_contract(trainer)
    isequal(snapshot.optimizer_config, optimizer_config) || throw(
        ArgumentError("resume optimizer configuration differs"),
    )
    snapshot.optimizer_config_fingerprint ===
        optimizer_config_fingerprint(optimizer_config) || throw(ArgumentError(
            "resume optimizer configuration fingerprint differs",
        ))
    snapshot.model_fingerprint === model_fingerprint(parameters) || throw(
        ArgumentError("resume model fingerprint differs"),
    )
    isequal(snapshot.run_contract, expected_run_contract) || throw(
        ArgumentError("resume run contract differs"),
    )
    snapshot.run_contract_fingerprint ===
        run_contract_fingerprint(expected_run_contract) || throw(ArgumentError(
            "resume run-contract fingerprint differs",
        ))
    restored_sampler = Sampler.restore_sampler(source_rows, snapshot.sampler)

    # All potentially failing semantic checks are above this line.  Shapes are
    # now fixed by equal model fingerprints, so the following copy is atomic at
    # the checkpoint-contract level.
    _copy_parameters!(parameters, snapshot.parameters)
    _copy_optimizer_state!(optimizer_state, snapshot.optimizer_state)
    Model.refresh_cache!(cache, parameters)
    return ResumeState(snapshot.update, restored_sampler)
end

end # module RelationGraphCheckpoint
