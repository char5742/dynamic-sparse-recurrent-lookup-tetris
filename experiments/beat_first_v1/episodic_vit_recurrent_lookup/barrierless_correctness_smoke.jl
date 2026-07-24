#!/usr/bin/env julia

"""
One-update correctness witness for the production barrierless executor.

The oracle is the direct, single-worker candidate state machine.  Both sides
are independently deserialized from the same immutable checkpoint and
consume the same next four *training* rows.  No validation subset or sealed
seed is constructed by this program.

Run with:

    JULIA_NUM_THREADS=20,0 julia barrierless_correctness_smoke.jl

The model-geometry defaults below are the geometry recorded by the immutable
checkpoint.  Explicit environment values are never overwritten, so an
incompatible invocation fails at checkpoint restoration rather than silently
changing the model.
"""

for (name, value) in (
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_ROUTER_TABLES" => "2",
    "EVRL_ROUTER_BITS" => "4",
    "EVRL_ROUTER_BUCKET_CAP" => "64",
    "EVRL_EPISODIC_SHORTLIST" => "64",
    "EVRL_EPISODIC_CANDIDATE_CAP" => "64",
    "EVRL_SPATIAL_ANCHORS" => "2",
    "EVRL_SPATIAL_SHORTLIST" => "2",
    "EVRL_SPATIAL_CANDIDATE_CAP" => "3",
    "EVRL_FFN_DIM" => "128",
    "EVRL_INITIAL_HALT_PROBABILITY" => "0.8",
)
    haskey(ENV, name) || (ENV[name] = value)
end

using LinearAlgebra
using JSON3
using Random
using SHA
using Serialization
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const Model = Training.Model
const TrainingCore = Training.TrainingCore

const DEFAULT_CHECKPOINT = raw"D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_fixed_k64_wta_mean_wm_u10000_20260723\checkpoints\checkpoint_000010000.jls"
const DEFAULT_CHECKPOINT_SHA256 =
    "c834611e07cec1743658cea90118253406c45bdb5b5c8db625259229df89906a"

const OUTPUT_ATOL = 1.0e-6
const OUTPUT_RTOL = 1.0e-6
# Dynamic trajectories can accumulate substantially larger visual-stem
# gradients than the former fixed-depth smoke.  Keep a strict relative-L2
# guard and use an absolute floor appropriate for Float32 reductions over all
# active cell tokens; optimizer/parameter state is checked again below.
const GRADIENT_ATOL = 2.0e-4
const GRADIENT_RELATIVE_L2 = 1.0e-5
const FORBIDDEN_TEACHER_SEEDS = Set{Int}(vcat(
    collect(8001:8008),
    collect(91001:91032),
))

_require(condition::Bool, message::AbstractString) =
    condition ? nothing : error(message)

mutable struct NumericError
    maximum_absolute::Float64
    squared_difference::Float64
    squared_reference::Float64
    values::Int
    tolerance_violations::Int
end

NumericError() = NumericError(0.0, 0.0, 0.0, 0, 0)

@inline function _observe!(
    result::NumericError,
    actual::Real,
    reference::Real;
    atol::Float64=Inf,
    rtol::Float64=0.0,
)
    actual64 = Float64(actual)
    reference64 = Float64(reference)
    isfinite(actual64) || error("non-finite actual value in correctness comparison")
    isfinite(reference64) || error("non-finite reference value in correctness comparison")
    difference = actual64 - reference64
    absolute = abs(difference)
    result.maximum_absolute = max(result.maximum_absolute, absolute)
    result.squared_difference += difference * difference
    result.squared_reference += reference64 * reference64
    result.values += 1
    absolute <= atol + rtol * abs(reference64) ||
        (result.tolerance_violations += 1)
    return result
end

function _observe_array!(
    result::NumericError,
    actual::AbstractArray{<:AbstractFloat},
    reference::AbstractArray{<:AbstractFloat};
    atol::Float64=Inf,
    rtol::Float64=0.0,
)
    size(actual) == size(reference) || error("numeric array shape differs")
    @inbounds for index in eachindex(actual, reference)
        _observe!(result, actual[index], reference[index]; atol, rtol)
    end
    return result
end

_relative_l2(error::NumericError) = sqrt(error.squared_difference) /
    max(sqrt(error.squared_reference), eps(Float64))

function _numeric_record(error::NumericError)
    return (;
        maximum_absolute=error.maximum_absolute,
        relative_l2=_relative_l2(error),
        values=error.values,
        tolerance_violations=error.tolerance_violations,
    )
end

function _training_only_split(dataset, metadata)
    # Reconstruct only the training-row universe recorded in the checkpoint.
    # Validation row IDs and all evaluation seeds deliberately remain unused.
    training_groups = Int.(metadata.training_groups)
    training_group_set = Set(training_groups)
    training_rows = findall(
        group -> Int(group) in training_group_set,
        dataset.split_group_ids,
    )
    isempty(training_rows) && error("checkpoint training-group set is empty")
    return (;
        training_rows,
        validation_rows=Int[],
        training_groups,
        validation_groups=Int.(metadata.validation_groups),
        predefined=Bool(metadata.predefined),
    )
end

function _restore_side(
    payload, split, manifest_sha256, scheduler::String, hyperparameters,
)
    trainer, sampler, _ = withenv(
        "EVRL_SCHEDULER" => scheduler,
        "EVRL_CPUSET_MODE" => "none",
        "EVRL_QUEUE_CHUNK" => "8",
        "EVRL_ADAPTIVE_TAIL" => "0",
    ) do
        Training.restore_checkpoint(
            payload,
            split,
            payload.split_metadata,
            manifest_sha256,
            hyperparameters,
        )
    end
    optimizer = payload.config.hyperparameters.optimizer
    Model.configure_bank_optimizer!(
        trainer.optimizer;
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        epsilon=optimizer.epsilon,
        bank_learning_rate=optimizer.bank_learning_rate,
        bank_weight_decay=optimizer.bank_weight_decay,
    )
    return trainer, sampler
end

function _reset_accumulators!(trainer)
    Model.reset_gradients!(trainer.scheduler.merged_accumulator)
    for accumulator in trainer.thread_accumulators
        Model.reset_gradients!(accumulator)
    end
    return trainer.scheduler.merged_accumulator
end

function _serial_oracle_accumulate!(trainer, batches, expected_update, hyperparameters)
    merged = _reset_accumulators!(trainer)
    state_results = Training._accumulate_dynamic_batches!(
        trainer,
        batches;
        expected_update,
        hyperparameters,
        baseline=trainer.baseline,
    )
    # This is intentionally the same fixed worker-index reduction and exact
    # four-state mean used by the training driver.
    for accumulator in trainer.thread_accumulators
        Model.merge_gradients!(merged, accumulator)
    end
    Model.scale_gradients!(merged, inv(Float32(length(state_results))))
    return state_results, merged
end

function _barrierless_accumulate!(
    executor,
    trainer,
    batches,
    expected_update,
    hyperparameters,
)
    _reset_accumulators!(trainer)
    # Invoke the production accumulation path, including its usage-recording
    # and candidate-region accounting, rather than a smoke-only scheduler.
    state_results = Training._accumulate_barrierless_batches!(
        trainer,
        batches;
        expected_update,
        hyperparameters,
        baseline=trainer.baseline,
    )
    # The production post phase owns scheduler.merged_accumulator and requires
    # it to remain empty until barrierless_reduce_and_optimizer!.  Build an
    # independent witness accumulator so the unclipped 4-state mean can still
    # be compared before the production parallel optimizer mutates gradients.
    witness = Model.GradientAccumulator(trainer.model)
    for accumulator in Training.barrierless_worker_accumulators(executor)
        Model.merge_gradients!(witness, accumulator)
    end
    Model.scale_gradients!(witness, inv(Float32(length(state_results))))
    return state_results, witness
end

function _finalize_update_state!(
    trainer,
    state_results,
    expected_update,
    hyperparameters,
)
    trainer.baseline = muladd(
        0.99f0,
        trainer.baseline,
        0.01f0 * mean(
            result.loss + hyperparameters.halting.compute_price *
                Float32(mean(result.depths))
            for result in state_results
        ),
    )
    trainer.update = expected_update
    return trainer
end

function _optimizer_step!(trainer, accumulator, state_results, expected_update, hyperparameters)
    optimizer = hyperparameters.optimizer
    episodic_scale = expected_update > optimizer.episodic_decay_after_update ?
        optimizer.episodic_decay_factor : 1.0f0
    record = Model.optimizer_step!(
        trainer.model,
        trainer.optimizer,
        accumulator;
        clip_norm=optimizer.gradient_clip_norm,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        epsilon=optimizer.epsilon,
        router_learning_rate=optimizer.router_learning_rate,
        lookup_alpha_learning_rate=optimizer.lookup_alpha_learning_rate,
        attention_learning_rate=optimizer.attention_learning_rate * episodic_scale,
        ffn_learning_rate=optimizer.ffn_learning_rate * episodic_scale,
        token_learning_rate=optimizer.token_learning_rate * episodic_scale,
        register_learning_rate=optimizer.register_learning_rate * episodic_scale,
        head_learning_rate=optimizer.head_learning_rate * episodic_scale,
        halt_learning_rate=optimizer.halt_learning_rate,
        dense_weight_decay=optimizer.dense_weight_decay,
    )
    _finalize_update_state!(
        trainer, state_results, expected_update, hyperparameters,
    )
    return record
end

function _rng_probe(rng)
    probe = copy(rng)
    return ntuple(_ -> Random.rand(probe, UInt64), 16)
end

function _serialized_bytes(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

_rng_state_exact(left, right) =
    _serialized_bytes(left) == _serialized_bytes(right)

function _sampler_state_exact(left, right)
    left_snapshot = TrainingCore.sampler_snapshot(left)
    right_snapshot = TrainingCore.sampler_snapshot(right)
    return left_snapshot.format_version == right_snapshot.format_version &&
        left_snapshot.source_rows == right_snapshot.source_rows &&
        left_snapshot.permutation == right_snapshot.permutation &&
        left_snapshot.cursor == right_snapshot.cursor &&
        left_snapshot.completed_epochs == right_snapshot.completed_epochs &&
        _rng_state_exact(left_snapshot.rng, right_snapshot.rng)
end

function _sampler_next_rows(sampler)
    snapshot = TrainingCore.sampler_snapshot(sampler)
    probe = TrainingCore.restore_sampler(snapshot.source_rows, snapshot)
    return Tuple(TrainingCore.next_batch!(probe, Training.TRAINING_STATE_BATCH))
end

@inline _matrix_integer_tuple(matrix) = Tuple(
    Tuple(Int(matrix[row, column]) for row in axes(matrix, 1))
    for column in axes(matrix, 2)
)

function _tape_token_edges(tape)
    return [begin
        cross = step.cross
        (;
            support="fixed-k$(Model.EPISODIC_SUPPORT)-register-specific",
            selected_ids=_matrix_integer_tuple(cross.selected_ids),
            key_count=size(cross.key, 2),
            attention_shape=size(cross.attention_weights),
            write_ids=Tuple(Int.(@view(
                step.memory_write.write_ids[
                    1:Int(step.memory_write.write_count)
                ],
            ))),
        )
    end for step in tape.steps]
end

function _tape_lookup_rows(tape)
    return [[[begin
        block_tape = step.lookup.blocks[block, register]
        ordered_columns = Int.(block_tape.columns)
        (;
            addresses=Int.(block_tape.addresses),
            ordered_columns,
            row_id_set=sort!(unique!(copy(ordered_columns))),
        )
    end for register in axes(step.lookup.blocks, 2)]
    for block in axes(step.lookup.blocks, 1)] for step in tape.steps]
end

function _discrete_snapshot(trainer, batches)
    counts = [Training._valid_candidate_count(batch) for batch in batches]
    candidate_seeds = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        copy(workspace.candidate_seeds[1:counts[state]])
    end for state in eachindex(batches)]
    forced_depths = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        Int.(workspace.forced_depths[1:counts[state]])
    end for state in eachindex(batches)]
    realized_depths = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        Int.(workspace.depths[1:counts[state]])
    end for state in eachindex(batches)]
    hard_halting = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [begin
            tape = workspace.tapes[candidate]
            tape === nothing && error("trajectory tape is missing")
            Int(workspace.depths[candidate]) == length(tape.steps) || error(
                "realized depth differs from trajectory length",
            )
            [(;
                stochastic=step.stochastic_decision,
                stopped=step.stopped,
                forced=step.forced_stop,
            ) for step in tape.steps]
        end for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    token_edges = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [_tape_token_edges(workspace.tapes[candidate])
         for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    lookup_rows = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [_tape_lookup_rows(workspace.tapes[candidate])
         for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    halt_probe_targets = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [reinterpret(UInt32, workspace.halt_probe_targets[candidate])
         for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    halt_probe_deltas = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [reinterpret(UInt32, workspace.halt_probe_deltas[candidate])
         for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    halt_trace_targets = [begin
        workspace = trainer.scheduler.state_workspaces[state]
        [copy(reinterpret(
            UInt32,
            @view(workspace.halt_trace_targets[:, candidate]),
        )) for candidate in 1:counts[state]]
    end for state in eachindex(batches)]
    return (;
        counts,
        candidate_seeds,
        forced_depths,
        realized_depths,
        hard_halting,
        token_edges,
        lookup_rows,
        halt_probe_targets,
        halt_probe_deltas,
        halt_trace_targets,
    )
end

function _signature_sha256(value)
    io = IOBuffer()
    serialize(io, value)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _output_error(serial_trainer, barrierless_trainer, batches)
    result = NumericError()
    for state in eachindex(batches)
        serial_raw = serial_trainer.scheduler.state_workspaces[state].raw
        async_raw = barrierless_trainer.scheduler.state_workspaces[state].raw
        _observe_array!(result, async_raw, serial_raw; atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    end
    return result
end

function _raw_vjp_error(serial_trainer, barrierless_trainer, batches)
    result = NumericError()
    for state in eachindex(batches)
        serial_vjp = serial_trainer.scheduler.raw_gradients[state]
        async_vjp = barrierless_trainer.scheduler.raw_gradients[state]
        _observe_array!(result, async_vjp, serial_vjp; atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    end
    return result
end

function _loss_error(serial_results, barrierless_results)
    length(serial_results) == length(barrierless_results) || error(
        "state result count differs",
    )
    result = NumericError()
    for state in eachindex(serial_results)
        _observe!(
            result,
            barrierless_results[state].loss,
            serial_results[state].loss;
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
        serial_components = serial_results[state].components
        async_components = barrierless_results[state].components
        propertynames(serial_components) == propertynames(async_components) ||
            error("loss component names differ")
        for name in propertynames(serial_components)
            _observe!(
                result,
                getproperty(async_components, name),
                getproperty(serial_components, name);
                atol=OUTPUT_ATOL,
                rtol=OUTPUT_RTOL,
            )
        end
    end
    return result
end

function _gradient_error(actual, reference)
    result = NumericError()
    actual.active_tokens == reference.active_tokens || error(
        "active-token mask differs",
    )
    actual.lookup.selected_row_events == reference.lookup.selected_row_events ||
        error("selected-row event count differs")

    for block in 1:Model.BLOCKS
        actual_bank = actual.lookup.bank_gradients[block]
        reference_bank = reference.lookup.bank_gradients[block]
        actual_columns = sort!(collect(keys(actual_bank)))
        reference_columns = sort!(collect(keys(reference_bank)))
        actual_columns == reference_columns || error(
            "sparse gradient row-ID support differs in block $block",
        )
        for column in reference_columns
            _observe_array!(result, actual_bank[column], reference_bank[column])
        end
        _observe_array!(
            result,
            actual.lookup.dbh4[block],
            reference.lookup.dbh4[block],
        )
    end
    for name in (
        :dalpha_logits, :dhead, :dbias, :dhalt_weight, :dhalt_bias,
        :dreinject_logit,
    )
        _observe_array!(
            result,
            getproperty(actual.lookup, name),
            getproperty(reference.lookup, name),
        )
    end
    actual_names = sort!(collect(keys(actual.dense)); by=String)
    reference_names = sort!(collect(keys(reference.dense)); by=String)
    actual_names == reference_names || error("dense gradient parameter set differs")
    for name in reference_names
        _observe_array!(result, actual.dense[name], reference.dense[name])
    end
    return result
end

function _gradient_diagnostics(actual, reference)
    records = NamedTuple[]
    for block in 1:Model.BLOCKS
        bank_error = NumericError()
        actual_bank = actual.lookup.bank_gradients[block]
        reference_bank = reference.lookup.bank_gradients[block]
        for column in sort!(collect(keys(reference_bank)))
            _observe_array!(
                bank_error, actual_bank[column], reference_bank[column],
            )
        end
        push!(records, (;
            name="lookup.bank_gradients[$block]",
            maximum_absolute=bank_error.maximum_absolute,
            relative_l2=_relative_l2(bank_error),
        ))
        dense_error = NumericError()
        _observe_array!(
            dense_error, actual.lookup.dbh4[block], reference.lookup.dbh4[block],
        )
        push!(records, (;
            name="lookup.dbh4[$block]",
            maximum_absolute=dense_error.maximum_absolute,
            relative_l2=_relative_l2(dense_error),
        ))
    end
    for name in (
        :dalpha_logits, :dhead, :dbias, :dhalt_weight, :dhalt_bias,
        :dreinject_logit,
    )
        error = NumericError()
        _observe_array!(
            error, getproperty(actual.lookup, name),
            getproperty(reference.lookup, name),
        )
        push!(records, (;
            name="lookup.$name",
            maximum_absolute=error.maximum_absolute,
            relative_l2=_relative_l2(error),
        ))
    end
    for name in sort!(collect(keys(reference.dense)); by=String)
        error = NumericError()
        _observe_array!(error, actual.dense[name], reference.dense[name])
        push!(records, (;
            name="dense.$name",
            maximum_absolute=error.maximum_absolute,
            relative_l2=_relative_l2(error),
        ))
    end
    sort!(records; by=record -> record.maximum_absolute, rev=true)
    return records[1:min(8, length(records))]
end

function _tree_error!(
    result::NumericError,
    actual,
    reference,
    path::String,
)
    typeof(actual) === typeof(reference) || error(
        "$path type differs: $(typeof(actual)) != $(typeof(reference))",
    )
    if actual isa AbstractArray{<:AbstractFloat}
        _observe_array!(result, actual, reference)
    elseif actual isa AbstractArray
        size(actual) == size(reference) || error("$path shape differs")
        if isbitstype(eltype(actual))
            actual == reference || error("$path exact array differs")
        else
            @inbounds for index in eachindex(actual, reference)
                _tree_error!(result, actual[index], reference[index], "$path[$index]")
            end
        end
    elseif actual isa AbstractFloat
        _observe!(result, actual, reference)
    elseif actual isa Dict
        actual_keys = sort!(collect(keys(actual)); by=string)
        reference_keys = sort!(collect(keys(reference)); by=string)
        actual_keys == reference_keys || error("$path dictionary keys differ")
        for key in reference_keys
            _tree_error!(result, actual[key], reference[key], "$path[$key]")
        end
    elseif actual isa Tuple
        length(actual) == length(reference) || error("$path tuple length differs")
        for index in eachindex(actual)
            _tree_error!(result, actual[index], reference[index], "$path[$index]")
        end
    elseif actual isa Integer || actual isa Bool || actual isa Symbol ||
           actual isa AbstractString || actual === nothing
        actual == reference || error("$path exact value differs")
    elseif isstructtype(typeof(actual))
        for name in fieldnames(typeof(actual))
            _tree_error!(
                result,
                getfield(actual, name),
                getfield(reference, name),
                "$path.$name",
            )
        end
    else
        isequal(actual, reference) || error("$path value differs")
    end
    return result
end

function _optimizer_clocks(optimizer)
    return (;
        lookup_step=optimizer.lookup.step,
        lookup_dense_step=optimizer.lookup.dense.step,
        bank_global_steps=Tuple(state.global_step for state in optimizer.lookup.bank_states),
        bank_global_log_decay_bits=Tuple(
            reinterpret(UInt64, state.global_log_decay)
            for state in optimizer.lookup.bank_states
        ),
        bank_last_log_decay_bits=Tuple(
            copy(reinterpret(UInt64, state.last_log_decay))
            for state in optimizer.lookup.bank_states
        ),
        dense_step=optimizer.dense_step,
        token_event_count=copy(optimizer.token_event_count),
        bank_event_count=Tuple(copy(state.event_count) for state in optimizer.lookup.bank_states),
        bank_last_event_step=Tuple(
            copy(state.last_event_step) for state in optimizer.lookup.bank_states
        ),
    )
end

function main()
    BLAS.set_num_threads(1)
    _require(BLAS.get_num_threads() == 1, "nested BLAS parallelism is forbidden")
    _require(
        Base.Threads.nthreads(:default) == 20,
        "correctness smoke requires JULIA_NUM_THREADS=20,0",
    )
    _require(
        Base.Threads.nthreads(:interactive) == 0,
        "correctness smoke forbids interactive Julia workers",
    )

    checkpoint = abspath(get(ENV, "EVRL_SMOKE_CHECKPOINT", DEFAULT_CHECKPOINT))
    checkpoint_sha256 = lowercase(get(
        ENV,
        "EVRL_SMOKE_CHECKPOINT_SHA256",
        DEFAULT_CHECKPOINT_SHA256,
    ))
    # Two deserialize operations are mandatory: neither side may share mutable
    # model, optimizer, RNG, usage, or sampler objects with the other.
    serial_payload, serial_artifact = Training.read_checkpoint(
        checkpoint,
        checkpoint_sha256,
    )
    barrierless_payload, barrierless_artifact = Training.read_checkpoint(
        checkpoint,
        checkpoint_sha256,
    )
    serial_artifact.sha256 == barrierless_artifact.sha256 || error(
        "checkpoint deserialize sides have different source hashes",
    )

    dataset_path = abspath(String(serial_payload.config.dataset_path))
    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    manifest_sha256 = Training._sha256_file(joinpath(dataset_path, "manifest.json"))
    split = _training_only_split(dataset, serial_payload.split_metadata)
    inherited_hyperparameters = Training._normalized_hyperparameters(
        serial_payload.config.hyperparameters,
    )
    probe_candidates_per_state = parse(Int, get(
        ENV,
        "EVRL_HALT_PROBES_PER_STATE",
        string(inherited_hyperparameters.halting.probe_candidates_per_state),
    ))
    probe_weight = parse(Float32, get(
        ENV,
        "EVRL_HALT_PROBE_WEIGHT",
        string(inherited_hyperparameters.halting.probe_weight),
    ))
    compute_price = parse(Float32, get(
        ENV,
        "EVRL_COMPUTE_PRICE",
        string(inherited_hyperparameters.halting.compute_price),
    ))
    halt_learning_rate = parse(Float32, get(
        ENV,
        "EVRL_LR_HALT",
        string(inherited_hyperparameters.optimizer.halt_learning_rate),
    ))
    dense_weight_decay = parse(Float32, get(
        ENV,
        "EVRL_WD_DENSE",
        string(inherited_hyperparameters.optimizer.dense_weight_decay),
    ))
    episodic_decay_after_update = parse(Int, get(
        ENV,
        "EVRL_EPISODIC_LR_DECAY_AFTER",
        string(inherited_hyperparameters.optimizer.episodic_decay_after_update),
    ))
    episodic_decay_factor = parse(Float32, get(
        ENV,
        "EVRL_EPISODIC_LR_DECAY_FACTOR",
        string(inherited_hyperparameters.optimizer.episodic_decay_factor),
    ))
    hyperparameters = merge(inherited_hyperparameters, (;
        optimizer=merge(inherited_hyperparameters.optimizer, (;
            halt_learning_rate,
            dense_weight_decay,
            episodic_decay_after_update,
            episodic_decay_factor,
        )),
        halting=merge(inherited_hyperparameters.halting, (;
            compute_price,
            probe_candidates_per_state,
            probe_weight,
        )),
    ))
    serial_trainer, serial_sampler = _restore_side(
        serial_payload,
        split,
        manifest_sha256,
        "serial",
        hyperparameters,
    )
    barrierless_trainer, barrierless_sampler = _restore_side(
        barrierless_payload,
        split,
        manifest_sha256,
        "barrierless",
        hyperparameters,
    )
    serial_trainer.update == barrierless_trainer.update || error(
        "restored update differs",
    )
    serial_payload.config.hyperparameters == barrierless_payload.config.hyperparameters || error(
        "restored hyperparameters differ",
    )
    serial_rows = TrainingCore.next_batch!(
        serial_sampler,
        Training.TRAINING_STATE_BATCH,
    )
    barrierless_rows = TrainingCore.next_batch!(
        barrierless_sampler,
        Training.TRAINING_STATE_BATCH,
    )
    serial_rows == barrierless_rows || error("training sampler row sequence differs")
    training_row_set = Set(split.training_rows)
    all(row -> row in training_row_set, serial_rows) || error(
        "smoke batch escaped the recorded training split",
    )
    training_seed_ids = Int.(dataset.split_group_ids[serial_rows])
    all(seed -> !(seed in FORBIDDEN_TEACHER_SEEDS), training_seed_ids) || error(
        "smoke batch contains a forbidden validation/sealed teacher seed",
    )
    next_training_rows = _sampler_next_rows(serial_sampler)
    next_training_seed_ids = Int.(
        dataset.split_group_ids[collect(next_training_rows)],
    )
    all(seed -> !(seed in FORBIDDEN_TEACHER_SEEDS), next_training_seed_ids) || error(
        "post-smoke sampler rows contain a forbidden validation/sealed teacher seed",
    )
    batches = [
        TrainingCore.allocate_host_batch(1; max_candidates=Training.LEARNER_WIDTH)
        for _ in 1:Training.TRAINING_STATE_BATCH
    ]
    smoke_candidate_cap = parse(Int, get(
        ENV, "EVRL_SMOKE_CANDIDATE_CAP", string(Training.LEARNER_WIDTH),
    ))
    1 <= smoke_candidate_cap <= Training.LEARNER_WIDTH || error(
        "EVRL_SMOKE_CANDIDATE_CAP is outside the learner width",
    )
    for (batch, row) in zip(batches, serial_rows)
        TrainingCore.pack_batch!(batch, dataset, [row])
        if smoke_candidate_cap < size(batch.mask, 1)
            @views fill!(batch.mask[(smoke_candidate_cap + 1):end, :], 0.0f0)
        end
    end

    expected_update = serial_trainer.update + 1
    serial_results, serial_gradient = _serial_oracle_accumulate!(
        serial_trainer,
        batches,
        expected_update,
        hyperparameters,
    )
    # A batch-eight discrete witness contains hundreds of deeply nested
    # trajectory tuples.  Keep it out of the coordinator closure's concrete
    # type so Julia does not recursively specialize the entire witness.
    serial_discrete_ref = Ref{Any}(
        _discrete_snapshot(serial_trainer, batches),
    )

    executor = barrierless_trainer.scheduler.barrierless_executor
    executor === nothing && error("barrierless executor was not attached")
    smoke_result = Training.run_with_barrierless_team!(executor) do live_executor
        barrierless_results, barrierless_gradient = _barrierless_accumulate!(
            live_executor,
            barrierless_trainer,
            batches,
            expected_update,
            hyperparameters,
        )
        serial_discrete = serial_discrete_ref[]
        barrierless_discrete_ref = Ref{Any}(
            _discrete_snapshot(barrierless_trainer, batches),
        )
        barrierless_discrete = barrierless_discrete_ref[]
        route_usage_exact =
            barrierless_trainer.usage.trajectories ==
                serial_trainer.usage.trajectories &&
            barrierless_trainer.usage.recurrent_steps ==
                serial_trainer.usage.recurrent_steps &&
            barrierless_trainer.usage.sparse.trajectories ==
                serial_trainer.usage.sparse.trajectories &&
            barrierless_trainer.usage.sparse.recurrent_steps ==
                serial_trainer.usage.sparse.recurrent_steps &&
            barrierless_trainer.usage.sparse.block_visits ==
                serial_trainer.usage.sparse.block_visits &&
            barrierless_trainer.usage.sparse.counts ==
                serial_trainer.usage.sparse.counts

        structural_exact = (;
            candidate_seed=
                barrierless_discrete.candidate_seeds == serial_discrete.candidate_seeds,
            forced_depth=
                barrierless_discrete.forced_depths == serial_discrete.forced_depths,
            realized_depth=
                barrierless_discrete.realized_depths == serial_discrete.realized_depths,
            hard_halting=
                barrierless_discrete.hard_halting == serial_discrete.hard_halting,
            selected_token_edge=
                barrierless_discrete.token_edges == serial_discrete.token_edges,
            lookup_row_id=
                barrierless_discrete.lookup_rows == serial_discrete.lookup_rows,
            halt_probe_target=
                barrierless_discrete.halt_probe_targets ==
                    serial_discrete.halt_probe_targets,
            halt_probe_delta=
                barrierless_discrete.halt_probe_deltas ==
                    serial_discrete.halt_probe_deltas,
            halt_trace_target=
                barrierless_discrete.halt_trace_targets ==
                    serial_discrete.halt_trace_targets,
            active_token_mask=
                barrierless_gradient.active_tokens == serial_gradient.active_tokens,
            selected_row_events=
                barrierless_gradient.lookup.selected_row_events ==
                    serial_gradient.lookup.selected_row_events,
            route_usage=route_usage_exact,
        )
        all(values(structural_exact)) || error(
            "one or more exact smoke invariants differ: $structural_exact",
        )

        output_error = _output_error(serial_trainer, barrierless_trainer, batches)
        loss_error = _loss_error(serial_results, barrierless_results)
        raw_vjp_error = _raw_vjp_error(serial_trainer, barrierless_trainer, batches)
        for (name, error_record) in (
            :output => output_error,
            :loss => loss_error,
            :raw_vjp => raw_vjp_error,
        )
            error_record.tolerance_violations == 0 || error(
                "$name exceeds atol=$OUTPUT_ATOL, rtol=$OUTPUT_RTOL",
            )
        end

        gradient_error = _gradient_error(barrierless_gradient, serial_gradient)
        gradient_relative_l2 = _relative_l2(gradient_error)
        gradient_error.maximum_absolute <= GRADIENT_ATOL || error(
            "parameter gradient exceeds atol=$GRADIENT_ATOL: " *
            "max_abs=$(gradient_error.maximum_absolute), " *
            "relative_l2=$gradient_relative_l2, " *
            "largest=$(_gradient_diagnostics(barrierless_gradient, serial_gradient))",
        )
        gradient_relative_l2 <= GRADIENT_RELATIVE_L2 || error(
            "parameter gradient exceeds relative L2=$GRADIENT_RELATIVE_L2",
        )

        serial_optimizer_record = _optimizer_step!(
            serial_trainer,
            serial_gradient,
            serial_results,
            expected_update,
            hyperparameters,
        )
        # Exercise the production parameter-parallel reduction and optimizer,
        # not a smoke-only copy of that code.
        barrierless_optimizer_record = Training.barrierless_reduce_and_optimizer!(
            live_executor,
            barrierless_trainer,
            hyperparameters,
            expected_update,
        )
        Training.finish_barrierless_update!(live_executor)
        _finalize_update_state!(
            barrierless_trainer,
            barrierless_results,
            expected_update,
            hyperparameters,
        )
        # Both optimizer paths leave their merged accumulator scaled by the
        # global clip factor.  This directly checks the production parallel
        # reduction result rather than only the smoke witness reduction.
        postphase_gradient_error = _gradient_error(
            barrierless_trainer.scheduler.merged_accumulator,
            serial_gradient,
        )
        postphase_gradient_relative_l2 = _relative_l2(
            postphase_gradient_error,
        )
        postphase_gradient_error.maximum_absolute <= GRADIENT_ATOL || error(
            "production postphase gradient exceeds atol=$GRADIENT_ATOL",
        )
        postphase_gradient_relative_l2 <= GRADIENT_RELATIVE_L2 || error(
            "production postphase gradient exceeds relative L2=$(GRADIENT_RELATIVE_L2)",
        )
        serial_clocks = _optimizer_clocks(serial_trainer.optimizer)
        barrierless_clocks = _optimizer_clocks(barrierless_trainer.optimizer)
        post_exact = (;
            optimizer_clock=barrierless_clocks == serial_clocks,
            halt_rng_state=
                _rng_state_exact(
                    barrierless_trainer.halt_rng,
                    serial_trainer.halt_rng,
                ),
            halt_rng_next_values=
                _rng_probe(barrierless_trainer.halt_rng) ==
                    _rng_probe(serial_trainer.halt_rng),
            sampler_state=_sampler_state_exact(
                barrierless_sampler, serial_sampler,
            ),
            sampler_next_rows=
                _sampler_next_rows(barrierless_sampler) ==
                    next_training_rows,
            trainer_update=barrierless_trainer.update == serial_trainer.update,
        )
        exact = merge(structural_exact, post_exact)
        all(values(post_exact)) || error(
            "one or more exact post-update invariants differ: $post_exact",
        )

        post_optimizer_error = NumericError()
        _tree_error!(
            post_optimizer_error,
            barrierless_trainer.model,
            serial_trainer.model,
            "model",
        )
        _tree_error!(
            post_optimizer_error,
            barrierless_trainer.optimizer,
            serial_trainer.optimizer,
            "optimizer",
        )
        _observe!(
            post_optimizer_error,
            barrierless_trainer.baseline,
            serial_trainer.baseline,
        )
        post_relative_l2 = _relative_l2(post_optimizer_error)
        post_optimizer_error.maximum_absolute <= GRADIENT_ATOL || error(
            "post-optimizer parameter/state exceeds atol=$GRADIENT_ATOL",
        )
        post_relative_l2 <= GRADIENT_RELATIVE_L2 || error(
            "post-optimizer parameter/state exceeds relative L2=$GRADIENT_RELATIVE_L2",
        )
        serial_optimizer_record.step == barrierless_optimizer_record.step || error(
            "optimizer record step differs",
        )
        serial_optimizer_record.active_columns ==
            barrierless_optimizer_record.active_columns || error(
            "optimizer active-column telemetry differs",
        )
        serial_optimizer_record.active_elements ==
            barrierless_optimizer_record.active_elements || error(
            "optimizer active-element telemetry differs",
        )
        optimizer_telemetry_error = NumericError()
        _observe!(
            optimizer_telemetry_error,
            barrierless_optimizer_record.gradient_norm,
            serial_optimizer_record.gradient_norm;
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
        _observe!(
            optimizer_telemetry_error,
            barrierless_optimizer_record.gradient_scale,
            serial_optimizer_record.gradient_scale;
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
        optimizer_telemetry_error.tolerance_violations == 0 || error(
            "optimizer telemetry exceeds output tolerance",
        )

        return (;
            status="pass",
            checkpoint=(;
                path=serial_artifact.path,
                sha256=serial_artifact.sha256,
                source_update=Int(serial_payload.update),
            ),
            update=expected_update,
            states=Training.TRAINING_STATE_BATCH,
            candidate_cap=smoke_candidate_cap,
            training_rows=Int.(serial_rows),
            training_seed_ids,
            candidate_counts=collect(serial_discrete.counts),
            workers=Base.Threads.nthreads(:default),
            exact,
            discrete_signature_sha256=_signature_sha256(serial_discrete),
            numeric=(;
                output=_numeric_record(output_error),
                loss=_numeric_record(loss_error),
                raw_vjp=_numeric_record(raw_vjp_error),
                worker_gradient_witness=_numeric_record(gradient_error),
                parameter_gradient=_numeric_record(
                    postphase_gradient_error,
                ),
                post_optimizer_parameter_state=
                    _numeric_record(post_optimizer_error),
                optimizer_telemetry=_numeric_record(
                    optimizer_telemetry_error,
                ),
            ),
            optimizer_clock=(;
                lookup_step=Int(serial_clocks.lookup_step),
                lookup_dense_step=Int(serial_clocks.lookup_dense_step),
                bank_global_steps=Int.(serial_clocks.bank_global_steps),
                dense_step=Int(serial_clocks.dense_step),
                token_clock_exact=true,
                sparse_row_clocks_exact=true,
            ),
            usage_exact=route_usage_exact,
        )
    end

    JSON3.pretty(stdout, smoke_result)
    write(stdout, '\n')
    return smoke_result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
