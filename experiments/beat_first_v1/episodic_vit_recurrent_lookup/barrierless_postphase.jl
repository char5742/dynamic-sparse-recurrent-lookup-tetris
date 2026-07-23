"""
Parameter-parallel reduction and optimizer tail for the barrierless executor.

This file is included inside `EpisodicViTRecurrentLookupTeacherTraining`.  It
does not alter the checkpointed model or optimizer types.  Post-phase runtime
is keyed by the non-serialized `BarrierlessExecutor` and uses the same bounded
MPMC queue and persistent native-worker team as candidate execution.

`barrierless_executor.jl` must route `BARRIERLESS_POST_JOB_KIND` before
constructing `BarrierlessJobKind(job.kind)` and must keep background workers in
the queue until `barrierless_postphase_complete(executor)` becomes true.  The
small integration hooks are listed at the end of this file.
"""

const BARRIERLESS_POST_JOB_KIND = UInt8(0xfe)

@enum BarrierlessPostOpcode::UInt8 begin
    BARRIERLESS_POST_NONE = 0
    BARRIERLESS_POST_REDUCE_BANK = 1
    BARRIERLESS_POST_REDUCE_LOOKUP = 2
    BARRIERLESS_POST_REDUCE_DENSE = 3
    BARRIERLESS_POST_REDUCE_ACTIVE_TOKENS = 4
    BARRIERLESS_POST_SCALE_BANK = 5
    BARRIERLESS_POST_SCALE_LOOKUP = 6
    BARRIERLESS_POST_SCALE_DENSE = 7
    BARRIERLESS_POST_OPTIMIZE_BANK = 8
    BARRIERLESS_POST_OPTIMIZE_LOOKUP_ROUTER = 9
    BARRIERLESS_POST_OPTIMIZE_LOOKUP_HEAD = 10
    BARRIERLESS_POST_OPTIMIZE_LOOKUP_HALT = 11
    BARRIERLESS_POST_OPTIMIZE_LOOKUP_REINJECT = 12
    BARRIERLESS_POST_OPTIMIZE_DENSE = 13
end

struct BarrierlessPostWork
    opcode::UInt8
    index::UInt16
end

Base.zero(::Type{BarrierlessPostWork}) = BarrierlessPostWork(
    UInt8(BARRIERLESS_POST_NONE), UInt16(0),
)

const BARRIERLESS_POST_WORK_CAPACITY = 128

# This order is both the model's `_dense_parameters` order and the stable order
# used by the legacy merge/update path.  It avoids `collect(keys(dict))` in the
# update hot path.
const BARRIERLESS_DENSE_PARAMETER_NAMES = (
    :cell_projection,
    :cell_bias,
    :cell_position,
    :visual_depthwise,
    :visual_channel_mix,
    :visual_pointwise,
    :visual_scale_logit,
    :next_projection,
    :next_bias,
    :next_position,
    :aux_value,
    :aux_position,
    :register_seed,
    :spatial_q,
    :spatial_k,
    :spatial_v,
    :spatial_o,
    :spatial_relative_bias,
    :spatial_scale_logit,
    :recurrent_depthwise,
    :recurrent_depthwise_scale_logit,
    :cross_q,
    :cross_k,
    :cross_v,
    :cross_o,
    :cross_scale_logit,
    :relation_scale_logit,
    :memory_write_v,
    :memory_write_o,
    :memory_write_scale_logit,
    :self_q,
    :self_k,
    :self_v,
    :self_o,
    :self_scale_logit,
    :ffn_gate,
    :ffn_up,
    :ffn_down,
    :ffn_scale_logit,
    :lookup_register_gate,
)

const BARRIERLESS_LOOKUP_GRADIENT_COUNT =
    Model.SparseLookup.BLOCKS + 6

mutable struct BarrierlessPostRuntime
    complete::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    works::Vector{BarrierlessPostWork}
    jobs::Vector{BarrierlessJob}
    work_count::Int
    row_ids::NTuple{Model.SparseLookup.BLOCKS,Vector{Int32}}
    worker_row_scratch::Vector{Vector{Int32}}
    scale::Float32
    next_step::UInt64
    bank_active_columns::Vector{Int}
    bank_active_elements::Vector{Int}
end

function BarrierlessPostRuntime(executor::BarrierlessExecutor)
    maximum_bank_columns =
        Model.SparseLookup.TABLES_PER_BLOCK * Model.SparseLookup.ROWS_PER_TABLE
    row_ids = ntuple(_ -> begin
        rows = Int32[]
        sizehint!(rows, maximum_bank_columns)
        rows
    end, Model.SparseLookup.BLOCKS)
    worker_row_scratch = [
        begin
            rows = Int32[]
            sizehint!(rows, maximum_bank_columns)
            rows
        end
        for _ in 1:(executor.active_workers * Model.SparseLookup.BLOCKS)
    ]
    return BarrierlessPostRuntime(
        Base.Threads.Atomic{UInt32}(UInt32(1)),
        Base.Threads.Atomic{Int}(0),
        fill(zero(BarrierlessPostWork), BARRIERLESS_POST_WORK_CAPACITY),
        fill(zero(BarrierlessJob), BARRIERLESS_POST_WORK_CAPACITY),
        0,
        row_ids,
        worker_row_scratch,
        1.0f0,
        UInt64(0),
        zeros(Int, Model.SparseLookup.BLOCKS),
        zeros(Int, Model.SparseLookup.BLOCKS),
    )
end

const _BARRIERLESS_POST_RUNTIME_LOCK = ReentrantLock()
const _BARRIERLESS_POST_RUNTIMES = IdDict{Any,BarrierlessPostRuntime}()

function _barrierless_post_runtime(executor::BarrierlessExecutor)
    runtime = get(_BARRIERLESS_POST_RUNTIMES, executor, nothing)
    runtime === nothing || return runtime
    lock(_BARRIERLESS_POST_RUNTIME_LOCK)
    try
        return get!(_BARRIERLESS_POST_RUNTIMES, executor) do
            BarrierlessPostRuntime(executor)
        end
    finally
        unlock(_BARRIERLESS_POST_RUNTIME_LOCK)
    end
end

"""Open the post-gradient boundary before publishing an update's PREPARE jobs."""
function barrierless_begin_postphase_update!(executor::BarrierlessExecutor)
    runtime = _barrierless_post_runtime(executor)
    runtime.remaining[] == 0 || error("stale barrierless post-phase work")
    runtime.complete[] = UInt32(0)
    runtime.work_count = 0
    runtime.scale = 1.0f0
    runtime.next_step = UInt64(0)
    fill!(runtime.bank_active_columns, 0)
    fill!(runtime.bank_active_elements, 0)
    return runtime
end

"""Restore the post-phase stop predicate when update preparation aborts."""
function barrierless_abort_postphase_update!(executor::BarrierlessExecutor)
    runtime = get(_BARRIERLESS_POST_RUNTIMES, executor, nothing)
    runtime === nothing && return executor
    runtime.remaining[] = 0
    runtime.work_count = 0
    runtime.complete[] = UInt32(1)
    Queue.wake_consumers!(executor.queue)
    return executor
end

"""Predicate used by persistent workers' update-boundary stop condition."""
function barrierless_postphase_complete(executor::BarrierlessExecutor)
    runtime = get(_BARRIERLESS_POST_RUNTIMES, executor, nothing)
    return runtime === nothing || runtime.complete[] != 0
end

@inline barrierless_is_post_job(kind::UInt8) = kind == BARRIERLESS_POST_JOB_KIND

@inline function _barrierless_post_worker_rows(
    runtime::BarrierlessPostRuntime,
    executor::BarrierlessExecutor,
    worker::Int,
    block::Int,
)
    return @inbounds runtime.worker_row_scratch[
        (block - 1) * executor.active_workers + worker
    ]
end

function _barrierless_prepare_bank_rows!(
    runtime::BarrierlessPostRuntime,
    executor::BarrierlessExecutor,
    destination,
)
    sparse_destination = destination.lookup
    @inbounds for block in 1:Model.SparseLookup.BLOCKS
        destination_dictionary = sparse_destination.bank_gradients[block]
        isempty(destination_dictionary) || error(
            "merged sparse gradient must be empty before post reduction",
        )
        rows = runtime.row_ids[block]
        empty!(rows)
        # Match the legacy merge: source workers in fixed ID order and sorted
        # columns within each source.  Scratch vectors are retained per worker.
        for worker in 1:executor.active_workers
            source_dictionary =
                executor.worker_accumulators[worker].lookup.bank_gradients[block]
            scratch = _barrierless_post_worker_rows(
                runtime, executor, worker, block,
            )
            resize!(scratch, length(source_dictionary))
            position = 0
            for column in keys(source_dictionary)
                position += 1
                scratch[position] = column
            end
            sort!(scratch)
            for column in scratch
                if !haskey(destination_dictionary, column)
                    Model.SparseLookup._bank_gradient_target!(
                        sparse_destination, block, column,
                    )
                    push!(rows, column)
                end
            end
        end
        isempty(rows) && error("lookup block $block has no selected gradient")
        events = UInt64(0)
        for worker in 1:executor.active_workers
            events += executor.worker_accumulators[worker].lookup.
                selected_row_events[block]
        end
        sparse_destination.selected_row_events[block] = events
    end
    return nothing
end

@inline function _barrierless_bank_range(runtime, executor, encoded::Int)
    workers = executor.active_workers
    block = fld(encoded - 1, workers) + 1
    lane = mod(encoded - 1, workers) + 1
    1 <= block <= Model.SparseLookup.BLOCKS || throw(BoundsError(
        runtime.row_ids, block,
    ))
    rows = runtime.row_ids[block]
    first = fld((lane - 1) * length(rows), workers) + 1
    last = fld(lane * length(rows), workers)
    return block, rows, first, last
end

@inline function _barrierless_lookup_gradient(accumulator, index::Int)
    lookup = accumulator.lookup
    blocks = Model.SparseLookup.BLOCKS
    index <= blocks && return lookup.dbh4[index]
    offset = index - blocks
    offset == 1 && return lookup.dalpha_logits
    offset == 2 && return lookup.dhead
    offset == 3 && return lookup.dbias
    offset == 4 && return lookup.dhalt_weight
    offset == 5 && return lookup.dhalt_bias
    offset == 6 && return lookup.dreinject_logit
    throw(BoundsError("lookup gradient", index))
end

function _barrierless_reduce_bank!(runtime, executor, destination, encoded::Int)
    block, rows, first, last = _barrierless_bank_range(
        runtime, executor, encoded,
    )
    first > last && return nothing
    target_dictionary = destination.lookup.bank_gradients[block]
    @inbounds for position in first:last
        column = rows[position]
        target = target_dictionary[column]
        for worker in 1:executor.active_workers
            source_dictionary =
                executor.worker_accumulators[worker].lookup.bank_gradients[block]
            source = get(source_dictionary, column, nothing)
            source === nothing && continue
            @simd for coordinate in eachindex(target)
                target[coordinate] += source[coordinate]
            end
        end
    end
    return nothing
end

function _barrierless_reduce_lookup!(executor, destination, index::Int)
    target = _barrierless_lookup_gradient(destination, index)
    @inbounds for worker in 1:executor.active_workers
        source = _barrierless_lookup_gradient(
            executor.worker_accumulators[worker], index,
        )
        target .+= source
    end
    return nothing
end

@inline function _barrierless_add_dense_array!(
    target::Array{Float32,N},
    source::Array{Float32,N},
) where {N}
    axes(target) == axes(source) || throw(DimensionMismatch(
        "dense gradient axes differ: $(axes(target)) != $(axes(source))",
    ))
    @inbounds @simd for element in eachindex(target, source)
        target[element] += source[element]
    end
    return nothing
end

function _barrierless_reduce_dense!(executor, destination, index::Int)
    name = BARRIERLESS_DENSE_PARAMETER_NAMES[index]
    target = destination.dense[name]
    @inbounds for worker in 1:executor.active_workers
        source = executor.worker_accumulators[worker].dense[name]
        _barrierless_add_dense_array!(target, source)
    end
    return nothing
end

function _barrierless_reduce_active_tokens!(executor, destination)
    @inbounds for worker in 1:executor.active_workers
        destination.active_tokens .|=
            executor.worker_accumulators[worker].active_tokens
    end
    return nothing
end

function _barrierless_scale_bank!(runtime, executor, destination, encoded::Int)
    block, rows, first, last = _barrierless_bank_range(
        runtime, executor, encoded,
    )
    first > last && return nothing
    dictionary = destination.lookup.bank_gradients[block]
    scale = runtime.scale
    @inbounds for position in first:last
        column = rows[position]
        gradient = dictionary[column]
        @simd for coordinate in eachindex(gradient)
            gradient[coordinate] *= scale
        end
    end
    return nothing
end

function _barrierless_scale_lookup!(runtime, destination, index::Int)
    gradient = _barrierless_lookup_gradient(destination, index)
    scale = runtime.scale
    @inbounds @simd for element in eachindex(gradient)
        gradient[element] *= scale
    end
    return nothing
end

@inline function _barrierless_scale_dense_array!(
    gradient::Array{Float32,N},
    scale::Float32,
) where {N}
    @inbounds @simd for element in eachindex(gradient)
        gradient[element] *= scale
    end
    return nothing
end

@inline function _barrierless_scale_selected_columns!(
    gradient::Matrix{Float32},
    active_tokens::BitVector,
    token_offset::Int,
    scale::Float32,
)
    @inbounds for token in eachindex(active_tokens)
        active_tokens[token] || continue
        column = token - token_offset
        1 <= column <= size(gradient, 2) || continue
        @simd for row in axes(gradient, 1)
            gradient[row, column] *= scale
        end
    end
    return nothing
end

function _barrierless_scale_dense!(runtime, destination, index::Int)
    name = BARRIERLESS_DENSE_PARAMETER_NAMES[index]
    gradient = destination.dense[name]
    scale = runtime.scale
    token_offset = Model._token_column_offset(name)
    if token_offset < 0
        _barrierless_scale_dense_array!(gradient, scale)
        return nothing
    end
    _barrierless_scale_selected_columns!(
        gradient, destination.active_tokens, token_offset, scale,
    )
    return nothing
end

function _barrierless_direct_sparse_adam_step!(
    theta,
    state,
    dictionary,
    columns,
    input_records,
)
    isempty(columns) && error("lookup block has no selected gradient")
    sort!(columns)
    next_step = state.global_step + UInt64(1)
    next_log_decay = state.global_log_decay +
        log1p(-Float64(state.learning_rate) * Float64(state.weight_decay))
    @inbounds for id in columns
        column = Int(id)
        gradient_vector = dictionary[id]
        event = state.event_count[column] + UInt64(1)
        correction1 = 1.0f0 - state.beta1^event
        correction2 = 1.0f0 - state.beta2^event
        decay_scale = Float32(exp(
            next_log_decay - state.last_log_decay[column],
        ))
        @simd for coordinate in axes(theta, 1)
            gradient = gradient_vector[coordinate]
            updated_m = muladd(
                state.beta1,
                state.m[coordinate, column],
                (1.0f0 - state.beta1) * gradient,
            )
            updated_v = muladd(
                state.beta2,
                state.v[coordinate, column],
                (1.0f0 - state.beta2) * gradient * gradient,
            )
            theta[coordinate, column] = theta[coordinate, column] * decay_scale -
                state.learning_rate * (updated_m / correction1) /
                (sqrt(updated_v / correction2) + state.epsilon)
            state.m[coordinate, column] = updated_m
            state.v[coordinate, column] = updated_v
        end
        state.event_count[column] = event
        state.last_event_step[column] = next_step
        state.last_log_decay[column] = next_log_decay
    end
    state.global_step = next_step
    state.global_log_decay = next_log_decay
    return (
        global_step=next_step,
        input_records=input_records,
        active_columns=length(columns),
        active_elements=length(columns) * size(theta, 1),
    )
end

function _barrierless_optimize_bank!(runtime, trainer, block::Int)
    model = trainer.model.lookup
    optimizer = trainer.optimizer.lookup
    accumulator = trainer.scheduler.merged_accumulator.lookup
    telemetry = _barrierless_direct_sparse_adam_step!(
        model.banks[block],
        optimizer.bank_states[block],
        accumulator.bank_gradients[block],
        runtime.row_ids[block],
        Int(accumulator.selected_row_events[block]),
    )
    runtime.bank_active_columns[block] = telemetry.active_columns
    runtime.bank_active_elements[block] = telemetry.active_elements
    return nothing
end

@inline function _barrierless_sparse_adam_kwargs(opt)
    return (
        beta1=opt.beta1,
        beta2=opt.beta2,
        epsilon=opt.epsilon,
        weight_decay=opt.dense_weight_decay,
    )
end

function _barrierless_optimize_lookup_router!(runtime, trainer, opt)
    model = trainer.model.lookup
    state = trainer.optimizer.lookup.dense
    gradient = trainer.scheduler.merged_accumulator.lookup
    kwargs = _barrierless_sparse_adam_kwargs(opt)
    @inbounds for block in 1:Model.SparseLookup.BLOCKS
        Model.SparseLookup._adam_update!(
            model.bh4_diagonals[block], state.mbh4[block], state.vbh4[block],
            gradient.dbh4[block], runtime.next_step, opt.router_learning_rate;
            kwargs...,
        )
    end
    Model.SparseLookup._adam_update!(
        model.alpha_logits, state.malpha, state.valpha, gradient.dalpha_logits,
        runtime.next_step, opt.lookup_alpha_learning_rate; kwargs...,
    )
    return nothing
end

function _barrierless_optimize_lookup_head!(runtime, trainer, opt, lr_scale)
    model = trainer.model.lookup
    state = trainer.optimizer.lookup.dense
    gradient = trainer.scheduler.merged_accumulator.lookup
    kwargs = _barrierless_sparse_adam_kwargs(opt)
    learning_rate = opt.head_learning_rate * lr_scale
    Model.SparseLookup._adam_update!(
        model.head, state.mhead, state.vhead, gradient.dhead,
        runtime.next_step, learning_rate; kwargs...,
    )
    Model.SparseLookup._adam_update!(
        model.bias, state.mbias, state.vbias, gradient.dbias,
        runtime.next_step, learning_rate; kwargs...,
    )
    return nothing
end

function _barrierless_optimize_lookup_halt!(
    runtime, trainer, opt, learning_rate,
)
    model = trainer.model.lookup
    state = trainer.optimizer.lookup.dense
    gradient = trainer.scheduler.merged_accumulator.lookup
    kwargs = _barrierless_sparse_adam_kwargs(opt)
    Model.SparseLookup._adam_update!(
        model.halt_weight, state.mhalt_weight, state.vhalt_weight,
        gradient.dhalt_weight, runtime.next_step, learning_rate;
        kwargs...,
    )
    Model.SparseLookup._adam_update!(
        model.halt_bias, state.mhalt_bias, state.vhalt_bias,
        gradient.dhalt_bias, runtime.next_step, learning_rate;
        kwargs...,
    )
    return nothing
end

function _barrierless_optimize_lookup_reinject!(runtime, trainer, opt, lr_scale)
    model = trainer.model.lookup
    state = trainer.optimizer.lookup.dense
    gradient = trainer.scheduler.merged_accumulator.lookup
    kwargs = _barrierless_sparse_adam_kwargs(opt)
    Model.SparseLookup._adam_update!(
        model.reinject_logit, state.mreinject, state.vreinject,
        gradient.dreinject_logit, runtime.next_step,
        opt.register_learning_rate * lr_scale; kwargs...,
    )
    return nothing
end

function _barrierless_optimize_dense_parameter!(
    runtime,
    trainer,
    opt,
    lr_scale,
    index::Int,
)
    name = BARRIERLESS_DENSE_PARAMETER_NAMES[index]
    parameters = Model._dense_parameters(trainer.model)
    parameter = getproperty(parameters, name)
    gradient = trainer.scheduler.merged_accumulator.dense[name]
    m, v = trainer.optimizer.dense_states[name]
    group = Model._dense_group(name)
    learning_rate = group === :token ? opt.token_learning_rate * lr_scale :
        group === :register ? opt.register_learning_rate * lr_scale :
        group === :ffn ? opt.ffn_learning_rate * lr_scale :
        opt.attention_learning_rate * lr_scale
    token_offset = Model._token_column_offset(name)
    if token_offset >= 0
        Model._adam_update_selected_token_columns!(
            parameter,
            m,
            v,
            gradient,
            trainer.scheduler.merged_accumulator.active_tokens,
            trainer.optimizer.token_event_count,
            token_offset,
            learning_rate;
            beta1=opt.beta1,
            beta2=opt.beta2,
            epsilon=opt.epsilon,
            weight_decay=opt.dense_weight_decay,
        )
    else
        Model.SparseLookup._adam_update!(
            parameter, m, v, gradient, runtime.next_step, learning_rate;
            beta1=opt.beta1,
            beta2=opt.beta2,
            epsilon=opt.epsilon,
            weight_decay=opt.dense_weight_decay,
        )
    end
    return nothing
end

function _barrierless_dispatch_post_work!(
    runtime::BarrierlessPostRuntime,
    executor::BarrierlessExecutor,
    context,
    work::BarrierlessPostWork,
)
    opcode = BarrierlessPostOpcode(work.opcode)
    index = Int(work.index)
    trainer = context.trainer
    destination = trainer.scheduler.merged_accumulator
    opt = context.hyperparameters.optimizer
    lr_scale = context.expected_update > opt.episodic_decay_after_update ?
        opt.episodic_decay_factor : 1.0f0
    if opcode == BARRIERLESS_POST_REDUCE_BANK
        _barrierless_reduce_bank!(runtime, executor, destination, index)
    elseif opcode == BARRIERLESS_POST_REDUCE_LOOKUP
        _barrierless_reduce_lookup!(executor, destination, index)
    elseif opcode == BARRIERLESS_POST_REDUCE_DENSE
        _barrierless_reduce_dense!(executor, destination, index)
    elseif opcode == BARRIERLESS_POST_REDUCE_ACTIVE_TOKENS
        _barrierless_reduce_active_tokens!(executor, destination)
    elseif opcode == BARRIERLESS_POST_SCALE_BANK
        _barrierless_scale_bank!(runtime, executor, destination, index)
    elseif opcode == BARRIERLESS_POST_SCALE_LOOKUP
        _barrierless_scale_lookup!(runtime, destination, index)
    elseif opcode == BARRIERLESS_POST_SCALE_DENSE
        _barrierless_scale_dense!(runtime, destination, index)
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_BANK
        _barrierless_optimize_bank!(runtime, trainer, index)
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_LOOKUP_ROUTER
        _barrierless_optimize_lookup_router!(runtime, trainer, opt)
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_LOOKUP_HEAD
        _barrierless_optimize_lookup_head!(runtime, trainer, opt, lr_scale)
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_LOOKUP_HALT
        _barrierless_optimize_lookup_halt!(
            runtime,
            trainer,
            opt,
            halting_learning_rate(
                context.expected_update, context.hyperparameters,
            ),
        )
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_LOOKUP_REINJECT
        _barrierless_optimize_lookup_reinject!(runtime, trainer, opt, lr_scale)
    elseif opcode == BARRIERLESS_POST_OPTIMIZE_DENSE
        _barrierless_optimize_dense_parameter!(
            runtime, trainer, opt, lr_scale, index,
        )
    else
        error("unknown barrierless post-phase opcode $(work.opcode)")
    end
    return nothing
end

"""
Dispatch hook for `_barrierless_dispatch_job!`.

It returns the unchanged continuation count because post jobs never enqueue a
candidate continuation.  The parent dispatcher must call this before casting
the raw job kind to `BarrierlessJobKind`.
"""
function barrierless_dispatch_post_job!(
    executor::BarrierlessExecutor,
    worker,
    worker_slot::Int,
    context,
    job::BarrierlessJob,
    continuation_count::Int,
)
    barrierless_is_post_job(job.kind) || error("not a barrierless post job")
    runtime = _BARRIERLESS_POST_RUNTIMES[executor]
    target = Int(job.target)
    1 <= target <= runtime.work_count || error("invalid post work target $target")
    _barrierless_dispatch_post_work!(
        runtime, executor, context, @inbounds(runtime.works[target]),
    )
    previous = Base.Threads.atomic_add!(runtime.remaining, -1)
    previous >= 1 || error("barrierless post-phase counter underflow")
    previous == 1 && Queue.wake_consumers!(executor.queue)
    return continuation_count
end

@inline function _barrierless_push_post_work!(runtime, opcode, index::Int=0)
    next = runtime.work_count + 1
    next <= length(runtime.works) || error("barrierless post work overflow")
    runtime.work_count = next
    runtime.works[next] = BarrierlessPostWork(UInt8(opcode), UInt16(index))
    return next
end

function _barrierless_drain_post_jobs!(executor, runtime)
    worker = executor.workers[1]
    while runtime.remaining[] != 0 || executor.active_dispatches[] != 0
        _barrierless_throw_failure(executor)
        count = Queue.try_dequeue_batch!(
            executor.queue, worker.dequeue_buffer, 1,
        )
        if count > 0
            _barrierless_process_batch!(executor, worker, 1, count)
            continue
        end
        expected = Queue.item_epoch(executor.queue)
        count = Queue.try_dequeue_batch!(
            executor.queue, worker.dequeue_buffer, 1,
        )
        if count > 0
            _barrierless_process_batch!(executor, worker, 1, count)
            continue
        end
        runtime.remaining[] == 0 && executor.active_dispatches[] == 0 && break
        Queue.wait_for_item_change!(executor.queue, expected; timeout_ms=100)
    end
    _barrierless_throw_failure(executor)
    runtime.remaining[] == 0 || error("post-phase drain ended early")
    executor.active_dispatches[] == 0 || error("post-phase dispatch still active")
    Queue.approx_length(executor.queue) == 0 || error("post-phase queue not empty")
    return nothing
end

function _barrierless_run_post_work!(executor, runtime)
    count = runtime.work_count
    count > 0 || return nothing
    runtime.remaining[] == 0 || error("post-phase work overlap")
    generation = executor.generation[]
    @inbounds for index in 1:count
        runtime.jobs[index] = BarrierlessJob(
            BARRIERLESS_POST_JOB_KIND, UInt16(index), generation,
        )
    end
    runtime.remaining[] = count
    _barrierless_enqueue_batch!(executor, runtime.jobs, count)
    _barrierless_drain_post_jobs!(executor, runtime)
    runtime.work_count = 0
    return nothing
end

function _barrierless_enqueue_reduction_work!(runtime)
    runtime.work_count = 0
    for block in 1:Model.SparseLookup.BLOCKS
        for lane in 1:length(runtime.worker_row_scratch) ÷ Model.SparseLookup.BLOCKS
            _barrierless_push_post_work!(
                runtime,
                BARRIERLESS_POST_REDUCE_BANK,
                (block - 1) *
                    (length(runtime.worker_row_scratch) ÷ Model.SparseLookup.BLOCKS) +
                    lane,
            )
        end
    end
    for index in 1:BARRIERLESS_LOOKUP_GRADIENT_COUNT
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_REDUCE_LOOKUP, index,
        )
    end
    for index in eachindex(BARRIERLESS_DENSE_PARAMETER_NAMES)
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_REDUCE_DENSE, index,
        )
    end
    _barrierless_push_post_work!(
        runtime, BARRIERLESS_POST_REDUCE_ACTIVE_TOKENS,
    )
    return nothing
end

function _barrierless_enqueue_scale_work!(runtime)
    runtime.work_count = 0
    for block in 1:Model.SparseLookup.BLOCKS
        for lane in 1:length(runtime.worker_row_scratch) ÷ Model.SparseLookup.BLOCKS
            _barrierless_push_post_work!(
                runtime,
                BARRIERLESS_POST_SCALE_BANK,
                (block - 1) *
                    (length(runtime.worker_row_scratch) ÷ Model.SparseLookup.BLOCKS) +
                    lane,
            )
        end
    end
    for index in 1:BARRIERLESS_LOOKUP_GRADIENT_COUNT
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_SCALE_LOOKUP, index,
        )
    end
    for index in eachindex(BARRIERLESS_DENSE_PARAMETER_NAMES)
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_SCALE_DENSE, index,
        )
    end
    return nothing
end

function _barrierless_enqueue_optimizer_work!(runtime)
    runtime.work_count = 0
    for block in 1:Model.SparseLookup.BLOCKS
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_OPTIMIZE_BANK, block,
        )
    end
    _barrierless_push_post_work!(
        runtime, BARRIERLESS_POST_OPTIMIZE_LOOKUP_ROUTER,
    )
    _barrierless_push_post_work!(
        runtime, BARRIERLESS_POST_OPTIMIZE_LOOKUP_HEAD,
    )
    _barrierless_push_post_work!(
        runtime, BARRIERLESS_POST_OPTIMIZE_LOOKUP_HALT,
    )
    _barrierless_push_post_work!(
        runtime, BARRIERLESS_POST_OPTIMIZE_LOOKUP_REINJECT,
    )
    for index in eachindex(BARRIERLESS_DENSE_PARAMETER_NAMES)
        _barrierless_push_post_work!(
            runtime, BARRIERLESS_POST_OPTIMIZE_DENSE, index,
        )
    end
    return nothing
end

"""
Deterministically reduce worker-local gradients, apply the configured state-batch
mean and one global clip scale, and update disjoint parameter/state groups in
parallel.  Optimizer clocks are committed only after every parameter job ends.
"""
function barrierless_reduce_and_optimizer!(
    executor::BarrierlessExecutor,
    trainer,
    hyperparameters,
    expected_update::Int,
)
    Base.Threads.threadid() == 1 || error(
        "barrierless post phase must be coordinated by worker slot 1",
    )
    executor.update_inflight[] == 1 || error("no barrierless update is active")
    executor.gradient_ready_states[] == TRAINING_STATE_BATCH || error(
        "post phase requires all state gradients",
    )
    executor.active_dispatches[] == 0 || error(
        "post phase overlaps candidate dispatch",
    )
    Queue.approx_length(executor.queue) == 0 || error(
        "post phase requires an empty candidate queue",
    )
    expected_update == trainer.update + 1 || error("non-adjacent optimizer update")
    BARRIERLESS_DENSE_PARAMETER_NAMES ==
        propertynames(Model._dense_parameters(trainer.model)) || error(
            "barrierless dense parameter registry differs from model registry",
        )
    runtime = _barrierless_post_runtime(executor)
    runtime.complete[] == 0 || error(
        "barrierless_begin_postphase_update! was not called before PREPARE",
    )
    destination = trainer.scheduler.merged_accumulator

    reduction_measurement = begin_barrierless_phase_measurement(
        executor, :gradient_reduce,
    )
    try
        _barrierless_prepare_bank_rows!(
            runtime, executor, destination,
        )
        _barrierless_enqueue_reduction_work!(runtime)
        _barrierless_run_post_work!(executor, runtime)
        runtime.scale = inv(Float32(TRAINING_STATE_BATCH))
        _barrierless_enqueue_scale_work!(runtime)
        _barrierless_run_post_work!(executor, runtime)
    finally
        finish_barrierless_phase_measurement!(
            executor, reduction_measurement,
        )
    end

    optimizer_measurement = begin_barrierless_phase_measurement(
        executor, :optimizer,
    )
    telemetry = try
        optimizer = trainer.optimizer
        optimizer.lookup.step == optimizer.dense_step || error(
            "optimizer clocks diverged",
        )
        all(
            state.global_step == optimizer.lookup.step
            for state in optimizer.lookup.bank_states
        ) || error("sparse clocks diverged")
        optimizer.lookup.dense.step == optimizer.lookup.step || error(
            "lookup dense optimizer clock diverged",
        )
        norm = Model.gradient_norm(destination)
        isfinite(norm) || error("gradient norm is non-finite")
        clip_norm = hyperparameters.optimizer.gradient_clip_norm
        clip_scale = norm > clip_norm ?
            Float32(clip_norm / norm) : 1.0f0
        runtime.scale = clip_scale
        _barrierless_enqueue_scale_work!(runtime)
        _barrierless_run_post_work!(executor, runtime)

        runtime.next_step = optimizer.lookup.step + UInt64(1)
        runtime.next_step == UInt64(expected_update) || error(
            "optimizer clock does not match the requested training update",
        )
        @inbounds for token in eachindex(destination.active_tokens)
            destination.active_tokens[token] || continue
            optimizer.token_event_count[token] += UInt64(1)
        end
        _barrierless_enqueue_optimizer_work!(runtime)
        _barrierless_run_post_work!(executor, runtime)

        all(
            state.global_step == runtime.next_step
            for state in optimizer.lookup.bank_states
        ) || error("parallel sparse optimizer clocks diverged")
        optimizer.lookup.dense.step = runtime.next_step
        optimizer.lookup.step = runtime.next_step
        optimizer.dense_step = runtime.next_step
        (
            step=Int(runtime.next_step),
            gradient_norm=norm,
            gradient_scale=Float64(clip_scale),
            active_columns=Tuple(runtime.bank_active_columns),
            active_elements=Tuple(runtime.bank_active_elements),
        )
    finally
        finish_barrierless_phase_measurement!(
            executor, optimizer_measurement,
        )
    end

    runtime.complete[] = UInt32(1)
    Queue.wake_consumers!(executor.queue)
    return telemetry
end

# Integration hooks required in `barrierless_executor.jl`:
#
# 1. `begin_barrierless_update!`, before publishing PREPARE jobs:
#       barrierless_begin_postphase_update!(executor)
#
# 2. `_barrierless_dispatch_job!`, after generation validation and before
#    `BarrierlessJobKind(job.kind)`:
#       if barrierless_is_post_job(job.kind)
#           return barrierless_dispatch_post_job!(
#               executor, worker, worker_slot, context, job, continuation_count,
#           )
#       end
#
# 3. Coordinator and background stop predicates must be distinct.  Slot 1's
#    `_barrierless_drain_update!` still stops at the all-gradient boundary so it
#    can enqueue reduction.  Background `_worker_entry!` stops only when all
#    gradients are ready *and* `barrierless_postphase_complete(executor)` is
#    true.  Do not move background workers into the next-generation wait at the
#    gradient-ready boundary.
#
# 4. Move the `boundary_workers` rendezvous from `_barrierless_drain_update!`
#    to `finish_barrierless_update!`, after the function above marks the post
#    phase complete and wakes consumers.  Keep `update_inflight == 1` until all
#    background workers publish that boundary.
