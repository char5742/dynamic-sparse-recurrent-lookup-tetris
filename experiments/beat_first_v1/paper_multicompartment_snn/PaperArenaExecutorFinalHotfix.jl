# Load immediately after PaperArenaExecutorFinal.jl.
#
# Julia cannot infer a dynamically selected NamedTuple field inside the
# element-wise reducer.  The prototype reducer therefore boxed every scalar
# load.  These Val-specialized helpers keep every field selection compile-time
# constant and make the warmed reduction allocation-free.

@inline _paper_final_worker_gradient_hotfix(
    worker,
    ::Val{name},
) where {name} = getproperty(worker.gradient, name)

@inline _paper_final_worker_field_hotfix(
    worker,
    ::Val{name},
) where {name} = getproperty(worker, name)

function _reduce_paper_final_gradient_array_hotfix!(
    destination,
    workers,
    field::Val,
    worker_count::Int,
)
    @inbounds for index in eachindex(destination)
        value = 0.0f0
        for slot in 1:worker_count
            value += _paper_final_worker_gradient_hotfix(
                workers[slot],
                field,
            )[index]
        end
        destination[index] = value
    end
    return nothing
end

function _reduce_paper_final_utility_array_hotfix!(
    destination,
    workers,
    field::Val,
    worker_count::Int,
    decay::Float32,
    inverse_candidates::Float32,
)
    @inbounds for index in eachindex(destination)
        observed = 0.0f0
        for slot in 1:worker_count
            observed += _paper_final_worker_field_hotfix(
                workers[slot],
                field,
            )[index]
        end
        destination[index] =
            decay * destination[index] +
            (1.0f0 - decay) *
            observed *
            inverse_candidates
    end
    return nothing
end

function _reduce_paper_final_workers_hotfix!(
    executor::PaperExecutorFinal;
    worker_count::Int=executor.active_workers,
)
    trainer = executor.trainer
    aux = register_paper_trainer_aux!(trainer)
    Optim.zero_parameter_tree!(trainer.gradient)
    workers = executor.workers

    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.input_conductance,
        workers,
        Val(:input_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.recurrent_conductance,
        workers,
        Val(:recurrent_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_conductance,
        workers,
        Val(:workspace_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.query_weight,
        workers,
        Val(:query_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_key,
        workers,
        Val(:workspace_key),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_decay_logit,
        workers,
        Val(:workspace_decay_logit),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.head_weight,
        workers,
        Val(:head_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.head_bias,
        workers,
        Val(:head_bias),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.output_weight,
        workers,
        Val(:output_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.output_bias,
        workers,
        Val(:output_bias),
        worker_count,
    )

    inverse_candidates =
        inv(Float32(max(trainer.tape.base.valid_count, 1)))
    _reduce_paper_final_utility_array_hotfix!(
        trainer.input_location_utility,
        workers,
        Val(:input_location_utility),
        worker_count,
        trainer.utility_decay,
        inverse_candidates,
    )
    _reduce_paper_final_utility_array_hotfix!(
        trainer.recurrent_location_utility,
        workers,
        Val(:recurrent_location_utility),
        worker_count,
        trainer.utility_decay,
        inverse_candidates,
    )
    _reduce_paper_final_utility_array_hotfix!(
        aux.workspace_location_utility,
        workers,
        Val(:workspace_location_utility),
        worker_count,
        trainer.utility_decay,
        inverse_candidates,
    )
    return nothing
end

# Redirect the post-replay path without redefining the original reducer.
function _paper_final_post_replay_hotfix!(
    executor::PaperExecutorFinal;
    worker_count::Int=executor.active_workers,
)
    trainer = executor.trainer
    _reduce_paper_final_workers_hotfix!(
        executor;
        worker_count,
    )
    _temporal_workspace_decay_gradient_final!(executor)
    trainer.metrics.gradient_norm = Optim.paper_adam_step!(
        trainer.optimizer,
        trainer.parameters,
        trainer.gradient,
    )
    moves = if isdefined(
        @__MODULE__,
        :_consolidate_one_location_canonical!,
    )
        _consolidate_one_location_canonical!(trainer)
    else
        _consolidate_one_location!(trainer)
    end
    moves += _consolidate_workspace_location!(trainer)
    trainer.metrics.location_moves = moves
    _refresh_paper_final_metrics!(executor)
    return trainer
end

function paper_arena_update_hotfinal!(
    executor::PaperExecutorFinal,
)
    executor.started || error("paper final team is not running")
    trainer = executor.trainer
    _clear_paper_final_workers!(executor)
    wall_started = time_ns()
    cpu_started =
        Point.CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()
    Point.prepare_batch_metadata!(
        trainer.tape.base,
        executor.dataset,
    )
    generation =
        Base.Threads.atomic_add!(
            executor.generation,
            UInt32(1),
        ) + UInt32(1)
    count = trainer.tape.base.valid_count
    _run_paper_final_phase!(
        executor,
        PAPER_FINAL_PACK,
        count,
        generation,
    )
    _run_paper_final_phase!(
        executor,
        PAPER_FINAL_FORWARD,
        count,
        generation,
    )
    trainer.last_loss = Point.loss_and_raw_gradient!(
        trainer.tape.base,
        trainer.loss_scratch,
        1.0f0,
        0.0f0,
    )
    _prepare_route_regularizer_final!(executor)
    _run_paper_final_phase!(
        executor,
        PAPER_FINAL_REPLAY,
        count,
        generation,
    )
    _paper_final_post_replay_hotfix!(executor)
    return _paper_final_finish_metrics!(
        executor,
        wall_started,
        cpu_started,
        gc_started,
    )
end

function paper_arena_update_serial_hotfinal!(
    executor::PaperExecutorFinal,
)
    trainer = executor.trainer
    _clear_paper_final_workers!(executor)
    wall_started = time_ns()
    cpu_started =
        Point.CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()
    Point.prepare_batch_metadata!(
        trainer.tape.base,
        executor.dataset,
    )
    generation =
        Base.Threads.atomic_add!(
            executor.generation,
            UInt32(1),
        ) + UInt32(1)
    count = trainer.tape.base.valid_count
    @inbounds for kind in (
        PAPER_FINAL_PACK,
        PAPER_FINAL_FORWARD,
    )
        for target in 1:count
            _paper_final_dispatch_body!(
                executor,
                1,
                PaperFinalWorkItem(kind, target, generation),
            )
        end
    end
    trainer.last_loss = Point.loss_and_raw_gradient!(
        trainer.tape.base,
        trainer.loss_scratch,
        1.0f0,
        0.0f0,
    )
    _prepare_route_regularizer_final!(executor)
    @inbounds for target in 1:count
        _paper_final_dispatch_body!(
            executor,
            1,
            PaperFinalWorkItem(
                PAPER_FINAL_REPLAY,
                target,
                generation,
            ),
        )
    end
    _paper_final_post_replay_hotfix!(
        executor;
        worker_count=1,
    )
    return _paper_final_finish_metrics!(
        executor,
        wall_started,
        cpu_started,
        gc_started,
    )
end

