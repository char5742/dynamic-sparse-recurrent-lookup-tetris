@enum PaperWorkKind::UInt8 begin
    PAPER_IDLE = 0
    PAPER_PACK = 1
    PAPER_FORWARD = 2
    PAPER_REPLAY = 3
end

mutable struct PaperExecutor{W,T,D}
    workers::W
    trainer::T
    dataset::D
    active_workers::Int
    julia_workers::Int
    stochastic_routing::Bool
    routing_seed::UInt64
    generation::Base.Threads.Atomic{UInt32}
    kind::Base.Threads.Atomic{UInt8}
    cursor::Base.Threads.Atomic{Int}
    count::Base.Threads.Atomic{Int}
    remaining::Base.Threads.Atomic{Int}
    shutdown::Base.Threads.Atomic{UInt32}
    ready::Base.Threads.Atomic{Int}
    failure_slot::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    started::Bool
end

function PaperExecutor(
    trainer::PaperTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    stochastic_routing::Bool=true,
    routing_seed::Integer=0x5041504552524f55,
    cpuset_mode::Symbol=:none,
)
    julia_workers = Base.Threads.nthreads(:default)
    2 <= active_workers <= julia_workers ||
        throw(ArgumentError("active_workers must be in 2:$julia_workers"))
    cpuset_mode === :none || @warn(
        "PaperExecutor atomic barrierless team currently uses OS placement; " *
        "cpuset_mode=$cpuset_mode is ignored",
    )
    register_paper_trainer_aux!(trainer)
    set_paper_ablation!(trainer, paper_ablation(trainer))
    return PaperExecutor(
        [PaperWorker(trainer) for _ in 1:active_workers],
        trainer,
        dataset,
        active_workers,
        julia_workers,
        stochastic_routing,
        UInt64(routing_seed),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{UInt8}(UInt8(PAPER_IDLE)),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        false,
    )
end

@inline function _paper_dispatch!(
    executor::PaperExecutor,
    slot::Int,
    kind::UInt8,
    target::Int,
)
    trainer = executor.trainer
    worker = executor.workers[slot]
    flat = Int(trainer.tape.base.valid_flats[target])
    if kind == UInt8(PAPER_PACK)
        Point.pack_candidate_rails!(
            trainer.tape.base,
            executor.dataset,
            worker.pack,
            flat,
        )
    elseif kind == UInt8(PAPER_FORWARD)
        nonce = executor.stochastic_routing ?
            _routing_nonce(
                executor.routing_seed,
                trainer.optimizer.step,
                flat,
            ) : UInt64(0)
        paper_forward_candidate!(
            worker,
            trainer,
            flat;
            stochastic_routing=executor.stochastic_routing,
            routing_nonce=nonce,
        )
    elseif kind == UInt8(PAPER_REPLAY)
        paper_local_replay_candidate!(worker, trainer, flat)
    else
        error("unknown paper work kind $kind")
    end
    worker.jobs += UInt64(1)
    return nothing
end

function _record_paper_failure!(
    executor::PaperExecutor,
    slot::Int,
    exception,
    backtrace,
)
    executor.failures[slot] = (exception, backtrace)
    Base.Threads.atomic_cas!(executor.failure_slot, 0, slot)
    Base.Threads.atomic_xchg!(executor.shutdown, UInt32(1))
    return nothing
end

function _throw_paper_failure(executor::PaperExecutor)
    slot = executor.failure_slot[]
    slot == 0 && return nothing
    payload = executor.failures[slot]
    payload === nothing &&
        error("paper worker $slot failed without payload")
    exception, backtrace = payload
    throw(Base.CapturedException(exception, backtrace))
end

function _paper_worker_generation!(
    executor::PaperExecutor,
    slot::Int,
)
    while true
        target =
            Base.Threads.atomic_add!(executor.cursor, 1) + 1
        target > executor.count[] && break
        _paper_dispatch!(
            executor,
            slot,
            executor.kind[],
            target,
        )
        previous =
            Base.Threads.atomic_add!(executor.remaining, -1)
        previous >= 1 ||
            error("paper phase counter underflow")
    end
    return nothing
end

function _paper_worker_loop!(
    executor::PaperExecutor,
    slot::Int,
)
    seen = UInt32(0)
    Base.Threads.atomic_add!(executor.ready, 1)
    while executor.shutdown[] == 0
        generation = executor.generation[]
        if generation == seen
            Base.yield()
            continue
        end
        seen = generation
        try
            _paper_worker_generation!(executor, slot)
        catch exception
            _record_paper_failure!(
                executor,
                slot,
                exception,
                catch_backtrace(),
            )
            return nothing
        end
    end
    return nothing
end

function _run_paper_phase!(
    executor::PaperExecutor,
    kind::PaperWorkKind,
    count::Int,
)
    count == 0 && return 0.0
    _throw_paper_failure(executor)
    started = time_ns()
    executor.kind[] = UInt8(kind)
    executor.cursor[] = 0
    executor.count[] = count
    executor.remaining[] = count
    Base.Threads.atomic_add!(
        executor.generation,
        UInt32(1),
    )
    _paper_worker_generation!(executor, 1)
    while executor.remaining[] != 0
        _throw_paper_failure(executor)
        Base.yield()
    end
    return (time_ns() - started) * 1.0e-9
end

function _clear_paper_workers!(executor::PaperExecutor)
    @inbounds for worker in executor.workers
        reset_worker_accumulator!(worker)
    end
    return nothing
end

function _reduce_location_utility!(
    destination,
    workers,
    field::Symbol,
    decay::Float32,
    inverse_candidates::Float32,
)
    @inbounds for index in eachindex(destination)
        observed = 0.0f0
        for worker in workers
            observed += getproperty(worker, field)[index]
        end
        destination[index] =
            decay * destination[index] +
            (1.0f0 - decay) *
            observed *
            inverse_candidates
    end
    return nothing
end

function _reduce_paper_workers!(executor::PaperExecutor)
    trainer = executor.trainer
    aux = register_paper_trainer_aux!(trainer)
    _reduce_worker_gradients!(trainer, executor.workers)
    inverse_candidates =
        inv(Float32(max(trainer.tape.base.valid_count, 1)))
    _reduce_location_utility!(
        trainer.input_location_utility,
        executor.workers,
        :input_location_utility,
        trainer.utility_decay,
        inverse_candidates,
    )
    _reduce_location_utility!(
        trainer.recurrent_location_utility,
        executor.workers,
        :recurrent_location_utility,
        trainer.utility_decay,
        inverse_candidates,
    )
    _reduce_location_utility!(
        aux.workspace_location_utility,
        executor.workers,
        :workspace_location_utility,
        trainer.utility_decay,
        inverse_candidates,
    )
    # Workspace-decay learning remains local to the public spike-trace plane.
    signal = 0.0f0
    arena = trainer.tape.base
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        signal += arena.listnet_q_gradient[flat]
    end
    probability = sigmoid(trainer.parameters.workspace_decay_logit[1])
    trainer.gradient.workspace_decay_logit[1] =
        0.94f0 * probability * (1.0f0 - probability) *
        signal *
        inverse_candidates
    return nothing
end

function _capacity_used(
    trainer::PaperTrainer,
    aux::PaperTrainerAux,
    block::Int,
    location::UInt8,
    kind::UInt8,
)
    used = 0
    model = trainer.model
    @inbounds for contact in 1:model.sensory_contacts
        rail = Int(model.input_rail[contact, block])
        _contact_kind(model, rail) == kind || continue
        trainer.input_location[contact, block] == location &&
            (used += 1)
    end
    @inbounds for contact in 1:model.recurrent_contacts
        source = Int(model.recurrent_source[contact, block])
        _source_block_kind(model, source) == kind || continue
        trainer.recurrent_location[contact, block] == location &&
            (used += 1)
    end
    @inbounds for contact in 1:model.workspace_contacts
        # Workspace slot source changes with routing. Reserve against the
        # stricter of the E/I capacities.
        aux.workspace_location[contact, block] == location &&
            (used += 1)
    end
    return used
end

function _consolidate_workspace_location!(
    trainer::PaperTrainer,
)
    trainer.optimizer.step % trainer.location_interval == 0 ||
        return 0
    aux = register_paper_trainer_aux!(trainer)
    model = trainer.model
    block = mod1(
        trainer.optimizer.step ÷ trainer.location_interval,
        model.blocks,
    )
    contact = mod1(
        trainer.optimizer.step ÷
        (trainer.location_interval * model.blocks) + 1,
        model.workspace_contacts,
    )
    current = aux.workspace_location[contact, block]
    current_slot = findfirst(==(current), aux.location_catalog)
    current_slot === nothing && return 0
    best = current_slot
    best_value =
        aux.workspace_location_utility[current_slot, contact, block]
    @inbounds for slot in eachindex(aux.location_catalog)
        location = aux.location_catalog[slot]
        capacity = min(
            aux.excitatory_capacity[Int(location)],
            aux.inhibitory_capacity[Int(location)],
        )
        _capacity_used(
            trainer,
            aux,
            block,
            location,
            Model.EXCITATORY,
        ) < capacity || continue
        value =
            aux.workspace_location_utility[slot, contact, block] -
            trainer.utility_connection_cost
        if value > best_value + 1.0f-4
            best = slot
            best_value = value
        end
    end
    best == current_slot && return 0
    aux.workspace_location[contact, block] =
        aux.location_catalog[best]
    trainer.optimizer.first_moment.workspace_conductance[
        contact,
        block,
    ] = 0.0f0
    trainer.optimizer.second_moment.workspace_conductance[
        contact,
        block,
    ] = 0.0f0
    return 1
end

function _refresh_paper_metrics!(
    executor::PaperExecutor,
)
    trainer = executor.trainer
    arena = trainer.tape.base
    spike = 0
    integrated = 0
    nmda = 0.0
    calcium = 0
    entropy = 0.0
    decisions = 0
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        spike += trainer.tape.soma_spike_count[flat]
        integrated += trainer.tape.integrated_cell_steps[flat]
        nmda += trainer.tape.nmda_current_sum[flat]
        calcium += trainer.tape.calcium_event_count[flat]
        for cycle in 1:trainer.model.cycles
            entropy +=
                arena.route_normalized_entropy[cycle, flat]
            decisions += 1
        end
    end
    trainer.metrics.firing_rate =
        spike / max(integrated, 1)
    trainer.metrics.nmda_current_mean =
        nmda / max(integrated, 1)
    trainer.metrics.calcium_event_rate =
        calcium / max(integrated, 1)
    trainer.metrics.routing_entropy =
        entropy / max(decisions, 1)
    trainer.metrics.detailed_cells_integrated = integrated
    return nothing
end

function paper_arena_update!(executor::PaperExecutor)
    executor.started || error("paper team is not running")
    trainer = executor.trainer
    _clear_paper_workers!(executor)
    wall_started = time_ns()
    cpu_started = Point.CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()

    Point.prepare_batch_metadata!(trainer.tape.base, executor.dataset)
    _run_paper_phase!(
        executor,
        PAPER_PACK,
        trainer.tape.base.valid_count,
    )
    _run_paper_phase!(
        executor,
        PAPER_FORWARD,
        trainer.tape.base.valid_count,
    )
    trainer.last_loss = Point.loss_and_raw_gradient!(
        trainer.tape.base,
        trainer.loss_scratch,
        1.0f0,
        0.0f0,
    )
    fill!(trainer.tape.base.route_regularizer_gradient, 0.0f0)
    _run_paper_phase!(
        executor,
        PAPER_REPLAY,
        trainer.tape.base.valid_count,
    )
    _reduce_paper_workers!(executor)
    trainer.metrics.gradient_norm = Optim.paper_adam_step!(
        trainer.optimizer,
        trainer.parameters,
        trainer.gradient,
    )
    moves = _consolidate_one_location!(trainer)
    moves += _consolidate_workspace_location!(trainer)
    trainer.metrics.location_moves = moves
    _refresh_paper_metrics!(executor)

    wall_seconds = (time_ns() - wall_started) * 1.0e-9
    cpu_seconds =
        (
            Point.CpuSets.process_cpu_ticks_100ns() -
            cpu_started
        ) * 1.0e-7
    gc_difference = Base.GC_Diff(Base.gc_num(), gc_started)
    trainer.metrics.wall_seconds = wall_seconds
    trainer.metrics.cpu_seconds = cpu_seconds
    trainer.metrics.allocation_bytes =
        Int128(gc_difference.allocd)
    trainer.metrics.gc_seconds =
        Float64(gc_difference.total_time) * 1.0e-9
    trainer.metrics.states_per_second =
        trainer.tape.base.state_batch /
        max(wall_seconds, eps(Float64))
    paper_internal_max_delta(trainer) == 0.0f0 ||
        error("frozen internal cell parameters changed")
    isfinite(trainer.last_loss.composite_loss) ||
        error("non-finite paper training loss")
    return trainer
end

function run_with_paper_team!(
    body::F,
    executor::PaperExecutor,
) where {F}
    executor.started && error("paper team already running")
    executor.shutdown[] = 0
    executor.ready[] = 0
    executor.failure_slot[] = 0
    executor.started = true
    result = Ref{Any}(nothing)
    failure = nothing
    try
        Base.Threads.threading_run(slot -> begin
            if slot == 1
                Base.Threads.atomic_add!(executor.ready, 1)
                while executor.ready[] < executor.julia_workers
                    Base.yield()
                end
                try
                    result[] = body(executor)
                catch exception
                    _record_paper_failure!(
                        executor,
                        slot,
                        exception,
                        catch_backtrace(),
                    )
                finally
                    Base.Threads.atomic_xchg!(
                        executor.shutdown,
                        UInt32(1),
                    )
                end
            elseif slot <= executor.active_workers
                _paper_worker_loop!(executor, slot)
            else
                Base.Threads.atomic_add!(executor.ready, 1)
                while executor.shutdown[] == 0
                    Base.yield()
                end
            end
        end)
        _throw_paper_failure(executor)
    catch exception
        failure = Base.CapturedException(
            exception,
            catch_backtrace(),
        )
    finally
        executor.started = false
    end
    failure === nothing || throw(failure)
    return result[]
end
