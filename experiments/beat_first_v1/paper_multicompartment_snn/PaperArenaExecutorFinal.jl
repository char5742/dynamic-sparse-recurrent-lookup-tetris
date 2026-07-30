# This file is intentionally additive.  Load it into
# `PaperArenaTrainingFinal` after `PaperArenaCanonicalOverrides.jl`.
#
# It replaces the prototype atomic-cursor executor with the same bounded,
# isbits MPMC design used by serial_workspace_snn, while preserving the
# candidate-level two-pass ListNet -> forward replay learning contract.

@enum PaperFinalWorkKind::UInt8 begin
    PAPER_FINAL_IDLE = 0
    PAPER_FINAL_PACK = 1
    PAPER_FINAL_FORWARD = 2
    PAPER_FINAL_REPLAY = 3
end

struct PaperFinalWorkItem
    kind::UInt8
    target::UInt32
    generation::UInt32
end

PaperFinalWorkItem(
    kind::PaperFinalWorkKind,
    target::Integer,
    generation::UInt32,
) = PaperFinalWorkItem(UInt8(kind), UInt32(target), generation)

Base.zero(::Type{PaperFinalWorkItem}) =
    PaperFinalWorkItem(UInt8(PAPER_FINAL_IDLE), UInt32(0), UInt32(0))

isbitstype(PaperFinalWorkItem) ||
    error("paper final work items must remain isbits")

mutable struct PaperFinalWorkerStatistics
    jobs::Matrix{UInt64}
    wall_ns::Matrix{UInt64}
    cpu_ticks::Matrix{UInt64}
end

function PaperFinalWorkerStatistics(workers::Int)
    return PaperFinalWorkerStatistics(
        zeros(UInt64, 3, workers),
        zeros(UInt64, 3, workers),
        zeros(UInt64, 3, workers),
    )
end

function reset_paper_final_statistics!(
    statistics::PaperFinalWorkerStatistics,
)
    fill!(statistics.jobs, 0)
    fill!(statistics.wall_ns, 0)
    fill!(statistics.cpu_ticks, 0)
    return statistics
end

function _paper_final_eligibility(model)
    return ReceptorEligibility(
        zeros(Float32, 3, model.sensory_contacts, model.blocks),
        zeros(Float32, 3, model.sensory_contacts, model.blocks),
        zeros(Float32, 3, model.recurrent_contacts, model.blocks),
        zeros(Float32, 3, model.recurrent_contacts, model.blocks),
        zeros(Float32, 3, model.workspace_contacts, model.blocks),
        zeros(Float32, 3, model.workspace_contacts, model.blocks),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, 11),
    )
end

function _paper_final_compartment_region()
    # `paper_hay_tree` allocates its reduced morphology.  It is called exactly
    # once per executor, never once per contact in the replay hot path.
    tree = Hay.paper_hay_tree()
    result = Vector{UInt8}(undef, length(tree.parent))
    @inbounds for compartment in eachindex(result)
        result[compartment] =
            UInt8(_region_coordinate(tree, compartment))
    end
    return result
end

function _paper_final_location_slot(catalog)
    largest = maximum(Int, catalog)
    result = zeros(Int16, largest)
    @inbounds for slot in eachindex(catalog)
        result[Int(catalog[slot])] = Int16(slot)
    end
    return result
end

mutable struct PaperExecutorFinal{W,E,T,D,Q}
    queue::Q
    workers::W
    eligibilities::E
    trainer::T
    dataset::D
    compartment_region::Vector{UInt8}
    location_slot::Vector{Int16}
    route_load::Vector{Float32}
    workspace_decay_trace::Vector{Float32}
    active_workers::Int
    julia_workers::Int
    cpuset_mode::Symbol
    stochastic_routing::Bool
    routing_seed::UInt64
    generation::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    shutdown::Base.Threads.Atomic{UInt32}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    failure_slot::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    bindings::Vector{Any}
    startup_event::Base.Event
    statistics::PaperFinalWorkerStatistics
    started::Bool
end

function PaperExecutorFinal(
    trainer::PaperTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    stochastic_routing::Bool=true,
    routing_seed::Integer=0x5041504552524f55,
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=512,
)
    julia_workers = Base.Threads.nthreads(:default)
    Base.Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0 for deterministic native worker slots",
    )
    2 <= active_workers <= julia_workers ||
        throw(ArgumentError(
            "active_workers must be in 2:$julia_workers",
        ))
    cpuset_mode in (:none, :all, :p_only) ||
        throw(ArgumentError(
            "cpuset_mode must be :none, :all, or :p_only",
        ))
    ispow2(queue_capacity) ||
        throw(ArgumentError("queue capacity must be a power of two"))
    queue_capacity >= 256 ||
        throw(ArgumentError("queue capacity must be at least 256"))

    register_paper_trainer_aux!(trainer)
    set_paper_ablation!(trainer, paper_ablation(trainer))
    workers = [paper_worker_final(trainer) for _ in 1:active_workers]
    # `falses` returns BitVector and is incompatible with PaperWorker's
    # Vector{Bool} field.  The final factory uses `fill(false, ...)`; keep an
    # executable contract here because a bit-packed selection mask is also
    # undesirable in the routing hot loop.
    all(worker -> worker.selected isa Vector{Bool}, workers) ||
        error("PaperWorker.selected must be a byte-addressable Vector{Bool}")
    eligibilities = [
        _paper_final_eligibility(trainer.model)
        for _ in 1:active_workers
    ]
    queue = Point.Queue.BoundedMPMCQueue{PaperFinalWorkItem}(
        queue_capacity,
        zero(PaperFinalWorkItem),
    )
    return PaperExecutorFinal(
        queue,
        workers,
        eligibilities,
        trainer,
        dataset,
        _paper_final_compartment_region(),
        _paper_final_location_slot(trainer.eligible_compartments),
        zeros(Float32, trainer.model.blocks),
        zeros(Float32, trainer.model.node_dim),
        active_workers,
        julia_workers,
        cpuset_mode,
        stochastic_routing,
        UInt64(routing_seed),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        Base.Event(true),
        PaperFinalWorkerStatistics(active_workers),
        false,
    )
end

@inline function _paper_final_state_credit(
    eligibility::ReceptorEligibility,
    runtime,
    compartment_region::Vector{UInt8},
    block::Int,
    compartment::Int,
)
    coordinate = Int(compartment_region[compartment])
    voltage = _compartment_voltage(runtime, block, compartment)
    nmda = _compartment_nmda(runtime, block, compartment)
    signal =
        eligibility.local_signal[coordinate] +
        0.25f0 * eligibility.local_signal[4 + coordinate] +
        0.20f0 * eligibility.local_signal[9] +
        0.35f0 * eligibility.local_signal[10] +
        0.20f0 * eligibility.local_signal[11]
    return signal *
        cell_surrogate(runtime, block) *
        (
            1.0f0 +
            0.01f0 * abs(voltage) +
            0.05f0 * abs(nmda)
        )
end

@inline function _paper_final_location_evidence!(
    utility,
    catalog,
    location_slot::Vector{Int16},
    compartment_region::Vector{UInt8},
    eligibility::ReceptorEligibility,
    runtime,
    contact::Int,
    block::Int,
    current_location::Int,
    synaptic_eligibility::Float32,
    millisecond::Int,
)
    current_slot = Int(location_slot[current_location])
    current_slot == 0 && return nothing
    current_credit = _paper_final_state_credit(
        eligibility,
        runtime,
        compartment_region,
        block,
        current_location,
    )
    utility[current_slot, contact, block] +=
        abs(current_credit * synaptic_eligibility)
    proposal = mod1(
        millisecond + 7contact + 13block,
        length(catalog),
    )
    proposal_location = Int(catalog[proposal])
    proposal_credit = _paper_final_state_credit(
        eligibility,
        runtime,
        compartment_region,
        block,
        proposal_location,
    )
    utility[proposal, contact, block] +=
        abs(proposal_credit * synaptic_eligibility)
    return nothing
end

"""
    paper_local_replay_candidate_final!(worker, eligibility, trainer, ...)

Forward-direction e-prop replay with worker-owned eligibility storage.  This
contains no global `IdDict` lookup and no morphology construction in the
contact loop.
"""
function paper_local_replay_candidate_final!(
    worker::PaperWorker,
    eligibility::ReceptorEligibility,
    trainer::PaperTrainer,
    compartment_region::Vector{UInt8},
    location_slot::Vector{Int16},
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)
    _reset_eligibility!(eligibility)
    reset_runtime!(worker.runtime)
    fill!(worker.previous_spike, 0.0f0)
    fill!(worker.current_spike, 0.0f0)
    _head_backward!(worker, trainer, flat)

    millisecond = 0
    @inbounds for cycle in 1:model.cycles
        for rank in 1:model.workspace_k
            worker.route_order[rank] =
                arena.route_order[rank, cycle, flat]
        end
        for _ in 1:model.substeps_per_cycle
            millisecond += 1
            copyto!(worker.previous_spike, worker.current_spike)
            fill!(worker.current_spike, 0.0f0)
            for rank in 1:model.workspace_k
                block = Int(worker.route_order[rank])
                reset_cell_drive!(worker.runtime, block)
                _add_input_events!(
                    worker,
                    trainer,
                    block,
                    flat,
                    millisecond,
                )
                _add_recurrent_events!(worker, trainer, block)
                _add_workspace_events!(worker, trainer, aux, block)
                spike = step_cell!(worker.runtime, block)
                worker.current_spike[block] = spike
                _local_prediction_error!(
                    eligibility,
                    worker,
                    trainer,
                    aux,
                    block,
                    flat,
                )

                for contact in 1:model.sensory_contacts
                    rail = Int(model.input_rail[contact, block])
                    event = _sensory_spike(
                        arena.rails[rail, flat],
                        rail,
                        millisecond,
                        flat,
                    ) ? 1.0f0 : 0.0f0
                    location =
                        Int(trainer.input_location[contact, block])
                    contact_eligibility = _contact_eligibility!(
                        eligibility.input_rise,
                        eligibility.input_decay,
                        contact,
                        block,
                        _contact_kind(model, rail),
                        event,
                        _compartment_voltage(
                            worker.runtime,
                            block,
                            location,
                        ),
                    )
                    credit = _paper_final_state_credit(
                        eligibility,
                        worker.runtime,
                        compartment_region,
                        block,
                        location,
                    )
                    worker.gradient.input_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _paper_final_location_evidence!(
                        worker.input_location_utility,
                        trainer.eligible_compartments,
                        location_slot,
                        compartment_region,
                        eligibility,
                        worker.runtime,
                        contact,
                        block,
                        location,
                        contact_eligibility,
                        millisecond,
                    )
                end
                for contact in 1:model.recurrent_contacts
                    source =
                        Int(model.recurrent_source[contact, block])
                    event = worker.previous_spike[source]
                    location =
                        Int(trainer.recurrent_location[contact, block])
                    contact_eligibility = _contact_eligibility!(
                        eligibility.recurrent_rise,
                        eligibility.recurrent_decay,
                        contact,
                        block,
                        _source_block_kind(model, source),
                        event,
                        _compartment_voltage(
                            worker.runtime,
                            block,
                            location,
                        ),
                    )
                    credit = _paper_final_state_credit(
                        eligibility,
                        worker.runtime,
                        compartment_region,
                        block,
                        location,
                    )
                    worker.gradient.recurrent_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _paper_final_location_evidence!(
                        worker.recurrent_location_utility,
                        trainer.eligible_compartments,
                        location_slot,
                        compartment_region,
                        eligibility,
                        worker.runtime,
                        contact,
                        block,
                        location,
                        contact_eligibility,
                        millisecond,
                    )
                end
                for contact in 1:model.workspace_contacts
                    source = Int(worker.route_order[contact])
                    event = worker.previous_spike[source]
                    location =
                        Int(aux.workspace_location[contact, block])
                    contact_eligibility = _contact_eligibility!(
                        eligibility.workspace_rise,
                        eligibility.workspace_decay,
                        contact,
                        block,
                        _source_block_kind(model, source),
                        event,
                        _compartment_voltage(
                            worker.runtime,
                            block,
                            location,
                        ),
                    )
                    credit = _paper_final_state_credit(
                        eligibility,
                        worker.runtime,
                        compartment_region,
                        block,
                        location,
                    )
                    worker.gradient.workspace_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _paper_final_location_evidence!(
                        worker.workspace_location_utility,
                        trainer.eligible_compartments,
                        location_slot,
                        compartment_region,
                        eligibility,
                        worker.runtime,
                        contact,
                        block,
                        location,
                        contact_eligibility,
                        millisecond,
                    )
                end
            end
        end
    end
    _routing_gradient!(worker, trainer, flat)
    return nothing
end

@inline function _paper_final_dispatch_body!(
    executor::PaperExecutorFinal,
    worker_slot::Int,
    work::PaperFinalWorkItem,
)
    work.generation == executor.generation[] ||
        error("stale paper final work generation")
    trainer = executor.trainer
    worker = executor.workers[worker_slot]
    target = Int(work.target)
    1 <= target <= trainer.tape.base.valid_count ||
        error("invalid paper final work target $target")
    flat = Int(trainer.tape.base.valid_flats[target])
    if work.kind == UInt8(PAPER_FINAL_PACK)
        Point.pack_candidate_rails!(
            trainer.tape.base,
            executor.dataset,
            worker.pack,
            flat,
        )
    elseif work.kind == UInt8(PAPER_FINAL_FORWARD)
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
    elseif work.kind == UInt8(PAPER_FINAL_REPLAY)
        paper_local_replay_candidate_final!(
            worker,
            executor.eligibilities[worker_slot],
            trainer,
            executor.compartment_region,
            executor.location_slot,
            flat,
        )
    else
        error("unknown paper final work kind $(work.kind)")
    end
    worker.jobs += UInt64(1)
    return nothing
end

@inline function _paper_final_record_work!(
    executor::PaperExecutorFinal,
    worker_slot::Int,
    kind::UInt8,
    wall_started::UInt64,
    cpu_started::UInt64,
)
    phase = Int(kind)
    statistics = executor.statistics
    statistics.jobs[phase, worker_slot] += UInt64(1)
    statistics.wall_ns[phase, worker_slot] +=
        time_ns() - wall_started
    statistics.cpu_ticks[phase, worker_slot] +=
        Point.CpuSets.thread_cpu_ticks_100ns() - cpu_started
    return nothing
end

@inline function _paper_final_complete!(
    executor::PaperExecutorFinal,
)
    previous =
        Base.Threads.atomic_add!(executor.remaining, -1)
    previous >= 1 ||
        error("paper final remaining-work counter underflow")
    previous == 1 &&
        Point.Queue.wake_consumers!(executor.queue)
    return nothing
end

function _paper_final_dispatch!(
    executor::PaperExecutorFinal,
    worker_slot::Int,
    work::PaperFinalWorkItem,
)
    wall_started = time_ns()
    cpu_started = Point.CpuSets.thread_cpu_ticks_100ns()
    _paper_final_dispatch_body!(executor, worker_slot, work)
    _paper_final_record_work!(
        executor,
        worker_slot,
        work.kind,
        wall_started,
        cpu_started,
    )
    _paper_final_complete!(executor)
    return nothing
end

function _paper_final_mark_failure!(
    executor::PaperExecutorFinal,
    worker_slot::Int,
    exception,
    backtrace,
)
    executor.failures[worker_slot] = (exception, backtrace)
    Base.Threads.atomic_cas!(
        executor.failure_slot,
        0,
        worker_slot,
    )
    Base.Threads.atomic_xchg!(executor.shutdown, UInt32(1))
    Point.Queue.close!(executor.queue)
    notify(executor.startup_event)
    return nothing
end

function _paper_final_throw_failure(
    executor::PaperExecutorFinal,
)
    slot = executor.failure_slot[]
    slot == 0 && return nothing
    payload = executor.failures[slot]
    payload === nothing &&
        error("paper final worker $slot failed without payload")
    exception, backtrace = payload
    throw(Base.CapturedException(exception, backtrace))
end

function _paper_final_worker_entry!(
    executor::PaperExecutorFinal,
    worker_slot::Int,
)
    while executor.shutdown[] == 0
        available, work = Point.Queue.dequeue_wait!(
            executor.queue;
            timeout_ms=100,
        )
        if !available
            Point.Queue.isclosed(executor.queue) &&
                return nothing
            continue
        end
        _paper_final_dispatch!(executor, worker_slot, work)
    end
    return nothing
end

function _paper_final_coordinator_drain!(
    executor::PaperExecutorFinal,
)
    while executor.remaining[] > 0
        _paper_final_throw_failure(executor)
        available, work =
            Point.Queue.try_dequeue!(executor.queue)
        if available
            _paper_final_dispatch!(executor, 1, work)
            continue
        end
        expected = Point.Queue.item_epoch(executor.queue)
        executor.remaining[] == 0 && break
        Point.Queue.wait_for_item_change!(
            executor.queue,
            expected;
            timeout_ms=10,
        )
    end
    _paper_final_throw_failure(executor)
    executor.remaining[] == 0 ||
        error("paper final phase ended before all work completed")
    return nothing
end

function _run_paper_final_phase!(
    executor::PaperExecutorFinal,
    kind::PaperFinalWorkKind,
    count::Int,
    generation::UInt32,
)
    count == 0 && return 0.0
    count <= typemax(UInt32) ||
        throw(ArgumentError("paper final phase is too large"))
    executor.remaining[] = count
    started = time_ns()
    @inbounds for target in 1:count
        Point.Queue.enqueue_wait!(
            executor.queue,
            PaperFinalWorkItem(kind, target, generation);
            timeout_ms=10_000,
        ) || error("paper final queue closed while publishing work")
    end
    _paper_final_coordinator_drain!(executor)
    return (time_ns() - started) * 1.0e-9
end

function _clear_paper_final_workers!(
    executor::PaperExecutorFinal,
)
    @inbounds for worker in executor.workers
        reset_worker_accumulator!(worker)
    end
    reset_paper_final_statistics!(executor.statistics)
    return nothing
end

function _reduce_paper_final_workers!(
    executor::PaperExecutorFinal;
    worker_count::Int=executor.active_workers,
)
    trainer = executor.trainer
    aux = register_paper_trainer_aux!(trainer)
    Optim.zero_parameter_tree!(trainer.gradient)
    # Fixed parameter -> element -> worker-slot order.  Dynamic dispatch may
    # change grouping, but the final sum order itself is deterministic.
    @inbounds for name in keys(trainer.gradient)
        destination = getproperty(trainer.gradient, name)
        for index in eachindex(destination)
            value = 0.0f0
            for slot in 1:worker_count
                value += getproperty(
                    executor.workers[slot].gradient,
                    name,
                )[index]
            end
            destination[index] = value
        end
    end
    inverse_candidates =
        inv(Float32(max(trainer.tape.base.valid_count, 1)))
    @inbounds for (
        destination,
        field,
    ) in (
        (
            trainer.input_location_utility,
            :input_location_utility,
        ),
        (
            trainer.recurrent_location_utility,
            :recurrent_location_utility,
        ),
        (
            aux.workspace_location_utility,
            :workspace_location_utility,
        ),
    )
        for index in eachindex(destination)
            observed = 0.0f0
            for slot in 1:worker_count
                observed += getproperty(
                    executor.workers[slot],
                    field,
                )[index]
            end
            destination[index] =
                trainer.utility_decay * destination[index] +
                (1.0f0 - trainer.utility_decay) *
                observed *
                inverse_candidates
        end
    end
    return nothing
end

"""
Prepare derivatives of explicit entropy-floor and batch-load losses with
respect to routing scores.  The load term includes the softmax centering term;
it is not a block-wise scalar pasted onto every candidate.
"""
function _prepare_route_regularizer_final!(
    executor::PaperExecutorFinal,
)
    trainer = executor.trainer
    arena = trainer.tape.base
    model = trainer.model
    load = executor.route_load
    fill!(arena.route_regularizer_gradient, 0.0f0)
    inverse_candidates =
        inv(Float32(max(arena.valid_count, 1)))
    target_load = inv(Float32(model.blocks))
    inverse_log_blocks =
        inv(log(Float32(max(model.blocks, 2))))
    inverse_temperature =
        inv(Float32(model.route_temperature))

    @inbounds for cycle in 1:model.cycles
        fill!(load, 0.0f0)
        for target in 1:arena.valid_count
            flat = Int(arena.valid_flats[target])
            for block in 1:model.blocks
                load[block] += arena.route_base_probability[
                    block,
                    cycle,
                    flat,
                ]
            end
        end
        for block in 1:model.blocks
            load[block] *= inverse_candidates
        end
        for target in 1:arena.valid_count
            flat = Int(arena.valid_flats[target])
            entropy_raw = 0.0f0
            load_center = 0.0f0
            for block in 1:model.blocks
                probability = max(
                    arena.route_base_probability[
                        block,
                        cycle,
                        flat,
                    ],
                    1.0f-12,
                )
                entropy_raw -= probability * log(probability)
                load_center = muladd(
                    probability,
                    load[block] - target_load,
                    load_center,
                )
            end
            entropy_normalized =
                entropy_raw * inverse_log_blocks
            entropy_gap = max(
                trainer.routing_entropy_floor -
                entropy_normalized,
                0.0f0,
            )
            for block in 1:model.blocks
                probability = max(
                    arena.route_base_probability[
                        block,
                        cycle,
                        flat,
                    ],
                    1.0f-12,
                )
                d_entropy_d_score =
                    -probability *
                    (log(probability) + entropy_raw) *
                    inverse_log_blocks *
                    inverse_temperature
                entropy_gradient =
                    -2.0f0 *
                    trainer.routing_entropy_weight *
                    entropy_gap *
                    d_entropy_d_score *
                    inverse_candidates
                load_gradient =
                    trainer.routing_load_weight *
                    probability *
                    (
                        load[block] -
                        target_load -
                        load_center
                    ) *
                    inverse_candidates *
                    inverse_temperature
                arena.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ] = entropy_gradient + load_gradient
            end
        end
    end
    return nothing
end

"""
Exact forward eligibility recurrence for

    workspace[t+1] = tanh(decay * workspace[t] + write[t]).

For each coordinate:

    e[t+1] = (1 - workspace[t+1]^2) *
             (workspace[t] + decay * e[t]).
"""
function _temporal_workspace_decay_gradient_final!(
    executor::PaperExecutorFinal,
)
    trainer = executor.trainer
    arena = trainer.tape.base
    model = trainer.model
    trace = executor.workspace_decay_trace
    decay = _workspace_decay(trainer.parameters)
    total = 0.0f0
    inverse =
        inv(Float32(max(arena.valid_count * model.cycles, 1)))
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        fill!(trace, 0.0f0)
        candidate_signal = arena.listnet_q_gradient[flat]
        for cycle in 1:model.cycles
            cycle_eligibility = 0.0f0
            for coordinate in 1:model.node_dim
                previous =
                    arena.workspace[coordinate, cycle, flat]
                next =
                    arena.workspace[coordinate, cycle + 1, flat]
                value =
                    (1.0f0 - next * next) *
                    (
                        previous +
                        decay * trace[coordinate]
                    )
                trace[coordinate] = value
                cycle_eligibility += value
            end
            total = muladd(
                candidate_signal,
                cycle_eligibility / Float32(model.node_dim),
                total,
            )
        end
    end
    probability =
        sigmoid(trainer.parameters.workspace_decay_logit[1])
    trainer.gradient.workspace_decay_logit[1] =
        0.94f0 *
        probability *
        (1.0f0 - probability) *
        total *
        inverse
    return nothing
end

function _refresh_paper_final_metrics!(
    executor::PaperExecutorFinal,
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

function _paper_final_post_replay!(
    executor::PaperExecutorFinal;
    worker_count::Int=executor.active_workers,
)
    trainer = executor.trainer
    _reduce_paper_final_workers!(
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

function _paper_final_finish_metrics!(
    executor::PaperExecutorFinal,
    wall_started::UInt64,
    cpu_started::UInt64,
    gc_started,
)
    trainer = executor.trainer
    wall_seconds =
        (time_ns() - wall_started) * 1.0e-9
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

function paper_arena_update!(
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
    _paper_final_post_replay!(executor)
    return _paper_final_finish_metrics!(
        executor,
        wall_started,
        cpu_started,
        gc_started,
    )
end

"""
Single-thread reference update using the exact final kernels and worker-owned
eligibility.  It exists for numerical-equivalence tests, not production.
"""
function paper_arena_update_serial_final!(
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
    _paper_final_post_replay!(executor; worker_count=1)
    return _paper_final_finish_metrics!(
        executor,
        wall_started,
        cpu_started,
        gc_started,
    )
end

function paper_final_phase_snapshot(
    executor::PaperExecutorFinal,
)
    statistics = executor.statistics
    phase_names = (:pack, :forward, :replay)
    return (;
        phases=NamedTuple{phase_names}(
            ntuple(3) do phase
                (;
                    jobs=sum(@view statistics.jobs[phase, :]),
                    wall_seconds=
                        sum(@view statistics.wall_ns[phase, :]) *
                        1.0e-9,
                    cpu_seconds=
                        sum(@view statistics.cpu_ticks[phase, :]) *
                        1.0e-7,
                )
            end,
        ),
        per_worker=[
            (;
                worker=slot,
                jobs=sum(@view statistics.jobs[:, slot]),
                wall_seconds=
                    sum(@view statistics.wall_ns[:, slot]) *
                    1.0e-9,
                cpu_seconds=
                    sum(@view statistics.cpu_ticks[:, slot]) *
                    1.0e-7,
            )
            for slot in 1:executor.active_workers
        ],
        process_cpu_utilization=
            executor.trainer.metrics.cpu_seconds /
            max(
                executor.trainer.metrics.wall_seconds *
                executor.active_workers,
                eps(Float64),
            ),
    )
end

function run_with_paper_team!(
    body::F,
    executor::PaperExecutorFinal,
) where {F}
    executor.started &&
        error("paper final team is already running")
    Point.Queue.isclosed(executor.queue) &&
        error("cannot restart a closed paper final queue")
    topology = Point.CpuSets.discover_topology()
    binding_plan = Point.CpuSets.configure_worker_bindings(
        executor.cpuset_mode,
        executor.active_workers,
        topology,
    )
    executor.ready_workers[] = 0
    executor.booted_workers[] = 0
    executor.failure_slot[] = 0
    executor.shutdown[] = 0
    reset(executor.startup_event)
    executor.started = true
    result = Ref{Any}(nothing)
    try
        Base.Threads.threading_run(worker_slot -> begin
            try
                binding =
                    Point.CpuSets.bind_current_worker!(worker_slot)
                executor.bindings[worker_slot] = binding
                booted = Base.Threads.atomic_add!(
                    executor.booted_workers,
                    1,
                ) + 1
                booted == executor.julia_workers &&
                    notify(executor.startup_event)
                worker_slot <= executor.active_workers ||
                    return nothing
                ready = Base.Threads.atomic_add!(
                    executor.ready_workers,
                    1,
                ) + 1
                ready == executor.active_workers &&
                    notify(executor.startup_event)
                if worker_slot == 1
                    while (
                        executor.booted_workers[] <
                        executor.julia_workers ||
                        executor.ready_workers[] <
                        executor.active_workers
                    )
                        _paper_final_throw_failure(executor)
                        wait(executor.startup_event)
                    end
                    result[] = body(executor)
                    Base.Threads.atomic_xchg!(
                        executor.shutdown,
                        UInt32(1),
                    )
                    Point.Queue.close!(executor.queue)
                else
                    _paper_final_worker_entry!(
                        executor,
                        worker_slot,
                    )
                end
                return nothing
            catch exception
                _paper_final_mark_failure!(
                    executor,
                    min(worker_slot, length(executor.failures)),
                    exception,
                    catch_backtrace(),
                )
                return nothing
            end
        end, true)
    finally
        executor.started = false
    end
    _paper_final_throw_failure(executor)
    return (
        result=result[],
        binding_plan,
        bindings=copy(executor.bindings),
    )
end

run_with_paper_team!(
    executor::PaperExecutorFinal,
    body::F,
) where {F} = run_with_paper_team!(body, executor)

