function _paper_worker_canonical(trainer::PaperTrainer)
    worker = paper_worker_final(trainer)
    # IdDict is fully populated before the persistent worker team starts.
    # Every replay subsequently performs read-only lookups; no lazy mutation
    # occurs under parallel execution.
    _eligibility_state(worker, trainer)
    return worker
end

PaperWorker(trainer::PaperTrainer) =
    _paper_worker_canonical(trainer)

function _prepare_route_regularizer_canonical!(
    trainer::PaperTrainer,
)
    arena = trainer.tape.base
    model = trainer.model
    fill!(arena.route_regularizer_gradient, 0.0f0)
    inverse_candidates =
        inv(Float32(max(arena.valid_count, 1)))
    target_load = inv(Float32(model.blocks))
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            load = 0.0f0
            for target in 1:arena.valid_count
                flat = Int(arena.valid_flats[target])
                load += arena.route_base_probability[
                    block,
                    cycle,
                    flat,
                ]
            end
            load *= inverse_candidates
            for target in 1:arena.valid_count
                flat = Int(arena.valid_flats[target])
                probability = max(
                    arena.route_base_probability[
                        block,
                        cycle,
                        flat,
                    ],
                    1.0f-8,
                )
                entropy = arena.route_normalized_entropy[cycle, flat]
                entropy_scale = entropy <
                    trainer.routing_entropy_floor ? 1.0f0 : 0.25f0
                entropy_gradient =
                    trainer.routing_entropy_weight *
                    entropy_scale *
                    (log(probability) + 1.0f0)
                load_gradient =
                    trainer.routing_load_weight *
                    (load - target_load)
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

function _temporal_workspace_decay_gradient!(
    trainer::PaperTrainer,
)
    arena = trainer.tape.base
    model = trainer.model
    total = 0.0f0
    inverse =
        inv(Float32(max(arena.valid_count * model.cycles, 1)))
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        candidate_signal = arena.listnet_q_gradient[flat]
        for cycle in 1:model.cycles
            eligibility = 0.0f0
            for coordinate in 1:model.node_dim
                previous =
                    arena.workspace[coordinate, cycle, flat]
                next =
                    arena.workspace[coordinate, cycle + 1, flat]
                eligibility = muladd(
                    previous,
                    1.0f0 - next * next,
                    eligibility,
                )
            end
            total = muladd(
                candidate_signal,
                eligibility / Float32(model.node_dim),
                total,
            )
        end
    end
    probability = sigmoid(
        trainer.parameters.workspace_decay_logit[1],
    )
    trainer.gradient.workspace_decay_logit[1] =
        0.94f0 *
        probability *
        (1.0f0 - probability) *
        total *
        inverse
    return nothing
end

function _location_capacity_available(
    trainer::PaperTrainer,
    aux::PaperTrainerAux,
    block::Int,
    location::UInt8,
    kind::UInt8,
)
    capacity = kind == Model.EXCITATORY ?
        aux.excitatory_capacity[Int(location)] :
        aux.inhibitory_capacity[Int(location)]
    return _capacity_used(
        trainer,
        aux,
        block,
        location,
        kind,
    ) < capacity
end

function _consolidate_one_location_canonical!(
    trainer::PaperTrainer,
)
    trainer.optimizer.step % trainer.location_interval == 0 ||
        return 0
    model = trainer.model
    aux = register_paper_trainer_aux!(trainer)
    trainer.location_cursor =
        mod(trainer.location_cursor, 2model.blocks) + 1
    input_contact = trainer.location_cursor <= model.blocks
    block = input_contact ?
        trainer.location_cursor :
        trainer.location_cursor - model.blocks
    contact = input_contact ?
        mod1(
            trainer.optimizer.step ÷ trainer.location_interval,
            model.sensory_contacts,
        ) :
        mod1(
            trainer.optimizer.step ÷ trainer.location_interval,
            model.recurrent_contacts,
        )
    utilities = input_contact ?
        trainer.input_location_utility :
        trainer.recurrent_location_utility
    location = input_contact ?
        trainer.input_location[contact, block] :
        trainer.recurrent_location[contact, block]
    kind = if input_contact
        rail = Int(model.input_rail[contact, block])
        _contact_kind(model, rail)
    else
        source = Int(model.recurrent_source[contact, block])
        _source_block_kind(model, source)
    end
    current_slot = findfirst(
        ==(location),
        trainer.eligible_compartments,
    )
    current_slot === nothing && return 0
    best = current_slot
    best_value = utilities[current_slot, contact, block]
    @inbounds for slot in eachindex(trainer.eligible_compartments)
        candidate = trainer.eligible_compartments[slot]
        candidate == location ||
            _location_capacity_available(
                trainer,
                aux,
                block,
                candidate,
                kind,
            ) || continue
        value =
            utilities[slot, contact, block] -
            trainer.utility_connection_cost
        if value > best_value + 1.0f-4
            best = slot
            best_value = value
        end
    end
    best == current_slot && return 0
    if input_contact
        trainer.input_location[contact, block] =
            trainer.eligible_compartments[best]
        trainer.optimizer.first_moment.input_conductance[
            contact,
            block,
        ] = 0.0f0
        trainer.optimizer.second_moment.input_conductance[
            contact,
            block,
        ] = 0.0f0
    else
        trainer.recurrent_location[contact, block] =
            trainer.eligible_compartments[best]
        trainer.optimizer.first_moment.recurrent_conductance[
            contact,
            block,
        ] = 0.0f0
        trainer.optimizer.second_moment.recurrent_conductance[
            contact,
            block,
        ] = 0.0f0
    end
    return 1
end

function paper_arena_update_canonical!(executor::PaperExecutor)
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
    _prepare_route_regularizer_canonical!(trainer)
    _run_paper_phase!(
        executor,
        PAPER_REPLAY,
        trainer.tape.base.valid_count,
    )
    _reduce_paper_workers!(executor)
    _temporal_workspace_decay_gradient!(trainer)
    trainer.metrics.gradient_norm = Optim.paper_adam_step!(
        trainer.optimizer,
        trainer.parameters,
        trainer.gradient,
    )
    moves = _consolidate_one_location_canonical!(trainer)
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

paper_arena_update!(executor::PaperExecutor) =
    paper_arena_update_canonical!(executor)
