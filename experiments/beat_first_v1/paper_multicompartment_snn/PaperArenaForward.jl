mutable struct PaperWorker{R,G}
    runtime::R
    gradient::G
    pack::Point.PackScratch
    scores::Vector{Float32}
    selected::Vector{Bool}
    soft_route::Vector{Float32}
    base_route::Vector{Float32}
    route_eligibility::Vector{Float32}
    route_standardized::Vector{Float32}
    route_logweight::Vector{Float32}
    route_alpha::Vector{Float32}
    route_key::Vector{Float32}
    route_order::Vector{Int16}
    trace::Matrix{Float32}
    previous_spike::Vector{Float32}
    current_spike::Vector{Float32}
    features::Vector{Float32}
    dfeatures::Vector{Float32}
    dhidden::Vector{Float32}
    block_signal::Matrix{Float32}
    local_state::Vector{Float32}
    input_location_utility::Array{Float32,3}
    recurrent_location_utility::Array{Float32,3}
    workspace_location_utility::Array{Float32,3}
    jobs::UInt64
end

function PaperWorker(trainer::PaperTrainer)
    model = trainer.model
    aux = register_paper_trainer_aux!(trainer)
    return PaperWorker(
        make_cell_runtime_v2(trainer),
        Optim.zero_parameter_tree(trainer.parameters),
        Point.PackScratch(),
        zeros(Float32, model.blocks),
        falses(model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Int16, model.workspace_k),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, model.hidden),
        zeros(Float32, 11, model.blocks),
        zeros(Float32, 11),
        zeros(Float32, size(trainer.input_location_utility)),
        zeros(Float32, size(trainer.recurrent_location_utility)),
        zeros(Float32, size(aux.workspace_location_utility)),
        UInt64(0),
    )
end

function reset_worker_accumulator!(worker::PaperWorker)
    Optim.zero_parameter_tree!(worker.gradient)
    fill!(worker.input_location_utility, 0.0f0)
    fill!(worker.recurrent_location_utility, 0.0f0)
    fill!(worker.workspace_location_utility, 0.0f0)
    worker.jobs = UInt64(0)
    return worker
end

@inline function _routing_nonce(
    seed::UInt64,
    update::Int,
    flat::Int,
)
    return Routing.routing_mix64(
        seed ⊻
        UInt64(update + 1) * UInt64(0x9e3779b97f4a7c15) ⊻
        UInt64(flat) * UInt64(0xd1b54a32d192ed03),
    )
end

function _prepare_route!(
    worker::PaperWorker,
    model,
    stochastic::Bool,
    nonce::UInt64,
    cycle::Int,
)
    Routing.prepare_policy!(
        worker.route_standardized,
        worker.base_route,
        worker.soft_route,
        worker.scores;
        temperature=model.route_temperature,
    )
    if stochastic
        Routing.sample_plackett_luce_topk!(
            worker.selected,
            worker.route_order,
            worker.route_key,
            worker.soft_route,
            model.workspace_k,
            nonce,
            cycle,
        )
    else
        Routing.deterministic_topk!(
            worker.selected,
            worker.route_order,
            worker.scores,
            model.workspace_k,
        )
    end
    Routing.ordered_score_eligibility!(
        worker.route_eligibility,
        worker.route_logweight,
        worker.route_alpha,
        worker.route_standardized,
        worker.base_route,
        worker.soft_route,
        worker.scores,
        worker.route_order,
        model.workspace_k;
        temperature=model.route_temperature,
    )
    return nothing
end

function _record_route!(
    arena,
    worker::PaperWorker,
    model,
    cycle::Int,
    flat::Int,
)
    entropy = 0.0f0
    score_square_sum = 0.0
    selected_cutoff = Inf32
    unselected_best = -Inf32
    fingerprint = UInt64(0xcbf29ce484222325)
    churn = 0
    @inbounds for block in 1:model.blocks
        selected = worker.selected[block]
        score = worker.scores[block]
        probability = worker.base_route[block]
        arena.block_mask[block, cycle, flat] =
            selected ? 1.0f0 : 0.0f0
        arena.route_policy_probability[block, cycle, flat] =
            worker.soft_route[block]
        arena.route_base_probability[block, cycle, flat] =
            probability
        arena.route_score[block, cycle, flat] = score
        arena.route_eligibility[block, cycle, flat] =
            worker.route_eligibility[block]
        score_square_sum = muladd(
            Float64(score),
            Float64(score),
            score_square_sum,
        )
        entropy -= probability * log(max(probability, 1.0f-12))
        if selected
            selected_cutoff = min(selected_cutoff, score)
            fingerprint = xor(fingerprint, UInt64(block))
            fingerprint *= UInt64(0x100000001b3)
        else
            unselected_best = max(unselected_best, score)
        end
        if cycle > 1
            previous =
                arena.block_mask[block, cycle - 1, flat] != 0.0f0
            churn += selected != previous
        end
    end
    @inbounds for rank in 1:model.workspace_k
        arena.route_order[rank, cycle, flat] =
            worker.route_order[rank]
    end
    arena.route_selection_gap_value[cycle, flat] =
        selected_cutoff - unselected_best
    arena.route_score_square_sum[cycle, flat] = score_square_sum
    arena.route_normalized_entropy[cycle, flat] =
        entropy / log(Float32(model.blocks))
    arena.route_mask_fingerprint[cycle, flat] = fingerprint
    arena.route_cycle_churn_count[cycle, flat] = Int16(churn)
    return nothing
end

@inline function _sensory_spike(
    active::Float32,
    rail::Int,
    millisecond::Int,
    flat::Int,
)
    active == 0.0f0 && return false
    phase = Int(
        (
            UInt32(rail) * UInt32(0x9e37) ⊻
            UInt32(flat) * UInt32(0x85eb)
        ) & UInt32(0x7),
    )
    return ((millisecond + phase) & 0x7) == 0
end

@inline function _add_input_events!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    block::Int,
    flat::Int,
    millisecond::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    @inbounds for contact in 1:model.sensory_contacts
        rail = Int(model.input_rail[contact, block])
        _sensory_spike(
            arena.rails[rail, flat],
            rail,
            millisecond,
            flat,
        ) || continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(trainer.input_location[contact, block]),
            _contact_kind(model, rail),
            trainer.parameters.input_conductance[contact, block],
        )
    end
    return nothing
end

@inline function _add_recurrent_events!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    block::Int,
)
    model = trainer.model
    @inbounds for contact in 1:model.recurrent_contacts
        source = Int(model.recurrent_source[contact, block])
        worker.previous_spike[source] == 0.0f0 && continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(trainer.recurrent_location[contact, block]),
            _source_block_kind(model, source),
            trainer.parameters.recurrent_conductance[contact, block],
        )
    end
    return nothing
end

@inline function _add_workspace_events!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    aux::PaperTrainerAux,
    block::Int,
)
    model = trainer.model
    @inbounds for rank in 1:model.workspace_contacts
        source = Int(worker.route_order[rank])
        worker.previous_spike[source] == 0.0f0 && continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(aux.workspace_location[rank, block]),
            _source_block_kind(model, source),
            trainer.parameters.workspace_conductance[rank, block],
        )
    end
    return nothing
end

function _compute_query!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    parameters = trainer.parameters
    arena = trainer.tape.base
    square_sum = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        value = 0.0f0
        for rail in 1:InputModel.INPUT_RAILS
            value = muladd(
                parameters.query_weight[coordinate, rail],
                arena.rails[rail, flat],
                value,
            )
        end
        arena.query_pre[coordinate, flat] = value
        square_sum = muladd(value, value, square_sum)
        arena.workspace[coordinate, 1, flat] = 0.0f0
    end
    inv_rms = inv(sqrt(
        square_sum / Float32(model.node_dim) +
        InputModel.RMS_NORM_EPS,
    ))
    arena.query_inv_rms[flat] = inv_rms
    @inbounds for coordinate in 1:model.node_dim
        arena.query[coordinate, flat] = tanh(
            InputModel.QUERY_NORM_SCALE *
            arena.query_pre[coordinate, flat] *
            inv_rms,
        )
    end
    return nothing
end

function _score_blocks!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    parameters = trainer.parameters
    arena = trainer.tape.base
    @inbounds for block in 1:model.blocks
        score = 0.0f0
        magnitude = 0.0f0
        for coordinate in 1:model.node_dim
            state = worker.trace[coordinate, block]
            # Query-only probe makes the first route input dependent while the
            # state term remains exclusively soma-spike-derived.
            score = muladd(
                parameters.workspace_key[coordinate, block],
                (state + 0.05f0) * arena.query[coordinate, flat],
                score,
            )
            magnitude += abs(state)
        end
        worker.scores[block] = score + 0.01f0 * magnitude
    end
    return nothing
end

function _write_cycle_state!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    cycle::Int,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    decay = _workspace_decay(trainer.parameters)
    @inbounds for block in 1:model.blocks
        offset = (block - 1) * model.node_dim
        for coordinate in 1:model.node_dim
            arena.membrane[
                offset + coordinate,
                cycle + 1,
                flat,
            ] = worker.trace[coordinate, block]
        end
    end
    @inbounds for coordinate in 1:model.node_dim
        write = 0.0f0
        for rank in 1:model.workspace_k
            block = Int(worker.route_order[rank])
            write += worker.trace[coordinate, block]
        end
        write /= Float32(model.workspace_k)
        arena.workspace[coordinate, cycle + 1, flat] = tanh(
            decay * arena.workspace[coordinate, cycle, flat] +
            write,
        )
    end
    return nothing
end

function _head_forward!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    parameters = trainer.parameters
    workspace_square = 0.0f0
    pool_square = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        pool = 0.0f0
        for rank in 1:model.workspace_k
            block = Int(worker.route_order[rank])
            pool += worker.trace[coordinate, block]
        end
        pool /= Float32(model.workspace_k)
        workspace =
            arena.workspace[coordinate, model.cycles + 1, flat]
        worker.features[coordinate] = workspace
        worker.features[model.node_dim + coordinate] = pool
        workspace_square =
            muladd(workspace, workspace, workspace_square)
        pool_square = muladd(pool, pool, pool_square)
    end
    workspace_inv = inv(sqrt(
        workspace_square / Float32(model.node_dim) +
        InputModel.RMS_NORM_EPS,
    ))
    pool_inv = inv(sqrt(
        pool_square / Float32(model.node_dim) +
        InputModel.RMS_NORM_EPS,
    ))
    arena.workspace_inv_rms[flat] = workspace_inv
    arena.selected_pool_inv_rms[flat] = pool_inv
    @inbounds for coordinate in 1:model.node_dim
        worker.features[coordinate] *= workspace_inv
        worker.features[model.node_dim + coordinate] *= pool_inv
    end
    hidden_square = 0.0f0
    @inbounds for hidden in 1:model.hidden
        value = parameters.head_bias[hidden]
        for feature in 1:(2model.node_dim)
            value = muladd(
                parameters.head_weight[hidden, feature],
                worker.features[feature],
                value,
            )
        end
        arena.hidden_pre[hidden, flat] = value
        hidden_square = muladd(value, value, hidden_square)
    end
    hidden_inv = inv(sqrt(
        hidden_square / Float32(model.hidden) +
        InputModel.RMS_NORM_EPS,
    ))
    arena.hidden_inv_rms[flat] = hidden_inv
    @inbounds for hidden in 1:model.hidden
        arena.hidden[hidden, flat] = tanh(
            InputModel.HIDDEN_NORM_SCALE *
            arena.hidden_pre[hidden, flat] *
            hidden_inv,
        )
    end
    @inbounds for output in 1:OUTPUT_DIM
        value = parameters.output_bias[output]
        for hidden in 1:model.hidden
            value = muladd(
                parameters.output_weight[output, hidden],
                arena.hidden[hidden, flat],
                value,
            )
        end
        arena.raw[output, flat] = value
    end
    return nothing
end

function paper_forward_candidate!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int;
    stochastic_routing::Bool=true,
    routing_nonce::UInt64=UInt64(0),
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)
    reset_runtime!(worker.runtime)
    fill!(worker.trace, 0.0f0)
    fill!(worker.previous_spike, 0.0f0)
    fill!(worker.current_spike, 0.0f0)
    trainer.tape.soma_spike_count[flat] = 0
    trainer.tape.integrated_cell_steps[flat] = 0
    trainer.tape.nmda_current_sum[flat] = 0.0
    trainer.tape.calcium_event_count[flat] = 0
    _compute_query!(worker, trainer, flat)
    fill!(
        @view(arena.membrane[:, 1, flat]),
        0.0f0,
    )

    millisecond = 0
    @inbounds for cycle in 1:model.cycles
        _score_blocks!(worker, trainer, flat)
        _prepare_route!(
            worker,
            model,
            stochastic_routing,
            routing_nonce,
            cycle,
        )
        _record_route!(arena, worker, model, cycle, flat)
        for _ in 1:model.substeps_per_cycle
            millisecond += 1
            copyto!(worker.previous_spike, worker.current_spike)
            fill!(worker.current_spike, 0.0f0)
            for block in 1:model.blocks
                for coordinate in 1:model.node_dim
                    worker.trace[coordinate, block] *=
                        model.trace_decay[coordinate]
                end
            end
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
                trainer.tape.soma_spike_count[flat] += Int32(spike)
                trainer.tape.integrated_cell_steps[flat] += 1
                trainer.tape.nmda_current_sum[flat] +=
                    cell_nmda_sum(worker.runtime, block)
                trainer.tape.calcium_event_count[flat] +=
                    Int32(cell_calcium_event(worker.runtime, block))
                for coordinate in 1:model.node_dim
                    worker.trace[coordinate, block] += spike
                end
            end
        end
        _write_cycle_state!(worker, trainer, cycle, flat)
    end
    _head_forward!(worker, trainer, flat)
    return nothing
end

@inline function _region_coordinate(
    tree::Hay.HayTree,
    compartment::Int,
)
    region = tree.region[compartment]
    region == Hay.BASAL && return 1
    region == Hay.APICAL_TRUNK && return 2
    region == Hay.APICAL_TUFT && return 4
    return 3
end

function _prepare_regional_signal!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    aux::PaperTrainerAux,
    flat::Int,
)
    arena = trainer.tape.base
    @inbounds for block in 1:trainer.model.blocks
        for state in 1:11
            signal = 0.0f0
            for output in 1:OUTPUT_DIM
                signal = muladd(
                    aux.regional_projection[state, output, block],
                    arena.raw_gradient[output, flat],
                    signal,
                )
            end
            worker.block_signal[state, block] =
                trainer.local_signal_scale * signal
        end
    end
    return nothing
end

@inline function _local_factor!(
    worker::PaperWorker,
    block::Int,
)
    cell_local_state!(worker.local_state, worker.runtime, block)
    factor = 0.0f0
    @inbounds for state in 1:11
        scale = state <= 4 || state == 9 || state == 10 ?
            100.0f0 : 1.0f0
        factor = muladd(
            worker.block_signal[state, block],
            tanh(worker.local_state[state] / scale),
            factor,
        )
    end
    return factor / 11.0f0 * cell_surrogate(worker.runtime, block)
end

function _update_location_evidence!(
    utility,
    catalog,
    tree,
    contact::Int,
    block::Int,
    current_location::Int,
    event_value::Float32,
    worker::PaperWorker,
    millisecond::Int,
)
    current_slot = findfirst(==(UInt8(current_location)), catalog)
    current_slot === nothing && return nothing
    coordinate = _region_coordinate(tree, current_location)
    observed = abs(
        worker.block_signal[coordinate, block] *
        event_value *
        cell_surrogate(worker.runtime, block),
    )
    utility[current_slot, contact, block] += observed
    proposal = mod1(
        millisecond + 7contact + 13block,
        length(catalog),
    )
    proposal_coordinate =
        _region_coordinate(tree, Int(catalog[proposal]))
    utility[proposal, contact, block] += abs(
        worker.block_signal[proposal_coordinate, block] *
        event_value *
        cell_surrogate(worker.runtime, block),
    )
    return nothing
end

function paper_local_replay_candidate!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)
    tree = Hay.paper_hay_tree()
    reset_runtime!(worker.runtime)
    fill!(worker.previous_spike, 0.0f0)
    fill!(worker.current_spike, 0.0f0)
    _prepare_regional_signal!(worker, trainer, aux, flat)
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
                local_factor = _local_factor!(worker, block)

                for contact in 1:model.sensory_contacts
                    rail = Int(model.input_rail[contact, block])
                    event = _sensory_spike(
                        arena.rails[rail, flat],
                        rail,
                        millisecond,
                        flat,
                    ) ? 1.0f0 : 0.0f0
                    event == 0.0f0 && continue
                    worker.gradient.input_conductance[
                        contact,
                        block,
                    ] += local_factor * event
                    _update_location_evidence!(
                        worker.input_location_utility,
                        trainer.eligible_compartments,
                        tree,
                        contact,
                        block,
                        Int(trainer.input_location[contact, block]),
                        event,
                        worker,
                        millisecond,
                    )
                end
                for contact in 1:model.recurrent_contacts
                    source = Int(model.recurrent_source[contact, block])
                    event = worker.previous_spike[source]
                    event == 0.0f0 && continue
                    worker.gradient.recurrent_conductance[
                        contact,
                        block,
                    ] += local_factor * event
                    _update_location_evidence!(
                        worker.recurrent_location_utility,
                        trainer.eligible_compartments,
                        tree,
                        contact,
                        block,
                        Int(trainer.recurrent_location[contact, block]),
                        event,
                        worker,
                        millisecond,
                    )
                end
                for contact in 1:model.workspace_contacts
                    source = Int(worker.route_order[contact])
                    event = worker.previous_spike[source]
                    event == 0.0f0 && continue
                    worker.gradient.workspace_conductance[
                        contact,
                        block,
                    ] += local_factor * event
                    _update_location_evidence!(
                        worker.workspace_location_utility,
                        trainer.eligible_compartments,
                        tree,
                        contact,
                        block,
                        Int(aux.workspace_location[contact, block]),
                        event,
                        worker,
                        millisecond,
                    )
                end
            end
        end
    end
    _routing_gradient!(worker, trainer, flat)
    return nothing
end
