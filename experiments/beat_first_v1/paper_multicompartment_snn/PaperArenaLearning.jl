@inline function _workspace_decay(parameters)
    probability = sigmoid(parameters.workspace_decay_logit[1])
    return muladd(0.94f0, probability, 0.02f0)
end

function _head_backward!(
    worker,
    trainer::PaperTrainer,
    flat::Int,
)
    arena = trainer.tape.base
    model = trainer.model
    parameters = trainer.parameters
    gradient = worker.gradient
    node_dim = model.node_dim
    fill!(worker.dfeatures, 0.0f0)
    fill!(worker.dhidden, 0.0f0)

    @inbounds for coordinate in 1:node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        worker.features[coordinate] =
            arena.workspace[coordinate, model.cycles + 1, flat] *
            arena.workspace_inv_rms[flat]
        worker.features[node_dim + coordinate] =
            selected_pool /
            Float32(model.workspace_k) *
            arena.selected_pool_inv_rms[flat]
    end

    @inbounds for output in 1:OUTPUT_DIM
        cotangent = arena.raw_gradient[output, flat]
        gradient.output_bias[output] += cotangent
        for hidden in 1:model.hidden
            gradient.output_weight[output, hidden] = muladd(
                cotangent,
                arena.hidden[hidden, flat],
                gradient.output_weight[output, hidden],
            )
            worker.dhidden[hidden] = muladd(
                parameters.output_weight[output, hidden],
                cotangent,
                worker.dhidden[hidden],
            )
        end
    end
    projection_mean = 0.0f0
    @inbounds for hidden in 1:model.hidden
        value = arena.hidden[hidden, flat]
        cotangent =
            worker.dhidden[hidden] *
            InputModel.HIDDEN_NORM_SCALE *
            (1.0f0 - value * value)
        worker.dhidden[hidden] = cotangent
        projection_mean = muladd(
            cotangent,
            arena.hidden_pre[hidden, flat],
            projection_mean,
        )
    end
    projection_mean /= Float32(model.hidden)
    inv_rms = arena.hidden_inv_rms[flat]
    inv_rms2 = inv_rms * inv_rms
    @inbounds for hidden in 1:model.hidden
        cotangent = inv_rms * (
            worker.dhidden[hidden] -
            arena.hidden_pre[hidden, flat] *
            inv_rms2 *
            projection_mean
        )
        gradient.head_bias[hidden] += cotangent
        for feature in 1:(2node_dim)
            gradient.head_weight[hidden, feature] = muladd(
                cotangent,
                worker.features[feature],
                gradient.head_weight[hidden, feature],
            )
            worker.dfeatures[feature] = muladd(
                parameters.head_weight[hidden, feature],
                cotangent,
                worker.dfeatures[feature],
            )
        end
    end
    return nothing
end

function _prepare_block_signal!(
    worker,
    trainer::PaperTrainer,
    flat::Int,
)
    arena = trainer.tape.base
    @inbounds for block in 1:trainer.model.blocks
        signal = 0.0f0
        for output in 1:OUTPUT_DIM
            signal = muladd(
                trainer.fixed_projection[block, output],
                arena.raw_gradient[output, flat],
                signal,
            )
        end
        worker.block_signal[block] =
            trainer.local_signal_scale * signal
    end
    return nothing
end

"""
Three-factor Plackett--Luce routing update.

The reward is the candidate-centered supervised ListNet/auxiliary surrogate,
never an environment return. Selection history and ordered eligibility are
read from the first forward pass; replay does not resample routes.
"""
function _routing_gradient!(
    worker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    gradient = worker.gradient
    advantage =
        trainer.global_signal_scale *
        arena.listnet_q_gradient[flat]
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            route_factor =
                advantage *
                arena.route_eligibility[block, cycle, flat]
            route_factor +=
                arena.route_regularizer_gradient[block, cycle, flat]
            route_factor == 0.0f0 && continue
            offset = (block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                state =
                    arena.membrane[offset + coordinate, cycle, flat]
                query = arena.query[coordinate, flat]
                gradient.workspace_key[coordinate, block] =
                    muladd(
                        route_factor * state,
                        query,
                        gradient.workspace_key[coordinate, block],
                    )
                gradient.query_weight[
                    coordinate,
                    model.input_rail[
                        mod1(coordinate, model.sensory_contacts),
                        block,
                    ],
                ] +=
                    0.05f0 * route_factor *
                    parametersafe(
                        trainer.parameters.workspace_key[
                            coordinate,
                            block,
                        ],
                    ) * state
            end
        end
    end
    return nothing
end

@inline parametersafe(value::Float32) =
    isfinite(value) ? value : 0.0f0

function _reduce_worker_gradients!(trainer, workers)
    Optim.zero_parameter_tree!(trainer.gradient)
    @inbounds for name in keys(trainer.gradient)
        destination = getproperty(trainer.gradient, name)
        for index in eachindex(destination)
            value = 0.0f0
            for worker in workers
                value += getproperty(worker.gradient, name)[index]
            end
            destination[index] = value
        end
    end
    return nothing
end

function _location_available(
    trainer::PaperTrainer,
    input_contact::Bool,
    block::Int,
    contact::Int,
    location::UInt8,
    kind::UInt8,
)
    model = trainer.model
    if input_contact
        @inbounds for other in 1:model.sensory_contacts
            other == contact && continue
            rail = Int(model.input_rail[other, block])
            _contact_kind(model, rail) == kind || continue
            trainer.input_location[other, block] == location &&
                return false
        end
    else
        @inbounds for other in 1:model.recurrent_contacts
            other == contact && continue
            source = Int(model.recurrent_source[other, block])
            _source_block_kind(model, source) == kind || continue
            trainer.recurrent_location[other, block] == location &&
                return false
        end
    end
    return true
end

function _consolidate_one_location!(trainer::PaperTrainer)
    trainer.optimizer.step % trainer.location_interval == 0 ||
        return 0
    model = trainer.model
    trainer.location_cursor =
        mod(trainer.location_cursor, 2model.blocks) + 1
    input_contact = trainer.location_cursor <= model.blocks
    block = input_contact ?
        trainer.location_cursor :
        trainer.location_cursor - model.blocks
    contact = input_contact ?
        mod1(trainer.optimizer.step ÷ trainer.location_interval, model.sensory_contacts) :
        mod1(trainer.optimizer.step ÷ trainer.location_interval, model.recurrent_contacts)
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
    current_slot = findfirst(==(location), trainer.eligible_compartments)
    current_slot === nothing && return 0
    best_slot = current_slot
    best_value = utilities[current_slot, contact, block]
    @inbounds for slot in eachindex(trainer.eligible_compartments)
        candidate = trainer.eligible_compartments[slot]
        _location_available(
            trainer,
            input_contact,
            block,
            contact,
            candidate,
            kind,
        ) || continue
        value =
            utilities[slot, contact, block] -
            trainer.utility_connection_cost
        if value > best_value + 1.0f-4
            best_value = value
            best_slot = slot
        end
    end
    best_slot == current_slot && return 0
    if input_contact
        trainer.input_location[contact, block] =
            trainer.eligible_compartments[best_slot]
        fill!(
            @view(
                trainer.optimizer.first_moment.input_conductance[
                    contact:contact,
                    block:block,
                ],
            ),
            0.0f0,
        )
        fill!(
            @view(
                trainer.optimizer.second_moment.input_conductance[
                    contact:contact,
                    block:block,
                ],
            ),
            0.0f0,
        )
    else
        trainer.recurrent_location[contact, block] =
            trainer.eligible_compartments[best_slot]
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
