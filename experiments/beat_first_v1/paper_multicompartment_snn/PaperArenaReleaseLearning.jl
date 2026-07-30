# Release-v2 local learning and official-location utility.

function _local_prediction_error_release!(
    eligibility::ReceptorEligibility,
    worker::PaperWorker{ReleaseCellRuntime},
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
    block::Int,
    flat::Int,
)
    arena = trainer.tape.base
    state_slot = div(flat - 1, arena.width) + 1
    candidate = flat - (state_slot - 1) * arena.width
    targets = arena.targets
    prediction = eligibility.prediction
    error = eligibility.local_error
    fill!(prediction, 0.0f0)
    fill!(error, 0.0f0)
    cell_local_state!(worker.local_state, worker.runtime, block)

    # Release coordinates are already semantically aligned and normalized.
    # Never divide voltage-like coordinates by 100 here.
    @inbounds for output in 1:OUTPUT_DIM
        value = 0.0f0
        for state in 1:11
            value = muladd(
                aux.regional_projection[state, output, block],
                worker.local_state[state],
                value,
            )
        end
        prediction[output] = value
    end
    inverse_valid = inv(Float32(max(arena.valid_count, 1)))
    error[1] =
        0.25f0 *
        clamp(
            prediction[1] -
            targets.teacher_q[candidate, state_slot],
            -1.0f0,
            1.0f0,
        ) *
        inverse_valid
    if targets.death_mask[candidate, state_slot] != 0.0f0
        error[2] =
            0.10f0 *
            (
                sigmoid(prediction[2]) -
                targets.death[candidate, state_slot]
            ) *
            inverse_valid
    end
    teacher_q = targets.teacher_q[candidate, state_slot]
    @inbounds for quantile in 1:16
        output = 2 + quantile
        residual = teacher_q - prediction[output]
        tau = (Float32(quantile) - 0.5f0) / 16.0f0
        weight = abs(
            tau -
            (residual < 0.0f0 ? 1.0f0 : 0.0f0),
        )
        error[output] =
            -0.05f0 *
            weight *
            clamp(residual, -1.0f0, 1.0f0) *
            inverse_valid /
            16.0f0
    end
    geometry = (
        targets.line_clear[candidate, state_slot] / 4.0f0,
        targets.max_height[candidate, state_slot] / 24.0f0,
        targets.holes[candidate, state_slot] / 240.0f0,
        targets.cavities[candidate, state_slot] / 240.0f0,
    )
    @inbounds for local_index in 1:4
        output = 18 + local_index
        error[output] =
            0.10f0 *
            clamp(
                prediction[output] - geometry[local_index],
                -1.0f0,
                1.0f0,
            ) *
            inverse_valid
    end

    @inbounds for state in 1:11
        signal = 0.0f0
        for output in 1:OUTPUT_DIM
            signal = muladd(
                aux.regional_projection[state, output, block],
                trainer.global_signal_scale *
                arena.raw_gradient[output, flat] +
                trainer.local_signal_scale * error[output],
                signal,
            )
        end
        coordinate = worker.local_state[state]
        derivative = state == 11 ?
            coordinate * (1.0f0 - coordinate) :
            max(0.0f0, 1.0f0 - coordinate * coordinate)
        eligibility.local_signal[state] = signal * derivative
        worker.block_signal[state, block] =
            eligibility.local_signal[state]
    end
    return nothing
end

@inline function _state_credit(
    eligibility::ReceptorEligibility,
    runtime::ReleaseCellRuntime,
    block::Int,
    location::Int,
)
    projection =
        runtime.trusted.parameters.compartment_projection
    signal = 0.0f0
    normalization = 0.0f0
    @inbounds for branch in 1:4
        placement = projection[branch, location]
        signal = muladd(
            placement,
            eligibility.local_signal[branch] +
            0.25f0 * eligibility.local_signal[4 + branch],
            signal,
        )
        normalization += placement
    end
    signal /= max(normalization, 1.0f-6)
    signal +=
        0.20f0 * eligibility.local_signal[9] +
        0.35f0 * eligibility.local_signal[10] +
        0.20f0 * eligibility.local_signal[11]
    voltage = _compartment_voltage(runtime, block, location)
    nmda = _compartment_nmda(runtime, block, location)
    return signal *
        cell_surrogate(runtime, block) *
        (1.0f0 + 0.01f0 * abs(voltage) + 0.05f0 * abs(nmda))
end

@inline function _release_location_evidence!(
    current_utility::Matrix{Float32},
    best_utility::Matrix{Float32},
    best_location::Matrix{UInt16},
    eligibility::ReceptorEligibility,
    runtime::ReleaseCellRuntime,
    contact::Int,
    block::Int,
    current_location::Int,
    synaptic_eligibility::Float32,
    millisecond::Int,
)
    current_credit = _state_credit(
        eligibility,
        runtime,
        block,
        current_location,
    )
    current_utility[contact, block] +=
        abs(current_credit * synaptic_eligibility)
    proposal = mod1(
        millisecond + 7contact + 13block,
        ReleaseCell.OFFICIAL_LOCATION_COUNT,
    )
    proposal == current_location && return nothing
    proposal_credit = _state_credit(
        eligibility,
        runtime,
        block,
        proposal,
    )
    evidence = abs(proposal_credit * synaptic_eligibility)
    if best_location[contact, block] == UInt16(proposal)
        best_utility[contact, block] += evidence
    elseif evidence > best_utility[contact, block]
        best_location[contact, block] = UInt16(proposal)
        best_utility[contact, block] = evidence
    end
    return nothing
end

function paper_local_replay_candidate!(
    worker::PaperWorker{ReleaseCellRuntime},
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)::PaperReleaseAux
    sparse = _RELEASE_WORKER_UTILITY[worker]
    eligibility = _reset_eligibility!(
        _eligibility_state(worker, trainer),
    )
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
                _local_prediction_error_release!(
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
                        Int(aux.input_location[contact, block])
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
                    credit = _state_credit(
                        eligibility,
                        worker.runtime,
                        block,
                        location,
                    )
                    worker.gradient.input_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _release_location_evidence!(
                        sparse.input_current,
                        sparse.input_best_value,
                        sparse.input_best_location,
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
                        Int(aux.recurrent_location[contact, block])
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
                    credit = _state_credit(
                        eligibility,
                        worker.runtime,
                        block,
                        location,
                    )
                    worker.gradient.recurrent_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _release_location_evidence!(
                        sparse.recurrent_current,
                        sparse.recurrent_best_value,
                        sparse.recurrent_best_location,
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
                    credit = _state_credit(
                        eligibility,
                        worker.runtime,
                        block,
                        location,
                    )
                    worker.gradient.workspace_conductance[
                        contact,
                        block,
                    ] += credit * contact_eligibility
                    _release_location_evidence!(
                        sparse.workspace_current,
                        sparse.workspace_best_value,
                        sparse.workspace_best_location,
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

function _decay_release_utility!(
    destination::Array{Float32,3},
    decay::Float32,
)
    @inbounds @simd for index in eachindex(destination)
        destination[index] *= decay
    end
    return destination
end

function _reduce_release_sparse_utility!(
    destination::Array{Float32,3},
    current_location::Matrix{UInt16},
    workers,
    current_field::Symbol,
    best_value_field::Symbol,
    best_location_field::Symbol,
    scale::Float32,
)
    @inbounds for worker in workers
        sparse = _RELEASE_WORKER_UTILITY[worker]
        current = getproperty(sparse, current_field)
        best_value = getproperty(sparse, best_value_field)
        best_location = getproperty(sparse, best_location_field)
        for block in axes(current, 2)
            for contact in axes(current, 1)
                location = Int(current_location[contact, block])
                destination[location, contact, block] +=
                    scale * current[contact, block]
                proposal = Int(best_location[contact, block])
                proposal == 0 && continue
                destination[proposal, contact, block] +=
                    scale * best_value[contact, block]
            end
        end
    end
    return destination
end

function _reduce_paper_workers!(executor::PaperExecutor)
    trainer = executor.trainer
    aux = register_paper_trainer_aux!(trainer)
    _reduce_worker_gradients!(trainer, executor.workers)
    inverse_candidates =
        inv(Float32(max(trainer.tape.base.valid_count, 1)))
    if aux isa PaperReleaseAux
        _decay_release_utility!(
            aux.input_location_utility,
            trainer.utility_decay,
        )
        _decay_release_utility!(
            aux.recurrent_location_utility,
            trainer.utility_decay,
        )
        _decay_release_utility!(
            aux.workspace_location_utility,
            trainer.utility_decay,
        )
        scale =
            (1.0f0 - trainer.utility_decay) *
            inverse_candidates
        _reduce_release_sparse_utility!(
            aux.input_location_utility,
            aux.input_location,
            executor.workers,
            :input_current,
            :input_best_value,
            :input_best_location,
            scale,
        )
        _reduce_release_sparse_utility!(
            aux.recurrent_location_utility,
            aux.recurrent_location,
            executor.workers,
            :recurrent_current,
            :recurrent_best_value,
            :recurrent_best_location,
            scale,
        )
        _reduce_release_sparse_utility!(
            aux.workspace_location_utility,
            aux.workspace_location,
            executor.workers,
            :workspace_current,
            :workspace_best_value,
            :workspace_best_location,
            scale,
        )
    else
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
    end
    signal = 0.0f0
    arena = trainer.tape.base
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        signal += arena.listnet_q_gradient[flat]
    end
    probability =
        sigmoid(trainer.parameters.workspace_decay_logit[1])
    trainer.gradient.workspace_decay_logit[1] =
        0.94f0 *
        probability *
        (1.0f0 - probability) *
        signal *
        inverse_candidates
    return nothing
end

