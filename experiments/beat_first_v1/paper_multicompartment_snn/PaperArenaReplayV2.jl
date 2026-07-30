mutable struct ReceptorEligibility
    input_rise::Array{Float32,3}
    input_decay::Array{Float32,3}
    recurrent_rise::Array{Float32,3}
    recurrent_decay::Array{Float32,3}
    workspace_rise::Array{Float32,3}
    workspace_decay::Array{Float32,3}
    prediction::Vector{Float32}
    local_error::Vector{Float32}
    local_signal::Vector{Float32}
end

const _WORKER_ELIGIBILITY = IdDict{Any,ReceptorEligibility}()

function _eligibility_state(worker::PaperWorker, trainer::PaperTrainer)
    return get!(_WORKER_ELIGIBILITY, worker) do
        model = trainer.model
        ReceptorEligibility(
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
end

function _reset_eligibility!(eligibility::ReceptorEligibility)
    fill!(eligibility.input_rise, 0.0f0)
    fill!(eligibility.input_decay, 0.0f0)
    fill!(eligibility.recurrent_rise, 0.0f0)
    fill!(eligibility.recurrent_decay, 0.0f0)
    fill!(eligibility.workspace_rise, 0.0f0)
    fill!(eligibility.workspace_decay, 0.0f0)
    return eligibility
end

@inline function _double_scale(
    rise::Float32,
    decay::Float32,
)
    peak_time =
        rise * decay / (decay - rise) *
        log(decay / rise)
    return inv(
        exp(-peak_time / decay) -
        exp(-peak_time / rise),
    )
end

const _RECEPTOR_RISE_DECAY = Float32[
    exp(-1.0f0 / 0.20f0),
    exp(-1.0f0 / 0.29f0),
    exp(-1.0f0 / 0.20f0),
]
const _RECEPTOR_DECAY_DECAY = Float32[
    exp(-1.0f0 / 1.70f0),
    exp(-1.0f0 / 43.0f0),
    exp(-1.0f0 / 8.0f0),
]
const _RECEPTOR_SCALE = Float32[
    _double_scale(0.20f0, 1.70f0),
    _double_scale(0.29f0, 43.0f0),
    _double_scale(0.20f0, 8.0f0),
]
const _RECEPTOR_MAX_NS = Float32[0.40f0, 0.30f0, 0.70f0]

@inline function _advance_contact_trace!(
    rise,
    decay,
    contact::Int,
    block::Int,
    receptor::Int,
    event::Float32,
)
    rise[receptor, contact, block] = muladd(
        _RECEPTOR_RISE_DECAY[receptor],
        rise[receptor, contact, block],
        event,
    )
    decay[receptor, contact, block] = muladd(
        _RECEPTOR_DECAY_DECAY[receptor],
        decay[receptor, contact, block],
        event,
    )
    return max(
        0.0f0,
        _RECEPTOR_SCALE[receptor] *
        (
            decay[receptor, contact, block] -
            rise[receptor, contact, block]
        ),
    ) * _RECEPTOR_MAX_NS[receptor]
end

@inline function _compartment_voltage(
    runtime::DetailedCellRuntime,
    block::Int,
    compartment::Int,
)
    return runtime.states[block].voltage_mv[compartment]
end

@inline function _compartment_nmda(
    runtime::DetailedCellRuntime,
    block::Int,
    compartment::Int,
)
    return abs(runtime.diagnostics[block].nmda_current[compartment])
end

@inline function _compartment_voltage(
    runtime::DistilledCellRuntime,
    block::Int,
    compartment::Int,
)
    projection =
        runtime.parameters.compartment_projection
    diagnostics = runtime.diagnostics[block]
    value = 0.0f0
    @inbounds for region in 1:4
        value = muladd(
            projection[region, compartment],
            diagnostics.dendritic_voltage_mv[region],
            value,
        )
    end
    return value
end

@inline function _compartment_nmda(
    runtime::DistilledCellRuntime,
    block::Int,
    compartment::Int,
)
    projection =
        runtime.parameters.compartment_projection
    diagnostics = runtime.diagnostics[block]
    value = 0.0f0
    @inbounds for region in 1:4
        value = muladd(
            projection[region, compartment],
            abs(diagnostics.nmda_current[region]),
            value,
        )
    end
    return value
end

@inline function _nmda_jacobian_factor(voltage::Float32)
    block = Hay.nmda_magnesium_block(voltage)
    derivative = 0.062f0 * block * (1.0f0 - block)
    return block + derivative * min(abs(voltage), 100.0f0)
end

@inline function _contact_eligibility!(
    rise,
    decay,
    contact::Int,
    block::Int,
    kind::UInt8,
    event::Float32,
    voltage::Float32,
)
    if kind == Model.EXCITATORY
        ampa = _advance_contact_trace!(
            rise,
            decay,
            contact,
            block,
            1,
            event,
        )
        nmda = _advance_contact_trace!(
            rise,
            decay,
            contact,
            block,
            2,
            event,
        )
        return ampa + nmda * _nmda_jacobian_factor(voltage)
    end
    gaba = _advance_contact_trace!(
        rise,
        decay,
        contact,
        block,
        3,
        event,
    )
    # Increasing GABA conductance suppresses the postsynaptic soma event.
    return -gaba
end

@inline function _normalized_state_value(
    index::Int,
    value::Float32,
)
    scale = index <= 4 || index == 9 || index == 10 ?
        100.0f0 : 1.0f0
    return tanh(value / scale)
end

function _local_prediction_error!(
    eligibility::ReceptorEligibility,
    worker::PaperWorker,
    trainer::PaperTrainer,
    aux::PaperTrainerAux,
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
    @inbounds for output in 1:OUTPUT_DIM
        value = 0.0f0
        for state in 1:11
            value = muladd(
                aux.regional_projection[state, output, block],
                _normalized_state_value(
                    state,
                    worker.local_state[state],
                ),
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
        weight = abs(tau - (residual < 0.0f0 ? 1.0f0 : 0.0f0))
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
        # Approximate the frozen 11-state Jacobian by its local activation
        # derivative. This is state-specific and has no global-head Jacobian.
        normalized = _normalized_state_value(
            state,
            worker.local_state[state],
        )
        eligibility.local_signal[state] =
            signal * (1.0f0 - normalized * normalized)
        worker.block_signal[state, block] =
            eligibility.local_signal[state]
    end
    return nothing
end

@inline function _state_credit(
    eligibility::ReceptorEligibility,
    runtime,
    block::Int,
    compartment::Int,
)
    tree = Hay.paper_hay_tree()
    coordinate = _region_coordinate(tree, compartment)
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
        (1.0f0 + 0.01f0 * abs(voltage) + 0.05f0 * abs(nmda))
end

function _location_evidence_v2!(
    utility,
    catalog,
    eligibility::ReceptorEligibility,
    runtime,
    contact::Int,
    block::Int,
    current_location::Int,
    synaptic_eligibility::Float32,
    millisecond::Int,
)
    current_slot = findfirst(==(UInt8(current_location)), catalog)
    current_slot === nothing && return nothing
    current_credit = _state_credit(
        eligibility,
        runtime,
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
    proposal_credit = _state_credit(
        eligibility,
        runtime,
        block,
        proposal_location,
    )
    # Same-region segments still differ because detailed voltage/NMDA or the
    # frozen compartment projection is location-specific.
    utility[proposal, contact, block] +=
        abs(proposal_credit * synaptic_eligibility)
    return nothing
end

function paper_local_replay_candidate_v2!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)
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
                    location = Int(
                        trainer.input_location[contact, block],
                    )
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
                    _location_evidence_v2!(
                        worker.input_location_utility,
                        trainer.eligible_compartments,
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
                    source = Int(model.recurrent_source[contact, block])
                    event = worker.previous_spike[source]
                    location = Int(
                        trainer.recurrent_location[contact, block],
                    )
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
                    _location_evidence_v2!(
                        worker.recurrent_location_utility,
                        trainer.eligible_compartments,
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
                    _location_evidence_v2!(
                        worker.workspace_location_utility,
                        trainer.eligible_compartments,
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
