# Post-overlay for PaperArenaReleaseAdapterV3.
#
# Include order inside the exact FinalProduction arena:
#   PaperArenaReleaseAdapterV3.jl
#   PaperArenaReleaseLearning.jl
#   PaperArenaReleaseStructure.jl
#   PaperArenaReleaseLegalContactMethodsV3.jl
#
# The generic release learning/structure files intentionally retain a
# 1:OFFICIAL_LOCATION_COUNT search.  This small final overlay restricts every
# learned proposal and consolidation move to official dendritic IDs 2:640.
# Utility tensors remain indexed by all 642 official morphology IDs.

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
    _release_location_is_legal(current_location) ||
        error("location evidence received forbidden soma/axon segment ID")
    current_credit = _state_credit(
        eligibility,
        runtime,
        block,
        current_location,
    )
    current_utility[contact, block] +=
        abs(current_credit * synaptic_eligibility)
    proposal = _release_legal_proposal(
        millisecond,
        contact,
        block,
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

function _consolidate_one_location_canonical!(
    trainer::PaperTrainer,
)
    trainer.optimizer.step % trainer.location_interval == 0 ||
        return 0
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux || return 0
    model = trainer.model
    trainer.location_cursor =
        mod(trainer.location_cursor, 2model.blocks) + 1
    input_contact =
        trainer.location_cursor <= model.blocks
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
        aux.input_location_utility :
        aux.recurrent_location_utility
    locations = input_contact ?
        aux.input_location :
        aux.recurrent_location
    location = locations[contact, block]
    _release_location_is_legal(location) ||
        error("consolidation received forbidden soma/axon segment ID")
    kind = if input_contact
        rail = Int(model.input_rail[contact, block])
        _contact_kind(model, rail)
    else
        source = Int(model.recurrent_source[contact, block])
        _source_block_kind(model, source)
    end
    current_slot = Int(location)
    best = current_slot
    best_value = utilities[current_slot, contact, block]
    @inbounds for slot in _release_legal_consolidation_slots()
        candidate = UInt16(slot)
        candidate == location ||
            _release_capacity_available(
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
    locations[contact, block] = UInt16(best)
    if input_contact
        trainer.optimizer.first_moment.input_conductance[
            contact,
            block,
        ] = 0.0f0
        trainer.optimizer.second_moment.input_conductance[
            contact,
            block,
        ] = 0.0f0
    else
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

function _consolidate_workspace_location!(
    trainer::PaperTrainer,
)
    trainer.optimizer.step % trainer.location_interval == 0 ||
        return 0
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux || return 0
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
    _release_location_is_legal(current) ||
        error("workspace consolidation received forbidden soma/axon segment ID")
    current_slot = Int(current)
    best = current_slot
    best_value =
        aux.workspace_location_utility[current_slot, contact, block]
    @inbounds for slot in _release_legal_consolidation_slots()
        candidate = UInt16(slot)
        candidate == current ||
            _release_workspace_location_free(
                trainer,
                aux,
                block,
                candidate,
                contact,
            ) || continue
        value =
            aux.workspace_location_utility[slot, contact, block] -
            trainer.utility_connection_cost
        if value > best_value + 1.0f-4
            best = slot
            best_value = value
        end
    end
    best == current_slot && return 0
    aux.workspace_location[contact, block] = UInt16(best)
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
