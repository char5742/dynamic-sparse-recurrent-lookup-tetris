# Runtime-version-independent Official Hay contact boundary.
#
# Include this file after a release adapter has established PaperReleaseAux,
# ReleaseCellRuntime, and the local-learning types in the exact arena module.
# It intentionally does not load an artifact or replace enable_release_runtime!;
# those responsibilities belong to the active release adapter.

const OFFICIAL_HAY_SEGMENT_COUNT = 642
const OFFICIAL_HAY_DENDRITIC_FIRST = 2
const OFFICIAL_HAY_DENDRITIC_LAST = 640
const OFFICIAL_HAY_DENDRITIC_COUNT = 639
const OFFICIAL_HAY_DENDRITIC_RANGE =
    OFFICIAL_HAY_DENDRITIC_FIRST:OFFICIAL_HAY_DENDRITIC_LAST

ReleaseCell.OFFICIAL_LOCATION_COUNT == OFFICIAL_HAY_SEGMENT_COUNT ||
    error("release adapter does not preserve the official 642-segment Hay mapping")
length(OFFICIAL_HAY_DENDRITIC_RANGE) ==
    OFFICIAL_HAY_DENDRITIC_COUNT ||
    error("official dendritic segment catalog must contain 639 IDs")

@inline _official_hay_contact_is_legal(location::Integer) =
    OFFICIAL_HAY_DENDRITIC_FIRST <= location <=
    OFFICIAL_HAY_DENDRITIC_LAST

@inline function _official_hay_contact_id(seed::Integer)
    return UInt16(
        OFFICIAL_HAY_DENDRITIC_FIRST +
        mod(seed - 1, OFFICIAL_HAY_DENDRITIC_COUNT),
    )
end

@inline function _official_hay_contact_proposal(
    millisecond::Int,
    contact::Int,
    block::Int,
)
    return Int(_official_hay_contact_id(
        millisecond + 7contact + 13block,
    ))
end

@inline _official_hay_consolidation_slots() =
    OFFICIAL_HAY_DENDRITIC_RANGE

_official_hay_contact_catalog() =
    UInt16.(OFFICIAL_HAY_DENDRITIC_RANGE)

function assert_official_hay_contact_boundary!(
    aux::PaperReleaseAux,
)
    aux.location_catalog == _official_hay_contact_catalog() ||
        error("release contact catalog must be UInt16 official dendrites 2:640")
    for (label, locations) in (
        ("input", aux.input_location),
        ("recurrent", aux.recurrent_location),
        ("workspace", aux.workspace_location),
    )
        @inbounds for location in locations
            _official_hay_contact_is_legal(location) ||
                error(
                    "$label contact uses forbidden soma/axon segment ID " *
                    string(location),
                )
        end
    end
    for (label, utility) in (
        ("input", aux.input_location_utility),
        ("recurrent", aux.recurrent_location_utility),
        ("workspace", aux.workspace_location_utility),
    )
        size(utility, 1) == OFFICIAL_HAY_SEGMENT_COUNT ||
            error("$label utility does not preserve 642 official indices")
    end
    length(aux.official_segment_region) ==
        OFFICIAL_HAY_SEGMENT_COUNT ||
        error("release adapter lost the complete 642-segment morphology map")
    return aux
end

function _release_initial_locations(model)
    inputs = Matrix{UInt16}(
        undef,
        model.sensory_contacts,
        model.blocks,
    )
    recurrent = Matrix{UInt16}(
        undef,
        model.recurrent_contacts,
        model.blocks,
    )
    workspace = Matrix{UInt16}(
        undef,
        model.workspace_contacts,
        model.blocks,
    )
    @inbounds for block in 1:model.blocks
        base = 53 * (block - 1)
        for contact in 1:model.sensory_contacts
            inputs[contact, block] =
                _official_hay_contact_id(base + contact)
        end
        for contact in 1:model.recurrent_contacts
            recurrent[contact, block] =
                _official_hay_contact_id(
                    base + model.sensory_contacts + contact,
                )
        end
        for contact in 1:model.workspace_contacts
            workspace[contact, block] =
                _official_hay_contact_id(
                    base +
                    model.sensory_contacts +
                    model.recurrent_contacts +
                    contact,
                )
        end
    end
    return inputs, recurrent, workspace
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
    _official_hay_contact_is_legal(current_location) ||
        error("location evidence received forbidden soma/axon segment ID")
    current_credit = _state_credit(
        eligibility,
        runtime,
        block,
        current_location,
    )
    current_utility[contact, block] +=
        abs(current_credit * synaptic_eligibility)
    proposal = _official_hay_contact_proposal(
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
    _official_hay_contact_is_legal(location) ||
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
    @inbounds for slot in _official_hay_consolidation_slots()
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
    _official_hay_contact_is_legal(current) ||
        error("workspace consolidation received forbidden soma/axon segment ID")
    current_slot = Int(current)
    best = current_slot
    best_value =
        aux.workspace_location_utility[current_slot, contact, block]
    @inbounds for slot in _official_hay_consolidation_slots()
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

