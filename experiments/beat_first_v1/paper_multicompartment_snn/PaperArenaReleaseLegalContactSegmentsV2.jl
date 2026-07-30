# Official Hay contact-location boundary for the release-v2 arena.
#
# The morphology and distilled projection retain all 642 official segment IDs:
#   1       soma
#   2:640   dendritic contact locations (639 legal IDs)
#   641:642 axon
#
# Only IDs 2:640 may be assigned to input, recurrent, or workspace contacts.
# Utility tensors deliberately retain a 642-entry first dimension so that the
# official morphology indexing and artifact hashes remain unchanged.

const RELEASE_OFFICIAL_SEGMENT_COUNT =
    ReleaseCell.OFFICIAL_LOCATION_COUNT
const RELEASE_LEGAL_CONTACT_FIRST = 2
const RELEASE_LEGAL_CONTACT_LAST = 640
const RELEASE_LEGAL_CONTACT_COUNT =
    RELEASE_LEGAL_CONTACT_LAST - RELEASE_LEGAL_CONTACT_FIRST + 1
const RELEASE_LEGAL_CONTACT_RANGE =
    RELEASE_LEGAL_CONTACT_FIRST:RELEASE_LEGAL_CONTACT_LAST

RELEASE_OFFICIAL_SEGMENT_COUNT == 642 ||
    error("official Hay morphology must contain exactly 642 segments")
RELEASE_LEGAL_CONTACT_COUNT == 639 ||
    error("official Hay contact catalog must contain exactly 639 dendritic segments")

@inline _release_location_is_legal(location::Integer) =
    RELEASE_LEGAL_CONTACT_FIRST <= location <= RELEASE_LEGAL_CONTACT_LAST

@inline function _release_legal_location(seed::Integer)
    return UInt16(
        RELEASE_LEGAL_CONTACT_FIRST +
        mod(seed - 1, RELEASE_LEGAL_CONTACT_COUNT),
    )
end

@inline function _release_legal_proposal(
    millisecond::Int,
    contact::Int,
    block::Int,
)
    return Int(_release_legal_location(
        millisecond + 7contact + 13block,
    ))
end

@inline _release_legal_consolidation_slots() =
    RELEASE_LEGAL_CONTACT_RANGE

function _release_legal_location_catalog()
    return UInt16.(RELEASE_LEGAL_CONTACT_RANGE)
end

function _assert_release_legal_location_matrix(
    locations::AbstractMatrix{UInt16},
    label::AbstractString,
)
    @inbounds for location in locations
        _release_location_is_legal(location) ||
            error(
                "$label contains forbidden soma/axon segment ID " *
                string(location),
            )
    end
    return locations
end

function _assert_release_legal_contact_state!(
    aux::PaperReleaseAux,
)
    aux.location_catalog == _release_legal_location_catalog() ||
        error("release contact catalog is not exactly UInt16 official IDs 2:640")
    _assert_release_legal_location_matrix(
        aux.input_location,
        "input locations",
    )
    _assert_release_legal_location_matrix(
        aux.recurrent_location,
        "recurrent locations",
    )
    _assert_release_legal_location_matrix(
        aux.workspace_location,
        "workspace locations",
    )
    size(aux.input_location_utility, 1) ==
        RELEASE_OFFICIAL_SEGMENT_COUNT ||
        error("input location utility no longer preserves 642 official indices")
    size(aux.recurrent_location_utility, 1) ==
        RELEASE_OFFICIAL_SEGMENT_COUNT ||
        error("recurrent location utility no longer preserves 642 official indices")
    size(aux.workspace_location_utility, 1) ==
        RELEASE_OFFICIAL_SEGMENT_COUNT ||
        error("workspace location utility no longer preserves 642 official indices")
    length(aux.official_segment_region) ==
        RELEASE_OFFICIAL_SEGMENT_COUNT ||
        error("official 642-segment morphology mapping is incomplete")
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
                _release_legal_location(base + contact)
        end
        for contact in 1:model.recurrent_contacts
            recurrent[contact, block] = _release_legal_location(
                base + model.sensory_contacts + contact,
            )
        end
        for contact in 1:model.workspace_contacts
            workspace[contact, block] = _release_legal_location(
                base +
                model.sensory_contacts +
                model.recurrent_contacts +
                contact,
            )
        end
    end
    return inputs, recurrent, workspace
end

function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact),
)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    path = abspath(artifact_path)
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from release artifact path")
    trusted = ReleaseCell.load_release_runtime(path)
    ReleaseCell.preflight_integrity!(trusted)
    data = JLD2.load(path)
    payload = data["payload"]
    regions = String.(collect(_payload_release_value(
        payload,
        :official_segment_region,
        (),
    )))
    length(regions) == RELEASE_OFFICIAL_SEGMENT_COUNT ||
        error("release artifact has no complete 642-segment region map")
    String(_payload_release_value(
        payload,
        :semantic_state_scale,
        "",
    )) == "normalized_unit_interval" ||
        error("release semantic states are not normalized_unit_interval")
    String(_payload_release_value(
        payload,
        :location_index_type,
        "",
    )) == "UInt16" ||
        error("release location index type is not UInt16")
    input_location, recurrent_location, workspace_location =
        _release_initial_locations(trainer.model)
    catalog = _release_legal_location_catalog()
    capacity = fill(
        Int16(1),
        RELEASE_OFFICIAL_SEGMENT_COUNT,
    )
    parameters = trusted.parameters
    lineage = PaperLineage(
        parameters.detailed_kernel_hash,
        parameters.frozen_twin_artifact_hash,
        trusted.artifact_sha256,
        trusted.expected_parameter_sha256,
        ReleaseCell.Final.DISTILLED_ARTIFACT_SCHEMA,
    )
    aux = PaperReleaseAux(
        trusted,
        parameters,
        deepcopy(parameters),
        lineage,
        catalog,
        regions,
        copy(capacity),
        copy(capacity),
        input_location,
        recurrent_location,
        workspace_location,
        zeros(
            Float32,
            RELEASE_OFFICIAL_SEGMENT_COUNT,
            trainer.model.sensory_contacts,
            trainer.model.blocks,
        ),
        zeros(
            Float32,
            RELEASE_OFFICIAL_SEGMENT_COUNT,
            trainer.model.recurrent_contacts,
            trainer.model.blocks,
        ),
        zeros(
            Float32,
            RELEASE_OFFICIAL_SEGMENT_COUNT,
            trainer.model.workspace_contacts,
            trainer.model.blocks,
        ),
        _regional_projection(trainer.model.blocks),
        trusted.location_mapping_sha256,
    )
    _assert_release_legal_contact_state!(aux)
    _TRAINER_AUX[trainer] = aux
    return aux
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

function paper_preflight_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production preflight has no release runtime")
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.preflight_integrity!(aux.trusted)
end

function paper_checkpoint_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production checkpoint has no release runtime")
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.checkpoint_integrity!(aux.trusted)
end

function paper_end_run_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production run has no release runtime")
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.end_run_integrity!(aux.trusted)
end

