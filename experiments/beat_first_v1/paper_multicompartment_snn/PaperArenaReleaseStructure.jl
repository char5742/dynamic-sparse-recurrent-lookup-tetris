# Shared E/I cable-capacity consolidation for all official location families.

@inline function _release_capacity_used(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
    block::Int,
    location::UInt16,
    kind::UInt8,
)
    used = 0
    model = trainer.model
    @inbounds for contact in 1:model.sensory_contacts
        rail = Int(model.input_rail[contact, block])
        _contact_kind(model, rail) == kind || continue
        aux.input_location[contact, block] == location &&
            (used += 1)
    end
    @inbounds for contact in 1:model.recurrent_contacts
        source = Int(model.recurrent_source[contact, block])
        _source_block_kind(model, source) == kind || continue
        aux.recurrent_location[contact, block] == location &&
            (used += 1)
    end
    # Workspace sources change with routing; reserve their segment against
    # both Dale types so all three location families share cable capacity.
    @inbounds for contact in 1:model.workspace_contacts
        aux.workspace_location[contact, block] == location &&
            (used += 1)
    end
    return used
end

@inline function _release_capacity_available(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
    block::Int,
    location::UInt16,
    kind::UInt8,
)
    capacity = kind == Model.EXCITATORY ?
        aux.excitatory_capacity[Int(location)] :
        aux.inhibitory_capacity[Int(location)]
    return _release_capacity_used(
        trainer,
        aux,
        block,
        location,
        kind,
    ) < capacity
end

@inline function _release_workspace_location_free(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
    block::Int,
    location::UInt16,
    current_contact::Int,
)
    model = trainer.model
    @inbounds for contact in 1:model.sensory_contacts
        aux.input_location[contact, block] == location &&
            return false
    end
    @inbounds for contact in 1:model.recurrent_contacts
        aux.recurrent_location[contact, block] == location &&
            return false
    end
    @inbounds for contact in 1:model.workspace_contacts
        contact == current_contact && continue
        aux.workspace_location[contact, block] == location &&
            return false
    end
    return true
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
    @inbounds for slot in 1:ReleaseCell.OFFICIAL_LOCATION_COUNT
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
    current_slot = Int(current)
    best = current_slot
    best_value =
        aux.workspace_location_utility[current_slot, contact, block]
    @inbounds for slot in 1:ReleaseCell.OFFICIAL_LOCATION_COUNT
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

function paper_aux_snapshot(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    if aux isa PaperReleaseAux
        return (;
            input_location=copy(aux.input_location),
            recurrent_location=copy(aux.recurrent_location),
            workspace_location=copy(aux.workspace_location),
            input_location_utility=
                copy(aux.input_location_utility),
            recurrent_location_utility=
                copy(aux.recurrent_location_utility),
            workspace_location_utility=
                copy(aux.workspace_location_utility),
            contact_capacity=(;
                excitatory=copy(aux.excitatory_capacity),
                inhibitory=copy(aux.inhibitory_capacity),
            ),
            regional_projection=copy(aux.regional_projection),
            official_segment_region=
                copy(aux.official_segment_region),
            location_mapping_sha256=
                aux.location_mapping_sha256,
            location_index_type="UInt16",
            official_segment_count=
                ReleaseCell.OFFICIAL_LOCATION_COUNT,
            lineage=aux.lineage,
        )
    end
    return (;
        input_location=copy(trainer.input_location),
        recurrent_location=copy(trainer.recurrent_location),
        workspace_location=copy(aux.workspace_location),
        input_location_utility=
            copy(trainer.input_location_utility),
        recurrent_location_utility=
            copy(trainer.recurrent_location_utility),
        workspace_location_utility=
            copy(aux.workspace_location_utility),
        contact_capacity=(;
            excitatory=copy(aux.excitatory_capacity),
            inhibitory=copy(aux.inhibitory_capacity),
        ),
        regional_projection=copy(aux.regional_projection),
        lineage=aux.lineage,
    )
end

function restore_paper_aux_snapshot!(
    trainer::PaperTrainer,
    snapshot,
)
    aux = register_paper_trainer_aux!(trainer)
    if aux isa PaperReleaseAux
        String(snapshot.location_index_type) == "UInt16" ||
            error("checkpoint location index type is not UInt16")
        Int(snapshot.official_segment_count) ==
            ReleaseCell.OFFICIAL_LOCATION_COUNT ||
            error("checkpoint does not contain 642 official segments")
        String(snapshot.location_mapping_sha256) ==
            aux.location_mapping_sha256 ||
            error("checkpoint official location mapping differs")
        copyto!(aux.input_location, snapshot.input_location)
        copyto!(
            aux.recurrent_location,
            snapshot.recurrent_location,
        )
        copyto!(
            aux.workspace_location,
            snapshot.workspace_location,
        )
        copyto!(
            aux.input_location_utility,
            snapshot.input_location_utility,
        )
        copyto!(
            aux.recurrent_location_utility,
            snapshot.recurrent_location_utility,
        )
        copyto!(
            aux.workspace_location_utility,
            snapshot.workspace_location_utility,
        )
        copyto!(
            aux.excitatory_capacity,
            snapshot.contact_capacity.excitatory,
        )
        copyto!(
            aux.inhibitory_capacity,
            snapshot.contact_capacity.inhibitory,
        )
        copyto!(
            aux.regional_projection,
            snapshot.regional_projection,
        )
        snapshot.lineage == aux.lineage ||
            error("checkpoint release lineage differs")
        return trainer
    end
    copyto!(trainer.input_location, snapshot.input_location)
    copyto!(
        trainer.recurrent_location,
        snapshot.recurrent_location,
    )
    copyto!(aux.workspace_location, snapshot.workspace_location)
    copyto!(
        trainer.input_location_utility,
        snapshot.input_location_utility,
    )
    copyto!(
        trainer.recurrent_location_utility,
        snapshot.recurrent_location_utility,
    )
    copyto!(
        aux.workspace_location_utility,
        snapshot.workspace_location_utility,
    )
    copyto!(
        aux.excitatory_capacity,
        snapshot.contact_capacity.excitatory,
    )
    copyto!(
        aux.inhibitory_capacity,
        snapshot.contact_capacity.inhibitory,
    )
    copyto!(
        aux.regional_projection,
        snapshot.regional_projection,
    )
    snapshot.lineage == aux.lineage ||
        error("checkpoint cell lineage differs")
    return trainer
end

