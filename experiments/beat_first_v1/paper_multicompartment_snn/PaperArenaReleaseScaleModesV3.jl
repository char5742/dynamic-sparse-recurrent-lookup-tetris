# Explicit scale-mode APIs for PaperArenaReleaseAdapterV3.
#
# Include immediately after PaperArenaReleaseAdapterV3.jl.  Production remains
# fail-closed on paper_scale=true.  A paper_scale=false V3 artifact is accepted
# only through the separately named development API and only when its explicit
# development_scale_chain flag is true.  Both paths use the same V3 artifact
# loader, semantic/multi-target/hash gates, frozen runtime, contact mapping, and
# arena state construction.

const _RELEASE_SCALE_MODE = IdDict{Any,Symbol}()

function _register_trusted_release_runtime_v3!(
    trainer::PaperTrainer,
    trusted::ReleaseCell.TrustedReleaseRuntime,
    path::AbstractString,
    mode::Symbol,
)
    mode in (:production, :development) ||
        error("unknown release scale mode $mode")
    data = JLD2.load(path)
    payload = data["payload"]
    regions = String.(collect(_payload_release_value(
        payload,
        :official_segment_region,
        (),
    )))
    length(regions) == ReleaseCell.OFFICIAL_LOCATION_COUNT ||
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
    _RELEASE_SCALE_MODE[trainer] = mode
    return aux
end

function _load_trusted_release_runtime_v3(
    trainer::PaperTrainer,
    artifact_path::AbstractString,
)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    path = abspath(artifact_path)
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from release artifact path")
    trusted = ReleaseCell.load_release_runtime(path)
    ReleaseCell.preflight_integrity!(trusted)
    return path, trusted
end

"""
Register the canonical production release runtime.

Only a V3 artifact declaring `paper_scale=true` is accepted.
"""
function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact),
)
    path, trusted = _load_trusted_release_runtime_v3(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === true ||
        error("production release API requires paper_scale=true")
    return _register_trusted_release_runtime_v3!(
        trainer,
        trusted,
        path,
        :production,
    )
end

"""
Register an explicitly development-scale release runtime.

This API cannot weaken the production path: the caller must pass
`development_scale_chain=true`, and the V3 artifact must declare
`paper_scale=false`.  All other V3 release gates remain identical.
"""
function enable_development_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    development_scale_chain::Bool,
)
    development_scale_chain === true ||
        error(
            "development release API requires " *
            "development_scale_chain=true",
        )
    path, trusted = _load_trusted_release_runtime_v3(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === false ||
        error(
            "development release API requires a " *
            "paper_scale=false V3 artifact",
        )
    return _register_trusted_release_runtime_v3!(
        trainer,
        trusted,
        path,
        :development,
    )
end

function paper_release_scale_mode(trainer::PaperTrainer)
    mode = get(_RELEASE_SCALE_MODE, trainer, :unregistered)
    mode in (:production, :development) ||
        error("release scale mode is not registered")
    return mode
end

function _assert_release_scale_mode!(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
)
    mode = paper_release_scale_mode(trainer)
    if mode === :production
        aux.trusted.paper_scale === true ||
            error("production release runtime lost paper_scale=true")
    else
        aux.trusted.paper_scale === false ||
            error("development release runtime lost paper_scale=false")
    end
    return mode
end

function paper_preflight_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production preflight has no release runtime")
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.preflight_integrity!(aux.trusted)
end

function paper_checkpoint_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production checkpoint has no release runtime")
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.checkpoint_integrity!(aux.trusted)
end

function paper_end_run_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production run has no release runtime")
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    return ReleaseCell.end_run_integrity!(aux.trusted)
end
