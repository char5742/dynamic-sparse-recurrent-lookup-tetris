# Canonical sealed-lineage security adapter for FinalProduction.
#
# RuntimeV5 deliberately reuses RuntimeV3's frozen 11-state numerical ABI, but
# no artifact may enter through the V3 loader.  The V3 adapter/scale files below
# provide only the already-tested arena structs, kernels, and contact methods;
# every public registration and lifecycle boundary is replaced here with a V5
# sealed-contract load.

isdefined(@__MODULE__, :ReleaseCell) &&
    error("a legacy release adapter was loaded before canonical V5")
isdefined(@__MODULE__, :DistilledElevenStateCellFinal) ||
    error("FinalProduction must alias the canonical Final cell before V5")

# Private numerical ABI base.  Its artifact entry points are superseded below.
include(joinpath(@__DIR__, "PaperArenaReleaseAdapterV3.jl"))
include(joinpath(@__DIR__, "PaperArenaReleaseScaleModesV3.jl"))

include(joinpath(
    @__DIR__,
    "DistilledElevenStateCellReleaseRuntimeV5.jl",
))
const ReleaseSecurityCell =
    DistilledElevenStateCellReleaseRuntimeV5
const RELEASE_ADAPTER_SECURITY_CONTRACT_VERSION = 5
const RELEASE_REQUIRED_ARTIFACT_RUNTIME = :sealed_runtime_v5

ReleaseSecurityCell.TrustedReleaseRuntime ===
    ReleaseCell.TrustedReleaseRuntime ||
    error("RuntimeV5 does not share the frozen 11-state numerical ABI")
ReleaseSecurityCell.OFFICIAL_ELM_INPUT_DIM == 1_278 ||
    error("RuntimeV5 is not the official1278 sealed runtime")

paper_release_adapter_contract_version() =
    RELEASE_ADAPTER_SECURITY_CONTRACT_VERSION

function _load_trusted_release_runtime_v5(
    trainer::PaperTrainer,
    artifact_path::AbstractString,
)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    path = abspath(artifact_path)
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from release artifact path")

    # This is the only artifact-deserialization authority in the canonical
    # adapter.  It verifies the exact SealedOfficialELMRelease lineage, signed
    # 1278-input NeuronIO windows, live/cache equality, and then delegates only
    # the frozen numerical payload decoding to RuntimeV3.
    trusted = ReleaseSecurityCell.load_release_runtime(path)
    ReleaseSecurityCell.preflight_integrity!(trusted)
    return path, trusted
end

function _register_trusted_release_runtime_v5!(
    trainer::PaperTrainer,
    trusted::ReleaseSecurityCell.TrustedReleaseRuntime,
    path::AbstractString,
    mode::Symbol,
)
    return _register_trusted_release_runtime_v3!(
        trainer,
        trusted,
        path,
        mode,
    )
end

"""
Register a paper-scale sealed RuntimeV5 artifact.

Legacy V3 artifacts fail inside `ReleaseSecurityCell.load_release_runtime`
before any arena state or trainer auxiliary object is constructed.
"""
function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact),
)
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === true ||
        error("production V5 release API requires paper_scale=true")
    return _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :production,
    )
end

"""
Register an explicitly non-promotable development-scale RuntimeV5 artifact.
"""
function enable_development_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    development_scale_chain::Bool,
)
    development_scale_chain === true ||
        error(
            "development V5 release API requires " *
            "development_scale_chain=true",
        )
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === false ||
        error(
            "development V5 release API requires paper_scale=false",
        )
    return _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :development,
    )
end

function _assert_runtime_v5_file_unchanged!(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
)
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    path = something(
        trainer.cell_artifact,
        error("V5 trainer lost its sealed artifact path"),
    )
    fresh = ReleaseSecurityCell.load_release_runtime(path)
    fresh.artifact_sha256 == aux.trusted.artifact_sha256 ||
        error("sealed RuntimeV5 artifact changed after registration")
    fresh.expected_parameter_sha256 ==
        aux.trusted.expected_parameter_sha256 ||
        error("sealed RuntimeV5 parameter identity changed")
    fresh.location_mapping_sha256 ==
        aux.trusted.location_mapping_sha256 ||
        error("sealed RuntimeV5 location mapping changed")
    fresh.source_segment_catalog_sha256 ==
        aux.trusted.source_segment_catalog_sha256 ||
        error("sealed RuntimeV5 segment catalog changed")
    fresh.paper_scale == aux.trusted.paper_scale ||
        error("sealed RuntimeV5 scale declaration changed")
    ReleaseSecurityCell.preflight_integrity!(aux.trusted)
    ReleaseSecurityCell.preflight_integrity!(fresh)
    return fresh.artifact_sha256
end

function paper_preflight_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production preflight has no V5 release runtime")
    return _assert_runtime_v5_file_unchanged!(trainer, aux)
end

function paper_checkpoint_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production checkpoint has no V5 release runtime")
    digest = _assert_runtime_v5_file_unchanged!(trainer, aux)
    ReleaseSecurityCell.checkpoint_integrity!(aux.trusted)
    return digest
end

function paper_end_run_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production run has no V5 release runtime")
    digest = _assert_runtime_v5_file_unchanged!(trainer, aux)
    ReleaseSecurityCell.end_run_integrity!(aux.trusted)
    return digest
end
