# Canonical FinalProduction security adapter:
# sealed official ELM final.v2 -> frozen 11-state RuntimeV5 artifact.
#
# RuntimeV3 is used only as the immutable numerical ABI under RuntimeV5Final.
# No public registration or lifecycle method deserializes through RuntimeV3.

isdefined(@__MODULE__, :ReleaseCell) &&
    error("a legacy release adapter was loaded before canonical V5Final")
isdefined(@__MODULE__, :DistilledElevenStateCellFinal) ||
    error("FinalProduction must alias the canonical Final cell first")

include(joinpath(@__DIR__, "PaperArenaReleaseAdapterV3.jl"))
include(joinpath(@__DIR__, "PaperArenaReleaseScaleModesV3.jl"))
include(joinpath(
    @__DIR__,
    "DistilledElevenStateCellReleaseRuntimeV5Final.jl",
))

const ReleaseSecurityCell =
    DistilledElevenStateCellReleaseRuntimeV5Final
const RELEASE_ADAPTER_SECURITY_CONTRACT_VERSION = 5
const RELEASE_REQUIRED_ARTIFACT_RUNTIME =
    :sealed_v2_runtime_v5

ReleaseSecurityCell.TrustedReleaseRuntime ===
    ReleaseCell.TrustedReleaseRuntime ||
    error("RuntimeV5Final does not share the frozen 11-state ABI")
ReleaseSecurityCell.OFFICIAL_ELM_INPUT_DIM == 1_278 ||
    error("RuntimeV5Final is not official1278")
ReleaseSecurityCell.SEALED_RELEASE_SCHEMA ==
    "hd_swsnn.paper_elm_v2.sealed_release.final.v2" ||
    error("RuntimeV5Final is not bound to sealed final.v2")
ReleaseSecurityCell.SEALED_RELEASE_ARTIFACT_KIND ==
    "SealedOfficialELMReleaseV2" ||
    error("RuntimeV5Final sealed artifact kind differs")

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

function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact),
)
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === true ||
        error("production V5Final API requires paper_scale=true")
    return _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :production,
    )
end

function enable_development_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    development_scale_chain::Bool,
)
    development_scale_chain === true ||
        error(
            "development V5Final API requires " *
            "development_scale_chain=true",
        )
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
    )
    trusted.paper_scale === false ||
        error(
            "development V5Final API requires paper_scale=false",
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
    trainer.cell_artifact === nothing &&
        error("V5Final trainer lost its sealed artifact path")
    path = trainer.cell_artifact::String
    fresh = ReleaseSecurityCell.load_release_runtime(path)
    fresh.artifact_sha256 == aux.trusted.artifact_sha256 ||
        error("sealed RuntimeV5Final artifact changed")
    fresh.expected_parameter_sha256 ==
        aux.trusted.expected_parameter_sha256 ||
        error("sealed RuntimeV5Final parameter identity changed")
    fresh.location_mapping_sha256 ==
        aux.trusted.location_mapping_sha256 ||
        error("sealed RuntimeV5Final location mapping changed")
    fresh.source_segment_catalog_sha256 ==
        aux.trusted.source_segment_catalog_sha256 ||
        error("sealed RuntimeV5Final segment catalog changed")
    fresh.paper_scale == aux.trusted.paper_scale ||
        error("sealed RuntimeV5Final scale declaration changed")
    ReleaseSecurityCell.preflight_integrity!(aux.trusted)
    ReleaseSecurityCell.preflight_integrity!(fresh)
    return fresh.artifact_sha256
end

function paper_preflight_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("preflight has no RuntimeV5Final")
    return _assert_runtime_v5_file_unchanged!(trainer, aux)
end

function paper_checkpoint_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("checkpoint has no RuntimeV5Final")
    digest = _assert_runtime_v5_file_unchanged!(trainer, aux)
    ReleaseSecurityCell.checkpoint_integrity!(aux.trusted)
    return digest
end

function paper_end_run_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("run end has no RuntimeV5Final")
    digest = _assert_runtime_v5_file_unchanged!(trainer, aux)
    ReleaseSecurityCell.end_run_integrity!(aux.trusted)
    return digest
end
