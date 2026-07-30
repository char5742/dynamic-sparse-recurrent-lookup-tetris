# Exact final development-chain composition.
#
# Downstream artifact owners expose only the stable verified/file-hash gate
# contract today.  This composer therefore does not invent stage-specific
# payload fields.  Instead it makes the only accepted control flow explicit:
#
# detailed Hay/NEURON teacher -> verified official ELM twin
# -> verified 11-state distillation -> verified frozen runtime artifact.

export EXACT_DEVELOPMENT_CHAIN_STAGES

const EXACT_DEVELOPMENT_CHAIN_STAGES =
    (:detailed, :twin, :distill, :freeze)

function verify_development_scale_chain(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST_DEV1500;
    expected_contract_sha256::AbstractString=
        DEVELOPMENT_TEACHER_CONTRACT_SHA256_DEV1500,
    expected_manifest_sha256::AbstractString=
        DEVELOPMENT_TEACHER_MANIFEST_SHA256_DEV1500,
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
    official_v2_gate,
    distilled_cell_gate,
    frozen_runtime_gate,
)
    detailed = verify_development_teacher_manifest(
        manifest_path;
        expected_contract_sha256,
        expected_manifest_sha256,
        modeldb_root,
    )
    detailed.chain_complete === false ||
        error("detailed teacher unexpectedly claims a completed chain")
    detailed.downstream_artifact_gate === :not_bound ||
        error("detailed teacher already has an untrusted downstream binding")

    twin = _verified_downstream_gate(
        official_v2_gate(detailed),
        "official_v2_gate",
    )
    distill = _verified_downstream_gate(
        distilled_cell_gate(detailed, twin),
        "distilled_cell_gate",
    )
    freeze = _verified_downstream_gate(
        frozen_runtime_gate(detailed, twin, distill),
        "frozen_runtime_gate",
    )

    return merge(
        detailed,
        (;
            chain_complete=true,
            chain_stages=EXACT_DEVELOPMENT_CHAIN_STAGES,
            downstream_artifact_gate=:verified,
            detailed_teacher_artifact=detailed,
            official_v2_artifact=twin,
            distilled_cell_artifact=distill,
            frozen_runtime_artifact=freeze,
        ),
    )
end
