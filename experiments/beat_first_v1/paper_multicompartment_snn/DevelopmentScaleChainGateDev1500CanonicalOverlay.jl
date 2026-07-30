# Canonical development teacher identity after the NeuronIO window audit.
#
# NeuronIO discards the first 500 ms and consumes a subsequent 500 ms window.
# A 500 ms recording therefore cannot feed the official1278 ELM path.  The
# pinned artifact records 1500 ms; alternate explicitly content-addressed
# development artifacts must still contain at least the 1000 ms required
# burn-in plus retained window.

export DEVELOPMENT_TEACHER_MANIFEST_DEV1500,
    DEVELOPMENT_TEACHER_CONTRACT_SHA256_DEV1500,
    DEVELOPMENT_TEACHER_MANIFEST_SHA256_DEV1500

const DEVELOPMENT_TEACHER_MANIFEST_DEV1500 =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release\manifest.json"
const DEVELOPMENT_TEACHER_CONTRACT_SHA256_DEV1500 =
    "4ee32b8070c361084e5334f1d131e99680e2c53f1ac9234b6ea4810f78d5b320"
const DEVELOPMENT_TEACHER_MANIFEST_SHA256_DEV1500 =
    "5c0efd11a7c807235bd27601769e47447114616c32f135b7687513251de9e968"

function _assert_neuronio_development_window(report)
    report.duration_ms >= 1_000 ||
        error(
            "development teacher is too short for NeuronIO: 500 ms burn-in " *
            "plus a subsequent 500 ms retained window are required",
        )
    report.sample_dt_ms == 1.0 ||
        error("NeuronIO development chain requires 1 ms samples")
    report.time_steps >= 1_000 ||
        error("NeuronIO development chain has fewer than 1000 time steps")
    return report
end

function verify_development_teacher_manifest(
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
)
    report = _verify_teacher(
        manifest_path,
        :development;
        expected_contract_sha256,
        expected_manifest_sha256,
        modeldb_root,
    )
    return _assert_neuronio_development_window(report)
end

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
)
    teacher = verify_development_teacher_manifest(
        manifest_path;
        expected_contract_sha256,
        expected_manifest_sha256,
        modeldb_root,
    )
    twin = _verified_downstream_gate(
        official_v2_gate(teacher),
        "official_v2_gate",
    )
    cell = _verified_downstream_gate(
        distilled_cell_gate(teacher, twin),
        "distilled_cell_gate",
    )
    return merge(
        teacher,
        (;
            chain_complete=true,
            downstream_artifact_gate=:verified,
            official_v2_artifact=twin,
            distilled_cell_artifact=cell,
        ),
    )
end
