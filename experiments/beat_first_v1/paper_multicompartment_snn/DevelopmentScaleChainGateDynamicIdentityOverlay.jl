# Loaded after `DevelopmentScaleChainGateFinalV2Overlay.jl`.
#
# The artifact identity remains explicit and content-addressed even when a
# caller selects another verified final.v2 development artifact.  This avoids
# both an implicit trust widening and the old coupling to one fixed duration.

function verify_development_teacher_manifest(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2;
    expected_contract_sha256::AbstractString=
        DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2,
    expected_manifest_sha256::AbstractString=
        DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2,
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
)
    return _verify_teacher(
        manifest_path,
        :development;
        expected_contract_sha256,
        expected_manifest_sha256,
        modeldb_root,
    )
end

function verify_development_scale_chain(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2;
    expected_contract_sha256::AbstractString=
        DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2,
    expected_manifest_sha256::AbstractString=
        DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2,
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
