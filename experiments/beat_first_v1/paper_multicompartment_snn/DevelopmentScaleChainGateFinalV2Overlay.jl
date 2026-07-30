# Loaded into `DevelopmentScaleChainGate` by
# `LoadDevelopmentScaleChainGateFinalV2.jl`.
#
# The development artifact is content-addressed below, but its declared trial
# counts and duration are read from the verified final.v2 manifest.  This keeps
# development-scale evidence exact without silently turning one smoke duration
# into a protocol requirement.  The separate paper-production branch remains
# fixed at the paper-scale 50k/2k/10s contract.

export DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2,
    DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2,
    DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2

const DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2 =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev500_release\manifest.json"
const DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2 =
    "a97b1e04d37c34b58e1483c09e94adcdb888804d05faf759b3e0e99273a71bb0"
const DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2 =
    "3297063a1d4951093c094ebfe9fb5d63a21bb92a216bd94956d063b912ec22a6"

function _scale_profile(manifest, contract, profile::Symbol)
    config = _required(manifest, :config)
    train_trials = Int(_required(config, :train_trials))
    held_out_test_trials = Int(_required(config, :test_trials))
    validation_from_train_trials =
        Int(_required(config, :validation_trials_from_train))
    duration_ms = Int(_required(config, :duration_ms))
    sample_dt_ms = Float64(_required(config, :sample_dt_ms))
    completed_trials = Int(_required(manifest, :completed_trials))
    conflict = _required(contract, :connectivity_scale_conflict)
    fully_paper_scale_claim =
        _required(conflict, :fully_paper_scale_claim)
    interpretation_acknowledged =
        _required(conflict, :interpretation_explicitly_acknowledged)

    if profile === :development
        train_trials > 0 ||
            error("development rich64 teacher has no train trials")
        held_out_test_trials > 0 ||
            error("development rich64 teacher has no held-out trials")
        completed_trials == train_trials + held_out_test_trials ||
            error(
                "development rich64 completed trial count does not equal " *
                "train plus held-out trials",
            )
        duration_ms > 0 ||
            error("development rich64 trial duration must be positive")
        sample_dt_ms == 1.0 ||
            error("development rich64 teacher must use 1 ms sampling")
        isinteger(duration_ms / sample_dt_ms) ||
            error(
                "development rich64 duration must contain an integral number " *
                "of samples",
            )
        validation_from_train_trials > 0 ||
            error("development rich64 must derive validation trials")
        Int(_required(config, :axons)) == 64 ||
            error("development rich64 teacher must use 64 axons")
        String(_required(config, :preset)) == "smoke" ||
            error("development rich64 teacher must use the smoke preset")
        fully_paper_scale_claim === false ||
            error("development teacher falsely claims paper scale")
        interpretation_acknowledged === false ||
            error("development teacher unexpectedly acknowledges production")
        Bool(_required(
            config,
            :connectivity_interpretation_acknowledged,
        )) === false ||
            error("development teacher must not claim production connectivity")
        paper_scale = false
        promotable_production = false
    elseif profile === :paper_production
        (
            train_trials,
            held_out_test_trials,
            duration_ms,
            sample_dt_ms,
            completed_trials,
        ) == (50_000, 2_000, 10_000, 1.0, 52_000) ||
            error(
                "paper-scale production accepts only 50k train, 2k held-out, " *
                "10 s at 1 ms sampling, and 52k complete trials",
            )
        fully_paper_scale_claim === true ||
            error("paper-scale contract does not claim full paper scale")
        interpretation_acknowledged === true ||
            error("paper-scale connectivity interpretation is unacknowledged")
        Bool(_required(
            config,
            :connectivity_interpretation_acknowledged,
        )) === true ||
            error("paper-scale config has no connectivity acknowledgement")
        String(_required(config, :preset)) == "production" ||
            error("paper-scale teacher must use the production preset")
        paper_scale = true
        promotable_production = true
    else
        error("unknown scale profile $profile")
    end

    0 <= validation_from_train_trials < train_trials ||
        error("validation-from-train count is invalid")
    validation_indices = Int.(
        collect(_required(manifest, :validation_from_train_indices)),
    )
    expected_validation = collect(
        (train_trials - validation_from_train_trials + 1):train_trials,
    )
    validation_indices == expected_validation ||
        error("validation_from_train_indices are not the final train trials")

    return (;
        scale_profile=profile,
        train_trials,
        validation_from_train_trials,
        fit_trials=train_trials - validation_from_train_trials,
        held_out_test_trials,
        completed_trials,
        duration_ms,
        sample_dt_ms,
        time_steps=round(Int, duration_ms / sample_dt_ms),
        paper_scale,
        promotable_production,
    )
end

function verify_development_teacher_manifest(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2;
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
)
    return _verify_teacher(
        manifest_path,
        :development;
        expected_contract_sha256=
            DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2,
        expected_manifest_sha256=
            DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2,
        modeldb_root,
    )
end

function verify_development_scale_chain(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST_FINAL_V2;
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
