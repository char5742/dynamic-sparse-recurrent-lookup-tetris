using Dates
using JSON3
using Random
using SHA

include(joinpath(@__DIR__, "contract.jl"))
using .OnlineCounterfactualTop2R1Contract

const TRAINING_SCHEDULE_SHA256 =
    "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
const CALIBRATION_SCHEDULE_SHA256 =
    "c08341a6891997301f112912a4fe969c5493603834cc3dbc114b3e24401617db"

function schedule_digest(schedules)
    payload = join((join(schedule, ",") for schedule in schedules), ";")
    return bytes2hex(sha256(payload))
end

function cluster_schedules(seed_ids::Vector{Int}, count::Int, seed::UInt64)
    isempty(seed_ids) && error("cannot bootstrap an empty episode role")
    rng = Xoshiro(seed)
    return [[seed_ids[rand(rng, eachindex(seed_ids))] for _ in eachindex(seed_ids)] for _ in 1:count]
end

function validate_eligibility(eligibility, contract)
    eligibility.status == "r1_design_eligibility_complete" || error("eligibility status mismatch")
    eligibility.experiment == contract.experiment_id || error("eligibility experiment mismatch")
    eligibility.contract_sha256 == hex_sha256(CONTRACT_PATH) || error("eligibility contract hash mismatch")
    Int.(eligibility.training_seed_ids) == Int.(contract.data_roles.training_seeds) || error("training seed role mismatch")
    Int.(eligibility.calibration_seed_ids) == Int.(contract.data_roles.calibration_seeds) || error("calibration seed role mismatch")
    Int.(eligibility.sample_piece_indices) == Int.(contract.data_roles.sample_pieces) || error("sample schedule mismatch")
    eligibility.planned_training_states == 288 || error("planned training cardinality mismatch")
    eligibility.planned_calibration_states == 144 || error("planned calibration cardinality mismatch")
    eligibility.minimum_training_states == 240 || error("training minimum mismatch")
    eligibility.minimum_calibration_states == 120 || error("calibration minimum mismatch")
    for name in (
        :real_data_loaded, :model_or_checkpoint_loaded, :game_run,
        :development_seed_loaded, :validation_seed_loaded, :sealed_test_seed_loaded,
        :existing_c10_c13_q1_dataset_loaded,
    )
        getproperty(eligibility, name) === false || error("forbidden pre-freeze activity: $name")
    end
    return true
end

function frozen_design(eligibility, eligibility_path::AbstractString)
    contract = load_contract()
    validate_eligibility(eligibility, contract)
    training_seeds = Int.(contract.data_roles.training_seeds)
    calibration_seeds = Int.(contract.data_roles.calibration_seeds)
    training_schedules = cluster_schedules(
        training_seeds,
        Int(contract.fit.ensemble_count),
        UInt64(contract.fit.bootstrap_seed_uint64),
    )
    calibration_schedules = cluster_schedules(
        calibration_seeds,
        Int(contract.calibration_gate.cluster_bootstrap_count),
        UInt64(contract.calibration_gate.cluster_bootstrap_seed_uint64),
    )
    training_digest = schedule_digest(training_schedules)
    calibration_digest = schedule_digest(calibration_schedules)
    training_digest == TRAINING_SCHEDULE_SHA256 || error(
        "training bootstrap schedule differs from the independent source anchor",
    )
    calibration_digest == CALIBRATION_SCHEDULE_SHA256 || error(
        "calibration bootstrap schedule differs from the independent source anchor",
    )
    return (;
        status="r1_design_frozen",
        generated_at=string(now(UTC)),
        experiment=String(contract.experiment_id),
        contract_path=abspath(CONTRACT_PATH),
        contract_sha256=hex_sha256(CONTRACT_PATH),
        eligibility_path=abspath(eligibility_path),
        eligibility_sha256=hex_sha256(eligibility_path),
        feature_schema_version=String(contract.feature_schema.version),
        feature_names=String.(contract.feature_schema.names),
        feature_names_sha256=String(contract.feature_schema.feature_names_sha256),
        feature_count=Int(contract.feature_schema.feature_count),
        coefficient_count=Int(contract.feature_schema.coefficient_count),
        canonical_policy=contract.canonical_policy,
        openvino_backend=contract.openvino_backend,
        immutable_inputs=contract.immutable_inputs,
        training_seed_ids=training_seeds,
        calibration_seed_ids=calibration_seeds,
        sample_piece_indices=Int.(contract.data_roles.sample_pieces),
        training_bootstrap_rng=String(contract.fit.bootstrap_rng),
        training_bootstrap_schedules=training_schedules,
        training_bootstrap_schedule_sha256=training_digest,
        calibration_bootstrap_rng=String(contract.calibration_gate.cluster_bootstrap_rng),
        calibration_bootstrap_schedules=calibration_schedules,
        calibration_bootstrap_schedule_sha256=calibration_digest,
        ridge_lambda=Float64(contract.fit.ridge_lambda),
        prediction_lower_quantile=Float64(contract.fit.prediction_lower_quantile),
        override_strict_threshold=Float64(contract.fit.override_strict_threshold),
        hyperparameter_sweep_authorized=false,
        model_or_checkpoint_loaded=false,
        game_run=false,
        development_seed_loaded=false,
        validation_seed_loaded=false,
        sealed_test_seed_loaded=false,
    )
end

function self_check()
    contract = load_contract()
    train = Int.(contract.data_roles.training_seeds)
    first = cluster_schedules(train, 256, UInt64(contract.fit.bootstrap_seed_uint64))
    second = cluster_schedules(train, 256, UInt64(contract.fit.bootstrap_seed_uint64))
    first == second || error("training cluster schedule is not deterministic")
    length(first) == 256 || error("training ensemble count mismatch")
    all(length(schedule) == 12 for schedule in first) || error("training cluster width mismatch")
    all(all(in(train), schedule) for schedule in first) || error("training cluster escaped seed role")
    schedule_digest(first) == TRAINING_SCHEDULE_SHA256 || error(
        "training schedule source anchor mismatch",
    )
    contract_calibration = Int.(contract.data_roles.calibration_seeds)
    calibration = cluster_schedules(
        contract_calibration,
        2000,
        UInt64(contract.calibration_gate.cluster_bootstrap_seed_uint64),
    )
    schedule_digest(calibration) == CALIBRATION_SCHEDULE_SHA256 || error(
        "calibration schedule source anchor mismatch",
    )
    return true
end

function main(args=ARGS)
    if args == ["--self-check"] || isempty(args)
        self_check()
        return
    end
    length(args) == 2 || error("usage: freeze_design.jl ELIGIBILITY_JSON OUTPUT_JSON")
    eligibility_path, output_path = abspath.(args)
    isfile(eligibility_path) || error("missing eligibility artifact")
    eligibility = JSON3.read(read(eligibility_path, String))
    atomic_write_json(output_path, frozen_design(eligibility, eligibility_path))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
