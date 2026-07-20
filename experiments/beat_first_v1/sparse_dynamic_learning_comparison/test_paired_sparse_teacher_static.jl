using Test
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CONTRACT_PATH = joinpath(@__DIR__, "paired_100_update_contract.toml")
const WRAPPER_PATH = joinpath(@__DIR__, "run_paired_sparse_teacher.ps1")
const ONE_TRAINER = joinpath(ROOT, "sparse_dynamic", "sparse_training.jl")
const THREE_TRAINER = joinpath(ROOT, "sparse_dynamic_3layer", "teacher_training.jl")
const ONE_OPTIMIZER = joinpath(ROOT, "sparse_dynamic", "sparse_optimizer.jl")
const THREE_OPTIMIZER = joinpath(ROOT, "sparse_dynamic_3layer", "optimizer.jl")

contract = TOML.parsefile(CONTRACT_PATH)
wrapper = read(WRAPPER_PATH, String)
one = read(ONE_TRAINER, String)
three = read(THREE_TRAINER, String)
one_optimizer = read(ONE_OPTIMIZER, String)
three_optimizer = read(THREE_OPTIMIZER, String)

function required_position(needle::AbstractString, haystack::AbstractString)
    found = findfirst(needle, haystack)
    found === nothing && error("missing static token: $needle")
    return first(found)
end

@testset "paired sparse 100-update static contract" begin
    shared = contract["shared"]
    @test contract["architecture_only_attribution"] === false
    @test shared["total_parameters"] == 19_924_022
    @test shared["state_batch"] == 1
    @test shared["learner_candidate_width"] == 80
    @test shared["maximum_updates"] == 100
    @test shared["evaluation_interval"] == 25
    @test shared["checkpoint_interval"] == 100
    @test shared["model_seed"] == 2026071901
    @test shared["split_seed"] == 2026071902
    @test shared["sampler_seed"] == 2026071903
    @test shared["learning_rate"] == 1.0e-4
    @test shared["bank_weight_decay"] == 0.0
    @test shared["exclusive_julia_mutex"] == "Local\\TetrisBeatFirstV1ExclusiveJulia"
    @test shared["overlap_poll_milliseconds"] == 100

    # Both trainers call the literal same supervised objective implementation.
    @test occursin("training\", \"core.jl", one)
    @test occursin("training\", \"core.jl", three)
    @test occursin("supervised_components(raw_output(candidate_outputs), batch)", one)
    @test occursin("margin_weight=trainer.objective_margin_weight", three)
    for trainer_source in (one, three)
        @test occursin("Xoshiro(UInt64(model_seed))", trainer_source)
        @test occursin("Xoshiro(UInt64(config.sampler_seed))", trainer_source)
        @test occursin("UInt64(config.split_seed) + UInt64(101)", trainer_source)
        @test occursin("UInt64(config.split_seed) + UInt64(202)", trainer_source)
        @test occursin("only(next_batch!(sampler, 1))", trainer_source)
        @test occursin("pack_batch!(host_batch, dataset, [row])", trainer_source)
    end

    # Storage stays immutable at 208 while both learner tensors are width 80.
    @test occursin("const LEARNER_WIDTH = 80", one)
    @test occursin("load_teacher_dataset(dataset_path; max_candidates=MAX_CANDIDATES)", one)
    @test occursin("allocate_host_batch(1; max_candidates=config.candidate_width)", one)
    @test occursin("const LEARNER_WIDTH = 80", three)
    @test occursin("max_candidates=MAX_CANDIDATES", three)
    @test occursin("allocate_host_batch(1; max_candidates=LEARNER_WIDTH)", three)

    # The wrapper launches one arm only and refuses any pre-existing Julia process.
    @test occursin("ValidateSet(\"one_layer\", \"three_layer\")", wrapper)
    @test occursin("Get-Process -Name \"julia\"", wrapper)
    @test occursin("BEAT_SPARSE_PAIRING_CONTRACT_SHA256", wrapper)
    @test occursin("Local\\TetrisBeatFirstV1ExclusiveJulia", wrapper)
    @test occursin("WaitOne(0)", wrapper)
    @test occursin("Start-Process", wrapper)
    @test occursin("WaitForExit(100)", wrapper)
    @test occursin("ownedJulia.Kill()", wrapper)
    @test occursin("ReleaseMutex()", wrapper)
    lease_position = required_position("WaitOne(0)", wrapper)
    process_check_position = required_position("Get-Process -Name \"julia\"", wrapper)
    launch_position = required_position("Start-Process", wrapper)
    release_position = required_position("ReleaseMutex()", wrapper)
    @test lease_position < process_check_position < launch_position < release_position
    @test length(findall("Get-Process -Name \"julia\"", wrapper)) == 3
    for literal in (
        "2026071901", "2026071902", "2026071903", "MAX_UPDATES = \"100\"",
        "EVAL_INTERVAL = \"25\"", "CHECKPOINT_INTERVAL = \"100\"",
        "TRAIN_EVAL_STATES = \"16\"", "VALIDATION_EVAL_STATES = \"32\"",
    )
        @test length(findall(literal, wrapper)) == 2
    end
    @test length(findall("0.0001", wrapper)) == 3 # 1L bank/head plus 3L shared LR.
    @test length(findall("WEIGHT_DECAY = \"0.0\"", wrapper)) == 2
    @test length(findall("BEAT_ALLOW_PARTIAL_DATASET = \"false\"", wrapper)) == 1
    @test occursin("pairing_contract_sha256=strip(get(", one)
    @test occursin("pairing_contract_sha256=strip(get(", three)
    @test occursin("BEAT_SPARSE_HEAD_BETA1 = \"0.9\"", wrapper)
    @test occursin("BEAT_SPARSE_HEAD_BETA2 = \"0.999\"", wrapper)
    @test occursin("BEAT_SPARSE_HEAD_EPSILON = \"1.0e-8\"", wrapper)
    @test occursin("head_beta1=_env_float(\"BEAT_SPARSE_HEAD_BETA1\"", one)
    @test occursin("head_beta2=_env_float(\"BEAT_SPARSE_HEAD_BETA2\"", one)
    @test occursin("head_epsilon=_env_float(\"BEAT_SPARSE_HEAD_EPSILON\"", one)
    for resume_field in (
        ":epochs,", ":maximum_updates,", ":eval_interval,",
        ":checkpoint_interval,", ":training_eval_states,",
        ":validation_eval_states,", ":evaluate_initial,", ":validation_fraction,",
    )
        @test occursin(resume_field, one)
    end

    # Existing last-step metrics already carry the sparse-work evidence; no
    # additional timed instrumentation is introduced by this comparison.
    for literal in (
        "unique_active_rows", "active_parameter_touches", "executed_linear_macs",
        "optimizer_rows_updated", "bank_coverage=bank_coverage_metrics",
    )
        @test occursin(literal, one)
    end
    for literal in (
        "unique_active_rows", "active_parameter_touches", "active_edge_touches",
        "routing_inclusive_forward_macs", "executed_training_macs",
        "moment_elements_written", "bank_coverage=bank_coverage_metrics",
    )
        @test occursin(literal, three)
    end

    # The sole optimizer confound is explicit rather than silently attributed to depth.
    differences = contract["remaining_difference"]
    @test differences[1]["name"] == "sparse_bank_optimizer_family"
    @test differences[1]["material"] === true
    @test occursin("init_sparse_adagradw", one_optimizer)
    @test occursin("init_eventtime_adamw", three_optimizer)
    @test occursin("architecture", lowercase(differences[1]["reason"]))

    combined = join((wrapper, read(CONTRACT_PATH, String)), '\n')
    @test !occursin("8001", combined)
    @test !occursin("91001", combined)
end
