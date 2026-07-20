using Test
using Random

include(joinpath(@__DIR__, "teacher_training.jl"))

const CBP = BeatFirstThreeLayerTeacherTraining
const CBS = Main.SparseDynamic3Layer
const CBO = CBS.MongooseSimHashOverlay
const CBW = CBS.WTALSHIndex

const CUTOFF_BOUNDARY_SERIALIZED_TESTS = (
    :frozen_v2_compute_and_learner_contract,
    :raw_score_then_stable_id_tie_ordering,
    :exact_k_kplus1_boundary,
    :fill_and_probe_exclusion,
    :missing_outside_candidate_rejection,
    :bounded_retrieved_scored_witness_only,
    :legacy_easy_extrema_unchanged,
    :checkpoint_cross_mode_rejection,
)

function _cutoff_scratch(
    retrieved::Vector{Int32};
    score_by_id::Dict{Int,Float64},
    natural_ids::Set{Int},
    neurons::Int=max(maximum(Int.(retrieved)), maximum(keys(score_by_id))),
)
    generation = UInt64(7)
    marks = zeros(UInt64, neurons)
    collisions = zeros(UInt16, neurons)
    scores = zeros(Float64, neurons)
    for neuron in retrieved
        id = Int(neuron)
        marks[id] = generation
        collisions[id] = id in natural_ids ? UInt16(1) : UInt16(0)
        scores[id] = score_by_id[id]
    end
    for (id, score) in score_by_id
        scores[id] = score
    end
    return CBW.WTAQueryScratch(
        generation,
        marks,
        collisions,
        scores,
        copy(retrieved),
        length(retrieved),
        length(retrieved),
        length(retrieved),
        0,
    )
end

function _metadata_config()
    digest = repeat("0", 64)
    base = (;
        variant=:k128,
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        routing_mode=CBP.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        routing_policy=CBP.MONGOOSE_V2_ROUTING_POLICY,
        mongoose_route_dim=CBO.ROUTE_DIM,
        mongoose_bits_per_table=CBO.BITS_PER_TABLE,
        mongoose_tables=CBO.TABLES,
        mongoose_router_parameters=CBO.ROUTER_PARAMETERS,
        mongoose_column_normalization=CBO.COLUMN_NORMALIZATION,
        mongoose_learning_rate=1.0e-4,
        mongoose_beta=1.0,
        mongoose_seed=0x4d4f4e47,
        mongoose_warmup_updates=2_000,
        mongoose_refresh_interval=2_000,
        objective_margin_weight=Float64(CBP.BeatFirstTrainingCore.MARGIN_WEIGHT),
        objective_margin_mode=
            CBP.BeatFirstTrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=
            CBP.BeatFirstTrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
    )
    return merge(base, CBP._mongoose_v2_policy_config())
end

@testset "cutoff arm preserves the frozen v2 compute and learner contract" begin
    @test length(CUTOFF_BOUNDARY_SERIALIZED_TESTS) == 8
    @test allunique(CUTOFF_BOUNDARY_SERIALIZED_TESTS)
    spec = CBP.named_variant_spec(:k128)
    policy = CBP._mongoose_v2_policy_config()
    @test spec.active_counts == (48, 40, 40)
    @test spec.training_probes == (6, 5, 5)
    @test CBO.ROUTER_PARAMETERS == 2_688
    @test CBP.LAYER_MAX_BUCKET_ENTRIES == (1_536, 1_280, 1_280)
    @test CBP.LAYER_MAX_SCORED_ROWS == (384, 640, 640)
    @test policy.mongoose_v2_lane_entry_caps == (96, 80, 80)
    @test CBP.BeatFirstTrainingCore.MARGIN_WEIGHT == 0.15f0
    @test isdefined(CBO, :pair_bce_gradient!)
end

@testset "MONGOOSE v2 cutoff uses exact raw-score/stable-ID boundary" begin
    # IDs 3 and 5 tie at the boundary; stable ID 3 must be inside.
    scratch = _cutoff_scratch(
        Int32[5, 7, 2, 3];
        score_by_id=Dict(7 => 10.0, 2 => 9.0, 3 => 8.0, 5 => 8.0),
        natural_ids=Set((2, 3, 5, 7)),
    )
    staging = Int32[]
    pair = CBP._stable_exploitation_cutoff_pair!(staging, scratch, 3)
    @test staging == Int32[7, 2, 3, 5]
    @test pair.positive == 3
    @test pair.negative == 5
    @test pair.positive_score == 8.0
    @test pair.negative_score == 8.0
    @test pair.positive_collisions == 1
    @test pair.negative_collisions == 1
    @test pair.cutoff_score_gap == 0.0
    @test pair.positive_rank == 3
    @test pair.negative_rank == 4
    @test pair.positive_in_exploitation
    @test !pair.negative_in_exploitation
end

@testset "fills and probes cannot enter either cutoff side" begin
    # Collision-zero IDs model both deterministic underflow fills and reserved
    # probes. Their larger raw scores must not alter the natural boundary.
    scratch = _cutoff_scratch(
        Int32[1, 10, 2, 20, 30];
        score_by_id=Dict(1 => 100.0, 2 => 99.0, 10 => 5.0, 20 => 4.0, 30 => 3.0),
        natural_ids=Set((10, 20, 30)),
    )
    pair = CBP._stable_exploitation_cutoff_pair!(Int32[], scratch, 2)
    @test pair.positive == 20
    @test pair.negative == 30
    @test pair.positive_collisions == 1
    @test pair.negative_collisions == 1
    @test pair.eligible_rows == 3
    @test !(pair.positive in (1, 2))
    @test !(pair.negative in (1, 2))
end

@testset "cutoff fails closed without a bounded outside natural row" begin
    scratch = _cutoff_scratch(
        Int32[1, 2, 3];
        score_by_id=Dict(1 => 9.0, 2 => 8.0, 3 => 100.0),
        natural_ids=Set((1, 2)),
    )
    @test_throws ErrorException CBP._stable_exploitation_cutoff_pair!(
        Int32[], scratch, 2,
    )
end

@testset "cutoff observes only bounded retrieved and exact-scored witnesses" begin
    # ID 9 has the largest bank-side score but was never retrieved. It must be
    # invisible without a bank scan.
    scratch = _cutoff_scratch(
        Int32[1, 2, 3];
        score_by_id=Dict(1 => 5.0, 2 => 4.0, 3 => 3.0, 9 => 1_000.0),
        natural_ids=Set((1, 2, 3)),
        neurons=9,
    )
    pair = CBP._stable_exploitation_cutoff_pair!(Int32[], scratch, 2)
    @test pair.positive == 2
    @test pair.negative == 3
    @test !(9 in Int.(scratch.retrieved))
end

@testset "legacy easy-extrema miner remains unchanged" begin
    scratch = _cutoff_scratch(
        Int32[4, 2, 7, 5];
        score_by_id=Dict(2 => 10.0, 4 => 10.0, 5 => -3.0, 7 => -3.0),
        natural_ids=Set((2, 4)),
    )
    positive, negative, positive_score, negative_score =
        CBP._stable_score_extrema(scratch)
    @test positive == 2
    @test negative == 5
    @test positive_score == 10.0
    @test negative_score == -3.0
end

@testset "cutoff mining has a distinct checkpoint metadata identity" begin
    easy = _metadata_config()
    cutoff = merge(easy, CBP._mongoose_v2_cutoff_pair_policy_config())
    @test CBP._config_mongoose_pair_mining_mode(easy) ===
        CBP.MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE
    @test CBP._config_mongoose_pair_mining_mode(cutoff) ===
        CBP.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
    @test cutoff.mongoose_pair_mining_exploitation_targets == (42, 35, 35)
    @test cutoff.mongoose_pair_mining_no_bank_scan === true
    @test cutoff.mongoose_pair_mining_fail_closed === true

    easy_metadata = CBP._checkpoint_metadata(easy)
    cutoff_metadata = CBP._checkpoint_metadata(cutoff)
    @test !haskey(easy_metadata, "mongoose_pair_mining_identity")
    @test cutoff_metadata["mongoose_pair_mining_identity"] ==
        CBP.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_IDENTITY
    @test_throws ErrorException CBP._validate_checkpoint_metadata(
        easy_metadata, cutoff,
    )
    @test_throws ErrorException CBP._validate_checkpoint_metadata(
        cutoff_metadata, easy,
    )

    model = CBS.initialize_model(
        Xoshiro(0x4355544f46465041);
        neuron_counts=(64, 64, 64),
        active_counts=(48, 40, 40),
    )
    runtime = CBS.initialize_runtime(model; weight_decay=0.0f0)
    state = CBO.initialize_v2_overlay(
        ntuple(i -> runtime.indexes[i].positions, 3);
        warmup_updates=2,
        refresh_interval=2,
        seed=0x4d4f4e47,
    )
    easy_trainer = CBP._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=CBP.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    cutoff_trainer = CBP._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=CBP.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        mongoose_pair_mining_mode=
            CBP.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE,
    )
    easy_bounded = merge(easy, (;
        variant=:bounded_test,
        bounded_test_contract=true,
        learner_width=1,
        mongoose_warmup_updates=2,
        mongoose_refresh_interval=2,
    ))
    cutoff_bounded = merge(easy_bounded, CBP._mongoose_v2_cutoff_pair_policy_config())
    @test CBP._assert_overlay_checkpoint_state!(
        easy_trainer,
        easy_bounded,
        0;
        _bounded_test_contract=true,
    )
    @test CBP._assert_overlay_checkpoint_state!(
        cutoff_trainer,
        cutoff_bounded,
        0;
        _bounded_test_contract=true,
    )
    @test_throws ErrorException CBP._assert_overlay_checkpoint_state!(
        cutoff_trainer,
        easy_bounded,
        0;
        _bounded_test_contract=true,
    )
    @test_throws ErrorException CBP._assert_overlay_checkpoint_state!(
        easy_trainer,
        cutoff_bounded,
        0;
        _bounded_test_contract=true,
    )
end
