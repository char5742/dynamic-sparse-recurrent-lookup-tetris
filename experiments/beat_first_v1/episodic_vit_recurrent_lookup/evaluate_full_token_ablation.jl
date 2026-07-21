using JLD2
using JSON3
using Dates
using LinearAlgebra
using Random
using SHA
using Statistics

const HERE = @__DIR__
const PREACT_CHECKPOINT = raw"D:\tetris-paper-plus\checkpoints\beat_first_v1\teacherv3_preact_s_av2_b4_r1_best.jld2"
const DEFAULT_FULL_TOKEN_CHECKPOINT = raw"D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_full283_fixed2_u12000_20260721_r1\checkpoints\checkpoint_000012000.jls"
const FULL_TOKEN_CHECKPOINT = abspath(get(
    ENV, "EVRL_EVALUATION_CHECKPOINT", DEFAULT_FULL_TOKEN_CHECKPOINT,
))
const ROUTED_COMPARISON = joinpath(HERE, "performance_comparison_2026-07-20.json")
const OUTPUT_PATH = abspath(get(
    ENV,
    "EVRL_EVALUATION_OUTPUT",
    joinpath(HERE, "token_routing_ablation_2026-07-21.json"),
))
const EVALUATION_ID = get(
    ENV, "EVRL_EVALUATION_ID", "evrl-token-routing-ablation-2026-07-21",
)
const TARGET_ARCHITECTURE = get(
    ENV, "EVRL_EVALUATION_ARCHITECTURE", "evrl-full-283-token-cross-attention",
)
const DESIGN_ADDITION = strip(get(ENV, "EVRL_EVALUATION_DESIGN_ADDITION", ""))
const EVALUATION_REPEATS = 3

for (name, value) in (
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_FFN_DIM" => "128",
    "EVRL_FIXED_DEPTH" => "2",
)
    ENV[name] = value
end

BLAS.set_num_threads(1)
Base.include(Main, joinpath(HERE, "teacher_training.jl"))
const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const ParentTraining = Training.ParentTraining
const TrainingCore = Training.TrainingCore

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function metric_record(metrics)
    return (;
        states=Int(metrics.states),
        composite_loss=Float64(metrics.composite_loss),
        listnet_loss=Float64(metrics.listnet_loss),
        old_q_loss=Float64(metrics.old_q_loss),
        margin_loss=Float64(metrics.margin_loss),
        top1_agreement=Float64(metrics.top1_agreement),
        ndcg=Float64(metrics.ndcg),
        pairwise_accuracy=Float64(metrics.pairwise_accuracy),
        action_margin=Float64(metrics.action_margin),
        q_mean=Float64(metrics.q_mean),
        q_std=Float64(metrics.q_std),
    )
end

function benchmark_predictor(dataset, rows, host_batch, predictor; repeats::Int)
    first_row = first(rows)
    TrainingCore.pack_batch!(host_batch, dataset, [first_row])
    warm = predictor(host_batch)
    checksum = sum(Float64, Array(warm.q))
    GC.gc()
    started = time_ns()
    candidates = 0
    for _ in 1:repeats, row in rows
        TrainingCore.pack_batch!(host_batch, dataset, [row])
        output = predictor(host_batch)
        checksum += sum(Float64, Array(output.q))
        candidates += Int(dataset.action_counts[row])
    end
    seconds = Float64(time_ns() - started) * 1.0e-9
    states = repeats * length(rows)
    return (;
        repeats,
        states,
        candidates,
        seconds,
        states_per_second=states / seconds,
        candidates_per_second=candidates / seconds,
        checksum,
    )
end

function difference(left, right)
    return (;
        top1=left.metrics.top1_agreement - right.metrics.top1_agreement,
        ndcg=left.metrics.ndcg - right.metrics.ndcg,
        pairwise=left.metrics.pairwise_accuracy - right.metrics.pairwise_accuracy,
        action_margin=left.metrics.action_margin - right.metrics.action_margin,
        composite_loss=left.metrics.composite_loss - right.metrics.composite_loss,
        inference_speed_ratio=
            left.inference.states_per_second / right.inference.states_per_second,
    )
end

function main()
    routed = JSON3.read(read(ROUTED_COMPARISON, String))
    preact = jldopen(PREACT_CHECKPOINT, "r") do file
        (;
            config=file["config"],
            update=Int(file["update"]),
        )
    end
    dataset_path = String(preact.config.dataset_path)
    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    split = ParentTraining.episode_separated_split(
        dataset;
        seed=UInt64(preact.config.split_seed),
        validation_fraction=0.10,
    )
    split.training_groups == preact.config.training_groups ||
        error("PreAct training split differs")
    split.validation_groups == preact.config.validation_groups ||
        error("PreAct validation split differs")
    rows = ParentTraining.fixed_evaluation_subset(
        split.validation_rows,
        128,
        UInt64(preact.config.seed) + UInt64(0x202),
    )
    rows_sha256 = bytes2hex(sha256(reinterpret(UInt8, rows)))
    rows_sha256 == String(routed.conditions.validation_rows_sha256) ||
        error("held-teacher row panel differs from routed comparison")
    manifest_sha256 = sha256_file(joinpath(dataset_path, "manifest.json"))
    manifest_sha256 == String(routed.conditions.dataset_manifest_sha256) ||
        error("teacher dataset manifest differs")

    payload, artifact = Training.read_checkpoint(FULL_TOKEN_CHECKPOINT)
    hyperparameters = Training.runtime_hyperparameters(Int(payload.update), payload)
    trainer, _, _ = Training.restore_checkpoint(
        payload,
        split,
        payload.split_metadata,
        manifest_sha256,
        hyperparameters,
    )
    host_batch = TrainingCore.allocate_host_batch(
        1; max_candidates=Training.LEARNER_WIDTH,
    )
    depths = Int[]
    predictor = batch -> begin
        raw, count = Training.predict_raw!(
            trainer,
            batch;
            training=false,
            expected_update=trainer.update,
            hyperparameters,
            record_tapes=false,
        )
        append!(depths, Int.(view(trainer.workspace.depths, 1:count)))
        Training.raw_output(raw)
    end
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        rows,
        host_batch,
        predictor;
        margin_weight=TrainingCore.MARGIN_WEIGHT,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    depth_record = (;
        mean=mean(depths),
        minimum=minimum(depths),
        maximum=maximum(depths),
    )
    empty!(depths)
    inference = benchmark_predictor(
        dataset,
        rows,
        host_batch,
        predictor;
        repeats=EVALUATION_REPEATS,
    )
    full_token = (;
        architecture=TARGET_ARCHITECTURE,
        checkpoint=artifact,
        update=Int(payload.update),
        consumed_states=Int(payload.consumed_states),
        parameter_count=Int(payload.config.total_parameter_count),
        metrics=metric_record(metrics),
        recurrent_depth=depth_record,
        inference,
    )

    old = routed.evrl_budget_matched
    baseline = routed.preact
    design_change = if isempty(DESIGN_ADDITION)
        (;
            removed="283-to-64 learned hash/WTA retrieval plus exact top-16 shortlist",
            replacement="all 283 token shared K/V exact register cross-attention",
            unchanged=(
                "input contract",
                "local 8-neighbour spatial attention",
                "register self-attention and SwiGLU",
                "LookupFFN parameter routing and active-only updates",
                "shared recurrence and fixed depth two evaluation setting",
                "teacher objective and optimizer semantics",
            ),
        )
    else
        (;
            added=DESIGN_ADDITION,
            unchanged=(
                "input contract and candidate-independent evaluation",
                "full 283-token register cross-attention",
                "local 8-neighbour recurrent cell attention",
                "LookupFFN parameter routing and active-only updates",
                "shared recurrence and fixed depth two evaluation setting",
                "teacher objective and optimizer semantics",
            ),
        )
    end
    result = (;
        evaluation_id=EVALUATION_ID,
        generated_at=string(Dates.now()),
        design_change,
        conditions=(;
            dataset_path,
            dataset_manifest_sha256=manifest_sha256,
            split_seed=UInt64(preact.config.split_seed),
            validation_subset_seed=UInt64(preact.config.seed) + UInt64(0x202),
            validation_states=length(rows),
            validation_rows_sha256=rows_sha256,
            evaluation_repeats=EVALUATION_REPEATS,
            blas_threads=BLAS.get_num_threads(),
            julia_threads=Base.Threads.nthreads(:default),
            teacher_q_and_rank_used_only_as_targets=true,
            candidate_independent_evaluation=true,
            game_validation_seeds_touched=false,
            game_sealed_seeds_touched=false,
        ),
        preact=baseline,
        routed_evrl=old,
        full_token_evrl=full_token,
        differences=(;
            full_token_vs_routed=difference(full_token, old),
            full_token_vs_preact=difference(full_token, baseline),
        ),
    )
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, result)
        write(io, '\n')
    end
    JSON3.pretty(stdout, result)
    println()
    return result
end

main()
