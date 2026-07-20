using JLD2
using JSON3
using Dates
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

const HERE = @__DIR__
const REPO_ROOT = normpath(joinpath(HERE, "..", "..", ".."))
const PREACT_CHECKPOINT = raw"D:\tetris-paper-plus\checkpoints\beat_first_v1\teacherv3_preact_s_av2_b4_r1_best.jld2"
const EVRL_RUN = raw"D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_spatial_lookup_fixed2_u20000_resume2000_20260720_r1"
const EVRL_CHECKPOINTS = (
    joinpath(EVRL_RUN, "checkpoints", "checkpoint_000012000.jls"),
    joinpath(EVRL_RUN, "checkpoints", "checkpoint_000020000.jls"),
)
const OUTPUT_PATH = joinpath(HERE, "performance_comparison_2026-07-20.json")
const EVALUATION_REPEATS = 3

# Checkpoint deserialization requires the live modules to be instantiated with
# exactly the geometry recorded by the final production run.
const EVRL_GEOMETRY_ENV = (
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_ROUTER_TABLES" => "2",
    "EVRL_ROUTER_BITS" => "4",
    "EVRL_ROUTER_BUCKET_CAP" => "64",
    "EVRL_EPISODIC_SHORTLIST" => "16",
    "EVRL_EPISODIC_CANDIDATE_CAP" => "64",
    "EVRL_CANDIDATE_SUPPORT_CAP" => "4",
    "EVRL_SPATIAL_ANCHORS" => "2",
    "EVRL_SPATIAL_SHORTLIST" => "2",
    "EVRL_SPATIAL_CANDIDATE_CAP" => "3",
    "EVRL_FFN_DIM" => "128",
)
for (name, value) in EVRL_GEOMETRY_ENV
    ENV[name] = value
end

BLAS.set_num_threads(1)
Base.include(Main, joinpath(HERE, "teacher_training.jl"))
const EVRLTraining = Main.EpisodicViTRecurrentLookupTeacherTraining
const ParentTraining = EVRLTraining.ParentTraining
const TrainingCore = EVRLTraining.TrainingCore

Base.include(Main, joinpath(HERE, "..", "models", "models.jl"))
const PreActModels = Main.BeatFirstModels

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function read_preact_checkpoint(path::AbstractString)
    isfile(path) || error("PreAct checkpoint does not exist: $path")
    return jldopen(path, "r") do file
        (;
            path=abspath(path),
            sha256=sha256_file(path),
            update=Int(file["update"]),
            ps=file["ps"],
            st=file["st"],
            config=file["config"],
            saved_metrics=file["metrics"],
        )
    end
end

function benchmark_predictor(dataset, rows, host_batch, predictor; repeats::Int)
    batch_size = size(host_batch.mask, 2)
    usable = length(rows) - mod(length(rows), batch_size)
    selected = @view rows[1:usable]
    first_partition = first(Iterators.partition(selected, batch_size))
    TrainingCore.pack_batch!(host_batch, dataset, collect(first_partition))
    warm_output = predictor(host_batch)
    checksum = sum(Float64, Array(warm_output.q))
    GC.gc()
    started = time_ns()
    candidates = 0
    for _ in 1:repeats
        for partition in Iterators.partition(selected, batch_size)
            packed_rows = collect(partition)
            TrainingCore.pack_batch!(host_batch, dataset, packed_rows)
            output = predictor(host_batch)
            checksum += sum(Float64, Array(output.q))
            candidates += sum(Int(dataset.action_counts[row]) for row in packed_rows)
        end
    end
    seconds = Float64(time_ns() - started) * 1.0e-9
    states = repeats * usable
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

function evaluate_evrl(path, dataset, split, validation_rows, dataset_manifest_sha256)
    payload, artifact = EVRLTraining.read_checkpoint(path)
    hyperparameters = EVRLTraining.runtime_hyperparameters(Int(payload.update), payload)
    trainer, _, _ = EVRLTraining.restore_checkpoint(
        payload,
        split,
        payload.split_metadata,
        dataset_manifest_sha256,
        hyperparameters,
    )
    host_batch = TrainingCore.allocate_host_batch(1; max_candidates=EVRLTraining.LEARNER_WIDTH)
    depths = Int[]
    predictor = batch -> begin
        raw, count = EVRLTraining.predict_raw!(
            trainer,
            batch;
            training=false,
            expected_update=trainer.update,
            hyperparameters,
            record_tapes=false,
        )
        append!(depths, Int.(view(trainer.workspace.depths, 1:count)))
        EVRLTraining.raw_output(raw)
    end
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        validation_rows,
        host_batch,
        predictor;
        margin_weight=TrainingCore.MARGIN_WEIGHT,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    empty!(depths)
    speed = benchmark_predictor(
        dataset,
        validation_rows,
        host_batch,
        predictor;
        repeats=EVALUATION_REPEATS,
    )
    return (;
        architecture="episodic_vit_recurrent_lookup",
        checkpoint=artifact,
        update=Int(payload.update),
        consumed_states=Int(payload.consumed_states),
        parameter_count=Int(payload.config.total_parameter_count),
        metrics=metric_record(metrics),
        recurrent_depth=(;
            mean=mean(depths),
            minimum=minimum(depths),
            maximum=maximum(depths),
        ),
        inference=speed,
    )
end

function main()
    preact = read_preact_checkpoint(PREACT_CHECKPOINT)
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
    split.training_groups == preact.config.training_groups || error(
        "PreAct training split differs",
    )
    split.validation_groups == preact.config.validation_groups || error(
        "PreAct validation split differs",
    )
    validation_rows = ParentTraining.fixed_evaluation_subset(
        split.validation_rows,
        128,
        UInt64(preact.config.seed) + UInt64(0x202),
    )
    length(validation_rows) == 128 || error("held panel is not 128 states")
    observed_width = maximum(dataset.action_counts)
    learner_width = 16 * cld(observed_width, 16)
    learner_width == 80 || error("learner width changed")
    manifest_sha256 = sha256_file(joinpath(dataset_path, "manifest.json"))

    setup = PreActModels.setup_model(
        :preact_eca,
        Xoshiro(UInt64(preact.config.seed));
        n_quantiles=Int(preact.config.n_quantiles),
    )
    Int(setup.meta.parameters) == Int(preact.config.total_parameter_count) || error(
        "PreAct parameter count differs",
    )
    preact_test_state = Lux.testmode(preact.st)
    preact_predictor = batch -> first(setup.model(batch.inputs, preact.ps, preact_test_state))
    preact_batch = TrainingCore.allocate_host_batch(4; max_candidates=learner_width)
    preact_metrics = TrainingCore.evaluation_metrics(
        dataset,
        validation_rows,
        preact_batch,
        preact_predictor;
        margin_weight=TrainingCore.MARGIN_WEIGHT,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    preact_speed = benchmark_predictor(
        dataset,
        validation_rows,
        preact_batch,
        preact_predictor;
        repeats=EVALUATION_REPEATS,
    )
    preact_record = (;
        architecture="preact_eca",
        checkpoint=(;
            path=preact.path,
            bytes=filesize(preact.path),
            sha256=preact.sha256,
        ),
        update=preact.update,
        consumed_states=preact.update * Int(preact.config.state_batch),
        parameter_count=Int(preact.config.total_parameter_count),
        metrics=metric_record(preact_metrics),
        inference=preact_speed,
    )

    evrl = map(
        path -> evaluate_evrl(
            path,
            dataset,
            split,
            validation_rows,
            manifest_sha256,
        ),
        EVRL_CHECKPOINTS,
    )
    budget_matched = first(evrl)
    final_model = last(evrl)
    result = (;
        evaluation_id="teacher-held-comparison-2026-07-20",
        generated_at=string(Dates.now()),
        conditions=(;
            dataset_path,
            dataset_manifest_sha256=manifest_sha256,
            split_seed=UInt64(preact.config.split_seed),
            validation_subset_seed=UInt64(preact.config.seed) + UInt64(0x202),
            validation_states=length(validation_rows),
            validation_rows_sha256=bytes2hex(sha256(reinterpret(UInt8, validation_rows))),
            observed_candidate_width=observed_width,
            learner_width,
            state_batch_preact=4,
            state_batch_evrl=1,
            blas_threads=BLAS.get_num_threads(),
            julia_threads=Base.Threads.nthreads(),
            evaluation_repeats=EVALUATION_REPEATS,
            objective_mode=String(TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE),
            margin_mode=String(TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE),
            teacher_q_and_rank_used_only_as_targets=true,
            candidate_independent_evaluation=true,
            game_validation_seeds_touched=false,
            game_sealed_seeds_touched=false,
        ),
        preact=preact_record,
        evrl_budget_matched=budget_matched,
        evrl_final=final_model,
        differences=(;
            budget_matched_vs_preact=(;
                top1=budget_matched.metrics.top1_agreement - preact_record.metrics.top1_agreement,
                ndcg=budget_matched.metrics.ndcg - preact_record.metrics.ndcg,
                pairwise=budget_matched.metrics.pairwise_accuracy - preact_record.metrics.pairwise_accuracy,
                action_margin=budget_matched.metrics.action_margin - preact_record.metrics.action_margin,
                composite_loss=budget_matched.metrics.composite_loss - preact_record.metrics.composite_loss,
                inference_speed_ratio=budget_matched.inference.states_per_second / preact_record.inference.states_per_second,
            ),
            final_vs_preact=(;
                top1=final_model.metrics.top1_agreement - preact_record.metrics.top1_agreement,
                ndcg=final_model.metrics.ndcg - preact_record.metrics.ndcg,
                pairwise=final_model.metrics.pairwise_accuracy - preact_record.metrics.pairwise_accuracy,
                action_margin=final_model.metrics.action_margin - preact_record.metrics.action_margin,
                composite_loss=final_model.metrics.composite_loss - preact_record.metrics.composite_loss,
                inference_speed_ratio=final_model.inference.states_per_second / preact_record.inference.states_per_second,
            ),
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
