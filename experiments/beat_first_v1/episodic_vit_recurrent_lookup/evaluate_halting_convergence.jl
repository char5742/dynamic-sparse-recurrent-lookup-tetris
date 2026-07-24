#!/usr/bin/env julia

"""
Evaluate EVRL checkpoints on one fixed training-only panel.

This program never constructs validation rows and never uses a sealed seed.
Pass semicolon-separated absolute checkpoint paths through
`EVRL_HALTING_EVAL_CHECKPOINTS`.  The JSON result is written to
`EVRL_HALTING_EVAL_OUTPUT` when that variable is nonempty and is always printed.
`EVRL_HALTING_EVAL_FORCE_DEPTH` may be set to a recurrent depth for an
evaluation-only ablation.  Checkpoint restoration always uses the inherited
training hyperparameters; the override is applied only to prediction.
"""

for (name, value) in (
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
    "EVRL_EPISODIC_SHORTLIST" => "64",
    "EVRL_EPISODIC_CANDIDATE_CAP" => "64",
    "EVRL_SPATIAL_ANCHORS" => "2",
    "EVRL_SPATIAL_SHORTLIST" => "2",
    "EVRL_SPATIAL_CANDIDATE_CAP" => "3",
    "EVRL_FFN_DIM" => "128",
    "EVRL_INITIAL_HALT_PROBABILITY" => "0.8",
)
    haskey(ENV, name) || (ENV[name] = value)
end

using JSON3
using LinearAlgebra
using SHA
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const TrainingCore = Training.TrainingCore
const ParentTraining = Training.ParentTraining
const PANEL_STATES = parse(
    Int, strip(get(ENV, "EVRL_HALTING_EVAL_STATES", "128")),
)
PANEL_STATES > 0 || error("EVRL_HALTING_EVAL_STATES must be positive")
const FORCE_DEPTH = parse(
    Int, strip(get(ENV, "EVRL_HALTING_EVAL_FORCE_DEPTH", "0")),
)
FORCE_DEPTH == 0 ||
    Training.Model.MIN_RECURRENT_STEPS <= FORCE_DEPTH <=
        Training.Model.MAX_RECURRENT_STEPS ||
    error("EVRL_HALTING_EVAL_FORCE_DEPTH must be zero or inside recurrent bounds")

function _required_paths()
    raw = strip(get(ENV, "EVRL_HALTING_EVAL_CHECKPOINTS", ""))
    isempty(raw) && error("EVRL_HALTING_EVAL_CHECKPOINTS is required")
    paths = filter(!isempty, strip.(split(raw, ';')))
    isempty(paths) && error("no halting evaluation checkpoint was supplied")
    length(paths) <= 16 || error("at most 16 checkpoints may be evaluated")
    absolute = abspath.(paths)
    all(isfile, absolute) || error("a halting evaluation checkpoint is missing")
    return absolute
end

function _evaluate_checkpoint(
    checkpoint_path,
    dataset,
    split,
    split_metadata,
    manifest_sha256,
    rows,
    host_batch,
)
    payload, artifact = Training.read_checkpoint(checkpoint_path)
    payload.split_metadata == split_metadata ||
        error("checkpoint split metadata differs")
    hyperparameters = Training.runtime_hyperparameters(
        max(Int(payload.update), 1), payload,
    )
    trainer, _, _ = Training.restore_checkpoint(
        payload,
        split,
        split_metadata,
        manifest_sha256,
        hyperparameters,
    )
    evaluation_hyperparameters = if iszero(FORCE_DEPTH)
        hyperparameters
    else
        merge(
            hyperparameters,
            (; halting=merge(
                hyperparameters.halting,
                (; fixed_depth=FORCE_DEPTH),
            )),
        )
    end
    depth_counts = zeros(Int, Training.Model.MAX_RECURRENT_STEPS)
    halt_evidence_counts = zeros(Int, Training.Model.MAX_RECURRENT_STEPS)
    halt_evidence_sums = zeros(Float64, Training.Model.MAX_RECURRENT_STEPS)
    halt_evidence_minima = fill(Inf, Training.Model.MAX_RECURRENT_STEPS)
    halt_evidence_maxima = fill(-Inf, Training.Model.MAX_RECURRENT_STEPS)
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        rows,
        host_batch,
        batch -> begin
            raw, count = Training.predict_raw!(
                trainer,
                batch;
                training=false,
                expected_update=trainer.update,
                hyperparameters=evaluation_hyperparameters,
                record_tapes=true,
            )
            @inbounds for candidate in 1:count
                depth_counts[trainer.workspace.depths[candidate]] += 1
                tape = trainer.workspace.tapes[candidate]
                tape === nothing && error("halting telemetry lost its tape")
                for (step_index, step) in enumerate(tape.steps)
                    probability = clamp(
                        Float64(step.halt_probability),
                        1.0e-7,
                        1.0 - 1.0e-7,
                    )
                    halt_logit = log(probability / (1.0 - probability))
                    prior = Float64(Training.Model.HALT_DEPTH_PRIOR_SLOPE) *
                        (
                            step_index -
                            Float64(Training.Model.HALT_DEPTH_PRIOR_CENTER)
                        )
                    evidence = (halt_logit - prior) /
                        Float64(Training.Model.HALT_RESIDUAL_LOGIT_SCALE)
                    halt_evidence_counts[step_index] += 1
                    halt_evidence_sums[step_index] += evidence
                    halt_evidence_minima[step_index] = min(
                        halt_evidence_minima[step_index],
                        evidence,
                    )
                    halt_evidence_maxima[step_index] = max(
                        halt_evidence_maxima[step_index],
                        evidence,
                    )
                end
            end
            Training.raw_output(raw)
        end;
        margin_weight=hyperparameters.loss.margin_weight,
        margin_mode=hyperparameters.loss.margin_mode,
        objective_mode=
            TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    weighted_loss = Training._weighted_loss(metrics, hyperparameters.loss)
    total_candidates = sum(depth_counts)
    total_candidates > 0 || error("fixed panel contained no candidate")
    mean_depth = sum(
        depth * depth_counts[depth]
        for depth in eachindex(depth_counts)
    ) / total_candidates
    active_depths = findall(!iszero, depth_counts)
    halt_evidence_by_step = [
        (;
            step,
            count=halt_evidence_counts[step],
            mean=halt_evidence_sums[step] / halt_evidence_counts[step],
            minimum=halt_evidence_minima[step],
            maximum=halt_evidence_maxima[step],
        )
        for step in eachindex(halt_evidence_counts)
        if !iszero(halt_evidence_counts[step])
    ]
    result = (;
        update=Int(payload.update),
        consumed_states=Int(payload.consumed_states),
        checkpoint=artifact,
        composite_loss=Float64(weighted_loss),
        top1=Float64(metrics.top1_agreement),
        ndcg=Float64(metrics.ndcg),
        pairwise=Float64(metrics.pairwise_accuracy),
        margin=Float64(metrics.action_margin),
        mean_depth=Float64(mean_depth),
        minimum_depth=minimum(active_depths),
        maximum_depth=maximum(active_depths),
        forced_depth=FORCE_DEPTH,
        depth_counts,
        halt_evidence_by_step,
        halt_weight_norm=Float64(norm(trainer.model.lookup.halt_weight)),
        halt_bias=Float64(trainer.model.lookup.halt_bias[1]),
    )
    trainer = nothing
    payload = nothing
    GC.gc()
    return result
end

function main()
    checkpoint_paths = _required_paths()
    first_payload, _ = Training.read_checkpoint(first(checkpoint_paths))
    dataset_path = String(first_payload.config.dataset_path)
    manifest_sha256 = Training._sha256_file(
        joinpath(dataset_path, "manifest.json"),
    )
    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    split = ParentTraining.episode_separated_split(
        dataset;
        seed=Training.SPLIT_SEED,
        validation_fraction=0.10,
    )
    split_metadata = first_payload.split_metadata
    Training._assert_current_split(first_payload, split)
    rows = Int.(ParentTraining.fixed_evaluation_subset(
        split.training_rows,
        PANEL_STATES,
        Training.TRAIN_EVAL_SEED,
    ))
    rows_sha256 = bytes2hex(sha256(reinterpret(UInt8, rows)))
    expected_rows_sha256 = lowercase(strip(get(
        ENV, "EVRL_EXPECTED_TRAINING_ROWS_SHA256", "",
    )))
    isempty(expected_rows_sha256) ||
        rows_sha256 == expected_rows_sha256 ||
        error(
            "training-only panel SHA-256 differs: expected=" *
            expected_rows_sha256 * ", actual=" * rows_sha256,
        )
    host_batch = TrainingCore.allocate_host_batch(
        1; max_candidates=Training.LEARNER_WIDTH,
    )
    results = [
        _evaluate_checkpoint(
            path,
            dataset,
            split,
            split_metadata,
            manifest_sha256,
            rows,
            host_batch,
        )
        for path in checkpoint_paths
    ]
    record = (;
        evaluation="evrl-halting-convergence-training-only-v1",
        dataset_path,
        dataset_manifest_sha256=manifest_sha256,
        training_rows=length(rows),
        training_rows_sha256=rows_sha256,
        forced_depth=FORCE_DEPTH,
        validation_rows_touched=false,
        sealed_seed_touched=false,
        checkpoints=results,
    )
    output_path = strip(get(ENV, "EVRL_HALTING_EVAL_OUTPUT", ""))
    if !isempty(output_path)
        mkpath(dirname(abspath(output_path)))
        open(abspath(output_path), "w") do io
            JSON3.pretty(io, record)
            write(io, '\n')
        end
    end
    JSON3.pretty(stdout, record)
    write(stdout, '\n')
    return record
end

main()
