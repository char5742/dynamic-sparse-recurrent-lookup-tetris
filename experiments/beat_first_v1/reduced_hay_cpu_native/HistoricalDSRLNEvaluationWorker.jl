#!/usr/bin/env julia

# Internal worker for evaluate_development_validation.jl.  The July DSRLN
# checkpoint serializes concrete Julia types from its source commit, so it must
# be loaded in an isolated process against that exact source tree.  This file
# is not a second evaluator: it returns the same frozen panel contract and the
# same Q-only ranking metrics to the canonical entrypoint.

length(ARGS) == 6 || error(
    "internal DSRLN worker expects source, checkpoint, SHA, dataset, output, repeats",
)
const HISTORICAL_ROOT = abspath(ARGS[1])
const CHECKPOINT = abspath(ARGS[2])
const EXPECTED_CHECKPOINT_SHA256 = lowercase(ARGS[3])
const DATASET_PATH = abspath(ARGS[4])
const OUTPUT = abspath(ARGS[5])
const REPEATS = parse(Int, ARGS[6])
REPEATS >= 1 || error("internal DSRLN repeats must be positive")

for (name, value) in (
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_BLOCKS" => "3",
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
    "EVRL_STATE_BATCH" => "4",
)
    ENV[name] = value
end

using JSON3
using LinearAlgebra
using SHA

include(joinpath(@__DIR__, "DevelopmentValidationPanel.jl"))
using .DevelopmentValidationPanel

training_path = joinpath(
    HISTORICAL_ROOT,
    "experiments",
    "beat_first_v1",
    "episodic_vit_recurrent_lookup",
    "teacher_training.jl",
)
isfile(training_path) || error("historical teacher_training.jl is missing")
Base.include(Main, training_path)
const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const Core = Training.TrainingCore
const Parent = Training.ParentTraining

BLAS.set_num_threads(1)
checkpoint_sha = sha256_file(CHECKPOINT)
checkpoint_sha == EXPECTED_CHECKPOINT_SHA256 || error(
    "historical DSRLN checkpoint SHA-256 mismatch",
)
payload, artifact = Training.read_checkpoint(
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
)
String(payload.config.dataset_path) == DATASET_PATH || error(
    "DSRLN checkpoint dataset path differs from the requested dataset",
)
source = Core.load_teacher_dataset(
    DATASET_PATH;
    max_candidates=Core.MAX_CANDIDATES,
    allow_partial_dataset=false,
    geometry_cache_max_states=1,
)
contract = load_contract(DATASET_PATH, source)
checkpoint_rows = Int.(payload.split_metadata.validation_eval_rows)
checkpoint_rows == contract.rows || error(
    "DSRLN checkpoint validation rows differ from the frozen development panel",
)
split = Parent.episode_separated_split(
    source;
    seed=UInt64(payload.config.split_seed),
    validation_fraction=0.10,
)
hyperparameters = payload.config.hyperparameters
trainer, _, _ = Training.restore_checkpoint(
    payload,
    split,
    payload.split_metadata,
    contract.dataset_manifest_sha256,
    hyperparameters,
)
Int(payload.config.total_parameter_count) == 20_577_789 || error(
    "canonical DSRLN parameter count changed",
)
width = 80
state_batch = 1
host_batch = Core.allocate_host_batch(state_batch; max_candidates=width)
score_sets = Vector{Vector{Float32}}(undef, contract.states)

function pass!(capture::Bool)
    checksum = 0.0
    @inbounds for panel_slot in eachindex(contract.rows)
        row = contract.rows[panel_slot]
        Core.pack_batch!(host_batch, source, Int[row])
        raw, count = Training.predict_raw!(
            trainer,
            host_batch;
            training=false,
            expected_update=trainer.update,
            hyperparameters,
            record_tapes=false,
        )
        expected_count = Int(source.action_counts[row])
        count == expected_count || error("DSRLN candidate count changed")
        output = Training.raw_output(raw)
        q = vec(output.q)
        length(q) >= count || error("DSRLN Q output is too short")
        checksum += sum(Float64, @view(q[1:count]))
        capture && (score_sets[panel_slot] = Float32.(q[1:count]))
    end
    return checksum
end

pass!(true)
metrics = evaluate_rankings(score_sets, source, contract)
GC.gc()
started = time_ns()
checksum = Ref(0.0)
for _ in 1:REPEATS
    checksum[] += pass!(false)
end
seconds = (time_ns() - started) * 1.0e-9

record = (;
    model=(;
        name="DSRLN",
        architecture="episodic_vit_recurrent_lookup_fixed_depth_2",
        checkpoint=(;
            absolute_path=CHECKPOINT,
            bytes=filesize(CHECKPOINT),
            sha256=checkpoint_sha,
        ),
        checkpoint_update=Int(payload.update),
        parameter_count=Int(payload.config.total_parameter_count),
        raw_float32_parameter_bytes=4 * Int(payload.config.total_parameter_count),
        parameter_resident_bytes=Base.summarysize(trainer.model),
        inference_resident_bytes=Base.summarysize((trainer, host_batch)),
        resident_memory_scope=
            "restored historical trainer plus host batch; includes training runtime retained by historical loader",
        historical_source_commit=
            "b1af779d8f490098705f77cfdbc354d01b46afd2",
    ),
    conditions=(;
        stage=String(contract.stage),
        scientific_status="development panel reused for model selection; not held test",
        dataset_path=contract.dataset_path,
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        panel_seed=string(PANEL_SEED),
        panel_rows_sha256=contract.rows_sha256,
        states=contract.states,
        candidates=contract.candidates,
        minimum_candidates=contract.minimum_candidates,
        maximum_candidates=contract.maximum_candidates,
        teacher_tie_states=contract.teacher_tie_states,
        held_test_touched=false,
        sealed_game_seed_touched=false,
        ranking_objective=(;
            q_only=true,
            standardization="per-state z-score",
            temperature=0.50,
            listnet_excess="cross entropy minus teacher entropy",
            legacy_top1="stable first maximum",
            tie_aware_tolerance=1.0e-6,
        ),
    ),
    metrics=(;
        states=metrics.states,
        candidates=metrics.candidates,
        listnet_excess=metrics.listnet_excess,
        listnet_cross_entropy=metrics.listnet_cross_entropy,
        listnet_teacher_entropy=metrics.listnet_teacher_entropy,
        legacy_stable_top1=metrics.legacy_stable_top1,
        tie_aware_top1=metrics.tie_aware_top1,
        ndcg=metrics.ndcg,
        pairwise_accuracy=metrics.pairwise_accuracy,
    ),
    inference=(;
        warmup_excluded=true,
        input_packing_included=true,
        repeats=REPEATS,
        logical_states=REPEATS * contract.states,
        logical_candidates=REPEATS * contract.candidates,
        wall_seconds=seconds,
        states_per_second=REPEATS * contract.states / seconds,
        candidates_per_second=REPEATS * contract.candidates / seconds,
        state_batch,
        julia_default_threads=Threads.nthreads(:default),
        julia_interactive_threads=Threads.nthreads(:interactive),
        blas_threads=BLAS.get_num_threads(),
        checksum=checksum[],
        process_peak_rss_bytes=Sys.maxrss(),
    ),
)
mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    JSON3.pretty(io, record)
    write(io, '\n')
end
