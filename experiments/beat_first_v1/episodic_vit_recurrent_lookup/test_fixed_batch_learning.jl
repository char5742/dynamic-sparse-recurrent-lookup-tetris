#!/usr/bin/env julia

using JSON3
using LinearAlgebra

for (name, value) in (
    "DSRL_BLOCKS" => "1",
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
    "EVRL_EPISODIC_SUPPORT" => "64",
    "EVRL_EPISODIC_CANDIDATE_CAP" => "128",
    "EVRL_EPISODIC_EXPLORATION_SLOTS" => "8",
    "EVRL_FFN_DIM" => "128",
    "EVRL_STATE_BATCH" => "4",
    "EVRL_SCHEDULER" => "barrierless",
    "EVRL_CPUSET_MODE" => "none",
    "EVRL_QUEUE_CHUNK" => "8",
    "EVRL_BACKWARD_QUEUE_CHUNK" => "1",
    "EVRL_ADAPTIVE_TAIL" => "0",
    # This diagnostic isolates recurrent body credit assignment.  Production
    # hard halting remains unchanged and is checked by the correctness smoke.
    "EVRL_FIXED_DEPTH" => "3",
    "EVRL_ROUTE_BALANCE_WEIGHT" => "0.05",
    "EVRL_WD_DENSE" => "0.0003",
)
    ENV[name] = get(ENV, name, value)
end

include(joinpath(@__DIR__, "teacher_training.jl"))
const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const TrainingCore = Training.TrainingCore

BLAS.set_num_threads(1)
Base.Threads.nthreads(:interactive) == 0 ||
    error("use JULIA_NUM_THREADS=<workers>,0")

const UPDATES = parse(Int, get(ENV, "EVRL_FIXED_BATCH_UPDATES", "100"))

function main()
    hyperparameters = Training.runtime_hyperparameters(UPDATES)
    dataset = TrainingCore.load_teacher_dataset(
        get(ENV, "EVRL_DATASET", Training.DEFAULT_DATASET);
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    split = Training.ParentTraining.episode_separated_split(
        dataset;
        seed=Training.SPLIT_SEED,
        validation_fraction=0.10,
    )
    rows = Int.(first(split.training_rows, Training.TRAINING_STATE_BATCH))
    batches = [
        TrainingCore.allocate_host_batch(
            1; max_candidates=Training.LEARNER_WIDTH,
        )
        for _ in 1:Training.TRAINING_STATE_BATCH
    ]
    for (batch, row) in zip(batches, rows)
        TrainingCore.pack_batch!(batch, dataset, [row])
    end
    evaluation_batch = TrainingCore.allocate_host_batch(
        1; max_candidates=Training.LEARNER_WIDTH,
    )
    trainer = Training.initialize_trainer(hyperparameters)
    before = Training.held_evaluation(
        trainer, dataset, rows, evaluation_batch; hyperparameters,
    )
    losses = Float64[]
    depths = Float64[]
    elapsed = @elapsed Training.run_with_barrierless_team!(
        trainer.scheduler.barrierless_executor,
    ) do _
        for update in 1:UPDATES
            step = Training.train_accumulated_step!(
                trainer,
                batches;
                expected_update=update,
                hyperparameters,
            )
            push!(losses, step.loss)
            push!(depths, step.mean_depth)
        end
    end
    after = Training.held_evaluation(
        trainer, dataset, rows, evaluation_batch; hyperparameters,
    )
    window = min(10, UPDATES)
    record = (;
        status="pass",
        rows,
        updates=UPDATES,
        elapsed_seconds=elapsed,
        updates_per_second=UPDATES / elapsed,
        first_window_loss=sum(@view(losses[1:window])) / window,
        last_window_loss=sum(@view(losses[(end - window + 1):end])) / window,
        loss_reduction_fraction=1.0 -
            (sum(@view(losses[(end - window + 1):end])) /
             sum(@view(losses[1:window]))),
        mean_depth=sum(depths) / length(depths),
        before,
        after,
    )
    JSON3.pretty(stdout, record)
    write(stdout, '\n')
    return record
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
