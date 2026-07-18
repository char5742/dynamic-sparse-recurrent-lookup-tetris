include(joinpath(@__DIR__, "fixedshape_learner.jl"))
include(joinpath(@__DIR__, "..", "..", "ad_backend_retry_2026", "common.jl"))

using .BeatFirstFixedShapeBackend

const ACTUAL_BATCH_SIZE = parse(Int, get(ENV, "BEAT_FIRST_SMOKE_BATCH", "2"))
const ACTUAL_STEPS = parse(Int, get(ENV, "BEAT_FIRST_SMOKE_STEPS", "2"))

struct ActualListwiseObjective{A,B}
    temperature::Float32
end

function (objective::ActualListwiseObjective{A,B})(model, ps, st, batch) where {A,B}
    raw, next_state = model(batch.inputs, ps, st)
    prediction = reshape(raw, A, B)
    loss = standardized_listwise_cross_entropy(
        prediction,
        batch.targets.teacher_q,
        batch.mask;
        temperature=objective.temperature,
    )
    return loss, next_state, NamedTuple()
end

function named_batch(input, teacher_q, mask)
    return (; inputs=input, targets=(; teacher_q), mask)
end

function main()
    ACTUAL_STEPS >= 2 || error("actual smoke requires at least two updates")
    problem = load_fixed_problem(ACTUAL_BATCH_SIZE)
    first_batch = named_batch(problem.batch...)

    dataset = load_teacher_dataset(DATASET_PATH)
    second_rows = collect((last(BASE_ROWS) - ACTUAL_BATCH_SIZE + 1):last(BASE_ROWS))
    second_input, second_targets, second_mask = candidate_batch(
        dataset, second_rows; teacher_targets=true
    )
    second_batch = named_batch(second_input, second_targets, second_mask)
    first_batch.targets.teacher_q == second_batch.targets.teacher_q && error(
        "actual smoke batches unexpectedly have identical teacher targets"
    )

    objective = ActualListwiseObjective{MAX_ACTIONS,ACTUAL_BATCH_SIZE}(TEMPERATURE)
    learner = init_backend(
        problem.model,
        problem.parameters,
        problem.state,
        OPTIMIZER,
        objective,
        first_batch;
        max_candidates=MAX_ACTIONS,
    )
    run_start = time_ns()
    first_result = train_step!(learner, first_batch)
    first_result.compiled_thunk_id === nothing && error("missing compiled thunk")
    final_result = first_result
    steady_wall = 0.0
    steady_pack = 0.0
    steady_transfer = 0.0
    steady_update = 0.0
    recompile_count = 0
    for step in 2:ACTUAL_STEPS
        batch = iseven(step) ? second_batch : first_batch
        result = train_step!(learner, batch)
        result.compiled_thunk_id == first_result.compiled_thunk_id || error(
            "changing actual batch contents recompiled the train step at $step"
        )
        result.recompiled && error("backend reported a recompile at $step")
        recompile_count += result.recompiled
        steady_wall += result.wall_seconds
        steady_pack += result.pack_seconds
        steady_transfer += result.transfer_seconds
        steady_update += result.update_seconds
        final_result = result
    end
    run_wall = (time_ns() - run_start) / 1.0e9
    snapshot = host_checkpoint(learner)
    finite_tree(snapshot.parameters) || error("non-finite synchronized parameters")
    snapshot.step == ACTUAL_STEPS || error("persistent TrainState step mismatch")
    memory = process_memory_bytes()

    @printf(
        "actual_compact_pass parameters=%d batch=%d steps=%d first_loss=%.9f final_loss=%.9f run_wall=%.6f first_wall=%.6f steady_updates_per_second=%.6f steady_pack_ms=%.6f steady_transfer_ms=%.6f steady_update_sync_ms=%.6f thunk=%d recompiles=%d peak_ram_bytes=%d current_ram_bytes=%d\n",
        Lux.parameterlength(snapshot.parameters),
        ACTUAL_BATCH_SIZE,
        ACTUAL_STEPS,
        first_result.loss,
        final_result.loss,
        run_wall,
        first_result.wall_seconds,
        (ACTUAL_STEPS - 1) / steady_wall,
        1.0e3 * steady_pack / (ACTUAL_STEPS - 1),
        1.0e3 * steady_transfer / (ACTUAL_STEPS - 1),
        1.0e3 * steady_update / (ACTUAL_STEPS - 1),
        final_result.compiled_thunk_id,
        recompile_count,
        memory.peak,
        memory.current,
    )
end

main()
