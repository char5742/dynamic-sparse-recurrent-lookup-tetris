using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using NPZ
using Optimisers
using Statistics
using TetrisPaperPlus
using Zygote

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyPartialTailTDContract
include(joinpath(@__DIR__, "tail_model.jl"))

function timing_record(timing)
    return (;
        seconds=timing.time,
        allocated_bytes=timing.bytes,
        gc_seconds=timing.gctime,
        compile_seconds=hasproperty(timing, :compile_time) ? timing.compile_time : nothing,
        recompile_seconds=hasproperty(timing, :recompile_time) ? timing.recompile_time : nothing,
    )
end

function write_stage(path, started, stage, details=(;))
    atomic_write_json(
        path,
        (;
            status="running",
            stage,
            stage_started_unix=time(),
            wall_seconds=time() - started,
            generated_at=string(now()),
            details,
        ),
    )
end

function collect_arrays!(output::Dict{String,Array}, value, path::AbstractString="")
    if value isa AbstractArray
        output[path] = Array(value)
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(path) ? string(name) : "$path.$name"
            collect_arrays!(output, getproperty(value, name), child)
        end
    end
    return output
end

function validate_training_data(data, row_freeze_path)
    rows = Int.(vec(data["source_rows"]))
    length(rows) == UPDATE_COUNT || error("training subset must contain exactly 300 rows")
    all(in(TRAIN_ROWS), rows) || error("training subset escaped rows 1--1500")
    length(unique(rows)) == UPDATE_COUNT || error("training subset contains duplicate rows")
    freeze = JSON3.read(read(row_freeze_path, String))
    rows == Int.(freeze.ordered_rows) || error("NPZ row order differs from preregistered freeze")
    episodes = Int.(vec(data["episode_ids"]))
    all(in(TRAIN_EPISODES), episodes) || error("training subset escaped episodes 1--6")
    steps = Int.(vec(data["episode_steps"]))
    counts = Int.(vec(data["action_counts"]))
    selected = Int.(vec(data["selected_actions"]))
    targets = Float32.(vec(data["targets"]))
    all((1 .<= selected) .& (selected .<= counts)) || error("selected action out of range")
    all(isfinite, targets) || error("non-finite frozen target")
    return (; rows, episodes, steps, counts, selected, targets)
end

function finite_difference_record(model, trainable, gradient, fixed_state, problem, label)
    parameter_array, gradient_array = if label == "score_net.layer_3.bias"
        trainable.score_net.layer_3.bias, gradient.score_net.layer_3.bias
    elseif label == "score_net.layer_1.weight"
        trainable.score_net.layer_1.weight, gradient.score_net.layer_1.weight
    elseif label == "board_net.resblocks.layer_31.layer_1.weight"
        trainable.board_net.resblocks.layer_31.layer_1.weight,
        gradient.board_net.resblocks.layer_31.layer_1.weight
    else
        error("unknown finite-difference coordinate")
    end
    index = first(eachindex(parameter_array))
    original = parameter_array[index]
    epsilon = FINITE_DIFFERENCE_EPSILON
    plus = minus = NaN
    try
        parameter_array[index] = original + epsilon
        plus = Float64(training_loss(model, trainable, fixed_state, problem))
        parameter_array[index] = original - epsilon
        minus = Float64(training_loss(model, trainable, fixed_state, problem))
    finally
        parameter_array[index] = original
    end
    fd = (plus - minus) / (2 * Float64(epsilon))
    ad = Float64(gradient_array[index])
    passed, tolerance = finite_difference_pass(fd, ad)
    return (;
        path=label,
        coordinate=Int(index),
        epsilon,
        plus,
        minus,
        finite_difference=fd,
        automatic_differentiation=ad,
        abs_error=abs(fd - ad),
        tolerance,
        passed,
    )
end

function differentiated_update(
    model, trainable, fixed_state, optimizer_state, problem; finite_difference=false
)
    differentiated = Zygote.withgradient(trainable) do current
        training_loss(model, current, fixed_state, problem)
    end
    gradient = only(differentiated.grad)
    validation = validate_gradient(trainable, gradient)
    validation.valid || return (;
        success=false,
        loss=Float64(differentiated.val),
        gradient,
        gradient_validation=validation,
        finite_differences=NamedTuple[],
        optimizer_state,
        trainable,
    )
    differences = if finite_difference
        [
            finite_difference_record(
                model, trainable, gradient, fixed_state, problem, label
            ) for label in (
                "score_net.layer_3.bias",
                "score_net.layer_1.weight",
                "board_net.resblocks.layer_31.layer_1.weight",
            )
        ]
    else
        NamedTuple[]
    end
    all(record.passed for record in differences) || return (;
        success=false,
        loss=Float64(differentiated.val),
        gradient,
        gradient_validation=validation,
        finite_differences=differences,
        optimizer_state,
        trainable,
    )
    next_optimizer_state, next_trainable = Optimisers.update(
        optimizer_state, trainable, gradient
    )
    return (;
        success=true,
        loss=Float64(differentiated.val),
        gradient,
        gradient_validation=validation,
        finite_differences=differences,
        optimizer_state=next_optimizer_state,
        trainable=next_trainable,
    )
end

function main(args=ARGS)
    length(args) == 5 || error(
        "usage: train_partial.jl OUTPUT_DIR TRAINING_NPZ CHECKPOINT FREEZE_JSON ROW_FREEZE_JSON"
    )
    output_directory, training_path, checkpoint_path, freeze_path, row_freeze_path =
        abspath.(args)
    isdir(output_directory) || error("one-shot wrapper must create output directory")
    phase_path = joinpath(output_directory, "training_phase.json")
    failure_path = joinpath(output_directory, "training_failure.json")
    for name in (
        "candidate_merged.jld2",
        "candidate_weights.npz",
        "final_reference.npz",
        "training_phase.json",
        "training_failure.json",
    )
        ispath(joinpath(output_directory, name)) && error("refusing to overwrite $name")
    end
    isfile(freeze_path) || error("missing one-shot freeze")
    isfile(row_freeze_path) || error("missing preregistered row freeze")
    require_hash(checkpoint_path, CHECKPOINT_SHA256, "legacy checkpoint")
    started = time()
    current_stage = Ref("initializing")
    failure_details = Ref{Any}((;))
    try
        VERSION == v"1.12.6" || error("Julia version mismatch: $(VERSION)")
        string(Base.pkgversion(Lux)) == "1.31.4" || error("Lux version mismatch")
        string(Base.pkgversion(Zygote)) == "0.7.11" || error("Zygote version mismatch")
        BLAS.set_num_threads(parse(Int, get(ENV, "P1_BLAS_THREADS", "10")))
        data = npzread(training_path)
        subset = validate_training_data(data, row_freeze_path)
        write_stage(phase_path, started, "training_subset_validated", (; rows=subset.rows))

        parameters, checkpoint_state = jldopen(checkpoint_path, "r") do file
            modernize_legacy_parameters(file["ps"]), file["st"]
        end
        model = LegacyQNetwork()
        Lux.parameterlength(parameters) == LEGACY_PARAMETER_COUNT || error(
            "full parameter count mismatch"
        )
        fixed_state = Lux.testmode(checkpoint_state)
        trainable = trainable_parameters(parameters)
        tree_array_elements(trainable) == TRAINABLE_PARAMETER_COUNT || error(
            "trainable count $(tree_array_elements(trainable)) != $TRAINABLE_PARAMETER_COUNT"
        )
        frozen_parameter_before = frozen_parameter_hashes(parameters)
        length(frozen_parameter_before) == 250 || error("frozen array leaf count must be 250")
        state_before = array_leaf_hashes(fixed_state)
        length(state_before) == 69 || error("running state array leaf count must be 69")
        trainable_before = array_leaf_hashes(trainable)
        length(trainable_before) == 34 || error("trainable array leaf count must be 34")
        tree_all_finite(parameters) || error("checkpoint contains non-finite parameters")
        tree_all_finite(fixed_state) || error("checkpoint contains non-finite state")

        current_stage[] = "split_tail_equivalence"
        write_stage(phase_path, started, current_stage[])
        # The first Xoshiro-frozen row is the preregistered step-0 witness.  Its
        # whole candidate list must contain full chunks and an actual tail, so
        # this tests both historical paths without spending the P1 wall budget
        # on 300 redundant full-model CPU passes.
        subset.rows[1] == STEP0_WITNESS_ROW || error("first Xoshiro witness row changed")
        subset.episodes[1] == STEP0_WITNESS_EPISODE || error("step-0 witness episode changed")
        subset.steps[1] == STEP0_WITNESS_STEP || error("step-0 witness step changed")
        witness_count = subset.counts[1]
        witness_count == STEP0_WITNESS_COUNT || error("step-0 witness candidate count changed")
        subset.selected[1] == STEP0_WITNESS_SELECTED || error(
            "step-0 witness selected action changed"
        )
        witness_input = row_input(data, 1, witness_count)
        split_tail_max_error = 0.0
        stored_q_max_error = 0.0
        equivalence_seconds = @elapsed begin
            full = full_historical_scores(
                model, parameters, fixed_state, witness_input, witness_count
            )
            split = split_historical_scores(
                model, trainable, parameters, fixed_state, witness_input, witness_count
            )
            old = Float32.(@view(data["stored_q"][1, 1:witness_count]))
            split_tail_max_error = maximum(abs, Float64.(full) .- Float64.(split))
            stored_q_max_error = maximum(abs, Float64.(full) .- Float64.(old))
        end
        split_tail_max_error <= SPLIT_TAIL_TOLERANCE || error(
            "split-tail equivalence failed: $split_tail_max_error"
        )
        stored_q_max_error <= STORED_Q_TOLERANCE || error(
            "old-Q equivalence failed: $stored_q_max_error"
        )
        array_leaf_hashes(fixed_state) == state_before || error(
            "running state changed during step-0 equivalence"
        )

        current_stage[] = "first_update_running"
        write_stage(
            phase_path,
            started,
            current_stage[],
            (; external_timeout_seconds=FIRST_UPDATE_SECONDS),
        )
        optimizer = Optimisers.AdamW(
            LEARNING_RATE, BETAS, WEIGHT_DECAY; couple=true
        )
        setup_timing = @timed Optimisers.setup(optimizer, trainable)
        optimizer_state = setup_timing.value
        tree_array_elements(optimizer_state) == OPTIMIZER_MOMENT_ELEMENTS || error(
            "optimizer moment count mismatch: $(tree_array_elements(optimizer_state))"
        )
        first_input = row_input(data, 1, subset.counts[1])
        first_problem = selected_training_problem(
            model,
            parameters,
            fixed_state,
            first_input,
            subset.selected[1],
            subset.counts[1],
            @view(data["stored_q"][1, :]),
            subset.targets[1],
        )
        first_timing = @timed differentiated_update(
            model,
            trainable,
            fixed_state,
            optimizer_state,
            first_problem;
            finite_difference=true,
        )
        first_result = first_timing.value
        first_update_seconds = setup_timing.time + first_timing.time
        failure_details[] = (;
            update=1,
            source_row=subset.rows[1],
            gradient_validation=first_result.gradient_validation,
            finite_differences=first_result.finite_differences,
        )
        first_result.success || error("first update gradient/finite-difference gate failed")
        first_update_seconds <= FIRST_UPDATE_SECONDS || error(
            "first update plus optimizer setup exceeded 180 seconds"
        )
        isfinite(first_result.loss) || error("non-finite first loss")
        trainable = first_result.trainable
        optimizer_state = first_result.optimizer_state
        tree_all_finite(trainable) || error("non-finite parameter after first update")
        tree_all_finite(optimizer_state) || error("non-finite optimizer after first update")
        update_records = Any[(;
            update=1,
            source_row=subset.rows[1],
            selected_range=collect(first_problem.range),
            loss=first_result.loss,
            timing=timing_record(first_timing),
            optimizer_setup=timing_record(setup_timing),
            gradient_elements=first_result.gradient_validation.gradient_elements,
            finite_differences=first_result.finite_differences,
        )]

        warm_seconds = Float64[]
        for update in 2:7
            warm_index = update - 1
            current_stage[] = "warm_update_$(warm_index)_running"
            write_stage(
                phase_path,
                started,
                current_stage[],
                (; update, source_row=subset.rows[update], external_timeout_seconds=WARM_UPDATE_SECONDS),
            )
            input = row_input(data, update, subset.counts[update])
            problem = selected_training_problem(
                model,
                parameters,
                fixed_state,
                input,
                subset.selected[update],
                subset.counts[update],
                @view(data["stored_q"][update, :]),
                subset.targets[update],
            )
            timing = @timed differentiated_update(
                model, trainable, fixed_state, optimizer_state, problem
            )
            result = timing.value
            failure_details[] = (;
                update,
                source_row=subset.rows[update],
                gradient_validation=result.gradient_validation,
            )
            result.success || error("gradient gate failed at update $update")
            timing.time <= WARM_UPDATE_SECONDS || error("warm update exceeded 15 seconds")
            isfinite(result.loss) || error("non-finite loss at update $update")
            trainable = result.trainable
            optimizer_state = result.optimizer_state
            tree_all_finite(trainable) || error("non-finite parameters at update $update")
            tree_all_finite(optimizer_state) || error("non-finite optimizer at update $update")
            push!(warm_seconds, timing.time)
            push!(update_records, (;
                update,
                source_row=subset.rows[update],
                selected_range=collect(problem.range),
                loss=result.loss,
                timing=timing_record(timing),
                gradient_elements=result.gradient_validation.gradient_elements,
            ))
        end
        warm_median = median(warm_seconds)
        warm_median <= WARM_MEDIAN_SECONDS || error(
            "six-update warm median $warm_median exceeds $WARM_MEDIAN_SECONDS"
        )
        local_elapsed_through_warm = time() - started
        one_shot_elapsed_through_warm = if haskey(ENV, "P1_HARD_DEADLINE_UNIX")
            deadline = parse(Float64, ENV["P1_HARD_DEADLINE_UNIX"])
            max(local_elapsed_through_warm, HARD_WALL_SECONDS - (deadline - time()))
        else
            local_elapsed_through_warm
        end
        projected_seconds = projected_total_seconds(
            one_shot_elapsed_through_warm, warm_median; completed_updates=7
        )
        projected_seconds <= HARD_WALL_SECONDS || error(
            "measured P1 projection $projected_seconds exceeds 2100 seconds"
        )

        for update in 8:UPDATE_COUNT
            current_stage[] = "update_$(update)_running"
            write_stage(
                phase_path, started, current_stage[], (; update, source_row=subset.rows[update])
            )
            input = row_input(data, update, subset.counts[update])
            problem = selected_training_problem(
                model,
                parameters,
                fixed_state,
                input,
                subset.selected[update],
                subset.counts[update],
                @view(data["stored_q"][update, :]),
                subset.targets[update],
            )
            timing = @timed differentiated_update(
                model, trainable, fixed_state, optimizer_state, problem
            )
            result = timing.value
            failure_details[] = (;
                update,
                source_row=subset.rows[update],
                gradient_validation=result.gradient_validation,
            )
            result.success || error("gradient gate failed at update $update")
            timing.time <= WARM_UPDATE_SECONDS || error(
                "update $update exceeded the 15-second per-update gate"
            )
            isfinite(result.loss) || error("non-finite loss at update $update")
            trainable = result.trainable
            optimizer_state = result.optimizer_state
            tree_all_finite(trainable) || error("non-finite parameters at update $update")
            tree_all_finite(optimizer_state) || error("non-finite optimizer at update $update")
            push!(update_records, (;
                update,
                source_row=subset.rows[update],
                selected_range=collect(problem.range),
                loss=result.loss,
                timing=timing_record(timing),
                gradient_elements=result.gradient_validation.gradient_elements,
            ))
        end

        current_stage[] = "merge_export_running"
        write_stage(phase_path, started, current_stage[])
        merged_parameters = merge_trainable(parameters, trainable)
        Lux.parameterlength(merged_parameters) == LEGACY_PARAMETER_COUNT || error(
            "merged parameter count mismatch"
        )
        frozen_parameter_after = frozen_parameter_hashes(merged_parameters)
        frozen_parameter_after == frozen_parameter_before || error(
            "a frozen parameter SHA-256 changed"
        )
        state_after = array_leaf_hashes(fixed_state)
        state_after == state_before || error("running state SHA-256 changed")
        trainable_after = array_leaf_hashes(trainable)
        changed_trainable_arrays = sort([
            path for (path, digest) in trainable_after if trainable_before[path] != digest
        ])
        isempty(changed_trainable_arrays) && error("no trainable array changed")

        candidate_checkpoint = joinpath(output_directory, "candidate_merged.jld2")
        jldsave(
            candidate_checkpoint;
            ps=merged_parameters,
            st=fixed_state,
            metadata=(;
                experiment=EXPERIMENT_ID,
                source_checkpoint_sha256=CHECKPOINT_SHA256,
                row_freeze_sha256=hex_sha256(row_freeze_path),
                updates=UPDATE_COUNT,
                trainable_paths=collect(TRAINABLE_PATHS),
            ),
        )
        arrays = Dict{String,Array}()
        collect_arrays!(arrays, merged_parameters, "ps")
        collect_arrays!(arrays, fixed_state, "st")
        length(arrays) == 353 || error("merged export must contain exactly 353 arrays")
        weights_path = joinpath(output_directory, "candidate_weights.npz")
        export_seconds = @elapsed npzwrite(weights_path, arrays)

        reference_input = row_input(data, 1, subset.counts[1])
        merged_output = full_historical_scores(
            model, merged_parameters, fixed_state, reference_input, subset.counts[1]
        )
        final_split = split_historical_scores(
            model, trainable, merged_parameters, fixed_state, reference_input, subset.counts[1]
        )
        final_merge_error = maximum(abs, Float64.(merged_output) .- Float64.(final_split))
        final_merge_error <= SPLIT_TAIL_TOLERANCE || error(
            "fresh merged model differs from split tail: $final_merge_error"
        )
        reference_path = joinpath(output_directory, "final_reference.npz")
        npzwrite(
            reference_path,
            Dict(
                "board" => reference_input[1],
                "placement" => reference_input[2],
                "ren" => reference_input[3],
                "back_to_back" => reference_input[4],
                "tspin" => reference_input[5],
                "queue" => reference_input[6],
                "lux_output" => merged_output,
                "action_count" => [subset.counts[1]],
            ),
        )
        final = (;
            status="training_phase_complete",
            generated_at=string(now()),
            wall_seconds=time() - started,
            constants=expected_constants(),
            source_checkpoint_sha256=CHECKPOINT_SHA256,
            training_subset_sha256=hex_sha256(training_path),
            training_subset_path=training_path,
            row_freeze_sha256=hex_sha256(row_freeze_path),
            row_freeze_path,
            freeze_sha256=hex_sha256(freeze_path),
            freeze_path,
            source_rows=subset.rows,
            update_count=length(update_records),
            step0=(;
                witness_source_row=subset.rows[1],
                witness_candidate_count=witness_count,
                witness_selected_action=subset.selected[1],
                witness_chunks=[collect(range) for range in chunk_ranges(witness_count)],
                split_tail_max_abs_error=split_tail_max_error,
                stored_old_q_max_abs_error=stored_q_max_error,
                seconds=equivalence_seconds,
            ),
            first_update_seconds,
            warm_update_seconds=warm_seconds,
            warm_median_seconds=warm_median,
            one_shot_elapsed_through_warm,
            projected_total_seconds=projected_seconds,
            gradient_elements_every_update=all(
                record.gradient_elements == TRAINABLE_PARAMETER_COUNT for record in update_records
            ),
            gradient_elements=TRAINABLE_PARAMETER_COUNT,
            optimizer_moment_elements=tree_array_elements(optimizer_state),
            finite_differences=first_result.finite_differences,
            update_records,
            trainable_array_count=length(trainable_after),
            changed_trainable_arrays,
            frozen_parameter_array_count=length(frozen_parameter_after),
            frozen_parameter_sha_unchanged=frozen_parameter_after == frozen_parameter_before,
            running_state_array_count=length(state_after),
            running_state_sha_unchanged=state_after == state_before,
            running_state_max_change=0.0,
            candidate_checkpoint,
            candidate_checkpoint_sha256=hex_sha256(candidate_checkpoint),
            weights_path,
            weights_sha256=hex_sha256(weights_path),
            export_array_count=length(arrays),
            export_seconds,
            reference_path,
            reference_sha256=hex_sha256(reference_path),
            final_merge_max_abs_error=final_merge_error,
            original_checkpoint_overwritten=false,
            existing_weight_artifact_overwritten=false,
            validation_or_test_seed_loaded=false,
            game_evaluation_run=false,
        )
        atomic_write_json(phase_path, final)
        println(JSON3.write((; status=final.status, output=phase_path)))
        return final
    catch exception
        failure = (;
            status="training_phase_failed",
            generated_at=string(now()),
            stage=current_stage[],
            wall_seconds=time() - started,
            error=sprint(showerror, exception, catch_backtrace()),
            details=failure_details[],
            original_checkpoint_overwritten=false,
            existing_weight_artifact_overwritten=false,
            validation_or_test_seed_loaded=false,
            game_evaluation_run=false,
        )
        ispath(failure_path) || atomic_write_json(failure_path, failure)
        rethrow()
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
