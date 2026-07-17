using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using NPZ
using Optimisers
using Random
using SHA
using Statistics
using Zygote

include(joinpath(@__DIR__, "contract.jl"))
using .FrozenOldAdditiveResidualQ1Contract
include(joinpath(@__DIR__, "model.jl"))

function replace_json(path::AbstractString, value)
    temporary = "$path.tmp"
    ispath(temporary) && rm(temporary; force=true)
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
        flush(io)
    end
    mv(temporary, path; force=true)
end

function huber(value; delta=HUBER_DELTA)
    absolute = abs.(value)
    return ifelse.(absolute .<= delta, 0.5f0 .* value .^ 2, delta .* (absolute .- 0.5f0 * delta))
end

function validate_data(data, first_row::Int, last_row::Int, eligible_count::Int)
    rows = Int.(vec(data["source_rows"]))
    rows == collect(first_row:last_row) || error("NPZ source rows are not the exact role range")
    counts = Int.(vec(data["action_counts"]))
    selected = Int.(vec(data["selected_actions"]))
    valid = Bool.(vec(data["target_valid"]))
    targets = Float32.(vec(data["targets"]))
    size(data["placements"], 2) == ACTIONS || error("candidate axis must be fixed 74")
    maximum(counts) == ACTIONS || error("effective candidate maximum must be 74")
    all((1 .<= selected) .& (selected .<= counts)) || error("selected action out of range")
    count(valid) == eligible_count || error("eligible target count mismatch")
    all(isfinite, targets[valid]) || error("non-finite eligible target")
    return (; rows, counts, selected, valid, targets)
end

flat_index(action::Int, state_slot::Int) = action + (state_slot - 1) * ACTIONS

function batch_input(data, slots::Vector{Int})
    batch = length(slots)
    flat = ACTIONS * batch
    board = zeros(Float32, 24, 10, 1, flat)
    placement = zeros(Float32, 24, 10, 1, flat)
    ren = zeros(Float32, 1, flat)
    back_to_back = zeros(Float32, 1, flat)
    tspin = zeros(Float32, 1, flat)
    queue = zeros(Float32, 7, 6, flat)
    mask = zeros(Float32, ACTIONS, batch)
    old_q = fill(Float32(NaN), ACTIONS, batch)
    selected = Vector{Int}(undef, batch)
    targets = Vector{Float32}(undef, batch)
    for (column, slot) in enumerate(slots)
        count = Int(data["action_counts"][slot])
        selected[column] = Int(data["selected_actions"][slot])
        targets[column] = Float32(data["targets"][slot])
        range = flat_index(1, column):flat_index(ACTIONS, column)
        board_one = permutedims(@view(data["boards"][slot:slot, :, :, :]), (2, 3, 4, 1))
        board[:, :, :, range] .= repeat(Float32.(board_one); outer=(1, 1, 1, ACTIONS))
        placement[:, :, :, range] .= permutedims(
            Float32.(@view(data["placements"][slot, :, :, :, :])), (2, 3, 4, 1)
        )
        ren[1, range] .= Float32(data["ren"][slot])
        back_to_back[1, range] .= Float32(data["back_to_back"][slot])
        tspin[1, range] .= Float32.(@view(data["tspin"][slot, :]))
        queue_one = Float32.(@view(data["queues"][slot, :, :]))
        queue[:, :, range] .= repeat(reshape(queue_one, 7, 6, 1); outer=(1, 1, ACTIONS))
        mask[1:count, column] .= 1f0
        old_q[1:count, column] .= Float32.(@view(data["stored_q"][slot, 1:count]))
    end
    return (board, placement, ren, back_to_back, tspin, queue), mask, old_q, selected, targets
end

function q1_loss(model, parameters, state, batch)
    input, mask, old_q, selected, targets = batch
    raw, _ = model(input, parameters, state)
    correction = reshape(raw, ACTIONS, length(selected))
    selected_values = [old_q[selected[column], column] + correction[selected[column], column] for column in eachindex(selected)]
    selected_loss = mean(huber(selected_values .- targets))
    anchor_loss = sum(huber(correction) .* mask) / sum(mask)
    return selected_loss + ANCHOR_WEIGHT * anchor_loss
end

function gradient_validation(parameters, gradient)
    ps_arrays = array_leaves(parameters)
    grad_arrays = gradient_arrays(gradient)
    keys(ps_arrays) == keys(grad_arrays) || return (valid=false, reason="gradient path mismatch")
    elements = sum(length, values(grad_arrays); init=0)
    finite = all(array -> all(isfinite, array), values(grad_arrays))
    return (;
        valid=finite && elements == PARAMETER_COUNT,
        gradient_elements=elements,
        gradient_array_count=length(grad_arrays),
        paths=sort!(collect(keys(grad_arrays))),
        all_finite=finite,
    )
end

function update_once(model, parameters, state, optimizer_state, batch)
    differentiated = Zygote.withgradient(parameters) do current
        q1_loss(model, current, state, batch)
    end
    gradient = only(differentiated.grad)
    validation = gradient_validation(parameters, gradient)
    validation.valid || error("invalid gradient: $validation")
    clipped_gradient, clip_statistics = clip_global_tree_l2(
        gradient; limit=Float64(GRADIENT_CLIP), tolerance=1.0e-6
    )
    next_optimizer_state, next_parameters = Optimisers.update(
        optimizer_state, parameters, clipped_gradient
    )
    return (
        Float64(differentiated.val),
        merge(validation, clip_statistics),
        next_optimizer_state,
        next_parameters,
    )
end

function zero_policy_gate(model, parameters, state, data)
    max_error = 0.0
    old_best = Int[]
    combined_best = Int[]
    valid_outputs = 0
    all_bitwise_positive_zero = true
    for start in 1:32:length(data["source_rows"])
        slots = collect(start:min(start + 31, length(data["source_rows"])))
        input, _, old_q, _, _ = batch_input(data, slots)
        raw, _ = model(input, parameters, state)
        correction = reshape(Float32.(Array(raw)), ACTIONS, length(slots))
        for (column, slot) in enumerate(slots)
            count = Int(data["action_counts"][slot])
            valid = @view correction[1:count, column]
            all_bitwise_positive_zero &= all(reinterpret(UInt32, value) == 0x00000000 for value in valid)
            combined = @view(old_q[1:count, column]) .+ valid
            max_error = max(max_error, maximum(abs, Float64.(combined) .- Float64.(@view(old_q[1:count, column]))))
            push!(old_best, argmax(@view old_q[1:count, column]))
            push!(combined_best, argmax(combined))
            valid_outputs += count
        end
    end
    agreement = mean(old_best .== combined_best)
    all_bitwise_positive_zero || error("update-0 correction is not bitwise +0 on every valid action")
    max_error == 0.0 || error("update-0 combined Q differs from stored old-Q")
    agreement == 1.0 || error("update-0 top-1 agreement is not 1")
    return (; valid_outputs, bitwise_zero=true, combined_stored_old_max_abs_error=max_error, top1_agreement=agreement)
end

function load_initializer(path, model)
    return jldopen(path, "r") do file
        config = file["model_config"]
        observed = (channels=Int(config.channels), blocks=Int(config.blocks), spatial_channels=Int(config.spatial_channels))
        observed == (channels=8, blocks=1, spatial_channels=2) || error("initializer model config mismatch: $observed")
        source = file["ps"]
        Lux.parameterlength(source) == PARAMETER_COUNT || error("initializer parameter count mismatch")
        state = Lux.testmode(file["st"])
        parameters = zero_scalar_head(source)
        only_scalar_head_changed(source, parameters) || error("zero initialization changed non-scalar parameters")
        model_arrays = array_leaves(parameters)
        all(iszero, model_arrays["head.layer_3.weight"]) || error("scalar weight is not zero")
        all(iszero, model_arrays["head.layer_3.bias"]) || error("scalar bias is not zero")
        return parameters, state, length(model_arrays)
    end
end

function save_snapshot(path, parameters, state, update, metadata)
    ispath(path) && error("refusing to overwrite snapshot $path")
    jldsave(path; ps=parameters, st=state, model_config=(channels=8, blocks=1, spatial_channels=2), parameter_count=PARAMETER_COUNT, update, metadata)
end

function collect_export_arrays(parameters)
    return Dict("ps.$path" => Array(value) for (path, value) in array_leaves(parameters))
end

function role_diagnostics(data, model, parameters, state, rows)
    residual_targets = Float64[]
    residual_predictions = Float64[]
    selected_losses = Float64[]
    for chunk in Iterators.partition(rows, 32)
        slots = collect(chunk)
        input, _, old_q, selected, targets = batch_input(data, slots)
        raw, _ = model(input, parameters, state)
        correction = reshape(Float32.(Array(raw)), ACTIONS, length(slots))
        for column in eachindex(slots)
            prediction = Float64(correction[selected[column], column])
            target_residual = Float64(targets[column] - old_q[selected[column], column])
            push!(residual_predictions, prediction)
            push!(residual_targets, target_residual)
            push!(selected_losses, Float64(only(huber([prediction - target_residual]))))
        end
    end
    return (;
        rows=length(rows),
        mean_target_residual=mean(residual_targets),
        mean_prediction=mean(residual_predictions),
        selected_huber=mean(selected_losses),
        correlation=std(residual_predictions) > 0 && std(residual_targets) > 0 ? cor(residual_predictions, residual_targets) : nothing,
        sign_agreement=mean(signbit.(residual_predictions) .== signbit.(residual_targets)),
    )
end

function self_check()
    rng = Xoshiro(0x51)
    model = q1_model()
    ps, st = Lux.setup(rng, model)
    Lux.parameterlength(ps) == PARAMETER_COUNT || error("synthetic model parameter count mismatch")
    zeroed = zero_scalar_head(ps)
    only_scalar_head_changed(ps, zeroed) || error("zero head changed backbone")
    input = (rand(rng, Float32, 24, 10, 1, 4), rand(rng, Float32, 24, 10, 1, 4), rand(rng, Float32, 1, 4), rand(rng, Float32, 1, 4), rand(rng, Float32, 1, 4), rand(rng, Float32, 7, 6, 4))
    evaluation_state = Lux.testmode(st)
    original_output, _ = model(input, ps, evaluation_state)
    for index in 1:4
        sliced = (input[1][:, :, :, index:index], input[2][:, :, :, index:index], input[3][:, index:index], input[4][:, index:index], input[5][:, index:index], input[6][:, :, index:index])
        isolated, _ = model(sliced, ps, evaluation_state)
        abs(Float64(original_output[index]) - Float64(isolated[1])) <= 1.0e-5 ||
            error("compact correction is not co-pack invariant within 1e-5")
    end
    output, _ = model(input, zeroed, evaluation_state)
    all(reinterpret(UInt32, value) == 0 for value in output) || error("synthetic zero-head output mismatch")
    flat_index(1, 1) == 1 || error("fixed74 flat index origin mismatch")
    flat_index(74, 4) == 296 || error("fixed74 flat index terminal mismatch")
    witness = global_clip_synthetic_witness()
    witness.statistics.global_gradient_norm_before == 13.0 || error("global clip witness pre-norm mismatch")
    witness.statistics.global_gradient_scale == 1 / 13 || error("global clip witness scale mismatch")
    isapprox(witness.statistics.global_gradient_norm_after, 1.0; atol=1.0e-6, rtol=0) || error("global clip witness post-norm mismatch")
    isapprox(witness.first_norm, 5 / 13; atol=1.0e-6, rtol=0) || error("first leaf clip norm mismatch")
    isapprox(witness.second_norm, 12 / 13; atol=1.0e-6, rtol=0) || error("second leaf clip norm mismatch")
    witness.empty_named_tuple_preserved || error("empty NamedTuple was not preserved")
    witness.nothing_preserved || error("Nothing gradient leaf was not preserved")
    return true
end

function main(args=ARGS)
    if args == ["--self-check"] || isempty(args)
        self_check()
        return
    end
    length(args) == 6 || error("usage: train_q1.jl OUTPUT_DIR TRAINING_NPZ INITIALIZER FREEZE_JSON ORDER_JSON OLD_CHECKPOINT")
    output_directory, training_path, initializer_path, freeze_path, order_path, old_checkpoint_path = abspath.(args)
    isdir(output_directory) || error("wrapper must create Q1 output directory")
    for name in ("training_phase.json", "training_failure.json", "correction_update2000.jld2", "correction_weights.npz", "combined_reference.npz")
        ispath(joinpath(output_directory, name)) && error("refusing to overwrite $name")
    end
    require_hash(initializer_path, INITIALIZER_SHA256, "compact initializer")
    require_hash(old_checkpoint_path, OLD_CHECKPOINT_SHA256, "frozen old checkpoint")
    VERSION == v"1.12.6" || error("Julia version mismatch")
    string(Base.pkgversion(Lux)) == "1.31.4" || error("Lux version mismatch")
    string(Base.pkgversion(Zygote)) == "0.7.11" || error("Zygote version mismatch")
    started = time()
    progress_path = joinpath(output_directory, "training_progress.json")
    try
        data = npzread(training_path)
        subset = validate_data(data, first(TRAIN_ROWS), last(TRAIN_ROWS), EXPECTED_TRAIN_ELIGIBLE)
        order_freeze = JSON3.read(read(order_path, String))
        order = Int.(order_freeze.ordered_rows)
        length(order) == ORDER_LENGTH || error("frozen order length mismatch")
        canonical = bytes2hex(SHA.sha256(join(string.(order), ",")))
        canonical == String(order_freeze.ordered_rows_sha256) || error("frozen order digest mismatch")
        all(row -> subset.valid[row], order) || error("frozen order contains ineligible row")
        model = q1_model()
        parameters, state, parameter_array_count = load_initializer(initializer_path, model)
        tree_all_finite(parameters) || error("initializer contains non-finite values")
        zero_gate = zero_policy_gate(model, parameters, state, data)
        input_hashes_before = Dict(
            "initializer" => hex_sha256(initializer_path),
            "old_checkpoint" => hex_sha256(old_checkpoint_path),
            "training_npz" => hex_sha256(training_path),
            "freeze" => hex_sha256(freeze_path),
            "order" => hex_sha256(order_path),
        )
        metadata = (; experiment=EXPERIMENT_ID, source_initializer_sha256=INITIALIZER_SHA256, old_checkpoint_sha256=OLD_CHECKPOINT_SHA256, order_sha256=input_hashes_before["order"])
        save_snapshot(joinpath(output_directory, "correction_update0.jld2"), parameters, state, 0, metadata)
        optimizer = Optimisers.AdamW(LEARNING_RATE, (0.9, 0.999), WEIGHT_DECAY)
        optimizer_state = Optimisers.setup(optimizer, parameters)
        update_records = Any[]
        reference_paths = nothing
        projected_total_seconds = NaN
        for update in 1:UPDATE_COUNT
            time() - started <= HARD_WALL_SECONDS || error("12-minute hard wall exceeded")
            rows = order[(STATE_BATCH * (update - 1) + 1):(STATE_BATCH * update)]
            timing = @timed begin
                batch = batch_input(data, rows)
                loss, validation, optimizer_state, parameters = update_once(model, parameters, state, optimizer_state, batch)
            end
            isfinite(loss) || error("non-finite loss at update $update")
            tree_all_finite(parameters) || error("non-finite parameters at update $update")
            reference_paths === nothing && (reference_paths = validation.paths)
            validation.paths == reference_paths || error("gradient paths changed at update $update")
            record = (;
                update,
                source_rows=rows,
                loss,
                seconds=timing.time,
                allocated_bytes=timing.bytes,
                gradient_elements=validation.gradient_elements,
                gradient_array_count=validation.gradient_array_count,
                clip_mode=validation.clip_mode,
                global_gradient_norm_before=validation.global_gradient_norm_before,
                global_gradient_norm_after=validation.global_gradient_norm_after,
                global_gradient_scale=validation.global_gradient_scale,
                all_gradient_leaves_same_scale=validation.all_leaves_same_scale,
                maximum_leaf_scale_error=validation.maximum_leaf_scale_error,
                parameter_elements=Lux.parameterlength(parameters),
                parameter_array_count=length(array_leaves(parameters)),
            )
            push!(update_records, record)
            update == 1 && timing.time > FIRST_UPDATE_LIMIT_SECONDS && error("first update exceeded 60 seconds")
            if 6 <= update <= 25 && timing.time > WARM_UPDATE_LIMIT_SECONDS
                error("warm update $update exceeded one second")
            end
            if update == 25
                warm = [entry.seconds for entry in update_records[6:25]]
                median(warm) <= WARM_MEDIAN_LIMIT_SECONDS || error("update 6:25 median exceeded 0.25 seconds")
                projected_total_seconds = time() - started + median(warm) * (UPDATE_COUNT - update)
                projected_total_seconds <= HARD_WALL_SECONDS || error("projected total exceeds 12-minute wall")
            end
            if update in (500, 1000, 2000)
                save_snapshot(joinpath(output_directory, "correction_update$(update).jld2"), parameters, state, update, metadata)
            end
            if update == 1 || 6 <= update <= 25 || update % 10 == 0
                replace_json(progress_path, (; status="training", update, seconds=time() - started, last_update_seconds=timing.time, process_id=getpid(), validation_or_test_seed_loaded=false, game_run=false))
            end
        end
        training_slots = findall(subset.valid)
        base_slots = [slot for slot in training_slots if subset.rows[slot] in BASE_ROWS]
        dagger_slots = [slot for slot in training_slots if subset.rows[slot] in DAGGER_ROWS]
        base_diagnostics = role_diagnostics(data, model, parameters, state, base_slots)
        dagger_diagnostics = role_diagnostics(data, model, parameters, state, dagger_slots)
        weights_path = joinpath(output_directory, "correction_weights.npz")
        npzwrite(weights_path, collect_export_arrays(parameters))
        witness_rows = order[1:STATE_BATCH]
        witness_input, witness_mask, witness_old_q, _, _ = batch_input(data, witness_rows)
        witness_raw, _ = model(witness_input, parameters, state)
        witness_correction = reshape(Float32.(Array(witness_raw)), ACTIONS, STATE_BATCH)
        reference_path = joinpath(output_directory, "combined_reference.npz")
        npzwrite(reference_path, Dict(
            "board" => witness_input[1], "placement" => witness_input[2], "ren" => witness_input[3],
            "back_to_back" => witness_input[4], "tspin" => witness_input[5], "queue" => witness_input[6],
            "mask" => witness_mask, "old_q" => witness_old_q,
            "lux_correction" => witness_correction,
            "lux_combined" => witness_old_q .+ witness_correction,
            "source_rows" => Int32.(witness_rows),
        ))
        input_hashes_after = Dict(key => hex_sha256(path) for (key, path) in (
            "initializer" => initializer_path, "old_checkpoint" => old_checkpoint_path,
            "training_npz" => training_path, "freeze" => freeze_path, "order" => order_path,
        ))
        input_hashes_after == input_hashes_before || error("an immutable input changed during training")
        warm = [entry.seconds for entry in update_records[6:25]]
        phase = (;
            status="training_phase_complete", generated_at=string(now()), constants=expected_constants(),
            wall_seconds=time() - started, update_count=length(update_records), zero_gate,
            parameter_count=Lux.parameterlength(parameters), parameter_array_count,
            gradient_paths=reference_paths, gradient_elements_every_update=all(entry.gradient_elements == PARAMETER_COUNT for entry in update_records),
            clip_mode="single_global_tree_l2",
            global_gradient_norm_tolerance=1.0e-6,
            global_gradient_norms_finite_every_update=all(
                isfinite(entry.global_gradient_norm_before) &&
                isfinite(entry.global_gradient_norm_after) &&
                isfinite(entry.global_gradient_scale) for entry in update_records
            ),
            global_gradient_post_norm_within_limit_every_update=all(
                entry.global_gradient_norm_after <= 1.0 + 1.0e-6 for entry in update_records
            ),
            global_gradient_uniform_scale_every_update=all(
                entry.clip_mode == "single_global_tree_l2" &&
                entry.all_gradient_leaves_same_scale &&
                entry.maximum_leaf_scale_error <= 1.0e-6 for entry in update_records
            ),
            first_update_seconds=update_records[1].seconds, warm_update_seconds=warm,
            warm_median_seconds=median(warm), update_records,
            projected_total_seconds,
            base_role_diagnostics=base_diagnostics, dagger_role_diagnostics=dagger_diagnostics,
            dagger_three_step_off_policy_bias_disclosed=true,
            initializer_exposed_to_offline_rows=true, offline_role="reused_development_guard",
            input_hashes_before, input_hashes_after, immutable_inputs_unchanged=true,
            candidate_checkpoint=joinpath(output_directory, "correction_update2000.jld2"),
            candidate_checkpoint_sha256=hex_sha256(joinpath(output_directory, "correction_update2000.jld2")),
            weights_path, weights_sha256=hex_sha256(weights_path),
            reference_path, reference_sha256=hex_sha256(reference_path),
            validation_or_test_seed_loaded=false, game_run=false,
        )
        atomic_write_json(joinpath(output_directory, "training_phase.json"), phase)
        replace_json(progress_path, (; status="complete", update=UPDATE_COUNT, seconds=time() - started, validation_or_test_seed_loaded=false, game_run=false))
    catch exception
        failure_path = joinpath(output_directory, "training_failure.json")
        ispath(failure_path) || atomic_write_json(failure_path, (;
            status="training_failed", generated_at=string(now()), wall_seconds=time() - started,
            error=sprint(showerror, exception, catch_backtrace()),
            validation_or_test_seed_loaded=false, game_run=false,
        ))
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
