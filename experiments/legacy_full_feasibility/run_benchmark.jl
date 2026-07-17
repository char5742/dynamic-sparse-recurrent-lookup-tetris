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
using .LegacyFullFeasibilityContract

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function timing_record(timing)
    return (;
        seconds=timing.time,
        allocated_bytes=timing.bytes,
        gc_seconds=timing.gctime,
        compile_seconds=hasproperty(timing, :compile_time) ? timing.compile_time : nothing,
        recompile_seconds=hasproperty(timing, :recompile_time) ? timing.recompile_time : nothing,
    )
end

function require_safe_output(path::AbstractString)
    isdir(path) || error("output directory must already be created by the one-shot wrapper")
    allowed_root = lowercase(raw"D:\tetris-paper-plus") * "\\"
    startswith(lowercase(abspath(path)), allowed_root) || error(
        "F benchmark results must live under D:\\tetris-paper-plus"
    )
    for name in (
        "julia_phase.json",
        "temporary_updated_weights.npz",
        "temporary_updated_reference.npz",
        "openvino_phase.json",
        "final_result.json",
    )
        ispath(joinpath(path, name)) && error("refusing to overwrite $name")
    end
end

function validate_subset(data)
    Int.(vec(data["source_rows"])) == collect(SOURCE_ROWS) || error("source row mismatch")
    Int.(vec(data["episode_ids"])) == collect(EXPECTED_EPISODE_IDS) || error(
        "episode ID mismatch"
    )
    Int.(vec(data["seeds"])) == collect(EXPECTED_SEEDS) || error("seed mismatch")
    targets = Float32.(vec(data["targets"]))
    targets == collect(EXPECTED_TARGETS) || error(
        "frozen target mismatch: $targets != $(collect(EXPECTED_TARGETS))"
    )
    counts = Int.(vec(data["action_counts"]))
    selected = Int.(vec(data["selected_actions"]))
    counts == [51, 43, 43, 51, 51, 26] || error("candidate count contract mismatch")
    selected == [35, 40, 30, 35, 45, 21] || error("selected action contract mismatch")
    all(selected .<= counts) || error("selected action exceeds candidate count")
    size(data["boards"]) == (6, 24, 10, 1) || error("board subset shape mismatch")
    size(data["placements"], 1) == 6 || error("placement subset shape mismatch")
    size(data["placements"], 2) == maximum(counts) || error(
        "placement storage width mismatch"
    )
    return (; counts, selected, targets)
end

function row_input(data, slot::Int, count::Int)
    board_one = permutedims(@view(data["boards"][slot:slot, :, :, :]), (2, 3, 4, 1))
    board = repeat(Float32.(board_one); outer=(1, 1, 1, count))
    placement = permutedims(
        Float32.(@view(data["placements"][slot, 1:count, :, :, :])),
        (2, 3, 4, 1),
    )
    ren = fill(Float32(data["ren"][slot]), 1, count)
    back_to_back = fill(Float32(data["back_to_back"][slot]), 1, count)
    tspin = reshape(Float32.(@view(data["tspin"][slot, 1:count])), 1, count)
    queue_one = Float32.(@view(data["queues"][slot, :, :]))
    queue = repeat(reshape(queue_one, 7, 6, 1); outer=(1, 1, count))
    return (board, placement, ren, back_to_back, tspin, queue)
end

function slice_input(input, range::UnitRange{Int})
    return (
        @view(input[1][:, :, :, range]),
        @view(input[2][:, :, :, range]),
        @view(input[3][:, range]),
        @view(input[4][:, range]),
        @view(input[5][:, range]),
        @view(input[6][:, :, range]),
    )
end

function historical_scores(model, parameters, fixed_state, input, count::Int)
    output = Vector{Float32}(undef, count)
    for range in chunk_ranges(count)
        values, returned_state = model(
            slice_input(input, range), parameters, fixed_state
        )
        returned_state === fixed_state || nothing # State value is checked separately.
        output[range] .= vec(Array(values))
    end
    return output
end

function selected_chunk(input, selected::Int, count::Int)
    start = fld(selected - 1, LEGACY_BATCH) * LEGACY_BATCH + 1
    stop = min(start + LEGACY_BATCH - 1, count)
    range = start:stop
    return slice_input(input, range), selected - start + 1, range
end

function selected_huber_objective(
    model, parameters, fixed_state, input, selected_local::Int, target::Float32
)
    prediction, _ = model(input, parameters, fixed_state)
    return huber_scalar(prediction[1, selected_local], target)
end

function complete_update(
    model,
    parameters,
    fixed_state,
    optimizer_state,
    input,
    selected_local::Int,
    target::Float32,
)
    differentiated = Zygote.withgradient(parameters) do current_parameters
        selected_huber_objective(
            model,
            current_parameters,
            fixed_state,
            input,
            selected_local,
            target,
        )
    end
    gradient = only(differentiated.grad)
    next_optimizer_state, next_parameters = Optimisers.update(
        optimizer_state, parameters, gradient
    )
    return (;
        loss=Float64(differentiated.val),
        gradient,
        optimizer_state=next_optimizer_state,
        parameters=next_parameters,
    )
end

function collect_arrays!(output::Dict{String,Array}, value, path::AbstractString="")
    if value isa AbstractArray
        output[path] = Array(value)
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(path) ? string(name) : string(path, ".", name)
            collect_arrays!(output, getproperty(value, name), child)
        end
    end
    return output
end

function write_partial(path, stage, started, details; stage_started_unix=time())
    atomic_write_json(
        path,
        (;
            status="running",
            stage,
            generated_at=string(now()),
            stage_started_unix,
            wall_seconds=time() - started,
            details,
        ),
    )
end

function main(args=ARGS)
    paths = resolve_benchmark_paths(args, ENV)
    output_directory = paths.output_directory
    subset_path = paths.subset_path
    checkpoint_path = paths.checkpoint_path
    freeze_path = paths.freeze_path
    require_safe_output(output_directory)
    isfile(freeze_path) || error("missing pre-execution freeze: $freeze_path")
    require_hash(checkpoint_path, CHECKPOINT_SHA256, "legacy checkpoint")
    partial_path = joinpath(output_directory, "julia_phase.json")
    started = time()

    VERSION == v"1.12.6" || error("Julia version mismatch: $(VERSION)")
    string(Base.pkgversion(Lux)) == "1.31.4" || error("Lux version mismatch")
    string(Base.pkgversion(Zygote)) == "0.7.11" || error("Zygote version mismatch")
    BLAS.set_num_threads(parse(Int, get(ENV, "F_BLAS_THREADS", "10")))

    data = npzread(subset_path)
    subset = validate_subset(data)
    size(data["terminal_mask"]) == (6, N_STEP) || error("terminal mask shape mismatch")
    any(Bool.(data["terminal_mask"])) && error("n-step source crosses terminal row")
    write_partial(
        partial_path,
        "subset_validated",
        started,
        (; constants=expected_constants(), subset_path, freeze_path),
    )

    parameters, checkpoint_state = jldopen(checkpoint_path, "r") do file
        modernize_legacy_parameters(file["ps"]), file["st"]
    end
    model = LegacyQNetwork()
    parameter_count = Lux.parameterlength(parameters)
    parameter_count == LEGACY_PARAMETER_COUNT || error(
        "legacy parameter count $parameter_count != $LEGACY_PARAMETER_COUNT"
    )
    fixed_state = Lux.testmode(checkpoint_state)
    fixed_state_before = deepcopy(fixed_state)
    tree_all_finite(parameters) || error("checkpoint contains non-finite parameters")
    tree_all_finite(fixed_state) || error("checkpoint contains non-finite state")

    inputs = [row_input(data, slot, subset.counts[slot]) for slot in 1:6]
    zero_errors = Float64[]
    zero_outputs = Vector{Vector{Float32}}()
    zero_seconds = @elapsed begin
        for slot in 1:6
            values = historical_scores(
                model, parameters, fixed_state, inputs[slot], subset.counts[slot]
            )
            reference = Float32.(@view(data["stored_q"][slot, 1:subset.counts[slot]]))
            all(isfinite, values) || error("non-finite zero-update output at slot $slot")
            push!(zero_outputs, values)
            push!(zero_errors, maximum(abs, Float64.(values) .- Float64.(reference)))
        end
    end
    maximum(zero_errors) <= ZERO_TOLERANCE || error(
        "zero-update max error $(maximum(zero_errors)) exceeds $ZERO_TOLERANCE"
    )
    tree_max_abs_difference(fixed_state_before, fixed_state) == 0.0 || error(
        "Lux.testmode state mutated during zero-update audit"
    )
    write_partial(
        partial_path,
        "zero_update_passed",
        started,
        (; zero_errors, zero_seconds, candidate_counts=subset.counts),
    )

    specialization_stage_started = time()
    write_partial(
        partial_path,
        "specialization_running",
        started,
        (; timeout_seconds=MAX_FIRST_SPECIALIZATION_SECONDS);
        stage_started_unix=specialization_stage_started,
    )
    optimizer = Optimisers.AdamW(
        LEARNING_RATE, BETAS, WEIGHT_DECAY; couple=true
    )
    optimizer_setup_timing = @timed Optimisers.setup(optimizer, parameters)
    optimizer_state = optimizer_setup_timing.value
    optimizer_array_elements = tree_array_elements(optimizer_state)
    optimizer_array_elements == 2 * parameter_count || error(
        "optimizer state does not cover every parameter: $optimizer_array_elements elements"
    )
    tree_all_finite(optimizer_state) || error("non-finite initial optimizer state")

    compile_input, compile_selected_local, compile_range = selected_chunk(
        inputs[1], subset.selected[1], subset.counts[1]
    )
    specialization_timing = @timed complete_update(
        model,
        parameters,
        fixed_state,
        optimizer_state,
        compile_input,
        compile_selected_local,
        subset.targets[1],
    )
    first_specialization_seconds =
        optimizer_setup_timing.time + specialization_timing.time
    specialization_record = timing_record(specialization_timing)
    write_partial(
        partial_path,
        "specialization_scanning",
        started,
        (; first_specialization_seconds, timeout_seconds=MAX_FIRST_SPECIALIZATION_SECONDS),
    )
    first_specialization_seconds <= MAX_FIRST_SPECIALIZATION_SECONDS || error(
        "first full specialization $first_specialization_seconds s exceeds $MAX_FIRST_SPECIALIZATION_SECONDS s"
    )
    probe = specialization_timing.value
    specialization_gradient_elements = 0
    specialization_scan_seconds = @elapsed begin
        isfinite(probe.loss) || error("non-finite specialization loss")
        tree_all_finite(probe.gradient) || error("non-finite specialization gradient")
        gradient_covers_parameters(parameters, probe.gradient) || error(
            "specialization gradient has a missing/nothing/mismatched parameter leaf"
        )
        specialization_gradient_elements = tree_array_elements(probe.gradient)
        specialization_gradient_elements == parameter_count || error(
            "specialization gradient covers $specialization_gradient_elements / $parameter_count parameters"
        )
        tree_all_finite(probe.parameters) || error("non-finite specialization parameters")
        tree_all_finite(probe.optimizer_state) || error(
            "non-finite specialization optimizer state"
        )
    end
    # The specialization update is deliberately discarded.  Every preregistered
    # row below is therefore consumed exactly once from the original checkpoint.
    probe = nothing
    specialization_timing = nothing
    GC.gc(true)
    write_partial(
        partial_path,
        "specialization_passed",
        started,
        (;
            optimizer_setup=timing_record(optimizer_setup_timing),
            specialization=specialization_record,
            first_specialization_seconds,
            specialization_scan_seconds,
            specialization_gradient_elements,
            compile_selected_range=collect(compile_range),
            optimizer_array_elements,
        ),
    )

    current_parameters = parameters
    current_optimizer_state = optimizer_state
    head_bias_before = Float64(parameters.score_net.layer_3.bias[1])
    update_records = NamedTuple[]
    for slot in 1:6
        input, selected_local, range = selected_chunk(
            inputs[slot], subset.selected[slot], subset.counts[slot]
        )
        update_stage_started = time()
        write_partial(
            partial_path,
            "warm_update_$(slot)_running",
            started,
            (;
                update=slot,
                source_row=SOURCE_ROWS[slot],
                timeout_seconds=MAX_WARM_UPDATE_SECONDS,
            );
            stage_started_unix=update_stage_started,
        )
        timing = @timed complete_update(
            model,
            current_parameters,
            fixed_state,
            current_optimizer_state,
            input,
            selected_local,
            subset.targets[slot],
        )
        write_partial(
            partial_path,
            "warm_update_$(slot)_scanning",
            started,
            (; update=slot, update_seconds=timing.time),
        )
        timing.time <= MAX_WARM_UPDATE_SECONDS || error(
            "warm update $slot took $(timing.time) s, exceeding $MAX_WARM_UPDATE_SECONDS s"
        )
        result = timing.value
        gradient_elements = 0
        scan_seconds = @elapsed begin
            isfinite(result.loss) || error("non-finite loss at update $slot")
            tree_all_finite(result.gradient) || error("non-finite gradient at update $slot")
            gradient_covers_parameters(current_parameters, result.gradient) || error(
                "update $slot gradient has a missing/nothing/mismatched parameter leaf"
            )
            gradient_elements = tree_array_elements(result.gradient)
            gradient_elements == parameter_count || error(
                "update $slot gradient covers $gradient_elements / $parameter_count parameters"
            )
            tree_all_finite(result.parameters) || error("non-finite parameters at update $slot")
            tree_all_finite(result.optimizer_state) || error(
                "non-finite optimizer state at update $slot"
            )
        end
        gradient_norm = sqrt(tree_sum_abs2(result.gradient))
        isfinite(gradient_norm) || error("non-finite gradient norm at update $slot")
        push!(
            update_records,
            (;
                update=slot,
                source_row=SOURCE_ROWS[slot],
                episode_id=EXPECTED_EPISODE_IDS[slot],
                seed=EXPECTED_SEEDS[slot],
                action_count=subset.counts[slot],
                selected_action=subset.selected[slot],
                selected_chunk=collect(range),
                selected_local,
                target=Float64(subset.targets[slot]),
                loss=result.loss,
                gradient_norm,
                gradient_elements,
                scan_seconds,
                timing=timing_record(timing),
            ),
        )
        current_parameters = result.parameters
        current_optimizer_state = result.optimizer_state
        write_partial(
            partial_path,
            "warm_update_$slot",
            started,
            (; updates=update_records),
        )
    end

    head_bias_after = Float64(current_parameters.score_net.layer_3.bias[1])
    head_bias_after != head_bias_before || error("six updates did not change proof parameter")
    tree_max_abs_difference(fixed_state_before, fixed_state) == 0.0 || error(
        "Lux.testmode running state changed during updates"
    )
    warm_seconds = [record.timing.seconds for record in update_records]
    warm_median_seconds = median(warm_seconds)

    weight_path = joinpath(output_directory, "temporary_updated_weights.npz")
    reference_path = joinpath(output_directory, "temporary_updated_reference.npz")
    isfile(weight_path) && error("refusing to overwrite updated weight output")
    isfile(reference_path) && error("refusing to overwrite updated reference output")
    updated_reference = Float32[]
    reference_seconds = @elapsed begin
        updated_reference = historical_scores(
            model,
            current_parameters,
            fixed_state,
            inputs[1],
            subset.counts[1],
        )
        all(isfinite, updated_reference) || error("non-finite updated reference output")
        npzwrite(
            reference_path,
            Dict(
                "board" => inputs[1][1],
                "placement" => inputs[1][2],
                "ren" => inputs[1][3],
                "back_to_back" => inputs[1][4],
                "tspin" => inputs[1][5],
                "queue" => inputs[1][6],
                "lux_output" => updated_reference,
                "action_count" => Int32[subset.counts[1]],
            ),
        )
    end
    arrays = Dict{String,Array}()
    export_seconds = @elapsed begin
        collect_arrays!(arrays, current_parameters, "ps")
        collect_arrays!(arrays, fixed_state, "st")
        length(arrays) == 353 || error("updated export array count $(length(arrays)) != 353")
        npzwrite(weight_path, arrays)
    end

    final = (;
        status="julia_phase_complete",
        generated_at=string(now()),
        wall_seconds=time() - started,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        zygote_version=string(Base.pkgversion(Zygote)),
        optimisers_version=string(Base.pkgversion(Optimisers)),
        backend="Lux+Zygote",
        threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        constants=expected_constants(),
        parameter_count,
        optimizer_array_elements,
        state_mode="Lux.testmode; fixed state passed and returned state discarded",
        zero_update=(;
            seconds=zero_seconds,
            per_row_max_abs_error=zero_errors,
            max_abs_error=maximum(zero_errors),
            tolerance=ZERO_TOLERANCE,
        ),
        optimizer_setup=timing_record(optimizer_setup_timing),
        specialization=specialization_record,
        first_specialization_seconds,
        specialization_scan_seconds,
        specialization_gradient_elements,
        warm_updates=update_records,
        warm_median_seconds,
        head_bias_before,
        head_bias_after,
        head_bias_abs_change=abs(head_bias_after - head_bias_before),
        fixed_state_max_abs_change=tree_max_abs_difference(fixed_state_before, fixed_state),
        reference_seconds,
        export_seconds,
        export_array_count=length(arrays),
        temporary_weight_path=weight_path,
        temporary_reference_path=reference_path,
        temporary_outputs_promoted=false,
        score_or_game_evaluation_run=false,
        validation_or_test_data_used=false,
        sys_maxrss=Sys.maxrss(),
        completion_reason="six preregistered warm updates and one temporary export completed",
    )
    atomic_write_json(partial_path, final)
    println(JSON3.write((; status=final.status, output=partial_path)))
    return final
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
