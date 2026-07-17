module LegacyPartialTailTDContract

using JSON3
using SHA

const EXPERIMENT_ID = "legacy_partial_tail_td_P1"
const AUTHORIZATION_REPORT_SHA256 =
    "a079330917571824fdbb0dd92d37db92dc1df9701012206bb27bc672d24ca906"
const CHECKPOINT_SHA256 =
    "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
const DATASET_SHA256 =
    "e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d"
const CANONICAL_BASELINE_WEIGHTS_SHA256 =
    "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"

const TRAINABLE_PATHS = (
    "board_net.resblocks.layer_29",
    "board_net.resblocks.layer_31",
    "board_net.conv2",
    "board_net.norm2",
    "score_net",
)
const LEGACY_PARAMETER_COUNT = 20_787_454
const TRAINABLE_PARAMETER_COUNT = 2_949_508
const FROZEN_PARAMETER_COUNT = 17_837_946
const OPTIMIZER_MOMENT_ELEMENTS = 5_899_016
const LEGACY_BATCH = 16

const TRAIN_ROWS = 1:1500
const TRAIN_EPISODES = 1:6
const TRAIN_SEEDS = 5742:5747
const OFFLINE_ROWS = 1501:2000
const OFFLINE_EPISODES = 7:8
const OFFLINE_SEEDS = 5748:5749
const DEVELOPMENT_SEEDS = (5756, 5757)
const DATA_ORDER_SEED = UInt64(0x1313_2026)
const UPDATE_COUNT = 300
const STEP0_WITNESS_ROW = 1055
const STEP0_WITNESS_EPISODE = 5
const STEP0_WITNESS_STEP = 55
const STEP0_WITNESS_COUNT = 52
const STEP0_WITNESS_SELECTED = 11
const N_STEP = 3
const DISCOUNT = 0.997f0
const HUBER_DELTA = 1.0f0
const ANCHOR_WEIGHT = 1.0f0

const LEARNING_RATE = 1.0f-5
const BETAS = (0.9, 0.999)
const WEIGHT_DECAY = 1.0f-4
const FINITE_DIFFERENCE_EPSILON = 1.0f-3
const FINITE_DIFFERENCE_ABS_TOLERANCE = 1.0e-3
const FINITE_DIFFERENCE_REL_TOLERANCE = 0.02

const SPLIT_TAIL_TOLERANCE = 1.0e-6
const STORED_Q_TOLERANCE = 1.0e-2
const CPU_TOLERANCE = 1.0e-4
const NPU_TOLERANCE = 1.0e-2
const OFFLINE_TOP1_AGREEMENT = 0.95
const DEVELOPMENT_MEAN_MINIMUM = 500.0

const FIRST_UPDATE_SECONDS = 180.0
const WARM_UPDATE_SECONDS = 15.0
const WARM_MEDIAN_SECONDS = 4.5
const HARD_WALL_SECONDS = 35 * 60
const MAX_PEAK_WORKING_SET_BYTES = 8 * 1024^3
const OFFLINE_RESERVE_SECONDS = 100.0
const EXPORT_RESERVE_SECONDS = 120.0
const ACTOR_SECONDS_PER_STEP = 0.411
const ACTOR_EVALUATION_STEPS = 400
const DEVELOPMENT_PIECES = 100
const GLOBAL_ONE_SHOT_MARKER =
    raw"D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json"

hex_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function require_hash(path::AbstractString, expected::AbstractString, label::AbstractString)
    isfile(path) || error("missing $label: $path")
    observed = hex_sha256(path)
    observed == lowercase(expected) || error(
        "$label SHA-256 mismatch: expected $(lowercase(expected)), observed $observed"
    )
    return observed
end

function expected_constants()
    return (;
        experiment_id=EXPERIMENT_ID,
        authorization_report_sha256=AUTHORIZATION_REPORT_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
        dataset_sha256=DATASET_SHA256,
        canonical_baseline_weights_sha256=CANONICAL_BASELINE_WEIGHTS_SHA256,
        trainable_paths=collect(TRAINABLE_PATHS),
        legacy_parameter_count=LEGACY_PARAMETER_COUNT,
        trainable_parameter_count=TRAINABLE_PARAMETER_COUNT,
        frozen_parameter_count=FROZEN_PARAMETER_COUNT,
        optimizer_moment_elements=OPTIMIZER_MOMENT_ELEMENTS,
        train_rows=[first(TRAIN_ROWS), last(TRAIN_ROWS)],
        train_episodes=collect(TRAIN_EPISODES),
        train_seeds=collect(TRAIN_SEEDS),
        offline_rows=[first(OFFLINE_ROWS), last(OFFLINE_ROWS)],
        offline_episodes=collect(OFFLINE_EPISODES),
        offline_seeds=collect(OFFLINE_SEEDS),
        development_seeds=collect(DEVELOPMENT_SEEDS),
        data_order_rng="Xoshiro(0x1313_2026)",
        update_count=UPDATE_COUNT,
        step0_witness=(;
            source_row=STEP0_WITNESS_ROW,
            episode=STEP0_WITNESS_EPISODE,
            step=STEP0_WITNESS_STEP,
            candidate_count=STEP0_WITNESS_COUNT,
            selected_action=STEP0_WITNESS_SELECTED,
        ),
        n_step=N_STEP,
        discount=DISCOUNT,
        huber_delta=HUBER_DELTA,
        anchor_weight=ANCHOR_WEIGHT,
        optimizer="Zygote AdamW(couple=true)",
        learning_rate=LEARNING_RATE,
        betas=collect(BETAS),
        weight_decay=WEIGHT_DECAY,
        historical_chunk=LEGACY_BATCH,
        split_tail_tolerance=SPLIT_TAIL_TOLERANCE,
        stored_q_tolerance=STORED_Q_TOLERANCE,
        cpu_tolerance=CPU_TOLERANCE,
        npu_tolerance=NPU_TOLERANCE,
        offline_top1_agreement=OFFLINE_TOP1_AGREEMENT,
        first_update_seconds=FIRST_UPDATE_SECONDS,
        warm_update_seconds=WARM_UPDATE_SECONDS,
        warm_median_seconds=WARM_MEDIAN_SECONDS,
        hard_wall_seconds=HARD_WALL_SECONDS,
        max_peak_working_set_bytes=MAX_PEAK_WORKING_SET_BYTES,
        global_one_shot_marker=GLOBAL_ONE_SHOT_MARKER,
    )
end

function chunk_ranges(count::Integer; batch::Integer=LEGACY_BATCH)
    count > 0 || throw(ArgumentError("candidate count must be positive"))
    batch > 0 || throw(ArgumentError("batch must be positive"))
    return [start:min(start + batch - 1, count) for start in 1:batch:count]
end

function selected_chunk_range(selected::Integer, count::Integer)
    1 <= selected <= count || throw(ArgumentError("selected action is out of range"))
    start = fld(selected - 1, LEGACY_BATCH) * LEGACY_BATCH + 1
    return start:min(start + LEGACY_BATCH - 1, count)
end

function huber_scalar(prediction::Real, target::Real; delta::Float32=HUBER_DELTA)
    difference = Float32(prediction) - Float32(target)
    absolute = abs(difference)
    return absolute <= delta ?
           0.5f0 * difference^2 : delta * (absolute - 0.5f0 * delta)
end

function anchored_loss(scores, selected_local::Integer, target::Real, old_scores)
    length(scores) == length(old_scores) || throw(DimensionMismatch("anchor shape mismatch"))
    1 <= selected_local <= length(scores) || throw(ArgumentError("local action is out of range"))
    selected_loss = huber_scalar(scores[selected_local], target)
    anchor = sum(
        huber_scalar(score, old_score) for (score, old_score) in zip(scores, old_scores);
        init=0.0f0,
    ) / Float32(length(scores))
    return selected_loss + ANCHOR_WEIGHT * anchor
end

function frozen_nstep_target(rewards, terminal, bootstrap_q)
    length(rewards) == N_STEP || throw(DimensionMismatch("P1 requires n=3 rewards"))
    length(terminal) == N_STEP || throw(DimensionMismatch("P1 requires n=3 terminal flags"))
    result = 0.0f0
    factor = 1.0f0
    for index in eachindex(rewards)
        isfinite(rewards[index]) || error("non-finite reward")
        result += factor * Float32(rewards[index])
        terminal[index] && return result
        factor *= DISCOUNT
    end
    isfinite(bootstrap_q) || error("non-finite bootstrap Q")
    return result + factor * Float32(bootstrap_q)
end

function tree_array_elements(value)
    if value isa AbstractArray
        return length(value)
    elseif value isa NamedTuple || value isa Tuple
        return sum(tree_array_elements, values(value); init=0)
    elseif isstructtype(typeof(value)) &&
           !(value isa Number || value isa AbstractString || value isa Symbol ||
             value isa Function || value isa Type)
        return sum(
            index -> tree_array_elements(getfield(value, index)),
            1:fieldcount(typeof(value));
            init=0,
        )
    end
    return 0
end

function tree_all_finite(value)
    if value isa AbstractArray
        return all(isfinite, value)
    elseif value isa Number
        return isfinite(value)
    elseif value === nothing || value isa AbstractString || value isa Symbol ||
           value isa Function || value isa Type
        return true
    elseif value isa NamedTuple || value isa Tuple
        return all(tree_all_finite, values(value))
    elseif isstructtype(typeof(value))
        return all(index -> tree_all_finite(getfield(value, index)), 1:fieldcount(typeof(value)))
    end
    return true
end

has_array_leaf(value) = tree_array_elements(value) > 0

function gradient_array_inventory!(records, value, path::AbstractString="")
    if value isa AbstractArray
        push!(records, (;
            path,
            shape=collect(size(value)),
            elements=length(value),
            finite=all(isfinite, value),
        ))
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(path) ? string(name) : "$path.$name"
            gradient_array_inventory!(records, getproperty(value, name), child)
        end
    elseif value isa Tuple
        for (index, child_value) in enumerate(value)
            child = isempty(path) ? string(index) : "$path.$index"
            gradient_array_inventory!(records, child_value, child)
        end
    end
    return records
end

function validate_gradient(parameters, gradient; root::AbstractString="trainable")
    inventory = NamedTuple[]
    gradient_array_inventory!(inventory, gradient, root)
    first_mismatch = Ref{Union{Nothing,String}}(nothing)
    function visit(parameter, derivative, path)
        first_mismatch[] === nothing || return
        if parameter isa AbstractArray
            if !(derivative isa AbstractArray)
                first_mismatch[] = "$path: missing array gradient"
            elseif size(derivative) != size(parameter)
                first_mismatch[] = "$path: gradient shape $(size(derivative)) != $(size(parameter))"
            elseif !all(isfinite, derivative)
                first_mismatch[] = "$path: non-finite array gradient"
            end
            return
        end
        if !has_array_leaf(parameter)
            if derivative === nothing || !has_array_leaf(derivative)
                return
            end
            first_mismatch[] = "$path: gradient exists for a no-parameter subtree"
            return
        end
        if parameter isa NamedTuple
            derivative isa NamedTuple || begin
                first_mismatch[] = "$path: expected NamedTuple gradient"
                return
            end
            for name in keys(parameter)
                child_path = isempty(path) ? string(name) : "$path.$name"
                child_gradient = hasproperty(derivative, name) ? getproperty(derivative, name) : nothing
                visit(getproperty(parameter, name), child_gradient, child_path)
            end
            if first_mismatch[] === nothing
                for name in keys(derivative)
                    if !hasproperty(parameter, name) && has_array_leaf(getproperty(derivative, name))
                        first_mismatch[] = "$path.$name: unexpected gradient array subtree"
                        break
                    end
                end
            end
        elseif parameter isa Tuple
            derivative isa Tuple || begin
                first_mismatch[] = "$path: expected Tuple gradient"
                return
            end
            for index in eachindex(parameter)
                child_gradient = index <= length(derivative) ? derivative[index] : nothing
                visit(parameter[index], child_gradient, "$path.$index")
            end
        else
            first_mismatch[] = "$path: unsupported parameter container $(typeof(parameter))"
        end
    end
    visit(parameters, gradient, root)
    gradient_elements = sum(record.elements for record in inventory; init=0)
    if first_mismatch[] === nothing && gradient_elements != tree_array_elements(parameters)
        first_mismatch[] = "$root: gradient element count $gradient_elements != $(tree_array_elements(parameters))"
    end
    return (;
        valid=first_mismatch[] === nothing,
        first_mismatch=first_mismatch[],
        parameter_elements=tree_array_elements(parameters),
        gradient_elements,
        inventory,
    )
end

function array_leaf_hashes!(output::Dict{String,String}, value, path::AbstractString="")
    if value isa AbstractArray
        bytes = reinterpret(UInt8, vec(Array(value)))
        output[path] = bytes2hex(sha256(bytes))
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(path) ? string(name) : "$path.$name"
            array_leaf_hashes!(output, getproperty(value, name), child)
        end
    elseif value isa Tuple
        for (index, child_value) in enumerate(value)
            child = isempty(path) ? string(index) : "$path.$index"
            array_leaf_hashes!(output, child_value, child)
        end
    end
    return output
end

array_leaf_hashes(value; root::AbstractString="") =
    array_leaf_hashes!(Dict{String,String}(), value, root)

function frozen_parameter_hashes(parameters)
    all_hashes = array_leaf_hashes(parameters)
    return Dict(
        path => digest for (path, digest) in all_hashes
        if !any(prefix -> path == prefix || startswith(path, prefix * "."), TRAINABLE_PATHS)
    )
end

function finite_difference_pass(fd::Real, ad::Real)
    tolerance = FINITE_DIFFERENCE_ABS_TOLERANCE +
                FINITE_DIFFERENCE_REL_TOLERANCE * max(abs(fd), abs(ad))
    return isfinite(fd) && isfinite(ad) && abs(fd - ad) <= tolerance, tolerance
end

function projected_total_seconds(
    elapsed_through_warm::Real, warm_median::Real; completed_updates::Integer=7
)
    1 <= completed_updates <= UPDATE_COUNT || throw(ArgumentError("invalid completed update count"))
    return Float64(elapsed_through_warm) +
           (UPDATE_COUNT - completed_updates) * Float64(warm_median) +
           OFFLINE_RESERVE_SECONDS + EXPORT_RESERVE_SECONDS +
           ACTOR_EVALUATION_STEPS * ACTOR_SECONDS_PER_STEP
end

function atomic_write_json(path::AbstractString, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, value)
    end
    mv(temporary, path; force=true)
    return path
end

export EXPERIMENT_ID,
    AUTHORIZATION_REPORT_SHA256,
    CHECKPOINT_SHA256,
    DATASET_SHA256,
    CANONICAL_BASELINE_WEIGHTS_SHA256,
    TRAINABLE_PATHS,
    LEGACY_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    FROZEN_PARAMETER_COUNT,
    OPTIMIZER_MOMENT_ELEMENTS,
    LEGACY_BATCH,
    TRAIN_ROWS,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
    OFFLINE_ROWS,
    OFFLINE_EPISODES,
    OFFLINE_SEEDS,
    DEVELOPMENT_SEEDS,
    DATA_ORDER_SEED,
    UPDATE_COUNT,
    STEP0_WITNESS_ROW,
    STEP0_WITNESS_EPISODE,
    STEP0_WITNESS_STEP,
    STEP0_WITNESS_COUNT,
    STEP0_WITNESS_SELECTED,
    N_STEP,
    DISCOUNT,
    HUBER_DELTA,
    ANCHOR_WEIGHT,
    LEARNING_RATE,
    BETAS,
    WEIGHT_DECAY,
    FINITE_DIFFERENCE_EPSILON,
    SPLIT_TAIL_TOLERANCE,
    STORED_Q_TOLERANCE,
    CPU_TOLERANCE,
    NPU_TOLERANCE,
    OFFLINE_TOP1_AGREEMENT,
    DEVELOPMENT_MEAN_MINIMUM,
    FIRST_UPDATE_SECONDS,
    WARM_UPDATE_SECONDS,
    WARM_MEDIAN_SECONDS,
    HARD_WALL_SECONDS,
    MAX_PEAK_WORKING_SET_BYTES,
    DEVELOPMENT_PIECES,
    GLOBAL_ONE_SHOT_MARKER,
    expected_constants,
    hex_sha256,
    require_hash,
    chunk_ranges,
    selected_chunk_range,
    huber_scalar,
    anchored_loss,
    frozen_nstep_target,
    tree_array_elements,
    tree_all_finite,
    has_array_leaf,
    validate_gradient,
    array_leaf_hashes,
    frozen_parameter_hashes,
    finite_difference_pass,
    projected_total_seconds,
    atomic_write_json

end
