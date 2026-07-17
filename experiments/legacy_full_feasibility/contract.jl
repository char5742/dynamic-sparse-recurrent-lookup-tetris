module LegacyFullFeasibilityContract

using JSON3
using SHA

const CHECKPOINT_SHA256 =
    "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
const DATASET_SHA256 =
    "e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d"
const SOURCE_ROWS = (1, 251, 501, 751, 1001, 1251)
const EXPECTED_EPISODE_IDS = (1, 2, 3, 4, 5, 6)
const EXPECTED_SEEDS = (5742, 5743, 5744, 5745, 5746, 5747)
const EXPECTED_TARGETS =
    (5.6397543f0, 5.567213f0, 5.616358f0, 5.702811f0, 5.58614f0, 5.53066f0)
const N_STEP = 3
const DISCOUNT = 0.997f0
const LEARNING_RATE = 1.0f-5
const BETAS = (0.9, 0.999)
const WEIGHT_DECAY = 1.0f-4
const LEGACY_PARAMETER_COUNT = 20_787_454
const LEGACY_BATCH = 16
const ZERO_TOLERANCE = 1.0f-2
const CPU_TOLERANCE = 1.0f-4
const NPU_TOLERANCE = 1.0f-2
const MAX_FIRST_SPECIALIZATION_SECONDS = 300.0
const MAX_WARM_UPDATE_SECONDS = 120.0
const MAX_PEAK_WORKING_SET_BYTES = 8 * 1024^3
const HARD_WALL_SECONDS = 25 * 60
const ACTOR_SECONDS_PER_STEP = 0.411
const ACTOR_REFRESH_COUNT = 4
const TARGET_T1000_SECONDS = 1800.0
const GLOBAL_ONE_SHOT_MARKER =
    raw"D:\tetris-paper-plus\runs\legacy_full_feasibility_F.started.json"

hex_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function expected_constants()
    return (;
        checkpoint_sha256=CHECKPOINT_SHA256,
        dataset_sha256=DATASET_SHA256,
        source_rows=collect(SOURCE_ROWS),
        expected_episode_ids=collect(EXPECTED_EPISODE_IDS),
        expected_seeds=collect(EXPECTED_SEEDS),
        expected_targets=collect(EXPECTED_TARGETS),
        n_step=N_STEP,
        discount=DISCOUNT,
        learning_rate=LEARNING_RATE,
        betas=collect(BETAS),
        weight_decay=WEIGHT_DECAY,
        legacy_parameter_count=LEGACY_PARAMETER_COUNT,
        legacy_batch=LEGACY_BATCH,
        zero_tolerance=ZERO_TOLERANCE,
        cpu_tolerance=CPU_TOLERANCE,
        npu_tolerance=NPU_TOLERANCE,
        max_first_specialization_seconds=MAX_FIRST_SPECIALIZATION_SECONDS,
        max_warm_update_seconds=MAX_WARM_UPDATE_SECONDS,
        max_peak_working_set_bytes=MAX_PEAK_WORKING_SET_BYTES,
        hard_wall_seconds=HARD_WALL_SECONDS,
        actor_seconds_per_step=ACTOR_SECONDS_PER_STEP,
        actor_refresh_count=ACTOR_REFRESH_COUNT,
        target_t1000_seconds=TARGET_T1000_SECONDS,
        global_one_shot_marker=GLOBAL_ONE_SHOT_MARKER,
    )
end

function resolve_benchmark_paths(args, environment)
    names = ("F_OUTPUT_DIRECTORY", "F_SUBSET_PATH", "F_CHECKPOINT_PATH", "F_FREEZE_PATH")
    values = if isempty(args)
        [get(environment, name, "") for name in names]
    elseif length(args) == 4
        collect(args)
    else
        error("F benchmark accepts either zero argv paths (environment contract) or exactly four")
    end
    all(value -> !isempty(value), values) || error("missing F benchmark path environment variable")
    return (;
        output_directory=abspath(values[1]),
        subset_path=abspath(values[2]),
        checkpoint_path=abspath(values[3]),
        freeze_path=abspath(values[4]),
    )
end

function require_hash(path::AbstractString, expected::AbstractString, label::AbstractString)
    isfile(path) || error("missing $label: $path")
    observed = hex_sha256(path)
    observed == lowercase(expected) || error(
        "$label SHA-256 mismatch: expected $(lowercase(expected)), observed $observed"
    )
    return observed
end

function chunk_ranges(count::Integer; batch::Integer=LEGACY_BATCH)
    count > 0 || throw(ArgumentError("candidate count must be positive"))
    batch > 0 || throw(ArgumentError("batch must be positive"))
    return [start:min(start + batch - 1, count) for start in 1:batch:count]
end

function huber_scalar(prediction::Real, target::Real; delta::Float32=1.0f0)
    difference = Float32(prediction) - Float32(target)
    absolute = abs(difference)
    return absolute <= delta ?
           0.5f0 * difference^2 : delta * (absolute - 0.5f0 * delta)
end

function frozen_nstep_target(rewards, bootstrap_q; discount::Float32=DISCOUNT)
    length(rewards) == N_STEP || throw(DimensionMismatch("F requires exactly n=3 rewards"))
    all(isfinite, rewards) || error("non-finite reward")
    isfinite(bootstrap_q) || error("non-finite stored bootstrap Q")
    result = 0.0f0
    factor = 1.0f0
    for reward in rewards
        result += factor * Float32(reward)
        factor *= discount
    end
    return result + factor * Float32(bootstrap_q)
end

function tree_all_finite(value)
    if value isa AbstractArray
        return all(isfinite, value)
    elseif value isa Number
        return isfinite(value)
    elseif value === nothing || value isa Symbol || value isa AbstractString ||
           value isa Function || value isa Type
        return true
    elseif value isa NamedTuple || value isa Tuple
        return all(tree_all_finite, values(value))
    elseif isstructtype(typeof(value))
        return all(index -> tree_all_finite(getfield(value, index)), 1:fieldcount(typeof(value)))
    end
    return true
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

function gradient_covers_parameters(parameters, gradient)
    if parameters isa AbstractArray
        return gradient isa AbstractArray && size(gradient) == size(parameters)
    elseif parameters isa NamedTuple
        gradient isa NamedTuple || return false
        keys(gradient) == keys(parameters) || return false
        return all(
            key -> gradient_covers_parameters(
                getproperty(parameters, key), getproperty(gradient, key)
            ),
            keys(parameters),
        )
    elseif parameters isa Tuple
        gradient isa Tuple || return false
        length(gradient) == length(parameters) || return false
        return all(
            gradient_covers_parameters(parameter, derivative)
            for (parameter, derivative) in zip(parameters, gradient)
        )
    end
    return gradient !== nothing
end

function tree_sum_abs2(value)
    if value isa AbstractArray
        return sum(abs2, value; init=0.0)
    elseif value isa Number
        return abs2(Float64(value))
    elseif value isa NamedTuple || value isa Tuple
        return sum(tree_sum_abs2, values(value); init=0.0)
    elseif isstructtype(typeof(value)) &&
           !(value isa AbstractString || value isa Symbol || value isa Function || value isa Type)
        return sum(
            index -> tree_sum_abs2(getfield(value, index)),
            1:fieldcount(typeof(value));
            init=0.0,
        )
    end
    return 0.0
end

function tree_max_abs_difference(left, right)
    if left isa AbstractArray && right isa AbstractArray
        size(left) == size(right) || throw(DimensionMismatch())
        return isempty(left) ? 0.0 : maximum(abs, Float64.(left) .- Float64.(right))
    elseif left isa NamedTuple && right isa NamedTuple
        keys(left) == keys(right) || error("parameter tree key mismatch")
        isempty(left) && return 0.0
        return maximum(
            tree_max_abs_difference(getproperty(left, key), getproperty(right, key))
            for key in keys(left)
        )
    elseif left isa Tuple && right isa Tuple
        length(left) == length(right) || throw(DimensionMismatch())
        return isempty(left) ? 0.0 : maximum(
            tree_max_abs_difference(a, b) for (a, b) in zip(left, right)
        )
    end
    return 0.0
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

export CHECKPOINT_SHA256,
    DATASET_SHA256,
    SOURCE_ROWS,
    EXPECTED_EPISODE_IDS,
    EXPECTED_SEEDS,
    EXPECTED_TARGETS,
    N_STEP,
    DISCOUNT,
    LEARNING_RATE,
    BETAS,
    WEIGHT_DECAY,
    LEGACY_PARAMETER_COUNT,
    LEGACY_BATCH,
    ZERO_TOLERANCE,
    CPU_TOLERANCE,
    NPU_TOLERANCE,
    MAX_FIRST_SPECIALIZATION_SECONDS,
    MAX_WARM_UPDATE_SECONDS,
    MAX_PEAK_WORKING_SET_BYTES,
    HARD_WALL_SECONDS,
    ACTOR_SECONDS_PER_STEP,
    ACTOR_REFRESH_COUNT,
    TARGET_T1000_SECONDS,
    GLOBAL_ONE_SHOT_MARKER,
    expected_constants,
    require_hash,
    chunk_ranges,
    huber_scalar,
    frozen_nstep_target,
    tree_all_finite,
    tree_array_elements,
    gradient_covers_parameters,
    tree_sum_abs2,
    tree_max_abs_difference,
    atomic_write_json,
    resolve_benchmark_paths

end
