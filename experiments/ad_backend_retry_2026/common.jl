using ADTypes: AutoEnzyme
using Dates
using Enzyme
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Printf
using SHA
using Statistics
using Zygote

const EXPERIMENT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
const TRACKED_TRAINER = joinpath(REPOSITORY_ROOT, "experiments", "learning", "train_distillation.jl")
include(TRACKED_TRAINER)

const DATASET_PATH = get(
    ENV,
    "AD_RETRY_DATASET",
    raw"D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2",
)
const CHECKPOINT_PATH = get(
    ENV,
    "AD_RETRY_CHECKPOINT",
    joinpath(REPOSITORY_ROOT, "checkpoints", "learning", "C11b_listwise_165k_lr1e3_3000_fixed74.jld2"),
)
const BASE_ROWS = [1, 750, 1501, 2160]
const MAX_ACTIONS = 74
const TEMPERATURE = 1.0f0
const LEARNING_RATE = 1.0f-3
const OPTIMIZER = Optimisers.AdamW(LEARNING_RATE, (0.9, 0.999), 1.0f-4)

struct FixedListwiseObjective{A,B}
    temperature::Float32
end

function (objective::FixedListwiseObjective{A,B})(model, parameters, state, batch) where {A,B}
    input, targets, mask = batch
    raw, next_state = model(input, parameters, state)
    predictions = reshape(raw, A, B)
    loss = standardized_listwise_cross_entropy(
        predictions, targets, mask; temperature=objective.temperature
    )
    return loss, next_state, NamedTuple()
end

loss_only(parameters, objective, model, state, batch) =
    first(objective(model, parameters, state, batch))

function tree_arrays!(destination, value)
    if value isa NamedTuple || value isa Tuple
        foreach(item -> tree_arrays!(destination, item), values(value))
    elseif value isa AbstractArray
        push!(destination, value)
    end
    return destination
end

tree_arrays(value) = tree_arrays!(Any[], value)

function zero_shadow!(shadow)
    foreach(array -> fill!(array, zero(eltype(array))), tree_arrays(shadow))
    return shadow
end

function flat_parameters(parameters)
    arrays = tree_arrays(parameters)
    isempty(arrays) && return Float32[]
    return reduce(vcat, vec.(Array.(arrays)))
end

function numerical_comparison(reference, candidate)
    ref = Float64.(flat_parameters(reference))
    cand = Float64.(flat_parameters(candidate))
    delta = cand .- ref
    denominator = norm(ref) * norm(cand)
    return (;
        cosine=iszero(denominator) ? (ref == cand ? 1.0 : NaN) : dot(ref, cand) / denominator,
        maximum_absolute_error=maximum(abs, delta; init=0.0),
        relative_l2=norm(delta) / max(norm(ref), eps(Float64)),
    )
end

finite_tree(value) = all(array -> all(isfinite, array), tree_arrays(value))

function file_sha256(path)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function load_fixed_problem(state_batch::Int)
    state_batch > 0 || error("state batch must be positive")
    dataset = load_teacher_dataset(DATASET_PATH)
    size(dataset.placements, 4) == MAX_ACTIONS || error(
        "expected $MAX_ACTIONS actions, observed $(size(dataset.placements, 4))"
    )
    rows = [BASE_ROWS[mod1(index, length(BASE_ROWS))] for index in 1:state_batch]
    all(dataset.episode_ids[rows] .<= 12) || error("fixed rows are not all training episodes")
    input, targets, mask = candidate_batch(dataset, rows; teacher_targets=true)
    model_config = (; channels=8, blocks=1, spatial_channels=2)
    model = CompactCandidateQ(; model_config...)
    initial = load_initial_checkpoint(CHECKPOINT_PATH, model_config)
    Lux.parameterlength(initial.parameters) == 165_051 || error("parameter count mismatch")
    objective = FixedListwiseObjective{MAX_ACTIONS,state_batch}(TEMPERATURE)
    return (;
        model,
        parameters=initial.parameters,
        state=initial.state,
        batch=(input, targets, mask),
        objective,
        rows,
        episode_ids=dataset.episode_ids[rows],
        action_counts=dataset.action_counts[rows],
        valid_candidates=Int(sum(mask)),
    )
end

function native_zygote_gradient(problem, parameters)
    loss, gradients = Zygote.withgradient(
        ps -> loss_only(ps, problem.objective, problem.model, problem.state, problem.batch),
        parameters,
    )
    return only(gradients), loss
end

function native_enzyme_gradient!(shadow, problem, parameters; runtime_activity::Bool=false)
    zero_shadow!(shadow)
    mode = runtime_activity ?
           Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal) :
           Enzyme.ReverseWithPrimal
    result = Enzyme.autodiff(
        mode,
        loss_only,
        Enzyme.Active,
        Enzyme.Duplicated(parameters, shadow),
        Enzyme.Const(problem.objective),
        Enzyme.Const(problem.model),
        Enzyme.Const(problem.state),
        Enzyme.Const(problem.batch),
    )
    return shadow, result[2]
end

function new_train_state(problem; reactant_values=nothing)
    parameters, state = reactant_values === nothing ?
                        (deepcopy(problem.parameters), deepcopy(problem.state)) :
                        reactant_values
    return Lux.Training.TrainState(problem.model, parameters, state, OPTIMIZER)
end

function process_memory_bytes()
    if Sys.iswindows()
        buffer = zeros(UInt8, 80)
        unsafe_store!(Ptr{UInt32}(pointer(buffer)), UInt32(length(buffer)))
        process = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        success = ccall(
            (:GetProcessMemoryInfo, "psapi"),
            Int32,
            (Ptr{Cvoid}, Ptr{Cvoid}, UInt32),
            process,
            pointer(buffer),
            UInt32(length(buffer)),
        )
        success == 0 && return (; peak=missing, current=missing)
        return (;
            peak=Int(unsafe_load(Ptr{UInt64}(pointer(buffer) + 8))),
            current=Int(unsafe_load(Ptr{UInt64}(pointer(buffer) + 16))),
        )
    end
    return (; peak=Int(Sys.maxrss()), current=missing)
end

function package_versions(; reactant=nothing)
    base = (;
        julia=string(VERSION),
        lux=string(Base.pkgversion(Lux)),
        zygote=string(Base.pkgversion(Zygote)),
        enzyme=string(Base.pkgversion(Enzyme)),
        optimisers=string(Base.pkgversion(Optimisers)),
        jld2=string(Base.pkgversion(JLD2)),
    )
    return reactant === nothing ? base : merge(base, (; reactant))
end

function input_provenance(problem)
    return (;
        model="CompactCandidateQ(channels=8, blocks=1, spatial_channels=2)",
        parameter_count=Lux.parameterlength(problem.parameters),
        action_width=MAX_ACTIONS,
        state_batch=length(problem.rows),
        fixed_rows=problem.rows,
        episode_ids=problem.episode_ids,
        action_counts=problem.action_counts,
        valid_candidates=problem.valid_candidates,
        dataset_path=abspath(DATASET_PATH),
        dataset_sha256=file_sha256(DATASET_PATH),
        checkpoint_path=abspath(CHECKPOINT_PATH),
        checkpoint_sha256=file_sha256(CHECKPOINT_PATH),
        tracked_model_sha256=file_sha256(joinpath(REPOSITORY_ROOT, "experiments", "learning", "compact_model.jl")),
        tracked_trainer_sha256=file_sha256(TRACKED_TRAINER),
    )
end

function write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end

function window_summaries(times, allocations, gc_times)
    result = Dict{String,Any}()
    for count in (10, 100, 500, 1000)
        count > length(times) && continue
        steady_start = min(6, count)
        result[string(count)] = (;
            updates=count,
            compile_inclusive_seconds=sum(@view times[1:count]),
            updates_per_second=count / sum(@view times[1:count]),
            first_update_seconds=first(times),
            steady_median_seconds=median(@view times[steady_start:count]),
            steady_p10_seconds=quantile(@view(times[steady_start:count]), 0.10),
            steady_p90_seconds=quantile(@view(times[steady_start:count]), 0.90),
            steady_median_allocated_bytes=Int(round(median(@view allocations[steady_start:count]))),
            gc_seconds=sum(@view gc_times[1:count]),
        )
    end
    return result
end
