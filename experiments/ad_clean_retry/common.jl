using Dates
using Enzyme
using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Printf
using Random
using SHA
using Statistics
using Zygote

const INPUT_WIDTH = 256
const HIDDEN_WIDTH = 256
const OUTPUT_WIDTH = 64
const LEARNING_RATE = 1.0f-3
const WEIGHT_DECAY = 1.0f-4
const OPTIMIZER = Optimisers.AdamW(LEARNING_RATE, (0.9, 0.999), WEIGHT_DECAY)
const WARMUP_UPDATES = 10

struct FixedMSEObjective end

function (::FixedMSEObjective)(model, parameters, state, batch)
    input, target = batch
    prediction, next_state = model(input, parameters, state)
    loss = sum(abs2, prediction .- target) / Float32(length(target))
    return loss, next_state, NamedTuple()
end

loss_only(parameters, objective, model, state, batch) =
    first(objective(model, parameters, state, batch))

function make_problem(batch_size::Int)
    batch_size > 0 || error("batch size must be positive")
    model = Lux.Chain(
        Lux.Dense(INPUT_WIDTH => HIDDEN_WIDTH, tanh),
        Lux.Dense(HIDDEN_WIDTH => OUTPUT_WIDTH),
    )
    parameter_rng = Random.Xoshiro(0x3141592653589793)
    data_rng = Random.Xoshiro(0x2718281828459045 + UInt64(batch_size))
    parameters, state = Lux.setup(parameter_rng, model)
    input = randn(data_rng, Float32, INPUT_WIDTH, batch_size)
    target = tanh.(randn(data_rng, Float32, OUTPUT_WIDTH, batch_size))
    objective = FixedMSEObjective()
    return (; model, parameters, state, batch=(input, target), objective, batch_size)
end

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

finite_tree(value) = all(array -> all(isfinite, Array(array)), tree_arrays(value))

function numeric_comparison(reference, candidate)
    ref = Float64.(flat_parameters(reference))
    cand = Float64.(flat_parameters(candidate))
    length(ref) == length(cand) || error("numeric tree length mismatch")
    delta = cand .- ref
    denominator = norm(ref) * norm(cand)
    cosine = iszero(denominator) ? (ref == cand ? 1.0 : NaN) : dot(ref, cand) / denominator
    return (;
        elements=length(ref),
        cosine,
        maximum_absolute_error=maximum(abs, delta; init=0.0),
        relative_l2=norm(delta) / max(norm(ref), eps(Float64)),
    )
end

function zygote_gradient(problem, parameters)
    loss, gradients = Zygote.withgradient(
        ps -> loss_only(ps, problem.objective, problem.model, problem.state, problem.batch),
        parameters,
    )
    return only(gradients), loss
end

function enzyme_mode(mode::Symbol)
    mode === :static && return Enzyme.ReverseWithPrimal
    mode === :runtime && return Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    error("unsupported Enzyme mode: $mode")
end

function enzyme_gradient!(shadow, problem, parameters, mode::Symbol)
    zero_shadow!(shadow)
    result = Enzyme.autodiff(
        enzyme_mode(mode),
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
        julia_commit=Base.GIT_VERSION_INFO.commit_short,
        lux=string(Base.pkgversion(Lux)),
        zygote=string(Base.pkgversion(Zygote)),
        enzyme=string(Base.pkgversion(Enzyme)),
        optimisers=string(Base.pkgversion(Optimisers)),
    )
    return reactant === nothing ? base : merge(base, (; reactant))
end

function array_sha256(arrays...)
    context = SHA.SHA256_CTX()
    for array in arrays
        SHA.update!(context, reinterpret(UInt8, vec(Array(array))))
    end
    return bytes2hex(SHA.digest!(context))
end

function workload_description(problem)
    input, target = problem.batch
    return (;
        model="Lux.Chain(Dense(256=>256,tanh),Dense(256=>64))",
        loss="sum(abs2,prediction-target)/length(target)",
        optimizer="Optimisers.AdamW(lr=1e-3,beta=(0.9,0.999),weight_decay=1e-4)",
        parameter_count=Lux.parameterlength(problem.parameters),
        batch_size=problem.batch_size,
        input_shape=size(input),
        target_shape=size(target),
        parameter_sha256=array_sha256(tree_arrays(problem.parameters)...),
        batch_sha256=array_sha256(input, target),
        dtype="Float32",
    )
end

function source_hashes()
    names = ("Project.toml", "Manifest.toml", "common.jl", "benchmark_native.jl", "benchmark_reactant.jl", "numerics_native.jl", "numerics_reactant.jl")
    return Dict(name => (isfile(joinpath(@__DIR__, name)) ? bytes2hex(open(sha256, joinpath(@__DIR__, name))) : nothing) for name in names)
end

function steady_summary(times, allocations, gc_times)
    first_steady = min(WARMUP_UPDATES + 1, length(times))
    indices = first_steady:length(times)
    steady_time = sum(@view times[indices])
    steady_updates = length(indices)
    return (;
        warmup_updates_excluded=first_steady - 1,
        measured_steady_updates=steady_updates,
        steady_wall_seconds=steady_time,
        steady_updates_per_second=steady_updates / steady_time,
        steady_median_seconds=median(@view times[indices]),
        steady_p10_seconds=quantile(@view(times[indices]), 0.10),
        steady_p90_seconds=quantile(@view(times[indices]), 0.90),
        steady_median_allocated_bytes=Int(round(median(@view allocations[indices]))),
        steady_mean_allocated_bytes=mean(@view allocations[indices]),
        steady_gc_seconds=sum(@view gc_times[indices]),
    )
end

function checkpoint_summary(times, allocations, count::Int)
    count <= length(times) || return nothing
    first_steady = min(WARMUP_UPDATES + 1, count)
    indices = first_steady:count
    elapsed = sum(@view times[indices])
    return (;
        updates=count,
        compile_inclusive_seconds=sum(@view times[1:count]),
        post_warmup_updates=length(indices),
        post_warmup_updates_per_second=length(indices) / elapsed,
        post_warmup_median_allocated_bytes=Int(round(median(@view allocations[indices]))),
    )
end

function checkpoint_summaries(times, allocations)
    result = Dict{String,Any}()
    for count in (16, 64, 100, 500, 1000)
        summary = checkpoint_summary(times, allocations, count)
        summary === nothing || (result[string(count)] = summary)
    end
    return result
end

function write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end
