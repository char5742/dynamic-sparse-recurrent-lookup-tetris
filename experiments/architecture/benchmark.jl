using Lux
using Zygote
using Random
using Statistics
using Printf
using JSON3
using LinearAlgebra

include(joinpath(@__DIR__, "models.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "ARCH_BLAS_THREADS", "12")))
const REPEATS_FORWARD = parse(Int, get(ENV, "ARCH_FORWARD_REPEATS", "3"))
const REPEATS_BACKWARD = parse(Int, get(ENV, "ARCH_BACKWARD_REPEATS", "1"))
const MEASURE_ALLOCATIONS = get(ENV, "ARCH_MEASURE_ALLOCATIONS", "false") == "true"

function make_input(rng, batch)
    board = rand(rng, Float32, 24, 10, 2, batch)
    queue = zeros(Float32, 7, 6, batch)
    for b in 1:batch, token in 1:6
        queue[rand(rng, 1:7), token, b] = 1.0f0
    end
    scalars = rand(rng, Float32, 3, batch)
    target = randn(rng, Float32, 1, batch)
    return (board, queue, scalars), target
end

function finite_tree(x)
    if x isa AbstractArray
        return all(isfinite, x)
    elseif x isa NamedTuple || x isa Tuple
        return all(finite_tree, values(x))
    end
    return true
end

function loss_only(model, input, target, ps, st)
    prediction, _ = model(input, ps, st)
    return mean(abs2, prediction .- target)
end

function time_forward(model, input, ps, st)
    prediction, _ = model(input, ps, st)
    @assert all(isfinite, prediction)
    times = Float64[]
    for _ in 1:REPEATS_FORWARD
        push!(times, @elapsed model(input, ps, st))
    end
    allocated = MEASURE_ALLOCATIONS ? (@allocated model(input, ps, st)) : 0
    return median(times), allocated
end

function time_backward(model, input, target, ps, st)
    loss, grads = Zygote.withgradient(ps) do current_ps
        loss_only(model, input, target, current_ps, st)
    end
    @assert isfinite(loss)
    @assert finite_tree(grads)
    times = Float64[]
    for _ in 1:REPEATS_BACKWARD
        push!(times, @elapsed Zygote.withgradient(ps) do current_ps
            loss_only(model, input, target, current_ps, st)
        end)
    end
    allocated = if MEASURE_ALLOCATIONS
        @allocated Zygote.withgradient(ps) do current_ps
            loss_only(model, input, target, current_ps, st)
        end
    else
        0
    end
    return loss, median(times), allocated
end

function benchmark_one(name::Symbol, factory, batch::Int)
    println("starting ", name, " batch=", batch)
    flush(stdout)
    rng = Xoshiro(0x1313 + batch)
    model = factory()
    ps, st = Lux.setup(rng, model)
    input, target = make_input(rng, batch)

    compile_forward = @elapsed model(input, ps, st)
    println("  forward compiled in ", round(compile_forward; digits=3), " s")
    flush(stdout)
    compile_backward = @elapsed Zygote.withgradient(ps) do current_ps
        loss_only(model, input, target, current_ps, st)
    end
    println("  backward compiled in ", round(compile_backward; digits=3), " s")
    flush(stdout)

    forward_seconds, forward_alloc = time_forward(model, input, ps, st)
    loss, backward_seconds, backward_alloc =
        time_backward(model, input, target, ps, st)
    result = (
        architecture=String(name),
        batch,
        parameters=Lux.parameterlength(model),
        macs=analytical_macs(name, batch),
        macs_per_candidate=analytical_macs(name, 1),
        activation_bytes=analytical_activation_bytes(name, batch),
        compile_forward_seconds=compile_forward,
        compile_backward_seconds=compile_backward,
        forward_seconds,
        forward_candidates_per_second=batch / forward_seconds,
        forward_allocated_bytes=forward_alloc,
        backward_seconds,
        backward_candidates_per_second=batch / backward_seconds,
        backward_allocated_bytes=backward_alloc,
        loss,
        finite=true,
    )
    @printf(
        "%-12s b=%3d params=%9d MAC/cand=%9d fwd=%7.1f cand/s bwd=%7.1f cand/s\n",
        name,
        batch,
        result.parameters,
        result.macs_per_candidate,
        result.forward_candidates_per_second,
        result.backward_candidates_per_second,
    )
    return result
end

all_factories = [
    (:preact_se, () -> PreActSEValueNet()),
    (:convnext, () -> ConvNeXtValueNet()),
    (:film, () -> FiLMValueNet()),
]
selected_names = Set(Symbol.(split(get(ENV, "ARCH_NAMES", "preact_se,convnext,film"), ',')))
factories = filter(item -> item[1] in selected_names, all_factories)
batches = parse.(Int, split(get(ENV, "ARCH_BATCHES", "16,64"), ','))

results = Any[]
for (name, factory) in factories, batch in batches
    push!(results, benchmark_one(name, factory, batch))
end

output = get(
    ENV,
    "ARCH_OUTPUT",
    joinpath(@__DIR__, "benchmark_results.json"),
)
open(output, "w") do io
    JSON3.pretty(io, results)
    println(io)
end
println("wrote ", output)
