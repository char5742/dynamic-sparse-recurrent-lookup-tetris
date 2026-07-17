using Dates
using Enzyme
using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Printf
using Random
using Statistics
using Zygote

const HEIGHT = 24
const WIDTH = 10
const CHANNELS = 2
const NEXT_ACTIONS = 4
const BASE_SEED = 0x5742
const LEARNING_RATE = 2.5f-4
const DISCOUNT = 0.93f0

"Stateless CNN proxy with Tetris-like board shape and candidate batches."
function build_model()
    return Lux.Chain(
        Lux.Conv((3, 3), CHANNELS => 32, gelu; pad=Lux.SamePad()),
        Lux.Conv((3, 3), 32 => 32, gelu; pad=Lux.SamePad()),
        Lux.Conv((1, 1), 32 => 4, gelu),
        Lux.FlattenLayer(3),
        Lux.Dense((HEIGHT * WIDTH * 4) => 128, gelu),
        Lux.Dense(128 => 1),
    )
end

function perturb_target(ps)
    return tree_map(ps) do x
        x isa AbstractArray ? x .+ 1.0f-3 .* sign.(x) : x
    end
end

function make_problem(batch_size::Int)
    rng = Xoshiro(BASE_SEED)
    model = build_model()
    ps, st = Lux.setup(rng, model)
    target_ps = perturb_target(ps)
    current = randn(rng, Float32, HEIGHT, WIDTH, CHANNELS, batch_size)
    next = randn(
        rng, Float32, HEIGHT, WIDTH, CHANNELS, NEXT_ACTIONS * batch_size
    )
    reward = randn(rng, Float32, 1, batch_size)
    terminal = Float32.(rand(rng, batch_size) .< 0.15)'
    data = (; current, next, reward, terminal, target_ps)
    return (; model, ps, st, data)
end

function td_objective(model, ps, st, data)
    q, next_st = model(data.current, ps, st)
    online_next, _ = model(data.next, ps, st)
    target_next, _ = model(data.next, data.target_ps, st)
    batch_size = size(data.current, 4)
    online_matrix = reshape(online_next, NEXT_ACTIONS, batch_size)
    target_matrix = reshape(target_next, NEXT_ACTIONS, batch_size)

    # Boolean selection makes the Double-DQN action choice explicitly
    # non-differentiable while leaving target evaluation inside the update.
    best_mask = online_matrix .== maximum(online_matrix; dims=1)
    selected_next = sum(target_matrix .* best_mask; dims=1) ./
                    sum(best_mask; dims=1)
    td_target = data.reward .+ DISCOUNT .* (1.0f0 .- data.terminal) .* selected_next
    difference = q .- td_target
    absolute = abs.(difference)
    loss = mean(
        ifelse.(
            absolute .<= 1.0f0,
            0.5f0 .* difference .^ 2,
            absolute .- 0.5f0,
        ),
    )
    return loss, next_st, (; q_mean=mean(q), target_mean=mean(td_target))
end

td_loss_only(ps, model, st, data) = first(td_objective(model, ps, st, data))

function tree_map(f, x)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(map(v -> tree_map(f, v), values(x)))
    elseif x isa Tuple
        return map(v -> tree_map(f, v), x)
    else
        return f(x)
    end
end

function tree_arrays!(destination, x)
    if x isa NamedTuple || x isa Tuple
        foreach(v -> tree_arrays!(destination, v), values(x))
    elseif x isa AbstractArray
        push!(destination, x)
    end
    return destination
end

tree_arrays(x) = tree_arrays!(Any[], x)
parameter_count(ps) = sum(length, tree_arrays(ps); init=0)

function flat_parameters(ps)
    arrays = tree_arrays(ps)
    isempty(arrays) && return Float32[]
    return reduce(vcat, vec.(Array.(arrays)))
end

function numerical_comparison(reference, candidate)
    r = Float64.(flat_parameters(reference))
    c = Float64.(flat_parameters(candidate))
    delta = c .- r
    denom = norm(r) * norm(c)
    cosine = iszero(denom) ? (r == c ? 1.0 : NaN) : dot(r, c) / denom
    return (;
        cosine,
        maximum_absolute_error=maximum(abs, delta; init=0.0),
        relative_l2=norm(delta) / max(norm(r), eps(Float64)),
    )
end

function finite_tree(x)
    return all(a -> all(isfinite, a), tree_arrays(x))
end

function native_enzyme_gradient(model, ps, st, data)
    gradient = Enzyme.make_zero(ps)
    result = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        td_loss_only,
        Enzyme.Active,
        Enzyme.Duplicated(ps, gradient),
        Enzyme.Const(model),
        Enzyme.Const(st),
        Enzyme.Const(data),
    )
    loss = result[2]
    return gradient, loss
end

function native_zygote_gradient(model, ps, st, data)
    objective(parameters) = first(td_objective(model, parameters, st, data))
    loss, pullback = Zygote.pullback(objective, ps)
    gradient = only(pullback(one(loss)))
    return gradient, loss
end

function optimizer_step(ps, gradient)
    optimizer_state = Optimisers.setup(Optimisers.Adam(LEARNING_RATE), ps)
    return Optimisers.update(optimizer_state, ps, gradient)
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
        peak = unsafe_load(Ptr{UInt64}(pointer(buffer) + 8))
        current = unsafe_load(Ptr{UInt64}(pointer(buffer) + 16))
        return (; peak=Int(peak), current=Int(current))
    end
    return (; peak=Int(Sys.maxrss()), current=missing)
end

function package_versions()
    return Dict(
        "julia" => string(VERSION),
        "lux" => string(Base.pkgversion(Lux)),
        "zygote" => string(Base.pkgversion(Zygote)),
        "enzyme" => string(Base.pkgversion(Enzyme)),
        "optimisers" => string(Base.pkgversion(Optimisers)),
    )
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end
