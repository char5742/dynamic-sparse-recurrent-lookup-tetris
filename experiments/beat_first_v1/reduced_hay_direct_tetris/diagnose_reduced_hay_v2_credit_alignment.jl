using LinearAlgebra
using Lux
using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))

using .BeatFirstTrainingCore
using .ReducedHayDirectTraining
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(0x44454e4453435241)
const ROUTING_SEED = UInt64(0x44454e44524f5554)

copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function model_parameters(parameters)
    fields = ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS
    return NamedTuple{fields}(
        map(name -> copy(getproperty(parameters, name)), fields),
    )
end

function worker_gradient(executor, name)
    result = zeros(Float32, size(getproperty(
        first(executor.workers).gradient,
        name,
    )))
    for worker in executor.workers
        result .+= getproperty(worker.gradient, name)
    end
    return result
end

function cosine(left, right)
    dot_product = dot(left, right)
    denominator = norm(left) * norm(right)
    return denominator == 0 ? NaN : dot_product / denominator
end

function group_cosine(exact, approximate, fields)
    dot_product = 0.0
    exact_square = 0.0
    local_square = 0.0
    for name in fields
        exact_array = getproperty(exact, name)
        local_array = getproperty(approximate, name)
        dot_product += dot(exact_array, local_array)
        exact_square += sum(abs2, exact_array; init=0.0)
        local_square += sum(abs2, local_array; init=0.0)
    end
    denominator = sqrt(exact_square * local_square)
    return denominator == 0 ? NaN : dot_product / denominator
end

function parse_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        index < length(arguments) || error("missing option value")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        state_batch=parse(Int, get(values, "state-batch", "4")),
        width=parse(Int, get(values, "width", "80")),
        workers=parse(Int, get(values, "workers", "4")),
        warmup=parse(Int, get(values, "warmup", "0")),
        global_scale=parse(Float32, get(values, "global-scale", "1")),
        local_scale=parse(Float32, get(values, "local-scale", "1")),
    )
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    rows = Int.(findall(==(:train), dataset.predefined_split))
    sampler = EpochSampler(rows, Xoshiro(0x4352454449545348))
    model = build_reduced_hay_model(:tiny_recurrent_v2)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    local_trainer = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=options.state_batch,
        width=options.width,
        global_signal_scale=options.global_scale,
        local_signal_scale=options.local_scale,
        routing_entropy_weight=0.0f0,
        routing_load_weight=0.0f0,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    executor = DendriticArenaExecutor(
        local_trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        routing_seed=ROUTING_SEED,
        recurrent_signal_scale=0.0f0,
    )
    comparison_rows = Int[]
    run_with_dendritic_team!(executor) do running
        for _ in 1:options.warmup
            local_trainer.tape.base.rows .=
                next_batch!(sampler, options.state_batch)
            dendritic_arena_update!(running)
        end
        comparison_rows = next_batch!(sampler, options.state_batch)
        local_trainer.tape.base.rows .= comparison_rows
        running.recurrent_signal_scale = 1.0f0
        exact_parameters = model_parameters(local_trainer.parameters)
        direct = ReducedHayDirectTrainer(
            model;
            rng=Xoshiro(MODEL_SEED),
            state_batch=options.state_batch,
            width=options.width,
        )
        direct.parameters = exact_parameters
        pack_rows!(direct, dataset, comparison_rows)
        exact_loss, exact_gradient = direct_gradient!(direct)
        dendritic_arena_update!(running)
        local_gradient = NamedTuple{
            ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS
        }(map(
            name -> worker_gradient(running, name),
            ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS,
        ))

        recurrent_fields =
            ReducedHayV2ArenaTraining.RECURRENT_PARAMETER_FIELDS
        head_fields = ReducedHayV2ArenaTraining.HEAD_PARAMETER_FIELDS
        @printf(
            "rows=%s warmup=%d exact_loss=%.6f local_loss=%.6f recurrent_cosine=%.9f head_cosine=%.9f\n",
            join(comparison_rows, ','),
            options.warmup,
            exact_loss.composite_loss,
            local_trainer.last_loss.composite_loss,
            group_cosine(exact_gradient, local_gradient, recurrent_fields),
            group_cosine(exact_gradient, local_gradient, head_fields),
        )
        for name in recurrent_fields
            @printf(
                "%s\tcosine=%.9f\texact_norm=%.9g\tlocal_norm=%.9g\n",
                name,
                cosine(
                    getproperty(exact_gradient, name),
                    getproperty(local_gradient, name),
                ),
                norm(getproperty(exact_gradient, name)),
                norm(getproperty(local_gradient, name)),
            )
        end
    end
    return nothing
end

main()
