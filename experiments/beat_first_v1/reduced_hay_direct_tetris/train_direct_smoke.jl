using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayDirectTraining

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"

function _options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        preset=Symbol(get(values, "preset", "tiny")),
        updates=parse(Int, get(values, "updates", "16")),
        state_batch=parse(Int, get(values, "state-batch", "1")),
        width=parse(Int, get(values, "width", "40")),
        learning_rate=parse(
            Float32,
            get(values, "learning-rate", "0.0003"),
        ),
        seed=parse(Int, get(values, "seed", "20260730")),
        fixed_panel=lowercase(get(
            values,
            "fixed-panel",
            "false",
        )) in ("1", "true", "yes"),
    )
end

function _eligible_rows(dataset, width)
    rows = findall(==(:train), dataset.predefined_split)
    isempty(rows) && (rows = collect(eachindex(dataset.action_counts)))
    rows = filter(row -> dataset.action_counts[row] <= width, rows)
    isempty(rows) &&
        error("no training state fits candidate width $width")
    return rows
end

function main(arguments=ARGS)
    options = _options(arguments)
    options.updates > 0 || error("updates must be positive")
    options.state_batch > 0 || error("state batch must be positive")
    dataset = load_teacher_dataset(options.dataset)
    rows = _eligible_rows(dataset, options.width)
    rng = MersenneTwister(options.seed)
    model = build_reduced_hay_model(options.preset)
    trainer = ReducedHayDirectTrainer(
        model;
        rng=MersenneTwister(options.seed ⊻ 0x52484454),
        state_batch=options.state_batch,
        width=options.width,
        learning_rate=options.learning_rate,
    )
    initial_parameters = deepcopy(trainer.parameters)
    fixed_rows = options.fixed_panel ?
        rows[1:options.state_batch] : Int[]
    started = time_ns()
    first_loss = NaN
    last_loss = NaN
    last_groups = nothing
    for update in 1:options.updates
        selected = options.fixed_panel ?
            fixed_rows : rand(rng, rows, options.state_batch)
        pack_rows!(trainer, dataset, selected)
        loss = direct_update!(trainer)
        update == 1 && (first_loss = loss.composite_loss)
        last_loss = loss.composite_loss
        last_groups = gradient_group_norms(trainer.gradient)
        @printf(
            "update=%d loss=%.6f grad=%.6f compartment=%.6f graph=%.6f routing=%.6f head=%.6f\n",
            update,
            loss.composite_loss,
            trainer.last_gradient_norm,
            last_groups.compartment,
            last_groups.graph,
            last_groups.routing,
            last_groups.head,
        )
    end
    elapsed = (time_ns() - started) * 1.0e-9
    result = (;
        updates=options.updates,
        first_loss,
        last_loss,
        parameter_max_delta=parameter_max_delta(
            initial_parameters,
            trainer.parameters,
        ),
        gradient_groups=last_groups,
        seconds=elapsed,
        updates_per_second=options.updates / elapsed,
        topology=reduced_hay_topology(model, trainer.parameters),
    )
    println("RESULT\t", repr(result))
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
