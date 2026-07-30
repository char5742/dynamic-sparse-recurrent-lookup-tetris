using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))
include(joinpath(@__DIR__, "BudgetMatchedPointSNN.jl"))
include(joinpath(@__DIR__, "BudgetMatchedGRU.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayDirectTraining
using .BudgetMatchedPointSNN
using .BudgetMatchedGRU

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"

function _parse(arguments)
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
        arm=Symbol(get(values, "arm", "reduced")),
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        updates=parse(Int, get(values, "updates", "6")),
        width=parse(Int, get(values, "width", "40")),
        learning_rate=parse(
            Float32,
            get(values, "learning-rate", "0.0002"),
        ),
        seed=parse(Int, get(values, "seed", "20260730")),
    )
end

function _arm(name::Symbol)
    name === :point && return (
        build_budget_point_snn(),
        budget_point_raw,
    )
    name === :reduced && return (
        build_reduced_hay_model(:tiny),
        reduced_hay_raw,
    )
    name === :reduced_v2 && return (
        build_reduced_hay_model(:tiny_recurrent_v2),
        reduced_hay_raw,
    )
    name === :gru && return (
        DiagonalGRUBaseline(),
        budget_gru_raw,
    )
    name === :frozen && error(
        "the frozen arm is fail-closed and artifact-backed; construct it " *
        "with BudgetMatchedFrozenElevenState.build_budget_frozen_trainer",
    )
    error(
        "unknown arm $name; use point, reduced, reduced_v2, gru, or frozen",
    )
end

function main(arguments=ARGS)
    options = _parse(arguments)
    options.updates > 0 || error("updates must be positive")
    dataset = load_teacher_dataset(options.dataset)
    rows = findall(==(:train), dataset.predefined_split)
    isempty(rows) && (rows = collect(eachindex(dataset.action_counts)))
    rows = filter(row -> dataset.action_counts[row] <= options.width, rows)
    isempty(rows) &&
        error("no training row fits width $(options.width)")
    fixed_row = first(rows)
    model, raw_function = _arm(options.arm)
    trainer = CanonicalDirectTrainer(
        model,
        raw_function;
        rng=MersenneTwister(options.seed),
        state_batch=1,
        width=options.width,
        learning_rate=options.learning_rate,
    )
    initial = deepcopy(trainer.parameters)
    first_loss = NaN
    last_loss = NaN
    started = time_ns()
    for update in 1:options.updates
        pack_rows!(trainer, dataset, [fixed_row])
        loss = direct_update!(trainer)
        update == 1 && (first_loss = loss.composite_loss)
        last_loss = loss.composite_loss
        @printf(
            "arm=%s update=%d loss=%.6f gradient=%.6f\n",
            String(options.arm),
            update,
            loss.composite_loss,
            trainer.last_gradient_norm,
        )
    end
    elapsed = (time_ns() - started) * 1.0e-9
    result = (;
        arm=options.arm,
        row=fixed_row,
        candidates=dataset.action_counts[fixed_row],
        updates=options.updates,
        first_loss,
        last_loss,
        parameter_max_delta=parameter_max_delta(
            initial,
            trainer.parameters,
        ),
        seconds=elapsed,
        updates_per_second=options.updates / elapsed,
    )
    println("RESULT\t", repr(result))
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
