using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "compare_reduced_hay_v2_sleep_shadow.jl"))
include(joinpath(@__DIR__, "SleepAlignmentDiagnostics.jl"))

using .SleepAlignmentDiagnostics

function parse_measurement_options(arguments)
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
    haskey(values, "checkpoint") || error("--checkpoint is required")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        repeats=parse(Int, get(values, "repeats", "4")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "sleep_measurement_gate.json"),
        )),
    )
end

function deterministic_snapshot!(trainer, dataset, rows, workers)
    result = wake_metrics_and_features!(trainer, dataset, rows, workers)
    base = trainer.tape.base
    valid = base.valid_count
    float64_loss = float64_statewise_loss(base)
    return (;
        raw=copy(@view base.raw[:, 1:valid]),
        masks=copy(result.route_masks),
        statewise=float64_loss,
        float32_metrics=result.metrics,
    )
end

function main(arguments=ARGS)
    options = parse_measurement_options(arguments)
    options.repeats >= 2 || error("repeats must be at least two")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    BLAS.set_num_threads(1)
    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    hasproperty(payload.run_config, :overfit_rows) ||
        error("measurement gate requires an overfit checkpoint")
    rows = Int.(payload.run_config.overfit_rows)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model = build_reduced_hay_model(Symbol(payload.run_config.preset))
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    trainer = trainer_from_checkpoint(model, parameters, payload)
    snapshots = Any[]
    for _ in 1:options.repeats
        restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
        push!(snapshots, deterministic_snapshot!(
            trainer,
            dataset,
            rows,
            options.workers,
        ))
    end
    reference = snapshots[1]
    raw_bitwise = true
    masks_bitwise = true
    maximum_excess_difference = 0.0
    aggregate_excess = Float64[]
    for snapshot in snapshots
        raw_bitwise &= isequal(snapshot.raw, reference.raw)
        masks_bitwise &= isequal(snapshot.masks, reference.masks)
        difference = maximum(abs.(
            snapshot.statewise.excess .-
            reference.statewise.excess
        ))
        maximum_excess_difference = max(
            maximum_excess_difference,
            difference,
        )
        push!(aggregate_excess, sum(snapshot.statewise.excess))
    end
    float64_composite = sum(reference.statewise.composite)
    float64_teacher_entropy = sum(reference.statewise.teacher_entropy)
    float64_excess = sum(reference.statewise.excess)
    float32_composite = reference.float32_metrics.composite_loss
    float32_excess = reference.float32_metrics.excess_loss
    measurement_width = maximum(aggregate_excess) - minimum(aggregate_excess)
    effect_floor = practical_effect_floor(float64_excess, measurement_width)
    pass =
        raw_bitwise &&
        masks_bitwise &&
        maximum_excess_difference == 0.0 &&
        abs(float64_composite - float32_composite) <= 5.0e-5 &&
        abs(float64_excess - float32_excess) <= 5.0e-5
    output = (;
        schema="reduced-hay-v2-sleep-measurement-gate-v1",
        checkpoint=options.checkpoint,
        repeats=options.repeats,
        raw_bitwise,
        masks_bitwise,
        maximum_statewise_excess_difference=maximum_excess_difference,
        repeated_measurement_width=measurement_width,
        practical_effect_floor=effect_floor,
        float64=(;
            composite=float64_composite,
            teacher_entropy=float64_teacher_entropy,
            excess=float64_excess,
            statewise_excess=reference.statewise.excess,
        ),
        float32=(;
            composite=float32_composite,
            excess=float32_excess,
        ),
        pass,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println(
        "pass=$(pass) float64_excess=$(float64_excess) " *
        "width=$(measurement_width) effect_floor=$(effect_floor)",
    )
    println("output=$(options.output)")
    pass || error("sleep measurement gate failed")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
