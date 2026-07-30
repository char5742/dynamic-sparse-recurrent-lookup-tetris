using LinearAlgebra
using Lux
using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(0x44454e4453435241)
const SAMPLER_SEED = UInt64(0x44454e4453414d50)
const ROUTING_SEED = UInt64(0x44454e44524f5554)

function parse_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") ||
            error("unexpected positional argument: $argument")
        index < length(arguments) ||
            error("missing value for $argument")
        values[argument[3:end]] = arguments[index + 1]
        index += 2
    end
    integer(name, default) =
        parse(Int, get(values, name, string(default)))
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        preset=Symbol(get(
            values,
            "preset",
            "reduced_hay_scaled_v2",
        )),
        warmup=integer("warmup", 8),
        repetitions=integer("repetitions", 32),
        state_batch=integer("state-batch", 8),
        width=integer("width", 80),
        workers=integer(
            "workers",
            min(20, Threads.nthreads(:default)),
        ),
    )
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    options.state_batch == 8 ||
        error("production benchmark fixes state batch at 8")
    options.width == 80 ||
        error("production benchmark fixes width at 80")
    options.warmup >= 0 ||
        error("warmup must be nonnegative")
    options.repetitions > 0 ||
        error("repetitions must be positive")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("invalid worker count")

    BLAS.set_num_threads(1)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    rows = Int.(findall(==(:train), dataset.predefined_split))
    isempty(rows) && error("teacher dataset has no train split")
    maximum(dataset.action_counts) <= options.width ||
        error("arena width is too small")

    model = build_reduced_hay_model(options.preset)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch=options.state_batch,
        width=options.width,
    )
    sampler = EpochSampler(rows, Xoshiro(SAMPLER_SEED))
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        stochastic_routing=true,
        routing_seed=ROUTING_SEED,
    )

    wall = 0.0
    cpu = 0.0
    pack = 0.0
    forward = 0.0
    loss = 0.0
    local_phase = 0.0
    optimizer = 0.0
    states_per_second = 0.0
    allocation_bytes = Int128(0)
    gc_seconds = 0.0

    run_with_dendritic_team!(executor) do running
        total = options.warmup + options.repetitions
        for update in 1:total
            ReducedHayV2ArenaTraining.Point.fill_next_rows!(
                trainer.tape.base.rows,
                sampler,
            )
            reduced_hay_v2_arena_update!(running)
            update <= options.warmup && continue
            metrics = trainer.metrics
            wall += metrics.wall_seconds
            cpu += metrics.cpu_seconds
            pack += metrics.pack_seconds
            forward += metrics.forward_seconds
            loss += metrics.loss_seconds
            local_phase += metrics.local_seconds
            optimizer += metrics.optimizer_seconds
            states_per_second += metrics.states_per_second
            allocation_bytes += metrics.allocation_bytes
            gc_seconds += metrics.gc_seconds
        end
    end

    inverse = inv(Float64(options.repetitions))
    measured_states =
        options.repetitions * options.state_batch
    throughput = measured_states / wall
    cpu_utilization =
        cpu / (wall * Float64(options.workers))
    @printf(
        "preset=%s batch=%d width=%d workers=%d warmup=%d repetitions=%d\n",
        String(options.preset),
        options.state_batch,
        options.width,
        options.workers,
        options.warmup,
        options.repetitions,
    )
    @printf(
        "throughput=%.3f states/s mean_update=%.6f s mean_reported=%.3f states/s cpu_utilization=%.3f\n",
        throughput,
        wall * inverse,
        states_per_second * inverse,
        cpu_utilization,
    )
    @printf(
        "phase_mean_s pack=%.6f forward=%.6f loss=%.6f local=%.6f optimizer=%.6f remainder=%.6f\n",
        pack * inverse,
        forward * inverse,
        loss * inverse,
        local_phase * inverse,
        optimizer * inverse,
        (
            wall -
            pack -
            forward -
            loss -
            local_phase -
            optimizer
        ) * inverse,
    )
    @printf(
        "hot_allocation_bytes=%d gc_seconds=%.9f final_loss=%.7f\n",
        allocation_bytes,
        gc_seconds,
        trainer.last_loss.composite_loss,
    )
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
