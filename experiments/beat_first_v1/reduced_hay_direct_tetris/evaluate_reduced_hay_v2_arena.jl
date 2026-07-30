using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_PANEL_SEED = UInt64(0x5248415956324556)

function parse_options(arguments)
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
    haskey(values, "checkpoint") ||
        error("--checkpoint is required")
    workers_default = min(20, Threads.nthreads(:default))
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        split=Symbol(get(values, "split", "validation")),
        states=parse(Int, get(values, "states", "128")),
        panel_seed=parse(
            UInt64,
            get(values, "panel-seed", string(DEFAULT_PANEL_SEED)),
        ),
        workers=parse(
            Int,
            get(values, "workers", string(workers_default)),
        ),
        cpuset_mode=Symbol(get(values, "cpuset-mode", "none")),
        output=get(values, "output", ""),
    )
end

function manifest_sha256(dataset_path)
    path = joinpath(dataset_path, "manifest.json")
    isfile(path) || error("dataset manifest is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function stable_panel_rows(
    dataset,
    split::Symbol,
    requested::Int,
    batch_size::Int,
    seed::UInt64,
)
    available = Int.(findall(==(split), dataset.predefined_split))
    isempty(available) ||
        requested > 0 ||
        error("requested states must be positive")
    isempty(available) &&
        error("dataset has no $split split")
    usable = min(requested, length(available))
    usable -= mod(usable, batch_size)
    usable > 0 ||
        error("panel is smaller than checkpoint state batch $batch_size")
    shuffle!(Xoshiro(seed), available)
    return sort!(available[1:usable])
end

function panel_sha256(rows)
    return bytes2hex(SHA.sha256(
        codeunits(join(rows, ',')),
    ))
end

function main(arguments=ARGS)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)
    options = parse_options(arguments)
    isfile(options.checkpoint) ||
        error("checkpoint is absent: $(options.checkpoint)")
    isdir(options.dataset) ||
        error("dataset is absent: $(options.dataset)")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers must be in 2:$(Threads.nthreads(:default))")
    options.cpuset_mode in (:none, :all, :p_only) ||
        error("cpuset-mode must be none, all, or p_only")

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    dataset_hash = manifest_sha256(options.dataset)
    payload.run_config.dataset_manifest_sha256 == dataset_hash ||
        error("checkpoint and evaluation dataset differ")
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    training_rows =
        Int.(findall(==(:train), dataset.predefined_split))
    preset = Symbol(payload.run_config.preset)
    model = build_reduced_hay_model(preset)
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    state_batch = Int(payload.arena_signature.state_batch)
    width = Int(payload.arena_signature.width)
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch,
        width,
    )
    restore_reduced_hay_v2_checkpoint!(
        trainer,
        payload,
        training_rows,
    )
    rows = stable_panel_rows(
        dataset,
        options.split,
        options.states,
        state_batch,
        options.panel_seed,
    )
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=options.cpuset_mode,
        stochastic_routing=false,
    )

    loss_fields = (
        :composite_loss,
        :listnet_loss,
        :q_huber_loss,
        :margin_loss,
        :death_loss,
        :quantile_teacher_loss,
        :geometry_loss,
    )
    loss_sums = zeros(Float64, length(loss_fields))
    top1_matches = 0
    ndcg_sum = 0.0
    pairwise_sum = 0.0
    firing_sum = 0.0
    plateau_sum = 0.0
    entropy_sum = 0.0
    forward_wall = 0.0
    forward_cpu = 0.0
    allocation_bytes = Int128(0)
    gc_seconds = 0.0
    batches = 0

    run_with_dendritic_team!(executor) do running
        for first_index in 1:state_batch:length(rows)
            batch_rows =
                @view rows[first_index:(first_index + state_batch - 1)]
            copyto!(trainer.tape.base.rows, batch_rows)
            reduced_hay_v2_arena_forward!(running)
            gate_sum = sum(trainer.cache.gate_probability)
            gate_density =
                gate_sum /
                Float32(length(trainer.cache.gate_probability))
            loss = ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
                trainer.tape.base,
                trainer.loss_scratch,
                gate_density,
                0.0f0,
            )
            for (index, name) in enumerate(loss_fields)
                loss_sums[index] +=
                    Float64(getproperty(loss, name))
            end
            base = trainer.tape.base
            for slot in 1:state_batch
                row = batch_rows[slot]
                count = Int(base.counts[slot])
                offset = (slot - 1) * width
                prediction = @view base.raw[
                    1,
                    (offset + 1):(offset + count),
                ]
                teacher =
                    @view dataset.teacher_q[1:count, row]
                top1_matches +=
                    argmax(prediction) == argmax(teacher)
                ndcg_sum += BeatFirstTrainingCore._ndcg(
                    prediction,
                    teacher,
                )
                pairwise_sum +=
                    BeatFirstTrainingCore._pairwise_accuracy(
                        prediction,
                        teacher,
                    )
            end
            firing_sum += trainer.metrics.firing_rate
            plateau_sum += trainer.metrics.plateau_mean
            entropy_sum += trainer.metrics.routing_entropy
            forward_wall += trainer.metrics.wall_seconds
            forward_cpu += trainer.metrics.cpu_seconds
            allocation_bytes +=
                trainer.metrics.allocation_bytes
            gc_seconds += trainer.metrics.gc_seconds
            batches += 1
        end
    end

    state_count = length(rows)
    inverse_batches = inv(Float64(batches))
    inverse_states = inv(Float64(state_count))
    losses = NamedTuple{loss_fields}(
        Tuple(loss_sums .* inverse_batches),
    )
    result = merge(losses, (;
        schema="reduced-hay-v2-fixed-panel-evaluation-v1",
        checkpoint=options.checkpoint,
        checkpoint_sha256=
            reduced_hay_v2_checkpoint_sha256(options.checkpoint),
        update=Int(payload.update),
        preset=String(preset),
        dataset=options.dataset,
        dataset_manifest_sha256=dataset_hash,
        split=String(options.split),
        states=state_count,
        panel_seed=string(options.panel_seed),
        panel_rows_sha256=panel_sha256(rows),
        deterministic_routing=true,
        top1_agreement=top1_matches * inverse_states,
        ndcg=ndcg_sum * inverse_states,
        pairwise_accuracy=pairwise_sum * inverse_states,
        firing_rate=firing_sum * inverse_batches,
        plateau_mean=plateau_sum * inverse_batches,
        routing_entropy=entropy_sum * inverse_batches,
        forward_wall_seconds=forward_wall,
        forward_cpu_seconds=forward_cpu,
        states_per_second=
            state_count / max(forward_wall, eps(Float64)),
        allocation_bytes=string(allocation_bytes),
        gc_seconds,
        workers=options.workers,
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
        claim_boundary=(
            "held-teacher ranking evaluation; not gameplay or " *
            "equal-wall-clock superiority evidence"
        ),
    ))
    destination = isempty(options.output) ?
        joinpath(
            dirname(dirname(options.checkpoint)),
            "evaluation_$(options.split)_$(state_count).json",
        ) : abspath(options.output)
    mkpath(dirname(destination))
    open(destination, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(JSON3.write(result))
    println("evaluation_path=$destination")
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
