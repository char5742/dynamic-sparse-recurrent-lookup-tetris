using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const MODEL_SEED = UInt64(2026072703)
const SPLIT_SEED = UInt64(2026071817)
const SAMPLER_SEED = UInt64(2026071801) + UInt64(0x9e3779b97f4a7c15)
const TRAIN_EVAL_SEED = UInt64(2026071801) + UInt64(0x101)
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const CHECKPOINT_FORMAT = "serial-workspace-snn-arena-checkpoint"
const CHECKPOINT_VERSION = 1

mutable struct ProgressTotals
    updates::Int
    teacher_states::Int
    candidates::Int
    hot_wall_seconds::Float64
    hot_cpu_seconds::Float64
    hot_allocation_bytes::Int128
    hot_gc_seconds::Float64
    pack_seconds::Float64
    forward_seconds::Float64
    loss_seconds::Float64
    backward_seconds::Float64
    optimizer_seconds::Float64
    consolidation_seconds::Float64
end

ProgressTotals() = ProgressTotals(
    0, 0, 0, 0.0, 0.0, Int128(0), 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

function progress_snapshot(progress::ProgressTotals)
    return (;
        updates=progress.updates,
        teacher_states=progress.teacher_states,
        candidates=progress.candidates,
        hot_wall_seconds=progress.hot_wall_seconds,
        hot_cpu_seconds=progress.hot_cpu_seconds,
        hot_allocation_bytes=progress.hot_allocation_bytes,
        hot_gc_seconds=progress.hot_gc_seconds,
        pack_seconds=progress.pack_seconds,
        forward_seconds=progress.forward_seconds,
        loss_seconds=progress.loss_seconds,
        backward_seconds=progress.backward_seconds,
        optimizer_seconds=progress.optimizer_seconds,
        consolidation_seconds=progress.consolidation_seconds,
    )
end

function restore_progress(snapshot)
    return ProgressTotals(
        Int(snapshot.updates),
        Int(snapshot.teacher_states),
        Int(snapshot.candidates),
        Float64(snapshot.hot_wall_seconds),
        Float64(snapshot.hot_cpu_seconds),
        Int128(snapshot.hot_allocation_bytes),
        Float64(snapshot.hot_gc_seconds),
        Float64(snapshot.pack_seconds),
        Float64(snapshot.forward_seconds),
        Float64(snapshot.loss_seconds),
        Float64(snapshot.backward_seconds),
        Float64(snapshot.optimizer_seconds),
        Float64(snapshot.consolidation_seconds),
    )
end

function accumulate!(
    progress::ProgressTotals,
    trainer::ArenaTrainer,
)
    metrics = trainer.metrics
    progress.updates += 1
    progress.teacher_states += trainer.arena.state_batch
    progress.candidates += trainer.arena.valid_count
    progress.hot_wall_seconds += metrics.wall_seconds
    progress.hot_cpu_seconds += metrics.cpu_seconds
    progress.hot_allocation_bytes += metrics.allocation_bytes
    progress.hot_gc_seconds += metrics.gc_seconds
    progress.pack_seconds += metrics.pack_seconds
    progress.forward_seconds += metrics.forward_seconds
    progress.loss_seconds += metrics.loss_seconds
    progress.backward_seconds += metrics.backward_seconds
    progress.optimizer_seconds += metrics.optimizer_seconds
    progress.consolidation_seconds += metrics.consolidation_seconds
    return progress
end

function env_int(name, default; minimum=0)
    value = parse(Int, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function env_float(name, default; minimum=0.0)
    value = parse(Float32, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) && error("manifest training split is empty")
        return Int.(rows)
    end
    groups = sort(unique(dataset.split_group_ids))
    shuffled = shuffle(Xoshiro(SPLIT_SEED), groups)
    validation_count =
        clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    forbidden = Set(shuffled[1:validation_count])
    return findall(
        group -> !(group in forbidden),
        dataset.split_group_ids,
    )
end

function fixed_training_panel(rows, count::Int)
    selected = copy(Int.(rows))
    shuffle!(Xoshiro(TRAIN_EVAL_SEED), selected)
    resize!(selected, min(count, length(selected)))
    return selected
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end

function append_trace(path, record)
    first_write = !isfile(path)
    open(path, "a") do io
        if first_write
            println(
                io,
                join((
                    "update",
                    "teacher_states",
                    "loss",
                    "gradient_norm",
                    "enabled_synapses",
                    "structural_flips_total",
                    "states_per_second",
                    "cpu_percent",
                    "hot_allocation_bytes",
                    "hot_gc_seconds",
                ), '\t'),
            )
        end
        println(io, join(values(record), '\t'))
    end
    return nothing
end

function source_fingerprint()
    files = (
        joinpath(@__DIR__, "SerialWorkspaceSNN.jl"),
        joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"),
        joinpath(@__DIR__, "train_arena_100k.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "bounded_mpmc_queue.jl",
        ),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "windows_cpu_sets.jl",
        ),
    )
    bytes = UInt8[]
    for path in files
        append!(bytes, read(path))
    end
    return bytes2hex(sha256(bytes))
end

function evaluate(model, parameters, states, dataset, rows, batch)
    return evaluation_metrics(
        dataset,
        rows,
        batch,
        packed -> first(model(packed.inputs, parameters, states)),
    )
end

function checkpoint_payload(
    trainer,
    sampler,
    initial_parameters,
    config,
    initial_metrics,
    progress,
)
    return (;
        format=CHECKPOINT_FORMAT,
        version=CHECKPOINT_VERSION,
        update=trainer.optimizer.step,
        parameters=trainer.parameters,
        optimizer=(;
            first_moment=trainer.optimizer.first_moment,
            second_moment=trainer.optimizer.second_moment,
            learning_rate=trainer.optimizer.learning_rate,
            beta1=trainer.optimizer.beta1,
            beta2=trainer.optimizer.beta2,
            beta1_power=trainer.optimizer.beta1_power,
            beta2_power=trainer.optimizer.beta2_power,
            epsilon=trainer.optimizer.epsilon,
            weight_decay=trainer.optimizer.weight_decay,
            step=trainer.optimizer.step,
        ),
        total_structural_flips=trainer.total_structural_flips,
        sampler_state=sampler_snapshot(sampler),
        initial_parameters,
        config,
        initial_metrics,
        progress=progress_snapshot(progress),
    )
end

function save_checkpoint!(
    path,
    trainer,
    sampler,
    initial_parameters,
    config,
    initial_metrics,
    progress,
)
    payload = checkpoint_payload(
        trainer,
        sampler,
        initial_parameters,
        config,
        initial_metrics,
        progress,
    )
    atomic_jldsave(path; payload)
    return (;
        path=abspath(path),
        bytes=filesize(path),
        sha256=bytes2hex(sha256(read(path))),
        update=trainer.optimizer.step,
    )
end

function load_checkpoint(path, expected_sha256)
    checkpoint_path = abspath(path)
    isfile(checkpoint_path) || error(
        "resume checkpoint does not exist: $checkpoint_path",
    )
    actual_sha256 = bytes2hex(sha256(read(checkpoint_path)))
    isempty(expected_sha256) ||
        lowercase(expected_sha256) == actual_sha256 ||
        error("resume checkpoint SHA-256 differs")
    file = JLD2.load(checkpoint_path)
    haskey(file, "payload") || error("resume checkpoint has no payload")
    payload = file["payload"]
    payload.format == CHECKPOINT_FORMAT || error(
        "resume checkpoint format differs",
    )
    Int(payload.version) == CHECKPOINT_VERSION || error(
        "unsupported resume checkpoint version",
    )
    return payload, (;
        path=checkpoint_path,
        bytes=filesize(checkpoint_path),
        sha256=actual_sha256,
    )
end

function restore_trainer!(trainer, payload)
    keys(payload.parameters) == keys(trainer.parameters) ||
        error("resume parameter registry differs")
    for name in keys(trainer.parameters)
        destination = getproperty(trainer.parameters, name)
        source = getproperty(payload.parameters, name)
        size(destination) == size(source) ||
            error("resume parameter shape differs for $name")
        destination .= source
        getproperty(trainer.optimizer.first_moment, name) .=
            getproperty(payload.optimizer.first_moment, name)
        getproperty(trainer.optimizer.second_moment, name) .=
            getproperty(payload.optimizer.second_moment, name)
    end
    trainer.optimizer.learning_rate =
        Float32(payload.optimizer.learning_rate)
    trainer.optimizer.beta1 = Float32(payload.optimizer.beta1)
    trainer.optimizer.beta2 = Float32(payload.optimizer.beta2)
    trainer.optimizer.beta1_power =
        Float32(payload.optimizer.beta1_power)
    trainer.optimizer.beta2_power =
        Float32(payload.optimizer.beta2_power)
    trainer.optimizer.epsilon = Float32(payload.optimizer.epsilon)
    trainer.optimizer.weight_decay =
        Float32(payload.optimizer.weight_decay)
    trainer.optimizer.step = Int(payload.optimizer.step)
    trainer.optimizer.step == Int(payload.update) ||
        error("resume optimizer clock differs")
    trainer.total_structural_flips =
        Int(payload.total_structural_flips)
    ArenaWorkspaceTraining.refresh_parameter_cache!(
        trainer.cache,
        trainer.parameters,
    )
    return trainer
end

function with_gc_boundary(body::F) where {F}
    GC.enable(true)
    try
        return body()
    finally
        GC.gc()
        GC.enable(false)
    end
end

function compile_hot_path!(
    model,
    parameters,
    dataset,
    rows,
    state_batch,
    width,
    active_workers,
    cpuset_mode,
)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch,
        width,
    )
    trainer.arena.rows .= @view rows[1:state_batch]
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers,
        cpuset_mode,
    )
    run_with_arena_team!(executor) do running
        arena_update!(running; structural_interval=1)
    end
    GC.gc()
    return nothing
end

function main()
    Threads.nthreads(:interactive) == 0 || error(
        "launch with --threads=N,0",
    )
    BLAS.set_num_threads(1)
    maximum_updates = env_int("SWSNN_MAX_UPDATES", 100_000; minimum=1)
    state_batch = env_int("SWSNN_STATE_BATCH", 8; minimum=1)
    active_workers = env_int(
        "SWSNN_ACTIVE_WORKERS",
        Threads.nthreads(:default);
        minimum=2,
    )
    cpuset_mode = Symbol(lowercase(get(
        ENV,
        "SWSNN_CPUSET_MODE",
        active_workers == Threads.nthreads(:default) ? "all" : "none",
    )))
    learning_rate =
        env_float("SWSNN_LEARNING_RATE", 5.0f-4; minimum=0.0)
    weight_decay =
        env_float("SWSNN_WEIGHT_DECAY", 1.0f-5; minimum=0.0)
    structure_weight =
        env_float("SWSNN_STRUCTURE_WEIGHT", 1.0f-2; minimum=0.0)
    structural_interval =
        env_int("SWSNN_STRUCTURAL_INTERVAL", 25; minimum=1)
    checkpoint_interval =
        env_int("SWSNN_CHECKPOINT_INTERVAL", 10_000; minimum=1)
    log_interval = env_int("SWSNN_LOG_INTERVAL", 1_000; minimum=1)
    evaluation_states =
        env_int("SWSNN_EVAL_STATES", 128; minimum=1)
    maximum_hot_allocation =
        env_int("SWSNN_MAX_HOT_ALLOCATION_BYTES", 4096; minimum=0)
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))
    output_root = abspath(get(
        ENV,
        "SWSNN_OUTPUT",
        joinpath(@__DIR__, "trained"),
    ))
    run_id = get(
        ENV,
        "SWSNN_RUN_ID",
        "arena_scaled_u100000_" * Dates.format(now(), "yyyymmddTHHMMSS"),
    )
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("run already exists: $run_dir")
    mkpath(joinpath(run_dir, "checkpoints"))

    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(state_batch, evaluation_states),
    )
    width = 16 * cld(maximum(dataset.action_counts), 16)
    width == 80 || error("teacher_v3 candidate width drift: $width")
    training_rows = training_rows_only(dataset)
    panel_rows = fixed_training_panel(training_rows, evaluation_states)
    panel_sha256 = bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    training_rows_sha256 =
        bytes2hex(sha256(reinterpret(UInt8, training_rows)))
    model = build_model(:scaled)
    fresh_parameters, states = Lux.setup(Xoshiro(MODEL_SEED), model)
    fingerprint = source_fingerprint()

    resume_path = strip(get(ENV, "SWSNN_RESUME_CHECKPOINT", ""))
    resume_sha256 = strip(get(ENV, "SWSNN_RESUME_SHA256", ""))
    resume_payload, parent_checkpoint = isempty(resume_path) ?
        (nothing, nothing) : load_checkpoint(resume_path, resume_sha256)

    config = (;
        experiment_id=:serial_workspace_snn_arena_v1,
        role="third_model_after_preact_and_dsrln",
        run_id,
        model=graph_topology(model, fresh_parameters),
        parameter_count=parameter_count(fresh_parameters),
        maximum_updates,
        state_batch,
        target_teacher_states=maximum_updates * state_batch,
        active_workers,
        cpuset_mode,
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
        learning_rate,
        weight_decay,
        structure_weight,
        structural_interval,
        checkpoint_interval,
        dataset_path,
        candidate_width=width,
        training_rows=length(training_rows),
        training_rows_sha256,
        training_eval_states=length(panel_rows),
        training_panel_rows_sha256=panel_sha256,
        validation_rows_used=false,
        game_validation_used=false,
        sealed_seeds_used=false,
        model_seed=MODEL_SEED,
        sampler_seed=SAMPLER_SEED,
        source_fingerprint=fingerprint,
        executor=(;
            fixed_candidate_arenas=true,
            analytic_vjp=true,
            worker_local_gradients=true,
            mpmc_isbits_jobs=true,
            parallel_in_place_adamw=true,
            gc_disabled_inside_hot_training=true,
        ),
    )
    if resume_payload !== nothing
        resume_payload.config.state_batch == state_batch ||
            error("resume state batch differs")
        resume_payload.config.training_rows_sha256 ==
            training_rows_sha256 ||
            error("resume training split differs")
        resume_payload.config.model == config.model ||
            error("resume model topology differs")
        resume_payload.config.source_fingerprint == fingerprint ||
            error("resume source fingerprint differs")
    end

    initial_parameters = resume_payload === nothing ?
        copy_parameters(fresh_parameters) :
        copy_parameters(resume_payload.initial_parameters)
    trainer = ArenaTrainer(
        model,
        fresh_parameters;
        state_batch,
        width,
        learning_rate,
        weight_decay,
        structure_weight,
    )
    sampler = resume_payload === nothing ?
        EpochSampler(training_rows, Xoshiro(SAMPLER_SEED)) :
        restore_sampler(training_rows, resume_payload.sampler_state)
    progress = resume_payload === nothing ?
        ProgressTotals() : restore_progress(resume_payload.progress)
    resume_payload === nothing ||
        restore_trainer!(trainer, resume_payload)
    trainer.optimizer.step < maximum_updates || error(
        "resume update already reached target",
    )

    compile_hot_path!(
        model,
        trainer.parameters,
        dataset,
        training_rows,
        state_batch,
        width,
        active_workers,
        cpuset_mode,
    )

    eval_batch = allocate_host_batch(1; max_candidates=width)
    initial_metrics = resume_payload === nothing ?
        evaluate(
            model,
            trainer.parameters,
            states,
            dataset,
            panel_rows,
            eval_batch,
        ) : resume_payload.initial_metrics
    trace_path = joinpath(run_dir, "training_trace.tsv")
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers,
        cpuset_mode,
    )
    overall_started = time()
    segment_start_update = trainer.optimizer.step
    last_checkpoint = nothing
    team = run_with_arena_team!(executor) do running
        GC.gc()
        GC.enable(false)
        try
            while trainer.optimizer.step < maximum_updates
                fill_next_rows!(trainer.arena.rows, sampler)
                arena_update!(
                    running;
                    structural_interval,
                )
                accumulate!(progress, trainer)
                trainer.metrics.gc_seconds == 0.0 || error(
                    "GC entered the arena hot update at $(trainer.optimizer.step)",
                )
                trainer.metrics.allocation_bytes <= maximum_hot_allocation ||
                    error(
                        "hot allocation $(trainer.metrics.allocation_bytes) exceeds " *
                        "$maximum_hot_allocation bytes at update " *
                        "$(trainer.optimizer.step)",
                    )

                update = trainer.optimizer.step
                if update == 1 ||
                   update % log_interval == 0 ||
                   update == maximum_updates
                    with_gc_boundary() do
                        record = (;
                            update,
                            teacher_states=progress.teacher_states,
                            loss=trainer.last_loss.composite_loss,
                            gradient_norm=trainer.last_gradient_norm,
                            enabled_synapses=enabled_synapse_count(
                                trainer.parameters,
                            ),
                            structural_flips_total=
                                trainer.total_structural_flips,
                            states_per_second=
                                progress.teacher_states /
                                progress.hot_wall_seconds,
                            cpu_percent=
                                100.0 * progress.hot_cpu_seconds /
                                (
                                    progress.hot_wall_seconds *
                                    Threads.nthreads(:default)
                                ),
                            hot_allocation_bytes=
                                progress.hot_allocation_bytes,
                            hot_gc_seconds=progress.hot_gc_seconds,
                        )
                        append_trace(trace_path, record)
                        @info "Arena workspace SNN training" record...
                    end
                end
                if update % checkpoint_interval == 0 ||
                   update == maximum_updates
                    last_checkpoint = with_gc_boundary() do
                        checkpoint_path = joinpath(
                            run_dir,
                            "checkpoints",
                            "checkpoint_" *
                            lpad(string(update), 9, '0') *
                            ".jld2",
                        )
                        artifact = save_checkpoint!(
                            checkpoint_path,
                            trainer,
                            sampler,
                            initial_parameters,
                            config,
                            initial_metrics,
                            progress,
                        )
                        @info "Arena workspace SNN checkpoint" artifact
                        artifact
                    end
                end
            end
        finally
            GC.enable(true)
        end
        return nothing
    end
    team.result === nothing || error("unexpected arena team result")
    overall_seconds = time() - overall_started
    segment_updates = trainer.optimizer.step - segment_start_update

    final_metrics = evaluate(
        model,
        trainer.parameters,
        states,
        dataset,
        panel_rows,
        eval_batch,
    )
    weight_delta = sqrt(sum(
        abs2,
        Float64.(
            trainer.parameters.synapse_weight .-
            initial_parameters.synapse_weight
        ),
    ))
    delay_delta = sqrt(sum(
        abs2,
        Float64.(
            sigmoid.(trainer.parameters.delay_logits) .-
            sigmoid.(initial_parameters.delay_logits)
        ),
    ))
    gate_flips = count(
        structural_mask(trainer.parameters) .!=
        structural_mask(initial_parameters),
    )
    results = (;
        config,
        parent_checkpoint,
        initial=initial_metrics,
        final=final_metrics,
        deltas=(;
            composite_loss=
                final_metrics.composite_loss -
                initial_metrics.composite_loss,
            top1_agreement=
                final_metrics.top1_agreement -
                initial_metrics.top1_agreement,
            ndcg=final_metrics.ndcg - initial_metrics.ndcg,
            pairwise_accuracy=
                final_metrics.pairwise_accuracy -
                initial_metrics.pairwise_accuracy,
        ),
        learning_witness=(;
            final_update=trainer.optimizer.step,
            consumed_teacher_states=progress.teacher_states,
            last_batch_loss=trainer.last_loss.composite_loss,
            last_gradient_norm=trainer.last_gradient_norm,
            synapse_weight_l2_delta=weight_delta,
            continuous_delay_l2_delta=delay_delta,
            structural_consolidation_flips=
                trainer.total_structural_flips,
            final_mask_flips_from_initial=gate_flips,
        ),
        throughput=merge(progress_snapshot(progress), (;
            states_per_second=
                progress.teacher_states / progress.hot_wall_seconds,
            updates_per_second=
                progress.updates / progress.hot_wall_seconds,
            average_cpu_percent=
                100.0 * progress.hot_cpu_seconds /
                (
                    progress.hot_wall_seconds *
                    Threads.nthreads(:default)
                ),
            overall_seconds,
            segment_updates,
            segment_updates_per_second=
                segment_updates / overall_seconds,
        )),
        checkpoint=last_checkpoint,
        bindings=(;
            verified=all(
                binding -> binding !== nothing && binding.verified,
                team.bindings,
            ),
            cpu_set_ids=[
                binding.cpu_set_id
                for binding in team.bindings[1:active_workers]
            ],
        ),
    )
    write_json(joinpath(run_dir, "results.json"), results)
    println(JSON3.write(results))
    return results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
