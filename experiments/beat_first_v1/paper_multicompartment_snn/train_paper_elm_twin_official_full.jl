# Resumable development-scale trainer for the canonical profiled ELM twin.
#
# This worker is deliberately fit/derived-validation only.  It never opens the
# held-out targets and therefore can run before the one-shot sealed V2 release
# evaluator.  A separate finalizer embeds the complete 3 x 35 training evidence
# after all workers have finished and selects exactly one checkpoint by raw
# validation voltage RMSE.

include(joinpath(
    @__DIR__,
    "train_paper_elm_twin_official_final_v2.jl",
))

module TrainPaperELMTwinOfficialFull

using Dates
using JLD2
using JSON3
using Lux
using LinearAlgebra
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const Development = Main.TrainPaperELMTwinOfficialFinal
const Twin = Development.Twin
const Sealed = Development.Sealed

const DEFAULT_RUN_ROOT = joinpath(
    dirname(dirname(dirname(@__DIR__))),
    "runs",
    "paper_elm_official_dev1500",
)
const BATCHES_PER_EPOCH = 4
const RELEASE_EPOCHS = 35
const RELEASE_RESTARTS = 3
const BASE_LEARNING_RATE = 5.0f-4
const COSINE_T_MAX = RELEASE_EPOCHS * BATCHES_PER_EPOCH

struct FullTrainerOptions
    dataset::String
    run_root::String
    trainer_run_id::String
    run_id::String
    restart_index::Int
    seed::UInt64
    epochs::Int
    batch_size::Int
    blas_threads::Int
    resume::Bool
end

function FullTrainerOptions(;
    dataset::AbstractString=Development.DEFAULT_DATASET,
    run_root::AbstractString=DEFAULT_RUN_ROOT,
    trainer_run_id::AbstractString="paper-elm-dev1500-3x35",
    run_id::AbstractString="elm-dev1500-restart-1",
    restart_index::Integer=1,
    seed::Integer=0x5457494e50524f50,
    epochs::Integer=RELEASE_EPOCHS,
    batch_size::Integer=8,
    blas_threads::Integer=6,
    resume::Bool=true,
)
    restart_index >= 1 ||
        throw(ArgumentError("restart_index must be positive"))
    epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    batch_size == 8 ||
        throw(ArgumentError("canonical development batch_size must be 8"))
    blas_threads >= 1 ||
        throw(ArgumentError("blas_threads must be positive"))
    isempty(trainer_run_id) &&
        throw(ArgumentError("trainer_run_id must not be empty"))
    isempty(run_id) && throw(ArgumentError("run_id must not be empty"))
    return FullTrainerOptions(
        abspath(String(dataset)),
        abspath(String(run_root)),
        String(trainer_run_id),
        String(run_id),
        Int(restart_index),
        UInt64(seed),
        Int(epochs),
        Int(batch_size),
        Int(blas_threads),
        resume,
    )
end

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _build_model()
    config = Twin.OfficialELMConfig(
        num_memory=1_000,
        hidden_size=2_000,
        nmda_regions=Development.NMDA_REGIONS,
        memory_tau_min_ms=0.1,
        memory_tau_max_ms=300.0,
        learn_memory_tau=false,
        delta_t_ms=1.0,
    )
    model = Twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    Twin.assert_profiled_official_elm_contract(model)
    return model
end

function _checkpoint_dir(options)
    return joinpath(
        options.run_root,
        "checkpoints",
        "restart_$(options.restart_index)",
    )
end

function _checkpoint_path(options, epoch)
    return joinpath(
        _checkpoint_dir(options),
        "epoch_$(lpad(epoch, 3, '0')).jld2",
    )
end

function _event_path(options)
    return joinpath(
        options.run_root,
        "events",
        "restart_$(options.restart_index).jsonl",
    )
end

function _append_event(path, value)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, value)
        write(io, '\n')
        flush(io)
    end
    return nothing
end

function _latest_checkpoint(options)
    for epoch in options.epochs:-1:1
        path = _checkpoint_path(options, epoch)
        isfile(path) && return path
    end
    return nothing
end

function _verify_checkpoint_identity!(checkpoint, options, dataset)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMTwinOfficialFullCheckpoint" ||
        error("checkpoint artifact kind differs")
    Int(checkpoint["format_version"]) == 1 ||
        error("checkpoint format version differs")
    String(checkpoint["trainer_run_id"]) == options.trainer_run_id ||
        error("checkpoint trainer_run_id differs")
    String(checkpoint["run_id"]) == options.run_id ||
        error("checkpoint run_id differs")
    Int(checkpoint["restart_index"]) == options.restart_index ||
        error("checkpoint restart_index differs")
    UInt64(checkpoint["seed"]) == options.seed ||
        error("checkpoint seed differs")
    String(checkpoint["manifest_sha256"]) ==
        dataset.manifest_sha256 ||
        error("checkpoint manifest digest differs")
    String(checkpoint["teacher_contract_sha256"]) ==
        dataset.teacher_contract_sha256 ||
        error("checkpoint teacher contract digest differs")
    actual = Twin.official_parameter_sha256(checkpoint["parameters"])
    actual == String(checkpoint["parameter_sha256"]) ||
        error("checkpoint parameter digest differs")
    return nothing
end

function _save_checkpoint(
    options,
    dataset,
    epoch,
    update_index,
    model,
    parameters,
    optimizer_state,
    normalizer,
    rng,
    validation_rmse,
    mean_training_total_loss,
)
    path = _checkpoint_path(options, epoch)
    mkpath(dirname(path))
    temporary = path * ".partial"
    parameter_sha256 = Twin.official_parameter_sha256(parameters)
    JLD2.jldsave(
        temporary;
        artifact_kind="PaperELMTwinOfficialFullCheckpoint",
        format_version=1,
        trainer_run_id=options.trainer_run_id,
        run_id=options.run_id,
        restart_index=options.restart_index,
        seed=options.seed,
        epoch=epoch,
        update_index=update_index,
        model=model,
        parameters=parameters,
        optimizer_state=optimizer_state,
        normalizer=normalizer,
        rng=rng,
        validation_physical_voltage_rmse_mv=validation_rmse,
        mean_training_total_loss=mean_training_total_loss,
        parameter_sha256=parameter_sha256,
        manifest_sha256=dataset.manifest_sha256,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
    )
    mv(temporary, path; force=true)
    return (
        path=path,
        checkpoint_file_sha256=_sha256_file(path),
        parameter_sha256=parameter_sha256,
    )
end

function _validation_physical_voltage_rmse(
    dataset,
    model,
    parameters,
)
    squared_error = 0.0
    observations = 0
    cache = Dict{Int,Any}()
    seen = Int[]
    for id in dataset.validation_ids
        record_index, _, item =
            Development._record_for_id(dataset, id)
        data = Development._numeric!(cache, dataset, record_index)
        Sealed._validate_numeric!(data)
        steps = size(data["target_voltage"], 1)
        steps == 1_500 ||
            error("derived-validation trial must contain 1500 bins")
        push!(seen, Int(id))
        for (window_index, start_step) in
            enumerate(Sealed._paper_window_starts(steps))
            input, actual_steps = Sealed._paper_window_input(
                data,
                item,
                start_step,
                steps,
            )
            # Reference NeuronIO evaluation resets recurrent state for every
            # overlapping 500-bin window.
            output = Twin.Core.official_elm_forward(
                model,
                parameters,
                input;
                initial_state=nothing,
            )
            local_keep_first =
                window_index == 1 ?
                1 :
                Sealed.PAPER_EVALUATION_OVERLAP_STEPS + 1
            local_keep_first <= actual_steps || continue
            global_keep_first = start_step + local_keep_first - 1
            global_keep_last = start_step + actual_steps - 1
            metric_global_first = max(global_keep_first, 501)
            metric_global_first > global_keep_last && continue
            local_metric_first =
                local_keep_first +
                metric_global_first - global_keep_first
            local_range = local_metric_first:actual_steps
            target_range = metric_global_first:global_keep_last
            predicted_mv = Twin.soma_voltage_from_coordinate(
                @view(output.voltage[local_range, :]),
            )
            target_mv = min.(
                @view(
                    data["target_voltage"][
                        target_range,
                        item:item,
                    ]
                ),
                Twin.OFFICIAL_SOMA_CLIP_MV,
            )
            squared_error += sum(
                abs2,
                Float64.(predicted_mv) .- Float64.(target_mv),
            )
            observations += length(target_mv)
        end
    end
    sort!(seen)
    seen == sort(Int.(collect(dataset.validation_ids))) ||
        error("derived-validation membership differs")
    observations == length(dataset.validation_ids) * 1_000 ||
        error("derived-validation fidelity bin count differs")
    return sqrt(squared_error / observations)
end

function _fresh_state(options, model)
    rng = Xoshiro(options.seed)
    parameters = Lux.initialparameters(rng, model)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(BASE_LEARNING_RATE),
        parameters,
    )
    return (
        epoch=0,
        update_index=0,
        parameters=parameters,
        optimizer_state=optimizer_state,
        rng=rng,
    )
end

function _load_or_initialize(options, dataset, model)
    latest = options.resume ? _latest_checkpoint(options) : nothing
    latest === nothing && return _fresh_state(options, model)
    checkpoint = JLD2.load(latest)
    _verify_checkpoint_identity!(checkpoint, options, dataset)
    return (
        epoch=Int(checkpoint["epoch"]),
        update_index=Int(checkpoint["update_index"]),
        parameters=checkpoint["parameters"],
        optimizer_state=checkpoint["optimizer_state"],
        rng=checkpoint["rng"],
    )
end

function run_worker(options::FullTrainerOptions)
    BLAS.set_num_threads(options.blas_threads)
    started = time()
    manifest_path = joinpath(
        options.dataset,
        Development.MANIFEST_NAME,
    )
    dataset = Sealed._verify_manifest_and_shards(
        manifest_path,
        options.dataset,
    )
    length(dataset.fit_ids) == 32 ||
        error("development fit split must contain 32 trials")
    length(dataset.validation_ids) == 8 ||
        error("derived-validation split must contain 8 trials")
    length(dataset.heldout_ids) == 8 ||
        error("sealed held-out split must contain 8 trials")
    normalizer = Development._fit_nmda_normalizer(dataset)
    model = _build_model()
    state = _load_or_initialize(options, dataset, model)
    parameters = state.parameters
    optimizer_state = state.optimizer_state
    rng = state.rng
    update_index = state.update_index
    event_path = _event_path(options)
    start_epoch = state.epoch + 1

    _append_event(event_path, (;
        event="worker_started",
        timestamp=string(now(UTC)),
        trainer_run_id=options.trainer_run_id,
        run_id=options.run_id,
        restart_index=options.restart_index,
        seed=string(options.seed),
        resume_epoch=state.epoch,
        target_epochs=options.epochs,
        pid=getpid(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
    ))

    for epoch in start_epoch:options.epochs
        epoch_started = time()
        epoch_losses = Float64[]
        for _ in 1:BATCHES_PER_EPOCH
            update_index += 1
            # Eight distinct fit trials per batch; batches are independent.
            order = randperm(rng, length(dataset.fit_ids))
            ids = dataset.fit_ids[order[1:options.batch_size]]
            starts = rand(
                rng,
                Development.FIRST_RANDOM_START:
                    Development.LAST_RANDOM_START,
                options.batch_size,
            )
            batch = Development._materialize_batch(
                dataset,
                ids,
                starts,
            )
            learning_rate = BASE_LEARNING_RATE * 0.5f0 * (
                1.0f0 +
                cos(
                    Float32(pi) *
                    Float32(update_index - 1) /
                    Float32(COSINE_T_MAX),
                )
            )
            Optimisers.adjust!(optimizer_state, learning_rate)
            objective, gradients = Zygote.withgradient(
                parameters,
            ) do candidate
                Development._objective(
                    model,
                    candidate,
                    normalizer,
                    batch,
                )
            end
            loss, _ = objective
            isfinite(loss) ||
                error("non-finite loss at update $update_index")
            gradient = only(gradients)
            squared_norm, finite_gradient, _ =
                Development._gradient_stats(gradient)
            finite_gradient ||
                error("non-finite gradient at update $update_index")
            squared_norm > 0 ||
                error("zero gradient at update $update_index")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                gradient,
            )
            push!(epoch_losses, Float64(loss))
        end

        validation_rmse = _validation_physical_voltage_rmse(
            dataset,
            model,
            parameters,
        )
        isfinite(validation_rmse) ||
            error("non-finite validation RMSE at epoch $epoch")
        saved = _save_checkpoint(
            options,
            dataset,
            epoch,
            update_index,
            model,
            parameters,
            optimizer_state,
            normalizer,
            rng,
            validation_rmse,
            mean(epoch_losses),
        )
        record = (;
            event="epoch_completed",
            timestamp=string(now(UTC)),
            trainer_run_id=options.trainer_run_id,
            run_id=options.run_id,
            restart_index=options.restart_index,
            seed=string(options.seed),
            epoch,
            update_index,
            mean_training_total_loss=mean(epoch_losses),
            validation_physical_voltage_rmse_mv=validation_rmse,
            checkpoint_file=replace(
                relpath(saved.path, options.run_root),
                '\\' => '/',
            ),
            checkpoint_file_sha256=
                saved.checkpoint_file_sha256,
            parameter_sha256=saved.parameter_sha256,
            elapsed_seconds=time() - epoch_started,
        )
        _append_event(event_path, record)
        println(JSON3.write(record))
        flush(stdout)
    end

    result = (;
        event="worker_completed",
        timestamp=string(now(UTC)),
        trainer_run_id=options.trainer_run_id,
        run_id=options.run_id,
        restart_index=options.restart_index,
        seed=string(options.seed),
        completed_epochs=options.epochs,
        update_index,
        elapsed_seconds=time() - started,
        manifest_sha256=dataset.manifest_sha256,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
    )
    _append_event(event_path, result)
    println(JSON3.write(result))
    return result
end

function _parse_bool(value)
    normalized = lowercase(String(value))
    normalized in ("true", "1", "yes") && return true
    normalized in ("false", "0", "no") && return false
    throw(ArgumentError("invalid boolean: $value"))
end

function _parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected positional argument: $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    restart_index = parse(
        Int,
        get(values, "restart-index", "1"),
    )
    return FullTrainerOptions(
        dataset=get(
            values,
            "dataset",
            Development.DEFAULT_DATASET,
        ),
        run_root=get(values, "run-root", DEFAULT_RUN_ROOT),
        trainer_run_id=get(
            values,
            "trainer-run-id",
            "paper-elm-dev1500-3x35",
        ),
        run_id=get(
            values,
            "run-id",
            "elm-dev1500-restart-$restart_index",
        ),
        restart_index=restart_index,
        seed=parse(UInt64, get(
            values,
            "seed",
            string(UInt64(0x5457494e50524f50)),
        )),
        epochs=parse(Int, get(
            values,
            "epochs",
            string(RELEASE_EPOCHS),
        )),
        batch_size=parse(Int, get(values, "batch-size", "8")),
        blas_threads=parse(
            Int,
            get(values, "blas-threads", "6"),
        ),
        resume=_parse_bool(get(values, "resume", "true")),
    )
end

main(arguments=ARGS) = run_worker(_parse_cli(arguments))

end # module TrainPaperELMTwinOfficialFull

if abspath(PROGRAM_FILE) == @__FILE__
    TrainPaperELMTwinOfficialFull.main(ARGS)
end
