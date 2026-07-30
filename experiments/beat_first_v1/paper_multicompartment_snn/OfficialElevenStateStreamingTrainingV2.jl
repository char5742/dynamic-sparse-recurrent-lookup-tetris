module OfficialElevenStateStreamingTrainingV2

using Optimisers
using Printf
using Random
using Serialization
using SHA
using Zygote

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :OfficialElevenStateStreamingTraining)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "OfficialElevenStateStreamingTraining.jl",
        ),
    )
end
const BaseTraining =
    getfield(_PARENT, :OfficialElevenStateStreamingTraining)
const Core = BaseTraining.Core
const Metrics = BaseTraining.Metrics

export evaluate_streaming_post_burnin,
    gate_metrics,
    neuronio_window_contract,
    train_streaming_neuronio_windows

const gate_metrics = BaseTraining.gate_metrics

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("NeuronIO contract lacks $(String(name))")
    return value
end

function _bounds(value, label)
    result = Int.(collect(value))
    length(result) == 2 || error("$label must have two bounds")
    return (result[1], result[2])
end

function _sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function neuronio_window_contract(dataset, config)
    training =
        _required(dataset.manifest, :neuronio_training_window)
    heldout =
        _required(dataset.manifest, :heldout_evaluation_window)
    full_steps = Int(_required(training, :full_time_steps))
    full_steps == dataset.time_steps ||
        error("stream does not preserve the full NeuronIO trajectory")
    Float64(_required(training, :sample_dt_ms)) == 1.0 ||
        error("NeuronIO dt differs")
    Float64(_required(training, :ignore_time_from_start_ms)) == 500.0 ||
        error("NeuronIO training ignore interval differs")
    window_steps = Int(_required(training, :input_window_steps))
    window_steps == 500 && config.window == 500 ||
        error("NeuronIO/distillation training window is not 500")
    starts = _bounds(
        _required(
            training,
            :valid_window_start_indices_one_based,
        ),
        "NeuronIO start range",
    )
    starts == (501, full_steps - window_steps) ||
        error("NeuronIO Python choice bounds differ")
    starts[2] + window_steps - 1 == full_steps - 1 ||
        error("NeuronIO exclusive final sample was not preserved")
    String(_required(training, :sampling)) ==
        "uniform_with_replacement" ||
        error("NeuronIO windows are not sampled with replacement")

    evaluation = _bounds(
        _required(
            heldout,
            :evaluated_time_indices_one_based,
        ),
        "held-out evaluation range",
    )
    evaluation == (501, full_steps) ||
        error("held-out evaluation must cover all post-burn-in steps")
    Int(_required(heldout, :evaluated_steps_per_trial)) ==
        full_steps - 500 ||
        error("held-out evaluation length differs")
    return (;
        full_time_steps=full_steps,
        training_ignore_steps=500,
        training_window_steps=window_steps,
        training_start_range=starts,
        training_sampling="uniform_with_replacement",
        heldout_burnin_steps=500,
        heldout_evaluation_range=evaluation,
        heldout_evaluated_steps=full_steps - 500,
    )
end

function _free_fraction(epoch, config)
    teacher_epochs =
        max(config.epochs - config.free_rollout_epochs, 1)
    epoch > teacher_epochs && return 1.0f0
    return Float32(epoch - 1) / Float32(teacher_epochs)
end

function train_streaming_neuronio_windows(
    rng,
    dataset,
    materialize_window,
    target_mean,
    target_scale,
    config,
)
    contract = neuronio_window_contract(dataset, config)
    valid_starts = collect(
        contract.training_start_range[1]:
        contract.training_start_range[2],
    )
    parameters = Core.initial_parameters(rng)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    sampled_starts = Int[]
    for epoch in 1:config.epochs
        free_fraction = _free_fraction(epoch, config)
        total_loss = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            first_time = rand(rng, valid_starts)
            push!(sampled_starts, first_time)
            samples =
                rand(rng, dataset.train_indices, config.batch)
            batch = materialize_window(
                dataset,
                samples,
                first_time,
                contract.training_window_steps,
            )
            loss, gradient = Zygote.withgradient(parameters) do candidate
                Core.sequence_loss(
                    candidate,
                    batch.raw_input,
                    batch.target,
                    target_mean,
                    target_scale,
                    free_fraction,
                )
            end
            isfinite(loss) ||
                error("sealed 11-state distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            Core.all_finite(parameters) ||
                error("sealed 11-state parameter became non-finite")
            total_loss += Float64(loss)
        end
        record = (;
            epoch,
            loss=total_loss / config.steps_per_epoch,
            free_rollout_fraction=free_fraction,
            elapsed_seconds=time() - started,
        )
        push!(history, record)
        @printf(
            "sealed ELM 1278 -> 11-state %d/%d loss=%.6f free=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            free_fraction,
            record.elapsed_seconds,
        )
        flush(stdout)
    end
    sampling_report = (;
        contract...,
        draws=length(sampled_starts),
        with_replacement=true,
        sampled_start_min=minimum(sampled_starts),
        sampled_start_max=maximum(sampled_starts),
        sampled_unique_starts=length(unique(sampled_starts)),
        sampled_start_sequence_sha256=_sha256(sampled_starts),
    )
    return parameters, history, sampling_report
end

"""
Free-rollout evaluation with a 500-step warm-up and metrics on steps 501:end.

The state is evolved during burn-in.  Sparse detailed Ca/dendritic and their
dependent semantic coordinates are gated only where `observed` is true.
"""
function evaluate_streaming_post_burnin(
    parameters,
    dataset,
    materialize_window,
    indices,
    target_mean,
    target_scale,
    config;
    time_chunk::Integer,
    auroc_bins::Integer,
)
    contract = neuronio_window_contract(dataset, config)
    selected = Int.(collect(indices))
    chunk = Int(time_chunk)
    chunk >= 1 || throw(ArgumentError("time_chunk must be positive"))
    metrics = Metrics.StreamingMetrics(
        length(selected),
        contract.heldout_evaluated_steps;
        auroc_bins=Int(auroc_bins),
        maximum_coordinate_rmse=
            BaseTraining.MAXIMUM_COORDINATE_RMSE,
        minimum_coordinate_correlation=
            BaseTraining.MINIMUM_COORDINATE_CORRELATION,
    )
    first_metric = contract.heldout_evaluation_range[1]
    for global_index in selected
        state = repeat(parameters.initial_state, 1, 1)
        for first_time in 1:chunk:dataset.time_steps
            count = min(chunk, dataset.time_steps - first_time + 1)
            batch = materialize_window(
                dataset,
                [global_index],
                first_time,
                count,
            )
            raw_input = @view batch.raw_input[:, :, 1]
            target = @view batch.target[:, :, 1]
            observed = @view batch.observed[:, :, 1]
            for local_time in 1:count
                global_time = first_time + local_time - 1
                input = Core.project_official_input(
                    @view(raw_input[:, local_time:local_time]),
                    parameters.location_logits,
                )
                state = Core.transition(parameters, state, input)
                global_time < first_metric && continue
                raw_output =
                    Core.structured_readout(parameters, state)
                normalized = Core.normalize_target(
                    @view(target[:, local_time:local_time]),
                    target_mean,
                    target_scale,
                )
                semantic = Core.semantic_target(normalized)
                physical = Core.physical_output(
                    @view(raw_output[:, 1]),
                    target_mean,
                    target_scale,
                )
                physical_validity =
                    @view observed[:, local_time]
                semantic_validity =
                    BaseTraining._semantic_validity(physical_validity)
                BaseTraining._update_metrics!(
                    metrics,
                    physical,
                    @view(target[:, local_time]),
                    @view(state[:, 1]),
                    @view(semantic[:, 1]),
                    physical_validity,
                    semantic_validity,
                )
            end
        end
    end
    result = BaseTraining._finalize_metrics(metrics)
    return merge(
        result,
        (;
            state_warmup_steps=contract.heldout_burnin_steps,
            evaluated_time_indices_one_based=
                contract.heldout_evaluation_range,
            evaluated_steps_per_trial=
                contract.heldout_evaluated_steps,
            training_window_contract_is_distinct=true,
        ),
    )
end

end # module OfficialElevenStateStreamingTrainingV2
