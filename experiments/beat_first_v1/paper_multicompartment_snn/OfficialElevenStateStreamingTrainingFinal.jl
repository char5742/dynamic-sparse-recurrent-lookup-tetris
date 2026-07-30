module OfficialElevenStateStreamingTrainingFinal

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

export evaluate_streaming,
    gate_metrics,
    paper_window_contract,
    train_streaming_paper_windows

const evaluate_streaming = BaseTraining.evaluate_streaming
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
        error("paper window contract lacks $(String(name))")
    return value
end

function _sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

"""
Validate the exact NeuronIO window-sampling contract.

For the current 1500-step development source this gives Julia starts
`501:1000`.  The last admissible window ends at sample 1499, matching Python
`choice(N - window - ignore)` rather than a generic inclusive crop.
"""
function paper_window_contract(dataset, config)
    manifest = dataset.manifest
    full_time_steps = Int(_required(manifest, :full_time_steps))
    ignore_from_start =
        Int(_required(manifest, :ignore_time_from_start))
    input_window_size =
        Int(_required(manifest, :input_window_size))
    sampling = String(_required(manifest, :window_sampling))
    sampling == "random_uniform_with_replacement" ||
        error("paper window sampling is not uniform with replacement")
    full_time_steps == dataset.time_steps ||
        error("stream does not retain the full source trajectory")
    config.window == input_window_size ||
        error("distillation window differs from NeuronIO input_window_size")
    0 <= ignore_from_start < full_time_steps ||
        error("paper ignore interval is invalid")
    input_window_size >= 1 ||
        error("paper input window is not positive")
    last_start = full_time_steps - input_window_size
    first_start = ignore_from_start + 1
    first_start <= last_start ||
        error("paper source has no post-ignore sampling window")
    expected_starts = collect(first_start:last_start)
    declared_starts = Int.(collect(_required(
        manifest,
        :valid_window_start_indices,
    )))
    declared_starts == expected_starts ||
        error("manifest valid starts differ from NeuronIO choice semantics")
    expected_starts[end] + input_window_size - 1 ==
        full_time_steps - 1 ||
        error("paper last source sample was incorrectly made selectable")
    return (;
        full_time_steps,
        ignore_time_from_start=ignore_from_start,
        input_window_size,
        valid_window_start_indices=expected_starts,
        window_sampling=sampling,
        last_source_sample_is_not_in_training_window=true,
    )
end

function _free_fraction(epoch, config)
    teacher_epochs =
        max(config.epochs - config.free_rollout_epochs, 1)
    epoch > teacher_epochs && return 1.0f0
    return Float32(epoch - 1) / Float32(teacher_epochs)
end

"""
Train using the exact post-ignore random-window distribution.

Both the time-window start and trial sampling use replacement.  The complete
start sequence is hashed into the returned sampling report.
"""
function train_streaming_paper_windows(
    rng,
    dataset,
    materialize_window,
    target_mean,
    target_scale,
    config,
)
    contract = paper_window_contract(dataset, config)
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
            first_time = rand(
                rng,
                contract.valid_window_start_indices,
            )
            push!(sampled_starts, first_time)
            samples =
                rand(rng, dataset.train_indices, config.batch)
            batch = materialize_window(
                dataset,
                samples,
                first_time,
                contract.input_window_size,
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
                error("official paper-window distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            Core.all_finite(parameters) ||
                error("official paper-window parameter became non-finite")
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
            "official paper-window 1278 -> 11-state %d/%d loss=%.6f free=%.3f time=%.2fs\n",
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

end # module OfficialElevenStateStreamingTrainingFinal
