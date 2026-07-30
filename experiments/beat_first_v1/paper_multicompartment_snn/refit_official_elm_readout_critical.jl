# Critical-path recovery for the development-scale Official ELM twin.
#
# This program never decodes held-out targets.  It treats an existing trained
# checkpoint as an immutable reservoir, extracts the exact reset/overlap
# evaluator memory features for fit and derived-validation trials, and refits
# only the six linear readout rows:
#   spike: class-balanced logistic Adam
#   voltage + four regional NMDA coordinates: ridge regression

include(joinpath(
    @__DIR__,
    "train_paper_elm_twin_official_final_v2.jl",
))

module RefitOfficialELMReadoutCritical

using JLD2
using JSON3
using LinearAlgebra
using SHA
using Statistics

const Development = Main.TrainPaperELMTwinOfficialFinal
const Twin = Development.Twin
const Sealed = Development.Sealed

const DEFAULT_DATASET =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
const DEFAULT_CHECKPOINT = joinpath(
    dirname(dirname(dirname(@__DIR__))),
    "runs",
    "paper_elm_official_dev1500",
    "paper_elm_dev1500_3x35_20260729t1053z",
    "checkpoints",
    "restart_3",
    "epoch_031.jld2",
)
const DEFAULT_RUN_ROOT = joinpath(
    dirname(dirname(dirname(@__DIR__))),
    "runs",
    "paper_elm_official_dev1500",
    "readout_recovery_r3e31",
)
const WINDOW_STARTS = (351, 701, 1_051)
const WINDOW_KEEP_FIRST = 151
const RIDGE_LAMBDAS =
    Float32[1.0f-6, 3.0f-6, 1.0f-5, 3.0f-5, 1.0f-4, 3.0f-4, 1.0f-3]

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

struct Options
    dataset::String
    checkpoint::String
    run_root::String
    logistic_iterations::Int
    logistic_learning_rate::Float32
    logistic_l2::Float32
    blas_threads::Int
end

function Options(;
    dataset::AbstractString=DEFAULT_DATASET,
    checkpoint::AbstractString=DEFAULT_CHECKPOINT,
    run_root::AbstractString=DEFAULT_RUN_ROOT,
    logistic_iterations::Integer=400,
    logistic_learning_rate::Real=0.02,
    logistic_l2::Real=1.0e-5,
    blas_threads::Integer=20,
)
    logistic_iterations >= 1 ||
        throw(ArgumentError("logistic_iterations must be positive"))
    logistic_learning_rate > 0 ||
        throw(ArgumentError("logistic_learning_rate must be positive"))
    logistic_l2 >= 0 ||
        throw(ArgumentError("logistic_l2 must be nonnegative"))
    blas_threads >= 1 ||
        throw(ArgumentError("blas_threads must be positive"))
    return Options(
        abspath(String(dataset)),
        abspath(String(checkpoint)),
        abspath(String(run_root)),
        Int(logistic_iterations),
        Float32(logistic_learning_rate),
        Float32(logistic_l2),
        Int(blas_threads),
    )
end

function _window_batch(dataset, ids, start_step, cache)
    inputs = Array{Float32,3}[]
    numeric = Any[]
    items = Int[]
    actual_steps = 0
    for id in ids
        record_index, _, item =
            Development._record_for_id(dataset, id)
        data = Development._numeric!(cache, dataset, record_index)
        Sealed._validate_numeric!(data)
        input, local_steps = Sealed._paper_window_input(
            data,
            item,
            start_step,
            size(data["target_voltage"], 1),
        )
        if actual_steps == 0
            actual_steps = local_steps
        elseif actual_steps != local_steps
            error("window batch has inconsistent time lengths")
        end
        push!(inputs, input)
        push!(numeric, data)
        push!(items, item)
    end
    return (
        input=cat(inputs...; dims=3),
        numeric,
        items,
        actual_steps,
    )
end

function _extract_features(
    dataset,
    ids,
    model,
    parameters,
    normalizer;
    batch_size=8,
)
    observations = length(ids) * 1_000
    memory_size = model.config.num_memory
    features = Matrix{Float32}(undef, memory_size, observations)
    target_voltage = Vector{Float32}(undef, observations)
    target_spike = Vector{Float32}(undef, observations)
    target_nmda =
        Matrix{Float32}(undef, Development.NMDA_REGIONS, observations)
    cache = Dict{Int,Any}()
    first_column = 1

    for first_id in 1:batch_size:length(ids)
        last_id = min(first_id + batch_size - 1, length(ids))
        group = ids[first_id:last_id]
        group_size = length(group)
        for start_step in WINDOW_STARTS
            window =
                _window_batch(dataset, group, start_step, cache)
            state = Twin.Core.initial_official_elm_state(
                model,
                group_size;
                element_type=Float32,
            )
            for local_step in 1:window.actual_steps
                result = Twin.Core.official_elm_step(
                    model,
                    parameters,
                    state,
                    @view(window.input[:, local_step, :]),
                )
                state = result.state
                local_step < WINDOW_KEEP_FIRST && continue
                columns =
                    first_column:(first_column + group_size - 1)
                features[:, columns] .= result.memory
                global_step = start_step + local_step - 1
                for local_item in 1:group_size
                    column = first_column + local_item - 1
                    data = window.numeric[local_item]
                    item = window.items[local_item]
                    target_voltage[column] =
                        Twin.preprocess_soma_voltage(
                            Float32(data["target_voltage"][
                                global_step,
                                item,
                            ]),
                        )
                    target_spike[column] =
                        Float32(data["target_spike"][
                            global_step,
                            item,
                        ])
                    @inbounds for region in
                        1:Development.NMDA_REGIONS
                        target_nmda[region, column] = (
                            Float32(data["target_nmda"][
                                region,
                                global_step,
                                item,
                            ]) -
                            normalizer.nmda_mean[region]
                        ) / normalizer.nmda_scale[region]
                    end
                end
                first_column += group_size
            end
        end
    end
    first_column == observations + 1 ||
        error("feature extraction observation count differs")
    return (;
        features,
        target_voltage,
        target_spike,
        target_nmda,
    )
end

function _auroc(score, target)
    label = target .== 1.0f0
    positives = count(label)
    negatives = length(label) - positives
    positives > 0 && negatives > 0 ||
        error("AUROC requires both spike classes")
    order = sortperm(score; alg=MergeSort)
    rank_sum = 0.0
    first = 1
    while first <= length(order)
        last = first
        while last < length(order) &&
              score[order[last + 1]] == score[order[first]]
            last += 1
        end
        average_rank = (first + last) / 2
        @inbounds for position in first:last
            label[order[position]] &&
                (rank_sum += average_rank)
        end
        first = last + 1
    end
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
end

@inline function _sigmoid(value::Float32)
    clipped = clamp(value, -40.0f0, 40.0f0)
    return inv(1.0f0 + exp(-clipped))
end

function _balanced_bce(logit, target)
    positives = sum(target)
    negatives = length(target) - positives
    positives > 0 && negatives > 0 ||
        error("balanced BCE requires both spike classes")
    positive_loss = 0.0
    negative_loss = 0.0
    @inbounds for index in eachindex(logit, target)
        value =
            max(logit[index], 0.0f0) -
            logit[index] * target[index] +
            log1p(exp(-abs(logit[index])))
        if target[index] == 1.0f0
            positive_loss += value
        else
            negative_loss += value
        end
    end
    return 0.5 * positive_loss / positives +
           0.5 * negative_loss / negatives
end

function _normalized_rmse(predicted, target)
    centered = target .- mean(target)
    denominator = sum(abs2, Float64.(centered))
    error2 = sum(abs2, Float64.(predicted .- target))
    if denominator <= eps(Float64)
        return error2 <= eps(Float64) ? 0.0 : Inf
    end
    return sqrt(error2 / denominator)
end

function _continuous_metrics(
    weight,
    bias,
    features,
    target_voltage,
    target_nmda,
)
    prediction =
        weight * features .+ reshape(bias, :, 1)
    voltage_rmse_mv = 30.0 * sqrt(mean(
        abs2,
        Float64.(
            @view(prediction[1, :]) .- target_voltage,
        ),
    ))
    nmda = [
        _normalized_rmse(
            @view(prediction[region + 1, :]),
            @view(target_nmda[region, :]),
        )
        for region in 1:Development.NMDA_REGIONS
    ]
    return (; voltage_rmse_mv, nmda_normalized_rmse=nmda)
end

function _fit_continuous(fit, validation)
    x_mean = mean(fit.features; dims=2)
    x_centered = fit.features .- x_mean
    target = vcat(
        reshape(fit.target_voltage, 1, :),
        fit.target_nmda,
    )
    target_mean = mean(target; dims=2)
    target_centered = target .- target_mean
    observations = size(fit.features, 2)
    gram =
        x_centered * transpose(x_centered) /
        Float32(observations)
    cross =
        target_centered * transpose(x_centered) /
        Float32(observations)
    best = nothing
    candidates = NamedTuple[]
    for lambda in RIDGE_LAMBDAS
        regularized = copy(gram)
        @inbounds for index in axes(regularized, 1)
            regularized[index, index] += lambda
        end
        factor = cholesky(Symmetric(regularized))
        weight = transpose(factor \ transpose(cross))
        bias = vec(target_mean - weight * x_mean)
        metrics = _continuous_metrics(
            weight,
            bias,
            validation.features,
            validation.target_voltage,
            validation.target_nmda,
        )
        finite_nmda = filter(isfinite, metrics.nmda_normalized_rmse)
        maximum_nmda = isempty(finite_nmda) ? Inf : maximum(finite_nmda)
        score = max(metrics.voltage_rmse_mv, maximum_nmda)
        candidate = (;
            lambda,
            weight,
            bias,
            metrics,
            score,
        )
        push!(candidates, (;
            lambda,
            validation_voltage_rmse_mv=metrics.voltage_rmse_mv,
            validation_nmda_normalized_rmse=
                metrics.nmda_normalized_rmse,
            score,
        ))
        if best === nothing ||
           (candidate.score, candidate.metrics.voltage_rmse_mv, lambda) <
           (best.score, best.metrics.voltage_rmse_mv, best.lambda)
            best = candidate
        end
    end
    return best, candidates
end

function _fit_spike(
    fit,
    validation,
    initial_weight,
    initial_bias,
    options,
)
    weight = copy(initial_weight)
    bias = Float32(initial_bias)
    first_moment_w = zeros(Float32, length(weight))
    second_moment_w = zeros(Float32, length(weight))
    first_moment_b = 0.0f0
    second_moment_b = 0.0f0
    positive_count = sum(fit.target_spike)
    negative_count = length(fit.target_spike) - positive_count
    positive_count > 0 && negative_count > 0 ||
        error("fit spike target lacks a class")
    best = nothing
    trace = NamedTuple[]

    for iteration in 1:options.logistic_iterations
        logit = vec(transpose(weight) * fit.features) .+ bias
        probability = _sigmoid.(logit)
        error = probability .- fit.target_spike
        @inbounds for index in eachindex(error)
            error[index] *= fit.target_spike[index] == 1.0f0 ?
                0.5f0 / positive_count :
                0.5f0 / negative_count
        end
        gradient_w =
            fit.features * error .+
            options.logistic_l2 .* weight
        gradient_b = sum(error)

        first_moment_w .=
            0.9f0 .* first_moment_w .+
            0.1f0 .* gradient_w
        second_moment_w .=
            0.999f0 .* second_moment_w .+
            0.001f0 .* abs2.(gradient_w)
        first_moment_b =
            0.9f0 * first_moment_b + 0.1f0 * gradient_b
        second_moment_b =
            0.999f0 * second_moment_b +
            0.001f0 * gradient_b * gradient_b
        correction1 = 1.0f0 - 0.9f0^iteration
        correction2 = 1.0f0 - 0.999f0^iteration
        step_scale = options.logistic_learning_rate
        weight .-= step_scale .* (
            (first_moment_w ./ correction1) ./
            (sqrt.(second_moment_w ./ correction2) .+ 1.0f-8)
        )
        bias -= step_scale * (
            (first_moment_b / correction1) /
            (sqrt(second_moment_b / correction2) + 1.0f-8)
        )

        if iteration == 1 ||
           iteration % 5 == 0 ||
           iteration == options.logistic_iterations
            validation_logit =
                vec(transpose(weight) * validation.features) .+ bias
            validation_auroc = _auroc(
                validation_logit,
                validation.target_spike,
            )
            validation_bce = _balanced_bce(
                validation_logit,
                validation.target_spike,
            )
            record = (;
                iteration,
                validation_auroc,
                validation_balanced_bce=validation_bce,
            )
            push!(trace, record)
            if best === nothing ||
               (-validation_auroc, validation_bce, iteration) <
               (
                   -best.validation_auroc,
                   best.validation_balanced_bce,
                   best.iteration,
               )
                best = (;
                    iteration,
                    validation_auroc,
                    validation_balanced_bce=validation_bce,
                    weight=copy(weight),
                    bias,
                )
            end
        end
    end
    return best, trace
end

function _reservoir_unchanged(before, after)
    for name in propertynames(before)
        name in (:output_weight, :output_bias) && continue
        isequal(getproperty(before, name), getproperty(after, name)) ||
            return false
    end
    return true
end

function run(options::Options)
    BLAS.set_num_threads(options.blas_threads)
    mkpath(options.run_root)
    dataset = Sealed._verify_manifest_and_shards(
        joinpath(options.dataset, Development.MANIFEST_NAME),
        options.dataset,
    )
    length(dataset.fit_ids) == 32 ||
        error("fit split must contain 32 trials")
    length(dataset.validation_ids) == 8 ||
        error("validation split must contain 8 trials")
    checkpoint = JLD2.load(options.checkpoint)
    model = checkpoint["model"]
    parent_parameters = checkpoint["parameters"]
    normalizer = checkpoint["normalizer"]
    parent_parameter_sha256 =
        Twin.official_parameter_sha256(parent_parameters)
    parent_parameter_sha256 ==
        String(checkpoint["parameter_sha256"]) ||
        error("parent parameter digest differs")

    println(JSON3.write((;
        event="feature_extraction_started",
        pid=getpid(),
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(options.checkpoint),
        parent_parameter_sha256,
        fit_trials=length(dataset.fit_ids),
        validation_trials=length(dataset.validation_ids),
        heldout_targets_opened=false,
    )))
    flush(stdout)

    fit = _extract_features(
        dataset,
        dataset.fit_ids,
        model,
        parent_parameters,
        normalizer,
    )
    validation = _extract_features(
        dataset,
        dataset.validation_ids,
        model,
        parent_parameters,
        normalizer,
    )
    continuous, ridge_trace =
        _fit_continuous(fit, validation)
    spike, spike_trace = _fit_spike(
        fit,
        validation,
        vec(@view(parent_parameters.output_weight[1, :])),
        parent_parameters.output_bias[1],
        options,
    )

    output_weight = copy(parent_parameters.output_weight)
    output_bias = copy(parent_parameters.output_bias)
    output_weight[1, :] .= spike.weight
    output_bias[1] = spike.bias
    output_weight[2:end, :] .= continuous.weight
    output_bias[2:end] .= continuous.bias
    recovered_parameters = merge(
        parent_parameters,
        (; output_weight, output_bias),
    )
    _reservoir_unchanged(parent_parameters, recovered_parameters) ||
        error("readout recovery changed reservoir parameters")
    recovered_parameter_sha256 =
        Twin.official_parameter_sha256(recovered_parameters)

    output_path =
        joinpath(options.run_root, "readout_recovered_checkpoint.jld2")
    JLD2.jldsave(
        output_path;
        artifact_kind="PaperELMReadoutRecoveryCheckpoint",
        format_version=1,
        model,
        parameters=recovered_parameters,
        normalizer,
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(options.checkpoint),
        parent_parameter_sha256,
        recovered_parameter_sha256,
        reservoir_unchanged=true,
        heldout_targets_opened=false,
        selected_ridge_lambda=continuous.lambda,
        selected_logistic_iteration=spike.iteration,
        validation_voltage_rmse_mv=
            continuous.metrics.voltage_rmse_mv,
        validation_nmda_normalized_rmse=
            continuous.metrics.nmda_normalized_rmse,
        validation_spike_auroc=spike.validation_auroc,
    )
    report = (;
        event="readout_recovery_completed",
        passed=true,
        pid=getpid(),
        output_path,
        output_file_sha256=_sha256_file(output_path),
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(options.checkpoint),
        parent_parameter_sha256,
        recovered_parameter_sha256,
        reservoir_unchanged=true,
        heldout_targets_opened=false,
        fit_observations=size(fit.features, 2),
        validation_observations=size(validation.features, 2),
        selected_ridge_lambda=continuous.lambda,
        selected_logistic_iteration=spike.iteration,
        validation_voltage_rmse_mv=
            continuous.metrics.voltage_rmse_mv,
        validation_nmda_normalized_rmse=
            continuous.metrics.nmda_normalized_rmse,
        validation_spike_auroc=spike.validation_auroc,
        validation_spike_balanced_bce=
            spike.validation_balanced_bce,
        ridge_trace,
        spike_trace,
    )
    report_path = joinpath(options.run_root, "report.json")
    open(report_path, "w") do io
        JSON3.write(io, report)
        write(io, '\n')
    end
    println(JSON3.write(report))
    flush(stdout)
    return report
end

function _parse(arguments)
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
    return Options(
        dataset=get(values, "dataset", DEFAULT_DATASET),
        checkpoint=get(values, "checkpoint", DEFAULT_CHECKPOINT),
        run_root=get(values, "run-root", DEFAULT_RUN_ROOT),
        logistic_iterations=parse(
            Int,
            get(values, "logistic-iterations", "400"),
        ),
        logistic_learning_rate=parse(
            Float32,
            get(values, "logistic-learning-rate", "0.02"),
        ),
        logistic_l2=parse(
            Float32,
            get(values, "logistic-l2", "1e-5"),
        ),
        blas_threads=parse(
            Int,
            get(values, "blas-threads", "20"),
        ),
    )
end

main(arguments=ARGS) = run(_parse(arguments))

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    RefitOfficialELMReadoutCritical.main(ARGS)
end
