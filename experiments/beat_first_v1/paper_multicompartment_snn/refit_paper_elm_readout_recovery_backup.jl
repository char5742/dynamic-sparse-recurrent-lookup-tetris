# Critical-path readout recovery for the development-scale Official ELM.
#
# The recurrent reservoir is loaded from an existing checkpoint and remains
# byte-identical.  Only the six output rows are replaced:
#   1       class-balanced logistic spike readout
#   2       ridge soma-voltage coordinate readout
#   3:6     ridge normalized regional-NMDA readouts
#
# Feature extraction uses the existing paper evaluator's reset/overlap
# semantics and calls the ELM step function directly.  Only fit32 and derived
# validation8 IDs are decoded; held-out targets are never opened.

include(joinpath(
    @__DIR__,
    "refit_official_elm_readout_critical.jl",
))

module RefitPaperELMReadoutRecoveryBackup

using JLD2
using JSON3
using LinearAlgebra
using SHA
using Statistics

const BaseRecovery = Main.RefitOfficialELMReadoutCritical
const Development = BaseRecovery.Development
const Twin = BaseRecovery.Twin
const Sealed = BaseRecovery.Sealed

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
    "readout_recovery_backup_r3e31",
)

const RIDGE_LAMBDAS =
    Float32[1.0f-6, 1.0f-5, 1.0f-4, 1.0f-3]
const LOGISTIC_LAMBDAS =
    Float32[1.0f-6, 1.0f-5, 1.0f-4]
const LOGISTIC_ITERATION_GRID = Int[25, 75, 150]

struct Options
    dataset::String
    checkpoint::String
    run_root::String
    logistic_learning_rate::Float32
    blas_threads::Int
end

function Options(;
    dataset::AbstractString=DEFAULT_DATASET,
    checkpoint::AbstractString=DEFAULT_CHECKPOINT,
    run_root::AbstractString=DEFAULT_RUN_ROOT,
    logistic_learning_rate::Real=0.02,
    blas_threads::Integer=20,
)
    logistic_learning_rate > 0 ||
        throw(ArgumentError("logistic_learning_rate must be positive"))
    blas_threads >= 1 ||
        throw(ArgumentError("blas_threads must be positive"))
    return Options(
        abspath(String(dataset)),
        abspath(String(checkpoint)),
        abspath(String(run_root)),
        Float32(logistic_learning_rate),
        Int(blas_threads),
    )
end

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _reservoir_sha256(parameters)
    io = IOBuffer()
    for name in (
        :proto_w_s,
        :input_weight,
        :input_bias,
        :memory_weight,
        :memory_bias,
        :proto_tau_m,
    )
        hasproperty(parameters, name) || continue
        value = getproperty(parameters, name)
        write(io, codeunits(String(name)))
        write(io, UInt8(0))
        write(io, Int64(ndims(value)))
        for dimension in size(value)
            write(io, Int64(dimension))
        end
        contiguous = vec(Array(value))
        write(io, reinterpret(UInt8, contiguous))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _assert_only_readout_changed(before, after)
    propertynames(before) == propertynames(after) ||
        error("parameter field set changed")
    for name in propertynames(before)
        name in (:output_weight, :output_bias) && continue
        isequal(getproperty(before, name), getproperty(after, name)) ||
            error("reservoir parameter `$name` changed")
    end
    _reservoir_sha256(before) == _reservoir_sha256(after) ||
        error("reservoir SHA changed")
    return true
end

function _continuous_metrics(
    weight,
    bias,
    features,
    target_voltage,
    target_nmda,
)
    prediction = weight * features .+ reshape(bias, :, 1)
    # Official coordinate is (clipped_mV + 67.7) * 0.1, hence 10 mV/unit.
    voltage_rmse_mv = 10.0 * sqrt(mean(
        abs2,
        Float64.(@view(prediction[1, :]) .- target_voltage),
    ))
    nmda_normalized_rmse = [
        sqrt(mean(
            abs2,
            Float64.(
                @view(prediction[region + 1, :]) .-
                @view(target_nmda[region, :]),
            ),
        ))
        for region in 1:Development.NMDA_REGIONS
    ]
    return (; voltage_rmse_mv, nmda_normalized_rmse)
end

function _fit_ridge_grid(fit, validation)
    x_mean = mean(fit.features; dims=2)
    x_centered = fit.features .- x_mean
    target = vcat(
        reshape(fit.target_voltage, 1, :),
        fit.target_nmda,
    )
    target_mean = mean(target; dims=2)
    target_centered = target .- target_mean
    observations = Float32(size(fit.features, 2))
    gram =
        x_centered * transpose(x_centered) / observations
    cross =
        target_centered * transpose(x_centered) / observations
    best = nothing
    trace = NamedTuple[]
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
        candidate = (;
            lambda,
            weight,
            bias,
            metrics,
        )
        push!(trace, (;
            lambda,
            validation_clipped_voltage_rmse_mv=
                metrics.voltage_rmse_mv,
            validation_nmda_normalized_rmse=
                metrics.nmda_normalized_rmse,
        ))
        if best === nothing ||
           (candidate.metrics.voltage_rmse_mv, lambda) <
           (best.metrics.voltage_rmse_mv, best.lambda)
            best = candidate
        end
    end
    return best, trace
end

@inline function _sigmoid(value::Float32)
    clipped = clamp(value, -40.0f0, 40.0f0)
    return inv(1.0f0 + exp(-clipped))
end

function _fit_logistic_grid(
    fit,
    validation,
    initial_weight,
    initial_bias,
    learning_rate,
)
    candidate_count = length(LOGISTIC_LAMBDAS)
    weights = repeat(
        reshape(Float32.(initial_weight), :, 1),
        1,
        candidate_count,
    )
    biases = fill(Float32(initial_bias), candidate_count)
    first_w = zeros(Float32, size(weights))
    second_w = zeros(Float32, size(weights))
    first_b = zeros(Float32, candidate_count)
    second_b = zeros(Float32, candidate_count)
    positive_count = sum(fit.target_spike)
    negative_count = length(fit.target_spike) - positive_count
    positive_count > 0 && negative_count > 0 ||
        error("fit spike target lacks a class")
    positive_weight = 0.5f0 / positive_count
    negative_weight = 0.5f0 / negative_count
    lambda_row = reshape(LOGISTIC_LAMBDAS, 1, :)
    best = nothing
    trace = NamedTuple[]
    maximum_iteration = maximum(LOGISTIC_ITERATION_GRID)

    for iteration in 1:maximum_iteration
        logits = transpose(fit.features) * weights
        logits .+= reshape(biases, 1, :)
        errors = _sigmoid.(logits)
        @inbounds for row in axes(errors, 1)
            target = fit.target_spike[row]
            scale = target == 1.0f0 ?
                positive_weight :
                negative_weight
            for column in axes(errors, 2)
                errors[row, column] =
                    (errors[row, column] - target) * scale
            end
        end
        gradient_w =
            fit.features * errors .+
            weights .* lambda_row
        gradient_b = vec(sum(errors; dims=1))

        first_w .= 0.9f0 .* first_w .+ 0.1f0 .* gradient_w
        second_w .=
            0.999f0 .* second_w .+
            0.001f0 .* abs2.(gradient_w)
        first_b .= 0.9f0 .* first_b .+ 0.1f0 .* gradient_b
        second_b .=
            0.999f0 .* second_b .+
            0.001f0 .* abs2.(gradient_b)
        correction1 = 1.0f0 - 0.9f0^iteration
        correction2 = 1.0f0 - 0.999f0^iteration
        weights .-= learning_rate .* (
            (first_w ./ correction1) ./
            (sqrt.(second_w ./ correction2) .+ 1.0f-8)
        )
        biases .-= learning_rate .* (
            (first_b ./ correction1) ./
            (sqrt.(second_b ./ correction2) .+ 1.0f-8)
        )

        iteration in LOGISTIC_ITERATION_GRID || continue
        validation_logits =
            transpose(validation.features) * weights
        validation_logits .+= reshape(biases, 1, :)
        for column in 1:candidate_count
            score = @view validation_logits[:, column]
            auroc = BaseRecovery._auroc(
                score,
                validation.target_spike,
            )
            balanced_bce = BaseRecovery._balanced_bce(
                score,
                validation.target_spike,
            )
            candidate = (;
                lambda=LOGISTIC_LAMBDAS[column],
                iteration,
                validation_auroc=auroc,
                validation_balanced_bce=balanced_bce,
                weight=copy(@view(weights[:, column])),
                bias=biases[column],
            )
            push!(trace, (;
                lambda=candidate.lambda,
                iteration,
                validation_auroc=auroc,
                validation_balanced_bce=balanced_bce,
            ))
            if best === nothing ||
               (
                   -candidate.validation_auroc,
                   candidate.validation_balanced_bce,
                   candidate.lambda,
                   candidate.iteration,
               ) <
               (
                   -best.validation_auroc,
                   best.validation_balanced_bce,
                   best.lambda,
                   best.iteration,
               )
                best = candidate
            end
        end
    end
    return best, trace
end

function run(options::Options)
    BLAS.set_num_threads(options.blas_threads)
    mkpath(options.run_root)
    isfile(options.checkpoint) ||
        error("parent checkpoint does not exist")
    checkpoint = JLD2.load(options.checkpoint)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMTwinOfficialFullCheckpoint" ||
        error("parent checkpoint kind differs")
    Int(checkpoint["format_version"]) == 1 ||
        error("parent checkpoint format differs")
    model = checkpoint["model"]
    parent_parameters = checkpoint["parameters"]
    normalizer = checkpoint["normalizer"]
    parent_parameter_sha256 =
        Twin.official_parameter_sha256(parent_parameters)
    parent_parameter_sha256 ==
        String(checkpoint["parameter_sha256"]) ||
        error("parent parameter SHA differs")
    parent_reservoir_sha256 =
        _reservoir_sha256(parent_parameters)

    dataset = Sealed._verify_manifest_and_shards(
        joinpath(options.dataset, Development.MANIFEST_NAME),
        options.dataset,
    )
    length(dataset.fit_ids) == 32 ||
        error("fit split must contain 32 trials")
    length(dataset.validation_ids) == 8 ||
        error("validation split must contain 8 trials")
    isempty(intersect(
        Set(Int.(dataset.fit_ids)),
        Set(Int.(dataset.heldout_ids)),
    )) || error("fit/heldout IDs overlap")
    isempty(intersect(
        Set(Int.(dataset.validation_ids)),
        Set(Int.(dataset.heldout_ids)),
    )) || error("validation/heldout IDs overlap")

    started = time()
    start_record = (;
        event="readout_recovery_started",
        pid=getpid(),
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(options.checkpoint),
        parent_parameter_sha256,
        parent_reservoir_sha256,
        fit_trials=length(dataset.fit_ids),
        validation_trials=length(dataset.validation_ids),
        heldout_targets_opened=false,
        blas_threads=BLAS.get_num_threads(),
    )
    println(JSON3.write(start_record))
    flush(stdout)

    # One batch per split keeps the recurrent step loop inside BLAS GEMM.
    fit = BaseRecovery._extract_features(
        dataset,
        dataset.fit_ids,
        model,
        parent_parameters,
        normalizer;
        batch_size=32,
    )
    validation = BaseRecovery._extract_features(
        dataset,
        dataset.validation_ids,
        model,
        parent_parameters,
        normalizer;
        batch_size=8,
    )
    ridge, ridge_trace = _fit_ridge_grid(fit, validation)
    logistic, logistic_trace = _fit_logistic_grid(
        fit,
        validation,
        @view(parent_parameters.output_weight[1, :]),
        parent_parameters.output_bias[1],
        options.logistic_learning_rate,
    )

    output_weight = copy(parent_parameters.output_weight)
    output_bias = copy(parent_parameters.output_bias)
    output_weight[1, :] .= logistic.weight
    output_bias[1] = logistic.bias
    output_weight[2:end, :] .= ridge.weight
    output_bias[2:end] .= ridge.bias
    recovered_parameters = merge(
        parent_parameters,
        (; output_weight, output_bias),
    )
    _assert_only_readout_changed(
        parent_parameters,
        recovered_parameters,
    )
    recovered_reservoir_sha256 =
        _reservoir_sha256(recovered_parameters)
    parent_reservoir_sha256 == recovered_reservoir_sha256 ||
        error("reservoir SHA before/after differs")
    recovered_parameter_sha256 =
        Twin.official_parameter_sha256(recovered_parameters)
    validation_selection_score =
        ridge.metrics.voltage_rmse_mv +
        (1.0 - logistic.validation_auroc)

    output_path = joinpath(
        options.run_root,
        "readout_recovered_backup_checkpoint.jld2",
    )
    JLD2.jldsave(
        output_path;
        artifact_kind="PaperELMReadoutRecoveryCheckpoint",
        format_version=1,
        recovery_implementation=
            "refit_paper_elm_readout_recovery_backup.jl",
        model,
        parameters=recovered_parameters,
        normalizer,
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(
            options.checkpoint,
        ),
        parent_parameter_sha256,
        recovered_parameter_sha256,
        parent_reservoir_sha256,
        recovered_reservoir_sha256,
        reservoir_unchanged=true,
        changed_parameter_groups=(
            "output_weight",
            "output_bias",
        ),
        decoded_split_ids=(
            fit=Int.(dataset.fit_ids),
            validation=Int.(dataset.validation_ids),
        ),
        heldout_targets_opened=false,
        selected_ridge_lambda=ridge.lambda,
        selected_logistic_lambda=logistic.lambda,
        selected_logistic_iteration=logistic.iteration,
        validation_selection_score,
        validation_clipped_voltage_rmse_mv=
            ridge.metrics.voltage_rmse_mv,
        validation_nmda_normalized_rmse=
            ridge.metrics.nmda_normalized_rmse,
        validation_spike_auroc=logistic.validation_auroc,
        validation_spike_balanced_bce=
            logistic.validation_balanced_bce,
    )

    report = (;
        event="readout_recovery_completed",
        passed=true,
        pid=getpid(),
        elapsed_seconds=time() - started,
        output_path,
        output_file_sha256=_sha256_file(output_path),
        parent_checkpoint=options.checkpoint,
        parent_checkpoint_sha256=_sha256_file(
            options.checkpoint,
        ),
        parent_parameter_sha256,
        recovered_parameter_sha256,
        parent_reservoir_sha256,
        recovered_reservoir_sha256,
        reservoir_unchanged=true,
        changed_parameter_groups=[
            "output_weight",
            "output_bias",
        ],
        fit_observations=size(fit.features, 2),
        validation_observations=
            size(validation.features, 2),
        heldout_targets_opened=false,
        selected_ridge_lambda=ridge.lambda,
        selected_logistic_lambda=logistic.lambda,
        selected_logistic_iteration=logistic.iteration,
        validation_selection_score,
        validation_clipped_voltage_rmse_mv=
            ridge.metrics.voltage_rmse_mv,
        validation_nmda_normalized_rmse=
            ridge.metrics.nmda_normalized_rmse,
        validation_spike_auroc=logistic.validation_auroc,
        validation_spike_balanced_bce=
            logistic.validation_balanced_bce,
        ridge_trace,
        logistic_trace,
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
        checkpoint=get(
            values,
            "checkpoint",
            DEFAULT_CHECKPOINT,
        ),
        run_root=get(values, "run-root", DEFAULT_RUN_ROOT),
        logistic_learning_rate=parse(
            Float32,
            get(values, "logistic-learning-rate", "0.02"),
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
    RefitPaperELMReadoutRecoveryBackup.main(ARGS)
end
