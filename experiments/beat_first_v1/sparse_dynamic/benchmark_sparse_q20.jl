using Random
using Statistics
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "SparseQ20.jl"))
using .SparseQ20
include(joinpath(@__DIR__, "..", "training", "core.jl"))
using .BeatFirstTrainingCore

const BENCHMARK_DENSE_ORACLE_ONLY = true
const INDEX_SCALING_MAX_TOTAL_LATENCY_RATIO = 2.50
const INDEX_SCALING_MAX_SCORED_FRACTION_RATIO = 1.50
const INDEX_SCALING_MAX_LATENCY_PER_SCORED_ROW_RATIO = 1.50

function synthetic_benchmark_input(rng::AbstractRNG)
    scale = 0.05f0
    return (
        candidate=scale .* randn(rng, Float32, 24, 10, 1, 1),
        difference=scale .* randn(rng, Float32, 24, 10, 1, 1),
        next_hold=scale .* randn(rng, Float32, 7, 6, 1),
        aux=scale .* randn(rng, Float32, 37, 1),
    )
end

function _environment_flag(name::AbstractString, default::Bool=false)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    throw(ArgumentError("$name must be a boolean flag; got $(repr(raw))"))
end

function real_teacher_benchmark_corpus(
    root::AbstractString;
    samples::Int=64,
)
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) || error("teacher manifest does not exist: $manifest_path")
    manifest = BeatFirstTrainingCore.JSON3.read(read(manifest_path, String))
    train_parts = filter(
        part -> String(part.split) == "train" && Int(part.row_count) >= samples,
        collect(manifest.parts),
    )
    isempty(train_parts) && error(
        "teacher manifest has no train part with at least $samples rows",
    )
    part = first(train_parts)
    part_path = normpath(joinpath(root, String(part.relative_path)))
    isfile(part_path) || error("teacher part does not exist: $part_path")
    filesize(part_path) == Int(part.bytes) || error("teacher part byte mismatch")
    part_sha256 = bytes2hex(open(BeatFirstTrainingCore.sha256, part_path))
    part_sha256 == lowercase(String(part.sha256)) || error("teacher part SHA mismatch")

    corpus = Any[]
    targets = Vector{Float32}[]
    sampled_rows = Int[]
    sampled_actions = Int[]
    BeatFirstTrainingCore.JLD2.jldopen(part_path, "r") do file
        action_counts = Int.(file["action_counts"])
        state_rows = round.(Int, range(1, length(action_counts); length=samples))
        length(unique(state_rows)) == samples || error("teacher row sampler repeated rows")
        dataset = (;
            placements=file["placements"],
            ren=Float32.(file["ren"]),
            back_to_back=Float32.(file["back_to_back"]),
            tspin=Float32.(file["tspin"]),
            geometry_cache=nothing,
        )
        boards = file["boards"]
        queues = file["queues"]
        teacher_q = Float32.(file["teacher_q"])
        death = haskey(file, "death") ? Bool.(file["death"]) : nothing

        for (sample_index, row) in pairs(state_rows)
            action_count = action_counts[row]
            action_count >= 1 || error("teacher row $row has no candidates")
            action = mod1(17 * sample_index + row, action_count)
            board = Float32.(@view boards[:, :, 1, row])
            queue = Float32.(@view queues[:, :, row])
            entry = BeatFirstTrainingCore._candidate_geometry_entry(
                dataset,
                board,
                row,
                action,
            )
            input = (;
                candidate=reshape(copy(entry.after), 24, 10, 1, 1),
                difference=reshape(copy(entry.difference), 24, 10, 1, 1),
                next_hold=reshape(copy(queue), 7, 6, 1),
                aux=reshape(copy(entry.aux), AUX_FEATURES, 1),
            )

            teacher = @view teacher_q[1:action_count, row]
            teacher_scale = max(std(teacher; corrected=false), 1.0f-4)
            teacher_z = (teacher[action] - mean(teacher)) / teacher_scale
            target = zeros(Float32, OUTPUT_DIM)
            target[Q_OUTPUT] = teacher_z
            target[DEATH_OUTPUT] = death === nothing ? 0.0f0 : Float32(death[action, row])
            target[QUANTILE_OUTPUTS] .= teacher_z
            target[GEOMETRY_OUTPUTS] .= Float32[
                entry.line_clear / 4,
                entry.geometry.max_height / 24,
                entry.geometry.holes / 240,
                entry.geometry.cavities / 240,
            ]
            push!(corpus, input)
            push!(targets, target)
            push!(sampled_rows, row)
            push!(sampled_actions, action)
        end
    end
    typed_corpus = convert(Vector{typeof(first(corpus))}, corpus)
    return (
        corpus=typed_corpus,
        targets=targets,
        source="teacher_train_part",
        e036_eligible=true,
        part_path=abspath(part_path),
        part_sha256,
        sampled_rows,
        sampled_actions,
    )
end

function benchmark_corpus(rng::AbstractRNG, arguments::Vector{String}=ARGS)
    dataset_arguments = filter(arg -> startswith(arg, "--dataset="), arguments)
    length(dataset_arguments) <= 1 || error("--dataset may be supplied only once")
    unknown = filter(
        arg -> arg != "--synthetic-smoke" && !startswith(arg, "--dataset="),
        arguments,
    )
    isempty(unknown) || error("unknown benchmark arguments: $(join(unknown, ' '))")
    synthetic_smoke = "--synthetic-smoke" in arguments ||
        _environment_flag("BEAT_SPARSE_BENCH_SYNTHETIC_SMOKE", false)
    if synthetic_smoke
        isempty(dataset_arguments) || error(
            "--dataset and --synthetic-smoke are mutually exclusive",
        )
        corpus = [synthetic_benchmark_input(rng) for _ in 1:64]
        targets = [0.05f0 .* randn(rng, Float32, OUTPUT_DIM) for _ in corpus]
        return (;
            corpus,
            targets,
            source="synthetic_smoke_only",
            e036_eligible=false,
            part_path="NONE",
            part_sha256="NONE",
            sampled_rows=Int[],
            sampled_actions=Int[],
        )
    end
    root = isempty(dataset_arguments) ?
        get(
            ENV,
            "BEAT_SPARSE_BENCH_DATASET",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        ) : split(only(dataset_arguments), "="; limit=2)[2]
    isempty(strip(root)) && error("teacher dataset path is empty")
    return real_teacher_benchmark_corpus(root)
end

function elapsed_ns(f)
    started = time_ns()
    value = f()
    return Float64(time_ns() - started), value
end

function time_repeated(f, samples::Int; warmup::Bool=true)
    warmup && f()
    timings = Vector{Float64}(undef, samples)
    sink = nothing
    for sample in 1:samples
        timings[sample], sink = elapsed_ns(f)
    end
    return timings, sink
end

function time_items!(items, f)
    timings = Vector{Float64}(undef, length(items))
    sink = nothing
    for i in eachindex(items)
        timings[i], sink = elapsed_ns(() -> f(items[i]))
    end
    return timings, sink
end

function report_latency(name::AbstractString, timings_ns::AbstractVector)
    milliseconds = timings_ns ./ 1.0e6
    @printf(
        "latency\t%s\tsamples=%d\tp50_ms=%.6f\tp95_ms=%.6f\tmean_ms=%.6f\tmin_ms=%.6f\n",
        name,
        length(milliseconds),
        median(milliseconds),
        quantile(milliseconds, 0.95),
        mean(milliseconds),
        minimum(milliseconds),
    )
end

function make_accumulator(vjp::SelectedVJP; copies::Int=1, reduce::Bool=false)
    accumulator = SparseRowGradientAccumulator(
        capacity=copies * length(vjp.selected_ids),
    )
    for _ in 1:copies
        accumulate_columns!(accumulator, vjp.selected_ids, vjp.dtheta)
    end
    reduce && reduce_gradients!(accumulator)
    return accumulator
end

"""Benchmark-only same-parameter dense twin.

This function deliberately evaluates every one of the 32,768 neurons.  It is
an explicit timing oracle and is never imported by `SparseQ20.jl`, never used
for routing, and never offered as a production fallback.
"""
function dense_twin_oracle(
    model::SparseNeuronBank,
    q::Vector{Float32},
    x::Vector{Float32},
)
    latent = zeros(Float32, LATENT_DIM)
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(VALUE_DIM)))
    bank_scale = inv(sqrt(Float32(NEURON_COUNT)))
    value_first = ROUTE_DIM + 1
    latent_first = ROUTE_DIM + VALUE_DIM + 1

    @inbounds for neuron in 1:NEURON_COUNT
        route_sum = 0.0f0
        for dimension in 1:ROUTE_DIM
            route_sum = muladd(model.theta[dimension, neuron], q[dimension], route_sum)
        end
        value_sum = 0.0f0
        for dimension in 1:VALUE_DIM
            value_sum = muladd(
                model.theta[value_first + dimension - 1, neuron],
                x[dimension],
                value_sum,
            )
        end
        activation = silu(route_sum * route_scale + value_sum * value_scale)
        for dimension in 1:LATENT_DIM
            latent[dimension] = muladd(
                activation,
                model.theta[latent_first + dimension - 1, neuron],
                latent[dimension],
            )
        end
    end
    @inbounds for dimension in 1:LATENT_DIM
        latent[dimension] *= bank_scale
    end

    output = Vector{Float32}(undef, OUTPUT_DIM)
    @inbounds for output_index in 1:OUTPUT_DIM
        accumulator = model.bias[output_index]
        for dimension in 1:LATENT_DIM
            accumulator = muladd(
                model.head[output_index, dimension],
                latent[dimension],
                accumulator,
            )
        end
        output[output_index] = accumulator
    end
    return output
end

# Everything below this marker is a benchmark-only dense training oracle.  It
# intentionally allocates a full-bank tape, gradient, and optimizer state so
# E036 compares the literal cost of training all 19.9M parameters against the
# selected-only production path.  None of these names is exported or included
# by `SparseQ20.jl`.
mutable struct DenseTwinTrainingWorkspace
    q::Vector{Float32}
    x::Vector{Float32}
    preactivation::Vector{Float32}
    activation::Vector{Float32}
    latent::Vector{Float32}
    output::Vector{Float32}
    output_cotangent::Vector{Float32}
    dlatent::Vector{Float32}
    activation_cotangent::Vector{Float32}
    scaled_cotangent::Vector{Float32}
    dtheta::Matrix{Float32}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
end

function DenseTwinTrainingWorkspace()
    return DenseTwinTrainingWorkspace(
        Vector{Float32}(undef, ROUTE_DIM),
        Vector{Float32}(undef, VALUE_DIM),
        Vector{Float32}(undef, NEURON_COUNT),
        Vector{Float32}(undef, NEURON_COUNT),
        zeros(Float32, LATENT_DIM),
        Vector{Float32}(undef, OUTPUT_DIM),
        Vector{Float32}(undef, OUTPUT_DIM),
        zeros(Float32, LATENT_DIM),
        zeros(Float32, NEURON_COUNT),
        zeros(Float32, NEURON_COUNT),
        Matrix{Float32}(undef, ROW_DIM, NEURON_COUNT),
        Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM),
        Vector{Float32}(undef, OUTPUT_DIM),
    )
end

mutable struct DenseTwinRowWiseAdaGradState
    accumulator_sq::Vector{Float32}
    learning_rate::Float32
    epsilon::Float32
    step::UInt64
end

function DenseTwinRowWiseAdaGradState(
    ;
    learning_rate::Real=1.0f-2,
    epsilon::Real=1.0f-8,
)
    lr = Float32(learning_rate)
    eps = Float32(epsilon)
    isfinite(lr) && lr > 0.0f0 || throw(ArgumentError("learning_rate"))
    isfinite(eps) && eps > 0.0f0 || throw(ArgumentError("epsilon"))
    return DenseTwinRowWiseAdaGradState(
        zeros(Float32, NEURON_COUNT),
        lr,
        eps,
        0,
    )
end

"""Full-bank forward with a reusable tape for the benchmark-only dense twin."""
function dense_twin_forward!(
    workspace::DenseTwinTrainingWorkspace,
    model::SparseNeuronBank,
)
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(VALUE_DIM)))
    bank_scale = inv(sqrt(Float32(NEURON_COUNT)))
    value_first = ROUTE_DIM + 1
    latent_first = ROUTE_DIM + VALUE_DIM + 1
    route_theta = @view model.theta[1:ROUTE_DIM, :]
    value_theta = @view model.theta[value_first:(latent_first - 1), :]
    latent_theta = @view model.theta[latent_first:ROW_DIM, :]

    # All full-bank linear algebra in the timed dense reference is dispatched
    # explicitly through native BLAS.  The row-block views preserve stride-1
    # within a column and use the original 608-element leading dimension. The scalar
    # oracle above is retained only as a diagnostic and is not the gate value.
    BLAS.gemv!(
        'T',
        route_scale,
        route_theta,
        workspace.q,
        0.0f0,
        workspace.preactivation,
    )
    BLAS.gemv!(
        'T',
        value_scale,
        value_theta,
        workspace.x,
        1.0f0,
        workspace.preactivation,
    )
    @inbounds @simd for neuron in 1:NEURON_COUNT
        workspace.activation[neuron] = silu(workspace.preactivation[neuron])
    end
    BLAS.gemv!(
        'N',
        bank_scale,
        latent_theta,
        workspace.activation,
        0.0f0,
        workspace.latent,
    )
    BLAS.gemv!(
        'N',
        1.0f0,
        model.head,
        workspace.latent,
        0.0f0,
        workspace.output,
    )
    @inbounds @simd for output_index in 1:OUTPUT_DIM
        workspace.output[output_index] += model.bias[output_index]
    end
    return workspace.output
end

"""Half mean-squared error and its output cotangent, without AD."""
function mse_loss_cotangent!(
    output_cotangent::Vector{Float32},
    output::AbstractVector{Float32},
    target::AbstractVector{Float32},
)
    length(output_cotangent) == OUTPUT_DIM || throw(DimensionMismatch("cotangent"))
    length(output) == OUTPUT_DIM || throw(DimensionMismatch("output"))
    length(target) == OUTPUT_DIM || throw(DimensionMismatch("target"))
    inverse_outputs = inv(Float32(OUTPUT_DIM))
    loss = 0.0f0
    @inbounds for output_index in 1:OUTPUT_DIM
        difference = output[output_index] - target[output_index]
        loss = muladd(0.5f0 * inverse_outputs, difference * difference, loss)
        output_cotangent[output_index] = difference * inverse_outputs
    end
    return loss
end

"""Manual full-bank parameter VJP for the benchmark-only dense twin.

Unlike `vjp_selected`, this writes every element of the 608x32,768 gradient
but, like the production sparse path, deliberately omits unused feature-input
cotangents.  The implementation is allocation-free after workspace construction
so allocator/GC cost does not artificially make the dense reference slower.
"""
function dense_twin_vjp!(
    workspace::DenseTwinTrainingWorkspace,
    model::SparseNeuronBank,
)
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(VALUE_DIM)))
    bank_scale = inv(sqrt(Float32(NEURON_COUNT)))
    value_first = ROUTE_DIM + 1
    latent_first = ROUTE_DIM + VALUE_DIM + 1
    latent_theta = @view model.theta[latent_first:ROW_DIM, :]
    droute_theta = @view workspace.dtheta[1:ROUTE_DIM, :]
    dvalue_theta = @view workspace.dtheta[value_first:(latent_first - 1), :]
    dlatent_theta = @view workspace.dtheta[latent_first:ROW_DIM, :]

    BLAS.gemm!(
        'N',
        'T',
        1.0f0,
        reshape(workspace.output_cotangent, OUTPUT_DIM, 1),
        reshape(workspace.latent, LATENT_DIM, 1),
        0.0f0,
        workspace.dhead,
    )
    BLAS.gemv!(
        'T',
        1.0f0,
        model.head,
        workspace.output_cotangent,
        0.0f0,
        workspace.dlatent,
    )
    copyto!(workspace.dbias, workspace.output_cotangent)

    BLAS.gemm!(
        'N',
        'T',
        bank_scale,
        reshape(workspace.dlatent, LATENT_DIM, 1),
        reshape(workspace.activation, NEURON_COUNT, 1),
        0.0f0,
        dlatent_theta,
    )
    BLAS.gemv!(
        'T',
        bank_scale,
        latent_theta,
        workspace.dlatent,
        0.0f0,
        workspace.activation_cotangent,
    )
    @inbounds @simd for neuron in 1:NEURON_COUNT
        workspace.activation_cotangent[neuron] *=
            silu_derivative(workspace.preactivation[neuron])
        workspace.scaled_cotangent[neuron] =
            workspace.activation_cotangent[neuron] * route_scale
    end
    BLAS.gemm!(
        'N',
        'T',
        1.0f0,
        reshape(workspace.q, ROUTE_DIM, 1),
        reshape(workspace.scaled_cotangent, NEURON_COUNT, 1),
        0.0f0,
        droute_theta,
    )
    @inbounds @simd for neuron in 1:NEURON_COUNT
        workspace.scaled_cotangent[neuron] =
            workspace.activation_cotangent[neuron] * value_scale
    end
    BLAS.gemm!(
        'N',
        'T',
        1.0f0,
        reshape(workspace.x, VALUE_DIM, 1),
        reshape(workspace.scaled_cotangent, NEURON_COUNT, 1),
        0.0f0,
        dvalue_theta,
    )
    return workspace
end

"""Update all bank rows with the same row-wise AdaGrad rule as sparse v1."""
function dense_twin_adagrad_step!(
    theta::Matrix{Float32},
    state::DenseTwinRowWiseAdaGradState,
    gradient::Matrix{Float32},
)
    size(theta) == (ROW_DIM, NEURON_COUNT) || throw(DimensionMismatch("theta"))
    size(gradient) == size(theta) || throw(DimensionMismatch("gradient"))
    state.step == typemax(UInt64) && error("dense twin optimizer step overflow")
    state.step += 1
    learning_rate = Float64(state.learning_rate)
    epsilon = Float64(state.epsilon)

    # Columns and row-wise states are disjoint.  Give the dense competitor the
    # strongest native CPU implementation available in this script instead of
    # serializing an embarrassingly parallel 20M-parameter update.
    Threads.@threads :static for neuron in 1:NEURON_COUNT
        @inbounds begin
            squared_norm = 0.0
            @simd for dimension in 1:ROW_DIM
                value = Float64(gradient[dimension, neuron])
                squared_norm += value * value
            end
            new_accumulator =
                Float64(state.accumulator_sq[neuron]) + squared_norm / ROW_DIM
            state.accumulator_sq[neuron] = Float32(new_accumulator)
            inverse_scale = learning_rate / (sqrt(new_accumulator) + epsilon)
            @simd for dimension in 1:ROW_DIM
                theta[dimension, neuron] = Float32(
                    Float64(theta[dimension, neuron]) -
                    inverse_scale * Float64(gradient[dimension, neuron]),
                )
            end
        end
    end
    return state
end

function dense_twin_complete_training_step!(
    workspace::DenseTwinTrainingWorkspace,
    model::SparseNeuronBank,
    bank_state::DenseTwinRowWiseAdaGradState,
    head_state::TinyDenseAdamWState,
    input,
    target::Vector{Float32},
)
    split_candidate_features!(workspace.q, workspace.x, input, 1)
    dense_twin_forward!(workspace, model)
    loss = mse_loss_cotangent!(
        workspace.output_cotangent,
        workspace.output,
        target,
    )
    dense_twin_vjp!(workspace, model)
    dense_twin_adagrad_step!(model.theta, bank_state, workspace.dtheta)
    tiny_dense_adamw_step!(
        model.head,
        model.bias,
        head_state,
        workspace.dhead,
        workspace.dbias,
    )
    return loss
end

function benchmark_index_scaling(
    rng::AbstractRNG,
    base_index::WTAIndex,
    base_theta::AbstractMatrix,
    query_key::Vector{Float32},
)
    println(
        "index_scaling\tbank_rows\tbuild_ms\troute_p50_ms\t" *
        "bucket_entries_visited\tkey_rows_scored",
    )
    # Preserve the production 608-element leading dimension and physical
    # column stride when doubling N. Only route rows are initialized/read, but
    # their addresses span the same padded per-neuron records as the base bank.
    try
        doubled_theta = Matrix{Float32}(undef, ROW_DIM, 2 * NEURON_COUNT)
        @inbounds for neuron in axes(doubled_theta, 2)
            for route_index in 1:ROUTE_DIM
                doubled_theta[route_index, neuron] = randn(rng, Float32)
            end
        end
        build_ns, doubled_index = elapsed_ns(
            () -> WTAIndex(
                doubled_theta;
                config=base_index.config,
                route_dims=ROUTE_DIM,
            ),
        )
        doubled_scratch = WTAQueryScratch(doubled_index)
        doubled_out = Int32[]
        sizehint!(doubled_out, ACTIVE_NEURONS)
        base_scratch = WTAQueryScratch(base_index)
        base_out = Int32[]
        sizehint!(base_out, ACTIVE_NEURONS)
        base_query = () -> query!(
            base_out,
            base_index,
            base_scratch,
            base_theta,
            query_key;
            max_retrieved=MAX_PROBED_KEY_ROWS,
            max_bucket_entries=MAX_BUCKET_ENTRIES,
        )
        doubled_query = () -> query!(
                doubled_out,
                doubled_index,
                doubled_scratch,
                doubled_theta,
                query_key;
                max_retrieved=MAX_PROBED_KEY_ROWS,
                max_bucket_entries=MAX_BUCKET_ENTRIES,
            )
        base_times, doubled_times = time_paired_alternating(
            base_query,
            doubled_query,
            50,
        )
        base_p50_ns = median(base_times)
        doubled_p50_ns = median(doubled_times)
        base_p50_ns > 0.0 || error("base route latency was not positive")
        base_scratch.key_rows_scored > 0 || error("base route scored no key rows")
        latency_ratio = doubled_p50_ns / base_p50_ns
        scored_rows_ratio =
            doubled_scratch.key_rows_scored / base_scratch.key_rows_scored
        base_scored_fraction =
            base_scratch.key_rows_scored / Float64(size(base_theta, 2))
        doubled_scored_fraction =
            doubled_scratch.key_rows_scored / Float64(size(doubled_theta, 2))
        scored_fraction_ratio = doubled_scored_fraction / base_scored_fraction
        base_ns_per_scored_row = base_p50_ns / base_scratch.key_rows_scored
        doubled_ns_per_scored_row =
            doubled_p50_ns / doubled_scratch.key_rows_scored
        latency_per_scored_row_ratio =
            doubled_ns_per_scored_row / base_ns_per_scored_row
        @printf(
            "index_scaling\t%d\t%s\t%.6f\t%d\t%d\n",
            size(base_theta, 2),
            "prebuilt",
            base_p50_ns / 1.0e6,
            base_scratch.bucket_entries_visited,
            base_scratch.key_rows_scored,
        )
        @printf(
            "index_scaling\t%d\t%.6f\t%.6f\t%d\t%d\n",
            size(doubled_theta, 2),
            build_ns / 1.0e6,
            doubled_p50_ns / 1.0e6,
            doubled_scratch.bucket_entries_visited,
            doubled_scratch.key_rows_scored,
        )
        @printf(
            "index_scaling_ratio\ttotal_latency=%.6f\tscored_rows=%.6f\tscored_fraction=%.6f\tlatency_per_scored_row=%.6f\ttotal_latency_limit=%.6f\tscored_fraction_limit=%.6f\tlatency_per_scored_row_limit=%.6f\n",
            latency_ratio,
            scored_rows_ratio,
            scored_fraction_ratio,
            latency_per_scored_row_ratio,
            INDEX_SCALING_MAX_TOTAL_LATENCY_RATIO,
            INDEX_SCALING_MAX_SCORED_FRACTION_RATIO,
            INDEX_SCALING_MAX_LATENCY_PER_SCORED_ROW_RATIO,
        )
        latency_ratio <= INDEX_SCALING_MAX_TOTAL_LATENCY_RATIO || error(
            "fixed-hash route latency scaled superlinearly: ratio=$latency_ratio",
        )
        scored_fraction_ratio <= INDEX_SCALING_MAX_SCORED_FRACTION_RATIO || error(
            "scored bank fraction degraded: ratio=$scored_fraction_ratio",
        )
        latency_per_scored_row_ratio <=
            INDEX_SCALING_MAX_LATENCY_PER_SCORED_ROW_RATIO || error(
            "latency per scored row degraded: ratio=$latency_per_scored_row_ratio",
        )
        println("gate\tindex_scaling\tPASS")
    catch exception
        println(
            "gate\tindex_scaling\tFAIL\tbank_rows=",
            2 * NEURON_COUNT,
            "\treason=",
            sprint(showerror, exception),
        )
        rethrow()
    end
end

function sweep_cache_lines!(buffer::Vector{UInt8}, sink::Base.RefValue{UInt64})
    accumulator = UInt64(0)
    @inbounds for index in 1:64:length(buffer)
        value = xor(buffer[index], 0x01)
        buffer[index] = value
        accumulator += UInt64(value)
    end
    sink[] = accumulator
    return accumulator
end

function time_cold_repeated(
    f,
    samples::Int,
    cache_buffer::Vector{UInt8},
    cache_sink::Base.RefValue{UInt64},
)
    f() # compile/warm outside the recorded samples
    timings = Vector{Float64}(undef, samples)
    value = nothing
    for sample in 1:samples
        sweep_cache_lines!(cache_buffer, cache_sink)
        timings[sample], value = elapsed_ns(f)
    end
    return timings, value
end

function time_paired_alternating(first, second, samples::Int)
    first()
    second()
    first_times = Vector{Float64}(undef, samples)
    second_times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        if isodd(sample)
            first_times[sample], _ = elapsed_ns(first)
            second_times[sample], _ = elapsed_ns(second)
        else
            second_times[sample], _ = elapsed_ns(second)
            first_times[sample], _ = elapsed_ns(first)
        end
    end
    return first_times, second_times
end

function time_cold_paired_alternating(
    first,
    second,
    samples::Int,
    cache_buffer::Vector{UInt8},
    cache_sink::Base.RefValue{UInt64},
)
    first()
    second()
    first_times = Vector{Float64}(undef, samples)
    second_times = Vector{Float64}(undef, samples)
    timed = function (f)
        sweep_cache_lines!(cache_buffer, cache_sink)
        elapsed, _ = elapsed_ns(f)
        return elapsed
    end
    for sample in 1:samples
        if isodd(sample)
            first_times[sample] = timed(first)
            second_times[sample] = timed(second)
        else
            second_times[sample] = timed(second)
            first_times[sample] = timed(first)
        end
    end
    return first_times, second_times
end

"""Measure a complete step and one timer nested inside that exact invocation."""
function time_nested_component(
    step,
    component_ns::Base.RefValue{Float64},
    samples::Int,
)
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    total_times = Vector{Float64}(undef, samples)
    component_times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        total_times[sample], _ = elapsed_ns(step)
        component = component_ns[]
        isfinite(component) && 0.0 <= component <= total_times[sample] || error(
            "nested component time is outside its complete-step interval",
        )
        component_times[sample] = component
    end
    return total_times, component_times
end

function corpus_step(corpus, counter::Base.RefValue{Int})
    counter[] += 1
    return mod1(counter[], length(corpus))
end

function profile_complete_step(f)
    GC.gc()
    return @timed f()
end

function time_real_rehash_samples!(
    rng::AbstractRNG,
    index::WTAIndex,
    theta::Matrix{Float32},
    selected_ids::Vector{Int32},
    samples::Int,
)
    timings = Vector{Float64}(undef, samples)
    changed_codes = Vector{Int}(undef, samples)
    previous_codes = Matrix{Int32}(undef, index.config.L, length(selected_ids))
    for sample in 1:samples
        @inbounds for (position, neuron) in pairs(selected_ids)
            for table in 1:index.config.L
                slot = (table - 1) * index.neurons + Int(neuron)
                previous_codes[table, position] = index.codes[slot]
            end
            # Fresh standard-normal route keys are generated outside the timed
            # boundary.  Each timed rehash therefore processes genuinely dirty
            # keys instead of repeatedly measuring an unchanged-code no-op.
            for route_index in 1:ROUTE_DIM
                theta[route_index, Int(neuron)] = randn(rng, Float32)
            end
        end
        timings[sample], _ = elapsed_ns(
            () -> rehash!(index, theta, selected_ids),
        )
        changed = 0
        @inbounds for (position, neuron) in pairs(selected_ids)
            for table in 1:index.config.L
                slot = (table - 1) * index.neurons + Int(neuron)
                changed +=
                    index.codes[slot] != previous_codes[table, position] ? 1 : 0
            end
        end
        changed > 0 || error("fresh route keys changed no WTA codes")
        changed_codes[sample] = changed
    end
    return timings, changed_codes
end

function main()
    BENCHMARK_DENSE_ORACLE_ONLY || error("dense oracle label was changed")
    PRODUCTION_DENSE_FALLBACK && error("production dense fallback must remain disabled")

    rng = MersenneTwister(0x534c494445)
    corpus_info = benchmark_corpus(rng)
    corpus = corpus_info.corpus
    targets = corpus_info.targets
    length(corpus) == 64 || error("benchmark corpus must contain exactly 64 candidates")
    length(targets) == length(corpus) || error("benchmark target count mismatch")
    println("setup\tinitializing literal 19.9M-parameter bank")
    model = initialize_model(rng)
    config = WTAConfig(
        m=8,
        K=4,
        L=16,
        target=64,
        min=48,
        max=80,
        training_probes=0,
        seed=0x513230,
    )
    build_ns, runtime = elapsed_ns(() -> SparseQ20Runtime(model; config=config))
    workspace = SparseQ20Workspace(runtime)
    input = corpus[1]
    split_candidate_features!(workspace.q, workspace.x, input, 1)

    # Measure whether the benchmark actually exercises input-dependent graph
    # changes rather than repeatedly hitting one hot set.
    unique_active = Set{Int32}()
    active_changes = Int[]
    probe_rows = Int[]
    bucket_visits = Int[]
    training_probe_rows = Int[]
    training_bucket_visits = Int[]
    training_route_forward_ops = Int[]
    training_unique_parameters = Int[]
    previous_ids = nothing
    for corpus_input in corpus
        result = route_forward!(runtime, workspace, corpus_input, 1)
        union!(unique_active, result.selected_ids)
        push!(probe_rows, result.accounting.probed_key_rows)
        push!(bucket_visits, result.accounting.bucket_entries_visited)
        if previous_ids !== nothing
            previous_set = Set(previous_ids)
            push!(
                active_changes,
                count(id -> !(id in previous_set), result.selected_ids),
            )
        end
        previous_ids = result.selected_ids
    end
    for (corpus_index, corpus_input) in pairs(corpus)
        result = route_forward!(
            runtime,
            workspace,
            corpus_input,
            1;
            training_probe_count=8,
            probe_token=corpus_index,
        )
        push!(training_probe_rows, result.accounting.probed_key_rows)
        push!(training_bucket_visits, result.accounting.bucket_entries_visited)
        push!(
            training_route_forward_ops,
            result.accounting.feature_plus_route_plus_forward_linear_ops,
        )
        push!(
            training_unique_parameters,
            result.accounting.routing_inclusive_unique_parameters_read,
        )
    end

    # WTA routing: only query hashing, bucket retrieval, key scoring, and rank.
    route_out = Int32[]
    sizehint!(route_out, ACTIVE_NEURONS)
    route_scratch = WTAQueryScratch(runtime.index)
    route_times, _ = time_repeated(
        () -> query!(
            route_out,
            runtime.index,
            route_scratch,
            model.theta,
            workspace.q;
            target=ACTIVE_NEURONS,
            max_retrieved=MAX_PROBED_KEY_ROWS,
            max_bucket_entries=MAX_BUCKET_ENTRIES,
        ),
        200,
    )
    selected_ids = copy(route_out)

    # E036's inference gate includes feature packing, WTA retrieval/ranking,
    # selected-row preparation, output mapping, and the neural kernel.  Keep
    # this separate from the kernel breakdown below.
    sparse_inference_counter = Ref(0)
    sparse_inference_step = () -> begin
        index = corpus_step(corpus, sparse_inference_counter)
        route_forward!(runtime, workspace, corpus[index], 1)
    end
    complete_inference_times, _ = time_repeated(
        sparse_inference_step,
        100,
    )

    # Selected preparation and neural forward are separated so lazy-state
    # bookkeeping cannot hide in the neural timing.
    prepare_times, _ = time_repeated(
        () -> prepare_selected_rows!(
            model.theta,
            runtime.bank_optimizer,
            selected_ids,
        ),
        100,
    )
    forward_times, forward_value = time_repeated(
        () -> forward_selected(model, workspace.q, workspace.x, selected_ids),
        100,
    )
    _, tape = forward_value
    dy = 0.1f0 .* randn(rng, Float32, OUTPUT_DIM)
    vjp_times, vjp = time_repeated(
        () -> vjp_selected(model, workspace.q, workspace.x, tape, dy),
        60,
    )

    # Stable duplicate-ID reduction is timed with two candidate records per
    # active row. Accumulator construction is outside this measurement.
    reduction_inputs = [make_accumulator(vjp; copies=2) for _ in 1:40]
    reduction_times, _ = time_items!(reduction_inputs, reduce_gradients!)

    # The optimizer inputs are already reduced; this timing therefore covers
    # only selected-row AdaGrad state/parameter mutation and its O(1) clock.
    optimizer_inputs = [
        make_accumulator(vjp; copies=1, reduce=true) for _ in 1:40
    ]
    prepare_selected_rows!(model.theta, runtime.bank_optimizer, selected_ids)
    optimizer_times, _ = time_items!(
        optimizer_inputs,
        accumulator -> sparse_adagradw_step!(
            model.theta,
            runtime.bank_optimizer,
            accumulator,
        ),
    )

    # Standalone rehash timing uses freshly perturbed keys on every sample.
    # The complete training boundary below separately measures the real
    # optimizer-induced rehash, including unchanged-code cases.
    dirty_ids = isempty(dirty_rows(runtime.bank_optimizer)) ?
        selected_ids : copy(dirty_rows(runtime.bank_optimizer))
    rehash!(runtime.index, model.theta, dirty_ids)
    rehash_times, rehash_changed_codes = time_real_rehash_samples!(
        rng,
        runtime.index,
        model.theta,
        dirty_ids,
        60,
    )

    # Component microbenchmarks above intentionally mutate their optimizer
    # state.  Reset only the state/counters (not parameters or index) so the
    # two complete-training twins both begin with zero AdaGrad history.
    runtime = SparseQ20Runtime(
        model,
        runtime.index,
        init_sparse_adagradw(model.theta; weight_decay=0.0f0),
    )
    workspace = SparseQ20Workspace(runtime)

    # Clone before either complete-training benchmark mutates the parameters.
    # Thus both start from the same literal 19.9M-parameter state even though
    # they subsequently execute different active supports.
    dense_model = SparseNeuronBank(
        copy(model.theta),
        copy(model.head),
        copy(model.bias),
    )
    dense_workspace = DenseTwinTrainingWorkspace()
    dense_bank_state = DenseTwinRowWiseAdaGradState()
    dense_head_state = init_tiny_dense_adamw(dense_model.head, dense_model.bias)

    head_state = init_tiny_dense_adamw(model.head, model.bias)
    sparse_training_counter = Ref(0)
    sparse_dy = Vector{Float32}(undef, OUTPUT_DIM)
    sparse_accumulator = SparseRowGradientAccumulator(capacity=ACTIVE_NEURONS)
    sparse_head_gradient = zeros(Float32, OUTPUT_DIM, LATENT_DIM)
    sparse_bias_gradient = zeros(Float32, OUTPUT_DIM)
    sparse_dlatent = zeros(Float32, LATENT_DIM)
    sparse_index_maintenance_ns = Ref(0.0)
    total_step = function ()
        case_index = corpus_step(corpus, sparse_training_counter)
        result = route_forward!(
            runtime,
            workspace,
            corpus[case_index],
            1;
            training_probe_count=8,
            probe_token=sparse_training_counter[],
        )
        loss = mse_loss_cotangent!(sparse_dy, result.raw, targets[case_index])
        reset!(sparse_accumulator)
        fill!(sparse_head_gradient, 0.0f0)
        fill!(sparse_bias_gradient, 0.0f0)
        first_value = reserve_gradient_records!(
            sparse_accumulator,
            result.tape.selected_ids,
        )
        vjp_selected_parameters!(
            model,
            workspace.q,
            workspace.x,
            result.tape,
            sparse_dy,
            sparse_accumulator.values,
            first_value,
            sparse_head_gradient,
            sparse_bias_gradient,
            sparse_dlatent,
        )
        reduce_gradients!(sparse_accumulator)
        sparse_adagradw_step!(
            model.theta,
            runtime.bank_optimizer,
            sparse_accumulator,
        )
        tiny_dense_adamw_step!(
            model.head,
            model.bias,
            head_state,
            sparse_head_gradient,
            sparse_bias_gradient,
        )
        index_maintenance_started = time_ns()
        changed_ids = take_dirty_rows!(runtime.bank_optimizer)
        rehash!(runtime.index, model.theta, changed_ids)
        sparse_index_maintenance_ns[] =
            Float64(time_ns() - index_maintenance_started)
        return loss
    end
    # The gate uses explicit optimized native BLAS, not the scalar diagnostic oracle.
    dense_inference_counter = Ref(0)
    dense_inference_step = () -> begin
        case_index = corpus_step(corpus, dense_inference_counter)
        split_candidate_features!(
            dense_workspace.q,
            dense_workspace.x,
            corpus[case_index],
            1,
        )
        dense_twin_forward!(dense_workspace, dense_model)
    end
    dense_training_counter = Ref(0)
    dense_training_step = () -> begin
        case_index = corpus_step(corpus, dense_training_counter)
        dense_twin_complete_training_step!(
            dense_workspace,
            dense_model,
            dense_bank_state,
            dense_head_state,
            corpus[case_index],
            targets[case_index],
        )
    end
    # Fair gates alternate order on every sample so turbo/thermal drift cannot
    # systematically favor the path that happened to run first.
    sparse_inference_counter[] = 0
    dense_inference_counter[] = 0
    fair_sparse_inference_times, dense_inference_times =
        time_paired_alternating(sparse_inference_step, dense_inference_step, 25)
    sparse_inference_check = sparse_inference_step()
    dense_inference_check = dense_inference_step()
    all(isfinite, sparse_inference_check.raw) || error("sparse forward was non-finite")
    all(isfinite, dense_inference_check) || error("dense forward was non-finite")

    sparse_training_counter[] = 0
    dense_training_counter[] = 0
    total_times, dense_training_times =
        time_paired_alternating(total_step, dense_training_step, 21)
    sparse_loss = total_step()
    dense_loss = dense_training_step()
    isfinite(sparse_loss) || error("sparse complete step produced non-finite loss")
    isfinite(dense_loss) || error("dense complete step produced non-finite loss")

    # A 48 MiB line sweep exceeds the 265K's 30 MiB LLC.  It happens before
    # the clock starts, so the reported cold-ish latency includes cache misses
    # in the model step but never charges either path for the sweep itself.
    cache_bytes = 48 * 1024 * 1024
    cache_buffer = fill(UInt8(0x5a), cache_bytes)
    cache_sink = Ref(UInt64(0))
    sparse_cold_times, dense_cold_times = time_cold_paired_alternating(
        total_step,
        dense_training_step,
        11,
        cache_buffer,
        cache_sink,
    )
    sparse_inference_cold_times, dense_inference_cold_times =
        time_cold_paired_alternating(
            sparse_inference_step,
            dense_inference_step,
            20,
            cache_buffer,
            cache_sink,
        )

    # Natural hot/cold timings above leave GC enabled.  These additionally
    # expose per-step Julia allocation and observed GC time; order alternates
    # so the diagnostic itself does not always favor one path.
    sparse_profiles = Any[]
    dense_profiles = Any[]
    for profile_index in 1:4
        if isodd(profile_index)
            push!(sparse_profiles, profile_complete_step(total_step))
            push!(dense_profiles, profile_complete_step(dense_training_step))
        else
            push!(dense_profiles, profile_complete_step(dense_training_step))
            push!(sparse_profiles, profile_complete_step(total_step))
        end
    end
    sparse_profile_bytes = Float64[profile.bytes for profile in sparse_profiles]
    dense_profile_bytes = Float64[profile.bytes for profile in dense_profiles]
    sparse_profile_gc = Float64[profile.gctime for profile in sparse_profiles]
    dense_profile_gc = Float64[profile.gctime for profile in dense_profiles]

    # This short series measures each rehash inside the exact complete sparse
    # step that contains it. It runs only after all paired sparse/dense timings,
    # so these extra sparse updates cannot bias either comparative speed gate.
    index_share_total_times, index_maintenance_times = time_nested_component(
        total_step,
        sparse_index_maintenance_ns,
        17,
    )
    index_maintenance_shares =
        index_maintenance_times ./ index_share_total_times
    index_maintenance_share_p50 = median(index_maintenance_shares)
    index_maintenance_share_p95 =
        quantile(index_maintenance_shares, 0.95)

    # Retain exactly one scalar full-bank timing as a diagnostic only.  It is
    # never used in either speedup ratio or GO/NO-GO gate.
    diagnostic_q = similar(workspace.q)
    diagnostic_x = similar(workspace.x)
    scalar_dense_times, scalar_dense_output = time_repeated(
        () -> begin
            split_candidate_features!(diagnostic_q, diagnostic_x, input, 1)
            dense_twin_oracle(dense_model, diagnostic_q, diagnostic_x)
        end,
        1,
        warmup=false,
    )
    all(isfinite, scalar_dense_output) || error("scalar dense oracle was non-finite")

    # Verify the optimized and scalar equations agree on one untouched input;
    # the dense model is identical for both calls at this point.
    split_candidate_features!(dense_workspace.q, dense_workspace.x, input, 1)
    optimized_check = copy(dense_twin_forward!(dense_workspace, dense_model))
    scalar_check = dense_twin_oracle(
        dense_model,
        dense_workspace.q,
        dense_workspace.x,
    )
    dense_equation_max_abs_error = maximum(abs.(optimized_check .- scalar_check))
    dense_equation_scale = max(1.0f0, maximum(abs, scalar_check))
    dense_equation_max_rel_error =
        dense_equation_max_abs_error / dense_equation_scale
    dense_equation_max_rel_error <= 5.0f-4 || error(
        "optimized/scalar dense forward mismatch: abs=$dense_equation_max_abs_error " *
        "scaled=$dense_equation_max_rel_error",
    )

    final_result = route_forward!(runtime, workspace, input, 1)
    accounting = final_result.accounting
    dense_forward_macs = NEURON_COUNT * ROW_DIM + OUTPUT_DIM * LATENT_DIM
    dense_vjp_macs = NEURON_COUNT * (ROW_DIM + LATENT_DIM) +
        2 * OUTPUT_DIM * LATENT_DIM
    feature_sketch_macs = accounting.feature_sketch_muladds
    dense_inference_linear_ops = feature_sketch_macs + dense_forward_macs
    dense_training_macs = dense_inference_linear_ops + dense_vjp_macs
    # Production omits feature-input cotangents. Per selected neuron it forms
    # one outgoing gradient, one outgoing dot for dactivation, and one
    # route/value parameter gradient. The tiny head contributes dhead+dlatent.
    sparse_vjp_macs = PARAMETER_VJP_MACS_K64
    sparse_training_macs_p50 = median(training_route_forward_ops) + sparse_vjp_macs
    sparse_training_macs_p95 =
        quantile(training_route_forward_ops, 0.95) + sparse_vjp_macs

    float_bytes = sizeof(Float32)
    theta_parameters = ROW_DIM * NEURON_COUNT
    head_parameters = TOTAL_PARAMETERS - theta_parameters
    head_weight_parameters = OUTPUT_DIM * LATENT_DIM
    active_theta_parameters = ACTIVE_NEURONS * ROW_DIM
    dense_vjp_parameter_reads =
        NEURON_COUNT * LATENT_DIM + head_weight_parameters
    sparse_vjp_parameter_reads =
        ACTIVE_NEURONS * LATENT_DIM + head_weight_parameters
    raw_input_floats =
        2 * CANDIDATE_BOARD_CELLS + NEXT_HOLD_FEATURES + AUX_FEATURES
    feature_pack_bytes =
        (raw_input_floats + ROUTE_DIM + VALUE_DIM) * float_bytes
    tiny_head_optimizer_bytes = head_parameters * 28
    dense_training_min_bytes =
        feature_pack_bytes +
        TOTAL_PARAMETERS * float_bytes +
        dense_vjp_parameter_reads * float_bytes +
        TOTAL_PARAMETERS * float_bytes +
        theta_parameters * 3 * float_bytes +
        NEURON_COUNT * 2 * float_bytes +
        tiny_head_optimizer_bytes
    sparse_training_bytes_for = function (probed_rows)
        return feature_pack_bytes +
            probed_rows * ROUTE_DIM * float_bytes +
            (active_theta_parameters + head_parameters) * float_bytes +
            sparse_vjp_parameter_reads * float_bytes +
            (active_theta_parameters + head_parameters) * float_bytes +
            4 * active_theta_parameters * float_bytes +
            active_theta_parameters * 3 * float_bytes +
            ACTIVE_NEURONS * 2 * float_bytes +
            tiny_head_optimizer_bytes +
            ACTIVE_NEURONS * ROUTE_DIM * float_bytes
    end
    sparse_training_min_bytes_p50 =
        sparse_training_bytes_for(median(training_probe_rows))
    sparse_training_min_bytes_p95 =
        sparse_training_bytes_for(quantile(training_probe_rows, 0.95))

    println("contract\tmetric\tvalue")
    println("contract\ttotal_parameters\t", accounting.total_parameters)
    println("contract\tactive_parameters\t", accounting.active_parameters)
    println("contract\tactive_fraction\t", accounting.active_parameters / accounting.total_parameters)
    println("contract\tfinal_active_rows\t", accounting.selected_rows)
    println("contract\tfinal_active_edges\t", accounting.active_edges)
    println("contract\tfinal_neural_linear_macs\t", accounting.neural_linear_macs)
    println("contract\tfinal_router_key_dot_macs\t", accounting.router_key_dot_macs)
    println("contract\tfeature_sketch_muladds\t", accounting.feature_sketch_muladds)
    println(
        "contract\tfinal_route_plus_forward_linear_macs\t",
        accounting.route_plus_forward_linear_macs,
    )
    println("contract\tprobed_key_rows\t", accounting.probed_key_rows)
    println("contract\tprobed_key_fraction\t", accounting.probed_key_fraction)
    println("contract\tbucket_entries_visited\t", accounting.bucket_entries_visited)
    println("contract\trouter_key_elements_read\t", accounting.router_key_elements_read)
    println("contract\tdense_twin_forward_linear_macs\t", dense_forward_macs)
    println("contract\tdense_complete_inference_linear_ops\t", dense_inference_linear_ops)
    println("contract\tdense_twin_vjp_linear_mac_equivalents\t", dense_vjp_macs)
    println("contract\tdense_parameter_vjp_omits_unused_input_cotangents\t", true)
    println("contract\tdense_complete_training_linear_mac_equivalents\t", dense_training_macs)
    println("contract\tsparse_selected_parameter_vjp_linear_mac_equivalents\t", sparse_vjp_macs)
    println("contract\tsparse_parameter_vjp_omits_unused_input_cotangents\t", true)
    println(
        "contract\tsparse_complete_training_linear_mac_equivalents_p50\t",
        sparse_training_macs_p50,
    )
    println(
        "contract\tsparse_complete_training_linear_mac_equivalents_p95\t",
        sparse_training_macs_p95,
    )
    println("contract\tdense_parameter_bytes\t", TOTAL_PARAMETERS * float_bytes)
    println("contract\tsparse_active_parameter_bytes\t", accounting.active_parameters * float_bytes)
    println("contract\tdense_full_gradient_bytes\t", TOTAL_PARAMETERS * float_bytes)
    println("contract\tsparse_selected_gradient_bytes\t", accounting.active_parameters * float_bytes)
    println("contract\tdense_complete_training_min_logical_float_bytes\t", dense_training_min_bytes)
    println(
        "contract\tsparse_complete_training_min_logical_float_bytes_p50\t",
        sparse_training_min_bytes_p50,
    )
    println(
        "contract\tsparse_complete_training_min_logical_float_bytes_p95\t",
        sparse_training_min_bytes_p95,
    )
    println("contract\tlogical_byte_accounting_excludes\thash_metadata_sort_allocator_cache_reuse")
    println("contract\tdense_backend\tLinearAlgebra.BLAS.gemv!_gemm!_native")
    println("contract\tblas_threads\t", BLAS.get_num_threads())
    println("contract\tblas_config\t", BLAS.get_config())
    println("contract\tjulia_threads\t", Threads.nthreads())
    println("contract\tdense_optimizer_parallel_columns\t", true)
    println("contract\tdense_optimizer_threads\t", Threads.nthreads())
    println(
        "contract\trouting_inclusive_unique_parameters_read\t",
        accounting.routing_inclusive_unique_parameters_read,
    )
    println(
        "contract\trouting_inclusive_unique_parameter_fraction\t",
        accounting.routing_inclusive_unique_parameter_fraction,
    )
    println("contract\tdecay_materialized_rows\t", accounting.decay_materialized_rows)
    println("contract\tcorpus_inputs\t", length(corpus))
    println("contract\tcorpus_source\t", corpus_info.source)
    println("contract\tcorpus_e036_eligible\t", corpus_info.e036_eligible)
    println("contract\tcorpus_part_path\t", corpus_info.part_path)
    println("contract\tcorpus_part_sha256\t", corpus_info.part_sha256)
    println("contract\tpaired_alternating_timing\t", true)
    println("contract\tnatural_gate_timings_gc_enabled\t", true)
    println("contract\tindex_maintenance_share_samples\t", length(index_maintenance_shares))
    println("contract\tindex_maintenance_share_p50\t", index_maintenance_share_p50)
    println("contract\tindex_maintenance_share_p95\t", index_maintenance_share_p95)
    println("contract\tsparse_complete_step_allocated_bytes_p50\t", median(sparse_profile_bytes))
    println("contract\tsparse_complete_step_allocated_bytes_p95\t", quantile(sparse_profile_bytes, 0.95))
    println("contract\tdense_complete_step_allocated_bytes_p50\t", median(dense_profile_bytes))
    println("contract\tdense_complete_step_allocated_bytes_p95\t", quantile(dense_profile_bytes, 0.95))
    println("contract\tsparse_complete_step_observed_gc_seconds_p50\t", median(sparse_profile_gc))
    println("contract\tsparse_complete_step_observed_gc_seconds_p95\t", quantile(sparse_profile_gc, 0.95))
    println("contract\tdense_complete_step_observed_gc_seconds_p50\t", median(dense_profile_gc))
    println("contract\tdense_complete_step_observed_gc_seconds_p95\t", quantile(dense_profile_gc, 0.95))
    println("contract\tcorpus_unique_active_rows\t", length(unique_active))
    println("contract\tcorpus_mean_changed_active_rows\t", mean(active_changes))
    println("contract\tcorpus_probe_rows_p50\t", median(probe_rows))
    println("contract\tcorpus_probe_rows_p95\t", quantile(probe_rows, 0.95))
    println("contract\tcorpus_key_rows_scored_p50\t", median(probe_rows))
    println("contract\tcorpus_key_rows_scored_p95\t", quantile(probe_rows, 0.95))
    println("contract\tcorpus_bucket_entries_p50\t", median(bucket_visits))
    println("contract\tcorpus_bucket_entries_p95\t", quantile(bucket_visits, 0.95))
    println(
        "contract\ttraining_probe_key_rows_scored_p50\t",
        median(training_probe_rows),
    )
    println(
        "contract\ttraining_probe_key_rows_scored_p95\t",
        quantile(training_probe_rows, 0.95),
    )
    println(
        "contract\ttraining_probe_bucket_entries_p50\t",
        median(training_bucket_visits),
    )
    println(
        "contract\ttraining_probe_bucket_entries_p95\t",
        quantile(training_bucket_visits, 0.95),
    )
    println(
        "contract\ttraining_routing_inclusive_unique_parameters_p50\t",
        median(training_unique_parameters),
    )
    println(
        "contract\ttraining_routing_inclusive_unique_parameters_p95\t",
        quantile(training_unique_parameters, 0.95),
    )
    println("contract\twta_micro_key_rows_scored\t", route_scratch.key_rows_scored)
    println(
        "contract\twta_micro_bucket_entries_visited\t",
        route_scratch.bucket_entries_visited,
    )
    println("contract\tproduction_max_probed_key_rows\t", MAX_PROBED_KEY_ROWS)
    println("contract\tproduction_max_bucket_entries\t", MAX_BUCKET_ENTRIES)
    println("contract\treal_rehash_changed_codes_p50\t", median(rehash_changed_codes))
    println("contract\treal_rehash_changed_codes_p95\t", quantile(rehash_changed_codes, 0.95))
    println("contract\tcold_cache_sweep_bytes\t", cache_bytes)
    println("contract\tdense_scalar_vs_blas_max_abs_error\t", dense_equation_max_abs_error)
    println("contract\tdense_scalar_vs_blas_scaled_error\t", dense_equation_max_rel_error)
    println("contract\tproduction_dense_fallback\t", PRODUCTION_DENSE_FALLBACK)
    @printf("setup\tindex_build_ms\t%.6f\n", build_ns / 1.0e6)

    report_latency("wta_route", route_times)
    report_latency("diagnostic_complete_sparse_inference_pre_microbench", complete_inference_times)
    report_latency("complete_sparse_inference_fair_clone_point", fair_sparse_inference_times)
    report_latency("selected_prepare", prepare_times)
    report_latency("sparse_forward", forward_times)
    report_latency("reference_allocating_selected_vjp_not_for_gate", vjp_times)
    report_latency("stable_sparse_reduction", reduction_times)
    report_latency("selected_adagrad_update", optimizer_times)
    report_latency("dirty_row_rehash", rehash_times)
    report_latency(
        "complete_sparse_training_index_maintenance_exact",
        index_maintenance_times,
    )
    report_latency(
        "complete_sparse_training_for_index_share_exact",
        index_share_total_times,
    )
    report_latency("complete_sparse_training_hot_varied_inputs", total_times)
    report_latency("complete_dense_training_hot_varied_inputs", dense_training_times)
    report_latency("complete_sparse_training_coldish_varied_inputs", sparse_cold_times)
    report_latency("complete_dense_training_coldish_varied_inputs", dense_cold_times)
    report_latency("dense_native_blas_complete_inference", dense_inference_times)
    report_latency("complete_sparse_inference_coldish", sparse_inference_cold_times)
    report_latency("dense_native_blas_complete_inference_coldish", dense_inference_cold_times)
    report_latency("diagnostic_scalar_dense_forward_not_for_gate", scalar_dense_times)
    @printf(
        "comparison\tdense_to_complete_sparse_inference_p50_ratio\t%.6f\n",
        median(dense_inference_times) / median(fair_sparse_inference_times),
    )
    @printf(
        "comparison\tdense_to_complete_sparse_inference_p95_ratio\t%.6f\n",
        quantile(dense_inference_times, 0.95) /
        quantile(fair_sparse_inference_times, 0.95),
    )
    @printf(
        "comparison\tdense_to_sparse_complete_training_hot_p50_ratio\t%.6f\n",
        median(dense_training_times) / median(total_times),
    )
    @printf(
        "comparison\tdense_to_sparse_complete_training_hot_p95_ratio\t%.6f\n",
        quantile(dense_training_times, 0.95) / quantile(total_times, 0.95),
    )
    @printf(
        "comparison\tdense_to_sparse_complete_training_coldish_p50_ratio\t%.6f\n",
        median(dense_cold_times) / median(sparse_cold_times),
    )
    @printf(
        "comparison\tdense_to_sparse_complete_training_coldish_p95_ratio\t%.6f\n",
        quantile(dense_cold_times, 0.95) / quantile(sparse_cold_times, 0.95),
    )

    inference_speedup =
        median(dense_inference_times) / median(fair_sparse_inference_times)
    training_speedup = median(dense_training_times) / median(total_times)
    if corpus_info.e036_eligible
        println("gate\te036_real_teacher_input\tPASS")
        println(
            "gate\te036_inference_speed_3x\t",
            inference_speedup >= 3.0 ? "PASS" : "FAIL",
        )
        println(
            "gate\te036_training_speed_2x\t",
            training_speedup >= 2.0 ? "PASS" : "FAIL",
        )
        println(
            "gate\te036_index_maintenance_under_25pct\t",
            index_maintenance_share_p50 < 0.25 ? "PASS" : "FAIL",
        )
    else
        println("gate\te036_real_teacher_input\tSMOKE_ONLY_NOT_AUTHORIZED")
        println("gate\te036_inference_speed_3x\tNOT_AUTHORIZED_SYNTHETIC")
        println("gate\te036_training_speed_2x\tNOT_AUTHORIZED_SYNTHETIC")
        println(
            "gate\te036_index_maintenance_under_25pct\t" *
            "NOT_AUTHORIZED_SYNTHETIC",
        )
    end

    try
        benchmark_index_scaling(
            rng,
            runtime.index,
            model.theta,
            workspace.q,
        )
    catch exception
        println(
            "gate\tindex_scaling_wrapper\tFAIL\treason=",
            sprint(showerror, exception),
        )
        rethrow()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
