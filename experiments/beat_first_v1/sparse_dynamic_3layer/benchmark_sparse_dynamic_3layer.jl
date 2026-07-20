using Printf
using Random
using Statistics

include("SparseDynamic3Layer.jl")
using .SparseDynamic3Layer

function _positive_env_int(name::String, default::Int)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || throw(ArgumentError("$name must be positive"))
    return value
end

const _ALLOWED_ACTIVE_COUNTS = (
    (24, 20, 20), # k64
    LAYER_ACTIVE_COUNTS, # frozen k70 baseline
    (48, 40, 40), # k128
    (96, 80, 80), # k256
)

function _active_counts_from_env()
    raw = strip(get(ENV, "SPARSE3_ACTIVE_COUNTS", "26,22,22"))
    pieces = split(raw, ',')
    length(pieces) == 3 || throw(ArgumentError(
        "SPARSE3_ACTIVE_COUNTS must contain exactly three comma-separated integers",
    ))
    values = Tuple(parse.(Int, strip.(pieces)))
    values in _ALLOWED_ACTIVE_COUNTS || throw(ArgumentError(
        "unsupported active counts $values; allowed=$(_ALLOWED_ACTIVE_COUNTS)",
    ))
    return values
end

_milliseconds(samples) = Float64.(samples) ./ 1.0e6

function _report_latency(label::String, nanoseconds::Vector{UInt64})
    milliseconds = _milliseconds(nanoseconds)
    @printf(
        "%s count=%d mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
        label,
        length(milliseconds),
        mean(milliseconds),
        quantile(milliseconds, 0.50),
        quantile(milliseconds, 0.95),
        minimum(milliseconds),
        maximum(milliseconds),
    )
end

function main()
    warmup = _positive_env_int("SPARSE3_WARMUP", 5)
    inference_iterations = _positive_env_int("SPARSE3_INFERENCE_ITERS", 100)
    training_iterations = _positive_env_int("SPARSE3_TRAINING_ITERS", 50)
    candidate_pool_size = _positive_env_int("SPARSE3_CANDIDATE_POOL", 128)
    seed = parse(Int, get(ENV, "SPARSE3_SEED", "20260719"))
    active_counts = _active_counts_from_env()
    rng = Xoshiro(seed)

    # This executable measures one independent candidate. It does not accept
    # an NNUE accumulator, parent position, sibling cache, or coarse expert.
    model = active_counts == LAYER_ACTIVE_COUNTS ?
        initialize_exact_model(rng) :
        initialize_model(rng; active_counts)
    parameter_count(model) == TOTAL_PARAMETERS || error(
        "active-width sweep changed the total parameter count",
    )
    runtime = initialize_runtime(model; learning_rate=1.0f-4, weight_decay=1.0f-4)
    workspace = ThreeLayerWorkspace(runtime)
    inputs = [
        ThreeLayerInput(
            randn(rng, Float32, ROUTE_DIM),
            randn(rng, Float32, RAW_VALUE_DIM),
        ) for _ in 1:candidate_pool_size
    ]
    dy = randn(rng, Float32, OUTPUT_DIM)
    accumulators = ntuple(3) do layer_id
        layer = runtime.model.layers[layer_id]
        EventTimeGradientAccumulator(
            size(layer.theta, 1),
            size(layer.theta, 2);
            initial_capacity=layer.active_count,
        )
    end

    for iteration in 1:warmup
        input = inputs[mod1(iteration, candidate_pool_size)]
        route_forward!(runtime, workspace, input)
    end

    forward_ns = Vector{UInt64}(undef, inference_iterations)
    routing_ns = Vector{UInt64}(undef, inference_iterations)
    selected_compute_ns = Vector{UInt64}(undef, inference_iterations)
    last_result = nothing
    for iteration in 1:inference_iterations
        input = inputs[mod1(warmup + iteration, candidate_pool_size)]
        result = route_forward!(runtime, workspace, input)
        telemetry = result.telemetry
        forward_ns[iteration] = telemetry.total_forward_nanoseconds
        routing_ns[iteration] = sum(telemetry.routing_nanoseconds)
        selected_compute_ns[iteration] = telemetry.selected_compute_nanoseconds
        last_result = result
    end

    vjp_ns = Vector{UInt64}(undef, training_iterations)
    optimizer_ns = Vector{UInt64}(undef, training_iterations)
    train_total_ns = Vector{UInt64}(undef, training_iterations)
    for iteration in 1:training_iterations
        input = inputs[mod1(warmup + inference_iterations + iteration, candidate_pool_size)]
        started = time_ns()
        result = route_forward!(runtime, workspace, input)
        vjp_started = time_ns()
        parameter_vjp = vjp_selected_parameters(runtime.model, result.tape, dy)
        vjp_finished = time_ns()
        apply_vjp_step!(runtime, parameter_vjp, accumulators)
        finished = time_ns()
        vjp_ns[iteration] = vjp_finished - vjp_started
        optimizer_ns[iteration] = finished - vjp_finished
        train_total_ns[iteration] = finished - started
        last_result = result
    end

    telemetry = last_result.telemetry
    println("scope=synthetic_single_candidate_kernel_microbenchmark")
    println("strength_or_real_input_evidence=false")
    println("synthetic_candidate_pool=$candidate_pool_size")
    println("model=Tetris-SLIDE-DeepNeuronBank-Q20-B3")
    println("active_counts=$(join(active_counts, ','))")
    println("active_neurons=$(sum(active_counts))")
    println("total_parameters=$(parameter_count(runtime.model))")
    println("active_parameters=$(telemetry.active_parameters)")
    println("active_edges=$(telemetry.active_edges)")
    println("model_forward_macs=$(telemetry.model_forward_macs)")
    println("sketch_forward_macs=$(telemetry.sketch_forward_macs)")
    println("routing_rerank_macs=$(sum(telemetry.rerank_macs))")
    println("routing_inclusive_forward_macs=$(telemetry.routing_inclusive_forward_macs)")
    println("scored_rows=$(telemetry.scored_rows)")
    println("bucket_entries=$(telemetry.bucket_entries)")
    println("gross_weight_gather_bytes=$(telemetry.gross_weight_gather_bytes)")
    println("unique_weight_gather_bytes=$(telemetry.unique_weight_gather_bytes)")
    println("parameter_vjp_macs=$(last_result.tape.accounting.parameter_vjp_macs)")
    println("parameter_training_macs=$(last_result.tape.accounting.parameter_training_macs)")
    dynamic_accounting = last_result.tape.accounting
    println(
        "parameter_vjp_inclusive_macs=$(dynamic_accounting.parameter_vjp_macs + dynamic_accounting.sketch_accumulates)",
    )
    println(
        "parameter_training_inclusive_macs=$(dynamic_accounting.parameter_training_macs + 2 * dynamic_accounting.sketch_accumulates)",
    )
    _report_latency("forward", forward_ns)
    _report_latency("routing", routing_ns)
    _report_latency("selected_compute", selected_compute_ns)
    _report_latency("parameter_vjp", vjp_ns)
    _report_latency("optimizer_rehash", optimizer_ns)
    _report_latency("training_step", train_total_ns)
    @printf(
        "training_candidates_per_second=%.6f\n",
        1.0e9 / mean(Float64.(train_total_ns)),
    )
    return nothing
end

main()
