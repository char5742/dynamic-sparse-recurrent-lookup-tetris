module SparseOneThreePairedCPU

using LinearAlgebra
using Printf
using Random
using SHA
using Statistics

const MODEL_PROVENANCE = (
    one_layer="experiments/beat_first_v1/sparse_dynamic/SparseQ20.jl",
    three_layer="experiments/beat_first_v1/sparse_dynamic_3layer/SparseDynamic3Layer.jl",
    dense_oracle="experiments/beat_first_v1/sparse_dynamic/benchmark_sparse_q20.jl",
)

# Isolate the two implementation namespaces.  In particular, ROUTE_DIM,
# TOTAL_PARAMETERS, and model types must never be resolved by an ambiguous
# `using` import.
module OneLayerSource
include(joinpath(@__DIR__, "..", "sparse_dynamic", "benchmark_sparse_q20.jl"))
end

module ThreeLayerSource
include(joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"))
end

const One = OneLayerSource.SparseQ20
const DenseOracle = OneLayerSource
const Three = ThreeLayerSource.SparseDynamic3Layer

const CORPUS_SIZE = 64
const SYNTHETIC_CORPUS_SEED = UInt64(0x7061697265646370)
const ONE_LAYER_MODEL_SEED = UInt64(0x70616972316c6d64)
const THREE_LAYER_MODEL_SEED = UInt64(0x70616972336c6d64)
const CORPUS_ORIGIN = "deterministic_synthetic_non_game_nonvalidation_unsealed"
const CORPUS_ORDER = "physical_1_to_64_cyclic_same_index_for_every_path"
const PRIMARY_WIDTH = :k64
const THREE_LAYER_WIDTHS = (
    k64=(24, 20, 20),
    k128=(48, 40, 40),
    k256=(96, 80, 80),
)

const DEFAULT_WARMUP = 6
const DEFAULT_FORWARD_SAMPLES = 24
const DEFAULT_TRAINING_SAMPLES = 24
const DEFAULT_COMPONENT_SAMPLES = 18
const MAX_WARMUP = 12
const MAX_FORWARD_SAMPLES = 48
const MAX_TRAINING_SAMPLES = 30
const MAX_COMPONENT_SAMPLES = 30
const REQUIRED_BLAS_THREADS = 1

# A cyclic schedule over all six permutations balances every member of the
# three-way group in every thermal position and every directed cross-sample
# boundary.  A path's input index is a function of sample number, never of
# invocation order.
const BALANCED_SIX_PERMUTATION_ORDERS = (
    (1, 2, 3),
    (1, 3, 2),
    (3, 2, 1),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
)
const PAIRED_RATIO_AUTHORITATIVE = "AUTHORITATIVE_NO_TIMED_GC"
const PAIRED_RATIO_NON_AUTHORITATIVE = "NON_AUTHORITATIVE_TIMED_GC_OBSERVED"
const BENCHMARK_FAIL_CLOSED_GC = "FAIL_CLOSED_TIMED_GC_OBSERVED"

struct BenchmarkConfig
    warmup::Int
    forward_samples::Int
    training_samples::Int
    component_samples::Int
    blas_threads::Int
end

function _bounded_env_int(
    name::String,
    default::Int,
    maximum::Int;
    multiple::Int=1,
)
    raw = strip(get(ENV, name, string(default)))
    value = tryparse(Int, raw)
    value === nothing && throw(ArgumentError("$name must be an integer"))
    1 <= value <= maximum || throw(ArgumentError(
        "$name must be in 1:$maximum; got $value",
    ))
    value % multiple == 0 || throw(ArgumentError(
        "$name must be a multiple of $multiple so every path occupies every " *
        "thermal position equally; got $value",
    ))
    return value
end

function benchmark_config()
    Threads.nthreads() == 1 || error(
        "paired CPU resource fairness requires exactly one Julia thread; " *
        "launch with JULIA_NUM_THREADS=1",
    )
    return BenchmarkConfig(
        _bounded_env_int(
            "PAIR_CPU_WARMUP",
            DEFAULT_WARMUP,
            MAX_WARMUP;
            multiple=6,
        ),
        _bounded_env_int(
            "PAIR_CPU_FORWARD_SAMPLES",
            DEFAULT_FORWARD_SAMPLES,
            MAX_FORWARD_SAMPLES;
            multiple=6,
        ),
        _bounded_env_int(
            "PAIR_CPU_TRAINING_SAMPLES",
            DEFAULT_TRAINING_SAMPLES,
            MAX_TRAINING_SAMPLES;
            multiple=6,
        ),
        _bounded_env_int(
            "PAIR_CPU_COMPONENT_SAMPLES",
            DEFAULT_COMPONENT_SAMPLES,
            MAX_COMPONENT_SAMPLES;
            multiple=6,
        ),
        REQUIRED_BLAS_THREADS,
    )
end

@inline function _write_u64_le(io::IO, value::Integer)
    write(io, htol(UInt64(value)))
    return nothing
end

function _write_array_provenance!(io::IO, label::Symbol, array)
    label_bytes = codeunits(String(label))
    _write_u64_le(io, length(label_bytes))
    write(io, label_bytes)
    _write_u64_le(io, ndims(array))
    for dimension in size(array)
        _write_u64_le(io, dimension)
    end
    for value in array
        write(io, htol(reinterpret(UInt32, Float32(value))))
    end
    return nothing
end

function _corpus_sha256(inputs, targets)
    io = IOBuffer()
    write(io, codeunits("paired_cpu_corpus_v1"))
    _write_u64_le(io, SYNTHETIC_CORPUS_SEED)
    _write_u64_le(io, length(inputs))
    for index in eachindex(inputs, targets)
        _write_u64_le(io, index)
        input = inputs[index]
        _write_array_provenance!(io, :candidate, input.candidate)
        _write_array_provenance!(io, :difference, input.difference)
        _write_array_provenance!(io, :next_hold, input.next_hold)
        _write_array_provenance!(io, :aux, input.aux)
        _write_array_provenance!(io, :target, targets[index])
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function build_synthetic_corpus()
    # The corpus RNG is dedicated: model initialization can never consume from
    # it or perturb the input/target sequence.
    rng = MersenneTwister(SYNTHETIC_CORPUS_SEED)
    inputs = [DenseOracle.synthetic_benchmark_input(rng) for _ in 1:CORPUS_SIZE]
    targets = [0.05f0 .* randn(rng, Float32, One.OUTPUT_DIM) for _ in inputs]
    length(inputs) == CORPUS_SIZE || error("synthetic corpus size changed")
    length(targets) == CORPUS_SIZE || error("synthetic target count changed")
    for index in eachindex(inputs, targets)
        input = inputs[index]
        One.validate_candidate_feature_input(input) == 1 || error(
            "one-layer feature contract rejected corpus item $index",
        )
        Three.BeatFirstSparseFeatures.validate_candidate_feature_input(input) == 1 ||
            error("three-layer feature contract rejected corpus item $index")
        length(targets[index]) == One.OUTPUT_DIM || error("target width changed")
        all(array -> all(isfinite, array), values(input)) || error(
            "synthetic corpus item $index contains a non-finite value",
        )
        all(isfinite, targets[index]) || error(
            "synthetic target $index contains a non-finite value",
        )
    end
    return (;
        inputs,
        targets,
        sha256=_corpus_sha256(inputs, targets),
        origin=CORPUS_ORIGIN,
        order=CORPUS_ORDER,
    )
end

function three_layer_width_accounting(active_counts::NTuple{3,Int})
    active_counts in values(THREE_LAYER_WIDTHS) || throw(ArgumentError(
        "unsupported three-layer accounting width $active_counts",
    ))
    active_bank_parameters = ntuple(
        i -> Three.LAYER_ROW_DIMS[i] * active_counts[i],
        3,
    )
    head_weight_parameters = Three.OUTPUT_DIM * Three.LATENT_DIM
    active_parameters = sum(active_bank_parameters) + Three.HEAD_PARAMETERS
    active_edges = sum(active_bank_parameters) + head_weight_parameters
    sketch_accumulates =
        2 * active_counts[1] + 2 * active_counts[2] + active_counts[3]
    parameter_vjp_macs =
        sum(active_bank_parameters) +
        head_weight_parameters + Three.OUTPUT_DIM * Three.INTERMEDIATE_SKETCH_DIM +
        (active_counts[2] + active_counts[3]) *
            (Three.ROUTE_DIM + Three.INTERMEDIATE_SKETCH_DIM)
    parameter_training_macs = active_edges + parameter_vjp_macs
    return (;
        total_parameters=Three.TOTAL_PARAMETERS,
        active_counts,
        active_parameters,
        active_edges,
        forward_macs=active_edges,
        parameter_vjp_macs,
        parameter_training_macs,
        sketch_accumulates,
        forward_inclusive_macs=active_edges + sketch_accumulates,
        parameter_training_inclusive_macs=
            parameter_training_macs + 2 * sketch_accumulates,
        active_weight_bytes=active_parameters * sizeof(Float32),
        routing_key_bytes_cap=Three.ROUTING_KEY_BYTES_CAP,
        route_plus_active_weight_bytes_cap=
            Three.ROUTING_KEY_BYTES_CAP + active_parameters * sizeof(Float32),
    )
end

function _assert_static_geometry()
    SYNTHETIC_CORPUS_SEED > UInt64(1_000_000) || error(
        "synthetic corpus seed must remain outside small game-seed namespaces",
    )
    One.PRODUCTION_DENSE_FALLBACK && error("one-layer dense fallback is enabled")
    Three.PRODUCTION_DENSE_FALLBACK && error("three-layer dense fallback is enabled")
    DenseOracle.BENCHMARK_DENSE_ORACLE_ONLY || error(
        "dense twin lost its benchmark-only provenance label",
    )
    One.ROUTE_DIM == Three.ROUTE_DIM == 64 || error("route dimension mismatch")
    One.VALUE_DIM == Three.RAW_VALUE_DIM == 496 || error("value dimension mismatch")
    One.OUTPUT_DIM == Three.OUTPUT_DIM == 22 || error("output dimension mismatch")
    One.TOTAL_PARAMETERS == Three.TOTAL_PARAMETERS == 19_924_022 || error(
        "paired models no longer have the same exact parameter count",
    )
    One.ACTIVE_NEURONS == 64 || error("one-layer primary width changed")
    THREE_LAYER_WIDTHS.k64 == (24, 20, 20) || error("three-layer k64 split changed")
    sum(THREE_LAYER_WIDTHS.k64) == 64 || error("three-layer k64 is not k64")
    sum(THREE_LAYER_WIDTHS.k128) == 128 || error("three-layer k128 is not k128")
    sum(THREE_LAYER_WIDTHS.k256) == 256 || error("three-layer k256 is not k256")

    length(BALANCED_SIX_PERMUTATION_ORDERS) == 6 || error(
        "paired thermal schedule must contain all six permutations",
    )
    all(
        order -> Set(order) == Set((1, 2, 3)),
        BALANCED_SIX_PERMUTATION_ORDERS,
    ) || error("paired thermal schedule contains a non-permutation")
    length(Set(BALANCED_SIX_PERMUTATION_ORDERS)) == 6 || error(
        "paired thermal schedule contains a duplicate permutation",
    )
    for path in 1:3
        position_counts = ntuple(
            position -> count(
                order -> order[position] == path,
                BALANCED_SIX_PERMUTATION_ORDERS,
            ),
            3,
        )
        position_counts == (2, 2, 2) || error(
            "paired thermal position balance changed for path $path: $position_counts",
        )
    end
    boundary_pairs = Set(
        (
            last(BALANCED_SIX_PERMUTATION_ORDERS[index]),
            first(BALANCED_SIX_PERMUTATION_ORDERS[
                mod1(index + 1, length(BALANCED_SIX_PERMUTATION_ORDERS))
            ]),
        ) for index in eachindex(BALANCED_SIX_PERMUTATION_ORDERS)
    )
    required_boundary_pairs = Set((
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ))
    boundary_pairs == required_boundary_pairs || error(
        "paired cyclic cross-sample boundary balance changed: $boundary_pairs",
    )

    expected = (
        k64=(31_934, 31_912, 46_376, 78_288, 108),
        k128=(58_214, 58_192, 82_896, 141_088, 216),
        k256=(110_774, 110_752, 155_936, 266_688, 432),
    )
    for (width, counts) in pairs(THREE_LAYER_WIDTHS)
        accounting = three_layer_width_accounting(counts)
        observed = (
            accounting.active_parameters,
            accounting.active_edges,
            accounting.parameter_vjp_macs,
            accounting.parameter_training_macs,
            accounting.sketch_accumulates,
        )
        observed == getproperty(expected, width) || error(
            "three-layer $width accounting changed: $observed",
        )
    end
    return true
end

const STATIC_GEOMETRY_VERIFIED = _assert_static_geometry()

function build_benchmark_states()
    one_model = One.initialize_model(MersenneTwister(ONE_LAYER_MODEL_SEED))
    One.assert_model_contract(one_model)
    One.parameter_count(one_model) == 19_924_022 || error(
        "one-layer parameter provenance mismatch",
    )

    # Clone before any runtime operation.  Equality and non-aliasing are checked
    # below, so the dense twin has the same initial 19,924,022 parameter values.
    dense_model = One.SparseNeuronBank(
        copy(one_model.theta),
        copy(one_model.head),
        copy(one_model.bias),
    )

    one_config = One.WTAConfig(
        m=8,
        K=4,
        L=16,
        target=64,
        min=48,
        max=80,
        training_probes=0,
        seed=0x513230,
    )
    one_runtime = One.SparseQ20Runtime(one_model; config=one_config)
    one_workspace = One.SparseQ20Workspace(one_runtime)
    one_state = (;
        runtime=one_runtime,
        workspace=one_workspace,
        head_optimizer=One.init_tiny_dense_adamw(one_model.head, one_model.bias),
        accumulator=One.SparseRowGradientAccumulator(capacity=One.ACTIVE_NEURONS),
        head_gradient=zeros(Float32, One.OUTPUT_DIM, One.LATENT_DIM),
        bias_gradient=zeros(Float32, One.OUTPUT_DIM),
        dlatent=zeros(Float32, One.LATENT_DIM),
        dy=zeros(Float32, One.OUTPUT_DIM),
        gather_theta=Matrix{Float32}(undef, One.ROW_DIM, One.ACTIVE_NEURONS),
        gather_head=similar(one_model.head),
        gather_bias=similar(one_model.bias),
    )

    three_model = Three.initialize_model(
        Xoshiro(THREE_LAYER_MODEL_SEED);
        active_counts=THREE_LAYER_WIDTHS.k64,
    )
    Three.parameter_count(three_model) == 19_924_022 || error(
        "three-layer parameter provenance mismatch",
    )
    ntuple(i -> size(three_model.layers[i].theta, 2), 3) ==
        Three.LAYER_NEURON_COUNTS || error("three-layer neuron geometry mismatch")
    ntuple(i -> three_model.layers[i].active_count, 3) ==
        THREE_LAYER_WIDTHS.k64 || error("three-layer primary active width mismatch")
    three_runtime = Three.initialize_runtime(
        three_model;
        learning_rate=1.0f-4,
        weight_decay=1.0f-4,
    )
    three_workspace = Three.ThreeLayerWorkspace(three_runtime)
    three_accumulators = ntuple(3) do layer_id
        layer = three_model.layers[layer_id]
        Three.EventTimeGradientAccumulator(
            size(layer.theta, 1),
            size(layer.theta, 2);
            initial_capacity=layer.active_count,
        )
    end
    three_gather = ntuple(3) do layer_id
        layer = three_model.layers[layer_id]
        Matrix{Float32}(undef, size(layer.theta, 1), layer.active_count)
    end
    three_state = (;
        runtime=three_runtime,
        workspace=three_workspace,
        accumulators=three_accumulators,
        dy=zeros(Float32, Three.OUTPUT_DIM),
        gather_theta=three_gather,
        gather_head=similar(three_model.head),
        gather_bias=similar(three_model.bias),
    )

    dense_workspace = DenseOracle.DenseTwinTrainingWorkspace()
    dense_state = (;
        model=dense_model,
        workspace=dense_workspace,
        bank_optimizer=DenseOracle.DenseTwinRowWiseAdaGradState(),
        head_optimizer=One.init_tiny_dense_adamw(dense_model.head, dense_model.bias),
    )

    dense_model.theta !== one_model.theta || error("dense theta aliases sparse theta")
    dense_model.head !== one_model.head || error("dense head aliases sparse head")
    dense_model.bias !== one_model.bias || error("dense bias aliases sparse bias")
    dense_model.theta == one_model.theta || error("dense theta clone mismatch")
    dense_model.head == one_model.head || error("dense head clone mismatch")
    dense_model.bias == one_model.bias || error("dense bias clone mismatch")
    One.parameter_count(dense_model) == One.parameter_count(one_model) || error(
        "dense twin parameter count mismatch",
    )
    return (; one=one_state, three=three_state, dense=dense_state)
end

function _one_accounting(result; training::Bool=false)
    accounting = result.accounting
    accounting.total_parameters == One.TOTAL_PARAMETERS || error(
        "one-layer runtime total changed",
    )
    accounting.active_parameters == One.ACTIVE_PARAMETERS_K64 || error(
        "one-layer active parameter count changed",
    )
    accounting.active_edges == One.ACTIVE_EDGES_K64 || error(
        "one-layer active edge count changed",
    )
    accounting.neural_linear_macs == One.FORWARD_MACS_K64 || error(
        "one-layer forward MAC count changed",
    )
    gross_bytes =
        (accounting.router_key_elements_read + accounting.active_parameters) *
        sizeof(Float32)
    unique_bytes = accounting.routing_inclusive_unique_parameters_read * sizeof(Float32)
    forward = (;
        total_parameters=accounting.total_parameters,
        active_parameters=accounting.active_parameters,
        active_edges=accounting.active_edges,
        executed_model_forward_macs=accounting.neural_linear_macs,
        executed_routing_rerank_macs=accounting.router_key_dot_macs,
        routing_inclusive_forward_linear_ops=
            accounting.feature_plus_route_plus_forward_linear_ops,
        gross_gathered_bytes=gross_bytes,
        unique_gathered_bytes=unique_bytes,
        scored_rows=accounting.probed_key_rows,
        bucket_entries=accounting.bucket_entries_visited,
    )
    training || return forward
    return merge(forward, (;
        executed_parameter_vjp_macs=One.PARAMETER_VJP_MACS_K64,
        executed_model_training_macs=One.TRAINING_LINEAR_MACS_K64,
        routing_inclusive_training_linear_ops=
            accounting.feature_plus_route_plus_forward_linear_ops +
            One.PARAMETER_VJP_MACS_K64,
    ))
end

function _three_accounting(result; training::Bool=false)
    telemetry = result.telemetry
    accounting = result.tape.accounting
    width = three_layer_width_accounting(THREE_LAYER_WIDTHS.k64)
    accounting.total_parameters == width.total_parameters || error(
        "three-layer runtime total changed",
    )
    telemetry.active_parameters == width.active_parameters || error(
        "three-layer active parameter count changed",
    )
    telemetry.active_edges == width.active_edges || error(
        "three-layer active edge count changed",
    )
    telemetry.model_forward_macs == width.forward_macs || error(
        "three-layer forward MAC count changed",
    )
    forward = (;
        total_parameters=accounting.total_parameters,
        active_parameters=telemetry.active_parameters,
        active_edges=telemetry.active_edges,
        executed_model_forward_macs=telemetry.model_forward_macs,
        executed_routing_rerank_macs=sum(telemetry.rerank_macs),
        routing_inclusive_forward_linear_ops=
            telemetry.routing_inclusive_forward_macs,
        gross_gathered_bytes=telemetry.gross_weight_gather_bytes,
        unique_gathered_bytes=telemetry.unique_weight_gather_bytes,
        scored_rows=sum(telemetry.scored_rows),
        bucket_entries=sum(telemetry.bucket_entries),
    )
    training || return forward
    return merge(forward, (;
        executed_parameter_vjp_macs=accounting.parameter_vjp_macs,
        executed_model_training_macs=accounting.parameter_training_macs,
        routing_inclusive_training_linear_ops=
            telemetry.routing_inclusive_forward_macs +
            accounting.parameter_vjp_macs + accounting.sketch_accumulates,
    ))
end

function _dense_accounting(; training::Bool=false)
    dense_forward_macs =
        One.NEURON_COUNT * One.ROW_DIM + One.OUTPUT_DIM * One.LATENT_DIM
    dense_vjp_macs =
        One.NEURON_COUNT * (One.ROW_DIM + One.LATENT_DIM) +
        2 * One.OUTPUT_DIM * One.LATENT_DIM
    forward = (;
        total_parameters=One.TOTAL_PARAMETERS,
        active_parameters=One.TOTAL_PARAMETERS,
        active_edges=dense_forward_macs,
        executed_model_forward_macs=dense_forward_macs,
        executed_routing_rerank_macs=0,
        routing_inclusive_forward_linear_ops=
            One.BOARD_ROUTE_SKETCH_MULADDS + dense_forward_macs,
        gross_gathered_bytes=One.TOTAL_PARAMETERS * sizeof(Float32),
        unique_gathered_bytes=One.TOTAL_PARAMETERS * sizeof(Float32),
        scored_rows=One.NEURON_COUNT,
        bucket_entries=0,
    )
    training || return forward
    return merge(forward, (;
        executed_parameter_vjp_macs=dense_vjp_macs,
        executed_model_training_macs=dense_forward_macs + dense_vjp_macs,
        routing_inclusive_training_linear_ops=
            One.BOARD_ROUTE_SKETCH_MULADDS + dense_forward_macs + dense_vjp_macs,
    ))
end

function one_forward_step!(state, corpus, case_index::Int)
    result = One.route_forward!(
        state.runtime,
        state.workspace,
        corpus.inputs[case_index],
        1;
        training_probe_count=0,
        probe_token=case_index,
    )
    return (;
        sink=Float64(result.raw[1]),
        phases=NamedTuple(),
        metrics=_one_accounting(result),
    )
end

function three_forward_step!(state, corpus, case_index::Int)
    result = Three.route_forward!(
        state.runtime,
        state.workspace,
        corpus.inputs[case_index],
        1;
        training_probes=(0, 0, 0),
        probe_token=case_index,
    )
    telemetry = result.telemetry
    return (;
        sink=Float64(result.output[1]),
        phases=(;
            routing_nanoseconds=Float64(sum(telemetry.routing_nanoseconds)),
            materialization_nanoseconds=
                Float64(sum(telemetry.materialization_nanoseconds)),
            fused_selected_gather_compute_nanoseconds=
                Float64(telemetry.selected_compute_nanoseconds),
        ),
        metrics=_three_accounting(result),
    )
end

function dense_forward_step!(state, corpus, case_index::Int)
    packing_started = time_ns()
    One.split_candidate_features!(
        state.workspace.q,
        state.workspace.x,
        corpus.inputs[case_index],
        1,
    )
    packing_nanoseconds = time_ns() - packing_started
    compute_started = time_ns()
    output = DenseOracle.dense_twin_forward!(state.workspace, state.model)
    compute_nanoseconds = time_ns() - compute_started
    return (;
        sink=Float64(output[1]),
        phases=(;
            feature_pack_nanoseconds=Float64(packing_nanoseconds),
            fused_fullbank_gather_compute_nanoseconds=Float64(compute_nanoseconds),
        ),
        metrics=_dense_accounting(),
    )
end

function _gather_one!(state, ids)
    length(ids) == One.ACTIVE_NEURONS || error("one-layer gather width changed")
    started = time_ns()
    for (slot, raw_id) in pairs(ids)
        neuron_id = Int(raw_id)
        copyto!(
            view(state.gather_theta, :, slot),
            view(state.runtime.model.theta, :, neuron_id),
        )
    end
    copyto!(state.gather_head, state.runtime.model.head)
    copyto!(state.gather_bias, state.runtime.model.bias)
    return Float64(time_ns() - started)
end

function _gather_three!(state, ids::NTuple{3,Vector{Int32}})
    started = time_ns()
    for layer_id in 1:3
        layer = state.runtime.model.layers[layer_id]
        length(ids[layer_id]) == layer.active_count || error(
            "three-layer gather width changed in layer $layer_id",
        )
        for (slot, raw_id) in pairs(ids[layer_id])
            copyto!(
                view(state.gather_theta[layer_id], :, slot),
                view(layer.theta, :, Int(raw_id)),
            )
        end
    end
    copyto!(state.gather_head, state.runtime.model.head)
    copyto!(state.gather_bias, state.runtime.model.bias)
    return Float64(time_ns() - started)
end

function _gather_dense!(state)
    # A non-additive copy diagnostic only.  Native BLAS reads these parameters
    # directly during the production dense oracle; this copy is never included
    # in either forward or training totals.
    started = time_ns()
    copyto!(state.workspace.dtheta, state.model.theta)
    copyto!(state.workspace.dhead, state.model.head)
    copyto!(state.workspace.dbias, state.model.bias)
    return Float64(time_ns() - started)
end

function one_component_probe!(state, corpus, case_index::Int)
    input = corpus.inputs[case_index]
    packing_started = time_ns()
    One._prepare_candidate!(state.workspace, input, 1)
    packing_nanoseconds = time_ns() - packing_started

    routing_started = time_ns()
    One._query_candidate!(
        state.runtime,
        state.workspace;
        training_probe_count=0,
        probe_token=case_index,
    )
    routing_nanoseconds = time_ns() - routing_started

    counters = state.runtime.bank_optimizer.counters
    rows_before = counters.decay_rows_materialized
    materialization_started = time_ns()
    One.prepare_selected_rows!(
        state.runtime.model.theta,
        state.runtime.bank_optimizer,
        state.workspace.selected_ids,
    )
    materialization_nanoseconds = time_ns() - materialization_started
    counters.decay_rows_materialized == rows_before || error(
        "one-layer zero-decay materialization touched a row",
    )

    compute_started = time_ns()
    output, tape = One.forward_selected(
        state.runtime.model,
        state.workspace.q,
        state.workspace.x,
        state.workspace.selected_ids,
    )
    One.map_outputs(output)
    compute_nanoseconds = time_ns() - compute_started
    tape.accounting.executed_linear_macs == One.FORWARD_MACS_K64 || error(
        "one-layer component probe changed executed MACs",
    )
    gather_nanoseconds = _gather_one!(state, tape.selected_ids)
    return (;
        sink=Float64(output[1]),
        phases=(;
            feature_pack_nanoseconds=Float64(packing_nanoseconds),
            routing_nanoseconds=Float64(routing_nanoseconds),
            materialization_nanoseconds=Float64(materialization_nanoseconds),
            selected_compute_nanoseconds=Float64(compute_nanoseconds),
            gather_copy_diagnostic_nanoseconds=gather_nanoseconds,
        ),
        metrics=(;
            gather_copy_diagnostic_bytes=
                One.ACTIVE_PARAMETERS_K64 * sizeof(Float32),
        ),
    )
end

function three_component_probe!(state, corpus, case_index::Int)
    started = time_ns()
    result = Three.route_forward!(
        state.runtime,
        state.workspace,
        corpus.inputs[case_index],
        1;
        training_probes=(0, 0, 0),
        probe_token=case_index,
    )
    elapsed = time_ns() - started
    telemetry = result.telemetry
    adapter_nanoseconds = elapsed >= telemetry.total_forward_nanoseconds ?
        elapsed - telemetry.total_forward_nanoseconds : UInt64(0)
    gather_nanoseconds = _gather_three!(state, result.tape.ids)
    return (;
        sink=Float64(result.output[1]),
        phases=(;
            feature_adapter_and_wrapper_nanoseconds=Float64(adapter_nanoseconds),
            routing_nanoseconds=Float64(sum(telemetry.routing_nanoseconds)),
            materialization_nanoseconds=
                Float64(sum(telemetry.materialization_nanoseconds)),
            fused_selected_gather_compute_nanoseconds=
                Float64(telemetry.selected_compute_nanoseconds),
            gather_copy_diagnostic_nanoseconds=gather_nanoseconds,
        ),
        metrics=(;
            gather_copy_diagnostic_bytes=three_layer_width_accounting(
                THREE_LAYER_WIDTHS.k64,
            ).active_weight_bytes,
        ),
    )
end

function dense_component_probe!(state, corpus, case_index::Int)
    forward = dense_forward_step!(state, corpus, case_index)
    gather_nanoseconds = _gather_dense!(state)
    phases = merge(
        forward.phases,
        (; gather_copy_diagnostic_nanoseconds=gather_nanoseconds),
    )
    return (;
        sink=forward.sink,
        phases,
        metrics=(;
            gather_copy_diagnostic_bytes=One.TOTAL_PARAMETERS * sizeof(Float32),
        ),
    )
end

function one_training_step!(state, corpus, case_index::Int)
    forward_started = time_ns()
    result = One.route_forward!(
        state.runtime,
        state.workspace,
        corpus.inputs[case_index],
        1;
        training_probe_count=0,
        probe_token=case_index,
    )
    forward_nanoseconds = time_ns() - forward_started

    loss_started = time_ns()
    loss = DenseOracle.mse_loss_cotangent!(
        state.dy,
        result.raw,
        corpus.targets[case_index],
    )
    loss_nanoseconds = time_ns() - loss_started

    vjp_started = time_ns()
    One.reset!(state.accumulator)
    fill!(state.head_gradient, 0.0f0)
    fill!(state.bias_gradient, 0.0f0)
    first_value = One.reserve_gradient_records!(
        state.accumulator,
        result.tape.selected_ids,
    )
    One.vjp_selected_parameters!(
        state.runtime.model,
        state.workspace.q,
        state.workspace.x,
        result.tape,
        state.dy,
        state.accumulator.values,
        first_value,
        state.head_gradient,
        state.bias_gradient,
        state.dlatent,
    )
    One.reduce_gradients!(state.accumulator)
    length(state.accumulator.reduced_ids) == One.ACTIVE_NEURONS || error(
        "one-layer active-only gradient support changed",
    )
    vjp_nanoseconds = time_ns() - vjp_started

    optimizer_started = time_ns()
    One.sparse_adagradw_step!(
        state.runtime.model.theta,
        state.runtime.bank_optimizer,
        state.accumulator,
    )
    One.tiny_dense_adamw_step!(
        state.runtime.model.head,
        state.runtime.model.bias,
        state.head_optimizer,
        state.head_gradient,
        state.bias_gradient,
    )
    mutation_finished = time_ns()
    changed_ids = One.take_dirty_rows!(state.runtime.bank_optimizer)
    One.rehash!(state.runtime.index, state.runtime.model.theta, changed_ids)
    optimizer_finished = time_ns()

    return (;
        sink=Float64(loss),
        phases=(;
            forward_total_nanoseconds=Float64(forward_nanoseconds),
            loss_cotangent_nanoseconds=Float64(loss_nanoseconds),
            parameter_vjp_and_reduction_nanoseconds=Float64(vjp_nanoseconds),
            optimizer_without_rehash_nanoseconds=
                Float64(mutation_finished - optimizer_started),
            rehash_nanoseconds=Float64(optimizer_finished - mutation_finished),
            optimizer_total_nanoseconds=
                Float64(optimizer_finished - optimizer_started),
        ),
        metrics=_one_accounting(result; training=true),
    )
end

function three_training_step!(state, corpus, case_index::Int)
    forward_started = time_ns()
    result = Three.route_forward!(
        state.runtime,
        state.workspace,
        corpus.inputs[case_index],
        1;
        training_probes=(0, 0, 0),
        probe_token=case_index,
    )
    forward_nanoseconds = time_ns() - forward_started

    loss_started = time_ns()
    loss = DenseOracle.mse_loss_cotangent!(
        state.dy,
        result.output,
        corpus.targets[case_index],
    )
    loss_nanoseconds = time_ns() - loss_started

    vjp_started = time_ns()
    parameter_vjp = Three.vjp_selected_parameters(
        state.runtime.model,
        result.tape,
        state.dy,
    )
    vjp_nanoseconds = time_ns() - vjp_started

    optimizer_started = time_ns()
    update = Three.apply_vjp_step!(
        state.runtime,
        parameter_vjp,
        state.accumulators,
    )
    optimizer_nanoseconds = time_ns() - optimizer_started
    timing = update.timing
    telemetry = result.telemetry
    return (;
        sink=Float64(loss),
        phases=(;
            forward_total_nanoseconds=Float64(forward_nanoseconds),
            routing_nanoseconds=Float64(sum(telemetry.routing_nanoseconds)),
            materialization_nanoseconds=
                Float64(sum(telemetry.materialization_nanoseconds)),
            fused_selected_gather_compute_nanoseconds=
                Float64(telemetry.selected_compute_nanoseconds),
            loss_cotangent_nanoseconds=Float64(loss_nanoseconds),
            parameter_vjp_nanoseconds=Float64(vjp_nanoseconds),
            optimizer_prepare_nanoseconds=Float64(timing.prepare_nanoseconds),
            optimizer_snapshot_nanoseconds=Float64(timing.snapshot_nanoseconds),
            optimizer_commit_nanoseconds=Float64(timing.commit_nanoseconds),
            rehash_nanoseconds=Float64(timing.rehash_nanoseconds),
            optimizer_total_nanoseconds=Float64(optimizer_nanoseconds),
        ),
        metrics=_three_accounting(result; training=true),
    )
end

function dense_training_step!(state, corpus, case_index::Int)
    packing_started = time_ns()
    One.split_candidate_features!(
        state.workspace.q,
        state.workspace.x,
        corpus.inputs[case_index],
        1,
    )
    packing_nanoseconds = time_ns() - packing_started

    forward_started = time_ns()
    output = DenseOracle.dense_twin_forward!(state.workspace, state.model)
    forward_nanoseconds = time_ns() - forward_started
    loss_started = time_ns()
    loss = DenseOracle.mse_loss_cotangent!(
        state.workspace.output_cotangent,
        output,
        corpus.targets[case_index],
    )
    loss_nanoseconds = time_ns() - loss_started

    vjp_started = time_ns()
    DenseOracle.dense_twin_vjp!(state.workspace, state.model)
    vjp_nanoseconds = time_ns() - vjp_started

    optimizer_started = time_ns()
    DenseOracle.dense_twin_adagrad_step!(
        state.model.theta,
        state.bank_optimizer,
        state.workspace.dtheta,
    )
    One.tiny_dense_adamw_step!(
        state.model.head,
        state.model.bias,
        state.head_optimizer,
        state.workspace.dhead,
        state.workspace.dbias,
    )
    optimizer_nanoseconds = time_ns() - optimizer_started
    return (;
        sink=Float64(loss),
        phases=(;
            feature_pack_nanoseconds=Float64(packing_nanoseconds),
            fused_fullbank_gather_forward_compute_nanoseconds=
                Float64(forward_nanoseconds),
            loss_cotangent_nanoseconds=Float64(loss_nanoseconds),
            parameter_vjp_nanoseconds=Float64(vjp_nanoseconds),
            optimizer_total_nanoseconds=Float64(optimizer_nanoseconds),
        ),
        metrics=_dense_accounting(training=true),
    )
end

mutable struct Series
    latency_nanoseconds::Vector{Float64}
    allocated_bytes::Vector{Float64}
    gc_nanoseconds::Vector{Float64}
    phases::Dict{Symbol,Vector{Float64}}
    metrics::Dict{Symbol,Vector{Float64}}
end

Series() = Series(
    Float64[],
    Float64[],
    Float64[],
    Dict{Symbol,Vector{Float64}}(),
    Dict{Symbol,Vector{Float64}}(),
)

function _record!(series::Series, timed)
    observation = timed.value
    isfinite(observation.sink) || error("benchmark path produced a non-finite sink")
    latency_nanoseconds = Float64(timed.time) * 1.0e9
    isfinite(latency_nanoseconds) && latency_nanoseconds > 0.0 || error(
        "benchmark path produced a non-positive latency",
    )
    gc_nanoseconds = Float64(timed.gctime) * 1.0e9
    isfinite(gc_nanoseconds) && gc_nanoseconds >= 0.0 || error(
        "benchmark path produced an invalid GC duration",
    )
    push!(series.latency_nanoseconds, latency_nanoseconds)
    push!(series.allocated_bytes, Float64(timed.bytes))
    push!(series.gc_nanoseconds, gc_nanoseconds)
    for (name, value) in pairs(observation.phases)
        numeric = Float64(value)
        isfinite(numeric) && numeric >= 0.0 || error("invalid phase sample $name")
        push!(get!(series.phases, name, Float64[]), numeric)
    end
    for (name, value) in pairs(observation.metrics)
        numeric = Float64(value)
        isfinite(numeric) && numeric >= 0.0 || error("invalid metric sample $name")
        push!(get!(series.metrics, name, Float64[]), numeric)
    end
    return series
end

@inline _case_index(offset::Int, sample::Int) = mod1(offset + sample, CORPUS_SIZE)

function _warmup_group!(steps::Tuple, warmup::Int; offset::Int=0)
    length(steps) == 3 || error("paired group must contain exactly three paths")
    for sample in 1:warmup
        case_index = _case_index(offset, sample)
        order = BALANCED_SIX_PERMUTATION_ORDERS[mod1(sample, 6)]
        for path in order
            observation = steps[path](case_index)
            isfinite(observation.sink) || error("warmup produced a non-finite sink")
        end
    end
    return nothing
end

function _benchmark_group(
    steps::Tuple,
    samples::Int;
    offset::Int=0,
)
    length(steps) == 3 || error("paired group must contain exactly three paths")
    series = (Series(), Series(), Series())
    for sample in 1:samples
        case_index = _case_index(offset, sample)
        order = BALANCED_SIX_PERMUTATION_ORDERS[mod1(sample, 6)]
        for path in order
            step = steps[path]
            timed = @timed step(case_index)
            _record!(series[path], timed)
        end
    end
    for path in 1:3
        length(series[path].latency_nanoseconds) == samples || error(
            "paired scheduler dropped a sample for path $path",
        )
    end
    return series
end

function _summary(values::Vector{Float64})
    isempty(values) && error("cannot summarize an empty sample vector")
    return (;
        p50=median(values),
        p95=quantile(values, 0.95),
        mean=mean(values),
        minimum=minimum(values),
        maximum=maximum(values),
    )
end

function _report_distribution(
    family::String,
    stage::String,
    model::String,
    width::String,
    metric::String,
    unit::String,
    values::Vector{Float64},
    scale::Float64=1.0,
)
    summary = _summary(values)
    @printf(
        "%s\t%s\t%s\t%s\t%s\t%s\tsamples=%d\tp50=%.6f\tp95=%.6f\tmean=%.6f\tmin=%.6f\tmax=%.6f\n",
        family,
        stage,
        model,
        width,
        metric,
        unit,
        length(values),
        summary.p50 / scale,
        summary.p95 / scale,
        summary.mean / scale,
        summary.minimum / scale,
        summary.maximum / scale,
    )
end

function _report_series(stage::String, model::String, width::String, series::Series)
    _report_distribution(
        "timing",
        stage,
        model,
        width,
        "total",
        "milliseconds",
        series.latency_nanoseconds,
        1.0e6,
    )
    _report_distribution(
        "profile",
        stage,
        model,
        width,
        "allocated",
        "bytes",
        series.allocated_bytes,
    )
    _report_distribution(
        "profile",
        stage,
        model,
        width,
        "observed_gc",
        "milliseconds",
        series.gc_nanoseconds,
        1.0e6,
    )
    for name in sort!(collect(keys(series.phases)); by=String)
        _report_distribution(
            "phase",
            stage,
            model,
            width,
            String(name),
            "milliseconds",
            series.phases[name],
            1.0e6,
        )
    end
    for name in sort!(collect(keys(series.metrics)); by=String)
        _report_distribution(
            "actual",
            stage,
            model,
            width,
            String(name),
            "count_or_bytes",
            series.metrics[name],
        )
    end
    return nothing
end

function _report_paired_latency_ratio(
    stage::String,
    numerator_model::String,
    denominator_model::String,
    numerator::Series,
    denominator::Series,
    authority::String,
)
    length(numerator.latency_nanoseconds) == length(denominator.latency_nanoseconds) ||
        error("paired latency sample counts differ")
    all(value -> value > 0.0, denominator.latency_nanoseconds) || error(
        "paired latency denominator is not positive",
    )
    ratios = numerator.latency_nanoseconds ./ denominator.latency_nanoseconds
    println(
        "comparison_authority\t$stage\t$numerator_model\t$denominator_model\t",
        "paired_latency_ratio\t$authority",
    )
    _report_distribution(
        "comparison",
        stage,
        numerator_model,
        denominator_model,
        "paired_latency_ratio",
        "ratio",
        ratios,
    )
    return nothing
end

function _print_static_accounting()
    one = (;
        total_parameters=One.TOTAL_PARAMETERS,
        active_parameters=One.ACTIVE_PARAMETERS_K64,
        active_edges=One.ACTIVE_EDGES_K64,
        forward_macs=One.FORWARD_MACS_K64,
        parameter_vjp_macs=One.PARAMETER_VJP_MACS_K64,
        parameter_training_macs=One.TRAINING_LINEAR_MACS_K64,
        active_weight_bytes=One.ACTIVE_PARAMETERS_K64 * sizeof(Float32),
    )
    for (metric, value) in pairs(one)
        println("accounting\tsparse_1layer\tk64\t$metric\t$value")
    end
    for (width, active_counts) in pairs(THREE_LAYER_WIDTHS)
        accounting = three_layer_width_accounting(active_counts)
        println(
            "accounting\tsparse_3layer\t$width\tactive_counts\t",
            join(active_counts, ','),
        )
        for (metric, value) in pairs(accounting)
            metric == :active_counts && continue
            println("accounting\tsparse_3layer\t$width\t$metric\t$value")
        end
        if width != PRIMARY_WIDTH
            println(
                "accounting\tsparse_3layer\t$width\ttiming_status\t",
                "accounting_only_not_instantiated",
            )
        end
    end
    dense_forward_macs =
        One.NEURON_COUNT * One.ROW_DIM + One.OUTPUT_DIM * One.LATENT_DIM
    dense_vjp_macs =
        One.NEURON_COUNT * (One.ROW_DIM + One.LATENT_DIM) +
        2 * One.OUTPUT_DIM * One.LATENT_DIM
    dense = (;
        total_parameters=One.TOTAL_PARAMETERS,
        active_parameters=One.TOTAL_PARAMETERS,
        active_edges=dense_forward_macs,
        forward_macs=dense_forward_macs,
        parameter_vjp_macs=dense_vjp_macs,
        parameter_training_macs=dense_forward_macs + dense_vjp_macs,
        active_weight_bytes=One.TOTAL_PARAMETERS * sizeof(Float32),
    )
    for (metric, value) in pairs(dense)
        println("accounting\tdense_1layer_twin\tall\t$metric\t$value")
    end
    return nothing
end

function _print_contract(config::BenchmarkConfig, corpus)
    println("contract\tschema\tpaired_sparse_cpu_benchmark_v1")
    println("contract\tscope\tbounded_synthetic_cpu_engineering_benchmark")
    println("contract\tcpu_only\ttrue")
    println("contract\tproduction_models_modified\tfalse")
    println("contract\tgame_or_dataset_inputs_used\tfalse")
    println("contract\tvalidation_or_sealed_game_seeds_used\tfalse")
    println("contract\tstrength_or_promotion_evidence\tfalse")
    println("contract\tcorpus_origin\t", corpus.origin)
    println("contract\tcorpus_seed_hex\t", string(SYNTHETIC_CORPUS_SEED; base=16))
    println("contract\tcorpus_sha256\t", corpus.sha256)
    println("contract\tcorpus_items\t", length(corpus.inputs))
    println("contract\tcorpus_order\t", corpus.order)
    println("contract\tpaired_input_index_schedule\tsame_index_per_sample_per_path")
    println("contract\tpaired_order\tbalanced_six_permutation_cyclic")
    println("contract\tpaired_order_exact_balance\tall_stage_counts_multiple_of_6")
    println("contract\tresource_envelope\tone_julia_thread_one_blas_thread")
    println("contract\tcpu_affinity\tUNPINNED")
    println("contract\tnatural_gc_enabled\ttrue")
    println(
        "contract\tpaired_ratio_gc_scope\t",
        "all_forward_and_full_training_samples_all_three_paths",
    )
    println("contract\tgather_timing_scope\tnonadditive_preallocated_copy_diagnostic")
    println("contract\tgathered_bytes_scope\trouting_plus_forward_parameter_reads")
    println(
        "contract\tmac_label_scope\t",
        "executed_*_macs_are_MACs_routing_inclusive_*_linear_ops_add_sketch_work",
    )
    println("contract\tfused_gather_compute_reported\ttrue")
    println("contract\tone_layer_training_support\tselected_k64_only")
    println("contract\tthree_layer_training_support\tselected_k64_split_24_20_20_only")
    println("contract\tdense_training_support\tfull_bank_benchmark_oracle")
    println("contract\tone_layer_bank_optimizer\trowwise_adagradw_zero_bank_decay")
    println("contract\tthree_layer_bank_optimizer\teventtime_adamw_weight_decay_1e-4")
    println("contract\tdense_bank_optimizer\tfullbank_rowwise_adagrad_benchmark_oracle")
    println("contract\twarmup\t", config.warmup)
    println("contract\tforward_samples\t", config.forward_samples)
    println("contract\ttraining_samples\t", config.training_samples)
    println("contract\tcomponent_samples\t", config.component_samples)
    println("contract\tblas_threads\t", BLAS.get_num_threads())
    println("contract\tjulia_threads\t", Threads.nthreads())
    for (role, relative_path) in pairs(MODEL_PROVENANCE)
        println("provenance\t$role\t$relative_path")
    end
    _print_static_accounting()
    return nothing
end

function run_benchmark(config::BenchmarkConfig)
    STATIC_GEOMETRY_VERIFIED || error("static geometry was not verified")
    corpus = build_synthetic_corpus()
    states = build_benchmark_states()
    _print_contract(config, corpus)

    labels = (
        ("sparse_1layer", "k64"),
        ("sparse_3layer", "k64"),
        ("dense_1layer_twin", "all"),
    )
    forward_steps = (
        index -> one_forward_step!(states.one, corpus, index),
        index -> three_forward_step!(states.three, corpus, index),
        index -> dense_forward_step!(states.dense, corpus, index),
    )
    _warmup_group!(forward_steps, config.warmup; offset=0)
    forward_series = _benchmark_group(
        forward_steps,
        config.forward_samples;
        offset=config.warmup,
    )

    component_steps = (
        index -> one_component_probe!(states.one, corpus, index),
        index -> three_component_probe!(states.three, corpus, index),
        index -> dense_component_probe!(states.dense, corpus, index),
    )
    _warmup_group!(component_steps, config.warmup; offset=0)
    component_series = _benchmark_group(
        component_steps,
        config.component_samples;
        offset=config.warmup,
    )

    training_steps = (
        index -> one_training_step!(states.one, corpus, index),
        index -> three_training_step!(states.three, corpus, index),
        index -> dense_training_step!(states.dense, corpus, index),
    )
    _warmup_group!(training_steps, config.warmup; offset=0)
    training_series = _benchmark_group(
        training_steps,
        config.training_samples;
        offset=config.warmup,
    )

    for path in 1:3
        model, width = labels[path]
        _report_series("forward", model, width, forward_series[path])
        _report_series("component_diagnostic", model, width, component_series[path])
        _report_series("full_training", model, width, training_series[path])
    end
    paired_ratios_gc_clean =
        all(series -> all(iszero, series.gc_nanoseconds), forward_series) &&
        all(series -> all(iszero, series.gc_nanoseconds), training_series)
    paired_ratio_authority = paired_ratios_gc_clean ?
        PAIRED_RATIO_AUTHORITATIVE : PAIRED_RATIO_NON_AUTHORITATIVE
    for (numerator, denominator) in ((2, 1), (3, 1), (3, 2))
        numerator_model = labels[numerator][1]
        denominator_model = labels[denominator][1]
        _report_paired_latency_ratio(
            "forward",
            numerator_model,
            denominator_model,
            forward_series[numerator],
            forward_series[denominator],
            paired_ratio_authority,
        )
        _report_paired_latency_ratio(
            "full_training",
            numerator_model,
            denominator_model,
            training_series[numerator],
            training_series[denominator],
            paired_ratio_authority,
        )
    end
    println("gate\tpaired_latency_ratios\t", paired_ratio_authority)
    benchmark_status = paired_ratios_gc_clean ? "PASS" : BENCHMARK_FAIL_CLOSED_GC
    println("gate\tbenchmark_completed\t", benchmark_status)
    println("gate\tpromotion_or_strength_claim\tNOT_AUTHORIZED_SYNTHETIC")
    return nothing
end

function main(arguments::Vector{String}=ARGS)
    isempty(arguments) || throw(ArgumentError(
        "this harness accepts no paths, datasets, checkpoints, or game arguments; " *
        "bounded sample counts are controlled only by PAIR_CPU_* variables",
    ))
    config = benchmark_config()
    config.blas_threads == REQUIRED_BLAS_THREADS || error(
        "paired CPU BLAS envelope changed",
    )
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(config.blas_threads)
    try
        BLAS.get_num_threads() == config.blas_threads || error(
            "requested CPU BLAS thread count was not applied",
        )
        return run_benchmark(config)
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
end

end # module SparseOneThreePairedCPU

if abspath(PROGRAM_FILE) == @__FILE__
    SparseOneThreePairedCPU.main()
end
