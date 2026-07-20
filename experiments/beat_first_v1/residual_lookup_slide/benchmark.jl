module ResidualLookupSlideR0Benchmark

using LinearAlgebra
using Printf
using Random
using SHA
using Serialization

include(joinpath(@__DIR__, "ResidualLookupSlide.jl"))
using .ResidualLookupSlide

const RLS = ResidualLookupSlide
const THREE_LAYER_TEACHER_SOURCE = normpath(joinpath(
    @__DIR__, "..", "sparse_dynamic_3layer", "teacher_training.jl",
))
if !isdefined(Main, :BeatFirstThreeLayerTeacherTraining)
    Base.include(Main, THREE_LAYER_TEACHER_SOURCE)
end

export R0Accounting,
       r0_accounting,
       run_benchmark,
       main

# This file is an accounting and measurement harness for the implementation in
# ResidualLookupSlide.jl. It deliberately does not define a second model.
const DIM = 256
const BLOCKS = 3
const TABLES = 76
const ROWS_PER_TABLE = 7^3
const OUTPUT_DIM = 22

const BANK_ROWS = BLOCKS * TABLES * ROWS_PER_TABLE
const BANK_PARAMETERS = BANK_ROWS * DIM
const HEAD_WEIGHT_PARAMETERS = OUTPUT_DIM * DIM
const HEAD_BIAS_PARAMETERS = OUTPUT_DIM
const ALPHA_PARAMETERS = BLOCKS
const HEAD_PARAMETERS = HEAD_WEIGHT_PARAMETERS + HEAD_BIAS_PARAMETERS
const TOTAL_PARAMETERS = BANK_PARAMETERS + HEAD_PARAMETERS + ALPHA_PARAMETERS

const ACTIVE_ROWS = BLOCKS * TABLES
const ACTIVE_BANK_PARAMETERS = ACTIVE_ROWS * DIM
const ACTIVE_PARAMETERS =
    ACTIVE_BANK_PARAMETERS + HEAD_PARAMETERS + ALPHA_PARAMETERS

# A trainable edge-use is one use of a trainable scalar in the forward graph.
# Each residual alpha is reused over 256 carrier coordinates; this is why edge
# uses and unique active parameters are intentionally different.
const RESIDUAL_ALPHA_EDGE_USES = BLOCKS * DIM
const TRAINABLE_EDGE_USES =
    ACTIVE_BANK_PARAMETERS + HEAD_WEIGHT_PARAMETERS +
    HEAD_BIAS_PARAMETERS + RESIDUAL_ALPHA_EDGE_USES

const LOOKUP_GATHER_BYTES = ACTIVE_BANK_PARAMETERS * sizeof(Float32)
const HEAD_WEIGHT_BYTES = HEAD_WEIGHT_PARAMETERS * sizeof(Float32)
const HEAD_BIAS_BYTES = HEAD_BIAS_PARAMETERS * sizeof(Float32)
const ALPHA_BYTES = ALPHA_PARAMETERS * sizeof(Float32)
const ACTIVE_PARAMETER_BYTES = ACTIVE_PARAMETERS * sizeof(Float32)

const FHT_BUTTERFLIES = BLOCKS * (DIM ÷ 2) * trailing_zeros(DIM)
const FHT_ADD_SUBS = 2 * FHT_BUTTERFLIES
const WTA_COMPARISONS = BLOCKS * TABLES * 3 * (7 - 1)
const BANK_ACCUMULATIONS = ACTIVE_BANK_PARAMETERS

# "MAC-equivalent" is a narrow execution counter, not a hardware FLOP claim.
# FHT additions, WTA comparisons, bank additions, and optimizer work are kept
# separate. The categories below follow the exact production forward.
const RAW_COUNTSKETCH_ACCUMULATES = 496
const RMS_SQUARE_ACCUMULATES = BLOCKS * DIM
const RMS_SIGNED_SCALE_MULTIPLIES = BLOCKS * DIM
const FHT_SIGNED_SCALE_MULTIPLIES = BLOCKS * DIM
const TABLE_MEAN_SCALE_MULTIPLIES = BLOCKS * DIM
const RESIDUAL_ALPHA_MULADDS = BLOCKS * DIM
const HEAD_FORWARD_MACS = HEAD_WEIGHT_PARAMETERS
const FORWARD_MAC_EQUIVALENTS =
    RAW_COUNTSKETCH_ACCUMULATES + RMS_SQUARE_ACCUMULATES +
    RMS_SIGNED_SCALE_MULTIPLIES + FHT_SIGNED_SCALE_MULTIPLIES +
    TABLE_MEAN_SCALE_MULTIPLIES + RESIDUAL_ALPHA_MULADDS +
    HEAD_FORWARD_MACS

const HEAD_INPUT_VJP_MACS = HEAD_WEIGHT_PARAMETERS
const HEAD_PARAMETER_VJP_MACS = HEAD_WEIGHT_PARAMETERS
const ALPHA_VJP_DOT_MACS = BLOCKS * DIM
const BACKWARD_MAC_EQUIVALENTS =
    HEAD_INPUT_VJP_MACS + HEAD_PARAMETER_VJP_MACS + ALPHA_VJP_DOT_MACS
const BANK_GRADIENT_SCALE_MULTIPLIES = ACTIVE_BANK_PARAMETERS
const RAW_COUNTSKETCH_TRANSPOSE_MULTIPLIES = 496

const CORPUS_SEED = UInt64(0x524c534c49444530)
const MODEL_SEED = UInt64(0x524c534c4d4f444c)
const EXPECTED_TEACHER_MANIFEST_SHA256 =
    "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
const FIXED_PARENT_CHECKPOINT = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\checkpoints\checkpoint_000002500.jls"
const FIXED_PARENT_SHA256 =
    "de9aac395fc9406f2a3c77de4fa2408ada62716155d1d1915a1aaedb62670b85"
const MONGOOSE_BASE_CHECKPOINT = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1\sparse_3l_20260719T213725\checkpoints\checkpoint_000002500.jls"
const MONGOOSE_BASE_SHA256 =
    "c0b8b350be2357c39c27a4cf73fb8efba9b4403ba4de8cee57a2248166eee2af"
const MONGOOSE_BASE_PAIR_CONTRACT_SHA256 =
    "6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450"
const MONGOOSE_BASE_LATEST = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1\sparse_3l_20260719T213725\latest.json"
const MONGOOSE_BASE_LATEST_SHA256 =
    "f033e0417f4d442c324fe7e8009cc163ee9d8a1f58f8678327d7c3ba66796325"
const MONGOOSE_BASE_CONTROLLER_STATUS = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1\controller_status.json"
const MONGOOSE_BASE_CONTROLLER_STATUS_SHA256 =
    "e4ab7a04f7c149983584cd77c9a02c9ed8adf460d06fc144972aa767efc95ca3"

struct R0Accounting
    total_rows::Int
    active_rows::Int
    bank_parameters::Int
    head_weight_parameters::Int
    head_bias_parameters::Int
    alpha_parameters::Int
    total_parameters::Int
    active_bank_parameters::Int
    active_parameters::Int
    trainable_edge_uses::Int
    lookup_gather_bytes::Int
    head_weight_bytes::Int
    head_bias_bytes::Int
    alpha_bytes::Int
    active_parameter_bytes::Int
    forward_mac_equivalents::Int
    fht_butterflies::Int
    fht_add_subs::Int
    wta_comparisons::Int
    bank_accumulations::Int
    backward_mac_equivalents::Int
    bank_gradient_scale_multiplies::Int
    raw_countsketch_transpose_multiplies::Int
end

function r0_accounting()
    return R0Accounting(
        BANK_ROWS,
        ACTIVE_ROWS,
        BANK_PARAMETERS,
        HEAD_WEIGHT_PARAMETERS,
        HEAD_BIAS_PARAMETERS,
        ALPHA_PARAMETERS,
        TOTAL_PARAMETERS,
        ACTIVE_BANK_PARAMETERS,
        ACTIVE_PARAMETERS,
        TRAINABLE_EDGE_USES,
        LOOKUP_GATHER_BYTES,
        HEAD_WEIGHT_BYTES,
        HEAD_BIAS_BYTES,
        ALPHA_BYTES,
        ACTIVE_PARAMETER_BYTES,
        FORWARD_MAC_EQUIVALENTS,
        FHT_BUTTERFLIES,
        FHT_ADD_SUBS,
        WTA_COMPARISONS,
        BANK_ACCUMULATIONS,
        BACKWARD_MAC_EQUIVALENTS,
        BANK_GRADIENT_SCALE_MULTIPLIES,
        RAW_COUNTSKETCH_TRANSPOSE_MULTIPLIES,
    )
end

@assert TOTAL_PARAMETERS == 20_025_881
@assert ACTIVE_PARAMETERS == 64_025
@assert TRAINABLE_EDGE_USES == 64_790
@assert FORWARD_MAC_EQUIVALENTS == 9_968
@assert BACKWARD_MAC_EQUIVALENTS == 12_032

mutable struct ComponentTiming
    sketch::UInt64
    norm::UInt64
    hash::UInt64
    gather::UInt64
    residual::UInt64
    head::UInt64
end

ComponentTiming() = ComponentTiming(0, 0, 0, 0, 0, 0)

function reset!(timing::ComponentTiming)
    timing.sketch = 0
    timing.norm = 0
    timing.hash = 0
    timing.gather = 0
    timing.residual = 0
    timing.head = 0
    return timing
end

mutable struct InstrumentedWorkspace
    carrier::Vector{Float32}
    route::Vector{Float32}
    gathered::Matrix{Float32}
    value::Vector{Float32}
    output::Vector{Float32}
    addresses::Matrix{Int16}
    columns::Matrix{Int32}
    timing::ComponentTiming
end

function InstrumentedWorkspace()
    return InstrumentedWorkspace(
        zeros(Float32, DIM),
        zeros(Float32, DIM),
        zeros(Float32, DIM, TABLES),
        zeros(Float32, DIM),
        zeros(Float32, OUTPUT_DIM),
        zeros(Int16, TABLES, BLOCKS),
        zeros(Int32, TABLES, BLOCKS),
        ComponentTiming(),
    )
end

@inline function _norm_first_sign!(route, carrier, layer)
    square_sum = 0.0f0
    @inbounds @simd for coordinate in 1:DIM
        square_sum = muladd(carrier[coordinate], carrier[coordinate], square_sum)
    end
    inverse_rms = inv(sqrt(square_sum / Float32(DIM) + 1.0f-6))
    seed = RLS.ROUTER_SEEDS[layer]
    @inbounds for coordinate in 1:DIM
        sign = RLS._hash_sign(RLS._hash_word(seed, 1, coordinate))
        route[coordinate] = sign * carrier[coordinate] * inverse_rms
    end
    return route
end

@inline function _fht_second_sign!(route, layer)
    half = 1
    while half < DIM
        stride = half << 1
        @inbounds for base in 1:stride:DIM
            @simd for offset in 0:(half - 1)
                left_index = base + offset
                right_index = left_index + half
                left = route[left_index]
                right = route[right_index]
                route[left_index] = left + right
                route[right_index] = left - right
            end
        end
        half = stride
    end
    seed = RLS.ROUTER_SEEDS[layer]
    scale = inv(sqrt(Float32(DIM)))
    @inbounds for coordinate in 1:DIM
        sign = RLS._hash_sign(RLS._hash_word(seed, 2, coordinate))
        route[coordinate] *= sign * scale
    end
    return route
end

"""Allocation-free serving kernel with diagnostic component timers.

The implementation is accepted for comparison only after a bitwise preflight
against `ResidualLookupSlide.forward`; otherwise the benchmark fails closed.
"""
function instrumented_forward!(
    model::RLS.ResidualLookupModel,
    workspace::InstrumentedWorkspace,
    input::RLS.ResidualLookupInput,
)
    timing = reset!(workspace.timing)

    started = time_ns()
    RLS.compose_carrier!(workspace.carrier, input)
    timing.sketch += time_ns() - started

    @inbounds for layer in 1:BLOCKS
        started = time_ns()
        _norm_first_sign!(workspace.route, workspace.carrier, layer)
        timing.norm += time_ns() - started

        started = time_ns()
        _fht_second_sign!(workspace.route, layer)
        for table in 1:TABLES
            address = RLS.route_address(workspace.route, layer, table)
            workspace.addresses[table, layer] = Int16(address)
            workspace.columns[table, layer] = Int32(
                RLS.flat_row_column(table, address),
            )
        end
        timing.hash += time_ns() - started

        started = time_ns()
        bank = model.banks[layer]
        for table in 1:TABLES
            column = Int(workspace.columns[table, layer])
            @simd for coordinate in 1:DIM
                workspace.gathered[coordinate, table] = bank[coordinate, column]
            end
        end
        timing.gather += time_ns() - started

        started = time_ns()
        fill!(workspace.value, 0.0f0)
        for table in 1:TABLES
            @simd for coordinate in 1:DIM
                workspace.value[coordinate] += workspace.gathered[coordinate, table]
            end
        end
        scale = inv(sqrt(Float32(TABLES)))
        @simd for coordinate in 1:DIM
            workspace.value[coordinate] *= scale
        end
        alpha = RLS.residual_alpha(model.alpha_logits[layer])
        @simd for coordinate in 1:DIM
            workspace.carrier[coordinate] = muladd(
                alpha,
                workspace.value[coordinate],
                workspace.carrier[coordinate],
            )
        end
        timing.residual += time_ns() - started
    end

    started = time_ns()
    copyto!(workspace.output, model.bias)
    @inbounds for output_id in 1:OUTPUT_DIM
        accumulator = workspace.output[output_id]
        @simd for coordinate in 1:DIM
            accumulator = muladd(
                model.head[output_id, coordinate],
                workspace.carrier[coordinate],
                accumulator,
            )
        end
        workspace.output[output_id] = accumulator
    end
    timing.head += time_ns() - started
    return workspace.output
end

@inline function _same_float32_bits(left, right)
    size(left) == size(right) || return false
    @inbounds for index in eachindex(left, right)
        reinterpret(UInt32, left[index]) == reinterpret(UInt32, right[index]) ||
            return false
    end
    return true
end

function preflight_instrumented_forward!(model, workspace, input)
    authoritative = RLS.forward(model, input)
    measured = instrumented_forward!(model, workspace, input)
    _same_float32_bits(authoritative.output, measured) || error(
        "instrumented R0 output differs bitwise from authoritative forward",
    )
    for layer in 1:BLOCKS
        Vector{Int16}(view(workspace.addresses, :, layer)) ==
            authoritative.tape.addresses[layer] || error(
                "instrumented R0 addresses differ at layer $layer",
            )
        Vector{Int32}(view(workspace.columns, :, layer)) ==
            authoritative.tape.columns[layer] || error(
                "instrumented R0 columns differ at layer $layer",
            )
    end
    return true
end

struct PairedInput{SparseInput}
    residual::RLS.ResidualLookupInput
    sparse::SparseInput
end

function _synthetic_candidate(rng)
    scale = 0.05f0
    return (
        candidate=scale .* randn(rng, Float32, 24, 10, 1, 1),
        difference=scale .* randn(rng, Float32, 24, 10, 1, 1),
        next_hold=scale .* randn(rng, Float32, 7, 6, 1),
        aux=scale .* randn(rng, Float32, 37, 1),
    )
end

function _write_array!(io, array)
    write(io, UInt64(ndims(array)))
    for dimension in size(array)
        write(io, UInt64(dimension))
    end
    write(io, reinterpret(UInt8, vec(array)))
    return io
end

function _corpus_sha256(
    sources, q_values, x_values, context_values, next_hold_values, targets,
)
    io = IOBuffer()
    write(io, codeunits("residual_lookup_slide_r0_paired_corpus_v3"))
    write(io, CORPUS_SEED)
    for index in eachindex(sources, q_values, x_values, targets)
        source = sources[index]
        _write_array!(io, source.candidate)
        _write_array!(io, source.difference)
        _write_array!(io, source.next_hold)
        _write_array!(io, source.aux)
        _write_array!(io, q_values[index])
        _write_array!(io, x_values[index])
        _write_array!(io, context_values[index])
        _write_array!(io, next_hold_values[index])
        _write_array!(io, targets[index])
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function build_paired_corpus(three, count::Int)
    count >= 3 || throw(ArgumentError("paired corpus needs at least three inputs"))
    rng = Xoshiro(CORPUS_SEED)
    sources = [_synthetic_candidate(rng) for _ in 1:count]
    q_values = [zeros(Float32, 64) for _ in 1:count]
    x_values = [zeros(Float32, 496) for _ in 1:count]
    context_values = [zeros(Float32, 22) for _ in 1:count]
    next_hold_values = [zeros(Float32, 42) for _ in 1:count]
    SparseInput = three.ThreeLayerInput
    paired = Vector{PairedInput{SparseInput}}(undef, count)
    for index in 1:count
        three.BeatFirstSparseFeatures.split_candidate_features!(
            q_values[index], x_values[index], sources[index], 1,
        )
        q = q_values[index]
        x = x_values[index]
        source = sources[index]
        @inbounds for (position, aux_index) in enumerate(
            three.BeatFirstSparseFeatures.ROUTE_AUX_INDICES,
        )
            context_values[index][position] = source.aux[aux_index, 1]
        end
        position = 1
        @inbounds for token in 1:three.BeatFirstSparseFeatures.NEXT_HOLD_TOKENS
            for piece in 1:three.BeatFirstSparseFeatures.NEXT_HOLD_PIECES
                next_hold_values[index][position] = source.next_hold[piece, token, 1]
                position += 1
            end
        end
        context = context_values[index]
        next_hold = next_hold_values[index]
        residual = RLS.ResidualLookupInput(x, context, next_hold)
        sparse = three.ThreeLayerInput((q, q, q), x, context, next_hold)
        paired[index] = PairedInput(residual, sparse)
    end
    targets = [0.05f0 .* randn(rng, Float32, OUTPUT_DIM) for _ in 1:count]
    sha = _corpus_sha256(
        sources,
        q_values,
        x_values,
        context_values,
        next_hold_values,
        targets,
    )
    return paired, targets, sha
end

function _sha256_file(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint does not exist: $path"))
    return bytes2hex(open(SHA.sha256, path))
end

function _require_sha256(value::AbstractString, label::AbstractString)
    normalized = lowercase(strip(String(value)))
    length(normalized) == 64 &&
        all(character -> isdigit(character) || 'a' <= character <= 'f', normalized) ||
        throw(ArgumentError("$label must be a lowercase SHA-256"))
    return normalized
end

function _load_training_modules!()
    isdefined(Main, :BeatFirstThreeLayerTeacherTraining) || error(
        "three-layer teacher training module did not load from " *
        THREE_LAYER_TEACHER_SOURCE,
    )
    isdefined(Main, :SparseDynamic3Layer) || error(
        "SparseDynamic3Layer did not load",
    )
    return getfield(Main, :SparseDynamic3Layer)
end

@inline function _required_property(object, name::Symbol, label::String)
    hasproperty(object, name) || error("$label omitted $name")
    return getproperty(object, name)
end

function _checkpoint_routing_mode(state)
    hasproperty(state, :routing_mode) && return state.routing_mode
    mongoose_state = hasproperty(state, :mongoose_state) ? state.mongoose_state : nothing
    dynamic_state = hasproperty(state, :dynamic_k_state) ? state.dynamic_k_state : nothing
    mongoose_state === nothing && dynamic_state === nothing || error(
        "checkpoint omitted routing_mode despite auxiliary routing state",
    )
    return :fixed_wta
end

function _checkpoint_pair_mining_mode(state, routing_mode)
    hasproperty(state, :mongoose_pair_mining_mode) &&
        return state.mongoose_pair_mining_mode
    routing_mode in (
        :fixed_wta,
        :mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2,
    ) || error("checkpoint omitted mongoose_pair_mining_mode")
    return :fixed_wta_easy_extrema_v1
end

function _checkpoint_objective(state, config, training)
    names = (
        :objective_margin_weight,
        :objective_margin_mode,
        :objective_mode,
    )
    present = ntuple(index -> hasproperty(config, names[index]), length(names))
    if all(present)
        return (
            margin_weight=Float32(config.objective_margin_weight),
            margin_mode=training.normalize_margin_mode(config.objective_margin_mode),
            objective_mode=training.normalize_objective_mode(config.objective_mode),
        )
    end
    any(present) && error("checkpoint contains a partial objective identity")
    _checkpoint_routing_mode(state) == :fixed_wta || error(
        "non-fixed checkpoint omitted its objective identity",
    )
    return (
        margin_weight=Float32(training.BeatFirstTrainingCore.MARGIN_WEIGHT),
        margin_mode=training.normalize_margin_mode(
            training.FIXED_TEACHER_TOP2_MARGIN_MODE,
        ),
        objective_mode=training.normalize_objective_mode(
            training.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        ),
    )
end

function _serialized_sha256(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _validate_teacher_checkpoint_state!(loaded, state)
    training = getfield(Main, :BeatFirstThreeLayerTeacherTraining)
    config = _required_property(state, :config, "teacher state")
    mongoose_state = hasproperty(state, :mongoose_state) ? state.mongoose_state : nothing
    dynamic_state = hasproperty(state, :dynamic_k_state) ? state.dynamic_k_state : nothing
    routing_mode = _checkpoint_routing_mode(state)
    pair_mining_mode = _checkpoint_pair_mining_mode(state, routing_mode)
    objective = _checkpoint_objective(state, config, training)
    trainer = training._trainer_from_runtime(
        loaded.runtime;
        variant=config.variant,
        training_probes=config.training_probes,
        candidate_width=config.learner_width,
        objective_margin_weight=objective.margin_weight,
        objective_margin_mode=objective.margin_mode,
        objective_mode=objective.objective_mode,
        initialization_nanoseconds=UInt64(state.initialization_nanoseconds),
        mongoose_state,
        dynamic_state,
        routing_mode,
        mongoose_pair_mining_mode=pair_mining_mode,
    )
    update = Int(state.update)
    training._assert_runtime_clocks(loaded.runtime, update)
    training._assert_overlay_checkpoint_state!(trainer, config, update)
    return true
end

function _materialize_all_pending_decay!(three, runtime)
    for layer_id in 1:3
        layer = runtime.model.layers[layer_id]
        three.materialize_rows!(
            layer.theta,
            runtime.bank_optimizers[layer_id],
            Base.OneTo(size(layer.theta, 2)),
        )
    end
    return runtime
end

Base.@kwdef struct BenchmarkConfig
    fixed_checkpoint::String = FIXED_PARENT_CHECKPOINT
    fixed_sha256::String = FIXED_PARENT_SHA256
    mongoose_checkpoint::String = MONGOOSE_BASE_CHECKPOINT
    mongoose_sha256::String = MONGOOSE_BASE_SHA256
    mongoose_pair_contract_sha256::String = MONGOOSE_BASE_PAIR_CONTRACT_SHA256
    mongoose_latest::String = MONGOOSE_BASE_LATEST
    mongoose_latest_sha256::String = MONGOOSE_BASE_LATEST_SHA256
    mongoose_controller_status::String = MONGOOSE_BASE_CONTROLLER_STATUS
    mongoose_controller_status_sha256::String =
        MONGOOSE_BASE_CONTROLLER_STATUS_SHA256
    samples::Int = 30
    warmup_cycles::Int = 6
end

function _assert_matched_base_pair!(fixed_state, mongoose_state, contract_sha256)
    fixed_config = _required_property(fixed_state, :config, "fixed state")
    mongoose_config = _required_property(mongoose_state, :config, "MONGOOSE state")
    for name in (
        :dataset_manifest_sha256,
        :seed,
        :model_seed,
        :split_seed,
        :sampler_seed,
        :variant,
        :active_counts,
        :training_probes,
        :learner_width,
        :learning_rate,
        :weight_decay,
        :beta1,
        :beta2,
        :epsilon,
    )
        _required_property(fixed_config, name, "fixed config") ==
            _required_property(mongoose_config, name, "MONGOOSE config") ||
            error("matched base-v2 field $name differs")
    end
    training = getfield(Main, :BeatFirstThreeLayerTeacherTraining)
    _checkpoint_objective(fixed_state, fixed_config, training) ==
        _checkpoint_objective(mongoose_state, mongoose_config, training) || error(
            "matched base-v2 objective identity differs",
        )
    mongoose_config.pairing_contract_sha256 == contract_sha256 || error(
        "base-v2 pairing-contract SHA differs",
    )
    _serialized_sha256(fixed_state.split_metadata) ==
        _serialized_sha256(mongoose_state.split_metadata) || error(
        "matched base-v2 split metadata differs",
    )
    _serialized_sha256(fixed_state.sampler_state) ==
        _serialized_sha256(mongoose_state.sampler_state) || error(
        "matched base-v2 sampler state differs",
    )
    return true
end

function load_comparison_states(config::BenchmarkConfig)
    three = _load_training_modules!()
    fixed_path = abspath(config.fixed_checkpoint)
    mongoose_path = abspath(config.mongoose_checkpoint)
    _sha256_file(fixed_path) == config.fixed_sha256 || error(
        "fixed checkpoint SHA mismatch",
    )
    _sha256_file(mongoose_path) == config.mongoose_sha256 || error(
        "MONGOOSE base-v2 checkpoint SHA mismatch",
    )
    _sha256_file(config.mongoose_latest) == config.mongoose_latest_sha256 ||
        error("MONGOOSE base-v2 latest receipt SHA mismatch")
    _sha256_file(config.mongoose_controller_status) ==
        config.mongoose_controller_status_sha256 || error(
            "MONGOOSE base-v2 controller receipt SHA mismatch",
        )
    fixed = three.load_checkpoint(fixed_path; full_validation=true)
    mongoose = three.load_checkpoint(mongoose_path; full_validation=false)
    fixed.training_state === nothing && error("fixed checkpoint has no training state")
    mongoose.training_state === nothing && error(
        "MONGOOSE base-v2 checkpoint has no training state",
    )
    fixed_state = fixed.training_state
    mongoose_state = mongoose.training_state
    _validate_teacher_checkpoint_state!(fixed, fixed_state)
    _validate_teacher_checkpoint_state!(mongoose, mongoose_state)
    _checkpoint_routing_mode(fixed_state) == :fixed_wta ||
        error("matched parent is not fixed-WTA")
    fixed_config = _required_property(fixed_state, :config, "fixed state")
    _required_property(
        fixed_config, :dataset_manifest_sha256, "fixed config",
    ) == EXPECTED_TEACHER_MANIFEST_SHA256 || error(
        "fixed parent teacher manifest changed",
    )
    _required_property(fixed_state, :update, "fixed state") == 2_500 ||
        error("fixed parent is not update 2500")
    _required_property(mongoose_state, :update, "MONGOOSE state") == 2_500 ||
        error("MONGOOSE base-v2 checkpoint is not update 2500")
    _required_property(mongoose_state, :routing_mode, "MONGOOSE state") ==
        three.MONGOOSE_V2_RUNTIME_ROUTING_MODE || error(
            "comparison checkpoint is not bounded MONGOOSE v2",
        )
    _checkpoint_pair_mining_mode(
        mongoose_state,
        _checkpoint_routing_mode(mongoose_state),
    ) == :fixed_wta_easy_extrema_v1 || error(
        "base-v2 checkpoint was mislabeled as the later cutoff-boundary arm",
    )
    _assert_matched_base_pair!(
        fixed_state,
        mongoose_state,
        config.mongoose_pair_contract_sha256,
    )
    overlay = _required_property(mongoose_state, :mongoose_state, "MONGOOSE state")
    overlay isa three.MongooseSimHashOverlay.MongooseV2OverlayState || error(
        "MONGOOSE base checkpoint has the wrong overlay type",
    )
    overlay.active || error("MONGOOSE base-v2 serving overlay is inactive")
    overlay.indexes === nothing && error("MONGOOSE base-v2 has no live index")

    for runtime in (fixed.runtime, mongoose.runtime)
        three.parameter_count(runtime.model) == 19_924_022 || error(
            "control parameter count changed",
        )
        ntuple(i -> runtime.model.layers[i].active_count, 3) == (48, 40, 40) ||
            error("control is not fixed-k128")
    end
    neuron_counts = ntuple(i -> size(mongoose.runtime.model.layers[i].theta, 2), 3)
    _materialize_all_pending_decay!(three, fixed.runtime)
    _materialize_all_pending_decay!(three, mongoose.runtime)
    return (
        three,
        fixed=(
            runtime=fixed.runtime,
            workspace=three.ThreeLayerWorkspace(fixed.runtime),
        ),
        mongoose=(
            runtime=mongoose.runtime,
            workspace=three.ThreeLayerWorkspace(mongoose.runtime),
            overlay,
            overlay_workspace=
                three.MongooseSimHashOverlay.V2OverlayQueryWorkspace(neuron_counts),
        ),
    )
end

mutable struct Series
    latency_ns::Vector{UInt64}
    allocation_bytes::Vector{UInt64}
    gc_ns::Vector{UInt64}
end

Series() = Series(UInt64[], UInt64[], UInt64[])

mutable struct ComponentSeries
    sketch::Vector{UInt64}
    norm::Vector{UInt64}
    hash::Vector{UInt64}
    gather::Vector{UInt64}
    residual::Vector{UInt64}
    head::Vector{UInt64}
end

ComponentSeries() = ComponentSeries(
    UInt64[], UInt64[], UInt64[], UInt64[], UInt64[], UInt64[],
)

function _record!(series::Series, timed)
    push!(series.latency_ns, UInt64(round(Int, timed.time * 1.0e9)))
    push!(series.allocation_bytes, UInt64(timed.bytes))
    push!(series.gc_ns, UInt64(round(Int, timed.gctime * 1.0e9)))
    return timed.value
end

function _record_components!(series::ComponentSeries, timing::ComponentTiming)
    for name in fieldnames(ComponentSeries)
        push!(getfield(series, name), getfield(timing, name))
    end
    return series
end

function _nearest_rank(values::AbstractVector{UInt64}, probability::Float64)
    isempty(values) && error("cannot summarize empty benchmark measurements")
    ordered = sort(collect(values))
    index = clamp(ceil(Int, probability * length(ordered)), 1, length(ordered))
    return ordered[index]
end

function _summary(values::AbstractVector{UInt64})
    return (
        samples=length(values),
        p50=_nearest_rank(values, 0.50),
        p95=_nearest_rank(values, 0.95),
        mean=sum(Float64, values) / length(values),
        minimum=minimum(values),
        maximum=maximum(values),
    )
end

function _print_summary(model, stage, unit, values)
    summary = _summary(values)
    @printf(
        "metric\t%s\t%s\t%s\tsamples=%d\tp50=%.3f\tp95=%.3f\tmean=%.3f\tmin=%d\tmax=%d\n",
        model,
        stage,
        unit,
        summary.samples,
        Float64(summary.p50),
        Float64(summary.p95),
        summary.mean,
        summary.minimum,
        summary.maximum,
    )
end

function _report_series(model, stage, series::Series)
    _print_summary(model, stage, "nanoseconds", series.latency_ns)
    _print_summary(model, stage * "_allocation", "bytes", series.allocation_bytes)
    _print_summary(model, stage * "_gc", "nanoseconds", series.gc_ns)
end

function _report_components(components::ComponentSeries)
    for name in fieldnames(ComponentSeries)
        _print_summary(
            "residual_lookup_r0",
            "forward_" * String(name),
            "nanoseconds",
            getfield(components, name),
        )
    end
end

@inline function _r0_sink!(model, workspace, input::PairedInput)
    result = RLS.forward(model, input.residual)
    return sum(Float64, result.output)
end

@inline function _fixed_sink!(comparison, input::PairedInput)
    result = comparison.three.route_forward!(
        comparison.fixed.runtime,
        comparison.fixed.workspace,
        input.sparse;
        training_probes=(0, 0, 0),
        probe_token=0,
    )
    return sum(Float64, result.output)
end

@inline function _mongoose_sink!(comparison, input::PairedInput)
    mongoose = comparison.mongoose
    result = comparison.three.route_forward!(
        mongoose.runtime,
        mongoose.workspace,
        input.sparse;
        training_probes=(0, 0, 0),
        probe_token=0,
        mongoose_state=mongoose.overlay,
        mongoose_workspace=mongoose.overlay_workspace,
        mongoose_routing_mode=comparison.three.MONGOOSE_V2_RUNTIME_ROUTING_MODE,
    )
    return sum(Float64, result.output)
end

function _run_path!(label, series, components, model, workspace, comparison, input)
    if label === :r0
        sink = _record!(series, @timed _r0_sink!(model, workspace, input))
    elseif label === :fixed_wta_k128
        sink = _record!(series, @timed _fixed_sink!(comparison, input))
    elseif label === :mongoose_v2_fixed_k_base
        sink = _record!(series, @timed _mongoose_sink!(comparison, input))
    else
        error("unknown benchmark path $label")
    end
    isfinite(Float64(sink)) || error("benchmark sink is non-finite")
    return nothing
end

function _warm_path!(label, model, workspace, comparison, input)
    if label === :r0
        _r0_sink!(model, workspace, input)
    elseif label === :fixed_wta_k128
        _fixed_sink!(comparison, input)
    elseif label === :mongoose_v2_fixed_k_base
        _mongoose_sink!(comparison, input)
    else
        error("unknown benchmark path $label")
    end
    return nothing
end

function _find_inactive_column(selected::AbstractVector{<:Integer}, width::Int)
    chosen = Set(Int.(selected))
    for column in 1:width
        column in chosen || return column
    end
    error("selected support unexpectedly covers the full bank")
end

function verify_active_only_step!(model, optimizer, input, target)
    result = RLS.forward(model, input)
    dy = result.output .- target
    vjp = RLS.vjp_selected_parameters(model, result.tape, dy)
    layer = 1
    selected = Int(vjp.columns[layer][1])
    inactive = _find_inactive_column(vjp.columns[layer], size(model.banks[layer], 2))
    state = optimizer.bank_states[layer]

    inactive_theta = copy(view(model.banks[layer], :, inactive))
    inactive_m = copy(view(state.m, :, inactive))
    inactive_v = copy(view(state.v, :, inactive))
    inactive_event = state.event_count[inactive]
    inactive_step = state.last_event_step[inactive]
    inactive_decay = reinterpret(UInt64, state.last_log_decay[inactive])
    selected_theta = copy(view(model.banks[layer], :, selected))
    selected_event = state.event_count[selected]

    RLS.optimizer_step!(model, optimizer, vjp)

    _same_float32_bits(view(model.banks[layer], :, inactive), inactive_theta) ||
        error("inactive bank parameter changed")
    _same_float32_bits(view(state.m, :, inactive), inactive_m) ||
        error("inactive first moment changed")
    _same_float32_bits(view(state.v, :, inactive), inactive_v) ||
        error("inactive second moment changed")
    state.event_count[inactive] == inactive_event || error(
        "inactive event count changed",
    )
    state.last_event_step[inactive] == inactive_step || error(
        "inactive event step changed",
    )
    reinterpret(UInt64, state.last_log_decay[inactive]) == inactive_decay ||
        error("inactive lazy-decay clock changed")
    state.event_count[selected] == selected_event + UInt64(1) || error(
        "selected event count did not advance",
    )
    !_same_float32_bits(view(model.banks[layer], :, selected), selected_theta) ||
        error("selected bank row did not change")
    return true
end

function _print_accounting()
    accounting = r0_accounting()
    for name in fieldnames(R0Accounting)
        println("accounting\t", name, "\t", getfield(accounting, name))
    end
    println("accounting\traw_countsketch_accumulates\t", RAW_COUNTSKETCH_ACCUMULATES)
    println("accounting\trms_square_accumulates\t", RMS_SQUARE_ACCUMULATES)
    println("accounting\trms_signed_scale_multiplies\t", RMS_SIGNED_SCALE_MULTIPLIES)
    println("accounting\tfht_signed_scale_multiplies\t", FHT_SIGNED_SCALE_MULTIPLIES)
    println("accounting\ttable_mean_scale_multiplies\t", TABLE_MEAN_SCALE_MULTIPLIES)
    println("accounting\tresidual_alpha_muladds\t", RESIDUAL_ALPHA_MULADDS)
    println("accounting\thead_forward_macs\t", HEAD_FORWARD_MACS)
    println("accounting\thead_input_vjp_macs\t", HEAD_INPUT_VJP_MACS)
    println("accounting\thead_parameter_vjp_macs\t", HEAD_PARAMETER_VJP_MACS)
    println("accounting\talpha_vjp_dot_macs\t", ALPHA_VJP_DOT_MACS)
end

function run_benchmark(config::BenchmarkConfig)
    Threads.nthreads() == 1 || error("benchmark requires JULIA_NUM_THREADS=1")
    BLAS.get_num_threads() == 1 || error("benchmark requires BLAS threads=1")
    config.samples >= 6 && config.samples % 3 == 0 || error(
        "samples must be a multiple of three and at least six",
    )
    config.warmup_cycles >= 3 && config.warmup_cycles % 3 == 0 || error(
        "warmup cycles must be a multiple of three and at least three",
    )

    comparison = load_comparison_states(config)
    corpus_count = max(config.samples, config.warmup_cycles)
    inputs, targets, corpus_sha256 = build_paired_corpus(
        comparison.three, corpus_count,
    )
    model = RLS.initialize_exact_model(Xoshiro(MODEL_SEED))
    RLS.parameter_count(model) == TOTAL_PARAMETERS || error(
        "runtime R0 parameter count differs from independent accounting",
    )
    RLS.active_parameter_count(model) == ACTIVE_PARAMETERS || error(
        "runtime R0 active count differs from independent accounting",
    )
    RLS.accounting(model) == RLS.EXACT_ACCOUNTING || error(
        "runtime R0 accounting is not exact",
    )
    workspace = InstrumentedWorkspace()
    preflight_instrumented_forward!(model, workspace, inputs[1].residual)
    optimizer = RLS.init_residual_lookup_optimizer(model)

    println("contract\tschema\tresidual_lookup_slide_r0_paired_benchmark_v4")
    println("contract\tstatus\tMEASUREMENT_ONLY_NO_SPEED_OR_STRENGTH_CLAIM")
    println("contract\tjulia_threads\t", Threads.nthreads())
    println("contract\tblas_threads\t", BLAS.get_num_threads())
    println("contract\tpaired_input_boundary\tprecomputed_q64_x496_raw_context22_raw_next_hold42")
    println("contract\tpaired_order\tbalanced_three-cycle_rotation")
    println("contract\tcorpus_seed_hex\t", string(CORPUS_SEED; base=16))
    println("contract\tcorpus_sha256\t", corpus_sha256)
    println("contract\tfixed_checkpoint_sha256\t", config.fixed_sha256)
    println("contract\tmongoose_checkpoint_sha256\t", config.mongoose_sha256)
    println("contract\tmongoose_pair_contract_sha256\t", config.mongoose_pair_contract_sha256)
    println("contract\tmongoose_latest_sha256\t", config.mongoose_latest_sha256)
    println("contract\tmongoose_controller_status_sha256\t", config.mongoose_controller_status_sha256)
    println("contract\tmongoose_scope\tSERVING_TIMING_ONLY_FAILED_STRENGTH_GATE")
    println("contract\tprotected_game_seed_sets_used\tfalse")
    println("contract\tr0_training_status\tFRESH_RANDOM_INITIALIZATION_TIMING_ONLY")
    println("gate\tinstrumented_vs_authoritative_bitwise\tPASS")
    _print_accounting()

    verify_active_only_step!(model, optimizer, inputs[1].residual, targets[1])
    println("gate\tactive_only_weight_moment_event_decay_witness\tPASS")

    labels = (:r0, :fixed_wta_k128, :mongoose_v2_fixed_k_base)
    for cycle in 1:config.warmup_cycles
        input = inputs[mod1(cycle, corpus_count)]
        offset = mod(cycle - 1, 3)
        for position in 1:3
            label = labels[mod1(position + offset, 3)]
            _warm_path!(label, model, workspace, comparison, input)
        end
    end
    GC.gc()

    forward_series = Dict(label => Series() for label in labels)
    for cycle in 1:config.samples
        input = inputs[cycle]
        offset = mod(cycle - 1, 3)
        for position in 1:3
            label = labels[mod1(position + offset, 3)]
            _run_path!(
                label,
                forward_series[label],
                nothing,
                model,
                workspace,
                comparison,
                input,
            )
        end
    end

    # Component probes are a separate diagnostic pass. They cannot contaminate
    # the authoritative three-way outer forward samples above.
    components = ComponentSeries()
    component_probe_series = Series()
    for index in 1:config.samples
        sink = _record!(component_probe_series, @timed sum(
            Float64,
            instrumented_forward!(model, workspace, inputs[index].residual),
        ))
        isfinite(sink) || error("component-probe sink is non-finite")
        _record_components!(components, workspace.timing)
    end

    backward_series = Series()
    optimizer_series = Series()
    full_training_series = Series()
    for index in 1:config.samples
        result = RLS.forward(model, inputs[index].residual)
        dy = result.output .- targets[index]
        vjp = _record!(
            backward_series,
            (@timed RLS.vjp_selected_parameters(model, result.tape, dy)),
        )
        telemetry = _record!(
            optimizer_series,
            (@timed RLS.optimizer_step!(model, optimizer, vjp)),
        )
        telemetry.global_step == optimizer.step || error(
            "optimizer telemetry clock differs from state",
        )
    end
    for index in 1:config.samples
        input = inputs[index].residual
        target = targets[index]
        sink = _record!(full_training_series, @timed begin
            result = RLS.forward(model, input)
            dy = result.output .- target
            vjp = RLS.vjp_selected_parameters(model, result.tape, dy)
            telemetry = RLS.optimizer_step!(model, optimizer, vjp)
            Float64(sum(abs2, dy)) + Float64(telemetry.global_step)
        end)
        isfinite(sink) || error("full training sink is non-finite")
    end

    for label in labels
        _report_series(String(label), "forward", forward_series[label])
    end
    _report_series(
        "residual_lookup_r0",
        "component_probe_forward",
        component_probe_series,
    )
    _report_components(components)
    _report_series("residual_lookup_r0", "backward", backward_series)
    _report_series("residual_lookup_r0", "sparse_optimizer", optimizer_series)
    _report_series("residual_lookup_r0", "full_training_step", full_training_series)
    total_training_seconds = sum(Float64, full_training_series.latency_ns) * 1.0e-9
    @printf(
        "metric\tresidual_lookup_r0\ttraining_throughput\tupdates_per_second\t%.9f\n",
        config.samples / total_training_seconds,
    )

    all_series = vcat(
        collect(values(forward_series)),
        Series[
            component_probe_series,
            backward_series,
            optimizer_series,
            full_training_series,
        ],
    )
    any_gc = any(series -> any(!iszero, series.gc_ns), all_series)
    println(
        "gate\tmeasurement_status\t",
        any_gc ? "RAW_RETAINED_GC_CONTAMINATED_NO_SPEED_CLAIM" :
                 "MEASURED_GC_CLEAN_NO_AUTOMATIC_SPEED_CLAIM",
    )
    return nothing
end

function _parse_arguments(arguments)
    values = Dict{String,String}()
    allowed = Set(("--samples", "--warmup-cycles"))
    index = 1
    while index <= length(arguments)
        flag = arguments[index]
        startswith(flag, "--") || throw(ArgumentError("unexpected argument $flag"))
        flag in allowed || throw(ArgumentError(
            "frozen benchmark does not permit override $flag",
        ))
        index == length(arguments) && throw(ArgumentError("missing value after $flag"))
        values[flag] = arguments[index + 1]
        index += 2
    end
    return BenchmarkConfig(
        samples=parse(Int, get(values, "--samples", "30")),
        warmup_cycles=parse(Int, get(values, "--warmup-cycles", "6")),
    )
end

function main(arguments::Vector{String}=ARGS)
    config = _parse_arguments(arguments)
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return run_benchmark(config)
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
end

end # module ResidualLookupSlideR0Benchmark

if abspath(PROGRAM_FILE) == @__FILE__
    ResidualLookupSlideR0Benchmark.main()
end
