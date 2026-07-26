#!/usr/bin/env julia

using LinearAlgebra
using Random
using Test

for (name, value) in (
    "DSRL_BLOCKS" => "1",
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_ROUTER_TABLES" => "2",
    "EVRL_ROUTER_BITS" => "4",
    "EVRL_ROUTER_BUCKET_CAP" => "64",
    "EVRL_EPISODIC_SUPPORT" => "64",
    "EVRL_FFN_DIM" => "128",
)
    ENV[name] = get(ENV, name, value)
end

include(joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))
const Model = Main.EpisodicViTRecurrentLookup
BLAS.set_num_threads(1)

function _input(rng)
    return Model.EpisodicCandidateInput(
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.PIECE_TYPES, Model.NEXT_HOLD_TOKENS),
        randn(rng, Float32, Model.AUX_FEATURES),
    )
end

function _discrete_signature(tape)
    return (
        cross=[
            copy(step.cross.selected_ids)
            for step in tape.steps
        ],
        lookup=[
            [
                copy(step.lookup.blocks[block, register].columns)
                for block in 1:Model.BLOCKS,
                    register in 1:Model.REGISTER_COUNT
            ]
            for step in tape.steps
        ],
    )
end

function _same_signature(left, right)
    length(left.cross) == length(right.cross) || return false
    length(left.lookup) == length(right.lookup) || return false
    for index in eachindex(left.cross)
        left.cross[index] == right.cross[index] || return false
        left.lookup[index] == right.lookup[index] || return false
    end
    return true
end

function _objective(model, input, output_cotangent)
    output, tape = Model.forward_trajectory(
        model,
        input;
        forced_depth=2,
        training=false,
        temperature=0.50f0,
    )
    objective = sum(
        Float64(output_cotangent[index]) * Float64(output[index])
        for index in eachindex(output)
    )
    return objective, tape
end

function _central_difference!(
    parameter,
    index,
    model,
    input,
    output_cotangent,
    reference_signature;
    epsilon=2.0f-3,
)
    original = parameter[index]
    for trial_epsilon in (
        epsilon,
        epsilon / 4.0f0,
        epsilon / 16.0f0,
        epsilon / 64.0f0,
        epsilon / 256.0f0,
    )
        parameter[index] = original + trial_epsilon
        positive, positive_tape = _objective(model, input, output_cotangent)
        parameter[index] = original - trial_epsilon
        negative, negative_tape = _objective(model, input, output_cotangent)
        parameter[index] = original
        positive_stable = _same_signature(
            reference_signature, _discrete_signature(positive_tape),
        )
        negative_stable = _same_signature(
            reference_signature, _discrete_signature(negative_tape),
        )
        positive_stable && negative_stable || continue
        return (
            (positive - negative) / (2.0 * Float64(trial_epsilon)),
            Float64(trial_epsilon),
        )
    end
    parameter[index] = original
    return nothing
end

function _strongest_indices(gradient; limit=64)
    count = min(Int(limit), length(gradient))
    return partialsortperm(vec(abs.(gradient)), 1:count; rev=true)
end

function _audit_parameter!(
    rows,
    name,
    parameter,
    gradient,
    model,
    input,
    output_cotangent,
    reference_signature;
    epsilon=2.0f-3,
)
    result = nothing
    index = 0
    for candidate_index in _strongest_indices(gradient)
        candidate_result = _central_difference!(
            parameter,
            candidate_index,
            model,
            input,
            output_cotangent,
            reference_signature;
            epsilon,
        )
        candidate_result === nothing && continue
        index = candidate_index
        result = candidate_result
        break
    end
    result === nothing && error(
        "no high-gradient coordinate preserved hard routing for $(name)",
    )
    numeric, used_epsilon = result
    analytic = Float64(gradient[index])
    absolute_error = abs(analytic - numeric)
    relative_error = absolute_error / max(abs(analytic), abs(numeric), 1.0e-7)
    push!(rows, (;
        name=String(name),
        analytic,
        numeric,
        absolute_error,
        relative_error,
        index=string(CartesianIndices(parameter)[index]),
        epsilon=used_epsilon,
    ))
    return nothing
end

@testset "full fixed-trajectory VJP directional audit" begin
    rng = Xoshiro(0x46554c4c564a5031)
    model = Model.initialize_model(rng)
    input = _input(rng)
    output_cotangent = randn(rng, Float32, Model.OUTPUT_DIM)
    _, tape = _objective(model, input, output_cotangent)
    reference_signature = _discrete_signature(tape)

    accumulator = Model.GradientAccumulator(model)
    Model.backward_trajectory!(
        accumulator,
        model,
        tape,
        output_cotangent;
        realized_loss=1.0f0,
        baseline=1.0f0,
        compute_price=0.0f0,
        policy_weight=0.0f0,
        entropy_weight=0.0f0,
        temperature=0.50f0,
        lookup_balance_weight=0.0f0,
    )

    # token_router and register_router intentionally use a hard-support STE.
    # Their task gradients are surrogate gradients and therefore cannot equal
    # the local finite difference of a trajectory whose hard support is fixed.
    exact_dense_names = (
        :cell_projection,
        :visual_depthwise,
        :visual_channel_mix,
        :visual_pointwise,
        :visual_scale_logit,
        :next_projection,
        :aux_value,
        :register_seed,
        :spatial_q,
        :spatial_k,
        :spatial_v,
        :spatial_o,
        :spatial_relative_bias,
        :spatial_scale_logit,
        :recurrent_depthwise,
        :recurrent_depthwise_scale_logit,
        :cross_q,
        :cross_k,
        :cross_v,
        :cross_o,
        :cross_scale_logit,
        :relation_scale_logit,
        :memory_write_v,
        :memory_write_o,
        :memory_write_scale_logit,
        :self_q,
        :self_k,
        :self_v,
        :self_o,
        :self_scale_logit,
        :ffn_gate,
        :ffn_up,
        :ffn_down,
        :ffn_scale_logit,
        :lookup_register_gate,
    )
    dense_parameters = Model._dense_parameters(model)
    rows = NamedTuple[]
    for name in exact_dense_names
        println("auditing ", name)
        _audit_parameter!(
            rows,
            name,
            getproperty(dense_parameters, name),
            accumulator.dense[name],
            model,
            input,
            output_cotangent,
            reference_signature,
        )
    end

    _audit_parameter!(
        rows,
        :lookup_head,
        model.lookup.head,
        accumulator.lookup.dhead,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    _audit_parameter!(
        rows,
        :lookup_bias,
        model.lookup.bias,
        accumulator.lookup.dbias,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    _audit_parameter!(
        rows,
        :lookup_bh4,
        model.lookup.bh4_diagonals[1],
        accumulator.lookup.dbh4[1],
        model,
        input,
        output_cotangent,
        reference_signature;
        epsilon=5.0f-4,
    )
    _audit_parameter!(
        rows,
        :lookup_alpha,
        model.lookup.alpha_logits,
        accumulator.lookup.dalpha_logits,
        model,
        input,
        output_cotangent,
        reference_signature,
    )

    bank_gradients = accumulator.lookup.bank_gradients[1]
    bank_column, bank_gradient = first(bank_gradients)
    for (column, gradient) in bank_gradients
        if maximum(abs, gradient) > maximum(abs, bank_gradient)
            bank_column, bank_gradient = column, gradient
        end
    end
    bank_coordinate = first(_strongest_indices(bank_gradient; limit=1))
    bank_index = CartesianIndex(bank_coordinate, Int(bank_column))
    _audit_parameter!(
        rows,
        :lookup_bank,
        @view(model.lookup.banks[1][:, Int(bank_column)]),
        bank_gradient,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    @test rows[end].index == string(CartesianIndex(bank_coordinate))
    println("lookup_bank_column\t", bank_column)

    println("name\tanalytic\tnumeric\trelative_error\tepsilon")
    for row in rows
        println(
            row.name,
            '\t',
            row.analytic,
            '\t',
            row.numeric,
            '\t',
            row.relative_error,
            '\t',
            row.epsilon,
        )
    end

    failures = filter(rows) do row
        row.absolute_error > 2.0e-3 &&
            row.relative_error > 7.5e-2
    end
    @test isempty(failures)
end
