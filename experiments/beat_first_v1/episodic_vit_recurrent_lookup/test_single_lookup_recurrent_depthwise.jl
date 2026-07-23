#!/usr/bin/env julia

using LinearAlgebra
using Random
using Test

for (name, value) in (
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_FFN_DIM" => "128",
    "EVRL_INITIAL_HALT_PROBABILITY" => "0.5",
)
    ENV[name] = value
end

include(joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))

const Model = Main.EpisodicViTRecurrentLookup

BLAS.set_num_threads(1)

function spatial_objective(model, memory, cotangent)
    output, _ = Model._spatial_attention_forward(model, memory)
    return dot(output, cotangent)
end

@testset "single lookup recurrent depthwise block" begin
    rng = Xoshiro(0x4457434f4e56)
    model = Model.initialize_model(rng)

    @test Model.SparseLookup.BLOCKS == 1
    @test Model.BLOCKS == 1
    @test size(model.recurrent_depthwise) == (Model.MODEL_DIM, 3, 3)
    @test Model.topology(model).long_memory_micro_calls_per_step ==
        Model.REGISTER_COUNT

    input = Model.EpisodicCandidateInput(
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.PIECE_TYPES, Model.NEXT_HOLD_TOKENS),
        randn(rng, Float32, Model.AUX_FEATURES),
    )
    memory = Model._tokenize(model, input)
    cotangent = randn(rng, Float32, size(memory))
    _, tape = Model._spatial_attention_forward(model, memory)
    accumulator = Model.GradientAccumulator(model)
    dinput = similar(memory)
    Model._spatial_attention_vjp!(
        accumulator,
        model,
        tape,
        cotangent,
        dinput,
        accumulator.backward_scratch,
    )

    kernel_gradient = accumulator.dense[:recurrent_depthwise]
    kernel_index = argmax(abs.(kernel_gradient))
    analytic_kernel = kernel_gradient[kernel_index]
    epsilon = 5.0f-4
    original_kernel = model.recurrent_depthwise[kernel_index]
    model.recurrent_depthwise[kernel_index] = original_kernel + epsilon
    positive = spatial_objective(model, memory, cotangent)
    model.recurrent_depthwise[kernel_index] = original_kernel - epsilon
    negative = spatial_objective(model, memory, cotangent)
    model.recurrent_depthwise[kernel_index] = original_kernel
    numeric_kernel = (positive - negative) / (2.0f0 * epsilon)

    scale_gradient =
        accumulator.dense[:recurrent_depthwise_scale_logit][1]
    original_scale = model.recurrent_depthwise_scale_logit[1]
    model.recurrent_depthwise_scale_logit[1] = original_scale + epsilon
    positive = spatial_objective(model, memory, cotangent)
    model.recurrent_depthwise_scale_logit[1] = original_scale - epsilon
    negative = spatial_objective(model, memory, cotangent)
    model.recurrent_depthwise_scale_logit[1] = original_scale
    numeric_scale = (positive - negative) / (2.0f0 * epsilon)

    @test isapprox(
        analytic_kernel,
        numeric_kernel;
        rtol=3.0f-2,
        atol=3.0f-2,
    )
    @test isapprox(
        scale_gradient,
        numeric_scale;
        rtol=3.0f-2,
        atol=3.0f-2,
    )
    @test all(isfinite, dinput)
    @test any(!iszero, kernel_gradient)
end
