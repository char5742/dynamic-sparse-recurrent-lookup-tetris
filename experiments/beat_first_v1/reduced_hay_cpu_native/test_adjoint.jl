using LinearAlgebra
using Test

module AdjointTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = AdjointTestHarness.ReducedHayCPU
const Model = Root.ReducedHayCPUNativeModel
const Oracle = Root.ExactOracle
const Optimizer = Root.CanonicalOptimizer

function objective(model, prepared, rails, cotangent)
    result = Model.forward_reference(model, prepared, rails)
    return dot(result.raw_output, cotangent)
end

@testset "exact oracle isolation and finite difference" begin
    model, staging, prepared = Root.build_model(0x303)
    rails = zeros(Float32, Model.INPUT_RAILS)
    rails[1:17:end] .= 1.0f0
    cotangent = collect(range(-0.3f0, 0.4f0; length=Model.OUTPUT_DIM))
    gradient = Optimizer.ParameterGradient(staging)
    Oracle.conditional_exact_vjp!(
        gradient,
        Oracle.ConditionalAdjointScratch(),
        model,
        prepared,
        rails,
        cotangent;
        expected_generation=Root.prepared_generation(prepared),
    )
    for field in fieldnames(Optimizer.ParameterGradient)
        @test all(isfinite, getfield(gradient, field))
    end
    @test norm(gradient.output_bias) > 0.0f0

    epsilon = 1.0f-3
    original = staging.output_bias[7]
    staging.output_bias[7] = original + epsilon
    Root.publish!(prepared, staging)
    plus = objective(model, prepared, rails, cotangent)
    staging.output_bias[7] = original - epsilon
    Root.publish!(prepared, staging)
    minus = objective(model, prepared, rails, cotangent)
    staging.output_bias[7] = original
    Root.publish!(prepared, staging)
    finite = (plus - minus) / (2.0f0 * epsilon)
    @test isapprox(gradient.output_bias[7], finite; rtol=2.0f-3, atol=2.0f-4)

    @test !(:conditional_exact_vjp! in Set(names(Root)))
    @test :conditional_exact_vjp! in Set(names(Oracle))
end
