#!/usr/bin/env julia

using LinearAlgebra
using Random
using Statistics
using Test

for (name, value) in (
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
    "EVRL_FFN_DIM" => "128",
)
    ENV[name] = value
end

include(joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))
const Model = Main.EpisodicViTRecurrentLookup
BLAS.set_num_threads(1)

@testset "fixed K=$(Model.EPISODIC_SUPPORT) episodic lookup" begin
    rng = Xoshiro(0x4b3634455049534f)
    model = Model.initialize_model(rng)
    input = Model.EpisodicCandidateInput(
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.PIECE_TYPES, Model.NEXT_HOLD_TOKENS),
        randn(rng, Float32, Model.AUX_FEATURES),
    )
    output, tape = Model.forward_trajectory(
        model, input; forced_depth=2, training=false,
    )
    @test length(output) == Model.OUTPUT_DIM
    @test all(isfinite, output)
    @test length(tape.steps) == 2

    for step in tape.steps
        selected = step.cross.selected_ids
        @test size(selected) ==
            (Model.EPISODIC_SUPPORT, Model.REGISTER_COUNT)
        for register in 1:Model.REGISTER_COUNT
            @test length(unique(@view selected[:, register])) ==
                Model.EPISODIC_SUPPORT
        end
        expected_mean = vec(mean(step.spatial.output_normalized; dims=2))
        @test isapprox(
            step.cross.global_summary, expected_mean;
            rtol=2.0f-6, atol=2.0f-6,
        )
        write_count = Int(step.memory_write.write_count)
        @test 1 <= write_count <= min(
            Model.TOKEN_COUNT,
            Model.EPISODIC_SUPPORT * Model.REGISTER_COUNT,
        )
        selected_union = Set(vec(selected))
        @test all(
            token -> token in selected_union,
            @view(step.memory_write.write_ids[1:write_count]),
        )
    end

    accumulator = Model.GradientAccumulator(model)
    Model.backward_trajectory!(
        accumulator,
        model,
        tape,
        randn(rng, Float32, Model.OUTPUT_DIM);
        realized_loss=1.0f0,
        baseline=1.0f0,
    )
    @test isfinite(Model.gradient_norm(accumulator))
    @test any(!iszero, accumulator.dense[:cross_q])
    @test any(!iszero, accumulator.dense[:token_router])
    @test any(!iszero, accumulator.dense[:register_router])
    @test any(!iszero, accumulator.dense[:relation_scale_logit])
    @test all(accumulator.active_tokens)
end
