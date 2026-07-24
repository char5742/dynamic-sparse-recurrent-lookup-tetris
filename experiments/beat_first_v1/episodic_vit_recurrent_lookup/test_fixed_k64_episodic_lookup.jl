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
    ENV[name] = get(ENV, name, value)
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
        union_ids = unique(vec(selected))
        union_count = Int(step.cross.union_count)
        @test union_count == length(union_ids)
        @test step.cross.union_ids[1:union_count] == union_ids
        for register in 1:Model.REGISTER_COUNT,
                support_index in 1:Model.EPISODIC_SUPPORT
            slot = Int(step.cross.support_slots[support_index, register])
            @test step.cross.union_ids[slot] ==
                selected[support_index, register]
        end
        expected_mean = vec(mean(step.spatial.output_normalized; dims=2))
        @test isapprox(
            step.cross.global_summary, expected_mean;
            rtol=2.0f-6, atol=2.0f-6,
        )
        write_count = Int(step.memory_write.write_count)
        @test write_count == union_count
        @test step.memory_write.write_ids[1:write_count] == union_ids
        expected_write_projection =
            model.memory_write_o *
            @view(step.memory_write.context[:, 1:write_count])
        @test isapprox(
            @view(step.memory_write.projected[:, 1:write_count]),
            expected_write_projection;
            rtol=3.0f-5,
            atol=3.0f-5,
        )
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

    # The factorized write VJP must equal the original token-wise dense
    # formulation, not merely produce finite gradients.
    write_tape = first(tape.steps).memory_write
    write_count = Int(write_tape.write_count)
    memory_cotangent = randn(
        rng, Float32, Model.MODEL_DIM, Model.TOKEN_COUNT,
    )
    reference_dprojected = Matrix{Float32}(
        undef, Model.MODEL_DIM, write_count,
    )
    for write_index in 1:write_count
        token = Int(write_tape.write_ids[write_index])
        @views reference_dprojected[:, write_index] .=
            write_tape.alpha .* memory_cotangent[:, token]
    end
    reference_do =
        reference_dprojected *
        transpose(@view(write_tape.context[:, 1:write_count]))
    reference_dcontext =
        transpose(model.memory_write_o) * reference_dprojected
    reference_dvalue = zeros(
        Float32, Model.ATTENTION_DIM, Model.REGISTER_COUNT,
    )
    reference_external = zeros(
        Float32,
        Model.EPISODIC_SUPPORT,
        Model.REGISTER_COUNT,
        Model.ATTENTION_HEADS,
    )
    token_projection = zeros(
        Float32, Model.TOKEN_COUNT, Model.ATTENTION_HEADS,
    )
    head_dim = Model.ATTENTION_DIM ÷ Model.ATTENTION_HEADS
    for register in 1:Model.REGISTER_COUNT,
            support_index in 1:Model.EPISODIC_SUPPORT
        token = Int(write_tape.selected_ids[support_index, register])
        write_index = Int(write_tape.token_slots[token])
        for head in 1:Model.ATTENTION_HEADS
            coordinates =
                ((head - 1) * head_dim + 1):(head * head_dim)
            dweight = dot(
                @view(reference_dcontext[coordinates, write_index]),
                @view(write_tape.value[coordinates, register]),
            )
            reference_external[support_index, register, head] = dweight
            token_projection[token, head] = muladd(
                write_tape.weights[support_index, register, head],
                dweight,
                token_projection[token, head],
            )
            @views reference_dvalue[coordinates, register] .+=
                write_tape.weights[support_index, register, head] .*
                reference_dcontext[coordinates, write_index]
        end
    end
    for register in 1:Model.REGISTER_COUNT,
            support_index in 1:Model.EPISODIC_SUPPORT
        token = Int(write_tape.selected_ids[support_index, register])
        for head in 1:Model.ATTENTION_HEADS
            reference_external[support_index, register, head] =
                write_tape.inverse_weight_sums[token, head] * (
                    reference_external[support_index, register, head] -
                    token_projection[token, head]
                )
        end
    end
    reference_dv =
        reference_dvalue * transpose(write_tape.registers)
    reference_dregisters =
        transpose(model.memory_write_v) * reference_dvalue
    write_accumulator = Model.GradientAccumulator(model)
    state_cotangent = zeros(
        Float32, Model.MODEL_DIM, Model.REGISTER_COUNT,
    )
    factorized_external = Model._working_memory_write_vjp!(
        write_accumulator,
        model,
        write_tape,
        memory_cotangent,
        state_cotangent,
        write_accumulator.backward_scratch,
    )
    @test isapprox(
        write_accumulator.dense[:memory_write_o], reference_do;
        rtol=3.0f-5, atol=3.0f-5,
    )
    @test isapprox(
        write_accumulator.dense[:memory_write_v], reference_dv;
        rtol=3.0f-5, atol=3.0f-5,
    )
    @test isapprox(
        state_cotangent, reference_dregisters;
        rtol=3.0f-5, atol=3.0f-5,
    )
    @test isapprox(
        @view(factorized_external[
            1:Model.EPISODIC_SUPPORT, :, :
        ]),
        reference_external;
        rtol=3.0f-5, atol=3.0f-5,
    )

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
