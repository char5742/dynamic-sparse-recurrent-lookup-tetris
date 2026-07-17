using Lux
using Random
using Test

include(joinpath(@__DIR__, "contract.jl"))
using .FrozenOldAdditiveResidualQ1Contract
include(joinpath(@__DIR__, "model.jl"))

function main(args=ARGS)
    @testset "Q1 frozen-old additive residual contract" begin
        @test TRAIN_ROWS == 1:2160
        @test OFFLINE_ROWS == 2161:2660
        @test ACTIONS == 74
        @test STATE_BATCH == 4
        @test UPDATE_COUNT == 2000
        @test RNG_SEED == UInt64(0x5131_2026)
        @test expected_constants().offline_role == "reused_development_guard"
        @test expected_constants().initializer_exposed_to_offline_rows
        @test !expected_constants().game_strength_evidence

        model = q1_model()
        ps, st = Lux.setup(Xoshiro(0x5131), model)
        @test Lux.parameterlength(ps) == PARAMETER_COUNT
        zeroed = zero_scalar_head(ps)
        @test only_scalar_head_changed(ps, zeroed)
        before = array_leaves(ps)
        after = array_leaves(zeroed)
        @test all(iszero, after["head.layer_3.weight"])
        @test all(iszero, after["head.layer_3.bias"])
        @test before["head.layer_2.weight"] == after["head.layer_2.weight"]

        # DAgger behavior is allowed to differ from the old-Q argmax. The
        # target contract therefore cannot assert selected == argmax.
        old_q = Float32[3, 1, 2]
        selected = 2
        @test selected != argmax(old_q)
        rewards = Float32[0.1, 0.2, 0.3]
        bootstrap = Float32[-2, 4, 1]
        y3 = rewards[1] + GAMMA * rewards[2] + GAMMA^2 * rewards[3] + GAMMA^3 * maximum(bootstrap)
        wrong_selected_bootstrap = rewards[1] + GAMMA * rewards[2] + GAMMA^2 * rewards[3] + GAMMA^3 * bootstrap[1]
        @test y3 != wrong_selected_bootstrap

        stored = Float32[1.25, -2.0, 0.0]
        correction = zeros(Float32, 3)
        combined = stored .+ correction
        @test reinterpret(UInt32, combined) == reinterpret(UInt32, stored)
        @test argmax(combined) == argmax(stored)

        witness = global_clip_synthetic_witness()
        @test witness.statistics.clip_mode == "single_global_tree_l2"
        @test witness.statistics.global_gradient_norm_before == 13.0
        @test witness.statistics.global_gradient_scale == 1 / 13
        @test isapprox(witness.statistics.global_gradient_norm_after, 1.0; atol=1.0e-6, rtol=0)
        @test isapprox(witness.first_norm, 5 / 13; atol=1.0e-6, rtol=0)
        @test isapprox(witness.second_norm, 12 / 13; atol=1.0e-6, rtol=0)
        @test witness.statistics.all_leaves_same_scale
        @test witness.empty_named_tuple_preserved
        @test witness.nothing_preserved
    end
    println("Q1 Julia synthetic contract checks passed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
