using Test

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyFullFeasibilityContract

struct MockOptimizerLeaf
    state
end

@testset "legacy full feasibility preregistration" begin
    @test SOURCE_ROWS == (1, 251, 501, 751, 1001, 1251)
    @test EXPECTED_EPISODE_IDS == (1, 2, 3, 4, 5, 6)
    @test EXPECTED_SEEDS == (5742, 5743, 5744, 5745, 5746, 5747)
    @test N_STEP == 3
    @test DISCOUNT === 0.997f0
    @test LEARNING_RATE === 1.0f-5
    @test WEIGHT_DECAY === 1.0f-4
    @test LEGACY_PARAMETER_COUNT == 20_787_454
    @test HARD_WALL_SECONDS == 1500
    @test GLOBAL_ONE_SHOT_MARKER ==
          raw"D:\tetris-paper-plus\runs\legacy_full_feasibility_F.started.json"
    environment = Dict(
        "F_OUTPUT_DIRECTORY" => raw"D:\tetris-paper-plus\runs\f output with spaces",
        "F_SUBSET_PATH" => raw"D:\tetris-paper-plus\runs\f output with spaces\subset.npz",
        "F_CHECKPOINT_PATH" => raw"C:\Users\fshuu\Documents\tetris\1313\mainmodel copy 3.jld2",
        "F_FREEZE_PATH" => raw"D:\tetris-paper-plus\runs\f output with spaces\freeze.json",
    )
    paths = resolve_benchmark_paths(String[], environment)
    @test paths.checkpoint_path == environment["F_CHECKPOINT_PATH"]
    @test occursin("mainmodel copy 3.jld2", paths.checkpoint_path)
    @test_throws ErrorException resolve_benchmark_paths(["only-one"], environment)
end

@testset "historical chunk semantics" begin
    @test chunk_ranges(51) == [1:16, 17:32, 33:48, 49:51]
    @test chunk_ranges(43) == [1:16, 17:32, 33:43]
    @test chunk_ranges(26) == [1:16, 17:26]
    @test_throws ArgumentError chunk_ranges(0)
end

@testset "frozen target and Huber" begin
    rewards = Float32[1 / 6, 0, 1 / 3]
    bootstrap = 5.25f0
    expected = rewards[1] + DISCOUNT * rewards[2] + DISCOUNT^2 * rewards[3] +
               DISCOUNT^3 * bootstrap
    @test frozen_nstep_target(rewards, bootstrap) == expected
    @test huber_scalar(0.5f0, 0.0f0) == 0.125f0
    @test huber_scalar(2.0f0, 0.0f0) == 1.5f0
end

@testset "recursive finite accounting" begin
    good = (; a=Float32[1, 2], leaf=MockOptimizerLeaf((Float32[3],)))
    bad = (; a=Float32[1], leaf=MockOptimizerLeaf((Float32[Inf],)))
    @test tree_all_finite(good)
    @test !tree_all_finite(bad)
    @test tree_array_elements(good) == 3
    @test tree_sum_abs2(good) == 14.0
    parameters = (; a=ones(Float32, 2), nested=(; b=ones(Float32, 3)))
    @test gradient_covers_parameters(
        parameters, (; a=zeros(Float32, 2), nested=(; b=zeros(Float32, 3)))
    )
    @test !gradient_covers_parameters(
        parameters, (; a=nothing, nested=(; b=zeros(Float32, 3)))
    )
end
