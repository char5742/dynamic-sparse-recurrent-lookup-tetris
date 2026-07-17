using Test

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyPartialTailTDContract

@testset "P1 fixed preregistration" begin
    @test TRAINABLE_PATHS == (
        "board_net.resblocks.layer_29",
        "board_net.resblocks.layer_31",
        "board_net.conv2",
        "board_net.norm2",
        "score_net",
    )
    @test TRAINABLE_PARAMETER_COUNT == 2_949_508
    @test FROZEN_PARAMETER_COUNT == 17_837_946
    @test TRAINABLE_PARAMETER_COUNT + FROZEN_PARAMETER_COUNT == LEGACY_PARAMETER_COUNT
    @test OPTIMIZER_MOMENT_ELEMENTS == 2 * TRAINABLE_PARAMETER_COUNT
    @test DATA_ORDER_SEED == UInt64(0x1313_2026)
    @test UPDATE_COUNT == 300
    @test (
        STEP0_WITNESS_ROW,
        STEP0_WITNESS_EPISODE,
        STEP0_WITNESS_STEP,
        STEP0_WITNESS_COUNT,
        STEP0_WITNESS_SELECTED,
    ) == (1055, 5, 55, 52, 11)
    @test DEVELOPMENT_SEEDS == (5756, 5757)
    @test GLOBAL_ONE_SHOT_MARKER ==
          raw"D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json"
end

@testset "historical chunks, targets, and anchored Huber" begin
    @test chunk_ranges(51) == [1:16, 17:32, 33:48, 49:51]
    @test selected_chunk_range(35, 51) == 33:48
    @test selected_chunk_range(51, 51) == 49:51
    rewards = Float32[1, 2, 3]
    @test frozen_nstep_target(rewards, Bool[false, true, false], 99) ==
          rewards[1] + DISCOUNT * rewards[2]
    @test frozen_nstep_target(rewards, falses(3), 4) ==
          rewards[1] + DISCOUNT * rewards[2] + DISCOUNT^2 * rewards[3] + DISCOUNT^3 * 4
    scores = Float32[0, 2]
    old = Float32[0, 0]
    @test anchored_loss(scores, 2, 0, old) == 1.5f0 + 0.75f0
end

@testset "actual-array gradient contract" begin
    parameters = (;
        block=(;
            weight=ones(Float32, 2, 3),
            empty=(; pool=NamedTuple()),
        ),
        bias=ones(Float32, 2),
    )
    good = (;
        block=(; weight=zeros(Float32, 2, 3), empty=nothing),
        bias=zeros(Float32, 2),
    )
    validation = validate_gradient(parameters, good)
    @test validation.valid
    @test validation.gradient_elements == 8
    empty_tuple = (;
        block=(; weight=zeros(Float32, 2, 3), empty=(; pool=NamedTuple())),
        bias=zeros(Float32, 2),
    )
    @test validate_gradient(parameters, empty_tuple).valid
    missing = (;
        block=(; weight=nothing, empty=nothing),
        bias=zeros(Float32, 2),
    )
    failure = validate_gradient(parameters, missing)
    @test !failure.valid
    @test occursin("block.weight", failure.first_mismatch)
    extra = (;
        block=(; weight=zeros(Float32, 2, 3), empty=nothing, extra=ones(Float32, 1)),
        bias=zeros(Float32, 2),
    )
    extra_failure = validate_gradient(parameters, extra)
    @test !extra_failure.valid
    @test occursin("block.extra", extra_failure.first_mismatch)
end

@testset "frozen hash and time gates" begin
    parameters = (;
        board_net=(;
            resblocks=(;
                layer_29=(; weight=Float32[1]),
                layer_31=(; weight=Float32[2]),
                layer_1=(; weight=Float32[3]),
            ),
            conv2=(; weight=Float32[4]),
            norm2=(; scale=Float32[5]),
            conv1=(; weight=Float32[6]),
        ),
        score_net=(; weight=Float32[7]),
        attention=(; weight=Float32[8]),
    )
    hashes = frozen_parameter_hashes(parameters)
    @test sort(collect(keys(hashes))) == [
        "attention.weight",
        "board_net.conv1.weight",
        "board_net.resblocks.layer_1.weight",
    ]
    @test projected_total_seconds(100, 4) == 100 + 293 * 4 + 100 + 120 + 400 * 0.411
    @test finite_difference_pass(1.0, 1.01)[1]
    @test !finite_difference_pass(1.0, 1.1)[1]
end
