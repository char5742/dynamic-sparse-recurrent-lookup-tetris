using Test
using JLD2

const HERE = @__DIR__
if !isdefined(Main, :CandidateDeltaInput)
    include(joinpath(HERE, "CandidateDeltaInput.jl"))
end
if !isdefined(Main, :TetrisRankingBatch)
    include(joinpath(HERE, "TetrisRankingBatch.jl"))
end

const DeltaInput = Main.CandidateDeltaInput
const RankingInput = Main.TetrisRankingBatch

function synthetic_clear_case(line_count::Int)
    0 <= line_count <= 4 || throw(ArgumentError("line_count must be 0:4"))
    common = DeltaInput.StateCommon()
    placement = zeros(UInt8, DeltaInput.BOARD_ROWS, DeltaInput.BOARD_COLUMNS)
    if iszero(line_count)
        placement[end, 1] = 0x01
    else
        for board_row in (DeltaInput.BOARD_ROWS - line_count + 1):DeltaInput.BOARD_ROWS
            @views common.board[board_row, 1:9] .= 0x01
            placement[board_row, 10] = 0x01
        end
    end
    common.queue[mod1(line_count + 1, 7), 1] = 0x01
    common.ren[1] = Float32(3line_count)
    common.back_to_back[1] = isodd(line_count) ? 1.0f0 : 0.0f0

    delta = DeltaInput.CandidateDelta()
    DeltaInput.prepare_candidate_delta!(
        delta,
        common,
        placement,
        iszero(line_count) ? 0.0f0 : 1.0f0,
    )
    output = DeltaInput.CandidateMaterialization()
    rails = zeros(Float32, DeltaInput.INPUT_RAILS)
    DeltaInput.pack_candidate_rails!(rails, common, delta, output)
    return common, delta, output, rails
end

function canonical_reference_rails(common, delta)
    dataset = (;
        boards=reshape(copy(common.board), 24, 10, 1, 1),
        placements=reshape(copy(delta.placement), 24, 10, 1, 1, 1),
        queues=reshape(copy(common.queue), 7, 6, 1),
        ren=copy(common.ren),
        back_to_back=copy(common.back_to_back),
        tspin=copy(delta.tspin),
    )
    scratch = RankingInput.PackScratch()
    line_count = RankingInput._fill_after_board!(scratch, dataset, 1, 1)
    cavities, aggregate_height, bumpiness, max_height =
        RankingInput._fill_geometry!(scratch)
    rails = zeros(Float32, RankingInput.INPUT_RAILS)
    rail = 0
    for source in (
        scratch.board,
        scratch.after,
        scratch.after .> scratch.board,
        scratch.after .< scratch.board,
    )
        for column in 1:10, board_row in 1:24
            rail += 1
            rails[rail] = source[board_row, column] > 0.5f0
        end
    end
    for token in 1:6, piece in 1:7
        rail += 1
        rails[rail] = common.queue[piece, token] != 0
    end
    for level in 1:RankingInput.AUX_LEVELS
        threshold = Float32(level) / Float32(RankingInput.AUX_LEVELS)
        for index in 1:RankingInput.AUX_FEATURES
            rail += 1
            rails[rail] = RankingInput._aux_value(
                scratch,
                dataset,
                1,
                1,
                index,
                cavities,
                aggregate_height,
                bumpiness,
                max_height,
            ) >= threshold
        end
    end
    @assert rail == RankingInput.INPUT_RAILS
    return line_count, scratch, rails
end

function real_teacher_part()
    part = joinpath(
        raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3\parts",
        "part__train__epsilon__seed110003__eps0p200.jld2",
    )
    isfile(part) || error("focused real-teacher shard is missing: $part")
    raw = JLD2.load(part)
    states = length(raw["action_counts"])
    width = size(raw["placements"], 4)
    return (;
        boards=raw["boards"],
        placements=raw["placements"],
        queues=raw["queues"],
        teacher_q=raw["teacher_q"],
        action_counts=Int.(raw["action_counts"]),
        selected_actions=Int.(raw["selected_actions"]),
        terminal=raw["terminal"],
        candidate_death=raw["death"],
        candidate_death_available=trues(states),
        line_clear=raw["line_clear"],
        max_height=raw["max_height"],
        holes=raw["holes"],
        cavities=raw["cavities"],
        ren=raw["ren"],
        back_to_back=raw["back_to_back"],
        tspin=raw["tspin"],
    ), width
end

function hot_allocation_probe(common, delta, output, rails, dataset)
    DeltaInput.prepare_state_common!(common, dataset, 1)
    DeltaInput.prepare_candidate_delta!(delta, common, dataset, 1, 1)
    DeltaInput.pack_candidate_rails!(rails, common, delta, output)
    state_bytes = @allocated DeltaInput.prepare_state_common!(common, dataset, 1)
    delta_bytes = @allocated DeltaInput.prepare_candidate_delta!(
        delta,
        common,
        dataset,
        1,
        1,
    )
    pack_bytes = @allocated DeltaInput.pack_candidate_rails!(
        rails,
        common,
        delta,
        output,
    )
    checksum = 0
    for index in 1:DeltaInput.placement_count(delta)
        checksum += Int(DeltaInput.placement_position(delta, index))
    end
    position_bytes = @allocated begin
        local total = 0
        for index in 1:DeltaInput.placement_count(delta)
            total += Int(DeltaInput.placement_position(delta, index))
        end
        total
    end
    return state_bytes, delta_bytes, pack_bytes, position_bytes, checksum
end

@testset "candidate-delta exact Tetris input" begin
    @test DeltaInput.INPUT_RAILS == 1_298

    @testset "synthetic 0:4 line-clear row maps" begin
        for line_count in 0:4
            common, delta, output, rails = synthetic_clear_case(line_count)
            reference_count, reference, reference_rails =
                canonical_reference_rails(common, delta)
            @test Int(delta.line_clear[1]) == line_count
            @test reference_count == line_count
            @test all(value -> value == 0.0f0 || value == 1.0f0, rails)
            @test length(rails) == 1_298
            @test reinterpret(UInt32, rails) == reinterpret(UInt32, reference_rails)
            @test output.after == UInt8.(reference.after .> 0.5f0)
            @test output.added == UInt8.(reference.after .> reference.board)
            @test output.removed == UInt8.(reference.after .< reference.board)

            first_cleared = DeltaInput.BOARD_ROWS - line_count + 1
            for source_row in 1:DeltaInput.BOARD_ROWS
                expected = !iszero(line_count) && source_row >= first_cleared ?
                    0 : source_row + line_count
                @test Int(delta.source_to_after[source_row]) == expected
            end
            for after_row in 1:DeltaInput.BOARD_ROWS
                expected = after_row <= line_count ? 0 : after_row - line_count
                @test Int(delta.after_to_source[after_row]) == expected
            end

            if iszero(line_count)
                @test count(!iszero, output.after) == 1
                @test count(!iszero, output.added) == 1
                @test count(!iszero, output.removed) == 0
            else
                @test count(!iszero, output.after) == 0
                @test count(!iszero, output.added) == 0
                @test count(!iszero, output.removed) == 9line_count
            end
            @test output.aux[35] === common.ren[1] / 30.0f0
            @test output.aux[36] === common.back_to_back[1]
            @test output.aux[37] === delta.tspin[1]

            expected_positions = UInt16[]
            for column in 1:DeltaInput.BOARD_COLUMNS,
                board_row in 1:DeltaInput.BOARD_ROWS
                !iszero(delta.placement[board_row, column]) && push!(
                    expected_positions,
                    UInt16(board_row +
                           (column - 1) * DeltaInput.BOARD_ROWS),
                )
            end
            @test DeltaInput.placement_count(delta) ==
                length(expected_positions)
            @test [DeltaInput.placement_position(delta, index)
                   for index in eachindex(expected_positions)] ==
                expected_positions
            @test_throws BoundsError DeltaInput.placement_position(delta, 0)
            @test_throws BoundsError DeltaInput.placement_position(
                delta,
                length(expected_positions) + 1,
            )
        end
    end

    @testset "raw placement positions are exact, fixed-capacity and fail closed" begin
        common = DeltaInput.StateCommon()
        delta = DeltaInput.CandidateDelta()
        @test delta.placement_positions isa Memory{UInt16}
        @test length(delta.placement_positions) == DeltaInput.PLACEMENT_CAPACITY
        @test size(delta.placement_count) == (1, 1)
        placement = zeros(UInt8, DeltaInput.BOARD_ROWS, DeltaInput.BOARD_COLUMNS)

        placement[24, 1] = 0x01
        placement[22, 2] = 0x01
        placement[21, 2] = 0x01
        placement[3, 10] = 0x01
        DeltaInput.prepare_candidate_delta!(delta, common, placement, 0.0f0)
        @test DeltaInput.placement_count(delta) == 4
        @test [DeltaInput.placement_position(delta, index) for index in 1:4] ==
            UInt16[24, 45, 46, 219]

        fill!(placement, 0x00)
        placement[7, 4] = 0x01
        DeltaInput.prepare_candidate_delta!(delta, common, placement, 0.0f0)
        @test DeltaInput.placement_count(delta) == 1
        @test DeltaInput.placement_position(delta, 1) == UInt16(79)
        @test collect(delta.placement_positions) == UInt16[79, 0, 0, 0]

        placement[8, 4] = 0x01
        placement[9, 4] = 0x01
        placement[10, 4] = 0x01
        placement[11, 4] = 0x01
        @test_throws ErrorException DeltaInput.prepare_candidate_delta!(
            delta,
            common,
            placement,
            0.0f0,
        )
        @test DeltaInput.placement_count(delta) == 0
        @test all(iszero, delta.placement_positions)
    end

    @testset "real teacher shard is bitwise identical" begin
        raw_dataset, width = real_teacher_part()
        dataset = RankingInput.validate_dataset(raw_dataset, width)
        batch = RankingInput.Batch(3, width)
        batch.rows .= (1, 17, 50)
        RankingInput.prepare_batch_metadata!(batch, dataset)
        reference_scratch = RankingInput.PackScratch()

        common = DeltaInput.StateCommon()
        delta = DeltaInput.CandidateDelta()
        output = DeltaInput.CandidateMaterialization()
        rails = zeros(Float32, DeltaInput.INPUT_RAILS)
        checked = 0
        for state_slot in 1:3
            row = batch.rows[state_slot]
            DeltaInput.prepare_state_common!(common, dataset, row)
            count = Int(batch.counts[state_slot])
            for candidate in unique((1, min(count, 2), count))
                flat = RankingInput.flat_index(candidate, state_slot, width)
                RankingInput.pack_candidate_rails!(
                    batch,
                    dataset,
                    reference_scratch,
                    flat,
                )
                DeltaInput.prepare_candidate_delta!(
                    delta,
                    common,
                    dataset,
                    row,
                    candidate,
                )
                DeltaInput.pack_candidate_rails!(rails, common, delta, output)
                @test reinterpret(UInt32, rails) ==
                    reinterpret(UInt32, batch.rails[:, flat])
                @test Int(delta.line_clear[1]) ==
                    Int(dataset.line_clear[candidate, row])
                expected_positions = findall(!iszero, vec(delta.placement))
                @test DeltaInput.placement_count(delta) ==
                    length(expected_positions)
                @test [Int(DeltaInput.placement_position(delta, index))
                       for index in eachindex(expected_positions)] ==
                    expected_positions
                @test output.after == UInt8.(reference_scratch.after .> 0.5f0)
                @test output.added == UInt8.(
                    reference_scratch.after .> reference_scratch.board
                )
                @test output.removed == UInt8.(
                    reference_scratch.after .< reference_scratch.board
                )
                checked += 1
            end
        end
        @test checked >= 6

        state_bytes, delta_bytes, pack_bytes, position_bytes, checksum =
            hot_allocation_probe(common, delta, output, rails, dataset)
        @test (state_bytes, delta_bytes, pack_bytes, position_bytes) ==
            (0, 0, 0, 0)
        @test checksum >= 0
    end
end
