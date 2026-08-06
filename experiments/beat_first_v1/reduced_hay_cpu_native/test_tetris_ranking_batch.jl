using Test

const HERE = @__DIR__
if !isdefined(Main, :TetrisRankingBatch)
    include(joinpath(HERE, "TetrisRankingBatch.jl"))
end
const Ranking = Main.TetrisRankingBatch

"""Independent, allocation-friendly test oracle for line clearing/geometry."""
function oracle_candidate(board, placement)
    combined = min.(UInt8(1), board .+ placement)
    full = vec(sum(combined .!= 0; dims=2) .== 10)
    after = zeros(UInt8, 24, 10)
    output_row = count(full) + 1
    for board_row in 1:24
        full[board_row] && continue
        @views after[output_row, :] .= combined[board_row, :]
        output_row += 1
    end

    heights = zeros(Int, 10)
    holes_by_column = zeros(Int, 10)
    for column in 1:10
        first_filled = findfirst(!iszero, @view(after[:, column]))
        first_filled === nothing && continue
        heights[column] = 24 - first_filled + 1
        holes_by_column[column] = count(
            iszero,
            @view(after[first_filled:24, column]),
        )
    end

    reachable = falses(24, 10)
    queue = Tuple{Int,Int}[]
    for column in 1:10
        if iszero(after[1, column])
            reachable[1, column] = true
            push!(queue, (1, column))
        end
    end
    head = 1
    while head <= length(queue)
        board_row, column = queue[head]
        head += 1
        for (next_row, next_column) in (
            (board_row - 1, column),
            (board_row + 1, column),
            (board_row, column - 1),
            (board_row, column + 1),
        )
            1 <= next_row <= 24 || continue
            1 <= next_column <= 10 || continue
            iszero(after[next_row, next_column]) || continue
            reachable[next_row, next_column] && continue
            reachable[next_row, next_column] = true
            push!(queue, (next_row, next_column))
        end
    end
    cavities = count(index ->
        iszero(after[index]) && !reachable[index],
        eachindex(after),
    )
    wells = zeros(Int, 10)
    for column in 1:10
        left = column == 1 ? 24 : heights[column - 1]
        right = column == 10 ? 24 : heights[column + 1]
        wells[column] = max(min(left, right) - heights[column], 0)
    end
    return (;
        after,
        heights,
        holes_by_column,
        wells,
        line_clear=count(full),
        max_height=maximum(heights),
        holes=sum(holes_by_column),
        cavities,
        aggregate_height=sum(heights),
        bumpiness=sum(abs(heights[index] - heights[index - 1]) for index in 2:10),
    )
end

function deterministic_fixture()
    state_count = 8
    width = 80
    action_counts = Int[1, 2, 7, 16, 31, 47, 63, 80]
    boards = zeros(UInt8, 24, 10, 1, state_count)
    placements = zeros(UInt8, 24, 10, 1, width, state_count)
    queues = zeros(UInt8, 7, 6, state_count)
    teacher_q = zeros(Float32, width, state_count)
    candidate_death = falses(width, state_count)
    line_clear = zeros(Int8, width, state_count)
    max_height = zeros(Int8, width, state_count)
    holes = zeros(Int16, width, state_count)
    cavities = zeros(Int16, width, state_count)
    tspin = zeros(Float32, width, state_count)

    for state in 1:state_count
        for column in 1:10
            height = mod(3state + 2column, 5)
            for offset in 0:(height - 1)
                boards[24 - offset, column, 1, state] = 0x01
            end
        end
        for token in 1:6
            queues[mod1(state + 2token, 7), token, state] = 0x01
        end
        for candidate in 1:action_counts[state]
            column = mod1(3candidate + state, 10)
            other = mod1(column + 1 + mod(candidate, 3), 10)
            first_row = max(
                1,
                24 - count(!iszero, @view(boards[:, column, 1, state])),
            )
            second_row = max(
                1,
                24 - count(!iszero, @view(boards[:, other, 1, state])),
            )
            placements[first_row, column, 1, candidate, state] = 0x01
            placements[second_row, other, 1, candidate, state] = 0x01
            if iszero(mod(candidate + state, 5)) && first_row > 1
                placements[first_row - 1, column, 1, candidate, state] = 0x01
            end
            oracle = oracle_candidate(
                @view(boards[:, :, 1, state]),
                @view(placements[:, :, 1, candidate, state]),
            )
            line_clear[candidate, state] = Int8(oracle.line_clear)
            max_height[candidate, state] = Int8(oracle.max_height)
            holes[candidate, state] = Int16(oracle.holes)
            cavities[candidate, state] = Int16(oracle.cavities)
            teacher_q[candidate, state] =
                0.03125f0 * Float32(candidate * candidate) -
                0.1875f0 * Float32(state * candidate) +
                0.125f0 * Float32(mod(candidate + 3state, 7))
            candidate_death[candidate, state] =
                iszero(mod(candidate + state, 11))
            tspin[candidate, state] =
                iszero(mod(2candidate + state, 9)) ? 1.0f0 : 0.0f0
        end
    end
    selected_actions = Int[
        mod1(5state + 1, action_counts[state])
        for state in 1:state_count
    ]
    return (;
        boards,
        placements,
        queues,
        teacher_q,
        action_counts,
        selected_actions,
        terminal=Bool[false, true, false, true, true, false, false, true],
        candidate_death,
        candidate_death_available=
            Bool[true, false, true, false, true, false, true, false],
        line_clear,
        max_height,
        holes,
        cavities,
        ren=reshape(Float32[0, 1, 3, 7, 11, 16, 23, 30], 1, :),
        back_to_back=
            reshape(Float32[0, 1, 0, 1, 1, 0, 1, 0], 1, :),
        tspin,
    )
end

function oracle_rails(dataset, row::Int, candidate::Int)
    board = @view dataset.boards[:, :, 1, row]
    placement = @view dataset.placements[:, :, 1, candidate, row]
    oracle = oracle_candidate(board, placement)
    rails = zeros(Float32, Ranking.INPUT_RAILS)
    rail = 0
    for source in (board, oracle.after)
        for column in 1:10, board_row in 1:24
            rail += 1
            rails[rail] = source[board_row, column] != 0
        end
    end
    for comparison in (>, <)
        for column in 1:10, board_row in 1:24
            rail += 1
            rails[rail] = comparison(
                oracle.after[board_row, column],
                board[board_row, column],
            )
        end
    end
    for token in 1:6, piece in 1:7
        rail += 1
        rails[rail] = dataset.queues[piece, token, row] != 0
    end
    aux = Float32[
        oracle.heights ./ 24;
        oracle.holes_by_column ./ 24;
        oracle.wells ./ 24;
        oracle.cavities / 240;
        oracle.aggregate_height / 240;
        oracle.bumpiness / 216;
        oracle.max_height / 24;
        dataset.ren[1, row] / 30;
        dataset.back_to_back[1, row];
        dataset.tspin[candidate, row];
    ]
    @test length(aux) == Ranking.AUX_FEATURES
    for level in 1:Ranking.AUX_LEVELS, value in aux
        rail += 1
        rails[rail] = value >= Float32(level) / Float32(Ranking.AUX_LEVELS)
    end
    @test rail == Ranking.INPUT_RAILS
    return rails
end

function expected_teacher_z(dataset, row::Int, count::Int)
    total = 0.0f0
    for candidate in 1:count
        total += dataset.teacher_q[candidate, row]
    end
    average = total / Float32(count)
    square = 0.0f0
    for candidate in 1:count
        centered = dataset.teacher_q[candidate, row] - average
        square = muladd(centered, centered, square)
    end
    scale = max(sqrt(square / Float32(count)), 1.0f-4)
    return Float32[
        (dataset.teacher_q[candidate, row] - average) / scale
        for candidate in 1:count
    ]
end

function fill_raw!(raw)
    @inbounds for flat in axes(raw, 2), output in axes(raw, 1)
        raw[output, flat] =
            0.001953125f0 * Float32(output * flat) -
            0.015625f0 * Float32(mod(7output + 3flat, 19))
    end
    return raw
end

function prepared_loss_fixture()
    dataset = Ranking.validate_dataset(deterministic_fixture(), 80)
    batch = Ranking.Batch(8, 80)
    batch.rows .= Int[8, 1, 6, 3, 7, 2, 5, 4]
    Ranking.pack_batch!(batch, dataset, Ranking.PackScratch())
    fill_raw!(batch.raw)
    return batch, Ranking.LossScratch(80, 8)
end

function listnet_oracle(batch)
    loss = 0.0
    for state_slot in 1:batch.state_batch
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * batch.width
        q = Float64[batch.raw[1, offset + candidate] for candidate in 1:count]
        q .-= sum(q) / count
        q ./= sqrt(sum(abs2, q) / count + 1.0e-4)
        teacher = Float64[
            batch.targets.teacher_z[candidate, state_slot]
            for candidate in 1:count
        ] ./ 0.5
        student = q ./ 0.5
        teacher_probability = exp.(teacher .- maximum(teacher))
        teacher_probability ./= sum(teacher_probability)
        student_probability = exp.(student .- maximum(student))
        student_probability ./= sum(student_probability)
        loss -= sum(teacher_probability .* log.(student_probability)) /
            batch.state_batch
    end
    return loss
end

@testset "validated Tetris ranking batch" begin
    @test Ranking.INPUT_RAILS == 1298
    @test Ranking.OUTPUT_DIM == 22
    @test :structure_loss ∉ fieldnames(Ranking.SupervisedLoss)

    raw_dataset = deterministic_fixture()
    dataset = Ranking.validate_dataset(raw_dataset, 80)
    @test dataset.state_count == 8
    @test dataset.candidate_width == 80

    batch = Ranking.Batch(8, 80)
    batch.rows .= Int[8, 1, 6, 3, 7, 2, 5, 4]
    scratch = Ranking.PackScratch()
    Ranking.pack_batch!(batch, dataset, scratch)
    expected_counts = raw_dataset.action_counts[batch.rows]
    @test Int.(batch.counts) == expected_counts
    expected_flats = Int32[]
    for (slot, count) in enumerate(expected_counts), candidate in 1:count
        push!(expected_flats, Int32(Ranking.flat_index(candidate, slot, 80)))
    end
    @test batch.valid_flats[1:batch.valid_count] == expected_flats
    @test batch.valid_count == sum(expected_counts)

    for slot in 1:batch.state_batch
        row = batch.rows[slot]
        count = Int(batch.counts[slot])
        @test batch.targets.teacher_q[1:count, slot] ==
            raw_dataset.teacher_q[1:count, row]
        @test reinterpret(UInt32, batch.targets.teacher_z[1:count, slot]) ==
            reinterpret(UInt32, expected_teacher_z(raw_dataset, row, count))
        for candidate in 1:count
            flat = Ranking.flat_index(candidate, slot, 80)
            @test reinterpret(UInt32, batch.rails[:, flat]) ==
                reinterpret(UInt32, oracle_rails(raw_dataset, row, candidate))
            expected_mask =
                raw_dataset.candidate_death_available[row] ||
                candidate == raw_dataset.selected_actions[row]
            expected_death = raw_dataset.candidate_death_available[row] ?
                raw_dataset.candidate_death[candidate, row] :
                candidate == raw_dataset.selected_actions[row] &&
                raw_dataset.terminal[row]
            @test Bool(batch.targets.death_mask[candidate, slot]) == expected_mask
            @test Bool(batch.targets.death[candidate, slot]) == expected_death
        end
    end

    # The unvalidated object has no hot-path compatibility overload.
    @test !applicable(Ranking.pack_batch!, batch, raw_dataset, scratch)
    @test !applicable(Ranking.prepare_batch_metadata!, batch, raw_dataset)

    Ranking.pack_batch!(batch, dataset, scratch)
    @test @allocated(Ranking.pack_batch!(batch, dataset, scratch)) == 0

    fill_raw!(batch.raw)
    loss_scratch = Ranking.LossScratch(80, 8)
    loss = Ranking.supervised_loss_and_raw_gradient!(batch, loss_scratch)
    @test loss.listnet_loss ≈ listnet_oracle(batch) rtol=2.0e-6 atol=2.0e-6
    analytic = copy(batch.raw_gradient)
    epsilon = 2.0f-3
    for ordinal in 1:batch.valid_count
        flat = Int(batch.valid_flats[ordinal])
        for output in 1:Ranking.OUTPUT_DIM
            original = batch.raw[output, flat]
            batch.raw[output, flat] = original + epsilon
            plus = Ranking.supervised_loss_and_raw_gradient!(batch, loss_scratch).composite_loss
            batch.raw[output, flat] = original - epsilon
            minus = Ranking.supervised_loss_and_raw_gradient!(batch, loss_scratch).composite_loss
            batch.raw[output, flat] = original
            finite_difference = (plus - minus) / (2.0f0 * epsilon)
            @test analytic[output, flat] ≈ finite_difference rtol=3.0e-2 atol=1.5e-3
        end
    end
    Ranking.supervised_loss_and_raw_gradient!(batch, loss_scratch)
    @test @allocated(
        Ranking.supervised_loss_and_raw_gradient!(batch, loss_scratch)
    ) == 0
    valid = falses(batch.capacity)
    valid[Int.(batch.valid_flats[1:batch.valid_count])] .= true
    @test all(iszero, @view(batch.raw_gradient[:, .!valid]))
end

@testset "dataset contract rejects malformed arrays" begin
    fixture = deterministic_fixture()
    @test_throws DimensionMismatch Ranking.validate_dataset(
        merge(fixture, (; boards=reshape(fixture.boards, 240, 8))),
        80,
    )
    @test_throws DimensionMismatch Ranking.validate_dataset(
        merge(fixture, (; queues=reshape(fixture.queues, 42, 8))),
        80,
    )
    @test_throws DimensionMismatch Ranking.validate_dataset(
        merge(fixture, (; placements=fixture.placements[:, :, :, 1:79, :])),
        80,
    )
    @test_throws ArgumentError Ranking.validate_dataset(
        merge(fixture, (; teacher_q=Float64.(fixture.teacher_q))),
        80,
    )
    @test_throws ArgumentError Ranking.validate_dataset(
        merge(fixture, (; action_counts=Int16.(fixture.action_counts))),
        80,
    )
    @test_throws ArgumentError Ranking.validate_dataset(
        merge(fixture, (; holes=Int32.(fixture.holes))),
        80,
    )
    bad_counts = copy(fixture.action_counts)
    bad_counts[1] = 81
    @test_throws ArgumentError Ranking.validate_dataset(
        merge(fixture, (; action_counts=bad_counts)),
        80,
    )
    bad_selected = copy(fixture.selected_actions)
    bad_selected[2] = 3
    @test_throws ArgumentError Ranking.validate_dataset(
        merge(fixture, (; selected_actions=bad_selected)),
        80,
    )
end

@testset "loss arena rejects shape and count corruption" begin
    batch, scratch = prepared_loss_fixture()

    # Every scratch backing is fixed-shape under normal Julia APIs, and the
    # immutable wrapper prevents replacing an individual buffer after setup.
    @test !ismutable(scratch)
    for name in fieldnames(Ranking.LossScratch)
        backing = getfield(scratch, name)
        @test backing isa Matrix{Float32}
        @test_throws MethodError resize!(backing, max(length(backing) - 1, 0))
    end
    @test_throws ErrorException setfield!(
        scratch,
        :teacher_probability,
        zeros(Float32, 79, 1),
    )
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        Ranking.LossScratch(79, 8),
    )
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        Ranking.LossScratch(80, 7),
    )

    # Batch vectors remain mutable arena metadata.  A resized or internally
    # inconsistent vector must be rejected before the bounds-elided kernel.
    original_counts = batch.counts
    batch.counts = batch.counts[1:7]
    fill!(batch.raw_gradient, 7.0f0)
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    @test all(==(7.0f0), batch.raw_gradient)
    batch.counts = original_counts

    original_listnet = batch.listnet_q_gradient
    batch.listnet_q_gradient = batch.listnet_q_gradient[1:(end - 1)]
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.listnet_q_gradient = original_listnet

    original_valid_flats = batch.valid_flats
    batch.valid_flats = batch.valid_flats[1:(end - 1)]
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.valid_flats = original_valid_flats

    original_count = batch.counts[1]
    batch.counts[1] = Int16(0)
    @test_throws ArgumentError Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.counts[1] = original_count
    batch.counts[1] = Int16(81)
    @test_throws ArgumentError Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.counts[1] = original_count

    original_valid_count = batch.valid_count
    batch.valid_count += 1
    @test_throws ArgumentError Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.valid_count = original_valid_count

    original_top1 = batch.targets.top1[1]
    batch.targets.top1[1] = Int16(0)
    @test_throws ArgumentError Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.targets.top1[1] = original_top1

    original_top2 = batch.targets.top2
    batch.targets.top2 = batch.targets.top2[1:7]
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.targets.top2 = original_top2

    # Every matrix that the loss indexes is shape-proved once.  Replacing any
    # one of them with a truncated matrix must fail without entering @inbounds.
    for name in (
        :teacher_q,
        :teacher_z,
        :death,
        :death_mask,
        :line_clear,
        :max_height,
        :holes,
        :cavities,
    )
        original = getfield(batch.targets, name)
        setfield!(
            batch.targets,
            name,
            zeros(Float32, size(original, 1) - 1, size(original, 2)),
        )
        @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
            batch,
            scratch,
        )
        setfield!(batch.targets, name, original)
    end

    original_raw = batch.raw
    batch.raw = zeros(Float32, Ranking.OUTPUT_DIM, batch.capacity - 1)
    @test_throws DimensionMismatch Ranking.supervised_loss_and_raw_gradient!(
        batch,
        scratch,
    )
    batch.raw = original_raw

    Ranking.supervised_loss_and_raw_gradient!(batch, scratch)
    @test @allocated(
        Ranking.supervised_loss_and_raw_gradient!(batch, scratch)
    ) == 0
end
