using Test
using Random
using LinearAlgebra

module SpatialProgramPacketsTestHarness
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "SpatialProgramPackets.jl"))
end

const H = SpatialProgramPacketsTestHarness
const Bank = H.DendriticProgramBank
const Spatial = H.SpatialProgramPackets

function local_sites(position)
    column = div(position - 1, Spatial.BOARD_ROWS) + 1
    row = position - (column - 1) * Spatial.BOARD_ROWS
    sites = Tuple{Int,Int}[]
    for column_offset in -1:1, row_offset in -1:1
        local_row = row + row_offset
        local_column = column + column_offset
        if 1 <= local_row <= Spatial.BOARD_ROWS &&
           1 <= local_column <= Spatial.BOARD_COLUMNS
            push!(sites, (local_row, local_column))
        end
    end
    return sites
end

@testset "exact collision-free spatial address exhausts all binary domains" begin
    @test Spatial.POSITION_COUNT == 240
    @test Spatial.PLANE_COUNT == 2
    @test Spatial.PACKET_WIDTH == 16
    @test Spatial.PACKET_COUNT == 480
    @test Spatial.packet_column(1, Spatial.BEFORE_PLANE) == 1
    @test Spatial.packet_column(240, Spatial.BEFORE_PLANE) == 240
    @test Spatial.packet_column(1, Spatial.AFTER_PLANE) == 241
    @test Spatial.packet_column(240, Spatial.AFTER_PLANE) == 480

    seen = ntuple(table -> falses(Bank.TABLE_ROW_COUNTS[table]), Bank.TABLE_COUNT)
    board = zeros(UInt8, Spatial.BOARD_ROWS, Spatial.BOARD_COLUMNS)
    table4_collision = false
    enumerated = 0
    @inbounds for plane in (Spatial.BEFORE_PLANE, Spatial.AFTER_PLANE)
        for position in 1:Spatial.POSITION_COUNT
            sites = local_sites(position)
            for pattern in 0:(Int(1) << length(sites)) - 1
                fill!(board, 0x00)
                for bit in eachindex(sites)
                    iszero(pattern & (Int(1) << (bit - 1))) && continue
                    row, column = sites[bit]
                    board[row, column] = 0x01
                end
                rows = Spatial.spatial_program_rows(board, position, plane)
                for table in 1:Bank.TABLE_COUNT
                    physical = Int(Bank.active_row(rows, table))
                    local_row = physical - Bank.TABLE_ROW_OFFSETS[table]
                    if table == 4 && seen[table][local_row]
                        table4_collision = true
                    end
                    seen[table][local_row] = true
                end
                enumerated += 1
            end
        end
    end
    @test enumerated == Bank.TABLE_ROW_COUNTS[4]
    @test !table4_collision
    @test all(all, seen)
end

@testset "before and after identity is explicit" begin
    board = zeros(UInt8, Spatial.BOARD_ROWS, Spatial.BOARD_COLUMNS)
    board[1, 1] = 1
    board[2, 1] = 1
    before = Spatial.spatial_program_rows(board, 1, Spatial.BEFORE_PLANE)
    after = Spatial.spatial_program_rows(board, 1, Spatial.AFTER_PLANE)
    @test before.rows[1:3] == after.rows[1:3]
    @test Int(after.rows[4] - before.rows[4]) == 94_016
    @test before.rows[4] != after.rows[4]

    @test_throws DimensionMismatch Spatial.spatial_program_rows(
        zeros(UInt8, 20, 10), 1, Spatial.BEFORE_PLANE,
    )
    @test_throws BoundsError Spatial.spatial_program_rows(
        board, 0, Spatial.BEFORE_PLANE,
    )
    @test_throws ArgumentError Spatial.spatial_program_rows(board, 1, 3)
end

@testset "base grid and selected candidate after packets are exact" begin
    rng = MersenneTwister(0x5a17)
    bank = Bank.ProgramBank(0x5a17)
    workspace = Spatial.SpatialPacketWorkspace()
    board = UInt8.(rand(rng, Bool, Spatial.BOARD_ROWS, Spatial.BOARD_COLUMNS))
    base_grid = zeros(Float32, Spatial.PACKET_WIDTH, Spatial.PACKET_COUNT)
    Spatial.base_packet_grid!(base_grid, workspace, bank, board)

    direct = zeros(Float32, Spatial.PACKET_WIDTH)
    for plane in (Spatial.BEFORE_PLANE, Spatial.AFTER_PLANE),
        position in (1, 27, 119, 240)
        rows = Spatial.spatial_program_rows(board, position, plane)
        Bank.program_packet!(direct, bank, rows)
        @test direct == base_grid[:, Spatial.packet_column(position, plane)]
    end
    @test base_grid[:, 1:240] != base_grid[:, 241:480]

    after = copy(board)
    after[8, 4] = xor(after[8, 4], 0x01)
    after[19, 9] = xor(after[19, 9], 0x01)
    positions = UInt16[80, 104, 211]
    candidate = zeros(Float32, Spatial.PACKET_WIDTH, length(positions))
    delta = similar(candidate)
    Spatial.candidate_after_packets!(
        candidate, delta, workspace, bank, base_grid, after, positions,
    )
    @inbounds for index in eachindex(positions)
        position = Int(positions[index])
        rows = Spatial.spatial_program_rows(after, position, Spatial.AFTER_PLANE)
        Bank.program_packet!(direct, bank, rows)
        @test candidate[:, index] == direct
        base_column = Spatial.packet_column(position, Spatial.AFTER_PLANE)
        @test delta[:, index] == direct - base_grid[:, base_column]
    end

    empty_packets = zeros(Float32, Spatial.PACKET_WIDTH, 0)
    empty_delta = similar(empty_packets)
    Spatial.candidate_after_packets!(
        empty_packets,
        empty_delta,
        workspace,
        bank,
        base_grid,
        after,
        UInt16[],
    )

    # Warm before measuring the production calls.
    Spatial.base_packet_grid!(base_grid, workspace, bank, board)
    Spatial.candidate_after_packets!(
        candidate, delta, workspace, bank, base_grid, after, positions,
    )
    @test @allocated(Spatial.base_packet_grid!(
        base_grid, workspace, bank, board,
    )) == 0
    @test @allocated(Spatial.candidate_after_packets!(
        candidate, delta, workspace, bank, base_grid, after, positions,
    )) == 0
end

function sparse_value(gradient, row, lane)
    for slot in 1:Bank.active_gradient_count(gradient)
        Bank.active_gradient_row(gradient, slot) == row || continue
        return gradient.values[lane, slot]
    end
    return 0.0f0
end

@testset "packet delta reverse is exact and sparse" begin
    rng = MersenneTwister(0x51fdf)
    bank = Bank.ProgramBank(0x51fdf)
    workspace = Spatial.SpatialPacketWorkspace()
    board = zeros(UInt8, Spatial.BOARD_ROWS, Spatial.BOARD_COLUMNS)
    board[17:24, 3] .= 0x01
    board[20:24, 8] .= 0x01
    after = copy(board)
    after[16, 3] = 0x01
    after[19, 8] = 0x01
    positions = UInt16[64, 184]

    base_grid = zeros(Float32, Spatial.PACKET_WIDTH, Spatial.PACKET_COUNT)
    candidate = zeros(Float32, Spatial.PACKET_WIDTH, length(positions))
    delta = similar(candidate)
    Spatial.base_packet_grid!(base_grid, workspace, bank, board)
    Spatial.candidate_after_packets!(
        candidate, delta, workspace, bank, base_grid, after, positions,
    )

    base_direct_bar = 0.01f0 .* randn(
        rng, Float32, Spatial.PACKET_WIDTH, Spatial.PACKET_COUNT,
    )
    candidate_bar = 0.03f0 .* randn(
        rng, Float32, Spatial.PACKET_WIDTH, length(positions),
    )
    delta_bar = 0.05f0 .* randn(
        rng, Float32, Spatial.PACKET_WIDTH, length(positions),
    )
    base_grid_bar = copy(base_direct_bar)
    gradient = Bank.SparseProgramGradient(bank, 4096)
    Spatial.candidate_after_packets_pullback!(
        gradient,
        base_grid_bar,
        workspace,
        bank,
        after,
        positions,
        candidate_bar,
        delta_bar,
    )
    @inbounds for index in eachindex(positions), lane in 1:Spatial.PACKET_WIDTH
        column = Spatial.packet_column(positions[index], Spatial.AFTER_PLANE)
        @test base_grid_bar[lane, column] ==
              base_direct_bar[lane, column] - delta_bar[lane, index]
    end
    Spatial.base_packet_grid_pullback!(
        gradient, workspace, bank, board, base_grid_bar,
    )

    function objective!()
        Spatial.base_packet_grid!(base_grid, workspace, bank, board)
        Spatial.candidate_after_packets!(
            candidate, delta, workspace, bank, base_grid, after, positions,
        )
        return dot(base_grid, base_direct_bar) +
               dot(candidate, candidate_bar) + dot(delta, delta_bar)
    end

    lane = 7
    candidate_rows = Spatial.spatial_program_rows(
        after, positions[1], Spatial.AFTER_PLANE,
    )
    base_rows = Spatial.spatial_program_rows(
        board, positions[1], Spatial.AFTER_PLANE,
    )
    for row in (Int(candidate_rows.rows[4]), Int(base_rows.rows[4]))
        original = bank.payload[lane, row]
        epsilon = 1.0f-3
        bank.payload[lane, row] = original + epsilon
        plus = objective!()
        bank.payload[lane, row] = original - epsilon
        minus = objective!()
        bank.payload[lane, row] = original
        finite_difference = (plus - minus) / (2.0f0 * epsilon)
        @test isapprox(
            sparse_value(gradient, row, lane),
            finite_difference;
            rtol=4.0f-3,
            atol=5.0f-4,
        )
    end

    # Reverse kernels own only caller-provided scratch and sparse slots.
    Bank.reset_sparse_gradient!(gradient)
    fill!(base_grid_bar, 0.0f0)
    Spatial.candidate_after_packets_pullback!(
        gradient,
        base_grid_bar,
        workspace,
        bank,
        after,
        positions,
        candidate_bar,
        delta_bar,
    )
    Spatial.base_packet_grid_pullback!(
        gradient, workspace, bank, board, base_grid_bar,
    )
    Bank.reset_sparse_gradient!(gradient)
    fill!(base_grid_bar, 0.0f0)
    @test @allocated(Spatial.candidate_after_packets_pullback!(
        gradient,
        base_grid_bar,
        workspace,
        bank,
        after,
        positions,
        candidate_bar,
        delta_bar,
    )) == 0
    @test @allocated(Spatial.base_packet_grid_pullback!(
        gradient, workspace, bank, board, base_grid_bar,
    )) == 0
    @test 0 < Bank.active_gradient_count(gradient) < Bank.ROW_COUNT
end

