using Test

module ArchitectureTestHarness
include(joinpath(@__DIR__, "Architecture.jl"))
end

const Contract = ArchitectureTestHarness.Architecture

@testset "canonical Reduced Hay architecture" begin
    @test Contract.BLOCK_COUNT == 30
    @test Contract.SPATIAL_POSITIONS_PER_BLOCK == 8
    @test Contract.LANES_PER_POSITION == 2
    @test Contract.CELLS_PER_BLOCK == 16
    @test Contract.TOTAL_CELLS == 480
    @test Contract.TOTAL_CELLS ==
          Contract.BLOCK_COUNT * Contract.CELLS_PER_BLOCK

    @test Contract.CYCLES == 10
    @test Contract.FANOUT == 48
    @test Contract.OUTPUT_POPULATION_PER_CHANNEL == 4
    @test Contract.Q_OUTPUT_CELLS_PER_BIT == 1
    @test Contract.OUTPUT_CELL_COUNT == 116
    @test Contract.Q_OUTPUT_CELL_COUNT == 32
    @test Contract.AUX_OUTPUT_CELL_COUNT == 84
    @test Contract.NUMERIC_OPERAND_BITS == 32
    @test Contract.OUTPUT_FANOUT == 16
    @test Contract.OUTPUT_FANOUT < Contract.OUTPUT_CELL_COUNT

    @test Contract.RAIL_COUNT == 1_298
    @test Contract.OUTPUT_COUNT == 22
    @test (Contract.BOARD_ROWS, Contract.BOARD_COLUMNS) == (24, 10)

    @test Contract.STATE_BATCH == 8
    @test Contract.CANDIDATE_WIDTH == 80
    @test Contract.CAPACITY == 640
    @test Contract.CAPACITY ==
          Contract.STATE_BATCH * Contract.CANDIDATE_WIDTH

    exported = Set(names(Contract))
    expected = Set((
        :Architecture,
        :BLOCK_COUNT,
        :SPATIAL_POSITIONS_PER_BLOCK,
        :LANES_PER_POSITION,
        :CELLS_PER_BLOCK,
        :TOTAL_CELLS,
        :CYCLES,
        :FANOUT,
        :OUTPUT_POPULATION_PER_CHANNEL,
        :Q_OUTPUT_CELLS_PER_BIT,
        :OUTPUT_CELL_COUNT,
        :Q_OUTPUT_CELL_COUNT,
        :AUX_OUTPUT_CELL_COUNT,
        :NUMERIC_OPERAND_BITS,
        :OUTPUT_FANOUT,
        :RAIL_COUNT,
        :OUTPUT_COUNT,
        :BOARD_ROWS,
        :BOARD_COLUMNS,
        :STATE_BATCH,
        :CANDIDATE_WIDTH,
        :CAPACITY,
    ))
    @test exported == expected
end
