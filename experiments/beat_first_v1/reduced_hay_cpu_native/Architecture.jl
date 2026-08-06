module Architecture

"""
Fixed production architecture for the CPU-native Reduced Hay experiment.

This module is the canonical owner of cross-component dimensions.  Cell-local
equation dimensions remain owned by `ActiveApicalCell`; every other module must
import these values instead of defining independent defaults.
"""

export BLOCK_COUNT,
    SPATIAL_POSITIONS_PER_BLOCK,
    LANES_PER_POSITION,
    CELLS_PER_BLOCK,
    TOTAL_CELLS,
    CYCLES,
    FANOUT,
    OUTPUT_POPULATION_PER_CHANNEL,
    Q_OUTPUT_CELLS_PER_BIT,
    Q_OUTPUT_CELL_COUNT,
    AUX_OUTPUT_CELL_COUNT,
    NUMERIC_OPERAND_BITS,
    OUTPUT_CELL_COUNT,
    OUTPUT_FANOUT,
    RAIL_COUNT,
    OUTPUT_COUNT,
    BOARD_ROWS,
    BOARD_COLUMNS,
    STATE_BATCH,
    CANDIDATE_WIDTH,
    CAPACITY

const BLOCK_COUNT = 30
# Each of the 240 board positions owns two independent high-dimensional Hay
# cells.  The first lane preserves the exact sensory identity; the second is
# an independently parameterized associative lane.  They share a spatial
# block but never share internal state or parameters.
const SPATIAL_POSITIONS_PER_BLOCK = 8
const LANES_PER_POSITION = 2
const CELLS_PER_BLOCK = SPATIAL_POSITIONS_PER_BLOCK * LANES_PER_POSITION
const TOTAL_CELLS = BLOCK_COUNT * CELLS_PER_BLOCK

# Every block advances on every cycle. Dynamic sparsity is owned by spikes and
# the fixed-fanout event graph, not by a central controller.
const CYCLES = 10
const FANOUT = 48

const RAIL_COUNT = 1_298
const OUTPUT_COUNT = 22
# The answer is emitted by hard-spiking Reduced Hay cells, not by a dense
# observation matrix.  Each recurrent source reaches only a seed-fixed subset
# of output cells; the source-major output graph therefore remains physically
# sparse as well as the recurrent graph.
const OUTPUT_POPULATION_PER_CHANNEL = 4
# One clocked high-dimensional cell emits one deterministic hard bit.
const Q_OUTPUT_CELLS_PER_BIT = 1
const NUMERIC_OPERAND_BITS = 32
const Q_OUTPUT_CELL_COUNT =
    NUMERIC_OPERAND_BITS * Q_OUTPUT_CELLS_PER_BIT
const AUX_OUTPUT_CELL_COUNT =
    (OUTPUT_COUNT - 1) * OUTPUT_POPULATION_PER_CHANNEL
const OUTPUT_CELL_COUNT = Q_OUTPUT_CELL_COUNT + AUX_OUTPUT_CELL_COUNT
const OUTPUT_FANOUT = 32
const BOARD_ROWS = 24
const BOARD_COLUMNS = 10

const STATE_BATCH = 8
const CANDIDATE_WIDTH = 80
const CAPACITY = STATE_BATCH * CANDIDATE_WIDTH

@assert TOTAL_CELLS == 480
@assert BOARD_ROWS * BOARD_COLUMNS * LANES_PER_POSITION == TOTAL_CELLS
@assert CAPACITY == 640
@assert OUTPUT_FANOUT < OUTPUT_CELL_COUNT
@assert Q_OUTPUT_CELL_COUNT ==
    NUMERIC_OPERAND_BITS * Q_OUTPUT_CELLS_PER_BIT

end # module Architecture
