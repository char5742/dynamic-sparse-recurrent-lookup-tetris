module ReducedHayCPU

"""
The sole production root for the typed candidate-delta relation/motif graph.

Historical point-SNN, routed workspace, dendritic-forest, exact-oracle, and
local-credit files may remain beside this root as research records.  None is
included here, so it cannot enter the canonical forward or optimizer path by
accident.
"""

# Model-neutral Tetris ranking and candidate-delta input contracts.
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))

# High-dimensional Reduced-Hay cell and typed anatomical primitives.
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "SpatialProgramPackets.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
include(joinpath(@__DIR__, "TypedDendriticAfferents.jl"))
include(joinpath(@__DIR__, "DendriticRelationTopology.jl"))
include(joinpath(@__DIR__, "DendriticMotifTopology.jl"))
include(joinpath(@__DIR__, "TypedRelationCellBank.jl"))
include(joinpath(@__DIR__, "TypedOutputCellBank.jl"))
include(joinpath(@__DIR__, "TypedRelationContext.jl"))
include(joinpath(@__DIR__, "StructuredMotifReadout.jl"))

# One canonical relation-to-motif model, optimizer, serial oracle, and
# production executor.
include(joinpath(@__DIR__, "CandidateDeltaRelationGraph.jl"))
include(joinpath(@__DIR__, "RelationGraphOptimizer.jl"))
include(joinpath(@__DIR__, "RelationGraphTraining.jl"))
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "RelationGraphBarrierless.jl"))
include(joinpath(@__DIR__, "Sampler.jl"))
include(joinpath(@__DIR__, "Checkpoint.jl"))

const CanonicalRanking = TetrisRankingBatch
const CanonicalInput = CandidateDeltaInput
const CanonicalCell = ActiveApicalCell
const CanonicalProgramBank = DendriticProgramBank
const CanonicalSpatialPackets = SpatialProgramPackets
const CanonicalPacket = HighDimensionalCellPacket
const CanonicalAfferents = TypedDendriticAfferents
const CanonicalTopology = DendriticRelationTopology
const CanonicalMotifTopology = DendriticMotifTopology
const CanonicalRelationCells = TypedRelationCellBank
const CanonicalOutputCells = TypedOutputCellBank
const CanonicalContext = TypedRelationContext
const CanonicalMotifReadout = StructuredMotifReadout
const CanonicalModel = CandidateDeltaRelationGraph
const CanonicalOptimizer = RelationGraphOptimizer

# The serial implementation is an explicit comparison oracle.  It is exposed
# only as a module namespace; no serial trainer or serial train_update! is
# lifted into the production root.
const CanonicalTraining = RelationGraphTraining
const CanonicalBarrierless = RelationGraphBarrierless
const CanonicalSampler = ReducedHayCPUSampler
const CanonicalCheckpoint = RelationGraphCheckpoint

using .CandidateDeltaRelationGraph: ModelCache,
    ModelForwardStats,
    ModelGradient,
    ModelParameters,
    ModelState,
    ModelWorker,
    initialize_model,
    stored_parameter_count
using .RelationGraphOptimizer: AdamWState,
    AdamWStepCounters,
    AdamWStepStats,
    OptimizerConfig,
    apply_adamw!,
    gradient_norm
using .RelationGraphBarrierless: BarrierlessRelationGraphSession,
    BarrierlessRelationGraphTrainer,
    forward_batch!,
    latest_gradient,
    latest_loss,
    run_trainer_team!,
    scheduler_report,
    train_update!

export CanonicalRanking,
    CanonicalInput,
    CanonicalCell,
    CanonicalProgramBank,
    CanonicalSpatialPackets,
    CanonicalPacket,
    CanonicalAfferents,
    CanonicalTopology,
    CanonicalMotifTopology,
    CanonicalRelationCells,
    CanonicalOutputCells,
    CanonicalContext,
    CanonicalMotifReadout,
    CanonicalModel,
    CanonicalOptimizer,
    CanonicalTraining,
    CanonicalBarrierless,
    CanonicalSampler,
    CanonicalCheckpoint,
    ModelCache,
    ModelForwardStats,
    ModelGradient,
    ModelParameters,
    ModelState,
    ModelWorker,
    initialize_model,
    stored_parameter_count,
    AdamWState,
    AdamWStepCounters,
    AdamWStepStats,
    OptimizerConfig,
    apply_adamw!,
    gradient_norm,
    BarrierlessRelationGraphSession,
    BarrierlessRelationGraphTrainer,
    forward_batch!,
    latest_gradient,
    latest_loss,
    run_trainer_team!,
    scheduler_report,
    train_update!

"""Fail closed if the typed relation modules disagree on public dimensions."""
function _assert_canonical_contract()
    checks = (
        CanonicalRanking.INPUT_RAILS == CanonicalInput.INPUT_RAILS,
        CanonicalRanking.BOARD_ROWS == CanonicalInput.BOARD_ROWS,
        CanonicalRanking.BOARD_COLUMNS == CanonicalInput.BOARD_COLUMNS,
        CanonicalRanking.OUTPUT_DIM == CanonicalOutputCells.OUTPUT_CELLS,
        CanonicalInput.BOARD_CELLS == CanonicalSpatialPackets.POSITION_COUNT,
        CanonicalSpatialPackets.PACKET_COUNT == CanonicalTopology.SOURCE_COUNT,
        CanonicalModel.PROGRAM_PACKET_DIM ==
            CanonicalSpatialPackets.PACKET_WIDTH,
        CanonicalModel.CELL_PACKET_DIM == CanonicalPacket.PACKET_DIM,
        CanonicalTopology.RELATION_COUNT ==
            CanonicalRelationCells.RELATION_CELLS,
        CanonicalMotifTopology.RELATION_SOURCE_COUNT ==
            CanonicalRelationCells.RELATION_CELLS,
        CanonicalMotifTopology.MOTIF_COUNT ==
            CanonicalRelationCells.RELATION_CELLS,
        CanonicalMotifReadout.SOURCE_COUNT ==
            CanonicalMotifTopology.MOTIF_COUNT,
        CanonicalTopology.BASAL_COMPARTMENT_COUNT == CanonicalCell.N_BASAL,
        CanonicalModel.PROGRAM_SOURCES == CanonicalSpatialPackets.PACKET_COUNT,
        CanonicalModel.RELATION_CELLS ==
            CanonicalRelationCells.RELATION_CELLS,
        CanonicalModel.OUTPUT_CELLS == CanonicalOutputCells.OUTPUT_CELLS,
        CanonicalProgramBank.PAYLOAD_WIDTH ==
            CanonicalModel.PROGRAM_PACKET_DIM,
    )
    all(checks) || error("typed candidate-delta relation contract drifted")
    return nothing
end

_assert_canonical_contract()

end # module ReducedHayCPU
