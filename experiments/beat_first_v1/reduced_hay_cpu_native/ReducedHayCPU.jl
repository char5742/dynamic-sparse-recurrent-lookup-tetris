module ReducedHayCPU

"""
The sole production root for the route-free, ordered multiscale Reduced-Hay
graph.

Historical program-bank, relation/motif, dense-output, routed-workspace and
exact-training files remain beside this root only as research records.  They
are deliberately outside this include closure and cannot enter the canonical
forward, local-learning, plasticity or optimizer path.
"""

# Numerical cell, target-free input and immutable typed anatomy.
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
include(joinpath(@__DIR__, "CanonicalExperimentData.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
include(joinpath(@__DIR__, "CanonicalListNet.jl"))

# Fixed-memory event execution and the sole optimizer/local-plasticity stack.
include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalPlasticity.jl"))

# One concrete graph, one barrierless executor and one production trainer.
include(joinpath(@__DIR__, "CanonicalDendriticGraph.jl"))
include(joinpath(@__DIR__, "CanonicalBarrierless.jl"))
include(joinpath(@__DIR__, "CanonicalValidation.jl"))
include(joinpath(@__DIR__, "CanonicalCheckpoint.jl"))
include(joinpath(@__DIR__, "CanonicalExactOracle.jl"))
include(joinpath(@__DIR__, "CanonicalTraining.jl"))

# The root exposes canonical subsystem ownership explicitly.  Low-level graph
# traversal, gradient, optimizer, barrierless and exact-oracle operations stay
# namespace-scoped so that the only root-level learning operation is the
# production CanonicalTraining.train_update!.
using .CanonicalDendriticGraph: GraphConfig,
    CanonicalModel,
    ModelState,
    ModelWorker,
    initialize_model,
    initialize_state,
    initialize_worker,
    stored_parameter_count
using .CanonicalTraining: CanonicalTrainingConfig,
    CanonicalTrainer,
    TrainingUpdateResult,
    with_training_team,
    train_update!,
    mechanism_counts,
    update_count

export ActiveApicalCell,
    CanonicalTetrisInput,
    DendriticAxonPacket,
    OrderedMultiscaleTopology,
    CanonicalSpatialDrive,
    CanonicalExperimentData,
    DendriticOutputPopulation,
    CanonicalListNet,
    CanonicalEventArena,
    CanonicalOptimizer,
    CanonicalLocalLearning,
    CanonicalPlasticity,
    CanonicalDendriticGraph,
    CanonicalBarrierless,
    CanonicalValidation,
    CanonicalCheckpoint,
    CanonicalExactOracle,
    CanonicalTraining,
    GraphConfig,
    CanonicalModel,
    ModelState,
    ModelWorker,
    initialize_model,
    initialize_state,
    initialize_worker,
    stored_parameter_count,
    CanonicalTrainingConfig,
    CanonicalTrainer,
    TrainingUpdateResult,
    with_training_team,
    train_update!,
    mechanism_counts,
    update_count

"""Fail closed if independently owned canonical dimensions drift apart."""
function _assert_canonical_contract()
    input = CanonicalTetrisInput
    ranking = TetrisRankingBatch
    axon = DendriticAxonPacket
    topology = OrderedMultiscaleTopology
    spatial = CanonicalSpatialDrive
    data = CanonicalExperimentData
    output = DendriticOutputPopulation
    events = CanonicalEventArena
    graph = CanonicalDendriticGraph

    checks = (
        input.BOARD_ROWS == ranking.BOARD_ROWS == topology.ROW_COUNT,
        input.BOARD_COLUMNS == ranking.BOARD_COLUMNS == topology.COLUMN_COUNT,
        input.BOARD_CELLS == topology.SPATIAL_COUNT_PER_PLANE,
        axon.PACKET_DIM == topology.FULL_PACKET_WIDTH == output.EVIDENCE_DIM,
        ActiveApicalCell.N_BASAL == spatial.NEIGHBOR_COUNT == output.MAX_EVIDENCE,
        topology.NODE_COUNT == graph.TOTAL_NODE_COUNT,
        graph.CORE_NODE_COUNT + output.OUTPUT_CELLS == topology.NODE_COUNT,
        ranking.OUTPUT_DIM == output.OUTPUT_DIM == topology.OUTPUT_COUNT,
        spatial.PHASE_COUNT == length(instances(graph.TransitionPhase)),
        GraphConfig().max_event_waves == events.CANONICAL_MAX_WAVES,
        data.CANDIDATE_WIDTH == 80,
    )
    all(checks) || error("canonical Reduced-Hay root contract drifted")
    return nothing
end

_assert_canonical_contract()

end # module ReducedHayCPU
