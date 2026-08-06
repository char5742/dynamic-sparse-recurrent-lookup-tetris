using Test

module RootModuleTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = RootModuleTestHarness.ReducedHayCPU

@testset "typed relation graph is the only canonical root" begin
    exported = Set(names(Root))
    required = (
        :CanonicalRanking,
        :CanonicalInput,
        :CanonicalCell,
        :CanonicalProgramBank,
        :CanonicalSpatialPackets,
        :CanonicalPacket,
        :CanonicalAfferents,
        :CanonicalTopology,
        :CanonicalMotifTopology,
        :CanonicalRelationCells,
        :CanonicalOutputCells,
        :CanonicalContext,
        :CanonicalMotifReadout,
        :CanonicalModel,
        :CanonicalOptimizer,
        :CanonicalTraining,
        :CanonicalBarrierless,
        :CanonicalSampler,
        :CanonicalCheckpoint,
        :initialize_model,
        :stored_parameter_count,
        :BarrierlessRelationGraphTrainer,
        :BarrierlessRelationGraphSession,
        :train_update!,
        :OptimizerConfig,
        :apply_adamw!,
    )
    @test all(name -> name in exported, required)

    @test Root.CanonicalModel === Root.CandidateDeltaRelationGraph
    @test Root.CanonicalTraining === Root.RelationGraphTraining
    @test Root.CanonicalBarrierless === Root.RelationGraphBarrierless
    @test Root.CanonicalOptimizer === Root.RelationGraphOptimizer
    @test Root.CanonicalSampler === Root.ReducedHayCPUSampler
    @test Root.CanonicalCheckpoint === Root.RelationGraphCheckpoint
    @test Root.train_update! === Root.CanonicalBarrierless.train_update!
    @test Root.train_update! !== Root.CanonicalTraining.train_update!
    @test :RelationGraphTrainer ∉ exported
    @test Root.CanonicalCheckpoint.CHECKPOINT_SCHEMA == UInt32(2)

    # Sampler/checkpoint stay namespace-scoped: the production entrypoint must
    # name their ownership instead of accidentally binding an obsolete API.
    for scoped in (
        :DeterministicEpochSampler,
        :next_batch!,
        :save_checkpoint,
        :load_checkpoint,
        :restore_checkpoint!,
    )
        @test scoped ∉ exported
    end

    parameters = Root.initialize_model()
    @test parameters isa Root.ModelParameters
    @test Root.stored_parameter_count(parameters) == 3_362_748
    @test Root.CanonicalRanking.INPUT_RAILS == 1_298
    @test Root.CanonicalRanking.OUTPUT_DIM == 22
    @test Root.CanonicalInput.BOARD_CELLS == 240
    @test Root.CanonicalSpatialPackets.PACKET_COUNT == 480
    @test Root.CanonicalModel.PROGRAM_PACKET_DIM == 16
    @test Root.CanonicalPacket.PACKET_DIM == 47
    @test Root.CanonicalModel.CELL_PACKET_DIM == 47
    @test Root.CanonicalTopology.RELATION_COUNT == 48
    @test Root.CanonicalMotifTopology.MOTIF_COUNT == 48
    @test Root.CanonicalMotifReadout.SOURCE_COUNT == 48
    @test Root.CanonicalRelationCells.RELATION_CELLS == 48
    @test Root.CanonicalOutputCells.OUTPUT_CELLS == 22

    for retired in (
        :CompactDendriticNode,
        :DendriticDeltaForestTopology,
        :DendriticDeltaForest,
        :DendriticForestOutput,
        :CandidateDeltaDendriticGraph,
        :CandidateDeltaDendriticTraining,
        :CandidateDeltaDendriticBarrierless,
        :CandidateDeltaDendriticOptimizer,
        :CanonicalNode,
        :CanonicalForest,
        :CanonicalOutput,
        :ExactBatchTrainer,
        :BarrierlessExactExecutor,
        :BarrierlessExactSession,
        :ReducedHayCPUCheckpoint,
        :Architecture,
        :Float32NumericCore,
        :ReducedHayCPUNativeModel,
        :CanonicalLocalLearner,
        :ControlPlane,
        :RoutingScratch,
        :RouteLoadSnapshot,
        :ROUTE_DETERMINISTIC,
        :ROUTE_STOCHASTIC,
        :CanonicalTrainer,
        :ExactOracle,
    )
        @test !isdefined(Root, retired)
        @test retired ∉ exported
    end
end
