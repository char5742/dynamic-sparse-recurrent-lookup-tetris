using Test

module RootModuleTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = RootModuleTestHarness.ReducedHayCPU

@testset "canonical Reduced-Hay root ownership" begin
    exported = Set(names(Root))

    canonical_modules = (
        :ActiveApicalCell,
        :CanonicalTetrisInput,
        :DendriticAxonPacket,
        :OrderedMultiscaleTopology,
        :CanonicalSpatialDrive,
        :CanonicalExperimentData,
        :DendriticOutputPopulation,
        :CanonicalListNet,
        :CanonicalEventArena,
        :CanonicalOptimizer,
        :CanonicalLocalLearning,
        :CanonicalPlasticity,
        :CanonicalDendriticGraph,
        :CanonicalBarrierless,
        :CanonicalValidation,
        :CanonicalCheckpoint,
        :CanonicalExactOracle,
        :CanonicalTraining,
    )
    @test all(name -> name in exported, canonical_modules)
    @test all(name -> getfield(Root, name) isa Module, canonical_modules)

    direct_graph = (
        :GraphConfig,
        :CanonicalModel,
        :ModelState,
        :ModelWorker,
        :initialize_model,
        :initialize_state,
        :initialize_worker,
        :stored_parameter_count,
    )
    direct_training = (
        :CanonicalTrainingConfig,
        :CanonicalTrainer,
        :TrainingUpdateResult,
        :with_training_team,
        :train_update!,
        :mechanism_counts,
        :update_count,
    )
    @test all(name -> name in exported, direct_graph)
    @test all(name -> name in exported, direct_training)

    for name in direct_graph
        @test getfield(Root, name) ===
            getfield(Root.CanonicalDendriticGraph, name)
    end
    for name in direct_training
        @test getfield(Root, name) === getfield(Root.CanonicalTraining, name)
    end
    @test Root.train_update! === Root.CanonicalTraining.train_update!

    model = Root.initialize_model()
    state = Root.initialize_state(model)
    worker = Root.initialize_worker(model)
    @test model isa Root.CanonicalModel
    @test state isa Root.ModelState
    @test worker isa Root.ModelWorker
    @test Root.stored_parameter_count(model) > 0
    @test Root.CanonicalDendriticGraph.TOTAL_NODE_COUNT == 1_458
    @test Root.DendriticOutputPopulation.OUTPUT_DIM == 22
    @test Root.DendriticAxonPacket.PACKET_DIM == 12
end

@testset "root include closure is canonical-only" begin
    source = read(joinpath(@__DIR__, "ReducedHayCPU.jl"), String)
    included = [match.captures[1] for match in eachmatch(
        r"include\(joinpath\(@__DIR__, \"([^\"]+)\"\)\)",
        source,
    )]
    @test included == [
        "ActiveApicalCell.jl",
        "CanonicalTetrisInput.jl",
        "TetrisRankingBatch.jl",
        "DendriticAxonPacket.jl",
        "OrderedMultiscaleTopology.jl",
        "CanonicalSpatialDrive.jl",
        "CanonicalExperimentData.jl",
        "DendriticOutputPopulation.jl",
        "CanonicalListNet.jl",
        "CanonicalEventArena.jl",
        "BarrierlessScheduler.jl",
        "CanonicalOptimizer.jl",
        "CanonicalLocalLearning.jl",
        "CanonicalPlasticity.jl",
        "CanonicalDendriticGraph.jl",
        "CanonicalBarrierless.jl",
        "CanonicalValidation.jl",
        "CanonicalCheckpoint.jl",
        "CanonicalExactOracle.jl",
        "CanonicalTraining.jl",
    ]

    # Neutral implementation dependencies are reachable by their real module
    # names but are not promoted into the root's public API.
    @test isdefined(Root, :TetrisRankingBatch)
    @test isdefined(Root, :BarrierlessScheduler)
    @test :TetrisRankingBatch ∉ names(Root)
    @test :BarrierlessScheduler ∉ names(Root)
end

@testset "retired modules and compatibility aliases are absent" begin
    exported = Set(names(Root))
    retired = (
        # Previous program/relation/motif production root.
        :CandidateDeltaInput,
        :DendriticProgramBank,
        :SpatialProgramPackets,
        :HighDimensionalCellPacket,
        :TypedDendriticAfferents,
        :DendriticRelationTopology,
        :DendriticMotifTopology,
        :TypedRelationCellBank,
        :TypedOutputCellBank,
        :TypedRelationContext,
        :StructuredMotifReadout,
        :ContinuousDendriticReadout,
        :CandidateDeltaRelationGraph,
        :RelationGraphOptimizer,
        :RelationGraphTraining,
        :RelationGraphBarrierless,
        :ReducedHayCPUSampler,
        :RelationGraphCheckpoint,

        # Earlier forest/point/dense-output and routed-workspace roots.
        :CompactDendriticNode,
        :DendriticDeltaForestTopology,
        :DendriticDeltaForest,
        :DendriticForestOutput,
        :CandidateDeltaDendriticGraph,
        :CandidateDeltaDendriticTraining,
        :CandidateDeltaDendriticBarrierless,
        :CandidateDeltaDendriticOptimizer,
        :ReducedHayCPUCheckpoint,
        :Architecture,
        :Float32NumericCore,
        :ReducedHayCPUNativeModel,
        :ControlPlane,
        :RoutingScratch,
        :RouteLoadSnapshot,
        :ROUTE_DETERMINISTIC,
        :ROUTE_STOCHASTIC,
        :route_kind,
        :route_temperature,
        :route_exploration,

        # Compatibility aliases are forbidden; use the actual module/type.
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
        :CanonicalSampler,
        :CanonicalLocalLearner,
        :CanonicalNode,
        :CanonicalForest,
        :CanonicalOutput,
        :TrainingConfig,
        :Trainer,
        :ExactOracle,
        :ExactBatchTrainer,
        :BarrierlessExactExecutor,
        :BarrierlessExactSession,
    )
    for name in retired
        @test !isdefined(Root, name)
        @test name ∉ exported
    end
end

@testset "low-level operations stay namespace-scoped" begin
    exported = Set(names(Root))
    scoped = (
        :ModelParameters,
        :ModelCache,
        :ModelGradient,
        :ModelLocalLearner,
        :initialize_gradient,
        :forward_candidate!,
        :conditional_reverse_candidate!,
        :local_replay_candidate!,
        :parameter_components,
        :gradient_components,
        :ParameterRegistry,
        :apply_adamw!,
        :CanonicalExecutor,
        :CanonicalSession,
        :serial_reference_update!,
        :AbstractExactOracleAdapter,
        :conditional_reverse!,
        :DendriticTrainingAdapter,
        :MechanismHooks,
        :hooks,
        :AuxiliaryLossConfig,
        :save_checkpoint,
        :load_checkpoint,
        :restore_checkpoint!,
    )
    for name in scoped
        @test !isdefined(Root, name)
        @test name ∉ exported
    end
end
