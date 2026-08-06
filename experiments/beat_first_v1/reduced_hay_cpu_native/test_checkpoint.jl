using Test
using Serialization

module RelationGraphCheckpointTestHarness
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
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
include(joinpath(@__DIR__, "CandidateDeltaRelationGraph.jl"))
include(joinpath(@__DIR__, "RelationGraphOptimizer.jl"))
include(joinpath(@__DIR__, "Sampler.jl"))
include(joinpath(@__DIR__, "Checkpoint.jl"))
end

const Harness = RelationGraphCheckpointTestHarness
const Checkpoint = Harness.RelationGraphCheckpoint
const Model = Harness.CandidateDeltaRelationGraph
const Optimizer = Harness.RelationGraphOptimizer
const Sampler = Harness.ReducedHayCPUSampler

struct FixtureTrainer
    parameters::Model.ModelParameters
    cache::Model.ModelCache
    optimizer_state::Optimizer.AdamWState
    optimizer_config::Optimizer.OptimizerConfig
end

function fixture()
    parameters = Model.initialize_model()
    config = Optimizer.OptimizerConfig(learning_rate=0.003)
    return FixtureTrainer(
        parameters,
        Model.ModelCache(parameters),
        Optimizer.AdamWState(parameters),
        config,
    )
end

function parameter_arrays(parameters)
    return (
        parameters.program_bank.payload,
        parameters.leaf_relation.raw_conductance,
        parameters.relation.cell_raw,
        parameters.relation_motif.raw_conductance,
        parameters.motif.cell_raw,
        parameters.context.common_relation.raw_conductance,
        parameters.context.common_output.raw_conductance,
        parameters.context.aux_relation.raw_conductance,
        parameters.placement_relation.raw_conductance,
        parameters.motif_readout.source_gain_raw,
        parameters.output.cell_raw,
        parameters.output.readout_weight,
        parameters.output.bias,
    )
end

function seed_optimizer_history!(trainer, update::Int)
    state = trainer.optimizer_state
    state.steps.total = update
    state.steps.program_batches = 2
    state.steps.program_rows = 3
    state.program_step_by_row[7] = UInt32(2)
    state.program_step_by_row[31] = UInt32(1)
    state.program_first[1, 7] = 0.125f0
    state.program_second[1, 7] = 0.25f0
    for name in fieldnames(Optimizer.DenseMoments)
        first = getfield(state.first, name)
        second = getfield(state.second, name)
        first[begin] = 0.01f0 * length(first)
        second[begin] = 0.001f0 * length(second)
        setfield!(state.steps, name, update)
    end
    trainer.parameters.program_bank.payload[1, 7] = 0.75f0
    trainer.parameters.relation_motif.raw_conductance[1] = 0.625f0
    trainer.parameters.motif.cell_raw[1] = -0.25f0
    trainer.parameters.motif_readout.source_gain_raw[1] = 0.5f0
    trainer.parameters.output.bias[1] = -0.375f0
    Model.refresh_cache!(trainer.cache, trainer.parameters)
    return trainer
end

function optimizer_equal(left, right)
    for name in fieldnames(Optimizer.DenseMoments)
        getfield(left.first, name) == getfield(right.first, name) || return false
        getfield(left.second, name) == getfield(right.second, name) || return false
    end
    return left.program_first == right.program_first &&
           left.program_second == right.program_second &&
           left.program_step_by_row == right.program_step_by_row &&
           all(
               getfield(left.steps, name) == getfield(right.steps, name)
               for name in fieldnames(Optimizer.AdamWStepCounters)
           )
end

const RUN_CONTRACT = (
    dataset_manifest_sha256=repeat("1", 64),
    training_split_sha256=repeat("2", 64),
    development_panel_sha256=repeat("3", 64),
    development_rows=(11, 29, 47, 61),
    state_batch=8,
    width=80,
    workers=20,
    candidate_chunk=4,
    schedule=(learning_rate=0.003f0, checkpoint_interval=1_000),
)

function advance_to_update!(sampler, update::Int)
    emitted = zeros(Int, update * RUN_CONTRACT.state_batch)
    Sampler.next_batch!(emitted, sampler)
    return emitted
end

@testset "canonical relation checkpoint roundtrip" begin
    mktempdir() do directory
        update = 17
        source_rows = collect(101:137)
        trainer = seed_optimizer_history!(fixture(), update)
        sampler = Sampler.DeterministicEpochSampler(source_rows, UInt64(0x707))
        advance_to_update!(sampler, update)
        expected_sampler = Sampler.sampler_snapshot(sampler)
        expected_parameters = deepcopy(trainer.parameters)
        expected_optimizer = deepcopy(trainer.optimizer_state)

        path = joinpath(directory, "relation-graph.jls")
        saved = Checkpoint.save_checkpoint(
            path,
            trainer,
            sampler;
            update=update,
            run_contract=RUN_CONTRACT,
        )
        @test saved == abspath(path)
        @test isfile(saved)
        snapshot = Checkpoint.load_checkpoint(saved)
        @test snapshot.schema == Checkpoint.CHECKPOINT_SCHEMA
        @test snapshot.schema == UInt32(2)
        @test snapshot.format ==
            "candidate-delta-relation-motif-graph-exact-v2"
        @test snapshot.update == update
        @test snapshot.run_contract == RUN_CONTRACT
        @test snapshot.model_fingerprint ==
            Checkpoint.model_fingerprint(expected_parameters)
        @test snapshot.optimizer_config_fingerprint ==
            Checkpoint.optimizer_config_fingerprint(trainer.optimizer_config)
        @test snapshot.source_closure ==
            Checkpoint.canonical_source_closure()
        source_paths = Set(file.path for file in snapshot.source_closure.files)
        @test "DendriticMotifTopology.jl" in source_paths
        @test "StructuredMotifReadout.jl" in source_paths

        for array in parameter_arrays(trainer.parameters)
            fill!(array, 0.0f0)
        end
        for name in fieldnames(Optimizer.DenseMoments)
            fill!(getfield(trainer.optimizer_state.first, name), 0.0f0)
            fill!(getfield(trainer.optimizer_state.second, name), 0.0f0)
        end
        fill!(trainer.optimizer_state.program_first, 0.0f0)
        fill!(trainer.optimizer_state.program_second, 0.0f0)
        fill!(trainer.optimizer_state.program_step_by_row, UInt32(0))
        for name in fieldnames(Optimizer.AdamWStepCounters)
            setfield!(trainer.optimizer_state.steps, name, 0)
        end

        resume = Checkpoint.restore_checkpoint!(
            trainer,
            snapshot,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
        @test resume.update == update
        @test Sampler.sampler_snapshot(resume.sampler) == expected_sampler
        @test all(
            left == right for (left, right) in
                zip(parameter_arrays(trainer.parameters),
                    parameter_arrays(expected_parameters))
        )
        @test optimizer_equal(trainer.optimizer_state, expected_optimizer)

        # A restored cache must expose the restored physical conductance.
        @test trainer.cache.leaf_relation.physical[1] ==
            Model.Afferents.conductance(
                trainer.parameters.leaf_relation.raw_conductance[1],
            )
    end
end

@testset "resume fails closed before mutation" begin
    mktempdir() do directory
        update = 9
        source_rows = collect(201:227)
        trainer = seed_optimizer_history!(fixture(), update)
        sampler = Sampler.DeterministicEpochSampler(source_rows, UInt64(0x919))
        advance_to_update!(sampler, update)
        path = Checkpoint.save_checkpoint(
            joinpath(directory, "canonical.jls"),
            trainer,
            sampler;
            update=update,
            run_contract=RUN_CONTRACT,
        )
        snapshot = Checkpoint.load_checkpoint(path)
        before_parameter = trainer.parameters.program_bank.payload[1, 7]
        before_moment = trainer.optimizer_state.program_first[1, 7]
        before_total = trainer.optimizer_state.steps.total

        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            snapshot,
            reverse(source_rows);
            expected_run_contract=RUN_CONTRACT,
        )
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            snapshot,
            source_rows;
            expected_run_contract=merge(RUN_CONTRACT, (; width=79)),
        )
        @test trainer.parameters.program_bank.payload[1, 7] == before_parameter
        @test trainer.optimizer_state.program_first[1, 7] == before_moment
        @test trainer.optimizer_state.steps.total == before_total

        changed_config_trainer = FixtureTrainer(
            trainer.parameters,
            trainer.cache,
            trainer.optimizer_state,
            Optimizer.OptimizerConfig(learning_rate=0.002),
        )
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            changed_config_trainer,
            snapshot,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )

        false_model = merge(snapshot, (; model_fingerprint=repeat("0", 64)))
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            false_model,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
        corrupted_parameters = deepcopy(snapshot.parameters)
        corrupted_parameters.program_bank.payload[1, 7] += 0.01f0
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            merge(snapshot, (; parameters=corrupted_parameters)),
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
        inconsistent_update = merge(snapshot, (; update=update + 1))
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            inconsistent_update,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
        drifted_sampler = merge(
            snapshot.sampler,
            (; cursor=snapshot.sampler.cursor - 1),
        )
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            merge(snapshot, (; sampler=drifted_sampler)),
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )

        legacy = merge(snapshot, (;
            schema=UInt32(1),
            format="candidate-delta-relation-graph-exact-v1",
        ))
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            legacy,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
        legacy_path = joinpath(directory, "legacy-v1.jls")
        open(legacy_path, "w") do io
            serialize(io, legacy)
        end
        @test_throws ArgumentError Checkpoint.load_checkpoint(legacy_path)
    end
end

@testset "source closure drift is rejected at load and restore" begin
    mktempdir() do directory
        update = 5
        source_rows = collect(301:319)
        trainer = seed_optimizer_history!(fixture(), update)
        sampler = Sampler.DeterministicEpochSampler(source_rows, UInt64(0xa11))
        advance_to_update!(sampler, update)
        path = Checkpoint.save_checkpoint(
            joinpath(directory, "source-bound.jls"),
            trainer,
            sampler;
            update=update,
            run_contract=RUN_CONTRACT,
        )
        snapshot = Checkpoint.load_checkpoint(path)
        false_closure = merge(
            snapshot.source_closure,
            (; aggregate=repeat("f", 64)),
        )
        drifted = merge(snapshot, (; source_closure=false_closure))
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            drifted,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )

        drifted_path = joinpath(directory, "source-drift.jls")
        open(drifted_path, "w") do io
            serialize(io, drifted)
        end
        @test_throws ArgumentError Checkpoint.load_checkpoint(drifted_path)

        false_file = merge(
            snapshot.source_closure.files[1],
            (; sha256=repeat("e", 64)),
        )
        false_files = copy(snapshot.source_closure.files)
        false_files[1] = false_file
        file_drift = merge(
            snapshot,
            (; source_closure=merge(
                snapshot.source_closure,
                (; files=false_files),
            )),
        )
        @test_throws ArgumentError Checkpoint.restore_checkpoint!(
            trainer,
            file_drift,
            source_rows;
            expected_run_contract=RUN_CONTRACT,
        )
    end
end
