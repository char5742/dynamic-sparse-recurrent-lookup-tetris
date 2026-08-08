using Random
using Serialization
using Test

module CanonicalCheckpointTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
include(joinpath(@__DIR__, "CanonicalExperimentData.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
include(joinpath(@__DIR__, "CanonicalListNet.jl"))
include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalPlasticity.jl"))
include(joinpath(@__DIR__, "CanonicalDendriticGraph.jl"))
include(joinpath(@__DIR__, "CanonicalBarrierless.jl"))
include(joinpath(@__DIR__, "Sampler.jl"))
include(joinpath(@__DIR__, "CanonicalCheckpoint.jl"))
include(joinpath(@__DIR__, "CanonicalTraining.jl"))
end
const Harness = CanonicalCheckpointTestHarness
const CP = Harness.CanonicalCheckpoint
const Data = Harness.CanonicalExperimentData
const Graph = Harness.CanonicalDendriticGraph
const Input = Harness.CanonicalTetrisInput
const Local = Harness.CanonicalLocalLearning
const Output = Harness.DendriticOutputPopulation
const Sampler = Harness.ReducedHayCPUSampler
const Training = Harness.CanonicalTraining

const STATE_BATCH = 8
const WIDTH = 80
const WORKERS = 20
const QUEUE_CAPACITY = 64
const CHUNK = 4

function canonical_batch()
    batch = Data.CanonicalBatch(STATE_BATCH, WIDTH)
    input = batch.input
    input.valid_count = 2 * STATE_BATCH
    ordinal = 1
    @inbounds for state in 1:STATE_BATCH
        input.rows[state] = state
        input.counts[state] = Int16(2)
        first_flat = (state - 1) * WIDTH + 1
        second_flat = first_flat + 1
        input.valid_flats[ordinal] = Int32(first_flat)
        input.valid_flats[ordinal + 1] = Int32(second_flat)
        input.raw_placement[mod1(state, Input.BOARD_ROWS), 1, second_flat] =
            Input.PRESENT
        input.positions[1, second_flat] = UInt16(state)
        input.placement_counts[second_flat] = UInt8(1)
        batch.teacher.teacher_q[1, state] = 1.0f0
        batch.teacher.teacher_q[2, state] = 0.0f0
        batch.teacher.raw22[Output.Q_INDEX, first_flat] = 1.0f0
        for output in Output.QUANTILE_RANGE
            batch.teacher.raw22[output, first_flat] = 1.0f0
        end
        ordinal += 2
    end
    return batch
end

function canonical_config()
    local_learning = Local.LocalLearningConfig(
        schedule=Local.LearningSchedule(
            analog_interval=1,
            hard_event_interval=2,
            homeostasis_interval=8,
            structure_interval=64,
        ),
        hard_event_multiplier=0.25f0,
        hard_event_energy_cost=0.125f0,
        plasticity=Local.PlasticityConfig(structure_enabled=false),
    )
    return Training.CanonicalTrainingConfig(local_learning=local_learning)
end

function run_contract(model, config; sampler_seed=UInt64(0x53414d504c455232))
    return CP.CanonicalRunContract(
        CP.RUN_CONTRACT_SCHEMA,
        1,
        repeat("1", 64),
        repeat("2", 64),
        repeat("3", 64),
        72,
        64,
        8,
        1_024,
        1,
        repeat("4", 64),
        repeat("5", 64),
        "training/validation-v1",
        repeat("a", 40),
        true,
        UInt64(0x4d4f44454c534545),
        sampler_seed,
        STATE_BATCH,
        WIDTH,
        WORKERS,
        QUEUE_CAPACITY,
        CHUNK,
        :none,
        Training.training_config_fingerprint(config),
        CP.architecture_fingerprint(model.config),
        CP.topology_fingerprint(model),
        repeat("6", 64),
        100_000,
        1_000,
        1_000,
        1_000,
        :one_successful_state_batch,
    )
end

@inline function consume_and_update!(trainer, sampler, batch, row_buffer)
    Sampler.next_batch!(row_buffer, sampler)
    copyto!(batch.input.rows, row_buffer)
    return Training.train_update!(trainer, batch)
end

function assert_parameter_state_equal(left, right)
    @test reinterpret(UInt32, vec(left.core_cell_raw)) ==
        reinterpret(UInt32, vec(right.core_cell_raw))
    @test reinterpret(UInt32, vec(left.semantic_projection_raw)) ==
        reinterpret(UInt32, vec(right.semantic_projection_raw))
    @test reinterpret(UInt32, left.event_raw) ==
        reinterpret(UInt32, right.event_raw)
    @test reinterpret(UInt32, vec(left.output_cell_raw)) ==
        reinterpret(UInt32, vec(right.output_cell_raw))
    @test reinterpret(UInt32, vec(left.output_projection_raw)) ==
        reinterpret(UInt32, vec(right.output_projection_raw))
end

function assert_persistent_equal(left, right)
    @test left.state_fingerprint == right.state_fingerprint
    @test left.training_updates == right.training_updates
    @test left.optimizer.total_step == right.optimizer.total_step
    @test left.optimizer.group_steps == right.optimizer.group_steps
    @test left.learning_clock == right.learning_clock
    @test left.cumulative_mechanisms == right.cumulative_mechanisms
    assert_parameter_state_equal(
        left.optimizer.parameters, right.optimizer.parameters,
    )
    assert_parameter_state_equal(
        left.optimizer.first_moments, right.optimizer.first_moments,
    )
    assert_parameter_state_equal(
        left.optimizer.second_moments, right.optimizer.second_moments,
    )
    @test reinterpret(UInt32, left.plasticity.firing_rate) ==
        reinterpret(UInt32, right.plasticity.firing_rate)
    @test reinterpret(UInt32, left.plasticity.activity_ema) ==
        reinterpret(UInt32, right.plasticity.activity_ema)
    @test reinterpret(UInt32, left.plasticity.incoming_conductance_ema) ==
        reinterpret(UInt32, right.plasticity.incoming_conductance_ema)
    @test reinterpret(UInt32, left.plasticity.utility) ==
        reinterpret(UInt32, right.plasticity.utility)
    @test left.plasticity.reduced_batches == right.plasticity.reduced_batches
    @test left.plasticity.homeostasis_events ==
        right.plasticity.homeostasis_events
    @test left.plasticity.synaptic_scaling_events ==
        right.plasticity.synaptic_scaling_events
    @test left.plasticity.utility_updates == right.plasticity.utility_updates
    @test left.plasticity.rewires == right.plasticity.rewires
    @test left.sampler.seed == right.sampler.seed
    @test left.sampler.epoch == right.sampler.epoch
    @test left.sampler.cursor == right.sampler.cursor
    @test left.sampler.source_rows == right.sampler.source_rows
    @test left.sampler.permutation == right.sampler.permutation
end

@testset "schema2 typed checkpoint and exact virgin resume" begin
    mktempdir() do directory
        rows = collect(1:64)
        seed = UInt64(0x53414d504c455232)
        batch = canonical_batch()
        config = canonical_config()
        model = Graph.initialize_model(MersenneTwister(0x4d4f44454c534545))
        contract = run_contract(model, config; sampler_seed=seed)
        sampler = Sampler.DeterministicEpochSampler(rows, seed)
        row_buffer = zeros(Int, STATE_BATCH)
        checkpoint_path = joinpath(directory, "canonical-schema2.jls")

        continuous_final = Training.with_training_team(
            model,
            batch,
            config;
            workers=WORKERS,
            queue_capacity=QUEUE_CAPACITY,
            candidate_chunk_size=CHUNK,
            binding_mode=:none,
            log_config=false,
        ) do trainer
            consume_and_update!(trainer, sampler, batch, row_buffer)
            after_k = Training.checkpoint_components(
                trainer, sampler, contract,
            )
            CP.save_checkpoint(checkpoint_path, after_k)
            @test CP.load_checkpoint(checkpoint_path).state_fingerprint ==
                after_k.state_fingerprint

            legacy_path = joinpath(directory, "legacy-schema1.jls")
            open(legacy_path, "w") do io
                serialize(io, (; schema=UInt32(1), format="legacy"))
            end
            @test_throws ArgumentError CP.load_checkpoint(legacy_path)

            for legacy_event_count in (2_040, 2_296, 2_301)
                stale = deepcopy(after_k)
                resize!(stale.optimizer.parameters.event_raw, legacy_event_count)
                @test_throws DimensionMismatch CP.save_checkpoint(
                    joinpath(directory, "legacy-event-$legacy_event_count.jls"),
                    stale,
                )
            end
            nonfinite = deepcopy(after_k)
            nonfinite.optimizer.parameters.event_raw[1] = Inf32
            @test_throws DomainError CP.save_checkpoint(
                joinpath(directory, "nonfinite.jls"), nonfinite,
            )

            consume_and_update!(trainer, sampler, batch, row_buffer)
            Training.checkpoint_components(trainer, sampler, contract)
        end

        restored_model = Graph.initialize_model(
            MersenneTwister(0x4d4f44454c534545),
        )
        restored_sampler = Sampler.DeterministicEpochSampler(rows, seed)
        restored_batch = canonical_batch()
        restored_rows = zeros(Int, STATE_BATCH)
        loaded = CP.load_checkpoint(checkpoint_path)
        restored_final = Training.with_training_team(
            restored_model,
            restored_batch,
            config;
            workers=WORKERS,
            queue_capacity=QUEUE_CAPACITY,
            candidate_chunk_size=CHUNK,
            binding_mode=:none,
            log_config=false,
        ) do trainer
            Training.restore_training_checkpoint!(
                trainer, restored_sampler, loaded, contract,
            )
            @test_throws ArgumentError Training.restore_training_checkpoint!(
                trainer, restored_sampler, loaded, contract,
            )
            consume_and_update!(
                trainer, restored_sampler, restored_batch, restored_rows,
            )
            Training.checkpoint_components(
                trainer, restored_sampler, contract,
            )
        end

        assert_persistent_equal(continuous_final, restored_final)
        @test continuous_final.cumulative_mechanisms.hard_event_control_updates > 0
        @test Sampler.sampler_consumed_rows(sampler) == UInt128(16)
        @test Sampler.sampler_consumed_rows(restored_sampler) == UInt128(16)
    end
end
