using Test
using Random

module CanonicalTrainingTestHarness
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "CanonicalBarrierless.jl"))
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "CanonicalExperimentData.jl"))
include(joinpath(@__DIR__, "CanonicalListNet.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalPlasticity.jl"))
include(joinpath(@__DIR__, "CanonicalDendriticGraph.jl"))
include(joinpath(@__DIR__, "CanonicalTraining.jl"))
end

const Graph = CanonicalTrainingTestHarness.CanonicalDendriticGraph
const Barrierless = CanonicalTrainingTestHarness.CanonicalBarrierless
const Data = CanonicalTrainingTestHarness.CanonicalExperimentData
const Local = CanonicalTrainingTestHarness.CanonicalLocalLearning
const Optimizer = CanonicalTrainingTestHarness.CanonicalOptimizer
const Plasticity = CanonicalTrainingTestHarness.CanonicalPlasticity
const Output = CanonicalTrainingTestHarness.DendriticOutputPopulation
const Training = CanonicalTrainingTestHarness.CanonicalTraining

function tiny_canonical_batch()
    batch = Data.CanonicalBatch(1)
    input = batch.input
    input.rows[1] = 1
    input.counts[1] = 2
    input.valid_count = 2
    input.valid_flats[1] = 1
    input.valid_flats[2] = 2
    # Candidate two differs by one explicit placement cell; no teacher field is
    # reachable from either graph input reference.
    input.raw_placement[24, 1, 2] =
        CanonicalTrainingTestHarness.CanonicalTetrisInput.PRESENT
    input.positions[1, 2] = UInt16(231)
    input.placement_counts[2] = UInt8(1)
    batch.teacher.teacher_q[1, 1] = 1.0f0
    batch.teacher.teacher_q[2, 1] = 0.0f0
    batch.teacher.raw22[1, 1] = 1.0f0
    @inbounds for output in Output.QUANTILE_RANGE
        batch.teacher.raw22[output, 1] = 1.0f0
    end
    return batch
end

@testset "canonical training owns one explicit configuration" begin
    schedule = Local.LearningSchedule(
        analog_interval=2,
        hard_event_interval=3,
        homeostasis_interval=5,
        structure_interval=7,
    )
    plasticity = Local.PlasticityConfig(
        conductance_floor=2.0f-4,
        conductance_ceiling=3.0f0,
        structure_enabled=false,
    )
    local_learning = Local.LocalLearningConfig(
        schedule=schedule,
        feedback_seed=0x51,
        feedback_scale=0.75f0,
        analog_multiplier=0.5f0,
        hard_event_multiplier=0.0f0,
        utility_mode=:combined,
        plasticity=plasticity,
    )
    config = Training.CanonicalTrainingConfig(local_learning=local_learning)
    summary = Training.training_config_summary(config)
    @test config.local_learning === local_learning
    @test config.groups.conductance_floor == 2.0f-4
    @test config.groups.conductance_ceiling == 3.0f0
    @test occursin("analog_interval=2", summary)
    @test occursin("hard_event_interval=3", summary)
    @test occursin("feedback_seed=81", summary)
    @test occursin("structure_enabled=false", summary)
    @test occursin("adam_learning_rate=", summary)
    @test occursin("output_projection_multiplier=", summary)
    @test length(Training.training_config_fingerprint(config)) == 64
    @test_throws ArgumentError Training.CanonicalTrainingConfig(
        local_learning=local_learning,
        groups=Training.OptimizerGroupConfig(
            conductance_floor=1.0f-4,
            conductance_ceiling=3.0f0,
        ),
    )
end

@testset "real production path keeps output fast and recurrent transactional" begin
    batch = tiny_canonical_batch()
    local_learning = Local.LocalLearningConfig(
        schedule=Local.LearningSchedule(
            analog_interval=2,
            hard_event_interval=4,
            homeostasis_interval=2,
            structure_interval=32,
        ),
        hard_event_multiplier=0.0f0,
        plasticity=Local.PlasticityConfig(
            target_rate_min=0.90f0,
            target_rate_max=0.91f0,
            threshold_homeostasis_step=0.01f0,
            adaptation_homeostasis_step=0.01f0,
            synaptic_scaling_rate=0.01f0,
            structure_enabled=false,
        ),
    )
    config = Training.CanonicalTrainingConfig(local_learning=local_learning)
    model = Graph.initialize_model(MersenneTwister(0x74524149))
    adapter = Training.DendriticTrainingAdapter(
        model, batch, config; candidate_chunk_size=1,
    )
    executor = Barrierless.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=1,
        candidate_chunk_size=1,
    )
    core_before = copy(adapter.registry.groups[1].parameter)
    output_before = copy(adapter.registry.groups[4].parameter)
    fill!(adapter.plasticity_state.utility, 0.25f0)
    utility_before = copy(reinterpret(UInt32, adapter.plasticity_state.utility))
    utility_updates_before = adapter.plasticity_state.utility_updates

    first_result = Barrierless.serial_reference_update!(executor)
    @test first_result.update == 1
    @test adapter.clocks.update == 1
    @test adapter.optimizer_state.group_steps == UInt64[0, 0, 0, 1, 1]
    @test adapter.registry.groups[1].parameter == core_before
    @test adapter.registry.groups[4].parameter != output_before
    @test first_result.mechanisms.decolle_signal_nonzero == 0
    @test reinterpret(UInt32, adapter.plasticity_state.utility) == utility_before
    @test adapter.plasticity_state.utility_updates == utility_updates_before
    @test all(!state.ready for state in adapter.common_states)

    # A due slow-plasticity counter overflow is detected after the real replay
    # publication joins but before Adam, plasticity, moments, or clocks mutate.
    adapter.plasticity_state.homeostasis_events = typemax(UInt64)
    parameters_before_failure = map(
        group -> copy(group.parameter), adapter.registry.groups,
    )
    first_moments_before_failure = map(
        moment -> copy(moment.first), adapter.optimizer_state.moments,
    )
    second_moments_before_failure = map(
        moment -> copy(moment.second), adapter.optimizer_state.moments,
    )
    group_steps_before_failure = copy(adapter.optimizer_state.group_steps)
    clock_before_failure = deepcopy(adapter.clocks)
    updates_before_failure = adapter.updates
    @test_throws OverflowError Barrierless.serial_reference_update!(executor)
    @test map(group -> group.parameter, adapter.registry.groups) ==
        parameters_before_failure
    @test map(moment -> moment.first, adapter.optimizer_state.moments) ==
        first_moments_before_failure
    @test map(moment -> moment.second, adapter.optimizer_state.moments) ==
        second_moments_before_failure
    @test adapter.optimizer_state.group_steps == group_steps_before_failure
    @test adapter.clocks.update == clock_before_failure.update
    @test adapter.clocks.analog_ticks == clock_before_failure.analog_ticks
    @test adapter.updates == updates_before_failure
    @test reinterpret(UInt32, adapter.plasticity_state.utility) == utility_before
    @test adapter.plasticity_state.utility_updates == utility_updates_before
    adapter.plasticity_state.homeostasis_events = UInt64(0)

    second_result = Barrierless.serial_reference_update!(executor)
    @test second_result.update == 2
    @test adapter.clocks.update == 2
    @test adapter.optimizer_state.group_steps == UInt64[1, 1, 1, 2, 2]
    @test second_result.mechanisms.decolle_signal_nonzero > 0
    @test second_result.mechanisms.subthreshold_updates > 0
    @test second_result.mechanisms.nonspiking_updates > 0
    @test second_result.mechanisms.homeostasis_events > 0
    @test second_result.mechanisms.synaptic_scaling_events > 0
    @test second_result.mechanisms.utility_updates > 0
    @test reinterpret(UInt32, adapter.plasticity_state.utility) != utility_before
    @test adapter.plasticity_state.utility_updates > utility_updates_before
    @test adapter.plasticity_state.reduced_batches == 2
    common_slot = cld(batch.input.valid_count, adapter.candidate_chunk_size) + 1
    @test adapter.slot_generation[common_slot] == adapter.active_generation
    @test adapter.slot_kind[common_slot] == 0x02
    @test adapter.slot_logical_first[common_slot] == 1
    @test adapter.slot_logical_last[common_slot] == 1
    @test all(==(adapter.plasticity_batch.generation),
              adapter.plasticity_batch.common.stamp[1:1])
    @test all(==(adapter.plasticity_batch.generation),
              adapter.plasticity_batch.candidate.stamp[1:2])
    @test all(!state.ready for state in adapter.common_states)
end

@testset "utility mode none freezes persistent utility on analog ticks" begin
    batch = tiny_canonical_batch()
    config = Training.CanonicalTrainingConfig(
        local_learning=Local.LocalLearningConfig(
            schedule=Local.LearningSchedule(analog_interval=1),
            utility_mode=:none,
        ),
    )
    model = Graph.initialize_model(MersenneTwister(0x554e4f4e45))
    adapter = Training.DendriticTrainingAdapter(
        model, batch, config; candidate_chunk_size=2,
    )
    executor = Barrierless.CanonicalExecutor(
        adapter, batch; worker_capacity=1, candidate_chunk_size=2,
    )
    fill!(adapter.plasticity_state.utility, 0.375f0)
    utility_before = copy(reinterpret(UInt32, adapter.plasticity_state.utility))
    result = Barrierless.serial_reference_update!(executor)
    @test result.update == 1
    @test result.mechanisms.decolle_signal_nonzero > 0
    @test result.mechanisms.utility_updates == 0
    @test reinterpret(UInt32, adapter.plasticity_state.utility) == utility_before
    @test adapter.plasticity_state.utility_updates == 0
    @test adapter.plasticity_state.reduced_batches == 1
end

@testset "concrete graph parameter registry is coordinator-owned" begin
    model = Graph.initialize_model(MersenneTwister(4))
    reduced = Graph.initialize_gradient(model)
    group_config = Training.OptimizerGroupConfig()
    registry = Training._parameter_registry(model, reduced, group_config)
    @test Optimizer.parameter_group_names(registry) == (
        :core_cell_raw,
        :semantic_projection_raw,
        :event_raw,
        :output_cell_raw,
        :output_projection_raw,
    )
    @test Optimizer.registry_group_count(registry) == 5
    parameters = Graph.parameter_components(model.parameters)
    gradients = Graph.gradient_components(reduced)
    @test registry.groups[1].parameter === parameters.core_cell_raw
    @test registry.groups[1].gradient === gradients.core_cell_raw
    @test registry.groups[2].parameter === parameters.semantic_projection_raw
    @test registry.groups[3].parameter === parameters.event_raw
    @test registry.groups[4].parameter === parameters.output_cell_raw
    @test registry.groups[5].parameter === parameters.output_projection_raw
    @test registry.groups[1].transform_kind == Optimizer.CELL_RAW
    @test registry.groups[2].transform_kind ==
        Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE
    @test registry.groups[3].transform_kind ==
        Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE
    @test registry.groups[4].transform_kind == Optimizer.CELL_RAW
    @test registry.groups[5].transform_kind ==
        Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE
end

@testset "due clocks map to five registry groups without zero-gradient emulation" begin
    model = Graph.initialize_model(MersenneTwister(8))
    reduced = Graph.initialize_gradient(model)
    config = Training.CanonicalTrainingConfig(
        local_learning=Local.LocalLearningConfig(
            schedule=Local.LearningSchedule(
                analog_interval=2,
                hard_event_interval=3,
                homeostasis_interval=5,
                structure_interval=7,
            ),
        ),
    )
    registry = Training._parameter_registry(model, reduced, config.groups)
    adam = Optimizer.AdamWState(registry)
    parameters_before = map(group -> copy(group.parameter), registry.groups)
    first_before = map(moment -> copy(moment.first), adam.moments)
    second_before = map(moment -> copy(moment.second), adam.moments)
    due = (false, false, false, true, true)
    for group in registry.groups
        fill!(group.gradient, 0.01f0)
    end
    stats = Optimizer.apply_optimizer_boundary!(
        adam, registry, config.optimizer; due_mask=due,
    )
    @test stats.active_groups == 2
    @test adam.group_steps == UInt64[0, 0, 0, 1, 1]
    for index in 1:3
        @test registry.groups[index].parameter == parameters_before[index]
        @test adam.moments[index].first == first_before[index]
        @test adam.moments[index].second == second_before[index]
    end
end

@testset "22D auxiliary credit reaches shared V and centered A exactly" begin
    components = [Output.OutputComponents(Float32) for _ in 1:4]
    bars = [Output.OutputComponentGradient(Float32) for _ in 1:4]
    raw_delta = zeros(Float32, Output.OUTPUT_DIM, 4)
    @inbounds for candidate in 1:4
        raw_delta[Output.Q_INDEX, candidate] = 0.1f0 * candidate
        for output in Output.QUANTILE_RANGE
            raw_delta[output, candidate] =
                Float32(output + candidate) * 1.0f-3
        end
    end
    qbar = [
        Output.q_cotangent(@view(raw_delta[:, candidate]))
        for candidate in 1:4
    ]
    total = Float32(sum(Float64, qbar))
    mean_bar = total / 4.0f0
    @inbounds for candidate in 1:4
        returned = Output.assemble_output_pullback!(
            bars[candidate],
            @view(raw_delta[:, candidate]),
            components[candidate],
            qbar[candidate] - mean_bar,
        )
        @test returned == qbar[candidate]
        bars[candidate].value = 0.0f0
    end
    @test total != sum(@view raw_delta[Output.Q_INDEX, :])
    @test isapprox(sum(bar.advantage for bar in bars), 0.0f0; atol=2eps(Float32))
    @test all(iszero(bar.value) for bar in bars)
end

@testset "candidate-set assembly is permutation invariant" begin
    components = [Output.OutputComponents(Float32) for _ in 1:5]
    @inbounds for candidate in eachindex(components)
        components[candidate].advantage = Float32(
            (-1)^candidate * candidate / 7,
        )
        components[candidate].death = 0.1f0 * candidate
        components[candidate].geometry .= Float32[
            candidate, candidate^2, -candidate, 0.25f0 * candidate,
        ]
        components[candidate].uncertainty_raw = -0.2f0 * candidate
    end
    reference_components = deepcopy(components)
    reference = zeros(Float32, Output.OUTPUT_DIM, 5)
    Graph.assemble_candidate_set!(reference, 0.375f0, reference_components, 5)

    permutation = (4, 1, 5, 2, 3)
    permuted_components = [deepcopy(components[index]) for index in permutation]
    permuted = zeros(Float32, Output.OUTPUT_DIM, 5)
    Graph.assemble_candidate_set!(permuted, 0.375f0, permuted_components, 5)
    @inbounds for candidate in 1:5
        permuted_slot = findfirst(==(candidate), permutation)
        @test reference[:, candidate] == permuted[:, permuted_slot]
    end
end

@testset "mechanism counters fail closed before a production mutation" begin
    expected = Training.MechanismActivation(
        true, true, true, false, false, true, false,
    )
    missing = Training.MechanismCounters(1, 2, 0, 0, 0, 4, 0)
    @test_throws ErrorException Training._assert_mechanisms!(
        expected, missing; include_slow=false,
    )
    complete = Training.MechanismCounters(1, 2, 3, 0, 0, 4, 0)
    @test isnothing(Training._assert_mechanisms!(
        expected, complete; include_slow=false,
    ))
end

@testset "production API contains no exact or arbitrary graph mode" begin
    @test !(:adapter in fieldnames(Graph.CanonicalModel))
    @test !(:exact in fieldnames(Training.CanonicalTrainingConfig))
    @test !(:mode in fieldnames(Training.CanonicalTrainingConfig))
    @test !isdefined(Training, :MechanismHooks)
    @test !isdefined(Training, :training_forward_candidate!)
    @test !isdefined(Training, :training_replay_candidate!)
end
