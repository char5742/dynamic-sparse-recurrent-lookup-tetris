using Test
using Serialization
using Random

module CanonicalCheckpointTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalDendriticGraph.jl"))
include(joinpath(@__DIR__, "CanonicalCheckpoint.jl"))
end

const CP = CanonicalCheckpointTestHarness.CanonicalCheckpoint
const Local = CanonicalCheckpointTestHarness.CanonicalLocalLearning
const Opt = CanonicalCheckpointTestHarness.CanonicalOptimizer
const Topology = CanonicalCheckpointTestHarness.OrderedMultiscaleTopology
const Input = CanonicalCheckpointTestHarness.CanonicalTetrisInput
const Graph = CanonicalCheckpointTestHarness.CanonicalDendriticGraph

struct TestParameters
    cell::Matrix{Float32}
    packet::Vector{Float32}
end

struct TestConfig
    max_candidates::Int
    max_event_waves::Int
    tape_capacity::Int
    event_overflow::Symbol
end

mutable struct TestCounters
    update::Int
    decolle_signal_nonzero::Int
    subthreshold_updates::Int
    rewires::Int
end

function fixture()
    model = Graph.initialize_model(MersenneTwister(0x5eed))
    parameters = TestParameters(
        reshape(Float32.(1:12) ./ 10.0f0, 3, 4),
        Float32[0.25, -0.5, 0.75],
    )
    first = TestParameters(
        fill(0.125f0, size(parameters.cell)),
        fill(-0.25f0, size(parameters.packet)),
    )
    second = TestParameters(
        fill(0.5f0, size(parameters.cell)),
        fill(0.75f0, size(parameters.packet)),
    )
    configs = (;
        architecture=model.config,
        input=CP.canonical_input_contract(Input),
        topology=model,
        learning=Local.LocalLearningConfig(
            schedule=Local.LearningSchedule(
                analog_interval=1,
                hard_event_interval=4,
                homeostasis_interval=128,
                structure_interval=4096,
            ),
            feedback_seed=UInt64(0xdec011e),
            plasticity=Local.PlasticityConfig(
                utility_decay=0.997,
                connection_cost=2.0e-6,
            ),
        ),
        optimizer=Opt.AdamWConfig(
            learning_rate=0.002f0,
            beta1=0.9f0,
            beta2=0.999f0,
            epsilon=1.0f-8,
            clip_norm=5.0f0,
            weight_decay=1.0f-4,
        ),
    )
    counters = TestCounters(37, 31, 19, 2)
    return parameters, first, second, configs, counters
end

function optimizer_fixture(model::Graph.CanonicalModel)
    parameters = Graph.parameter_components(model.parameters)
    gradient = Graph.initialize_gradient(model)
    gradients = Graph.gradient_components(gradient)
    groups = (
        Opt.ParameterGroup(
            :core_cell_raw,
            parameters.core_cell_raw,
            gradients.core_cell_raw,
            Opt.CELL_RAW;
            multiplier=0.5f0,
        ),
        Opt.ParameterGroup(
            :semantic_projection_raw,
            parameters.semantic_projection_raw,
            gradients.semantic_projection_raw,
            Opt.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=0.75f0,
            lower_bound=1.0f-4,
            upper_bound=4.0f0,
        ),
        Opt.ParameterGroup(
            :event_raw,
            parameters.event_raw,
            gradients.event_raw,
            Opt.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=0.5f0,
            lower_bound=1.0f-4,
            upper_bound=4.0f0,
        ),
        Opt.ParameterGroup(
            :output_cell_raw,
            parameters.output_cell_raw,
            gradients.output_cell_raw,
            Opt.CELL_RAW;
            multiplier=1.0f0,
        ),
        Opt.ParameterGroup(
            :output_projection_raw,
            parameters.output_projection_raw,
            gradients.output_projection_raw,
            Opt.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=1.0f0,
            lower_bound=1.0f-4,
            upper_bound=4.0f0,
        ),
    )
    registry = Opt.ParameterRegistry(groups...)
    state = Opt.AdamWState(registry)
    @inbounds for (index, moment) in enumerate(state.moments)
        fill!(moment.first, 0.025f0 * index)
        fill!(moment.second, 0.05f0 * index)
    end
    state.group_steps .= UInt64[5, 6, 5, 7, 7]
    state.total_step = UInt64(7)
    return registry, state
end

function registry_with_event_length(registry, event_count::Int)
    original = registry.groups[3]
    event_group = Opt.ParameterGroup(
        :event_raw,
        zeros(Float32, event_count),
        zeros(Float32, event_count),
        original.transform_kind;
        multiplier=original.multiplier,
        lower_bound=original.lower_bound,
        upper_bound=original.upper_bound,
    )
    return Opt.ParameterRegistry(
        Base.setindex(registry.groups, event_group, 3),
    )
end

function training_counters(config, update::Int)
    clock = Local.LearningClockState()
    for _ in 1:update
        Local.advance_clocks!(clock, config.learning.schedule)
    end
    mechanisms = (;
        decolle_signal_nonzero=Int64(31),
        subthreshold_updates=Int64(19),
        nonspiking_updates=Int64(11),
        homeostasis_events=Int64(0),
        synaptic_scaling_events=Int64(0),
        utility_updates=Int64(7),
        rewires=Int64(0),
    )
    return (;
        learning_clock=clock,
        mechanisms,
        training_updates=UInt64(update),
    )
end

function with_utility_decay(config::Local.LocalLearningConfig, decay::Real)
    original = config.plasticity
    plasticity = Local.PlasticityConfig(
        firing_ema_decay=original.firing_ema_decay,
        target_rate_min=original.target_rate_min,
        target_rate_max=original.target_rate_max,
        threshold_homeostasis_step=original.threshold_homeostasis_step,
        adaptation_homeostasis_step=original.adaptation_homeostasis_step,
        synaptic_scaling_rate=original.synaptic_scaling_rate,
        conductance_floor=original.conductance_floor,
        conductance_ceiling=original.conductance_ceiling,
        structure_enabled=original.structure_enabled,
        utility_decay=decay,
        connection_cost=original.connection_cost,
        max_swaps_per_node=original.max_swaps_per_node,
    )
    return Local.LocalLearningConfig(
        schedule=config.schedule,
        feedback_seed=config.feedback_seed,
        feedback_scale=config.feedback_scale,
        predictor_scale=config.predictor_scale,
        predictor_dim=config.predictor_dim,
        eligibility_decay=config.eligibility_decay,
        analog_multiplier=config.analog_multiplier,
        hard_event_multiplier=config.hard_event_multiplier,
        utility_mode=config.utility_mode,
        plasticity=plasticity,
    )
end

function save_fixture(path, parameters, first, second, configs, counters;
                      step=37)
    return CP.save_checkpoint(
        path;
        parameters=(core=parameters,),
        first_moments=(core=first,),
        second_moments=(core=second,),
        optimizer_step=step,
        counters=counters,
        architecture_config=configs.architecture,
        input_config=configs.input,
        topology_config=configs.topology,
        learning_config=configs.learning,
        optimizer_config=configs.optimizer,
    )
end

@testset "canonical optimizer registry roundtrip" begin
    mktempdir() do directory
        _, _, _, configs, _ = fixture()
        counters = training_counters(configs, 7)
        registry, state = optimizer_fixture(configs.topology)
        expected_parameters = map(group -> copy(group.parameter), registry.groups)
        expected_first = map(moment -> copy(moment.first), state.moments)
        expected_second = map(moment -> copy(moment.second), state.moments)
        expected_event_weight = copy(configs.topology.cache.event_weight)
        path = CP.save_checkpoint(
            joinpath(directory, "optimizer.jls"),
            registry,
            state;
            counters,
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )
        snapshot = CP.load_checkpoint(path)
        @test Opt.parameter_group_names(registry) == CP.CANONICAL_PARAMETER_GROUPS
        @test length(registry.groups[3].parameter) == 2_125
        @test Graph.stored_parameter_count(configs.topology) == 69_445
        @test snapshot.optimizer_step == 7
        @test snapshot.counters.optimizer_group_steps ==
            (UInt64(5), UInt64(6), UInt64(5), UInt64(7), UInt64(7))

        for group in registry.groups
            fill!(group.parameter, 0.0f0)
        end
        Graph.refresh_cache!(configs.topology)
        @test configs.topology.cache.event_weight != expected_event_weight
        for moment in state.moments
            fill!(moment.first, 0.0f0)
            fill!(moment.second, 0.0f0)
        end
        state.group_steps .= 0
        state.total_step = 0
        resume = CP.restore_checkpoint!(
            registry,
            state,
            snapshot;
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )
        @test map(group -> group.parameter, registry.groups) == expected_parameters
        @test map(moment -> moment.first, state.moments) == expected_first
        @test map(moment -> moment.second, state.moments) == expected_second
        @test configs.topology.cache.event_weight == expected_event_weight
        @test state.group_steps == UInt64[5, 6, 5, 7, 7]
        @test state.total_step == 7
        @test resume.optimizer_step == 7
        @test resume.counters.learning_clock.update == 7
        @test resume.counters.learning_clock.analog_ticks == 7
        @test resume.counters.learning_clock.hard_event_ticks == 1
        @test resume.counters.mechanisms.decolle_signal_nonzero == 31
        learning_contract = CP.canonical_learning_contract(configs.learning)
        @test learning_contract.fixed_local_signal_map.output_dim == 22
        @test learning_contract.fixed_local_signal_map.continuous_observation_dim == 47
        @test learning_contract.fixed_local_signal_map.packet_observation_dim == 12
        @test learning_contract.fixed_local_signal_map.cell_count == 1_436

        bad_clock_counters = deepcopy(counters)
        bad_clock_counters.learning_clock.hard_event_ticks += 1
        @test_throws ArgumentError CP.save_checkpoint(
            joinpath(directory, "bad-clock.jls"),
            registry,
            state;
            counters=bad_clock_counters,
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )

        wrong_core = Opt.ParameterGroup(
            :core_cell_raw,
            zeros(Float32, 45, 1_436),
            zeros(Float32, 45, 1_436),
            Opt.CELL_RAW;
            multiplier=registry.groups[1].multiplier,
        )
        wrong_shape_registry = Opt.ParameterRegistry(
            Base.setindex(registry.groups, wrong_core, 1),
        )
        wrong_shape_state = Opt.AdamWState(wrong_shape_registry)
        @test_throws DimensionMismatch CP.save_checkpoint(
            joinpath(directory, "wrong-shape.jls"),
            wrong_shape_registry,
            wrong_shape_state;
            counters=training_counters(configs, 0),
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )

        short_registry = Opt.ParameterRegistry(registry.groups[1:4])
        short_state = Opt.AdamWState(short_registry)
        @test_throws ArgumentError CP.save_checkpoint(
            joinpath(directory, "missing-group.jls"),
            short_registry,
            short_state;
            counters=training_counters(configs, 0),
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )

        # Historical event layouts are semantically incompatible even when
        # their static source-major prefix matches the canonical graph.
        for legacy_event_count in (2_040, 2_296, 2_301)
            legacy_registry = registry_with_event_length(
                registry,
                legacy_event_count,
            )
            legacy_state = Opt.AdamWState(legacy_registry)
            @test_throws DimensionMismatch CP.save_checkpoint(
                joinpath(directory, "legacy-event-$legacy_event_count.jls"),
                legacy_registry,
                legacy_state;
                counters=training_counters(configs, 0),
                architecture_config=configs.architecture,
                input_config=configs.input,
                topology_config=configs.topology,
                learning_config=configs.learning,
                optimizer_config=configs.optimizer,
            )
        end

        # Metadata is part of the optimizer fingerprint, not mutable payload.
        changed = deepcopy(registry)
        original = changed.groups[1]
        changed_group = Opt.ParameterGroup(
            changed.groups[1].name,
            changed.groups[1].parameter,
            changed.groups[1].gradient,
            changed.groups[1].transform_kind;
            multiplier=0.25f0,
            lower_bound=original.lower_bound,
            upper_bound=original.upper_bound,
        )
        changed_registry = Opt.ParameterRegistry(
            Base.setindex(changed.groups, changed_group, 1),
        )
        @test_throws ArgumentError CP.restore_checkpoint!(
            changed_registry,
            state,
            snapshot;
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )
    end
end

function restore_fixture!(parameters, first, second, snapshot, configs)
    return CP.restore_checkpoint!(
        (core=parameters,),
        (core=first,),
        (core=second,),
        snapshot;
        architecture_config=configs.architecture,
        input_config=configs.input,
        topology_config=configs.topology,
        learning_config=configs.learning,
        optimizer_config=configs.optimizer,
    )
end

@testset "canonical checkpoint normal roundtrip" begin
    mktempdir() do directory
        parameters, first, second, configs, counters = fixture()
        expected_parameters = deepcopy(parameters)
        expected_first = deepcopy(first)
        expected_second = deepcopy(second)
        path = save_fixture(
            joinpath(directory, "canonical.jls"),
            parameters,
            first,
            second,
            configs,
            counters,
        )
        snapshot = CP.load_checkpoint(path)
        @test snapshot.magic == CP.CHECKPOINT_MAGIC
        @test snapshot.schema == CP.CHECKPOINT_SCHEMA
        @test snapshot.format == CP.CHECKPOINT_FORMAT
        @test snapshot.parameter_registry == CP.parameter_registry((core=parameters,))
        @test snapshot.architecture_fingerprint ==
            CP.architecture_fingerprint(configs.architecture)
        @test snapshot.input_fingerprint == CP.input_fingerprint(configs.input)
        @test snapshot.topology_fingerprint ==
            CP.topology_fingerprint(configs.topology)
        @test snapshot.learning_fingerprint ==
            CP.learning_fingerprint(configs.learning)
        @test snapshot.optimizer_fingerprint ==
            CP.optimizer_fingerprint(configs.optimizer)
        @test configs.input.board.empty != configs.input.board.occupied
        @test keys(configs.input.pieces) ==
            (:none, :i, :o, :t, :s, :z, :j, :l)
        swapped_topology = deepcopy(configs.topology)
        swapped_topology.topology.edge_sources[1] = UInt16(
            Int(swapped_topology.topology.edge_sources[1]) + 1,
        )
        @test CP.topology_fingerprint(swapped_topology) !=
            CP.topology_fingerprint(configs.topology)
        changed_event_semantics = deepcopy(configs.topology)
        changed_event_semantics.cache.event_graph.destination[1] += UInt16(1)
        @test CP.topology_fingerprint(changed_event_semantics) !=
            CP.topology_fingerprint(configs.topology)
        @test_throws ArgumentError CP.topology_fingerprint(
            configs.topology.topology,
        )
        changed_adam = Opt.AdamWConfig(
            learning_rate=configs.optimizer.learning_rate,
            beta1=configs.optimizer.beta1,
            beta2=0.995f0,
            epsilon=configs.optimizer.epsilon,
            clip_norm=configs.optimizer.clip_norm,
            weight_decay=configs.optimizer.weight_decay,
        )
        @test CP.optimizer_fingerprint(changed_adam) !=
            CP.optimizer_fingerprint(configs.optimizer)
        changed_learning = with_utility_decay(configs.learning, 0.996f0)
        @test CP.learning_fingerprint(changed_learning) !=
            CP.learning_fingerprint(configs.learning)
        @test CP.architecture_fingerprint(
            TestConfig(128, 7, 11_488, :error),
        ) isa String
        @test_throws ArgumentError CP.architecture_fingerprint(
            TestConfig(128, 7, 11_487, :error),
        )
        @test_throws ArgumentError CP.architecture_fingerprint(
            TestConfig(128, 8, 13_000, :error),
        )
        @test_throws ArgumentError CP.architecture_fingerprint(
            TestConfig(128, 4, 1_436, :error),
        )

        fill!(parameters.cell, 0.0f0)
        fill!(parameters.packet, 0.0f0)
        fill!(first.cell, 0.0f0)
        fill!(first.packet, 0.0f0)
        fill!(second.cell, 0.0f0)
        fill!(second.packet, 0.0f0)
        resume = restore_fixture!(parameters, first, second, snapshot, configs)
        @test parameters.cell == expected_parameters.cell
        @test parameters.packet == expected_parameters.packet
        @test first.cell == expected_first.cell
        @test first.packet == expected_first.packet
        @test second.cell == expected_second.cell
        @test second.packet == expected_second.packet
        @test resume.optimizer_step == 37
        @test resume.counters isa TestCounters
        @test resume.counters.update == 37
        @test resume.counters.decolle_signal_nonzero == 31
        @test resume.counters.subthreshold_updates == 19
        @test resume.counters.rewires == 2
    end
end

@testset "canonical checkpoint fails closed" begin
    mktempdir() do directory
        parameters, first, second, configs, counters = fixture()
        path = save_fixture(
            joinpath(directory, "canonical.jls"),
            parameters,
            first,
            second,
            configs,
            counters,
        )
        snapshot = CP.load_checkpoint(path)

        before = copy(parameters.cell)
        wrong_topology = deepcopy(configs.topology)
        wrong_topology.topology.edge_roles[1] = xor(
            wrong_topology.topology.edge_roles[1],
            0x01,
        )
        wrong_configs = merge(configs, (; topology=wrong_topology))
        @test_throws ArgumentError restore_fixture!(
            parameters,
            first,
            second,
            snapshot,
            wrong_configs,
        )
        @test parameters.cell == before

        wrong_parameters = (;
            core=TestParameters(copy(parameters.cell), copy(parameters.packet)),
            unexpected=Float32[1],
        )
        @test_throws ArgumentError CP.restore_checkpoint!(
            wrong_parameters,
            (core=first,),
            (core=second,),
            snapshot;
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )

        corrupted_values = deepcopy(snapshot.parameter_values)
        corrupted_values[1][1] += 1.0f0
        @test_throws ArgumentError CP.restore_checkpoint!(
            (core=parameters,),
            (core=first,),
            (core=second,),
            merge(snapshot, (; parameter_values=corrupted_values));
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )
        @test parameters.cell == before

        legacy = (;
            schema=UInt32(2),
            format="candidate-delta-relation-motif-graph-exact-v2",
            parameters=(relation=zeros(Float32, 2), motif=zeros(Float32, 2)),
        )
        legacy_path = joinpath(directory, "legacy-relation-motif.jls")
        open(legacy_path, "w") do io
            serialize(io, legacy)
        end
        @test_throws ArgumentError CP.load_checkpoint(legacy_path)

        truncated_path = joinpath(directory, "truncated.jls")
        open(truncated_path, "w") do io
            write(io, UInt8[0x01, 0x02, 0x03])
        end
        @test_throws ArgumentError CP.load_checkpoint(truncated_path)
    end
end

@testset "canonical checkpoint bounds and registry guards" begin
    mktempdir() do directory
        parameters, first, second, configs, counters = fixture()
        @test_throws ArgumentError save_fixture(
            joinpath(directory, "negative-step.jls"),
            parameters,
            first,
            second,
            configs,
            counters;
            step=-1,
        )
        @test_throws ArgumentError save_fixture(
            joinpath(directory, "bool-step.jls"),
            parameters,
            first,
            second,
            configs,
            counters;
            step=true,
        )

        negative_counters = TestCounters(0, -1, 0, 0)
        @test_throws ArgumentError save_fixture(
            joinpath(directory, "negative-counter.jls"),
            parameters,
            first,
            second,
            configs,
            negative_counters;
            step=0,
        )

        bad_second = deepcopy(second)
        bad_second.cell[1] = -eps(Float32)
        @test_throws DomainError save_fixture(
            joinpath(directory, "negative-second.jls"),
            parameters,
            first,
            bad_second,
            configs,
            counters,
        )

        bad_parameters = deepcopy(parameters)
        bad_parameters.cell[1] = Inf32
        @test_throws DomainError save_fixture(
            joinpath(directory, "nonfinite.jls"),
            bad_parameters,
            first,
            second,
            configs,
            counters,
        )

        mismatched_first = (
            core=TestParameters(
                zeros(Float32, 4, 3),
                zeros(Float32, 3),
            ),
        )
        @test_throws ArgumentError CP.save_checkpoint(
            joinpath(directory, "registry-mismatch.jls");
            parameters=(core=parameters,),
            first_moments=mismatched_first,
            second_moments=(core=second,),
            optimizer_step=37,
            counters=counters,
            architecture_config=configs.architecture,
            input_config=configs.input,
            topology_config=configs.topology,
            learning_config=configs.learning,
            optimizer_config=configs.optimizer,
        )

        shared = Float32[1, 2]
        @test_throws ArgumentError CP.parameter_registry((left=shared, right=shared))
        @test_throws ArgumentError CP.parameter_registry((scalar=1.0f0,))

        path = save_fixture(
            joinpath(directory, "zero-step.jls"),
            parameters,
            first,
            second,
            configs,
            TestCounters(0, 0, 0, 0);
            step=0,
        )
        @test CP.load_checkpoint(path).optimizer_step == 0
    end
end
