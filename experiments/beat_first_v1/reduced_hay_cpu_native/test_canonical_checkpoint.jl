using Test
using Serialization

module CanonicalCheckpointTestHarness
include(joinpath(@__DIR__, "CanonicalCheckpoint.jl"))
end

const CP = CanonicalCheckpointTestHarness.CanonicalCheckpoint

struct TestParameters
    cell::Matrix{Float32}
    packet::Vector{Float32}
end

struct TestConfig
    count::Int
    scale::Float32
    labels::Tuple{Symbol,Symbol}
end

mutable struct TestCounters
    update::Int
    decolle_signal_nonzero::Int
    subthreshold_updates::Int
    rewires::Int
end

@enum TestTransformKind::UInt8 TEST_SIGNED=1 TEST_CONDUCTANCE=2

struct TestGroup
    name::Symbol
    parameter::Array{Float32}
    gradient::Array{Float32}
    transform_kind::TestTransformKind
    multiplier::Float32
    lower_bound::Float32
    upper_bound::Float32
end

struct TestRegistry
    groups::Tuple{TestGroup,TestGroup}
end

struct TestGroupMoments
    name::Symbol
    transform_kind::TestTransformKind
    multiplier::Float32
    lower_bound::Float32
    upper_bound::Float32
    first::Array{Float32}
    second::Array{Float32}
end

mutable struct TestOptimizerState
    moments::Tuple{TestGroupMoments,TestGroupMoments}
    group_steps::Vector{UInt64}
    total_step::UInt64
end

function fixture()
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
        architecture=TestConfig(1_458, 12.0f0, (:analog, :event)),
        input=(rows=24, columns=10, raw_placement=true, ren_encoding=:exact),
        topology=(
            packet_width=12,
            maximum_waves=4,
            fanout=4,
            source=UInt16[1, 2, 3],
            destination=UInt16[4, 5, 6],
            branch=UInt8[1, 2, 3],
            receptor=UInt8[1, 2, 3],
        ),
        learning=(eligibility_decay=0.8f0, feedback_seed=UInt64(0xdec011e)),
        optimizer=(learning_rate=0.002f0, beta1=0.9f0, beta2=0.999f0),
    )
    counters = TestCounters(37, 31, 19, 2)
    return parameters, first, second, configs, counters
end

function optimizer_fixture()
    left = reshape(Float32.(1:6), 2, 3)
    right = Float32[-1, 2, -3]
    groups = (
        TestGroup(
            :cell_raw,
            left,
            zeros(Float32, size(left)),
            TEST_CONDUCTANCE,
            0.5f0,
            0.01f0,
            4.0f0,
        ),
        TestGroup(
            :signed_output,
            right,
            zeros(Float32, size(right)),
            TEST_SIGNED,
            1.0f0,
            -Inf32,
            Inf32,
        ),
    )
    moments = map(groups) do group
        TestGroupMoments(
            group.name,
            group.transform_kind,
            group.multiplier,
            group.lower_bound,
            group.upper_bound,
            fill(0.1f0, size(group.parameter)),
            fill(0.2f0, size(group.parameter)),
        )
    end
    return TestRegistry(groups), TestOptimizerState(
        moments,
        UInt64[5, 7],
        UInt64(7),
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
        _, _, _, configs, counters = fixture()
        registry, state = optimizer_fixture()
        expected_parameters = map(group -> copy(group.parameter), registry.groups)
        expected_first = map(moment -> copy(moment.first), state.moments)
        expected_second = map(moment -> copy(moment.second), state.moments)
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
        @test snapshot.optimizer_step == 7
        @test snapshot.counters.optimizer_group_steps == (UInt64(5), UInt64(7))

        for group in registry.groups
            fill!(group.parameter, 0.0f0)
        end
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
        @test state.group_steps == UInt64[5, 7]
        @test state.total_step == 7
        @test resume.optimizer_step == 7
        @test resume.counters.update == counters.update

        # Metadata is part of the optimizer fingerprint, not mutable payload.
        changed = deepcopy(registry)
        changed_group = TestGroup(
            changed.groups[1].name,
            changed.groups[1].parameter,
            changed.groups[1].gradient,
            changed.groups[1].transform_kind,
            0.25f0,
            changed.groups[1].lower_bound,
            changed.groups[1].upper_bound,
        )
        changed_registry = TestRegistry((changed_group, changed.groups[2]))
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
        swapped_topology = merge(configs.topology, (;
            source=reverse(configs.topology.source),
        ))
        @test CP.topology_fingerprint(swapped_topology) !=
            CP.topology_fingerprint(configs.topology)
        changed_adam = merge(configs.optimizer, (; beta2=0.995f0))
        @test CP.optimizer_fingerprint(changed_adam) !=
            CP.optimizer_fingerprint(configs.optimizer)

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
        wrong_configs = merge(configs, (;
            topology=merge(configs.topology, (; packet_width=11)),
        ))
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
