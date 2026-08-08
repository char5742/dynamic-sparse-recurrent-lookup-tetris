using LinearAlgebra
using Test

module CanonicalLocalLearningTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
end

const Cell = CanonicalLocalLearningTestHarness.ActiveApicalCell
const Axon = CanonicalLocalLearningTestHarness.DendriticAxonPacket
const Local = CanonicalLocalLearningTestHarness.CanonicalLocalLearning

function nonspiking_transition()
    raw = Cell.default_raw_parameters()
    cache = Cell.transform_parameters(raw)
    state = Cell.initial_state(cache)
    state[Cell.ADAPTATION_INDEX] = 0.2f0
    input = zeros(Float32, Cell.INPUT_DIM)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        input[Cell.input_index(compartment, Cell.INPUT_AMPA)] = 0.20f0
        input[Cell.input_index(compartment, Cell.INPUT_NMDA)] = 0.15f0
        input[Cell.input_index(compartment, Cell.INPUT_GABA)] = 0.05f0
    end
    next_state = Cell.cell_step_cached_functional(state, input, cache)
    @assert next_state[Cell.SPIKE_INDEX] == 0.0f0
    @assert abs(Cell.spike_margin_from_transition(state, next_state, cache)) <
        Cell.SPIKE_SURROGATE_WIDTH
    return raw, state, input, next_state
end

@testset "canonical local learning configuration and fixed maps" begin
    default_config = Local.LocalLearningConfig()
    copied_config = Local.LocalLearningConfig()
    @test default_config == copied_config
    @test !default_config.plasticity.structure_enabled
    @test default_config.utility_mode === :packet
    @test default_config.plasticity.max_swaps_per_node == 1
    summary = Local.config_summary(default_config)
    @test occursin("feedback_seed=", summary)
    @test occursin("conductance_floor=", summary)
    @test occursin("structure_enabled=false", summary)
    fingerprint = Local.config_fingerprint(default_config)
    @test length(fingerprint) == 64
    @test fingerprint == Local.config_fingerprint(copied_config)
    @test fingerprint != Local.config_fingerprint(Local.LocalLearningConfig(
        feedback_seed=default_config.feedback_seed + 1,
    ))
    @test fingerprint != Local.config_fingerprint(Local.LocalLearningConfig(
        plasticity=Local.PlasticityConfig(target_rate_max=0.2),
    ))
    @test occursin("LocalLearningConfig(", sprint(show, default_config))

    configured = Local.LocalLearningConfig(
        feedback_scale=2.0,
        predictor_scale=0.5,
        predictor_dim=5,
        eligibility_decay=0.8,
        analog_multiplier=0.25,
        hard_event_multiplier=0.1,
        utility_mode=:continuous,
    )
    configured_map = Local.FixedLocalSignalMap(
        22, configured; family=3, cell=17,
    )
    @test size(configured_map.global_feedback) ==
        (Local.LOCAL_OBSERVATION_DIM, 22)
    @test size(configured_map.predictor_feedback) ==
        (Local.LOCAL_OBSERVATION_DIM, 5)
    @test configured.eligibility_decay == 0.8f0

    map_a = Local.FixedLocalSignalMap(
        22, 5; seed=0x91, family=3, cell=17,
    )
    map_copy = Local.FixedLocalSignalMap(
        22, 5; seed=0x91, family=3, cell=17,
    )
    map_seed = Local.FixedLocalSignalMap(
        22, 5; seed=0x92, family=3, cell=17,
    )
    map_cell = Local.FixedLocalSignalMap(
        22, 5; seed=0x91, family=3, cell=18,
    )
    @test map_a.global_feedback == map_copy.global_feedback
    @test map_a.predictor_feedback == map_copy.predictor_feedback
    @test map_a.global_feedback != map_seed.global_feedback
    @test map_a.global_feedback != map_cell.global_feedback
    @test size(map_a.global_feedback) == (Local.LOCAL_OBSERVATION_DIM, 22)
    @test size(map_a.predictor_feedback) == (Local.LOCAL_OBSERVATION_DIM, 5)

    raw_delta = collect(range(-0.3f0, 0.4f0; length=22))
    local_error = collect(range(0.2f0, -0.1f0; length=5))
    signal = zeros(Float32, Local.LOCAL_OBSERVATION_DIM)
    Local.project_learning_signal!(signal, map_a, raw_delta, local_error)
    @test isapprox(
        signal,
        map_a.global_feedback * raw_delta +
            map_a.predictor_feedback * local_error;
        rtol=2eps(Float32),
        atol=2eps(Float32),
    )
    @test_throws DimensionMismatch Local.project_learning_signal!(
        signal, map_a, raw_delta[1:end-1], local_error,
    )
    @test_throws DimensionMismatch Local.project_learning_signal!(
        signal, map_a, raw_delta, local_error[1:end-1],
    )
    @test_throws ArgumentError Local.project_learning_signal!(
        signal, map_a, raw_delta,
    )
    @test_throws ArgumentError Local.FixedLocalSignalMap(0)
    @test_throws ArgumentError Local.FixedLocalSignalMap(22, -1)
    @test_throws ArgumentError Local.PlasticityConfig(firing_ema_decay=1.0)
    @test_throws ArgumentError Local.PlasticityConfig(
        target_rate_min=0.3, target_rate_max=0.2,
    )
    @test_throws ArgumentError Local.PlasticityConfig(
        conductance_floor=1.0, conductance_ceiling=1.0,
    )
    @test_throws ArgumentError Local.PlasticityConfig(utility_decay=1.0)
    @test_throws ArgumentError Local.PlasticityConfig(max_swaps_per_node=2)
    @test_throws ArgumentError Local.LocalLearningConfig(feedback_seed=-1)
    @test_throws ArgumentError Local.LocalLearningConfig(predictor_dim=-1)
    @test_throws ArgumentError Local.LocalLearningConfig(eligibility_decay=1.0)
    @test_throws ArgumentError Local.LocalLearningConfig(analog_multiplier=-1)
    @test_throws ArgumentError Local.LocalLearningConfig(
        hard_event_multiplier=-1,
    )
    @test_throws ArgumentError Local.LocalLearningConfig(utility_mode=:invalid)

    schedule = Local.LearningSchedule(
        analog_interval=1,
        hard_event_interval=2,
        homeostasis_interval=3,
        structure_interval=6,
    )
    clock = Local.LearningClockState()
    due = [Local.advance_clocks!(clock, schedule) for _ in 1:6]
    @test all(item.analog for item in due)
    @test [item.hard_event for item in due] ==
        Bool[false, true, false, true, false, true]
    @test [item.homeostasis for item in due] ==
        Bool[false, false, true, false, false, true]
    @test [item.structure for item in due] ==
        Bool[false, false, false, false, false, true]
    @test clock.analog_ticks == 6
    @test clock.hard_event_ticks == 3
    @test clock.homeostasis_ticks == 2
    @test clock.structure_ticks == 1
    @test_throws ArgumentError Local.LearningSchedule(analog_interval=0)
end

@testset "two-pass ListNet replay boundary" begin
    replay = Local.TwoPassListNetReplay(22, 3)
    Local.record_teacher_free_forward!(replay, 1, 0x101)
    Local.record_teacher_free_forward!(replay, 2, 0x202)
    delta = reshape(collect(Float32, 1:66), 22, 3)
    @test_throws ArgumentError Local.seal_listnet_deltas!(replay, delta, 3)
    Local.record_teacher_free_forward!(replay, 3, 0x303)
    @test_throws ArgumentError Local.record_teacher_free_forward!(replay, 3, 0x303)
    @test_throws DimensionMismatch Local.seal_listnet_deltas!(
        replay, delta[:, 1:2], 3,
    )
    Local.seal_listnet_deltas!(replay, delta, 3)
    @test replay.phase == Local.DELTAS_SEALED
    destination = zeros(Float32, 22)
    @test_throws ArgumentError Local.copy_replay_delta!(
        destination, replay, 1, 0x999,
    )
    Local.copy_replay_delta!(destination, replay, 1, 0x101)
    @test destination == delta[:, 1]
    @test_throws ArgumentError Local.copy_replay_delta!(
        destination, replay, 1, 0x101,
    )
    @test_throws ArgumentError Local.finish_replay!(replay)
    Local.copy_replay_delta!(destination, replay, 2, 0x202)
    Local.copy_replay_delta!(destination, replay, 3, 0x303)
    Local.finish_replay!(replay)
    @test replay.phase == Local.REPLAY_COMPLETE
    @test_throws ArgumentError Local.record_teacher_free_forward!(
        replay, 1, 0x101,
    )

    Local.reset_replay!(replay)
    @test replay.phase == Local.COLLECTING_FORWARD
    @test replay.candidate_count == 0
    @test all(iszero, replay.raw_delta)
    @test_throws BoundsError Local.record_teacher_free_forward!(replay, 0, 0x1)
    @test_throws BoundsError Local.record_teacher_free_forward!(replay, 4, 0x1)
    @test_throws ArgumentError Local.TwoPassListNetReplay(0, 3)
    @test_throws ArgumentError Local.TwoPassListNetReplay(22, 0)
end

@testset "teacher-free multi-compartment causal eligibility" begin
    raw, state, input, next_state = nonspiking_transition()
    parameter_count = Cell.PARAM_DIM + 3
    basis = Local.LocalParameterBasis(parameter_count)
    # Three graph/contact parameters drive typed AMPA, NMDA and GABA inputs.
    basis.input_basis[Cell.input_index(1, Cell.INPUT_AMPA), Cell.PARAM_DIM + 1] = 0.7f0
    basis.input_basis[Cell.input_index(2, Cell.INPUT_NMDA), Cell.PARAM_DIM + 2] = 0.6f0
    basis.input_basis[Cell.input_index(3, Cell.INPUT_GABA), Cell.PARAM_DIM + 3] = 0.5f0

    analog = Local.AnalogEligibilityState(parameter_count)
    event = Local.HardEventEligibilityState(parameter_count)
    scratch = Local.EligibilityScratch(parameter_count)
    @test Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=true,
    )
    @test analog.touched
    @test event.touched
    @test analog.transition_count == 1
    @test event.transition_count == 1
    @test next_state[Cell.SPIKE_INDEX] == 0.0f0

    for field in (
        Cell.FIELD_VOLTAGE,
        Cell.FIELD_AMPA,
        Cell.FIELD_NMDA,
        Cell.FIELD_GABA,
        Cell.FIELD_PLATEAU,
    )
        rows = [Cell.state_index(compartment, field) for
            compartment in 1:Cell.N_COMPARTMENTS]
        @test norm(@view(analog.eligibility[rows, :])) > 0.0f0
    end
    @test norm(@view(analog.eligibility[
        (Local.LOCAL_OBSERVATION_DIM - 1):Local.LOCAL_OBSERVATION_DIM,
        :,
    ])) > 0.0f0
    @test all(column -> norm(@view(analog.eligibility[:, column])) > 0.0f0,
        (Cell.PARAM_DIM + 1):parameter_count)
    @test norm(event.trace) > 0.0f0
    @test size(analog.packet_eligibility) == (Axon.PACKET_DIM, parameter_count)
    @test all(column ->
        norm(@view(analog.packet_eligibility[:, column])) > 0.0f0,
        (Cell.PARAM_DIM + 1):parameter_count,
    )
    first_event_trace = copy(event.trace)

    # Eligibility carries through local temporal dynamics without reading a
    # teacher. A second replayed transition changes the recurrent sensitivity.
    first_eligibility = copy(analog.eligibility)
    cache = Cell.transform_parameters(raw)
    next_next_state = Cell.cell_step_cached_functional(next_state, input, cache)
    Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        next_state,
        input,
        raw,
        next_next_state;
        touched=true,
    )
    @test analog.transition_count == 2
    @test analog.eligibility != first_eligibility

    map = Local.FixedLocalSignalMap(22; seed=0x1234, family=4, cell=9)
    raw_delta = collect(range(-0.25f0, 0.35f0; length=22))
    learning_signal = zeros(Float32, Local.LOCAL_OBSERVATION_DIM)
    Local.project_learning_signal!(learning_signal, map, raw_delta)
    eligibility_before_modulation = copy(analog.eligibility)

    analog_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        analog_gradient, learning_signal, analog,
    )
    @test norm(analog_gradient) > 0.0f0
    scaled_config = Local.LocalLearningConfig(
        analog_multiplier=0.25,
        hard_event_multiplier=0.5,
        utility_mode=:packet,
        plasticity=Local.PlasticityConfig(utility_decay=0.9),
    )
    scaled_analog_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        scaled_analog_gradient, learning_signal, analog, scaled_config,
    )
    @test isapprox(
        scaled_analog_gradient,
        0.25f0 .* analog_gradient;
        rtol=2eps(Float32),
        atol=2eps(Float32),
    )
    @test analog.eligibility == eligibility_before_modulation
    changed_map = Local.FixedLocalSignalMap(
        22; seed=0x1235, family=4, cell=9,
    )
    changed_signal = similar(learning_signal)
    Local.project_learning_signal!(changed_signal, changed_map, raw_delta)
    changed_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        changed_gradient, changed_signal, analog,
    )
    @test changed_gradient != analog_gradient
    packet_map = Local.FixedLocalSignalMap(
        22; observation_dim=Axon.PACKET_DIM, seed=0x512, family=4, cell=9,
    )
    packet_signal = zeros(Float32, Axon.PACKET_DIM)
    Local.project_learning_signal!(packet_signal, packet_map, raw_delta)
    packet_gradient = zeros(Float32, parameter_count)
    Local.accumulate_packet_gradient!(
        packet_gradient, packet_signal, analog,
    )
    @test norm(packet_gradient) > 0.0f0

    # M=0 and eligibility=0 are independent causal stops.
    zero_modulation_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        zero_modulation_gradient,
        zeros(Float32, Local.LOCAL_OBSERVATION_DIM),
        analog,
    )
    @test all(iszero, zero_modulation_gradient)

    zero_eligibility = Local.AnalogEligibilityState(parameter_count)
    zero_eligibility.touched = true
    @test all(iszero, zero_eligibility.packet_eligibility)
    zero_trace_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        zero_trace_gradient, learning_signal, zero_eligibility,
    )
    @test all(iszero, zero_trace_gradient)

    # Hard-event control is a disjoint estimator and clock.
    event_for_control = Local.HardEventEligibilityState{Float32}(
        first_event_trace, true, 1,
    )
    event_gradient = zeros(Float32, parameter_count)
    Local.accumulate_hard_event_gradient!(
        event_gradient, 0.75f0, event_for_control,
    )
    @test norm(event_gradient) > 0.0f0
    scaled_event_gradient = zeros(Float32, parameter_count)
    Local.accumulate_hard_event_gradient!(
        scaled_event_gradient,
        0.75f0,
        event_for_control,
        scaled_config,
    )
    @test isapprox(
        scaled_event_gradient,
        0.5f0 .* event_gradient;
        rtol=2eps(Float32),
        atol=2eps(Float32),
    )
    @test event_gradient != analog_gradient
    event_stopped = zeros(Float32, parameter_count)
    Local.accumulate_hard_event_gradient!(
        event_stopped, 0.0f0, event_for_control,
    )
    @test all(iszero, event_stopped)

    # Structural utility depends on both the third factor and eligibility.
    utility = Local.StructuralUtilityState(parameter_count)
    Local.update_structural_utility!(utility, learning_signal, analog; decay=0.9f0)
    @test any(>(0.0f0), utility.utility)
    utility_zero_m = Local.StructuralUtilityState(parameter_count)
    Local.update_structural_utility!(
        utility_zero_m,
        zeros(Float32, Local.LOCAL_OBSERVATION_DIM),
        analog;
        decay=0.9f0,
    )
    @test all(iszero, utility_zero_m.utility)
    utility_zero_e = Local.StructuralUtilityState(parameter_count)
    Local.update_structural_utility!(
        utility_zero_e, learning_signal, zero_eligibility; decay=0.9f0,
    )
    @test all(iszero, utility_zero_e.utility)
    packet_utility = Local.StructuralUtilityState(parameter_count)
    Local.update_packet_structural_utility!(
        packet_utility, packet_signal, analog; decay=0.9f0,
    )
    @test any(>(0.0f0), packet_utility.utility)
    configured_utility = Local.StructuralUtilityState(parameter_count)
    Local.update_configured_utility!(
        configured_utility, packet_signal, analog, scaled_config,
    )
    @test configured_utility.utility == packet_utility.utility

    # A fresh unvisited cell remains exactly unchanged; no synthetic task
    # update is created merely because another cell was active.
    unvisited_analog = Local.AnalogEligibilityState(parameter_count)
    unvisited_event = Local.HardEventEligibilityState(parameter_count)
    unvisited_scratch = Local.EligibilityScratch(parameter_count)
    @test !Local.accumulate_active_apical_transition!(
        unvisited_analog,
        unvisited_event,
        unvisited_scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=false,
    )
    @test !unvisited_analog.touched
    @test !unvisited_event.touched
    @test all(iszero, unvisited_analog.eligibility)
    @test all(iszero, unvisited_analog.packet_eligibility)
    @test all(iszero, unvisited_event.trace)
    unvisited_gradient = zeros(Float32, parameter_count)
    Local.accumulate_analog_gradient!(
        unvisited_gradient, learning_signal, unvisited_analog,
    )
    @test all(iszero, unvisited_gradient)

    Local.reset_eligibility!(analog, event)
    @test !analog.touched
    @test !event.touched
    @test analog.transition_count == 0
    @test event.transition_count == 0
    @test all(iszero, analog.state_sensitivity)
    @test all(iszero, analog.eligibility)
    @test all(iszero, analog.packet_eligibility)
    @test all(iszero, event.trace)
end

@testset "continuous observation and bounds" begin
    raw, state, input, next_state = nonspiking_transition()
    cache = Cell.transform_parameters(raw)
    observation = zeros(Float32, Local.LOCAL_OBSERVATION_DIM)
    Local.continuous_observation!(observation, state, next_state, cache)
    @test observation[1:45] == next_state[1:45]
    @test observation[end] == next_state[Cell.ADAPTATION_INDEX]
    @test observation[end - 1] ==
        Cell.spike_margin_from_transition(state, next_state, cache)
    @test_throws DimensionMismatch Local.continuous_observation!(
        observation[1:end-1], state, next_state, cache,
    )

    @test_throws ArgumentError Local.LocalParameterBasis(0)
    @test_throws ArgumentError Local.LocalParameterBasis(
        Cell.PARAM_DIM - 1,
    )
    @test_throws ArgumentError Local.AnalogEligibilityState(0)
    @test_throws ArgumentError Local.HardEventEligibilityState(0)
    @test_throws ArgumentError Local.StructuralUtilityState(0)
    @test_throws ArgumentError Local.EligibilityScratch(0)

    basis = Local.LocalParameterBasis(Cell.PARAM_DIM)
    analog = Local.AnalogEligibilityState(Cell.PARAM_DIM)
    event = Local.HardEventEligibilityState(Cell.PARAM_DIM)
    scratch = Local.EligibilityScratch(Cell.PARAM_DIM)
    @test_throws ArgumentError Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=true,
        event_decay=1.1f0,
    )
    @test_throws DimensionMismatch Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state[1:end-1],
        input,
        raw,
        next_state;
        touched=true,
    )

    # The local Jacobian reference is expensive but fixed-memory. This is the
    # correctness kernel against which the later packet-specialized SIMD
    # implementation is compared.
    Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=true,
    )
    Local.reset_eligibility!(analog, event)
    allocated = @allocated Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=true,
    )
    @test allocated == 0
end

@testset "two-step continuous eligibility finite difference" begin
    raw32, state32, input32, _ = nonspiking_transition()
    raw = Float64.(raw32)
    state = Float64.(state32)
    input = Float64.(input32)
    input .*= 0.5
    raw_parameter = findfirst(==(:basal_to_soma), Cell.PARAMETER_NAMES)
    input_channel = Cell.input_index(1, Cell.INPUT_AMPA)

    basis = Local.LocalParameterBasis(
        2; T=Float64, include_cell_parameters=false,
    )
    basis.raw_basis[raw_parameter, 1] = 1.0
    basis.input_basis[input_channel, 2] = 0.7
    analog = Local.AnalogEligibilityState(2; T=Float64)
    event = Local.HardEventEligibilityState(2; T=Float64)
    scratch = Local.EligibilityScratch(2; T=Float64)
    cache = Cell.transform_parameters(raw)
    next_state = Cell.cell_step_cached_functional(state, input, cache)
    next_next_state = Cell.cell_step_cached_functional(next_state, input, cache)
    Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        state,
        input,
        raw,
        next_state;
        touched=true,
    )
    Local.accumulate_active_apical_transition!(
        analog,
        event,
        scratch,
        basis,
        next_state,
        input,
        raw,
        next_next_state;
        touched=true,
    )

    function two_step_observation(raw_offset, input_offset)
        candidate_raw = copy(raw)
        candidate_raw[raw_parameter] += raw_offset
        candidate_input = copy(input)
        candidate_input[input_channel] += 0.7 * input_offset
        candidate_cache = Cell.transform_parameters(candidate_raw)
        first = Cell.cell_step_cached_functional(
            state, candidate_input, candidate_cache,
        )
        second = Cell.cell_step_cached_functional(
            first, candidate_input, candidate_cache,
        )
        @assert first[Cell.SPIKE_INDEX] == next_state[Cell.SPIKE_INDEX]
        @assert second[Cell.SPIKE_INDEX] == next_next_state[Cell.SPIKE_INDEX]
        observation = zeros(Float64, Local.LOCAL_OBSERVATION_DIM)
        Local.continuous_observation!(
            observation, first, second, candidate_cache,
        )
        return observation
    end

    function two_step_packet(raw_offset, input_offset)
        candidate_raw = copy(raw)
        candidate_raw[raw_parameter] += raw_offset
        candidate_input = copy(input)
        candidate_input[input_channel] += 0.7 * input_offset
        candidate_cache = Cell.transform_parameters(candidate_raw)
        first = Cell.cell_step_cached_functional(
            state, candidate_input, candidate_cache,
        )
        second = Cell.cell_step_cached_functional(
            first, candidate_input, candidate_cache,
        )
        @assert first[Cell.SPIKE_INDEX] == next_state[Cell.SPIKE_INDEX]
        @assert second[Cell.SPIKE_INDEX] == next_next_state[Cell.SPIKE_INDEX]
        packet = zeros(Float64, Axon.PACKET_DIM)
        Axon.axon_packet!(packet, first, second, candidate_cache)
        return packet
    end

    epsilon = 1.0e-5
    raw_numerical = (
        two_step_observation(epsilon, 0.0) -
        two_step_observation(-epsilon, 0.0)
    ) / (2epsilon)
    input_numerical = (
        two_step_observation(0.0, epsilon) -
        two_step_observation(0.0, -epsilon)
    ) / (2epsilon)
    packet_raw_numerical = (
        two_step_packet(epsilon, 0.0) -
        two_step_packet(-epsilon, 0.0)
    ) / (2epsilon)
    packet_input_numerical = (
        two_step_packet(0.0, epsilon) -
        two_step_packet(0.0, -epsilon)
    ) / (2epsilon)
    @test isapprox(
        @view(analog.eligibility[:, 1]),
        raw_numerical;
        rtol=8.0e-4,
        atol=3.0e-7,
    )
    @test isapprox(
        @view(analog.eligibility[:, 2]),
        input_numerical;
        rtol=8.0e-4,
        atol=3.0e-7,
    )
    @test isapprox(
        @view(analog.packet_eligibility[:, 1]),
        packet_raw_numerical;
        rtol=1.5e-3,
        atol=3.0e-7,
    )
    @test isapprox(
        @view(analog.packet_eligibility[:, 2]),
        packet_input_numerical;
        rtol=1.5e-3,
        atol=3.0e-7,
    )
    @test next_state[Cell.SPIKE_INDEX] == 0.0
    @test next_next_state[Cell.SPIKE_INDEX] == 1.0
end
