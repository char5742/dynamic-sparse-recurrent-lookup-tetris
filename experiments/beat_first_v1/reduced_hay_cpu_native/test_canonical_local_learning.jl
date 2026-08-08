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
    @test default_config.utility_mode === :combined
    @test default_config.plasticity.max_swaps_per_node == 1
    summary = Local.config_summary(default_config)
    @test occursin("feedback_seed=", summary)
    @test occursin("conductance_floor=", summary)
    @test occursin("structure_enabled=false", summary)
    @test occursin("utility_mode=combined", summary)
    fingerprint = Local.config_fingerprint(default_config)
    @test length(fingerprint) == 64
    @test fingerprint == Local.config_fingerprint(copied_config)
    @test fingerprint != Local.config_fingerprint(Local.LocalLearningConfig(
        feedback_seed=default_config.feedback_seed + 1,
    ))
    @test fingerprint != Local.config_fingerprint(Local.LocalLearningConfig(
        plasticity=Local.PlasticityConfig(target_rate_max=0.2),
    ))
    @test fingerprint != Local.config_fingerprint(Local.LocalLearningConfig(
        utility_mode=:none,
    ))
    @test occursin("LocalLearningConfig(", sprint(show, default_config))

    configured = Local.LocalLearningConfig(
        feedback_scale=2.0,
        predictor_scale=0.5,
        predictor_dim=5,
        eligibility_decay=0.8,
        analog_multiplier=0.25,
        hard_event_multiplier=0.1,
        utility_mode=:none,
    )
    configured_map = Local.FixedLocalSignalMap(
        22, configured; family=3, cell=17,
    )
    @test size(configured_map.global_feedback) ==
        (Local.LOCAL_OBSERVATION_DIM, 22)
    @test size(configured_map.predictor_feedback) ==
        (Local.LOCAL_OBSERVATION_DIM, 5)
    @test configured.eligibility_decay == 0.8f0
    @test configured.utility_mode === :none

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
    @test_throws ArgumentError Local.LocalLearningConfig(utility_mode=:packet)
    @test_throws ArgumentError Local.LocalLearningConfig(utility_mode=:continuous)
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

clock_snapshot(clock) = (
    clock.update,
    clock.analog_ticks,
    clock.hard_event_ticks,
    clock.homeostasis_ticks,
    clock.structure_ticks,
)

@testset "transaction-safe learning clocks" begin
    schedule = Local.LearningSchedule(
        analog_interval=2,
        hard_event_interval=3,
        homeostasis_interval=4,
        structure_interval=6,
    )
    clock = Local.LearningClockState()
    before_preview = clock_snapshot(clock)
    due = Local.preview_clocks(clock, schedule)
    @test clock_snapshot(clock) == before_preview
    @test due.expected_update == 1
    @test due.expected_ticks == (0, 0, 0, 0)
    @test due.schedule_intervals == (2, 3, 4, 6)
    @test due == Local.preview_clocks(clock, schedule)

    committed = Local.commit_clocks!(clock, schedule, due)
    @test committed == due
    @test clock_snapshot(clock) == (1, 0, 0, 0, 0)
    after_commit = clock_snapshot(clock)
    @test_throws ArgumentError Local.commit_clocks!(clock, schedule, due)
    @test clock_snapshot(clock) == after_commit

    next_due = Local.preview_clocks(clock, schedule)
    wrong_due = Local.DuePlasticityClocks(
        !next_due.analog,
        next_due.hard_event,
        next_due.homeostasis,
        next_due.structure,
        next_due.expected_update,
        next_due.expected_ticks,
        next_due.schedule_intervals,
    )
    before_mismatch = clock_snapshot(clock)
    @test_throws ArgumentError Local.commit_clocks!(clock, schedule, wrong_due)
    @test clock_snapshot(clock) == before_mismatch
    @test_throws ArgumentError Local.commit_clocks!(
        clock,
        Local.LearningSchedule(
            analog_interval=2,
            hard_event_interval=3,
            homeostasis_interval=5,
            structure_interval=6,
        ),
        next_due,
    )
    @test clock_snapshot(clock) == before_mismatch
    @test_throws ArgumentError Local.commit_clocks!(
        clock,
        schedule,
        Local.DuePlasticityClocks(false, false, false, false),
    )
    @test clock_snapshot(clock) == before_mismatch

    overflow_update = Local.LearningClockState(typemax(Int), 3, 4, 5, 6)
    overflow_update_before = clock_snapshot(overflow_update)
    @test_throws OverflowError Local.preview_clocks(overflow_update, schedule)
    @test clock_snapshot(overflow_update) == overflow_update_before

    overflow_tick = Local.LearningClockState(0, typemax(Int), 4, 5, 6)
    every_update = Local.LearningSchedule(
        analog_interval=1,
        hard_event_interval=2,
        homeostasis_interval=3,
        structure_interval=4,
    )
    overflow_tick_before = clock_snapshot(overflow_tick)
    @test_throws OverflowError Local.preview_clocks(overflow_tick, every_update)
    @test clock_snapshot(overflow_tick) == overflow_tick_before
    stale_overflow_due = Local.DuePlasticityClocks(
        true,
        false,
        false,
        false,
        1,
        (typemax(Int), 4, 5, 6),
        (1, 2, 3, 4),
    )
    @test_throws OverflowError Local.commit_clocks!(
        overflow_tick, every_update, stale_overflow_due,
    )
    @test clock_snapshot(overflow_tick) == overflow_tick_before

    hot_clock = Local.LearningClockState()
    Local.preview_clocks(hot_clock, schedule)
    preview_allocated = @allocated Local.preview_clocks(hot_clock, schedule)
    @test preview_allocated == 0
    hot_due = Local.preview_clocks(hot_clock, schedule)
    commit_allocated = @allocated Local.commit_clocks!(
        hot_clock, schedule, hot_due,
    )
    @test commit_allocated == 0
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

function make_local_trajectory(step_count::Int; T=Float64)
    raw32, initial32, input32, _ = nonspiking_transition()
    raw = T.(raw32)
    initial = T.(initial32)
    input = T.(input32)
    input .*= T(0.45)
    cache = Cell.transform_parameters(raw)
    states = Vector{Vector{T}}(undef, step_count + 1)
    inputs = Vector{Vector{T}}(undef, step_count)
    states[1] = initial
    for step in 1:step_count
        inputs[step] = copy(input)
        states[step + 1] = Cell.cell_step_cached_functional(
            states[step], inputs[step], cache,
        )
    end
    continuous = [
        T[sin(T(0.07) * T(step + lane)) for
            lane in 1:Local.LOCAL_OBSERVATION_DIM]
        for step in 1:step_count
    ]
    packet = [
        T[cos(T(0.11) * T(2step + lane)) for lane in 1:Axon.PACKET_DIM]
        for step in 1:step_count
    ]
    return raw, states, inputs, continuous, packet
end

# Test-only forward full-eligibility oracle. Production never constructs these
# matrices; this intentionally expensive reference proves the contraction
# identity against the former RTRL-style implementation.
function full_eligibility_contraction(
    raw,
    states,
    inputs,
    continuous_signals,
    packet_signals,
    raw_parameter,
    input_channel,
    input_scale,
)
    T = eltype(raw)
    parameter_count = 2
    state_sensitivity = zeros(T, Cell.STATE_DIM, parameter_count)
    gradient = zeros(T, parameter_count)
    cache, derivative_cache = Cell.parameter_caches(raw)
    dstate = zeros(T, Cell.STATE_DIM)
    dinput = zeros(T, Cell.INPUT_DIM)
    draw = zeros(T, Cell.PARAM_DIM)
    dnext = zeros(T, Cell.STATE_DIM)
    state_jacobian = zeros(T, Cell.STATE_DIM, Cell.STATE_DIM)
    input_jacobian = zeros(T, Cell.STATE_DIM, Cell.INPUT_DIM)
    raw_jacobian = zeros(T, Cell.STATE_DIM, Cell.PARAM_DIM)
    next_sensitivity = similar(state_sensitivity)
    margin_sensitivity = zeros(T, parameter_count)
    continuous_eligibility = zeros(T, Local.LOCAL_OBSERVATION_DIM, parameter_count)
    packet_eligibility = zeros(T, Axon.PACKET_DIM, parameter_count)
    packet_bar = zeros(T, Axon.PACKET_DIM)
    packet_dnext = zeros(T, Cell.STATE_DIM)

    for step in eachindex(inputs)
        previous_state = states[step]
        next_state = states[step + 1]
        input = inputs[step]
        fill!(state_jacobian, zero(T))
        fill!(input_jacobian, zero(T))
        fill!(raw_jacobian, zero(T))
        for output in 1:(Cell.STATE_DIM - 1)
            fill!(dnext, zero(T))
            dnext[output] = one(T)
            Cell.cell_step_conditional_pullback!(
                dstate, dinput, draw, previous_state, input, cache,
                derivative_cache, next_state, dnext,
            )
            state_jacobian[output, :] .= dstate
            input_jacobian[output, :] .= dinput
            raw_jacobian[output, :] .= draw
        end
        mul!(next_sensitivity, state_jacobian, state_sensitivity)
        next_sensitivity[:, 1] .+= @view(raw_jacobian[:, raw_parameter])
        next_sensitivity[:, 2] .+=
            input_scale .* @view(input_jacobian[:, input_channel])

        fill!(dnext, zero(T))
        Cell.cell_step_conditional_pullback!(
            dstate, dinput, draw, previous_state, input, cache,
            derivative_cache, next_state, dnext, zero(T), zero(T), one(T),
        )
        margin_sensitivity[1] =
            dot(dstate, @view(state_sensitivity[:, 1])) + draw[raw_parameter]
        margin_sensitivity[2] =
            dot(dstate, @view(state_sensitivity[:, 2])) +
            input_scale * dinput[input_channel]

        continuous_count = Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM
        continuous_eligibility[1:continuous_count, :] .=
            @view(next_sensitivity[1:continuous_count, :])
        continuous_eligibility[end - 1, :] .= margin_sensitivity
        continuous_eligibility[end, :] .=
            @view(next_sensitivity[Cell.ADAPTATION_INDEX, :])

        for lane in 1:Axon.PACKET_DIM
            fill!(packet_bar, zero(T))
            packet_bar[lane] = one(T)
            margin_bar = Axon.axon_packet_pullback!(
                packet_dnext, packet_bar, previous_state, next_state, cache,
            )
            for parameter in 1:parameter_count
                packet_eligibility[lane, parameter] =
                    dot(packet_dnext, @view(next_sensitivity[:, parameter])) +
                    margin_bar * margin_sensitivity[parameter]
            end
        end
        mul!(
            gradient,
            transpose(continuous_eligibility),
            continuous_signals[step],
            one(T),
            one(T),
        )
        mul!(
            gradient,
            transpose(packet_eligibility),
            packet_signals[step],
            one(T),
            one(T),
        )
        copyto!(state_sensitivity, next_sensitivity)
    end
    return gradient
end

function contracted_gradient(
    raw,
    states,
    inputs,
    continuous_signals,
    packet_signals;
    eligibility_scale=one(eltype(raw)),
    touched=true,
    terminal_seed=nothing,
)
    T = eltype(raw)
    step_count = length(inputs)
    adjoint = Local.ContractedLocalAdjoint(; T=T)
    scratch = Local.ContractedAdjointScratch(; T=T)
    Local.begin_local_adjoint!(
        adjoint,
        step_count;
        terminal_seed=terminal_seed,
    )
    raw_gradient = zeros(T, Cell.PARAM_DIM)
    input_gradient = zeros(T, Cell.INPUT_DIM)
    for step in step_count:-1:1
        changed = Local.contract_replayed_transition!(
            adjoint,
            scratch,
            Local.ChronologicalTransitionLink(step, step - 1, 100 + step),
            states[step],
            inputs[step],
            raw,
            states[step + 1],
            continuous_signals[step],
            packet_signals[step];
            touched=touched,
            eligibility_scale=T(eligibility_scale),
        )
        if changed
            raw_gradient .+= Local.raw_parameter_cotangent(scratch)
            input_gradient .+= Local.input_cotangent(scratch)
        end
    end
    if touched
        root_bar = copy(Local.finish_local_adjoint!(adjoint, 0))
    else
        root_bar = copy(adjoint.state_bar)
    end
    return raw_gradient, input_gradient, root_bar, adjoint
end

@testset "factorized local-adjoint equals full eligibility contraction" begin
    raw_parameter = findfirst(==(:basal_to_soma), Cell.PARAMETER_NAMES)
    input_channel = Cell.input_index(1, Cell.INPUT_AMPA)
    input_scale = 0.7
    for step_count in (1, 2, 4)
        raw, states, inputs, continuous, packet =
            make_local_trajectory(step_count)
        oracle = full_eligibility_contraction(
            raw,
            states,
            inputs,
            continuous,
            packet,
            raw_parameter,
            input_channel,
            input_scale,
        )
        raw_gradient, input_gradient, _, adjoint = contracted_gradient(
            raw, states, inputs, continuous, packet,
        )
        contracted = [
            raw_gradient[raw_parameter],
            input_scale * input_gradient[input_channel],
        ]
        @test isapprox(contracted, oracle; rtol=2.0e-11, atol=2.0e-12)
        @test adjoint.visited_transition_count == step_count
        @test adjoint.conditional_pullback_count == step_count
        @test !adjoint.active
    end
end

@testset "teacher-free causal gates, nonspiking credit, and chronology" begin
    raw32, state32, input32, next32 = nonspiking_transition()
    raw = Float64.(raw32)
    state = Float64.(state32)
    input = Float64.(input32)
    next_state = Float64.(next32)
    continuous = [collect(range(-0.2, 0.3; length=Local.LOCAL_OBSERVATION_DIM))]
    packet = [collect(range(0.15, -0.1; length=Axon.PACKET_DIM))]
    states = [state, next_state]
    inputs = [input]
    @test next_state[Cell.SPIKE_INDEX] == 0.0

    raw_gradient, input_gradient, root_bar, adjoint = contracted_gradient(
        raw, states, inputs, continuous, packet,
    )
    @test norm(raw_gradient) > 0
    @test norm(input_gradient) > 0
    @test norm(root_bar) > 0
    @test adjoint.touched

    zeros_continuous = [zeros(Float64, Local.LOCAL_OBSERVATION_DIM)]
    zeros_packet = [zeros(Float64, Axon.PACKET_DIM)]
    zero_m_raw, zero_m_input, zero_m_root, _ = contracted_gradient(
        raw, states, inputs, zeros_continuous, zeros_packet,
    )
    @test all(iszero, zero_m_raw)
    @test all(iszero, zero_m_input)
    @test all(iszero, zero_m_root)

    zero_e_raw, zero_e_input, zero_e_root, _ = contracted_gradient(
        raw,
        states,
        inputs,
        continuous,
        packet;
        eligibility_scale=0.0,
    )
    @test all(iszero, zero_e_raw)
    @test all(iszero, zero_e_input)
    @test all(iszero, zero_e_root)

    unvisited_raw, unvisited_input, unvisited_root, unvisited =
        contracted_gradient(
            raw, states, inputs, continuous, packet; touched=false,
        )
    @test all(iszero, unvisited_raw)
    @test all(iszero, unvisited_input)
    @test all(iszero, unvisited_root)
    @test !unvisited.touched
    @test unvisited.visited_transition_count == 0
    @test unvisited.conditional_pullback_count == 0

    adjoint_order = Local.ContractedLocalAdjoint(; T=Float64)
    scratch = Local.ContractedAdjointScratch(; T=Float64)
    Local.begin_local_adjoint!(adjoint_order, 1)
    @test_throws ArgumentError Local.contract_replayed_transition!(
        adjoint_order,
        scratch,
        Local.ChronologicalTransitionLink(2, 1, 202),
        state,
        input,
        raw,
        next_state,
        continuous[1],
        packet[1];
        touched=true,
    )
    @test_throws ArgumentError Local.finish_local_adjoint!(adjoint_order, 0)
    @test Local.ChronologicalTransitionLink(5, 3, 91).packet_version == 91
    @test_throws ArgumentError Local.ChronologicalTransitionLink(0, 0)
    @test_throws ArgumentError Local.ChronologicalTransitionLink(2, 2)

    Local.begin_local_adjoint!(adjoint_order, 1)
    connected = Local.CausalEventControl(0.2; connected=true)
    @test_throws ArgumentError Local.contract_replayed_transition!(
        adjoint_order,
        scratch,
        Local.ChronologicalTransitionLink(1, 0),
        state,
        input,
        raw,
        next_state,
        continuous[1],
        packet[1];
        touched=true,
        event_control=connected,
    )
    disconnected_nonzero = Local.CausalEventControl(0.2; connected=false)
    @test_throws ArgumentError Local.contract_replayed_transition!(
        adjoint_order,
        scratch,
        Local.ChronologicalTransitionLink(1, 0),
        state,
        input,
        raw,
        next_state,
        continuous[1],
        packet[1];
        touched=true,
        event_control=disconnected_nonzero,
    )
end

@testset "linearity and generic seeded-adjoint duality" begin
    raw, states, inputs, continuous_a, packet_a = make_local_trajectory(2)
    continuous_b = [0.37 .* signal for signal in continuous_a]
    packet_b = [-0.21 .* signal for signal in packet_a]
    continuous_sum = [continuous_a[i] .+ continuous_b[i] for i in 1:2]
    packet_sum = [packet_a[i] .+ packet_b[i] for i in 1:2]
    raw_a, input_a, root_a, _ = contracted_gradient(
        raw, states, inputs, continuous_a, packet_a,
    )
    raw_b, input_b, root_b, _ = contracted_gradient(
        raw, states, inputs, continuous_b, packet_b,
    )
    raw_sum, input_sum, root_sum, _ = contracted_gradient(
        raw, states, inputs, continuous_sum, packet_sum,
    )
    @test isapprox(raw_sum, raw_a + raw_b; rtol=3eps(Float64), atol=3eps(Float64))
    @test isapprox(
        input_sum, input_a + input_b; rtol=3eps(Float64), atol=3eps(Float64),
    )
    @test isapprox(root_sum, root_a + root_b; rtol=3eps(Float64), atol=3eps(Float64))

    # Generic/oracle seed duality only. Canonical Graph candidate worlds stop
    # at initial_core and never use this root-to-common propagation; canonical
    # common instead receives aggregate raw-22D fixed feedback once per state.
    common_raw, common_states, common_inputs, _, _ = make_local_trajectory(1)
    zero_continuous = [zeros(Float64, Local.LOCAL_OBSERVATION_DIM)]
    zero_packet = [zeros(Float64, Axon.PACKET_DIM)]
    seed_a = root_a
    seed_b = root_b
    raw_common_a, input_common_a, _, _ = contracted_gradient(
        common_raw,
        common_states,
        common_inputs,
        zero_continuous,
        zero_packet;
        terminal_seed=seed_a,
    )
    raw_common_b, input_common_b, _, _ = contracted_gradient(
        common_raw,
        common_states,
        common_inputs,
        zero_continuous,
        zero_packet;
        terminal_seed=seed_b,
    )
    combined = seed_a + seed_b
    raw_common_once, input_common_once, _, _ = contracted_gradient(
        common_raw,
        common_states,
        common_inputs,
        zero_continuous,
        zero_packet;
        terminal_seed=combined,
    )
    @test isapprox(
        raw_common_once,
        raw_common_a + raw_common_b;
        rtol=3eps(Float64),
        atol=3eps(Float64),
    )
    @test isapprox(
        input_common_once,
        input_common_a + input_common_b;
        rtol=3eps(Float64),
        atol=3eps(Float64),
    )

    common_seed_state = Local.ContractedLocalAdjoint(; T=Float64)
    Local.begin_local_adjoint!(common_seed_state, 1)
    Local.add_terminal_seed!(common_seed_state, seed_a)
    Local.add_terminal_seed!(common_seed_state, seed_b)
    @test common_seed_state.state_bar == combined
end

@testset "continuous observation, utility, bounds, and allocation" begin
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

    @test_throws ArgumentError Local.StructuralUtilityState(0)
    utility = Local.StructuralUtilityState(3)
    Local.update_structural_utility!(
        utility, Float32[0.2, 0.0, -0.4]; decay=0.9f0, normalization=2.0f0,
    )
    @test utility.utility[1] > 0
    @test utility.utility[2] == 0
    @test utility.utility[3] > utility.utility[1]
    zero_utility = Local.StructuralUtilityState(3)
    Local.update_structural_utility!(
        zero_utility, zeros(Float32, 3); decay=0.9f0,
    )
    @test all(iszero, zero_utility.utility)
    @test_throws DimensionMismatch Local.update_structural_utility!(
        utility, zeros(Float32, 2),
    )

    # Scratch contains fixed cell/packet vectors only, never a P-wide matrix.
    scratch = Local.ContractedAdjointScratch()
    @test all(field -> !(field isa AbstractMatrix),
        (getfield(scratch, name) for name in fieldnames(typeof(scratch))))
    @test length(scratch.draw) == Cell.PARAM_DIM
    @test length(scratch.dinput) == Cell.INPUT_DIM

    continuous = collect(range(-0.2f0, 0.3f0; length=Local.LOCAL_OBSERVATION_DIM))
    packet = collect(range(0.15f0, -0.1f0; length=Axon.PACKET_DIM))
    adjoint = Local.ContractedLocalAdjoint()
    event_control = Local.CausalEventControl()
    link = Local.ChronologicalTransitionLink(1, 0, 71)
    Local.begin_local_adjoint!(adjoint, 1)
    Local.contract_replayed_transition!(
        adjoint,
        scratch,
        link,
        state,
        input,
        raw,
        next_state,
        continuous,
        packet;
        touched=true,
        event_control=event_control,
    )
    Local.begin_local_adjoint!(adjoint, 1)
    allocated = @allocated Local.contract_replayed_transition!(
        adjoint,
        scratch,
        link,
        state,
        input,
        raw,
        next_state,
        continuous,
        packet;
        touched=true,
        event_control=event_control,
    )
    @test allocated == 0
    @test adjoint.conditional_pullback_count == 1

    # Production path: generation-stamped 48×N slab, integer column address,
    # precomputed caches, no per-cell view/object and no cache reconstruction.
    arena = Local.ContractedAdjointArena(1436)
    selected = 713
    cache, derivative_cache = Cell.parameter_caches(raw)
    Local.begin_local_adjoint!(arena, selected, 1)
    Local.contract_replayed_transition!(
        arena,
        selected,
        scratch,
        link,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous,
        packet;
        touched=true,
        event_control=event_control,
    )
    Local.finish_local_adjoint!(arena, selected, 0)
    Local.begin_local_adjoint!(arena, selected, 1)
    arena_allocated = @allocated Local.contract_replayed_transition!(
        arena,
        selected,
        scratch,
        link,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous,
        packet;
        touched=true,
        event_control=event_control,
    )
    @test arena_allocated == 0
    @test arena.visited_transition_count[selected] == 1
    @test arena.conditional_pullback_count[selected] == 1
    @test size(arena.state_bar) == (Cell.STATE_DIM, 1436)
    Local.finish_local_adjoint!(arena, selected, 0)
    Local.reset_adjoint_arena!(arena)
    @test_throws ArgumentError Local.finish_local_adjoint!(arena, selected, 0)

    candidate_arena = Local.ContractedAdjointArena(2)
    common_arena = Local.ContractedAdjointArena(2)
    candidate_seed = collect(range(-0.1f0, 0.2f0; length=Cell.STATE_DIM))
    Local.begin_local_adjoint!(
        candidate_arena, 1, 0; terminal_seed=candidate_seed,
    )
    Local.begin_local_adjoint!(common_arena, 1, 0)
    Local.add_terminal_seed!(common_arena, 1, candidate_arena, 1)
    @test common_arena.state_bar[:, 1] == candidate_seed
    fill!(@view(common_arena.state_bar[:, 1]), 0.0f0)
    seed_reduce_allocated = @allocated Local.add_terminal_seed!(
        common_arena, 1, candidate_arena, 1,
    )
    @test seed_reduce_allocated == 0

    @test !isdefined(Local, :AnalogEligibilityState)
    @test !isdefined(Local, :HardEventEligibilityState)
    @test !isdefined(Local, :LocalParameterBasis)
    @test !isdefined(Local, :EligibilityScratch)
    @test Local.LocalLearningConfig().predictor_dim == 0
    @test Local.LocalLearningConfig().hard_event_multiplier == 0.0f0
end
