using Test

module PayloadTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "StateCodec.jl"))
include(joinpath(@__DIR__, "Payload.jl"))
end

const Cell = PayloadTestHarness.ActiveApicalCell
using .PayloadTestHarness.Payload

function normalized_fixture(::Type{T}=Float32) where {T<:AbstractFloat}
    state = zeros(T, Cell.STATE_DIM)
    state[Cell.SOMA_INDEX] = T(0.6)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_NMDA)] = T(0.1 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] =
            T(0.05 * (compartment + 1))
    end
    state[Cell.SPIKE_INDEX] = one(T)
    state[Cell.ADAPTATION_INDEX] = T(0.25)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] =
            T(0.05 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_AMPA)] =
            T(0.03 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_GABA)] =
            T(0.02 * compartment)
    end
    return state
end

function central_difference!(values, index::Int, objective; epsilon=1.0e-6)
    original = values[index]
    values[index] = original + epsilon
    positive = objective()
    values[index] = original - epsilon
    negative = objective()
    values[index] = original
    return (positive - negative) / (2 * epsilon)
end

@testset "compartment/receptor payload VJP" begin
    state = normalized_fixture(Float64)
    raw_gains = Float64[0.3, -0.2, 0.7]
    gains = zeros(Float64, ANALOG_GAIN_COUNT)
    derivatives = similar(gains)
    transform_payload_gains!(gains, derivatives, raw_gains)
    channels = zeros(Float64, PAYLOAD_DIM)
    payload_channels_cached_unchecked!(channels, state, gains)
    @test length(channels) == Cell.INPUT_DIM
    @test channels[Cell.input_index(1, Cell.INPUT_AMPA)] !=
          channels[Cell.input_index(2, Cell.INPUT_AMPA)]
    @test channels[Cell.input_index(1, Cell.INPUT_AMPA)] !=
          channels[Cell.input_index(1, Cell.INPUT_NMDA)]

    channel_bar = [0.07 * index - 0.31 for index in 1:PAYLOAD_DIM]
    dstate = zeros(Float64, length(state))
    draw = zeros(Float64, length(raw_gains))
    payload_channels_cached_raw_vjp_unchecked!(
        dstate,
        draw,
        state,
        gains,
        derivatives,
        channel_bar,
    )
    objective = function ()
        local_gains = zeros(Float64, ANALOG_GAIN_COUNT)
        local_derivatives = similar(local_gains)
        transform_payload_gains!(local_gains, local_derivatives, raw_gains)
        local_channels = zeros(Float64, PAYLOAD_DIM)
        payload_channels_cached_unchecked!(local_channels, state, local_gains)
        return sum(local_channels .* channel_bar)
    end
    for index in eachindex(state)
        observed = central_difference!(state, index, objective)
        @test dstate[index] ≈ observed rtol=2.0e-7 atol=2.0e-8
    end
    for index in eachindex(raw_gains)
        observed = central_difference!(raw_gains, index, objective)
        @test draw[index] ≈ observed rtol=2.0e-7 atol=2.0e-8
    end

    payload_channels_cached_unchecked!(channels, state, gains)
    payload_channels_cached_raw_vjp_unchecked!(
        dstate, draw, state, gains, derivatives, channel_bar,
    )
    @test @allocated(payload_channels_cached_unchecked!(channels, state, gains)) == 0
    @test @allocated(payload_channels_cached_raw_vjp_unchecked!(
        dstate, draw, state, gains, derivatives, channel_bar,
    )) == 0
end

@testset "branch-local hard plateau event payload" begin
    state = normalized_fixture(Float64)
    state[Cell.SPIKE_INDEX] = 0.0
    for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = 0.0
    end
    active_compartment = 2
    plateau_index = Cell.state_index(
        active_compartment,
        Cell.FIELD_PLATEAU,
    )
    state[plateau_index] = 0.03
    raw_gains = Float64[0.3, -0.2, 0.7]
    gains = zeros(Float64, ANALOG_GAIN_COUNT)
    derivatives = similar(gains)
    transform_payload_gains!(gains, derivatives, raw_gains)
    channels = zeros(Float64, PAYLOAD_DIM)
    @test has_payload_event(state, 0.0)
    @test payload_channels_event_masked_cached_unchecked!(
        channels,
        state,
        gains,
        0.0,
    )
    for compartment in 1:Cell.N_COMPARTMENTS
        channel_range = (
            Cell.input_index(compartment, 1):
            Cell.input_index(compartment, Cell.INPUT_CHANNELS)
        )
        values = @view channels[channel_range]
        if compartment == active_compartment
            @test any(!iszero, values)
        else
            @test all(iszero, values)
        end
    end

    channel_bar = [0.03 * index - 0.2 for index in 1:PAYLOAD_DIM]
    dstate = zeros(Float64, Cell.STATE_DIM)
    draw = zeros(Float64, ANALOG_GAIN_COUNT)
    scaled_bar = zeros(Float64, PAYLOAD_DIM)
    payload_value = zeros(Float64, PAYLOAD_DIM)
    payload_channels_event_masked_cached_raw_vjp_unchecked!(
        dstate,
        draw,
        state,
        gains,
        derivatives,
        channel_bar,
        0.0,
        scaled_bar,
        payload_value,
    )
    objective = function ()
        transform_payload_gains!(gains, derivatives, raw_gains)
        payload_channels_event_masked_cached_unchecked!(
            channels,
            state,
            gains,
            0.0,
        )
        return sum(channels .* channel_bar)
    end
    for index in (
        Cell.state_index(active_compartment, Cell.FIELD_VOLTAGE),
        Cell.state_index(active_compartment, Cell.FIELD_AMPA),
        Cell.state_index(active_compartment, Cell.FIELD_NMDA),
        Cell.state_index(active_compartment, Cell.FIELD_GABA),
        plateau_index,
    )
        observed = central_difference!(state, index, objective)
        @test dstate[index] ≈ observed rtol=2.0e-7 atol=2.0e-8
    end
    for index in eachindex(raw_gains)
        observed = central_difference!(raw_gains, index, objective)
        @test draw[index] ≈ observed rtol=2.0e-7 atol=2.0e-8
    end
    @test @allocated(payload_channels_event_masked_cached_unchecked!(
        channels, state, gains, 0.0,
    )) == 0
    @test @allocated(payload_channels_event_masked_cached_raw_vjp_unchecked!(
        dstate, draw, state, gains, derivatives, channel_bar, 0.0,
        scaled_bar, payload_value,
    )) == 0

    state[plateau_index] = 0.0
    @test !has_payload_event(state, 0.0)
    @test has_payload_event(state, 0.05)
end

function hot_allocations(
    state::Vector{Float32},
    raw_gains::Vector{Float32},
    gains::Vector{Float32},
    raw_derivatives::Vector{Float32},
    dstate::Vector{Float32},
    draw_gains::Vector{Float32},
)
    payload_amplitude_raw(state, raw_gains)
    payload_amplitude_cached_unchecked(state, gains)
    payload_amplitude_raw_vjp!(dstate, draw_gains, state, raw_gains, 0.75f0)
    fill!(dstate, 0.0f0)
    fill!(draw_gains, 0.0f0)
    payload_amplitude_cached_raw_vjp_unchecked!(
        dstate,
        draw_gains,
        state,
        gains,
        raw_derivatives,
        0.75f0,
    )
    fill!(dstate, 0.0f0)
    fill!(draw_gains, 0.0f0)
    raw_forward = @allocated payload_amplitude_raw(state, raw_gains)
    cached_forward = @allocated payload_amplitude_cached_unchecked(state, gains)
    raw_reverse = @allocated payload_amplitude_raw_vjp!(
        dstate,
        draw_gains,
        state,
        raw_gains,
        0.75f0,
    )
    fill!(dstate, 0.0f0)
    cached_reverse = @allocated payload_amplitude_cached_raw_vjp_unchecked!(
        dstate,
        draw_gains,
        state,
        gains,
        raw_derivatives,
        0.75f0,
    )
    return raw_forward, cached_forward, raw_reverse, cached_reverse
end

@testset "one scalar payload and cached forward" begin
    state = normalized_fixture()
    raw_gains = zeros(Float32, ANALOG_GAIN_COUNT)
    gains = zeros(Float32, ANALOG_GAIN_COUNT)
    raw_derivatives = zeros(Float32, ANALOG_GAIN_COUNT)
    transform_payload_gains!(gains, raw_derivatives, raw_gains)
    @test gains == Float32[0.5, 0.5, 0.5]
    @test raw_derivatives == Float32[0.25, 0.25, 0.25]

    mean_nmda = sum(Float32(0.1 * compartment)
                    for compartment in 1:Cell.N_COMPARTMENTS) /
                Float32(Cell.N_COMPARTMENTS)
    mean_plateau = sum(Float32(0.05 * (compartment + 1))
                       for compartment in 1:Cell.N_COMPARTMENTS) /
                   Float32(Cell.N_COMPARTMENTS)
    expected = 1.0f0 + 0.5f0 * 0.6f0 +
               0.5f0 * mean_nmda + 0.5f0 * mean_plateau
    @test payload_amplitude_raw(state, raw_gains) ≈ expected
    @test payload_amplitude_cached(state, gains) ≈ expected
    @test payload_amplitude_cached_unchecked(state, gains) ≈ expected
    @test payload_amplitude_raw(state, raw_gains) ≈
          payload_amplitude_cached(state, gains) rtol=eps(Float32) atol=0

    # The active apical compartment contributes equally to NMDA and plateau.
    apical_only = zeros(Float32, Cell.STATE_DIM)
    apical = Cell.N_COMPARTMENTS
    apical_only[Cell.state_index(apical, Cell.FIELD_NMDA)] = 1.0f0
    apical_only[Cell.state_index(apical, Cell.FIELD_PLATEAU)] = 0.5f0
    compartment_scale = inv(Float32(Cell.N_COMPARTMENTS))
    expected_apical = 0.5f0 * compartment_scale +
                       0.5f0 * 0.5f0 * compartment_scale
    @test payload_amplitude_cached(apical_only, gains) ≈ expected_apical
    apical_only[Cell.SOMA_INDEX] = -0.75f0
    @test payload_amplitude_cached(apical_only, gains) ≈ expected_apical
end

@testset "zero analog and fixed spike coefficient" begin
    state = normalized_fixture()
    zero_analog_raw = fill(Float32(RAW_GAIN_LOW), ANALOG_GAIN_COUNT)
    zero_gains = zeros(Float32, ANALOG_GAIN_COUNT)
    zero_derivatives = zeros(Float32, ANALOG_GAIN_COUNT)
    transform_payload_gains!(zero_gains, zero_derivatives, zero_analog_raw)
    @test payload_amplitude_cached(state, zero_gains) == 1.0f0

    state[Cell.SPIKE_INDEX] = 0.0f0
    silent = payload_amplitude_cached(state, zero_gains)
    @test silent == 0.0f0
    state[Cell.SPIKE_INDEX] = 1.0f0
    @test payload_amplitude_cached(state, zero_gains) - silent == 1.0f0

    dstate = zeros(Float32, Cell.STATE_DIM)
    draw_gains = zeros(Float32, ANALOG_GAIN_COUNT)
    payload_amplitude_raw_vjp!(
        dstate,
        draw_gains,
        state,
        zero_analog_raw,
        1.0f0,
    )
    @test dstate[Cell.SPIKE_INDEX] == 1.0f0
    @test count(!iszero, dstate) == 1
    @test all(iszero, draw_gains)

    saturated = Float32[RAW_GAIN_LOW - 1, RAW_GAIN_HIGH + 1, 0]
    transformed = zeros(Float32, ANALOG_GAIN_COUNT)
    transformed_derivatives = zeros(Float32, ANALOG_GAIN_COUNT)
    transform_payload_gains!(transformed, transformed_derivatives, saturated)
    @test transformed == Float32[0, 1, 0.5]
    @test transformed_derivatives == Float32[0, 0, 0.25]
end

@testset "raw scalar VJP finite differences" begin
    state = normalized_fixture(Float64)
    raw_gains = Float64[-0.4, 0.2, 1.1]
    amplitude_cotangent = 1.6
    dstate = zeros(Float64, Cell.STATE_DIM)
    draw_gains = zeros(Float64, ANALOG_GAIN_COUNT)
    payload_amplitude_raw_vjp!(
        dstate,
        draw_gains,
        state,
        raw_gains,
        amplitude_cotangent,
    )
    objective = () -> amplitude_cotangent * payload_amplitude_raw(state, raw_gains)
    @test dstate[Cell.SPIKE_INDEX] == amplitude_cotangent
    continuous_indices = Int[Cell.SOMA_INDEX]
    for compartment in 1:Cell.N_COMPARTMENTS
        push!(continuous_indices, Cell.state_index(compartment, Cell.FIELD_NMDA))
        push!(continuous_indices, Cell.state_index(compartment, Cell.FIELD_PLATEAU))
    end
    for index in continuous_indices
        finite_difference = central_difference!(state, index, objective)
        @test dstate[index] ≈ finite_difference rtol=1.0e-8 atol=1.0e-9
    end
    for index in 1:ANALOG_GAIN_COUNT
        finite_difference = central_difference!(raw_gains, index, objective)
        @test draw_gains[index] ≈ finite_difference rtol=1.0e-8 atol=1.0e-9
    end

    used = Set(vcat(continuous_indices, Cell.SPIKE_INDEX))
    for index in 1:Cell.STATE_DIM
        index in used || @test dstate[index] == 0.0
    end

    first_state = copy(dstate)
    first_gains = copy(draw_gains)
    payload_amplitude_raw_vjp!(
        dstate,
        draw_gains,
        state,
        raw_gains,
        amplitude_cotangent,
    )
    @test dstate ≈ 2 .* first_state
    @test draw_gains ≈ 2 .* first_gains
end

@testset "cached scalar raw VJP finite differences" begin
    state = normalized_fixture(Float64)
    raw_gains = Float64[-0.8, 0.2, 1.2]
    gains = zeros(Float64, ANALOG_GAIN_COUNT)
    raw_derivatives = similar(gains)
    transform_payload_gains!(gains, raw_derivatives, raw_gains)
    amplitude_cotangent = -0.7
    dstate = zeros(Float64, Cell.STATE_DIM)
    draw_gains = zeros(Float64, ANALOG_GAIN_COUNT)
    payload_amplitude_cached_raw_vjp!(
        dstate,
        draw_gains,
        state,
        gains,
        raw_derivatives,
        amplitude_cotangent,
    )
    unchecked_dstate = zeros(Float64, Cell.STATE_DIM)
    unchecked_draw = zeros(Float64, ANALOG_GAIN_COUNT)
    payload_amplitude_cached_raw_vjp_unchecked!(
        unchecked_dstate,
        unchecked_draw,
        state,
        gains,
        raw_derivatives,
        amplitude_cotangent,
    )
    @test unchecked_dstate == dstate
    @test unchecked_draw == draw_gains
    objective = () -> amplitude_cotangent * payload_amplitude_cached(state, gains)
    state_indices = Int[Cell.SOMA_INDEX]
    for compartment in 1:Cell.N_COMPARTMENTS
        push!(state_indices, Cell.state_index(compartment, Cell.FIELD_NMDA))
        push!(state_indices, Cell.state_index(compartment, Cell.FIELD_PLATEAU))
    end
    for index in state_indices
        finite_difference = central_difference!(state, index, objective)
        @test dstate[index] ≈ finite_difference rtol=1.0e-8 atol=1.0e-9
    end
    for index in 1:ANALOG_GAIN_COUNT
        raw_objective = () -> begin
            transform_payload_gains!(gains, raw_derivatives, raw_gains)
            amplitude_cotangent * payload_amplitude_cached(state, gains)
        end
        finite_difference = central_difference!(raw_gains, index, raw_objective)
        @test draw_gains[index] ≈ finite_difference rtol=1.0e-8 atol=1.0e-9
    end

    cold_dstate = zeros(Float64, Cell.STATE_DIM)
    cold_draw = zeros(Float64, ANALOG_GAIN_COUNT)
    payload_amplitude_raw_vjp!(
        cold_dstate,
        cold_draw,
        state,
        raw_gains,
        amplitude_cotangent,
    )
    @test dstate == cold_dstate
    @test draw_gains == cold_draw
end

@testset "payload validation and allocation" begin
    state = normalized_fixture()
    raw_gains = zeros(Float32, ANALOG_GAIN_COUNT)
    gains = zeros(Float32, ANALOG_GAIN_COUNT)
    raw_derivatives = zeros(Float32, ANALOG_GAIN_COUNT)
    transform_payload_gains!(gains, raw_derivatives, raw_gains)
    @test_throws DimensionMismatch payload_amplitude_raw(state[1:end-1], raw_gains)
    @test_throws DimensionMismatch payload_amplitude_raw(state, raw_gains[1:2])
    @test_throws DimensionMismatch payload_amplitude_cached(state, gains[1:2])
    invalid_state = copy(state)
    invalid_state[Cell.SPIKE_INDEX] = 0.5f0
    @test_throws ArgumentError payload_amplitude_cached(invalid_state, gains)
    invalid_gains = copy(gains)
    invalid_gains[1] = 1.1f0
    @test_throws ArgumentError payload_amplitude_cached(state, invalid_gains)
    invalid_raw = copy(raw_gains)
    invalid_raw[1] = NaN32
    @test_throws ArgumentError transform_payload_gains!(
        gains,
        raw_derivatives,
        invalid_raw,
    )

    dstate = zeros(Float32, Cell.STATE_DIM)
    draw_gains = zeros(Float32, ANALOG_GAIN_COUNT)
    allocations = hot_allocations(
        state,
        raw_gains,
        gains,
        raw_derivatives,
        dstate,
        draw_gains,
    )
    @test allocations == (0, 0, 0, 0)

    transform_payload_gains!(gains, raw_derivatives, raw_gains)
    transform_bytes = @allocated transform_payload_gains!(
        gains,
        raw_derivatives,
        raw_gains,
    )
    @test transform_bytes == 0
end
