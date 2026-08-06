using Random
using Test

module MultiStepCellCodecHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "StateCodec.jl"))
end

const Cell = MultiStepCellCodecHarness.ActiveApicalCell
const Codec = MultiStepCellCodecHarness.StateCodec

function objective(raw, inputs, encoded_seed)
    cache = Cell.transform_parameters(raw)
    state = Cell.initial_state(cache)
    next = similar(state)
    encoded = similar(state)
    value = 0.0f0
    for step in axes(inputs, 2)
        Cell.cell_step!(next, state, @view(inputs[:, step]), cache, 1.0f0)
        Codec.encode_state!(encoded, next)
        value += sum(encoded .* @view(encoded_seed[:, step]))
        state, next = next, state
    end
    return value
end

@testset "six-step cell plus codec exact VJP" begin
    rng = Xoshiro(0xc011c0de)
    raw = Cell.default_raw_parameters(Float32)
    cache = Cell.transform_parameters(raw)
    derivative = Cell.transform_parameter_derivatives(raw)
    inputs = 0.05f0 .* rand(rng, Float32, Cell.INPUT_DIM, 6)
    encoded_seed = 0.1f0 .* randn(rng, Float32, Cell.STATE_DIM, 6)
    physical = zeros(Float32, Cell.STATE_DIM, 7)
    encoded = zeros(Float32, Cell.STATE_DIM, 6)
    Cell.initial_state!(@view(physical[:, 1]), cache)
    for step in 1:6
        Cell.cell_step!(
            @view(physical[:, step + 1]),
            @view(physical[:, step]),
            @view(inputs[:, step]),
            cache,
            1.0f0,
        )
        Codec.encode_state!(
            @view(encoded[:, step]),
            @view(physical[:, step + 1]),
        )
    end

    successor_bar = zeros(Float32, Cell.STATE_DIM)
    local_bar = zeros(Float32, Cell.STATE_DIM)
    state_bar = zeros(Float32, Cell.STATE_DIM)
    input_bar = zeros(Float32, Cell.INPUT_DIM)
    raw_bar = zeros(Float32, Cell.PARAM_DIM)
    step_raw_bar = similar(raw_bar)
    for step in 6:-1:1
        Codec.state_codec_pullback!(
            local_bar,
            @view(physical[:, step + 1]),
            @view(encoded_seed[:, step]),
        )
        local_bar .+= successor_bar
        Cell.cell_step_pullback!(
            state_bar,
            input_bar,
            step_raw_bar,
            @view(physical[:, step]),
            @view(inputs[:, step]),
            cache,
            derivative,
            @view(physical[:, step + 1]),
            local_bar,
            0.0f0,
            1.0f0,
        )
        raw_bar .+= step_raw_bar
        copyto!(successor_bar, state_bar)
    end
    Cell.initial_state_pullback!(raw_bar, successor_bar, derivative)

    for index in (
        Cell.P_COMPARTMENT_REST,
        Cell.P_SOMA_REST,
        Cell.P_SOMA_THRESHOLD_GAP,
    )
        epsilon = 5.0f-4
        original = raw[index]
        raw[index] = original + epsilon
        positive = objective(raw, inputs, encoded_seed)
        raw[index] = original - epsilon
        negative = objective(raw, inputs, encoded_seed)
        raw[index] = original
        numerical = (positive - negative) / (2.0f0 * epsilon)
        @test isapprox(raw_bar[index], numerical; atol=2.0f-3, rtol=4.0f-2)
    end
end
