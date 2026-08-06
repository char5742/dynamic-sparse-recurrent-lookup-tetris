using Test
using Random
using LinearAlgebra
using Zygote

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "StateCodec.jl"))
using .ActiveApicalCell
using .StateCodec

const Cell = ActiveApicalCell
const Codec = StateCodec

function physical_probe_state(::Type{T}=Float32) where {T<:AbstractFloat}
    state = zeros(T, Cell.STATE_DIM)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] = T(-62 + compartment)
        state[Cell.state_index(compartment, Cell.FIELD_AMPA)] = T(0.04 + 0.01 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_NMDA)] = T(0.12 + 0.02 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_GABA)] = T(0.08 + 0.015 * compartment)
        state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = T(0.1 + 0.05 * compartment)
    end
    state[Cell.SOMA_INDEX] = T(-58)
    state[Cell.ADAPTATION_INDEX] = T(0.7)
    state[Cell.SPIKE_INDEX] = T(0.25)
    return state
end

@testset "canonical codec contract and resting zero" begin
    @test Codec.CODEC_DIM == Cell.STATE_DIM == 48
    raw = Cell.default_raw_parameters()
    rest = Cell.initial_state(raw)
    @test all(iszero, Codec.encode_state_functional(rest))
    destination = similar(rest)
    Codec.encode_state!(destination, rest)
    @test all(iszero, destination)
end

@testset "same position axes and functional equivalence" begin
    state = physical_probe_state()
    expected = Codec.encode_state_functional(state)
    destination = similar(state)
    Codec.encode_state!(destination, state)
    @test destination == expected

    for index in eachindex(state)
        perturbed = copy(state)
        if index == Cell.SPIKE_INDEX
            perturbed[index] += 0.25f0
        elseif (index - 1) % Cell.COMPARTMENT_STATE_DIM == Cell.FIELD_PLATEAU - 1
            perturbed[index] = min(0.9f0, perturbed[index] + 0.05f0)
        else
            perturbed[index] += 0.01f0
        end
        difference = Codec.encode_state_functional(perturbed) .- expected
        @test findall(!iszero, difference) == [index]
    end

    Codec.encode_state!(destination, state)
    @test @allocated(Codec.encode_state!(destination, state)) == 0
    alias_state = copy(state)
    Codec.encode_state!(alias_state, alias_state)
    @test alias_state == expected
end

@testset "range and monotonicity by physical state class" begin
    state = physical_probe_state()
    encoded = Codec.encode_state_functional(state)
    for compartment in 1:Cell.N_COMPARTMENTS
        @test -1.0f0 < encoded[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] < 1.0f0
        for field in (Cell.FIELD_AMPA, Cell.FIELD_NMDA, Cell.FIELD_GABA, Cell.FIELD_PLATEAU)
            @test 0.0f0 <= encoded[Cell.state_index(compartment, field)] <= 1.0f0
        end
    end
    @test -1.0f0 < encoded[Cell.SOMA_INDEX] < 1.0f0
    @test 0.0f0 <= encoded[Cell.ADAPTATION_INDEX] < 1.0f0
    @test encoded[Cell.SPIKE_INDEX] == state[Cell.SPIKE_INDEX]

    for (field, values) in (
        (Cell.FIELD_AMPA, (0.0f0, 0.01f0, 0.25f0, 1.0f0, 1.0f4)),
        (Cell.FIELD_NMDA, (0.0f0, 0.01f0, 0.25f0, 1.0f0, 1.0f4)),
        (Cell.FIELD_GABA, (0.0f0, 0.01f0, 0.25f0, 1.0f0, 1.0f4)),
    )
        outputs = Float32[]
        for value in values
            probe = copy(state)
            probe[Cell.state_index(1, field)] = value
            push!(outputs, Codec.encode_state_functional(probe)[Cell.state_index(1, field)])
        end
        @test issorted(outputs)
        @test first(outputs) == 0.0f0
        @test last(outputs) < 1.0f0
    end

    voltage_outputs = [
        begin
            probe = copy(state)
            probe[Cell.state_index(1, Cell.FIELD_VOLTAGE)] = voltage
            Codec.encode_state_functional(probe)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
        end for voltage in (-1.0f4, -100.0f0, -65.0f0, -30.0f0, 1.0f4)
    ]
    @test issorted(voltage_outputs)
    @test first(voltage_outputs) == -1.0f0
    @test last(voltage_outputs) == 1.0f0
end

@testset "analytic VJP versus Zygote and every-coordinate finite difference" begin
    rng = MersenneTwister(0xC0DEC)
    state = physical_probe_state(Float64)
    direction = randn(rng, Float64, Codec.CODEC_DIM)
    objective(s) = dot(Codec.encode_state_functional(s), direction)
    zygote_gradient = Zygote.gradient(objective, state)[1]
    analytic_gradient = similar(state)
    Codec.state_codec_pullback!(analytic_gradient, state, direction)
    @test isapprox(analytic_gradient, zygote_gradient; rtol=2.0e-12, atol=2.0e-13)

    epsilon = 1.0e-5
    for index in eachindex(state)
        plus = copy(state)
        minus = copy(state)
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2epsilon)
        @test isapprox(analytic_gradient[index], numerical; rtol=2.0e-6, atol=2.0e-8)
    end

    state32 = physical_probe_state()
    direction32 = Float32.(direction)
    gradient32 = similar(state32)
    Codec.state_codec_pullback!(gradient32, state32, direction32)
    @test @allocated(Codec.state_codec_pullback!(gradient32, state32, direction32)) == 0
end

@testset "full 240-cell resting-anchor head scale" begin
    rng = MersenneTwister(0x240CE11)
    cells = 30 * 8
    anchor = Vector{Float32}(undef, Codec.CODEC_DIM * cells)
    encoded = Vector{Float32}(undef, Codec.CODEC_DIM)
    raw = Cell.default_raw_parameters()
    rest = Cell.initial_state(raw)
    for cell in 1:cells
        state = copy(rest)
        for compartment in 1:Cell.N_COMPARTMENTS
            state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] += 2.0f0 * randn(rng, Float32)
            state[Cell.state_index(compartment, Cell.FIELD_AMPA)] = 0.03f0 * abs(randn(rng, Float32))
            state[Cell.state_index(compartment, Cell.FIELD_NMDA)] = 0.05f0 * abs(randn(rng, Float32))
            state[Cell.state_index(compartment, Cell.FIELD_GABA)] = 0.03f0 * abs(randn(rng, Float32))
            state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = clamp(0.02f0 * abs(randn(rng, Float32)), 0.0f0, 1.0f0)
        end
        state[Cell.SOMA_INDEX] += randn(rng, Float32)
        state[Cell.ADAPTATION_INDEX] = 0.05f0 * abs(randn(rng, Float32))
        state[Cell.SPIKE_INDEX] = rand(rng, Float32) < 0.01f0 ? 1.0f0 : 0.0f0
        Codec.encode_state!(encoded, state)
        copyto!(anchor, (cell - 1) * Codec.CODEC_DIM + 1, encoded, 1, Codec.CODEC_DIM)
    end

    output_dim = 22
    head = randn(rng, Float32, output_dim, length(anchor)) .* sqrt(2.0f0 / (length(anchor) + output_dim))
    logits = head * anchor
    @test all(isfinite, logits)
    @test sqrt(sum(abs2, logits) / length(logits)) < 0.5f0
    @test maximum(abs, logits) < 1.0f0
end
