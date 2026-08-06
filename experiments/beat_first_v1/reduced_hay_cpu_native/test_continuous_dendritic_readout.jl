using Test
using Random

module ContinuousDendriticReadoutTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "ContinuousDendriticReadout.jl"))
end

const Cell = ContinuousDendriticReadoutTestHarness.ActiveApicalCell
const Readout = ContinuousDendriticReadoutTestHarness.ContinuousDendriticReadout

function seeded_drive()
    rng = MersenneTwister(0xc011ec7)
    return 0.03f0 .+ 0.04f0 .* rand(
        rng,
        Float32,
        Cell.INPUT_DIM,
        Readout.DECISION_CELLS,
        Readout.PHASES,
    )
end

function objective!(output, tape, drive, parameters, cache, direction)
    Readout.readout_forward!(output, tape, drive, parameters, cache)
    return sum(output .* direction)
end

@testset "all-cell continuous dendritic readout contract" begin
    parameters = Readout.initialize_parameters()
    @test Readout.OUTPUT_CHANNELS == 22
    @test Readout.DECISION_CELLS == 50
    @test Readout.PHASES == 3
    @test length(parameters.shared_cell_raw) == Cell.PARAM_DIM
    @test size(parameters.gain) == (22, 50)
    @test length(parameters.bias) == 22
    @test all(!iszero, parameters.gain)
    @test all(channel -> length(unique(parameters.gain[channel, :])) > 40, 1:22)
    @test_throws DimensionMismatch Readout.ReadoutParameters(
        copy(parameters.shared_cell_raw),
        zeros(Float32, 22, 49),
        zeros(Float32, 22),
    )
end

@testset "hard event is control-only at the continuous output" begin
    parameters = Readout.initialize_parameters()
    cache = Readout.ReadoutCache(parameters)
    drive = seeded_drive()
    tape = Readout.ReadoutTape()
    output = zeros(Float32, 22)
    Readout.readout_forward!(output, tape, drive, parameters, cache)

    altered = Readout.ReadoutTape(copy(tape.physical), copy(tape.margin))
    @inbounds for cell in 1:Readout.DECISION_CELLS
        altered.physical[Cell.SPIKE_INDEX, cell, end] =
            1.0f0 - altered.physical[Cell.SPIKE_INDEX, cell, end]
        altered.physical[Cell.SOMA_INDEX, cell, end] += 100.0f0
    end
    altered_output = similar(output)
    Readout.readout_from_tape!(altered_output, altered, parameters, cache)
    @test altered_output == output

    # No private output population remains: changing any one cell gain changes
    # both Q and an auxiliary channel without another cell transition.
    cell = 37
    margin = tape.margin[cell, end]
    original_q = output[1]
    original_aux = output[12]
    parameters.gain[1, cell] += 0.5f0
    parameters.gain[12, cell] -= 0.25f0
    Readout.readout_from_tape!(altered_output, tape, parameters, cache)
    normalization = inv(sqrt(Float32(Readout.DECISION_CELLS)))
    @test altered_output[1] - original_q ≈ 0.5f0 * margin * normalization
    @test altered_output[12] - original_aux ≈ -0.25f0 * margin * normalization
end

@testset "conditional analytic pullback matches finite differences" begin
    rng = MersenneTwister(0xad70117)
    parameters = Readout.initialize_parameters()
    cache = Readout.ReadoutCache(parameters)
    drive = seeded_drive()
    tape = Readout.ReadoutTape()
    scratch = Readout.ReadoutScratch()
    gradient = Readout.ReadoutGradient()
    drive_bar = similar(drive)
    output = zeros(Float32, 22)
    direction = randn(rng, Float32, 22)
    Readout.readout_forward!(output, tape, drive, parameters, cache)
    Readout.readout_pullback!(
        drive_bar,
        gradient,
        tape,
        scratch,
        drive,
        parameters,
        cache,
        direction,
    )

    epsilon = 5.0f-3
    function finite_difference!(array, index; refresh=false)
        original = array[index]
        array[index] = original + epsilon
        refresh && Readout.refresh_cache!(cache, parameters)
        plus = objective!(output, tape, drive, parameters, cache, direction)
        array[index] = original - epsilon
        refresh && Readout.refresh_cache!(cache, parameters)
        minus = objective!(output, tape, drive, parameters, cache, direction)
        array[index] = original
        refresh && Readout.refresh_cache!(cache, parameters)
        return (plus - minus) / (2.0f0 * epsilon)
    end

    drive_index = argmax(abs.(drive_bar))
    gain_index = argmax(abs.(gradient.gain))
    bias_index = argmax(abs.(gradient.bias))
    raw_index = argmax(abs.(gradient.shared_cell_raw))
    @test drive_bar[drive_index] ≈ finite_difference!(drive, drive_index) rtol=2.0f-2 atol=2.0f-3
    @test gradient.gain[gain_index] ≈ finite_difference!(parameters.gain, gain_index) rtol=2.0f-2 atol=2.0f-3
    @test gradient.bias[bias_index] ≈ finite_difference!(parameters.bias, bias_index) rtol=2.0f-3 atol=2.0f-3
    @test gradient.shared_cell_raw[raw_index] ≈
          finite_difference!(parameters.shared_cell_raw, raw_index; refresh=true) rtol=3.0f-2 atol=3.0f-3

    q_only = zeros(Float32, 22)
    q_only[1] = 1.0f0
    Readout.readout_forward!(output, tape, drive, parameters, cache)
    Readout.readout_pullback!(
        drive_bar, gradient, tape, scratch, drive, parameters, cache, q_only,
    )
    @test all(cell -> any(!iszero, @view drive_bar[:, cell, :]), 1:50)
    @test @allocated(
        Readout.readout_forward!(output, tape, drive, parameters, cache)
    ) == 0
end
