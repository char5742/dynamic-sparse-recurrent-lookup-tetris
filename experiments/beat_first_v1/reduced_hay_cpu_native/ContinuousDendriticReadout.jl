module ContinuousDendriticReadout

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export OUTPUT_CHANNELS,
       PHASES,
       DECISION_CELLS,
       ReadoutParameters,
       ReadoutCache,
       ReadoutTape,
       ReadoutScratch,
       ReadoutGradient,
       initialize_parameters,
       refresh_cache!,
       clear_gradient!,
       readout_from_tape!,
       readout_forward!,
       readout_pullback!

const OUTPUT_CHANNELS = 22
const PHASES = 3
const DECISION_CELLS = 50
const OUTPUT_NORMALIZATION = inv(sqrt(Float32(DECISION_CELLS)))

"""
Continuous all-cell readout.

Every high-dimensional decision cell contributes to every supervised channel.
The 22 x 50 projection is only 1,100 scalar products per candidate and avoids
the fatal capacity loss of assigning eight cells to Q and the other 42 cells
exclusively to auxiliary outputs.  Hard soma events remain in the physical
tape for control and diagnostics; outputs read only pre-reset analog margins.
"""
struct ReadoutParameters
    shared_cell_raw::Vector{Float32}
    gain::Matrix{Float32}
    bias::Vector{Float32}

    function ReadoutParameters(
        shared_cell_raw::Vector{Float32},
        gain::Matrix{Float32},
        bias::Vector{Float32},
    )
        length(shared_cell_raw) == Cell.PARAM_DIM || throw(DimensionMismatch(
            "shared cell raw vector has the wrong length",
        ))
        size(gain) == (OUTPUT_CHANNELS, DECISION_CELLS) || throw(
            DimensionMismatch("readout gain must have shape (22, 50)"),
        )
        length(bias) == OUTPUT_CHANNELS || throw(DimensionMismatch(
            "readout bias must have 22 values",
        ))
        return new(shared_cell_raw, gain, bias)
    end
end

@inline function _initial_gain(channel::Int, cell::Int)
    word = UInt64(channel) * UInt64(0x9e3779b97f4a7c15) ⊻
           UInt64(cell) * UInt64(0xbf58476d1ce4e5b9)
    word = xor(word, word >> 30) * UInt64(0xbf58476d1ce4e5b9)
    word = xor(word, word >> 27) * UInt64(0x94d049bb133111eb)
    word = xor(word, word >> 31)
    magnitude = 0.75f0 + Float32((word >> 40) & UInt64(0x00ff_ffff)) /
        Float32(0x0100_0000) * 0.5f0
    return isodd(count_ones(word)) ? magnitude : -magnitude
end

function initialize_parameters()
    gain = Matrix{Float32}(undef, OUTPUT_CHANNELS, DECISION_CELLS)
    @inbounds for cell in 1:DECISION_CELLS, channel in 1:OUTPUT_CHANNELS
        gain[channel, cell] = _initial_gain(channel, cell)
    end
    return ReadoutParameters(
        Cell.default_raw_parameters(Float32),
        gain,
        zeros(Float32, OUTPUT_CHANNELS),
    )
end

mutable struct ReadoutCache
    cell::Cell.CellParameterCache{Float32}
    cell_derivative::Cell.CellParameterDerivativeCache{Float32}
end

function ReadoutCache(parameters::ReadoutParameters)
    cache, derivative = Cell.parameter_caches(parameters.shared_cell_raw)
    return ReadoutCache(cache, derivative)
end

function refresh_cache!(cache::ReadoutCache, parameters::ReadoutParameters)
    cache.cell = Cell.transform_parameters(parameters.shared_cell_raw)
    cache.cell_derivative =
        Cell.transform_parameter_derivatives(parameters.shared_cell_raw)
    return cache
end

"""Caller-owned fixed trajectory for all three integration phases."""
struct ReadoutTape
    physical::Array{Float32,3}
    margin::Matrix{Float32}

    function ReadoutTape(physical::Array{Float32,3}, margin::Matrix{Float32})
        size(physical) == (Cell.STATE_DIM, DECISION_CELLS, PHASES + 1) ||
            throw(DimensionMismatch("decision-cell physical tape has the wrong shape"))
        size(margin) == (DECISION_CELLS, PHASES) || throw(
            DimensionMismatch("decision-cell margin tape has the wrong shape"),
        )
        return new(physical, margin)
    end
end

ReadoutTape() = ReadoutTape(
    zeros(Float32, Cell.STATE_DIM, DECISION_CELLS, PHASES + 1),
    zeros(Float32, DECISION_CELLS, PHASES),
)

"""Caller-owned one-cell reverse workspace."""
struct ReadoutScratch
    state_bar::Vector{Float32}
    previous_state_bar::Vector{Float32}
    input_bar::Vector{Float32}
    raw_bar::Vector{Float32}
end

ReadoutScratch() = ReadoutScratch(
    zeros(Float32, Cell.STATE_DIM),
    zeros(Float32, Cell.STATE_DIM),
    zeros(Float32, Cell.INPUT_DIM),
    zeros(Float32, Cell.PARAM_DIM),
)

struct ReadoutGradient
    shared_cell_raw::Vector{Float32}
    gain::Matrix{Float32}
    bias::Vector{Float32}

    function ReadoutGradient(
        shared_cell_raw::Vector{Float32},
        gain::Matrix{Float32},
        bias::Vector{Float32},
    )
        length(shared_cell_raw) == Cell.PARAM_DIM || throw(DimensionMismatch(
            "shared cell gradient has the wrong length",
        ))
        size(gain) == (OUTPUT_CHANNELS, DECISION_CELLS) || throw(
            DimensionMismatch("gain gradient has the wrong shape"),
        )
        length(bias) == OUTPUT_CHANNELS || throw(DimensionMismatch(
            "bias gradient has the wrong length",
        ))
        return new(shared_cell_raw, gain, bias)
    end
end

ReadoutGradient() = ReadoutGradient(
    zeros(Float32, Cell.PARAM_DIM),
    zeros(Float32, OUTPUT_CHANNELS, DECISION_CELLS),
    zeros(Float32, OUTPUT_CHANNELS),
)

function clear_gradient!(gradient::ReadoutGradient)
    fill!(gradient.shared_cell_raw, 0.0f0)
    fill!(gradient.gain, 0.0f0)
    fill!(gradient.bias, 0.0f0)
    return gradient
end

@inline function _check_drive(drive::AbstractArray{Float32,3})
    size(drive) == (Cell.INPUT_DIM, DECISION_CELLS, PHASES) || throw(
        DimensionMismatch("decision-cell conductance drive has the wrong shape"),
    )
    return nothing
end

@inline function _check_tape(tape::ReadoutTape)
    size(tape.physical) == (Cell.STATE_DIM, DECISION_CELLS, PHASES + 1) ||
        throw(DimensionMismatch("decision-cell physical tape has the wrong shape"))
    size(tape.margin) == (DECISION_CELLS, PHASES) || throw(
        DimensionMismatch("decision-cell margin tape has the wrong shape"),
    )
    return nothing
end

"""Project every final continuous soma margin to all 22 outputs."""
function readout_from_tape!(
    raw_output::AbstractVector{Float32},
    tape::ReadoutTape,
    parameters::ReadoutParameters,
    cache::ReadoutCache,
)
    length(raw_output) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "continuous readout must produce 22 values",
    ))
    _check_tape(tape)
    @inbounds for cell in 1:DECISION_CELLS
        tape.margin[cell, PHASES] = Cell.spike_margin_from_transition(
            @view(tape.physical[:, cell, PHASES]),
            @view(tape.physical[:, cell, PHASES + 1]),
            cache.cell,
        )
    end
    @inbounds for channel in 1:OUTPUT_CHANNELS
        total = parameters.bias[channel]
        for cell in 1:DECISION_CELLS
            total = muladd(
                parameters.gain[channel, cell] * OUTPUT_NORMALIZATION,
                tape.margin[cell, PHASES],
                total,
            )
        end
        raw_output[channel] = total
    end
    return raw_output
end


function readout_forward!(
    raw_output::AbstractVector{Float32},
    tape::ReadoutTape,
    drive::AbstractArray{Float32,3},
    parameters::ReadoutParameters,
    cache::ReadoutCache,
)
    _check_drive(drive)
    _check_tape(tape)
    @inbounds for cell in 1:DECISION_CELLS
        Cell.initial_state!(@view(tape.physical[:, cell, 1]), cache.cell)
    end
    @inbounds for phase in 1:PHASES
        for cell in 1:DECISION_CELLS
            previous = @view tape.physical[:, cell, phase]
            next = @view tape.physical[:, cell, phase + 1]
            Cell.cell_step!(
                next,
                previous,
                @view(drive[:, cell, phase]),
                cache.cell,
                0.0f0,
            )
            tape.margin[cell, phase] =
                Cell.spike_margin_from_transition(previous, next, cache.cell)
        end
    end
    return readout_from_tape!(raw_output, tape, parameters, cache)
end

"""
Exact reverse of the continuous paths conditional on the recorded hard-event
sequence.  Every cell is traversed once; its margin seed is the signed sum of
all 22 channel cotangents.  No output channel owns or hides a private cell.
"""
function readout_pullback!(
    drive_bar::AbstractArray{Float32,3},
    gradient::ReadoutGradient,
    tape::ReadoutTape,
    scratch::ReadoutScratch,
    drive::AbstractArray{Float32,3},
    parameters::ReadoutParameters,
    cache::ReadoutCache,
    raw_bar::AbstractVector{Float32},
)
    _check_drive(drive)
    _check_drive(drive_bar)
    _check_tape(tape)
    length(raw_bar) == OUTPUT_CHANNELS || throw(DimensionMismatch(
        "continuous output cotangent must have 22 values",
    ))
    fill!(drive_bar, 0.0f0)
    clear_gradient!(gradient)

    @inbounds for channel in 1:OUTPUT_CHANNELS
        gradient.bias[channel] = raw_bar[channel]
        for cell in 1:DECISION_CELLS
            gradient.gain[channel, cell] = raw_bar[channel] *
                tape.margin[cell, PHASES] * OUTPUT_NORMALIZATION
        end
    end

    @inbounds for cell in 1:DECISION_CELLS
        margin_seed = 0.0f0
        for channel in 1:OUTPUT_CHANNELS
            margin_seed = muladd(
                raw_bar[channel],
                parameters.gain[channel, cell] * OUTPUT_NORMALIZATION,
                margin_seed,
            )
        end
        fill!(scratch.state_bar, 0.0f0)
        for phase in PHASES:-1:1
            Cell.cell_step_conditional_pullback!(
                scratch.previous_state_bar,
                scratch.input_bar,
                scratch.raw_bar,
                @view(tape.physical[:, cell, phase]),
                @view(drive[:, cell, phase]),
                cache.cell,
                cache.cell_derivative,
                @view(tape.physical[:, cell, phase + 1]),
                scratch.state_bar,
                0.0f0,
                0.0f0,
                phase == PHASES ? margin_seed : 0.0f0,
            )
            @simd for parameter in 1:Cell.PARAM_DIM
                gradient.shared_cell_raw[parameter] += scratch.raw_bar[parameter]
            end
            @simd for input_channel in 1:Cell.INPUT_DIM
                drive_bar[input_channel, cell, phase] = scratch.input_bar[input_channel]
            end
            copyto!(scratch.state_bar, scratch.previous_state_bar)
        end
        fill!(scratch.raw_bar, 0.0f0)
        Cell.initial_state_pullback!(
            scratch.raw_bar,
            scratch.state_bar,
            cache.cell_derivative,
        )
        @simd for parameter in 1:Cell.PARAM_DIM
            gradient.shared_cell_raw[parameter] += scratch.raw_bar[parameter]
        end
    end
    return gradient
end

end # module ContinuousDendriticReadout
