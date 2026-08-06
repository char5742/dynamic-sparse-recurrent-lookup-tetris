using Test
using Random
using LinearAlgebra

module TypedOutputCellBankTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
include(joinpath(@__DIR__, "TypedOutputCellBank.jl"))
end

const H = TypedOutputCellBankTestHarness
const Cell = H.ActiveApicalCell
const Packet = H.HighDimensionalCellPacket
const Output = H.TypedOutputCellBank

function _raw_for_value!(raw, name::Symbol, value)
    index = something(findfirst(==(name), Cell.PARAMETER_NAMES))
    lower = eltype(raw)(Cell.PARAMETER_LOWER[index])
    upper = eltype(raw)(Cell.PARAMETER_UPPER[index])
    probability = clamp(
        (eltype(raw)(value) - lower) / (upper - lower),
        eltype(raw)(1.0e-6),
        one(eltype(raw)) - eltype(raw)(1.0e-6),
    )
    raw[index] = log(probability / (one(probability) - probability))
    return raw
end

function _central_difference!(values, index, objective; epsilon)
    original = values[index]
    values[index] = original + epsilon
    plus, plus_events = objective()
    values[index] = original - epsilon
    minus, minus_events = objective()
    values[index] = original
    return (plus - minus) / (2epsilon), plus_events, minus_events
end

function _resting_base(cache::Output.TypedOutputCache{T}) where {T}
    base_state = Matrix{T}(undef, Cell.STATE_DIM, Output.OUTPUT_CELLS)
    return Output.output_initial_state!(base_state, cache)
end

function _interior_base(cache::Output.TypedOutputCache{T}) where {T}
    base_state = _resting_base(cache)
    @inbounds for output in 1:Output.OUTPUT_CELLS
        for compartment in 1:Cell.N_COMPARTMENTS
            base_state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE), output] +=
                T(0.1)
            base_state[Cell.state_index(compartment, Cell.FIELD_AMPA), output] =
                T(0.02)
            base_state[Cell.state_index(compartment, Cell.FIELD_NMDA), output] =
                T(0.015)
            base_state[Cell.state_index(compartment, Cell.FIELD_GABA), output] =
                T(0.01)
            base_state[Cell.state_index(compartment, Cell.FIELD_PLATEAU), output] =
                T(0.02)
        end
        base_state[Cell.SOMA_INDEX, output] += T(0.05)
        base_state[Cell.ADAPTATION_INDEX, output] = T(0.1)
        base_state[Cell.SPIKE_INDEX, output] = zero(T)
    end
    return base_state
end

@testset "typed 22-cell output contract" begin
    parameters = Output.initialize_parameters()
    @test size(parameters.cell_raw) == (Cell.PARAM_DIM, 22)
    @test Output.READOUT_DIM == Packet.PACKET_DIM == 47
    @test size(parameters.readout_weight) == (47, 22)
    @test length(parameters.bias) == 22
    @test Output.stored_parameter_count(parameters) ==
          22 * (Cell.PARAM_DIM + 47 + 1)

    tape = Output.TypedOutputTape()
    @test size(tape.base_state) == (Cell.STATE_DIM, 22)
    @test size(tape.next_state) == (Cell.STATE_DIM, 22)
    @test size(tape.inbox) == (Cell.INPUT_DIM, 22)
    @test size(tape.packet) == (47, 22)
    @test length(tape.event) == 22
    @test Output.hard_event_denominator() == 22

    cache = Output.TypedOutputCache(parameters)
    base_state = _resting_base(cache)
    inbox = zeros(Float32, Cell.INPUT_DIM, 22)
    continuous = fill(Float32(NaN), 22)
    hard_event = similar(continuous)
    Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    # The fixed physical voltage center is independent of each cell's
    # trainable rest potential, so a silent packet may carry a small common
    # offset. It must remain finite, local and identical across identical
    # initialized cells.
    @test all(isfinite, continuous)
    @test all(==(continuous[1]), continuous)
    @test maximum(abs, continuous) < 0.02f0
    @test hard_event == zeros(Float32, 22)
    @test Output.hard_event_count(tape) == 0
    @test tape.inbox == inbox
end

@testset "typed receptor identity and one mandatory event transition" begin
    parameters = Output.initialize_parameters()
    cache = Output.TypedOutputCache(parameters)
    base_state = _resting_base(cache)
    tape = Output.TypedOutputTape()
    inbox = zeros(Float32, Cell.INPUT_DIM, 22)
    @inbounds for branch in 1:Cell.N_BASAL
        inbox[Cell.input_index(branch, Cell.INPUT_AMPA), 1] = 0.5f0
        inbox[Cell.input_index(branch, Cell.INPUT_NMDA), 1] = 0.5f0
    end
    continuous = zeros(Float32, 22)
    hard_event = zeros(Float32, 22)
    Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    @test tape.event[1] == 1.0f0
    @test hard_event[1] == 1.0f0
    @test all(iszero, @view(hard_event[2:end]))
    @test tape.inbox == inbox
    @test tape.event == hard_event

    # AMPA and GABA are independent typed receptor coordinates.  The bank
    # consumes them directly; it never reconstructs them through a signed
    # semantic-drive adapter.
    quiet = Output.TypedOutputTape()
    ampa = Output.TypedOutputTape()
    gaba = Output.TypedOutputTape()
    zero_inbox = zeros(Float32, Cell.INPUT_DIM, 22)
    ampa_inbox = copy(zero_inbox)
    gaba_inbox = copy(zero_inbox)
    ampa_inbox[Cell.input_index(1, Cell.INPUT_AMPA), 2] = 0.2f0
    gaba_inbox[Cell.input_index(1, Cell.INPUT_GABA), 2] = 0.2f0
    quiet_value = zeros(Float32, 22)
    ampa_value = similar(quiet_value)
    gaba_value = similar(quiet_value)
    events = similar(quiet_value)
    Output.typed_output_forward!(
        quiet_value, events, quiet, base_state, zero_inbox, parameters, cache,
    )
    Output.typed_output_forward!(
        ampa_value, events, ampa, base_state, ampa_inbox, parameters, cache,
    )
    Output.typed_output_forward!(
        gaba_value, events, gaba, base_state, gaba_inbox, parameters, cache,
    )
    @test ampa_value[2] != quiet_value[2]
    @test gaba_value[2] != quiet_value[2]
    @test ampa_value[2] != gaba_value[2]
end

@testset "47-lane packet rank and strictly local output readout" begin
    rng = MersenneTwister(0x16a9e)
    parameters = Output.initialize_parameters(Float64)
    cache = Output.TypedOutputCache(parameters)
    base_state = _resting_base(cache)
    inbox = abs.(0.2 .* randn(
        rng,
        Float64,
        Cell.INPUT_DIM,
        Output.OUTPUT_CELLS,
    ))
    tape = Output.TypedOutputTape(Float64)
    continuous = zeros(Float64, Output.OUTPUT_CELLS)
    hard_event = similar(continuous)
    Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )

    # Forty-seven coordinates across 22 independent cells can have rank at
    # most 22. Reaching that algebraic maximum rules out the former scalar
    # packet bottleneck.
    @test rank(tape.packet) == min(size(tape.packet)...)

    # The explicit packet-to-output Jacobian is block diagonal: it has full
    # output rank but contains no learned connection between distinct outputs.
    packet_jacobian = zeros(
        Float64,
        Output.OUTPUT_CELLS,
        Output.READOUT_DIM * Output.OUTPUT_CELLS,
    )
    @inbounds for output in 1:Output.OUTPUT_CELLS
        first = (output - 1) * Output.READOUT_DIM + 1
        packet_jacobian[
            output,
            first:(first + Output.READOUT_DIM - 1),
        ] .= @view(parameters.readout_weight[:, output])
    end
    @test rank(packet_jacobian) == Output.OUTPUT_CELLS

    selected = 7
    direction = zeros(Float64, Output.OUTPUT_CELLS)
    direction[selected] = 1.0
    gradient = Output.TypedOutputGradient(Float64)
    scratch = Output.TypedOutputScratch(Float64)
    dbase_state = similar(base_state)
    dinbox = similar(inbox)
    Output.typed_output_pullback!(
        dbase_state,
        dinbox,
        gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )
    @test gradient.readout_weight[:, selected] == tape.packet[:, selected]
    # Exact-zero plateau/adaptation coordinates are legitimate for one hard
    # trajectory; every nonzero packet coordinate must still receive its exact
    # local readout gradient.
    @test count(!iszero, @view(gradient.readout_weight[:, selected])) >= 37
    @test all(iszero, @view(gradient.readout_weight[:, 1:(selected - 1)]))
    @test all(iszero, @view(gradient.readout_weight[:, (selected + 1):end]))
    @test all(iszero, @view(dinbox[:, 1:(selected - 1)]))
    @test all(iszero, @view(dinbox[:, (selected + 1):end]))
    @test all(iszero, @view(dbase_state[:, 1:(selected - 1)]))
    @test all(iszero, @view(dbase_state[:, (selected + 1):end]))
end

function _conditional_vjp_check(; at_bounds::Bool)
    rng = MersenneTwister(at_bounds ? 0xb01d5 : 0x0a11ce)
    parameters = Output.initialize_parameters(Float64)
    if at_bounds
        @inbounds for parameter in 1:Cell.PARAM_DIM
            parameters.cell_raw[parameter, 1] = isodd(parameter) ? -9.0 : 9.0
        end
    end
    inbox = abs.(0.05 .* randn(
        rng,
        Float64,
        Cell.INPUT_DIM,
        Output.OUTPUT_CELLS,
    ))
    direction = randn(rng, Float64, Output.OUTPUT_CELLS)
    cache = Output.TypedOutputCache(parameters)
    base_state = _interior_base(cache)
    tape = Output.TypedOutputTape(Float64)
    scratch = Output.TypedOutputScratch(Float64)
    gradient = Output.TypedOutputGradient(Float64)
    continuous = zeros(Float64, Output.OUTPUT_CELLS)
    events = similar(continuous)
    Output.typed_output_forward!(
        continuous,
        events,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    baseline_events = copy(events)
    dbase_state = similar(base_state)
    dinbox = similar(inbox)
    Output.typed_output_pullback!(
        dbase_state,
        dinbox,
        gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )

    function objective()
        Output.refresh_cache!(cache, parameters)
        Output.typed_output_forward!(
            continuous,
            events,
            tape,
            base_state,
            inbox,
            parameters,
            cache,
        )
        return dot(continuous, direction), copy(events)
    end

    epsilon = at_bounds ? 1.0e-4 : 1.0e-5
    relative_tolerance = at_bounds ? 3.0e-3 : 3.0e-4
    absolute_tolerance = at_bounds ? 2.0e-7 : 5.0e-8

    # One complete cell covers every bounded transform and cross-coupled raw
    # parameter, including normalized basal roles.
    @inbounds for parameter in 1:Cell.PARAM_DIM
        numerical, plus_events, minus_events = _central_difference!(
            parameters.cell_raw,
            CartesianIndex(parameter, 1),
            objective;
            epsilon,
        )
        @test plus_events == baseline_events
        @test minus_events == baseline_events
        @test isapprox(
            gradient.cell_raw[parameter, 1],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end

    # The typed inbox VJP is checked on every receptor coordinate of one cell.
    @inbounds for channel in 1:Cell.INPUT_DIM
        numerical, plus_events, minus_events = _central_difference!(
            inbox,
            CartesianIndex(channel, 1),
            objective;
            epsilon,
        )
        @test plus_events == baseline_events
        @test minus_events == baseline_events
        @test isapprox(
            dinbox[channel, 1],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end

    # The mandatory candidate transition returns its full base-state
    # cotangent to the common-pass integrator instead of resetting internally.
    @inbounds for state in 1:Cell.STATE_DIM
        numerical, plus_events, minus_events = _central_difference!(
            base_state,
            CartesianIndex(state, 1),
            objective;
            epsilon,
        )
        @test plus_events == baseline_events
        @test minus_events == baseline_events
        @test isapprox(
            dbase_state[state, 1],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end

    # Every anatomical lane has an independent local readout parameter. Check
    # all 47 x 22 weights rather than sampling the former scalar shortcut.
    @inbounds for output in 1:Output.OUTPUT_CELLS
        for lane in 1:Output.READOUT_DIM
            numerical, plus_events, minus_events = _central_difference!(
                parameters.readout_weight,
                CartesianIndex(lane, output),
                objective;
                epsilon,
            )
            @test plus_events == baseline_events
            @test minus_events == baseline_events
            @test isapprox(
                gradient.readout_weight[lane, output],
                numerical;
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        end
    end
    @inbounds for output in 1:Output.OUTPUT_CELLS
        numerical, plus_events, minus_events = _central_difference!(
            parameters.bias,
            output,
            objective;
            epsilon,
        )
        @test plus_events == baseline_events
        @test minus_events == baseline_events
        @test isapprox(
            gradient.bias[output],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end
    return nothing
end

@testset "Float64 conditional VJP at ordinary parameters" begin
    _conditional_vjp_check(at_bounds=false)
end

@testset "Float64 conditional VJP near transform bounds" begin
    _conditional_vjp_check(at_bounds=true)
end

function _initial_state_vjp_check(; at_bounds::Bool)
    rng = MersenneTwister(at_bounds ? 0x1b01d : 0x1a11ce)
    parameters = Output.initialize_parameters(Float64)
    if at_bounds
        @inbounds for parameter in 1:Cell.PARAM_DIM
            parameters.cell_raw[parameter, 1] = isodd(parameter) ? -9.0 : 9.0
        end
    end
    cache = Output.TypedOutputCache(parameters)
    state = Matrix{Float64}(undef, Cell.STATE_DIM, Output.OUTPUT_CELLS)
    direction = randn(rng, Float64, size(state))
    Output.output_initial_state!(state, cache)
    gradient = Output.TypedOutputGradient(Float64)
    scratch = Output.TypedOutputScratch(Float64)
    Output.output_initial_state_pullback!(
        gradient,
        scratch,
        direction,
        cache,
    )
    objective() = begin
        Output.refresh_cache!(cache, parameters)
        Output.output_initial_state!(state, cache)
        return dot(state, direction), zeros(Float64, 0)
    end
    epsilon = at_bounds ? 1.0e-4 : 1.0e-5
    @inbounds for parameter in 1:Cell.PARAM_DIM
        numerical, _, _ = _central_difference!(
            parameters.cell_raw,
            CartesianIndex(parameter, 1),
            objective;
            epsilon,
        )
        @test isapprox(
            gradient.cell_raw[parameter, 1],
            numerical;
            rtol=at_bounds ? 3.0e-3 : 3.0e-4,
            atol=at_bounds ? 2.0e-7 : 5.0e-8,
        )
    end
    return nothing
end

@testset "separate output initial-state VJP normal and bounds" begin
    _initial_state_vjp_check(at_bounds=false)
    _initial_state_vjp_check(at_bounds=true)
end

@testset "hard event is control-only and absent from task VJP" begin
    rng = MersenneTwister(0xc017201)
    parameters = Output.initialize_parameters()
    cache = Output.TypedOutputCache(parameters)
    base_state = _interior_base(cache)
    tape = Output.TypedOutputTape()
    scratch = Output.TypedOutputScratch()
    inbox = abs.(0.1f0 .* randn(
        rng,
        Float32,
        Cell.INPUT_DIM,
        Output.OUTPUT_CELLS,
    ))
    continuous = zeros(Float32, 22)
    hard_event = similar(continuous)
    direction = randn(rng, Float32, 22)
    Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    first_gradient = Output.TypedOutputGradient()
    first_dbase_state = similar(base_state)
    first_dinbox = similar(inbox)
    Output.typed_output_pullback!(
        first_dbase_state,
        first_dinbox,
        first_gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )

    # The recorded event is a diagnostic only.  Changing it cannot alter
    # the continuous conditional VJP; the physical spike states remain fixed
    # in the cell trajectory, exactly as required by conditional BPTT.
    tape.event .= 1.0f0 .- tape.event
    second_gradient = Output.TypedOutputGradient()
    second_dbase_state = similar(base_state)
    second_dinbox = similar(inbox)
    Output.typed_output_pullback!(
        second_dbase_state,
        second_dinbox,
        second_gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )
    @test second_dbase_state == first_dbase_state
    @test second_dinbox == first_dinbox
    @test second_gradient.cell_raw == first_gradient.cell_raw
    @test second_gradient.readout_weight == first_gradient.readout_weight
    @test second_gradient.bias == first_gradient.bias
end

function _allocation_probe()
    rng = MersenneTwister(0xa110c)
    parameters = Output.initialize_parameters()
    cache = Output.TypedOutputCache(parameters)
    base_state = _interior_base(cache)
    tape = Output.TypedOutputTape()
    scratch = Output.TypedOutputScratch()
    gradient = Output.TypedOutputGradient()
    inbox = abs.(0.1f0 .* randn(
        rng,
        Float32,
        Cell.INPUT_DIM,
        Output.OUTPUT_CELLS,
    ))
    continuous = zeros(Float32, 22)
    hard_event = similar(continuous)
    direction = randn(rng, Float32, 22)
    dbase_state = similar(base_state)
    dinbox = similar(inbox)
    initial_state = similar(base_state)
    initial_bar = randn(rng, Float32, size(base_state))
    Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    Output.typed_output_pullback!(
        dbase_state,
        dinbox,
        gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )
    Output.output_initial_state!(initial_state, cache)
    Output.output_initial_state_pullback!(
        gradient,
        scratch,
        initial_bar,
        cache,
    )
    Output.clear_gradient!(gradient)
    initial_forward_bytes = @allocated Output.output_initial_state!(
        initial_state,
        cache,
    )
    initial_pullback_bytes = @allocated Output.output_initial_state_pullback!(
        gradient,
        scratch,
        initial_bar,
        cache,
    )
    Output.clear_gradient!(gradient)
    forward_bytes = @allocated Output.typed_output_forward!(
        continuous,
        hard_event,
        tape,
        base_state,
        inbox,
        parameters,
        cache,
    )
    pullback_bytes = @allocated Output.typed_output_pullback!(
        dbase_state,
        dinbox,
        gradient,
        scratch,
        tape,
        parameters,
        cache,
        direction,
    )
    count_bytes = @allocated Output.hard_event_count(tape)
    return (
        initial_forward_bytes,
        initial_pullback_bytes,
        forward_bytes,
        pullback_bytes,
        count_bytes,
    )
end

@testset "Float32 fixed hot path allocates nothing" begin
    @test _allocation_probe() == (0, 0, 0, 0, 0)
end
