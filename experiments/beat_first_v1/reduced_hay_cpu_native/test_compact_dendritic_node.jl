using Test
using Random
using LinearAlgebra

module CompactDendriticNodeTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CompactDendriticNode.jl"))
end

const H = CompactDendriticNodeTestHarness
const Cell = H.ActiveApicalCell
const Node = H.CompactDendriticNode

function evaluate_node(
    raw::Vector{T},
    drive::Vector{T},
) where {T<:AbstractFloat}
    cache = Cell.transform_parameters(raw)
    trace = Node.NodeTrace(T)
    payload = Vector{T}(undef, Node.PAYLOAD_DIM)
    Node.node_forward!(payload, trace, drive, cache)
    return payload, trace
end

function finite_difference(values, index, objective; epsilon=1.0e-5)
    plus = copy(values)
    minus = copy(values)
    plus[index] += epsilon
    minus[index] -= epsilon
    return (objective(plus) - objective(minus)) / (2epsilon)
end

function raw_for_value(raw, name::Symbol, value)
    index = findfirst(==(name), Cell.PARAMETER_NAMES)
    lo = Cell.PARAMETER_LOWER[index]
    hi = Cell.PARAMETER_UPPER[index]
    probability = clamp((value - lo) / (hi - lo), 1.0f-6, 1.0f0 - 1.0f-6)
    result = copy(raw)
    result[index] = log(probability / (1.0f0 - probability))
    return result
end

@testset "compact dendritic node contract" begin
    @test Node.PHASE_COUNT == 3
    @test Node.DRIVE_DIM == 9
    @test Node.ANALOG_DIM == 2
    @test Node.PAYLOAD_DIM == 3
    @test Node.CENTERED_MARGIN_SCALE == 5.0f0
    @test Node.MEAN_PLATEAU_SCALE == 0.03125f0
    @test Node.EXCITATORY_DRIVE_SCALE == 0.035f0
    @test Node.INHIBITORY_DRIVE_SCALE == 1.0f0
    @test Cell.STATE_DIM == 48
    @test size(Node.NodeTrace().states) == (48, 4)
    @test length(Node.NodeTrace().driven_input) == 27
end

@testset "signed E/I/NMDA drive and silent relaxation" begin
    raw = Cell.default_raw_parameters()
    cache = Cell.transform_parameters(raw)
    quiet = zeros(Float32, Node.DRIVE_DIM)
    positive = copy(quiet)
    negative = copy(quiet)
    positive[1] = 1.5f0
    negative[1] = -1.5f0

    quiet_payload, quiet_trace = evaluate_node(raw, quiet)
    positive_payload, positive_trace = evaluate_node(raw, positive)
    negative_payload, negative_trace = evaluate_node(raw, negative)

    phase1_positive = @view positive_trace.states[:, 2]
    phase1_negative = @view negative_trace.states[:, 2]
    @test phase1_positive[Cell.state_index(1, Cell.FIELD_AMPA)] > 0.0f0
    @test phase1_positive[Cell.state_index(1, Cell.FIELD_NMDA)] > 0.0f0
    @test phase1_positive[Cell.state_index(1, Cell.FIELD_GABA)] == 0.0f0
    @test phase1_negative[Cell.state_index(1, Cell.FIELD_AMPA)] == 0.0f0
    @test phase1_negative[Cell.state_index(1, Cell.FIELD_NMDA)] == 0.0f0
    @test phase1_negative[Cell.state_index(1, Cell.FIELD_GABA)] > 0.0f0
    @test positive_trace.driven_input[Cell.input_index(1, Cell.INPUT_AMPA)] ==
          positive[1] * Node.EXCITATORY_DRIVE_SCALE
    @test positive_trace.driven_input[Cell.input_index(1, Cell.INPUT_NMDA)] ==
          positive[1] * Node.EXCITATORY_DRIVE_SCALE
    @test negative_trace.driven_input[Cell.input_index(1, Cell.INPUT_GABA)] ==
          -negative[1] * Node.INHIBITORY_DRIVE_SCALE

    # Slow NMDA evidence remains present while the external phase input is
    # exactly silent; this is relaxation, not repeated DC injection.
    @test all(iszero, positive_trace.silent_input)
    @test positive_trace.states[Cell.state_index(1, Cell.FIELD_NMDA), 3] > 0.0f0
    @test positive_trace.states[Cell.state_index(1, Cell.FIELD_NMDA), 4] > 0.0f0
    @test positive_payload[Node.CENTERED_MARGIN_INDEX] >
          quiet_payload[Node.CENTERED_MARGIN_INDEX] >
          negative_payload[Node.CENTERED_MARGIN_INDEX]
    @test positive_payload[Node.MEAN_PLATEAU_INDEX] >=
          quiet_payload[Node.MEAN_PLATEAU_INDEX]

    final_state = @view positive_trace.states[:, end]
    physical_plateau_mean = sum(
        final_state[Cell.state_index(branch, Cell.FIELD_PLATEAU)]
        for branch in 1:Cell.N_BASAL
    ) / Cell.N_BASAL
    expected_centered_margin = (
        positive_trace.margins[end] -
        (cache.soma_rest - cache.soma_threshold)
    ) / Node.CENTERED_MARGIN_SCALE
    @test positive_payload[Node.CENTERED_MARGIN_INDEX] ==
          expected_centered_margin
    @test positive_payload[Node.MEAN_PLATEAU_INDEX] ==
          physical_plateau_mean / Node.MEAN_PLATEAU_SCALE
    @test isfinite(positive_payload[Node.CENTERED_MARGIN_INDEX])
    @test 0.0f0 <= positive_payload[Node.MEAN_PLATEAU_INDEX] <=
          inv(Node.MEAN_PLATEAU_SCALE)

    # Reconstruct the stored trajectory independently: input is present once,
    # then two exactly zero-input cell steps.
    manual0 = Cell.initial_state(cache)
    manual1 = Cell.cell_step_cached_functional(
        manual0,
        positive_trace.driven_input,
        cache,
    )
    manual2 = Cell.cell_step_cached_functional(
        manual1,
        positive_trace.silent_input,
        cache,
    )
    manual3 = Cell.cell_step_cached_functional(
        manual2,
        positive_trace.silent_input,
        cache,
    )
    @test manual1 == positive_trace.states[:, 2]
    @test manual2 == positive_trace.states[:, 3]
    @test manual3 == positive_trace.states[:, 4]

    repeated2 = Cell.cell_step_cached_functional(
        manual1,
        positive_trace.driven_input,
        cache,
    )
    @test repeated2 != manual2

    apical_positive = copy(quiet)
    apical_negative = copy(quiet)
    apical_positive[end] = 1.5f0
    apical_negative[end] = -1.5f0
    positive_apical_payload, _ = evaluate_node(raw, apical_positive)
    negative_apical_payload, _ = evaluate_node(raw, apical_negative)
    @test positive_apical_payload[Node.CENTERED_MARGIN_INDEX] >
          negative_apical_payload[Node.CENTERED_MARGIN_INDEX]

    zero_payload, zero_trace = evaluate_node(raw, zeros(Float32, Node.DRIVE_DIM))
    @test all(iszero, zero_trace.driven_input)
    @test zero_payload[Node.CENTERED_MARGIN_INDEX] == 0.0f0
end

@testset "calibrated positive and negative semantic response" begin
    raw = Cell.default_raw_parameters()
    quiet_payload, _ = evaluate_node(raw, zeros(Float32, Node.DRIVE_DIM))
    quiet_margin = quiet_payload[Node.CENTERED_MARGIN_INDEX]
    for amplitude in (0.05f0, 0.1f0, 0.2f0, 0.5f0)
        positive_drive = zeros(Float32, Node.DRIVE_DIM)
        negative_drive = zeros(Float32, Node.DRIVE_DIM)
        positive_drive[1] = amplitude
        negative_drive[1] = -amplitude
        positive_payload, _ = evaluate_node(raw, positive_drive)
        negative_payload, _ = evaluate_node(raw, negative_drive)
        positive_response = abs(
            positive_payload[Node.CENTERED_MARGIN_INDEX] - quiet_margin,
        )
        negative_response = abs(
            negative_payload[Node.CENTERED_MARGIN_INDEX] - quiet_margin,
        )
        @test positive_response > 0.0f0
        @test negative_response > 0.0f0
        @test max(positive_response, negative_response) /
              min(positive_response, negative_response) <= 2.0f0
    end
end

@testset "hard event is an explicit control coordinate" begin
    raw = Cell.default_raw_parameters()
    cache, derivative_cache = Cell.parameter_caches(raw)
    payload = zeros(Float32, Node.PAYLOAD_DIM)
    trace = Node.NodeTrace()

    # The output must remain a true hard bit for every drive, never a soft
    # probability hidden among the analog coordinates.
    for amplitude in range(-8.0f0, 8.0f0; length=65)
        drive = fill(amplitude, Node.DRIVE_DIM)
        event = Node.node_forward!(payload, trace, drive, cache)
        @test event == 0.0f0 || event == 1.0f0
        @test payload[Node.HARD_EVENT_INDEX] === event
    end

    # Reset and adaptation can make an early spike disappear by phase three.
    # The node-level control coordinate is OR over the complete integration
    # window, so this phase-one-only event must remain visible.
    early_raw = raw_for_value(raw, :soma_threshold_gap, 1.01f0)
    early_raw = raw_for_value(early_raw, :adaptation_gain, 2.0f0)
    early_raw = raw_for_value(early_raw, :adaptation_coupling, 5.0f0)
    early_raw = raw_for_value(early_raw, :adaptation_decay, 0.1f0)
    early_cache = Cell.transform_parameters(early_raw)
    early_drive = fill(
        0.06f0 / Node.EXCITATORY_DRIVE_SCALE,
        Node.DRIVE_DIM,
    )
    early_event = Node.node_forward!(payload, trace, early_drive, early_cache)
    @test trace.events == Float32[1.0, 0.0, 0.0]
    @test argmax(trace.margins) == 1
    @test early_event == 1.0f0
    @test payload[Node.HARD_EVENT_INDEX] == 1.0f0

    drive = fill(0.35f0, Node.DRIVE_DIM)
    Node.node_forward!(payload, trace, drive, cache)
    ddrive = zeros(Float32, Node.DRIVE_DIM)
    draw = zeros(Float32, Cell.PARAM_DIM)
    scratch = Node.NodeScratch()
    danalog = zeros(Float32, Node.ANALOG_DIM)
    Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        danalog,
    )
    @test all(iszero, ddrive)
    @test all(iszero, draw)
    @test_throws DimensionMismatch Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        zeros(Float32, Node.PAYLOAD_DIM),
    )

    # ActiveApicalCell explicitly supports a triangular final-event surrogate.
    # Find a drive whose final margin lies within that compact support and
    # verify that only the separate event cotangent enables the path.
    near_drive = nothing
    for amplitude in range(0.01f0, 20.0f0; length=2000)
        candidate = fill(amplitude, Node.DRIVE_DIM)
        Node.node_forward!(payload, trace, candidate, cache)
        if abs(trace.margins[end]) < Cell.SPIKE_SURROGATE_WIDTH
            near_drive = candidate
            break
        end
    end
    @test near_drive !== nothing
    if near_drive !== nothing
        Node.node_forward!(payload, trace, near_drive, cache)
        Node.node_pullback!(
            ddrive,
            draw,
            scratch,
            trace,
            near_drive,
            cache,
            derivative_cache,
            danalog,
            1.0f0,
        )
        @test norm(ddrive) > 0.0f0
        @test norm(draw) > 0.0f0
    end
end


@testset "OR surrogate is the deterministic maximum-margin phase" begin
    raw32 = Cell.default_raw_parameters()
    raw32 = raw_for_value(raw32, :soma_threshold_gap, 1.01f0)
    raw32 = raw_for_value(raw32, :adaptation_gain, 2.0f0)
    raw32 = raw_for_value(raw32, :adaptation_coupling, 5.0f0)
    raw32 = raw_for_value(raw32, :adaptation_decay, 0.1f0)
    raw = Float64.(raw32)
    drive = fill(0.06 / Float64(Node.EXCITATORY_DRIVE_SCALE), Node.DRIVE_DIM)
    cache, derivative_cache = Cell.parameter_caches(raw)
    payload = zeros(Float64, Node.PAYLOAD_DIM)
    trace = Node.NodeTrace(Float64)
    Node.node_forward!(payload, trace, drive, cache)
    @test trace.events == Float64[1.0, 0.0, 0.0]
    @test argmax(trace.margins) == 1
    @test abs(maximum(trace.margins)) < Float64(Cell.SPIKE_SURROGATE_WIDTH)

    ddrive = zeros(Float64, Node.DRIVE_DIM)
    draw = zeros(Float64, Cell.PARAM_DIM)
    scratch = Node.NodeScratch(Float64)
    Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        zeros(Float64, Node.ANALOG_DIM),
        1.0,
    )

    function surrogate_objective(candidate_drive)
        _, candidate_trace = evaluate_node(raw, candidate_drive)
        @test argmax(candidate_trace.margins) == 1
        return Cell.spike_surrogate_value(maximum(candidate_trace.margins))
    end
    for index in eachindex(drive)
        numerical = finite_difference(drive, index, surrogate_objective)
        @test isapprox(ddrive[index], numerical; rtol=7.0e-4, atol=5.0e-7)
    end
end

@testset "event-stable exact VJP of both continuous outputs" begin
    rng = MersenneTwister(0xC04D)
    raw = Float64.(Cell.default_raw_parameters())
    drive = [
        0.31,
        -0.27,
        0.43,
        -0.36,
        0.22,
        -0.19,
        0.38,
        -0.24,
        0.17,
    ]
    direction = randn(rng, Float64, Node.ANALOG_DIM)
    cache, derivative_cache = Cell.parameter_caches(raw)
    payload = zeros(Float64, Node.PAYLOAD_DIM)
    trace = Node.NodeTrace(Float64)
    Node.node_forward!(payload, trace, drive, cache)
    baseline_events = copy(trace.events)

    ddrive = zeros(Float64, Node.DRIVE_DIM)
    draw = zeros(Float64, Cell.PARAM_DIM)
    scratch = Node.NodeScratch(Float64)
    Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        direction,
    )

    drive_objective(candidate) = begin
        candidate_payload, candidate_trace = evaluate_node(raw, candidate)
        @test candidate_trace.events == baseline_events
        dot(@view(candidate_payload[1:Node.ANALOG_DIM]), direction)
    end
    raw_objective(candidate) = begin
        candidate_payload, candidate_trace = evaluate_node(candidate, drive)
        @test candidate_trace.events == baseline_events
        dot(@view(candidate_payload[1:Node.ANALOG_DIM]), direction)
    end

    for index in eachindex(drive)
        numerical = finite_difference(drive, index, drive_objective)
        @test isapprox(ddrive[index], numerical; rtol=5.0e-4, atol=4.0e-7)
    end
    for index in eachindex(raw)
        numerical = finite_difference(raw, index, raw_objective)
        @test isapprox(draw[index], numerical; rtol=1.5e-3, atol=2.0e-6)
    end
end

@testset "local high-dimensional state and allocation-free Float32 hot path" begin
    raw = Cell.default_raw_parameters()
    cache, derivative_cache = Cell.parameter_caches(raw)
    drive = Float32[0.6, -0.4, 0.7, -0.3, 0.5, -0.2, 0.8, -0.1, 0.4]
    payload = zeros(Float32, Node.PAYLOAD_DIM)
    trace = Node.NodeTrace()
    scratch = Node.NodeScratch()
    ddrive = zeros(Float32, Node.DRIVE_DIM)
    draw = zeros(Float32, Cell.PARAM_DIM)
    danalog = Float32[0.7, -0.4]

    Node.node_forward!(payload, trace, drive, cache)
    Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        danalog,
    )

    final_state = @view trace.states[:, end]
    @test length(final_state) == 48
    @test count(!iszero, final_state) > Node.PAYLOAD_DIM
    @test any(
        final_state[Cell.state_index(branch, Cell.FIELD_NMDA)] > 0.0f0
        for branch in 1:Cell.N_BASAL
    )
    @test any(
        final_state[Cell.state_index(branch, Cell.FIELD_GABA)] > 0.0f0
        for branch in 1:Cell.N_BASAL
    )
    @test @allocated(Node.node_forward!(payload, trace, drive, cache)) == 0
    @test @allocated(Node.node_pullback!(
        ddrive,
        draw,
        scratch,
        trace,
        drive,
        cache,
        derivative_cache,
        danalog,
    )) == 0
end
