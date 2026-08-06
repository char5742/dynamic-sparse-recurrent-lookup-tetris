using Test
using Random
using LinearAlgebra

module DendriticForestOutputTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "CompactDendriticNode.jl"))
include(joinpath(@__DIR__, "DendriticForestOutput.jl"))
end

const H = DendriticForestOutputTestHarness
const Cell = H.ActiveApicalCell
const Delta = H.CandidateDeltaInput
const Node = H.CompactDendriticNode
const Output = H.DendriticForestOutput

function evaluate(
    parameters,
    anchors,
    context,
    placement=zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS),
)
    cache = Output.ForestOutputCache(parameters)
    tape = Output.ForestOutputTape(eltype(anchors))
    raw = zeros(eltype(anchors), Output.OUTPUT_CHANNELS)
    events = similar(raw)
    Output.forest_output_forward!(
        raw, events, tape, anchors, context, placement, parameters, cache,
    )
    return raw, events, tape
end

function central_difference!(values, index, objective; epsilon=1.0e-5)
    original = values[index]
    values[index] = original + epsilon
    plus = objective()
    values[index] = original - epsilon
    minus = objective()
    values[index] = original
    return (plus - minus) / (2epsilon)
end

@testset "22 private output cells and signed context contract" begin
    parameters = Output.initialize_parameters()
    @test size(parameters.cell_raw) == (Cell.PARAM_DIM, 22)
    @test size(parameters.anchor_weight) == (3, 34, 2, 22)
    @test size(parameters.context_weight) == (81, 22)
    @test size(parameters.placement_weight) == (240, 22)
    @test size(parameters.cascade_weight) == (3, 21)
    @test Output.stored_parameter_count(parameters) == 12_669

    common = Delta.StateCommon()
    materialization = Delta.CandidateMaterialization()
    common.queue[3, 2] = 0x01
    common.ren[1] = 15.0f0
    common.back_to_back[1] = 1.0f0
    materialization.aux[1] = 0.25f0
    context = zeros(Float32, Output.CONTEXT_DIM)
    Output.fill_context!(context, common, materialization)
    @test count(==(1.0f0), @view(context[1:42])) == 1
    @test context[43] == 0.0f0
    @test context[44] == 1.0f0
    @test context[45] == -0.5f0
end

@testset "plane, anchor and hard-event coordinates remain causal" begin
    parameters = Output.initialize_parameters()
    fill!(parameters.anchor_weight, 0.0f0)
    fill!(parameters.context_weight, 0.0f0)
    parameters.anchor_weight[1, 1, Output.BEFORE_PLANE, 1] = 1.5f0
    parameters.anchor_weight[1, 1, Output.AFTER_PLANE, 1] = -2.0f0
    parameters.anchor_weight[1, 2, Output.BEFORE_PLANE, 1] = 2.5f0
    parameters.anchor_weight[3, 3, Output.AFTER_PLANE, 1] = 3.0f0
    context = zeros(Float32, 81)

    function response(coordinate, anchor, plane)
        anchors = zeros(Float32, 3, 34, 2)
        anchors[coordinate, anchor, plane] = 1.0f0
        raw, _, _ = evaluate(parameters, anchors, context)
        return raw[1]
    end
    quiet = response(1, 4, 1)
    @test response(1, 1, 1) != quiet
    @test response(1, 1, 2) != response(1, 1, 1)
    @test response(1, 2, 1) != response(1, 1, 1)
    @test response(3, 3, 2) != quiet

    anchors = zeros(Float32, 3, 34, 2)
    anchors[3, 3, 2] = 1.0f0
    raw, events, tape = evaluate(parameters, anchors, context)
    cache = Output.ForestOutputCache(parameters)
    gradient = Output.ForestOutputGradient()
    scratch = Output.ForestOutputScratch()
    anchor_bar = similar(anchors)
    context_bar = similar(context)
    raw_bar = zeros(Float32, 22)
    placement = zeros(UInt8, 24, 10)
    raw_bar[1] = 1.0f0
    Output.forest_output_pullback!(
        anchor_bar, context_bar, gradient, scratch, tape, anchors, context,
        placement, parameters, cache, raw_bar,
    )
    @test anchor_bar[3, 3, 2] != 0.0f0
    @test all(event -> event == 0.0f0 || event == 1.0f0, events)
    @test 0 <= Output.hard_event_count(tape) <= Output.hard_event_denominator()
end

@testset "placement identity and auxiliary cascade remove structural aliases" begin
    parameters = Output.initialize_parameters()
    fill!(parameters.anchor_weight, 0.0f0)
    fill!(parameters.context_weight, 0.0f0)
    fill!(parameters.placement_weight, 0.0f0)
    fill!(parameters.cascade_weight, 0.0f0)
    anchors = zeros(Float32, 3, 34, 2)
    context = zeros(Float32, 81)

    placement_a = zeros(UInt8, 24, 10)
    placement_b = zeros(UInt8, 24, 10)
    placement_a[24, 1:4] .= 0x01
    placement_b[23, 1:4] .= 0x01
    @inbounds for column in 1:4
        position_a = 24 + (column - 1) * 24
        position_b = 23 + (column - 1) * 24
        parameters.placement_weight[position_a, 1] = 2.0f0
        parameters.placement_weight[position_b, 1] = -2.0f0
    end
    raw_a, _, _ = evaluate(parameters, anchors, context, placement_a)
    raw_b, _, _ = evaluate(parameters, anchors, context, placement_b)
    @test raw_a[1] != raw_b[1]

    # Force auxiliary cell 2 to emit an exact final hard event, then show that
    # coordinate three of its compact payload causally drives the Q cell.
    fill!(parameters.placement_weight, 0.0f0)
    # Positive semantic drives are intentionally calibrated at the compact
    # node boundary.  Compensate by the named scale here so this mechanism
    # isolation test reproduces its pre-calibration physical AMPA/NMDA drive;
    # do not hide the boundary contract behind an unrelated large literal.
    excitatory_contact_calibration = inv(Node.EXCITATORY_DRIVE_SCALE)
    @views parameters.anchor_weight[:, :, :, 2] .=
        excitatory_contact_calibration
    anchors[1:2, :, :] .= 0.1f0
    anchors[3, :, :] .= 1.0f0
    without_event, _, tape = evaluate(parameters, anchors, context, placement_a)
    @test tape.payload[3, 2] == 1.0f0
    parameters.cascade_weight[3, 1] =
        3.0f0 * excitatory_contact_calibration
    with_event, _, _ = evaluate(parameters, anchors, context, placement_a)
    @test with_event[1] != without_event[1]
end

@testset "Q context gradient is state-dependent, not a single ridge" begin
    rng = MersenneTwister(0xc45cade)
    parameters = Output.initialize_parameters()
    # Restore the pre-boundary-calibration physical basal/apical drive so this
    # test probes nonlinear state dependence rather than the infinitesimal
    # linear regime of the newly calibrated semantic input.
    semantic_contact_calibration = inv(Node.EXCITATORY_DRIVE_SCALE)
    parameters.anchor_weight .*= semantic_contact_calibration
    parameters.context_weight .*= semantic_contact_calibration
    cache = Output.ForestOutputCache(parameters)
    context = randn(rng, Float32, 81)
    placement = zeros(UInt8, 24, 10)
    raw_bar = zeros(Float32, 22)
    raw_bar[1] = 1.0f0

    function q_context_bar(anchors)
        _, _, tape = evaluate(parameters, anchors, context, placement)
        gradient = Output.ForestOutputGradient()
        scratch = Output.ForestOutputScratch()
        anchor_bar = zeros(Float32, 3, 34, 2)
        context_bar = zeros(Float32, 81)
        Output.forest_output_pullback!(
            anchor_bar, context_bar, gradient, scratch, tape, anchors, context,
            placement, parameters, cache, raw_bar,
        )
        return context_bar
    end

    quiet_anchors = zeros(Float32, 3, 34, 2)
    active_anchors = randn(rng, Float32, 3, 34, 2)
    @views active_anchors[3, :, :] .= Float32.(rand(rng, Bool, 34, 2))
    quiet_bar = q_context_bar(quiet_anchors)
    active_bar = q_context_bar(active_anchors)
    cosine = dot(quiet_bar, active_bar) / (norm(quiet_bar) * norm(active_bar))
    @test norm(quiet_bar) > 0.0f0
    @test norm(active_bar) > 0.0f0
    @test abs(cosine) < 0.99f0
end

@testset "representative event-stable VJP for every parameter group" begin
    rng = MersenneTwister(0xded1c8)
    parameters = Output.initialize_parameters(Float64)
    # Keep the representative trajectory in the same physical drive regime
    # after CompactDendriticNode introduced its semantic E/I boundary scale.
    semantic_contact_calibration = inv(Float64(Node.EXCITATORY_DRIVE_SCALE))
    parameters.anchor_weight .*= semantic_contact_calibration
    parameters.context_weight .*= semantic_contact_calibration
    parameters.placement_weight .*= semantic_contact_calibration
    # This test compares the conditional continuous VJP with finite
    # differences of the hard forward.  A nonzero hard-event cascade contact
    # deliberately activates the explicit surrogate path, which is not the
    # derivative of that hard objective.  Keep those contacts at zero here;
    # their weight gradients remain testable because the recorded event is a
    # constant input.  CompactDendriticNode's focused test separately checks
    # the max-margin event surrogate against its smooth surrogate objective.
    fill!(
        @view(parameters.cascade_weight[Node.HARD_EVENT_INDEX, :]),
        0.0,
    )
    anchors = 0.2 .* randn(rng, Float64, 3, 34, 2)
    # Hard root events remain exact bits in forward while their returned bar is
    # reserved for the upstream node's explicit event surrogate.
    @views anchors[3, :, :] .= Float64.(rand(rng, Bool, 34, 2))
    context = 0.3 .* randn(rng, Float64, 81)
    placement = zeros(UInt8, 24, 10)
    placement[24, 1] = placement[24, 2] = 0x01
    placement[23, 1] = placement[23, 2] = 0x01
    direction = randn(rng, Float64, 22)
    raw, baseline_events, tape = evaluate(parameters, anchors, context, placement)
    cache = Output.ForestOutputCache(parameters)
    gradient = Output.ForestOutputGradient(Float64)
    scratch = Output.ForestOutputScratch(Float64)
    anchor_bar = zeros(Float64, 3, 34, 2)
    context_bar = zeros(Float64, 81)
    Output.forest_output_pullback!(
        anchor_bar, context_bar, gradient, scratch, tape, anchors, context,
        placement, parameters, cache, direction,
    )

    objective() = begin
        candidate_raw, candidate_events, _ = evaluate(
            parameters, anchors, context, placement,
        )
        @test candidate_events == baseline_events
        dot(candidate_raw, direction)
    end
    cases = (
        (anchors, CartesianIndex(1, 7, 1), anchor_bar[1, 7, 1]),
        (anchors, CartesianIndex(3, 9, 2), anchor_bar[3, 9, 2]),
        (context, 17, context_bar[17]),
        (
            parameters.anchor_weight,
            CartesianIndex(2, 11, 2, 5),
            gradient.anchor_weight[2, 11, 2, 5],
        ),
        (
            parameters.context_weight,
            CartesianIndex(44, 6),
            gradient.context_weight[44, 6],
        ),
        (
            parameters.placement_weight,
            CartesianIndex(24, 7),
            gradient.placement_weight[24, 7],
        ),
        (
            parameters.cascade_weight,
            CartesianIndex(1, 6),
            gradient.cascade_weight[1, 6],
        ),
        (
            parameters.cell_raw,
            CartesianIndex(20, 8),
            gradient.cell_raw[20, 8],
        ),
        (parameters.gain, 4, gradient.gain[4]),
        (parameters.bias, 5, gradient.bias[5]),
    )
    for (values, index, analytic) in cases
        numerical = central_difference!(values, index, objective)
        @test isapprox(analytic, numerical; rtol=3.0e-3, atol=3.0e-5)
    end
    event_auxiliary = findfirst(!iszero, @view(tape.payload[3, 2:22]))
    @test event_auxiliary !== nothing
    if event_auxiliary !== nothing
        event_contact = CartesianIndex(3, event_auxiliary)
        numerical = central_difference!(
            parameters.cascade_weight,
            event_contact,
            objective,
        )
        @test isapprox(
            gradient.cascade_weight[event_contact],
            numerical;
            rtol=3.0e-3,
            atol=3.0e-5,
        )
    end

    # Every supervised channel owns and updates its own cell and scalar head.
    @test all(!iszero, gradient.bias)
    @test all(!iszero, gradient.gain)
    @test all(output -> norm(@view(gradient.cell_raw[:, output])) > 0.0, 1:22)

    accumulated = Output.ForestOutputGradient(Float64)
    Output.accumulate_gradient!(accumulated, gradient)
    Output.accumulate_gradient!(accumulated, gradient)
    @test accumulated.cascade_weight == 2 .* gradient.cascade_weight
    @test accumulated.placement_weight == 2 .* gradient.placement_weight
end

@testset "Float32 hot path allocates nothing" begin
    rng = MersenneTwister(0xa110c)
    parameters = Output.initialize_parameters()
    cache = Output.ForestOutputCache(parameters)
    tape = Output.ForestOutputTape()
    scratch = Output.ForestOutputScratch()
    gradient = Output.ForestOutputGradient()
    anchors = randn(rng, Float32, 3, 34, 2)
    @views anchors[3, :, :] .= Float32.(rand(rng, Bool, 34, 2))
    context = randn(rng, Float32, 81)
    placement = zeros(UInt8, 24, 10)
    placement[24, 1] = placement[24, 2] = 0x01
    placement[23, 1] = placement[23, 2] = 0x01
    raw = zeros(Float32, 22)
    events = zeros(Float32, 22)
    raw_bar = randn(rng, Float32, 22)
    anchor_bar = similar(anchors)
    context_bar = similar(context)

    Output.forest_output_forward!(
        raw, events, tape, anchors, context, placement, parameters, cache,
    )
    Output.forest_output_pullback!(
        anchor_bar, context_bar, gradient, scratch, tape, anchors, context,
        placement, parameters, cache, raw_bar,
    )
    @test @allocated(Output.forest_output_forward!(
        raw, events, tape, anchors, context, placement, parameters, cache,
    )) == 0
    @test @allocated(Output.forest_output_pullback!(
        anchor_bar, context_bar, gradient, scratch, tape, anchors, context,
        placement, parameters, cache, raw_bar,
    )) == 0
    @test @allocated(Output.hard_event_count(tape)) == 0
end
