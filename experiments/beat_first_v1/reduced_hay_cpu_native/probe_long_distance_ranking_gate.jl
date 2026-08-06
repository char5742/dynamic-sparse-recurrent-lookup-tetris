using Test
using LinearAlgebra

module LongDistanceRankingGateHarness
for file in (
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SharedDendriticFactor.jl",
    "TypedSparseAfferents.jl",
    "ContextAfferents.jl",
    "ContinuousDendriticReadout.jl",
    "SpatialDendriticFactors.jl",
    "DendriticDecisionGraph.jl",
    "CandidateDeltaDendriticGraph.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const GateHarness = LongDistanceRankingGateHarness
const GateDelta = GateHarness.CandidateDeltaInput
const GateSpatial = GateHarness.SpatialDendriticFactors
const GateModel = GateHarness.CandidateDeltaDendriticGraph

const GATE_CANDIDATES = 2
const GATE_CUES = 2
const GATE_Q_CHANNEL = 1

"""
Deterministic two-candidate ranking problem.

Both candidates fill one of two mirror-image bottom cavities.  Their local
placement patches are unchanged when the cue changes.  The correct preference
is nevertheless reversed by either (a) a board marker nineteen rows away or
(b) the identity of the first queue token.  Thus an additive
`state_score + candidate_score` model cannot solve the gate: it needs a
state-candidate interaction.
"""
struct LongDistanceRankingProbe
    placement_a::Matrix{UInt8}
    placement_b::Matrix{UInt8}
    teacher_q::Matrix{Float32}
end

function LongDistanceRankingProbe()
    placement_a = zeros(UInt8, GateDelta.BOARD_ROWS, GateDelta.BOARD_COLUMNS)
    placement_b = zeros(UInt8, GateDelta.BOARD_ROWS, GateDelta.BOARD_COLUMNS)
    @inbounds for board_row in 23:24
        placement_a[board_row, 2] = 0x01
        placement_a[board_row, 3] = 0x01
        placement_b[board_row, 8] = 0x01
        placement_b[board_row, 9] = 0x01
    end
    # Column 1 is cue 1 (candidate A wins); column 2 is cue 2 (B wins).
    teacher_q = Float32[1 -1; -1 1]
    return LongDistanceRankingProbe(placement_a, placement_b, teacher_q)
end

@inline function gate_placement(probe::LongDistanceRankingProbe, candidate::Int)
    candidate == 1 && return probe.placement_a
    candidate == 2 && return probe.placement_b
    throw(BoundsError(1:GATE_CANDIDATES, candidate))
end

@inline function gate_preferred_candidate(
    probe::LongDistanceRankingProbe,
    cue::Int,
)
    1 <= cue <= GATE_CUES || throw(BoundsError(1:GATE_CUES, cue))
    return probe.teacher_q[1, cue] > probe.teacher_q[2, cue] ? 1 : 2
end

function _fill_symmetric_cavities!(board)
    fill!(board, 0x00)
    # Two identical 2 x 2 cavities.  Filling one never completes a row, so the
    # exact line-clear map is identical under both remote cues.
    @inbounds for board_row in 23:24, column in (1, 4, 5, 6, 7, 10)
        board[board_row, column] = 0x01
    end
    return board
end

function _fill_neutral_queue!(queue)
    fill!(queue, 0x00)
    @inbounds for token in 1:GateDelta.QUEUE_TOKENS
        queue[mod1(token + 2, GateDelta.QUEUE_PIECES), token] = 0x01
    end
    return queue
end

"""
Prepare one cue state.  `cue_kind == :board` moves a single marker between
row 4 / column 2 and row 4 / column 9.  `cue_kind == :queue` changes only the
first queue token.  In both cases occupancy cardinality is held constant.
"""
function prepare_gate_state!(state, cue_kind::Symbol, cue::Int)
    1 <= cue <= GATE_CUES || throw(BoundsError(1:GATE_CUES, cue))
    common = state.common
    _fill_symmetric_cavities!(common.board)
    _fill_neutral_queue!(common.queue)
    if cue_kind === :board
        common.board[4, cue == 1 ? 2 : 9] = 0x01
    elseif cue_kind === :queue
        common.queue[:, 1] .= 0x00
        common.queue[cue == 1 ? 1 : 7, 1] = 0x01
    else
        throw(ArgumentError("cue_kind must be :board or :queue"))
    end
    common.ren[1] = 0.0f0
    common.back_to_back[1] = 0.0f0
    return state
end

function gate_outputs!(
    output,
    probe::LongDistanceRankingProbe,
    cue_kind::Symbol,
    parameters,
    cache,
    state,
    worker,
)
    size(output) == (22, GATE_CANDIDATES, GATE_CUES) ||
        throw(DimensionMismatch("gate output must have shape (22, 2, 2)"))
    raw = zeros(Float32, 22)
    @inbounds for cue in 1:GATE_CUES
        prepare_gate_state!(state, cue_kind, cue)
        GateModel.prepare_state!(state, worker, parameters, cache)
        for candidate in 1:GATE_CANDIDATES
            GateModel.forward_candidate!(
                raw,
                worker,
                state,
                parameters,
                cache,
                gate_placement(probe, candidate),
                0.0f0,
            )
            output[:, candidate, cue] .= raw
        end
    end
    return output
end

"""
Mixed state-candidate difference.

For one output channel this is

    (q_B(cue2) - q_A(cue2)) - (q_B(cue1) - q_A(cue1)).

It is identically zero for every additive state/candidate score, whereas a
positive value has exactly the preference orientation required by the gate.
"""
@inline function gate_mixed_difference(output, channel::Int=GATE_Q_CHANNEL)
    return (output[channel, 2, 2] - output[channel, 1, 2]) -
           (output[channel, 2, 1] - output[channel, 1, 1])
end

function gate_mixed_gradient!(
    gradient,
    probe::LongDistanceRankingProbe,
    cue_kind::Symbol,
    parameters,
    cache,
    state,
    worker;
    channel::Int=GATE_Q_CHANNEL,
)
    GateModel.clear_gradient!(gradient)
    raw = zeros(Float32, 22)
    raw_bar = zeros(Float32, 22)
    @inbounds for cue in 1:GATE_CUES
        prepare_gate_state!(state, cue_kind, cue)
        GateModel.prepare_state!(state, worker, parameters, cache)
        for candidate in 1:GATE_CANDIDATES
            GateModel.forward_candidate!(
                raw,
                worker,
                state,
                parameters,
                cache,
                gate_placement(probe, candidate),
                0.0f0,
            )
            fill!(raw_bar, 0.0f0)
            # +B2 -A2 -B1 +A1
            raw_bar[channel] = Float32((cue == 2 ? 1 : -1) *
                                       (candidate == 2 ? 1 : -1))
            GateModel.pullback_candidate!(
                gradient,
                raw,
                raw_bar,
                worker,
                state,
                parameters,
                cache,
            )
        end
        GateModel.finish_state_pullback!(
            gradient,
            worker,
            state,
            parameters,
            cache,
        )
    end
    return gradient
end

function _capture_local_candidate_features!(
    destination,
    positions,
    probe,
    cue_kind,
    cue,
    candidate,
    state,
    worker,
    parameters,
    cache,
)
    prepare_gate_state!(state, cue_kind, cue)
    GateModel.prepare_state!(state, worker, parameters, cache)
    GateModel.prepare_candidate!(
        worker,
        state,
        parameters,
        cache,
        gate_placement(probe, candidate),
        0.0f0,
    )
    empty!(positions)
    @inbounds for value in worker.affected
        push!(positions, Int(value))
    end
    size(destination) == (
        GateHarness.SharedDendriticFactor.FEATURE_DIM,
        GateSpatial.POSITION_COUNT,
    ) || throw(DimensionMismatch(
        "local feature capture must have shape (27, 240)",
    ))
    @inbounds for (slot, position) in enumerate(positions)
        destination[:, slot] .= worker.candidate_features[:, position]
    end
    return destination, positions
end

@testset "long-distance ranking reversal probe construction" begin
    probe = LongDistanceRankingProbe()
    @test gate_preferred_candidate(probe, 1) == 1
    @test gate_preferred_candidate(probe, 2) == 2
    @test count(value -> !iszero(value), probe.placement_a) == 4
    @test count(value -> !iszero(value), probe.placement_b) == 4

    parameters = GateModel.initialize_model()
    cache = GateModel.ModelCache(parameters)
    state = GateModel.ModelState()
    worker = GateModel.ModelWorker()
    left_features = zeros(
        Float32,
        GateHarness.SharedDendriticFactor.FEATURE_DIM,
        GateSpatial.POSITION_COUNT,
    )
    right_features = similar(left_features)
    left_positions = Int[]
    right_positions = Int[]

    for cue_kind in (:board, :queue), candidate in 1:GATE_CANDIDATES
        _capture_local_candidate_features!(
            left_features,
            left_positions,
            probe,
            cue_kind,
            1,
            candidate,
            state,
            worker,
            parameters,
            cache,
        )
        saved_feature_count = length(left_positions)
        saved_features = copy(@view left_features[:, 1:saved_feature_count])
        saved_positions = copy(left_positions)
        _capture_local_candidate_features!(
            right_features,
            right_positions,
            probe,
            cue_kind,
            2,
            candidate,
            state,
            worker,
            parameters,
            cache,
        )
        @test right_positions == saved_positions
        @test @view(right_features[:, 1:length(right_positions)]) == saved_features
        @test all(position -> begin
            row = mod1(position, GateDelta.BOARD_ROWS)
            row >= 22
        end, right_positions)
    end
end

@testset "current graph exposes a trainable mixed state-candidate feature" begin
    probe = LongDistanceRankingProbe()
    parameters = GateModel.initialize_model()
    cache = GateModel.ModelCache(parameters)
    state = GateModel.ModelState()
    worker = GateModel.ModelWorker()
    gradient = GateModel.ModelGradient(parameters; active_program_capacity=4_096)
    outputs = zeros(Float32, 22, GATE_CANDIDATES, GATE_CUES)

    for cue_kind in (:board, :queue)
        gate_outputs!(
            outputs,
            probe,
            cue_kind,
            parameters,
            cache,
            state,
            worker,
        )
        initial_mixed = gate_mixed_difference(outputs)
        gate_mixed_gradient!(
            gradient,
            probe,
            cue_kind,
            parameters,
            cache,
            state,
            worker,
        )

        gain_gradient = @view gradient.decision.readout.gain[GATE_Q_CHANNEL, :]
        gain_norm = norm(gain_gradient)
        @test isfinite(initial_mixed)
        @test gain_norm > 1.0f-6
        # An output bias can encode a cue-independent preference only; it must
        # cancel exactly from the mixed difference.
        @test abs(gradient.decision.readout.bias[GATE_Q_CHANNEL]) <= 2.0f-6

        # Because the final projection is linear, this is a particularly clean
        # directional check of the grouped exact reverse: cell dynamics and
        # hard-event tapes stay fixed while the mixed feature is exposed.
        direction = Vector{Float32}(gain_gradient ./ gain_norm)
        gain = @view parameters.decision.readout.gain[GATE_Q_CHANNEL, :]
        epsilon = 2.0f-2
        @. gain += epsilon * direction
        gate_outputs!(outputs, probe, cue_kind, parameters, cache, state, worker)
        plus = gate_mixed_difference(outputs)
        @. gain -= 2.0f0 * epsilon * direction
        gate_outputs!(outputs, probe, cue_kind, parameters, cache, state, worker)
        minus = gate_mixed_difference(outputs)
        @. gain += epsilon * direction
        finite_difference = (plus - minus) / (2.0f0 * epsilon)
        @test finite_difference ≈ gain_norm rtol=3.0f-3 atol=2.0f-5

        if cue_kind === :queue
            @test norm(gradient.decision.context_raw) > 1.0f-7
        else
            spatial_norm = hypot(
                norm(gradient.decision.before_raw),
                norm(gradient.decision.after_raw),
            )
            @test spatial_norm > 1.0f-7
        end

        @info "long-distance mixed-difference gate" cue_kind initial_mixed gain_norm finite_difference
    end
end
