using Test
using Random
using LinearAlgebra

module DendriticDeltaForestTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "CompactDendriticNode.jl"))
include(joinpath(@__DIR__, "DendriticDeltaForestTopology.jl"))
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "DendriticDeltaForest.jl"))
end

const H = DendriticDeltaForestTestHarness
const Cell = H.ActiveApicalCell
const Delta = H.CandidateDeltaInput
const Node = H.CompactDendriticNode
const Topology = H.DendriticDeltaForestTopology
const Bank = H.DendriticProgramBank
const Forest = H.DendriticDeltaForest

function affected_factors(before, after)
    marked = falses(Topology.LEAF_COUNT)
    @inbounds for column in 1:Delta.BOARD_COLUMNS, row in 1:Delta.BOARD_ROWS
        before[row, column] == after[row, column] && continue
        for factor_column in max(1, column - 1):min(Delta.BOARD_COLUMNS, column + 1)
            for factor_row in max(1, row - 1):min(Delta.BOARD_ROWS, row + 1)
                marked[factor_row + (factor_column - 1) * Delta.BOARD_ROWS] = true
            end
        end
    end
    return UInt16[index for index in eachindex(marked) if marked[index]]
end

function make_candidate(common; line_clear=false)
    placement = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    if line_clear
        placement[end, end] = 1
    else
        placement[20, 5] = 1
        placement[21, 5] = 1
        placement[21, 6] = 1
        placement[22, 6] = 1
    end
    delta = Delta.CandidateDelta()
    Delta.prepare_candidate_delta!(delta, common, placement, 0.0f0)
    materialized = Delta.CandidateMaterialization()
    Delta.reconstruct_candidate!(materialized, common, delta)
    return delta, materialized
end

function setup(::Type{T}=Float32) where {T<:AbstractFloat}
    parameters = Forest.initialize_parameters(0x517, T)
    cache = Forest.PlaneCache(parameters)
    bank = Bank.ProgramBank()
    shared_raw = Cell.default_raw_parameters(T)
    shared_cache, shared_derivative = Cell.parameter_caches(shared_raw)
    return parameters, cache, bank, shared_raw, shared_cache, shared_derivative
end

function seed_node_bar(anchor_bar, ::Type{T}=eltype(anchor_bar)) where {T}
    result = zeros(T, Forest.PAYLOAD_DIM, Topology.NODE_COUNT)
    @inbounds for anchor in 1:Topology.ANCHOR_COUNT
        node = Int(Topology.anchor_node(anchor))
        result[:, node] .= anchor_bar[:, anchor]
    end
    return result
end

@testset "full plane equals structural COW for placement and line clear" begin
    parameters, cache, bank, _, shared_cache, _ = setup()
    workspace = Forest.PlaneWorkspace()
    topology = Topology.canonical_topology()

    common = Delta.StateCommon()
    common.board[19, 4] = 1
    common.board[22, 7] = 1
    base = Forest.PlaneState()
    Forest.base_forward!(
        base, workspace, parameters, cache, bank, common.board,
        Forest.AFTER_PLANE, shared_cache, topology,
    )

    for line_clear in (false, true)
        if line_clear
            fill!(common.board, 0x00)
            @views common.board[end, 1:9] .= 0x01
            Forest.base_forward!(
                base, workspace, parameters, cache, bank, common.board,
                Forest.AFTER_PLANE, shared_cache, topology,
            )
        end
        _, materialized = make_candidate(common; line_clear)
        affected = affected_factors(common.board, materialized.after)
        closure = Topology.DirtyClosure()
        cow = Forest.COWPlaneState()
        stats = Forest.candidate_forward!(
            cow, base, workspace, parameters, cache, bank,
            materialized.after, Forest.AFTER_PLANE, shared_cache,
            affected, closure, topology,
        )
        full = Forest.PlaneState()
        Forest.base_forward!(
            full, workspace, parameters, cache, bank, materialized.after,
            Forest.AFTER_PLANE, shared_cache, topology,
        )
        full_anchors = zeros(Float32, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT)
        cow_anchors = similar(full_anchors)
        Forest.copy_anchor_payloads!(full_anchors, full, topology)
        Forest.copy_anchor_payloads!(cow_anchors, cow, base, topology)
        @test cow_anchors == full_anchors
        @test stats.dirty_leaves == length(affected)
        @test stats.dirty_ancestors == Topology.dirty_count(closure) - length(affected)
        @test stats.compact_messages > 0
    end
end

function dense_grouped_reverse_case()
    rng = MersenneTwister(0xC0A)
    parameters, cache, bank, _, shared_cache, shared_derivative = setup()
    topology = Topology.canonical_topology()
    common = Delta.StateCommon()
    @views common.board[end, 1:9] .= 0x01
    common.board[20, 3] = 1

    base = Forest.PlaneState()
    workspace = Forest.PlaneWorkspace()
    Forest.base_forward!(
        base, workspace, parameters, cache, bank, common.board,
        Forest.AFTER_PLANE, shared_cache, topology,
    )
    candidates = (
        make_candidate(common; line_clear=false)[2],
        make_candidate(common; line_clear=true)[2],
    )
    anchor_bars = (
        0.05f0 .* randn(rng, Float32, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT),
        0.05f0 .* randn(rng, Float32, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT),
    )

    naive_gradient = Forest.PlaneGradient(parameters)
    naive_shared = zeros(Float32, Cell.PARAM_DIM)
    naive_bank = zeros(Float32, Bank.PAYLOAD_WIDTH, Bank.ROW_COUNT)
    for candidate_index in eachindex(candidates)
        full = Forest.PlaneState()
        Forest.base_forward!(
            full, workspace, parameters, cache, bank,
            candidates[candidate_index].after, Forest.AFTER_PLANE,
            shared_cache, topology,
        )
        node_bar = seed_node_bar(anchor_bars[candidate_index])
        Forest.base_reverse!(
            naive_gradient, naive_shared, naive_bank, node_bar,
            workspace, full, parameters, cache, bank,
            candidates[candidate_index].after, Forest.AFTER_PLANE,
            shared_cache, shared_derivative, topology,
        )
    end

    grouped_gradient = Forest.PlaneGradient(parameters)
    grouped_shared = zeros(Float32, Cell.PARAM_DIM)
    grouped_bank = zeros(Float32, Bank.PAYLOAD_WIDTH, Bank.ROW_COUNT)
    base_accumulator = zeros(Float32, Forest.PAYLOAD_DIM, Topology.NODE_COUNT)
    cow = Forest.COWPlaneState()
    closure = Topology.DirtyClosure()
    for candidate_index in eachindex(candidates)
        candidate = candidates[candidate_index]
        affected = affected_factors(common.board, candidate.after)
        Forest.candidate_forward!(
            cow, base, workspace, parameters, cache, bank, candidate.after,
            Forest.AFTER_PLANE, shared_cache, affected, closure, topology,
        )
        Forest.candidate_reverse!(
            grouped_gradient, grouped_shared, grouped_bank, base_accumulator,
            workspace, cow, base, parameters, cache, bank, candidate.after,
            Forest.AFTER_PLANE, shared_cache, shared_derivative,
            anchor_bars[candidate_index], closure, topology,
        )
    end
    Forest.base_reverse!(
        grouped_gradient, grouped_shared, grouped_bank, base_accumulator,
        workspace, base, parameters, cache, bank, common.board,
        Forest.AFTER_PLANE, shared_cache, shared_derivative, topology,
    )
    return naive_gradient, naive_shared, naive_bank,
           grouped_gradient, grouped_shared, grouped_bank
end

@testset "candidate reverse plus one grouped base reverse is exact" begin
    naive, naive_shared, naive_bank, grouped, grouped_shared, grouped_bank =
        dense_grouped_reverse_case()
    @test isapprox(grouped.internal_raw, naive.internal_raw; rtol=2f-5, atol=2f-6)
    @test isapprox(grouped.child_contact, naive.child_contact; rtol=2f-5, atol=2f-6)
    @test isapprox(grouped_shared, naive_shared; rtol=2f-5, atol=2f-6)
    @test isapprox(grouped_bank, naive_bank; rtol=2f-5, atol=2f-6)
end

@testset "hard child event is a causal typed parent contact" begin
    parameters, cache, _, _, _, _ = setup()
    topology = Topology.canonical_topology()
    workspace = Forest.PlaneWorkspace()
    source = Forest.PlaneState()
    parent = Int(Topology.row_half_node(1, 1))
    edge = Int(topology.child_offsets[parent])
    child = Int(topology.children[edge])
    parameters.child_contact[Node.HARD_EVENT_INDEX, edge] = 2.0f0
    low = zeros(Float32, Forest.PAYLOAD_DIM)
    high = similar(low)
    Forest._internal_forward!(
        low, workspace, topology, parent, parameters, cache, source,
    )
    source.payload[Node.HARD_EVENT_INDEX, child] = 1.0f0
    Forest._internal_forward!(
        high, workspace, topology, parent, parameters, cache, source,
    )
    @test high[Node.CENTERED_MARGIN_INDEX] != low[Node.CENTERED_MARGIN_INDEX]

    # A root event cotangent must use the explicit surrogate path and must not
    # be smuggled into either analog coordinate.
    gradient = Forest.PlaneGradient(parameters)
    anchor_bar = zeros(Float32, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT)
    anchor_bar[Node.HARD_EVENT_INDEX, 1] = 1.0f0
    node_bar = seed_node_bar(anchor_bar)
    state = Forest.PlaneState()
    bank = Bank.ProgramBank()
    shared_raw = Cell.default_raw_parameters()
    shared_cache, shared_derivative = Cell.parameter_caches(shared_raw)
    board = zeros(UInt8, 24, 10)
    Forest.base_forward!(
        state, workspace, parameters, cache, bank, board,
        Forest.AFTER_PLANE, shared_cache, topology,
    )
    shared_bar = zeros(Float32, Cell.PARAM_DIM)
    bank_bar = Bank.SparseProgramGradient(bank, 2048)
    Forest.base_reverse!(
        gradient, shared_bar, bank_bar, node_bar, workspace, state,
        parameters, cache, bank, board, Forest.AFTER_PLANE,
        shared_cache, shared_derivative, topology,
    )
    @test norm(gradient.internal_raw) > 0.0f0
end

function objective_and_events!(
    state,
    workspace,
    parameters,
    cache,
    bank,
    board,
    shared_cache,
    anchor_bar,
)
    Forest.refresh_cache!(cache, parameters)
    Forest.base_forward!(
        state, workspace, parameters, cache, bank, board,
        Forest.AFTER_PLANE, shared_cache,
    )
    anchors = zeros(eltype(state.payload), Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT)
    Forest.copy_anchor_payloads!(anchors, state)
    return dot(anchors, anchor_bar), copy(@view state.payload[3, :])
end

@testset "representative event-stable plane finite differences" begin
    rng = MersenneTwister(0xFD5)
    parameters, cache, bank, shared_raw, shared_cache, shared_derivative =
        setup(Float64)
    # Conditional/event-stable FD differentiates the analog graph with hard
    # event coordinates held fixed.  The explicit event-surrogate path is
    # tested separately above, so remove that intentionally non-FD path here.
    parameters.child_contact[Node.HARD_EVENT_INDEX, :] .= 0.0
    board = zeros(UInt8, 24, 10)
    board[18:24, 2] .= 1
    board[21:24, 7] .= 1
    workspace = Forest.PlaneWorkspace(Float64)
    state = Forest.PlaneState(Float64)
    anchor_bar = zeros(Float64, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT)
    anchor_bar[1:2, :] .= 0.03 .* randn(rng, 2, Topology.ANCHOR_COUNT)

    Forest.base_forward!(
        state, workspace, parameters, cache, bank, board,
        Forest.AFTER_PLANE, shared_cache,
    )
    node_bar = seed_node_bar(anchor_bar, Float64)
    gradient = Forest.PlaneGradient(parameters)
    shared_bar = zeros(Float64, Cell.PARAM_DIM)
    bank_bar = zeros(Float64, Bank.PAYLOAD_WIDTH, Bank.ROW_COUNT)
    Forest.base_reverse!(
        gradient, shared_bar, bank_bar, node_bar, workspace, state,
        parameters, cache, bank, board, Forest.AFTER_PLANE,
        shared_cache, shared_derivative,
    )

    baseline, events = objective_and_events!(
        state, workspace, parameters, cache, bank, board, shared_cache, anchor_bar,
    )
    @test isfinite(baseline)

    function central_parameter!(array, index, epsilon, refresh_shared=false)
        original = array[index]
        array[index] = original + epsilon
        if refresh_shared
            plus_cache = Cell.transform_parameters(shared_raw)
            plus, plus_events = objective_and_events!(
                state, workspace, parameters, cache, bank, board,
                plus_cache, anchor_bar,
            )
        else
            plus, plus_events = objective_and_events!(
                state, workspace, parameters, cache, bank, board,
                shared_cache, anchor_bar,
            )
        end
        array[index] = original - epsilon
        if refresh_shared
            minus_cache = Cell.transform_parameters(shared_raw)
            minus, minus_events = objective_and_events!(
                state, workspace, parameters, cache, bank, board,
                minus_cache, anchor_bar,
            )
        else
            minus, minus_events = objective_and_events!(
                state, workspace, parameters, cache, bank, board,
                shared_cache, anchor_bar,
            )
        end
        array[index] = original
        @test plus_events == events == minus_events
        return (plus - minus) / (2epsilon)
    end

    contact_index = argmax(abs.(gradient.child_contact))
    numerical_contact = central_parameter!(
        parameters.child_contact, contact_index, 1e-5,
    )
    @test isapprox(
        gradient.child_contact[contact_index], numerical_contact;
        rtol=4e-3, atol=2e-6,
    )

    internal_index = argmax(abs.(gradient.internal_raw))
    numerical_internal = central_parameter!(
        parameters.internal_raw, internal_index, 1e-5,
    )
    @test isapprox(
        gradient.internal_raw[internal_index], numerical_internal;
        rtol=8e-3, atol=3e-6,
    )

    shared_index = argmax(abs.(shared_bar))
    numerical_shared = central_parameter!(shared_raw, shared_index, 1e-5, true)
    @test isapprox(
        shared_bar[shared_index], numerical_shared;
        rtol=8e-3, atol=3e-6,
    )

    program_index = argmax(abs.(bank_bar))
    original_program = bank.payload[program_index]
    epsilon = 2.0f-3
    bank.payload[program_index] = original_program + epsilon
    plus, plus_events = objective_and_events!(
        state, workspace, parameters, cache, bank, board, shared_cache, anchor_bar,
    )
    bank.payload[program_index] = original_program - epsilon
    minus, minus_events = objective_and_events!(
        state, workspace, parameters, cache, bank, board, shared_cache, anchor_bar,
    )
    bank.payload[program_index] = original_program
    @test plus_events == events == minus_events
    numerical_program = (plus - minus) / (2Float64(epsilon))
    @test isapprox(bank_bar[program_index], numerical_program; rtol=2e-2, atol=3e-6)
end

function hot_allocations()
    parameters, cache, bank, _, shared_cache, shared_derivative = setup()
    topology = Topology.canonical_topology()
    board = zeros(UInt8, 24, 10)
    after = copy(board)
    after[21, 5] = 1
    affected = affected_factors(board, after)
    closure = Topology.DirtyClosure()
    workspace = Forest.PlaneWorkspace()
    base = Forest.PlaneState()
    cow = Forest.COWPlaneState()
    gradient = Forest.PlaneGradient(parameters)
    shared_bar = zeros(Float32, Cell.PARAM_DIM)
    bank_bar = Bank.SparseProgramGradient(bank, 4096)
    base_bar = zeros(Float32, Forest.PAYLOAD_DIM, Topology.NODE_COUNT)
    anchor_bar = zeros(Float32, Forest.PAYLOAD_DIM, Topology.ANCHOR_COUNT)
    anchor_bar[1, 1] = 1.0f0

    Forest.base_forward!(
        base, workspace, parameters, cache, bank, board,
        Forest.AFTER_PLANE, shared_cache, topology,
    )
    full_forward = @allocated Forest.base_forward!(
        base, workspace, parameters, cache, bank, board,
        Forest.AFTER_PLANE, shared_cache, topology,
    )
    Forest.candidate_forward!(
        cow, base, workspace, parameters, cache, bank, after,
        Forest.AFTER_PLANE, shared_cache, affected, closure, topology,
    )
    cow_forward = @allocated Forest.candidate_forward!(
        cow, base, workspace, parameters, cache, bank, after,
        Forest.AFTER_PLANE, shared_cache, affected, closure, topology,
    )
    Forest.candidate_reverse!(
        gradient, shared_bar, bank_bar, base_bar, workspace, cow, base,
        parameters, cache, bank, after, Forest.AFTER_PLANE,
        shared_cache, shared_derivative, anchor_bar, closure, topology,
    )
    fill!(base_bar, 0.0f0)
    cow_reverse = @allocated Forest.candidate_reverse!(
        gradient, shared_bar, bank_bar, base_bar, workspace, cow, base,
        parameters, cache, bank, after, Forest.AFTER_PLANE,
        shared_cache, shared_derivative, anchor_bar, closure, topology,
    )
    Forest.base_reverse!(
        gradient, shared_bar, bank_bar, base_bar, workspace, base,
        parameters, cache, bank, board, Forest.AFTER_PLANE,
        shared_cache, shared_derivative, topology,
    )
    fill!(base_bar, 0.0f0)
    base_bar[1, Int(Topology.anchor_node(1))] = 1.0f0
    full_reverse = @allocated Forest.base_reverse!(
        gradient, shared_bar, bank_bar, base_bar, workspace, base,
        parameters, cache, bank, board, Forest.AFTER_PLANE,
        shared_cache, shared_derivative, topology,
    )
    return full_forward, cow_forward, cow_reverse, full_reverse
end

@testset "plane identity, size, and allocation-free hot paths" begin
    before = Forest.initialize_parameters(7)
    after = Forest.initialize_parameters(7)
    @test before.internal_raw !== after.internal_raw
    @test before.child_contact !== after.child_contact
    before.child_contact[1] += 1.0f0
    @test before.child_contact[1] != after.child_contact[1]
    @test Forest.stored_parameter_count(before) ==
          4 * Cell.PARAM_DIM + 3 * Topology.CHILD_EDGE_COUNT
    @test size(Forest.PlaneState().payload) == (3, 362)
    @test hot_allocations() == (0, 0, 0, 0)
end
