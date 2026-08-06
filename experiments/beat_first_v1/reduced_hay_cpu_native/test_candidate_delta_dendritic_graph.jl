using Test
using Random

module CandidateDeltaDendriticGraphTestHarness
for file in (
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "CompactDendriticNode.jl",
    "DendriticDeltaForestTopology.jl",
    "DendriticDeltaForest.jl",
    "DendriticForestOutput.jl",
    "CandidateDeltaDendriticGraph.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = CandidateDeltaDendriticGraphTestHarness
const Delta = H.CandidateDeltaInput
const Bank = H.DendriticProgramBank
const Topology = H.DendriticDeltaForestTopology
const Model = H.CandidateDeltaDendriticGraph

function fixture_state!(state)
    fill!(state.common.board, 0x00)
    state.common.board[20, 3] = 0x01
    state.common.board[21, 7] = 0x01
    state.common.board[23, 4] = 0x01
    state.common.board[24, 2] = 0x01
    fill!(state.common.queue, 0x00)
    @inbounds for role in 1:Delta.QUEUE_TOKENS
        state.common.queue[mod1(2role + 1, Delta.QUEUE_PIECES), role] = 0x01
    end
    state.common.ren[1] = 2.0f0
    state.common.back_to_back[1] = 1.0f0
    return state
end

function fixture_placement()
    placement = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement[22, 5] = 0x01
    placement[22, 6] = 0x01
    placement[23, 5] = 0x01
    placement[23, 6] = 0x01
    return placement
end

function set_all_affected!(affected)
    @inbounds for position in 1:Topology.LEAF_COUNT
        affected.positions[position] = UInt16(position)
        affected.marked[position] = 0x01
    end
    affected.count = Topology.LEAF_COUNT
    return affected
end

function program_gradient_map(gradient)
    result = Dict{Int,Vector{Float32}}()
    @inbounds for slot in 1:Bank.active_gradient_count(gradient)
        row = Int(Bank.active_gradient_row(gradient, slot))
        result[row] = copy(@view gradient.values[:, slot])
    end
    return result
end

function run_path!(raw, gradient, state, worker, parameters, cache, placement,
                   direction; full_candidate=false)
    Model.prepare_state!(state, worker, parameters, cache)
    Model.prepare_candidate!(worker, state, parameters, cache, placement, 0.0f0)
    full_candidate && set_all_affected!(worker.affected)
    Model._forward_prepared_candidate!(raw, worker, state, parameters, cache)
    Model.clear_gradient!(gradient)
    Model.pullback_candidate!(
        gradient, raw, direction, worker, state, parameters, cache,
    )
    Model.finish_state_pullback!(gradient, worker, state, parameters, cache)
    return raw
end

@testset "canonical DDF model contract and COW forward" begin
    parameters = Model.initialize_model()
    cache = Model.ModelCache(parameters)
    placement = fixture_placement()

    sparse_state = fixture_state!(Model.ModelState())
    sparse_worker = Model.ModelWorker()
    Model.prepare_state!(sparse_state, sparse_worker, parameters, cache)
    sparse_raw = zeros(Float32, 22)
    Model.forward_candidate!(
        sparse_raw, sparse_worker, sparse_state, parameters, cache,
        placement, 0.0f0,
    )
    sparse_affected = length(sparse_worker.affected)

    full_state = fixture_state!(Model.ModelState())
    full_worker = Model.ModelWorker()
    Model.prepare_state!(full_state, full_worker, parameters, cache)
    Model.prepare_candidate!(
        full_worker, full_state, parameters, cache, placement, 0.0f0,
    )
    set_all_affected!(full_worker.affected)
    full_raw = zeros(Float32, 22)
    Model._forward_prepared_candidate!(
        full_raw, full_worker, full_state, parameters, cache,
    )

    @test sparse_raw ≈ full_raw rtol=3.0f-5 atol=3.0f-5
    @test sparse_affected == 16
    @test sparse_affected < Topology.LEAF_COUNT ÷ 10
    @test Model.stored_parameter_count(parameters) == 3_349_771
    @test count(!iszero, sparse_worker.delta.placement) == 4
    stats = Model.forward_stats(sparse_state, sparse_worker)
    @test stats.candidate_after.dirty_leaves == sparse_affected
    @test stats.candidate_after.dirty_ancestors > 0
    @test stats.candidate_after.compact_messages > 0
    @test stats.output_event_denominator == 22 * 3

    Model.forward_candidate!(
        sparse_raw, sparse_worker, sparse_state, parameters, cache,
        placement, 0.0f0,
    )
    @test @allocated(Model.forward_candidate!(
        sparse_raw, sparse_worker, sparse_state, parameters, cache,
        placement, 0.0f0,
    )) == 0

    source = read(joinpath(@__DIR__, "CandidateDeltaDendriticGraph.jl"), String)
    for retired in (
        "SharedDendriticFactor", "SpatialDendriticFactors",
        "TypedSparseAfferents", "ContextAfferents",
        "ContinuousDendriticReadout", "DendriticDecisionGraph",
    )
        @test !occursin("using .." * retired, source)
    end
end

@testset "COW plus grouped base reverse equals full forest reverse" begin
    rng = MersenneTwister(0x51a7d3a7)
    parameters = Model.initialize_model()
    cache = Model.ModelCache(parameters)
    placement = fixture_placement()
    direction = randn(rng, Float32, 22)

    sparse_state = fixture_state!(Model.ModelState())
    sparse_worker = Model.ModelWorker()
    sparse_gradient = Model.ModelGradient(
        parameters; active_program_capacity=4_096,
    )
    sparse_raw = zeros(Float32, 22)
    run_path!(
        sparse_raw, sparse_gradient, sparse_state, sparse_worker,
        parameters, cache, placement, direction,
    )

    full_state = fixture_state!(Model.ModelState())
    full_worker = Model.ModelWorker()
    full_gradient = Model.ModelGradient(
        parameters; active_program_capacity=4_096,
    )
    full_raw = zeros(Float32, 22)
    run_path!(
        full_raw, full_gradient, full_state, full_worker,
        parameters, cache, placement, direction; full_candidate=true,
    )

    @test sparse_raw ≈ full_raw rtol=3.0f-5 atol=3.0f-5
    @test sparse_gradient.leaf_shared_raw ≈
          full_gradient.leaf_shared_raw rtol=8.0f-5 atol=4.0f-5
    @test sparse_gradient.forest.internal_raw ≈
          full_gradient.forest.internal_raw rtol=8.0f-5 atol=4.0f-5
    @test sparse_gradient.forest.child_contact ≈
          full_gradient.forest.child_contact rtol=8.0f-5 atol=4.0f-5
    for field in (
        :cell_raw, :anchor_weight, :context_weight, :placement_weight,
        :cascade_weight, :gain, :bias,
    )
        @test getfield(sparse_gradient.output, field) ≈
              getfield(full_gradient.output, field) rtol=8.0f-5 atol=4.0f-5
    end
    sparse_program = program_gradient_map(sparse_gradient.program)
    full_program = program_gradient_map(full_gradient.program)
    @test keys(sparse_program) == keys(full_program)
    for row in keys(sparse_program)
        @test sparse_program[row] ≈ full_program[row] rtol=1.0f-4 atol=5.0f-5
    end
end
