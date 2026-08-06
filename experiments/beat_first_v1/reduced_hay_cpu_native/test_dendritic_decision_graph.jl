using Test
using Random

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "TypedSparseAfferents.jl"))
include(joinpath(@__DIR__, "ContextAfferents.jl"))
include(joinpath(@__DIR__, "ContinuousDendriticReadout.jl"))
include(joinpath(@__DIR__, "DendriticDecisionGraph.jl"))

using .CandidateDeltaInput
using .TypedSparseAfferents
using .ContextAfferents
using .ContinuousDendriticReadout
using .DendriticDecisionGraph

function synthetic_common()
    common = StateCommon()
    @inbounds for role in 1:QUEUE_TOKENS
        common.queue[mod1(role + 2, QUEUE_PIECES), role] = 0x01
    end
    common.ren[1] = 2.0f0
    common.back_to_back[1] = 1.0f0
    return common
end

function naive_forward!(raw, input, drive, tape, base, candidate, aux,
                        common, parameters, cache)
    fill!(input, 0.0f0)
    deposit_full!(input, parameters.before, base)
    deposit_full!(input, parameters.after, candidate)
    deposit_state_common!(
        input,
        common,
        parameters.context_topology,
        parameters.context_raw,
    )
    deposit_candidate_aux!(
        input,
        aux,
        parameters.context_topology,
        parameters.context_raw,
    )
    @inbounds for phase in 1:PHASES, cell in 1:DECISION_CELL_COUNT,
                  channel in 1:INPUT_COUNT
        drive[channel, cell, phase] = input[channel, cell]
    end
    readout_forward!(
        raw,
        tape,
        drive,
        parameters.readout,
        cache.readout,
    )
    return raw
end

function objective!(raw, worker, state, parameters, cache, candidate, base,
                    affected, aux, common, direction)
    prepare_state!(state, parameters, common, base, base)
    forward_candidate!(
        raw,
        worker,
        state,
        parameters,
        cache,
        candidate,
        base,
        affected,
        aux,
    )
    return sum(raw .* direction)
end

@testset "candidate-common decision graph forward" begin
    rng = MersenneTwister(0xd3c1510)
    parameters = DendriticDecisionGraph.initialize_parameters()
    cache = DecisionCache(parameters)
    state = DecisionState()
    worker = DecisionWorker()
    common = synthetic_common()
    base = 0.25f0 .* randn(rng, Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[2, 39, 111, 200, 239]
    @inbounds for position in affected
        @views candidate[:, position] .+=
            0.15f0 .* randn(rng, Float32, FEATURE_COUNT)
    end
    aux = rand(rng, Float32, AUXILIARY_FEATURES, 1)
    raw = zeros(Float32, OUTPUT_CHANNELS)
    prepare_state!(state, parameters, common, base, base)
    forward_candidate!(
        raw,
        worker,
        state,
        parameters,
        cache,
        candidate,
        base,
        affected,
        aux,
    )

    naive_raw = similar(raw)
    naive_input = zeros(Float32, INPUT_COUNT, DECISION_CELL_COUNT)
    naive_drive = zeros(Float32, INPUT_COUNT, DECISION_CELL_COUNT, PHASES)
    naive_tape = ReadoutTape()
    naive_forward!(
        naive_raw,
        naive_input,
        naive_drive,
        naive_tape,
        base,
        candidate,
        aux,
        common,
        parameters,
        cache,
    )
    @test raw ≈ naive_raw rtol=2.0f-5 atol=2.0f-5
    @test worker.candidate_input ≈ naive_input rtol=2.0f-5 atol=2.0f-5
    @test minimum(worker.candidate_input) >= -2.0f-5
    @test parameters.before.destination_cell != parameters.after.destination_cell
    @test parameters.before.destination_compartment !=
          parameters.after.destination_compartment
end

@testset "shared-DAG analytic reverse" begin
    rng = MersenneTwister(0xad50117)
    parameters = DendriticDecisionGraph.initialize_parameters()
    cache = DecisionCache(parameters)
    state = DecisionState()
    worker = DecisionWorker()
    gradient = DecisionGradient(parameters)
    common = synthetic_common()
    base = 0.15f0 .* randn(rng, Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[7, 88, 177, 224]
    @inbounds for position in affected
        @views candidate[:, position] .+=
            0.08f0 .* randn(rng, Float32, FEATURE_COUNT)
    end
    aux = 0.1f0 .+ 0.8f0 .* rand(rng, Float32, AUXILIARY_FEATURES, 1)
    direction = randn(rng, Float32, OUTPUT_CHANNELS)
    raw = zeros(Float32, OUTPUT_CHANNELS)
    raw_bar = copy(direction)
    candidate_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    before_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    after_base_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    aux_bar = zeros(Float32, AUXILIARY_FEATURES, 1)

    prepare_state!(state, parameters, common, base, base)
    DendriticDecisionGraph.clear_gradient!(gradient)
    pullback_candidate!(
        candidate_bar,
        after_base_bar,
        aux_bar,
        gradient,
        raw,
        raw_bar,
        worker,
        state,
        parameters,
        cache,
        candidate,
        base,
        affected,
        aux,
    )
    finish_state_pullback!(
        before_bar,
        after_base_bar,
        gradient,
        state,
        parameters,
        common,
        base,
        base,
    )
    base_bar = before_bar + after_base_bar
    common_reverse_nonzero = any(!iszero, state.common_input_bar)
    queue_reverse_nonzero = any(!iszero, state.queue_bar)

    # The integrated objective is accumulated in Float32 over 50 cells.  A
    # 1e-2 central step keeps the upstream sparse-edge signal safely above
    # Float32 cancellation while remaining local relative to raw magnitudes.
    epsilon = 1.0f-2
    function finite_difference!(array, index; refresh=false)
        original = array[index]
        array[index] = original + epsilon
        refresh && DendriticDecisionGraph.refresh_cache!(cache, parameters)
        plus = objective!(
            raw, worker, state, parameters, cache, candidate, base,
            affected, aux, common, direction,
        )
        array[index] = original - epsilon
        refresh && DendriticDecisionGraph.refresh_cache!(cache, parameters)
        minus = objective!(
            raw, worker, state, parameters, cache, candidate, base,
            affected, aux, common, direction,
        )
        array[index] = original
        refresh && DendriticDecisionGraph.refresh_cache!(cache, parameters)
        return (plus - minus) / (2.0f0 * epsilon)
    end

    before_slot = argmax(abs.(gradient.before_raw))
    after_slot = argmax(abs.(gradient.after_raw))
    context_slot = argmax(abs.(gradient.context_raw))
    gain_slot = argmax(abs.(gradient.readout.gain))
    shared_slot = argmax(abs.(gradient.readout.shared_cell_raw))
    base_slot = argmax(abs.(base_bar))
    candidate_indices = CartesianIndices(candidate_bar[:, affected])
    local_slot = argmax(abs.(candidate_bar[:, affected]))
    local_index = candidate_indices[local_slot]
    candidate_slot = CartesianIndex(local_index[1], affected[local_index[2]])
    aux_slot = argmax(abs.(aux_bar))

    @test gradient.before_raw[before_slot] ≈
          finite_difference!(parameters.before.raw_magnitude, before_slot) rtol=1.5f-2 atol=2.0f-3
    @test gradient.after_raw[after_slot] ≈
          finite_difference!(parameters.after.raw_magnitude, after_slot) rtol=1.5f-2 atol=2.0f-3
    @test gradient.context_raw[context_slot] ≈
          finite_difference!(parameters.context_raw, context_slot) rtol=1.5f-2 atol=2.0f-3
    @test gradient.readout.gain[gain_slot] ≈
          finite_difference!(parameters.readout.gain, gain_slot) rtol=1.5f-2 atol=2.0f-3
    @test gradient.readout.shared_cell_raw[shared_slot] ≈
          finite_difference!(parameters.readout.shared_cell_raw, shared_slot; refresh=true) rtol=2.5f-2 atol=3.0f-3
    @test base_bar[base_slot] ≈
          finite_difference!(base, base_slot) rtol=2.0f-2 atol=3.0f-3
    @test candidate_bar[candidate_slot] ≈
          finite_difference!(candidate, candidate_slot) rtol=2.0f-2 atol=3.0f-3
    @test aux_bar[aux_slot] ≈
          finite_difference!(aux, aux_slot) rtol=2.0f-2 atol=3.0f-3
    @test common_reverse_nonzero
    @test queue_reverse_nonzero
end

@testset "decision hot path uses caller-owned storage" begin
    rng = MersenneTwister(0xa110ca7e)
    parameters = DendriticDecisionGraph.initialize_parameters()
    cache = DecisionCache(parameters)
    state = DecisionState()
    worker = DecisionWorker()
    common = synthetic_common()
    base = 0.1f0 .* randn(rng, Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[12, 144, 233]
    @inbounds for position in affected
        @views candidate[:, position] .+= 0.05f0
    end
    aux = rand(rng, Float32, AUXILIARY_FEATURES, 1)
    raw = zeros(Float32, OUTPUT_CHANNELS)
    prepare_state!(state, parameters, common, base, base)
    forward_candidate!(raw, worker, state, parameters, cache, candidate, base,
                       affected, aux)
    @test @allocated(
        forward_candidate!(raw, worker, state, parameters, cache, candidate,
                           base, affected, aux)
    ) == 0
end
