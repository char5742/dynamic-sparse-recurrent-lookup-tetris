module DendriticDecisionGraph

using ..CandidateDeltaInput
using ..TypedSparseAfferents
using ..ContextAfferents
using ..ContinuousDendriticReadout

const Spatial = TypedSparseAfferents
const Context = ContextAfferents
const Readout = ContinuousDendriticReadout

export DecisionParameters,
       DecisionCache,
       DecisionGradient,
       DecisionState,
       DecisionWorker,
       clear_gradient!,
       clear_state_reverse!,
       finish_state_pullback!,
       forward_candidate!,
       initialize_parameters,
       prepare_state!,
       pullback_candidate!,
       refresh_cache!

"""
Trainable decision graph with distinct before/after spatial planes.

The two spatial graphs deliberately use different fixed topologies.  Thus the
same local feature at the same board position cannot lose whether it belongs
to the current board or the candidate result.  Only positive edge magnitudes,
context magnitudes, decision-cell dynamics and continuous population gains are
learned; all destination identities remain fixed.
"""
struct DecisionParameters
    before::Spatial.TypedSparseAfferentGraph{Float32}
    after::Spatial.TypedSparseAfferentGraph{Float32}
    context_topology::Context.ContextTopology
    context_raw::Matrix{Float32}
    readout::Readout.ReadoutParameters
end

function initialize_parameters(
    ;
    before_seed::Integer=0x4245464f524501,
    after_seed::Integer=0x414654455201,
    context_seed::Integer=0x434f4e5445585401,
)
    Readout.DECISION_CELLS == Spatial.DECISION_CELL_COUNT || error(
        "readout and typed spatial graph disagree on decision-cell count",
    )
    return DecisionParameters(
        Spatial.build_typed_sparse_afferents(before_seed),
        Spatial.build_typed_sparse_afferents(after_seed),
        Context.build_topology(context_seed),
        Context.default_raw_magnitudes(Float32),
        Readout.initialize_parameters(),
    )
end

mutable struct DecisionCache
    readout::Readout.ReadoutCache
end

DecisionCache(parameters::DecisionParameters) =
    DecisionCache(Readout.ReadoutCache(parameters.readout))

function refresh_cache!(cache::DecisionCache, parameters::DecisionParameters)
    Readout.refresh_cache!(cache.readout, parameters.readout)
    return cache
end

"""Additive gradient for every trainable field in the decision graph."""
struct DecisionGradient
    before_raw::Vector{Float32}
    after_raw::Vector{Float32}
    context_raw::Matrix{Float32}
    readout::Readout.ReadoutGradient
end

function DecisionGradient(
    parameters::DecisionParameters,
)
    return DecisionGradient(
        zeros(Float32, Spatial.edge_count(parameters.before)),
        zeros(Float32, Spatial.edge_count(parameters.after)),
        zeros(Float32, size(parameters.context_raw)),
        Readout.ReadoutGradient(),
    )
end

function clear_gradient!(gradient::DecisionGradient)
    fill!(gradient.before_raw, 0.0f0)
    fill!(gradient.after_raw, 0.0f0)
    fill!(gradient.context_raw, 0.0f0)
    Readout.clear_gradient!(gradient.readout)
    return gradient
end

"""
State-shared forward value and reverse accumulator.

`base_input` is constructed once from the before board, the identical
candidate baseline for the after plane, and queue/REN/B2B.  Candidate-local
line-clear/placement changes and exact auxiliary values are added later.
"""
struct DecisionState
    base_input::Matrix{Float32}
    common_input_bar::Matrix{Float32}
    queue_bar::Matrix{Float32}
    ren_bar::Matrix{Float32}
    back_to_back_bar::Matrix{Float32}
end

DecisionState() = DecisionState(
    zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT),
    zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT),
    zeros(Float32, CandidateDeltaInput.QUEUE_PIECES, CandidateDeltaInput.QUEUE_TOKENS),
    zeros(Float32, 1, 1),
    zeros(Float32, 1, 1),
)

function clear_state_reverse!(state::DecisionState)
    fill!(state.common_input_bar, 0.0f0)
    fill!(state.queue_bar, 0.0f0)
    fill!(state.ren_bar, 0.0f0)
    fill!(state.back_to_back_bar, 0.0f0)
    return state
end

"""Fixed worker storage.  Candidate trajectories are replayed for reverse."""
struct DecisionWorker
    candidate_input::Matrix{Float32}
    drive::Array{Float32,3}
    drive_bar::Array{Float32,3}
    input_bar::Matrix{Float32}
    tape::Readout.ReadoutTape
    readout_scratch::Readout.ReadoutScratch
    candidate_readout_gradient::Readout.ReadoutGradient
end

function DecisionWorker()
    return DecisionWorker(
        zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT),
        zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT, Readout.PHASES),
        zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT, Readout.PHASES),
        zeros(Float32, Spatial.INPUT_COUNT, Spatial.DECISION_CELL_COUNT),
        Readout.ReadoutTape(),
        Readout.ReadoutScratch(),
        Readout.ReadoutGradient(),
    )
end

@inline function _copy_phase_drive!(drive, input)
    @inbounds for phase in 1:Readout.PHASES
        for cell in 1:Spatial.DECISION_CELL_COUNT
            @simd for channel in 1:Spatial.INPUT_COUNT
                drive[channel, cell, phase] = input[channel, cell]
            end
        end
    end
    return drive
end

@inline function _sum_phase_bar!(destination, drive_bar)
    fill!(destination, 0.0f0)
    @inbounds for phase in 1:Readout.PHASES
        for cell in 1:Spatial.DECISION_CELL_COUNT
            @simd for channel in 1:Spatial.INPUT_COUNT
                destination[channel, cell] += drive_bar[channel, cell, phase]
            end
        end
    end
    return destination
end

"""Build the candidate-common part of the decision input exactly once."""
function prepare_state!(
    state::DecisionState,
    parameters::DecisionParameters,
    common::CandidateDeltaInput.StateCommon,
    before_features::AbstractMatrix,
    after_base_features::AbstractMatrix,
)
    fill!(state.base_input, 0.0f0)
    Spatial.deposit_full!(state.base_input, parameters.before, before_features)
    Spatial.deposit_full!(state.base_input, parameters.after, after_base_features)
    Context.deposit_state_common!(
        state.base_input,
        common,
        parameters.context_topology,
        parameters.context_raw,
    )
    clear_state_reverse!(state)
    return state
end

"""
Candidate-local delta forward followed by continuous 22-channel readout.

The same typed conductance drive is integrated for three physical phases.  It
is not a dense layer: each lane was produced by a fixed sparse anatomical
contact, and time is expressed by the Reduced Hay state transition itself.
"""
function forward_candidate!(
    raw_output::AbstractVector{Float32},
    worker::DecisionWorker,
    state::DecisionState,
    parameters::DecisionParameters,
    cache::DecisionCache,
    candidate_features::AbstractMatrix,
    after_base_features::AbstractMatrix,
    affected_positions::AbstractVector{<:Integer},
    candidate_aux,
)
    copyto!(worker.candidate_input, state.base_input)
    Spatial.deposit_affected_delta!(
        worker.candidate_input,
        parameters.after,
        candidate_features,
        after_base_features,
        affected_positions,
    )
    Context.deposit_candidate_aux!(
        worker.candidate_input,
        candidate_aux,
        parameters.context_topology,
        parameters.context_raw,
    )
    _copy_phase_drive!(worker.drive, worker.candidate_input)
    return Readout.readout_forward!(
        raw_output,
        worker.tape,
        worker.drive,
        parameters.readout,
        cache.readout,
    )
end

@inline function _accumulate_readout_gradient!(destination, source)
    @simd for parameter in eachindex(destination.shared_cell_raw)
        destination.shared_cell_raw[parameter] += source.shared_cell_raw[parameter]
    end
    @simd for cell in eachindex(destination.gain)
        destination.gain[cell] += source.gain[cell]
    end
    @simd for channel in eachindex(destination.bias)
        destination.bias[channel] += source.bias[channel]
    end
    return destination
end

"""
Replay one candidate and accumulate its exact reverse contribution.

Every candidate contributes the same `input_bar` to the common DAG node.
Candidate auxiliary and affected-after paths are reversed immediately, while
the common node is traversed exactly once by [`finish_state_pullback!`](@ref).
"""
function pullback_candidate!(
    candidate_feature_bar::AbstractMatrix,
    after_base_feature_bar::AbstractMatrix,
    aux_bar::AbstractMatrix,
    gradient::DecisionGradient,
    raw_output::AbstractVector{Float32},
    raw_bar::AbstractVector{Float32},
    worker::DecisionWorker,
    state::DecisionState,
    parameters::DecisionParameters,
    cache::DecisionCache,
    candidate_features::AbstractMatrix,
    after_base_features::AbstractMatrix,
    affected_positions::AbstractVector{<:Integer},
    candidate_aux,
)
    forward_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
        candidate_features,
        after_base_features,
        affected_positions,
        candidate_aux,
    )
    Readout.readout_pullback!(
        worker.drive_bar,
        worker.candidate_readout_gradient,
        worker.tape,
        worker.readout_scratch,
        worker.drive,
        parameters.readout,
        cache.readout,
        raw_bar,
    )
    _accumulate_readout_gradient!(
        gradient.readout,
        worker.candidate_readout_gradient,
    )
    _sum_phase_bar!(worker.input_bar, worker.drive_bar)
    @inbounds for index in eachindex(state.common_input_bar)
        state.common_input_bar[index] += worker.input_bar[index]
    end
    Context.candidate_aux_pullback!(
        aux_bar,
        gradient.context_raw,
        worker.input_bar,
        candidate_aux,
        parameters.context_topology,
        parameters.context_raw,
    )
    Spatial.deposit_affected_delta_pullback!(
        candidate_feature_bar,
        after_base_feature_bar,
        gradient.after_raw,
        parameters.after,
        candidate_features,
        after_base_features,
        affected_positions,
        worker.input_bar,
    )
    return raw_output
end

"""Traverse the state-common reverse node once after all candidates."""
function finish_state_pullback!(
    before_feature_bar::AbstractMatrix,
    after_base_feature_bar::AbstractMatrix,
    gradient::DecisionGradient,
    state::DecisionState,
    parameters::DecisionParameters,
    common::CandidateDeltaInput.StateCommon,
    before_features::AbstractMatrix,
    after_base_features::AbstractMatrix,
)
    Spatial.deposit_full_pullback!(
        before_feature_bar,
        gradient.before_raw,
        parameters.before,
        before_features,
        state.common_input_bar,
    )
    Spatial.deposit_full_pullback!(
        after_base_feature_bar,
        gradient.after_raw,
        parameters.after,
        after_base_features,
        state.common_input_bar,
    )
    Context.state_common_pullback!(
        state.queue_bar,
        state.ren_bar,
        state.back_to_back_bar,
        gradient.context_raw,
        state.common_input_bar,
        common,
        parameters.context_topology,
        parameters.context_raw,
    )
    return before_feature_bar, after_base_feature_bar, gradient
end

end # module DendriticDecisionGraph
