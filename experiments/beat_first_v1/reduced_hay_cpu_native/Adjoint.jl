module ExactOracle

using ..ActiveApicalCell
using ..StateCodec
using ..ReducedHayCPUNativeEventGraph
using ..Payload
using ..SensoryEncoder
using ..OutputCellBank
using ..Topology
using ..ReducedHayCPUNativeModel
using ..CanonicalOptimizer

export Gradient,
    ConditionalAdjointScratch,
    clear_gradient!,
    conditional_exact_vjp!

const Cell = ActiveApicalCell
const Codec = StateCodec
const Graph = ReducedHayCPUNativeEventGraph
const EventPayload = Payload
const Encoder = SensoryEncoder
const OutputBank = OutputCellBank
const NetworkTopology = Topology
const Model = ReducedHayCPUNativeModel
const Gradient = CanonicalOptimizer.ParameterGradient
const clear_gradient! = CanonicalOptimizer.clear_gradient!

"""
Worker-local storage for one exact reverse pass.

The forward is replayed through `Model.forward_candidate!`; no second forward
equation exists here.  The high-dimensional output-cell trajectory is reversed
first, then its anchor/recurrent cotangents continue through the sparse event
graph and the sensory root.
"""
struct ConditionalAdjointScratch{
    V3<:AbstractVector{Float32},
    V4<:AbstractVector{Float32},
}
    buffers::Model.ForwardBuffers{Float32}
    forward::Model.ForwardScratch{Float32}
    output::OutputBank.OutputScratch
    anchor_physical::Vector{V3}
    anchor_physical_bar::Vector{V3}
    recurrent_physical::Vector{V4}
    recurrent_encoded::Vector{V4}
    recurrent_encoded_bar::Vector{V4}
    recurrent_physical_bar::Vector{V4}
    recurrent_input::Vector{V4}
    encoded_anchor_bar::Array{Float32,3}
    encoded_recurrent_bar::Array{Float32,4}
    physical_anchor_bar::Array{Float32,3}
    physical_recurrent_bar::Array{Float32,4}
    pending_bar::Graph.ConductanceInbox{Float32}
    ring::Graph.DelayedPayloadRing{Float32}
    ring_bar::Graph.DelayedPayloadRing{Float32}
    strength_bar::Vector{Float32}
    sensory_input_bar::Array{Float32,3}
    sensory_gain_bar::Matrix{Float32}
    rail_bar::Vector{Float32}
    state_bar::Vector{Float32}
    payload_bar::Vector{Float32}
    payload_value::Vector{Float32}
    input_bar::Vector{Float32}
    raw_cell_bar::Vector{Float32}
    physical_bar::Vector{Float32}
    initial_state::Vector{Float32}
    current_sources::Vector{Int}
    previous_sources::Vector{Int}
    active_sources::Vector{Int}
end

@inline _recurrent_slot(cell::Int, block::Int, step::Int) =
    cell + Model.CELLS_PER_BLOCK * (
        (block - 1) + Model.BLOCKS * (step - 1)
    )

@inline _anchor_slot(cell::Int, block::Int) =
    cell + Model.CELLS_PER_BLOCK * (block - 1)

function _anchor_views(array::Array{Float32,3})
    first_view = @view array[:, 1, 1]
    views = Vector{typeof(first_view)}(
        undef,
        Model.CELLS_PER_BLOCK * Model.BLOCKS,
    )
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            views[_anchor_slot(cell, block)] = @view array[:, cell, block]
        end
    end
    return views
end

function _recurrent_views(array::Array{Float32,4})
    first_view = @view array[:, 1, 1, 1]
    views = Vector{typeof(first_view)}(
        undef,
        Model.CELLS_PER_BLOCK * Model.BLOCKS * Model.RECURRENT_STEPS,
    )
    @inbounds for step in 1:Model.RECURRENT_STEPS
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                views[_recurrent_slot(cell, block, step)] =
                    @view array[:, cell, block, step]
            end
        end
    end
    return views
end

function ConditionalAdjointScratch()
    buffers = Model.ForwardBuffers(Float32)
    forward = Model.ForwardScratch(Float32)
    encoded_anchor_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
    )
    encoded_recurrent_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
        Model.RECURRENT_STEPS,
    )
    physical_recurrent_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
        Model.RECURRENT_STEPS,
    )
    physical_anchor_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
    )
    return ConditionalAdjointScratch(
        buffers,
        forward,
        OutputBank.OutputScratch(),
        _anchor_views(buffers.physical_anchor),
        _anchor_views(physical_anchor_bar),
        _recurrent_views(buffers.physical_recurrent),
        _recurrent_views(forward.encoded_recurrent),
        _recurrent_views(encoded_recurrent_bar),
        _recurrent_views(physical_recurrent_bar),
        _recurrent_views(forward.recurrent_inputs),
        encoded_anchor_bar,
        encoded_recurrent_bar,
        physical_anchor_bar,
        physical_recurrent_bar,
        Graph.ConductanceInbox(Model.TOTAL_CELLS, Float32),
        Graph.DelayedPayloadRing(Model.TOTAL_CELLS, Float32),
        Graph.DelayedPayloadRing(Model.TOTAL_CELLS, Float32),
        zeros(Float32, Model.FANOUT * Model.TOTAL_CELLS),
        zeros(Float32, Cell.INPUT_DIM, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, 2, Model.INPUT_RAILS),
        zeros(Float32, Model.INPUT_RAILS),
        zeros(Float32, Cell.STATE_DIM),
        zeros(Float32, EventPayload.PAYLOAD_DIM),
        zeros(Float32, EventPayload.PAYLOAD_DIM),
        zeros(Float32, Cell.INPUT_DIM),
        zeros(Float32, Cell.PARAM_DIM),
        zeros(Float32, Cell.STATE_DIM),
        zeros(Float32, Cell.STATE_DIM),
        Vector{Int}(undef, Model.BLOCKS * Model.CELLS_PER_BLOCK),
        Vector{Int}(undef, Model.BLOCKS * Model.CELLS_PER_BLOCK),
        Vector{Int}(undef, 2 * Model.BLOCKS * Model.CELLS_PER_BLOCK),
    )
end

@inline _global_cell(block::Int, cell::Int) =
    (block - 1) * Model.CELLS_PER_BLOCK + cell

@inline function _has_payload_event(
    state::AbstractVector{Float32},
    event_floor::Float32,
)
    return EventPayload.has_payload_event(state, event_floor)
end

@inline function _output_gradient(gradient::Gradient)
    return OutputBank.OutputGradient(
        gradient.output_cell_raw,
        gradient.output_edge_raw,
        gradient.output_q_basal_bias_raw,
        gradient.output_gain,
        gradient.output_bias,
    )
end

@inline function _fill_sources!(
    destination::Vector{Int},
    encoded_recurrent::AbstractVector{<:AbstractVector{Float32}},
    step::Int,
    event_floor::Float32,
)
    count = 0
    step == 0 && return count
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            state = encoded_recurrent[_recurrent_slot(cell, block, step)]
            _has_payload_event(state, event_floor) || continue
            count += 1
            destination[count] = _global_cell(block, cell)
        end
    end
    return count
end

@inline function _merge_sorted_sources!(
    destination::Vector{Int},
    current::Vector{Int},
    current_count::Int,
    previous::Vector{Int},
    previous_count::Int,
)
    current_index = 1
    previous_index = 1
    destination_index = 0
    @inbounds while current_index <= current_count && previous_index <= previous_count
        current_source = current[current_index]
        previous_source = previous[previous_index]
        if current_source < previous_source
            destination_index += 1
            destination[destination_index] = current_source
            current_index += 1
        elseif previous_source < current_source
            destination_index += 1
            destination[destination_index] = previous_source
            previous_index += 1
        else
            destination_index += 1
            destination[destination_index] = current_source
            current_index += 1
            previous_index += 1
        end
    end
    @inbounds while current_index <= current_count
        destination_index += 1
        destination[destination_index] = current[current_index]
        current_index += 1
    end
    @inbounds while previous_index <= previous_count
        destination_index += 1
        destination[destination_index] = previous[previous_index]
        previous_index += 1
    end
    return destination_index
end

@inline function _set_step_payloads!(
    ring::Graph.DelayedPayloadRing{Float32},
    encoded_recurrent::AbstractVector{<:AbstractVector{Float32}},
    step::Int,
    payload_gain::AbstractVector{Float32},
    tap::UInt8,
    event_floor::Float32,
)
    step == 0 && return nothing
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            source = _global_cell(block, cell)
            state = encoded_recurrent[_recurrent_slot(cell, block, step)]
            _has_payload_event(state, event_floor) || continue
            if tap == 0x00
                EventPayload.payload_channels_event_masked_cached_unchecked!(
                    @view(ring.current[:, source]),
                    state,
                    payload_gain,
                    event_floor,
                )
            else
                EventPayload.payload_channels_event_masked_cached_unchecked!(
                    @view(ring.previous[:, source]),
                    state,
                    payload_gain,
                    event_floor,
                )
            end
        end
    end
    return nothing
end

@inline function _accumulate_step_payload_bar!(
    encoded_bar::AbstractVector{<:AbstractVector{Float32}},
    payload_gain_bar::AbstractVector{Float32},
    encoded_recurrent::AbstractVector{<:AbstractVector{Float32}},
    step::Int,
    source_bar::AbstractMatrix{Float32},
    payload_gain::AbstractVector{Float32},
    payload_gain_derivative::AbstractVector{Float32},
    event_floor::Float32,
    scaled_source_bar::AbstractVector{Float32},
    payload_value::AbstractVector{Float32},
)
    step == 0 && return nothing
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            source = _global_cell(block, cell)
            state = encoded_recurrent[_recurrent_slot(cell, block, step)]
            _has_payload_event(state, event_floor) || continue
            EventPayload.payload_channels_event_masked_cached_raw_vjp_unchecked!(
                encoded_bar[_recurrent_slot(cell, block, step)],
                payload_gain_bar,
                state,
                payload_gain,
                payload_gain_derivative,
                @view(source_bar[:, source]),
                event_floor,
                scaled_source_bar,
                payload_value,
            )
        end
    end
    return nothing
end

@inline function _accumulate_anchor_payload_bar!(
    anchor_bar::AbstractArray{Float32,3},
    payload_gain_bar::AbstractVector{Float32},
    encoded_anchor::AbstractArray{Float32,3},
    source_bar::AbstractMatrix{Float32},
    payload_gain::AbstractVector{Float32},
    payload_gain_derivative::AbstractVector{Float32},
)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            source = _global_cell(block, cell)
            EventPayload.payload_channels_cached_raw_vjp_unchecked!(
                @view(anchor_bar[:, cell, block]),
                payload_gain_bar,
                @view(encoded_anchor[:, cell, block]),
                payload_gain,
                payload_gain_derivative,
                @view(source_bar[:, source]),
            )
        end
    end
    return nothing
end

@inline function _set_pending_bar!(
    pending_bar::Graph.ConductanceInbox{Float32},
    source::Int,
    input_bar::AbstractVector{Float32},
)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        pending_bar.ampa[compartment, source] =
            input_bar[Cell.input_index(compartment, Cell.INPUT_AMPA)]
        pending_bar.nmda[compartment, source] =
            input_bar[Cell.input_index(compartment, Cell.INPUT_NMDA)]
        pending_bar.gaba[compartment, source] =
            input_bar[Cell.input_index(compartment, Cell.INPUT_GABA)]
    end
    return nothing
end

@inline function _accumulate_cell_raw!(
    destination::AbstractVector{Float32},
    source::AbstractVector{Float32},
)
    @inbounds for parameter in 1:Cell.PARAM_DIM
        destination[parameter] += source[parameter]
    end
    return nothing
end

@inline function _encoded_to_physical_bar!(
    scratch::ConditionalAdjointScratch,
    physical_state::AbstractVector{Float32},
    encoded_bar::AbstractVector{Float32},
    spike_credit::Bool,
)
    event_bar = encoded_bar[Cell.SPIKE_INDEX]
    encoded_bar[Cell.SPIKE_INDEX] = 0.0f0
    Codec.state_codec_pullback!(scratch.physical_bar, physical_state, encoded_bar)
    return spike_credit ? event_bar : 0.0f0
end

"""
    conditional_exact_vjp!(gradient, scratch, model, prepared, rails,
                           output_cotangent; expected_generation)

Replay the route-free recurrent trajectory and accumulate its composed
continuous/surrogate VJP through the output-cell bank, recurrent graph, and
sensory root.  The same prepared generation is checked before and after the
replay/reverse transaction.
"""
function conditional_exact_vjp!(
    gradient::Gradient,
    scratch::ConditionalAdjointScratch,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    output_cotangent::AbstractVector{Float32};
    expected_generation::UInt64=Model.prepared_generation(prepared),
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
    spike_credit::Bool=true,
)
    length(rails) == Model.INPUT_RAILS || throw(DimensionMismatch(
        "rails must have length $(Model.INPUT_RAILS)",
    ))
    length(output_cotangent) == Model.OUTPUT_DIM || throw(DimensionMismatch(
        "output cotangent must have length $(Model.OUTPUT_DIM)",
    ))

    slot = Model.assert_generation(prepared, expected_generation)
    parameters = slot.parameters
    cache = slot.cache
    Model.forward_candidate!(
        scratch.buffers,
        scratch.forward,
        model,
        prepared,
        rails,
        Model.NoForwardDiagnostics();
        expected_generation,
        event_floor,
        spike_smoothing,
    )

    fill!(scratch.physical_anchor_bar, 0.0f0)
    fill!(scratch.physical_recurrent_bar, 0.0f0)
    fill!(scratch.encoded_anchor_bar, 0.0f0)
    fill!(scratch.encoded_recurrent_bar, 0.0f0)
    Graph.clear_inbox!(scratch.pending_bar)
    Graph.clear_payload_ring!(scratch.ring)
    Graph.clear_payload_ring!(scratch.ring_bar)
    fill!(scratch.strength_bar, 0.0f0)
    fill!(scratch.sensory_input_bar, 0.0f0)

    OutputBank.output_pullback!(
        scratch.encoded_anchor_bar,
        scratch.encoded_recurrent_bar,
        _output_gradient(gradient),
        gradient.payload_gain_raw,
        scratch.buffers.output_trajectory,
        scratch.output,
        model.output_topology,
        Model._output_parameters(parameters),
        cache.output,
        cache.payload_gain,
        cache.payload_gain_derivative,
        scratch.forward.encoded_anchor,
        scratch.forward.encoded_recurrent,
        output_cotangent;
        event_floor,
        spike_smoothing,
    )

    @inbounds for step in Model.RECURRENT_STEPS:-1:1
        if step < Model.RECURRENT_STEPS
            Graph.clear_payload_ring!(scratch.ring)
            Graph.clear_payload_ring!(scratch.ring_bar)
            _set_step_payloads!(
                scratch.ring,
                scratch.recurrent_encoded,
                step,
                cache.payload_gain,
                0x00,
                event_floor,
            )
            _set_step_payloads!(
                scratch.ring,
                scratch.recurrent_encoded,
                step - 1,
                cache.payload_gain,
                0x01,
                event_floor,
            )
            current_count = _fill_sources!(
                scratch.current_sources,
                scratch.recurrent_encoded,
                step,
                event_floor,
            )
            previous_count = _fill_sources!(
                scratch.previous_sources,
                scratch.recurrent_encoded,
                step - 1,
                event_floor,
            )
            active_count = _merge_sorted_sources!(
                scratch.active_sources,
                scratch.current_sources,
                current_count,
                scratch.previous_sources,
                previous_count,
            )
            Graph.deliver_payloads_vjp!(
                scratch.strength_bar,
                scratch.ring_bar,
                model.graph,
                cache.edge_strength,
                scratch.ring,
                @view(scratch.active_sources[1:active_count]),
                scratch.pending_bar,
            )
            _accumulate_step_payload_bar!(
                scratch.recurrent_encoded_bar,
                gradient.payload_gain_raw,
                scratch.recurrent_encoded,
                step,
                scratch.ring_bar.current,
                cache.payload_gain,
                cache.payload_gain_derivative,
                event_floor,
                scratch.payload_bar,
                scratch.payload_value,
            )
            _accumulate_step_payload_bar!(
                scratch.recurrent_encoded_bar,
                gradient.payload_gain_raw,
                scratch.recurrent_encoded,
                step - 1,
                scratch.ring_bar.previous,
                cache.payload_gain,
                cache.payload_gain_derivative,
                event_floor,
                scratch.payload_bar,
                scratch.payload_value,
            )
        end

        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                recurrent_slot = _recurrent_slot(cell, block, step)
                physical_next = scratch.recurrent_physical[recurrent_slot]
                encoded_bar = scratch.recurrent_encoded_bar[recurrent_slot]
                event_bar = _encoded_to_physical_bar!(
                    scratch,
                    physical_next,
                    encoded_bar,
                    spike_credit,
                )
                recurrent_physical_bar = scratch.recurrent_physical_bar[recurrent_slot]
                for state in 1:Model.STATE_DIM
                    scratch.physical_bar[state] += recurrent_physical_bar[state]
                end

                if step == 1
                    anchor_slot = _anchor_slot(cell, block)
                    Cell.cell_step_pullback!(
                        scratch.state_bar,
                        scratch.input_bar,
                        scratch.raw_cell_bar,
                        scratch.anchor_physical[anchor_slot],
                        scratch.recurrent_input[recurrent_slot],
                        cache.cell[cell, block],
                        cache.cell_derivative[cell, block],
                        physical_next,
                        scratch.physical_bar,
                        event_bar,
                        spike_smoothing,
                    )
                    predecessor_bar = scratch.anchor_physical_bar[anchor_slot]
                    for state in 1:Model.STATE_DIM
                        predecessor_bar[state] += scratch.state_bar[state]
                    end
                else
                    previous_slot = _recurrent_slot(cell, block, step - 1)
                    Cell.cell_step_pullback!(
                        scratch.state_bar,
                        scratch.input_bar,
                        scratch.raw_cell_bar,
                        scratch.recurrent_physical[previous_slot],
                        scratch.recurrent_input[recurrent_slot],
                        cache.cell[cell, block],
                        cache.cell_derivative[cell, block],
                        physical_next,
                        scratch.physical_bar,
                        event_bar,
                        spike_smoothing,
                    )
                    predecessor_bar =
                        scratch.recurrent_physical_bar[previous_slot]
                    for state in 1:Model.STATE_DIM
                        predecessor_bar[state] += scratch.state_bar[state]
                    end
                end
                _accumulate_cell_raw!(
                    @view(gradient.cell_raw[:, cell, block]),
                    scratch.raw_cell_bar,
                )
                _set_pending_bar!(
                    scratch.pending_bar,
                    _global_cell(block, cell),
                    scratch.input_bar,
                )
            end
        end
    end

    # Reverse the all-cell anchor precharge.  Both delay taps alias the same
    # anchor payload in forward, so their cotangents are summed here.
    Graph.clear_payload_ring!(scratch.ring)
    Graph.clear_payload_ring!(scratch.ring_bar)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            source = _global_cell(block, cell)
            EventPayload.payload_channels_cached_unchecked!(
                @view(scratch.ring.current[:, source]),
                @view(scratch.forward.encoded_anchor[:, cell, block]),
                cache.payload_gain,
            )
            copyto!(
                @view(scratch.ring.previous[:, source]),
                @view(scratch.ring.current[:, source]),
            )
        end
    end
    Graph.deliver_payloads_vjp!(
        scratch.strength_bar,
        scratch.ring_bar,
        model.graph,
        cache.edge_strength,
        scratch.ring,
        1:Model.TOTAL_CELLS,
        scratch.pending_bar,
    )
    @inbounds for source in 1:Model.TOTAL_CELLS
        for channel in axes(scratch.ring_bar.current, 1)
            scratch.ring_bar.current[channel, source] +=
                scratch.ring_bar.previous[channel, source]
        end
    end
    _accumulate_anchor_payload_bar!(
        scratch.encoded_anchor_bar,
        gradient.payload_gain_raw,
        scratch.forward.encoded_anchor,
        scratch.ring_bar.current,
        cache.payload_gain,
        cache.payload_gain_derivative,
    )

    # Anchor cell transition and the shared sensory encoder are the trajectory
    # root.  The canonical resting state derivative is accumulated once per cell.
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            anchor_slot = _anchor_slot(cell, block)
            physical_anchor = scratch.anchor_physical[anchor_slot]
            encoded_bar = @view scratch.encoded_anchor_bar[:, cell, block]
            event_bar = _encoded_to_physical_bar!(
                scratch,
                physical_anchor,
                encoded_bar,
                spike_credit,
            )
            anchor_physical_bar = scratch.anchor_physical_bar[anchor_slot]
            for state in 1:Model.STATE_DIM
                scratch.physical_bar[state] += anchor_physical_bar[state]
            end
            Cell.initial_state!(scratch.initial_state, cache.cell[cell, block])
            Cell.cell_step_pullback!(
                scratch.state_bar,
                scratch.input_bar,
                scratch.raw_cell_bar,
                scratch.initial_state,
                @view(scratch.forward.sensory_input[:, cell, block]),
                cache.cell[cell, block],
                cache.cell_derivative[cell, block],
                physical_anchor,
                scratch.physical_bar,
                event_bar,
            )
            Cell.initial_state_pullback!(
                scratch.raw_cell_bar,
                scratch.state_bar,
                cache.cell_derivative[cell, block],
            )
            _accumulate_cell_raw!(
                @view(gradient.cell_raw[:, cell, block]),
                scratch.raw_cell_bar,
            )
            copyto!(
                @view(scratch.sensory_input_bar[:, cell, block]),
                scratch.input_bar,
            )
        end
    end

    Encoder.sensory_cached_raw_vjp!(
        scratch.rail_bar,
        scratch.sensory_gain_bar,
        scratch.sensory_input_bar,
        rails,
        cache.sensory_gain,
        cache.sensory_gain_derivative,
    )
    @inbounds for index in eachindex(gradient.sensory_gain_raw)
        gradient.sensory_gain_raw[index] += scratch.sensory_gain_bar[index]
    end
    NetworkTopology.edge_strength_cached_raw_vjp!(
        gradient.edge_strength_raw,
        cache.edge_strength_derivative,
        scratch.strength_bar,
    )

    Model.assert_generation(prepared, expected_generation)
    return scratch.buffers
end

end # module ExactOracle
