module LearningPrimitives

using ..ReducedHayCPUNativeEventGraph
using ..Payload
using ..OutputCellBank
using ..ReducedHayCPUNativeModel
using ..CanonicalOptimizer

export output_gradient,
    q_output_gradient,
    fill_sources!,
    merge_sorted_sources!,
    set_step_payloads!,
    accumulate_step_payload_bar!,
    accumulate_anchor_payload_bar!

const Graph = ReducedHayCPUNativeEventGraph
const EventPayload = Payload
const OutputBank = OutputCellBank
const Model = ReducedHayCPUNativeModel
const Optimizer = CanonicalOptimizer

@inline recurrent_slot(cell::Int, block::Int, step::Int) =
    cell + Model.CELLS_PER_BLOCK * (
        (block - 1) + Model.BLOCKS * (step - 1)
    )

@inline global_cell(block::Int, cell::Int) =
    (block - 1) * Model.CELLS_PER_BLOCK + cell

@inline function output_gradient(gradient::Optimizer.ParameterGradient)
    return OutputBank.OutputGradient(
        gradient.output_cell_raw,
        gradient.output_edge_raw,
        gradient.output_q_basal_bias_raw,
        gradient.output_gain,
        gradient.output_bias,
    )
end


@inline function q_output_gradient(gradient::Optimizer.ParameterGradient)
    return OutputBank.OutputGradient(
        gradient.output_cell_raw,
        gradient.output_q_edge_raw,
        gradient.output_q_basal_bias_raw,
        gradient.output_gain,
        gradient.output_bias,
    )
end

@inline function fill_sources!(
    destination::Vector{Int},
    encoded_recurrent::AbstractVector{<:AbstractVector{Float32}},
    step::Int,
    event_floor::Float32,
)
    count = 0
    step == 0 && return count
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            state = encoded_recurrent[recurrent_slot(cell, block, step)]
            EventPayload.has_payload_event(state, event_floor) || continue
            count += 1
            destination[count] = global_cell(block, cell)
        end
    end
    return count
end

@inline function merge_sorted_sources!(
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
        destination_index += 1
        if current_source < previous_source
            destination[destination_index] = current_source
            current_index += 1
        elseif previous_source < current_source
            destination[destination_index] = previous_source
            previous_index += 1
        else
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

@inline function set_step_payloads!(
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
            source = global_cell(block, cell)
            state = encoded_recurrent[recurrent_slot(cell, block, step)]
            EventPayload.has_payload_event(state, event_floor) || continue
            if tap == 0x00
                EventPayload.payload_channels_event_masked_cached_unchecked!(
                    @view(ring.current[:, source]), state, payload_gain, event_floor,
                )
            else
                EventPayload.payload_channels_event_masked_cached_unchecked!(
                    @view(ring.previous[:, source]), state, payload_gain, event_floor,
                )
            end
        end
    end
    return nothing
end

@inline function accumulate_step_payload_bar!(
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
            source = global_cell(block, cell)
            state = encoded_recurrent[recurrent_slot(cell, block, step)]
            EventPayload.has_payload_event(state, event_floor) || continue
            EventPayload.payload_channels_event_masked_cached_raw_vjp_unchecked!(
                encoded_bar[recurrent_slot(cell, block, step)],
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

@inline function accumulate_anchor_payload_bar!(
    anchor_bar::AbstractArray{Float32,3},
    payload_gain_bar::AbstractVector{Float32},
    encoded_anchor::AbstractArray{Float32,3},
    source_bar::AbstractMatrix{Float32},
    payload_gain::AbstractVector{Float32},
    payload_gain_derivative::AbstractVector{Float32},
)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            source = global_cell(block, cell)
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

end # module LearningPrimitives
