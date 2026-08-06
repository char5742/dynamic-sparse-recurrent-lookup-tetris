module ActivityPlasticity

using LinearAlgebra

using ..ActiveApicalCell
using ..ReducedHayCPUNativeModel
using ..ReducedHayCPUNativeArena
using ..TetrisRankingBatch
using ..Topology
using ..ReducedHayCPUNativeEventGraph
using ..OutputCellBank
using ..LearningConfig
using ..CanonicalOptimizer

export PlasticityState,
    AlignmentReport,
    update_firing_rates!,
    measure_subthreshold_alignment!,
    apply_subthreshold_eprop!,
    update_structural_utility!,
    apply_intrinsic_homeostasis!,
    apply_utility_rewiring!,
    reset_activity_counts!,
    accumulate_activity!

const Cell = ActiveApicalCell
const Model = ReducedHayCPUNativeModel
const Arena = ReducedHayCPUNativeArena
const Ranking = TetrisRankingBatch
const Graph = ReducedHayCPUNativeEventGraph
const OutputBank = OutputCellBank
const Config = LearningConfig
const Optimizer = CanonicalOptimizer

struct AlignmentReport
    reference_norm::Float64
    local_norm::Float64
    cosine::Float64
    nonspiking_eligibility_fraction::Float64
    third_factor_norm::Float64
end

const ZERO_ALIGNMENT_REPORT = AlignmentReport(0.0, 0.0, 0.0, 0.0, 0.0)

"""
Persistent slow state. Eligibility is recomputed from the fixed arena at the
configured local interval; only its normalized absolute task-tagged value is retained as
structural utility.  No sample, rail, or teacher target is stored here.
"""
mutable struct PlasticityState
    config::Config.LocalLearningConfig
    recurrent_rate::Matrix{Float32}
    output_rate::Vector{Float32}
    recurrent_dead_age::Matrix{UInt16}
    output_dead_age::Vector{UInt16}
    eligibility::Array{Float32,3}
    task_tagged_edge::Array{Float32,3}
    utility::Array{Float32,3}
    cell_signal::Vector{Float32}
    touched_block::BitVector
    touched_output_channel::BitVector
    recurrent_cursor::Int
    recurrent_cell_cursor::Int
    output_channel_cursor::Int
    updates::Int
    homeostatic_events::Int
    synaptic_scaling_events::Int
    subthreshold_updates::Int
    nonspiking_updates::Int
    utility_updates::Int
    rewires::Int
    last_alignment::AlignmentReport
end

function PlasticityState(config::Config.LocalLearningConfig=Config.LocalLearningConfig())
    initial_rate = clamp(
        2.0f0 * config.minimum_rate,
        config.minimum_rate,
        config.maximum_rate,
    )
    return PlasticityState(
        config,
        fill(initial_rate, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        fill(initial_rate, OutputBank.OUTPUT_CELLS),
        zeros(UInt16, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(UInt16, OutputBank.OUTPUT_CELLS),
        zeros(Float32, Model.FANOUT, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, Model.FANOUT, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, Model.FANOUT, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, Model.TOTAL_CELLS),
        falses(Model.BLOCKS),
        falses(OutputBank.OUTPUT_CHANNELS),
        1,
        1,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ZERO_ALIGNMENT_REPORT,
    )
end

function reset_activity_counts!(recurrent_counts, output_counts)
    for counts in recurrent_counts
        fill!(counts, UInt64(0))
    end
    for counts in output_counts
        fill!(counts, UInt64(0))
    end
    return nothing
end

"""Accumulate exact hard-event counts from one worker-owned forward tape."""
function accumulate_activity!(
    recurrent_count::AbstractMatrix{UInt64},
    output_count::AbstractVector{UInt64},
    buffers::Model.ForwardBuffers{Float32},
)
    size(recurrent_count) == (Model.CELLS_PER_BLOCK, Model.BLOCKS) || throw(
        DimensionMismatch("recurrent activity count has the wrong shape"),
    )
    length(output_count) == OutputBank.OUTPUT_CELLS || throw(
        DimensionMismatch("output activity count has the wrong shape"),
    )
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            recurrent_count[cell, block] += UInt64(
                buffers.physical_anchor[Cell.SPIKE_INDEX, cell, block] > 0.5f0,
            )
        end
    end
    @inbounds for step in 1:Model.RECURRENT_STEPS
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                recurrent_count[cell, block] += UInt64(
                    buffers.physical_recurrent[
                        Cell.SPIKE_INDEX,
                        cell,
                        block,
                        step,
                    ] > 0.5f0,
                )
            end
        end
        for output in 1:OutputBank.OUTPUT_CELLS
            output_count[output] += UInt64(
                buffers.output_trajectory.physical[
                    Cell.SPIKE_INDEX,
                    output,
                    step + 1,
                ] > 0.5f0,
            )
        end
    end
    return nothing
end

function update_firing_rates!(
    state::PlasticityState,
    recurrent_counts,
    output_counts,
    candidate_count::Integer,
)
    candidate_count > 0 || throw(ArgumentError("candidate count must be positive"))
    recurrent_observations = Float32(candidate_count * (Model.RECURRENT_STEPS + 1))
    output_observations = Float32(candidate_count * Model.RECURRENT_STEPS)
    decay = state.config.ema_decay
    complement = 1.0f0 - decay
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            count = UInt64(0)
            for worker_count in recurrent_counts
                count += worker_count[cell, block]
            end
            rate = Float32(count) / recurrent_observations
            state.recurrent_rate[cell, block] = muladd(
                decay,
                state.recurrent_rate[cell, block],
                complement * rate,
            )
        end
    end
    @inbounds for output in 1:OutputBank.OUTPUT_CELLS
        count = UInt64(0)
        for worker_count in output_counts
            count += worker_count[output]
        end
        rate = Float32(count) / output_observations
        state.output_rate[output] = muladd(
            decay,
            state.output_rate[output],
            complement * rate,
        )
    end
    state.updates += 1
    return state
end

@inline function _saturating_increment(value::UInt16)
    return value == typemax(UInt16) ? value : value + UInt16(1)
end

@inline function _reset_cell_moments!(first, second, cell::Int, block::Int)
    @inbounds for parameter in (
        Cell.P_BASAL_TO_SOMA,
        Cell.P_SOMA_THRESHOLD_GAP,
        Cell.P_ADAPTATION_GAIN,
        Cell.P_ADAPTATION_COUPLING,
    )
        first.cell_raw[parameter, cell, block] = 0.0f0
        second.cell_raw[parameter, cell, block] = 0.0f0
    end
    return nothing
end

@inline function _reset_output_cell_moments!(first, second, output::Int)
    @inbounds for parameter in (
        Cell.P_BASAL_TO_SOMA,
        Cell.P_SOMA_THRESHOLD_GAP,
        Cell.P_ADAPTATION_GAIN,
        Cell.P_ADAPTATION_COUPLING,
    )
        first.output_cell_raw[parameter, output] = 0.0f0
        second.output_cell_raw[parameter, output] = 0.0f0
    end
    return nothing
end

function _scale_recurrent_incoming!(
    parameters::Model.Parameters,
    first,
    second,
    model::Model.CPUHayModel,
    destination::Int,
    amount::Float32,
)
    graph = model.graph
    changed = 0
    @inbounds for source in 1:Model.TOTAL_CELLS
        source_cell = mod1(source, Model.CELLS_PER_BLOCK)
        source_block = (source - 1) ÷ Model.CELLS_PER_BLOCK + 1
        for relation in 1:Model.FANOUT
            slot = Graph.edge_slot(graph, source, relation)
            Int(graph.destination_cell[slot]) == destination || continue
            parameters.edge_strength_raw[relation, source_cell, source_block] += amount
            first.edge_strength_raw[relation, source_cell, source_block] = 0.0f0
            second.edge_strength_raw[relation, source_cell, source_block] = 0.0f0
            changed += 1
        end
    end
    return changed
end

function _scale_auxiliary_output_incoming!(
    parameters::Model.Parameters,
    first,
    second,
    model::Model.CPUHayModel,
    output::Int,
    amount::Float32,
)
    output > OutputBank.Q_OUTPUT_CELLS || throw(ArgumentError(
        "generic output homeostasis must not mutate shared numeric-Q cells",
    ))
    topology = model.output_topology
    changed = 0
    @inbounds for source in 1:OutputBank.SOURCE_CELLS
        for relation in 1:OutputBank.OUTPUT_FANOUT
            Int(topology.destination[relation, source]) == output || continue
            parameters.output_edge_raw[relation, source] += amount
            first.output_edge_raw[relation, source] = 0.0f0
            second.output_edge_raw[relation, source] = 0.0f0
            changed += 1
        end
    end
    return changed
end

function _prime_recurrent_cell!(state, parameters, first, second, model, cell, block)
    step = state.config.intrinsic_step
    @inbounds begin
        parameters.cell_raw[Cell.P_SOMA_THRESHOLD_GAP, cell, block] -= step
        parameters.cell_raw[Cell.P_ADAPTATION_GAIN, cell, block] -= 0.5f0 * step
        parameters.cell_raw[Cell.P_ADAPTATION_COUPLING, cell, block] -= 0.5f0 * step
        parameters.cell_raw[Cell.P_BASAL_TO_SOMA, cell, block] += step
    end
    _reset_cell_moments!(first, second, cell, block)
    destination = (block - 1) * Model.CELLS_PER_BLOCK + cell
    scaled = _scale_recurrent_incoming!(
        parameters,
        first,
        second,
        model,
        destination,
        state.config.synaptic_scale_step,
    )
    scaled > 0 && (state.synaptic_scaling_events += 1)
    fill!(@view(state.eligibility[:, cell, block]), 0.0f0)
    fill!(@view(state.task_tagged_edge[:, cell, block]), 0.0f0)
    return nothing
end

function _prime_auxiliary_output_cell!(
    state,
    parameters,
    first,
    second,
    model,
    output,
)
    output > OutputBank.Q_OUTPUT_CELLS || throw(ArgumentError(
        "generic output homeostasis must not mutate shared numeric-Q cells",
    ))
    step = state.config.intrinsic_step
    @inbounds begin
        parameters.output_cell_raw[Cell.P_SOMA_THRESHOLD_GAP, output] -= step
        parameters.output_cell_raw[Cell.P_ADAPTATION_GAIN, output] -= 0.5f0 * step
        parameters.output_cell_raw[Cell.P_ADAPTATION_COUPLING, output] -= 0.5f0 * step
        parameters.output_cell_raw[Cell.P_BASAL_TO_SOMA, output] += step
    end
    _reset_output_cell_moments!(first, second, output)
    scaled = _scale_auxiliary_output_incoming!(
        parameters,
        first,
        second,
        model,
        output,
        state.config.synaptic_scale_step,
    )
    scaled > 0 && (state.synaptic_scaling_events += 1)
    return nothing
end

"""
Apply bounded intrinsic plasticity.  At most one recurrent cell per block and
one output member per channel is changed at an interval, so a whole population
cannot be revived or suppressed in one discontinuous jump.
"""
function apply_intrinsic_homeostasis!(
    state::PlasticityState,
    parameters::Model.Parameters,
    first,
    second,
    model::Model.CPUHayModel,
)
    config = state.config
    config.mode == Config.LEARNING_ACTIVE || return 0
    state.updates >= config.warmup_updates || return 0
    state.updates % config.homeostasis_interval == 0 || return 0
    recurrent_active =
        state.updates <= config.recurrent_homeostasis_until
    output_active = state.updates <= config.output_homeostasis_until
    recurrent_active || output_active || return 0
    changed = 0
    fill!(state.touched_block, false)
    fill!(state.touched_output_channel, false)

    # Recurrent structural metabolism may be intentionally bounded while the
    # hard output population keeps intrinsic rate control for the whole run.
    # Conflating those lifetimes previously let output members die after the
    # structural warmup and collapsed a four-cell answer to one active cell.
    if recurrent_active
    # First update the slow dead-cell age for every cell.  Selection below is
    # global and bounded; it is not one simultaneous mutation per block.
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            rate = state.recurrent_rate[cell, block]
            if rate < config.minimum_rate
                state.recurrent_dead_age[cell, block] = _saturating_increment(
                    state.recurrent_dead_age[cell, block],
                )
            else
                state.recurrent_dead_age[cell, block] = UInt16(0)
            end
        end
    end
    recurrent_suppression_budget = config.maximum_recurrent_adjustments ÷ 2
    recurrent_changed = 0
    @inbounds for _ in 1:recurrent_suppression_budget
        selected_cell = 0
        selected_block = 0
        selected_rate = config.maximum_rate
        for block in 1:Model.BLOCKS
            state.touched_block[block] && continue
            for cell in 1:Model.CELLS_PER_BLOCK
                rate = state.recurrent_rate[cell, block]
                if rate > selected_rate
                    selected_cell = cell
                    selected_block = block
                    selected_rate = rate
                end
            end
        end
        if selected_cell != 0
            parameters.cell_raw[
                Cell.P_SOMA_THRESHOLD_GAP,
                selected_cell,
                selected_block,
            ] += config.intrinsic_step
            parameters.cell_raw[
                Cell.P_ADAPTATION_GAIN,
                selected_cell,
                selected_block,
            ] += 0.5f0 * config.intrinsic_step
            _reset_cell_moments!(
                first,
                second,
                selected_cell,
                selected_block,
            )
            destination = (selected_block - 1) * Model.CELLS_PER_BLOCK +
                          selected_cell
            scaled = _scale_recurrent_incoming!(
                parameters,
                first,
                second,
                model,
                destination,
                -config.synaptic_scale_step,
            )
            scaled > 0 && (state.synaptic_scaling_events += 1)
            state.touched_block[selected_block] = true
            changed += 1
            recurrent_changed += 1
        end
    end
    recurrent_prime_budget = config.maximum_recurrent_adjustments -
        recurrent_changed
    @inbounds for _ in 1:recurrent_prime_budget
        selected_cell = 0
        selected_block = 0
        for offset in 0:(Model.BLOCKS - 1)
            block = mod1(state.recurrent_cursor + offset, Model.BLOCKS)
            state.touched_block[block] && continue
            selected_rate = Inf32
            for cell_offset in 0:(Model.CELLS_PER_BLOCK - 1)
                cell = mod1(
                    state.recurrent_cell_cursor + cell_offset,
                    Model.CELLS_PER_BLOCK,
                )
                state.recurrent_dead_age[cell, block] >= config.dead_patience ||
                    continue
                rate = state.recurrent_rate[cell, block]
                if rate < selected_rate
                    selected_cell = cell
                    selected_block = block
                    selected_rate = rate
                end
            end
            selected_cell == 0 || break
        end
        selected_cell == 0 && break
        _prime_recurrent_cell!(
            state,
            parameters,
            first,
            second,
            model,
            selected_cell,
            selected_block,
        )
        state.recurrent_dead_age[selected_cell, selected_block] = UInt16(0)
        state.touched_block[selected_block] = true
        state.recurrent_cursor = mod1(selected_block + 1, Model.BLOCKS)
        state.recurrent_cell_cursor = mod1(
            selected_cell + 1,
            Model.CELLS_PER_BLOCK,
        )
        changed += 1
        recurrent_changed += 1
    end
    end # recurrent_active

    if output_active
    # Numeric-register bits have task-defined, strongly nonuniform firing
    # priors. Generic rate homeostasis would corrupt exponent/sign semantics;
    # silent Q bits recover through teacher-free subthreshold eligibility and
    # the supervised Q learning signal applied after candidate comparison.
    @inbounds for channel in 2:OutputBank.OUTPUT_CHANNELS
        for output in OutputBank.channel_output_range(channel)
            rate = state.output_rate[output]
            if rate < config.minimum_rate
                state.output_dead_age[output] = _saturating_increment(
                    state.output_dead_age[output],
                )
            else
                state.output_dead_age[output] = UInt16(0)
            end
        end
    end
    output_suppression_budget = config.maximum_output_adjustments ÷ 2
    output_changed = 0
    @inbounds for _ in 1:output_suppression_budget
        selected_output = 0
        selected_channel = 0
        selected_rate = config.maximum_rate
        for channel in 2:OutputBank.OUTPUT_CHANNELS
            state.touched_output_channel[channel] && continue
            for output in OutputBank.channel_output_range(channel)
                rate = state.output_rate[output]
                if rate > selected_rate
                    selected_output = output
                    selected_channel = channel
                    selected_rate = rate
                end
            end
        end
        if selected_output != 0
            parameters.output_cell_raw[
                Cell.P_SOMA_THRESHOLD_GAP,
                selected_output,
            ] += config.intrinsic_step
            parameters.output_cell_raw[
                Cell.P_ADAPTATION_GAIN,
                selected_output,
            ] += 0.5f0 * config.intrinsic_step
            _reset_output_cell_moments!(first, second, selected_output)
            scaled = _scale_auxiliary_output_incoming!(
                parameters,
                first,
                second,
                model,
                selected_output,
                -config.synaptic_scale_step,
            )
            scaled > 0 && (state.synaptic_scaling_events += 1)
            state.touched_output_channel[selected_channel] = true
            changed += 1
            output_changed += 1
        end
    end
    output_prime_budget = config.maximum_output_adjustments -
        output_changed
    @inbounds for _ in 1:output_prime_budget
        selected_output = 0
        selected_channel = 0
        for offset in 0:(OutputBank.OUTPUT_CHANNELS - 2)
            channel = 2 + mod(
                max(state.output_channel_cursor, 2) - 2 + offset,
                OutputBank.OUTPUT_CHANNELS - 1,
            )
            state.touched_output_channel[channel] && continue
            selected_rate = Inf32
            for output in OutputBank.channel_output_range(channel)
                state.output_dead_age[output] >= config.dead_patience || continue
                rate = state.output_rate[output]
                if rate < selected_rate
                    selected_output = output
                    selected_channel = channel
                    selected_rate = rate
                end
            end
            selected_output == 0 || break
        end
        selected_output == 0 && break
        _prime_auxiliary_output_cell!(
            state,
            parameters,
            first,
            second,
            model,
            selected_output,
        )
        state.output_dead_age[selected_output] = UInt16(0)
        state.touched_output_channel[selected_channel] = true
        state.output_channel_cursor = selected_channel == OutputBank.OUTPUT_CHANNELS ?
            2 : selected_channel + 1
        changed += 1
        output_changed += 1
    end
    end # output_active
    state.homeostatic_events += changed
    return changed
end

@inline function _cell_third_factor(
    raw_bar,
    flat::Int,
    source::Int,
    model,
    parameters,
    cache,
)
    value = 0.0f0
    topology = model.output_topology
    @inbounds for relation in 1:OutputBank.OUTPUT_FANOUT
        output = Int(topology.destination[relation, source])
        channel = OutputBank.output_channel(output)
        mean_gain = if output <= OutputBank.Q_OUTPUT_CELLS
            1.0f0
        else
            auxiliary = output - OutputBank.Q_OUTPUT_CELLS
            sum(@view parameters.output_gain[:, auxiliary]) /
                Float32(OutputBank.RECURRENT_STEPS)
        end
        feedback = cache.output.edge_strength[relation, source] *
            mean_gain
        value = muladd(feedback, raw_bar[channel, flat], value)
    end
    return value
end

@inline function _fixed_cell_third_factor(
    raw_bar,
    flat::Int,
    source::Int,
    projection,
)
    cell = mod1(source, Model.CELLS_PER_BLOCK)
    block = (source - 1) ÷ Model.CELLS_PER_BLOCK + 1
    value = 0.0f0
    @inbounds @simd for output in 1:Model.OUTPUT_DIM
        value = muladd(
            projection[output, cell, block],
            raw_bar[output, flat],
            value,
        )
    end
    return value
end

@inline function _recurrent_state(arena, cell, block, step, flat)
    return @view arena.physical_recurrent[:, cell, block, step, flat]
end

@inline function _subthreshold_psi(physical, cache, compartment::Int)
    voltage = physical[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)]
    nmda = physical[Cell.state_index(compartment, Cell.FIELD_NMDA)]
    plateau = physical[Cell.state_index(compartment, Cell.FIELD_PLATEAU)]
    soma = physical[Cell.SOMA_INDEX]
    spike = physical[Cell.SPIKE_INDEX]
    voltage_drive = clamp(
        abs(voltage - cache.compartment_rest) / 20.0f0,
        0.0f0,
        1.0f0,
    )
    nmda_drive = clamp(nmda / max(cache.nmda_max, 1.0f-4), 0.0f0, 1.0f0)
    plateau_drive = clamp(plateau, 0.0f0, 1.0f0)
    margin_drive = clamp(
        1.0f0 - abs(soma - cache.soma_threshold) /
            Cell.SPIKE_SURROGATE_WIDTH,
        0.0f0,
        1.0f0,
    )
    # The first four terms remain informative with a hard zero spike.
    return 0.20f0 * voltage_drive + 0.30f0 * nmda_drive +
           0.30f0 * plateau_drive + 0.20f0 * margin_drive,
           spike <= 0.5f0
end

@inline function _pre_event(arena, state, source_cell, source_block, step, delay, flat)
    source_step = step - (delay == 0x01 ? 2 : 1)
    source_step < 0 && return 0.0f0
    spike = if source_step == 0
        @inbounds arena.physical_anchor[
            Cell.SPIKE_INDEX,
            source_cell,
            source_block,
            flat,
        ]
    else
        @inbounds arena.physical_recurrent[
            Cell.SPIKE_INDEX,
            source_cell,
            source_block,
            source_step,
            flat,
        ]
    end
    return spike > 0.5f0 ? 1.0f0 : 0.0f0
end

@inline _task_tagged_eligibility(third_factor::Float32, trace::Float32) =
    third_factor * trace

"""Compute the task-tagged subthreshold eligibility trace without mutation."""
function _compute_subthreshold_trace!(
    state::PlasticityState,
    arena::Arena.FixedBatchArena,
    batch::Ranking.Batch,
    model::Model.CPUHayModel,
    prepared::Model.PreparedModelState,
    reference::Optimizer.ParameterGradient;
    third_factor_scale::Float32=1.0f0,
    fixed_third_factor=nothing,
)
    config = state.config
    config.mode == Config.LEARNING_DISABLED && return ZERO_ALIGNMENT_REPORT
    if fixed_third_factor !== nothing
        size(fixed_third_factor) == (
            Model.OUTPUT_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
        ) || throw(DimensionMismatch(
            "fixed third-factor projection has the wrong shape",
        ))
    end
    fill!(state.task_tagged_edge, 0.0f0)
    slot = Model.assert_generation(prepared, Arena.arena_generation(arena))
    cache = slot.cache
    parameters = slot.parameters
    graph = model.graph
    nonspiking_mass = 0.0
    total_mass = 0.0
    third_norm2 = 0.0
    @inbounds for ordinal in 1:batch.valid_count
        flat = Int(batch.valid_flats[ordinal])
        fill!(state.eligibility, 0.0f0)
        for source in 1:Model.TOTAL_CELLS
            signal = if fixed_third_factor === nothing
                _cell_third_factor(
                    batch.raw_gradient,
                    flat,
                    source,
                    model,
                    parameters,
                    cache,
                )
            else
                _fixed_cell_third_factor(
                    batch.raw_gradient,
                    flat,
                    source,
                    fixed_third_factor,
                )
            end
            state.cell_signal[source] = third_factor_scale * signal
        end
        for source in 1:Model.TOTAL_CELLS
            third_norm2 = muladd(
                Float64(state.cell_signal[source]),
                Float64(state.cell_signal[source]),
                third_norm2,
            )
        end
        for step in 1:Model.RECURRENT_STEPS
            for source_block in 1:Model.BLOCKS
                for source_cell in 1:Model.CELLS_PER_BLOCK
                    source = (source_block - 1) * Model.CELLS_PER_BLOCK + source_cell
                    for relation in 1:Model.FANOUT
                        graph_slot = Graph.edge_slot(graph, source, relation)
                        destination = Int(graph.destination_cell[graph_slot])
                        destination_block = (destination - 1) ÷ Model.CELLS_PER_BLOCK + 1
                        destination_cell = mod1(destination, Model.CELLS_PER_BLOCK)
                        compartment = Int(graph.destination_compartment[graph_slot])
                        pre = _pre_event(
                            arena,
                            state,
                            source_cell,
                            source_block,
                            step,
                            graph.delay_previous[graph_slot],
                            flat,
                        )
                        post = _recurrent_state(
                            arena,
                            destination_cell,
                            destination_block,
                            step,
                            flat,
                        )
                        psi, nonspiking = _subthreshold_psi(
                            post,
                            cache.cell[destination_cell, destination_block],
                            compartment,
                        )
                        polarity = graph.polarity[graph_slot] == Graph.EXCITATORY ?
                            1.0f0 : -1.0f0
                        previous = state.eligibility[
                            relation,
                            source_cell,
                            source_block,
                        ]
                        trace = muladd(
                            config.eligibility_decay,
                            previous,
                            polarity * pre * psi *
                                cache.edge_strength_derivative[graph_slot],
                        )
                        state.eligibility[
                            relation,
                            source_cell,
                            source_block,
                        ] = trace
                        contribution = _task_tagged_eligibility(
                            state.cell_signal[destination],
                            trace,
                        )
                        state.task_tagged_edge[
                            relation,
                            source_cell,
                            source_block,
                        ] += contribution
                        mass = abs(Float64(contribution))
                        total_mass += mass
                        nonspiking && (nonspiking_mass += mass)
                    end
                end
            end
        end
    end
    reference_flat = vec(reference.edge_strength_raw)
    local_flat = vec(state.task_tagged_edge)
    reference_norm = norm(reference_flat)
    local_norm = norm(local_flat)
    cosine = if reference_norm > 0 && local_norm > 0
        dot(reference_flat, local_flat) / (reference_norm * local_norm)
    else
        0.0
    end
    report = AlignmentReport(
        reference_norm,
        local_norm,
        cosine,
        total_mass > 0 ? nonspiking_mass / total_mass : 0.0,
        sqrt(third_norm2),
    )
    return report
end

"""Shadow-only alignment measurement. Never changes an optimizer gradient."""
function measure_subthreshold_alignment!(state, arena, batch, model, prepared,
                                         reference;
                                         third_factor_scale::Float32=1.0f0,
                                         fixed_third_factor=nothing)
    report = _compute_subthreshold_trace!(
        state, arena, batch, model, prepared, reference;
        third_factor_scale, fixed_third_factor,
    )
    state.last_alignment = report
    return report
end

"""Active e-prop update using the already unified fixed block signal."""
function apply_subthreshold_eprop!(state, arena, batch, model, prepared,
                                   gradient::Optimizer.ParameterGradient,
                                   fixed_third_factor)
    config = state.config
    config.mode == Config.LEARNING_ACTIVE || return ZERO_ALIGNMENT_REPORT
    state.updates % config.subthreshold_interval == 0 || return ZERO_ALIGNMENT_REPORT
    report = _compute_subthreshold_trace!(
        state, arena, batch, model, prepared, gradient;
        fixed_third_factor,
    )
    reference_norm = report.reference_norm
    trace_norm = report.local_norm
    if config.subthreshold_edge_scale > 0.0f0 && reference_norm > 0.0 && trace_norm > 0.0
        retained = 1.0f0 - config.subthreshold_edge_scale
        scale = Float32(config.subthreshold_edge_scale * reference_norm / trace_norm)
        @inbounds @simd for index in eachindex(gradient.edge_strength_raw)
            gradient.edge_strength_raw[index] = muladd(
                scale, state.task_tagged_edge[index],
                retained * gradient.edge_strength_raw[index],
            )
        end
        state.subthreshold_updates += 1
        report.nonspiking_eligibility_fraction > 0.0 &&
            (state.nonspiking_updates += 1)
    end
    return report
end

"""Update structural utility from the most recently computed active trace."""
function update_structural_utility!(state::PlasticityState)
    config = state.config
    config.mode == Config.LEARNING_ACTIVE || return 0
    mean_abs = sum(abs, state.task_tagged_edge) /
        max(length(state.task_tagged_edge), 1)
    mean_abs > 0.0f0 || return 0
    inverse_scale = inv(Float32(mean_abs))
    @inbounds for index in eachindex(state.utility)
        normalized = abs(state.task_tagged_edge[index]) * inverse_scale
        state.utility[index] = muladd(
            config.utility_decay,
            state.utility[index],
            (1.0f0 - config.utility_decay) * normalized,
        )
    end
    state.utility_updates += 1
    return 1
end

@inline function _source_has_destination(graph, source::Int, destination::Int)
    @inbounds for relation in 1:graph.fanout
        slot = Graph.edge_slot(graph, source, relation)
        Int(graph.destination_cell[slot]) == destination && return true
    end
    return false
end

"""
Low-frequency one-edge structural metabolism.  A dead destination receives an
edge from an active source; the source's least useful nonduplicate relation is
replaced.  Fanout, Dale polarity and delay are preserved, and only the changed
edge's optimizer moments and stale trace are reset.
"""
function apply_utility_rewiring!(
    state::PlasticityState,
    model::Model.CPUHayModel,
    parameters::Model.Parameters,
    first,
    second,
)
    config = state.config
    config.mode == Config.LEARNING_ACTIVE || return 0
    state.updates <= config.structure_until || return 0
    config.maximum_rewires > 0 || return 0
    state.updates >= config.warmup_updates || return 0
    state.updates % config.structure_interval == 0 || return 0
    graph = model.graph
    changed = 0
    @inbounds for destination_block in 1:Model.BLOCKS
        changed >= config.maximum_rewires && break
        destination_cell = argmin(@view state.recurrent_rate[:, destination_block])
        state.recurrent_rate[destination_cell, destination_block] <
            config.minimum_rate || continue
        state.recurrent_dead_age[destination_cell, destination_block] > 0 || continue
        destination = (destination_block - 1) * Model.CELLS_PER_BLOCK +
            destination_cell

        source = 0
        source_rate = -Inf32
        for candidate_block in 1:Model.BLOCKS
            for candidate_cell in 1:Model.CELLS_PER_BLOCK
                candidate = (candidate_block - 1) * Model.CELLS_PER_BLOCK +
                    candidate_cell
                _source_has_destination(graph, candidate, destination) && continue
                rate = state.recurrent_rate[candidate_cell, candidate_block]
                if rate > source_rate
                    source = candidate
                    source_rate = rate
                end
            end
        end
        source == 0 && continue
        source_block = (source - 1) ÷ Model.CELLS_PER_BLOCK + 1
        source_cell = mod1(source, Model.CELLS_PER_BLOCK)
        relation = argmin(@view state.utility[:, source_cell, source_block])
        graph_slot = Graph.edge_slot(graph, source, relation)
        graph.destination_cell[graph_slot] = Int32(destination)
        # Rotate the recipient branch so repeated recycling cannot collapse
        # every new contact onto one compartment.
        graph.destination_compartment[graph_slot] = UInt8(
            mod(state.rewires + changed, Cell.N_COMPARTMENTS) + 1,
        )
        parameters.edge_strength_raw[relation, source_cell, source_block] = 0.0f0
        first.edge_strength_raw[relation, source_cell, source_block] = 0.0f0
        second.edge_strength_raw[relation, source_cell, source_block] = 0.0f0
        state.eligibility[relation, source_cell, source_block] = 0.0f0
        state.task_tagged_edge[relation, source_cell, source_block] = 0.0f0
        state.utility[relation, source_cell, source_block] = 0.0f0
        changed += 1
    end
    state.rewires += changed
    return changed
end

end # module ActivityPlasticity
