@inline function _accumulate_exact_recurrent_cycle!(
    scratch,
    tape,
    model,
    parameters,
    cache,
    branch_for_edge,
    flat::Int,
    cycle::Int,
)
    cycle >= 2 || return nothing
    base = tape.base
    cells = model.blocks * model.cells_per_block
    fill!(scratch.branch_inbox, 0.0f0)

    # Reconstruct the exact recurrent inbox used by this cycle.  Unlike the
    # event-only eligibility replay, the reverse scan below must visit every
    # enabled edge: a silent source still has a non-zero surrogate derivative.
    @inbounds for source in 1:cells
        current = tape.cell_spikes[source, cycle - 1, flat]
        previous = cycle <= 2 ? 0.0f0 :
            tape.cell_spikes[source, cycle - 2, flat]
        for relation in 1:model.fanout
            gate = cache.gate_hard[source, relation]
            gate == 0.0f0 && continue
            delay = cache.delay[source, relation]
            pre = muladd(1.0f0 - delay, current, delay * previous)
            destination = model.destination_for_source[source, relation]
            branch = Int(branch_for_edge[source, relation])
            scratch.branch_inbox[destination, branch] = muladd(
                parameters.synapse_weight[source, relation] * gate,
                pre,
                scratch.branch_inbox[destination, branch],
            )
        end
    end

    @inbounds for source in 1:cells
        current_time = cycle - 1
        previous_time = cycle - 2
        current = tape.cell_spikes[source, current_time, flat]
        previous = previous_time < 1 ? 0.0f0 :
            tape.cell_spikes[source, previous_time, flat]
        source_block = div(source - 1, model.cells_per_block) + 1
        for relation in 1:model.fanout
            destination = model.destination_for_source[source, relation]
            branch = Int(branch_for_edge[source, relation])
            recurrent_drive = scratch.branch_inbox[destination, branch]
            drive_signal = if recurrent_drive > 0.0f0
                scratch.branch_exc_drive_signal[
                    destination, branch, cycle,
                ]
            elseif recurrent_drive < 0.0f0
                -scratch.branch_inh_drive_signal[
                    destination, branch, cycle,
                ]
            else
                0.5f0 * (
                    scratch.branch_exc_drive_signal[
                        destination, branch, cycle,
                    ] -
                    scratch.branch_inh_drive_signal[
                        destination, branch, cycle,
                    ]
                )
            end
            drive_signal == 0.0f0 && continue
            gate = cache.gate_hard[source, relation]
            delay = cache.delay[source, relation]
            weight = parameters.synapse_weight[source, relation]
            pre = muladd(1.0f0 - delay, current, delay * previous)

            scratch.gradient.synapse_weight[source, relation] = muladd(
                drive_signal,
                gate * pre,
                scratch.gradient.synapse_weight[source, relation],
            )
            scratch.gate_cotangent[source, relation] = muladd(
                drive_signal,
                weight * pre,
                scratch.gate_cotangent[source, relation],
            )
            scratch.gradient.delay_logits[source, relation] = muladd(
                drive_signal * weight * gate * (previous - current),
                cache.delay_derivative[source, relation],
                scratch.gradient.delay_logits[source, relation],
            )

            weighted = drive_signal * weight * gate
            current_signal = weighted * (1.0f0 - delay)
            current_mask = base.block_mask[
                source_block, current_time, flat,
            ]
            scratch.spike_signal[source, current_time] = muladd(
                current_signal,
                current_mask,
                scratch.spike_signal[source, current_time],
            )
            scratch.route_mask_signal[source_block, current_time] = muladd(
                current_signal,
                tape.soma_spikes[source, current_time, flat],
                scratch.route_mask_signal[source_block, current_time],
            )
            if previous_time >= 1
                previous_signal = weighted * delay
                previous_mask = base.block_mask[
                    source_block, previous_time, flat,
                ]
                scratch.spike_signal[source, previous_time] = muladd(
                    previous_signal,
                    previous_mask,
                    scratch.spike_signal[source, previous_time],
                )
                scratch.route_mask_signal[
                    source_block, previous_time,
                ] = muladd(
                    previous_signal,
                    tape.soma_spikes[source, previous_time, flat],
                    scratch.route_mask_signal[source_block, previous_time],
                )
            end
        end
    end
    return nothing
end

@inline function _finish_exact_gate_vjp!(
    scratch,
    model,
    parameters,
)
    keep = Float32(model.fixed_recurrent_fanout)
    inverse_temperature = inv(Float32(model.route_temperature))
    @inbounds for source in axes(parameters.gate_logits, 1)
        maximum_logit = -Inf32
        for relation in axes(parameters.gate_logits, 2)
            maximum_logit = max(
                maximum_logit,
                parameters.gate_logits[source, relation],
            )
        end
        normalizer = 0.0f0
        expected_cotangent = 0.0f0
        for relation in axes(parameters.gate_logits, 2)
            probability = exp(
                (parameters.gate_logits[source, relation] - maximum_logit) *
                inverse_temperature,
            )
            scratch.route_logweight[relation] = probability
            normalizer += probability
        end
        inverse_normalizer = inv(max(normalizer, eps(Float32)))
        for relation in axes(parameters.gate_logits, 2)
            probability =
                scratch.route_logweight[relation] * inverse_normalizer
            expected_cotangent = muladd(
                probability,
                scratch.gate_cotangent[source, relation],
                expected_cotangent,
            )
        end
        for relation in axes(parameters.gate_logits, 2)
            probability =
                scratch.route_logweight[relation] * inverse_normalizer
            scratch.gradient.gate_logits[source, relation] +=
                keep * inverse_temperature * probability *
                (scratch.gate_cotangent[source, relation] -
                 expected_cotangent)
        end
    end
    return nothing
end

"""
Reverse one v10 bound workspace write.

`workspace_root_signal[:, cycle]` is the direct head cotangent of
`workspace[:, cycle + 1]`.  `dworkspace_a` carries the cotangent arriving from
later workspace recurrence and from the apical read performed by the next
cell transition.  Spatial binding is an orthogonal signed permutation, so its
transpose is the inverse permutation used below when the selected write is
returned to raw cell coordinates.
"""
@inline function _reverse_bound_workspace_cycle!(
    scratch,
    tape,
    model,
    parameters,
    cache,
    flat::Int,
    cycle::Int,
    routing_temperature::Float32,
    routing_logit_limit::Float32,
    exact_graph_bptt::Bool,
)
    base = tape.base
    inverse_write_normalization = inv(sqrt(Float32(model.workspace_k)))
    fill!(scratch.feedback_error, 0.0f0)
    fill!(scratch.route_alpha, 0.0f0)

    @inbounds for bound_coordinate in 1:model.node_dim
        workspace_cotangent =
            scratch.point_scratch.dworkspace_a[bound_coordinate] +
            scratch.workspace_root_signal[bound_coordinate, cycle]
        old_workspace = base.workspace[bound_coordinate, cycle, flat]
        write = 0.0f0
        for block in 1:model.blocks
            raw_coordinate = Int(tape.spatial_bound_coordinate[
                bound_coordinate,
                block,
            ])
            sign_value = tape.spatial_bound_sign[
                bound_coordinate,
                block,
            ]
            raw_state = base.membrane[
                raw_coordinate + (block - 1) * model.node_dim,
                cycle + 1,
                flat,
            ]
            bound_state = sign_value * raw_state
            selected = base.block_mask[block, cycle, flat]
            write = muladd(selected, bound_state, write)
        end
        write *= inverse_write_normalization

        scratch.gradient.workspace_decay_logit[1] +=
            workspace_cotangent * (old_workspace - write) *
            cache.workspace_decay_derivative
        write_cotangent =
            workspace_cotangent * (1.0f0 - cache.workspace_decay)

        for block in 1:model.blocks
            raw_coordinate = Int(tape.spatial_bound_coordinate[
                bound_coordinate,
                block,
            ])
            sign_value = tape.spatial_bound_sign[
                bound_coordinate,
                block,
            ]
            raw_state = base.membrane[
                raw_coordinate + (block - 1) * model.node_dim,
                cycle + 1,
                flat,
            ]
            bound_state = sign_value * raw_state
            mask_cotangent =
                write_cotangent * inverse_write_normalization
            scratch.route_alpha[block] = muladd(
                mask_cotangent,
                bound_state,
                scratch.route_alpha[block],
            )
            base.block_mask[block, cycle, flat] == 0.0f0 && continue
            scratch.feedback_error[raw_coordinate, block] = muladd(
                mask_cotangent,
                sign_value,
                scratch.feedback_error[raw_coordinate, block],
            )
        end
        scratch.point_scratch.dworkspace_a[bound_coordinate] =
            workspace_cotangent * cache.workspace_decay
    end

    if exact_graph_bptt
        @inbounds for block in 1:model.blocks
            scratch.route_alpha[block] +=
                scratch.route_mask_signal[block, cycle]
        end
    end
    _accumulate_pathwise_route_cycle!(
        scratch,
        tape,
        model,
        parameters,
        flat,
        cycle,
        routing_temperature,
        routing_logit_limit,
    )
    @inbounds for block in 1:model.blocks
        for raw_coordinate in 1:model.node_dim
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                raw_coordinate,
                block,
                cycle,
                flat,
                scratch.feedback_error[raw_coordinate, block],
            )
        end
    end
    return nothing
end

"""
Reconstruct the v11 exact-slot workspace and the selected global route
context for one candidate.

The forward worker scratch is reused by every candidate in an arena batch, so
neither trajectory may be assumed to still belong to `flat` when its backward
pass starts.  Both recurrences are replayed from the immutable membrane and
hard-route tapes without allocation.
"""
@inline function _replay_exact_slot_trajectory!(
    scratch,
    tape,
    model,
    parameters,
    cache,
    flat::Int,
)
    base = tape.base
    node_dim = model.node_dim
    inverse_selected = inv(sqrt(Float32(model.workspace_k)))
    fill!(scratch.exact_workspace, 0.0f0)
    fill!(scratch.route_context, 0.0f0)
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            selected = base.block_mask[block, cycle, flat]
            block_offset = (block - 1) * node_dim
            for coordinate in 1:node_dim
                previous = scratch.exact_workspace[
                    coordinate, block, cycle,
                ]
                state = base.membrane[
                    block_offset + coordinate, cycle + 1, flat,
                ]
                scratch.exact_workspace[
                    coordinate, block, cycle + 1,
                ] = muladd(
                    selected,
                    state,
                    (1.0f0 - selected) * cache.workspace_decay * previous,
                )
            end
        end
        _replay_v11_route_state_cycle!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle,
        )
        for route in 1:model.route_dim
            context = 0.0f0
            for block in 1:model.blocks
                context = muladd(
                    base.block_mask[block, cycle, flat] *
                    _v11_route_block_sign(model, block, route),
                    scratch.route_state[route, block],
                    context,
                )
            end
            scratch.route_context[route, cycle + 1] =
                context * inverse_selected
        end
    end
    return nothing
end

"""Rebuild `P_route * X[cycle+1]` for every v11 block."""
@inline function _replay_v11_route_state_cycle!(
    scratch,
    tape,
    model,
    parameters,
    flat::Int,
    cycle::Int,
)
    base = tape.base
    state_rank = size(parameters.route_state_projection, 1)
    fill!(scratch.route_state, 0.0f0)
    @inbounds for block in 1:model.blocks
        block_offset = (block - 1) * model.node_dim
        for local_cell in 1:model.cells_per_block
            cell_offset =
                block_offset + (local_cell - 1) * model.readout_per_cell
            route_offset = (local_cell - 1) * state_rank
            for state in 1:model.readout_per_cell
                value = base.membrane[
                    cell_offset + state, cycle + 1, flat,
                ]
                for rank in 1:state_rank
                    route = route_offset + rank
                    scratch.route_state[route, block] = muladd(
                        parameters.route_state_projection[
                            rank, state, local_cell,
                        ],
                        value,
                        scratch.route_state[route, block],
                    )
                end
            end
        end
    end
    return nothing
end

"""
Reverse the v11 exact block-slot recurrence and selected route context for one
cycle.  `slot_adjoint` and `global_context_signal` carry cotangents from later
cell transitions.  Direct state roots are accumulated in `feedback_error`;
route-score/query VJP is deliberately delegated to the shared route helper so
all contributions pass through `P_route` exactly once.
"""
@inline function _reverse_exact_slot_cycle!(
    scratch,
    tape,
    model,
    parameters,
    cache,
    flat::Int,
    cycle::Int,
    routing_temperature::Float32,
    routing_logit_limit::Float32,
    pathwise_route_parameters::Bool,
)
    base = tape.base
    node_dim = model.node_dim
    inverse_selected = inv(sqrt(Float32(model.workspace_k)))
    fill!(scratch.feedback_error, 0.0f0)
    fill!(scratch.route_alpha, 0.0f0)
    fill!(scratch.route_state_signal, 0.0f0)
    _replay_v11_route_state_cycle!(
        scratch,
        tape,
        model,
        parameters,
        flat,
        cycle,
    )

    # W[c+1,b] = m[b,c] * X[c+1,b]
    #            + (1-m[b,c]) * decay * W[c,b].
    @inbounds for block in 1:model.blocks
        selected = base.block_mask[block, cycle, flat]
        unselected = 1.0f0 - selected
        block_offset = (block - 1) * node_dim
        for coordinate in 1:node_dim
            slot_signal = scratch.slot_adjoint[coordinate, block]
            previous = scratch.exact_workspace[
                coordinate, block, cycle,
            ]
            state = base.membrane[
                block_offset + coordinate, cycle + 1, flat,
            ]
            scratch.feedback_error[coordinate, block] = muladd(
                selected,
                slot_signal,
                scratch.feedback_error[coordinate, block],
            )
            scratch.route_alpha[block] = muladd(
                slot_signal,
                state - cache.workspace_decay * previous,
                scratch.route_alpha[block],
            )
            scratch.gradient.workspace_decay_logit[1] +=
                slot_signal * unselected * previous *
                cache.workspace_decay_derivative
            scratch.slot_adjoint[coordinate, block] =
                slot_signal * unselected * cache.workspace_decay
        end
    end

    # C[c+1,r] = sum_b m[b,c] code[b,r] R[c,b,r] / sqrt(K).
    # Consume the present context cotangent here; the cell reverse below will
    # subsequently accumulate dC[c] for the preceding route cycle.
    @inbounds for route in 1:model.route_dim
        context_signal = scratch.global_context_signal[route]
        scratch.global_context_signal[route] = 0.0f0
        context_signal == 0.0f0 && continue
        scaled_signal = context_signal * inverse_selected
        for block in 1:model.blocks
            selected = base.block_mask[block, cycle, flat]
            code = _v11_route_block_sign(model, block, route)
            route_state = scratch.route_state[route, block]
            scratch.route_state_signal[route, block] = muladd(
                selected * code,
                scaled_signal,
                scratch.route_state_signal[route, block],
            )
            scratch.route_alpha[block] = muladd(
                code * route_state,
                scaled_signal,
                scratch.route_alpha[block],
            )
        end
    end

    # History-head and exact recurrent-event roots share the hard mask.  v11
    # is admitted only in exact-BPTT mode, so these are pathwise roots rather
    # than a block-local reward surrogate.
    @inbounds for block in 1:model.blocks
        scratch.route_alpha[block] +=
            scratch.route_mask_signal[block, cycle]
        tape.block_supervised_reward[block, cycle, flat] =
            -scratch.route_alpha[block]
    end
    _accumulate_pathwise_route_cycle!(
        scratch,
        tape,
        model,
        parameters,
        flat,
        cycle,
        routing_temperature,
        routing_logit_limit,
        pathwise_route_parameters,
    )
    # The v11 route VJP reuses this vector for the query-input cotangent after
    # the actual dC[c+1] has been consumed above.  Do not let that temporary
    # value leak into the persistent dC[c] subsequently written by the cell
    # feedback reverse.
    fill!(scratch.global_context_signal, 0.0f0)
    @inbounds for block in 1:model.blocks
        for coordinate in 1:node_dim
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                cycle,
                flat,
                scratch.feedback_error[coordinate, block],
            )
        end
    end
    return nothing
end

"""
Fixed-memory temporal adjoint for the Reduced Hay state.

The default remains the block-local DECOLLE/e-prop path.  With
`exact_graph_bptt=true`, the same reverse cycle also propagates destination
drive cotangents through recurrent edges into earlier source spikes and route
masks.  That mode is the allocation-free analytic counterpart of the Zygote
reference BPTT and is used only as a performance-ceiling control.
"""
function _accumulate_cell_temporal_gradients!(
    scratch,
    tape,
    model,
    parameters,
    cache,
    flat::Int;
    routing_temperature::Float32=model.route_temperature,
    routing_logit_limit::Float32=Routing.DEFAULT_LOGIT_LIMIT,
    branch_for_edge=nothing,
    exact_graph_bptt::Bool=false,
    pathwise_route_parameters::Bool=true,
)
    base = tape.base
    cells = model.blocks * model.cells_per_block
    readout = model.readout_per_cell
    sensory_normalization = inv(sqrt(Float32(model.sensory_fanin)))
    sensory_cycle_scale = Model.reduced_hay_sensory_cycle_scale(model)
    fill!(scratch.adjoint_ampa, 0.0f0)
    fill!(scratch.adjoint_nmda, 0.0f0)
    fill!(scratch.adjoint_gaba, 0.0f0)
    fill!(scratch.adjoint_branch, 0.0f0)
    fill!(scratch.adjoint_plateau, 0.0f0)
    fill!(scratch.adjoint_apical, 0.0f0)
    fill!(scratch.adjoint_soma, 0.0f0)
    fill!(scratch.adjoint_adaptation, 0.0f0)
    fill!(scratch.branch_exc_drive_signal, 0.0f0)
    fill!(scratch.branch_inh_drive_signal, 0.0f0)
    fill!(scratch.point_scratch.dworkspace_a, 0.0f0)
    exact_block_slots = _uses_exact_block_slots(model)
    if exact_block_slots
        fill!(scratch.slot_adjoint, 0.0f0)
        fill!(scratch.route_state_signal, 0.0f0)
        fill!(scratch.route_query_signal, 0.0f0)
        fill!(scratch.global_context_signal, 0.0f0)
        _replay_exact_slot_trajectory!(
            scratch,
            tape,
            model,
            parameters,
            cache,
            flat,
        )
    end
    if exact_graph_bptt
        branch_for_edge === nothing &&
            error("exact graph BPTT requires branch_for_edge")
        fill!(scratch.gate_cotangent, 0.0f0)
    end

    @inbounds for cycle in model.cycles:-1:1
        # The full-state v10 workspace is a linear, spatially bound recurrence.
        # Reverse w[t+1] before the cell transition that produced z[t+1], so
        # its selected-write cotangent is present when that transition is
        # visited below.  Legacy models retain the historical after-cell tanh
        # workspace reverse unchanged.
        if exact_block_slots
            _reverse_exact_slot_cycle!(
                scratch,
                tape,
                model,
                parameters,
                cache,
                flat,
                cycle,
                routing_temperature,
                routing_logit_limit,
                pathwise_route_parameters,
            )
        elseif model.workspace_binding !== :none
            _reverse_bound_workspace_cycle!(
                scratch,
                tape,
                model,
                parameters,
                cache,
                flat,
                cycle,
                routing_temperature,
                routing_logit_limit,
                exact_graph_bptt,
            )
        end
        for cell in 1:cells
            block = div(cell - 1, model.cells_per_block) + 1
            local_cell = cell - (block - 1) * model.cells_per_block
            old_apical = tape.apical[cell, cycle, flat]
            next_apical = tape.apical[cell, cycle + 1, flat]
            old_soma = tape.soma[cell, cycle, flat]
            old_adaptation = tape.adaptation[cell, cycle, flat]
            spike = tape.soma_spikes[cell, cycle, flat]
            threshold = cache.soma_threshold[cell]
            soma_pre = tape.soma[cell, cycle + 1, flat] + spike * threshold
            post_surrogate = _spike_surrogate(
                soma_pre,
                threshold,
                model.spike_temperature,
            )

            basal = 0.0f0
            for branch in 1:model.branches
                basal = muladd(
                    parameters.soma_coupling[branch, cell],
                    tape.branch_voltage[cell, branch, cycle + 1, flat] +
                    tape.plateau[cell, branch, cycle + 1, flat],
                    basal,
                )
            end
            apical_activation =
                _apical_activation(model, next_apical)
            modulation = 1.0f0 +
                cache.apical_gain[cell] * apical_activation

            adjoint_next_soma = scratch.adjoint_soma[cell] +
                scratch.soma_signal[cell, cycle]
            adjoint_next_adaptation =
                scratch.adjoint_adaptation[cell] +
                (model.cell_export === :full24 ?
                 scratch.adaptation_signal[cell, cycle] : 0.0f0)
            # spike_signal contains both an exported-event root cotangent and,
            # in exact graph mode, later recurrent-edge cotangents.  Even when
            # graph BPTT is disabled the former must pass through the soma
            # surrogate rather than being silently dropped.
            external_spike_signal =
                (model.cell_export === :full24 || exact_graph_bptt) ?
                scratch.spike_signal[cell, cycle] : 0.0f0
            adjoint_soma_pre =
                adjoint_next_soma *
                (1.0f0 - threshold * post_surrogate) +
                adjoint_next_adaptation *
                cache.adaptation_gain[cell] * post_surrogate +
                external_spike_signal * post_surrogate

            threshold_effect =
                adjoint_next_soma *
                (-spike + threshold * post_surrogate) -
                adjoint_next_adaptation *
                cache.adaptation_gain[cell] * post_surrogate -
                external_spike_signal * post_surrogate
            scratch.gradient.soma_threshold_logits[cell] +=
                threshold_effect * cache.soma_threshold_derivative[cell]
            scratch.gradient.adaptation_gain_logits[cell] +=
                adjoint_next_adaptation * spike *
                cache.adaptation_gain_derivative[cell]
            scratch.gradient.adaptation_decay_logits[cell] +=
                adjoint_next_adaptation * old_adaptation *
                cache.adaptation_decay_derivative[cell]
            scratch.gradient.soma_leak_logits[cell] +=
                adjoint_soma_pre * old_soma *
                cache.soma_leak_derivative[cell]

            adjoint_old_adaptation =
                adjoint_next_adaptation * cache.adaptation_decay[cell] -
                adjoint_soma_pre
            adjoint_old_soma =
                adjoint_soma_pre * cache.soma_leak[cell]
            adjoint_basal = adjoint_soma_pre * modulation
            adjoint_modulation = adjoint_soma_pre * basal
            adjoint_next_apical = scratch.adjoint_apical[cell] +
                scratch.apical_signal[cell, cycle] +
                adjoint_modulation * cache.apical_gain[cell] *
                _hard_sigmoid_derivative(next_apical)
            scratch.gradient.apical_gain_logits[cell] +=
                adjoint_modulation * apical_activation *
                cache.apical_gain_derivative[cell]
            scratch.gradient.apical_leak_logits[cell] +=
                adjoint_next_apical * old_apical *
                cache.apical_leak_derivative[cell]
            adjoint_old_apical =
                adjoint_next_apical * cache.apical_leak[cell]

            if exact_block_slots
                inverse_local_feedback = inv(sqrt(Float32(readout)))
                cell_offset = (local_cell - 1) * readout
                for state in 1:readout
                    coordinate = cell_offset + state
                    workspace_value = scratch.exact_workspace[
                        coordinate, block, cycle,
                    ]
                    scratch.gradient.feedback_gain[
                        state, local_cell, block,
                    ] = muladd(
                        adjoint_next_apical * inverse_local_feedback,
                        workspace_value,
                        scratch.gradient.feedback_gain[
                            state, local_cell, block,
                        ],
                    )
                    scratch.slot_adjoint[coordinate, block] = muladd(
                        parameters.feedback_gain[
                            state, local_cell, block,
                        ] * inverse_local_feedback,
                        adjoint_next_apical,
                        scratch.slot_adjoint[coordinate, block],
                    )
                end
                inverse_global_feedback =
                    inv(sqrt(Float32(model.route_dim)))
                for route in 1:model.route_dim
                    context = scratch.route_context[route, cycle]
                    scratch.gradient.global_feedback_gain[
                        route, local_cell, block,
                    ] = muladd(
                        adjoint_next_apical * inverse_global_feedback,
                        context,
                        scratch.gradient.global_feedback_gain[
                            route, local_cell, block,
                        ],
                    )
                    scratch.global_context_signal[route] = muladd(
                        parameters.global_feedback_gain[
                            route, local_cell, block,
                        ] * inverse_global_feedback,
                        adjoint_next_apical,
                        scratch.global_context_signal[route],
                    )
                end
            else
                feedback_normalization = model.workspace_binding !== :none ?
                    sqrt(Float32(readout)) : Float32(readout)
                for channel in 1:readout
                    raw_coordinate = channel + (local_cell - 1) * readout
                    workspace_coordinate = model.workspace_binding !== :none ?
                        Int(tape.spatial_inverse_coordinate[
                            raw_coordinate,
                            block,
                        ]) : raw_coordinate
                    workspace_sign = model.workspace_binding !== :none ?
                        tape.spatial_inverse_sign[
                            raw_coordinate,
                            block,
                        ] : 1.0f0
                    unbound_workspace = workspace_sign * base.workspace[
                        workspace_coordinate,
                        cycle,
                        flat,
                    ]
                    scratch.gradient.feedback_gain[
                        raw_coordinate,
                        block,
                    ] = muladd(
                        adjoint_next_apical / feedback_normalization,
                        unbound_workspace,
                        scratch.gradient.feedback_gain[
                            raw_coordinate,
                            block,
                        ],
                    )
                    scratch.point_scratch.dworkspace_a[
                        workspace_coordinate,
                    ] = muladd(
                        parameters.feedback_gain[raw_coordinate, block] *
                        workspace_sign / feedback_normalization,
                        adjoint_next_apical,
                        scratch.point_scratch.dworkspace_a[
                            workspace_coordinate,
                        ],
                    )
                end
            end

            for branch in 1:model.branches
                old_branch = tape.branch_voltage[
                    cell, branch, cycle, flat,
                ]
                next_branch = tape.branch_voltage[
                    cell, branch, cycle + 1, flat,
                ]
                old_ampa = tape.ampa[cell, branch, cycle, flat]
                next_ampa = tape.ampa[cell, branch, cycle + 1, flat]
                old_nmda = tape.nmda[cell, branch, cycle, flat]
                next_nmda = tape.nmda[cell, branch, cycle + 1, flat]
                old_gaba = tape.gaba[cell, branch, cycle, flat]
                next_gaba = tape.gaba[cell, branch, cycle + 1, flat]
                old_plateau = tape.plateau[
                    cell, branch, cycle, flat,
                ]
                next_plateau = tape.plateau[
                    cell, branch, cycle + 1, flat,
                ]

                coupling = parameters.soma_coupling[branch, cell]
                scratch.gradient.soma_coupling[branch, cell] +=
                    adjoint_basal * (next_branch + next_plateau)
                adjoint_next_branch =
                    scratch.adjoint_branch[cell, branch] +
                    scratch.branch_signal[cell, branch, cycle] +
                    adjoint_basal * coupling
                adjoint_next_plateau =
                    scratch.adjoint_plateau[cell, branch] +
                    (model.cell_export === :full24 ?
                     scratch.plateau_signal[cell, branch, cycle] : 0.0f0) +
                    adjoint_basal * coupling

                plateau_argument =
                    cache.plateau_slope[branch, cell] *
                    (next_branch -
                     cache.plateau_threshold[branch, cell])
                coincidence = _hard_sigmoid(plateau_argument)
                coincidence_derivative =
                    _hard_sigmoid_derivative(plateau_argument)
                raw_plateau =
                    cache.plateau_decay[branch, cell] * old_plateau +
                    cache.plateau_gain[branch, cell] *
                    next_nmda * coincidence
                plateau_clamp_derivative =
                    0.0f0 < raw_plateau < 4.0f0 ? 1.0f0 : 0.0f0
                adjoint_raw_plateau =
                    adjoint_next_plateau * plateau_clamp_derivative
                scratch.gradient.plateau_decay_logits[branch, cell] +=
                    adjoint_raw_plateau * old_plateau *
                    cache.plateau_decay_derivative[branch, cell]
                scratch.gradient.plateau_gain_logits[branch, cell] +=
                    adjoint_raw_plateau * next_nmda * coincidence *
                    cache.plateau_gain_derivative[branch, cell]
                adjoint_old_plateau = adjoint_raw_plateau *
                    cache.plateau_decay[branch, cell]
                adjoint_next_nmda =
                    scratch.adjoint_nmda[cell, branch] +
                    (model.cell_export === :full24 ?
                     scratch.nmda_signal[cell, branch, cycle] : 0.0f0) +
                    adjoint_raw_plateau *
                    cache.plateau_gain[branch, cell] * coincidence
                adjoint_coincidence =
                    adjoint_raw_plateau *
                    cache.plateau_gain[branch, cell] * next_nmda
                scratch.gradient.plateau_threshold_logits[
                    branch, cell,
                ] +=
                    adjoint_coincidence * coincidence_derivative *
                    (-cache.plateau_slope[branch, cell]) *
                    cache.plateau_threshold_derivative[branch, cell]
                scratch.gradient.plateau_slope_logits[branch, cell] +=
                    adjoint_coincidence * coincidence_derivative *
                    (next_branch -
                     cache.plateau_threshold[branch, cell]) *
                    cache.plateau_slope_derivative[branch, cell]
                adjoint_next_branch +=
                    adjoint_coincidence * coincidence_derivative *
                    cache.plateau_slope[branch, cell]

                unblock_argument =
                    cache.nmda_slope[branch, cell] *
                    (old_branch - cache.nmda_half[branch, cell])
                unblock = sigmoid(unblock_argument)
                unblock_derivative = unblock * (1.0f0 - unblock)
                excitatory_current =
                    (next_ampa + next_nmda * unblock) *
                    (1.0f0 - old_branch)
                inhibitory_current =
                    next_gaba * (-1.0f0 - old_branch)
                total_current = excitatory_current + inhibitory_current
                raw_branch =
                    cache.branch_leak[branch, cell] * old_branch +
                    cache.current_gain[branch, cell] * total_current +
                    cache.axial_gain[branch, cell] *
                    (old_soma - old_branch) +
                    cache.plateau_feedback[branch, cell] * old_plateau
                branch_clamp_derivative =
                    -2.0f0 < raw_branch < 3.0f0 ? 1.0f0 : 0.0f0
                adjoint_raw_branch =
                    adjoint_next_branch * branch_clamp_derivative
                scratch.gradient.branch_leak_logits[branch, cell] +=
                    adjoint_raw_branch * old_branch *
                    cache.branch_leak_derivative[branch, cell]
                scratch.gradient.current_gain_logits[branch, cell] +=
                    adjoint_raw_branch * total_current *
                    cache.current_gain_derivative[branch, cell]
                scratch.gradient.axial_gain_logits[branch, cell] +=
                    adjoint_raw_branch * (old_soma - old_branch) *
                    cache.axial_gain_derivative[branch, cell]
                scratch.gradient.plateau_feedback_logits[branch, cell] +=
                    adjoint_raw_branch * old_plateau *
                    cache.plateau_feedback_derivative[branch, cell]

                adjoint_current =
                    adjoint_raw_branch * cache.current_gain[branch, cell]
                adjoint_next_ampa =
                    scratch.adjoint_ampa[cell, branch] +
                    (model.cell_export === :full24 ?
                     scratch.ampa_signal[cell, branch, cycle] : 0.0f0) +
                    adjoint_current * (1.0f0 - old_branch)
                adjoint_next_nmda +=
                    adjoint_current * unblock *
                    (1.0f0 - old_branch)
                adjoint_next_gaba =
                    scratch.adjoint_gaba[cell, branch] +
                    (model.cell_export === :full24 ?
                     scratch.gaba_signal[cell, branch, cycle] : 0.0f0) +
                    adjoint_current * (-1.0f0 - old_branch)
                # These are the exact cell-local cotangents of the E/I drive.
                # The inter-cell graph is not traversed backwards: the later
                # source-major replay only combines them with each edge's
                # delayed presynaptic factor.  This is the factorized e-prop
                # decomposition, but without the former one-branch trace that
                # lost soma-mediated cross-branch paths.
                scratch.branch_exc_drive_signal[
                    cell, branch, cycle,
                ] = adjoint_next_ampa + 0.72f0 * adjoint_next_nmda
                scratch.branch_inh_drive_signal[
                    cell, branch, cycle,
                ] = adjoint_next_gaba
                adjoint_unblock =
                    adjoint_current * next_nmda *
                    (1.0f0 - old_branch)
                adjoint_old_branch =
                    adjoint_raw_branch *
                    (cache.branch_leak[branch, cell] -
                     cache.axial_gain[branch, cell]) -
                    adjoint_current *
                    (next_ampa + next_nmda * unblock + next_gaba) +
                    adjoint_unblock * unblock_derivative *
                    cache.nmda_slope[branch, cell]
                adjoint_old_soma +=
                    adjoint_raw_branch * cache.axial_gain[branch, cell]
                adjoint_old_plateau +=
                    adjoint_raw_branch *
                    cache.plateau_feedback[branch, cell]
                scratch.gradient.nmda_slope_logits[branch, cell] +=
                    adjoint_unblock * unblock_derivative *
                    (old_branch - cache.nmda_half[branch, cell]) *
                    cache.nmda_slope_derivative[branch, cell]
                scratch.gradient.nmda_half_logits[branch, cell] +=
                    -adjoint_unblock * unblock_derivative *
                    cache.nmda_slope[branch, cell] *
                    cache.nmda_half_derivative[branch, cell]

                scratch.gradient.ampa_decay_logits[branch, cell] +=
                    adjoint_next_ampa * old_ampa *
                    cache.ampa_decay_derivative[branch, cell]
                scratch.gradient.nmda_decay_logits[branch, cell] +=
                    adjoint_next_nmda * old_nmda *
                    cache.nmda_decay_derivative[branch, cell]
                scratch.gradient.gaba_decay_logits[branch, cell] +=
                    adjoint_next_gaba * old_gaba *
                    cache.gaba_decay_derivative[branch, cell]
                adjoint_old_ampa = adjoint_next_ampa *
                    cache.ampa_decay[branch, cell]
                adjoint_old_nmda = adjoint_next_nmda *
                    cache.nmda_decay[branch, cell]
                adjoint_old_gaba = adjoint_next_gaba *
                    cache.gaba_decay[branch, cell]

                if cycle <= model.sensory_cycles
                    excitatory_drive_effect =
                        adjoint_next_ampa + 0.72f0 * adjoint_next_nmda
                    inhibitory_drive_effect = adjoint_next_gaba
                    branch_bias_derivative =
                        if model.branch_bias_mode ===
                           Model.BOUNDED_POSITIVE_BRANCH_BIAS
                            sensory_cycle_scale *
                            Model.reduced_hay_branch_bias_derivative(
                                model,
                                parameters.branch_bias[branch, cell],
                            )
                        else
                            # Preserve the established raw-bias adjoint for
                            # legacy/v10/v11/v12 bit-for-bit.  Only the v13
                            # bounded-positive parameterization introduces a
                            # logit Jacobian here.
                            1.0f0
                        end
                    scratch.gradient.branch_bias[branch, cell] +=
                        excitatory_drive_effect * branch_bias_derivative
                    for contact in 1:model.sensory_fanin
                        exc_rail = model.excitatory_feature[
                            contact, branch, cell,
                        ]
                        inh_rail = model.inhibitory_feature[
                            contact, branch, cell,
                        ]
                        scratch.gradient.input_exc_logits[
                            contact, branch, cell,
                        ] = muladd(
                            excitatory_drive_effect *
                            cache.input_exc_derivative[
                                contact, branch, cell,
                            ],
                            base.rails[exc_rail, flat] *
                            sensory_normalization,
                            scratch.gradient.input_exc_logits[
                                contact, branch, cell,
                            ],
                        )
                        scratch.gradient.input_inh_logits[
                            contact, branch, cell,
                        ] = muladd(
                            inhibitory_drive_effect *
                            cache.input_inh_derivative[
                                contact, branch, cell,
                            ],
                            base.rails[inh_rail, flat] *
                            sensory_normalization,
                            scratch.gradient.input_inh_logits[
                                contact, branch, cell,
                            ],
                        )
                    end
                end

                scratch.adjoint_ampa[cell, branch] = adjoint_old_ampa
                scratch.adjoint_nmda[cell, branch] = adjoint_old_nmda
                scratch.adjoint_gaba[cell, branch] = adjoint_old_gaba
                scratch.adjoint_branch[cell, branch] = adjoint_old_branch
                scratch.adjoint_plateau[cell, branch] = adjoint_old_plateau
            end
            scratch.adjoint_apical[cell] = adjoint_old_apical
            scratch.adjoint_soma[cell] = adjoint_old_soma
            scratch.adjoint_adaptation[cell] = adjoint_old_adaptation
        end
        exact_graph_bptt && _accumulate_exact_recurrent_cycle!(
            scratch,
            tape,
            model,
            parameters,
            cache,
            branch_for_edge,
            flat,
            cycle,
        )
        if !exact_block_slots &&
           model.workspace_binding === :none && cycle >= 2
            # workspace[:, cycle] was written by route cycle-1 and then read
            # by the apical compartments above.  Propagate that cotangent now,
            # before the reverse loop reaches the cell transition that created
            # the selected block states at time=cycle.
            previous_cycle = cycle - 1
            inverse_workspace_k = inv(Float32(model.workspace_k))
            fill!(scratch.feedback_error, 0.0f0)
            fill!(scratch.route_alpha, 0.0f0)
            for coordinate in 1:model.node_dim
                workspace_value = base.workspace[coordinate, cycle, flat]
                write_signal =
                    scratch.point_scratch.dworkspace_a[coordinate] *
                    (1.0f0 - workspace_value * workspace_value)
                scratch.gradient.workspace_decay_logit[1] +=
                    write_signal *
                    base.workspace[coordinate, cycle - 1, flat] *
                    cache.workspace_decay_derivative
                for block in 1:model.blocks
                    state = base.membrane[
                        coordinate + (block - 1) * model.node_dim,
                        cycle,
                        flat,
                    ]
                    scratch.route_alpha[block] = muladd(
                        write_signal * inverse_workspace_k,
                        state,
                        scratch.route_alpha[block],
                    )
                    if base.block_mask[
                        block, previous_cycle, flat,
                    ] != 0.0f0
                        scratch.feedback_error[coordinate, block] +=
                            write_signal * inverse_workspace_k
                    end
                end
                scratch.point_scratch.dworkspace_a[coordinate] =
                    write_signal * cache.workspace_decay
            end
            if exact_graph_bptt
                for block in 1:model.blocks
                    scratch.route_alpha[block] +=
                        scratch.route_mask_signal[block, previous_cycle]
                end
            end
            _accumulate_pathwise_route_cycle!(
                scratch,
                tape,
                model,
                parameters,
                flat,
                previous_cycle,
                routing_temperature,
                routing_logit_limit,
            )
            for block in 1:model.blocks
                for coordinate in 1:model.node_dim
                    state = base.membrane[
                        coordinate + (block - 1) * model.node_dim,
                        cycle,
                        flat,
                    ]
                    local_signal =
                        scratch.feedback_error[coordinate, block] *
                        (1.0f0 - state * state)
                    cell = _cell_for_coordinate(model, coordinate, block)
                    channel = _channel_for_coordinate(model, coordinate)
                    if channel == 1
                        scratch.soma_signal[cell, previous_cycle] +=
                            local_signal
                    elseif channel == 2
                        scratch.apical_signal[cell, previous_cycle] +=
                            local_signal
                    else
                        scratch.branch_signal[
                            cell, channel - 2, previous_cycle,
                        ] += local_signal
                    end
                end
            end
        end
    end
    exact_graph_bptt && _finish_exact_gate_vjp!(
        scratch,
        model,
        parameters,
    )
    return nothing
end
