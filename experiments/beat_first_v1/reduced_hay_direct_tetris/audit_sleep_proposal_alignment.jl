using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "audit_sleep_replay_identity.jl"))
include(joinpath(@__DIR__, "SleepAlignmentDiagnostics.jl"))

using .SleepAlignmentDiagnostics

const ALIGNMENT_SEED = UInt64(0x534c5050524f504f)

function parse_alignment_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    haskey(values, "checkpoint-dir") ||
        error("--checkpoint-dir is required")
    parse_ints(value) = parse.(Int, split(value, ','))
    parse_reals(value) = parse.(Float64, split(value, ','))
    return (;
        checkpoint_dir=abspath(values["checkpoint-dir"]),
        updates=parse_ints(get(values, "updates", "100,300,1000")),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        trajectories=parse(Int, get(values, "trajectories", "8")),
        noise_seeds=parse(Int, get(values, "noise-seeds", "4")),
        routing_eval_seeds=parse(
            Int,
            get(values, "routing-eval-seeds", "8"),
        ),
        internal_noise_scale=parse(
            Float32,
            get(values, "internal-noise-scale", "2.0"),
        ),
        key_fraction=parse(Float32, get(values, "key-fraction", "0.25")),
        prototypes_per_block=parse(
            Int,
            get(values, "prototypes-per-block", "4"),
        ),
        local_credit_mode=Symbol(get(
            values,
            "local-credit-mode",
            "apical_predictive_online",
        )),
        doses=parse_reals(get(
            values,
            "doses",
            "1e-5,3e-5,1e-4,3e-4,1e-3",
        )),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "sleep_proposal_alignment_gate.json"),
        )),
    )
end

@inline function checkpoint_path(directory, update)
    return joinpath(
        directory,
        "checkpoints",
        "checkpoint_" * lpad(string(update), 9, '0') * ".jld2",
    )
end

function direction_metrics(direction, reference)
    dot_product = 0.0
    direction_square = 0.0
    reference_square = 0.0
    @inbounds for index in eachindex(direction, reference)
        left = Float64(direction[index])
        right = Float64(reference[index])
        dot_product = muladd(left, right, dot_product)
        direction_square = muladd(left, left, direction_square)
        reference_square = muladd(right, right, reference_square)
    end
    denominator = sqrt(direction_square * reference_square)
    return (;
        cosine=denominator > 0.0 ? dot_product / denominator : 0.0,
        dot=dot_product,
        direction_rms=sqrt(direction_square / length(direction)),
        reference_rms=sqrt(reference_square / length(reference)),
    )
end

function combined_direction_metrics(direction, reference)
    dot_product = 0.0
    direction_square = 0.0
    reference_square = 0.0
    count = 0
    for name in (:synapse, :threshold, :route)
        left = getproperty(direction, name)
        right = getproperty(reference, name)
        @inbounds for index in eachindex(left, right)
            left_value = Float64(left[index])
            right_value = Float64(right[index])
            dot_product = muladd(left_value, right_value, dot_product)
            direction_square = muladd(
                left_value,
                left_value,
                direction_square,
            )
            reference_square = muladd(
                right_value,
                right_value,
                reference_square,
            )
            count += 1
        end
    end
    denominator = sqrt(direction_square * reference_square)
    return (;
        cosine=denominator > 0.0 ? dot_product / denominator : 0.0,
        dot=dot_product,
        direction_rms=sqrt(direction_square / max(count, 1)),
        reference_rms=sqrt(reference_square / max(count, 1)),
    )
end

function capture_gradient_direction!(
    trainer,
    payload,
    dataset,
    rows,
    workers,
    credit_mode,
)
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    copyto!(trainer.tape.base.rows, rows)
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode,
    )
    run_with_dendritic_team!(executor) do running
        running.recurrent_signal_scale = 1.0f0
        reduced_hay_v2_arena_gradient!(running)
    end
    loss = float64_statewise_loss(trainer.tape.base)
    return (;
        direction=(;
            synapse=.-copy(trainer.gradient.synapse_weight),
            threshold=.-copy(trainer.gradient.soma_threshold_logits),
            route=.-copy(trainer.gradient.workspace_key),
        ),
        composite=sum(loss.composite),
        teacher_entropy=sum(loss.teacher_entropy),
        excess=sum(loss.excess),
    )
end

function prepare_candidate_gradient!(scratch, trainer, flat, credit_mode)
    # The production worker intentionally reuses a fixed scratch.  Clear all
    # candidate-dependent buffers here so this diagnostic also detects any
    # hidden cross-candidate state instead of inheriting job-scheduling order.
    ReducedHayV2ArenaTraining._fill_parameter_tree!(scratch.gradient)
    for name in fieldnames(typeof(scratch))
        name === :gradient && continue
        value = getfield(scratch, name)
        if value isa AbstractArray
            fill!(value, zero(eltype(value)))
        elseif name === :point_scratch || name === :pack
            for child_name in fieldnames(typeof(value))
                child = getfield(value, child_name)
                child isa AbstractArray &&
                    fill!(child, zero(eltype(child)))
            end
        end
    end
    scratch.active_edge_count = 0
    scratch.local_q_loss = 0.0
    scratch.local_death_loss = 0.0
    scratch.local_quantile_loss = 0.0
    scratch.local_geometry_loss = 0.0
    scratch.apical_predictor_loss = 0.0
    scratch.feedback_alignment_loss = 0.0
    scratch.burst_gate_sum = 0.0
    scratch.burst_gate_count = 0
    apical = credit_mode === :apical_predictive_online
    exact = credit_mode === :exact_bptt
    root_feedback = apical ? trainer.parameters.root_feedback : nothing
    dendritic_prepare_workspace_root_signal_candidate!(
        scratch,
        trainer.tape,
        trainer.model,
        trainer.parameters,
        trainer.cache,
        trainer.branch_for_edge,
        flat,
        0,
        0.5f0,
        root_feedback,
        apical,
        apical,
        trainer.model.route_temperature,
        exact,
    )
    ReducedHayV2ArenaTraining.dendritic_local_candidate!(
        scratch,
        trainer.tape,
        trainer.model,
        trainer.parameters,
        flat,
        false,
    )
    return scratch.gradient
end

"""
Bind wake plasticity to the low-rank block sequence engram that caused it.
Only destination-block parameter slices are retained, so sleep can reactivate
one factor without storing a dataset row or a candidate trajectory.
"""
function capture_clustered_parameter_tags!(
    trainer,
    payload,
    dataset,
    rows,
    workers,
    state_tags,
    credit_mode,
)
    snapshot = capture_gradient_direction!(
        trainer,
        payload,
        dataset,
        rows,
        workers,
        credit_mode,
    )
    model = trainer.model
    ranks = size(state_tags.prototypes, 3)
    synapse = zeros(
        Float32,
        size(trainer.parameters.synapse_weight)...,
        ranks,
    )
    threshold = zeros(
        Float32,
        length(trainer.parameters.soma_threshold_logits),
        ranks,
    )
    route = zeros(
        Float32,
        size(trainer.parameters.workspace_key)...,
        ranks,
    )
    observations = zeros(Int, ranks, model.blocks)
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    base = trainer.tape.base
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        gradient = prepare_candidate_gradient!(
            scratch,
            trainer,
            flat,
            credit_mode,
        )
        for block in 1:model.blocks
            rank = Int(state_tags.prototype_assignment[block, target])
            rank == 0 && continue
            observations[rank, block] += 1
            first_cell = (block - 1) * model.cells_per_block + 1
            last_cell = block * model.cells_per_block
            for cell in first_cell:last_cell
                threshold[cell, rank] -=
                    gradient.soma_threshold_logits[cell]
            end
            for coordinate in 1:model.node_dim
                route[coordinate, block, rank] -=
                    gradient.workspace_key[coordinate, block]
            end
        end
        for source in axes(trainer.parameters.synapse_weight, 1)
            for relation in axes(trainer.parameters.synapse_weight, 2)
                destination = model.destination_for_source[source, relation]
                block = div(destination - 1, model.cells_per_block) + 1
                rank = Int(state_tags.prototype_assignment[block, target])
                rank == 0 && continue
                synapse[source, relation, rank] -=
                    gradient.synapse_weight[source, relation]
            end
        end
    end
    @inbounds for block in 1:model.blocks
        first_cell = (block - 1) * model.cells_per_block + 1
        last_cell = block * model.cells_per_block
        for rank in 1:ranks
            count = observations[rank, block]
            count == 0 && continue
            inverse = inv(Float32(count))
            for cell in first_cell:last_cell
                threshold[cell, rank] *= inverse
            end
            for coordinate in 1:model.node_dim
                route[coordinate, block, rank] *= inverse
            end
        end
    end
    @inbounds for source in axes(trainer.parameters.synapse_weight, 1)
        for relation in axes(trainer.parameters.synapse_weight, 2)
            destination = model.destination_for_source[source, relation]
            block = div(destination - 1, model.cells_per_block) + 1
            for rank in 1:ranks
                count = observations[rank, block]
                count == 0 && continue
                synapse[source, relation, rank] /= Float32(count)
            end
        end
    end
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    return (;
        synapse,
        threshold,
        route,
        observations,
        aggregate=snapshot.direction,
        loss=(;
            composite=snapshot.composite,
            teacher_entropy=snapshot.teacher_entropy,
            excess=snapshot.excess,
        ),
    )
end

function raw_sleep_statistics(trainer, trajectories)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    cells = model.blocks * model.cells_per_block
    edge = zeros(Float32, size(trainer.parameters.synapse_weight))
    route = zeros(Float32, size(trainer.parameters.workspace_key))
    threshold = zeros(
        Float32,
        size(trainer.parameters.soma_threshold_logits),
    )
    block_activity = zeros(Float32, model.blocks)
    block_load = zeros(Float32, model.blocks)
    continuation_sum = 0
    @inbounds for flat in 1:trajectories
        continuation = 0
        alive = true
        for cycle in 1:model.cycles
            any_spike = false
            for block in 1:model.blocks
                block_load[block] += base.block_mask[block, cycle, flat]
                first_cell = (block - 1) * model.cells_per_block + 1
                last_cell = block * model.cells_per_block
                spikes = 0.0f0
                for cell in first_cell:last_cell
                    spike = tape.cell_spikes[cell, cycle, flat]
                    spikes += spike
                    any_spike |= spike != 0.0f0
                    threshold[cell] += abs(
                        ReducedHayV2ArenaTraining._spike_surrogate(
                            tape.soma[cell, cycle + 1, flat],
                            trainer.cache.soma_threshold[cell],
                            model.spike_temperature,
                        ),
                    )
                end
                block_activity[block] +=
                    spikes / Float32(model.cells_per_block)
                for coordinate in 1:model.node_dim
                    node = coordinate + (block - 1) * model.node_dim
                    route[coordinate, block] = muladd(
                        abs(base.route_eligibility[block, cycle, flat]) *
                        abs(tape.state_query[coordinate, cycle, flat]),
                        abs(base.membrane[node, cycle + 1, flat]),
                        route[coordinate, block],
                    )
                end
            end
            if alive && any_spike
                continuation += 1
            else
                alive = false
            end
            cycle > 1 || continue
            for source in 1:cells
                current = tape.cell_spikes[source, cycle - 1, flat]
                previous = cycle <= 2 ? 0.0f0 :
                    tape.cell_spikes[source, cycle - 2, flat]
                (current != 0.0f0 || previous != 0.0f0) || continue
                for relation in 1:model.fanout
                    gate = trainer.cache.gate_hard[source, relation]
                    gate == 0.0f0 &&
                        continue
                    destination =
                        model.destination_for_source[source, relation]
                    delay = trainer.cache.delay[source, relation]
                    pre = muladd(
                        1.0f0 - delay,
                        current,
                        delay * previous,
                    )
                    pre == 0.0f0 && continue
                    branch = Int(trainer.branch_for_edge[source, relation])
                    branch_state = tanh(tape.branch_voltage[
                        destination,
                        branch,
                        cycle + 1,
                        flat,
                    ])
                    branch_sensitivity =
                        (1.0f0 - branch_state * branch_state) *
                        abs(trainer.parameters.soma_coupling[
                            branch,
                            destination,
                        ])
                    post_sensitivity = abs(
                        ReducedHayV2ArenaTraining._spike_surrogate(
                            tape.soma[destination, cycle + 1, flat],
                            trainer.cache.soma_threshold[destination],
                            model.spike_temperature,
                        ),
                    )
                    edge[source, relation] = muladd(
                        abs(gate * pre) *
                        branch_sensitivity *
                        post_sensitivity,
                        1.0f0,
                        edge[source, relation],
                    )
                end
            end
        end
        continuation_sum += continuation
    end
    normalizer = inv(Float32(trajectories * model.cycles))
    edge .*= normalizer
    route .*= normalizer
    threshold .*= normalizer
    block_activity .*= normalizer
    block_load .*= normalizer
    route_entropy = mean(view(
        base.route_normalized_entropy,
        :,
        1:trajectories,
    ))
    route_entropy_finite = isfinite(route_entropy)
    return (;
        edge,
        route,
        threshold,
        block_activity,
        block_load,
        continuation_length=continuation_sum / trajectories,
        route_entropy=route_entropy_finite ? route_entropy : 0.0f0,
        route_entropy_finite,
    )
end

function proposal_from_replay(
    trainer,
    parameter_tags,
    state_tags,
    statistics,
    seed_blocks,
    recalled_ranks,
    ;
    gate_by_replay::Bool=true,
    aggregate_guard=nothing,
    use_slow_aggregate_tag::Bool=false,
    modulation_floor_value::Float32=0.25f0,
)
    model = trainer.model
    ranks = size(parameter_tags.synapse, 3)
    modulation_floor = modulation_floor_value
    0.0f0 <= modulation_floor <= 1.0f0 ||
        error("modulation floor must be in [0, 1]")
    rank_weight = zeros(Float32, ranks, model.blocks)
    @inbounds for trajectory in eachindex(seed_blocks, recalled_ranks)
        block = seed_blocks[trajectory]
        rank = recalled_ranks[trajectory]
        consistency = abs(state_tags.prototype_signed_credit[rank, block]) /
            max(state_tags.prototype_weight[rank, block], 1.0f-8)
        rank_weight[rank, block] += consistency
    end
    maximum_weight = maximum(rank_weight)
    maximum_weight > 0.0f0 && (rank_weight ./= maximum_weight)
    edge_maximum = max(maximum(statistics.edge), 1.0f-8)
    threshold_maximum = max(maximum(statistics.threshold), 1.0f-8)
    route_maximum = max(maximum(statistics.route), 1.0f-8)
    synapse = zeros(Float32, size(trainer.parameters.synapse_weight))
    threshold = zeros(
        Float32,
        size(trainer.parameters.soma_threshold_logits),
    )
    route = zeros(Float32, size(trainer.parameters.workspace_key))
    @inbounds for source in axes(synapse, 1)
        for relation in axes(synapse, 2)
            destination = model.destination_for_source[source, relation]
            block = div(destination - 1, model.cells_per_block) + 1
            activation = clamp(sum(@view rank_weight[:, block]), 0.0f0, 1.0f0)
            block_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) * activation : 1.0f0
            tag = if use_slow_aggregate_tag
                parameter_tags.aggregate.synapse[source, relation]
            else
                sum(
                    rank_weight[rank, block] *
                    parameter_tags.synapse[source, relation, rank]
                    for rank in 1:ranks
                )
            end
            eligibility_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) *
                abs(statistics.edge[source, relation]) / edge_maximum :
                1.0f0
            proposed = tag * block_modulation * eligibility_modulation
            synapse[source, relation] = if aggregate_guard === nothing ||
                                           proposed * aggregate_guard.synapse[
                                               source,
                                               relation,
                                           ] > 0.0f0
                proposed
            else
                0.0f0
            end
        end
    end
    @inbounds for block in 1:model.blocks
        first_cell = (block - 1) * model.cells_per_block + 1
        last_cell = block * model.cells_per_block
        for cell in first_cell:last_cell
            activation = clamp(sum(@view rank_weight[:, block]), 0.0f0, 1.0f0)
            block_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) * activation : 1.0f0
            tag = if use_slow_aggregate_tag
                parameter_tags.aggregate.threshold[cell]
            else
                sum(
                    rank_weight[rank, block] *
                    parameter_tags.threshold[cell, rank]
                    for rank in 1:ranks
                )
            end
            eligibility_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) *
                statistics.threshold[cell] / threshold_maximum : 1.0f0
            proposed = tag * block_modulation * eligibility_modulation
            threshold[cell] = if aggregate_guard === nothing ||
                                 proposed * aggregate_guard.threshold[cell] >
                                 0.0f0
                proposed
            else
                0.0f0
            end
        end
        for coordinate in 1:model.node_dim
            activation = clamp(sum(@view rank_weight[:, block]), 0.0f0, 1.0f0)
            block_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) * activation : 1.0f0
            tag = if use_slow_aggregate_tag
                parameter_tags.aggregate.route[coordinate, block]
            else
                sum(
                    rank_weight[rank, block] *
                    parameter_tags.route[coordinate, block, rank]
                    for rank in 1:ranks
                )
            end
            eligibility_modulation = gate_by_replay ?
                modulation_floor +
                (1.0f0 - modulation_floor) *
                abs(statistics.route[coordinate, block]) / route_maximum :
                1.0f0
            proposed = tag * block_modulation * eligibility_modulation
            route[coordinate, block] = if aggregate_guard === nothing ||
                                          proposed * aggregate_guard.route[
                                              coordinate,
                                              block,
                                          ] > 0.0f0
                proposed
            else
                0.0f0
            end
        end
    end
    return (; synapse, threshold, route)
end

function zero_proposal(trainer)
    return (;
        synapse=zeros(Float32, size(trainer.parameters.synapse_weight)),
        threshold=zeros(
            Float32,
            size(trainer.parameters.soma_threshold_logits),
        ),
        route=zeros(Float32, size(trainer.parameters.workspace_key)),
    )
end

function add_proposal!(destination, source)
    destination.synapse .+= source.synapse
    destination.threshold .+= source.threshold
    destination.route .+= source.route
    return destination
end

function scale_proposal!(proposal, scale)
    proposal.synapse .*= scale
    proposal.threshold .*= scale
    proposal.route .*= scale
    return proposal
end

function proposal_maximum(proposal, groups)
    maximum_value = 0.0f0
    :recurrent in groups && begin
        maximum_value = max(maximum_value, maximum(abs, proposal.synapse))
        maximum_value = max(maximum_value, maximum(abs, proposal.threshold))
    end
    :route in groups &&
        (maximum_value = max(maximum_value, maximum(abs, proposal.route)))
    return maximum_value
end

function evaluate_proposal_loss!(
    trainer,
    payload,
    rows,
    dataset,
    workers,
    proposal,
    signed_dose,
    groups,
    ;
    routing_seeds::Int=0,
)
    maximum_value = proposal_maximum(proposal, groups)
    maximum_value > 0.0f0 || return nothing
    scale = Float32(signed_dose / maximum_value)
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    :recurrent in groups && begin
        trainer.parameters.synapse_weight .+= scale .* proposal.synapse
        trainer.parameters.soma_threshold_logits .+=
            scale .* proposal.threshold
    end
    :route in groups &&
        (trainer.parameters.workspace_key .+= scale .* proposal.route)
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    losses = Float64[]
    seeds = routing_seeds == 0 ? (0,) : (1:routing_seeds)
    for seed_index in seeds
        copyto!(trainer.tape.base.rows, rows)
        executor = ReducedHayV2ArenaExecutor(
            trainer,
            dataset;
            active_workers=workers,
            cpuset_mode=:none,
            stochastic_routing=routing_seeds != 0,
            routing_seed=ALIGNMENT_SEED ⊻
                UInt64(seed_index) * UInt64(0xd6e8feb86659fd93),
            credit_mode=:block_teacher,
        )
        run_with_dendritic_team!(executor) do running
            reduced_hay_v2_arena_forward!(running)
        end
        loss = float64_statewise_loss(trainer.tape.base)
        push!(losses, sum(loss.excess))
    end
    return (;
        excess=mean(losses),
        excess_standard_error=length(losses) > 1 ?
            std(losses) / sqrt(length(losses)) : 0.0,
        scale,
        maximum_parameter_delta=abs(signed_dose),
        routing_seeds,
    )
end

function finite_difference_table!(
    trainer,
    payload,
    rows,
    dataset,
    workers,
    proposal,
    doses,
    groups,
    ;
    routing_seeds::Int=0,
)
    result = Any[]
    baseline = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        workers,
        proposal,
        0.0,
        groups;
        routing_seeds,
    )
    for dose in doses
        positive = evaluate_proposal_loss!(
            trainer,
            payload,
            rows,
            dataset,
            workers,
            proposal,
            dose,
            groups;
            routing_seeds,
        )
        negative = evaluate_proposal_loss!(
            trainer,
            payload,
            rows,
            dataset,
            workers,
            proposal,
            -dose,
            groups;
            routing_seeds,
        )
        if baseline === nothing || positive === nothing || negative === nothing
            push!(result, (;
                dose,
                available=false,
                baseline_excess=nothing,
                positive_excess=nothing,
                negative_excess=nothing,
                positive_improvement=0.0,
                negative_improvement=0.0,
                signed_benefit=0.0,
            ))
        else
            push!(result, (;
                dose,
                available=true,
                baseline_excess=baseline.excess,
                positive_excess=positive.excess,
                negative_excess=negative.excess,
                positive_improvement=
                    baseline.excess - positive.excess,
                negative_improvement=
                    baseline.excess - negative.excess,
                signed_benefit=negative.excess - positive.excess,
                routing_seeds,
            ))
        end
    end
    return result
end

function evaluate_checkpoint(options, checkpoint)
    payload = load_reduced_hay_v2_checkpoint(checkpoint)
    rows = Int.(payload.run_config.overfit_rows)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model = build_reduced_hay_model(Symbol(payload.run_config.preset))
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    trainer = trainer_from_checkpoint(model, parameters, payload)
    length(rows) == trainer.tape.base.state_batch ||
        error("alignment panel must fill one arena batch")

    state_tags = capture_wake_state_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        options.prototypes_per_block;
        credit_mode=options.local_credit_mode,
    )
    local_tags = capture_clustered_parameter_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        state_tags,
        options.local_credit_mode,
    )
    exact_tags = capture_clustered_parameter_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        state_tags,
        :exact_bptt,
    )
    slow_direction = (;
        synapse=.-copy(payload.optimizer_first_moment.synapse_weight),
        threshold=.-copy(
            payload.optimizer_first_moment.soma_threshold_logits,
        ),
        route=.-copy(payload.optimizer_first_moment.workspace_key),
    )
    slow_tags = merge(local_tags, (; aggregate=slow_direction))
    direct = (;
        recurrent=combined_direction_metrics(
            (;
                synapse=local_tags.aggregate.synapse,
                threshold=local_tags.aggregate.threshold,
                route=zeros(Float32, size(local_tags.aggregate.route)),
            ),
            (;
                synapse=exact_tags.aggregate.synapse,
                threshold=exact_tags.aggregate.threshold,
                route=zeros(Float32, size(exact_tags.aggregate.route)),
            ),
        ),
        route=direction_metrics(
            local_tags.aggregate.route,
            exact_tags.aggregate.route,
        ),
        joint=combined_direction_metrics(
            local_tags.aggregate,
            exact_tags.aggregate,
        ),
        slow_recurrent=combined_direction_metrics(
            (;
                synapse=slow_direction.synapse,
                threshold=slow_direction.threshold,
                route=zeros(Float32, size(slow_direction.route)),
            ),
            (;
                synapse=exact_tags.aggregate.synapse,
                threshold=exact_tags.aggregate.threshold,
                route=zeros(Float32, size(exact_tags.aggregate.route)),
            ),
        ),
        slow_route=direction_metrics(
            slow_direction.route,
            exact_tags.aggregate.route,
        ),
        slow_joint=combined_direction_metrics(
            slow_direction,
            exact_tags.aggregate,
        ),
    )

    priority_pairs = [
        (
            weight=state_tags.prototype_weight[rank, block],
            block,
            rank,
        )
        for block in 1:model.blocks
        for rank in 1:options.prototypes_per_block
        if state_tags.prototype_weight[rank, block] > 0.0f0
    ]
    sort!(priority_pairs; by=pair -> pair.weight, rev=true)
    length(priority_pairs) >= 2 ||
        error(
            "wake checkpoint has fewer than two nonzero task-tagged engrams",
        )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    audit = SleepAudit()
    local_sum = zero_proposal(trainer)
    oracle_sum = zero_proposal(trainer)
    shuffled_sum = zero_proposal(trainer)
    selected_reference_sum = zero_proposal(trainer)
    seed_results = Any[]
    for seed_index in 1:options.noise_seeds
        noise_seed = ALIGNMENT_SEED ⊻
            UInt64(seed_index) * UInt64(0x9e3779b97f4a7c15)
        # Priority is combined with deterministic novelty rotation: each
        # seed consumes the next tranche instead of replaying the same
        # strongest attractors forever.
        offset = (seed_index - 1) * options.trajectories
        selected = [
            priority_pairs[mod1(offset + trajectory, length(priority_pairs))]
            for trajectory in 1:options.trajectories
        ]
        seed_blocks = [pair.block for pair in selected]
        seed_ranks = [pair.rank for pair in selected]
        seed_keys = build_seed_state_keys(
            state_tags,
            seed_blocks,
            seed_ranks,
            1,
        )
        shuffled_blocks = [
            mod1(block + 17, model.blocks) for block in seed_blocks
        ]
        shuffled_ranks = [
            mod1(rank + 1, options.prototypes_per_block)
            for rank in seed_ranks
        ]
        shuffled_keys = build_seed_state_keys(
            state_tags,
            shuffled_blocks,
            shuffled_ranks,
            1,
        )
        tagged_keys, recalled_ranks = complete_seed_state_keys(
            trainer,
            state_tags,
            seed_blocks,
            seed_keys,
            noise_seed,
            options.key_fraction,
        )
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            seed_blocks,
            seed_state_keys=tagged_keys,
            key_fraction=1.0f0,
            key_gain=1.0f0,
        )
        statistics = raw_sleep_statistics(trainer, options.trajectories)
        local_proposal = proposal_from_replay(
            trainer,
            local_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks,
            aggregate_guard=local_tags.aggregate,
            use_slow_aggregate_tag=true,
        )
        oracle_proposal = proposal_from_replay(
            trainer,
            exact_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks,
            aggregate_guard=exact_tags.aggregate,
            use_slow_aggregate_tag=true,
        )
        selected_reference = proposal_from_replay(
            trainer,
            exact_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks;
            gate_by_replay=false,
            aggregate_guard=exact_tags.aggregate,
            use_slow_aggregate_tag=true,
        )
        binding_reference = proposal_from_replay(
            trainer,
            exact_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks;
            gate_by_replay=true,
            aggregate_guard=nothing,
            use_slow_aggregate_tag=false,
            modulation_floor_value=0.0f0,
        )
        binding_tagged = proposal_from_replay(
            trainer,
            local_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks;
            gate_by_replay=true,
            aggregate_guard=nothing,
            use_slow_aggregate_tag=false,
            modulation_floor_value=0.0f0,
        )

        wrong_keys, wrong_ranks = complete_seed_state_keys(
            trainer,
            state_tags,
            seed_blocks,
            shuffled_keys,
            noise_seed,
            options.key_fraction,
        )
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            seed_blocks,
            seed_state_keys=wrong_keys,
            key_fraction=1.0f0,
            key_gain=1.0f0,
        )
        wrong_statistics = raw_sleep_statistics(
            trainer,
            options.trajectories,
        )
        shuffled_proposal = proposal_from_replay(
            trainer,
            local_tags,
            state_tags,
            wrong_statistics,
            seed_blocks,
            wrong_ranks,
            aggregate_guard=local_tags.aggregate,
            use_slow_aggregate_tag=true,
        )
        binding_shuffled = proposal_from_replay(
            trainer,
            local_tags,
            state_tags,
            wrong_statistics,
            seed_blocks,
            wrong_ranks;
            gate_by_replay=true,
            aggregate_guard=nothing,
            use_slow_aggregate_tag=false,
            modulation_floor_value=0.0f0,
        )
        local_metrics = combined_direction_metrics(
            local_proposal,
            selected_reference,
        )
        oracle_metrics = combined_direction_metrics(
            oracle_proposal,
            selected_reference,
        )
        shuffled_metrics = combined_direction_metrics(
            shuffled_proposal,
            selected_reference,
        )
        binding_tagged_metrics = combined_direction_metrics(
            binding_tagged,
            binding_reference,
        )
        binding_shuffled_metrics = combined_direction_metrics(
            binding_shuffled,
            binding_reference,
        )
        push!(seed_results, (;
            seed_index,
            selected_blocks=seed_blocks,
            selected_ranks=seed_ranks,
            deployable=local_metrics,
            oracle=oracle_metrics,
            shuffled=shuffled_metrics,
            binding_tagged=binding_tagged_metrics,
            binding_shuffled=binding_shuffled_metrics,
            continuation_length=statistics.continuation_length,
            route_entropy=statistics.route_entropy,
            engram_recall=mean(recalled_ranks .== seed_ranks),
            shuffled_engram_recall=mean(wrong_ranks .== seed_ranks),
        ))
        add_proposal!(local_sum, local_proposal)
        add_proposal!(oracle_sum, oracle_proposal)
        add_proposal!(shuffled_sum, shuffled_proposal)
        add_proposal!(selected_reference_sum, selected_reference)
    end
    inverse_seeds = inv(Float32(options.noise_seeds))
    scale_proposal!(local_sum, inverse_seeds)
    scale_proposal!(oracle_sum, inverse_seeds)
    scale_proposal!(shuffled_sum, inverse_seeds)
    scale_proposal!(selected_reference_sum, inverse_seeds)
    aggregate = (;
        deployable=(;
            recurrent=combined_direction_metrics(
                (;
                    synapse=local_sum.synapse,
                    threshold=local_sum.threshold,
                    route=zeros(Float32, size(local_sum.route)),
                ),
                (;
                    synapse=selected_reference_sum.synapse,
                    threshold=selected_reference_sum.threshold,
                    route=zeros(Float32, size(selected_reference_sum.route)),
                ),
            ),
            route=direction_metrics(
                local_sum.route,
                selected_reference_sum.route,
            ),
            joint=combined_direction_metrics(
                local_sum,
                selected_reference_sum,
            ),
        ),
        global_wake=combined_direction_metrics(
            local_sum,
            exact_tags.aggregate,
        ),
        oracle=combined_direction_metrics(
            oracle_sum,
            selected_reference_sum,
        ),
        shuffled=combined_direction_metrics(
            shuffled_sum,
            selected_reference_sum,
        ),
        reversed=combined_direction_metrics(
            (;
                synapse=.-local_sum.synapse,
                threshold=.-local_sum.threshold,
                route=.-local_sum.route,
            ),
            selected_reference_sum,
        ),
    )
    finite_difference = (;
        exact_recurrent=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            exact_tags.aggregate,
            options.doses,
            (:recurrent,),
        ),
        exact_route_deterministic=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            exact_tags.aggregate,
            options.doses,
            (:route,),
        ),
        exact_route=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            exact_tags.aggregate,
            options.doses,
            (:route,);
            routing_seeds=options.routing_eval_seeds,
        ),
        exact_joint=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            exact_tags.aggregate,
            options.doses,
            (:recurrent, :route),
        ),
        local_aggregate_joint=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            local_tags.aggregate,
            options.doses,
            (:recurrent, :route),
        ),
        recurrent=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            local_sum,
            options.doses,
            (:recurrent,),
        ),
        route_deterministic=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            local_sum,
            options.doses,
            (:route,),
        ),
        route=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            local_sum,
            options.doses,
            (:route,);
            routing_seeds=options.routing_eval_seeds,
        ),
        joint=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            local_sum,
            options.doses,
            (:recurrent, :route),
        ),
    )
    positive_seed_count = count(
        result -> result.deployable.dot > 0.0,
        seed_results,
    )
    shuffled_mean = mean(result.shuffled.cosine for result in seed_results)
    binding_tagged_mean = mean(
        result.binding_tagged.cosine for result in seed_results
    )
    binding_shuffled_mean = mean(
        result.binding_shuffled.cosine for result in seed_results
    )
    engram_recall_mean = mean(
        result.engram_recall for result in seed_results
    )
    shuffled_engram_recall_mean = mean(
        result.shuffled_engram_recall for result in seed_results
    )
    local_mean = mean(
        result.deployable.cosine for result in seed_results
    )
    practical_improvement_floor = max(
        1.0e-4,
        0.005 * abs(exact_tags.loss.excess),
    )
    joint_positive_doses = count(
        row -> row.available &&
            row.positive_improvement >= practical_improvement_floor,
        finite_difference.joint,
    )
    recurrent_positive_doses = count(
        row -> row.available &&
            row.positive_improvement >= practical_improvement_floor,
        finite_difference.recurrent,
    )
    route_positive_doses = count(
        row -> row.available &&
            row.positive_improvement >= practical_improvement_floor,
        finite_difference.route_deterministic,
    )
    deterministic_baseline = first(
        finite_difference.exact_recurrent,
    ).baseline_excess
    baseline_consistent = deterministic_baseline !== nothing &&
        abs(deterministic_baseline - exact_tags.loss.excess) <= 1.0e-6
    pass =
        baseline_consistent &&
        aggregate.oracle.dot > 0.0 &&
        aggregate.deployable.recurrent.dot > 0.0 &&
        aggregate.deployable.route.dot > 0.0 &&
        aggregate.deployable.joint.dot > 0.0 &&
        aggregate.reversed.dot < 0.0 &&
        positive_seed_count == options.noise_seeds &&
        binding_tagged_mean > binding_shuffled_mean + 0.01 &&
        engram_recall_mean >= 0.75 &&
        shuffled_engram_recall_mean <= engram_recall_mean - 0.25 &&
        recurrent_positive_doses >= 2 &&
        route_positive_doses >= 1 &&
        joint_positive_doses >= 2 &&
        audit.nonzero_rail_observations == 0
    return (;
        checkpoint,
        update=Int(payload.update),
        baseline=exact_tags.loss,
        direct,
        aggregate,
        seed_results,
        finite_difference,
        audit=(;
            zero_rail_checks=audit.zero_rail_checks,
            nonzero_rail_observations=audit.nonzero_rail_observations,
            dataset_reads_during_sleep=0,
            teacher_target_reads_during_sleep=0,
        ),
        positive_seed_count,
        local_mean_cosine=local_mean,
        shuffled_mean_cosine=shuffled_mean,
        binding_tagged_mean_cosine=binding_tagged_mean,
        binding_shuffled_mean_cosine=binding_shuffled_mean,
        engram_recall_mean,
        shuffled_engram_recall_mean,
        practical_improvement_floor,
        baseline_consistent,
        joint_positive_doses,
        recurrent_positive_doses,
        route_positive_doses,
        pass,
    )
end

function collect_nonfinite!(paths, value, path="root")
    if value isa AbstractFloat
        isfinite(value) || push!(paths, path * "=" * string(value))
    elseif value isa NamedTuple
        for name in keys(value)
            collect_nonfinite!(
                paths,
                getproperty(value, name),
                path * "." * String(name),
            )
        end
    elseif value isa AbstractArray
        for index in eachindex(value)
            collect_nonfinite!(
                paths,
                value[index],
                path * "[" * string(index) * "]",
            )
        end
    end
    return paths
end

function main(arguments=ARGS)
    options = parse_alignment_options(arguments)
    options.local_credit_mode in (
        :apical_predictive_online,
        :workspace_root_control,
        :reciprocal_apical,
        :adaptive_apical,
    ) || error(
        "unsupported local-credit-mode: $(options.local_credit_mode)",
    )
    options.noise_seeds >= 3 || error("noise-seeds must be at least three")
    options.trajectories > 0 || error("trajectories must be positive")
    options.prototypes_per_block > 1 ||
        error("prototypes-per-block must exceed one")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    all(>(0.0), options.doses) || error("doses must be positive")
    BLAS.set_num_threads(1)
    results = Any[]
    for update in options.updates
        checkpoint = checkpoint_path(options.checkpoint_dir, update)
        isfile(checkpoint) || error("checkpoint absent: $checkpoint")
        println("alignment checkpoint update=$update")
        push!(results, evaluate_checkpoint(options, checkpoint))
        println(
            "update=$update pass=$(results[end].pass) " *
            "direct=$(round(results[end].direct.joint.cosine; digits=6)) " *
            "sleep=$(round(results[end].aggregate.deployable.joint.cosine; digits=6)) " *
            "oracle=$(round(results[end].aggregate.oracle.cosine; digits=6)) " *
            "shuffle=$(round(results[end].aggregate.shuffled.cosine; digits=6))",
        )
        flush(stdout)
    end
    pass = all(result.pass for result in results)
    output = (;
        schema="reduced-hay-v2-sleep-proposal-alignment-gate-v1",
        options=(;
            checkpoint_dir=options.checkpoint_dir,
            updates=options.updates,
            trajectories=options.trajectories,
            noise_seeds=options.noise_seeds,
            routing_eval_seeds=options.routing_eval_seeds,
            internal_noise_scale=options.internal_noise_scale,
            key_fraction=options.key_fraction,
            prototypes_per_block=options.prototypes_per_block,
            local_credit_mode=String(options.local_credit_mode),
            doses=options.doses,
        ),
        semantics=(;
            exact_gradient="evaluation oracle only",
            local_tag=String(options.local_credit_mode) *
                " wake plasticity",
            sleep_proposal="unapplied until wake-loss audit",
            reward="supervised reward surrogate, never environment return",
        ),
        results,
        pass,
    )
    nonfinite_paths = collect_nonfinite!(String[], output)
    isempty(nonfinite_paths) || error(
        "non-finite alignment diagnostics: " *
        join(first(nonfinite_paths, min(length(nonfinite_paths), 16)), ", "),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println("pass=$pass output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
