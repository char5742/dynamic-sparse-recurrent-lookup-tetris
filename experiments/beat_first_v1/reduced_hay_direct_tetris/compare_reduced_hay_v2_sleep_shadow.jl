using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const SLEEP_ARMS = (
    :wake_only,
    :recurrent_only,
    :route_only,
    :alternating,
    :simultaneous,
)
const SLEEP_SEED = UInt64(0x534c454550563253)

function parse_boolean(value)
    normalized = lowercase(strip(value))
    normalized in ("true", "1", "yes") && return true
    normalized in ("false", "0", "no") && return false
    error("invalid boolean value $value")
end

function parse_options(arguments)
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
    haskey(values, "checkpoint") || error("--checkpoint is required")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        microcycles=parse(Int, get(values, "microcycles", "8")),
        trajectories=parse(Int, get(values, "trajectories", "8")),
        internal_noise_scale=parse(
            Float32,
            get(values, "internal-noise-scale", "1.25"),
        ),
        recurrent_rate=parse(
            Float32,
            get(values, "recurrent-rate", "0.00001"),
        ),
        route_rate=parse(
            Float32,
            get(values, "route-rate", "0.000002"),
        ),
        downscale_rate=parse(
            Float32,
            get(values, "downscale-rate", "0.000001"),
        ),
        homeostasis_rate=parse(
            Float32,
            get(values, "homeostasis-rate", "0.000001"),
        ),
        counterfactual=parse_boolean(get(
            values,
            "counterfactual",
            "false",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "sleep_shadow_comparison.json"),
        )),
    )
end

mutable struct SleepAudit
    zero_rail_checks::Int
    nonzero_rail_observations::Int
    dataset_reads::Int
    teacher_target_reads::Int
end

SleepAudit() = SleepAudit(0, 0, 0, 0)

struct WakePlasticityTags
    synapse_direction::Matrix{Float32}
    threshold_direction::Vector{Float32}
    workspace_key_direction::Matrix{Float32}
    raw_synapse_direction::Matrix{Float32}
    raw_threshold_direction::Vector{Float32}
    raw_workspace_key_direction::Matrix{Float32}
    block_utility::Vector{Float32}
    wake_firing_rate::Float64
    wake_workspace_transition::Float64
    route_margin::Matrix{Float32}
    route_selected::Matrix{Int16}
    route_challenger::Matrix{Int16}
    route_selected_key_factor::Array{Float32,3}
    route_challenger_key_factor::Array{Float32,3}
    synapse_margin_reference_step::Matrix{Float32}
    threshold_margin_reference_step::Matrix{Float32}
    margin_reference_scale::Float32
    raw_synapse_tag_rms::Float32
    raw_threshold_tag_rms::Float32
    raw_route_tag_rms::Float32
    raw_block_utility_maximum::Float32
end

mutable struct SleepReplayState
    reference_membrane::Array{Float32,4}
    previous_prediction_error::Vector{Float32}
    replay_count::Vector{Float32}
    teacher_synapse::Matrix{Float32}
    teacher_workspace_key::Matrix{Float32}
end

function rms_normalize!(
    values;
    epsilon::Float32=1.0f-8,
    limit::Float32=4.0f0,
)
    square_sum = 0.0
    @inbounds for value in values
        square_sum = muladd(Float64(value), Float64(value), square_sum)
    end
    inverse = inv(Float32(sqrt(square_sum / length(values)) + epsilon))
    @inbounds for index in eachindex(values)
        values[index] = clamp(values[index] * inverse, -limit, limit)
    end
    return values
end

function copy_parameter!(destination, source)
    @inbounds for name in keys(destination)
        copyto!(getproperty(destination, name), getproperty(source, name))
    end
    return destination
end

@inline function checkpoint_config_value(config, name::Symbol, default)
    return hasproperty(config, name) ? getproperty(config, name) : default
end

"""Rebuild the trainer with the hyperparameters recorded by its checkpoint."""
function trainer_from_checkpoint(model, parameters, payload)
    config = payload.run_config
    return ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch=Int(payload.arena_signature.state_batch),
        width=Int(payload.arena_signature.width),
        learning_rate=checkpoint_config_value(
            config,
            :learning_rate,
            5.0e-4,
        ),
        weight_decay=checkpoint_config_value(
            config,
            :weight_decay,
            1.0e-5,
        ),
        structural_interval=Int(checkpoint_config_value(
            config,
            :structural_interval,
            25,
        )),
        branch_interval=Int(checkpoint_config_value(
            config,
            :branch_interval,
            128,
        )),
        recurrent_learning_rate_multiplier=checkpoint_config_value(
            config,
            :recurrent_learning_rate_multiplier,
            0.001,
        ),
        routing_entropy_weight=checkpoint_config_value(
            config,
            :routing_entropy_weight,
            4.0,
        ),
        routing_entropy_floor=checkpoint_config_value(
            config,
            :routing_entropy_floor,
            0.85,
        ),
        routing_load_weight=checkpoint_config_value(
            config,
            :routing_load_weight,
            0.10,
        ),
    )
end

function sanitize_sleep_arena!(trainer, trajectories)
    base = trainer.tape.base
    trajectories <= base.capacity || error("sleep trajectories exceed arena")
    fill!(base.rows, 0)
    fill!(base.counts, 0)
    fill!(base.valid_flats, 0)
    base.valid_count = trajectories
    @inbounds for flat in 1:trajectories
        base.valid_flats[flat] = Int32(flat)
    end
    fill!(base.rails, 0.0f0)
    for field in fieldnames(typeof(base.targets))
        fill!(getfield(base.targets, field), 0)
    end
    return nothing
end

function wake_metrics_and_features!(trainer, dataset, rows, workers)
    copyto!(trainer.tape.base.rows, rows)
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    run_with_dendritic_team!(executor) do running
        reduced_hay_v2_arena_forward!(running)
    end
    base = trainer.tape.base
    gate_density = sum(trainer.cache.gate_probability) /
        Float32(length(trainer.cache.gate_probability))
    loss = ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
        base,
        trainer.loss_scratch,
        gate_density,
        0.0f0,
    )
    matches = 0
    ndcg = 0.0
    pairwise = 0.0
    @inbounds for slot in 1:base.state_batch
        count = Int(base.counts[slot])
        offset = (slot - 1) * base.width
        prediction = @view base.raw[1, (offset + 1):(offset + count)]
        teacher = @view base.targets.teacher_q[1:count, slot]
        matches += argmax(prediction) == argmax(teacher)
        ndcg += BeatFirstTrainingCore._ndcg(prediction, teacher)
        pairwise += BeatFirstTrainingCore._pairwise_accuracy(
            prediction,
            teacher,
        )
    end
    features = zeros(Float32, 2 * trainer.model.node_dim, base.valid_count)
    route_masks = zeros(
        UInt8,
        trainer.model.blocks,
        trainer.model.cycles,
        base.valid_count,
    )
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for cycle in 1:trainer.model.cycles
            for block in 1:trainer.model.blocks
                route_masks[block, cycle, target] = UInt8(
                    base.block_mask[block, cycle, flat] != 0.0f0,
                )
            end
        end
        for coordinate in 1:trainer.model.node_dim
            features[coordinate, target] =
                base.workspace[
                    coordinate,
                    trainer.model.cycles + 1,
                    flat,
                ] * base.workspace_inv_rms[flat]
            pooled = 0.0f0
            for block in 1:trainer.model.blocks
                node = coordinate +
                    (block - 1) * trainer.model.node_dim
                pooled = muladd(
                    base.membrane[
                        node,
                        trainer.model.cycles + 1,
                        flat,
                    ],
                    base.block_mask[
                        block,
                        trainer.model.cycles,
                        flat,
                    ],
                    pooled,
                )
            end
            pooled /= Float32(trainer.model.workspace_k)
            features[trainer.model.node_dim + coordinate, target] =
                pooled * base.selected_pool_inv_rms[flat]
        end
    end
    inverse_states = inv(Float64(base.state_batch))
    return (;
        metrics=(;
            excess_loss=Float64(loss.composite_loss - loss.teacher_entropy),
            composite_loss=Float64(loss.composite_loss),
            top1=matches * inverse_states,
            ndcg=ndcg * inverse_states,
            pairwise=pairwise * inverse_states,
            firing_rate=trainer.metrics.firing_rate,
            routing_entropy=trainer.metrics.routing_entropy,
        ),
        features,
        route_masks,
    )
end

function capture_route_margin_basis(trainer)
    model = trainer.model
    base = trainer.tape.base
    targets = base.valid_count
    margin = zeros(Float32, model.cycles, targets)
    selected = zeros(Int16, model.cycles, targets)
    challenger = zeros(Int16, model.cycles, targets)
    selected_key_factor = zeros(
        Float32,
        model.node_dim,
        model.cycles,
        targets,
    )
    challenger_key_factor = similar(selected_key_factor)
    @inbounds for target in 1:targets
        flat = Int(base.valid_flats[target])
        for cycle in 1:model.cycles
            selected_block = 0
            selected_score = Inf32
            challenger_block = 0
            challenger_score = -Inf32
            for block in 1:model.blocks
                score = base.route_score[block, cycle, flat]
                if base.block_mask[block, cycle, flat] != 0.0f0
                    if score < selected_score
                        selected_score = score
                        selected_block = block
                    end
                elseif score > challenger_score
                    challenger_score = score
                    challenger_block = block
                end
            end
            selected_block > 0 || error("wake route has no selected block")
            challenger_block > 0 || error("wake route has no challenger")
            selected[cycle, target] = Int16(selected_block)
            challenger[cycle, target] = Int16(challenger_block)
            margin[cycle, target] = selected_score - challenger_score
            selected_offset = (selected_block - 1) * model.node_dim
            challenger_offset = (challenger_block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                query = trainer.tape.state_query[
                    coordinate,
                    cycle,
                    flat,
                ]
                selected_key_factor[coordinate, cycle, target] =
                    query * base.membrane[
                        selected_offset + coordinate,
                        cycle + 1,
                        flat,
                    ]
                challenger_key_factor[coordinate, cycle, target] =
                    query * base.membrane[
                        challenger_offset + coordinate,
                        cycle + 1,
                        flat,
                    ]
            end
        end
    end
    return (;
        margin,
        selected,
        challenger,
        selected_key_factor,
        challenger_key_factor,
    )
end

function capture_anchor_pair_margins(trainer, anchors)
    model = trainer.model
    base = trainer.tape.base
    size(anchors.margin, 2) == base.valid_count ||
        error("wake route anchor count changed")
    result = similar(anchors.margin)
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for cycle in 1:model.cycles
            selected = Int(anchors.selected[cycle, target])
            challenger = Int(anchors.challenger[cycle, target])
            result[cycle, target] =
                base.route_score[selected, cycle, flat] -
                base.route_score[challenger, cycle, flat]
        end
    end
    return result
end

function capture_wake_tags!(trainer, payload, rows, dataset, workers)
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    wake_metrics_and_features!(trainer, dataset, rows, workers)
    route_anchors = capture_route_margin_basis(trainer)
    copyto!(trainer.tape.base.rows, rows)
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    run_with_dendritic_team!(executor) do running
        running.recurrent_signal_scale = 1.0f0
        reduced_hay_v2_arena_gradient!(running)
    end
    # Keep the task tag upstream of Adam, family clipping, weight decay, and
    # the recurrent learning-rate multiplier.  `trainer.gradient` is dL/dp;
    # sleep consumes a signed descent direction.
    synapse_direction = .-copy(trainer.gradient.synapse_weight)
    threshold_direction =
        .-copy(trainer.gradient.soma_threshold_logits)
    workspace_key_direction = .-copy(trainer.gradient.workspace_key)
    raw_synapse_direction = copy(synapse_direction)
    raw_threshold_direction = copy(threshold_direction)
    raw_workspace_key_direction = copy(workspace_key_direction)
    raw_synapse_tag_rms = Float32(sqrt(mean(abs2, synapse_direction)))
    raw_threshold_tag_rms = Float32(sqrt(mean(abs2, threshold_direction)))
    raw_route_tag_rms = Float32(sqrt(mean(abs2, workspace_key_direction)))
    rms_normalize!(synapse_direction)
    rms_normalize!(threshold_direction)
    rms_normalize!(workspace_key_direction)
    blocks = trainer.model.blocks
    block_utility = zeros(Float32, blocks)
    base = trainer.tape.base
    @inbounds for block in 1:blocks
        square_sum = 0.0f0
        count = 0
        for flat_index in 1:base.valid_count
            flat = Int(base.valid_flats[flat_index])
            for cycle in 1:trainer.model.cycles
                value = trainer.tape.block_supervised_reward[
                    block,
                    cycle,
                    flat,
                ]
                square_sum = muladd(value, value, square_sum)
                count += 1
            end
        end
        key_square = sum(abs2, @view workspace_key_direction[:, block]) /
            Float32(trainer.model.node_dim)
        block_utility[block] = sqrt(square_sum / max(count, 1) + key_square)
    end
    maximum_utility = maximum(block_utility)
    raw_block_utility_maximum = maximum_utility
    maximum_utility > 0.0f0 && (block_utility ./= maximum_utility)
    transition = 0.0
    transition_count = 0
    @inbounds for flat_index in 1:base.valid_count
        flat = Int(base.valid_flats[flat_index])
        for cycle in 1:trainer.model.cycles
            for coordinate in 1:trainer.model.node_dim
                difference =
                    base.workspace[coordinate, cycle + 1, flat] -
                    base.workspace[coordinate, cycle, flat]
                transition += difference * difference
                transition_count += 1
            end
        end
    end
    wake_firing_rate = trainer.metrics.firing_rate
    wake_workspace_transition =
        sqrt(transition / max(transition_count, 1))

    # Wake stores only a local first-order route-boundary tag.  These two
    # finite differences are generated before sleep and contain neither a
    # dataset row nor a teacher target.  Sleep can therefore bound the effect
    # of a proposed recurrent step without reopening the external dataset.
    margin_reference_scale = 1.0f-4
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    trainer.parameters.synapse_weight .+=
        margin_reference_scale .* synapse_direction
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    wake_metrics_and_features!(trainer, dataset, rows, workers)
    synapse_margin_reference_step =
        capture_anchor_pair_margins(trainer, route_anchors) .-
        route_anchors.margin

    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    trainer.parameters.soma_threshold_logits .+=
        margin_reference_scale .* threshold_direction
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    wake_metrics_and_features!(trainer, dataset, rows, workers)
    threshold_margin_reference_step =
        capture_anchor_pair_margins(trainer, route_anchors) .-
        route_anchors.margin

    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    return WakePlasticityTags(
        synapse_direction,
        threshold_direction,
        workspace_key_direction,
        raw_synapse_direction,
        raw_threshold_direction,
        raw_workspace_key_direction,
        block_utility,
        wake_firing_rate,
        wake_workspace_transition,
        route_anchors.margin,
        route_anchors.selected,
        route_anchors.challenger,
        route_anchors.selected_key_factor,
        route_anchors.challenger_key_factor,
        synapse_margin_reference_step,
        threshold_margin_reference_step,
        margin_reference_scale,
        raw_synapse_tag_rms,
        raw_threshold_tag_rms,
        raw_route_tag_rms,
        raw_block_utility_maximum,
    )
end

function run_internal_sleep_trajectories!(
    trainer,
    scratch,
    trajectories,
    noise_seed,
    noise_scale,
    audit;
    silenced_block::Int=0,
    seed_blocks=nothing,
    seed_state_keys=nothing,
    key_fraction::Float32=0.0f0,
    key_gain::Float32=0.0f0,
)
    sanitize_sleep_arena!(trainer, trajectories)
    base = trainer.tape.base
    @inbounds for trajectory in 1:trajectories
        seed_block = seed_blocks === nothing ?
            mod1(trajectory, trainer.model.blocks) :
            Int(seed_blocks[trajectory])
        1 <= seed_block <= trainer.model.blocks ||
            error("sleep seed block is outside the model")
        state_key = seed_state_keys === nothing ? nothing :
            view(seed_state_keys, :, trajectory)
        dendritic_forward_candidate!(
            trainer.tape,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            scratch,
            trainer.branch_for_edge,
            trajectory;
            stochastic_routing=true,
            routing_nonce=noise_seed ⊻ UInt64(trajectory),
            routing_temperature=trainer.model.route_temperature,
            internal_noise_seed=noise_seed,
            internal_noise_scale=noise_scale,
            internal_noise_block=seed_block,
            require_zero_rails=true,
            silenced_block,
            internal_state_key=state_key,
            internal_key_fraction=key_fraction,
            internal_key_gain=key_gain,
        )
        audit.zero_rail_checks += 1
    end
    nonzero = count(!iszero, base.rails)
    audit.nonzero_rail_observations += nonzero
    nonzero == 0 || error("sleep contaminated the zero rail arena")
    return nothing
end

function internal_replay_outcome(trainer, replay, trajectories)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    continuation_sum = 0.0f0
    completion_successes = 0.0f0
    energy = 0.0f0
    prediction_error = 0.0f0
    @inbounds for flat in 1:trajectories
        seed_block = mod1(flat, model.blocks)
        alive = true
        completed = false
        for cycle in 1:model.cycles
            any_spike = false
            for block in 1:model.blocks
                first_cell = (block - 1) * model.cells_per_block + 1
                last_cell = block * model.cells_per_block
                block_spikes = 0.0f0
                for cell in first_cell:last_cell
                    spike = tape.cell_spikes[cell, cycle, flat]
                    block_spikes += spike
                    any_spike |= spike != 0.0f0
                end
                energy += block_spikes / Float32(model.cells_per_block)
                if cycle > 1 && block != seed_block && block_spikes > 0.0f0
                    completed = true
                end
                offset = (block - 1) * model.node_dim
                for coordinate in 1:model.node_dim
                    node = offset + coordinate
                    difference =
                        base.membrane[node, cycle + 1, flat] -
                        replay.reference_membrane[
                            node,
                            cycle + 1,
                            flat,
                            1,
                        ]
                    prediction_error = muladd(
                        difference,
                        difference,
                        prediction_error,
                    )
                end
            end
            if alive && any_spike
                continuation_sum += 1.0f0
            else
                alive = false
            end
        end
        completion_successes += completed
    end
    continuation = continuation_sum /
        Float32(trajectories * model.cycles)
    completion = completion_successes / Float32(trajectories)
    mean_energy = energy /
        Float32(trajectories * model.cycles * model.blocks)
    mean_prediction_error = prediction_error / Float32(
        trajectories * model.cycles * model.blocks * model.node_dim,
    )
    score =
        0.45f0 * completion +
        0.35f0 * continuation -
        0.15f0 * mean_prediction_error -
        0.05f0 * mean_energy
    return (;
        score,
        continuation,
        completion,
        mean_energy,
        mean_prediction_error,
    )
end

function add_block_silencing_advantage!(
    trainer,
    tags,
    statistics,
    replay,
    scratch,
    trajectories,
    noise_seed,
    options,
    audit,
)
    factual = internal_replay_outcome(trainer, replay, trajectories)
    order = sortperm(statistics.block_load; rev=true)
    maximum_blocks = min(12, trainer.model.blocks)
    tested = Int[]
    for block in order
        statistics.block_load[block] > 0.0f0 || break
        push!(tested, block)
        length(tested) == maximum_blocks && break
    end
    raw_advantage = zeros(Float32, trainer.model.blocks)
    for block in tested
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            silenced_block=block,
        )
        counterfactual = internal_replay_outcome(
            trainer,
            replay,
            trajectories,
        )
        raw_advantage[block] = factual.score - counterfactual.score
    end
    # Restore the factual trajectory before any parameter update or teacher
    # snapshot.  The paired counterfactual differs only by the silenced block.
    run_internal_sleep_trajectories!(
        trainer,
        scratch,
        trajectories,
        noise_seed,
        options.internal_noise_scale,
        audit,
    )
    direction = zeros(Float32, trainer.model.blocks)
    if !isempty(tested)
        center = mean(@view raw_advantage[tested])
        square_sum = 0.0f0
        for block in tested
            difference = raw_advantage[block] - center
            direction[block] = difference
            square_sum = muladd(difference, difference, square_sum)
        end
        inverse_rms = inv(sqrt(
            square_sum / Float32(length(tested)) + 1.0f-8,
        ))
        for block in tested
            direction[block] = clamp(
                direction[block] * inverse_rms,
                -4.0f0,
                4.0f0,
            )
        end
    end
    cost = 0.35f0 .* statistics.cost .- 0.65f0 .* direction
    cost .-= mean(cost)
    rms_normalize!(cost)
    return merge(statistics, (;
        cost,
        causal_advantage=raw_advantage,
        causal_direction=direction,
        causal_tested_blocks=length(tested),
        causal_positive_blocks=count(>(0.0f0), raw_advantage[tested]),
        causal_mean_advantage=
            isempty(tested) ? 0.0f0 : mean(@view raw_advantage[tested]),
        causal_maximum_advantage=
            isempty(tested) ? 0.0f0 : maximum(@view raw_advantage[tested]),
        factual_replay_score=factual.score,
    ))
end

function initialize_replay_state(trainer, trajectories)
    model = trainer.model
    reference = zeros(
        Float32,
        model.blocks * model.node_dim,
        model.cycles + 1,
        trajectories,
        1,
    )
    @views copyto!(
        reference[:, :, :, 1],
        trainer.tape.base.membrane[:, :, 1:trajectories],
    )
    return SleepReplayState(
        reference,
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        copy(trainer.parameters.synapse_weight),
        copy(trainer.parameters.workspace_key),
    )
end

function sleep_statistics!(trainer, tags, replay, trajectories)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    blocks = model.blocks
    cells = blocks * model.cells_per_block
    edge_eligibility = zeros(Float32, size(trainer.parameters.synapse_weight))
    route_gradient = zeros(Float32, size(trainer.parameters.workspace_key))
    block_activity = zeros(Float32, blocks)
    block_energy = zeros(Float32, blocks)
    block_load = zeros(Float32, blocks)
    prediction_error = zeros(Float32, blocks)
    completion = zeros(Float32, blocks)
    continuation_sum = 0.0
    completion_successes = 0
    @inbounds for flat in 1:trajectories
        seed_block = mod1(flat, blocks)
        continuation = 0
        alive = true
        completed = false
        for cycle in 1:model.cycles
            any_spike = false
            for block in 1:blocks
                selected = base.block_mask[block, cycle, flat]
                block_load[block] += selected
                block_spikes = 0.0f0
                first_cell = (block - 1) * model.cells_per_block + 1
                last_cell = block * model.cells_per_block
                for cell in first_cell:last_cell
                    spike = tape.cell_spikes[cell, cycle, flat]
                    block_spikes += spike
                    any_spike |= spike != 0.0f0
                end
                rate = block_spikes / Float32(model.cells_per_block)
                block_activity[block] += rate
                block_energy[block] += rate
                if cycle > 1 && block != seed_block && rate > 0.0f0
                    completion[block] += rate
                    completed = true
                end
                for coordinate in 1:model.node_dim
                    node = coordinate + (block - 1) * model.node_dim
                    value = base.membrane[node, cycle + 1, flat]
                    reference = replay.reference_membrane[
                        node,
                        cycle + 1,
                        flat,
                        1,
                    ]
                    difference = value - reference
                    prediction_error[block] = muladd(
                        difference,
                        difference,
                        prediction_error[block],
                    )
                    eligibility = base.route_eligibility[
                        block,
                        cycle,
                        flat,
                    ]
                    route_gradient[coordinate, block] = muladd(
                        eligibility *
                        tape.state_query[coordinate, cycle, flat],
                        value,
                        route_gradient[coordinate, block],
                    )
                end
            end
            if alive && any_spike
                continuation += 1
            else
                alive = false
            end
            if cycle > 1
                for source in 1:cells
                    pre = tape.cell_spikes[source, cycle - 1, flat]
                    pre == 0.0f0 && continue
                    for relation in 1:model.fanout
                        trainer.cache.gate_hard[source, relation] == 0.0f0 &&
                            continue
                        destination = model.destination_for_source[
                            source,
                            relation,
                        ]
                        post = tape.soma_spikes[destination, cycle, flat]
                        edge_eligibility[source, relation] = muladd(
                            pre,
                            post,
                            edge_eligibility[source, relation],
                        )
                    end
                end
            end
        end
        continuation_sum += continuation
        completion_successes += completed
    end
    normalizer = inv(Float32(trajectories * model.cycles))
    block_activity .*= normalizer
    block_energy .*= normalizer
    block_load .*= normalizer
    completion .*= normalizer
    prediction_error ./=
        Float32(trajectories * model.cycles * model.node_dim)
    prediction_reduction = replay.previous_prediction_error .-
        prediction_error
    copyto!(replay.previous_prediction_error, prediction_error)
    replay.replay_count .+= block_load
    novelty = @. inv(sqrt(1.0f0 + replay.replay_count))
    expected_load = Float32(model.workspace_k / model.blocks)
    task_consistency = tags.block_utility .* block_activity
    reward =
        0.20f0 .* prediction_reduction .+
        0.35f0 .* completion .+
        0.35f0 .* task_consistency .+
        0.10f0 .* novelty .* completion .-
        0.08f0 .* block_energy .-
        0.20f0 .* max.(block_load .- expected_load, 0.0f0)
    cost = .-reward
    cost .-= mean(cost)
    raw_edge_eligibility = edge_eligibility ./
        Float32(max(trajectories * model.cycles, 1))
    raw_route_gradient = route_gradient ./
        Float32(max(trajectories * model.cycles, 1))
    rms_normalize!(cost)
    rms_normalize!(edge_eligibility)
    rms_normalize!(route_gradient)
    return (;
        cost,
        edge_eligibility,
        route_gradient,
        raw_edge_eligibility,
        raw_route_gradient,
        block_activity,
        block_energy,
        block_load,
        prediction_error,
        prediction_reduction,
        completion,
        continuation_length=continuation_sum / trajectories,
        pattern_completion_success=completion_successes / trajectories,
        route_entropy=mean(view(
            base.route_normalized_entropy,
            :,
            1:trajectories,
        )),
    )
end

function apply_recurrent_sleep!(
    trainer,
    tags,
    statistics,
    options,
    rate_scale::Float32=1.0f0,
)
    model = trainer.model
    weights = trainer.parameters.synapse_weight
    utility = abs.(tags.synapse_direction)
    maximum_utility = maximum(utility)
    maximum_utility > 0.0f0 && (utility ./= maximum_utility)
    @inbounds for source in axes(weights, 1)
        for relation in axes(weights, 2)
            destination = model.destination_for_source[source, relation]
            block = div(destination - 1, model.cells_per_block) + 1
            eligibility = statistics.edge_eligibility[source, relation]
            task_update =
                tags.synapse_direction[source, relation] * abs(eligibility)
            homeostatic_update =
                -statistics.cost[block] *
                eligibility *
                sign(weights[source, relation])
            weights[source, relation] +=
                options.recurrent_rate * rate_scale * (
                0.75f0 * task_update +
                0.25f0 * homeostatic_update
            )
            protection = utility[source, relation]
            weights[source, relation] *=
                1.0f0 - options.downscale_rate * rate_scale *
                (1.0f0 - protection)
        end
    end
    target_rate = Float32(clamp(tags.wake_firing_rate, 0.002, 0.05))
    @inbounds for cell in eachindex(trainer.parameters.soma_threshold_logits)
        block = div(cell - 1, model.cells_per_block) + 1
        rate_error = statistics.block_activity[block] - target_rate
        trainer.parameters.soma_threshold_logits[cell] +=
            options.homeostasis_rate * rate_scale * rate_error
    end
    return nothing
end

function direction_projection_and_residual(delta, direction)
    denominator = sum(abs2, direction)
    denominator > 0.0f0 || return (0.0f0, 0.0f0)
    projection = dot(delta, direction) / denominator
    residual_square = 0.0
    @inbounds for index in eachindex(delta, direction)
        residual = delta[index] - projection * direction[index]
        residual_square = muladd(
            Float64(residual),
            Float64(residual),
            residual_square,
        )
    end
    return projection, Float32(sqrt(residual_square / length(delta)))
end

function wake_route_margin_guard(
    trainer,
    tags,
    anchor_synapse,
    anchor_threshold,
    anchor_key,
)
    synapse_delta = trainer.parameters.synapse_weight .- anchor_synapse
    threshold_delta =
        trainer.parameters.soma_threshold_logits .- anchor_threshold
    synapse_projection, synapse_residual =
        direction_projection_and_residual(
            synapse_delta,
            tags.synapse_direction,
        )
    threshold_projection, threshold_residual =
        direction_projection_and_residual(
            threshold_delta,
            tags.threshold_direction,
        )
    reference_scale = tags.margin_reference_scale
    uncertainty = 8.0f0 * (synapse_residual + threshold_residual)
    minimum_predicted = Inf32
    minimum_lower_bound = Inf32
    violations = 0
    @inbounds for target in axes(tags.route_margin, 2)
        for cycle in axes(tags.route_margin, 1)
            selected = Int(tags.route_selected[cycle, target])
            challenger = Int(tags.route_challenger[cycle, target])
            predicted = tags.route_margin[cycle, target]
            predicted +=
                (synapse_projection / reference_scale) *
                tags.synapse_margin_reference_step[cycle, target]
            predicted +=
                (threshold_projection / reference_scale) *
                tags.threshold_margin_reference_step[cycle, target]
            for coordinate in axes(anchor_key, 1)
                predicted = muladd(
                    tags.route_selected_key_factor[
                        coordinate,
                        cycle,
                        target,
                    ],
                    trainer.parameters.workspace_key[coordinate, selected] -
                    anchor_key[coordinate, selected],
                    predicted,
                )
                predicted = muladd(
                    -tags.route_challenger_key_factor[
                        coordinate,
                        cycle,
                        target,
                    ],
                    trainer.parameters.workspace_key[
                        coordinate,
                        challenger,
                    ] - anchor_key[coordinate, challenger],
                    predicted,
                )
            end
            lower_bound = predicted - uncertainty
            minimum_predicted = min(minimum_predicted, predicted)
            minimum_lower_bound = min(minimum_lower_bound, lower_bound)
            violations += lower_bound <= 0.0f0
        end
    end
    return (;
        safe=violations == 0,
        minimum_anchor_margin=minimum(tags.route_margin),
        minimum_predicted,
        minimum_lower_bound,
        violations,
        synapse_projection,
        threshold_projection,
        synapse_residual,
        threshold_residual,
        uncertainty,
    )
end

function apply_recurrent_sleep_trust_region!(
    trainer,
    tags,
    statistics,
    options,
    anchor_synapse,
    anchor_threshold,
    anchor_key,
)
    weights_before = copy(trainer.parameters.synapse_weight)
    threshold_before = copy(trainer.parameters.soma_threshold_logits)
    last_guard = wake_route_margin_guard(
        trainer,
        tags,
        anchor_synapse,
        anchor_threshold,
        anchor_key,
    )
    for attempt in 1:8
        copyto!(trainer.parameters.synapse_weight, weights_before)
        copyto!(trainer.parameters.soma_threshold_logits, threshold_before)
        scale = Float32(0.5^(attempt - 1))
        apply_recurrent_sleep!(trainer, tags, statistics, options, scale)
        ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
            trainer.cache,
            trainer.parameters,
            trainer.gate_mask,
        )
        guard = wake_route_margin_guard(
            trainer,
            tags,
            anchor_synapse,
            anchor_threshold,
            anchor_key,
        )
        last_guard = guard
        guard.safe && return true, scale, guard
    end
    copyto!(trainer.parameters.synapse_weight, weights_before)
    copyto!(trainer.parameters.soma_threshold_logits, threshold_before)
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    return false, 0.0f0, last_guard
end

function apply_route_sleep!(
    trainer,
    tags,
    statistics,
    options,
    rate_scale::Float32=1.0f0,
)
    key = trainer.parameters.workspace_key
    @inbounds for block in axes(key, 2)
        replay_gate = clamp(
            statistics.block_load[block] *
            Float32(trainer.model.blocks / trainer.model.workspace_k),
            0.0f0,
            1.0f0,
        )
        for coordinate in axes(key, 1)
            task_update =
                tags.workspace_key_direction[coordinate, block] *
                replay_gate
            cost_update =
                -statistics.cost[block] *
                statistics.route_gradient[coordinate, block]
            key[coordinate, block] += options.route_rate * rate_scale * (
                0.65f0 * task_update +
                0.35f0 * cost_update
            )
        end
    end
    return nothing
end

function apply_route_sleep_trust_region!(
    trainer,
    tags,
    statistics,
    replay,
    scratch,
    trajectories,
    seed,
    options,
    audit,
    anchor_synapse,
    anchor_threshold,
    anchor_key,
)
    key_before = copy(trainer.parameters.workspace_key)
    previous_error = copy(replay.previous_prediction_error)
    replay_count = copy(replay.replay_count)
    baseline_error = mean(statistics.prediction_error)
    for attempt in 1:6
        copyto!(trainer.parameters.workspace_key, key_before)
        copyto!(replay.previous_prediction_error, previous_error)
        copyto!(replay.replay_count, replay_count)
        scale = Float32(0.5^(attempt - 1))
        apply_route_sleep!(trainer, tags, statistics, options, scale)
        ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
            trainer.cache,
            trainer.parameters,
            trainer.gate_mask,
        )
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            trajectories,
            seed,
            options.internal_noise_scale,
            audit,
        )
        candidate = sleep_statistics!(
            trainer,
            tags,
            replay,
            trajectories,
        )
        candidate_error = mean(candidate.prediction_error)
        entropy_ok = candidate.route_entropy >= 0.85f0
        drift_ok = candidate_error <= baseline_error + 1.0f-7
        margin_guard = wake_route_margin_guard(
            trainer,
            tags,
            anchor_synapse,
            anchor_threshold,
            anchor_key,
        )
        copyto!(replay.replay_count, replay_count)
        if entropy_ok && drift_ok && margin_guard.safe
            return candidate, true, scale, margin_guard
        end
    end
    copyto!(trainer.parameters.workspace_key, key_before)
    copyto!(replay.previous_prediction_error, previous_error)
    copyto!(replay.replay_count, replay_count)
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    margin_guard = wake_route_margin_guard(
        trainer,
        tags,
        anchor_synapse,
        anchor_threshold,
        anchor_key,
    )
    return statistics, false, 0.0f0, margin_guard
end

function update_teacher_snapshot!(trainer, replay, trajectories)
    decay = 0.995f0
    complement = 1.0f0 - decay
    @views replay.reference_membrane[:, :, :, 1] .=
        decay .* replay.reference_membrane[:, :, :, 1] .+
        complement .* trainer.tape.base.membrane[:, :, 1:trajectories]
    replay.teacher_synapse .=
        decay .* replay.teacher_synapse .+
        complement .* trainer.parameters.synapse_weight
    replay.teacher_workspace_key .=
        decay .* replay.teacher_workspace_key .+
        complement .* trainer.parameters.workspace_key
    return nothing
end

function parameter_drift(before, after)
    maximum_delta = 0.0f0
    square_sum = 0.0
    @inbounds for index in eachindex(before, after)
        difference = after[index] - before[index]
        maximum_delta = max(maximum_delta, abs(difference))
        square_sum = muladd(Float64(difference), Float64(difference), square_sum)
    end
    return (;
        maximum=maximum_delta,
        rms=sqrt(square_sum / length(before)),
    )
end

function run_sleep_arm!(trainer, arm, tags, options)
    audit = SleepAudit()
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        trainer.model,
        trainer.parameters,
    )
    before_synapse = copy(trainer.parameters.synapse_weight)
    before_threshold = copy(trainer.parameters.soma_threshold_logits)
    before_key = copy(trainer.parameters.workspace_key)
    before_head = copy(trainer.parameters.head_weight)
    replay = nothing
    final_statistics = nothing
    route_trust_accepts = 0
    route_trust_rejects = 0
    route_trust_last_scale = 0.0f0
    recurrent_trust_accepts = 0
    recurrent_trust_rejects = 0
    recurrent_trust_last_scale = 0.0f0
    causal_evaluations = 0
    causal_tested_blocks = 0
    causal_positive_blocks = 0
    causal_advantage_sum = 0.0f0
    causal_advantage_maximum = -Inf32
    final_margin_guard = wake_route_margin_guard(
        trainer,
        tags,
        before_synapse,
        before_threshold,
        before_key,
    )
    if arm !== :wake_only
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            SLEEP_SEED,
            options.internal_noise_scale,
            audit,
        )
        replay = initialize_replay_state(trainer, options.trajectories)
        for microcycle in 1:options.microcycles
            seed = SLEEP_SEED ⊻ UInt64(microcycle) << 32
            run_internal_sleep_trajectories!(
                trainer,
                scratch,
                options.trajectories,
                seed,
                options.internal_noise_scale,
                audit,
            )
            statistics = sleep_statistics!(
                trainer,
                tags,
                replay,
                options.trajectories,
            )
            if options.counterfactual
                statistics = add_block_silencing_advantage!(
                    trainer,
                    tags,
                    statistics,
                    replay,
                    scratch,
                    options.trajectories,
                    seed,
                    options,
                    audit,
                )
                causal_evaluations += 1
                causal_tested_blocks += statistics.causal_tested_blocks
                causal_positive_blocks += statistics.causal_positive_blocks
                causal_advantage_sum += statistics.causal_mean_advantage
                causal_advantage_maximum = max(
                    causal_advantage_maximum,
                    statistics.causal_maximum_advantage,
                )
            end
            if arm === :recurrent_only
                accepted, accepted_scale, margin_guard =
                    apply_recurrent_sleep_trust_region!(
                        trainer,
                        tags,
                        statistics,
                        options,
                        before_synapse,
                        before_threshold,
                        before_key,
                    )
                recurrent_trust_accepts += accepted
                recurrent_trust_rejects += !accepted
                recurrent_trust_last_scale = accepted_scale
                final_margin_guard = margin_guard
            elseif arm === :route_only
                statistics, accepted, accepted_scale, margin_guard =
                    apply_route_sleep_trust_region!(
                        trainer,
                        tags,
                        statistics,
                        replay,
                        scratch,
                        options.trajectories,
                        seed ⊻ UInt64(0xa17e2a7e),
                        options,
                        audit,
                        before_synapse,
                        before_threshold,
                        before_key,
                    )
                route_trust_accepts += accepted
                route_trust_rejects += !accepted
                route_trust_last_scale = accepted_scale
                final_margin_guard = margin_guard
            elseif arm === :alternating
                recurrent_accepted, recurrent_scale, margin_guard =
                    apply_recurrent_sleep_trust_region!(
                        trainer,
                        tags,
                        statistics,
                        options,
                        before_synapse,
                        before_threshold,
                        before_key,
                    )
                recurrent_trust_accepts += recurrent_accepted
                recurrent_trust_rejects += !recurrent_accepted
                recurrent_trust_last_scale = recurrent_scale
                final_margin_guard = margin_guard
                run_internal_sleep_trajectories!(
                    trainer,
                    scratch,
                    options.trajectories,
                    seed ⊻ UInt64(0xa17e2a7e),
                    options.internal_noise_scale,
                    audit,
                )
                statistics = sleep_statistics!(
                    trainer,
                    tags,
                    replay,
                    options.trajectories,
                )
                if options.counterfactual
                    statistics = add_block_silencing_advantage!(
                        trainer,
                        tags,
                        statistics,
                        replay,
                        scratch,
                        options.trajectories,
                        seed ⊻ UInt64(0xa17e2a7e),
                        options,
                        audit,
                    )
                    causal_evaluations += 1
                    causal_tested_blocks +=
                        statistics.causal_tested_blocks
                    causal_positive_blocks +=
                        statistics.causal_positive_blocks
                    causal_advantage_sum +=
                        statistics.causal_mean_advantage
                    causal_advantage_maximum = max(
                        causal_advantage_maximum,
                        statistics.causal_maximum_advantage,
                    )
                end
                statistics, accepted, accepted_scale, margin_guard =
                    apply_route_sleep_trust_region!(
                        trainer,
                        tags,
                        statistics,
                        replay,
                        scratch,
                        options.trajectories,
                        seed ⊻ UInt64(0xa17e2a7e),
                        options,
                        audit,
                        before_synapse,
                        before_threshold,
                        before_key,
                    )
                route_trust_accepts += accepted
                route_trust_rejects += !accepted
                route_trust_last_scale = accepted_scale
                final_margin_guard = margin_guard
            elseif arm === :simultaneous
                apply_recurrent_sleep!(trainer, tags, statistics, options)
                apply_route_sleep!(trainer, tags, statistics, options)
                ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
                    trainer.cache,
                    trainer.parameters,
                    trainer.gate_mask,
                )
            else
                error("unknown sleep arm $arm")
            end
            update_teacher_snapshot!(
                trainer,
                replay,
                options.trajectories,
            )
            final_statistics = statistics
        end
    end
    final_margin_guard = wake_route_margin_guard(
        trainer,
        tags,
        before_synapse,
        before_threshold,
        before_key,
    )
    audit.dataset_reads == 0 || error("sleep read the dataset")
    audit.teacher_target_reads == 0 || error("sleep read teacher targets")
    head_drift = parameter_drift(before_head, trainer.parameters.head_weight)
    head_drift.maximum == 0.0f0 || error("sleep changed the supervised head")
    return (;
        audit=(;
            zero_rail_checks=audit.zero_rail_checks,
            nonzero_rail_observations=audit.nonzero_rail_observations,
            dataset_reads=audit.dataset_reads,
            teacher_target_reads=audit.teacher_target_reads,
        ),
        recurrent_delta=parameter_drift(
            before_synapse,
            trainer.parameters.synapse_weight,
        ),
        route_delta=parameter_drift(
            before_key,
            trainer.parameters.workspace_key,
        ),
        head_delta=head_drift,
        wake_margin_guard=final_margin_guard,
        replay=final_statistics === nothing ? nothing : (;
            route_entropy=final_statistics.route_entropy,
            continuation_length=final_statistics.continuation_length,
            pattern_completion_success=
                final_statistics.pattern_completion_success,
            mean_block_load=mean(final_statistics.block_load),
            maximum_block_load=maximum(final_statistics.block_load),
            mean_prediction_error=
                mean(final_statistics.prediction_error),
            mean_prediction_error_reduction=
                mean(final_statistics.prediction_reduction),
            mean_energy=mean(final_statistics.block_energy),
            route_trust_accepts,
            route_trust_rejects,
            route_trust_last_scale,
            recurrent_trust_accepts,
            recurrent_trust_rejects,
            recurrent_trust_last_scale,
            causal_evaluations,
            causal_tested_blocks,
            causal_positive_blocks,
            causal_mean_advantage=
                causal_evaluations == 0 ? 0.0f0 :
                causal_advantage_sum / Float32(causal_evaluations),
            causal_maximum_advantage=
                causal_evaluations == 0 ? 0.0f0 :
                causal_advantage_maximum,
        ),
    )
end

function relative_feature_drift(before, after)
    size(before) == size(after) || error("head feature shape changed")
    numerator = sum(abs2, after .- before)
    denominator = max(sum(abs2, before), eps(Float32))
    cosine = dot(vec(before), vec(after)) /
        max(norm(before) * norm(after), eps(Float32))
    return (;
        relative_rms=sqrt(Float64(numerator / denominator)),
        cosine=Float64(cosine),
    )
end

function route_mask_drift(before, after)
    size(before) == size(after) || error("route mask shape changed")
    changed = 0
    @inbounds for index in eachindex(before, after)
        changed += before[index] != after[index]
    end
    return changed / length(before)
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    options.microcycles > 0 || error("microcycles must be positive")
    options.trajectories > 0 || error("trajectories must be positive")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    BLAS.set_num_threads(1)
    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    hasproperty(payload.run_config, :overfit_rows) ||
        error("sleep shadow currently requires an overfit checkpoint")
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
        error("overfit panel must fill one arena batch")
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    baseline = wake_metrics_and_features!(
        trainer,
        dataset,
        rows,
        options.workers,
    )
    tags = capture_wake_tags!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
    )
    tag_summary = (;
        synapse_rms=sqrt(mean(abs2, tags.synapse_direction)),
        threshold_rms=sqrt(mean(abs2, tags.threshold_direction)),
        route_rms=sqrt(mean(abs2, tags.workspace_key_direction)),
        block_utility_mean=mean(tags.block_utility),
        block_utility_maximum=maximum(tags.block_utility),
        wake_firing_rate=tags.wake_firing_rate,
        wake_workspace_transition=tags.wake_workspace_transition,
        minimum_route_margin=minimum(tags.route_margin),
        median_route_margin=median(vec(tags.route_margin)),
        margin_reference_scale=tags.margin_reference_scale,
        raw_synapse_tag_rms=tags.raw_synapse_tag_rms,
        raw_threshold_tag_rms=tags.raw_threshold_tag_rms,
        raw_route_tag_rms=tags.raw_route_tag_rms,
        raw_block_utility_maximum=tags.raw_block_utility_maximum,
    )
    results = Dict{String,Any}()
    for arm in SLEEP_ARMS
        restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
        sleep = run_sleep_arm!(trainer, arm, tags, options)
        after = wake_metrics_and_features!(
            trainer,
            dataset,
            rows,
            options.workers,
        )
        drift = relative_feature_drift(
            baseline.features,
            after.features,
        )
        mask_drift = route_mask_drift(
            baseline.route_masks,
            after.route_masks,
        )
        results[String(arm)] = (;
            before=baseline.metrics,
            after=after.metrics,
            head_input_drift=drift,
            route_mask_change_fraction=mask_drift,
            sleep,
        )
        println(
            "arm=$(arm) excess=$(round(after.metrics.excess_loss; digits=6)) " *
            "top1=$(round(after.metrics.top1; digits=6)) " *
            "rec_delta=$(round(sleep.recurrent_delta.maximum; digits=7)) " *
            "route_delta=$(round(sleep.route_delta.maximum; digits=7)) " *
            "drift=$(round(drift.relative_rms; digits=7)) " *
            "mask_change=$(round(mask_drift; digits=7)) " *
            "margin_lower=$(round(sleep.wake_margin_guard.minimum_lower_bound; digits=7)) " *
            "margin_violations=$(sleep.wake_margin_guard.violations)",
        )
    end
    output = (;
        schema="reduced-hay-v2-brain-internal-sleep-shadow-v2",
        checkpoint=options.checkpoint,
        checkpoint_update=Int(payload.update),
        rows,
        constraints=(;
            external_rails="strict_zero",
            stored_samples=false,
            world_model=false,
            separate_generator=false,
            supervised_head_frozen=true,
            teacher_block_direct_connection=false,
            sleep_reward_semantics="internal_predictive_cost_advantage",
        ),
        options=(;
            microcycles=options.microcycles,
            trajectories=options.trajectories,
            internal_noise_scale=options.internal_noise_scale,
            recurrent_rate=options.recurrent_rate,
            route_rate=options.route_rate,
            downscale_rate=options.downscale_rate,
            homeostasis_rate=options.homeostasis_rate,
            counterfactual=options.counterfactual,
        ),
        tag_summary,
        results,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println("output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
