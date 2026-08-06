using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "compare_reduced_hay_v2_sleep_shadow.jl"))

function parse_identity_options(arguments)
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
        workers=parse(Int, get(values, "workers", "20")),
        trajectories=parse(Int, get(values, "trajectories", "8")),
        noise_seeds=parse(Int, get(values, "noise-seeds", "8")),
        internal_noise_scale=parse(
            Float32,
            get(values, "internal-noise-scale", "2.0"),
        ),
        key_fraction=parse(
            Float32,
            get(values, "key-fraction", "0.25"),
        ),
        key_gain=parse(Float32, get(values, "key-gain", "0.75")),
        score_cycle_start=parse(
            Int,
            get(values, "score-cycle-start", "2"),
        ),
        prototypes_per_block=parse(
            Int,
            get(values, "prototypes-per-block", "4"),
        ),
        engram_completion=parse_boolean(get(
            values,
            "engram-completion",
            "false",
        )),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "sleep_replay_identity_gate.json"),
        )),
    )
end

@inline function sequence_cosine(sequence, prototype)
    denominator = norm(sequence) * norm(prototype)
    denominator > 1.0f-8 || return -Inf32
    return dot(sequence, prototype) / denominator
end

function capture_wake_state_tags!(
    trainer,
    payload,
    dataset,
    rows,
    workers,
    prototypes_per_block,
    ;
    credit_mode::Symbol=:exact_bptt,
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
        # Capture wake trajectories and block-local signed task tags without
        # advancing Adam.  Sleep identity must describe the current network,
        # not an optimizer-mutated successor.
        reduced_hay_v2_arena_gradient!(running)
    end
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    # A few block-local sequence prototypes preserve wake-state identity
    # without retaining a dataset row or candidate trajectory.
    prototypes = zeros(
        Float32,
        model.node_dim,
        model.cycles,
        prototypes_per_block,
        model.blocks,
    )
    prototype_weight = zeros(
        Float32,
        prototypes_per_block,
        model.blocks,
    )
    # This is an accumulator, not a write-once buffer.  `similar` leaves
    # memory uninitialized and made replay ranking depend on allocator history
    # (occasionally even injecting NaN into a shuffled control).
    prototype_signed_credit = zeros(
        Float32,
        prototypes_per_block,
        model.blocks,
    )
    prototype_assignment = zeros(
        UInt8,
        model.blocks,
        trainer.tape.base.valid_count,
    )
    priority = zeros(Float32, model.blocks)
    sequence = zeros(Float32, model.node_dim, model.cycles)
    @inbounds for block in 1:model.blocks
        offset = (block - 1) * model.node_dim
        for target in 1:base.valid_count
            flat = Int(base.valid_flats[target])
            absolute_credit = 0.0f0
            signed_credit = 0.0f0
            for cycle in 1:model.cycles
                reward = tape.block_supervised_reward[block, cycle, flat]
                absolute_credit += abs(reward)
                signed_credit += reward
                for coordinate in 1:model.node_dim
                    sequence[coordinate, cycle] = base.membrane[
                        offset + coordinate,
                        cycle + 1,
                        flat,
                    ]
                end
            end
            absolute_credit > 0.0f0 || continue
            sequence_norm = norm(sequence)
            sequence_norm > 1.0f-8 || continue
            sequence ./= sequence_norm
            empty_rank = findfirst(
                iszero,
                view(prototype_weight, :, block),
            )
            rank = if empty_rank === nothing
                best_rank = 1
                best_score = -Inf32
                for candidate_rank in 1:prototypes_per_block
                    score = sequence_cosine(
                        sequence,
                        view(
                            prototypes,
                            :,
                            :,
                            candidate_rank,
                            block,
                        ),
                    )
                    if score > best_score
                        best_score = score
                        best_rank = candidate_rank
                    end
                end
                best_rank
            else
                Int(empty_rank)
            end
            prototype_assignment[block, target] = UInt8(rank)
            for cycle in 1:model.cycles
                for coordinate in 1:model.node_dim
                    prototypes[coordinate, cycle, rank, block] = muladd(
                        absolute_credit,
                        sequence[coordinate, cycle],
                        prototypes[coordinate, cycle, rank, block],
                    )
                end
            end
            prototype_weight[rank, block] += absolute_credit
            prototype_signed_credit[rank, block] += signed_credit
            priority[block] += absolute_credit
        end
    end
    @inbounds for block in 1:model.blocks
        for rank in 1:prototypes_per_block
            prototype_weight[rank, block] > 0.0f0 || continue
            prototype = @view prototypes[:, :, rank, block]
            prototype ./= max(norm(prototype), 1.0f-8)
        end
    end
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    return (;
        prototypes,
        prototype_weight,
        prototype_signed_credit,
        prototype_assignment,
        priority,
    )
end

@inline function prototype_cosine(state, prototype)
    denominator = norm(state) * norm(prototype)
    denominator > 1.0f-8 || return 0.0f0
    return dot(state, prototype) / denominator
end

function best_prototype_score(state, prototypes, cycle, block)
    best = -Inf32
    @inbounds for rank in axes(prototypes, 3)
        best = max(
            best,
            prototype_cosine(
                state,
                view(prototypes, :, cycle, rank, block),
            ),
        )
    end
    return best
end

function score_sleep_identity(
    trainer,
    tags,
    seed_blocks,
    seed_ranks,
    trajectories,
    score_cycle_start,
)
    model = trainer.model
    base = trainer.tape.base
    target_score = 0.0
    block_shuffle_score = 0.0
    key_shuffle_score = 0.0
    temporal_reverse_score = 0.0
    target_margin = 0.0
    top1 = 0
    target_rank_margin = 0.0
    rank_top1 = 0
    observations = 0
    reversed_coordinates = model.node_dim:-1:1
    @inbounds for trajectory in 1:trajectories
        target_block = seed_blocks[trajectory]
        target_rank = seed_ranks[trajectory]
        for cycle in score_cycle_start:model.cycles
            offset = (target_block - 1) * model.node_dim
            state = @view base.membrane[
                (offset + 1):(offset + model.node_dim),
                cycle + 1,
                trajectory,
            ]
            score = prototype_cosine(
                state,
                view(
                    tags.prototypes,
                    :,
                    cycle,
                    target_rank,
                    target_block,
                ),
            )
            best_other = -Inf32
            best_non_target = -Inf32
            best_block = 0
            for block in 1:model.blocks
                candidate = best_prototype_score(
                    state,
                    tags.prototypes,
                    cycle,
                    block,
                )
                if candidate > best_other
                    best_other = candidate
                    best_block = block
                end
                block != target_block &&
                    (best_non_target = max(best_non_target, candidate))
            end
            best_rank_score = -Inf32
            best_non_target_rank = -Inf32
            best_rank = 0
            for rank in axes(tags.prototypes, 3)
                candidate = prototype_cosine(
                    state,
                    view(
                        tags.prototypes,
                        :,
                        cycle,
                        rank,
                        target_block,
                    ),
                )
                if candidate > best_rank_score
                    best_rank_score = candidate
                    best_rank = rank
                end
                rank != target_rank &&
                    (best_non_target_rank = max(
                        best_non_target_rank,
                        candidate,
                    ))
            end
            shuffled_block = mod1(target_block + 17, model.blocks)
            shuffled_rank = mod1(
                target_rank + 1,
                size(tags.prototypes, 3),
            )
            shuffled_score = prototype_cosine(
                state,
                view(
                    tags.prototypes,
                    :,
                    cycle,
                    shuffled_rank,
                    shuffled_block,
                ),
            )
            key_score = prototype_cosine(
                state,
                view(
                    tags.prototypes,
                    reversed_coordinates,
                    cycle,
                    target_rank,
                    target_block,
                ),
            )
            reverse_cycle = model.cycles - cycle + 1
            reverse_score = prototype_cosine(
                state,
                view(
                    tags.prototypes,
                    :,
                    reverse_cycle,
                    target_rank,
                    target_block,
                ),
            )
            target_score += score
            block_shuffle_score += shuffled_score
            key_shuffle_score += key_score
            temporal_reverse_score += reverse_score
            target_margin += score - best_non_target
            top1 += best_block == target_block
            target_rank_margin += score - best_non_target_rank
            rank_top1 += best_rank == target_rank
            observations += 1
        end
    end
    inverse = inv(Float64(observations))
    return (;
        target_cosine=target_score * inverse,
        block_shuffle_cosine=block_shuffle_score * inverse,
        key_shuffle_cosine=key_shuffle_score * inverse,
        temporal_reverse_cosine=temporal_reverse_score * inverse,
        target_margin=target_margin * inverse,
        block_top1=top1 * inverse,
        target_rank_margin=target_rank_margin * inverse,
        rank_top1=rank_top1 * inverse,
        observations,
    )
end

function build_seed_state_keys(tags, seed_blocks, seed_ranks, cycle::Int)
    node_dim = size(tags.prototypes, 1)
    keys = zeros(Float32, node_dim, length(seed_blocks))
    @inbounds for trajectory in eachindex(seed_blocks)
        block = seed_blocks[trajectory]
        rank = seed_ranks[trajectory]
        copyto!(
            view(keys, :, trajectory),
            view(tags.prototypes, :, cycle, rank, block),
        )
    end
    return keys
end

function cue_coordinate_selected(
    model,
    noise_seed::UInt64,
    trajectory::Int,
    coordinate::Int,
    fraction::Float32,
)
    channel = mod(coordinate - 1, model.readout_per_cell) + 1
    channel == 2 && return false
    selector_seed = channel == 1 ?
        noise_seed ⊻ UInt64(0x4355454b45593232) :
        noise_seed ⊻ UInt64(0x4355454b45593131)
    selector = 0.5f0 * (
        ReducedHayV2ArenaTraining._internal_sleep_noise(
            selector_seed,
            trajectory,
            coordinate,
        ) + 1.0f0
    )
    return selector < fraction
end

function masked_key_cosine(left, right, mask)
    numerator = 0.0f0
    left_square = 0.0f0
    right_square = 0.0f0
    @inbounds for coordinate in eachindex(left, right, mask)
        mask[coordinate] || continue
        numerator = muladd(left[coordinate], right[coordinate], numerator)
        left_square = muladd(
            left[coordinate],
            left[coordinate],
            left_square,
        )
        right_square = muladd(
            right[coordinate],
            right[coordinate],
            right_square,
        )
    end
    denominator = sqrt(left_square * right_square)
    denominator > 1.0f-8 || return -Inf32
    return numerator / denominator
end

function complete_seed_state_keys(
    trainer,
    tags,
    seed_blocks,
    source_keys,
    noise_seed::UInt64,
    fraction::Float32,
)
    model = trainer.model
    completed = similar(source_keys)
    selected_ranks = zeros(Int, length(seed_blocks))
    mask = falses(model.node_dim)
    @inbounds for trajectory in eachindex(seed_blocks)
        block = seed_blocks[trajectory]
        for coordinate in 1:model.node_dim
            mask[coordinate] = cue_coordinate_selected(
                model,
                noise_seed,
                trajectory,
                coordinate,
                fraction,
            )
        end
        source = view(source_keys, :, trajectory)
        best_rank = 1
        best_score = -Inf32
        for rank in axes(tags.prototypes, 3)
            candidate = view(tags.prototypes, :, 1, rank, block)
            score = masked_key_cosine(source, candidate, mask)
            if score > best_score
                best_score = score
                best_rank = rank
            end
        end
        selected_ranks[trajectory] = best_rank
        copyto!(
            view(completed, :, trajectory),
            view(tags.prototypes, :, 1, best_rank, block),
        )
    end
    return completed, selected_ranks
end

function main(arguments=ARGS)
    options = parse_identity_options(arguments)
    options.trajectories > 0 || error("trajectories must be positive")
    options.noise_seeds > 1 || error("noise-seeds must exceed one")
    options.prototypes_per_block > 0 ||
        error("prototypes-per-block must be positive")
    BLAS.set_num_threads(1)
    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    hasproperty(payload.run_config, :overfit_rows) ||
        error("identity gate requires an overfit checkpoint")
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
    tags = capture_wake_state_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        options.prototypes_per_block,
    )
    priority_pairs = [
        (
            weight=tags.prototype_weight[rank, block],
            block=block,
            rank=rank,
        )
        for block in 1:model.blocks
        for rank in 1:options.prototypes_per_block
        if tags.prototype_weight[rank, block] > 0.0f0
    ]
    sort!(priority_pairs; by=pair -> pair.weight, rev=true)
    length(priority_pairs) >= options.trajectories ||
        error("not enough wake state prototypes")
    selected_pairs = priority_pairs[1:options.trajectories]
    seed_blocks = [pair.block for pair in selected_pairs]
    seed_ranks = [pair.rank for pair in selected_pairs]
    1 <= options.score_cycle_start <= model.cycles ||
        error("score-cycle-start is outside the model")
    seed_state_keys = build_seed_state_keys(
        tags,
        seed_blocks,
        seed_ranks,
        1,
    )
    shuffled_seed_blocks = [
        mod1(block + 17, model.blocks) for block in seed_blocks
    ]
    shuffled_seed_ranks = [
        mod1(rank + 1, options.prototypes_per_block) for rank in seed_ranks
    ]
    shuffled_state_keys = build_seed_state_keys(
        tags,
        shuffled_seed_blocks,
        shuffled_seed_ranks,
        1,
    )
    priority_coverage =
        sum(pair.weight for pair in selected_pairs) /
        max(sum(tags.prototype_weight), eps(Float32))
    audit = SleepAudit()
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    before_synapse = copy(trainer.parameters.synapse_weight)
    before_key = copy(trainer.parameters.workspace_key)
    before_head = copy(trainer.parameters.head_weight)
    results = Any[]
    for seed_index in 1:options.noise_seeds
        noise_seed =
            UInt64(0x4944454e54495459) ⊻
            UInt64(seed_index) * UInt64(0x9e3779b97f4a7c15)
        tagged_forward_keys = seed_state_keys
        shuffled_forward_keys = shuffled_state_keys
        tagged_recalled_ranks = copy(seed_ranks)
        shuffled_recalled_ranks = copy(shuffled_seed_ranks)
        if options.engram_completion
            tagged_forward_keys, tagged_recalled_ranks =
                complete_seed_state_keys(
                    trainer,
                    tags,
                    seed_blocks,
                    seed_state_keys,
                    noise_seed,
                    options.key_fraction,
                )
            shuffled_forward_keys, shuffled_recalled_ranks =
                complete_seed_state_keys(
                    trainer,
                    tags,
                    seed_blocks,
                    shuffled_state_keys,
                    noise_seed,
                    options.key_fraction,
                )
        end
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            seed_blocks,
        )
        no_cue = score_sleep_identity(
            trainer,
            tags,
            seed_blocks,
            seed_ranks,
            options.trajectories,
            options.score_cycle_start,
        )
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            seed_blocks,
            seed_state_keys=tagged_forward_keys,
            key_fraction=options.engram_completion ? 1.0f0 :
                options.key_fraction,
            key_gain=options.engram_completion ? 1.0f0 : options.key_gain,
        )
        tagged = score_sleep_identity(
            trainer,
            tags,
            seed_blocks,
            seed_ranks,
            options.trajectories,
            options.score_cycle_start,
        )
        run_internal_sleep_trajectories!(
            trainer,
            scratch,
            options.trajectories,
            noise_seed,
            options.internal_noise_scale,
            audit;
            seed_blocks,
            seed_state_keys=shuffled_forward_keys,
            key_fraction=options.engram_completion ? 1.0f0 :
                options.key_fraction,
            key_gain=options.engram_completion ? 1.0f0 : options.key_gain,
        )
        shuffled_cue = score_sleep_identity(
            trainer,
            tags,
            seed_blocks,
            seed_ranks,
            options.trajectories,
            options.score_cycle_start,
        )
        push!(results, (;
            no_cue,
            tagged,
            shuffled_cue,
            tagged_engram_recall=mean(
                tagged_recalled_ranks .== seed_ranks,
            ),
            shuffled_engram_recall=mean(
                shuffled_recalled_ranks .== seed_ranks,
            ),
        ))
    end
    target = mean(result.tagged.target_cosine for result in results)
    no_cue_target = mean(result.no_cue.target_cosine for result in results)
    shuffled_cue_target = mean(
        result.shuffled_cue.target_cosine for result in results
    )
    block_shuffle = mean(
        result.tagged.block_shuffle_cosine for result in results
    )
    key_shuffle = mean(result.tagged.key_shuffle_cosine for result in results)
    temporal_reverse = mean(
        result.tagged.temporal_reverse_cosine for result in results
    )
    top1 = mean(result.tagged.block_top1 for result in results)
    margin = mean(result.tagged.target_margin for result in results)
    rank_top1 = mean(result.tagged.rank_top1 for result in results)
    rank_margin = mean(
        result.tagged.target_rank_margin for result in results
    )
    engram_recall = mean(result.tagged_engram_recall for result in results)
    shuffled_engram_recall = mean(
        result.shuffled_engram_recall for result in results
    )
    parameters_frozen =
        isequal(before_synapse, trainer.parameters.synapse_weight) &&
        isequal(before_key, trainer.parameters.workspace_key) &&
        isequal(before_head, trainer.parameters.head_weight)
    pass =
        parameters_frozen &&
        target >= no_cue_target + 0.02 &&
        target >= shuffled_cue_target + 0.02 &&
        target >= key_shuffle + 0.02 &&
        target >= temporal_reverse + 0.01 &&
        rank_top1 >= 0.50 &&
        rank_margin > 0.0 &&
        (!options.engram_completion || engram_recall >= 0.75)
    output = (;
        schema="reduced-hay-v2-frozen-task-tagged-replay-gate-v1",
        checkpoint=options.checkpoint,
        constraints=(;
            external_rails="strict_zero",
            parameters_frozen=true,
            dataset_reads_during_sleep=0,
            teacher_target_reads_during_sleep=0,
        ),
        options=(;
            trajectories=options.trajectories,
            noise_seeds=options.noise_seeds,
            internal_noise_scale=options.internal_noise_scale,
            key_fraction=options.key_fraction,
            key_gain=options.key_gain,
            score_cycle_start=options.score_cycle_start,
            prototypes_per_block=options.prototypes_per_block,
            engram_completion=options.engram_completion,
        ),
        priority=(;
            selected_blocks=seed_blocks,
            selected_ranks=seed_ranks,
            coverage=priority_coverage,
            raw_maximum=maximum(tags.priority),
            raw_mean=mean(tags.priority),
        ),
        aggregate=(;
            target_cosine=target,
            no_cue_target_cosine=no_cue_target,
            shuffled_cue_target_cosine=shuffled_cue_target,
            block_shuffle_cosine=block_shuffle,
            key_shuffle_cosine=key_shuffle,
            temporal_reverse_cosine=temporal_reverse,
            block_top1=top1,
            target_margin=margin,
            rank_top1,
            target_rank_margin=rank_margin,
            engram_recall,
            shuffled_engram_recall,
        ),
        audit=(;
            zero_rail_checks=audit.zero_rail_checks,
            nonzero_rail_observations=audit.nonzero_rail_observations,
            dataset_reads=audit.dataset_reads,
            teacher_target_reads=audit.teacher_target_reads,
            parameters_frozen,
        ),
        results,
        pass,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println(
        "pass=$(pass) target=$(round(target; digits=6)) " *
        "no_cue=$(round(no_cue_target; digits=6)) " *
        "shuffled_cue=$(round(shuffled_cue_target; digits=6)) " *
        "block_shuffle=$(round(block_shuffle; digits=6)) " *
        "key_shuffle=$(round(key_shuffle; digits=6)) " *
        "reverse=$(round(temporal_reverse; digits=6)) " *
        "block_top1=$(round(top1; digits=6)) " *
        "rank_top1=$(round(rank_top1; digits=6)) " *
        "rank_margin=$(round(rank_margin; digits=6)) " *
        "engram_recall=$(round(engram_recall; digits=6))",
    )
    println("output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
