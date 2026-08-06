using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "audit_route_counterfactual_credit.jl"))

const CAUSAL_SLEEP_ARMS = (
    :wake_only,
    :recurrent_only,
    :route_only,
    :alternating,
    :simultaneous,
)
const CAUSAL_SLEEP_SEED = UInt64(0x43415553414c534c)

function parse_causal_sleep_options(arguments)
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
    promote_checkpoint = get(values, "promote-checkpoint", "")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        trajectories=parse(Int, get(values, "trajectories", "8")),
        replay_seeds=parse(Int, get(values, "replay-seeds", "4")),
        microcycles=parse(Int, get(values, "microcycles", "4")),
        route_fit_seeds=parse(Int, get(values, "route-fit-seeds", "32")),
        candidates_per_state=parse(
            Int,
            get(values, "candidates-per-state", "4"),
        ),
        prototypes_per_block=parse(
            Int,
            get(values, "prototypes-per-block", "4"),
        ),
        internal_noise_scale=parse(
            Float32,
            get(values, "internal-noise-scale", "2.0"),
        ),
        key_fraction=parse(Float32, get(values, "key-fraction", "0.25")),
        recurrent_dose=parse(
            Float32,
            get(values, "recurrent-dose", "0.0003"),
        ),
        route_dose=parse(Float32, get(values, "route-dose", "0.001")),
        tag_credit_mode=Symbol(get(values, "tag-credit-mode", "auto")),
        promote_checkpoint=isempty(promote_checkpoint) ? nothing :
            abspath(promote_checkpoint),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "causal_sleep_arms.json"),
        )),
    )
end

function causal_sleep_trainer(payload)
    model = build_reduced_hay_model(Symbol(payload.run_config.preset))
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    return model, trainer_from_checkpoint(model, parameters, payload)
end

function priority_engram_pairs(state_tags, model, prototypes_per_block)
    pairs = [
        (
            weight=state_tags.prototype_weight[rank, block],
            block,
            rank,
        )
        for block in 1:model.blocks
        for rank in 1:prototypes_per_block
        if state_tags.prototype_weight[rank, block] > 0.0f0
    ]
    sort!(pairs; by=pair -> pair.weight, rev=true)
    return pairs
end

function build_internal_replay_proposal!(
    trainer,
    state_tags,
    parameter_tags,
    priority_pairs,
    options,
    seed_offset::UInt64,
)
    length(priority_pairs) >= options.trajectories ||
        error("not enough wake engrams for sleep trajectories")
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        trainer.model,
        trainer.parameters,
    )
    audit = SleepAudit()
    proposal_sum = zero_proposal(trainer)
    block_load = zeros(Float32, trainer.model.blocks)
    continuation = 0.0
    entropy = 0.0
    pattern_completion = 0.0
    target_margin = 0.0
    for replay_seed in 1:options.replay_seeds
        offset = (replay_seed - 1) * options.trajectories
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
        noise_seed = CAUSAL_SLEEP_SEED ⊻ seed_offset ⊻
            UInt64(replay_seed) * UInt64(0x9e3779b97f4a7c15)
        completed_keys, recalled_ranks = complete_seed_state_keys(
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
            seed_state_keys=completed_keys,
            key_fraction=1.0f0,
            key_gain=1.0f0,
        )
        statistics = raw_sleep_statistics(trainer, options.trajectories)
        proposal = proposal_from_replay(
            trainer,
            parameter_tags,
            state_tags,
            statistics,
            seed_blocks,
            recalled_ranks;
            aggregate_guard=parameter_tags.aggregate,
            use_slow_aggregate_tag=true,
        )
        fill!(proposal.route, 0.0f0)
        add_proposal!(proposal_sum, proposal)
        block_load .+= statistics.block_load
        continuation += statistics.continuation_length
        entropy += statistics.route_entropy
        identity = score_sleep_identity(
            trainer,
            state_tags,
            seed_blocks,
            seed_ranks,
            options.trajectories,
            1,
        )
        pattern_completion += identity.rank_top1
        target_margin += identity.target_rank_margin
    end
    inverse = inv(Float32(options.replay_seeds))
    scale_proposal!(proposal_sum, inverse)
    block_load .*= inverse
    audit.dataset_reads == 0 || error("sleep read the dataset")
    audit.teacher_target_reads == 0 || error("sleep read teacher targets")
    audit.nonzero_rail_observations == 0 ||
        error("sleep observed a nonzero external rail")
    return (;
        proposal=proposal_sum,
        block_load,
        continuation_length=continuation / options.replay_seeds,
        route_entropy=entropy / options.replay_seeds,
        pattern_completion=pattern_completion / options.replay_seeds,
        target_rank_margin=target_margin / options.replay_seeds,
        audit=(;
            zero_rail_checks=audit.zero_rail_checks,
            nonzero_rail_observations=audit.nonzero_rail_observations,
            dataset_reads=audit.dataset_reads,
            teacher_target_reads=audit.teacher_target_reads,
        ),
    )
end

function replay_modulated_route(
    trainer,
    causal_direction,
    replay,
    route_samples,
)
    # A hard route boundary is sensitive to rotating the parameter direction.
    # Internal replay therefore validates the wake-captured causal tag without
    # changing its direction.  Incomplete replay attenuates the whole tag; a
    # completed attractor expresses it at its calibrated wake magnitude.
    completion_gate = clamp(
        Float32(replay.pattern_completion) / 0.50f0,
        0.0f0,
        1.0f0,
    )
    replay.target_rank_margin > 0.0 || (completion_gate = 0.0f0)
    block_modulation = copy(replay.block_load)
    maximum_load = maximum(block_modulation)
    maximum_load > 0.0f0 && (block_modulation ./= maximum_load)
    block_modulation .= 0.90f0 .+ 0.10f0 .* block_modulation
    route = copy(causal_direction)
    @inbounds for block in axes(route, 2)
        @views route[:, block] .*= Float64(
            completion_gate * block_modulation[block],
        )
    end
    project_counterfactual_constraints!(route, route_samples)
    return route_only_proposal(trainer, route)
end

function apply_causal_sleep_proposal!(
    trainer,
    anchor_synapse,
    anchor_threshold,
    anchor_route,
    accumulator,
    recurrent_proposal,
    route_proposal,
    recurrent_dose::Float32,
    route_dose::Float32,
)
    recurrent_maximum = max(
        maximum(abs, recurrent_proposal.synapse),
        maximum(abs, recurrent_proposal.threshold),
    )
    route_maximum = maximum(abs, route_proposal.route)
    recurrent_scale = recurrent_dose == 0.0f0 || recurrent_maximum == 0.0f0 ?
        0.0f0 : recurrent_dose / recurrent_maximum
    route_scale = route_dose == 0.0f0 || route_maximum == 0.0f0 ?
        0.0f0 : route_dose / route_maximum
    # A causal route margin can require a total change smaller than one ULP of
    # an individual Float32 micro-step.  Keep the slow sleep tag in Float64 and
    # materialise anchor + cumulative tag, rather than repeatedly rounding
    # `parameter += tiny_delta` to zero.
    @inbounds for index in eachindex(accumulator.synapse)
        accumulator.synapse[index] = muladd(
            Float64(recurrent_scale),
            Float64(recurrent_proposal.synapse[index]),
            accumulator.synapse[index],
        )
        trainer.parameters.synapse_weight[index] = Float32(
            Float64(anchor_synapse[index]) + accumulator.synapse[index],
        )
    end
    @inbounds for index in eachindex(accumulator.threshold)
        accumulator.threshold[index] = muladd(
            Float64(recurrent_scale),
            Float64(recurrent_proposal.threshold[index]),
            accumulator.threshold[index],
        )
        trainer.parameters.soma_threshold_logits[index] = Float32(
            Float64(anchor_threshold[index]) + accumulator.threshold[index],
        )
    end
    @inbounds for index in eachindex(accumulator.route)
        accumulator.route[index] = muladd(
            Float64(route_scale),
            Float64(route_proposal.route[index]),
            accumulator.route[index],
        )
        trainer.parameters.workspace_key[index] = Float32(
            Float64(anchor_route[index]) + accumulator.route[index],
        )
    end
    ReducedHayV2ArenaTraining.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    return nothing
end

function proposal_split_metrics(first, second)
    return combined_direction_metrics(first, second)
end

function evaluate_causal_sleep_arm!(
    trainer,
    payload,
    rows,
    dataset,
    baseline,
    arm,
    state_tags,
    parameter_tags,
    priority_pairs,
    causal_route_direction,
    route_samples,
    route_confident,
    options,
)
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    before_synapse = copy(trainer.parameters.synapse_weight)
    before_threshold = copy(trainer.parameters.soma_threshold_logits)
    before_route = copy(trainer.parameters.workspace_key)
    before_head = copy(trainer.parameters.head_weight)
    accumulator = (;
        synapse=zeros(Float64, size(before_synapse)),
        threshold=zeros(Float64, size(before_threshold)),
        route=zeros(Float64, size(before_route)),
    )
    pre_replay = build_internal_replay_proposal!(
        trainer,
        state_tags,
        parameter_tags,
        priority_pairs,
        options,
        UInt64(0),
    )
    post_replay = pre_replay
    if arm !== :wake_only && !(arm in CAUSAL_SLEEP_ARMS)
        error("unknown causal sleep arm $arm")
    end
    recurrent_step = options.recurrent_dose / Float32(options.microcycles)
    route_step = options.route_dose / Float32(options.microcycles)
    if arm !== :wake_only
        for microcycle in 1:options.microcycles
            seed_offset = UInt64(microcycle) << 32
            pre_replay = build_internal_replay_proposal!(
                trainer,
                state_tags,
                parameter_tags,
                priority_pairs,
                options,
                seed_offset,
            )
            route_pre = replay_modulated_route(
                trainer,
                route_confident ? causal_route_direction :
                zero(causal_route_direction),
                pre_replay,
                route_samples,
            )
            if arm === :recurrent_only
                apply_causal_sleep_proposal!(
                    trainer,
                    before_synapse,
                    before_threshold,
                    before_route,
                    accumulator,
                    pre_replay.proposal,
                    route_pre,
                    recurrent_step,
                    0.0f0,
                )
            elseif arm === :route_only
                apply_causal_sleep_proposal!(
                    trainer,
                    before_synapse,
                    before_threshold,
                    before_route,
                    accumulator,
                    pre_replay.proposal,
                    route_pre,
                    0.0f0,
                    route_step,
                )
            elseif arm === :alternating
                apply_causal_sleep_proposal!(
                    trainer,
                    before_synapse,
                    before_threshold,
                    before_route,
                    accumulator,
                    pre_replay.proposal,
                    route_pre,
                    recurrent_step,
                    0.0f0,
                )
                post_replay = build_internal_replay_proposal!(
                    trainer,
                    state_tags,
                    parameter_tags,
                    priority_pairs,
                    options,
                    seed_offset ⊻ UInt64(0xa17e2a7e),
                )
                route_post = replay_modulated_route(
                    trainer,
                    route_confident ? causal_route_direction :
                    zero(causal_route_direction),
                    post_replay,
                    route_samples,
                )
                apply_causal_sleep_proposal!(
                    trainer,
                    before_synapse,
                    before_threshold,
                    before_route,
                    accumulator,
                    post_replay.proposal,
                    route_post,
                    0.0f0,
                    route_step,
                )
            else
                # Literal simultaneous control: both proposals are computed
                # from the same pre-update trajectory.
                apply_causal_sleep_proposal!(
                    trainer,
                    before_synapse,
                    before_threshold,
                    before_route,
                    accumulator,
                    pre_replay.proposal,
                    route_pre,
                    recurrent_step,
                    route_step,
                )
            end
        end
    end
    head_delta = parameter_drift(before_head, trainer.parameters.head_weight)
    head_delta.maximum == 0.0f0 || error("sleep changed supervised head")
    after = wake_metrics_and_features!(
        trainer,
        dataset,
        rows,
        options.workers,
    )
    return (;
        before=baseline.metrics,
        after=after.metrics,
        excess_gain=baseline.metrics.excess_loss - after.metrics.excess_loss,
        head_input_drift=relative_feature_drift(
            baseline.features,
            after.features,
        ),
        route_mask_change_fraction=route_mask_drift(
            baseline.route_masks,
            after.route_masks,
        ),
        recurrent_delta=(;
            synapse=parameter_drift(
                before_synapse,
                trainer.parameters.synapse_weight,
            ),
            threshold=parameter_drift(
                before_threshold,
                trainer.parameters.soma_threshold_logits,
            ),
        ),
        route_delta=parameter_drift(
            before_route,
            trainer.parameters.workspace_key,
        ),
        head_delta,
        replay=(;
            before=(;
                continuation_length=pre_replay.continuation_length,
                route_entropy=pre_replay.route_entropy,
                pattern_completion=pre_replay.pattern_completion,
                target_rank_margin=pre_replay.target_rank_margin,
                audit=pre_replay.audit,
            ),
            after_recurrent=(;
                continuation_length=post_replay.continuation_length,
                route_entropy=post_replay.route_entropy,
                pattern_completion=post_replay.pattern_completion,
                target_rank_margin=post_replay.target_rank_margin,
                audit=post_replay.audit,
            ),
        ),
    )
end

function main_causal_sleep(arguments=ARGS)
    options = parse_causal_sleep_options(arguments)
    options.trajectories > 0 || error("trajectories must be positive")
    options.replay_seeds >= 2 || error("replay-seeds must be at least two")
    options.microcycles > 0 || error("microcycles must be positive")
    options.route_fit_seeds >= 4 ||
        error("route-fit-seeds must be at least four")
    options.candidates_per_state > 0 ||
        error("candidates-per-state must be positive")
    options.prototypes_per_block > 0 ||
        error("prototypes-per-block must be positive")
    options.recurrent_dose > 0.0f0 || error("recurrent-dose must be positive")
    options.route_dose > 0.0f0 || error("route-dose must be positive")
    options.tag_credit_mode in (
        :auto,
        :workspace_root_control,
        :workspace_root_reciprocal_control,
        :apical_predictive_online,
    ) || error("unsupported tag-credit-mode")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    BLAS.set_num_threads(1)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    rows = Int.(payload.run_config.overfit_rows)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model, trainer = causal_sleep_trainer(payload)
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    baseline = wake_metrics_and_features!(
        trainer,
        dataset,
        rows,
        options.workers,
    )
    state_tags = capture_wake_state_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        options.prototypes_per_block;
        # The task-dependent wake payload is limited to block-local scalar
        # priority and a low-rank state prototype.  Parameter-specific tags
        # below still come from local apical/e-prop credit, not exact BPTT.
        credit_mode=:exact_bptt,
    )
    checkpoint_credit_mode = hasproperty(payload.run_config, :credit_mode) ?
        Symbol(payload.run_config.credit_mode) : :apical_predictive_online
    tag_credit_mode = options.tag_credit_mode === :auto ?
        (checkpoint_credit_mode === :exact_bptt ?
         :workspace_root_control : :apical_predictive_online) :
        options.tag_credit_mode
    parameter_tags = capture_clustered_parameter_tags!(
        trainer,
        payload,
        dataset,
        rows,
        options.workers,
        state_tags,
        tag_credit_mode,
    )
    priority_pairs = priority_engram_pairs(
        state_tags,
        model,
        options.prototypes_per_block,
    )
    route_samples = collect_counterfactual_samples!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        options.route_fit_seeds,
        options.candidates_per_state,
        false,
    )
    raw_route_direction = mean_sample_direction(
        route_samples.sample_direction,
    )
    causal_route_direction = copy(raw_route_direction)
    route_projection = project_counterfactual_constraints!(
        causal_route_direction,
        route_samples,
    )
    half = div(options.route_fit_seeds, 2)
    route_split = direction_metrics(
        mean_sample_direction(view(
            route_samples.sample_direction,
            :,
            :,
            1:half,
        )),
        mean_sample_direction(view(
            route_samples.sample_direction,
            :,
            :,
            (half + 1):options.route_fit_seeds,
        )),
    )
    causal_validation_proposal = route_only_proposal(
        trainer,
        causal_route_direction,
    )
    shuffled_validation_proposal = route_only_proposal(
        trainer,
        shuffled_blocks(causal_route_direction),
    )
    reversed_validation_proposal = route_only_proposal(
        trainer,
        .-causal_route_direction,
    )
    route_validation_baseline = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        causal_validation_proposal,
        0.0,
        (:route,),
    )
    route_validation_causal = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        causal_validation_proposal,
        options.route_dose,
        (:route,),
    )
    route_validation_shuffled = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        shuffled_validation_proposal,
        options.route_dose,
        (:route,),
    )
    route_validation_reversed = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        reversed_validation_proposal,
        options.route_dose,
        (:route,),
    )
    validation_floor = max(
        1.0e-4,
        0.0025 * route_validation_baseline.excess,
    )
    causal_gain = route_validation_baseline.excess -
        route_validation_causal.excess
    shuffled_gain = route_validation_baseline.excess -
        route_validation_shuffled.excess
    reversed_gain = route_validation_baseline.excess -
        route_validation_reversed.excess
    control_gain = max(shuffled_gain, reversed_gain, 0.0)
    route_confident = route_split.cosine >= 0.50 &&
        route_projection.maximum_violation_after <= 1.0e-6 &&
        causal_gain >= validation_floor &&
        causal_gain > 2.0 * control_gain

    results = Dict{String,Any}()
    for arm in CAUSAL_SLEEP_ARMS
        result = evaluate_causal_sleep_arm!(
            trainer,
            payload,
            rows,
            dataset,
            baseline,
            arm,
            state_tags,
            parameter_tags,
            priority_pairs,
            causal_route_direction,
            route_samples,
            route_confident,
            options,
        )
        results[String(arm)] = result
        println(
            "arm=$arm excess=$(round(result.after.excess_loss; digits=9)) " *
            "gain=$(round(result.excess_gain; digits=9)) " *
            "top1=$(round(result.after.top1; digits=6)) " *
            "rec=$(round(result.recurrent_delta.synapse.maximum; digits=9)) " *
            "route=$(round(result.route_delta.maximum; digits=9)) " *
            "drift=$(round(result.head_input_drift.relative_rms; digits=9)) " *
            "mask=$(round(result.route_mask_change_fraction; digits=9))",
        )
    end
    wake = results["wake_only"]
    alternating = results["alternating"]
    effect_floor = max(1.0e-4, 0.005 * wake.after.excess_loss)
    pass = route_confident &&
        alternating.excess_gain >= effect_floor &&
        alternating.recurrent_delta.synapse.maximum > 0.0f0 &&
        alternating.route_delta.maximum > 0.0f0 &&
        alternating.head_input_drift.relative_rms <= 0.01 &&
        alternating.route_mask_change_fraction <= 0.001 &&
        alternating.after.top1 >= wake.after.top1
    promotion = nothing
    if options.promote_checkpoint !== nothing
        pass || error("refusing to promote a failed causal sleep arm")
        sampler = restore_reduced_hay_v2_checkpoint!(
            trainer,
            payload,
            rows,
        )
        promoted = evaluate_causal_sleep_arm!(
            trainer,
            payload,
            rows,
            dataset,
            baseline,
            :alternating,
            state_tags,
            parameter_tags,
            priority_pairs,
            causal_route_direction,
            route_samples,
            route_confident,
            options,
        )
        promoted_config = merge(
            payload.run_config,
            (;
                sleep_promotion=(;
                    schema="causal-alternating-sleep-v1",
                    source_checkpoint=options.checkpoint,
                    recurrent_dose=options.recurrent_dose,
                    route_dose=options.route_dose,
                    microcycles=options.microcycles,
                    replay_seeds=options.replay_seeds,
                    excess_gain=promoted.excess_gain,
                    head_input_drift=promoted.head_input_drift.relative_rms,
                ),
            ),
        )
        promotion = save_reduced_hay_v2_checkpoint(
            options.promote_checkpoint,
            trainer,
            sampler,
            promoted_config;
            update=Int(payload.update),
        )
        _, restored_trainer = causal_sleep_trainer(
            load_reduced_hay_v2_checkpoint(promotion.path),
        )
        restored_payload = load_reduced_hay_v2_checkpoint(promotion.path)
        restore_reduced_hay_v2_checkpoint!(
            restored_trainer,
            restored_payload,
            rows,
        )
        restored = wake_metrics_and_features!(
            restored_trainer,
            dataset,
            rows,
            options.workers,
        )
        abs(restored.metrics.excess_loss - promoted.after.excess_loss) <=
            5.0e-7 || error("promoted checkpoint changed wake loss")
    end
    output = (;
        schema="reduced-hay-v2-causal-alternating-sleep-arms-v1",
        checkpoint=options.checkpoint,
        checkpoint_update=Int(payload.update),
        semantics=(;
            wake_route="deterministic hard top-k",
            sleep_route="stochastic internal replay; deterministic wake counterfactual margin tag",
            recurrent_tag=String(tag_credit_mode) *
                " low-dimensional root signal times local eligibility",
            block_teacher_target=false,
            external_rails="strict_zero",
            stored_dataset_samples=false,
            world_model=false,
            separate_generator=false,
            supervised_head_frozen=true,
        ),
        options=(;
            trajectories=options.trajectories,
            replay_seeds=options.replay_seeds,
            microcycles=options.microcycles,
            route_fit_seeds=options.route_fit_seeds,
            candidates_per_state=options.candidates_per_state,
            prototypes_per_block=options.prototypes_per_block,
            internal_noise_scale=options.internal_noise_scale,
            key_fraction=options.key_fraction,
            recurrent_dose=options.recurrent_dose,
            route_dose=options.route_dose,
            tag_credit_mode,
        ),
        route_gate=(;
            confident=route_confident,
            split_half=route_split,
            projection=route_projection,
            projected_to_raw=direction_metrics(
                causal_route_direction,
                raw_route_direction,
            ),
            wake_validation=(;
                baseline=route_validation_baseline,
                causal=route_validation_causal,
                shuffled=route_validation_shuffled,
                reversed=route_validation_reversed,
                validation_floor,
                causal_gain,
                shuffled_gain,
                reversed_gain,
            ),
        ),
        effect_floor,
        results,
        promotion,
        pass,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println("pass=$pass output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_causal_sleep()
