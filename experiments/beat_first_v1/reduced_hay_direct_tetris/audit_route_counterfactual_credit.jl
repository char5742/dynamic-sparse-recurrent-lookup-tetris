using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "audit_sleep_proposal_alignment.jl"))

const ROUTE_COUNTERFACTUAL_SEED = UInt64(0x525445434e544652)
const ROUTE_COUNTERFACTUAL_STRIDE = UInt64(0xd6e8feb86659fd93)

function parse_counterfactual_options(arguments)
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
    parse_reals(value) = parse.(Float64, split(value, ','))
    stochastic_value = lowercase(get(values, "stochastic-routing", "false"))
    stochastic_value in ("true", "false") ||
        error("stochastic-routing must be true or false")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        fit_seeds=parse(Int, get(values, "fit-seeds", "32")),
        evaluation_seeds=parse(Int, get(values, "evaluation-seeds", "128")),
        candidates_per_state=parse(
            Int,
            get(values, "candidates-per-state", "4"),
        ),
        stochastic_routing=stochastic_value == "true",
        doses=parse_reals(get(values, "doses", "3e-5,1e-4,3e-4")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "route_counterfactual_credit_gate.json"),
        )),
    )
end

function highest_influence_candidates(base, state_slot, requested)
    count = Int(base.counts[state_slot])
    offset = (state_slot - 1) * base.width
    ranked = Vector{Tuple{Float32,Int}}(undef, count)
    @inbounds for candidate in 1:count
        flat = offset + candidate
        influence = 0.0f0
        for output in axes(base.raw_gradient, 1)
            influence += abs(base.raw_gradient[output, flat])
        end
        ranked[candidate] = (influence, candidate)
    end
    sort!(ranked; by=first, rev=true)
    keep = min(requested, count)
    return ranked[1:keep]
end

function boundary_swap(base, model, flat, cycle)
    selected = 0
    selected_score = Inf32
    challenger = 0
    challenger_score = -Inf32
    @inbounds for block in 1:model.blocks
        score = base.route_score[block, cycle, flat]
        if base.block_mask[block, cycle, flat] != 0.0f0
            if score < selected_score
                selected = block
                selected_score = score
            end
        elseif score > challenger_score
            challenger = block
            challenger_score = score
        end
    end
    selected > 0 || error("counterfactual route has no selected block")
    challenger > 0 || error("counterfactual route has no challenger")
    return (; selected, challenger, margin=selected_score - challenger_score)
end

function forced_swap_prefix(base, model, flat, cycle, swap)
    forced = zeros(Int16, model.workspace_k, model.cycles)
    @inbounds for prefix_cycle in 1:cycle
        for rank in 1:model.workspace_k
            forced[rank, prefix_cycle] =
                base.route_order[rank, prefix_cycle, flat]
        end
    end
    selected_rank = findfirst(
        ==(Int16(swap.selected)),
        view(forced, :, cycle),
    )
    selected_rank === nothing && error("selected block is absent from order")
    forced[selected_rank, cycle] = Int16(swap.challenger)
    return forced
end

function collect_counterfactual_samples!(
    trainer,
    payload,
    rows,
    dataset,
    workers,
    sample_count,
    candidates_per_state,
    stochastic_routing,
)
    model = trainer.model
    states = trainer.tape.base.state_batch
    sample_direction = zeros(
        Float64,
        model.node_dim,
        model.blocks,
        sample_count,
    )
    interventions = states * candidates_per_state
    loss_difference = zeros(Float64, interventions, sample_count)
    selected_block = zeros(Int16, interventions, sample_count)
    challenger_block = zeros(Int16, interventions, sample_count)
    selected_cycle = zeros(Int8, interventions, sample_count)
    selected_candidate = zeros(Int16, interventions, sample_count)
    route_margin = zeros(Float32, interventions, sample_count)
    selected_factor = zeros(
        Float32,
        model.node_dim,
        interventions,
        sample_count,
    )
    challenger_factor = similar(selected_factor)
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing,
        routing_seed=ROUTE_COUNTERFACTUAL_SEED,
        credit_mode=:block_teacher,
    )
    run_with_dendritic_team!(executor) do running
        for sample in 1:sample_count
            restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
            copyto!(trainer.tape.base.rows, rows)
            routing_seed = xor(
                ROUTE_COUNTERFACTUAL_SEED,
                UInt64(sample) * ROUTE_COUNTERFACTUAL_STRIDE,
            )
            running.routing_seed = routing_seed
            reduced_hay_v2_arena_forward!(running)
            base = trainer.tape.base
            gate_density = sum(trainer.cache.gate_probability) /
                Float32(length(trainer.cache.gate_probability))
            trainer.last_loss =
                ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
                    base,
                    trainer.loss_scratch,
                    gate_density,
                    0.0f0,
                )
            baseline_loss = float64_statewise_loss(base)

            @inbounds for state in 1:states
                ranked_candidates = highest_influence_candidates(
                    base,
                    state,
                    candidates_per_state,
                )
                for candidate_rank in eachindex(ranked_candidates)
                    _, candidate = ranked_candidates[candidate_rank]
                    intervention =
                        (state - 1) * candidates_per_state + candidate_rank
                    flat = (state - 1) * base.width + candidate
                    cycle = mod1(
                        sample + state + candidate_rank - 2,
                        model.cycles,
                    )
                swap = boundary_swap(base, model, flat, cycle)
                forced = forced_swap_prefix(
                    base,
                    model,
                    flat,
                    cycle,
                    swap,
                )
                baseline_raw = copy(@view base.raw[:, flat])
                factual_eligibility = copy(view(
                    base.route_eligibility,
                    :,
                    cycle,
                    flat,
                ))
                score_factor = zeros(
                    Float32,
                    model.node_dim,
                    model.blocks,
                )
                for block in 1:model.blocks
                    block_offset = (block - 1) * model.node_dim
                    for coordinate in 1:model.node_dim
                        score_factor[coordinate, block] =
                            trainer.tape.state_query[
                                coordinate,
                                cycle,
                                flat,
                            ] * base.membrane[
                                block_offset + coordinate,
                                cycle + 1,
                                flat,
                            ]
                    end
                end
                nonce = ReducedHayV2ArenaTraining._routing_nonce(
                    routing_seed,
                    trainer.optimizer.step,
                    flat,
                )
                dendritic_forward_candidate!(
                    trainer.tape,
                    model,
                    trainer.parameters,
                    trainer.cache,
                    scratch,
                    trainer.branch_for_edge,
                    flat;
                    stochastic_routing,
                    routing_nonce=nonce,
                    routing_temperature=running.routing_temperature,
                    forced_route_order=forced,
                )
                alternative_loss = float64_statewise_loss(base)
                delta =
                    alternative_loss.composite[state] -
                    baseline_loss.composite[state]
                plastic_delta = min(delta, 0.0)
                    loss_difference[intervention, sample] = delta
                    selected_block[intervention, sample] = Int16(swap.selected)
                    challenger_block[intervention, sample] =
                        Int16(swap.challenger)
                    selected_cycle[intervention, sample] = Int8(cycle)
                    selected_candidate[intervention, sample] = Int16(candidate)
                    route_margin[intervention, sample] = swap.margin
                    for coordinate in 1:model.node_dim
                        selected_factor[coordinate, intervention, sample] =
                            score_factor[coordinate, swap.selected]
                        challenger_factor[coordinate, intervention, sample] =
                            score_factor[coordinate, swap.challenger]
                    end
                for block in 1:model.blocks
                    pair_logpolicy_gradient = if stochastic_routing
                        Float64(
                            factual_eligibility[block] -
                            base.route_eligibility[block, cycle, flat],
                        )
                    elseif block == swap.selected
                        1.0
                    elseif block == swap.challenger
                        -1.0
                    else
                        0.0
                    end
                    for coordinate in 1:model.node_dim
                        sample_direction[
                            coordinate,
                            block,
                            sample,
                        ] = muladd(
                            plastic_delta * pair_logpolicy_gradient,
                            Float64(score_factor[coordinate, block]),
                            sample_direction[
                                coordinate,
                                block,
                                sample,
                            ],
                        )
                    end
                end
                    copyto!(@view(base.raw[:, flat]), baseline_raw)
                end
            end
        end
    end
    return (;
        sample_direction,
        loss_difference,
        selected_block,
        challenger_block,
        selected_cycle,
        selected_candidate,
        route_margin,
        selected_factor,
        challenger_factor,
    )
end

function mean_sample_direction(sample_direction)
    return dropdims(mean(sample_direction; dims=3), dims=3)
end

function shuffled_blocks(direction)
    result = similar(direction)
    blocks = size(direction, 2)
    @inbounds for block in 1:blocks
        result[:, block] .= direction[:, mod1(block + 17, blocks)]
    end
    return result
end

function project_counterfactual_constraints!(
    direction,
    samples;
    passes::Int=12,
)
    before = 0
    after = 0
    maximum_violation_before = 0.0
    maximum_violation_after = 0.0
    interventions, sample_count = size(samples.loss_difference)
    function signed_margin_change(intervention, sample)
        selected = Int(samples.selected_block[intervention, sample])
        challenger = Int(samples.challenger_block[intervention, sample])
        value = 0.0
        @inbounds for coordinate in axes(direction, 1)
            value = muladd(
                Float64(samples.selected_factor[
                    coordinate,
                    intervention,
                    sample,
                ]),
                Float64(direction[coordinate, selected]),
                value,
            )
            value = muladd(
                -Float64(samples.challenger_factor[
                    coordinate,
                    intervention,
                    sample,
                ]),
                Float64(direction[coordinate, challenger]),
                value,
            )
        end
        sign_constraint =
            samples.loss_difference[intervention, sample] < 0.0 ? -1.0 : 1.0
        return sign_constraint * value
    end
    @inbounds for sample in 1:sample_count
        for intervention in 1:interventions
            signed = signed_margin_change(intervention, sample)
            if signed < 0.0
                before += 1
                maximum_violation_before = max(
                    maximum_violation_before,
                    -signed,
                )
            end
        end
    end
    for _ in 1:passes
        @inbounds for sample in 1:sample_count
            for intervention in 1:interventions
                delta = samples.loss_difference[intervention, sample]
                delta == 0.0 && continue
                selected = Int(samples.selected_block[intervention, sample])
                challenger =
                    Int(samples.challenger_block[intervention, sample])
                sign_constraint = delta < 0.0 ? -1.0 : 1.0
                margin_change = 0.0
                norm_square = 0.0
                for coordinate in axes(direction, 1)
                    selected_value = Float64(samples.selected_factor[
                        coordinate,
                        intervention,
                        sample,
                    ])
                    challenger_value = Float64(samples.challenger_factor[
                        coordinate,
                        intervention,
                        sample,
                    ])
                    margin_change = muladd(
                        selected_value,
                        Float64(direction[coordinate, selected]),
                        margin_change,
                    )
                    margin_change = muladd(
                        -challenger_value,
                        Float64(direction[coordinate, challenger]),
                        margin_change,
                    )
                    norm_square = muladd(
                        selected_value,
                        selected_value,
                        norm_square,
                    )
                    norm_square = muladd(
                        challenger_value,
                        challenger_value,
                        norm_square,
                    )
                end
                signed = sign_constraint * margin_change
                signed >= 0.0 && continue
                norm_square > 0.0 || continue
                correction = -signed / norm_square
                for coordinate in axes(direction, 1)
                    direction[coordinate, selected] += correction *
                        sign_constraint * Float64(samples.selected_factor[
                            coordinate,
                            intervention,
                            sample,
                        ])
                    direction[coordinate, challenger] -= correction *
                        sign_constraint * Float64(samples.challenger_factor[
                            coordinate,
                            intervention,
                            sample,
                        ])
                end
            end
        end
    end
    @inbounds for sample in 1:sample_count
        for intervention in 1:interventions
            signed = signed_margin_change(intervention, sample)
            if signed < -1.0e-10
                after += 1
                maximum_violation_after = max(
                    maximum_violation_after,
                    -signed,
                )
            end
        end
    end
    return (;
        violations_before=before,
        violations_after=after,
        maximum_violation_before,
        maximum_violation_after,
        passes,
    )
end

function route_only_proposal(trainer, route)
    return (;
        synapse=zeros(Float32, size(trainer.parameters.synapse_weight)),
        threshold=zeros(
            Float32,
            size(trainer.parameters.soma_threshold_logits),
        ),
        route=Float32.(route),
    )
end

function main_counterfactual(arguments=ARGS)
    options = parse_counterfactual_options(arguments)
    options.fit_seeds >= 4 || error("fit-seeds must be at least four")
    options.evaluation_seeds >= 8 ||
        error("evaluation-seeds must be at least eight")
    options.candidates_per_state >= 1 ||
        error("candidates-per-state must be positive")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    all(>(0.0), options.doses) || error("doses must be positive")
    BLAS.set_num_threads(1)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
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
    samples = collect_counterfactual_samples!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        options.fit_seeds,
        options.candidates_per_state,
        options.stochastic_routing,
    )
    raw_direction = mean_sample_direction(samples.sample_direction)
    direction = copy(raw_direction)
    projection = project_counterfactual_constraints!(direction, samples)
    half = div(options.fit_seeds, 2)
    first_direction = mean_sample_direction(
        view(samples.sample_direction, :, :, 1:half),
    )
    second_direction = mean_sample_direction(
        view(
            samples.sample_direction,
            :,
            :,
            (half + 1):options.fit_seeds,
        ),
    )
    split_half = direction_metrics(first_direction, second_direction)
    shuffled = shuffled_blocks(direction)
    proposal = route_only_proposal(trainer, direction)
    raw_proposal = route_only_proposal(trainer, raw_direction)
    raw_reversed_proposal = route_only_proposal(trainer, .-raw_direction)
    shuffled_proposal = route_only_proposal(trainer, shuffled)
    reversed_proposal = route_only_proposal(trainer, .-direction)
    routing_evaluation_seeds = options.stochastic_routing ?
        options.evaluation_seeds : 0
    baseline = evaluate_proposal_loss!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        proposal,
        0.0,
        (:route,);
        routing_seeds=routing_evaluation_seeds,
    )
    finite_difference = (;
        raw=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            raw_proposal,
            options.doses,
            (:route,);
            routing_seeds=routing_evaluation_seeds,
        ),
        raw_reversed=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            raw_reversed_proposal,
            options.doses,
            (:route,);
            routing_seeds=routing_evaluation_seeds,
        ),
        counterfactual=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            proposal,
            options.doses,
            (:route,);
            routing_seeds=routing_evaluation_seeds,
        ),
        shuffled=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            shuffled_proposal,
            options.doses,
            (:route,);
            routing_seeds=routing_evaluation_seeds,
        ),
        reversed=finite_difference_table!(
            trainer,
            payload,
            rows,
            dataset,
            options.workers,
            reversed_proposal,
            options.doses,
            (:route,);
            routing_seeds=routing_evaluation_seeds,
        ),
    )
    best_benefit(table) = maximum(
        row.available ? row.signed_benefit : -Inf for row in table
    )
    causal_benefit = best_benefit(finite_difference.counterfactual)
    shuffled_benefit = best_benefit(finite_difference.shuffled)
    reversed_benefit = best_benefit(finite_difference.reversed)
    paired_successes = count(eachindex(options.doses)) do index
        causal = finite_difference.counterfactual[index].signed_benefit
        shuffled_value = finite_difference.shuffled[index].signed_benefit
        reversed_value = finite_difference.reversed[index].signed_benefit
        causal_gain = baseline.excess -
            finite_difference.counterfactual[index].positive_excess
        shuffled_gain = baseline.excess -
            finite_difference.shuffled[index].positive_excess
        reversed_gain = baseline.excess -
            finite_difference.reversed[index].positive_excess
        effect_floor = practical_effect_floor(baseline.excess, 0.0)
        control_gain = max(shuffled_gain, reversed_gain, 0.0)
        causal > 0.0 && causal_gain >= effect_floor &&
            causal_gain > 2.0 * control_gain &&
            shuffled_value < causal && reversed_value < causal
    end
    nonzero_fraction = count(!iszero, samples.loss_difference) /
        length(samples.loss_difference)
    positive_gains = [
        baseline.excess - row.positive_excess
        for row in finite_difference.counterfactual
    ]
    effect_floor = practical_effect_floor(baseline.excess, 0.0)
    pass = split_half.cosine > 0.20 &&
        projection.maximum_violation_after <= 1.0e-6 &&
        paired_successes >= 1
    result = (;
        schema="reduced-hay-v2-route-counterfactual-credit-gate-v1",
        checkpoint=options.checkpoint,
        fit_seeds=options.fit_seeds,
        evaluation_seeds=options.evaluation_seeds,
        candidates_per_state=options.candidates_per_state,
        stochastic_routing=options.stochastic_routing,
        semantics=(;
            advantage="global supervised state loss of one forced route swap minus factual loss",
            plasticity="only swaps with lower counterfactual loss update; factual-better pairs are protection anchors",
            block_target="none",
            candidate_sampling="highest absolute supervised output cotangent candidates per state",
            cycle_sampling="deterministic round-robin",
            swap="weakest selected score versus strongest unselected score",
            sleep_tag="signed pairwise route-margin update; no sample retained",
        ),
        diagnostics=(;
            split_half,
            projection,
            projected_to_raw=direction_metrics(direction, raw_direction),
            nonzero_loss_difference_fraction=nonzero_fraction,
            mean_absolute_loss_difference=mean(abs, samples.loss_difference),
            maximum_absolute_loss_difference=maximum(
                abs,
                samples.loss_difference,
            ),
            mean_route_margin=mean(samples.route_margin),
        ),
        finite_difference,
        baseline_excess=baseline.excess,
        practical_effect_floor=effect_floor,
        positive_gains,
        best_signed_benefit=(;
            counterfactual=causal_benefit,
            shuffled=shuffled_benefit,
            reversed=reversed_benefit,
        ),
        paired_successes,
        pass,
    )
    nonfinite = collect_nonfinite!(String[], result)
    isempty(nonfinite) || error(
        "non-finite counterfactual diagnostics: " * join(nonfinite, ", "),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "route counterfactual pass=$pass split=$(split_half.cosine) " *
        "benefit=$causal_benefit shuffled=$shuffled_benefit " *
        "reversed=$reversed_benefit output=$(options.output)",
    )
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_counterfactual()
