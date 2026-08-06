using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "audit_sleep_proposal_alignment.jl"))

const ROUTE_SCORE_FIT_SEED = UInt64(0x5254455343464954)
const ROUTE_SCORE_SEED_STRIDE = UInt64(0xd6e8feb86659fd93)

function parse_route_score_options(arguments)
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
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        fit_seeds=parse(Int, get(values, "fit-seeds", "64")),
        evaluation_seeds=parse(Int, get(values, "evaluation-seeds", "64")),
        doses=parse_reals(get(
            values,
            "doses",
            "1e-6,3e-6,1e-5,3e-5,1e-4",
        )),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "route_score_function_gate.json"),
        )),
    )
end

function route_logpolicy_key_gradient_by_state!(destination, trainer)
    fill!(destination, 0.0)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    @inbounds for state_slot in 1:base.state_batch
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        for candidate in 1:count
            flat = offset + candidate
            for cycle in 1:model.cycles
                for block in 1:model.blocks
                    eligibility = Float64(
                        base.route_eligibility[block, cycle, flat],
                    )
                    block_offset = (block - 1) * model.node_dim
                    for coordinate in 1:model.node_dim
                        state = Float64(base.membrane[
                            block_offset + coordinate,
                            cycle + 1,
                            flat,
                        ])
                        query = Float64(
                            tape.state_query[coordinate, cycle, flat],
                        )
                        destination[
                            coordinate,
                            block,
                            state_slot,
                        ] = muladd(
                            eligibility,
                            state * query,
                            destination[coordinate, block, state_slot],
                        )
                    end
                end
            end
        end
    end
    return destination
end

function root_reward_by_state!(destination, trainer)
    fill!(destination, 0.0)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    @inbounds for state_slot in 1:base.state_batch
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        for candidate in 1:count
            flat = offset + candidate
            for cycle in 1:model.cycles
                for block in 1:model.blocks
                    base.block_mask[block, cycle, flat] == 0.0f0 &&
                        continue
                    destination[state_slot] += Float64(
                        tape.block_supervised_reward[
                            block,
                            cycle,
                            flat,
                        ],
                    )
                end
            end
        end
    end
    return destination
end

function collect_route_score_samples!(
    trainer,
    payload,
    rows,
    dataset,
    workers,
    sample_count,
)
    model = trainer.model
    states = trainer.tape.base.state_batch
    key_loggradient = zeros(
        Float64,
        model.node_dim,
        model.blocks,
        states,
        sample_count,
    )
    composite = zeros(Float64, states, sample_count)
    excess = zeros(Float64, states, sample_count)
    core_route_descent = zeros(
        Float64,
        model.node_dim,
        model.blocks,
        sample_count,
    )
    temporary_loggradient = zeros(
        Float64,
        model.node_dim,
        model.blocks,
        states,
    )
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    trainer.routing_entropy_weight = 0.0f0
    trainer.routing_load_weight = 0.0f0
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=ROUTE_SCORE_FIT_SEED,
        credit_mode=:block_teacher,
    )
    run_with_dendritic_team!(executor) do running
        for sample in 1:sample_count
            restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
            trainer.routing_entropy_weight = 0.0f0
            trainer.routing_load_weight = 0.0f0
            copyto!(trainer.tape.base.rows, rows)
            running.routing_seed = xor(
                ROUTE_SCORE_FIT_SEED,
                UInt64(sample) * ROUTE_SCORE_SEED_STRIDE,
            )
            reduced_hay_v2_arena_forward!(running)
            gate_density = sum(trainer.cache.gate_probability) /
                Float32(length(trainer.cache.gate_probability))
            trainer.last_loss =
                ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
                    trainer.tape.base,
                    trainer.loss_scratch,
                    gate_density,
                    0.0f0,
                )
            statewise = float64_statewise_loss(trainer.tape.base)
            composite[:, sample] .= statewise.composite
            excess[:, sample] .= statewise.excess
            route_logpolicy_key_gradient_by_state!(
                temporary_loggradient,
                trainer,
            )
            @views key_loggradient[:, :, :, sample] .=
                temporary_loggradient
            total_excess = 0.0f0
            for state in 1:states
                total_excess +=
                    trainer.loss_scratch.state_composite[state] -
                    trainer.loss_scratch.state_teacher_entropy[state]
            end
            inverse_other = states > 1 ?
                inv(Float32(states - 1)) : 0.0f0
            for state in 1:states
                state_excess =
                    trainer.loss_scratch.state_composite[state] -
                    trainer.loss_scratch.state_teacher_entropy[state]
                baseline = states > 1 ?
                    (total_excess - state_excess) * inverse_other : 0.0f0
                reward = -(state_excess - baseline)
                for block in 1:model.blocks
                    for coordinate in 1:model.node_dim
                        core_route_descent[coordinate, block, sample] =
                            muladd(
                                Float64(reward),
                                temporary_loggradient[
                                    coordinate,
                                    block,
                                    state,
                                ],
                                core_route_descent[
                                    coordinate,
                                    block,
                                    sample,
                                ],
                            )
                    end
                end
            end
        end
    end
    return (;
        key_loggradient,
        composite,
        excess,
        core_route_descent,
    )
end

function covariance_descent(samples, reward)
    size(samples, 3) == size(reward, 1) ||
        throw(DimensionMismatch("route state axis"))
    size(samples, 4) == size(reward, 2) ||
        throw(DimensionMismatch("route sample axis"))
    destination = zeros(Float64, size(samples, 1), size(samples, 2))
    state_count, sample_count = size(reward)
    inverse_count = inv(Float64(sample_count))
    @inbounds for state in 1:state_count
        baseline = mean(@view reward[state, :])
        for sample in 1:sample_count
            advantage = reward[state, sample] - baseline
            for block in axes(samples, 2)
                for coordinate in axes(samples, 1)
                    destination[coordinate, block] = muladd(
                        advantage * inverse_count,
                        samples[coordinate, block, state, sample],
                        destination[coordinate, block],
                    )
                end
            end
        end
    end
    return destination
end

function sample_average_descent(samples, reward)
    size(samples, 3) == size(reward, 1) ||
        throw(DimensionMismatch("route state axis"))
    size(samples, 4) == size(reward, 2) ||
        throw(DimensionMismatch("route sample axis"))
    destination = zeros(Float64, size(samples, 1), size(samples, 2))
    state_count, sample_count = size(reward)
    inverse_count = inv(Float64(sample_count))
    @inbounds for sample in 1:sample_count
        for state in 1:state_count
            advantage = reward[state, sample]
            for block in axes(samples, 2)
                for coordinate in axes(samples, 1)
                    destination[coordinate, block] = muladd(
                        advantage * inverse_count,
                        samples[coordinate, block, state, sample],
                        destination[coordinate, block],
                    )
                end
            end
        end
    end
    return destination
end

function leave_one_state_out!(destination, values)
    size(destination) == size(values) ||
        throw(DimensionMismatch("state reward buffer"))
    state_count, sample_count = size(values)
    @inbounds for sample in 1:sample_count
        total = sum(@view values[:, sample])
        for state in 1:state_count
            baseline = state_count == 1 ? 0.0 :
                (total - values[state, sample]) / Float64(state_count - 1)
            destination[state, sample] = values[state, sample] - baseline
        end
    end
    return destination
end

function proposal_from_route(trainer, route)
    return (;
        synapse=zeros(Float32, size(trainer.parameters.synapse_weight)),
        threshold=zeros(
            Float32,
            size(trainer.parameters.soma_threshold_logits),
        ),
        route=Float32.(route),
    )
end

function main_route_score(arguments=ARGS)
    options = parse_route_score_options(arguments)
    options.fit_seeds >= 8 || error("fit-seeds must be at least eight")
    options.evaluation_seeds >= 8 ||
        error("evaluation-seeds must be at least eight")
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
    samples = collect_route_score_samples!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        options.fit_seeds,
    )

    # A negative task loss is reward, so covariance(reward, score) is the
    # parameter descent direction.  The workspace-root surrogate is already
    # stored using reward convention by the recurrent VJP.
    oracle_reward = .-samples.composite
    oracle_descent = covariance_descent(
        samples.key_loggradient,
        oracle_reward,
    )
    production_reward = similar(samples.excess)
    production_reward_source = .-samples.excess
    leave_one_state_out!(production_reward, production_reward_source)
    production_descent = sample_average_descent(
        samples.key_loggradient,
        production_reward,
    )
    core_descent = dropdims(
        mean(samples.core_route_descent; dims=3),
        dims=3,
    )
    production_alignment = direction_metrics(
        production_descent,
        oracle_descent,
    )
    core_alignment = direction_metrics(core_descent, oracle_descent)
    core_to_offline = direction_metrics(
        core_descent,
        production_descent,
    )
    half = div(options.fit_seeds, 2)
    oracle_first = covariance_descent(
        @view(samples.key_loggradient[:, :, :, 1:half]),
        @view(oracle_reward[:, 1:half]),
    )
    oracle_second = covariance_descent(
        @view(samples.key_loggradient[:, :, :, (half + 1):end]),
        @view(oracle_reward[:, (half + 1):end]),
    )
    production_first_reward = similar(view(samples.excess, :, 1:half))
    production_second_reward = similar(
        view(samples.excess, :, (half + 1):options.fit_seeds),
    )
    leave_one_state_out!(
        production_first_reward,
        @view(production_reward_source[:, 1:half]),
    )
    leave_one_state_out!(
        production_second_reward,
        @view(production_reward_source[:, (half + 1):end]),
    )
    production_first = sample_average_descent(
        @view(samples.key_loggradient[:, :, :, 1:half]),
        production_first_reward,
    )
    production_second = sample_average_descent(
        @view(samples.key_loggradient[:, :, :, (half + 1):end]),
        production_second_reward,
    )
    split_half = (;
        oracle=direction_metrics(oracle_first, oracle_second),
        production=direction_metrics(
            production_first,
            production_second,
        ),
    )
    oracle_table = finite_difference_table!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        proposal_from_route(trainer, oracle_descent),
        options.doses,
        (:route,);
        routing_seeds=options.evaluation_seeds,
    )
    core_table = finite_difference_table!(
        trainer,
        payload,
        rows,
        dataset,
        options.workers,
        proposal_from_route(trainer, core_descent),
        options.doses,
        (:route,);
        routing_seeds=options.evaluation_seeds,
    )
    positive_count(table) = count(
        row -> row.available && row.signed_benefit > 0.0,
        table,
    )
    oracle_positive = positive_count(oracle_table)
    core_positive = positive_count(core_table)
    required_positive = max(1, length(options.doses) - 1)
    pass = oracle_positive >= required_positive &&
        production_alignment.cosine > 0.0 &&
        core_to_offline.cosine > 0.999 &&
        core_positive >= required_positive
    result = (;
        schema="reduced-hay-v2-route-score-function-gate-v1",
        checkpoint=options.checkpoint,
        fit_seeds=options.fit_seeds,
        evaluation_seeds=options.evaluation_seeds,
        semantics=(;
            route_eligibility="ordered Plackett-Luce dlogP/dscore",
            oracle_reward="negative Float64 Tetris state loss; evaluation only",
            production_reward="negative Float64 state excess with leave-one-state-out baseline",
            baseline="per-state sample mean over independent route draws",
            finite_difference="held routing seed family",
        ),
        alignment=(;
            production_to_oracle=production_alignment,
            core_to_oracle=core_alignment,
            core_to_offline_production=core_to_offline,
            split_half,
        ),
        finite_difference=(;
            oracle=oracle_table,
            integrated_core=core_table,
        ),
        positive_doses=(;
            oracle=oracle_positive,
            integrated_core=core_positive,
            required=required_positive,
        ),
        pass,
    )
    nonfinite = collect_nonfinite!(String[], result)
    isempty(nonfinite) || error(
        "non-finite route score diagnostics: " * join(nonfinite, ", "),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "route score pass=$pass oracle_positive=$oracle_positive " *
        "production_cosine=$(production_alignment.cosine) " *
        "core_positive=$core_positive " *
        "output=$(options.output)",
    )
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_route_score()
