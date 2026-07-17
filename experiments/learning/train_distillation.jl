using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Random
using Statistics
using Zygote

include(joinpath(@__DIR__, "compact_model.jl"))
include(joinpath(@__DIR__, "replay.jl"))

function load_teacher_dataset(path)
    return jldopen(path, "r") do file
        action_counts = Int.(file["action_counts"])
        # Preserve every candidate while trimming only never-used fixed-schema
        # padding. The resulting learner tensor is still fixed shape, but does
        # not pay for 128 actions when this dataset's true maximum is smaller.
        effective_max_actions = maximum(action_counts)
        (;
            boards=file["boards"],
            placements=file["placements"][:, :, :, 1:effective_max_actions, :],
            ren=file["ren"],
            back_to_back=file["back_to_back"],
            tspin=file["tspin"][1:effective_max_actions, :],
            queues=file["queues"],
            teacher_q=file["teacher_q"][1:effective_max_actions, :],
            action_counts,
            selected_actions=Int.(file["selected_actions"]),
            rewards=file["rewards"],
            episode_ids=Int.(file["episode_ids"]),
            episode_steps=Int.(file["episode_steps"]),
            terminal=Bool.(file["terminal"]),
            metadata=file["metadata"],
        )
    end
end

flat_index(action::Int, slot::Int, max_actions::Int) = action + (slot - 1) * max_actions

function candidate_batch(dataset, rows::AbstractVector{Int}; teacher_targets::Bool=false)
    max_actions = size(dataset.placements, 4)
    state_count = length(rows)
    flat_count = max_actions * state_count
    board = zeros(Float32, 24, 10, 1, flat_count)
    placement = zeros(Float32, 24, 10, 1, flat_count)
    ren = zeros(Float32, 1, flat_count)
    back_to_back = zeros(Float32, 1, flat_count)
    tspin = zeros(Float32, 1, flat_count)
    queue = zeros(Float32, 7, 6, flat_count)
    mask = zeros(Float32, max_actions, state_count)
    targets = zeros(Float32, max_actions, state_count)
    for (slot, row) in enumerate(rows)
        count = dataset.action_counts[row]
        teacher = @view dataset.teacher_q[1:count, row]
        if teacher_targets
            center = mean(teacher)
            scale = max(std(teacher; corrected=false), 1.0f-4)
            targets[1:count, slot] .= (teacher .- center) ./ scale
        end
        flat_start = flat_index(1, slot, max_actions)
        flat_stop = flat_index(max_actions, slot, max_actions)
        flat_range = flat_start:flat_stop
        board[:, :, :, flat_range] .= reshape(
            Float32.(@view(dataset.boards[:, :, :, row])), 24, 10, 1, 1
        )
        ren[1, flat_range] .= dataset.ren[1, row]
        back_to_back[1, flat_range] .= dataset.back_to_back[1, row]
        queue[:, :, flat_range] .= reshape(
            Float32.(@view(dataset.queues[:, :, row])), 7, 6, 1
        )

        valid_range = flat_start:(flat_start + count - 1)
        placement[:, :, :, valid_range] .= Float32.(
            @view dataset.placements[:, :, :, 1:count, row]
        )
        tspin[1, valid_range] .= @view dataset.tspin[1:count, row]
        mask[1:count, slot] .= 1.0f0
    end
    return (board, placement, ren, back_to_back, tspin, queue), targets, mask
end

function selected_batch(dataset, rows::AbstractVector{Int})
    count = length(rows)
    board = zeros(Float32, 24, 10, 1, count)
    placement = zeros(Float32, 24, 10, 1, count)
    ren = zeros(Float32, 1, count)
    back_to_back = zeros(Float32, 1, count)
    tspin = zeros(Float32, 1, count)
    queue = zeros(Float32, 7, 6, count)
    for (slot, row) in enumerate(rows)
        action = dataset.selected_actions[row]
        board[:, :, :, slot] .= Float32.(@view dataset.boards[:, :, :, row])
        placement[:, :, :, slot] .= Float32.(
            @view dataset.placements[:, :, :, action, row]
        )
        ren[1, slot] = dataset.ren[1, row]
        back_to_back[1, slot] = dataset.back_to_back[1, row]
        tspin[1, slot] = dataset.tspin[action, row]
        queue[:, :, slot] .= Float32.(@view dataset.queues[:, :, row])
    end
    return (board, placement, ren, back_to_back, tspin, queue)
end

function masked_huber(predictions, targets, mask; delta::Float32=1.0f0)
    difference = predictions .- targets
    absolute = abs.(difference)
    loss = ifelse.(
        absolute .<= delta,
        0.5f0 .* difference .^ 2,
        delta .* (absolute .- 0.5f0 * delta),
    )
    return sum(loss .* mask) / max(sum(mask), 1.0f0)
end

"""Per-state standardized, masked ListNet-style cross entropy.

Both teacher and student logits are standardized within each complete candidate
set.  This makes the objective invariant to the arbitrary Q offset/scale of a
state while retaining the full teacher ranking, rather than supervising only
the argmax action.
"""
function standardized_listwise_cross_entropy(
    predictions, standardized_teacher, mask; temperature::Float32=1.0f0
)
    counts = max.(sum(mask; dims=1), 1.0f0)
    prediction_mean = sum(predictions .* mask; dims=1) ./ counts
    centered = (predictions .- prediction_mean) .* mask
    prediction_scale = sqrt.(sum(centered .^ 2; dims=1) ./ counts .+ 1.0f-4)
    prediction_z = centered ./ prediction_scale

    invalid = (1.0f0 .- mask) .* -1.0f4
    teacher_logits = standardized_teacher ./ temperature .+ invalid
    student_logits = prediction_z ./ temperature .+ invalid
    teacher_logits = teacher_logits .- maximum(teacher_logits; dims=1)
    student_logits = student_logits .- maximum(student_logits; dims=1)
    teacher_exp = exp.(teacher_logits) .* mask
    teacher_probability = teacher_exp ./ sum(teacher_exp; dims=1)
    student_log_probability = student_logits .- log.(sum(exp.(student_logits) .* mask; dims=1))
    return -sum(teacher_probability .* student_log_probability .* mask) /
           size(predictions, 2)
end

function distillation_objective(model, parameters, state, batch)
    input, targets, mask, max_actions, state_count, temperature = batch
    raw, next_state = model(input, parameters, state)
    predictions = reshape(raw, max_actions, state_count)
    loss = standardized_listwise_cross_entropy(
        predictions, targets, mask; temperature
    )
    return loss, next_state, (;
        prediction_mean=sum(predictions .* mask) / max(sum(mask), 1.0f0),
        valid_candidates=sum(mask),
    )
end

function td_objective(model, parameters, state, batch)
    input, targets, weights = batch
    predictions, next_state = model(input, parameters, state)
    difference = predictions .- targets
    absolute = abs.(difference)
    element_loss = ifelse.(
        absolute .<= 1.0f0,
        0.5f0 .* difference .^ 2,
        absolute .- 0.5f0,
    )
    loss = sum(element_loss .* reshape(weights, 1, :)) / sum(weights)
    return loss, next_state, (;
        q_mean=mean(predictions), target_mean=mean(targets), td_abs_mean=mean(absolute)
    )
end

function policy_metrics(dataset, rows, model, parameters, state; state_batch::Int=8)
    agreements = Bool[]
    reciprocal_ranks = Float64[]
    correlations = Float64[]
    listwise_losses = Float64[]
    for row_batch in Iterators.partition(rows, state_batch)
        batch_rows = collect(row_batch)
        input, _, _ = candidate_batch(dataset, batch_rows)
        raw, _ = model(input, parameters, state)
        predictions = reshape(raw, size(dataset.placements, 4), length(batch_rows))
        for (slot, row) in enumerate(batch_rows)
            count = dataset.action_counts[row]
            prediction = @view predictions[1:count, slot]
            teacher = @view dataset.teacher_q[1:count, row]
            teacher_best = argmax(teacher)
            push!(agreements, argmax(prediction) == teacher_best)
            ordering = sortperm(prediction; rev=true)
            rank = findfirst(==(teacher_best), ordering)
            push!(reciprocal_ranks, 1.0 / rank)
            if std(prediction; corrected=false) > 1.0f-8 &&
               std(teacher; corrected=false) > 1.0f-8
                push!(correlations, cor(prediction, teacher))
            end
            teacher_z = (teacher .- mean(teacher)) ./
                        max(std(teacher; corrected=false), 1.0f-4)
            prediction_z = (prediction .- mean(prediction)) ./
                           max(std(prediction; corrected=false), 1.0f-4)
            teacher_logits = teacher_z .- maximum(teacher_z)
            prediction_logits = prediction_z .- maximum(prediction_z)
            teacher_probability = exp.(teacher_logits)
            teacher_probability ./= sum(teacher_probability)
            student_log_probability = prediction_logits .- log(sum(exp.(prediction_logits)))
            push!(listwise_losses, -sum(teacher_probability .* student_log_probability))
        end
    end
    return (;
        states=length(rows),
        top1_agreement=mean(agreements),
        mean_reciprocal_rank=mean(reciprocal_ranks),
        mean_correlation=isempty(correlations) ? NaN : mean(correlations),
        mean_listwise_cross_entropy=mean(listwise_losses),
        random_top1=mean(1.0 ./ dataset.action_counts[rows]),
    )
end

json_number(value::Real) = isfinite(value) ? value : nothing

function double_dqn_targets(
    dataset,
    transition_data,
    transition_indices,
    model,
    online_parameters,
    online_state,
    target_parameters,
    target_state,
)
    bootstrap_rows = transition_data.bootstrap_rows[transition_indices]
    safe_rows = [row == 0 ? 1 : row for row in bootstrap_rows]
    input, _, mask = candidate_batch(dataset, safe_rows)
    online_raw, _ = model(input, online_parameters, online_state)
    target_raw, _ = model(input, target_parameters, target_state)
    max_actions, batch_size = size(mask)
    online = reshape(online_raw, max_actions, batch_size)
    target = reshape(target_raw, max_actions, batch_size)
    targets = copy(transition_data.returns[transition_indices])
    for slot in 1:batch_size
        row = bootstrap_rows[slot]
        row == 0 && continue
        count = dataset.action_counts[row]
        best = argmax(@view online[1:count, slot])
        targets[slot] += transition_data.bootstrap_discounts[transition_indices[slot]] *
                         target[best, slot]
    end
    return reshape(Float32.(targets), 1, :)
end

function append_ledger(path, record)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
    end
end

function main()
    dataset_path = get(
        ENV,
        "TEACHER_DATASET_PATH",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_dev_smoke.jld2",
    )
    checkpoint_root = get(
        ENV,
        "LEARNING_CHECKPOINT_ROOT",
        raw"D:\tetris-paper-plus\checkpoints\learning",
    )
    mkpath(checkpoint_root)
    dataset = load_teacher_dataset(dataset_path)
    unique_episodes = sort(unique(dataset.episode_ids))
    length(unique_episodes) >= 2 || error("at least two development episodes are required")
    validation_episode_count = parse(
        Int, get(ENV, "DISTILL_VALIDATION_EPISODES", "2")
    )
    1 <= validation_episode_count < length(unique_episodes) || error(
        "DISTILL_VALIDATION_EPISODES must leave at least one training episode"
    )
    development_episodes = unique_episodes[(end - validation_episode_count + 1):end]
    training_rows = findall(id -> !(id in development_episodes), dataset.episode_ids)
    development_rows = findall(id -> id in development_episodes, dataset.episode_ids)
    isempty(training_rows) && error("empty training split")
    isempty(development_rows) && error("empty development split")

    rng = Xoshiro(parse(UInt64, get(ENV, "LEARNING_SEED", "517812839")))
    channels = parse(Int, get(ENV, "STUDENT_CHANNELS", "8"))
    blocks = parse(Int, get(ENV, "STUDENT_BLOCKS", "1"))
    spatial_channels = parse(Int, get(ENV, "STUDENT_SPATIAL_CHANNELS", "2"))
    state_batch = parse(Int, get(ENV, "DISTILL_STATE_BATCH", "2"))
    distill_steps = parse(Int, get(ENV, "DISTILL_STEPS", "20"))
    td_steps = parse(Int, get(ENV, "TD_STEPS", "20"))
    td_batch = parse(Int, get(ENV, "TD_BATCH", "16"))
    learning_rate = parse(Float32, get(ENV, "LEARNING_RATE", "3e-4"))
    distill_temperature = parse(
        Float32, get(ENV, "DISTILL_TEMPERATURE", "1.0")
    )
    distill_temperature > 0 || error("DISTILL_TEMPERATURE must be positive")
    discount = parse(Float32, get(ENV, "RL_DISCOUNT", "0.997"))
    n_step = parse(Int, get(ENV, "RL_NSTEP", "3"))
    target_interval = parse(Int, get(ENV, "RL_TARGET_INTERVAL", "20"))
    anchor_interval = parse(Int, get(ENV, "RL_ANCHOR_INTERVAL", "5"))

    model = CompactCandidateQ(; channels, blocks, spatial_channels)
    parameters, state = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(
        model, parameters, state, Optimisers.AdamW(learning_rate, (0.9, 0.999), 1.0f-4)
    )
    parameter_count = Lux.parameterlength(parameters)
    initial_metrics = policy_metrics(
        dataset, development_rows, model, train_state.parameters, train_state.states
    )
    started = time()
    distill_losses = Float64[]
    @info "Starting end-to-end listwise distillation" parameter_count initial_metrics

    for update in 1:distill_steps
        rows = rand(rng, training_rows, state_batch)
        input, targets, mask = candidate_batch(dataset, rows; teacher_targets=true)
        batch = (
            input,
            targets,
            mask,
            size(mask, 1),
            size(mask, 2),
            distill_temperature,
        )
        _, loss, statistics, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(), distillation_objective, batch, train_state
        )
        isfinite(loss) || error("non-finite distillation loss at update $update")
        push!(distill_losses, Float64(loss))
        if update == 1 || update % 10 == 0
            @info "Distillation update" update loss=Float64(loss) statistics elapsed=time()-started
        end
    end

    after_distillation_metrics = policy_metrics(
        dataset, development_rows, model, train_state.parameters, train_state.states
    )
    @info "Completed offline distillation validation" after_distillation_metrics elapsed=time()-started
    target_parameters = deepcopy(train_state.parameters)
    target_state = deepcopy(train_state.states)
    transition_data = nstep_transitions(
        dataset.rewards,
        dataset.episode_ids,
        dataset.episode_steps,
        dataset.terminal;
        n=n_step,
        discount,
    )
    replay = PrioritizedIndexReplay(length(training_rows); alpha=0.6, epsilon=1.0e-3)
    initialize!(replay, length(training_rows))
    td_losses = Float64[]

    for update in 1:td_steps
        replay_indices, weights = sample_indices(rng, replay, td_batch; beta=0.4)
        rows = training_rows[replay_indices]
        targets = double_dqn_targets(
            dataset,
            transition_data,
            rows,
            model,
            train_state.parameters,
            train_state.states,
            target_parameters,
            target_state,
        )
        input = selected_batch(dataset, rows)
        _, loss, statistics, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(),
            td_objective,
            (input, targets, weights),
            train_state,
        )
        isfinite(loss) || error("non-finite TD loss at update $update")
        push!(td_losses, Float64(loss))
        predictions, _ = model(
            input, train_state.parameters, train_state.states
        )
        update_priorities!(replay, replay_indices, vec(targets .- predictions))
        if target_interval > 0 && update % target_interval == 0
            target_parameters = deepcopy(train_state.parameters)
            target_state = deepcopy(train_state.states)
        end
        if anchor_interval > 0 && update % anchor_interval == 0
            anchor_rows = rand(rng, training_rows, state_batch)
            anchor_input, anchor_targets, anchor_mask = candidate_batch(
                dataset, anchor_rows; teacher_targets=true
            )
            anchor_batch = (
                anchor_input,
                anchor_targets,
                anchor_mask,
                size(anchor_mask, 1),
                size(anchor_mask, 2),
                distill_temperature,
            )
            _, anchor_loss, _, train_state = Lux.Training.single_train_step!(
                Lux.Training.AutoZygote(),
                distillation_objective,
                anchor_batch,
                train_state,
            )
            isfinite(anchor_loss) || error("non-finite anchor loss")
        end
        if update == 1 || update % 10 == 0
            @info "Offline n-step Double-DQN update" update loss=Float64(loss) statistics priority_total=priority_total(replay.tree) elapsed=time()-started
        end
    end

    final_metrics = policy_metrics(
        dataset, development_rows, model, train_state.parameters, train_state.states
    )
    experiment_tag = get(
        ENV,
        "LEARNING_EXPERIMENT_TAG",
        "compact_q_listwise_$(parameter_count)p_$(distill_steps)steps",
    )
    checkpoint_path = joinpath(checkpoint_root, "$(experiment_tag).jld2")
    jldsave(
        checkpoint_path;
        ps=train_state.parameters,
        st=train_state.states,
        target_ps=target_parameters,
        target_st=target_state,
        model_config=(; channels, blocks, spatial_channels),
        optimizer="AdamW",
        learning_rate,
        distill_temperature,
        discount,
        n_step,
        distill_steps,
        td_steps,
        parameter_count,
        effective_max_actions=size(dataset.placements, 4),
        dataset_path,
        initial_metrics,
        after_distillation_metrics,
        final_metrics,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
    )
    summary = (;
        experiment_id="C01_C02_smoke_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        generated_at=string(now()),
        dataset_path,
        checkpoint_path,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        model_config=(; channels, blocks, spatial_channels),
        parameter_count,
        effective_max_actions=size(dataset.placements, 4),
        backend="Lux+Zygote",
        learning_rate,
        distill_temperature,
        distill_steps,
        td_steps,
        n_step,
        discount,
        target_interval,
        anchor_interval,
        training_episode_ids=sort(unique(dataset.episode_ids[training_rows])),
        development_episode_ids=development_episodes,
        held_out_test_seeds_used=false,
        initial_metrics,
        after_distillation_metrics,
        final_metrics,
        distill_loss_first=isempty(distill_losses) ? nothing : json_number(first(distill_losses)),
        distill_loss_last=isempty(distill_losses) ? nothing : json_number(last(distill_losses)),
        td_loss_first=isempty(td_losses) ? nothing : json_number(first(td_losses)),
        td_loss_last=isempty(td_losses) ? nothing : json_number(last(td_losses)),
        wall_seconds=time() - started,
        completion_reason="configured smoke budgets completed",
    )
    summary_path = replace(checkpoint_path, r"\.jld2$" => ".json")
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
    append_ledger(joinpath(@__DIR__, "ledger.jsonl"), summary)
    @info "Completed learning smoke experiment" summary
end

main()
