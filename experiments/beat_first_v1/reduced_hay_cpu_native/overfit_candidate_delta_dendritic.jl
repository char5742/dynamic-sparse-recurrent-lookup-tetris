using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))

module CandidateDeltaDendriticOverfit
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "CompactDendriticNode.jl",
    "DendriticDeltaForestTopology.jl",
    "DendriticDeltaForest.jl",
    "DendriticForestOutput.jl",
    "CandidateDeltaDendriticGraph.jl",
    "CandidateDeltaDendriticTraining.jl",
    "CandidateDeltaDendriticOptimizer.jl",
    "BarrierlessScheduler.jl",
    "CandidateDeltaDendriticBarrierless.jl",
)
    include(joinpath(@__DIR__, file))
end
end

using .BeatFirstTrainingCore

const Canonical = CandidateDeltaDendriticOverfit
const Ranking = Canonical.TetrisRankingBatch
const Model = Canonical.CandidateDeltaDendriticGraph
const Training = Canonical.CandidateDeltaDendriticTraining
const Optimizer = Canonical.CandidateDeltaDendriticOptimizer
const Barrierless = Canonical.CandidateDeltaDendriticBarrierless
const Output = Canonical.DendriticForestOutput
const Topology = Canonical.DendriticDeltaForestTopology

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const CANDIDATE_WIDTH = 80
const ALLOWED_STATE_COUNTS = (8, 16, 64)

Base.@kwdef struct OverfitOptions
    states::Int = 8
    updates::Int = 2_000
    seed::Int = Int(0x48415939)
    log_every::Int = 100
    workers::Int = min(20, Threads.nthreads(:default))
    candidate_chunk::Int = 4
    learning_rate::Float32 = 1.0f-3
    warmup_updates::Int = 100
    min_learning_rate_ratio::Float32 = 0.01f0
    clip_norm::Float32 = 1.0f0
    weight_decay::Float32 = 1.0f-4
    cell_weight_decay::Float32 = 0.0f0
    leaf_cell_multiplier::Float32 = 0.1f0
    forest_internal_multiplier::Float32 = 0.1f0
    forest_contact_multiplier::Float32 = 1.0f0
    program_multiplier::Float32 = 1.0f0
    output_cell_multiplier::Float32 = 0.1f0
    output_anchor_multiplier::Float32 = 1.0f0
    output_context_multiplier::Float32 = 1.0f0
    output_placement_multiplier::Float32 = 1.0f0
    output_cascade_multiplier::Float32 = 1.0f0
    output_gain_multiplier::Float32 = 1.0f0
    output_bias_multiplier::Float32 = 1.0f0
    dataset::String = DEFAULT_DATASET
end

function _usage(io::IO=stdout)
    println(io, "usage: julia --project=. overfit_candidate_delta_dendritic.jl [options]")
    println(io, "  --states 8|16|64              fixed training panel size")
    println(io, "  --updates N                   exact DDF AdamW updates")
    println(io, "  --seed N                      deterministic train-row sampling seed")
    println(io, "  --log-every N                 progress interval")
    println(io, "  --workers N                   1=serial; >1=persistent barrierless team")
    println(io, "  --candidate-chunk N           candidates per MPMC job")
    println(io, "  --learning-rate X             base AdamW learning rate")
    println(io, "  --warmup-updates N            linear warmup cap (default 100)")
    println(io, "  --min-learning-rate-ratio X   cosine floor/base ratio (default 0.01)")
    println(io, "  --clip-norm X                 global gradient clip")
    println(io, "  --weight-decay X              signed-contact/program decay")
    println(io, "  --cell-weight-decay X         bounded cell-raw decay")
    println(io, "  --leaf-cell-multiplier X      shared leaf-cell multiplier")
    println(io, "  --forest-internal-multiplier X")
    println(io, "  --forest-contact-multiplier X")
    println(io, "  --program-multiplier X")
    println(io, "  --output-cell-multiplier X")
    println(io, "  --output-anchor-multiplier X")
    println(io, "  --output-context-multiplier X")
    println(io, "  --output-placement-multiplier X")
    println(io, "  --output-cascade-multiplier X")
    println(io, "  --output-gain-multiplier X")
    println(io, "  --output-bias-multiplier X")
    println(io, "  --dataset PATH                complete teacher_v3 dataset")
end

@inline function _argument_value(args, index::Int, token::String)
    equals = findfirst(==('='), token)
    if !isnothing(equals)
        return token[(equals + 1):end], index
    end
    index < length(args) || error("missing value after $token")
    return args[index + 1], index + 1
end

function parse_options(args=ARGS)
    defaults = OverfitOptions()
    values = Dict{Symbol,Any}(
        name => getfield(defaults, name) for name in fieldnames(OverfitOptions)
    )
    option = Dict(
        "--states" => (:states, Int),
        "--updates" => (:updates, Int),
        "--seed" => (:seed, Int),
        "--log-every" => (:log_every, Int),
        "--workers" => (:workers, Int),
        "--candidate-chunk" => (:candidate_chunk, Int),
        "--learning-rate" => (:learning_rate, Float32),
        "--warmup-updates" => (:warmup_updates, Int),
        "--min-learning-rate-ratio" =>
            (:min_learning_rate_ratio, Float32),
        "--clip-norm" => (:clip_norm, Float32),
        "--weight-decay" => (:weight_decay, Float32),
        "--cell-weight-decay" => (:cell_weight_decay, Float32),
        "--leaf-cell-multiplier" => (:leaf_cell_multiplier, Float32),
        "--forest-internal-multiplier" =>
            (:forest_internal_multiplier, Float32),
        "--forest-contact-multiplier" =>
            (:forest_contact_multiplier, Float32),
        "--program-multiplier" => (:program_multiplier, Float32),
        "--output-cell-multiplier" => (:output_cell_multiplier, Float32),
        "--output-anchor-multiplier" =>
            (:output_anchor_multiplier, Float32),
        "--output-context-multiplier" =>
            (:output_context_multiplier, Float32),
        "--output-placement-multiplier" =>
            (:output_placement_multiplier, Float32),
        "--output-cascade-multiplier" =>
            (:output_cascade_multiplier, Float32),
        "--output-gain-multiplier" => (:output_gain_multiplier, Float32),
        "--output-bias-multiplier" => (:output_bias_multiplier, Float32),
        "--dataset" => (:dataset, String),
    )
    index = 1
    while index <= length(args)
        token = args[index]
        token in ("-h", "--help") && (_usage(); return nothing)
        name = first(split(token, '='; limit=2))
        haskey(option, name) || error("unknown option $token")
        value, index = _argument_value(args, index, token)
        field, type = option[name]
        values[field] = type === String ? String(value) : parse(type, value)
        index += 1
    end
    options = OverfitOptions(;
        (name => values[name] for name in fieldnames(OverfitOptions))...,
    )
    options.states in ALLOWED_STATE_COUNTS || error(
        "--states must be one of $(join(ALLOWED_STATE_COUNTS, ','))",
    )
    options.updates >= 1 || error("--updates must be positive")
    options.log_every >= 1 || error("--log-every must be positive")
    1 <= options.workers <= Threads.nthreads(:default) || error(
        "--workers must be within 1:$(Threads.nthreads(:default))",
    )
    options.candidate_chunk >= 1 || error("--candidate-chunk must be positive")
    options.learning_rate > 0.0f0 || error("--learning-rate must be positive")
    options.warmup_updates >= 0 || error("--warmup-updates must be non-negative")
    0.0f0 <= options.min_learning_rate_ratio <= 1.0f0 || error(
        "--min-learning-rate-ratio must lie in [0,1]",
    )
    options.workers > 1 && Threads.nthreads(:interactive) != 0 && error(
        "barrierless execution requires Julia with no interactive pool; " *
        "start it with --threads=$(Threads.nthreads(:default)),0",
    )
    return options
end

"""Borrow the complete teacher tensors through the canonical width-80 view."""
function load_width80_dataset(path::AbstractString)
    source = BeatFirstTrainingCore.load_teacher_dataset(
        abspath(path);
        max_candidates=BeatFirstTrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    maximum(source.action_counts) <= CANDIDATE_WIDTH || error(
        "teacher dataset contains more than $CANDIDATE_WIDTH legal candidates",
    )
    dataset = Ranking.validate_dataset((;
        boards=source.boards,
        placements=@view(source.placements[:, :, :, 1:CANDIDATE_WIDTH, :]),
        queues=source.queues,
        teacher_q=@view(source.teacher_q[1:CANDIDATE_WIDTH, :]),
        action_counts=source.action_counts,
        selected_actions=source.selected_actions,
        terminal=source.terminal,
        candidate_death=@view(source.candidate_death[1:CANDIDATE_WIDTH, :]),
        candidate_death_available=source.candidate_death_available,
        line_clear=@view(source.line_clear[1:CANDIDATE_WIDTH, :]),
        max_height=@view(source.max_height[1:CANDIDATE_WIDTH, :]),
        holes=@view(source.holes[1:CANDIDATE_WIDTH, :]),
        cavities=@view(source.cavities[1:CANDIDATE_WIDTH, :]),
        ren=source.ren,
        back_to_back=source.back_to_back,
        tspin=@view(source.tspin[1:CANDIDATE_WIDTH, :]),
    ), CANDIDATE_WIDTH)
    return source, dataset
end

"""Select a reproducible training-only overfit panel without replacement."""
function select_train_rows(source, count::Int, seed::Int)
    rows = findall(==(:train), source.predefined_split)
    length(rows) >= count || error("teacher_v3 has fewer than $count train rows")
    shuffle!(MersenneTwister(seed), rows)
    resize!(rows, count)
    sort!(rows)
    return rows
end

function top1_metrics(batch::Ranking.Batch)
    legacy_correct = 0
    tie_correct = 0
    tolerance = 1.0f-6
    @inbounds for state in 1:batch.state_batch
        count = Int(batch.counts[state])
        offset = (state - 1) * batch.width
        predicted = 1
        teacher = 1
        predicted_value = batch.raw[1, offset + 1]
        teacher_value = batch.targets.teacher_q[1, state]
        for candidate in 2:count
            value = batch.raw[1, offset + candidate]
            if value > predicted_value
                predicted = candidate
                predicted_value = value
            end
            target = batch.targets.teacher_q[candidate, state]
            if target > teacher_value
                teacher = candidate
                teacher_value = target
            end
        end
        legacy_correct += predicted == teacher
        tie_correct += batch.targets.teacher_q[predicted, state] >=
            teacher_value - tolerance
    end
    denominator = Float32(batch.state_batch)
    return Float32(legacy_correct) / denominator,
        Float32(tie_correct) / denominator
end

@inline function _accumulate_plane!(
    counters::Vector{Int64},
    stats,
    candidate_delta::Bool,
)
    counters[1] += stats.leaf_hard_events
    counters[2] += stats.internal_hard_events
    counters[3] += stats.root_hard_events
    counters[5] += stats.dirty_leaves + stats.dirty_ancestors
    counters[8] += stats.compact_messages
    if candidate_delta
        counters[6] += stats.dirty_leaves
        counters[7] += stats.dirty_ancestors
    end
    return counters
end

"""Evaluate one fixed hard panel and expose the canonical DDF counters."""
function evaluate!(trainer, batch, dataset)
    Ranking.prepare_batch_metadata!(batch, dataset)
    Training.refresh_model_cache!(trainer)
    counters = zeros(Int64, 8)
    state = trainer.state
    worker = trainer.worker
    parameters = trainer.parameters
    cache = trainer.cache
    width = batch.width
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        _accumulate_plane!(counters, state.before.stats, false)
        _accumulate_plane!(counters, state.after_base.stats, false)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            Model.forward_candidate!(
                @view(batch.raw[:, offset + candidate]),
                worker,
                state,
                parameters,
                cache,
                dataset,
                row,
                candidate,
            )
            _accumulate_plane!(counters, worker.candidate_after.stats, true)
            counters[4] += Output.hard_event_count(worker.output_tape)
        end
    end
    loss = Training.loss_and_raw_gradient!(trainer, batch)
    legacy_top1, tie_top1 = top1_metrics(batch)
    diagnostics = Barrierless.ForwardDiagnostics(counters...)
    denominator = batch.valid_count * Output.hard_event_denominator()
    output_event_rate = denominator == 0 ? 0.0f0 :
        Float32(diagnostics.output_events / denominator)
    return loss, legacy_top1, tie_top1, diagnostics, output_event_rate
end

function optimizer_config(
    options::OverfitOptions;
    learning_rate::Float32=options.learning_rate,
)
    multipliers = Optimizer.GroupLearningRateMultipliers(
        leaf_cell=options.leaf_cell_multiplier,
        forest_internal=options.forest_internal_multiplier,
        forest_contact=options.forest_contact_multiplier,
        program=options.program_multiplier,
        output_cell=options.output_cell_multiplier,
        output_anchor=options.output_anchor_multiplier,
        output_context=options.output_context_multiplier,
        output_placement=options.output_placement_multiplier,
        output_cascade=options.output_cascade_multiplier,
        output_gain=options.output_gain_multiplier,
        output_bias=options.output_bias_multiplier,
    )
    return Optimizer.AdamWConfig(
        learning_rate=learning_rate,
        clip_norm=options.clip_norm,
        weight_decay=options.weight_decay,
        cell_weight_decay=options.cell_weight_decay,
        multipliers=multipliers,
    )
end

function print_progress(
    update::Int,
    loss,
    legacy_top1::Float32,
    tie_top1::Float32,
    gradient_norm::Float64,
    clip_scale::Float32,
    active_program_rows::Int,
    diagnostics::Barrierless.ForwardDiagnostics,
    output_event_rate::Float32,
    started::Float64,
    training_seconds::Float64,
    states::Int,
    learning_rate::Float32,
)
    elapsed = max(time() - started, eps(Float64))
    updates_per_s = update == 0 ? 0.0 :
        update / max(training_seconds, eps(Float64))
    wall_updates_per_s = update == 0 ? 0.0 : update / elapsed
    @printf(
        "candidate_delta_ddf_progress update=%d composite=%.6f excess=%.6f listnet_kl=%.6f legacy_top1=%.6f tie_top1=%.6f learning_rate=%.9g grad_norm=%.6f clip=%.6f active_program_rows=%d leaf_events=%d internal_events=%d root_events=%d output_events=%d output_event_rate=%.8f evaluated_nodes=%d dirty_leaves=%d dirty_ancestors=%d compact_messages=%d training_only_updates_per_s=%.3f training_only_states_per_s=%.3f wall_updates_per_s=%.3f\n",
        update,
        loss.composite_loss,
        loss.composite_loss - loss.teacher_entropy,
        loss.listnet_kl,
        legacy_top1,
        tie_top1,
        learning_rate,
        gradient_norm,
        clip_scale,
        active_program_rows,
        diagnostics.leaf_events,
        diagnostics.internal_events,
        diagnostics.root_events,
        diagnostics.output_events,
        output_event_rate,
        diagnostics.evaluated_nodes,
        diagnostics.dirty_leaves,
        diagnostics.dirty_ancestors,
        diagnostics.compact_messages,
        updates_per_s,
        updates_per_s * states,
        wall_updates_per_s,
    )
    flush(stdout)
end

struct SerialOverfitExecution{T}
    trainer::T
end

struct BarrierlessOverfitExecution{S,E}
    session::S
    executor::E
end

function _training_update!(
    execution::SerialOverfitExecution,
    optimizer,
    parameters,
    config,
    batch,
    dataset,
)
    trainer = execution.trainer
    Training.forward_loss_backward!(trainer, batch, dataset)
    stats = Optimizer.apply_adamw!(
        optimizer,
        parameters,
        trainer.gradient,
        config;
        gradient_scale=1.0,
    )
    Training.refresh_model_cache!(trainer)
    return stats
end

function _training_update!(
    execution::BarrierlessOverfitExecution,
    optimizer,
    parameters,
    config,
    ::Ranking.Batch,
    ::Ranking.ValidatedDataset,
)
    Barrierless.forward_loss_backward!(
        execution.session,
        execution.executor.loss_sink,
    )
    stats = Optimizer.apply_adamw!(
        optimizer,
        parameters,
        execution.executor.gradient,
        config;
        # The ranking loss already scales each supervised cotangent by the
        # state batch. Worker/state gradients are summed without re-averaging.
        gradient_scale=1.0,
    )
    Model.refresh_cache!(execution.executor.cache, parameters)
    return stats
end

function _run_training_updates!(
    execution,
    options::OverfitOptions,
    optimizer,
    parameters,
    config,
    schedule::Optimizer.LearningRateSchedule,
    evaluation_trainer,
    batch,
    dataset,
    started::Float64,
)
    last_stats = Optimizer.AdamWStepStats(0.0, 1.0f0, 0)
    training_seconds = 0.0
    for update in 1:options.updates
        learning_rate = Optimizer.learning_rate_at(schedule, update)
        step_config = Optimizer.AdamWConfig(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            clip_norm=config.clip_norm,
            weight_decay=config.weight_decay,
            cell_weight_decay=config.cell_weight_decay,
            multipliers=config.multipliers,
        )
        update_started = time()
        last_stats = _training_update!(
            execution,
            optimizer,
            parameters,
            step_config,
            batch,
            dataset,
        )
        training_seconds += time() - update_started
        if update % options.log_every == 0 || update == options.updates
            loss, legacy, tied, diagnostics, event_rate =
                evaluate!(evaluation_trainer, batch, dataset)
            print_progress(
                update,
                loss,
                legacy,
                tied,
                last_stats.gradient_norm,
                last_stats.clip_scale,
                last_stats.active_program_rows,
                diagnostics,
                event_rate,
                started,
                training_seconds,
                options.states,
                learning_rate,
            )
        end
    end
    return last_stats, training_seconds
end

function run_overfit(options::OverfitOptions)
    source, dataset = load_width80_dataset(options.dataset)
    rows = select_train_rows(source, options.states, options.seed)
    parameters = Model.initialize_model()
    trainer = Training.ExactBatchTrainer(
        parameters,
        options.states,
        CANDIDATE_WIDTH,
    )
    optimizer = Optimizer.AdamWState(parameters)
    config = optimizer_config(options)
    effective_warmup = min(options.warmup_updates, options.updates ÷ 20)
    schedule = Optimizer.LearningRateSchedule(
        options.learning_rate,
        effective_warmup,
        max(options.updates - effective_warmup, 1),
        options.min_learning_rate_ratio,
    )
    batch = Ranking.Batch(options.states, CANDIDATE_WIDTH)
    copyto!(batch.rows, rows)
    barrierless_executor = options.workers == 1 ? nothing :
        Barrierless.BarrierlessExactExecutor(
            parameters,
            batch,
            dataset;
            worker_capacity=options.workers,
            candidate_chunk_size=options.candidate_chunk,
        )
    execution_kind = options.workers == 1 ? "serial" : "barrierless"

    println(
        "candidate_delta_ddf_start states=", options.states,
        " rows=", join(rows, ','),
        " candidates=", sum(dataset.action_counts[row] for row in rows),
        " seed=", options.seed,
        " leaves=", Topology.LEAF_COUNT,
        " forest_nodes=", Topology.NODE_COUNT,
        " forest_edges=", Topology.CHILD_EDGE_COUNT,
        " anchors=", Topology.ANCHOR_COUNT,
        " output_cells=", Output.OUTPUT_CHANNELS,
        " address_scheme=", Canonical.DendriticProgramBank.ADDRESS_SCHEME,
        " program_rows=", Canonical.DendriticProgramBank.ROW_COUNT,
        " stored_parameters=", Model.stored_parameter_count(parameters),
        " updates=", options.updates,
        " execution=", execution_kind,
        " workers=", options.workers,
        " candidate_chunk=", options.candidate_chunk,
        " optimizer=", config,
        " learning_rate_schedule=", schedule,
        " exact_conditional_reverse=true hard_events=true full_22d_loss=true width=80",
        " throughput_scope=training_only gradient_scale=1.0",
    )
    started = time()
    initial, legacy, tied, diagnostics, event_rate =
        evaluate!(trainer, batch, dataset)
    print_progress(
        0,
        initial,
        legacy,
        tied,
        0.0,
        1.0f0,
        0,
        diagnostics,
        event_rate,
        started,
        0.0,
        options.states,
        0.0f0,
    )

    last_stats, training_seconds = if options.workers == 1
        _run_training_updates!(
            SerialOverfitExecution(trainer),
            options,
            optimizer,
            parameters,
            config,
            schedule,
            trainer,
            batch,
            dataset,
            started,
        )
    else
        Barrierless.run_executor_team!(
            barrierless_executor;
            workers=options.workers,
            queue_capacity=256,
        ) do session
            _run_training_updates!(
                BarrierlessOverfitExecution(session, barrierless_executor),
                options,
                optimizer,
                parameters,
                config,
                schedule,
                trainer,
                batch,
                dataset,
                started,
            )
        end
    end
    final_loss, legacy, tied, diagnostics, event_rate =
        evaluate!(trainer, batch, dataset)
    training_only_updates_per_s = options.updates /
        max(training_seconds, eps(Float64))
    println(
        "candidate_delta_ddf_result states=", options.states,
        " updates=", options.updates,
        " execution=", execution_kind,
        " workers=", options.workers,
        " candidate_chunk=", options.candidate_chunk,
        " composite=", final_loss.composite_loss,
        " excess=", final_loss.composite_loss - final_loss.teacher_entropy,
        " legacy_top1=", legacy,
        " tie_top1=", tied,
        " leaf_events=", diagnostics.leaf_events,
        " internal_events=", diagnostics.internal_events,
        " root_events=", diagnostics.root_events,
        " output_events=", diagnostics.output_events,
        " output_event_rate=", event_rate,
        " evaluated_nodes=", diagnostics.evaluated_nodes,
        " dirty_leaves=", diagnostics.dirty_leaves,
        " dirty_ancestors=", diagnostics.dirty_ancestors,
        " compact_messages=", diagnostics.compact_messages,
        " active_program_rows=", last_stats.active_program_rows,
        " training_seconds=", training_seconds,
        " training_only_updates_per_s=", training_only_updates_per_s,
        " training_only_states_per_s=",
            training_only_updates_per_s * options.states,
    )
    return (;
        trainer,
        barrierless_executor,
        optimizer,
        batch,
        dataset,
        rows,
        final_loss,
        diagnostics,
        training_seconds,
    )
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) || run_overfit(options)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
