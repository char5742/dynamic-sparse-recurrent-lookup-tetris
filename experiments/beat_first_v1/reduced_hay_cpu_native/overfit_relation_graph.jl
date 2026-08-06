using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))

module RelationGraphOverfitCanonical
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SpatialProgramPackets.jl",
    "DendriticRelationTopology.jl",
    "DendriticMotifTopology.jl",
    "TypedDendriticAfferents.jl",
    "HighDimensionalCellPacket.jl",
    "TypedRelationCellBank.jl",
    "TypedRelationContext.jl",
    "TypedOutputCellBank.jl",
    "StructuredMotifReadout.jl",
    "CandidateDeltaRelationGraph.jl",
    "RelationGraphOptimizer.jl",
    "RelationGraphTraining.jl",
    "BarrierlessScheduler.jl",
    "RelationGraphBarrierless.jl",
)
    include(joinpath(@__DIR__, file))
end
end

using .BeatFirstTrainingCore

const Canonical = RelationGraphOverfitCanonical
const Ranking = Canonical.TetrisRankingBatch
const Model = Canonical.CandidateDeltaRelationGraph
const Optimizer = Canonical.RelationGraphOptimizer
const Training = Canonical.RelationGraphTraining
const Barrierless = Canonical.RelationGraphBarrierless

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const CANDIDATE_WIDTH = 80
const ALLOWED_STATE_COUNTS = (1, 8, 64)
const ALLOWED_TRAINERS = (:serial, :barrierless)
const REQUIRED_64_STATE_BATCH = 8
const TAIL_EVALUATIONS = 5
const MIN_GATE_TAIL_EVALUATIONS = 3

Base.@kwdef struct OverfitOptions
    states::Int = 8
    updates::Int = 2_000
    log_every::Int = 100
    eval_every::Int = 100
    batch_size::Int = 0
    learning_rate::Float32 = 1.0f-3
    finish_learning_rate::Float32 = 0.0f0
    finish_at::Int = 0
    seed::Int = Int(0x4844_4344)
    dataset::String = DEFAULT_DATASET
    trainer::Symbol = :serial
    workers::Int = Base.Threads.nthreads(:default)
    candidate_chunk_size::Int = 4
    diagnose_final::Bool = false
end

function _usage(io::IO=stdout)
    println(io, "usage: julia --project=. overfit_relation_graph.jl [options]")
    println(io, "  --states 1|8|64       fixed train-only overfit panel")
    println(io, "  --updates N           exact optimizer updates")
    println(io, "  --log-every N         progress interval")
    println(io, "  --eval-every N        complete-panel evaluation interval")
    println(io, "  --batch-size N        training minibatch (default min(states,8); 64 requires 8)")
    println(io, "  --learning-rate X     fixed AdamW learning rate")
    println(io, "  --finish-learning-rate X  optional second-stage AdamW rate")
    println(io, "  --finish-at N         first update using the finish learning rate")
    println(io, "  --seed N              deterministic train-row selection")
    println(io, "  --dataset PATH        complete teacher_v3 dataset")
    println(io, "  --trainer NAME        serial or barrierless")
    println(io, "  --workers N           persistent barrierless workers")
    println(io, "  --candidate-chunk N   barrierless candidate chunk size")
    println(io, "  --diagnose-final BOOL print per-state ranking-margin diagnostics")
    println(io, "  barrierless launch:   julia --threads=N,0 --project=. ...")
    return nothing
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
    schema = Dict(
        "--states" => (:states, Int),
        "--updates" => (:updates, Int),
        "--log-every" => (:log_every, Int),
        "--eval-every" => (:eval_every, Int),
        "--batch-size" => (:batch_size, Int),
        "--learning-rate" => (:learning_rate, Float32),
        "--finish-learning-rate" => (:finish_learning_rate, Float32),
        "--finish-at" => (:finish_at, Int),
        "--seed" => (:seed, Int),
        "--dataset" => (:dataset, String),
        "--trainer" => (:trainer, Symbol),
        "--workers" => (:workers, Int),
        "--candidate-chunk" => (:candidate_chunk_size, Int),
        "--diagnose-final" => (:diagnose_final, Bool),
    )
    index = 1
    while index <= length(args)
        token = args[index]
        token in ("-h", "--help") && (_usage(); return nothing)
        option_name = first(split(token, '='; limit=2))
        haskey(schema, option_name) || error("unknown option $token")
        value, index = _argument_value(args, index, token)
        field, field_type = schema[option_name]
        values[field] = field_type === String ? String(value) :
            field_type === Symbol ? Symbol(lowercase(value)) :
            parse(field_type, value)
        index += 1
    end
    options = OverfitOptions(;
        (name => values[name] for name in fieldnames(OverfitOptions))...,
    )
    if options.batch_size == 0
        values[:batch_size] = min(options.states, REQUIRED_64_STATE_BATCH)
        options = OverfitOptions(;
            (name => values[name] for name in fieldnames(OverfitOptions))...,
        )
    end
    options.states in ALLOWED_STATE_COUNTS || error(
        "--states must be one of $(join(ALLOWED_STATE_COUNTS, ','))",
    )
    options.updates >= 1 || error("--updates must be positive")
    options.log_every >= 1 || error("--log-every must be positive")
    options.eval_every >= 1 || error("--eval-every must be positive")
    1 <= options.batch_size <= options.states || error(
        "--batch-size must lie within 1:states",
    )
    options.states % options.batch_size == 0 || error(
        "--batch-size must divide the fixed state panel exactly",
    )
    options.states == 64 && options.batch_size != REQUIRED_64_STATE_BATCH &&
        error("the 64-state gate requires --batch-size 8")
    isfinite(options.learning_rate) && options.learning_rate > 0.0f0 ||
        error("--learning-rate must be finite and positive")
    finish_enabled = options.finish_at != 0 ||
        options.finish_learning_rate != 0.0f0
    if finish_enabled
        1 <= options.finish_at <= options.updates || error(
            "--finish-at must lie within 1:updates when finish mode is enabled",
        )
        isfinite(options.finish_learning_rate) &&
            options.finish_learning_rate > 0.0f0 || error(
                "--finish-learning-rate must be finite and positive when " *
                "finish mode is enabled",
            )
    end
    isempty(options.dataset) && error("--dataset cannot be empty")
    options.trainer in ALLOWED_TRAINERS || error(
        "--trainer must be serial or barrierless",
    )
    options.workers >= 1 || error("--workers must be positive")
    options.candidate_chunk_size >= 1 ||
        error("--candidate-chunk must be positive")
    return options
end

"""Load only the canonical width-80 tensor view."""
function load_dataset(path::AbstractString)
    source = BeatFirstTrainingCore.load_teacher_dataset(
        abspath(path);
        max_candidates=BeatFirstTrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    maximum(source.action_counts) <= CANDIDATE_WIDTH || error(
        "teacher dataset contains more than $CANDIDATE_WIDTH candidates",
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

function select_training_rows(source, count::Int, seed::Int)
    rows = findall(==(:train), source.predefined_split)
    length(rows) >= count || error("training split has fewer than $count rows")
    shuffle!(MersenneTwister(seed), rows)
    resize!(rows, count)
    sort!(rows)
    @inbounds for row in rows
        source.predefined_split[row] === :train || error(
            "overfit row $row is not in the training split",
        )
    end
    return rows
end

@inline function _print_progress(update, metrics, elapsed, states)
    updates_per_second = Float64(update) / max(elapsed, eps(Float64))
    @printf(
        "relation_graph_overfit update=%d composite=%.6f excess=%.6f tie_top1=%.6f relation_events=%d motif_events=%d output_events=%d changed=%d affected_positions=%d affected_relations=%d affected_motifs=%d base_contacts=%d candidate_contacts=%d grad_norm=%.6f clip=%.6f active_rows=%d updates_per_s=%.3f states_per_s=%.3f\n",
        update,
        metrics.loss.composite_loss,
        metrics.excess_loss,
        metrics.tie_top1,
        metrics.base_relation_events + metrics.candidate_relation_events,
        metrics.base_motif_events + metrics.candidate_motif_events,
        metrics.base_output_events + metrics.candidate_output_events,
        metrics.changed_positions,
        metrics.affected_positions,
        metrics.affected_relations,
        metrics.affected_motifs,
        metrics.base_contact_visits,
        metrics.candidate_contact_visits,
        metrics.gradient_norm,
        metrics.clip_scale,
        metrics.active_program_rows,
        updates_per_second,
        updates_per_second * states,
    )
    flush(stdout)
    return nothing
end

"""Copy the next deterministic contiguous minibatch from a fixed panel."""
function select_cyclic_minibatch!(
    batch::Ranking.Batch,
    panel_rows::AbstractVector{<:Integer},
    update::Int,
)
    update >= 1 || throw(ArgumentError("update must be positive"))
    length(panel_rows) % batch.state_batch == 0 || throw(DimensionMismatch(
        "panel length must be divisible by the training minibatch",
    ))
    first_row = mod((update - 1) * batch.state_batch, length(panel_rows)) + 1
    last_row = first_row + batch.state_batch - 1
    last_row <= length(panel_rows) || error("cyclic minibatch crossed panel end")
    @inbounds for state_slot in 1:batch.state_batch
        batch.rows[state_slot] = Int(panel_rows[first_row + state_slot - 1])
    end
    return batch
end

"""Fixed storage for complete-panel evaluation without an optimizer update."""
struct PanelEvaluator{F}
    forward_inputs::F
    state::Model.ModelState
    worker::Model.ModelWorker
    batch::Ranking.Batch
    loss_scratch::Ranking.LossScratch
end

function PanelEvaluator(
    dataset::Ranking.ValidatedDataset,
    rows::AbstractVector{<:Integer},
)
    batch = Ranking.Batch(length(rows), CANDIDATE_WIDTH)
    @inbounds for state_slot in eachindex(rows)
        batch.rows[state_slot] = Int(rows[state_slot])
    end
    return PanelEvaluator(
        Training.ForwardInputs(dataset),
        Model.ModelState(),
        Model.ModelWorker(),
        batch,
        Ranking.LossScratch(CANDIDATE_WIDTH, length(rows)),
    )
end

@inline function _evaluation_placement(
    evaluator::PanelEvaluator,
    row::Int,
    candidate::Int,
)
    return @view evaluator.forward_inputs.placements[:, :, 1, candidate, row]
end

@inline function _evaluation_tspin(
    evaluator::PanelEvaluator,
    row::Int,
    candidate::Int,
)
    return @inbounds Float32(evaluator.forward_inputs.tspin[candidate, row])
end

"""Evaluate every fixed panel state under one immutable parameter snapshot."""
function evaluate_panel!(
    evaluator::PanelEvaluator,
    parameters::Model.ModelParameters,
    cache::Model.ModelCache,
    dataset::Ranking.ValidatedDataset,
)
    batch = evaluator.batch
    Ranking.prepare_batch_metadata!(batch, dataset)
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Delta = Model.Delta
        Delta.prepare_state_common!(
            evaluator.state.common,
            evaluator.forward_inputs,
            row,
        )
        Model.prepare_state!(
            evaluator.state,
            evaluator.worker,
            parameters,
            cache,
        )
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * batch.width
        for candidate in 1:count
            Model.forward_candidate!(
                @view(batch.raw[:, offset + candidate]),
                evaluator.worker,
                evaluator.state,
                parameters,
                cache,
                _evaluation_placement(evaluator, row, candidate),
                _evaluation_tspin(evaluator, row, candidate),
            )
        end
    end
    loss = Ranking.supervised_loss_and_raw_gradient!(
        batch,
        evaluator.loss_scratch,
    )
    return (
        loss=loss,
        excess_loss=loss.composite_loss - loss.teacher_entropy,
        tie_top1=Training._tie_top1(batch),
    )
end

"""Print the exact states whose hard top-1 is wrong or nearest to changing.

The diagnostic intentionally reads only an already evaluated fixed panel.  It
does not change the objective, parameters, optimizer, or forward trajectory.
`teacher_set_margin` is the best student score among teacher-optimal candidates
minus the best score outside that set; a negative value is a genuine hard
top-1 error, while a small positive value is vulnerable to minibatch drift.
"""
function print_panel_ranking_diagnostics!(
    evaluator::PanelEvaluator,
    dataset::Ranking.ValidatedDataset;
    nearest::Int=8,
)
    batch = evaluator.batch
    records = NamedTuple[]
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * batch.width

        teacher_best = 1
        predicted_best = 1
        for candidate in 2:count
            batch.targets.teacher_q[candidate, state_slot] >
                batch.targets.teacher_q[teacher_best, state_slot] &&
                (teacher_best = candidate)
            batch.raw[1, offset + candidate] >
                batch.raw[1, offset + predicted_best] &&
                (predicted_best = candidate)
        end
        predicted_second = count == 1 ? 1 : predicted_best == 1 ? 2 : 1
        if count >= 2
            for candidate in 1:count
                candidate == predicted_best && continue
                batch.raw[1, offset + candidate] >
                    batch.raw[1, offset + predicted_second] &&
                    (predicted_second = candidate)
            end
        end

        teacher_max = batch.targets.teacher_q[teacher_best, state_slot]
        teacher_second = -Inf32
        teacher_ties = 0
        best_teacher_set_score = -Inf32
        best_outside_score = -Inf32
        for candidate in 1:count
            teacher_q = batch.targets.teacher_q[candidate, state_slot]
            student_q = batch.raw[1, offset + candidate]
            if teacher_q >= teacher_max - 1.0f-6
                teacher_ties += 1
                best_teacher_set_score = max(best_teacher_set_score, student_q)
            else
                teacher_second = max(teacher_second, teacher_q)
                best_outside_score = max(best_outside_score, student_q)
            end
        end
        teacher_gap = isfinite(teacher_second) ?
            teacher_max - teacher_second : Inf32
        teacher_set_margin = isfinite(best_outside_score) ?
            best_teacher_set_score - best_outside_score : Inf32
        predicted_teacher_gap = teacher_max -
            batch.targets.teacher_q[predicted_best, state_slot]
        model_top_gap = batch.raw[1, offset + predicted_best] -
            batch.raw[1, offset + predicted_second]
        placement_hamming = 0
        if predicted_best != teacher_best
            teacher_placement = @view evaluator.forward_inputs.placements[
                :, :, 1, teacher_best, row,
            ]
            predicted_placement = @view evaluator.forward_inputs.placements[
                :, :, 1, predicted_best, row,
            ]
            for index in eachindex(teacher_placement, predicted_placement)
                placement_hamming +=
                    teacher_placement[index] != predicted_placement[index]
            end
        end
        correct = predicted_teacher_gap <= 1.0f-6
        push!(records, (;
            state_slot,
            row,
            count,
            correct,
            teacher_best,
            predicted_best,
            teacher_ties,
            teacher_gap,
            predicted_teacher_gap,
            teacher_set_margin,
            model_top_gap,
            target_entropy=evaluator.loss_scratch.state_teacher_entropy[
                state_slot,
            ],
            state_excess=evaluator.loss_scratch.state_composite[state_slot] -
                evaluator.loss_scratch.state_teacher_entropy[state_slot],
            teacher_student_q=batch.raw[1, offset + teacher_best],
            predicted_student_q=batch.raw[1, offset + predicted_best],
            placement_hamming,
        ))
    end
    sort!(records; by=record -> (record.correct, record.teacher_set_margin))
    wrong = count(record -> !record.correct, records)
    @printf(
        "relation_graph_panel_diagnostic_summary states=%d wrong=%d nearest=%d\n",
        length(records),
        wrong,
        min(nearest, length(records)),
    )
    printed = 0
    for record in records
        (!record.correct || printed < nearest) || continue
        @printf(
            "relation_graph_panel_diagnostic slot=%d row=%d candidates=%d correct=%s teacher_best=%d predicted_best=%d teacher_ties=%d teacher_gap=%.9g predicted_teacher_gap=%.9g teacher_set_margin=%.9g model_top_gap=%.9g target_entropy=%.9g state_excess=%.9g teacher_student_q=%.9g predicted_student_q=%.9g placement_hamming=%d\n",
            record.state_slot,
            record.row,
            record.count,
            record.correct,
            record.teacher_best,
            record.predicted_best,
            record.teacher_ties,
            record.teacher_gap,
            record.predicted_teacher_gap,
            record.teacher_set_margin,
            record.model_top_gap,
            record.target_entropy,
            record.state_excess,
            record.teacher_student_q,
            record.predicted_student_q,
            record.placement_hamming,
        )
        printed += 1
    end
    flush(stdout)
    return records
end

mutable struct EvaluationTail
    excess::Vector{Float32}
    top1::Vector{Float32}
    count::Int
end

EvaluationTail(capacity::Int=TAIL_EVALUATIONS) = capacity >= 1 ?
    EvaluationTail(zeros(Float32, capacity), zeros(Float32, capacity), 0) :
    throw(ArgumentError("tail capacity must be positive"))

function record_evaluation!(
    tail::EvaluationTail,
    excess::Real,
    top1::Real,
)
    tail.count += 1
    slot = mod1(tail.count, length(tail.excess))
    tail.excess[slot] = Float32(excess)
    tail.top1[slot] = Float32(top1)
    return tail
end

function summarize_tail(tail::EvaluationTail)
    used = min(tail.count, length(tail.excess))
    used >= 1 || error("cannot summarize an empty evaluation tail")
    excess = @view tail.excess[1:used]
    top1 = @view tail.top1[1:used]
    maximum_excess = maximum(excess)
    minimum_top1 = minimum(top1)
    stable = used >= MIN_GATE_TAIL_EVALUATIONS &&
        maximum_excess < 0.05f0 && minimum_top1 >= 1.0f0 - 1.0f-6
    return (
        count=used,
        excess_mean=sum(excess) / Float32(used),
        excess_max=maximum_excess,
        excess_range=maximum_excess - minimum(excess),
        top1_min=minimum_top1,
        stable=stable,
    )
end

@inline function _should_evaluate(update::Int, options::OverfitOptions)
    return update == 1 || update % options.eval_every == 0 ||
           update == options.updates
end

function _run_updates!(
    update_step!::F,
    set_learning_rate!::G,
    parameters::Model.ModelParameters,
    cache::Model.ModelCache,
    dataset::Ranking.ValidatedDataset,
    batch::Ranking.Batch,
    panel_rows::Vector{Int},
    options::OverfitOptions,
) where {F,G}
    evaluator = PanelEvaluator(dataset, panel_rows)
    tail = EvaluationTail()
    start_ns = time_ns()
    final_evaluation = nothing
    for update in 1:options.updates
        if update == options.finish_at
            set_learning_rate!(options.finish_learning_rate)
            @printf(
                "relation_graph_learning_rate_transition update=%d learning_rate=%.9g\n",
                update,
                options.finish_learning_rate,
            )
            flush(stdout)
        end
        select_cyclic_minibatch!(batch, panel_rows, update)
        metrics = update_step!()
        if _should_evaluate(update, options)
            evaluation = evaluate_panel!(evaluator, parameters, cache, dataset)
            final_evaluation = evaluation
            record_evaluation!(
                tail,
                evaluation.excess_loss,
                evaluation.tie_top1,
            )
            elapsed = (time_ns() - start_ns) / 1.0e9
            _print_progress(update, metrics, elapsed, batch.state_batch)
            @printf(
                "relation_graph_panel_eval update=%d panel_states=%d excess=%.6f tie_top1=%.6f tail_observations=%d\n",
                update,
                length(panel_rows),
                evaluation.excess_loss,
                evaluation.tie_top1,
                min(tail.count, length(tail.excess)),
            )
            flush(stdout)
        elseif update % options.log_every == 0
            elapsed = (time_ns() - start_ns) / 1.0e9
            _print_progress(update, metrics, elapsed, batch.state_batch)
        end
    end
    return final_evaluation, summarize_tail(tail), evaluator
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return nothing
    source, dataset = load_dataset(options.dataset)
    panel_rows = select_training_rows(source, options.states, options.seed)
    parameters = Model.initialize_model()
    batch = Ranking.Batch(options.batch_size, CANDIDATE_WIDTH)
    optimizer_config = Optimizer.OptimizerConfig(
        learning_rate=options.learning_rate,
    )

    @printf(
        "relation_graph_overfit_start states=%d batch_size=%d updates=%d lr=%.9g finish_lr=%.9g finish_at=%d parameters=%d trainer=%s workers=%d split=train rows=%s\n",
        options.states,
        options.batch_size,
        options.updates,
        options.learning_rate,
        options.finish_learning_rate,
        options.finish_at,
        Model.stored_parameter_count(parameters),
        options.trainer,
        options.workers,
        join(panel_rows, ','),
    )
    flush(stdout)

    final_evaluation, tail, evaluator = if options.trainer === :serial
        trainer = Training.RelationGraphTrainer(
            parameters,
            dataset,
            options.batch_size,
            CANDIDATE_WIDTH;
            optimizer_config,
        )
        _run_updates!(
            () -> Training.train_update!(trainer, batch),
            learning_rate -> Training.set_learning_rate!(
                trainer,
                learning_rate,
            ),
            parameters,
            trainer.cache,
            dataset,
            batch,
            panel_rows,
            options,
        )
    else
        trainer = Barrierless.BarrierlessRelationGraphTrainer(
            parameters,
            batch,
            dataset;
            optimizer_config,
            worker_capacity=options.workers,
            candidate_chunk_size=options.candidate_chunk_size,
        )
        Barrierless.run_trainer_team!(
            trainer;
            workers=options.workers,
        ) do session
            _run_updates!(
                () -> Barrierless.train_update!(session),
                learning_rate -> Barrierless.set_learning_rate!(
                    trainer,
                    learning_rate,
                ),
                parameters,
                trainer.cache,
                dataset,
                batch,
                panel_rows,
                options,
            )
        end
    end

    options.diagnose_final &&
        print_panel_ranking_diagnostics!(evaluator, dataset)

    gate_pass = options.states != 64 || (
        final_evaluation.tie_top1 >= 1.0f0 - 1.0f-6 &&
        final_evaluation.excess_loss < 0.05f0 && tail.stable
    )
    @printf(
        "relation_graph_overfit_final states=%d batch_size=%d excess=%.6f tie_top1=%.6f tail_count=%d tail_excess_mean=%.6f tail_excess_max=%.6f tail_excess_range=%.6f tail_top1_min=%.6f tail_stable=%s gate_pass=%s\n",
        options.states,
        options.batch_size,
        final_evaluation.excess_loss,
        final_evaluation.tie_top1,
        tail.count,
        tail.excess_mean,
        tail.excess_max,
        tail.excess_range,
        tail.top1_min,
        tail.stable,
        gate_pass,
    )
    flush(stdout)
    gate_pass || error(
        "64-state gate failed: require tie_top1=1, excess<0.05 and stable tail",
    )
    return final_evaluation
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
