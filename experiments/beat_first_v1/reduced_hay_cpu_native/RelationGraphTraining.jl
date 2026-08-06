module RelationGraphTraining

using ..CandidateDeltaInput
using ..CandidateDeltaRelationGraph
using ..DendriticProgramBank
using ..RelationGraphOptimizer
using ..StructuredMotifReadout
using ..TetrisRankingBatch

const Delta = CandidateDeltaInput
const Model = CandidateDeltaRelationGraph
const Bank = DendriticProgramBank
const Optimizer = RelationGraphOptimizer
const Readout = StructuredMotifReadout
const Ranking = TetrisRankingBatch

const READOUT_CONTACTS_PER_SOURCE = sum(
    Readout.output_field_count(output) for output in 1:Readout.OUTPUT_COUNT
)

export RelationGraphTrainer,
       TrainUpdateMetrics,
       set_learning_rate!,
       train_update!

"""
The input-only view used by the model forward path.

This type deliberately has no teacher-Q, target, selected-action, death, or
geometry-label field.  Supervised arrays remain owned by `ValidatedDataset`
and are consumed only by `prepare_batch_metadata!` and the loss kernel.  The
model therefore cannot inspect a teacher while producing a candidate output.
"""
struct ForwardInputs{B,P,Q,R,BT,T}
    boards::B
    placements::P
    queues::Q
    ren::R
    back_to_back::BT
    tspin::T
end

@inline ForwardInputs(dataset::Ranking.ValidatedDataset) = ForwardInputs(
    dataset.boards,
    dataset.placements,
    dataset.queues,
    dataset.ren,
    dataset.back_to_back,
    dataset.tspin,
)

"""One complete pre-update observation of the canonical serial learner."""
struct TrainUpdateMetrics
    loss::Ranking.SupervisedLoss
    excess_loss::Float32
    tie_top1::Float32
    changed_positions::Int64
    affected_positions::Int64
    affected_relations::Int64
    affected_motifs::Int64
    base_relation_events::Int64
    candidate_relation_events::Int64
    base_motif_events::Int64
    candidate_motif_events::Int64
    base_output_events::Int64
    candidate_output_events::Int64
    base_contact_visits::Int64
    candidate_contact_visits::Int64
    gradient_norm::Float64
    clip_scale::Float32
    active_program_rows::Int
end

"""
Fixed-memory serial exact trainer for the high-dimensional relation graph.

`train_update!` is the sole public learning operation.  One invocation runs:

```
all candidate forwards
-> complete 22-D supervised ListNet cotangent
-> deterministic candidate replay and exact conditional reverse
-> one grouped common-state reverse per Tetris state
-> one globally clipped AdamW update
```

The trainer owns the validated teacher dataset, but the forward model receives
only `ForwardInputs`, direct placement bits, and the T-spin input scalar.
"""
mutable struct RelationGraphTrainer{D,F}
    parameters::Model.ModelParameters
    cache::Model.ModelCache
    gradient::Model.ModelGradient
    optimizer_state::Optimizer.AdamWState
    optimizer_config::Optimizer.OptimizerConfig
    teacher_data::D
    forward_inputs::F
    state::Model.ModelState
    worker::Model.ModelWorker
    loss_scratch::Ranking.LossScratch
    state_batch::Int
    width::Int
end

"""Change only future AdamW step size; parameters, moments and clocks persist."""
function set_learning_rate!(trainer::RelationGraphTrainer, learning_rate::Real)
    trainer.optimizer_config = Optimizer.with_learning_rate(
        trainer.optimizer_config,
        learning_rate,
    )
    return trainer
end

@inline function _required_active_program_capacity(
    parameters::Model.ModelParameters,
    state_batch::Int,
    width::Int,
)
    # A state materializes all 480 common before/after packets.  In the
    # adversarial candidate, every one of the 240 after-plane positions can be
    # in the exact changed closure.  Each packet names exactly four program
    # rows.  Repeated semantic addresses only reduce the number of unique rows,
    # so this is a safe pre-update bound.
    common_visits = Base.checked_mul(Model.PROGRAM_SOURCES, Bank.MAX_ACTIVE_ROWS)
    candidate_visits = Base.checked_mul(
        Base.checked_mul(Model.Spatial.POSITION_COUNT, Bank.MAX_ACTIVE_ROWS),
        width,
    )
    batch_visits = Base.checked_mul(
        Base.checked_add(common_visits, candidate_visits),
        state_batch,
    )
    return min(Bank.bank_row_count(parameters.program_bank), batch_visits)
end

function RelationGraphTrainer(
    parameters::Model.ModelParameters,
    dataset::Ranking.ValidatedDataset,
    state_batch::Integer,
    width::Integer=dataset.candidate_width;
    optimizer_config::Optimizer.OptimizerConfig=Optimizer.OptimizerConfig(),
    active_program_capacity::Integer=
        Bank.bank_row_count(parameters.program_bank),
)
    states = Int(state_batch)
    candidates = Int(width)
    states >= 1 || throw(ArgumentError("state_batch must be positive"))
    candidates >= 1 || throw(ArgumentError("width must be positive"))
    candidates == dataset.candidate_width || throw(DimensionMismatch(
        "trainer width $candidates differs from dataset width " *
        "$(dataset.candidate_width)",
    ))
    capacity = Int(active_program_capacity)
    1 <= capacity <= Bank.bank_row_count(parameters.program_bank) ||
        throw(ArgumentError(
            "active_program_capacity must lie within the program bank",
        ))
    required_capacity = _required_active_program_capacity(
        parameters,
        states,
        candidates,
    )
    capacity >= required_capacity || throw(ArgumentError(
        "active_program_capacity $capacity is below the fail-closed batch " *
        "bound $required_capacity",
    ))
    return RelationGraphTrainer(
        parameters,
        Model.ModelCache(parameters),
        Model.ModelGradient(
            parameters;
            active_program_capacity=capacity,
        ),
        Optimizer.AdamWState(parameters),
        optimizer_config,
        dataset,
        ForwardInputs(dataset),
        Model.ModelState(),
        Model.ModelWorker(),
        Ranking.LossScratch(candidates, states),
        states,
        candidates,
    )
end

@inline function _check_batch(
    trainer::RelationGraphTrainer,
    batch::Ranking.Batch,
)
    batch.state_batch == trainer.state_batch || throw(DimensionMismatch(
        "batch state_batch $(batch.state_batch) differs from trainer " *
        "state_batch $(trainer.state_batch)",
    ))
    batch.width == trainer.width || throw(DimensionMismatch(
        "batch width $(batch.width) differs from trainer width " *
        "$(trainer.width)",
    ))
    return nothing
end

@inline function _prepare_state!(
    trainer::RelationGraphTrainer,
    row::Int,
)
    Delta.prepare_state_common!(
        trainer.state.common,
        trainer.forward_inputs,
        row,
    )
    Model.prepare_state!(
        trainer.state,
        trainer.worker,
        trainer.parameters,
        trainer.cache,
    )
    return trainer.state
end

@inline function _placement(trainer::RelationGraphTrainer, row::Int, candidate::Int)
    return @view trainer.forward_inputs.placements[:, :, 1, candidate, row]
end

@inline function _tspin(trainer::RelationGraphTrainer, row::Int, candidate::Int)
    return @inbounds Float32(trainer.forward_inputs.tspin[candidate, row])
end

@inline function _changed_positions(
    state::Model.ModelState,
    worker::Model.ModelWorker,
)
    changed = 0
    @inbounds for column in 1:Delta.BOARD_COLUMNS, row in 1:Delta.BOARD_ROWS
        changed += state.common.board[row, column] !=
            worker.materialization.after[row, column]
    end
    return changed
end

@inline function _base_contact_visits(parameters::Model.ModelParameters)
    Afferents = Model.Afferents
    return Afferents.contact_count(parameters.leaf_relation) +
           Afferents.contact_count(parameters.relation_motif) +
           Afferents.contact_count(parameters.context.common_relation) +
           Afferents.contact_count(parameters.context.common_output) +
           (Model.RELATION_CELLS + Model.MOTIF_CELLS) *
               READOUT_CONTACTS_PER_SOURCE
end

@inline function _candidate_contact_visits(
    parameters::Model.ModelParameters,
    worker::Model.ModelWorker,
    stats::Model.ModelForwardStats,
)
    Afferents = Model.Afferents
    changed_sources = Int(stats.affected_positions)
    changed_relations = Int(stats.affected_relations)
    changed_motifs = Int(stats.affected_motifs)
    placements = Delta.placement_count(worker.delta)
    return changed_sources * parameters.leaf_relation.fanout +
           changed_relations * parameters.relation_motif.fanout +
           (changed_relations + changed_motifs) *
               READOUT_CONTACTS_PER_SOURCE +
           Afferents.contact_count(parameters.context.aux_relation) +
           placements * parameters.placement_relation.fanout
end

@inline function _tie_top1(batch::Ranking.Batch)
    correct = 0
    width = batch.width
    @inbounds for state_slot in 1:batch.state_batch
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        predicted = 1
        teacher_best = 1
        for candidate in 2:count
            batch.raw[1, offset + candidate] >
                batch.raw[1, offset + predicted] && (predicted = candidate)
            batch.targets.teacher_q[candidate, state_slot] >
                batch.targets.teacher_q[teacher_best, state_slot] &&
                (teacher_best = candidate)
        end
        correct += batch.targets.teacher_q[predicted, state_slot] >=
            batch.targets.teacher_q[teacher_best, state_slot] - 1.0f-6
    end
    return Float32(correct) / Float32(batch.state_batch)
end

"""
Run one complete serial exact update.

The forward pass is completed for every candidate before the teacher loss is
evaluated.  During reverse, each candidate is replayed from input-only data so
its fixed worker tape is current; all candidate cotangents accumulate into the
same state-common bars, which `finish_state_pullback!` traverses once.
"""
function train_update!(
    trainer::RelationGraphTrainer,
    batch::Ranking.Batch,
)
    _check_batch(trainer, batch)
    Ranking.prepare_batch_metadata!(batch, trainer.teacher_data)

    state = trainer.state
    worker = trainer.worker
    parameters = trainer.parameters
    cache = trainer.cache
    width = trainer.width

    changed_positions = 0
    affected_positions = 0
    affected_relations = 0
    affected_motifs = 0
    base_relation_events = 0
    candidate_relation_events = 0
    base_motif_events = 0
    candidate_motif_events = 0
    base_output_events = 0
    candidate_output_events = 0
    base_contact_visits = 0
    candidate_contact_visits = 0

    # Forward has no access path to teacher or target arrays.
    @inbounds for state_slot in 1:trainer.state_batch
        row = batch.rows[state_slot]
        _prepare_state!(trainer, row)
        base_contact_visits += _base_contact_visits(parameters)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            Model.forward_candidate!(
                @view(batch.raw[:, offset + candidate]),
                worker,
                state,
                parameters,
                cache,
                _placement(trainer, row, candidate),
                _tspin(trainer, row, candidate),
            )
            stats = Model.forward_stats(state, worker)
            if candidate == 1
                base_relation_events += Int(stats.base_relation_events)
                base_motif_events += Int(stats.base_motif_events)
                base_output_events += Int(stats.base_output_events)
            end
            changed_positions += _changed_positions(state, worker)
            affected_positions += Int(stats.affected_positions)
            affected_relations += Int(stats.affected_relations)
            affected_motifs += Int(stats.affected_motifs)
            candidate_relation_events += Int(stats.candidate_relation_events)
            candidate_motif_events += Int(stats.candidate_motif_events)
            candidate_output_events += Int(stats.candidate_output_events)
            candidate_contact_visits +=
                _candidate_contact_visits(parameters, worker, stats)
        end
    end

    loss = Ranking.supervised_loss_and_raw_gradient!(batch, trainer.loss_scratch)
    tied = _tie_top1(batch)
    Model.clear_gradient!(trainer.gradient)

    # Deterministic replay refreshes candidate-local fixed tape before VJP.
    @inbounds for state_slot in 1:trainer.state_batch
        row = batch.rows[state_slot]
        _prepare_state!(trainer, row)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            flat = offset + candidate
            Model.forward_candidate!(
                @view(batch.raw[:, flat]),
                worker,
                state,
                parameters,
                cache,
                _placement(trainer, row, candidate),
                _tspin(trainer, row, candidate),
            )
            Model.pullback_candidate!(
                trainer.gradient,
                @view(batch.raw_gradient[:, flat]),
                worker,
                state,
                parameters,
                cache,
            )
        end
        Model.finish_state_pullback!(
            trainer.gradient,
            worker,
            state,
            parameters,
            cache,
        )
    end

    step = Optimizer.apply_adamw!(
        trainer.optimizer_state,
        parameters,
        trainer.gradient,
        trainer.optimizer_config,
    )
    Model.refresh_cache!(cache, parameters)
    return TrainUpdateMetrics(
        loss,
        loss.composite_loss - loss.teacher_entropy,
        tied,
        changed_positions,
        affected_positions,
        affected_relations,
        affected_motifs,
        base_relation_events,
        candidate_relation_events,
        base_motif_events,
        candidate_motif_events,
        base_output_events,
        candidate_output_events,
        base_contact_visits,
        candidate_contact_visits,
        step.gradient_norm,
        step.clip_scale,
        step.active_program_rows,
    )
end

end # module RelationGraphTraining
