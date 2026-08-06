module RelationGraphBarrierless

using ..BarrierlessScheduler
using ..CandidateDeltaInput
using ..CandidateDeltaRelationGraph
using ..DendriticProgramBank
using ..RelationGraphOptimizer
using ..RelationGraphTraining
using ..TetrisRankingBatch

const SchedulerCore = BarrierlessScheduler
const Delta = CandidateDeltaInput
const Model = CandidateDeltaRelationGraph
const Bank = DendriticProgramBank
const Optimizer = RelationGraphOptimizer
const Serial = RelationGraphTraining
const Ranking = TetrisRankingBatch

export BarrierlessRelationGraphSession,
       BarrierlessRelationGraphTrainer,
       forward_batch!,
       latest_gradient,
       latest_loss,
       run_trainer_team!,
       scheduler_report,
       set_learning_rate!,
       train_update!

const _PREPARE_STATE = UInt16(1)
const _FORWARD_CANDIDATE = UInt16(2)
const _BACKWARD_STATE = UInt16(3)

# Each worker owns one padded diagnostic column.  Sixteen Int64 values keep
# adjacent hot columns 128 bytes apart and reduce false-sharing risk.
const _CHANGED_POSITIONS = 1
const _AFFECTED_POSITIONS = 2
const _AFFECTED_RELATIONS = 3
const _AFFECTED_MOTIFS = 4
const _BASE_RELATION_EVENTS = 5
const _CANDIDATE_RELATION_EVENTS = 6
const _BASE_MOTIF_EVENTS = 7
const _CANDIDATE_MOTIF_EVENTS = 8
const _BASE_OUTPUT_EVENTS = 9
const _CANDIDATE_OUTPUT_EVENTS = 10
const _BASE_CONTACT_VISITS = 11
const _CANDIDATE_CONTACT_VISITS = 12
const _DIAGNOSTIC_STRIDE = 16

const _HOMEOSTASIS_FLOOR_RATIO = 0.25f0
const _HOMEOSTASIS_CEILING_RATIO = 4.0f0

"""
Fixed-memory exact trainer for the canonical candidate-delta relation graph.

Parameters and transformed caches are immutable while a scheduler phase is
active.  A state slot owns its common trajectory and exact gradient.  A native
worker owns only candidate COW/replay scratch.  Consequently candidate forward
may be stolen freely, whereas exact reverse is deliberately one job per state.
"""
mutable struct BarrierlessRelationGraphTrainer{D,F}
    parameters::Model.ModelParameters
    cache::Model.ModelCache
    gradient::Model.ModelGradient
    optimizer_state::Optimizer.AdamWState
    optimizer_config::Optimizer.OptimizerConfig
    teacher_data::D
    forward_inputs::F
    batch::Ranking.Batch
    states::Vector{Model.ModelState}
    workers::Vector{Model.ModelWorker}
    state_gradients::Vector{Model.ModelGradient}
    loss_scratch::Ranking.LossScratch
    loss_sink::Base.RefValue{Ranking.SupervisedLoss}
    state_batch::Int
    width::Int
    candidate_chunk_size::Int
    diagnostic_counters::Matrix{Int64}
end

"""Change only future AdamW step size; parameters, moments and clocks persist."""
function set_learning_rate!(
    trainer::BarrierlessRelationGraphTrainer,
    learning_rate::Real,
)
    trainer.optimizer_config = Optimizer.with_learning_rate(
        trainer.optimizer_config,
        learning_rate,
    )
    return trainer
end

@inline function _state_program_capacity(
    parameters::Model.ModelParameters,
    width::Int,
)
    common_visits = Base.checked_mul(
        Model.PROGRAM_SOURCES,
        Bank.MAX_ACTIVE_ROWS,
    )
    candidate_visits = Base.checked_mul(
        Base.checked_mul(
            Model.Spatial.POSITION_COUNT,
            Bank.MAX_ACTIVE_ROWS,
        ),
        width,
    )
    return min(
        Bank.bank_row_count(parameters.program_bank),
        Base.checked_add(common_visits, candidate_visits),
    )
end

@inline function _batch_program_capacity(
    parameters::Model.ModelParameters,
    state_batch::Int,
    state_capacity::Int,
)
    return min(
        Bank.bank_row_count(parameters.program_bank),
        Base.checked_mul(state_batch, state_capacity),
    )
end

function BarrierlessRelationGraphTrainer(
    parameters::Model.ModelParameters,
    batch::Ranking.Batch,
    dataset::D;
    optimizer_config::Optimizer.OptimizerConfig=Optimizer.OptimizerConfig(),
    worker_capacity::Integer=Base.Threads.nthreads(:default),
    candidate_chunk_size::Integer=4,
    state_program_capacity::Union{Nothing,Integer}=nothing,
    batch_program_capacity::Union{Nothing,Integer}=nothing,
) where {D<:Ranking.ValidatedDataset}
    batch.width == dataset.candidate_width || throw(DimensionMismatch(
        "batch width $(batch.width) differs from dataset width " *
        "$(dataset.candidate_width)",
    ))
    state_batch = batch.state_batch
    width = batch.width
    worker_count = Int(worker_capacity)
    worker_count >= 1 || throw(ArgumentError(
        "worker_capacity must be positive",
    ))
    chunk = Int(candidate_chunk_size)
    chunk >= 1 || throw(ArgumentError(
        "candidate_chunk_size must be positive",
    ))

    required_state_capacity = _state_program_capacity(parameters, width)
    state_capacity = isnothing(state_program_capacity) ?
        required_state_capacity : Int(state_program_capacity)
    required_batch_capacity = _batch_program_capacity(
        parameters,
        state_batch,
        required_state_capacity,
    )
    batch_capacity = isnothing(batch_program_capacity) ?
        required_batch_capacity : Int(batch_program_capacity)
    row_count = Bank.bank_row_count(parameters.program_bank)
    required_state_capacity <= state_capacity <= row_count ||
        throw(ArgumentError(
            "state_program_capacity $state_capacity is below the " *
            "fail-closed bound $required_state_capacity or exceeds the bank",
        ))
    required_batch_capacity <= batch_capacity <= row_count ||
        throw(ArgumentError(
            "batch_program_capacity $batch_capacity is below the " *
            "fail-closed bound $required_batch_capacity or exceeds the bank",
        ))

    return BarrierlessRelationGraphTrainer(
        parameters,
        Model.ModelCache(parameters),
        Model.ModelGradient(
            parameters;
            active_program_capacity=batch_capacity,
        ),
        Optimizer.AdamWState(parameters),
        optimizer_config,
        dataset,
        Serial.ForwardInputs(dataset),
        batch,
        [Model.ModelState() for _ in 1:state_batch],
        [Model.ModelWorker() for _ in 1:worker_count],
        [Model.ModelGradient(
            parameters;
            active_program_capacity=state_capacity,
        ) for _ in 1:state_batch],
        Ranking.LossScratch(width, state_batch),
        Ref{Ranking.SupervisedLoss}(),
        state_batch,
        width,
        chunk,
        zeros(Int64, _DIAGNOSTIC_STRIDE, worker_count),
    )
end

@inline function _placement(
    trainer::BarrierlessRelationGraphTrainer,
    row::Int,
    candidate::Int,
)
    return @view trainer.forward_inputs.placements[:, :, 1, candidate, row]
end

@inline function _tspin(
    trainer::BarrierlessRelationGraphTrainer,
    row::Int,
    candidate::Int,
)
    return @inbounds Float32(trainer.forward_inputs.tspin[candidate, row])
end

@inline function _prepare_state_core!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    state_slot::Int,
)
    row = @inbounds trainer.batch.rows[state_slot]
    state = @inbounds trainer.states[state_slot]
    worker = @inbounds trainer.workers[worker_slot]
    Delta.prepare_state_common!(
        state.common,
        trainer.forward_inputs,
        row,
    )
    Model.prepare_state!(
        state,
        worker,
        trainer.parameters,
        trainer.cache,
    )
    return state
end

@inline function _record_base_diagnostics!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    state::Model.ModelState,
)
    counters = trainer.diagnostic_counters
    @inbounds begin
        counters[_BASE_RELATION_EVENTS, worker_slot] +=
            Model.Relations.hard_event_count(state.relation_tape)
        counters[_BASE_MOTIF_EVENTS, worker_slot] +=
            Model.Relations.hard_event_count(state.motif_tape)
        counters[_BASE_OUTPUT_EVENTS, worker_slot] +=
            Model.Outputs.hard_event_count(state.output_tape)
        counters[_BASE_CONTACT_VISITS, worker_slot] +=
            Serial._base_contact_visits(trainer.parameters)
    end
    return nothing
end

@inline function _prepare_state!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    state_slot::Int,
)
    state = _prepare_state_core!(trainer, worker_slot, state_slot)
    _record_base_diagnostics!(trainer, worker_slot, state)
    return nothing
end

@inline function _forward_candidate_range!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    first_ordinal::Int,
    last_ordinal::Int,
)
    batch = trainer.batch
    worker = @inbounds trainer.workers[worker_slot]
    counters = trainer.diagnostic_counters
    @inbounds for ordinal in first_ordinal:last_ordinal
        flat = Int(batch.valid_flats[ordinal])
        state_slot, candidate = Ranking.state_candidate(flat, trainer.width)
        row = batch.rows[state_slot]
        state = trainer.states[state_slot]
        Model.forward_candidate!(
            @view(batch.raw[:, flat]),
            worker,
            state,
            trainer.parameters,
            trainer.cache,
            _placement(trainer, row, candidate),
            _tspin(trainer, row, candidate),
        )
        stats = Model.forward_stats(state, worker)
        counters[_CHANGED_POSITIONS, worker_slot] +=
            Serial._changed_positions(state, worker)
        counters[_AFFECTED_POSITIONS, worker_slot] +=
            Int(stats.affected_positions)
        counters[_AFFECTED_RELATIONS, worker_slot] +=
            Int(stats.affected_relations)
        counters[_AFFECTED_MOTIFS, worker_slot] +=
            Int(stats.affected_motifs)
        counters[_CANDIDATE_RELATION_EVENTS, worker_slot] +=
            Int(stats.candidate_relation_events)
        counters[_CANDIDATE_MOTIF_EVENTS, worker_slot] +=
            Int(stats.candidate_motif_events)
        counters[_CANDIDATE_OUTPUT_EVENTS, worker_slot] +=
            Int(stats.candidate_output_events)
        counters[_CANDIDATE_CONTACT_VISITS, worker_slot] +=
            Serial._candidate_contact_visits(trainer.parameters, worker, stats)
    end
    return nothing
end

@inline function _backward_state!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    state_slot::Int,
)
    batch = trainer.batch
    state = @inbounds trainer.states[state_slot]
    worker = @inbounds trainer.workers[worker_slot]
    gradient = @inbounds trainer.state_gradients[state_slot]
    Model.clear_gradient!(gradient)

    # Rebuild the common tape exactly as the serial oracle does.  This is a
    # small state-level cost and keeps the initial barrierless implementation
    # maximally conservative; candidate tapes are still replayed one at a time.
    _prepare_state_core!(trainer, worker_slot, state_slot)
    row = @inbounds batch.rows[state_slot]
    count = Int(@inbounds batch.counts[state_slot])
    offset = (state_slot - 1) * trainer.width
    @inbounds for candidate in 1:count
        flat = offset + candidate
        Model.forward_candidate!(
            @view(batch.raw[:, flat]),
            worker,
            state,
            trainer.parameters,
            trainer.cache,
            _placement(trainer, row, candidate),
            _tspin(trainer, row, candidate),
        )
        Model.pullback_candidate!(
            gradient,
            @view(batch.raw_gradient[:, flat]),
            worker,
            state,
            trainer.parameters,
            trainer.cache,
        )
    end
    Model.finish_state_pullback!(
        gradient,
        worker,
        state,
        trainer.parameters,
        trainer.cache,
    )
    return nothing
end

function SchedulerCore.dispatch_work!(
    trainer::BarrierlessRelationGraphTrainer,
    worker_slot::Int,
    item::SchedulerCore.WorkItem,
)
    phase = item.phase
    first = Int(item.first)
    last = Int(item.last)
    if phase == _PREPARE_STATE
        @inbounds for state_slot in first:last
            _prepare_state!(trainer, worker_slot, state_slot)
        end
    elseif phase == _FORWARD_CANDIDATE
        _forward_candidate_range!(trainer, worker_slot, first, last)
    elseif phase == _BACKWARD_STATE
        @inbounds for state_slot in first:last
            _backward_state!(trainer, worker_slot, state_slot)
        end
    else
        throw(ArgumentError("unknown relation-graph phase $phase"))
    end
    return nothing
end

struct BarrierlessRelationGraphSession{E,S}
    trainer::E
    scheduler::S
end

function run_trainer_team!(
    body::F,
    trainer::BarrierlessRelationGraphTrainer;
    workers::Integer=Base.Threads.nthreads(:default),
    queue_capacity::Integer=64,
    binding_mode::Symbol=:none,
) where {F}
    worker_count = Int(workers)
    worker_count <= length(trainer.workers) || throw(ArgumentError(
        "workers exceeds trainer worker_capacity $(length(trainer.workers))",
    ))
    scheduler = SchedulerCore.Scheduler(
        trainer;
        workers=worker_count,
        queue_capacity,
        binding_mode,
    )
    return SchedulerCore.run_team!(scheduler) do active_scheduler
        body(BarrierlessRelationGraphSession(trainer, active_scheduler))
    end
end

@inline scheduler_report(session::BarrierlessRelationGraphSession) =
    SchedulerCore.teardown_report(session.scheduler)

@inline latest_gradient(session::BarrierlessRelationGraphSession) =
    session.trainer.gradient

@inline latest_loss(session::BarrierlessRelationGraphSession) =
    session.trainer.loss_sink[]

@inline function _check_rows(trainer::BarrierlessRelationGraphTrainer)
    @inbounds for state_slot in 1:trainer.state_batch
        row = trainer.batch.rows[state_slot]
        1 <= row <= trainer.teacher_data.state_count || throw(BoundsError(
            1:trainer.teacher_data.state_count,
            row,
        ))
    end
    return nothing
end

@inline function _add_array!(destination, source)
    @inbounds @simd for index in eachindex(destination, source)
        destination[index] += source[index]
    end
    return destination
end

@inline function _merge_sparse_program!(
    destination::Bank.SparseProgramGradient,
    source::Bank.SparseProgramGradient,
)
    @inbounds for slot in 1:Bank.active_gradient_count(source)
        row = Int(Bank.active_gradient_row(source, slot))
        Bank.accumulate_program_gradient!(
            destination,
            row,
            @view(source.values[:, slot]),
            1.0f0,
        )
    end
    return destination
end

@inline function _merge_gradient!(
    destination::Model.ModelGradient,
    source::Model.ModelGradient,
)
    _merge_sparse_program!(destination.program, source.program)
    _add_array!(destination.leaf_relation, source.leaf_relation)
    Model.Relations.accumulate_gradient!(
        destination.relation,
        source.relation,
    )
    _add_array!(destination.relation_motif, source.relation_motif)
    Model.Relations.accumulate_gradient!(
        destination.motif,
        source.motif,
    )
    _add_array!(
        destination.context.common_relation_raw,
        source.context.common_relation_raw,
    )
    _add_array!(
        destination.context.common_output_raw,
        source.context.common_output_raw,
    )
    _add_array!(
        destination.context.aux_relation_raw,
        source.context.aux_relation_raw,
    )
    _add_array!(destination.placement_relation, source.placement_relation)
    Model.Readout.accumulate_gradient!(
        destination.motif_readout,
        source.motif_readout,
    )
    Model.Outputs.accumulate_gradient!(destination.output, source.output)
    return destination
end

function _reduce_state_gradients!(trainer::BarrierlessRelationGraphTrainer)
    Model.clear_gradient!(trainer.gradient)
    @inbounds for state_slot in 1:trainer.state_batch
        _merge_gradient!(
            trainer.gradient,
            trainer.state_gradients[state_slot],
        )
    end
    return trainer.gradient
end

@inline function _project_afferent_group!(
    multiplier::Float32,
    cache,
    graph,
    target::Float32,
)
    multiplier > 0.0f0 || return nothing
    Model.Afferents.project_conductance_homeostasis!(
        cache,
        graph,
        target,
        _HOMEOSTASIS_FLOOR_RATIO,
        _HOMEOSTASIS_CEILING_RATIO,
    )
    return nothing
end

"""
Apply the canonical projected-Adam conductance prior after an optimizer step.

Each graph is projected only when its Adam group is active, preserving the
zero-multiplier strict-freeze contract. Multi-contact typed-input groups retain
their target mean within a fourfold box; singleton groups are box-clamped only
so a learned in-range conductance is not erased.
"""
function _project_conductance_homeostasis!(
    trainer::BarrierlessRelationGraphTrainer,
)
    parameters = trainer.parameters
    cache = trainer.cache
    config = trainer.optimizer_config
    _project_afferent_group!(
        config.leaf_relation_multiplier,
        cache.leaf_relation,
        parameters.leaf_relation,
        0.1f0,
    )
    _project_afferent_group!(
        config.relation_motif_multiplier,
        cache.relation_motif,
        parameters.relation_motif,
        0.1f0,
    )
    _project_afferent_group!(
        config.common_relation_multiplier,
        cache.common_relation,
        parameters.context.common_relation,
        0.1f0,
    )
    _project_afferent_group!(
        config.common_output_multiplier,
        cache.common_output,
        parameters.context.common_output,
        0.1f0,
    )
    _project_afferent_group!(
        config.auxiliary_relation_multiplier,
        cache.aux_relation,
        parameters.context.aux_relation,
        0.05f0,
    )
    _project_afferent_group!(
        config.placement_relation_multiplier,
        cache.placement_relation,
        parameters.placement_relation,
        0.25f0,
    )
    return nothing
end

@inline function _diagnostic_total(
    trainer::BarrierlessRelationGraphTrainer,
    lane::Int,
)
    total = Int64(0)
    @inbounds @simd for worker in axes(trainer.diagnostic_counters, 2)
        total += trainer.diagnostic_counters[lane, worker]
    end
    return total
end

"""
Run only the canonical input-only forward phases for the current fixed batch.

This is the production inference/evaluation boundary.  It prepares batch
metadata, builds every state-common trajectory, and evaluates every valid
candidate.  It deliberately performs no supervised loss, reverse traversal,
gradient reduction, optimizer update, or transformed-cache refresh.
"""
function forward_batch!(session::BarrierlessRelationGraphSession)
    trainer = session.trainer
    scheduler = session.scheduler
    Base.Threads.threadid() == 1 || error(
        "only coordinator worker slot one may start a forward batch",
    )
    _check_rows(trainer)
    Ranking.prepare_batch_metadata!(trainer.batch, trainer.teacher_data)
    fill!(trainer.diagnostic_counters, 0)

    SchedulerCore.run_phase!(
        scheduler,
        _PREPARE_STATE,
        1,
        trainer.state_batch;
        chunk_size=1,
    )
    SchedulerCore.run_phase!(
        scheduler,
        _FORWARD_CANDIDATE,
        1,
        trainer.batch.valid_count;
        chunk_size=trainer.candidate_chunk_size,
    )
    return trainer.batch.raw
end

"""
Run one complete exact update inside an already active persistent worker team.

The only synchronization boundaries are state preparation, completion of all
candidate outputs before ListNet, and completion of every state-owned reverse
before deterministic reduction and AdamW.
"""
function train_update!(session::BarrierlessRelationGraphSession)
    trainer = session.trainer
    scheduler = session.scheduler
    Base.Threads.threadid() == 1 || error(
        "only coordinator worker slot one may start a training update",
    )
    forward_batch!(session)

    loss = Ranking.supervised_loss_and_raw_gradient!(
        trainer.batch,
        trainer.loss_scratch,
    )
    trainer.loss_sink[] = loss
    tied = Serial._tie_top1(trainer.batch)

    SchedulerCore.run_phase!(
        scheduler,
        _BACKWARD_STATE,
        1,
        trainer.state_batch;
        chunk_size=1,
    )
    _reduce_state_gradients!(trainer)
    step = Optimizer.apply_adamw!(
        trainer.optimizer_state,
        trainer.parameters,
        trainer.gradient,
        trainer.optimizer_config,
    )
    _project_conductance_homeostasis!(trainer)
    Model.refresh_cache!(trainer.cache, trainer.parameters)

    return Serial.TrainUpdateMetrics(
        loss,
        loss.composite_loss - loss.teacher_entropy,
        tied,
        _diagnostic_total(trainer, _CHANGED_POSITIONS),
        _diagnostic_total(trainer, _AFFECTED_POSITIONS),
        _diagnostic_total(trainer, _AFFECTED_RELATIONS),
        _diagnostic_total(trainer, _AFFECTED_MOTIFS),
        _diagnostic_total(trainer, _BASE_RELATION_EVENTS),
        _diagnostic_total(trainer, _CANDIDATE_RELATION_EVENTS),
        _diagnostic_total(trainer, _BASE_MOTIF_EVENTS),
        _diagnostic_total(trainer, _CANDIDATE_MOTIF_EVENTS),
        _diagnostic_total(trainer, _BASE_OUTPUT_EVENTS),
        _diagnostic_total(trainer, _CANDIDATE_OUTPUT_EVENTS),
        _diagnostic_total(trainer, _BASE_CONTACT_VISITS),
        _diagnostic_total(trainer, _CANDIDATE_CONTACT_VISITS),
        step.gradient_norm,
        step.clip_scale,
        step.active_program_rows,
    )
end

end # module RelationGraphBarrierless
