module CandidateDeltaDendriticTraining

using ..CandidateDeltaDendriticGraph
using ..TetrisRankingBatch

const Model = CandidateDeltaDendriticGraph
const Ranking = TetrisRankingBatch

export ExactBatchTrainer,
       backward_batch!,
       clear_batch_gradient!,
       forward_batch!,
       forward_loss_backward!,
       loss_and_raw_gradient!,
       refresh_model_cache!

"""
Serial exact trainer for the candidate-delta dendritic graph.

The trainer deliberately owns no optimizer.  It connects the real Tetris
candidate grouping and complete 22-channel supervised objective directly to
the model's exact conditional reverse.  State-common factors are evaluated
once in the forward pass and traversed once in the reverse pass; candidate
placements remain in the stable order stored by `TetrisRankingBatch.Batch`.
"""
struct ExactBatchTrainer
    parameters::Model.ModelParameters
    cache::Model.ModelCache
    gradient::Model.ModelGradient
    state::Model.ModelState
    worker::Model.ModelWorker
    loss_scratch::Ranking.LossScratch
    state_batch::Int
    width::Int
end

@inline function _maximum_program_rows(
    parameters::Model.ModelParameters,
    state_batch::Int,
    width::Int,
)
    # Per state: before and after-baseline planes plus, in the conservative
    # upper bound, a full after plane for every candidate.  Actual candidate
    # deltas touch only their exact 3x3 closure and therefore use far fewer
    # rows.  Capping at physical bank size keeps the sparse accumulator exact.
    positions = Model.Topology.LEAF_COUNT
    probes = Model.Bank.MAX_ACTIVE_ROWS
    factor_visits = Base.checked_mul(
        state_batch,
        Base.checked_add(
            Base.checked_mul(2, positions),
            Base.checked_mul(width, positions),
        ),
    )
    return min(
        Model.Bank.bank_row_count(parameters.program_bank),
        Base.checked_mul(factor_visits, probes),
    )
end

function ExactBatchTrainer(
    parameters::Model.ModelParameters,
    state_batch::Integer,
    width::Integer;
    active_program_capacity::Union{Nothing,Integer}=nothing,
)
    states = Int(state_batch)
    candidates = Int(width)
    states >= 1 || throw(ArgumentError("state_batch must be positive"))
    candidates >= 1 || throw(ArgumentError("width must be positive"))
    capacity = isnothing(active_program_capacity) ?
        _maximum_program_rows(parameters, states, candidates) :
        Int(active_program_capacity)
    return ExactBatchTrainer(
        parameters,
        Model.ModelCache(parameters),
        Model.ModelGradient(
            parameters;
            active_program_capacity=capacity,
        ),
        Model.ModelState(),
        Model.ModelWorker(),
        Ranking.LossScratch(candidates, states),
        states,
        candidates,
    )
end

@inline function _check_contract(
    trainer::ExactBatchTrainer,
    batch::Ranking.Batch,
    dataset::Ranking.ValidatedDataset,
)
    batch.state_batch == trainer.state_batch || throw(DimensionMismatch(
        "batch state_batch $(batch.state_batch) differs from trainer " *
        "state_batch $(trainer.state_batch)",
    ))
    batch.width == trainer.width || throw(DimensionMismatch(
        "batch width $(batch.width) differs from trainer width " *
        "$(trainer.width)",
    ))
    dataset.candidate_width == trainer.width || throw(DimensionMismatch(
        "dataset width $(dataset.candidate_width) differs from trainer width " *
        "$(trainer.width)",
    ))
    return nothing
end

"""Refresh transformed cell parameters after an external optimizer update."""
function refresh_model_cache!(trainer::ExactBatchTrainer)
    Model.refresh_cache!(trainer.cache, trainer.parameters)
    return trainer
end

"""Clear every dense and sparse gradient owned by this exact batch."""
function clear_batch_gradient!(trainer::ExactBatchTrainer)
    Model.clear_gradient!(trainer.gradient)
    return trainer.gradient
end

"""
Evaluate all legal candidates in stable state-major order.

Only `prepare_batch_metadata!` is used from the legacy batch container.  The
1,298-rail matrix is neither packed nor read by this path: the model consumes
the validated board, queue, placement and scalar context tensors directly.
"""
function forward_batch!(
    trainer::ExactBatchTrainer,
    batch::Ranking.Batch,
    dataset::Ranking.ValidatedDataset,
)
    _check_contract(trainer, batch, dataset)
    Ranking.prepare_batch_metadata!(batch, dataset)
    refresh_model_cache!(trainer)
    state = trainer.state
    worker = trainer.worker
    parameters = trainer.parameters
    cache = trainer.cache
    width = batch.width
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
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
        end
    end
    return batch.raw
end

"""Compute the complete exact 22-channel Tetris cotangent."""
function loss_and_raw_gradient!(
    trainer::ExactBatchTrainer,
    batch::Ranking.Batch,
)
    batch.state_batch == trainer.state_batch || throw(DimensionMismatch(
        "batch state_batch differs from trainer state_batch",
    ))
    batch.width == trainer.width || throw(DimensionMismatch(
        "batch width differs from trainer width",
    ))
    return Ranking.supervised_loss_and_raw_gradient!(
        batch,
        trainer.loss_scratch,
    )
end

"""
Accumulate the existing raw-output cotangent into the model gradient.

This method intentionally does not clear the gradient, which permits an
external reducer to combine microbatches.  Call `clear_batch_gradient!` first
for a single-batch gradient.  Candidate-local paths are replayed in their
stable order; after all candidates of one state, its shared before/after DAG is
reversed exactly once.
"""
function backward_batch!(
    trainer::ExactBatchTrainer,
    batch::Ranking.Batch,
    dataset::Ranking.ValidatedDataset,
)
    _check_contract(trainer, batch, dataset)
    state = trainer.state
    worker = trainer.worker
    parameters = trainer.parameters
    cache = trainer.cache
    gradient = trainer.gradient
    width = batch.width
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            flat = offset + candidate
            Model.prepare_candidate!(
                worker,
                state,
                parameters,
                cache,
                dataset,
                row,
                candidate,
            )
            Model.pullback_candidate!(
                gradient,
                @view(batch.raw[:, flat]),
                @view(batch.raw_gradient[:, flat]),
                worker,
                state,
                parameters,
                cache,
            )
        end
        Model.finish_state_pullback!(
            gradient,
            worker,
            state,
            parameters,
            cache,
        )
    end
    return gradient
end

"""Run forward, full supervised loss, gradient clear and exact reverse."""
function forward_loss_backward!(
    trainer::ExactBatchTrainer,
    batch::Ranking.Batch,
    dataset::Ranking.ValidatedDataset,
)
    forward_batch!(trainer, batch, dataset)
    loss = loss_and_raw_gradient!(trainer, batch)
    clear_batch_gradient!(trainer)
    backward_batch!(trainer, batch, dataset)
    return loss
end

end # module CandidateDeltaDendriticTraining
