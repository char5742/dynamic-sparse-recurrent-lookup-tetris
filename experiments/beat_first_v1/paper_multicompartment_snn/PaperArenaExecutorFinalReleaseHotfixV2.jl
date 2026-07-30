# Allocation-safe release post-replay reducer with compile-time field access.

function _paper_final_post_replay_hotfix!(
    executor::PaperExecutorFinal;
    worker_count::Int=executor.active_workers,
)
    trainer = executor.trainer
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("final post-replay reducer has no release aux")
    Optim.zero_parameter_tree!(trainer.gradient)
    workers = executor.workers
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.input_conductance,
        workers,
        Val(:input_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.recurrent_conductance,
        workers,
        Val(:recurrent_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_conductance,
        workers,
        Val(:workspace_conductance),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.query_weight,
        workers,
        Val(:query_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_key,
        workers,
        Val(:workspace_key),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.workspace_decay_logit,
        workers,
        Val(:workspace_decay_logit),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.head_weight,
        workers,
        Val(:head_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.head_bias,
        workers,
        Val(:head_bias),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.output_weight,
        workers,
        Val(:output_weight),
        worker_count,
    )
    _reduce_paper_final_gradient_array_hotfix!(
        trainer.gradient.output_bias,
        workers,
        Val(:output_bias),
        worker_count,
    )

    inverse_candidates =
        inv(Float32(max(trainer.tape.base.valid_count, 1)))
    _decay_release_utility!(
        aux.input_location_utility,
        trainer.utility_decay,
    )
    _decay_release_utility!(
        aux.recurrent_location_utility,
        trainer.utility_decay,
    )
    _decay_release_utility!(
        aux.workspace_location_utility,
        trainer.utility_decay,
    )
    scale =
        (1.0f0 - trainer.utility_decay) *
        inverse_candidates
    _reduce_release_sparse_utility_final!(
        aux.input_location_utility,
        aux.input_location,
        workers,
        worker_count,
        :input_current,
        :input_best_value,
        :input_best_location,
        scale,
    )
    _reduce_release_sparse_utility_final!(
        aux.recurrent_location_utility,
        aux.recurrent_location,
        workers,
        worker_count,
        :recurrent_current,
        :recurrent_best_value,
        :recurrent_best_location,
        scale,
    )
    _reduce_release_sparse_utility_final!(
        aux.workspace_location_utility,
        aux.workspace_location,
        workers,
        worker_count,
        :workspace_current,
        :workspace_best_value,
        :workspace_best_location,
        scale,
    )
    _temporal_workspace_decay_gradient_final!(executor)
    trainer.metrics.gradient_norm = Optim.paper_adam_step!(
        trainer.optimizer,
        trainer.parameters,
        trainer.gradient,
    )
    moves = _consolidate_one_location_canonical!(trainer)
    moves += _consolidate_workspace_location!(trainer)
    trainer.metrics.location_moves = moves
    _refresh_paper_final_metrics!(executor)
    return trainer
end
