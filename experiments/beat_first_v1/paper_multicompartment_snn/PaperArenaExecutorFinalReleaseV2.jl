# Release-v2 bindings for the exact PaperExecutorFinal MPMC type.

function _release_official_coordinate(
    regions::Vector{String},
)
    length(regions) == ReleaseCell.OFFICIAL_LOCATION_COUNT ||
        error("official segment region map must contain 642 entries")
    result = Vector{UInt8}(undef, length(regions))
    @inbounds for index in eachindex(regions)
        name = lowercase(regions[index])
        result[index] =
            occursin("basal", name) ? UInt8(1) :
            occursin("tuft", name) ? UInt8(4) :
            occursin("apic", name) ? UInt8(2) : UInt8(3)
    end
    return result
end

function PaperExecutorFinal(
    trainer::PaperTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    stochastic_routing::Bool=true,
    routing_seed::Integer=0x5041504552524f55,
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=512,
)
    julia_workers = Base.Threads.nthreads(:default)
    Base.Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0 for deterministic native workers",
    )
    2 <= active_workers <= julia_workers ||
        throw(ArgumentError(
            "active_workers must be in 2:$julia_workers",
        ))
    cpuset_mode in (:none, :all, :p_only) ||
        throw(ArgumentError(
            "cpuset_mode must be :none, :all, or :p_only",
        ))
    ispow2(queue_capacity) ||
        throw(ArgumentError("queue capacity must be a power of two"))
    queue_capacity >= 256 ||
        throw(ArgumentError(
            "queue capacity must be at least 256",
        ))

    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux || error(
        "PaperExecutorFinal production requires enable_release_runtime!",
    )
    set_paper_ablation!(trainer, paper_ablation(trainer))
    workers = [PaperWorker(trainer) for _ in 1:active_workers]
    all(worker -> worker.runtime isa ReleaseCellRuntime, workers) ||
        error("final executor contains a non-release cell runtime")
    all(worker -> worker.selected isa Vector{Bool}, workers) ||
        error("PaperWorker.selected must be Vector{Bool}")
    eligibilities = [
        _paper_final_eligibility(trainer.model)
        for _ in 1:active_workers
    ]
    # Bind the executor-owned traces once. Release replay performs one
    # allocation-free IdDict lookup per candidate, never per contact.
    @inbounds for slot in eachindex(workers)
        _WORKER_ELIGIBILITY[workers[slot]] =
            eligibilities[slot]
    end
    queue = Point.Queue.BoundedMPMCQueue{PaperFinalWorkItem}(
        queue_capacity,
        zero(PaperFinalWorkItem),
    )
    return PaperExecutorFinal(
        queue,
        workers,
        eligibilities,
        trainer,
        dataset,
        _release_official_coordinate(
            aux.official_segment_region,
        ),
        Int16.(1:ReleaseCell.OFFICIAL_LOCATION_COUNT),
        zeros(Float32, trainer.model.blocks),
        zeros(Float32, trainer.model.node_dim),
        active_workers,
        julia_workers,
        cpuset_mode,
        stochastic_routing,
        UInt64(routing_seed),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        Base.Event(true),
        PaperFinalWorkerStatistics(active_workers),
        false,
    )
end

function paper_local_replay_candidate_final!(
    worker::PaperWorker{ReleaseCellRuntime},
    eligibility::ReceptorEligibility,
    trainer::PaperTrainer,
    compartment_region::Vector{UInt8},
    location_slot::Vector{Int16},
    flat::Int,
)
    _WORKER_ELIGIBILITY[worker] === eligibility ||
        error("final executor eligibility ownership differs")
    length(compartment_region) ==
        ReleaseCell.OFFICIAL_LOCATION_COUNT ||
        error("final executor does not expose 642 locations")
    length(location_slot) ==
        ReleaseCell.OFFICIAL_LOCATION_COUNT ||
        error("final executor location-slot map is not 642")
    return paper_local_replay_candidate!(
        worker,
        trainer,
        flat,
    )
end

function _reduce_release_sparse_utility_final!(
    destination::Array{Float32,3},
    current_location::Matrix{UInt16},
    workers,
    worker_count::Int,
    current_field::Symbol,
    best_value_field::Symbol,
    best_location_field::Symbol,
    scale::Float32,
)
    @inbounds for slot in 1:worker_count
        sparse = _RELEASE_WORKER_UTILITY[workers[slot]]
        current = getproperty(sparse, current_field)
        best_value = getproperty(sparse, best_value_field)
        best_location = getproperty(sparse, best_location_field)
        for block in axes(current, 2)
            for contact in axes(current, 1)
                location = Int(current_location[contact, block])
                destination[location, contact, block] +=
                    scale * current[contact, block]
                proposal = Int(best_location[contact, block])
                proposal == 0 && continue
                destination[proposal, contact, block] +=
                    scale * best_value[contact, block]
            end
        end
    end
    return destination
end

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
    for name in keys(trainer.gradient)
        _reduce_paper_final_gradient_array_hotfix!(
            getproperty(trainer.gradient, name),
            workers,
            Val(name),
            worker_count,
        )
    end
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

