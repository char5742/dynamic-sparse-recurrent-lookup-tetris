module EpisodicViTRecurrentLookupTeacherTraining

using Dates
using JSON3
using LinearAlgebra
using Profile
using Random
using Serialization
using SHA
using Statistics
using Zygote

if !isdefined(Main, :BeatFirstThreeLayerTeacherTraining)
    Base.include(Main, joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "teacher_training.jl"))
end
if !isdefined(Main, :EpisodicViTRecurrentLookup)
    Base.include(Main, joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))
end
if !isdefined(Main, :WinCpuSets)
    Base.include(Main, joinpath(@__DIR__, "windows_cpu_sets.jl"))
end
if !isdefined(Main, :BoundedMPMCRing)
    Base.include(Main, joinpath(@__DIR__, "bounded_mpmc_queue.jl"))
end

const ParentTraining = Main.BeatFirstThreeLayerTeacherTraining
const TrainingCore = ParentTraining.BeatFirstTrainingCore
const Model = Main.EpisodicViTRecurrentLookup
const CpuSets = Main.WinCpuSets
const Queue = Main.BoundedMPMCRing

const EXPERIMENT_ID = :episodic_vit_recurrent_lookup_v1
const LEARNER_WIDTH = 80
const OUTPUT_DIM = 22
const MODEL_SEED = UInt64(2026071801)
const HALT_SEED = UInt64(0x4556524c5f48414c)
const TRAIN_SEED = UInt64(2026071801)
const SPLIT_SEED = UInt64(2026071817)
const SAMPLER_SEED = TRAIN_SEED + UInt64(0x9e3779b97f4a7c15)
const TRAIN_EVAL_SEED = TRAIN_SEED + UInt64(0x101)
const VALIDATION_EVAL_SEED = TRAIN_SEED + UInt64(0x202)
const TRAIN_EVAL_STATES = 64
const VALIDATION_EVAL_STATES = 128
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT = raw"D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup"

@inline _property_or(value, name::Symbol, default) =
    value !== nothing && hasproperty(value, name) ? getproperty(value, name) : default

function _float32_env(name::AbstractString, default; nonnegative::Bool=false)
    value = parse(Float32, strip(get(ENV, name, string(default))))
    isfinite(value) || error("$name must be finite")
    if nonnegative
        value >= 0.0f0 || error("$name must be nonnegative")
    else
        value > 0.0f0 || error("$name must be positive")
    end
    return value
end

function _int_env(name::AbstractString, default; minimum::Int=0)
    value = parse(Int, strip(get(ENV, name, string(default))))
    value >= minimum || error("$name must be at least $minimum")
    return value
end

function _margin_mode(raw)
    normalized = replace(lowercase(strip(String(raw))), '-' => '_')
    normalized in ("fixed_top2", "fixed_teacher_top2") &&
        return TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE
    normalized in ("hard_negative", "student_hard_negative") &&
        return TrainingCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE
    error("EVRL_MARGIN_MODE must be fixed_top2 or student_hard_negative")
end

function runtime_hyperparameters(maximum_updates::Int, payload=nothing)
    inherited = payload === nothing ? nothing : payload.config.hyperparameters
    inherited_optimizer = _property_or(inherited, :optimizer, nothing)
    inherited_routing = _property_or(inherited, :routing, nothing)
    inherited_halting = _property_or(inherited, :halting, nothing)
    inherited_loss = _property_or(inherited, :loss, nothing)
    optimizer = (;
        gradient_clip_norm=_float32_env(
            "EVRL_GRADIENT_CLIP_NORM",
            _property_or(inherited_optimizer, :gradient_clip_norm, 5.0f0),
        ),
        beta1=_float32_env("EVRL_ADAM_BETA1", _property_or(inherited_optimizer, :beta1, 0.9f0); nonnegative=true),
        beta2=_float32_env("EVRL_ADAM_BETA2", _property_or(inherited_optimizer, :beta2, 0.999f0); nonnegative=true),
        epsilon=_float32_env("EVRL_ADAM_EPSILON", _property_or(inherited_optimizer, :epsilon, 1.0f-8)),
        bank_learning_rate=_float32_env("EVRL_LR_BANK", _property_or(inherited_optimizer, :bank_learning_rate, 2.0f-4)),
        router_learning_rate=_float32_env("EVRL_LR_ROUTER", _property_or(inherited_optimizer, :router_learning_rate, 4.0f-4)),
        lookup_alpha_learning_rate=_float32_env("EVRL_LR_LOOKUP_ALPHA", _property_or(inherited_optimizer, :lookup_alpha_learning_rate, 2.0f-4)),
        attention_learning_rate=_float32_env("EVRL_LR_ATTENTION", _property_or(inherited_optimizer, :attention_learning_rate, 2.0f-4)),
        ffn_learning_rate=_float32_env("EVRL_LR_FFN", _property_or(inherited_optimizer, :ffn_learning_rate, 2.0f-4)),
        token_learning_rate=_float32_env("EVRL_LR_TOKEN", _property_or(inherited_optimizer, :token_learning_rate, 2.0f-4)),
        register_learning_rate=_float32_env("EVRL_LR_REGISTER", _property_or(inherited_optimizer, :register_learning_rate, 2.0f-4)),
        head_learning_rate=_float32_env("EVRL_LR_HEAD", _property_or(inherited_optimizer, :head_learning_rate, 2.0f-4)),
        halt_learning_rate=_float32_env("EVRL_LR_HALT", _property_or(inherited_optimizer, :halt_learning_rate, 5.0f-5)),
        episodic_decay_after_update=_int_env(
            "EVRL_EPISODIC_LR_DECAY_AFTER",
            _property_or(inherited_optimizer, :episodic_decay_after_update, maximum_updates);
            minimum=0,
        ),
        episodic_decay_factor=_float32_env(
            "EVRL_EPISODIC_LR_DECAY_FACTOR",
            _property_or(inherited_optimizer, :episodic_decay_factor, 1.0f0);
            nonnegative=true,
        ),
        bank_weight_decay=_float32_env("EVRL_WD_BANK", _property_or(inherited_optimizer, :bank_weight_decay, 0.0f0); nonnegative=true),
        dense_weight_decay=_float32_env("EVRL_WD_DENSE", _property_or(inherited_optimizer, :dense_weight_decay, 1.0f-4); nonnegative=true),
    )
    0.0f0 <= optimizer.beta1 < 1.0f0 || error("EVRL_ADAM_BETA1 must be in [0,1)")
    0.0f0 <= optimizer.beta2 < 1.0f0 || error("EVRL_ADAM_BETA2 must be in [0,1)")
    optimizer.episodic_decay_factor <= 1.0f0 ||
        error("EVRL_EPISODIC_LR_DECAY_FACTOR must be in [0,1]")
    routing = (;
        start_temperature=_float32_env("EVRL_ROUTE_TEMP_START", _property_or(inherited_routing, :start_temperature, 1.0f0)),
        end_temperature=_float32_env("EVRL_ROUTE_TEMP_END", _property_or(inherited_routing, :end_temperature, 0.25f0)),
        anneal_start_update=_int_env("EVRL_ROUTE_ANNEAL_START", _property_or(inherited_routing, :anneal_start_update, 0)),
        anneal_end_update=_int_env("EVRL_ROUTE_ANNEAL_END", _property_or(inherited_routing, :anneal_end_update, maximum_updates); minimum=1),
    )
    routing.anneal_end_update > routing.anneal_start_update || error("routing anneal interval is empty")
    halting = (;
        warmup_updates=_int_env("EVRL_WARMUP_UPDATES", _property_or(inherited_halting, :warmup_updates, min(1_000, maximum_updates))),
        fixed_depth=_int_env(
            "EVRL_FIXED_DEPTH",
            _property_or(inherited_halting, :fixed_depth, 0);
            minimum=0,
        ),
        compute_price=_float32_env("EVRL_COMPUTE_PRICE", _property_or(inherited_halting, :compute_price, 0.02f0); nonnegative=true),
        policy_weight=_float32_env("EVRL_POLICY_WEIGHT", _property_or(inherited_halting, :policy_weight, 0.05f0); nonnegative=true),
        entropy_weight=_float32_env("EVRL_ENTROPY_WEIGHT", _property_or(inherited_halting, :entropy_weight, 0.001f0); nonnegative=true),
        probe_candidates_per_state=_int_env(
            "EVRL_HALT_PROBES_PER_STATE",
            _property_or(inherited_halting, :probe_candidates_per_state, 0);
            minimum=0,
        ),
        probe_weight=_float32_env(
            "EVRL_HALT_PROBE_WEIGHT",
            _property_or(inherited_halting, :probe_weight, 1.0f0);
            nonnegative=true,
        ),
    )
    iszero(halting.fixed_depth) ||
        Model.MIN_RECURRENT_STEPS <= halting.fixed_depth <= Model.MAX_RECURRENT_STEPS ||
        error("EVRL_FIXED_DEPTH must be zero or inside the recurrent bounds")
    loss = (;
        listnet_weight=Float32(_property_or(inherited_loss, :listnet_weight, 1.0f0)),
        old_q_weight=Float32(_property_or(inherited_loss, :old_q_weight, 0.25f0)),
        margin_weight=_float32_env(
            "EVRL_LOSS_MARGIN",
            _property_or(inherited_loss, :margin_weight, 0.15f0);
            nonnegative=true,
        ),
        death_weight=Float32(_property_or(inherited_loss, :death_weight, 0.10f0)),
        quantile_weight=Float32(_property_or(inherited_loss, :quantile_weight, 0.05f0)),
        geometry_weight=Float32(_property_or(inherited_loss, :geometry_weight, 0.10f0)),
        margin_mode=_margin_mode(get(
            ENV,
            "EVRL_MARGIN_MODE",
            string(_property_or(
                inherited_loss,
                :margin_mode,
                TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
            )),
        )),
        hard_negative_margin_floor=_float32_env(
            "EVRL_HARDNEG_MARGIN_FLOOR",
            _property_or(inherited_loss, :hard_negative_margin_floor, 0.0f0);
            nonnegative=true,
        ),
    )
    all(isfinite, (
        loss.listnet_weight,
        loss.old_q_weight,
        loss.margin_weight,
        loss.death_weight,
        loss.quantile_weight,
        loss.geometry_weight,
        loss.hard_negative_margin_floor,
    )) || error("loss weights must be finite")
    all(value -> value >= 0.0f0, (
        loss.listnet_weight,
        loss.old_q_weight,
        loss.margin_weight,
        loss.death_weight,
        loss.quantile_weight,
        loss.geometry_weight,
        loss.hard_negative_margin_floor,
    )) || error("loss weights must be nonnegative")
    return (; optimizer, routing, halting, loss)
end

function _normalized_hyperparameters(value)
    halting = value.halting
    normalized_halting = (;
        warmup_updates=halting.warmup_updates,
        fixed_depth=halting.fixed_depth,
        compute_price=halting.compute_price,
        policy_weight=halting.policy_weight,
        entropy_weight=halting.entropy_weight,
        probe_candidates_per_state=_property_or(
            halting, :probe_candidates_per_state, 0,
        ),
        probe_weight=Float32(_property_or(halting, :probe_weight, 1.0f0)),
    )
    return (;
        optimizer=value.optimizer,
        routing=value.routing,
        halting=normalized_halting,
        loss=value.loss,
    )
end

@inline function routing_temperature(update::Int, schedule)
    progress = clamp(
        Float32(update - schedule.anneal_start_update) /
        Float32(max(schedule.anneal_end_update - schedule.anneal_start_update, 1)),
        0.0f0,
        1.0f0,
    )
    return schedule.start_temperature -
        (schedule.start_temperature - schedule.end_temperature) * progress
end

function objective_contract(hyperparameters)
    loss = hyperparameters.loss
    return (;
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        margin_mode=loss.margin_mode,
        listnet_temperature=TrainingCore.LISTNET_TEMPERATURE,
        listnet_weight=loss.listnet_weight,
        old_q_weight=loss.old_q_weight,
        margin_weight=loss.margin_weight,
        hard_negative_margin_floor=loss.hard_negative_margin_floor,
        death_weight=loss.death_weight,
        quantile_weight=loss.quantile_weight,
        geometry_weight=loss.geometry_weight,
    )
end

mutable struct TeacherWorkspace
    raw::Matrix{Float32}
    tapes::Vector{Union{Nothing,Model.TrajectoryTape}}
    depths::Vector{Int16}
    candidate_seeds::Vector{UInt64}
    forced_depths::Vector{Int16}
    memory_buffers::Vector{Matrix{Float32}}
    inverse_rms_buffers::Vector{Vector{Float32}}
    forward_arenas::Vector{Model.ForwardCandidateArena}
    halt_probe_targets::Vector{Float32}
    halt_probe_deltas::Vector{Float32}
end

TeacherWorkspace() = TeacherWorkspace(
    zeros(Float32, OUTPUT_DIM, LEARNER_WIDTH),
    Union{Nothing,Model.TrajectoryTape}[nothing for _ in 1:LEARNER_WIDTH],
    zeros(Int16, LEARNER_WIDTH),
    zeros(UInt64, LEARNER_WIDTH),
    zeros(Int16, LEARNER_WIDTH),
    [Matrix{Float32}(undef, Model.MODEL_DIM, Model.TOKEN_COUNT) for _ in 1:LEARNER_WIDTH],
    [Vector{Float32}(undef, Model.TOKEN_COUNT) for _ in 1:LEARNER_WIDTH],
    [Model.ForwardCandidateArena() for _ in 1:LEARNER_WIDTH],
    fill(Float32(NaN), LEARNER_WIDTH),
    fill(Float32(NaN), LEARNER_WIDTH),
)

const TRAINING_STATE_BATCH = let
    raw = strip(get(ENV, "EVRL_STATE_BATCH", "4"))
    value = tryparse(Int, raw)
    value === nothing && error("EVRL_STATE_BATCH must be an integer")
    value in (4, 8) || error("EVRL_STATE_BATCH must be 4 or 8")
    value
end
const MAX_FLAT_CANDIDATES = TRAINING_STATE_BATCH * LEARNER_WIDTH

struct CandidateSchedulerConfig
    mode::Symbol
    cpuset_mode::Symbol
    chunk_size::Int
    adaptive_tail::Bool
    active_workers::Int
    detected_workers::Int
    topology
end

function candidate_scheduler_config()
    mode_raw = Symbol(lowercase(strip(get(ENV, "EVRL_SCHEDULER", "barrierless"))))
    mode = mode_raw in (:async, :asynchronous) ? :barrierless : mode_raw
    mode in (:static, :dynamic, :barrierless, :serial) || error(
        "EVRL_SCHEDULER must be static, dynamic, barrierless, or serial",
    )
    cpuset_mode = Symbol(lowercase(strip(get(ENV, "EVRL_CPUSET_MODE", "all"))))
    cpuset_mode in (:none, :all, :p_only) || error(
        "EVRL_CPUSET_MODE must be none, all, or p_only",
    )
    chunk_size = _int_env("EVRL_QUEUE_CHUNK", 8; minimum=1)
    chunk_size in (8, 16, 32) || error("EVRL_QUEUE_CHUNK must be 8, 16, or 32")
    adaptive_raw = strip(get(ENV, "EVRL_ADAPTIVE_TAIL", "0"))
    adaptive_raw in ("0", "1") || error("EVRL_ADAPTIVE_TAIL must be 0 or 1")
    adaptive_tail = adaptive_raw == "1"
    topology = CpuSets.discover_topology()
    detected_workers = length(topology.physical_cores)
    requested = cpuset_mode === :p_only ? length(topology.p_cores) :
        Base.Threads.nthreads(:default)
    active_workers = mode === :serial ? 1 : requested
    active_workers <= Base.Threads.nthreads(:default) || error(
        "scheduler requests more workers than Julia provides",
    )
    if mode !== :serial && cpuset_mode === :all
        detected_workers == Base.Threads.nthreads(:default) || error(
            "Julia worker count must equal detected physical P/E core count",
        )
    end
    CpuSets.configure_worker_bindings(cpuset_mode, active_workers, topology)
    return CandidateSchedulerConfig(
        mode, cpuset_mode, chunk_size, adaptive_tail,
        active_workers, detected_workers, topology,
    )
end

mutable struct CandidateSchedulerWorkspace
    config::CandidateSchedulerConfig
    state_workspaces::Vector{TeacherWorkspace}
    trajectory_states::Vector{Union{Nothing,Model.ForwardTrajectoryState}}
    job_states::Vector{Int16}
    job_candidates::Vector{Int16}
    active_a::Vector{Int16}
    active_b::Vector{Int16}
    cursor::Base.Threads.Atomic{Int}
    ready_workers::Base.Threads.Atomic{Int}
    raw_gradients::Vector{Matrix{Float32}}
    state_losses::Vector{Float32}
    merged_accumulator::Model.GradientAccumulator
    worker_jobs::Vector{UInt64}
    worker_chunks::Vector{UInt64}
    probe_arenas::Vector{Model.ForwardCandidateArena}
    probe_outputs::Vector{Vector{Float32}}
    candidate_wall_nanoseconds::UInt128
    candidate_cpu_ticks_100ns::UInt128
    barrierless_executor::Any
end

function CandidateSchedulerWorkspace(model)
    config = candidate_scheduler_config()
    return CandidateSchedulerWorkspace(
        config,
        [TeacherWorkspace() for _ in 1:TRAINING_STATE_BATCH],
        Union{Nothing,Model.ForwardTrajectoryState}[
            nothing for _ in 1:MAX_FLAT_CANDIDATES
        ],
        zeros(Int16, MAX_FLAT_CANDIDATES),
        zeros(Int16, MAX_FLAT_CANDIDATES),
        zeros(Int16, MAX_FLAT_CANDIDATES),
        zeros(Int16, MAX_FLAT_CANDIDATES),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        [zeros(Float32, OUTPUT_DIM, LEARNER_WIDTH) for _ in 1:TRAINING_STATE_BATCH],
        zeros(Float32, TRAINING_STATE_BATCH),
        Model.GradientAccumulator(model),
        zeros(UInt64, Base.Threads.maxthreadid()),
        zeros(UInt64, Base.Threads.maxthreadid()),
        [Model.ForwardCandidateArena() for _ in 1:Base.Threads.maxthreadid()],
        [zeros(Float32, OUTPUT_DIM) for _ in 1:Base.Threads.maxthreadid()],
        UInt128(0),
        UInt128(0),
        nothing,
    )
end

# Julia 1.12 on Windows must enter these include-defined specializations once
# on the caller before workers share them.  That is a one-time compile barrier,
# not work that belongs on every state in every update.
const FORWARD_EVAL_THREAD_WARMED = Ref(false)
const FORWARD_TRAIN_THREAD_WARMED = Ref(false)
const BACKWARD_THREAD_WARMED = Ref(false)

mutable struct TeacherTrainer
    model::Model.EpisodicViTLookupModel
    optimizer::Model.Optimizer
    halt_rng::Xoshiro
    usage::Model.RouteUsage
    workspace::TeacherWorkspace
    scheduler::CandidateSchedulerWorkspace
    thread_accumulators::Vector{Model.GradientAccumulator}
    baseline::Float32
    update::Int
    timed_updates::UInt64
    timed_states::UInt64
    timed_candidates::UInt64
    timed_recurrent_steps::UInt64
    training_nanoseconds::UInt128
    training_cpu_ticks_100ns::UInt128
end

function initialize_trainer(hyperparameters=runtime_hyperparameters(1_000))
    model = Model.initialize_model(Xoshiro(MODEL_SEED))
    optimizer = hyperparameters.optimizer
    scheduler = CandidateSchedulerWorkspace(model)
    thread_accumulators = [
        Model.GradientAccumulator(model) for _ in 1:Base.Threads.maxthreadid()
    ]
    trainer = TeacherTrainer(
        model,
        Model.initialize_optimizer(
            model;
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
            bank_learning_rate=optimizer.bank_learning_rate,
            bank_weight_decay=optimizer.bank_weight_decay,
        ),
        Xoshiro(HALT_SEED),
        Model.RouteUsage(),
        TeacherWorkspace(),
        scheduler,
        # `threadid()` is a global ID and may include the interactive pool;
        # size storage by the maximum live ID, not the default-pool count.
        thread_accumulators,
        8.0f0,
        0, 0, 0, 0, 0, 0, 0,
    )
    return _attach_barrierless_executor!(trainer)
end

function _valid_candidate_count(batch)
    size(batch.mask) == (LEARNER_WIDTH, 1) || throw(DimensionMismatch(
        "teacher batches must be 80x1",
    ))
    count = Int(sum(@view batch.mask[:, 1]))
    1 <= count <= LEARNER_WIDTH || error("invalid candidate count")
    all(@view(batch.mask[1:count, 1]) .== 1.0f0) || error("candidate mask is not prefix-valid")
    count < LEARNER_WIDTH && !all(@view(batch.mask[(count + 1):end, 1]) .== 0.0f0) &&
        error("candidate padding mask is nonzero")
    return count
end

@inline function _candidate_input(batch, candidate::Int)
    return Model.EpisodicCandidateInput(
        @view(batch.inputs.board[:, :, 1, candidate]),
        @view(batch.inputs.candidate[:, :, 1, candidate]),
        @view(batch.inputs.difference[:, :, 1, candidate]),
        @view(batch.inputs.next_hold[:, :, candidate]),
        @view(batch.inputs.aux[:, candidate]),
    )
end

@inline function _predict_candidate!(
    trainer::TeacherTrainer,
    batch,
    candidate::Int,
    workspace::TeacherWorkspace,
    training::Bool,
    record_tapes::Bool,
    temperature::Float32,
    materialize,
)
    forced_code = workspace.forced_depths[candidate]
    forced_depth = iszero(forced_code) ? nothing : Int(forced_code)
    owned_rng = training ? Xoshiro(workspace.candidate_seeds[candidate]) : nothing
    result = Model.forward_trajectory(
        trainer.model,
        _candidate_input(batch, candidate);
        rng=owned_rng,
        training,
        forced_depth,
        temperature,
        usage=nothing,
        materialize,
        memory_buffer=workspace.memory_buffers[candidate],
        inverse_rms_buffer=workspace.inverse_rms_buffers[candidate],
    )
    workspace.raw[:, candidate] .= result.output
    workspace.depths[candidate] = Int16(result.depth)
    (training || record_tapes) && (workspace.tapes[candidate] = result.tape)
    return nothing
end

@inline function _reset_workspace!(workspace::TeacherWorkspace)
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.tapes, nothing)
    fill!(workspace.depths, 0)
    fill!(workspace.halt_probe_targets, Float32(NaN))
    fill!(workspace.halt_probe_deltas, Float32(NaN))
    return workspace
end

function _timed_candidate_region!(body, trainer::TeacherTrainer)
    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    result = body()
    trainer.scheduler.candidate_wall_nanoseconds += UInt128(time_ns() - wall_started)
    trainer.scheduler.candidate_cpu_ticks_100ns += UInt128(
        CpuSets.process_cpu_ticks_100ns() - cpu_started,
    )
    return result
end

"""Run persistent native-worker consumers over one shared Atomic chunk queue."""
function _run_candidate_queue!(body, trainer::TeacherTrainer, count::Int)
    count <= 0 && return nothing
    scheduler = trainer.scheduler
    config = scheduler.config
    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    if config.mode === :serial
        @inbounds for job in 1:count
            body(job, 1)
        end
        scheduler.worker_jobs[1] += UInt64(count)
        scheduler.worker_chunks[1] += UInt64(cld(count, config.chunk_size))
    else
        scheduler.cursor[] = 0
        scheduler.ready_workers[] = 0
        Base.Threads.threading_run(worker_slot -> begin
            worker_slot <= config.active_workers || return nothing
            config.cpuset_mode === :none ||
                CpuSets.bind_current_worker!(worker_slot)
            Base.Threads.atomic_add!(scheduler.ready_workers, 1)
            while scheduler.ready_workers[] < config.active_workers
                yield()
            end
            local_jobs = UInt64(0)
            local_chunks = UInt64(0)
            while true
                first = Base.Threads.atomic_add!(
                    scheduler.cursor, config.chunk_size,
                ) + 1
                first > count && break
                last_job = min(first + config.chunk_size - 1, count)
                local_chunks += 1
                @inbounds for job in first:last_job
                    body(job, worker_slot)
                    local_jobs += 1
                end
            end
            thread = Base.Threads.threadid()
            scheduler.worker_jobs[thread] += local_jobs
            scheduler.worker_chunks[thread] += local_chunks
            return nothing
        end, true)
    end
    scheduler.candidate_wall_nanoseconds += UInt128(time_ns() - wall_started)
    scheduler.candidate_cpu_ticks_100ns += UInt128(
        CpuSets.process_cpu_ticks_100ns() - cpu_started,
    )
    return nothing
end

function _prepare_flat_jobs!(
    trainer::TeacherTrainer,
    batches,
    expected_update::Int,
    hyperparameters,
)
    length(batches) == TRAINING_STATE_BATCH || error(
        "dynamic scheduler received a state batch inconsistent with EVRL_STATE_BATCH",
    )
    scheduler = trainer.scheduler
    fixed_depth = hyperparameters.halting.fixed_depth
    warmup = iszero(fixed_depth) &&
        expected_update <= hyperparameters.halting.warmup_updates
    total = 0
    # Preserve the former serial RNG draw order exactly: state-major,
    # candidate-major, then halt seed followed by forced warmup depth.
    for (state_slot, batch) in enumerate(batches)
        workspace = _reset_workspace!(scheduler.state_workspaces[state_slot])
        count = _valid_candidate_count(batch)
        @inbounds for candidate in 1:count
            workspace.candidate_seeds[candidate] = rand(trainer.halt_rng, UInt64)
            workspace.forced_depths[candidate] = !iszero(fixed_depth) ?
                Int16(fixed_depth) :
                (warmup ? Int16(rand(
                    trainer.halt_rng,
                    Model.MIN_RECURRENT_STEPS:Model.WARMUP_MAX_STEPS,
                )) : Int16(0))
            total += 1
            scheduler.job_states[total] = Int16(state_slot)
            scheduler.job_candidates[total] = Int16(candidate)
            scheduler.active_a[total] = Int16(total)
            scheduler.trajectory_states[total] = nothing
        end
    end
    return total
end

function _dynamic_forward_batches!(
    trainer::TeacherTrainer,
    batches;
    expected_update::Int,
    hyperparameters,
)
    iszero(hyperparameters.optimizer.bank_weight_decay) || error(
        "dynamic queue requires zero bank decay so forward remains read-only",
    )
    scheduler = trainer.scheduler
    total = _prepare_flat_jobs!(trainer, batches, expected_update, hyperparameters)
    temperature = Float32(routing_temperature(expected_update, hyperparameters.routing))

    _run_candidate_queue!(trainer, total) do job, _
        state_slot = Int(scheduler.job_states[job])
        candidate = Int(scheduler.job_candidates[job])
        workspace = scheduler.state_workspaces[state_slot]
        forced_code = workspace.forced_depths[candidate]
        trajectory = Model.prepare_trajectory(
            trainer.model,
            _candidate_input(batches[state_slot], candidate);
            rng=Xoshiro(workspace.candidate_seeds[candidate]),
            training=true,
            forced_depth=iszero(forced_code) ? nothing : Int(forced_code),
            temperature,
            memory_buffer=workspace.memory_buffers[candidate],
            inverse_rms_buffer=workspace.inverse_rms_buffers[candidate],
            arena=workspace.forward_arenas[candidate],
        )
        scheduler.trajectory_states[job] = trajectory
        # No trajectory may halt before MIN_RECURRENT_STEPS.  Execute that
        # mandatory prefix in the same claimed chunk so short-lived native
        # teams do not dominate the common depth-two case.  The first actual
        # hard-halting decision is still followed by the same active compact.
        for _ in 1:Model.MIN_RECURRENT_STEPS
            Model.advance_trajectory!(trainer.model, trajectory)
            trajectory.stopped && break
        end
        if trajectory.stopped
            result = Model.finalize_trajectory(trainer.model, trajectory)
            workspace.raw[:, candidate] .= result.output
            workspace.depths[candidate] = Int16(result.depth)
            workspace.tapes[candidate] = result.tape
        end
    end

    active_count = 0
    @inbounds for job in 1:total
        trajectory = scheduler.trajectory_states[job]
        trajectory === nothing && error("candidate trajectory was not prepared")
        if !trajectory.stopped
            active_count += 1
            scheduler.active_b[active_count] = Int16(job)
        end
    end
    scheduler.active_a, scheduler.active_b = scheduler.active_b, scheduler.active_a
    while active_count > 0
        _run_candidate_queue!(trainer, active_count) do active_index, _
            job = Int(scheduler.active_a[active_index])
            trajectory = scheduler.trajectory_states[job]
            trajectory === nothing && error("candidate trajectory was not prepared")
            stopped = Model.advance_trajectory!(trainer.model, trajectory)
            if stopped
                state_slot = Int(scheduler.job_states[job])
                candidate = Int(scheduler.job_candidates[job])
                workspace = scheduler.state_workspaces[state_slot]
                result = Model.finalize_trajectory(trainer.model, trajectory)
                workspace.raw[:, candidate] .= result.output
                workspace.depths[candidate] = Int16(result.depth)
                workspace.tapes[candidate] = result.tape
            end
        end
        next_count = 0
        @inbounds for active_index in 1:active_count
            job16 = scheduler.active_a[active_index]
            trajectory = scheduler.trajectory_states[Int(job16)]
            trajectory === nothing && error("candidate trajectory disappeared")
            if !trajectory.stopped
                next_count += 1
                scheduler.active_b[next_count] = job16
            end
        end
        scheduler.active_a, scheduler.active_b = scheduler.active_b, scheduler.active_a
        active_count = next_count
    end

    # Telemetry remains deterministic and race-free.
    for state_slot in eachindex(batches)
        workspace = scheduler.state_workspaces[state_slot]
        count = _valid_candidate_count(batches[state_slot])
        @inbounds for candidate in 1:count
            tape = workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            Model.record_usage!(trainer.usage, tape)
        end
        all(isfinite, workspace.raw) || error("model output is non-finite")
    end
    return total
end

function predict_raw!(
    trainer::TeacherTrainer,
    batch;
    training::Bool,
    expected_update::Int=trainer.update,
    hyperparameters=runtime_hyperparameters(1_000),
    record_tapes::Bool=training,
)
    count = _valid_candidate_count(batch)
    workspace = trainer.workspace
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.tapes, nothing)
    fill!(workspace.depths, 0)
    temperature = routing_temperature(expected_update, hyperparameters.routing)
    # Lazy AdamW decay needs a pre-gather barrier only when bank decay is
    # nonzero.  At the production default (zero), calling it for every routed
    # register/block merely sorts and uniques the already-selected columns.
    materialize = training && !iszero(hyperparameters.optimizer.bank_weight_decay) ? ((block, columns) ->
        Model.materialize_selected_columns!(
            trainer.model, trainer.optimizer, block, columns,
        )) : nothing
    fixed_depth = hyperparameters.halting.fixed_depth
    warmup = training && iszero(fixed_depth) &&
        expected_update <= hyperparameters.halting.warmup_updates
    @inbounds for candidate in 1:count
        workspace.candidate_seeds[candidate] = training ?
            rand(trainer.halt_rng, UInt64) : UInt64(0)
        workspace.forced_depths[candidate] = !iszero(fixed_depth) ?
            Int16(fixed_depth) :
            (warmup ? Int16(rand(
                trainer.halt_rng, Model.MIN_RECURRENT_STEPS:Model.WARMUP_MAX_STEPS,
            )) : Int16(0))
    end
    # A nonzero lazy bank decay materializes model columns before each gather
    # and is therefore intentionally kept on the serial path.  The production
    # zero-decay path is read-only and safe to distribute by candidate.
    # The barrierless production driver already owns Julia's one native
    # threaded region.  A diagnostic evaluation inside that lifetime must not
    # nest `@threads`; use the identical serial candidate loop in that case.
    in_threaded_region = !iszero(ccall(:jl_in_threaded_region, Cint, ()))
    if Base.Threads.nthreads() > 1 && materialize === nothing && !in_threaded_region
        warmed = training ? FORWARD_TRAIN_THREAD_WARMED : FORWARD_EVAL_THREAD_WARMED
        first_parallel_candidate = 1
        if !warmed[]
            _predict_candidate!(
                trainer, batch, 1, workspace, training, record_tapes,
                Float32(temperature), materialize,
            )
            warmed[] = true
            first_parallel_candidate = 2
        end
        Base.Threads.@threads :static for candidate in first_parallel_candidate:count
            _predict_candidate!(
                trainer, batch, candidate, workspace, training, record_tapes,
                Float32(temperature), materialize,
            )
        end
    else
        for candidate in 1:count
            _predict_candidate!(
                trainer, batch, candidate, workspace, training, record_tapes,
                Float32(temperature), materialize,
            )
        end
    end
    if training
        # Forward workers never share telemetry.  Merge completed trajectories
        # in candidate order so usage is race-free and reproducible.
        for candidate in 1:count
            tape = workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            Model.record_usage!(trainer.usage, tape)
        end
    end
    all(isfinite, workspace.raw) || error("model output is non-finite")
    return workspace.raw, count
end

raw_output(raw::AbstractMatrix) = ParentTraining.raw_output(raw)

function _hard_negative_selection(raw::AbstractMatrix, batch, margin_mode)
    margin_mode === TrainingCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE || return nothing
    output = raw_output(raw)
    return TrainingCore.hard_negative_selection(
        reshape(output.q, size(batch.mask)),
        batch,
    )
end

@inline function _weighted_loss(components, weights)
    return weights.listnet_weight * components.listnet_loss +
        weights.old_q_weight * components.old_q_loss +
        weights.margin_weight * components.margin_loss +
        weights.death_weight * components.death_loss +
        weights.quantile_weight * components.quantile_teacher_loss +
        weights.geometry_weight * components.geometry_loss
end

function _weighted_components(raw, batch, hyperparameters; hard_negative=nothing)
    weights = hyperparameters.loss
    components = TrainingCore.supervised_components(
        raw_output(raw),
        batch;
        margin_weight=weights.margin_weight,
        margin_mode=weights.margin_mode,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        hard_negative,
        hard_negative_margin_floor=weights.hard_negative_margin_floor,
    )
    return merge(components, (; composite_loss=_weighted_loss(components, weights)))
end

function _loss_output_vjp(
    raw::Matrix{Float32}, batch, hyperparameters; hard_negative=nothing,
)
    loss, pullback = Zygote.pullback(raw) do candidate_outputs
        _weighted_components(
            candidate_outputs, batch, hyperparameters; hard_negative,
        ).composite_loss
    end
    gradient = only(pullback(one(loss)))
    gradient === nothing && error("loss VJP returned no gradient")
    result = Matrix{Float32}(gradient)
    all(isfinite, result) || error("loss VJP is non-finite")
    return Float32(loss), result
end

function _component_record(components)
    names = (
        :composite_loss, :listnet_loss, :old_q_loss, :q_huber_loss,
        :margin_loss, :raw_top_gap_loss, :death_loss, :quantile_teacher_loss,
        :geometry_loss, :line_clear_loss, :max_height_loss, :holes_loss,
        :cavities_loss, :valid_candidates,
    )
    return NamedTuple{names}(Tuple(Float64(getproperty(components, name)) for name in names))
end

const Q_OUTPUT_INDEX = 1

"""ListNet + margin value used only for counterfactual halting labels."""
function _halting_ranking_loss(raw, batch, hyperparameters)
    hard_negative = _hard_negative_selection(
        raw, batch, hyperparameters.loss.margin_mode,
    )
    components = _weighted_components(
        raw, batch, hyperparameters; hard_negative,
    )
    weights = hyperparameters.loss
    return Float32(
        weights.listnet_weight * components.listnet_loss +
        weights.margin_weight * components.margin_loss
    )
end

@inline function _halt_probe_eligible(tape)
    tape === nothing && return false
    length(tape.steps) < Model.MAX_RECURRENT_STEPS || return false
    step = last(tape.steps)
    return step.stochastic_decision && step.stopped && !step.forced_stop
end

"""Attach candidate-local stop labels from a physically sparse one-step probe.

Only the probed candidate's scalar Q is replaced while computing the
counterfactual ListNet + margin value.  The task output matrix is restored
before return, so the normal task loss and VJP remain bit-for-bit independent
of this supervision path.
"""
function _apply_halt_probes!(
    trainer::TeacherTrainer,
    batch,
    workspace::TeacherWorkspace,
    state_slot::Int,
    expected_update::Int,
    hyperparameters,
    worker_slot::Int,
)
    requested = hyperparameters.halting.probe_candidates_per_state
    requested <= 0 && return (;
        count=0, continue_count=0, stop_count=0, mean_delta=0.0f0,
    )
    count = _valid_candidate_count(batch)
    count == 0 && return (;
        count=0, continue_count=0, stop_count=0, mean_delta=0.0f0,
    )
    scheduler = trainer.scheduler
    arena = scheduler.probe_arenas[worker_slot]
    probe_output = scheduler.probe_outputs[worker_slot]
    stop_loss = _halting_ranking_loss(workspace.raw, batch, hyperparameters)
    selected = 0
    continue_count = 0
    delta_sum = 0.0f0
    start = mod(expected_update + 31 * state_slot - 2, count) + 1
    @inbounds for offset in 0:(count - 1)
        selected >= requested && break
        candidate = mod(start + offset - 1, count) + 1
        tape = workspace.tapes[candidate]
        _halt_probe_eligible(tape) || continue
        Model.probe_one_step!(
            probe_output,
            trainer.model,
            tape,
            arena;
            temperature=routing_temperature(
                expected_update, hyperparameters.routing,
            ),
        )
        previous_q = workspace.raw[Q_OUTPUT_INDEX, candidate]
        continue_loss = stop_loss
        try
            workspace.raw[Q_OUTPUT_INDEX, candidate] =
                probe_output[Q_OUTPUT_INDEX]
            continue_loss = _halting_ranking_loss(
                workspace.raw, batch, hyperparameters,
            )
        finally
            workspace.raw[Q_OUTPUT_INDEX, candidate] = previous_q
        end
        delta = stop_loss - continue_loss
        target = delta > hyperparameters.halting.compute_price ? 0.0f0 : 1.0f0
        workspace.halt_probe_deltas[candidate] = delta
        workspace.halt_probe_targets[candidate] = target
        selected += 1
        continue_count += iszero(target)
        delta_sum += delta
    end
    return (;
        count=selected,
        continue_count,
        stop_count=selected - continue_count,
        mean_delta=selected == 0 ? 0.0f0 : delta_sum / Float32(selected),
    )
end

# Scheduling-only implementation.  It is included after the loss helpers and
# candidate preparation API it dispatches, while remaining outside checkpoint
# payloads.
include(joinpath(@__DIR__, "barrierless_executor.jl"))
include(joinpath(@__DIR__, "barrierless_postphase.jl"))

function _attach_barrierless_executor!(trainer::TeacherTrainer)
    scheduler = trainer.scheduler
    if scheduler.config.mode === :barrierless
        active = scheduler.config.active_workers
        scheduler.barrierless_executor = BarrierlessExecutor(
            trainer.thread_accumulators[1:active];
            active_workers=active,
            fixed_chunk_size=scheduler.config.chunk_size,
            adaptive_tail=scheduler.config.adaptive_tail,
            queue_capacity=max(1024, nextpow(2, 2 * MAX_FLAT_CANDIDATES)),
        )
    else
        scheduler.barrierless_executor = nothing
    end
    return trainer
end

"""Accumulate one configured state batch on the persistent async DAG."""
function _accumulate_barrierless_batches!(
    trainer::TeacherTrainer,
    batches;
    expected_update::Int,
    hyperparameters,
    baseline::Float32,
)
    iszero(hyperparameters.optimizer.bank_weight_decay) || error(
        "barrierless execution requires zero bank decay so parameters stay read-only",
    )
    length(batches) == TRAINING_STATE_BATCH || error(
        "barrierless execution received a state batch inconsistent with EVRL_STATE_BATCH",
    )
    scheduler = trainer.scheduler
    executor = scheduler.barrierless_executor
    executor isa BarrierlessExecutor || error("barrierless executor is not attached")
    executor.started || error("barrierless persistent worker team is not running")

    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    begin_barrierless_update!(
        executor,
        trainer,
        batches;
        expected_update,
        hyperparameters,
        baseline,
    )
    state_results = wait_barrierless_gradients!(executor)
    scheduler.candidate_wall_nanoseconds += UInt128(time_ns() - wall_started)
    scheduler.candidate_cpu_ticks_100ns += UInt128(
        CpuSets.process_cpu_ticks_100ns() - cpu_started,
    )

    # Usage is persistent optimizer/checkpoint state.  Preserve the former
    # state-major/candidate-major recording order rather than racing workers.
    for (state_slot, batch) in enumerate(batches)
        workspace = scheduler.state_workspaces[state_slot]
        count = _valid_candidate_count(batch)
        @inbounds for candidate in 1:count
            tape = workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            Model.record_usage!(trainer.usage, tape)
        end
        all(isfinite, workspace.raw) || error("model output is non-finite")
    end
    return state_results
end

function _accumulate_dynamic_batches!(
    trainer::TeacherTrainer,
    batches;
    expected_update::Int,
    hyperparameters,
    baseline::Float32,
)
    total = _dynamic_forward_batches!(
        trainer, batches; expected_update, hyperparameters,
    )
    scheduler = trainer.scheduler
    state_results = Vector{Any}(undef, length(batches))
    # Loss semantics stay state-local and the four state gradients are averaged
    # only after backward, exactly as in the original implementation.
    for (state_slot, batch) in enumerate(batches)
        workspace = scheduler.state_workspaces[state_slot]
        raw = workspace.raw
        hard_negative = _hard_negative_selection(
            raw, batch, hyperparameters.loss.margin_mode,
        )
        components = _weighted_components(
            raw, batch, hyperparameters; hard_negative,
        )
        loss, raw_gradient = _loss_output_vjp(
            raw, batch, hyperparameters; hard_negative,
        )
        scheduler.raw_gradients[state_slot] .= raw_gradient
        scheduler.state_losses[state_slot] = loss
        count = _valid_candidate_count(batch)
        halt_probe = _apply_halt_probes!(
            trainer,
            batch,
            workspace,
            state_slot,
            expected_update,
            hyperparameters,
            1,
        )
        state_results[state_slot] = (;
            loss,
            components=_component_record(components),
            candidate_count=count,
            depths=Int.(view(workspace.depths, 1:count)),
            halt_probe,
        )
    end

    temperature = routing_temperature(expected_update, hyperparameters.routing)
    local_accumulators = trainer.thread_accumulators
    _run_candidate_queue!(trainer, total) do job, _
        state_slot = Int(scheduler.job_states[job])
        candidate = Int(scheduler.job_candidates[job])
        tape = scheduler.state_workspaces[state_slot].tapes[candidate]
        tape === nothing && error("training trajectory is missing")
        Model.backward_trajectory!(
            local_accumulators[Base.Threads.threadid()],
            trainer.model,
            tape,
            @view(scheduler.raw_gradients[state_slot][:, candidate]);
            realized_loss=scheduler.state_losses[state_slot],
            baseline,
            compute_price=hyperparameters.halting.compute_price,
            policy_weight=hyperparameters.halting.policy_weight,
            entropy_weight=hyperparameters.halting.entropy_weight,
            halt_probe_mode=hyperparameters.halting.probe_candidates_per_state > 0,
            halt_probe_target=scheduler.state_workspaces[state_slot].halt_probe_targets[candidate],
            halt_probe_weight=hyperparameters.halting.probe_weight,
            temperature,
        )
        scheduler.trajectory_states[job] = nothing
    end
    return state_results
end

function _accumulate_state_gradient!(
    trainer::TeacherTrainer,
    batch,
    accumulator::Model.GradientAccumulator;
    expected_update::Int,
    hyperparameters,
    baseline::Float32,
)
    raw, candidate_count = _timed_candidate_region!(trainer) do
        predict_raw!(
            trainer, batch; training=true, expected_update, hyperparameters,
        )
    end
    hard_negative = _hard_negative_selection(
        raw, batch, hyperparameters.loss.margin_mode,
    )
    components = _weighted_components(
        raw, batch, hyperparameters; hard_negative,
    )
    loss, raw_gradient = _loss_output_vjp(
        raw, batch, hyperparameters; hard_negative,
    )
    halt_probe = _apply_halt_probes!(
        trainer,
        batch,
        trainer.workspace,
        1,
        expected_update,
        hyperparameters,
        1,
    )
    temperature = routing_temperature(expected_update, hyperparameters.routing)
    local_accumulators = trainer.thread_accumulators
    _timed_candidate_region!(trainer) do
      if Base.Threads.nthreads() > 1
        first_parallel_candidate = 1
        if !BACKWARD_THREAD_WARMED[]
            first_tape = trainer.workspace.tapes[1]
            first_tape === nothing && error("training trajectory is missing")
            Model.backward_trajectory!(
                local_accumulators[1],
                trainer.model,
                first_tape,
                @view(raw_gradient[:, 1]);
                realized_loss=loss,
                baseline,
                compute_price=hyperparameters.halting.compute_price,
                policy_weight=hyperparameters.halting.policy_weight,
                entropy_weight=hyperparameters.halting.entropy_weight,
                halt_probe_mode=hyperparameters.halting.probe_candidates_per_state > 0,
                halt_probe_target=trainer.workspace.halt_probe_targets[1],
                halt_probe_weight=hyperparameters.halting.probe_weight,
                temperature,
            )
            BACKWARD_THREAD_WARMED[] = true
            first_parallel_candidate = 2
        end
        Base.Threads.@threads :static for candidate in first_parallel_candidate:candidate_count
            tape = trainer.workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            Model.backward_trajectory!(
                local_accumulators[Base.Threads.threadid()],
                trainer.model,
                tape,
                @view(raw_gradient[:, candidate]);
                realized_loss=loss,
                baseline,
                compute_price=hyperparameters.halting.compute_price,
                policy_weight=hyperparameters.halting.policy_weight,
                entropy_weight=hyperparameters.halting.entropy_weight,
                halt_probe_mode=hyperparameters.halting.probe_candidates_per_state > 0,
                halt_probe_target=trainer.workspace.halt_probe_targets[candidate],
                halt_probe_weight=hyperparameters.halting.probe_weight,
                temperature,
            )
        end
      else
        for candidate in 1:candidate_count
            tape = trainer.workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            Model.backward_trajectory!(
                local_accumulators[1],
                trainer.model,
                tape,
                @view(raw_gradient[:, candidate]);
                realized_loss=loss,
                baseline,
                compute_price=hyperparameters.halting.compute_price,
                policy_weight=hyperparameters.halting.policy_weight,
                entropy_weight=hyperparameters.halting.entropy_weight,
                halt_probe_mode=hyperparameters.halting.probe_candidates_per_state > 0,
                halt_probe_target=trainer.workspace.halt_probe_targets[candidate],
                halt_probe_weight=hyperparameters.halting.probe_weight,
                temperature,
            )
        end
      end
    end
    return (;
        loss,
        components=_component_record(components),
        candidate_count,
        depths=Int.(view(trainer.workspace.depths, 1:candidate_count)),
        halt_probe,
    )
end

function _mean_records(records)
    names = propertynames(first(records))
    return NamedTuple{names}(Tuple(
        mean(getproperty(record, name) for record in records) for name in names
    ))
end

function train_accumulated_step!(
    trainer::TeacherTrainer,
    batches;
    expected_update::Int,
    hyperparameters,
)
    expected_update == trainer.update + 1 || error("non-adjacent training update")
    isempty(batches) && error("empty state batch")
    started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    scheduler_mode = trainer.scheduler.config.mode
    barrierless_executor = scheduler_mode === :barrierless ?
        trainer.scheduler.barrierless_executor : nothing
    accumulator = trainer.scheduler.merged_accumulator
    Model.reset_gradients!(accumulator)
    # A training update contains four independently packed states.  Keep each
    # thread's gradients live across all four states, then perform one ordered
    # reduction.  Resetting and reducing every dense thread-local buffer once
    # per state made synchronization dominate this physically sparse model.
    if scheduler_mode !== :barrierless
        for local_accumulator in trainer.thread_accumulators
            Model.reset_gradients!(local_accumulator)
        end
    end
    baseline = trainer.baseline
    state_results = if scheduler_mode === :static
        map(batches) do batch
            _accumulate_state_gradient!(
                trainer,
                batch,
                accumulator;
                expected_update,
                hyperparameters,
                baseline,
            )
        end
    elseif scheduler_mode === :barrierless
        _accumulate_barrierless_batches!(
            trainer,
            batches;
            expected_update,
            hyperparameters,
            baseline,
        )
    else
        _accumulate_dynamic_batches!(
            trainer,
            batches;
            expected_update,
            hyperparameters,
            baseline,
        )
    end
    state_batch = length(state_results)
    opt = hyperparameters.optimizer
    episodic_lr_scale = expected_update > opt.episodic_decay_after_update ?
        opt.episodic_decay_factor : 1.0f0
    optimizer_record = if scheduler_mode === :barrierless
        barrierless_reduce_and_optimizer!(
            barrierless_executor,
            trainer,
            hyperparameters,
            expected_update,
        )
    else
        # Fixed worker-slot order preserves the previous deterministic
        # reduction semantics.  The barrierless path performs the same order
        # through parameter jobs on its persistent native-worker team.
        for local_accumulator in trainer.thread_accumulators
            Model.merge_gradients!(accumulator, local_accumulator)
        end
        Model.scale_gradients!(accumulator, inv(Float32(state_batch)))
        Model.optimizer_step!(
            trainer.model,
            trainer.optimizer,
            accumulator;
            clip_norm=opt.gradient_clip_norm,
            beta1=opt.beta1,
            beta2=opt.beta2,
            epsilon=opt.epsilon,
            router_learning_rate=opt.router_learning_rate,
            lookup_alpha_learning_rate=opt.lookup_alpha_learning_rate,
            attention_learning_rate=opt.attention_learning_rate * episodic_lr_scale,
            ffn_learning_rate=opt.ffn_learning_rate * episodic_lr_scale,
            token_learning_rate=opt.token_learning_rate * episodic_lr_scale,
            register_learning_rate=opt.register_learning_rate * episodic_lr_scale,
            head_learning_rate=opt.head_learning_rate * episodic_lr_scale,
            halt_learning_rate=opt.halt_learning_rate,
            dense_weight_decay=opt.dense_weight_decay,
        )
    end
    if scheduler_mode === :barrierless
        # This is the sole update boundary: all gradients were reduced and the
        # optimizer completed before any next-update forward can be enqueued.
        # Keep the exact overall-update measurement open until the returned
        # step record below has also been fully constructed.
        finish_barrierless_update!(
            trainer.scheduler.barrierless_executor;
            finish_measurement=false,
        )
    end
    losses = [result.loss for result in state_results]
    depths = reduce(vcat, (result.depths for result in state_results))
    halt_probe_count = sum(result.halt_probe.count for result in state_results)
    halt_probe_continue_count = sum(
        result.halt_probe.continue_count for result in state_results
    )
    halt_probe_stop_count = sum(
        result.halt_probe.stop_count for result in state_results
    )
    halt_probe_mean_delta = halt_probe_count == 0 ? 0.0 : sum(
        Float64(result.halt_probe.mean_delta) * result.halt_probe.count
        for result in state_results
    ) / halt_probe_count
    trainer.baseline = muladd(
        0.99f0,
        trainer.baseline,
        0.01f0 * mean(
            result.loss + hyperparameters.halting.compute_price * Float32(mean(result.depths))
            for result in state_results
        ),
    )
    trainer.update = expected_update
    trainer.timed_updates += 1
    trainer.timed_states += UInt64(state_batch)
    trainer.timed_candidates += UInt64(sum(result.candidate_count for result in state_results))
    trainer.timed_recurrent_steps += UInt64(sum(depths))
    elapsed = UInt128(time_ns() - started)
    trainer.training_nanoseconds += elapsed
    trainer.training_cpu_ticks_100ns += UInt128(
        CpuSets.process_cpu_ticks_100ns() - cpu_started,
    )
    step_record = (;
        update=expected_update,
        state_batch,
        candidate_count=sum(result.candidate_count for result in state_results),
        loss=Float64(mean(losses)),
        components=_mean_records([result.components for result in state_results]),
        mean_depth=mean(depths),
        minimum_depth=minimum(depths),
        maximum_depth=maximum(depths),
        halt_probe_count,
        halt_probe_continue_count,
        halt_probe_stop_count,
        halt_probe_mean_delta,
        routing_temperature=Float64(routing_temperature(expected_update, hyperparameters.routing)),
        episodic_learning_rate_scale=Float64(episodic_lr_scale),
        baseline=Float64(trainer.baseline),
        optimizer=optimizer_record,
        training_seconds=Float64(elapsed) * 1.0e-9,
    )
    if scheduler_mode === :barrierless
        finish_barrierless_update_measurement!(barrierless_executor)
    end
    return step_record
end

train_step!(trainer, batch; expected_update::Int, hyperparameters) =
    train_accumulated_step!(trainer, (batch,); expected_update, hyperparameters)

function held_evaluation(trainer, dataset, rows, host_batch; hyperparameters)
    depths = Int[]
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        Int.(rows),
        host_batch,
        batch -> begin
            raw, count = predict_raw!(
                trainer,
                batch;
                training=false,
                expected_update=trainer.update,
                hyperparameters,
                record_tapes=false,
            )
            append!(depths, Int.(view(trainer.workspace.depths, 1:count)))
            raw_output(raw)
        end;
        margin_weight=hyperparameters.loss.margin_weight,
        margin_mode=hyperparameters.loss.margin_mode,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    return (;
        metrics=merge(metrics, (; composite_loss=_weighted_loss(metrics, hyperparameters.loss))),
        mean_depth=mean(depths),
        minimum_depth=minimum(depths),
        maximum_depth=maximum(depths),
    )
end

function _throughput(trainer, start=nothing)
    timed_updates = start === nothing ? trainer.timed_updates : trainer.timed_updates - start.timed_updates
    timed_states = start === nothing ? trainer.timed_states : trainer.timed_states - start.timed_states
    timed_candidates = start === nothing ? trainer.timed_candidates : trainer.timed_candidates - start.timed_candidates
    timed_steps = start === nothing ? trainer.timed_recurrent_steps : trainer.timed_recurrent_steps - start.timed_recurrent_steps
    nanoseconds = start === nothing ? trainer.training_nanoseconds : trainer.training_nanoseconds - start.training_nanoseconds
    cpu_ticks = start === nothing ? trainer.training_cpu_ticks_100ns :
        trainer.training_cpu_ticks_100ns - start.training_cpu_ticks_100ns
    candidate_wall = start === nothing ? trainer.scheduler.candidate_wall_nanoseconds :
        trainer.scheduler.candidate_wall_nanoseconds - start.candidate_wall_nanoseconds
    candidate_cpu_ticks = start === nothing ? trainer.scheduler.candidate_cpu_ticks_100ns :
        trainer.scheduler.candidate_cpu_ticks_100ns - start.candidate_cpu_ticks_100ns
    seconds = Float64(nanoseconds) * 1.0e-9
    detected = max(trainer.scheduler.config.detected_workers, 1)
    assigned = max(trainer.scheduler.config.active_workers, 1)
    average_cpu = nanoseconds == 0 ? 0.0 :
        100.0 * (Float64(cpu_ticks) * 100.0) / (Float64(nanoseconds) * detected)
    candidate_cpu = candidate_wall == 0 ? 0.0 :
        100.0 * (Float64(candidate_cpu_ticks) * 100.0) /
        (Float64(candidate_wall) * detected)
    candidate_assigned_cpu = candidate_wall == 0 ? 0.0 :
        100.0 * (Float64(candidate_cpu_ticks) * 100.0) /
        (Float64(candidate_wall) * assigned)
    return (;
        training_seconds=seconds,
        updates_per_second=seconds == 0 ? 0.0 : Float64(timed_updates) / seconds,
        states_per_second=seconds == 0 ? 0.0 : Float64(timed_states) / seconds,
        candidates_per_second=seconds == 0 ? 0.0 : Float64(timed_candidates) / seconds,
        recurrent_steps_per_second=seconds == 0 ? 0.0 : Float64(timed_steps) / seconds,
        average_cpu_percent=average_cpu,
        candidate_cpu_percent=candidate_cpu,
        candidate_assigned_cpu_percent=candidate_assigned_cpu,
    )
end

_counter_snapshot(trainer) = (;
    timed_updates=trainer.timed_updates,
    timed_states=trainer.timed_states,
    timed_candidates=trainer.timed_candidates,
    timed_recurrent_steps=trainer.timed_recurrent_steps,
    training_nanoseconds=trainer.training_nanoseconds,
    training_cpu_ticks_100ns=trainer.training_cpu_ticks_100ns,
    candidate_wall_nanoseconds=trainer.scheduler.candidate_wall_nanoseconds,
    candidate_cpu_ticks_100ns=trainer.scheduler.candidate_cpu_ticks_100ns,
)

_sha256_file(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function source_fingerprint()
    io = IOBuffer()
    for path in (
        joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"),
        joinpath(@__DIR__, "windows_cpu_sets.jl"),
        joinpath(@__DIR__, "bounded_mpmc_queue.jl"),
        joinpath(@__DIR__, "barrierless_executor.jl"),
        joinpath(@__DIR__, "barrierless_postphase.jl"),
        joinpath(@__DIR__, "..", "dynamic_sparse_recurrent_lookup",
                 "DynamicSparseRecurrentLookup.jl"),
        @__FILE__,
    )
        write(io, codeunits(abspath(path)))
        write(io, read(path))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _append_jsonl(path, record)
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
end

function _write_json(path, record)
    open(path, "w") do io
        JSON3.pretty(io, record)
        write(io, '\n')
        flush(io)
    end
end

function save_checkpoint(path, trainer, sampler, history, config, split_metadata)
    consumed_states = TrainingCore.sampler_consumed_states(sampler)
    payload = (;
        format="episodic-vit-recurrent-lookup-checkpoint",
        version=1,
        update=trainer.update,
        model=trainer.model,
        optimizer=trainer.optimizer,
        halt_rng=copy(trainer.halt_rng),
        usage=trainer.usage,
        baseline=trainer.baseline,
        timed_updates=trainer.timed_updates,
        timed_states=trainer.timed_states,
        timed_candidates=trainer.timed_candidates,
        timed_recurrent_steps=trainer.timed_recurrent_steps,
        training_nanoseconds=trainer.training_nanoseconds,
        training_cpu_ticks_100ns=trainer.training_cpu_ticks_100ns,
        candidate_wall_nanoseconds=trainer.scheduler.candidate_wall_nanoseconds,
        candidate_cpu_ticks_100ns=trainer.scheduler.candidate_cpu_ticks_100ns,
        consumed_states,
        sampler_snapshot=TrainingCore.sampler_snapshot(sampler),
        history,
        config,
        split_metadata,
    )
    temporary = path * ".tmp-" * string(getpid())
    open(temporary, "w") do io
        serialize(io, payload)
        flush(io)
    end
    mv(temporary, path; force=true)
    return (;
        path=abspath(path),
        bytes=filesize(path),
        sha256=_sha256_file(path),
        update=trainer.update,
        consumed_states,
    )
end

function read_checkpoint(path::AbstractString, expected_sha256::AbstractString="")
    checkpoint_path = abspath(path)
    isfile(checkpoint_path) || error("resume checkpoint does not exist: $checkpoint_path")
    actual_sha256 = _sha256_file(checkpoint_path)
    expected = lowercase(strip(expected_sha256))
    isempty(expected) || actual_sha256 == expected || error(
        "resume checkpoint SHA-256 differs",
    )
    payload = open(checkpoint_path, "r") do io
        deserialize(io)
    end
    hasproperty(payload, :format) || error("resume checkpoint format is missing")
    payload.format == "episodic-vit-recurrent-lookup-checkpoint" || error(
        "resume checkpoint format differs",
    )
    hasproperty(payload, :version) && Int(payload.version) == 1 || error(
        "unsupported resume checkpoint version",
    )
    return payload, (;
        path=checkpoint_path,
        bytes=filesize(checkpoint_path),
        sha256=actual_sha256,
    )
end

function _assert_current_split(payload, split)
    metadata = payload.split_metadata
    hasproperty(metadata, :training_groups) || error(
        "resume split metadata lacks training groups",
    )
    hasproperty(metadata, :validation_groups) || error(
        "resume split metadata lacks validation groups",
    )
    hasproperty(metadata, :predefined) || error(
        "resume split metadata lacks predefined split identity",
    )
    metadata.training_groups == split.training_groups || error(
        "resume training groups differ",
    )
    metadata.validation_groups == split.validation_groups || error(
        "resume validation groups differ",
    )
    metadata.predefined == split.predefined || error(
        "resume predefined split identity differs",
    )
    return nothing
end

function restore_checkpoint(
    payload,
    split,
    split_metadata,
    dataset_manifest_sha256,
    hyperparameters,
)
    required = (
        :update, :model, :optimizer, :halt_rng, :usage, :baseline,
        :timed_updates, :timed_states, :timed_candidates,
        :timed_recurrent_steps, :training_nanoseconds, :sampler_snapshot,
        :history, :config, :split_metadata,
    )
    all(name -> hasproperty(payload, name), required) || error(
        "resume checkpoint is incomplete",
    )
    update = Int(payload.update)
    update >= 0 || error("resume update is negative")
    hasproperty(payload.config, :experiment_id) &&
        payload.config.experiment_id == EXPERIMENT_ID || error(
            "resume experiment identity differs",
        )
    state_batch = Int(_property_or(payload.config, :state_batch, 0))
    state_batch == TRAINING_STATE_BATCH || error(
        "resume state batch differs from EVRL_STATE_BATCH",
    )
    hasproperty(payload.config, :model) || error("resume config lacks model topology")
    payload.config.model == Model.topology(payload.model) || error(
        "resume model topology differs from the live EVRL geometry",
    )
    hasproperty(payload.config, :total_parameter_count) || error(
        "resume config lacks total parameter count",
    )
    Model.parameter_count(payload.model) == Int(payload.config.total_parameter_count) ||
        error("resume model parameter count differs")
    hasproperty(payload.config, :hyperparameters) || error(
        "resume config lacks hyperparameters",
    )
    inherited_normalized = _normalized_hyperparameters(
        payload.config.hyperparameters,
    )
    requested_normalized = _normalized_hyperparameters(hyperparameters)
    if inherited_normalized != requested_normalized
        transition_raw = strip(get(ENV, "EVRL_ENABLE_DYNAMIC_HALTING_TRANSITION", "0"))
        transition_raw in ("0", "1") || error(
            "EVRL_ENABLE_DYNAMIC_HALTING_TRANSITION must be 0 or 1",
        )
        inherited = inherited_normalized
        inherited_halting = inherited.halting
        requested_halting = hyperparameters.halting
        dynamic_halting_transition = transition_raw == "1" &&
            inherited.optimizer == hyperparameters.optimizer &&
            inherited.routing == hyperparameters.routing &&
            inherited.loss == hyperparameters.loss &&
            requested_halting.warmup_updates >= inherited_halting.warmup_updates &&
            inherited_halting.compute_price == requested_halting.compute_price &&
            inherited_halting.policy_weight == requested_halting.policy_weight &&
            inherited_halting.entropy_weight == requested_halting.entropy_weight &&
            inherited_halting.probe_candidates_per_state ==
                requested_halting.probe_candidates_per_state &&
            inherited_halting.probe_weight == requested_halting.probe_weight &&
            inherited_halting.fixed_depth != 0 &&
            requested_halting.fixed_depth == 0
        probe_transition_raw = strip(get(
            ENV, "EVRL_ENABLE_HALT_PROBE_TRANSITION", "0",
        ))
        probe_transition_raw in ("0", "1") || error(
            "EVRL_ENABLE_HALT_PROBE_TRANSITION must be 0 or 1",
        )
        halt_probe_transition = probe_transition_raw == "1" &&
            inherited.optimizer == requested_normalized.optimizer &&
            inherited.routing == requested_normalized.routing &&
            inherited.loss == requested_normalized.loss &&
            inherited_halting.warmup_updates == requested_halting.warmup_updates &&
            inherited_halting.fixed_depth == requested_halting.fixed_depth &&
            inherited_halting.policy_weight == requested_halting.policy_weight &&
            inherited_halting.entropy_weight == requested_halting.entropy_weight &&
            inherited_halting.probe_candidates_per_state == 0 &&
            requested_halting.probe_candidates_per_state > 0
        halt_lr_transition_raw = strip(get(
            ENV, "EVRL_ENABLE_HALT_LR_TRANSITION", "0",
        ))
        halt_lr_transition_raw in ("0", "1") || error(
            "EVRL_ENABLE_HALT_LR_TRANSITION must be 0 or 1",
        )
        halt_lr_transition = halt_lr_transition_raw == "1" &&
            inherited.routing == requested_normalized.routing &&
            inherited.loss == requested_normalized.loss &&
            inherited.halting == requested_normalized.halting &&
            merge(inherited.optimizer, (;
                halt_learning_rate=requested_normalized.optimizer.halt_learning_rate,
            )) == requested_normalized.optimizer
        dense_wd_transition_raw = strip(get(
            ENV, "EVRL_ENABLE_DENSE_WD_TRANSITION", "0",
        ))
        dense_wd_transition_raw in ("0", "1") || error(
            "EVRL_ENABLE_DENSE_WD_TRANSITION must be 0 or 1",
        )
        dense_wd_transition = dense_wd_transition_raw == "1" &&
            inherited.routing == requested_normalized.routing &&
            inherited.loss == requested_normalized.loss &&
            inherited.halting == requested_normalized.halting &&
            merge(inherited.optimizer, (;
                dense_weight_decay=requested_normalized.optimizer.dense_weight_decay,
            )) == requested_normalized.optimizer
        dynamic_halting_transition || halt_probe_transition || halt_lr_transition ||
            dense_wd_transition || error(
            "resume hyperparameters differ; only an explicit dynamic-halting, halt-probe, halt-LR, or dense-WD transition is allowed",
        )
        dynamic_halting_transition && @info(
            "EVRL fixed-depth checkpoint transitions to sampled hard halting",
            previous_fixed_depth=inherited_halting.fixed_depth,
            requested_fixed_depth=requested_halting.fixed_depth,
        )
        halt_probe_transition && @info(
            "EVRL sampled halting transitions to sparse one-step probe supervision",
            probes_per_state=requested_halting.probe_candidates_per_state,
            probe_weight=requested_halting.probe_weight,
            continue_threshold=requested_halting.compute_price,
        )
        halt_lr_transition && @info(
            "EVRL halt learning rate transition",
            previous_halt_learning_rate=inherited.optimizer.halt_learning_rate,
            requested_halt_learning_rate=requested_normalized.optimizer.halt_learning_rate,
        )
        dense_wd_transition && @info(
            "EVRL dense weight decay transition",
            previous_dense_weight_decay=inherited.optimizer.dense_weight_decay,
            requested_dense_weight_decay=requested_normalized.optimizer.dense_weight_decay,
        )
    end
    hasproperty(payload.config, :objective) || error("resume config lacks objective")
    payload.config.objective == objective_contract(hyperparameters) || error(
        "resume objective differs",
    )
    payload.split_metadata == split_metadata || error("resume split metadata differs")
    _assert_current_split(payload, split)
    hasproperty(payload.config, :dataset_manifest_sha256) || error(
        "resume config lacks dataset manifest SHA-256",
    )
    payload.config.dataset_manifest_sha256 == dataset_manifest_sha256 || error(
        "resume dataset manifest differs",
    )
    hasproperty(payload.config, :julia_version) &&
        payload.config.julia_version == string(VERSION) || error(
            "resume Julia version differs",
        )

    expected_step = UInt64(update)
    payload.optimizer.lookup.step == expected_step || error(
        "resume lookup optimizer clock differs",
    )
    payload.optimizer.lookup.dense.step == expected_step || error(
        "resume lookup dense optimizer clock differs",
    )
    payload.optimizer.dense_step == expected_step || error(
        "resume EVRL dense optimizer clock differs",
    )
    all(state.global_step == expected_step for state in payload.optimizer.lookup.bank_states) ||
        error("resume sparse optimizer clocks differ")
    halt_reset_raw = strip(get(ENV, "EVRL_DYNAMIC_HALT_RESET_PROBABILITY", ""))
    if !isempty(halt_reset_raw)
        transition_raw = strip(get(ENV, "EVRL_ENABLE_DYNAMIC_HALTING_TRANSITION", "0"))
        inherited_halting = payload.config.hyperparameters.halting
        requested_halting = hyperparameters.halting
        transition_raw == "1" && inherited_halting.fixed_depth != 0 &&
            requested_halting.fixed_depth == 0 || error(
                "halt-head reset is allowed only during the explicit fixed-depth-to-dynamic transition",
            )
        reset_probability = tryparse(Float32, halt_reset_raw)
        reset_probability !== nothing && 0.0f0 < reset_probability < 1.0f0 ||
            error("EVRL_DYNAMIC_HALT_RESET_PROBABILITY must be strictly between zero and one")
        fill!(payload.model.lookup.halt_weight, 0.0f0)
        payload.model.lookup.halt_bias[1] = log(
            reset_probability / (1.0f0 - reset_probability),
        )
        dense = payload.optimizer.lookup.dense
        fill!(dense.mhalt_weight, 0.0f0)
        fill!(dense.vhalt_weight, 0.0f0)
        fill!(dense.mhalt_bias, 0.0f0)
        fill!(dense.vhalt_bias, 0.0f0)
        @info "EVRL halt head reset for recurrent-depth acquisition" reset_probability
    end
    length(payload.optimizer.token_event_count) == Model.TOKEN_COUNT || error(
        "resume token optimizer geometry differs",
    )
    expected_usage_shape = (
        Model.SparseLookup.ROWS_PER_TABLE,
        Model.SparseLookup.TABLES_PER_BLOCK,
        Model.SparseLookup.BLOCKS,
    )
    size(payload.usage.sparse.counts) == expected_usage_shape || error(
        "resume route-usage geometry differs",
    )

    sampler = TrainingCore.restore_sampler(split.training_rows, payload.sampler_snapshot)
    expected_consumed_states = hasproperty(payload, :consumed_states) ?
        Int(payload.consumed_states) : update * state_batch
    TrainingCore.sampler_consumed_states(sampler) == expected_consumed_states || error(
        "resume sampler position differs from recorded consumed states",
    )
    expected_consumed_states == update * state_batch || error(
        "resume sampler position differs from update/state-batch contract",
    )

    scheduler = CandidateSchedulerWorkspace(payload.model)
    scheduler.candidate_wall_nanoseconds = UInt128(_property_or(
        payload, :candidate_wall_nanoseconds, 0,
    ))
    scheduler.candidate_cpu_ticks_100ns = UInt128(_property_or(
        payload, :candidate_cpu_ticks_100ns, 0,
    ))
    trainer = TeacherTrainer(
        payload.model,
        payload.optimizer,
        copy(payload.halt_rng),
        payload.usage,
        TeacherWorkspace(),
        scheduler,
        [Model.GradientAccumulator(payload.model) for _ in 1:Base.Threads.maxthreadid()],
        Float32(payload.baseline),
        update,
        UInt64(payload.timed_updates),
        UInt64(payload.timed_states),
        UInt64(payload.timed_candidates),
        UInt64(payload.timed_recurrent_steps),
        UInt128(payload.training_nanoseconds),
        UInt128(_property_or(payload, :training_cpu_ticks_100ns, 0)),
    )
    _attach_barrierless_executor!(trainer)
    return trainer, sampler, copy(payload.history)
end

function _metric_summary(metrics)
    return (;
        composite_loss=Float64(metrics.composite_loss),
        top1_agreement=Float64(metrics.top1_agreement),
        ndcg=Float64(metrics.ndcg),
        pairwise_accuracy=Float64(metrics.pairwise_accuracy),
        action_margin=Float64(metrics.action_margin),
        states=Int(metrics.states),
    )
end

function _evaluation_record(
    trainer,
    dataset,
    training_rows,
    validation_rows,
    host_batch,
    last_step;
    hyperparameters,
    segment_start,
)
    training = held_evaluation(trainer, dataset, training_rows, host_batch; hyperparameters)
    validation = held_evaluation(trainer, dataset, validation_rows, host_batch; hyperparameters)
    return (;
        update=trainer.update,
        experiment_id=EXPERIMENT_ID,
        packed_states=Int(trainer.timed_states),
        training=_metric_summary(training.metrics),
        validation=_metric_summary(validation.metrics),
        recurrent_depth=(;
            training_mean=training.mean_depth,
            validation_mean=validation.mean_depth,
            validation_minimum=validation.minimum_depth,
            validation_maximum=validation.maximum_depth,
        ),
        route_usage=Model.usage_summary(trainer.usage),
        throughput=_throughput(trainer),
        segment_throughput=_throughput(trainer, segment_start),
        last_step,
        parameter_contract=Model.topology(trainer.model),
        objective=objective_contract(hyperparameters),
        source_fingerprint=source_fingerprint(),
    )
end

function teacher_signal_cli_main()
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 || error("nested BLAS parallelism is forbidden")
    Base.Threads.nthreads(:interactive) == 0 || error(
        "EVRL requires JULIA_NUM_THREADS=<workers>,0",
    )
    maximum_updates = _int_env("EVRL_MAX_UPDATES", 1_000; minimum=1)
    benchmark_raw = strip(get(ENV, "EVRL_BENCHMARK_ONLY", "0"))
    benchmark_raw in ("0", "1") || error("EVRL_BENCHMARK_ONLY must be 0 or 1")
    benchmark_only = benchmark_raw == "1"
    detailed_raw = strip(get(ENV, "EVRL_DETAILED_BENCHMARK", benchmark_only ? "1" : "0"))
    detailed_raw in ("0", "1") || error(
        "EVRL_DETAILED_BENCHMARK must be 0 or 1",
    )
    detailed_benchmark = detailed_raw == "1"
    write_checkpoints_raw = strip(get(ENV, "EVRL_WRITE_CHECKPOINTS", "1"))
    write_checkpoints_raw in ("0", "1") || error(
        "EVRL_WRITE_CHECKPOINTS must be 0 or 1",
    )
    write_checkpoints = write_checkpoints_raw == "1"
    minimum_updates_per_second_raw = strip(get(
        ENV, "EVRL_MIN_UPDATES_PER_SECOND", "0",
    ))
    minimum_updates_per_second = tryparse(
        Float64, minimum_updates_per_second_raw,
    )
    minimum_updates_per_second === nothing && error(
        "EVRL_MIN_UPDATES_PER_SECOND must be a finite nonnegative number",
    )
    isfinite(minimum_updates_per_second) && minimum_updates_per_second >= 0 ||
        error("EVRL_MIN_UPDATES_PER_SECOND must be a finite nonnegative number")
    minimum_throughput_updates = _int_env(
        "EVRL_MIN_THROUGHPUT_UPDATES", 100; minimum=1,
    )
    benchmark_warmup_updates = benchmark_only ?
        _int_env("EVRL_BENCHMARK_WARMUP_UPDATES", 0; minimum=0) : 0
    allocation_profile_sample_rate = if detailed_benchmark
        raw = strip(get(ENV, "EVRL_ALLOC_PROFILE_SAMPLE_RATE", "0.01"))
        rate = tryparse(Float64, raw)
        rate === nothing && error(
            "EVRL_ALLOC_PROFILE_SAMPLE_RATE must be in (0,1]",
        )
        isfinite(rate) && 0.0 < rate <= 1.0 || error(
            "EVRL_ALLOC_PROFILE_SAMPLE_RATE must be in (0,1]",
        )
        rate
    else
        0.0
    end
    resume_path = strip(get(ENV, "EVRL_RESUME_CHECKPOINT", ""))
    resume_sha256 = strip(get(ENV, "EVRL_RESUME_SHA256", ""))
    resume_payload, resume_artifact = isempty(resume_path) ? (nothing, nothing) :
        read_checkpoint(resume_path, resume_sha256)
    inherited_state_batch = resume_payload === nothing ? TRAINING_STATE_BATCH :
        Int(_property_or(resume_payload.config, :state_batch, TRAINING_STATE_BATCH))
    state_batch = _int_env("EVRL_STATE_BATCH", inherited_state_batch; minimum=1)
    state_batch == TRAINING_STATE_BATCH || error(
        "EVRL_STATE_BATCH changed after the training module was loaded",
    )
    evaluation_interval = _int_env("EVRL_EVAL_INTERVAL", maximum_updates; minimum=1)
    checkpoint_interval = _int_env("EVRL_CHECKPOINT_INTERVAL", maximum_updates; minimum=1)
    inherited_dataset_path = resume_payload === nothing ? DEFAULT_DATASET :
        String(_property_or(resume_payload.config, :dataset_path, DEFAULT_DATASET))
    dataset_path = abspath(get(ENV, "EVRL_DATASET", inherited_dataset_path))
    output_root = abspath(get(ENV, "EVRL_OUTPUT", DEFAULT_OUTPUT))
    run_id = strip(get(ENV, "EVRL_RUN_ID", ""))
    isempty(run_id) && (run_id = "episodic_vit_recurrent_lookup_" * Dates.format(now(), "yyyymmddTHHMMSS"))
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("fresh run already exists: $run_dir")
    hyperparameters = runtime_hyperparameters(maximum_updates, resume_payload)

    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    observed_width = maximum(dataset.action_counts)
    observed_width == 76 || error("teacher_v3 observed candidate width changed: $observed_width")
    16 * cld(observed_width, 16) == LEARNER_WIDTH || error("rounded learner width changed")
    split = ParentTraining.episode_separated_split(
        dataset;
        seed=SPLIT_SEED,
        validation_fraction=0.10,
    )
    training_eval_rows, validation_eval_rows, split_metadata = if benchmark_only
        metadata = resume_payload === nothing ? (;
            training_eval_rows=Int[],
            validation_eval_rows=Int[],
            training_groups=copy(split.training_groups),
            validation_groups=copy(split.validation_groups),
            predefined=split.predefined,
        ) : resume_payload.split_metadata
        Int[], Int[], metadata
    else
        training_rows = ParentTraining.fixed_evaluation_subset(
            split.training_rows, TRAIN_EVAL_STATES, TRAIN_EVAL_SEED,
        )
        validation_rows = ParentTraining.fixed_evaluation_subset(
            split.validation_rows, VALIDATION_EVAL_STATES, VALIDATION_EVAL_SEED,
        )
        metadata = (;
            training_eval_rows=Int.(training_rows),
            validation_eval_rows=Int.(validation_rows),
            training_groups=copy(split.training_groups),
            validation_groups=copy(split.validation_groups),
            predefined=split.predefined,
        )
        training_rows, validation_rows, metadata
    end
    manifest_path = joinpath(dataset_path, "manifest.json")
    dataset_manifest_sha256 = _sha256_file(manifest_path)
    trainer, sampler, history = if resume_payload === nothing
        fresh_trainer = initialize_trainer(hyperparameters)
        fresh_sampler = TrainingCore.EpochSampler(split.training_rows, Xoshiro(SAMPLER_SEED))
        fresh_trainer, fresh_sampler, Any[]
    else
        restored_trainer, restored_sampler, restored_history = restore_checkpoint(
            resume_payload,
            split,
            split_metadata,
            dataset_manifest_sha256,
            hyperparameters,
        )
        optimizer = hyperparameters.optimizer
        Model.configure_bank_optimizer!(
            restored_trainer.optimizer;
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
            bank_learning_rate=optimizer.bank_learning_rate,
            bank_weight_decay=optimizer.bank_weight_decay,
        )
        restored_trainer, restored_sampler, restored_history
    end
    maximum_updates > trainer.update || error(
        "EVRL_MAX_UPDATES is an absolute target and must exceed the starting update",
    )
    starting_update = trainer.update
    starting_consumed_states = TrainingCore.sampler_consumed_states(sampler)
    starting_update + benchmark_warmup_updates < maximum_updates || error(
        "benchmark warmup must leave at least one measured update",
    )
    # Profile.Allocs decoding/clearing can itself perturb GC state.  Profile
    # the penultimate warmup update so one ordinary warmup update always runs
    # after decoding and before the measured counters are reset.
    allocation_profile_update = benchmark_only && detailed_benchmark &&
        benchmark_warmup_updates >= 2 ?
        starting_update + benchmark_warmup_updates - 1 : nothing
    segment_start = _counter_snapshot(trainer)
    parent_checkpoint = resume_payload === nothing ? nothing : (;
        resume_artifact...,
        update=Int(resume_payload.update),
        run_id=String(_property_or(resume_payload.config, :run_id, "")),
        source_fingerprint=String(_property_or(
            resume_payload.config, :source_fingerprint, "",
        )),
    )
    config = (;
        experiment_id=EXPERIMENT_ID,
        run_id,
        dataset_path,
        dataset_manifest_sha256,
        source_fingerprint=source_fingerprint(),
        julia_version=string(VERSION),
        maximum_updates,
        starting_update,
        starting_consumed_states,
        state_batch,
        evaluation_interval,
        checkpoint_interval,
        benchmark_only,
        detailed_benchmark,
        write_checkpoints,
        minimum_updates_per_second,
        minimum_throughput_updates,
        benchmark_warmup_updates,
        benchmark_measured_updates=maximum_updates - starting_update - benchmark_warmup_updates,
        allocation_profile_sample_rate,
        allocation_profile_update,
        parent_checkpoint,
        training_seed=TRAIN_SEED,
        split_seed=SPLIT_SEED,
        sampler_seed=SAMPLER_SEED,
        training_eval_seed=TRAIN_EVAL_SEED,
        validation_eval_seed=VALIDATION_EVAL_SEED,
        training_eval_states=benchmark_only ? 0 : length(training_eval_rows),
        validation_eval_states=benchmark_only ? 0 : length(validation_eval_rows),
        observed_candidate_width=observed_width,
        learner_width=LEARNER_WIDTH,
        model=Model.topology(trainer.model),
        total_parameter_count=Model.parameter_count(trainer.model),
        scheduler=(;
            mode=trainer.scheduler.config.mode,
            cpuset_mode=trainer.scheduler.config.cpuset_mode,
            chunk_size=trainer.scheduler.config.chunk_size,
            adaptive_tail=trainer.scheduler.config.adaptive_tail,
            active_workers=trainer.scheduler.config.active_workers,
            julia_default_workers=Base.Threads.nthreads(:default),
            julia_interactive_workers=Base.Threads.nthreads(:interactive),
            blas_threads=BLAS.get_num_threads(),
            topology=CpuSets.topology_summary(trainer.scheduler.config.topology),
        ),
        hyperparameters,
        objective=objective_contract(hyperparameters),
        input_contract=(;
            board=(24, 10, 1),
            candidate=(24, 10, 1),
            difference=(24, 10, 1),
            next_hold=(7, 6),
            aux=37,
            local_mask_used=false,
            candidate_set_features_used=false,
            countsketch_used=false,
            candidates_evaluated_independently=true,
        ),
        forbidden_game_seeds_touched=false,
    )

    mkpath(joinpath(run_dir, "checkpoints"))
    _write_json(joinpath(run_dir, "config.json"), config)
    metrics_path = joinpath(run_dir, "metrics.jsonl")
    host_batches = [
        TrainingCore.allocate_host_batch(1; max_candidates=LEARNER_WIDTH)
        for _ in 1:state_batch
    ]
    evaluation_batch = benchmark_only ? nothing :
        TrainingCore.allocate_host_batch(1; max_candidates=LEARNER_WIDTH)
    if !benchmark_only
        initial = _evaluation_record(
            trainer,
            dataset,
            training_eval_rows,
            validation_eval_rows,
            evaluation_batch,
            nothing;
            hyperparameters,
            segment_start,
        )
        push!(history, initial)
        _append_jsonl(metrics_path, initial)
        @info "Episodic ViT recurrent Lookup initial teacher signal" training=initial.training validation=initial.validation
    end

    stopped_for_throughput = Ref(false)
    sampled_allocation_profile = Ref{Any}(nothing)
    training_driver = function (_executor)
        if _executor isa BarrierlessExecutor
            reset_barrierless_benchmark_statistics!(
                _executor; enabled=detailed_benchmark,
            )
        end
        local_last_step = nothing
        while trainer.update < maximum_updates
            pack_measurement = nothing
            if detailed_benchmark && _executor isa BarrierlessExecutor
                begin_barrierless_update_measurement!(_executor)
                pack_measurement = begin_barrierless_phase_measurement(
                    _executor, :data_pack,
                )
            end
            try
                rows = TrainingCore.next_batch!(sampler, state_batch)
                for (batch, row) in zip(host_batches, rows)
                    TrainingCore.pack_batch!(batch, dataset, [row])
                end
            catch
                if _executor isa BarrierlessExecutor &&
                   barrierless_update_measurement_open(_executor)
                    abort_barrierless_update_measurement!(_executor)
                end
                rethrow()
            finally
                if pack_measurement !== nothing &&
                   barrierless_update_measurement_open(_executor)
                    finish_barrierless_phase_measurement!(
                        _executor, pack_measurement,
                    )
                end
            end
            profile_this_update = allocation_profile_update !== nothing &&
                _executor isa BarrierlessExecutor &&
                trainer.update + 1 == allocation_profile_update
            if profile_this_update
                Profile.Allocs.clear()
                Profile.Allocs.start(; sample_rate=allocation_profile_sample_rate)
                profile_succeeded = false
                try
                    local_last_step = train_accumulated_step!(
                        trainer,
                        host_batches;
                        expected_update=trainer.update + 1,
                        hyperparameters,
                    )
                    profile_succeeded = true
                finally
                    Profile.Allocs.stop()
                    profile_succeeded || Profile.Allocs.clear()
                end
                allocation_results = Profile.Allocs.fetch()
                sampled_allocation_profile[] =
                    summarize_barrierless_allocation_profile(
                        allocation_results;
                        sample_rate=allocation_profile_sample_rate,
                        update=trainer.update,
                    )
                Profile.Allocs.clear()
            else
                local_last_step = train_accumulated_step!(
                    trainer,
                    host_batches;
                    expected_update=trainer.update + 1,
                    hyperparameters,
                )
            end
            if benchmark_only && benchmark_warmup_updates > 0 &&
               trainer.update == starting_update + benchmark_warmup_updates
                segment_start = _counter_snapshot(trainer)
                if detailed_benchmark && _executor isa BarrierlessExecutor
                    reset_barrierless_benchmark_statistics!(
                        _executor; enabled=true,
                    )
                end
            end
            trainer.update % 100 == 0 && @info(
                "Episodic ViT recurrent Lookup progress",
                update=trainer.update,
                loss=local_last_step.loss,
                throughput=_throughput(trainer, segment_start),
            )
            if !benchmark_only &&
               (trainer.update % evaluation_interval == 0 || trainer.update == maximum_updates)
                record = _evaluation_record(
                    trainer,
                    dataset,
                    training_eval_rows,
                    validation_eval_rows,
                    evaluation_batch,
                    local_last_step;
                    hyperparameters,
                    segment_start,
                )
                push!(history, record)
                _append_jsonl(metrics_path, record)
                @info "Episodic ViT recurrent Lookup teacher signal" update=trainer.update training=record.training validation=record.validation throughput=record.segment_throughput
            end
            if write_checkpoints &&
               (trainer.update % checkpoint_interval == 0 || trainer.update == maximum_updates)
                checkpoint_path = joinpath(
                    run_dir,
                    "checkpoints",
                    "checkpoint_" * lpad(string(trainer.update), 9, '0') * ".jls",
                )
                artifact = save_checkpoint(
                    checkpoint_path,
                    trainer,
                    sampler,
                    history,
                    config,
                    split_metadata,
                )
                _write_json(joinpath(run_dir, "latest.json"), artifact)
                @info "Episodic ViT recurrent Lookup checkpoint" artifact
                measured_updates = trainer.timed_updates - segment_start.timed_updates
                if minimum_updates_per_second > 0 &&
                   measured_updates >= minimum_throughput_updates
                    observed = _throughput(trainer, segment_start).updates_per_second
                    if observed < minimum_updates_per_second
                        stopped_for_throughput[] = true
                        @warn(
                            "steady throughput fell below the configured floor; " *
                            "stopping after the saved update boundary",
                            update=trainer.update,
                            observed_updates_per_second=observed,
                            minimum_updates_per_second,
                            checkpoint_path,
                        )
                        break
                    end
                end
            end
        end
        return local_last_step
    end
    last_step = if trainer.scheduler.config.mode === :barrierless
        run_with_barrierless_team!(
            training_driver,
            trainer.scheduler.barrierless_executor,
        )
    else
        training_driver(nothing)
    end
    summary = (;
        status=stopped_for_throughput[] ? "stopped_low_throughput" : "complete",
        run_dir=abspath(run_dir),
        metrics_path=abspath(metrics_path),
        update=trainer.update,
        consumed_states=TrainingCore.sampler_consumed_states(sampler),
        final=isempty(history) ? nothing : last(history),
        last_step,
        throughput=_throughput(trainer),
        segment_throughput=_throughput(trainer, segment_start),
        scheduler_benchmark=trainer.scheduler.config.mode === :barrierless ?
            barrierless_benchmark_summary(
                trainer.scheduler.barrierless_executor,
                sampled_allocation_profile=sampled_allocation_profile[],
            ) : nothing,
        measured_updates=trainer.timed_updates - segment_start.timed_updates,
        target_update=maximum_updates,
        stopped_for_throughput=stopped_for_throughput[],
        benchmark_only,
        parent_checkpoint,
        source_fingerprint=config.source_fingerprint,
    )
    _write_json(joinpath(run_dir, "summary.json"), summary)
    return summary
end

export TeacherTrainer, initialize_trainer, predict_raw!, train_step!, train_accumulated_step!,
       held_evaluation, save_checkpoint, read_checkpoint, restore_checkpoint,
       teacher_signal_cli_main, runtime_hyperparameters, source_fingerprint

end
