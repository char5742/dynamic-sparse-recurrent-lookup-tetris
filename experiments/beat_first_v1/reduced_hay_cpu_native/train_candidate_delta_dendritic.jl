using Printf

if !isdefined(Main, :BeatFirstTrainingCore)
    include(joinpath(@__DIR__, "..", "training", "core.jl"))
end

module CandidateDeltaDendriticScratchTraining

using Dates
using JSON3
using Printf
using Random
using Serialization
using SHA

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
    "BarrierlessScheduler.jl",
    "CandidateDeltaDendriticBarrierless.jl",
    "CandidateDeltaDendriticOptimizer.jl",
)
    include(joinpath(@__DIR__, file))
end

const Core = Main.BeatFirstTrainingCore
const Ranking = TetrisRankingBatch
const Bank = DendriticProgramBank
const Model = CandidateDeltaDendriticGraph
const Output = DendriticForestOutput
const Parallel = CandidateDeltaDendriticBarrierless
const Optimizer = CandidateDeltaDendriticOptimizer

export CHECKPOINT_SCHEMA,
       EXPECTED_MANIFEST_SHA256,
       EXPECTED_VALIDATION_ROWS_SHA256,
       RandomTrainingSampler,
       ScratchOptions,
       append_json!,
       batch_metrics,
       build_checkpoint,
       fixed_validation_rows,
       load_checkpoint,
       load_width80_dataset,
       main,
       next_batch!,
       options_record,
       parse_options,
       preflight_diagnostics,
       restore_checkpoint,
       rows_sha256,
       run_training,
       save_checkpoint_atomic,
       semantic_config,
       source_fingerprint,
       validate_checkpoint!

const CANDIDATE_WIDTH = 80
const STATE_BATCH = 8
const VALIDATION_STATES = 128
const VALIDATION_SUBSET_SEED = UInt64(2026072315)
const DEFAULT_SAMPLER_SEED = UInt64(0x4344534450475341)
const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_RESULTS =
    raw"D:\tetris-paper-plus\runs\beat_first_v1\candidate_delta_dendritic"
const DEFAULT_CHECKPOINT = joinpath(DEFAULT_RESULTS, "latest.jls")
const DEFAULT_PROGRESS = joinpath(DEFAULT_RESULTS, "progress.jsonl")
const _PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const EXPECTED_MANIFEST_SHA256 =
    "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
const EXPECTED_VALIDATION_ROWS_SHA256 =
    "fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25"
const CHECKPOINT_MAGIC = "candidate_delta_dendritic_forest_scratch"
const CHECKPOINT_SCHEMA = 3

# Every source that changes the numerical model, sparse address contract,
# supervised objective, optimizer, scheduler reduction, or sampler/checkpoint
# semantics participates in the resume fingerprint.
const _SOURCE_PATHS = (
    joinpath(@__DIR__, "train_candidate_delta_dendritic.jl"),
    joinpath(@__DIR__, "TetrisRankingBatch.jl"),
    joinpath(@__DIR__, "ActiveApicalCell.jl"),
    joinpath(@__DIR__, "CandidateDeltaInput.jl"),
    joinpath(@__DIR__, "DendriticProgramBank.jl"),
    joinpath(@__DIR__, "CompactDendriticNode.jl"),
    joinpath(@__DIR__, "DendriticDeltaForestTopology.jl"),
    joinpath(@__DIR__, "DendriticDeltaForest.jl"),
    joinpath(@__DIR__, "DendriticForestOutput.jl"),
    joinpath(@__DIR__, "CandidateDeltaDendriticGraph.jl"),
    joinpath(@__DIR__, "CandidateDeltaDendriticTraining.jl"),
    joinpath(@__DIR__, "BarrierlessScheduler.jl"),
    joinpath(@__DIR__, "CandidateDeltaDendriticBarrierless.jl"),
    joinpath(@__DIR__, "CandidateDeltaDendriticOptimizer.jl"),
    joinpath(@__DIR__, "..", "training", "core.jl"),
    joinpath(
        @__DIR__, "..", "episodic_vit_recurrent_lookup",
        "bounded_mpmc_queue.jl",
    ),
    joinpath(
        @__DIR__, "..", "episodic_vit_recurrent_lookup",
        "windows_cpu_sets.jl",
    ),
    joinpath(_PROJECT_ROOT, "Project.toml"),
    joinpath(_PROJECT_ROOT, "Manifest.toml"),
)

Base.@kwdef struct ScratchOptions
    updates::Int = 1_000
    workers::Int = min(20, Threads.nthreads(:default))
    log_every::Int = 100
    evaluate_every::Int = 1_000
    checkpoint_every::Int = 100
    candidate_chunk_size::Int = 4
    queue_capacity::Int = 64
    binding_mode::Symbol = :none
    sampler_seed::UInt64 = DEFAULT_SAMPLER_SEED
    learning_rate::Float32 = 1.0f-3
    warmup_updates::Int = 1_000
    learning_rate_schedule_updates::Int = 100_000
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
    checkpoint::String = DEFAULT_CHECKPOINT
    progress::String = DEFAULT_PROGRESS
    resume::Union{Nothing,String} = nothing
end

function _usage(io::IO=stdout)
    println(io, "usage: julia --project=. --threads=20,0 train_candidate_delta_dendritic.jl [options]")
    println(io, "  --updates N                 target optimizer update (default 1000)")
    println(io, "  --workers N                 persistent barrierless workers (default 20)")
    println(io, "  --log-every N               stdout/JSONL interval (default 100)")
    println(io, "  --evaluate-every N          fixed validation interval (default 1000)")
    println(io, "  --checkpoint-every N        atomic latest interval (default 100)")
    println(io, "  --candidate-chunk-size N    forward scheduler chunk (default 4)")
    println(io, "  --queue-capacity N          power-of-two MPMC capacity (default 64)")
    println(io, "  --binding-mode none|all|p_only")
    println(io, "  --sampler-seed N")
    println(io, "  --learning-rate X --warmup-updates N")
    println(io, "  --learning-rate-schedule-updates N --min-learning-rate-ratio X")
    println(io, "  --clip-norm X --weight-decay X")
    println(io, "  --cell-weight-decay X")
    println(io, "  --leaf-cell-multiplier X --forest-internal-multiplier X")
    println(io, "  --forest-contact-multiplier X --program-multiplier X")
    println(io, "  --output-cell-multiplier X --output-anchor-multiplier X")
    println(io, "  --output-context-multiplier X --output-placement-multiplier X")
    println(io, "  --output-cascade-multiplier X --output-gain-multiplier X")
    println(io, "  --output-bias-multiplier X")
    println(io, "  --dataset PATH --checkpoint PATH --progress PATH")
    println(io, "  --resume PATH              resume only after full identity checks")
end

@inline function _argument_value(args, index::Int, token::String)
    equals = findfirst(==('='), token)
    if !isnothing(equals)
        return token[(equals + 1):end], index
    end
    index < length(args) || error("missing value after $token")
    return args[index + 1], index + 1
end

function _validated_options(options::ScratchOptions)
    active_project = Base.active_project()
    active_project !== nothing &&
        abspath(active_project) == abspath(joinpath(_PROJECT_ROOT, "Project.toml")) ||
        error("launch from the repository with --project=.")
    Threads.nthreads(:interactive) == 0 || error(
        "the interactive thread pool must be disabled; launch with --threads=N,0",
    )
    options.updates >= 1 || error("--updates must be positive")
    1 <= options.workers <= Threads.nthreads(:default) || error(
        "--workers must lie in 1:$(Threads.nthreads(:default))",
    )
    options.log_every >= 1 || error("--log-every must be positive")
    options.evaluate_every >= 1 || error("--evaluate-every must be positive")
    options.checkpoint_every >= 1 || error(
        "--checkpoint-every must be positive",
    )
    options.candidate_chunk_size >= 1 || error(
        "--candidate-chunk-size must be positive",
    )
    options.queue_capacity >= 2 && ispow2(options.queue_capacity) || error(
        "--queue-capacity must be a power of two at least two",
    )
    options.binding_mode in (:none, :all, :p_only) || error(
        "--binding-mode must be none, all, or p_only",
    )
    options.warmup_updates >= 0 || error(
        "--warmup-updates must be non-negative",
    )
    options.learning_rate_schedule_updates >= 1 || error(
        "--learning-rate-schedule-updates must be positive",
    )
    0.0f0 <= options.min_learning_rate_ratio <= 1.0f0 || error(
        "--min-learning-rate-ratio must lie in [0, 1]",
    )
    for name in (
        :learning_rate, :clip_norm, :weight_decay, :cell_weight_decay,
        :leaf_cell_multiplier, :forest_internal_multiplier,
        :forest_contact_multiplier, :program_multiplier,
        :output_cell_multiplier, :output_anchor_multiplier,
        :output_context_multiplier, :output_placement_multiplier,
        :output_cascade_multiplier, :output_gain_multiplier,
        :output_bias_multiplier,
    )
        value = getfield(options, name)
        isfinite(value) && value >= 0.0f0 || error(
            "--$(replace(String(name), '_' => '-')) must be finite and non-negative",
        )
    end
    options.learning_rate > 0.0f0 || error("--learning-rate must be positive")
    options.clip_norm > 0.0f0 || error("--clip-norm must be positive")
    abspath(options.checkpoint) != abspath(options.progress) || error(
        "checkpoint and progress paths must differ",
    )
    return options
end

function parse_options(args=ARGS)
    defaults = ScratchOptions()
    values = Dict{Symbol,Any}(
        name => getfield(defaults, name) for name in fieldnames(ScratchOptions)
    )
    specifications = Dict{String,Tuple{Symbol,Any}}(
        "--updates" => (:updates, Int),
        "--workers" => (:workers, Int),
        "--log-every" => (:log_every, Int),
        "--evaluate-every" => (:evaluate_every, Int),
        "--checkpoint-every" => (:checkpoint_every, Int),
        "--candidate-chunk-size" => (:candidate_chunk_size, Int),
        "--queue-capacity" => (:queue_capacity, Int),
        "--binding-mode" => (:binding_mode, Symbol),
        "--sampler-seed" => (:sampler_seed, UInt64),
        "--learning-rate" => (:learning_rate, Float32),
        "--warmup-updates" => (:warmup_updates, Int),
        "--learning-rate-schedule-updates" =>
            (:learning_rate_schedule_updates, Int),
        "--min-learning-rate-ratio" =>
            (:min_learning_rate_ratio, Float32),
        "--clip-norm" => (:clip_norm, Float32),
        "--weight-decay" => (:weight_decay, Float32),
        "--cell-weight-decay" => (:cell_weight_decay, Float32),
        "--leaf-cell-multiplier" => (:leaf_cell_multiplier, Float32),
        "--forest-internal-multiplier" => (:forest_internal_multiplier, Float32),
        "--forest-contact-multiplier" => (:forest_contact_multiplier, Float32),
        "--program-multiplier" => (:program_multiplier, Float32),
        "--output-cell-multiplier" => (:output_cell_multiplier, Float32),
        "--output-anchor-multiplier" => (:output_anchor_multiplier, Float32),
        "--output-context-multiplier" => (:output_context_multiplier, Float32),
        "--output-placement-multiplier" => (:output_placement_multiplier, Float32),
        "--output-cascade-multiplier" => (:output_cascade_multiplier, Float32),
        "--output-gain-multiplier" => (:output_gain_multiplier, Float32),
        "--output-bias-multiplier" => (:output_bias_multiplier, Float32),
        "--dataset" => (:dataset, String),
        "--checkpoint" => (:checkpoint, String),
        "--progress" => (:progress, String),
        "--resume" => (:resume, String),
    )
    index = 1
    while index <= length(args)
        token = String(args[index])
        token in ("-h", "--help") && (_usage(); return nothing)
        name = first(split(token, '='; limit=2))
        haskey(specifications, name) || error("unknown option $token")
        value, index = _argument_value(args, index, token)
        field, type = specifications[name]
        values[field] = type === String ? String(value) :
            type === Symbol ? Symbol(lowercase(String(value))) :
            parse(type, value)
        index += 1
    end
    options = ScratchOptions(;
        (name => values[name] for name in fieldnames(ScratchOptions))...,
    )
    return _validated_options(options)
end

function options_record(options::ScratchOptions)
    return NamedTuple{
        fieldnames(ScratchOptions)
    }(Tuple(getfield(options, name) for name in fieldnames(ScratchOptions)))
end

function semantic_config(options::ScratchOptions)
    return (
        model_family=CHECKPOINT_MAGIC,
        state_batch=STATE_BATCH,
        candidate_width=CANDIDATE_WIDTH,
        address_scheme=Bank.ADDRESS_SCHEME,
        program_rows=Bank.ROW_COUNT,
        program_payload_width=Bank.PAYLOAD_WIDTH,
        forest_nodes=Model.Topology.NODE_COUNT,
        forest_anchors=Model.Topology.ANCHOR_COUNT,
        compact_payload_width=Model.Forest.PAYLOAD_DIM,
        output_channels=Output.OUTPUT_CHANNELS,
        sampler_seed=options.sampler_seed,
        gradient_scale=1.0f0,
        base_learning_rate=options.learning_rate,
        warmup_updates=options.warmup_updates,
        learning_rate_schedule_updates=options.learning_rate_schedule_updates,
        learning_rate_decay_updates=max(
            options.learning_rate_schedule_updates - options.warmup_updates,
            1,
        ),
        min_learning_rate_ratio=options.min_learning_rate_ratio,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
        clip_norm=options.clip_norm,
        weight_decay=options.weight_decay,
        cell_weight_decay=options.cell_weight_decay,
        leaf_cell_multiplier=options.leaf_cell_multiplier,
        forest_internal_multiplier=options.forest_internal_multiplier,
        forest_contact_multiplier=options.forest_contact_multiplier,
        program_multiplier=options.program_multiplier,
        output_cell_multiplier=options.output_cell_multiplier,
        output_anchor_multiplier=options.output_anchor_multiplier,
        output_context_multiplier=options.output_context_multiplier,
        output_placement_multiplier=options.output_placement_multiplier,
        output_cascade_multiplier=options.output_cascade_multiplier,
        output_gain_multiplier=options.output_gain_multiplier,
        output_bias_multiplier=options.output_bias_multiplier,
    )
end

function _semantic_config_record(record)
    required = (
        :sampler_seed, :learning_rate, :warmup_updates,
        :learning_rate_schedule_updates, :min_learning_rate_ratio,
        :clip_norm, :weight_decay,
        :cell_weight_decay, :leaf_cell_multiplier,
        :forest_internal_multiplier, :forest_contact_multiplier,
        :program_multiplier, :output_cell_multiplier,
        :output_anchor_multiplier, :output_context_multiplier,
        :output_placement_multiplier, :output_cascade_multiplier,
        :output_gain_multiplier, :output_bias_multiplier,
    )
    _require_properties(record, required, "checkpoint options")
    return (
        model_family=CHECKPOINT_MAGIC,
        state_batch=STATE_BATCH,
        candidate_width=CANDIDATE_WIDTH,
        address_scheme=Bank.ADDRESS_SCHEME,
        program_rows=Bank.ROW_COUNT,
        program_payload_width=Bank.PAYLOAD_WIDTH,
        forest_nodes=Model.Topology.NODE_COUNT,
        forest_anchors=Model.Topology.ANCHOR_COUNT,
        compact_payload_width=Model.Forest.PAYLOAD_DIM,
        output_channels=Output.OUTPUT_CHANNELS,
        sampler_seed=record.sampler_seed,
        gradient_scale=1.0f0,
        base_learning_rate=record.learning_rate,
        warmup_updates=record.warmup_updates,
        learning_rate_schedule_updates=record.learning_rate_schedule_updates,
        learning_rate_decay_updates=max(
            record.learning_rate_schedule_updates - record.warmup_updates,
            1,
        ),
        min_learning_rate_ratio=record.min_learning_rate_ratio,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
        clip_norm=record.clip_norm,
        weight_decay=record.weight_decay,
        cell_weight_decay=record.cell_weight_decay,
        leaf_cell_multiplier=record.leaf_cell_multiplier,
        forest_internal_multiplier=record.forest_internal_multiplier,
        forest_contact_multiplier=record.forest_contact_multiplier,
        program_multiplier=record.program_multiplier,
        output_cell_multiplier=record.output_cell_multiplier,
        output_anchor_multiplier=record.output_anchor_multiplier,
        output_context_multiplier=record.output_context_multiplier,
        output_placement_multiplier=record.output_placement_multiplier,
        output_cascade_multiplier=record.output_cascade_multiplier,
        output_gain_multiplier=record.output_gain_multiplier,
        output_bias_multiplier=record.output_bias_multiplier,
    )
end

function source_fingerprint(paths=_SOURCE_PATHS)
    buffer = IOBuffer()
    for source_path in paths
        path = abspath(source_path)
        isfile(path) || error("source fingerprint input is missing: $path")
        label = replace(relpath(path, @__DIR__), '\\' => '/')
        payload = read(path)
        write(buffer, codeunits(label))
        write(buffer, UInt8(0))
        write(buffer, codeunits(string(length(payload))))
        write(buffer, UInt8(0))
        write(buffer, payload)
        write(buffer, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(buffer)))
end

@inline file_sha256(path::AbstractString) =
    bytes2hex(SHA.sha256(read(abspath(path))))

function rows_sha256(rows::AbstractVector{<:Integer})
    materialized = Int.(rows)
    return bytes2hex(SHA.sha256(reinterpret(UInt8, materialized)))
end

"""Load and validate the complete width-80 teacher_v3 tensor contract."""
function load_width80_dataset(path::AbstractString)
    source = Core.load_teacher_dataset(
        abspath(path);
        max_candidates=Core.MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    maximum(source.action_counts) <= CANDIDATE_WIDTH || error(
        "teacher dataset contains more than $CANDIDATE_WIDTH legal candidates",
    )
    source.part_integrity_verified || error(
        "teacher dataset part-integrity verification did not complete",
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
    manifest = file_sha256(source.manifest_path)
    return source, dataset, manifest
end

function fixed_validation_rows(
    source;
    count::Integer=VALIDATION_STATES,
    seed::UInt64=VALIDATION_SUBSET_SEED,
)
    rows = Int.(findall(==(:validation), source.predefined_split))
    length(rows) >= count || error(
        "predefined validation split has fewer than $count rows",
    )
    shuffle!(Xoshiro(seed), rows)
    resize!(rows, Int(count))
    return rows
end

mutable struct RandomTrainingSampler
    rows::Vector{Int}
    seed::UInt64
    rng::Xoshiro
    draws::UInt64
end

function RandomTrainingSampler(rows::AbstractVector{<:Integer}, seed::UInt64)
    materialized = Int.(rows)
    isempty(materialized) && error("training rows are empty")
    length(unique(materialized)) == length(materialized) || error(
        "training rows contain duplicates",
    )
    return RandomTrainingSampler(materialized, seed, Xoshiro(seed), UInt64(0))
end

function next_batch!(destination::AbstractVector{Int}, sampler::RandomTrainingSampler)
    @inbounds for slot in eachindex(destination)
        destination[slot] = sampler.rows[rand(sampler.rng, 1:length(sampler.rows))]
    end
    sampler.draws = Base.checked_add(sampler.draws, UInt64(length(destination)))
    return destination
end

@inline function _stable_argmax(values, count::Int)
    best = 1
    @inbounds for candidate in 2:count
        values[candidate] > values[best] && (best = candidate)
    end
    return best
end

function _ndcg(prediction, teacher)
    count = length(teacher)
    teacher_order = sortperm(teacher; rev=true, alg=MergeSort)
    relevance = zeros(Float64, count)
    @inbounds for (rank, candidate) in enumerate(teacher_order)
        relevance[candidate] = count - rank
    end
    prediction_order = sortperm(prediction; rev=true, alg=MergeSort)
    dcg = 0.0
    idcg = 0.0
    @inbounds for rank in 1:count
        discount = inv(log2(rank + 1.0))
        dcg += relevance[prediction_order[rank]] * discount
        idcg += (count - rank) * discount
    end
    return iszero(idcg) ? 1.0 : dcg / idcg
end

function _pairwise_accuracy(prediction, teacher)
    correct = 0
    compared = 0
    @inbounds for left in 1:(length(teacher) - 1)
        for right in (left + 1):length(teacher)
            teacher_difference = teacher[left] - teacher[right]
            iszero(teacher_difference) && continue
            correct +=
                (prediction[left] - prediction[right]) * teacher_difference > 0
            compared += 1
        end
    end
    return iszero(compared) ? 1.0 : correct / compared
end

"""Per-state ranking metrics plus canonical forest/output execution telemetry."""
function batch_metrics(
    batch::Ranking.Batch,
    loss::Ranking.SupervisedLoss,
    diagnostics::Parallel.ForwardDiagnostics,
)
    legacy_correct = 0
    tie_correct = 0
    ndcg_sum = 0.0
    pairwise_sum = 0.0
    @inbounds for state in 1:batch.state_batch
        count = Int(batch.counts[state])
        offset = (state - 1) * batch.width
        prediction = @view batch.raw[1, (offset + 1):(offset + count)]
        teacher = @view batch.targets.teacher_q[1:count, state]
        predicted = _stable_argmax(prediction, count)
        teacher_best = _stable_argmax(teacher, count)
        legacy_correct += predicted == teacher_best
        tie_correct += teacher[predicted] >= teacher[teacher_best] - 1.0f-6
        ndcg_sum += _ndcg(prediction, teacher)
        pairwise_sum += _pairwise_accuracy(prediction, teacher)
    end
    state_denominator = Float64(batch.state_batch)
    forest_events = diagnostics.leaf_events + diagnostics.internal_events +
                    diagnostics.root_events
    output_denominator = batch.valid_count * Output.hard_event_denominator()
    return (
        states=batch.state_batch,
        candidates=batch.valid_count,
        composite=Float64(loss.composite_loss),
        excess=Float64(loss.composite_loss - loss.teacher_entropy),
        listnet_kl=Float64(loss.listnet_kl),
        legacy_top1=legacy_correct / state_denominator,
        tie_top1=tie_correct / state_denominator,
        ndcg=ndcg_sum / state_denominator,
        pairwise=pairwise_sum / state_denominator,
        leaf_events=diagnostics.leaf_events,
        internal_events=diagnostics.internal_events,
        root_events=diagnostics.root_events,
        output_events=diagnostics.output_events,
        evaluated_nodes=diagnostics.evaluated_nodes,
        dirty_leaves=diagnostics.dirty_leaves,
        dirty_ancestors=diagnostics.dirty_ancestors,
        compact_messages=diagnostics.compact_messages,
        forest_event_rate=iszero(diagnostics.evaluated_nodes) ? 0.0 :
            forest_events / diagnostics.evaluated_nodes,
        output_event_rate=iszero(output_denominator) ? 0.0 :
            diagnostics.output_events / output_denominator,
    )
end

"""
Measure the canonical DDF execution regime before the first optimizer update.

This cold path traverses the same crossed forest, candidate COW closure, and
22-cell output cascade as training.  It exists to fail closed when the forest
is not evaluated or emits an impossible event rate; it does not resurrect any
retired factor/readout diagnostic.
"""
function preflight_diagnostics(parameters, dataset, rows)
    length(rows) >= STATE_BATCH || error(
        "preflight requires at least $STATE_BATCH development rows",
    )
    cache = Model.ModelCache(parameters)
    state = Model.ModelState()
    worker = Model.ModelWorker()
    raw = zeros(Float32, Ranking.OUTPUT_DIM)
    leaf_events = Int64(0)
    internal_events = Int64(0)
    root_events = Int64(0)
    output_events = Int64(0)
    evaluated_nodes = Int64(0)
    dirty_leaves = Int64(0)
    dirty_ancestors = Int64(0)
    compact_messages = Int64(0)
    output_denominator = Int64(0)
    candidates = 0
    @inbounds for row in @view(rows[1:STATE_BATCH])
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        for stats in (state.before.stats, state.after_base.stats)
            leaf_events += stats.leaf_hard_events
            internal_events += stats.internal_hard_events
            root_events += stats.root_hard_events
            evaluated_nodes += stats.dirty_leaves + stats.dirty_ancestors
            compact_messages += stats.compact_messages
        end
        count = Int(dataset.action_counts[row])
        for candidate in 1:count
            Model.forward_candidate!(
                raw, worker, state, parameters, cache, dataset, row, candidate,
            )
            stats = worker.candidate_after.stats
            leaf_events += stats.leaf_hard_events
            internal_events += stats.internal_hard_events
            root_events += stats.root_hard_events
            evaluated_nodes += stats.dirty_leaves + stats.dirty_ancestors
            dirty_leaves += stats.dirty_leaves
            dirty_ancestors += stats.dirty_ancestors
            compact_messages += stats.compact_messages
            output_events += Output.hard_event_count(worker.output_tape)
            output_denominator += Output.hard_event_denominator()
            candidates += 1
        end
    end
    forest_events = leaf_events + internal_events + root_events
    return (
        states=STATE_BATCH,
        candidates,
        leaf_events,
        internal_events,
        root_events,
        output_events,
        evaluated_nodes,
        dirty_leaves,
        dirty_ancestors,
        compact_messages,
        forest_event_rate=iszero(evaluated_nodes) ? 0.0 :
            forest_events / evaluated_nodes,
        output_event_rate=iszero(output_denominator) ? 0.0 :
            output_events / output_denominator,
    )
end

function _assert_preflight_gate(diagnostics)
    diagnostics.candidates > 0 || error("preflight evaluated no candidates")
    diagnostics.evaluated_nodes > 0 || error("preflight evaluated no forest nodes")
    diagnostics.compact_messages > 0 || error("preflight delivered no compact messages")
    0.0 < diagnostics.forest_event_rate < 1.0 || error(
        "forest has a degenerate $(100diagnostics.forest_event_rate)% event rate; " *
        "refusing scratch training",
    )
    0.0 <= diagnostics.output_event_rate <= 1.0 || error(
        "output hard-event rate is outside [0, 1]",
    )
    return diagnostics
end

function _optimizer_config(options::ScratchOptions)
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
        learning_rate=options.learning_rate,
        clip_norm=options.clip_norm,
        weight_decay=options.weight_decay,
        cell_weight_decay=options.cell_weight_decay,
        multipliers=multipliers,
    )
end

@inline function _learning_rate_schedule(options::ScratchOptions)
    decay_updates = max(
        options.learning_rate_schedule_updates - options.warmup_updates,
        1,
    )
    return Optimizer.LearningRateSchedule(
        options.learning_rate,
        options.warmup_updates,
        decay_updates,
        options.min_learning_rate_ratio,
    )
end

@inline function _with_learning_rate(
    config::Optimizer.AdamWConfig,
    learning_rate::Float32,
)
    return Optimizer.AdamWConfig(
        learning_rate=learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        clip_norm=config.clip_norm,
        weight_decay=config.weight_decay,
        cell_weight_decay=config.cell_weight_decay,
        multipliers=config.multipliers,
    )
end

@inline function _dense_moments_snapshot(moments)
    return (
        leaf_shared_raw=copy(moments.leaf_shared_raw),
        forest_internal_raw=copy(moments.forest_internal_raw),
        forest_child_contact=copy(moments.forest_child_contact),
        output_cell_raw=copy(moments.output_cell_raw),
        output_anchor_weight=copy(moments.output_anchor_weight),
        output_context_weight=copy(moments.output_context_weight),
        output_placement_weight=copy(moments.output_placement_weight),
        output_cascade_weight=copy(moments.output_cascade_weight),
        output_gain=copy(moments.output_gain),
        output_bias=copy(moments.output_bias),
    )
end

function _model_snapshot(parameters::Model.ModelParameters)
    return (
        leaf_shared_raw=copy(parameters.leaf_shared_raw),
        program_payload=copy(parameters.program_bank.payload),
        forest_internal_raw=copy(parameters.forest.internal_raw),
        forest_child_contact=copy(parameters.forest.child_contact),
        output_cell_raw=copy(parameters.output.cell_raw),
        output_anchor_weight=copy(parameters.output.anchor_weight),
        output_context_weight=copy(parameters.output.context_weight),
        output_placement_weight=copy(parameters.output.placement_weight),
        output_cascade_weight=copy(parameters.output.cascade_weight),
        output_gain=copy(parameters.output.gain),
        output_bias=copy(parameters.output.bias),
    )
end

function _optimizer_snapshot(state::Optimizer.AdamWState)
    step_names = fieldnames(Optimizer.AdamWStepCounters)
    steps = NamedTuple{step_names}(
        Tuple(getfield(state.steps, name) for name in step_names),
    )
    return (
        first=_dense_moments_snapshot(state.first),
        second=_dense_moments_snapshot(state.second),
        program_first=copy(state.program_first),
        program_second=copy(state.program_second),
        program_step_by_row=copy(state.program_step_by_row),
        placement_step_by_coordinate=copy(state.placement_step_by_coordinate),
        steps,
    )
end

@inline _sampler_snapshot(sampler::RandomTrainingSampler) = (
    rows=copy(sampler.rows),
    seed=sampler.seed,
    draws=sampler.draws,
)

function build_checkpoint(
    update::Integer,
    consumed_states::Integer,
    consumed_candidates::Integer,
    training_nanoseconds::UInt128,
    parameters::Model.ModelParameters,
    optimizer::Optimizer.AdamWState,
    sampler::RandomTrainingSampler,
    options::ScratchOptions,
    model_source_fingerprint::AbstractString,
    dataset_manifest_sha256::AbstractString,
    training_rows::AbstractVector{<:Integer},
    validation_rows::AbstractVector{<:Integer},
)
    return (
        magic=CHECKPOINT_MAGIC,
        schema=CHECKPOINT_SCHEMA,
        julia_version=string(VERSION),
        update=Int(update),
        consumed_states=Int(consumed_states),
        consumed_candidates=Int(consumed_candidates),
        training_nanoseconds,
        options=options_record(options),
        semantic_config=semantic_config(options),
        source_fingerprint=String(model_source_fingerprint),
        dataset_manifest_sha256=String(dataset_manifest_sha256),
        training_rows_sha256=rows_sha256(training_rows),
        validation_rows=Int.(validation_rows),
        validation_rows_sha256=rows_sha256(validation_rows),
        parameters=_model_snapshot(parameters),
        optimizer=_optimizer_snapshot(optimizer),
        sampler=_sampler_snapshot(sampler),
    )
end

function save_checkpoint_atomic(path::AbstractString, snapshot)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid()) * "." * string(time_ns())
    try
        open(temporary, "w") do io
            serialize(io, snapshot)
            flush(io)
        end
        # libuv rename uses replace-existing semantics on Windows and remains
        # within one directory/volume, so readers see either complete image.
        Base.Filesystem.rename(temporary, destination)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

function load_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("checkpoint does not exist: $source")
    snapshot = try
        open(deserialize, source)
    catch exception
        error("invalid or truncated DDF checkpoint $source: $(sprint(showerror, exception))")
    end
    snapshot isa NamedTuple || error(
        "checkpoint is not a plain DDF snapshot",
    )
    hasproperty(snapshot, :magic) && snapshot.magic == CHECKPOINT_MAGIC || error(
        "checkpoint belongs to another model family",
    )
    hasproperty(snapshot, :schema) && snapshot.schema == CHECKPOINT_SCHEMA || error(
        "unsupported DDF checkpoint schema",
    )
    return snapshot
end

@inline function _require_properties(value, names, label)
    for name in names
        hasproperty(value, name) || error("$label is missing `$name`")
    end
    return value
end

@inline function _copy_float_array!(destination, source, label)
    source isa AbstractArray{Float32} || error("$label must be Float32")
    size(source) == size(destination) || error(
        "$label shape $(size(source)); expected $(size(destination))",
    )
    all(isfinite, source) || error("$label contains a non-finite value")
    copyto!(destination, source)
    return destination
end

function _restore_model(snapshot)
    names = (
        :leaf_shared_raw, :program_payload, :forest_internal_raw,
        :forest_child_contact, :output_cell_raw, :output_anchor_weight,
        :output_context_weight, :output_placement_weight,
        :output_cascade_weight, :output_gain, :output_bias,
    )
    _require_properties(snapshot, names, "parameter snapshot")
    parameters = Model.initialize_model()
    _copy_float_array!(parameters.leaf_shared_raw, snapshot.leaf_shared_raw,
                       "leaf_shared_raw")
    _copy_float_array!(parameters.program_bank.payload, snapshot.program_payload,
                       "program_payload")
    _copy_float_array!(parameters.forest.internal_raw,
                       snapshot.forest_internal_raw, "forest_internal_raw")
    _copy_float_array!(parameters.forest.child_contact,
                       snapshot.forest_child_contact, "forest_child_contact")
    _copy_float_array!(parameters.output.cell_raw,
                       snapshot.output_cell_raw, "output_cell_raw")
    _copy_float_array!(parameters.output.anchor_weight,
                       snapshot.output_anchor_weight, "output_anchor_weight")
    _copy_float_array!(parameters.output.context_weight,
                       snapshot.output_context_weight, "output_context_weight")
    _copy_float_array!(parameters.output.placement_weight,
                       snapshot.output_placement_weight, "output_placement_weight")
    _copy_float_array!(parameters.output.cascade_weight,
                       snapshot.output_cascade_weight, "output_cascade_weight")
    _copy_float_array!(parameters.output.gain,
                       snapshot.output_gain, "output_gain")
    _copy_float_array!(parameters.output.bias,
                       snapshot.output_bias, "output_bias")
    return parameters
end

function _restore_dense_moments!(destination, snapshot, label)
    for name in fieldnames(Optimizer.DenseMoments)
        hasproperty(snapshot, name) || error("$label is missing `$name`")
        _copy_float_array!(
            getfield(destination, name), getproperty(snapshot, name),
            "$label.$name",
        )
    end
    return destination
end

function _restore_optimizer(snapshot, parameters, update::Int)
    _require_properties(
        snapshot,
        (:first, :second, :program_first, :program_second,
         :program_step_by_row, :placement_step_by_coordinate, :steps),
        "optimizer snapshot",
    )
    state = Optimizer.AdamWState(parameters)
    _restore_dense_moments!(state.first, snapshot.first, "optimizer.first")
    _restore_dense_moments!(state.second, snapshot.second, "optimizer.second")
    _copy_float_array!(state.program_first, snapshot.program_first,
                       "optimizer.program_first")
    _copy_float_array!(state.program_second, snapshot.program_second,
                       "optimizer.program_second")
    for name in fieldnames(Optimizer.DenseMoments)
        array = getfield(state.second, name)
        all(value -> value >= 0.0f0, array) || error(
            "optimizer second moments `$name` contain a negative value",
        )
    end
    all(value -> value >= 0.0f0, state.program_second) || error(
        "optimizer program second moments contain a negative value",
    )
    source_steps = snapshot.program_step_by_row
    source_steps isa AbstractVector{UInt32} || error(
        "optimizer.program_step_by_row must be UInt32",
    )
    length(source_steps) == length(state.program_step_by_row) || error(
        "optimizer.program_step_by_row length mismatch",
    )
    copyto!(state.program_step_by_row, source_steps)
    source_placement_steps = snapshot.placement_step_by_coordinate
    source_placement_steps isa AbstractMatrix{UInt32} || error(
        "optimizer.placement_step_by_coordinate must be UInt32",
    )
    size(source_placement_steps) == size(state.placement_step_by_coordinate) ||
        error("optimizer.placement_step_by_coordinate shape mismatch")
    copyto!(state.placement_step_by_coordinate, source_placement_steps)
    for name in fieldnames(Optimizer.AdamWStepCounters)
        hasproperty(snapshot.steps, name) || error(
            "optimizer steps are missing `$name`",
        )
        value = Int(getproperty(snapshot.steps, name))
        0 <= value || error("optimizer step `$name` is negative")
        setfield!(state.steps, name, value)
    end
    state.steps.total == update || error(
        "optimizer total $(state.steps.total) differs from update $update",
    )
    sum(UInt128, state.program_step_by_row) == UInt128(state.steps.program_rows) ||
        error("program row clocks do not sum to program_rows")
    sum(UInt128, state.placement_step_by_coordinate) ==
        UInt128(state.steps.output_placement_coordinates) || error(
            "placement coordinate clocks do not sum to output_placement_coordinates",
        )
    return state
end

function _restore_sampler(snapshot, expected_rows, expected_seed::UInt64)
    _require_properties(snapshot, (:rows, :seed, :draws), "sampler snapshot")
    rows = Int.(snapshot.rows)
    rows == Int.(expected_rows) || error("sampler training rows changed")
    UInt64(snapshot.seed) == expected_seed || error("sampler seed changed")
    draws = UInt64(snapshot.draws)
    sampler = RandomTrainingSampler(rows, expected_seed)
    draws <= UInt64(typemax(Int)) || error("sampler draw count is too large")
    @inbounds for _ in 1:Int(draws)
        rand(sampler.rng, 1:length(rows))
    end
    sampler.draws = draws
    return sampler
end

function validate_checkpoint!(
    snapshot,
    options::ScratchOptions,
    model_source_fingerprint::AbstractString,
    dataset_manifest_sha256::AbstractString,
    training_rows::AbstractVector{<:Integer},
    validation_rows::AbstractVector{<:Integer},
)
    _require_properties(
        snapshot,
        (:magic, :schema, :julia_version, :update, :consumed_states, :consumed_candidates,
         :training_nanoseconds, :options, :semantic_config,
         :source_fingerprint, :dataset_manifest_sha256,
         :training_rows_sha256, :validation_rows,
         :validation_rows_sha256, :parameters, :optimizer, :sampler),
        "checkpoint",
    )
    snapshot.magic == CHECKPOINT_MAGIC || error(
        "checkpoint belongs to another model family",
    )
    snapshot.schema == CHECKPOINT_SCHEMA || error(
        "unsupported DDF checkpoint schema",
    )
    snapshot.julia_version == string(VERSION) || error(
        "checkpoint Julia version changed",
    )
    snapshot.semantic_config == _semantic_config_record(snapshot.options) ||
        error("checkpoint options disagree with its semantic config")
    snapshot.semantic_config == semantic_config(options) || error(
        "checkpoint semantic training options changed",
    )
    snapshot.source_fingerprint == model_source_fingerprint || error(
        "checkpoint source fingerprint changed",
    )
    snapshot.dataset_manifest_sha256 == dataset_manifest_sha256 || error(
        "checkpoint dataset manifest changed",
    )
    snapshot.training_rows_sha256 == rows_sha256(training_rows) || error(
        "checkpoint training row set changed",
    )
    snapshot.validation_rows == Int.(validation_rows) || error(
        "checkpoint validation panel changed",
    )
    snapshot.validation_rows_sha256 == rows_sha256(validation_rows) || error(
        "checkpoint validation row hash changed",
    )
    update = Int(snapshot.update)
    update >= 0 || error("checkpoint update is negative")
    update <= options.updates || error(
        "checkpoint update exceeds requested target",
    )
    Int(snapshot.consumed_states) == update * STATE_BATCH || error(
        "checkpoint consumed-state count is inconsistent",
    )
    UInt64(snapshot.sampler.draws) == UInt64(update * STATE_BATCH) || error(
        "checkpoint sampler draw count is inconsistent",
    )
    Int(snapshot.consumed_candidates) >= Int(snapshot.consumed_states) || error(
        "checkpoint consumed-candidate count is inconsistent",
    )
    return nothing
end

function restore_checkpoint(
    snapshot,
    options::ScratchOptions,
    model_source_fingerprint::AbstractString,
    dataset_manifest_sha256::AbstractString,
    training_rows::AbstractVector{<:Integer},
    validation_rows::AbstractVector{<:Integer},
)
    validate_checkpoint!(
        snapshot, options, model_source_fingerprint,
        dataset_manifest_sha256, training_rows, validation_rows,
    )
    update = Int(snapshot.update)
    parameters = _restore_model(snapshot.parameters)
    optimizer = _restore_optimizer(snapshot.optimizer, parameters, update)
    sampler = _restore_sampler(
        snapshot.sampler, training_rows, options.sampler_seed,
    )
    return (;
        parameters,
        optimizer,
        sampler,
        update,
        consumed_states=Int(snapshot.consumed_states),
        consumed_candidates=Int(snapshot.consumed_candidates),
        training_nanoseconds=UInt128(snapshot.training_nanoseconds),
    )
end

function append_json!(path::AbstractString, record)
    destination = abspath(path)
    mkpath(dirname(destination))
    open(destination, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
    return destination
end

@inline timestamp_utc() = string(Dates.now(Dates.UTC)) * "Z"

function _training_record(
    update, consumed_states, consumed_candidates, metrics, stats,
    learning_rate, window_updates, window_states, window_nanoseconds,
)
    seconds = max(Float64(window_nanoseconds) * 1.0e-9, eps(Float64))
    return (
        kind="training",
        timestamp_utc=timestamp_utc(),
        update,
        consumed_states,
        consumed_candidates,
        composite=metrics.composite,
        excess=metrics.excess,
        listnet_kl=metrics.listnet_kl,
        legacy_top1=metrics.legacy_top1,
        tie_top1=metrics.tie_top1,
        ndcg=metrics.ndcg,
        pairwise=metrics.pairwise,
        forest_event_rate=metrics.forest_event_rate,
        output_event_rate=metrics.output_event_rate,
        evaluated_nodes=metrics.evaluated_nodes,
        dirty_leaves=metrics.dirty_leaves,
        dirty_ancestors=metrics.dirty_ancestors,
        compact_messages=metrics.compact_messages,
        learning_rate=Float64(learning_rate),
        training_only_updates_per_s=window_updates / seconds,
        training_only_states_per_s=window_states / seconds,
        active_program_rows=stats.active_program_rows,
        gradient_norm=stats.gradient_norm,
        clip_scale=Float64(stats.clip_scale),
    )
end

function _print_record(record)
    @printf(
        "candidate_delta_%s update=%d composite=%.6f excess=%.6f listnet_kl=%.6f legacy_top1=%.6f tie_top1=%.6f ndcg=%.6f pairwise=%.6f forest_event_rate=%.8f output_event_rate=%.8f learning_rate=%.9g updates_per_s=%.3f states_per_s=%.3f active_rows=%d grad_norm=%.6f\n",
        record.kind,
        record.update,
        record.composite,
        record.excess,
        record.listnet_kl,
        record.legacy_top1,
        record.tie_top1,
        record.ndcg,
        record.pairwise,
        record.forest_event_rate,
        record.output_event_rate,
        get(record, :learning_rate, 0.0),
        get(record, :training_only_updates_per_s, 0.0),
        get(record, :training_only_states_per_s, 0.0),
        get(record, :active_program_rows, 0),
        get(record, :gradient_norm, 0.0),
    )
    flush(stdout)
    return nothing
end

function _evaluate_validation!(session, validation_rows)
    executor = session.executor
    batch = executor.batch
    length(validation_rows) == VALIDATION_STATES || error(
        "validation panel must contain $VALIDATION_STATES states",
    )
    total_states = 0
    total_candidates = 0
    composite = 0.0
    excess = 0.0
    listnet_kl = 0.0
    legacy = 0.0
    tied = 0.0
    ndcg = 0.0
    pairwise = 0.0
    leaf_events = Int64(0)
    internal_events = Int64(0)
    root_events = Int64(0)
    output_events = Int64(0)
    evaluated_nodes = Int64(0)
    dirty_leaves = Int64(0)
    dirty_ancestors = Int64(0)
    compact_messages = Int64(0)
    for first in 1:STATE_BATCH:length(validation_rows)
        copyto!(batch.rows, @view(validation_rows[first:(first + STATE_BATCH - 1)]))
        Parallel.forward_batch!(session)
        loss = Parallel.loss_and_raw_gradient!(session)
        metrics = batch_metrics(
            batch,
            loss,
            Parallel.latest_forward_diagnostics(session),
        )
        weight = metrics.states
        total_states += weight
        total_candidates += metrics.candidates
        composite += metrics.composite * weight
        excess += metrics.excess * weight
        listnet_kl += metrics.listnet_kl * weight
        legacy += metrics.legacy_top1 * weight
        tied += metrics.tie_top1 * weight
        ndcg += metrics.ndcg * weight
        pairwise += metrics.pairwise * weight
        leaf_events += metrics.leaf_events
        internal_events += metrics.internal_events
        root_events += metrics.root_events
        output_events += metrics.output_events
        evaluated_nodes += metrics.evaluated_nodes
        dirty_leaves += metrics.dirty_leaves
        dirty_ancestors += metrics.dirty_ancestors
        compact_messages += metrics.compact_messages
    end
    forest_events = leaf_events + internal_events + root_events
    output_denominator = total_candidates * Output.hard_event_denominator()
    return (
        kind="validation",
        timestamp_utc=timestamp_utc(),
        states=total_states,
        candidates=total_candidates,
        composite=composite / total_states,
        excess=excess / total_states,
        listnet_kl=listnet_kl / total_states,
        legacy_top1=legacy / total_states,
        tie_top1=tied / total_states,
        ndcg=ndcg / total_states,
        pairwise=pairwise / total_states,
        forest_event_rate=iszero(evaluated_nodes) ? 0.0 :
            forest_events / evaluated_nodes,
        output_event_rate=iszero(output_denominator) ? 0.0 :
            output_events / output_denominator,
        evaluated_nodes,
        dirty_leaves,
        dirty_ancestors,
        compact_messages,
    )
end

function run_training(options::ScratchOptions)
    _validated_options(options)
    source, dataset, manifest_sha256 = load_width80_dataset(options.dataset)
    manifest_sha256 == EXPECTED_MANIFEST_SHA256 || error(
        "teacher_v3 manifest changed: $manifest_sha256",
    )
    training_rows = Int.(findall(==(:train), source.predefined_split))
    isempty(training_rows) && error("predefined teacher split has no training rows")
    validation_rows = fixed_validation_rows(source)
    validation_hash = rows_sha256(validation_rows)
    validation_hash == EXPECTED_VALIDATION_ROWS_SHA256 || error(
        "shared validation panel changed: $validation_hash",
    )
    fingerprint = source_fingerprint()

    parameters = Model.initialize_model()
    optimizer = Optimizer.AdamWState(parameters)
    sampler = RandomTrainingSampler(training_rows, options.sampler_seed)
    update = 0
    consumed_states = 0
    consumed_candidates = 0
    training_nanoseconds = UInt128(0)
    if options.resume !== nothing
        restored = restore_checkpoint(
            load_checkpoint(options.resume), options, fingerprint,
            manifest_sha256, training_rows, validation_rows,
        )
        parameters = restored.parameters
        optimizer = restored.optimizer
        sampler = restored.sampler
        update = restored.update
        consumed_states = restored.consumed_states
        consumed_candidates = restored.consumed_candidates
        training_nanoseconds = restored.training_nanoseconds
        update < options.updates || error("checkpoint already reached target")
    end

    diagnostics = preflight_diagnostics(
        parameters,
        dataset,
        validation_rows,
    )
    diagnostic_record = merge(
        (;
            kind="preflight_diagnostics",
            timestamp_utc=timestamp_utc(),
            update,
        ),
        diagnostics,
    )
    append_json!(options.progress, diagnostic_record)
    println(
        "candidate_delta_preflight update=", update,
        " forest_event_rate=", diagnostics.forest_event_rate,
        " output_event_rate=", diagnostics.output_event_rate,
        " evaluated_nodes=", diagnostics.evaluated_nodes,
        " dirty_leaves=", diagnostics.dirty_leaves,
        " dirty_ancestors=", diagnostics.dirty_ancestors,
        " compact_messages=", diagnostics.compact_messages,
    )
    diagnostics.output_event_rate in (0.0, 1.0) && println(
        "candidate_delta_preflight_warning output hard events are degenerate; " *
        "continuous Reduced Hay state credit remains active",
    )
    flush(stdout)
    _assert_preflight_gate(diagnostics)

    base_config = _optimizer_config(options)
    learning_rate_schedule = _learning_rate_schedule(options)
    batch = Ranking.Batch(STATE_BATCH, CANDIDATE_WIDTH)
    executor = Parallel.BarrierlessExactExecutor(
        parameters,
        batch,
        dataset;
        worker_capacity=options.workers,
        candidate_chunk_size=options.candidate_chunk_size,
    )
    start_record = (
        kind=options.resume === nothing ? "start" : "resume",
        timestamp_utc=timestamp_utc(),
        update,
        target_updates=options.updates,
        state_batch=STATE_BATCH,
        candidate_width=CANDIDATE_WIDTH,
        workers=options.workers,
        address_scheme=Bank.ADDRESS_SCHEME,
        program_rows=Bank.ROW_COUNT,
        stored_parameters=Model.stored_parameter_count(parameters),
        train_rows=length(training_rows),
        training_rows_sha256=rows_sha256(training_rows),
        validation_rows_sha256=validation_hash,
        dataset_manifest_sha256=manifest_sha256,
        source_fingerprint=fingerprint,
        semantic_config=semantic_config(options),
        learning_rate_schedule=(;
            base_learning_rate=learning_rate_schedule.base_learning_rate,
            warmup_updates=learning_rate_schedule.warmup_updates,
            decay_updates=learning_rate_schedule.decay_updates,
            total_updates=options.learning_rate_schedule_updates,
            min_ratio=learning_rate_schedule.min_ratio,
        ),
    )
    append_json!(options.progress, start_record)
    println(
        "candidate_delta_start update=", update,
        " target_updates=", options.updates,
        " workers=", options.workers,
        " state_batch=", STATE_BATCH,
        " width=", CANDIDATE_WIDTH,
        " address_scheme=", Bank.ADDRESS_SCHEME,
        " program_rows=", Bank.ROW_COUNT,
        " parameters=", Model.stored_parameter_count(parameters),
        " base_learning_rate=", learning_rate_schedule.base_learning_rate,
        " warmup_updates=", learning_rate_schedule.warmup_updates,
        " schedule_updates=", options.learning_rate_schedule_updates,
        " decay_updates=", learning_rate_schedule.decay_updates,
        " min_learning_rate_ratio=", learning_rate_schedule.min_ratio,
        " train_rows=", length(training_rows),
        " validation_rows_sha256=", validation_hash,
        " source_fingerprint=", fingerprint,
    )
    flush(stdout)

    last_stats = Optimizer.AdamWStepStats(0.0, 1.0f0, 0)
    last_training_record_update = -1
    last_validation_update = -1
    window_updates = 0
    window_states = 0
    window_nanoseconds = UInt128(0)
    Parallel.run_executor_team!(
        executor;
        workers=options.workers,
        queue_capacity=options.queue_capacity,
        binding_mode=options.binding_mode,
    ) do session
        while update < options.updates
            next_update = update + 1
            learning_rate = Optimizer.learning_rate_at(
                learning_rate_schedule,
                next_update,
            )
            step_config = _with_learning_rate(base_config, learning_rate)
            next_batch!(batch.rows, sampler)
            started_ns = time_ns()
            Parallel.forward_loss_backward!(session, executor.loss_sink)
            loss = Parallel.latest_loss(session)
            metrics = batch_metrics(
                batch, loss, Parallel.latest_forward_diagnostics(session),
            )
            last_stats = Optimizer.apply_adamw!(
                optimizer,
                parameters,
                executor.gradient,
                step_config,
            )
            elapsed_ns = UInt128(time_ns() - started_ns)
            update += 1
            consumed_states += STATE_BATCH
            consumed_candidates += batch.valid_count
            training_nanoseconds += elapsed_ns
            window_updates += 1
            window_states += STATE_BATCH
            window_nanoseconds += elapsed_ns

            if update % options.log_every == 0 || update == options.updates
                record = _training_record(
                    update, consumed_states, consumed_candidates, metrics,
                    last_stats, learning_rate,
                    window_updates, window_states, window_nanoseconds,
                )
                append_json!(options.progress, record)
                _print_record(record)
                last_training_record_update = update
                window_updates = 0
                window_states = 0
                window_nanoseconds = UInt128(0)
            end

            if update % options.evaluate_every == 0 || update == options.updates
                validation = _evaluate_validation!(session, validation_rows)
                cumulative_seconds = max(
                    Float64(training_nanoseconds) * 1.0e-9,
                    eps(Float64),
                )
                validation = merge(validation, (;
                    update,
                    training_only_updates_per_s=update / cumulative_seconds,
                    training_only_states_per_s=consumed_states / cumulative_seconds,
                    active_program_rows=last_stats.active_program_rows,
                    gradient_norm=last_stats.gradient_norm,
                    clip_scale=Float64(last_stats.clip_scale),
                    learning_rate=Float64(learning_rate),
                ))
                append_json!(options.progress, validation)
                _print_record(validation)
                last_validation_update = update
            end

            if update % options.checkpoint_every == 0 || update == options.updates
                snapshot = build_checkpoint(
                    update, consumed_states, consumed_candidates,
                    training_nanoseconds, parameters, optimizer, sampler,
                    options, fingerprint, manifest_sha256,
                    training_rows, validation_rows,
                )
                save_checkpoint_atomic(options.checkpoint, snapshot)
            end
        end
    end
    last_training_record_update == update || error(
        "final training metrics were not recorded",
    )
    last_validation_update == update || error(
        "final validation metrics were not recorded",
    )
    println(
        "candidate_delta_complete update=", update,
        " consumed_states=", consumed_states,
        " consumed_candidates=", consumed_candidates,
        " checkpoint=", abspath(options.checkpoint),
        " progress=", abspath(options.progress),
    )
    return nothing
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) || run_training(options)
    return nothing
end

end # module CandidateDeltaDendriticScratchTraining

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    CandidateDeltaDendriticScratchTraining.main()
