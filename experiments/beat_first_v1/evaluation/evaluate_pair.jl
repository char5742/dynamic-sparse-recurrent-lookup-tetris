ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

using Dates
using JSON3
using PythonCall
using Random
using Statistics

const EVALUATION_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(EVALUATION_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))

const DEV_SEEDS = collect(5756:5757)
const VALIDATION_SEEDS = collect(8001:8008)
const SEALED_SEEDS = collect(91001:91032)
const EPISODE_PIECES = 250
const NEXT_COUNT = 5
const OLD_BATCH_SIZE = 16
const BOOTSTRAP_REPLICATES = 10_000
const BOOTSTRAP_SEED = UInt64(0xb347_f1a5_2026_0718)

function parse_cli(args)
    options = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--promoted", "--authorize-sealed")
            push!(flags, arg)
            index += 1
        elseif arg in ("--stage", "--adapter", "--checkpoint", "--output")
            index == length(args) && error("missing value after $arg")
            haskey(options, arg) && error("duplicate option $arg")
            options[arg] = args[index + 1]
            index += 2
        else
            error("unknown argument: $arg")
        end
    end
    for required in ("--stage", "--adapter", "--checkpoint", "--output")
        haskey(options, required) || error("missing required option $required")
    end
    return (; options, flags)
end

const PAIR_EVALUATOR_EXECUTED = abspath(PROGRAM_FILE) == @__FILE__
const PAIR_EVALUATOR_CLI = PAIR_EVALUATOR_EXECUTED ? parse_cli(ARGS) : nothing

# Load the supplied policy adapter at top level.  This avoids Julia world-age
# failures while keeping model-specific code out of the canonical evaluator.
if PAIR_EVALUATOR_EXECUTED
    adapter_path = normpath(abspath(PAIR_EVALUATOR_CLI.options["--adapter"]))
    isfile(adapter_path) || error("candidate adapter does not exist: $adapter_path")
    Base.include(Main, adapter_path)
    isdefined(Main, :BeatFirstCandidateAdapter) || error(
        "adapter must define module BeatFirstCandidateAdapter"
    )
end

function stage_seeds(stage::String, flags::Set{String})
    if stage == "dev"
        return DEV_SEEDS
    elseif stage == "validation"
        "--promoted" in flags || error(
            "validation is locked until the candidate is explicitly promoted"
        )
        return VALIDATION_SEEDS
    elseif stage == "sealed"
        "--authorize-sealed" in flags || error(
            "sealed evaluation requires explicit root authorization"
        )
        return SEALED_SEEDS
    end
    error("stage must be dev, validation, or sealed")
end

function first_maximum_index(scores::AbstractVector)
    isempty(scores) && error("cannot select from an empty score vector")
    best_index = firstindex(scores)
    best_score = scores[best_index]
    @inbounds for index in (firstindex(scores) + 1):lastindex(scores)
        # Strict comparison intentionally preserves the first candidate on ties.
        if scores[index] > best_score
            best_index = index
            best_score = scores[index]
        end
    end
    return best_index
end

function candidate_score_result(adapter, policy, state, nodes)
    raw = adapter.score_candidate_set(policy, state, nodes)
    hasproperty(raw, :scores) || error("adapter result is missing scores")
    hasproperty(raw, :physical_requests) || error(
        "adapter result is missing physical_requests"
    )
    raw.scores isa AbstractVector || error("adapter scores must be a vector")
    scores = Float32.(collect(raw.scores))
    length(scores) == length(nodes) || error(
        "candidate adapter returned $(length(scores)) scores for $(length(nodes)) nodes"
    )
    all(isfinite, scores) || error("candidate adapter produced a non-finite score")
    physical_requests = Int(raw.physical_requests)
    physical_requests >= 1 || error("physical_requests must be positive")
    return (; scores, physical_requests)
end

function old_score_result(inference, state, nodes)
    input = legacy_candidate_batch(state, nodes; next_count=NEXT_COUNT)
    scores = openvino_scores(inference, input)
    length(scores) == length(nodes) || error("old OpenVINO candidate count mismatch")
    all(isfinite, scores) || error("old OpenVINO policy produced a non-finite score")
    return (;
        scores,
        physical_requests=chunked_backend_requests(length(nodes), OLD_BATCH_SIZE),
    )
end

function evaluate_episode(score_candidate_set, seed::Integer, role::String)
    state = GameState(Xoshiro(seed))
    steps = 0
    tetrises = 0
    tspins = 0
    perfect_clears = 0
    candidate_evaluations = 0
    logical_model_calls = 0
    physical_backend_requests = 0
    candidate_generation_seconds = 0.0
    inference_wall_seconds = 0.0
    started = time()

    while !state.game_over_flag && steps < EPISODE_PIECES
        nodes = Node[]
        candidate_generation_seconds += @elapsed nodes = stable_node_list(state)
        isempty(nodes) && break

        scored = nothing
        # This wall time includes model-specific feature packing on both sides.
        inference_wall_seconds += @elapsed scored = score_candidate_set(state, nodes)
        index = first_maximum_index(scored.scores)
        node = nodes[index]

        candidate_evaluations += length(nodes)
        logical_model_calls += 1
        physical_backend_requests += scored.physical_requests
        blocks_before = sum(state.current_game_board.binary)
        apply_node!(state, node)
        cleared = (blocks_before + 4 - sum(state.current_game_board.binary)) ÷ 10
        tetrises += cleared == 4
        tspins += node.tspin > 0
        perfect_clears += cleared > 0 && iszero(sum(state.current_game_board.binary))
        steps += 1
    end

    completed = steps == EPISODE_PIECES && !state.game_over_flag
    return (;
        seed=Int(seed),
        role,
        score=state.score,
        steps,
        completed,
        game_over=state.game_over_flag,
        survival_fraction=steps / EPISODE_PIECES,
        tetrises,
        tspins,
        perfect_clears,
        candidate_evaluations,
        logical_model_calls,
        physical_backend_requests,
        candidate_generation_seconds,
        inference_wall_seconds,
        wall_seconds=time() - started,
        candidates_per_inference_second=(
            inference_wall_seconds > 0 ?
            candidate_evaluations / inference_wall_seconds : nothing
        ),
    )
end

function distribution_summary(values::AbstractVector{<:Real})
    isempty(values) && error("cannot summarize an empty vector")
    data = Float64.(values)
    return (;
        mean=mean(data),
        median=median(data),
        maximum=maximum(data),
        minimum=minimum(data),
        p10=quantile(data, 0.10),
        p25=quantile(data, 0.25),
        p75=quantile(data, 0.75),
        p90=quantile(data, 0.90),
    )
end

function paired_bootstrap_mean_ci(
    differences::AbstractVector{<:Real};
    replicates::Int=BOOTSTRAP_REPLICATES,
    seed::UInt64=BOOTSTRAP_SEED,
)
    isempty(differences) && error("cannot bootstrap an empty vector")
    replicates > 0 || error("bootstrap replicates must be positive")
    data = Float64.(differences)
    rng = Xoshiro(seed)
    means = Vector{Float64}(undef, replicates)
    for replicate in 1:replicates
        total = 0.0
        for _ in eachindex(data)
            total += data[rand(rng, eachindex(data))]
        end
        means[replicate] = total / length(data)
    end
    return (;
        statistic="paired mean score difference (candidate - old)",
        replicates,
        seed=string(seed),
        lower_95=quantile(means, 0.025),
        upper_95=quantile(means, 0.975),
    )
end

function aggregate_results(candidate_records, old_records, pairs)
    length(candidate_records) == length(old_records) == length(pairs) || error(
        "paired result lengths differ"
    )
    differences = Float64[pair.score_difference for pair in pairs]
    candidate_scores = Float64[record.score for record in candidate_records]
    old_scores = Float64[record.score for record in old_records]
    candidate_completed = count(record -> record.completed, candidate_records)
    old_completed = count(record -> record.completed, old_records)
    count_pairs = length(pairs)
    return (;
        episodes=count_pairs,
        candidate_scores=distribution_summary(candidate_scores),
        old_scores=distribution_summary(old_scores),
        paired_differences=distribution_summary(differences),
        paired_bootstrap=paired_bootstrap_mean_ci(differences),
        wins=count(>(0), differences),
        ties=count(==(0), differences),
        losses=count(<(0), differences),
        candidate_completion_rate=candidate_completed / count_pairs,
        old_completion_rate=old_completed / count_pairs,
        paired_completion_rate_difference=(candidate_completed - old_completed) / count_pairs,
        candidate_mean_survival_steps=mean(record.steps for record in candidate_records),
        old_mean_survival_steps=mean(record.steps for record in old_records),
        candidate_totals=(;
            candidate_evaluations=sum(record.candidate_evaluations for record in candidate_records),
            logical_model_calls=sum(record.logical_model_calls for record in candidate_records),
            physical_backend_requests=sum(record.physical_backend_requests for record in candidate_records),
            inference_wall_seconds=sum(record.inference_wall_seconds for record in candidate_records),
            episode_wall_seconds=sum(record.wall_seconds for record in candidate_records),
        ),
        old_totals=(;
            candidate_evaluations=sum(record.candidate_evaluations for record in old_records),
            logical_model_calls=sum(record.logical_model_calls for record in old_records),
            physical_backend_requests=sum(record.physical_backend_requests for record in old_records),
            inference_wall_seconds=sum(record.inference_wall_seconds for record in old_records),
            episode_wall_seconds=sum(record.wall_seconds for record in old_records),
        ),
    )
end

function write_json_atomic(path::AbstractString, value)
    parent = dirname(path)
    mkpath(parent)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    mv(temporary, path; force=true)
    return path
end

function progress_document(
    status,
    stage,
    seeds,
    candidate_metadata,
    checkpoint,
    old_weights,
    compilation,
    candidate_records,
    old_records,
    pairs,
    started,
)
    return (;
        status,
        generated_at=string(now()),
        stage,
        seeds,
        rules=(;
            rng="Random.Xoshiro",
            board="24x10 (20 visible rows)",
            max_pieces=EPISODE_PIECES,
            next_count=NEXT_COUNT,
            hold_enabled=true,
            candidate_order="stable_node_key (duplicates preserved)",
            tie_break="strict first maximum",
            lookahead_expansions=0,
            logical_full_candidate_score_calls_per_decision=1,
        ),
        candidate=(;
            checkpoint,
            metadata=candidate_metadata,
            compilation_seconds=compilation.candidate_seconds,
        ),
        old=(;
            weights=old_weights,
            backend="OpenVINO NPU static batch 16 plus actual-size CPU tail",
            compilation_seconds=compilation.old_seconds,
        ),
        candidate_records,
        old_records,
        pairs,
        summary=isempty(pairs) ? nothing : aggregate_results(
            candidate_records, old_records, pairs
        ),
        wall_seconds=time() - started,
    )
end

function main(cli=PAIR_EVALUATOR_CLI)
    isnothing(cli) && error("evaluate_pair.jl must be run as a program")
    stage = cli.options["--stage"]
    seeds = stage_seeds(stage, cli.flags)
    checkpoint = normpath(abspath(cli.options["--checkpoint"]))
    output = normpath(abspath(cli.options["--output"]))
    isfile(checkpoint) || error("candidate checkpoint does not exist: $checkpoint")
    ispath(output) && error("refusing to overwrite evaluation output: $output")
    checkpoint = checkpoint_file_fingerprint(checkpoint)
    old_weights = checkpoint_file_fingerprint(
        joinpath(
            EVALUATION_ROOT,
            "artifacts",
            "legacy_openvino",
            "legacy_1313_weights.npz",
        ),
    )

    adapter = getfield(Main, :BeatFirstCandidateAdapter)
    for function_name in (
        :build_candidate_policy,
        :score_candidate_set,
        :candidate_policy_metadata,
    )
        isdefined(adapter, function_name) || error(
            "BeatFirstCandidateAdapter is missing $function_name"
        )
    end

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(EVALUATION_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    started = time()
    candidate_policy = nothing
    candidate_compile_seconds = @elapsed candidate_policy =
        adapter.build_candidate_policy(checkpoint.absolute_path)
    candidate_metadata = adapter.candidate_policy_metadata(candidate_policy)
    for field in (:architecture, :backend, :parameter_count)
        hasproperty(candidate_metadata, field) || error(
            "candidate metadata is missing $field"
        )
    end
    Int(candidate_metadata.parameter_count) > 0 || error(
        "candidate parameter_count must be positive"
    )
    old_inference = nothing
    old_compile_seconds = @elapsed old_inference =
        legacy_openvino.LegacyOpenVINOInference("NPU", OLD_BATCH_SIZE)
    compilation = (;
        candidate_seconds=candidate_compile_seconds,
        old_seconds=old_compile_seconds,
    )

    candidate_records = Any[]
    old_records = Any[]
    pairs = Any[]
    write_json_atomic(
        output,
        progress_document(
            "running",
            stage,
            seeds,
            candidate_metadata,
            checkpoint,
            old_weights,
            compilation,
            candidate_records,
            old_records,
            pairs,
            started,
        ),
    )

    for seed in seeds
        candidate = evaluate_episode(seed, "candidate") do state, nodes
            candidate_score_result(adapter, candidate_policy, state, nodes)
        end
        push!(candidate_records, candidate)
        old = evaluate_episode(seed, "canonical_old") do state, nodes
            old_score_result(old_inference, state, nodes)
        end
        push!(old_records, old)
        push!(
            pairs,
            (;
                seed,
                candidate_score=candidate.score,
                old_score=old.score,
                score_difference=candidate.score - old.score,
                candidate_steps=candidate.steps,
                old_steps=old.steps,
                survival_steps_difference=candidate.steps - old.steps,
                candidate_completed=candidate.completed,
                old_completed=old.completed,
                completion_difference=Int(candidate.completed) - Int(old.completed),
            ),
        )
        write_json_atomic(
            output,
            progress_document(
                "running",
                stage,
                seeds,
                candidate_metadata,
                checkpoint,
                old_weights,
                compilation,
                candidate_records,
                old_records,
                pairs,
                started,
            ),
        )
    end

    final = progress_document(
        "complete",
        stage,
        seeds,
        candidate_metadata,
        checkpoint,
        old_weights,
        compilation,
        candidate_records,
        old_records,
        pairs,
        started,
    )
    write_json_atomic(output, final)
    println(JSON3.write((; status=final.status, stage, output)))
end

PAIR_EVALUATOR_EXECUTED && main()
