include(joinpath(@__DIR__, "evaluate_legacy_checkpoint.jl"))
include(joinpath(@__DIR__, "evaluation_artifact_helpers.jl"))
include(joinpath(ROOT, "experiments", "learning", "compact_model.jl"))

using Dates
using JLD2
using JSON3
using Statistics

function load_compact_checkpoint(path::AbstractString)
    return jldopen(path, "r") do file
        config = file["model_config"]
        model = CompactCandidateQ(;
            channels=Int(config.channels),
            blocks=Int(config.blocks),
            spatial_channels=Int(config.spatial_channels),
        )
        return (;
            model,
            parameters=file["ps"],
            state=Lux.testmode(file["st"]),
            config,
            parameter_count=Lux.parameterlength(file["ps"]),
        )
    end
end

function select_compact_node(state::GameState, checkpoint; next_count::Int=5)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nothing, generation_seconds, 0.0, 0
    input = legacy_candidate_batch(state, nodes; next_count)
    inference_seconds = @elapsed raw, _ = checkpoint.model(
        input, checkpoint.parameters, checkpoint.state
    )
    scores = vec(Array(raw))
    length(scores) == length(nodes) || error("candidate/output length mismatch")
    all(isfinite, scores) || error("compact checkpoint produced a non-finite score")
    return nodes[argmax(scores)], generation_seconds, inference_seconds, length(nodes)
end

function evaluate_compact_episode(
    seed::Integer,
    checkpoint;
    next_count::Int=5,
    max_steps::Int=PAPER_EPISODE_STEPS,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    candidate_count = 0
    network_calls = 0
    generation_seconds = 0.0
    inference_seconds = 0.0
    started = time()
    while !state.game_over_flag && steps < max_steps
        node, generation, inference, candidates = select_compact_node(
            state, checkpoint; next_count
        )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference
        candidate_count += candidates
        network_calls += 1
        apply_node!(state, node)
        steps += 1
        if steps == 1 || steps % 25 == 0 || state.game_over_flag
            @info "Compact checkpoint episode" seed steps score=state.score candidate_count network_calls generation_seconds inference_seconds wall_seconds=time()-started
        end
    end
    compute_accounting = evaluation_compute_accounting(
        candidate_count, network_calls, network_calls
    )
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        candidate_count,
        network_calls,
        compute_accounting...,
        lookahead_expansions=0,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
        candidates_per_inference_second=candidate_count / max(inference_seconds, eps()),
    )
end

function main()
    checkpoint_path = abspath(get(
        ENV,
        "COMPACT_CHECKPOINT",
        raw"D:\tetris-paper-plus\checkpoints\learning\compact_q_smoke.jld2",
    ))
    seeds = parse.(Int, split(get(ENV, "COMPACT_EVAL_SEEDS", "5742"), ','))
    max_steps = parse(Int, get(ENV, "COMPACT_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    next_count = parse(Int, get(ENV, "COMPACT_EVAL_NEXT", "5"))
    checkpoint_fingerprint = checkpoint_file_fingerprint(checkpoint_path)
    checkpoint = load_compact_checkpoint(checkpoint_path)
    compile_seconds = @elapsed begin
        # Compile on a real candidate set but do not advance the scored episode.
        warm_state = GameState(Xoshiro(first(seeds)))
        warm_nodes = stable_node_list(warm_state)
        warm_input = legacy_candidate_batch(warm_state, warm_nodes; next_count)
        checkpoint.model(warm_input, checkpoint.parameters, checkpoint.state)
    end
    episodes = [
        evaluate_compact_episode(seed, checkpoint; next_count, max_steps)
        for seed in seeds
    ]
    scores = getproperty.(episodes, :score)
    result = (;
        timestamp=string(now()),
        experiment="compact_model_fixed_budget_evaluation",
        checkpoint_path,
        checkpoint_fingerprint,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        backend="Lux native",
        device="CPU",
        model_config=checkpoint.config,
        parameter_count=checkpoint.parameter_count,
        next_count,
        max_steps,
        network_calls_per_decision=1,
        lookahead_expansions=0,
        search_budget=(;
            episode_piece_limit=max_steps,
            next_count,
            hold_enabled=true,
            candidate_order="stable_node_key",
            lookahead_expansions=0,
            logical_network_calls_per_decision=1,
            selection="argmax_candidate_value",
            inference_batch_size="all_candidates_in_one_forward",
        ),
        compile_seconds,
        episodes,
        mean_score=mean(scores),
        median_score=median(scores),
        maximum_score=maximum(scores),
        completion_rate=mean(item.steps >= max_steps && !item.game_over for item in episodes),
    )
    output_root = get(
        ENV, "COMPACT_EVAL_OUTPUT_ROOT", raw"D:\tetris-paper-plus\runs\learning"
    )
    mkpath(output_root)
    seed_tag = join(seeds, "-")
    output_path = joinpath(
        output_root,
        evaluation_result_filename(
            "compact_eval", seed_tag, next_count, max_steps
        ),
    )
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Saved compact fixed-budget evaluation" output_path result.mean_score result.median_score
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
