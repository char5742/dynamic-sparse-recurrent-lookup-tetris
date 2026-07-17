ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

using SHA

function sha256_text(value::AbstractString)
    return bytes2hex(sha256(Vector{UInt8}(codeunits(value))))
end

function mino_type_index(mino)
    return something(
        findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS),
        0,
    )
end

function canonical_candidate_record(node::Node)
    return (;
        board=vec(Int8.(node.game_state.current_game_board.binary)),
        mino_index=mino_type_index(node.mino),
        direction=Int(node.mino.direction),
        y=Int(node.position.y),
        x=Int(node.position.x),
        tspin=Int(node.tspin),
        actions=[string(nameof(typeof(action))) for action in node.action_list],
    )
end

candidate_sha256(node::Node) = sha256_text(JSON3.write(canonical_candidate_record(node)))

function candidate_order_sha256(nodes::Vector{Node})
    hashes = candidate_sha256.(nodes)
    return hashes, sha256_text(join(hashes, "\n"))
end

function q_sha256(values)
    io = IOBuffer()
    for value in Float32.(values)
        write(io, reinterpret(UInt32, value))
    end
    return bytes2hex(sha256(take!(io)))
end

function canonical_lux_scores(
    model,
    parameters,
    model_state,
    input;
    inference_batch_size::Int=16,
)
    candidate_count = size(input[1], ndims(input[1]))
    chunks = Vector{Vector{Float32}}()
    elapsed = @elapsed for start_index in 1:inference_batch_size:candidate_count
        end_index = min(start_index + inference_batch_size - 1, candidate_count)
        batch_input = map(
            value -> selectdim(value, ndims(value), start_index:end_index), input
        )
        batch_scores, _ = model(batch_input, parameters, model_state)
        push!(chunks, vec(Array{Float32}(batch_scores)))
    end
    return reduce(vcat, chunks), elapsed
end

function top2_margin(scores::AbstractVector)
    length(scores) >= 2 || return Inf32
    indices = partialsortperm(scores, 1:2; rev=true)
    return Float32(scores[indices[1]] - scores[indices[2]])
end

function state_sha256(state::GameState)
    record = (;
        board=vec(Int8.(state.current_game_board.binary)),
        score=Int(state.score),
        ren=Int(state.ren),
        back_to_back=Bool(state.back_to_back_flag),
        current=mino_type_index(state.current_mino),
        hold=isnothing(state.hold_mino) ? 0 : mino_type_index(state.hold_mino),
        next=[mino_type_index(mino) for mino in state.mino_list],
        game_over=Bool(state.game_over_flag),
    )
    return sha256_text(JSON3.write(record))
end

function audit_decision(
    decision::Int,
    state::GameState,
    model,
    parameters,
    model_state,
    cpu_inference,
    npu_inference;
    next_count::Int=5,
    batch_size::Int=16,
)
    nodes = stable_node_list(state)
    isempty(nodes) && error("no candidates at audit decision $decision")
    candidate_hashes, order_hash = candidate_order_sha256(nodes)
    input = legacy_candidate_batch(state, nodes; next_count)

    lux_scores, lux_seconds = canonical_lux_scores(
        model, parameters, model_state, input; inference_batch_size=batch_size
    )
    cpu_seconds = @elapsed cpu_scores = openvino_scores(cpu_inference, input)
    npu_seconds = @elapsed npu_scores = openvino_scores(npu_inference, input)

    length(lux_scores) == length(cpu_scores) == length(npu_scores) == length(nodes) ||
        error("score length mismatch at decision $decision")
    lux_index = argmax(lux_scores)
    cpu_index = argmax(cpu_scores)
    npu_index = argmax(npu_scores)
    return (;
        decision,
        state_sha256=state_sha256(state),
        candidate_count=length(nodes),
        full_batches=fld(length(nodes), batch_size),
        tail_size=rem(length(nodes), batch_size),
        candidate_order_sha256=order_hash,
        candidate_sha256=candidate_hashes,
        q=(;
            lux_fp32=lux_scores,
            openvino_cpu=cpu_scores,
            openvino_npu=npu_scores,
            lux_sha256=q_sha256(lux_scores),
            openvino_cpu_sha256=q_sha256(cpu_scores),
            openvino_npu_sha256=q_sha256(npu_scores),
            cpu_vs_lux_max_abs=maximum(abs.(cpu_scores .- lux_scores)),
            npu_vs_lux_max_abs=maximum(abs.(npu_scores .- lux_scores)),
            cpu_vs_npu_max_abs=maximum(abs.(cpu_scores .- npu_scores)),
        ),
        argmax=(;
            lux=lux_index,
            openvino_cpu=cpu_index,
            openvino_npu=npu_index,
            cpu_matches_lux=cpu_index == lux_index,
            npu_matches_lux=npu_index == lux_index,
            lux_candidate_sha256=candidate_hashes[lux_index],
            cpu_candidate_sha256=candidate_hashes[cpu_index],
            npu_candidate_sha256=candidate_hashes[npu_index],
        ),
        top2_margin=(;
            lux=top2_margin(lux_scores),
            openvino_cpu=top2_margin(cpu_scores),
            openvino_npu=top2_margin(npu_scores),
        ),
        inference_seconds=(;
            lux_fp32=lux_seconds,
            openvino_cpu=cpu_seconds,
            openvino_npu=npu_seconds,
        ),
        selected_node=nodes[lux_index],
    )
end

function baseline_conformance_main()
    seed = parse(Int, get(ENV, "CONFORMANCE_SEED", "5742"))
    decisions_requested = parse(Int, get(ENV, "CONFORMANCE_DECISIONS", "4"))
    batch_size = parse(Int, get(ENV, "CONFORMANCE_BATCH", "16"))
    next_count = parse(Int, get(ENV, "CONFORMANCE_NEXT", "5"))
    output_path = get(
        ENV,
        "CONFORMANCE_OUTPUT",
        joinpath(ROOT, "artifacts", "baseline_conformance", "seed$(seed).json"),
    )

    checkpoint_path = joinpath(ROOT, "1313", "mainmodel copy 3.jld2")
    parameters, model_state = jldopen(checkpoint_path, "r") do file
        modernize_legacy_parameters(file["ps"]), Lux.testmode(file["st"])
    end
    model = LegacyQNetwork()
    BLAS.set_num_threads(min(parse(Int, get(ENV, "LEGACY_BLAS_THREADS", "12")), Threads.nthreads()))

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    @info "Compiling baseline conformance backends" batch_size
    cpu_inference = legacy_openvino.LegacyOpenVINOInference("CPU", batch_size)
    npu_inference = legacy_openvino.LegacyOpenVINOInference("NPU", batch_size)

    state = GameState(Xoshiro(seed))
    decisions = NamedTuple[]
    started = time()
    for decision in 1:decisions_requested
        state.game_over_flag && break
        result = audit_decision(
            decision,
            state,
            model,
            parameters,
            model_state,
            cpu_inference,
            npu_inference;
            next_count,
            batch_size,
        )
        push!(decisions, Base.structdiff(result, (; selected_node=result.selected_node)))
        apply_node!(state, result.selected_node)
        @info "Audited canonical trajectory decision" decision candidate_count=result.candidate_count cpu_argmax=result.argmax.cpu_matches_lux npu_argmax=result.argmax.npu_matches_lux
    end

    saw_full_batch = any(item.full_batches > 0 for item in decisions)
    saw_cpu_tail = any(item.tail_size > 0 for item in decisions)
    saw_full_batch || error("audit did not exercise a complete batch")
    saw_cpu_tail || error("audit did not exercise a short CPU tail")
    cpu_action_match = all(item.argmax.cpu_matches_lux for item in decisions)
    npu_action_match = all(item.argmax.npu_matches_lux for item in decisions)
    result = (;
        generated_at=string(now()),
        audit_schema="baseline-conformance-v1",
        canonical_policy="Lux FP32 CPU, stable candidate order, batch 16 with actual-size tail",
        trajectory_driver="canonical Lux argmax",
        seed_set="development",
        seed,
        next_count,
        batch_size,
        decisions_requested,
        decisions_observed=length(decisions),
        checkpoint_path,
        checkpoint_sha256=bytes2hex(open(sha256, checkpoint_path)),
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        backends=(lux="CPU FP32", openvino_cpu="CPU", openvino_npu="NPU with dynamic CPU tail"),
        batch_coverage=(; saw_full_batch, saw_cpu_tail),
        action_conformance=(;
            openvino_cpu_all=cpu_action_match,
            openvino_npu_all=npu_action_match,
            trajectory_action_sha256=sha256_text(
                join((item.argmax.lux_candidate_sha256 for item in decisions), "\n")
            ),
        ),
        maxima=(;
            cpu_vs_lux_max_abs=maximum(item.q.cpu_vs_lux_max_abs for item in decisions),
            npu_vs_lux_max_abs=maximum(item.q.npu_vs_lux_max_abs for item in decisions),
            minimum_lux_top2_margin=minimum(item.top2_margin.lux for item in decisions),
        ),
        latency_valid=lowercase(get(ENV, "CONFORMANCE_LATENCY_VALID", "false")) == "true",
        latency_note=get(
            ENV,
            "CONFORMANCE_LATENCY_NOTE",
            "Numerical/action conformance run; latency is uncontrolled unless explicitly marked valid.",
        ),
        wall_seconds=time() - started,
        decisions,
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Saved baseline conformance audit" output_path result.action_conformance result.maxima
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    baseline_conformance_main()
end
